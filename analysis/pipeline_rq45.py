"""Statistical analysis for the Hindi MGT study."""
from __future__ import annotations

import argparse

import itertools

import os

import warnings

from pathlib import Path

from typing import Dict, List, Tuple, Optional

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

from matching import (
    normalize_text_for_matching,
    safe_paired_matrix,
    dedupe_cell,
)

from stats_tests import (
    mcnemar_pairwise,
    holm_correction,
    cochrans_q_test,
    friedman_across_detectors,
    wilcoxon_paired,
)

from study import DATASETS, LLMS, TRANSFER_CAPABLE_MODELS

RQ4_DATASET = "coling_hindi"

RQ4_TEST_LLM = "gpt4o"

RQ5_LLMS = {"gemini", "qwen3"}

RQ45_MODELS = [
    "XLM-RoBERTa",
    "Indic-BERT",
    "RADAR",
    "CONDA",
    "Text-Fluoroscopy",
    "GPTZero",
    "DetectGPT",
]

MODEL_ORDER = RQ45_MODELS.copy()

def cell_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute hard-label metrics and AUC when a real continuous score exists."""
    y_true = df["true_label"].to_numpy()
    y_pred = df["predicted_label"].to_numpy()

    out = {
        "n": len(df),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "ai_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "human_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),
    }

    scored = df.dropna(subset=["score"])

    if (
        len(scored) >= 4
        and scored["true_label"].nunique() == 2
        and scored["score"].notna().all()
    ):
        try:
            out["auc"] = float(
                roc_auc_score(
                    scored["true_label"].to_numpy(),
                    scored["score"].to_numpy(),
                )
            )
            out["auc_n"] = len(scored)
        except Exception:
            out["auc"] = np.nan
            out["auc_n"] = len(scored)
    else:
        out["auc"] = np.nan
        out["auc_n"] = len(scored)

    return out

def prompt_setting(
    train_dataset: str,
    train_llm: str,
    test_dataset: str,
    test_llm: str,
) -> str:
    """
    Classify an RQ5 cell using the same three shift types used by RQ1-RQ3.
    """
    if (
        train_dataset == test_dataset
        and train_llm == test_llm
    ):
        return "in_domain"

    if (
        train_dataset == test_dataset
        and train_llm != test_llm
    ):
        return "llm_cross_domain"

    if (
        train_dataset != test_dataset
        and train_llm == test_llm
    ):
        return "dataset_cross_domain"

    return "other_unseen_both"

def rq4_metrics(
    rq4: pd.DataFrame,
    outdir: Path,
) -> pd.DataFrame:
    """Compute one row per detector/source dataset/source LLM."""
    rows = []

    group_cols = [
        "model",
        "train_dataset",
        "train_llm",
    ]

    for keys, grp in rq4.groupby(group_cols):

        model, train_ds, train_llm = keys

        metrics = cell_metrics(grp)

        rows.append({
            **metrics,
            "model": model,
            "train_dataset": train_ds,
            "train_llm": train_llm,
            "test_dataset": RQ4_DATASET,
            "test_llm": RQ4_TEST_LLM,
        })

    metrics_df = pd.DataFrame(rows)

    metrics_df["model"] = pd.Categorical(
        metrics_df["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    metrics_df = metrics_df.sort_values(
        [
            "train_dataset",
            "train_llm",
            "model",
        ]
    )

    metrics_df.to_csv(
        outdir / "rq4_unseen_both_metrics.csv",
        index=False,
    )

    return metrics_df

def rq4_detector_comparison(
    rq4: pd.DataFrame,
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare the seven detectors on the exact same COLING examples.

    Cochran's Q:
        overall detector difference within each source training cell.

    McNemar:
        detector-vs-detector differences within each source training cell.
    """
    cochran_rows = []
    pairwise_rows = []

    for train_ds in DATASETS:
        for train_llm in LLMS:

            grp = rq4[
                (rq4["train_dataset"] == train_ds)
                & (rq4["train_llm"] == train_llm)
                & (rq4["model"].isin(MODEL_ORDER))
            ].copy()

            if grp.empty:
                continue

            pivot, truth, audit = safe_paired_matrix(
                grp,
                value_col="predicted_label",
            )

            if pivot.empty:
                continue

            # Only run a five-detector comparison when all expected
            # detectors are present.
            missing_models = [
                m for m in MODEL_ORDER
                if m not in pivot.columns
            ]

            if missing_models:
                warnings.warn(
                    f"[RQ4] Missing detectors for "
                    f"{train_ds}/{train_llm}: {missing_models}"
                )
                continue

            pivot = pivot[MODEL_ORDER]

            correctness = (
                pivot.eq(
                    truth,
                    axis=0,
                )
                .astype(int)
            )

            q, df_q, p_q = cochrans_q_test(
                correctness.to_numpy()
            )

            cochran_rows.append({
                "train_dataset": train_ds,
                "train_llm": train_llm,
                "n_items": len(pivot),
                "n_detectors": len(MODEL_ORDER),
                "Q": q,
                "df": df_q,
                "p_value": p_q,
            })

            for model_a, model_b in itertools.combinations(
                MODEL_ORDER,
                2,
            ):

                p_value, table, detail = mcnemar_pairwise(
                    truth.to_numpy(),
                    pivot[model_a].to_numpy(),
                    pivot[model_b].to_numpy(),
                )

                pairwise_rows.append({
                    "train_dataset": train_ds,
                    "train_llm": train_llm,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_items": len(pivot),
                    "p_value": p_value,
                    **detail,
                })

    cochran_df = pd.DataFrame(cochran_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    if not pairwise_df.empty:
        pairwise_df["p_holm_global"] = holm_correction(
            pairwise_df["p_value"]
            .fillna(1.0)
            .to_numpy()
        )

        pairwise_df["significant_holm_global"] = (
            pairwise_df["p_holm_global"] < 0.05
        )

    cochran_df.to_csv(
        outdir / "rq4_cochrans_q.csv",
        index=False,
    )

    pairwise_df.to_csv(
        outdir / "rq4_mcnemar_pairwise.csv",
        index=False,
    )

    return cochran_df, pairwise_df

def rq4_training_dataset_comparison(
    rq4: pd.DataFrame,
    outdir: Path,
) -> pd.DataFrame:
    """
    Compare HindiSumm-trained and XQuad-trained models on the same COLING
    test examples.

    A separate paired McNemar comparison is made for:
        detector x training LLM

    There are 5 detectors x 3 LLMs = 15 comparisons.
    """
    rows = []

    for model in MODEL_ORDER:
        for train_llm in LLMS:

            hs = rq4[
                (rq4["model"] == model)
                & (rq4["train_dataset"] == "hindisumm")
                & (rq4["train_llm"] == train_llm)
            ].copy()

            xq = rq4[
                (rq4["model"] == model)
                & (rq4["train_dataset"] == "xquad")
                & (rq4["train_llm"] == train_llm)
            ].copy()

            if hs.empty or xq.empty:
                continue

            # The COLING set contains a repeated normalized text key. Collapse
            # duplicate rows only when truth and prediction agree; conflicting
            # duplicates are excluded rather than entering a many-to-many join.
            hs_deduped, hs_conflicts = dedupe_cell(hs)
            xq_deduped, xq_conflicts = dedupe_cell(xq)

            # Make sure we pair by normalized text rather than raw text.
            hs_deduped = hs_deduped[
                ["text_key", "true_label", "predicted_label"]
            ].rename(columns={
                "true_label": "true_hs", "predicted_label": "pred_hs",
            })
            xq_deduped = xq_deduped[
                ["text_key", "true_label", "predicted_label"]
            ].rename(columns={
                "true_label": "true_xq", "predicted_label": "pred_xq",
            })
            merged = hs_deduped.merge(
                xq_deduped, on="text_key", how="inner", validate="one_to_one"
            )

            if merged.empty:
                continue

            # A paired detector comparison requires a common ground truth.
            merged = merged[
                merged["true_hs"] == merged["true_xq"]
            ]

            if len(merged) < 2:
                continue

            p_value, table, detail = mcnemar_pairwise(
                merged["true_hs"].to_numpy(),
                merged["pred_hs"].to_numpy(),
                merged["pred_xq"].to_numpy(),
            )

            rows.append({
                "model": model,
                "train_llm": train_llm,
                "n_items": len(merged),
                "n_conflicting_duplicate_keys_hindisumm": hs_conflicts,
                "n_conflicting_duplicate_keys_xquad": xq_conflicts,
                "p_value": p_value,
                **detail,
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["p_holm_global"] = holm_correction(
            result["p_value"]
            .fillna(1.0)
            .to_numpy()
        )

        result["significant_holm_global"] = (
            result["p_holm_global"] < 0.05
        )

    result.to_csv(
        outdir / "rq4_training_dataset_comparison.csv",
        index=False,
    )

    return result

def rq4_friedman(
    rq4_metrics_df: pd.DataFrame,
    outdir: Path,
) -> pd.DataFrame:
    """
    Compare detectors across the six unseen-source configurations.

    Blocks:
        train_dataset x train_llm

    Columns:
        detectors

    Metrics:
        AI Recall
        Human Recall
        Accuracy
    """
    rows = []

    block_cols = [
        "train_dataset",
        "train_llm",
    ]

    for metric in [
        "ai_recall",
        "human_recall",
        "accuracy",
    ]:

        wide = (
            rq4_metrics_df
            .pivot_table(
                index=block_cols,
                columns="model",
                values=metric,
                aggfunc="first",
            )
            .reindex(columns=MODEL_ORDER)
        )

        wide = wide.dropna(
            how="any"
        )

        if (
            len(wide) < 2
            or wide.shape[1] < 2
        ):
            continue

        stat, p = friedman_across_detectors(
            wide.to_numpy()
        )

        rows.append({
            "metric": metric,
            "n_blocks": len(wide),
            "n_detectors": wide.shape[1],
            "chi2": stat,
            "p_value": p,
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        outdir / "rq4_friedman.csv",
        index=False,
    )

    return result

def make_rq4_plots(
    metrics_df: pd.DataFrame,
    outdir: Path,
) -> None:
    """Create simple RQ4 plots."""
    if metrics_df.empty:
        return

    # AI Recall by training configuration.
    plot_df = metrics_df.copy()

    plot_df["source"] = (
        plot_df["train_dataset"]
        + " / "
        + plot_df["train_llm"]
    )

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=plot_df,
        x="source",
        y="ai_recall",
        hue="model",
        errorbar=None,
    )

    plt.ylim(0, 1)
    plt.ylabel("AI Recall")
    plt.xlabel("Training dataset / LLM")
    plt.title(
        "RQ4: AI Recall on the Unseen COLING Hindi Dataset"
    )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        outdir / "fig_rq4_ai_recall.png",
        dpi=180,
    )
    plt.close()

    # Human Recall.
    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=plot_df,
        x="source",
        y="human_recall",
        hue="model",
        errorbar=None,
    )

    plt.ylim(0, 1)
    plt.ylabel("Human Recall")
    plt.xlabel("Training dataset / LLM")
    plt.title(
        "RQ4: Human Recall on the Unseen COLING Hindi Dataset"
    )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        outdir / "fig_rq4_human_recall.png",
        dpi=180,
    )
    plt.close()

def compute_cell_metric_table(
    df: pd.DataFrame,
    prompt_name: str,
) -> pd.DataFrame:
    """Convert example-level rows into one row per experimental cell."""
    rows = []

    group_cols = [
        "model",
        "train_dataset",
        "train_llm",
        "test_dataset",
        "test_llm",
    ]

    for keys, grp in df.groupby(group_cols):

        (
            model,
            train_ds,
            train_llm,
            test_ds,
            test_llm,
        ) = keys

        metrics = cell_metrics(grp)

        rows.append({
            **metrics,
            "model": model,
            "train_dataset": train_ds,
            "train_llm": train_llm,
            "test_dataset": test_ds,
            "test_llm": test_llm,
            "setting": prompt_setting(
                train_ds,
                train_llm,
                test_ds,
                test_llm,
            ),
            "prompt": prompt_name,
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["model"] = pd.Categorical(
            result["model"],
            categories=MODEL_ORDER,
            ordered=True,
        )

    return result

def match_rq5_original_and_new(
    original_metrics: pd.DataFrame,
    new_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pair original-prompt and new-prompt results by the same experimental cell.
    """
    key_cols = [
        "model",
        "train_dataset",
        "train_llm",
        "test_dataset",
        "test_llm",
    ]

    merged = original_metrics.merge(
        new_metrics,
        on=key_cols,
        how="inner",
        suffixes=("_original", "_new"),
    )

    return merged

def rq5_prompt_change_wilcoxon(
    merged: pd.DataFrame,
    outdir: Path,
) -> pd.DataFrame:
    """
    Primary RQ5 inference:

    For each of:
        in_domain
        llm_cross_domain
        dataset_cross_domain

    compare old-prompt vs new-prompt metrics using paired Wilcoxon.

    Three metrics:
        accuracy
        ai_recall
        human_recall

    Nine primary tests total.
    Holm correction is applied globally over these nine tests.
    """
    primary_rows = []

    settings = [
        "in_domain",
        "llm_cross_domain",
        "dataset_cross_domain",
    ]

    metrics = [
        "accuracy",
        "ai_recall",
        "human_recall",
    ]

    for setting in settings:
        subset = merged[
            merged["setting_original"] == setting
        ].copy()

        for metric in metrics:

            original = subset[
                f"{metric}_original"
            ].to_numpy(dtype=float)

            new = subset[
                f"{metric}_new"
            ].to_numpy(dtype=float)

            mask = ~(
                np.isnan(original)
                | np.isnan(new)
            )

            original = original[mask]
            new = new[mask]

            if len(original) < 2:
                stat = np.nan
                p = np.nan
            else:
                stat, p = wilcoxon_paired(
                    original,
                    new,
                )

            primary_rows.append({
                "setting": setting,
                "metric": metric,
                "n_pairs": len(original),
                "mean_original": (
                    float(np.mean(original))
                    if len(original)
                    else np.nan
                ),
                "mean_new": (
                    float(np.mean(new))
                    if len(new)
                    else np.nan
                ),
                "mean_change": (
                    float(np.mean(new - original))
                    if len(original)
                    else np.nan
                ),
                "wilcoxon_stat": stat,
                "p_value": p,
            })

    result = pd.DataFrame(
        primary_rows
    )

    if not result.empty:
        result["p_holm_global"] = holm_correction(
            result["p_value"]
            .fillna(1.0)
            .to_numpy()
        )

        result["significant_holm"] = (
            result["p_holm_global"] < 0.05
        )

    result.to_csv(
        outdir / "rq5_prompt_change_wilcoxon.csv",
        index=False,
    )

    return result

def rq5_per_detector_wilcoxon(
    merged: pd.DataFrame,
    outdir: Path,
) -> pd.DataFrame:
    """
    Secondary analysis:
    old-vs-new prompt within each detector and each setting.

    These are exploratory/supporting results. The globally corrected pooled
    nine-test table above is the primary RQ5 significance analysis.
    """
    rows = []

    for model in MODEL_ORDER:

        for setting in [
            "in_domain",
            "llm_cross_domain",
            "dataset_cross_domain",
        ]:

            subset = merged[
                (merged["model"] == model)
                & (
                    merged["setting_original"]
                    == setting
                )
            ].copy()

            for metric in [
                "accuracy",
                "ai_recall",
                "human_recall",
            ]:

                original = subset[
                    f"{metric}_original"
                ].to_numpy(dtype=float)

                new = subset[
                    f"{metric}_new"
                ].to_numpy(dtype=float)

                mask = ~(
                    np.isnan(original)
                    | np.isnan(new)
                )

                original = original[mask]
                new = new[mask]

                if len(original) >= 2:
                    stat, p = wilcoxon_paired(
                        original,
                        new,
                    )
                else:
                    stat, p = np.nan, np.nan

                rows.append({
                    "model": model,
                    "setting": setting,
                    "metric": metric,
                    "n_pairs": len(original),
                    "mean_original": (
                        float(np.mean(original))
                        if len(original)
                        else np.nan
                    ),
                    "mean_new": (
                        float(np.mean(new))
                        if len(new)
                        else np.nan
                    ),
                    "mean_change": (
                        float(np.mean(new - original))
                        if len(original)
                        else np.nan
                    ),
                    "wilcoxon_stat": stat,
                    "p_value": p,
                })

    result = pd.DataFrame(rows)

    if not result.empty:
        result["p_holm_within_model_setting"] = (
            result.groupby(
                ["model", "setting"]
            )["p_value"]
            .transform(
                lambda s: holm_correction(
                    s.fillna(1.0).to_numpy()
                )
            )
        )

    result.to_csv(
        outdir / "rq5_per_detector_wilcoxon.csv",
        index=False,
    )

    return result

def rq5_largest_changes(
    merged: pd.DataFrame,
    outdir: Path,
) -> pd.DataFrame:
    """
    Identify concrete prompt-change failures.

    This is descriptive, not a significance test.
    """
    rows = []

    metric_labels = [
        "accuracy",
        "ai_recall",
        "human_recall",
        "f1",
    ]

    for _, row in merged.iterrows():

        setting = row["setting_original"]

        for metric in metric_labels:

            old = row.get(
                f"{metric}_original",
                np.nan,
            )
            new = row.get(
                f"{metric}_new",
                np.nan,
            )

            if np.isnan(old) or np.isnan(new):
                continue

            change = new - old

            rows.append({
                "model": row["model"],
                "train_dataset": row["train_dataset"],
                "train_llm": row["train_llm"],
                "test_dataset": row["test_dataset"],
                "test_llm": row["test_llm"],
                "setting": setting,
                "metric": metric,
                "original": old,
                "new": new,
                "change": change,
                "absolute_change": abs(change),
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "absolute_change",
            ascending=False,
        )

    result.to_csv(
        outdir / "rq5_prompt_change_largest_changes.csv",
        index=False,
    )

    return result

def make_rq5_plots(
    merged: pd.DataFrame,
    outdir: Path,
) -> None:
    """Create simple original-vs-new prompt plots."""
    if merged.empty:
        return

    for metric, ylabel in [
        ("ai_recall", "AI Recall"),
        ("human_recall", "Human Recall"),
        ("accuracy", "Accuracy"),
    ]:

        plot_rows = []

        for _, row in merged.iterrows():

            for prompt_name, suffix in [
                ("Original prompt", "_original"),
                ("New prompt", "_new"),
            ]:

                value = row[
                    f"{metric}{suffix}"
                ]

                if np.isnan(value):
                    continue

                plot_rows.append({
                    "model": row["model"],
                    "setting": row[
                        "setting_original"
                    ],
                    "prompt": prompt_name,
                    "value": value,
                })

        plot_df = pd.DataFrame(
            plot_rows
        )

        if plot_df.empty:
            continue

        plt.figure(figsize=(11, 6))

        sns.boxplot(
            data=plot_df,
            x="model",
            y="value",
            hue="prompt",
        )

        plt.ylim(0, 1)
        plt.ylabel(ylabel)
        plt.xlabel("Detector")
        plt.title(
            f"RQ5: {ylabel} before and after prompt change"
        )
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        filename = (
            f"fig_rq5_{metric}_prompt_change.png"
        )

        plt.savefig(
            outdir / filename,
            dpi=180,
        )

        plt.close()
