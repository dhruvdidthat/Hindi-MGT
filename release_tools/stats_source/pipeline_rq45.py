"""
pipeline_rq45.py
================

RQ4:
    How adaptable are detectors when BOTH the dataset and LLM are unknown?

RQ5:
    How flexible are detectors to prompt change?

This pipeline is intentionally separate from the RQ1-RQ3 pipeline, but it
uses the same canonical loaders and source-calibration policy. All seven
detectors are supported. GPTZero/DetectGPT thresholds are fitted on the
source calibration split and transferred unchanged to COLING/new-prompt tests.

It:
    1. Loads RQ4 COLING-2025 Hindi predictions.
    2. Loads RQ5 new-prompt predictions.
    3. Uses the existing RQ1-RQ3 loader output as the ORIGINAL-PROMPT
       baseline for RQ5.
    4. Canonicalizes labels/scores using the detector-specific conventions.
    5. Computes Accuracy, F1, AI Recall, Human Recall, and valid AUC.
    6. Performs:
         RQ4:
            - Cochran's Q
            - pairwise McNemar + global Holm correction
            - HindiSumm-vs-XQuad training-dataset comparison
            - Friedman detector comparison
         RQ5:
            - paired Wilcoxon old-vs-new prompt comparisons
            - pooled and per-detector comparisons
            - Holm correction for primary pooled tests
            - largest individual prompt-change effects
    7. Writes CSVs, plots, and findings.md.

IMPORTANT:
    - XLM-RoBERTa RQ4 predictions.csv has no continuous score, so AUC is NaN.
    - RQ5 Indic-BERT `predicted_prob` and RADAR `predicted_prob` are NOT used
      for AUC because the provided schema does not establish whether they are
      P(AI), P(Human), or another probability.
    - RQ5 XLM-RoBERTa `predicted_prob_label0` is treated as P(AI), because
      the audited raw label convention for XLM-RoBERTa is 0=AI.
    - RQ5 includes Gemini and Qwen3 only, matching the paper's stated RQ5 scope.
    - Llama3 prompt-change files are ignored even though they exist on disk.
"""

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

# ---------------------------------------------------------------------------
# Existing validated pipeline utilities
# ---------------------------------------------------------------------------

from loaders import (
    load_all,
    DATASETS,
    LLMS,
    TRANSFER_CAPABLE_MODELS,
    ZERO_SHOT_SPECS,
    load_zero_shot_rq4_all,
    load_zero_shot_rq5_all,
    set_detectgpt_rebuilt_root,
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RQ4_DATASET = "coling_hindi"

# The COLING Hindi test set contains GPT-4o generated text according to
# the current paper/evaluation setup.
RQ4_TEST_LLM = "gpt4o"

# RQ5 follows the paper's stated scope: Gemini and Qwen3.
RQ5_LLMS = {"gemini", "qwen3"}

# Same all seven detectors used for RQ2/RQ3.
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


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV without crashing the whole pipeline."""
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def canonicalize_label(
    values: pd.Series,
    raw_ai_label: int,
) -> np.ndarray:
    """
    Convert detector-specific raw labels to:
        1 = AI
        0 = Human
    """
    arr = values.astype(int).to_numpy()

    if raw_ai_label == 1:
        return arr

    if raw_ai_label == 0:
        return 1 - arr

    raise ValueError(f"raw_ai_label must be 0 or 1, got {raw_ai_label}")


def canonicalize_score(
    values: pd.Series,
    direction: str,
) -> np.ndarray:
    """
    Convert a detector score into:
        higher = more AI

    direction:
        ai_native   -> score already means P/score of AI
        human_prob  -> score is P(Human), so use 1-score
        none        -> all NaN
    """
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    if direction == "ai_native":
        return arr

    if direction == "human_prob":
        return 1.0 - arr

    if direction == "none":
        return np.full(len(arr), np.nan)

    raise ValueError(f"Unknown score direction: {direction}")


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


def metric_direction_text(
    original: float,
    new: float,
) -> str:
    if np.isnan(original) or np.isnan(new):
        return "undefined"

    if new > original:
        return "increase"

    if new < original:
        return "decrease"

    return "unchanged"


# ===========================================================================
# RQ4
# ===========================================================================

# Exact detector-specific schemas and directory conventions supplied by user.
RQ4_CONFIG = {
    "XLM-RoBERTa": {
        "subdir": "xlm_roberta",
        "results_dir": "inference_results",
        "train_dir": "{ds}_{llm}",
        "label_col": "label",
        "pred_col": "predicted_label",
        "raw_ai_label": 0,
        "score_col": None,
        "score_direction": "none",
    },

    "Indic-BERT": {
        "subdir": "indicBert",
        "results_dir": "inference_results",
        "train_dir": "{ds}_{llm}",
        "label_col": "label",
        "pred_col": "predicted_label",
        "raw_ai_label": 0,
        "score_col": "predicted_prob_human",
        "score_direction": "human_prob",
    },

    "RADAR": {
        "subdir": "RADAR",
        "results_dir": "inference_results",
        "train_dir": "{ds}_{llm}",
        "label_col": "true_label",
        "pred_col": "predicted_label",
        "raw_ai_label": 0,
        "score_col": "prediction_probability",
        "score_direction": "human_prob",
    },

    "CONDA": {
        "subdir": "CONDA",
        "results_dir": "inference_results",
        # CONDA uniquely doubles the training LLM in its directory name.
        "train_dir": "{ds}_{llm}_{llm}",
        "label_col": "true_label",
        "pred_col": "predicted_label",
        "raw_ai_label": 0,
        "score_col": "prediction_probability_class1",
        "score_direction": "human_prob",
    },

    "Text-Fluoroscopy": {
        "subdir": "text_fluoroscopy",
        "results_dir": "inference_results_coling",
        "train_dir": "{ds}_{llm}",
        "label_col": "true_label",
        "pred_col": "predicted_label",
        "raw_ai_label": 1,
        "score_col": "predicted_prob_label1",
        "score_direction": "ai_native",
    },
}


def load_rq4_coling(
    rq4_root: Path,
    calibration_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load all seven detector families for the fully-unseen COLING target.

    Trainable detectors use their existing per-source result structure.
    GPTZero reads saved COLING perplexities and applies the six thresholds
    from its source-result master. DetectGPT reads the v3 cache-only canonical
    rebuild. No detector inference or target recalibration is performed.
    """
    rows: List[pd.DataFrame] = []

    # Existing five fine-tuned/adaptation detectors: keep their schemas.
    for model_name, cfg in RQ4_CONFIG.items():
        subdir_candidates = [cfg["subdir"]]
        if model_name == "Text-Fluoroscopy":
            subdir_candidates = ["text_fluoroscopy", "text_fluroscopy"]
        base = next(
            (rq4_root / sd / cfg["results_dir"] for sd in subdir_candidates
             if (rq4_root / sd / cfg["results_dir"]).is_dir()),
            rq4_root / cfg["subdir"] / cfg["results_dir"],
        )
        for train_ds in DATASETS:
            for train_llm in LLMS:
                train_dir = cfg["train_dir"].format(ds=train_ds, llm=train_llm)
                path = base / train_dir / "tested_on_coling_hindi_data" / "predictions.csv"
                df = safe_read_csv(path)
                if df is None:
                    warnings.warn(f"[RQ4/{model_name}] missing: {path}")
                    continue
                required = {"text", cfg["label_col"], cfg["pred_col"]}
                missing = required - set(df.columns)
                if missing:
                    warnings.warn(f"[RQ4/{model_name}] missing columns {sorted(missing)} in {path}")
                    continue
                true_label = canonicalize_label(df[cfg["label_col"]], cfg["raw_ai_label"])
                predicted_label = canonicalize_label(df[cfg["pred_col"]], cfg["raw_ai_label"])
                if cfg["score_col"] is not None and cfg["score_col"] in df.columns:
                    score = canonicalize_score(df[cfg["score_col"]], cfg["score_direction"])
                else:
                    score = np.full(len(df), np.nan)
                out = pd.DataFrame({
                    "model": model_name,
                    "train_dataset": train_ds,
                    "train_llm": train_llm,
                    "test_dataset": RQ4_DATASET,
                    "test_llm": RQ4_TEST_LLM,
                    "text": df["text"].astype(str),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "score": score,
                    "setting": "other_unseen_both",
                    "rq": "RQ4",
                    "source_path": str(path),
                })
                rows.append(out)

    # Score-only detectors: use source-frozen GPTZero thresholds and the
    # v3 cache-rebuilt, canonical DetectGPT outputs. No score inference is run.
    zero_shot = load_zero_shot_rq4_all(calibration_root or rq4_root, rq4_root)
    if not zero_shot.empty:
        zero_shot["setting"] = "other_unseen_both"
        zero_shot["rq"] = "RQ4"
        rows.append(zero_shot)

    if not rows:
        raise RuntimeError(
            "No RQ4 prediction files were loaded. Check --rq4-root and source calibration files."
        )

    result = pd.concat(rows, ignore_index=True)
    result["text_key"] = result["text"].map(normalize_text_for_matching)
    return result


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


# ===========================================================================
# RQ5
# ===========================================================================

# Exact prompt-change schemas supplied by user.
#
# NOTE:
# Indic-BERT `predicted_prob` and RADAR `predicted_prob` are deliberately
# NOT used for AUC because their probability semantics were not established.
#
# XLM-RoBERTa:
#   raw 0 = AI
#   predicted_prob_label0 = P(AI)
#
# CONDA:
#   raw 0 = AI
#   prediction_probability_class1 = P(Human)
#
# Text-Fluoroscopy:
#   raw 1 = AI
#   predicted_prob_label1 = P(AI)
#
RQ5_CONFIG = {
    "XLM-RoBERTa": {
        "subdir": "xlm_roberta",
        "results_dir": "inference_results_new_prompt",
        "layout": "flat",
        "label_col": "true_label",
        "pred_col": "predicted_label",
        "raw_ai_label": 0,
        "score_col": "predicted_prob_label0",
        "score_direction": "ai_native",
    },

    "Indic-BERT": {
        "subdir": "indicBert",
        "results_dir": "inference_results",
        "layout": "flat",
        "label_col": "true_label",
        "pred_col": "predicted_label",
        "raw_ai_label": 0,
        "score_col": None,
        "score_direction": "none",
    },

    "RADAR": {
        "subdir": "RADAR",
        "results_dir": "inference_results",
        "layout": "flat",
        "label_col": "true_label",
        "pred_col": "predicted_label",
        "raw_ai_label": 0,
        "score_col": None,
        "score_direction": "none",
    },

    "CONDA": {
        "subdir": "CONDA",
        "results_dir": "inference_results",
        "layout": "nested",
        "label_col": "true_label",
        "pred_col": "predicted_label",
        "raw_ai_label": 0,
        "score_col": "prediction_probability_class1",
        "score_direction": "human_prob",
    },

    "Text-Fluoroscopy": {
        "subdir": "text_fluoroscopy",
        "results_dir": "inference_results_new_prompt",
        "layout": "flat",
        "label_col": "true_label",
        "pred_col": "predicted_label",
        "raw_ai_label": 1,
        "score_col": "predicted_prob_label1",
        "score_direction": "ai_native",
    },

    # Score-only detectors: `predicted_label` is deliberately not required.
    # The source-train Youden threshold is fitted from `calibration.csv` and
    # applied to the score in this new-prompt target file.
    "GPTZero": {
        "subdir": "gptzero",
        "results_dir": "inference_results_new_prompt",
        "layout": "flat",
        "zero_shot": True,
    },
    "DetectGPT": {
        "subdir": "detect_gpt",
        "results_dir": "inference_results_new_prompt",
        "layout": "flat",
        "zero_shot": True,
    },
}


def parse_prompt_dirname(
    dirname: str,
) -> Optional[Tuple[str, str, str, str]]:
    """
    Parse:
        hindisumm_gemini_tested_on_xquad_qwen3
    """
    marker = "_tested_on_"

    if marker not in dirname:
        return None

    left, right = dirname.split(
        marker,
        1,
    )

    left_parts = left.split("_")
    right_parts = right.split("_")

    if (
        len(left_parts) != 2
        or len(right_parts) != 2
    ):
        return None

    train_ds, train_llm = left_parts
    test_ds, test_llm = right_parts

    if train_ds not in DATASETS:
        return None

    if train_llm not in LLMS:
        return None

    if test_ds not in DATASETS:
        return None

    if test_llm not in LLMS:
        return None

    return (
        train_ds,
        train_llm,
        test_ds,
        test_llm,
    )


def load_rq5_new_prompt(
    rq5_root: Path,
    calibration_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load new-prompt predictions for all seven detectors.

    For GPTZero/DetectGPT, recalibration on new-prompt labels is forbidden:
    the threshold is fitted once on the original source training split and
    transferred to each new-prompt target file.
    """
    rows: List[pd.DataFrame] = []
    calibration_root = calibration_root or rq5_root

    for model_name, cfg in RQ5_CONFIG.items():
        # GPTZero and DetectGPT are loaded by dedicated source-aware loaders.
        # This avoids applying the old DetectGPT raw-label/score-direction
        # assumptions and reads the v3 cache-rebuilt results directly.
        if cfg.get("zero_shot", False):
            cell = load_zero_shot_rq5_all(calibration_root, rq5_root)
            if not cell.empty:
                cell["prompt"] = "new"
                cell["setting"] = cell.apply(
                    lambda r: prompt_setting(
                        r["train_dataset"], r["train_llm"],
                        r["test_dataset"], r["test_llm"]
                    ), axis=1
                )
                cell["rq"] = "RQ5"
                rows.append(cell)
            # Both zero-shot detectors are loaded together to avoid repeating
            # the same 12 file scans once for GPTZero and again for DetectGPT.
            break

        subdir_candidates = [cfg["subdir"]]
        if model_name == "Text-Fluoroscopy":
            subdir_candidates = ["text_fluoroscopy", "text_fluroscopy"]
        base = next(
            (rq5_root / sd / cfg["results_dir"] for sd in subdir_candidates
             if (rq5_root / sd / cfg["results_dir"]).is_dir()),
            rq5_root / cfg["subdir"] / cfg["results_dir"],
        )
        if not base.exists():
            warnings.warn(f"[RQ5/{model_name}] missing: {base}")
            continue

        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            parsed = parse_prompt_dirname(child.name)
            if parsed is None:
                continue
            train_ds, train_llm, test_ds, test_llm = parsed
            if train_llm not in RQ5_LLMS or test_llm not in RQ5_LLMS:
                continue
            prediction_path = child / "predictions.csv" if cfg["layout"] == "flat" else (
                child / f"tested_on_{test_ds}_{test_llm}" / "predictions.csv"
            )
            if not prediction_path.is_file():
                warnings.warn(f"[RQ5/{model_name}] missing: {prediction_path}")
                continue

            df = safe_read_csv(prediction_path)
            if df is None:
                continue
            required = {"text", cfg["label_col"], cfg["pred_col"]}
            missing = required - set(df.columns)
            if missing:
                warnings.warn(f"[RQ5/{model_name}] missing columns {sorted(missing)} in {prediction_path}")
                continue
            true_label = canonicalize_label(df[cfg["label_col"]], cfg["raw_ai_label"])
            predicted_label = canonicalize_label(df[cfg["pred_col"]], cfg["raw_ai_label"])
            if cfg["score_col"] is not None and cfg["score_col"] in df.columns:
                score = canonicalize_score(df[cfg["score_col"]], cfg["score_direction"])
            else:
                score = np.full(len(df), np.nan)
            out = pd.DataFrame({
                "model": model_name,
                "train_dataset": train_ds,
                "train_llm": train_llm,
                "test_dataset": test_ds,
                "test_llm": test_llm,
                "text": df["text"].astype(str),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "score": score,
                "prompt": "new",
                "setting": [prompt_setting(train_ds, train_llm, test_ds, test_llm)] * len(df),
                "rq": "RQ5",
                "source_path": str(prediction_path),
            })
            rows.append(out)

    if not rows:
        raise RuntimeError(
            "No RQ5 new-prompt files were loaded. Check --rq5-root and source calibration files."
        )
    result = pd.concat(rows, ignore_index=True)
    result["text_key"] = result["text"].map(normalize_text_for_matching)
    return result


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


def load_original_rq5_baseline(
    training_models_root: Path,
) -> pd.DataFrame:
    """Load original-prompt baseline for all seven detectors.

    Only Gemini/Qwen3 source and target LLMs are retained, matching RQ5's
    original scope. Zero-shot rows already contain source-fitted thresholds.
    """
    master = load_all(str(training_models_root), models=RQ45_MODELS)
    master = master[
        master["model"].isin(RQ45_MODELS)
        & master["train_llm"].isin(RQ5_LLMS)
        & master["test_llm"].isin(RQ5_LLMS)
        & master["train_dataset"].isin(DATASETS)
        & master["test_dataset"].isin(DATASETS)
    ].copy()
    return master


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


# ===========================================================================
# Findings
# ===========================================================================

def write_rq45_findings(
    outdir: Path,
    rq4_metrics_df: pd.DataFrame,
    rq4_cochran_df: pd.DataFrame,
    rq4_pairwise_df: pd.DataFrame,
    rq4_training_df: pd.DataFrame,
    rq4_friedman_df: pd.DataFrame,
    rq5_primary_df: pd.DataFrame,
    rq5_detector_df: pd.DataFrame,
    rq5_largest_df: pd.DataFrame,
) -> None:

    lines = [
        "# RQ4 and RQ5 Findings",
        "",
        "All labels use the canonical convention 1=AI and 0=Human.",
        "Scores, where available, are oriented so that higher means more AI-like.",
        "",
    ]

    # -----------------------------------------------------------------------
    # RQ4
    # -----------------------------------------------------------------------

    lines.append("## RQ4 — Dataset and LLM both unseen")
    lines.append("")

    if rq4_metrics_df.empty:
        lines.append(
            "- No RQ4 metrics were produced."
        )
    else:
        lines.append(
            f"- RQ4 contains {len(rq4_metrics_df)} "
            "detector/source-model evaluations on the unseen "
            "COLING Hindi test set."
        )

        # Best mean AI Recall.
        mean_ai = (
            rq4_metrics_df
            .groupby("model")["ai_recall"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        if not mean_ai.empty:
            best_model = mean_ai.index[0]

            lines.append(
                f"- Highest mean AI Recall across the six unseen-source "
                f"training configurations: {best_model} "
                f"({mean_ai.iloc[0]:.3f})."
            )

        # Lowest mean AI Recall.
        worst_model = mean_ai.index[-1]

        lines.append(
            f"- Lowest mean AI Recall across the six unseen-source "
            f"training configurations: {worst_model} "
            f"({mean_ai.iloc[-1]:.3f})."
        )

    # Cochran.
    if not rq4_cochran_df.empty:

        n_sig = int(
            (
                rq4_cochran_df["p_value"]
                < 0.05
            ).sum()
        )

        lines.append(
            f"- Detector error rates differed significantly in "
            f"{n_sig}/{len(rq4_cochran_df)} COLING evaluations according "
            f"to Cochran's Q test."
        )

    # McNemar.
    if not rq4_pairwise_df.empty:

        sig = rq4_pairwise_df[
            rq4_pairwise_df[
                "p_holm_global"
            ] < 0.05
        ]

        lines.append(
            f"- {len(sig)} of {len(rq4_pairwise_df)} "
            "paired detector comparisons remained significant after "
            "global Holm correction."
        )

        # Strongest pairwise direction summary.
        if not sig.empty:

            directional_rows = []

            for _, row in sig.iterrows():

                if row["favors_a"] > row["favors_b"]:
                    winner = row["model_a"]
                    loser = row["model_b"]
                    strength = (
                        row["favors_a"]
                        - row["favors_b"]
                    )
                else:
                    winner = row["model_b"]
                    loser = row["model_a"]
                    strength = (
                        row["favors_b"]
                        - row["favors_a"]
                    )

                directional_rows.append({
                    "winner": winner,
                    "loser": loser,
                    "strength": strength,
                })

            if directional_rows:

                directional_df = pd.DataFrame(
                    directional_rows
                )

                pair_counts = (
                    directional_df
                    .groupby(
                        ["winner", "loser"]
                    )
                    .size()
                    .sort_values(
                        ascending=False
                    )
                )

                if not pair_counts.empty:
                    (winner, loser), count = (
                        pair_counts.index[0],
                        pair_counts.iloc[0],
                    )

                    lines.append(
                        f"- The most frequent significant pairwise "
                        f"direction was {winner} over {loser}, occurring "
                        f"in {int(count)} COLING source configurations."
                    )

    # Training dataset.
    if not rq4_training_df.empty:

        sig = rq4_training_df[
            rq4_training_df[
                "p_holm_global"
            ] < 0.05
        ]

        lines.append(
            f"- HindiSumm-trained and XQuad-trained models differed "
            f"significantly on the same COLING examples in "
            f"{len(sig)}/{len(rq4_training_df)} "
            "detector/source-LLM comparisons after global Holm correction."
        )

    # Friedman.
    if not rq4_friedman_df.empty:

        for _, row in rq4_friedman_df.iterrows():

            if row["p_value"] < 0.05:
                lines.append(
                    f"- Detector AI Recall/accuracy differences across "
                    f"the six unseen-source configurations were significant "
                    f"for {row['metric']} "
                    f"(Friedman $\\chi^2$={row['chi2']:.3f}, "
                    f"p={row['p_value']:.4g})."
                )

    # -----------------------------------------------------------------------
    # RQ5
    # -----------------------------------------------------------------------

    lines.append("")
    lines.append("## RQ5 — Prompt-change robustness")
    lines.append("")

    if rq5_primary_df.empty:
        lines.append(
            "- No RQ5 primary comparisons were produced."
        )
    else:
        for _, row in rq5_primary_df.iterrows():

            if np.isnan(row["p_holm_global"]):
                continue

            direction = (
                "increased"
                if row["mean_change"] > 0
                else "decreased"
                if row["mean_change"] < 0
                else "did not change"
            )

            significance = (
                "significant"
                if row["p_holm_global"] < 0.05
                else "not significant"
            )

            lines.append(
                f"- Under {row['setting']}, changing the prompt "
                f"{direction} mean {row['metric']} by "
                f"{abs(row['mean_change']):.3f}; the difference was "
                f"{significance} after Holm correction "
                f"(p={row['p_holm_global']:.4g})."
            )

    # Largest concrete failures.
    if not rq5_largest_df.empty:

        lines.append("")
        lines.append(
            "### Largest prompt-change effects"
        )

        for _, row in (
            rq5_largest_df
            .head(10)
            .iterrows()
        ):

            lines.append(
                f"- {row['model']}, "
                f"{row['train_dataset']}-{row['train_llm']} "
                f"to {row['test_dataset']}-{row['test_llm']}, "
                f"{row['metric']}: "
                f"{row['original']:.3f} -> {row['new']:.3f} "
                f"(change={row['change']:+.3f})."
            )

    # Explicit AUC note.
    lines.extend([
        "",
        "### AUC handling",
        "",
        "- AUC is computed only when a valid continuous score with a known "
        "AI-score direction is available.",
        "- XLM-RoBERTa RQ4 predictions do not contain a score column, so "
        "row-level AUC is not fabricated from hard predictions.",
        "- RQ5 Indic-BERT and RADAR probability columns are not used for AUC "
        "because their score semantics were not established by the supplied "
        "schema.",
    ])

    with open(
        outdir / "findings.md",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(lines)
        )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="RQ4/RQ5 statistics pipeline"
    )

    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Existing training_models root containing the validated "
            "original-prompt RQ1-RQ3 predictions."
        ),
    )

    parser.add_argument(
        "--rq4-root",
        required=True,
        help=(
            "COLING_2025_MGT_Task1 directory."
        ),
    )

    parser.add_argument(
        "--rq5-root",
        required=True,
        help=(
            "prompt_change directory containing new-prompt predictions."
        ),
    )

    parser.add_argument(
        "--outdir",
        default="./mgt_stats_rq45_out",
    )

    parser.add_argument(
        "--detectgpt-rebuilt-root",
        default=None,
        help=(
            "v3 audit directory containing rebuilt_from_score_cache/{rq123,rq4} "
            "and rq5_repaired. If omitted, the newest complete directory is auto-discovered."
        ),
    )

    args = parser.parse_args()
    set_detectgpt_rebuilt_root(args.detectgpt_rebuilt_root)

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    root = Path(
        args.root
    )

    rq4_root = Path(
        args.rq4_root
    )

    rq5_root = Path(
        args.rq5_root
    )

    # ======================================================================
    # RQ4
    # ======================================================================

    print(
        "\n============================================================"
    )
    print(
        "RQ4: unseen dataset + unseen LLM"
    )
    print(
        "============================================================"
    )

    rq4 = load_rq4_coling(
        rq4_root,
        calibration_root=root,
    )

    rq4.to_csv(
        outdir / "rq4_master_predictions.csv",
        index=False,
    )
    if "calibration_threshold" in rq4.columns:
        rq4.loc[rq4["model"].isin(ZERO_SHOT_SPECS)].drop_duplicates(
            subset=["model", "train_dataset", "train_llm"]
        ).to_csv(outdir / "rq4_zero_shot_threshold_audit.csv", index=False)

    print(
        f"Loaded {len(rq4)} RQ4 rows."
    )

    rq4_metrics_df = rq4_metrics(
        rq4,
        outdir,
    )

    rq4_cochran_df, rq4_pairwise_df = (
        rq4_detector_comparison(
            rq4,
            outdir,
        )
    )

    rq4_training_df = (
        rq4_training_dataset_comparison(
            rq4,
            outdir,
        )
    )

    rq4_friedman_df = rq4_friedman(
        rq4_metrics_df,
        outdir,
    )

    make_rq4_plots(
        rq4_metrics_df,
        outdir,
    )

    # ======================================================================
    # RQ5 ORIGINAL PROMPT BASELINE
    # ======================================================================

    print(
        "\n============================================================"
    )
    print(
        "RQ5: loading ORIGINAL prompt baseline"
    )
    print(
        "============================================================"
    )

    original = load_original_rq5_baseline(
        root
    )

    print(
        f"Loaded {len(original)} original-prompt rows."
    )

    original.to_csv(
        outdir / "rq5_original_master.csv",
        index=False,
    )

    original_metrics = compute_cell_metric_table(
        original,
        prompt_name="original",
    )

    original_metrics.to_csv(
        outdir / "rq5_original_prompt_metrics.csv",
        index=False,
    )

    # ======================================================================
    # RQ5 NEW PROMPT
    # ======================================================================

    print(
        "\n============================================================"
    )
    print(
        "RQ5: loading NEW prompt predictions"
    )
    print(
        "============================================================"
    )

    new_prompt = load_rq5_new_prompt(
        rq5_root,
        calibration_root=root,
    )

    print(
        f"Loaded {len(new_prompt)} new-prompt rows."
    )

    new_prompt.to_csv(
        outdir / "rq5_new_prompt_master.csv",
        index=False,
    )
    if "calibration_threshold" in new_prompt.columns:
        new_prompt.loc[new_prompt["model"].isin(ZERO_SHOT_SPECS)].drop_duplicates(
            subset=["model", "train_dataset", "train_llm"]
        ).to_csv(outdir / "rq5_zero_shot_threshold_audit.csv", index=False)

    new_metrics = compute_cell_metric_table(
        new_prompt,
        prompt_name="new",
    )

    new_metrics.to_csv(
        outdir / "rq5_new_prompt_metrics.csv",
        index=False,
    )

    # ======================================================================
    # RQ5 matched cell comparison
    # ======================================================================

    merged = match_rq5_original_and_new(
        original_metrics,
        new_metrics,
    )

    merged.to_csv(
        outdir / "rq5_original_vs_new_prompt.csv",
        index=False,
    )

    print(
        f"Matched {len(merged)} original/new prompt experimental cells."
    )

    rq5_primary_df = (
        rq5_prompt_change_wilcoxon(
            merged,
            outdir,
        )
    )

    rq5_detector_df = (
        rq5_per_detector_wilcoxon(
            merged,
            outdir,
        )
    )

    rq5_largest_df = (
        rq5_largest_changes(
            merged,
            outdir,
        )
    )

    make_rq5_plots(
        merged,
        outdir,
    )

    # ======================================================================
    # Findings
    # ======================================================================

    write_rq45_findings(
        outdir=outdir,
        rq4_metrics_df=rq4_metrics_df,
        rq4_cochran_df=rq4_cochran_df,
        rq4_pairwise_df=rq4_pairwise_df,
        rq4_training_df=rq4_training_df,
        rq4_friedman_df=rq4_friedman_df,
        rq5_primary_df=rq5_primary_df,
        rq5_detector_df=rq5_detector_df,
        rq5_largest_df=rq5_largest_df,
    )

    print(
        "\n============================================================"
    )
    print(
        "DONE"
    )
    print(
        f"Outputs written to: {outdir.resolve()}"
    )
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()