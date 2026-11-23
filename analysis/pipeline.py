"""Statistical analysis for the Hindi MGT study."""
import argparse

import os

import warnings

import itertools

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_auc_score, recall_score, f1_score, accuracy_score

from matching import add_text_key, per_cell_audit, safe_paired_matrix

from stats_tests import (
    mcnemar_pairwise, holm_correction, cochrans_q_test, bootstrap_metric_ci, error_overlap,
    levene_stability_test, friedman_across_detectors, wilcoxon_paired,
    cluster_logit, regression_is_unstable, cluster_bootstrap_logit,
)

from study import DATASETS, LLMS, TRANSFER_CAPABLE_MODELS

def ai_recall(y_true, y_pred):
    positive = np.asarray(y_true) == 1
    return float(np.mean(np.asarray(y_pred)[positive] == 1)) if positive.any() else 0.0

def human_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)

def cell_metrics(df):
    y_true, y_pred = df["true_label"].values, df["predicted_label"].values
    out = dict(
        n=len(df),
        accuracy=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, zero_division=0),
        ai_recall=ai_recall(y_true, y_pred),
        human_recall=human_recall(y_true, y_pred),
    )
    # AUC: use only rows with an actual score (never impute a missing score
    # with the column median — that fabricates information and biases AUC
    # toward 0.5). If fewer than 2 examples of either class have a real
    # score, or no score exists at all, report NaN rather than a computed
    # number that doesn't mean what it looks like it means.
    scored = df.dropna(subset=["score"])
    if len(scored) >= 4 and scored["true_label"].nunique() == 2:
        try:
            out["auc_score_col"] = roc_auc_score(scored["true_label"].values, scored["score"].values)
            out["auc_n"] = len(scored)
        except Exception:
            out["auc_score_col"] = np.nan
            out["auc_n"] = len(scored)
    else:
        out["auc_score_col"] = np.nan
        out["auc_n"] = len(scored)
    return out

def run_rq1(master, outdir):
    print("\n=== RQ1: in-domain performance ===")
    sub = master[master["setting"].isin(["in_domain", "zero_shot"])].copy()
    rows = []
    for (ds, llm), grp in sub.groupby(["test_dataset", "test_llm"]):
        for model, mgrp in grp.groupby("model"):
            m = cell_metrics(mgrp)
            m.update(dataset=ds, llm=llm, model=model)
            mean, lo, hi = bootstrap_metric_ci(
                mgrp["true_label"].values, mgrp["predicted_label"].values, ai_recall
            )
            m.update(ai_recall_boot_mean=mean, ai_recall_ci_lo=lo, ai_recall_ci_hi=hi)
            rows.append(m)
    rq1_df = pd.DataFrame(rows)
    rq1_df.to_csv(os.path.join(outdir, "rq1_in_domain_metrics.csv"), index=False)

    # Pairwise McNemar + Cochran's Q per test cell, using the audited
    # conflict-safe pairing from matching.safe_paired_matrix instead of a
    # bare pivot_table(aggfunc="first").
    mcnemar_rows, cochran_rows, cell_audit_rows = [], [], []
    for (ds, llm), grp in sub.groupby(["test_dataset", "test_llm"]):
        pivot, truth, audit = safe_paired_matrix(grp, value_col="predicted_label")
        audit.update(dataset=ds, llm=llm)
        cell_audit_rows.append(audit)
        if pivot.empty or pivot.shape[1] < 2 or pivot.shape[0] < 5:
            continue
        correctness = pivot.eq(truth, axis=0).astype(int)

        Q, df_, p_q = cochrans_q_test(correctness.values)
        cochran_rows.append(dict(dataset=ds, llm=llm, n_items=len(pivot),
                                  n_detectors=pivot.shape[1], Q=Q, df=df_, p_value=p_q))

        for a, b in itertools.combinations(pivot.columns, 2):
            p, table, detail = mcnemar_pairwise(truth.values, pivot[a].values, pivot[b].values)
            row = dict(dataset=ds, llm=llm, model_a=a, model_b=b,
                       p_value=p, n_items=len(pivot))
            row.update(detail)
            mcnemar_rows.append(row)

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    cochran_df = pd.DataFrame(cochran_rows)
    if not mcnemar_df.empty:
        # Holm correction WITHIN each (dataset, llm) cell's family of
        # comparisons, and separately across the FULL set of comparisons —
        # report both, since "significant after correcting within this
        # cell" and "significant after correcting across everything we
        # tested" are both defensible framings and give different counts.
        mcnemar_df["p_holm_within_cell"] = np.nan
        for (ds, llm), idx in mcnemar_df.groupby(["dataset", "llm"]).groups.items():
            mcnemar_df.loc[idx, "p_holm_within_cell"] = holm_correction(mcnemar_df.loc[idx, "p_value"].values)
        mcnemar_df["p_holm_global"] = holm_correction(mcnemar_df["p_value"].values)
    mcnemar_df.to_csv(os.path.join(outdir, "rq1_mcnemar_pairwise.csv"), index=False)
    cochran_df.to_csv(os.path.join(outdir, "rq1_cochrans_q.csv"), index=False)
    pairing_audit_df = pd.DataFrame(cell_audit_rows)
    pairing_audit_df.to_csv(os.path.join(outdir, "rq1_pairing_audit.csv"), index=False)

    if not rq1_df.empty:
        g = sns.catplot(
            data=rq1_df, x="llm", y="ai_recall_boot_mean", hue="model",
            col="dataset", kind="bar", height=4.5, aspect=1.1, errorbar=None,
        )
        g.set_axis_labels("Generating LLM (in-domain test)", "AI Recall (bootstrap mean)")
        g.savefig(os.path.join(outdir, "fig_rq1_ai_recall_by_model.png"), dpi=150)
        plt.close("all")

    return rq1_df, mcnemar_df, cochran_df, pairing_audit_df

def run_rq2(master, outdir):
    print("\n=== RQ2: LLM cross-domain transfer ===")
    sub = master[master["setting"] == "llm_cross_domain"].copy()
    sub = sub[sub["model"].isin(TRANSFER_CAPABLE_MODELS)]
    if sub.empty:
        warnings.warn("No llm_cross_domain rows found; skipping RQ2.")
        return pd.DataFrame(), {}

    rows = []
    for (model, ds, tr_llm, te_llm), grp in sub.groupby(
        ["model", "train_dataset", "train_llm", "test_llm"]
    ):
        m = cell_metrics(grp)
        m.update(model=model, dataset=ds, train_llm=tr_llm, test_llm=te_llm)
        rows.append(m)
    rq2_df = pd.DataFrame(rows)
    rq2_df.to_csv(os.path.join(outdir, "rq2_llm_cross_domain_metrics.csv"), index=False)

    findings = {}

    gemini_target = rq2_df[rq2_df["test_llm"] == "gemini"].set_index(["model", "dataset", "train_llm"])["ai_recall"]
    other_target = (
        rq2_df[rq2_df["test_llm"] != "gemini"]
        .groupby(["model", "dataset", "train_llm"])["ai_recall"].mean()
    )
    common_idx = gemini_target.index.intersection(other_target.index)
    if len(common_idx) >= 2:
        stat, p = wilcoxon_paired(gemini_target.loc[common_idx].values, other_target.loc[common_idx].values)
        findings["gemini_target_wilcoxon"] = dict(
            n_pairs=len(common_idx),
            mean_gemini_target_air=float(gemini_target.loc[common_idx].mean()),
            mean_other_target_air=float(other_target.loc[common_idx].mean()),
            wilcoxon_stat=stat, p_value=p,
        )

    stability_rows = []
    detector_groups, detector_names = [], []
    for model, grp in rq2_df.groupby("model"):
        vals = grp["ai_recall"].values
        detector_groups.append(vals)
        detector_names.append(model)
        stability_rows.append(dict(model=model, n_cells=len(vals),
                                    mean_ai_recall=float(np.mean(vals)),
                                    std_ai_recall=float(np.std(vals)),
                                    range_ai_recall=float(np.max(vals) - np.min(vals)),
                                    cv=float(np.std(vals) / np.mean(vals)) if np.mean(vals) else np.nan))
    stability_df = pd.DataFrame(stability_rows).sort_values("cv")
    stability_df.to_csv(os.path.join(outdir, "rq2_detector_stability.csv"), index=False)
    lev_stat, lev_p = levene_stability_test(detector_groups)
    findings["levene_variance_across_detectors"] = dict(statistic=lev_stat, p_value=lev_p,
                                                          detectors=detector_names)

    block_pivot = rq2_df.pivot_table(index=["dataset", "train_llm", "test_llm"],
                                      columns="model", values="ai_recall")
    block_pivot_complete = block_pivot.dropna()
    if block_pivot_complete.shape[0] >= 2 and block_pivot_complete.shape[1] >= 2:
        fstat, fp = friedman_across_detectors(block_pivot_complete.values)
        findings["friedman_detectors_llm_cross"] = dict(
            statistic=fstat, p_value=fp, n_blocks=block_pivot_complete.shape[0],
            n_detectors=block_pivot_complete.shape[1],
        )

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=rq2_df, x="model", y="ai_recall")
    sns.stripplot(data=rq2_df, x="model", y="ai_recall", color="black", alpha=0.5, size=4)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("AI Recall (each point = one train_llm -> test_llm transfer cell)")
    plt.title("RQ2: AI-recall spread under LLM cross-domain transfer")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_rq2_stability_boxplot.png"), dpi=150)
    plt.close("all")

    plt.figure(figsize=(7, 5))
    heat = rq2_df.groupby(["train_llm", "test_llm"])["ai_recall"].mean().unstack()
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("RQ2: mean AI recall by (train LLM -> test LLM), averaged over detectors/datasets")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_rq2_llm_transfer_heatmap.png"), dpi=150)
    plt.close("all")

    return rq2_df, findings

def run_rq3(master, outdir):
    print("\n=== RQ3: dataset cross-domain transfer ===")
    sub = master[master["setting"] == "dataset_cross_domain"].copy()
    sub = sub[sub["model"].isin(TRANSFER_CAPABLE_MODELS)]
    if sub.empty:
        warnings.warn("No dataset_cross_domain rows found; skipping RQ3.")
        return pd.DataFrame(), {}

    rows = []
    for (model, tr_ds, te_ds, llm), grp in sub.groupby(
        ["model", "train_dataset", "test_dataset", "test_llm"]
    ):
        m = cell_metrics(grp)
        m.update(model=model, train_dataset=tr_ds, test_dataset=te_ds, llm=llm)
        rows.append(m)
    rq3_df = pd.DataFrame(rows)
    rq3_df.to_csv(os.path.join(outdir, "rq3_dataset_cross_domain_metrics.csv"), index=False)

    findings = {}

    in_domain_hr = master[master["setting"].isin(["in_domain"])]
    id_hr_by_cell = (
        in_domain_hr.groupby(["model", "test_dataset", "test_llm"])
        .apply(lambda g: human_recall(g["true_label"], g["predicted_label"]))
        .rename("in_domain_human_recall")
    )
    shifted_hr = rq3_df.set_index(["model", "test_dataset", "llm"])["human_recall"]
    common = id_hr_by_cell.index.intersection(shifted_hr.index)
    if len(common) >= 2:
        stat, p = wilcoxon_paired(id_hr_by_cell.loc[common].values, shifted_hr.loc[common].values)
        findings["human_recall_collapse_wilcoxon"] = dict(
            n_pairs=len(common),
            mean_in_domain_hr=float(id_hr_by_cell.loc[common].mean()),
            mean_dataset_shift_hr=float(shifted_hr.loc[common].mean()),
            wilcoxon_stat=stat, p_value=p,
        )

    groups, names = [], []
    for model, grp in rq3_df.groupby("model"):
        groups.append(grp["human_recall"].values)
        names.append(model)
    lev_stat, lev_p = levene_stability_test(groups)
    findings["levene_human_recall_variance_by_model"] = dict(statistic=lev_stat, p_value=lev_p, detectors=names)

    # Report range for EVERY detector on an equal footing (not "CONDA vs the
    # rest") so the comparison can't be read as pre-endorsing CONDA.
    range_by_model = rq3_df.groupby("model")["human_recall"].agg(lambda s: float(s.max() - s.min()))
    mean_by_model = rq3_df.groupby("model")["human_recall"].mean()
    findings["human_recall_range_by_model"] = dict(
        range=range_by_model.to_dict(), mean=mean_by_model.to_dict(),
        most_stable_by_range=range_by_model.idxmin() if not range_by_model.empty else None,
    )

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=rq3_df, x="model", y="human_recall")
    sns.stripplot(data=rq3_df, x="model", y="human_recall", color="black", alpha=0.5, size=4)
    plt.xticks(rotation=30, ha="right")
    plt.title("RQ3: Human-recall spread under dataset cross-domain shift")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_rq3_human_recall_boxplot.png"), dpi=150)
    plt.close("all")

    return rq3_df, findings

def run_error_overlap(master, outdir):
    print("\n=== Error overlap across detectors (in-domain cells) ===")
    sub = master[master["setting"].isin(["in_domain", "zero_shot"])].copy()
    overlap_rows = []
    hard_examples_all = []
    for (ds, llm), grp in sub.groupby(["test_dataset", "test_llm"]):
        pivot, truth, audit = safe_paired_matrix(grp, value_col="predicted_label")
        if pivot.empty or pivot.shape[1] < 2:
            continue

        for a, b in itertools.combinations(pivot.columns, 2):
            jac, kappa = error_overlap(truth.values, pivot[a].values, pivot[b].values)
            overlap_rows.append(dict(dataset=ds, llm=llm, model_a=a, model_b=b,
                                      jaccard=jac, cohen_kappa=kappa))

        wrong_frac = (pivot.ne(truth, axis=0)).mean(axis=1)
        hard = wrong_frac[wrong_frac >= 0.6].sort_values(ascending=False)
        # attach one representative raw text per text_key for readability
        key_to_text = grp.drop_duplicates("text_key").set_index("text_key")["text"]
        for text_key, frac in hard.items():
            wrong_models = pivot.columns[pivot.loc[text_key].ne(truth.loc[text_key])].tolist()
            hard_examples_all.append(dict(
                dataset=ds, llm=llm, text=key_to_text.get(text_key, text_key),
                frac_detectors_wrong=frac, n_detectors=pivot.shape[1],
                detectors_wrong=";".join(wrong_models),
                true_label=int(truth.loc[text_key]),
            ))

    overlap_df = pd.DataFrame(overlap_rows)
    hard_df = pd.DataFrame(hard_examples_all)
    overlap_df.to_csv(os.path.join(outdir, "error_overlap_pairwise.csv"), index=False)
    hard_df.to_csv(os.path.join(outdir, "hard_examples_majority_wrong.csv"), index=False)

    if not overlap_df.empty:
        heat = overlap_df.groupby(["model_a", "model_b"])["jaccard"].mean().unstack()
        plt.figure(figsize=(7, 6))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Mean Jaccard overlap of error sets (higher = shared, text-intrinsic errors)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_error_overlap_jaccard.png"), dpi=150)
        plt.close("all")

    return overlap_df, hard_df

FEATURE_COLS_Z = ["ASW_z", "ASL_z", "AWL_z", "PSW30_z", "JUK_z", "n_words_z"]

def _prep_regression_frame(master):
    df = master.dropna(subset=["ASW", "ASL", "AWL", "PSW30", "JUK", "n_words"]).copy()
    df = df[df["setting"].isin(["in_domain", "llm_cross_domain", "dataset_cross_domain", "zero_shot"])]
    for c in ["ASW", "ASL", "AWL", "PSW30", "JUK", "n_words"]:
        df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std(ddof=0)
    df["text_id"] = df["text_key"].astype("category").cat.codes
    return df

def _fit_one_outcome(df, subset_mask, outcome_col, outdir, tag):
    """Fit `outcome ~ features + n_words_z + C(model) + C(setting)`,
    cluster-robust by text_id. If the standard MLE shows separation
    symptoms, fall back to the L2-regularized cluster-bootstrap estimator
    and say so explicitly in the summary file rather than silently
    reporting whichever one ran."""
    sub = df[subset_mask].copy()
    if sub.empty or sub[outcome_col].nunique() < 2:
        return None, f"[{tag}] insufficient data/variation — skipped.\n"

    formula = (
        f"{outcome_col} ~ ASW_z + ASL_z + AWL_z + PSW30_z + JUK_z "
        f"+ n_words_z + C(model) + C(setting)"
    )
    res = cluster_logit(sub, formula, cluster_col="text_id")
    unstable = regression_is_unstable(res)

    summary_lines = [f"=== {tag} : {formula} ===\n"]
    if not unstable:
        summary_lines.append(str(res.summary()))
        coef_df = pd.DataFrame({
            "model_type": "cluster_robust_glm",
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.bse.values,
            "p_value": res.pvalues.values,
        })
    else:
        summary_lines.append(
            "Standard cluster-robust GLM showed separation symptoms "
            "(non-finite or implausibly large SE/coefficients) and its "
            "p-values are NOT reported. Falling back to an L2-regularized "
            "logistic regression with cluster-bootstrap confidence "
            "intervals (500 resamples of whole text clusters).\n"
        )
        boot_df = cluster_bootstrap_logit(
            sub, FEATURE_COLS_Z, outcome_col, cluster_col="text_id",
        )
        if boot_df.empty:
            summary_lines.append("Bootstrap fallback also failed (insufficient data).\n")
            return None, "\n".join(summary_lines)
        summary_lines.append(boot_df.to_string(index=False))
        coef_df = boot_df.rename(columns={"feature": "term"}).copy()
        coef_df["model_type"] = "l2_cluster_bootstrap"
        coef_df["p_value"] = np.nan  # deliberately not reported; use frac_boot_same_sign instead

    return coef_df, "\n".join(summary_lines)

def run_regression(master, outdir):
    print("\n=== Linguistic-feature regression: AI false-negatives / Human false-positives ===")
    df = _prep_regression_frame(master)
    if df.empty:
        warnings.warn("Not enough data/variation for regression; skipping.")
        return None

    # AI false negatives: among true-AI examples, what predicts being missed
    # (predicted Human)? true_label==1 canonical AI.
    df["ai_missed"] = ((df["true_label"] == 1) & (df["predicted_label"] == 0)).astype(int)
    # Human false positives: among true-Human examples, what predicts being
    # falsely flagged as AI? true_label==0 canonical Human.
    df["human_flagged"] = ((df["true_label"] == 0) & (df["predicted_label"] == 1)).astype(int)

    coef_fn, summary_fn = _fit_one_outcome(
        df, df["true_label"] == 1, "ai_missed", outdir, "AI false-negative model"
    )
    coef_fp, summary_fp = _fit_one_outcome(
        df, df["true_label"] == 0, "human_flagged", outdir, "Human false-positive model"
    )

    with open(os.path.join(outdir, "regression_summary.txt"), "w") as f:
        f.write(summary_fn + "\n\n" + summary_fp)

    parts = []
    if coef_fn is not None:
        coef_fn = coef_fn.copy()
        coef_fn["outcome"] = "ai_missed (AI text predicted as Human)"
        parts.append(coef_fn)
    if coef_fp is not None:
        coef_fp = coef_fp.copy()
        coef_fp["outcome"] = "human_flagged (Human text predicted as AI)"
        parts.append(coef_fp)
    if not parts:
        return None
    coef_df = pd.concat(parts, ignore_index=True)
    coef_df.to_csv(os.path.join(outdir, "regression_coefficients.csv"), index=False)
    return coef_df
