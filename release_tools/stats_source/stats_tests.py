"""
stats_tests.py
--------------
All the "is this difference real" machinery used by pipeline.py:

  - mcnemar_pairwise        : paired detector-vs-detector comparison on same test set,
                               now also returns the discordant counts (b, c) and direction
  - holm_correction          : Holm-Bonferroni step-down adjustment for a family of p-values
  - cochrans_q_test          : k-detector joint comparison on same test set
  - bootstrap_metric_ci      : bootstrap CI for any sklearn-style metric
  - error_overlap             : Jaccard / Cohen's kappa between two detectors' error sets
  - levene_stability_test     : are detectors' recalls equally variable across settings?
  - friedman_across_detectors : repeated-measures test blocked by test cell
  - wilcoxon_paired            : paired non-parametric test (e.g. Gemini-target vs rest)
  - cluster_logit               : logistic regression with cluster-robust SE
  - regression_is_unstable      : flags quasi/perfect-separation symptoms in a fitted GLM
  - cluster_bootstrap_logit     : L2-regularized logit + cluster bootstrap CIs, for use when
                                   cluster_logit's asymptotic SEs are degenerate (separation)
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf


def mcnemar_pairwise(y_true, pred_a, pred_b):
    """Returns (p_value, table, detail) where detail has the discordant
    counts and which model each discordant pair favors — "77/126 significant"
    without this is not reportable; the direction and magnitude are what
    make a McNemar result meaningful (see README)."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    table = pd.crosstab(correct_a, correct_b)
    table = table.reindex(index=[True, False], columns=[True, False], fill_value=0)
    vals = table.values
    # vals[0,1] = A correct, B wrong (discordant favoring A)
    # vals[1,0] = A wrong, B correct (discordant favoring B)
    favors_a = int(vals[0, 1])
    favors_b = int(vals[1, 0])
    use_exact = vals.min() < 25
    try:
        res = mcnemar(vals, exact=use_exact, correction=True)
        p = float(res.pvalue)
    except Exception:
        p = np.nan
    detail = dict(favors_a=favors_a, favors_b=favors_b,
                   n_discordant=favors_a + favors_b, method="exact" if use_exact else "asymptotic")
    return p, vals, detail


def holm_correction(pvalues):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    SAME order as the input (not sorted), each capped at 1.0 and enforced
    monotonic non-decreasing along the sorted order (standard Holm step-up
    enforcement so adjusted p-values never decrease as raw p increases).
    Use this instead of reporting raw p<0.05 across a family of pairwise
    tests (e.g. all McNemar comparisons within one RQ1 cell, or across all
    cells) — with ~126 pairwise tests, an uncorrected count of "77
    significant" is inflated by multiple comparisons."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n)
    prev = 0.0
    for i, pv in enumerate(ranked):
        val = min((n - i) * pv, 1.0)
        val = max(val, prev)  # enforce monotonicity
        adj[i] = val
        prev = val
    out = np.empty(n)
    out[order] = adj
    return out


def cochrans_q_test(correctness_matrix: np.ndarray):
    """correctness_matrix: rows=items, cols=detectors, binary (1=correct)."""
    X = np.asarray(correctness_matrix, dtype=float)
    N, k = X.shape
    if k < 2 or N < 2:
        return np.nan, np.nan, np.nan
    col_sums = X.sum(axis=0)
    row_sums = X.sum(axis=1)
    T = X.sum()
    denom = (k * T - np.sum(row_sums ** 2))
    if denom == 0:
        return np.nan, k - 1, np.nan
    Q = (k - 1) * (k * np.sum(col_sums ** 2) - T ** 2) / denom
    df = k - 1
    p = 1 - stats.chi2.cdf(Q, df)
    return float(Q), df, float(p)


def bootstrap_metric_ci(y_true, y_pred_or_score, metric_fn, n_boot=2000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred_or_score = np.asarray(y_pred_or_score)
    n = len(y_true)
    if n == 0:
        return np.nan, np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred_or_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            boots.append(metric_fn(yt, yp))
        except Exception:
            continue
    if not boots:
        return np.nan, np.nan, np.nan
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(np.mean(boots)), float(lo), float(hi)


def error_overlap(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    err_a = set(np.where(pred_a != y_true)[0].tolist())
    err_b = set(np.where(pred_b != y_true)[0].tolist())
    union = err_a | err_b
    jaccard = len(err_a & err_b) / len(union) if union else np.nan
    try:
        kappa = cohen_kappa_score((pred_a == y_true).astype(int), (pred_b == y_true).astype(int))
    except Exception:
        kappa = np.nan
    return jaccard, kappa


def levene_stability_test(groups: list):
    groups = [np.asarray(g) for g in groups if len(g) > 0]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return np.nan, np.nan
    stat, p = stats.levene(*groups)
    return float(stat), float(p)


def friedman_across_detectors(block_matrix: np.ndarray):
    """block_matrix: rows = blocks (test cells), cols = detectors, values = a metric (e.g. AI recall).
    All rows must be complete (no NaN) - caller should filter."""
    X = np.asarray(block_matrix, dtype=float)
    if X.shape[0] < 2 or X.shape[1] < 3 or np.isnan(X).any():
        return np.nan, np.nan
    stat, p = stats.friedmanchisquare(*[X[:, j] for j in range(X.shape[1])])
    return float(stat), float(p)


def wilcoxon_paired(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.allclose(a, b):
        return np.nan, np.nan
    try:
        stat, p = stats.wilcoxon(a, b)
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def cluster_logit(df, formula, cluster_col):
    """Logistic regression (outcome ~ features) with cluster-robust SEs.
    Returns the fitted statsmodels result object (or None on failure). This
    is the asymptotic-MLE estimator — check `regression_is_unstable()`
    before trusting its p-values, since quasi/perfect separation (near-
    ceiling predictors like RADAR/XLM-R correctness) can blow up SEs even
    when the fit "succeeds" without raising."""
    try:
        model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
        res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
        return res
    except Exception as e:
        print(f"[cluster_logit] failed: {e}")
        return None


def regression_is_unstable(res, se_threshold=25.0, coef_threshold=15.0):
    """Symptom check for quasi/perfect separation in a fitted GLM: any
    standard error non-finite or absurdly large, or any coefficient
    magnitude implausibly large for a z-scored-predictor logistic model.
    This is exactly the failure mode in the original regression output
    (SE columns of 1e13-1e22, p-values collapsing to ~1, pseudo-R^2 far
    below what the huge coefficients would imply) — do not report p-values
    from a fit that trips this check; use cluster_bootstrap_logit instead."""
    if res is None:
        return True
    bse = np.asarray(res.bse, dtype=float)
    coef = np.asarray(res.params, dtype=float)
    if not np.all(np.isfinite(bse)) or not np.all(np.isfinite(coef)):
        return True
    if np.nanmax(bse) > se_threshold:
        return True
    if np.nanmax(np.abs(coef)) > coef_threshold:
        return True
    return False




def cluster_bootstrap_logit(df, feature_cols, outcome_col, cluster_col,
                             n_boot=500, seed=42, C=1.0):
    """Separation-robust alternative to cluster_logit: an L2-regularized
    logistic regression (sklearn, standardized features) as the point
    estimate, with cluster-bootstrap (resampling whole text clusters, not
    individual rows, matching cluster_logit's clustering unit) confidence
    intervals instead of asymptotic standard errors. Regularization keeps
    coefficients finite under quasi-separation; the cluster bootstrap keeps
    the CI honest about the fact that observations within one text (e.g.
    repeated across settings) aren't independent.

    Returns a DataFrame: feature, coef (point estimate), boot_mean,
    ci_lo, ci_hi (2.5/97.5 percentile), frac_boot_same_sign (stability of
    direction across bootstrap resamples — treat this as the practical
    substitute for a p-value under separation: consistently signed across
    >=95% of resamples is reasonable evidence of a real effect; anything
    close to 50/50 is not).
    """
    work = df.dropna(subset=feature_cols + [outcome_col, cluster_col]).copy()
    if work.empty or work[outcome_col].nunique() < 2:
        return pd.DataFrame()

    X_full = work[feature_cols].values
    y_full = work[outcome_col].values.astype(int)
    scaler = StandardScaler().fit(X_full)

    def _fit(X, y):
        Xs = scaler.transform(X)
        clf = LogisticRegression(C=C, max_iter=2000)  # penalty="l2" is the sklearn default
        clf.fit(Xs, y)
        return clf.coef_.ravel()

    point = _fit(X_full, y_full)

    clusters = work[cluster_col].unique()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        boot_df = work[work[cluster_col].isin(sampled_clusters)]
        # sampling clusters with replacement but selecting rows by isin()
        # under-duplicates repeated clusters; correct via explicit concat
        boot_parts = [work[work[cluster_col] == c] for c in sampled_clusters]
        boot_df = pd.concat(boot_parts, ignore_index=True)
        if boot_df[outcome_col].nunique() < 2:
            continue
        try:
            boots.append(_fit(boot_df[feature_cols].values, boot_df[outcome_col].values.astype(int)))
        except Exception:
            continue

    if not boots:
        return pd.DataFrame({
            "feature": feature_cols, "coef": point,
            "boot_mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "frac_boot_same_sign": np.nan,
        })

    boots = np.vstack(boots)
    ci_lo = np.percentile(boots, 2.5, axis=0)
    ci_hi = np.percentile(boots, 97.5, axis=0)
    boot_mean = boots.mean(axis=0)
    same_sign = (np.sign(boots) == np.sign(point)).mean(axis=0)

    return pd.DataFrame({
        "feature": feature_cols,
        "coef": point,
        "boot_mean": boot_mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "frac_boot_same_sign": same_sign,
        "n_boot_successful": len(boots),
    })
