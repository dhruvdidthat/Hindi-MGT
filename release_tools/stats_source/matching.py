"""
matching.py
-----------
Cross-detector text matching and data-integrity auditing.

Why this exists
----------------
Per the detector source-code audit (see README "Corrections applied"), the
seven detectors do NOT preprocess text identically before saving it to
predictions.csv:

  - XLM-RoBERTa        : raw test text, unmodified.
  - Indic-BERT         : strips U+200B (zero-width space) and converts
                          Devanagari digits to ASCII before inference; the
                          SAVED text reflects this transform.
  - RADAR               : same transform as Indic-BERT.
  - CONDA               : calls a `PreProcess.fit()` that currently returns
                          the text unchanged, so CONDA's saved text is
                          whatever was fed in (no guaranteed extra
                          normalization on top of that).
  - DetectGPT            : computes its score on a truncated/newline-stripped
                          copy internally but saves the ORIGINAL text.
  - GPTZero               : same — truncates internally for scoring, saves
                          the original text.
  - Text-Fluoroscopy      : uses the test text directly, saves that text.

Joining seven CSVs on raw `text` equality therefore silently drops rows
whenever one detector's copy has a zero-width space, a Devanagari-vs-ASCII
digit, or incidental whitespace that another detector's copy doesn't. This
module normalizes a join KEY (never the text used for linguistic-feature
extraction, which must stay tied to what that specific detector actually
scored) and provides audited, conflict-safe pairing instead of a bare
`pivot_table(..., aggfunc="first")`, which would silently pick an arbitrary
row when detectors disagree about a text's label.
"""
import re
import unicodedata
import numpy as np
import pandas as pd

_ZW_CHARS = "\u200b\u200c\u200d\ufeff"  # zero-width space/non-joiner/joiner/BOM
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_WS_RE = re.compile(r"\s+")


def normalize_text_for_matching(text) -> str:
    """Canonicalize a text string into a join key that is robust to the
    known preprocessing differences between detectors (zero-width chars,
    Devanagari vs ASCII digits, incidental whitespace). This key is used
    ONLY for pairing rows across detectors — never for linguistic-feature
    extraction or for display, both of which should use the detector's own
    saved `text`.
    """
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    for ch in _ZW_CHARS:
        t = t.replace(ch, "")
    t = t.translate(_DEVANAGARI_DIGITS)
    t = _WS_RE.sub(" ", t).strip()
    return t


def add_text_key(master: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    master = master.copy()
    master["text_key"] = master[text_col].map(normalize_text_for_matching)
    return master


def per_cell_audit(master: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, train_dataset, train_llm, test_dataset, test_llm)
    describing raw row counts, duplicate/conflicting text_keys, label
    balance, and score availability. Write this to disk and READ it before
    trusting any downstream paired test — a cell with a large
    conflicting-duplicate count means the paired sample for that cell is
    smaller and less trustworthy than `n` in the metrics tables suggests.
    """
    rows = []
    group_cols = ["model", "train_dataset", "train_llm", "test_dataset", "test_llm"]
    for keys, grp in master.groupby(group_cols):
        n = len(grp)
        n_unique_keys = grp["text_key"].nunique()
        dup_groups = grp.groupby("text_key")
        conflicting_label = 0
        conflicting_pred = 0
        for _, g in dup_groups:
            if len(g) > 1:
                if g["true_label"].nunique() > 1:
                    conflicting_label += 1
                if g["predicted_label"].nunique() > 1:
                    conflicting_pred += 1
        n_missing_score = grp["score"].isna().sum()
        row = dict(zip(group_cols, keys))
        row.update(
            n_rows=n,
            n_unique_text_keys=n_unique_keys,
            n_duplicate_text_key_groups=int((dup_groups.size() > 1).sum()),
            n_conflicting_label_groups=conflicting_label,
            n_conflicting_prediction_groups=conflicting_pred,
            n_true_ai=int((grp["true_label"] == 1).sum()),
            n_true_human=int((grp["true_label"] == 0).sum()),
            n_missing_score=int(n_missing_score),
            frac_score_available=float(1 - n_missing_score / n) if n else np.nan,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def dedupe_cell(cell_df: pd.DataFrame):
    """Collapse a single (dataset, llm, model) slice to one row per
    text_key. Rows whose duplicates fully agree (same true_label, same
    predicted_label) are safely collapsed to one row (score averaged if it
    varies). Rows whose duplicates DISAGREE on true_label or predicted_label
    are dropped entirely rather than resolved with `aggfunc="first"`, since
    silently picking an arbitrary row would fabricate a paired observation
    that doesn't actually exist. Returns (deduped_df, n_dropped_conflicting).
    """
    if cell_df.empty:
        return cell_df, 0
    keep_rows = []
    n_dropped = 0
    for key, g in cell_df.groupby("text_key"):
        if len(g) == 1:
            keep_rows.append(g.iloc[0])
            continue
        if g["true_label"].nunique() > 1 or g["predicted_label"].nunique() > 1:
            n_dropped += 1
            continue
        rep = g.iloc[0].copy()
        if g["score"].notna().any():
            rep["score"] = g["score"].mean()
        keep_rows.append(rep)
    if not keep_rows:
        return cell_df.iloc[0:0], n_dropped
    out = pd.DataFrame(keep_rows).reset_index(drop=True)
    return out, n_dropped


def safe_paired_matrix(grp: pd.DataFrame, value_col: str = "predicted_label"):
    """Build a text_key x model matrix for a single test cell, safely.

    Steps:
      1. Dedupe each model's rows within the cell (drop conflicting dupes).
      2. Keep only text_keys present for EVERY model in the cell after
         dedup (an inner join — a text_key that only 4 of 7 detectors have
         a row for cannot be used in a paired test that needs all of them).
      3. Verify true_label agreement across detectors for each surviving
         text_key (detectors should be scoring the same underlying item);
         drop any text_key with disagreeing true_label across detectors —
         that is a data-integrity problem, not real signal, and should not
         silently feed a McNemar/Cochran/Jaccard computation.

    Returns (pivot_df[text_key x model], truth[text_key], audit_dict).
    """
    models = sorted(grp["model"].unique())
    per_model = {}
    n_dropped_conflicting = {}
    for m in models:
        deduped, n_drop = dedupe_cell(grp[grp["model"] == m])
        per_model[m] = deduped.set_index("text_key")
        n_dropped_conflicting[m] = n_drop

    common_keys = None
    for m, d in per_model.items():
        idx = set(d.index)
        common_keys = idx if common_keys is None else (common_keys & idx)
    common_keys = sorted(common_keys) if common_keys else []

    if not common_keys:
        return pd.DataFrame(), pd.Series(dtype=float), dict(
            n_common_before_truth_check=0, n_dropped_truth_mismatch=0,
            n_dropped_conflicting_by_model=n_dropped_conflicting,
        )

    truth_by_model = pd.DataFrame(
        {m: per_model[m].loc[common_keys, "true_label"] for m in models}
    )
    truth_agrees = truth_by_model.nunique(axis=1) == 1
    n_truth_mismatch = int((~truth_agrees).sum())
    good_keys = truth_agrees[truth_agrees].index.tolist()

    pivot = pd.DataFrame({m: per_model[m].loc[good_keys, value_col] for m in models})
    truth = truth_by_model.loc[good_keys, models[0]]
    truth.name = "true_label"

    audit = dict(
        n_common_before_truth_check=len(common_keys),
        n_dropped_truth_mismatch=n_truth_mismatch,
        n_paired_items=len(good_keys),
        n_dropped_conflicting_by_model=n_dropped_conflicting,
    )
    return pivot, truth, audit
