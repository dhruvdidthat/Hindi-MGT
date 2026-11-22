"""Canonical data loaders for the Hindi MGT statistical analyses (RQ1–RQ5).

Every loader returns one long-format row per detector/source/target/text with:
  true_label=1 / predicted_label=1 -> AI-generated
  true_label=0 / predicted_label=0 -> Human-written
  score higher -> more AI-like (or NaN if no continuous score was saved)

Source-label and saved-prediction encodings are intentionally treated as
separate concepts. In particular:
  * Shared training/testing CSVs used by GPTZero and DetectGPT have raw 0=AI.
  * GPTZero uses low perplexity as AI evidence; canonical score = -perplexity.
  * DetectGPT's v3 cache rebuild has already canonicalized its output labels;
    it uses the original LLR without negation (higher LLR = more AI-like).
  * RQ4 COLING and RQ5 new-prompt labels are separately checked/mapped using
    their source metadata and the completed v3 audit.

GPTZero thresholds are read from the existing GPTZero source-calibration
master, and DetectGPT thresholds/predictions are read from the completed v3
cache-only rebuild. No detector or GPU inference is performed here. Target
labels are used only for evaluation, never for threshold selection.
"""
import os
import warnings
from typing import Optional, Dict, Any
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd

from config import DATASETS, LLMS, ZERO_SHOT_SPECS


def _safe_read_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        warnings.warn(f"Could not read {path}: {e}")
        return None


def _canonicalize(raw_true, raw_pred, raw_ai_label):
    """raw_ai_label: the integer value that means 'AI' in this detector's
    OWN raw label column. Returns (canonical_true, canonical_pred) where
    1 always means AI and 0 always means Human, regardless of the input
    detector's own convention."""
    raw_true = np.asarray(raw_true, dtype=int)
    raw_pred = np.asarray(raw_pred, dtype=int)
    if raw_ai_label == 1:
        return raw_true, raw_pred
    elif raw_ai_label == 0:
        return 1 - raw_true, 1 - raw_pred
    else:
        raise ValueError(f"raw_ai_label must be 0 or 1, got {raw_ai_label}")


def _canonicalize_score(raw_score, direction):
    """direction:
       'human_prob'      -> raw score is P(Human); canonical = 1 - raw
       'neg_perplexity'  -> raw score is perplexity, lower=more AI; canonical = -raw
       'ai_native'       -> raw score already means higher=more AI; unchanged
       'none'            -> no score available
    """
    if direction == "none" or raw_score is None:
        return np.nan
    if direction == "human_prob":
        return 1.0 - raw_score
    if direction == "neg_perplexity":
        return -raw_score
    if direction == "ai_native":
        return raw_score
    if direction == "lower_is_ai":
        return -raw_score
    raise ValueError(f"unknown score direction: {direction}")


# ---------------------------------------------------------------------------
# Score-only detectors: source-calibrated GPTZero + cache-rebuilt DetectGPT
# ---------------------------------------------------------------------------

_DETECTGPT_REBUILT_ROOT = None


def set_detectgpt_rebuilt_root(path=None):
    """Set an explicit v3 audit directory; None enables newest-valid auto-discovery."""
    global _DETECTGPT_REBUILT_ROOT
    _DETECTGPT_REBUILT_ROOT = Path(path).expanduser().resolve() if path else None


def _project_root_from_training_root(root):
    root = Path(root).expanduser().resolve()
    # CLI uses CODE/training_models as --root.
    return root.parent if root.name.lower() == "training_models" else root


def discover_detectgpt_rebuilt_root(training_models_root):
    """Find the newest complete v3 cache-only rebuild (RQ123/RQ4/RQ5)."""
    if _DETECTGPT_REBUILT_ROOT is not None:
        candidate = _DETECTGPT_REBUILT_ROOT
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Explicit DetectGPT rebuild directory not found: {candidate}")

    project_root = _project_root_from_training_root(training_models_root)
    candidates = sorted(
        project_root.glob("detectgpt_audit_*"),
        key=lambda p: (p.stat().st_mtime if p.exists() else 0, p.name),
        reverse=True,
    )
    for audit_root in candidates:
        rq123 = audit_root / "rebuilt_from_score_cache" / "rq123"
        rq4 = audit_root / "rebuilt_from_score_cache" / "rq4"
        rq5 = audit_root / "rq5_repaired"
        checks = [
            (rq123 / "rq123_recomputed_metrics.csv", 24,
             list(rq123.glob("train_*__test_*/predictions.csv"))),
            (rq4 / "rq4_recomputed_metrics.csv", 6,
             list(rq4.glob("*_tested_on_coling_hindi/predictions.csv"))),
            (rq5 / "detect_gpt_calibrated_new_prompt_results_REPAIRED.csv", 12,
             list(rq5.glob("*_tested_on_*/predictions.csv"))),
        ]
        if all(master.is_file() and len(files) == expected for master, expected, files in checks):
            return audit_root
    raise FileNotFoundError(
        "No complete DetectGPT v3 cache rebuild found. Expected a CODE/detectgpt_audit_*/ "
        "directory containing 24 RQ1-RQ3, 6 RQ4 and 12 repaired RQ5 predictions. "
        "Pass --detectgpt-rebuilt-root explicitly when needed."
    )


def _read_csv_required(path, detector, context):
    df = _safe_read_csv(path)
    if df is None:
        warnings.warn(f"[{detector}/{context}] missing or unreadable: {path}")
    return df


def _normalise_key(value):
    if pd.isna(value):
        return ""
    s = unicodedata.normalize("NFC", str(value))
    s = s.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        s = s.replace(ch, "")
    return " ".join(s.split()).strip()


def _align_scored_rows(reference_df, scored_df, score_col, label_col="label"):
    """Attach continuous scores to the reference rows without collapsing duplicates.

    Prefer exact row-order alignment when the normalized text sequences match.
    Otherwise use text + within-text occurrence index. This preserves repeated
    COLING rows and avoids a many-to-many merge.
    """
    if reference_df is None or scored_df is None or "text" not in reference_df or "text" not in scored_df or score_col not in scored_df:
        return pd.DataFrame(), {"mode": "missing_columns", "coverage": 0.0}
    left = reference_df.copy().reset_index(drop=True)
    right = scored_df.copy().reset_index(drop=True)
    left["_key"] = left["text"].map(_normalise_key)
    right["_key"] = right["text"].map(_normalise_key)
    right["_score"] = pd.to_numeric(right[score_col], errors="coerce")
    left["_row_order"] = np.arange(len(left))

    if len(left) == len(right) and np.array_equal(left["_key"].to_numpy(), right["_key"].to_numpy()):
        left[score_col] = right["_score"].to_numpy(dtype=float)
        if label_col in scored_df.columns:
            left["_score_file_label"] = pd.to_numeric(right[label_col], errors="coerce").to_numpy()
        # Keep the scorer's own exports under explicit diagnostic names. Do
        # not confuse them with a benchmark/source `predicted_label` column.
        if "predicted_label" in right.columns:
            left["_score_file_predicted_label"] = pd.to_numeric(right["predicted_label"], errors="coerce").to_numpy()
        if "true_label" in right.columns:
            left["_score_file_true_label"] = pd.to_numeric(right["true_label"], errors="coerce").to_numpy()
        coverage = float(np.isfinite(pd.to_numeric(left[score_col], errors="coerce")).mean())
        return left.sort_values("_row_order").reset_index(drop=True), {
            "mode": "exact_row_order", "coverage": coverage,
            "reference_rows": len(left), "score_rows": len(right),
        }

    left["_occ"] = left.groupby("_key", sort=False).cumcount()
    right["_occ"] = right.groupby("_key", sort=False).cumcount()
    keep = ["_key", "_occ", "_score"]
    if label_col in right.columns:
        right["_score_file_label"] = pd.to_numeric(right[label_col], errors="coerce")
        keep.append("_score_file_label")
    if "predicted_label" in right.columns:
        right["_score_file_predicted_label"] = pd.to_numeric(right["predicted_label"], errors="coerce")
        keep.append("_score_file_predicted_label")
    if "true_label" in right.columns:
        right["_score_file_true_label"] = pd.to_numeric(right["true_label"], errors="coerce")
        keep.append("_score_file_true_label")
    right = right[keep]
    aligned = left.merge(right, on=["_key", "_occ"], how="left", validate="one_to_one", sort=False)
    aligned = aligned.sort_values("_row_order").reset_index(drop=True)
    aligned[score_col] = aligned["_score"]
    finite = np.isfinite(pd.to_numeric(aligned[score_col], errors="coerce"))
    coverage = float(finite.sum() / max(1, len(reference_df)))
    return aligned, {
        "mode": "text_occurrence_order", "coverage": coverage,
        "reference_rows": len(reference_df), "score_rows": len(scored_df),
    }


def _gptzero_threshold_map(training_models_root):
    """Read the six frozen source thresholds from GPTZero's original result master."""
    root = Path(training_models_root).expanduser().resolve()
    candidates = [
        root / "gptzero" / "inference_results" / "gptzero_all_results.csv",
        root / "gptzero_in_domain" / "inference_results" / "gptzero_all_results.csv",
        root / "gptzero_60_threshold_fixed" / "inference_results" / "gptzero_all_results.csv",
    ]
    master_path = next((p for p in candidates if p.is_file()), None)
    if master_path is None:
        for p in root.glob("**/gptzero_all_results.csv"):
            if "inference_results" in p.parts:
                master_path = p
                break
    if master_path is None:
        raise FileNotFoundError(
            "GPTZero threshold master not found. Expected training_models/gptzero/inference_results/gptzero_all_results.csv."
        )
    df = _read_csv_required(master_path, "GPTZero", "threshold master")
    required = {"train_dataset", "train_llm", "threshold"}
    if df is None or not required.issubset(df.columns):
        raise ValueError(f"GPTZero master {master_path} must contain {sorted(required)}")
    if {"test_dataset", "test_llm"}.issubset(df.columns):
        in_domain = df[(df.train_dataset == df.test_dataset) & (df.train_llm == df.test_llm)]
        if not in_domain.empty:
            df = in_domain
    out = {}
    for (ds, llm), group in df.groupby(["train_dataset", "train_llm"]):
        vals = pd.to_numeric(group.threshold, errors="coerce").dropna().unique()
        if len(vals) == 1:
            out[(str(ds), str(llm))] = float(vals[0])
        elif len(vals) > 1:
            raise ValueError(f"Conflicting GPTZero source thresholds for {ds}/{llm}: {vals.tolist()}")
    if len(out) != 6:
        raise ValueError(f"Expected six GPTZero source thresholds in {master_path}; found {len(out)}: {out}")
    return out, master_path


def _gptzero_prediction_paths(root, src_ds, src_llm, tgt_ds, tgt_llm):
    base = Path(root) / "gptzero" / "inference_results"
    paths = []
    if src_ds == tgt_ds and src_llm == tgt_llm:
        paths.append(base / f"{tgt_ds}_{tgt_llm}" / "predictions.csv")
    else:
        paths.append(base / f"cross_train_{src_ds}_{src_llm}__test_{tgt_ds}_{tgt_llm}" / "predictions.csv")
    # Compatibility candidates for variants of the writer layout.
    paths.extend([
        base / f"{src_ds}_{src_llm}" / f"tested_on_{tgt_ds}_{tgt_llm}" / "predictions.csv",
        base / f"{src_ds}_{src_llm}_tested_on_{tgt_ds}_{tgt_llm}" / "predictions.csv",
    ])
    return next((p for p in paths if p.is_file()), None)


def _safe_ai_score_from_perplexity(perplexity):
    ppl = np.asarray(perplexity, dtype=float)
    finite = ppl[np.isfinite(ppl)]
    fallback = float(finite.max() * 10.0) if finite.size else 1e6
    return -np.where(np.isfinite(ppl), ppl, fallback), fallback


def _build_gptzero_cell(reference_df, scored_df, threshold, train_ds, train_llm,
                        test_ds, test_llm, source_path, context,
                        saved_pred_encoding="raw0_ai"):
    """Build canonical GPTZero rows from source truth + saved perplexity.

    `saved_pred_encoding` is used only for the diagnostic agreement column:
      raw0_ai: legacy RQ1-RQ3 predictions (0=AI, 1=Human)
      canonical_ai1: RQ4/RQ5 outputs (1=AI, 0=Human)
    The predictions used by statistics are always regenerated from perplexity
    and the original source threshold.
    """
    aligned, alignment = _align_scored_rows(reference_df, scored_df, "perplexity", label_col="label")
    if aligned.empty or "label" not in aligned.columns:
        warnings.warn(f"[GPTZero/{context}] cannot align scores/labels for {source_path}")
        return pd.DataFrame()
    raw_label = pd.to_numeric(aligned.label, errors="coerce")
    ppl = pd.to_numeric(aligned.perplexity, errors="coerce").to_numpy(dtype=float)
    label_valid = raw_label.isin([0, 1]).to_numpy()
    # An infinite perplexity is retained as an unusable/very-Human-like
    # model score, matching the GPTZero inference export's fallback behavior.
    score_valid = np.isfinite(ppl) | np.isinf(ppl)
    valid = label_valid & score_valid
    if not valid.any():
        return pd.DataFrame()
    raw = raw_label.to_numpy(dtype=float)[valid].astype(int)
    ppl = ppl[valid]
    y_true = (raw == 0).astype(int)  # GPTZero/source convention: raw 0=AI, 1=Human
    y_pred = (ppl <= float(threshold)).astype(int)  # low perplexity => AI
    score, fallback = _safe_ai_score_from_perplexity(ppl)

    saved_agreement = np.nan
    if "_score_file_predicted_label" in aligned.columns:
        saved_raw_pred = pd.to_numeric(aligned.loc[valid, "_score_file_predicted_label"], errors="coerce").to_numpy(dtype=float)
        pmask = np.isfinite(saved_raw_pred) & np.isin(saved_raw_pred, [0, 1])
        if pmask.any():
            saved_values = saved_raw_pred[pmask].astype(int)
            saved_canonical = (1 - saved_values) if saved_pred_encoding == "raw0_ai" else saved_values
            saved_agreement = float(np.mean(saved_canonical == y_pred[pmask]))
    saved_truth_agreement = np.nan
    if "_score_file_true_label" in aligned.columns:
        saved_true = pd.to_numeric(aligned.loc[valid, "_score_file_true_label"], errors="coerce").to_numpy(dtype=float)
        tmask = np.isfinite(saved_true) & np.isin(saved_true, [0, 1])
        if tmask.any():
            saved_truth_agreement = float(np.mean(saved_true[tmask].astype(int) == y_true[tmask]))
    if alignment["coverage"] < 0.98:
        warnings.warn(f"[GPTZero/{context}] score alignment coverage only {alignment['coverage']:.2%}: {source_path}")

    out = pd.DataFrame({
        "model": "GPTZero", "train_dataset": train_ds, "train_llm": train_llm,
        "test_dataset": test_ds, "test_llm": test_llm,
        "text": aligned.loc[valid, "text"].astype(str).to_numpy(),
        "true_label": y_true, "predicted_label": y_pred, "score": score,
        "calibration_threshold": float(threshold),
        "calibration_threshold_raw_perplexity": float(threshold),
        "threshold_policy": "source_train_youden_j",
        "score_direction": "negative_perplexity_higher_is_ai",
        "score_alignment_mode": alignment["mode"],
        "score_alignment_coverage": alignment["coverage"],
        "stored_prediction_rule_agreement": saved_agreement,
        "stored_truth_agreement": saved_truth_agreement,
        "source_path": str(source_path),
    })
    return out


def load_gptzero(root):
    """Load the existing GPTZero score/prediction files for all RQ1-RQ3 cells.

    Uses the frozen threshold in gptzero_all_results.csv. Ground truth is read
    from the shared testing_data files (raw 0=AI), perplexity is read from the
    existing target predictions, and predictions are regenerated from the
    source threshold. No GPU inference and no target-label calibration.
    """
    root = Path(root).expanduser().resolve()
    thresholds, master_path = _gptzero_threshold_map(root)
    test_root = root / "testing_data"
    rows = []
    for src_ds in DATASETS:
        for src_llm in LLMS:
            threshold = thresholds[(src_ds, src_llm)]
            for tgt_ds in DATASETS:
                for tgt_llm in LLMS:
                    if src_ds != tgt_ds and src_llm != tgt_llm:
                        continue  # RQ1-RQ3: 6 + 12 + 6; both-shift is not in these RQs.
                    prediction_path = _gptzero_prediction_paths(root, src_ds, src_llm, tgt_ds, tgt_llm)
                    ref_path = test_root / tgt_ds / f"{tgt_ds}_test_{tgt_llm}.csv"
                    reference = _read_csv_required(ref_path, "GPTZero", "shared test truth")
                    scored = _read_csv_required(prediction_path, "GPTZero", "target perplexities") if prediction_path else None
                    if reference is None or scored is None:
                        warnings.warn(f"[GPTZero] missing source-target cell {src_ds}/{src_llm} -> {tgt_ds}/{tgt_llm}")
                        continue
                    cell = _build_gptzero_cell(reference, scored, threshold, src_ds, src_llm,
                                               tgt_ds, tgt_llm, prediction_path, "RQ1-RQ3")
                    if not cell.empty:
                        cell["calibration_master"] = str(master_path)
                        rows.append(cell)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _detectgpt_threshold_metadata(audit_root):
    path = Path(audit_root) / "rebuilt_from_score_cache" / "rq123" / "source_thresholds_recomputed.csv"
    df = _safe_read_csv(path)
    if df is None or not {"train_dataset", "train_llm", "threshold"}.issubset(df.columns):
        raise FileNotFoundError(f"DetectGPT rebuilt source threshold table missing/incomplete: {path}")
    out = {}
    for _, row in df.iterrows():
        key = (str(row.train_dataset), str(row.train_llm))
        out[key] = {
            "calibration_threshold": float(row.threshold),
            "calibration_youden_j": float(row.train_youden_j) if "train_youden_j" in row and pd.notna(row.train_youden_j) else np.nan,
            "calibration_auc": float(row.train_auroc_ai_higher_score) if "train_auroc_ai_higher_score" in row and pd.notna(row.train_auroc_ai_higher_score) else np.nan,
            "calibration_n": int(row.n_calibration) if "n_calibration" in row and pd.notna(row.n_calibration) else np.nan,
            "calibration_n_ai": int(row.ai_count) if "ai_count" in row and pd.notna(row.ai_count) else np.nan,
            "calibration_n_human": int(row.human_count) if "human_count" in row and pd.notna(row.human_count) else np.nan,
            "calibration_path": "v3 score_cache/train_*.csv",
            "threshold_policy": "source_train_youden_j",
        }
    return out


def _detectgpt_canonical_cell(path, train_ds, train_llm, test_ds, test_llm, metadata=None, context=""):
    df = _read_csv_required(path, "DetectGPT", context)
    needed = {"text", "true_label", "predicted_label", "llr"}
    if df is None or not needed.issubset(df.columns):
        if df is not None:
            warnings.warn(f"[DetectGPT/{context}] {path} missing {sorted(needed - set(df.columns))}")
        return pd.DataFrame()
    true = pd.to_numeric(df.true_label, errors="coerce")
    pred = pd.to_numeric(df.predicted_label, errors="coerce")
    score = pd.to_numeric(df.llr, errors="coerce")
    valid = true.isin([0, 1]) & pred.isin([0, 1]) & np.isfinite(score) & df.text.notna()
    if not valid.any():
        return pd.DataFrame()
    out = pd.DataFrame({
        "model": "DetectGPT", "train_dataset": train_ds, "train_llm": train_llm,
        "test_dataset": test_ds, "test_llm": test_llm,
        "text": df.loc[valid, "text"].astype(str).to_numpy(),
        "true_label": true.loc[valid].astype(int).to_numpy(),
        "predicted_label": pred.loc[valid].astype(int).to_numpy(),
        "score": score.loc[valid].astype(float).to_numpy(),
        "source_path": str(path), "score_direction": "llr_higher_is_ai",
    })
    if "threshold" in df.columns:
        out["calibration_threshold"] = pd.to_numeric(df.loc[valid, "threshold"], errors="coerce").to_numpy()
    if metadata:
        for key, value in metadata.items():
            out[key] = value
    return out


def load_detectgpt(root):
    """Load RQ1-RQ3 DetectGPT from the completed v3 cache-only rebuild.

    The v3 outputs are canonical (1=AI) and use recalibrated thresholds from
    raw source labels 0=AI. The invalid old DetectGPT prediction folders are
    intentionally ignored. Existing llr scores are reused; no inference runs.
    """
    audit_root = discover_detectgpt_rebuilt_root(root)
    base = audit_root / "rebuilt_from_score_cache" / "rq123"
    threshold_meta = _detectgpt_threshold_metadata(audit_root)
    rows = []
    for src_ds in DATASETS:
        for src_llm in LLMS:
            meta = threshold_meta.get((src_ds, src_llm), {})
            for tgt_ds in DATASETS:
                for tgt_llm in LLMS:
                    if src_ds != tgt_ds and src_llm != tgt_llm:
                        continue
                    case = f"train_{src_ds}_{src_llm}__test_{tgt_ds}_{tgt_llm}"
                    path = base / case / "predictions.csv"
                    cell = _detectgpt_canonical_cell(path, src_ds, src_llm, tgt_ds, tgt_llm, meta, "RQ1-RQ3 rebuilt")
                    if not cell.empty:
                        rows.append(cell)
    if len(rows) != 24:
        warnings.warn(f"[DetectGPT] expected 24 rebuilt RQ1-RQ3 cells; loaded {len(rows)}. Rebuild root: {audit_root}")
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not result.empty:
        result["detectgpt_rebuild_root"] = str(audit_root)
    return result


def _find_gptzero_rq4_file(rq4_root, train_ds, train_llm):
    base = Path(rq4_root) / "gptzero" / "inference_results_coling"
    candidates = [
        base / f"{train_ds}_{train_llm}_tested_on_coling_hindi_gpt4o" / "predictions.csv",
        base / f"{train_ds}_{train_llm}_tested_on_coling_hindi" / "predictions.csv",
        base / f"{train_ds}_{train_llm}" / "tested_on_coling_hindi_data" / "predictions.csv",
    ]
    return next((p for p in candidates if p.is_file()), None)


def load_zero_shot_rq4(training_models_root, rq4_root, model):
    """Load one score-only detector family for RQ4 (six source thresholds)."""
    training_models_root = Path(training_models_root).expanduser().resolve()
    rq4_root = Path(rq4_root).expanduser().resolve()
    rows = []
    if model == "DetectGPT":
        audit_root = discover_detectgpt_rebuilt_root(training_models_root)
        base = audit_root / "rebuilt_from_score_cache" / "rq4"
        threshold_meta = _detectgpt_threshold_metadata(audit_root)
        for ds in DATASETS:
            for llm in LLMS:
                case = f"{ds}_{llm}_tested_on_coling_hindi"
                path = base / case / "predictions.csv"
                cell = _detectgpt_canonical_cell(path, ds, llm, "coling_hindi", "gpt4o", threshold_meta.get((ds, llm), {}), "RQ4 rebuilt")
                if not cell.empty:
                    rows.append(cell)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    if model != "GPTZero":
        raise KeyError(f"Unsupported score-only detector: {model}")
    thresholds, master_path = _gptzero_threshold_map(training_models_root)
    source_path = rq4_root / "hindi_data.csv"
    reference = _read_csv_required(source_path, "GPTZero", "COLING source truth")
    if reference is None or not {"text", "label", "model"}.issubset(reference.columns):
        warnings.warn(f"[GPTZero/RQ4] COLING source must contain text,label,model: {source_path}")
        return pd.DataFrame()
    raw_labels = pd.to_numeric(reference.label, errors="coerce")
    model_tags = reference.model.astype(str).str.strip().str.lower()
    human = model_tags.eq("human")
    # Verified convention: all model=='human' rows are raw label 1 and all
    # GPT-4o-generated rows are raw label 0. Abort rather than silently invert.
    if not human.any() or not (~human).any() or not (raw_labels[human] == 1).all() or not (raw_labels[~human] == 0).all():
        raise ValueError("GPTZero RQ4: COLING model metadata does not support raw label 0=AI, 1=Human.")
    for ds in DATASETS:
        for llm in LLMS:
            path = _find_gptzero_rq4_file(rq4_root, ds, llm)
            if path is None:
                warnings.warn(f"[GPTZero/RQ4] missing score file for source {ds}/{llm}")
                continue
            scored = _read_csv_required(path, "GPTZero", "COLING perplexity")
            if scored is None:
                continue
            cell = _build_gptzero_cell(reference, scored, thresholds[(ds, llm)], ds, llm,
                                       "coling_hindi", "gpt4o", path, "RQ4",
                                       saved_pred_encoding="canonical_ai1")
            if not cell.empty:
                cell["calibration_master"] = str(master_path)
                rows.append(cell)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_zero_shot_rq5(training_models_root, rq5_root, model):
    """Load original-source-calibrated GPTZero or repaired DetectGPT RQ5 cells."""
    training_models_root = Path(training_models_root).expanduser().resolve()
    rq5_root = Path(rq5_root).expanduser().resolve()
    rows = []
    if model == "DetectGPT":
        audit_root = discover_detectgpt_rebuilt_root(training_models_root)
        base = audit_root / "rq5_repaired"
        threshold_meta = _detectgpt_threshold_metadata(audit_root)
        for tgt_ds in DATASETS:
            for tgt_llm in ("gemini", "qwen3"):
                for src_ds in DATASETS:
                    for src_llm in ("gemini", "qwen3"):
                        if src_ds != tgt_ds and src_llm != tgt_llm:
                            continue
                        case = f"{src_ds}_{src_llm}_tested_on_{tgt_ds}_{tgt_llm}"
                        path = base / case / "predictions.csv"
                        cell = _detectgpt_canonical_cell(path, src_ds, src_llm, tgt_ds, tgt_llm,
                                                       threshold_meta.get((src_ds, src_llm), {}), "RQ5 repaired")
                        if not cell.empty:
                            cell["prompt"] = "new"
                            rows.append(cell)
        result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if len(result) and result.groupby(["train_dataset", "train_llm", "test_dataset", "test_llm"]).ngroups != 12:
            warnings.warn(f"[DetectGPT/RQ5] expected 12 repaired cells; loaded {result.groupby(['train_dataset','train_llm','test_dataset','test_llm']).ngroups}")
        return result

    if model != "GPTZero":
        raise KeyError(f"Unsupported score-only detector: {model}")
    thresholds, master_path = _gptzero_threshold_map(training_models_root)
    source_data_root = rq5_root / "new_prompt_data"
    score_root = rq5_root / "gptzero" / "inference_results_new_prompt"
    for tgt_ds in DATASETS:
        for tgt_llm in ("gemini", "qwen3"):
            reference_path = source_data_root / f"{tgt_ds}_{tgt_llm}_new_prompt.csv"
            reference = _read_csv_required(reference_path, "GPTZero", "new-prompt source truth")
            if reference is None or not {"text", "label"}.issubset(reference.columns):
                continue
            # Independent audit supports raw new-prompt 0=AI, 1=Human for all four targets.
            for src_ds in DATASETS:
                for src_llm in ("gemini", "qwen3"):
                    if src_ds != tgt_ds and src_llm != tgt_llm:
                        continue
                    case = f"{src_ds}_{src_llm}_tested_on_{tgt_ds}_{tgt_llm}"
                    path = score_root / case / "predictions.csv"
                    scored = _read_csv_required(path, "GPTZero", "new-prompt perplexity")
                    if scored is None:
                        continue
                    cell = _build_gptzero_cell(reference, scored, thresholds[(src_ds, src_llm)],
                                               src_ds, src_llm, tgt_ds, tgt_llm, path, "RQ5",
                                               saved_pred_encoding="canonical_ai1")
                    if not cell.empty:
                        cell["calibration_master"] = str(master_path)
                        cell["prompt"] = "new"
                        rows.append(cell)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(result) and result.groupby(["train_dataset", "train_llm", "test_dataset", "test_llm"]).ngroups != 12:
        warnings.warn(f"[GPTZero/RQ5] expected 12 Table 7 cells; loaded {result.groupby(['train_dataset','train_llm','test_dataset','test_llm']).ngroups}")
    return result


def load_zero_shot_rq4_all(training_models_root, rq4_root):
    frames = [load_zero_shot_rq4(training_models_root, rq4_root, model) for model in ZERO_SHOT_SPECS]
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_zero_shot_rq5_all(training_models_root, rq5_root):
    frames = [load_zero_shot_rq5(training_models_root, rq5_root, model) for model in ZERO_SHOT_SPECS]
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# End of score-only detector loaders. Trainable detector loaders follow.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Trainable models with the nested "{train}/tested_on_{test}/predictions.csv"
# layout: XLM-RoBERTa, IndicBERT, RADAR
# ---------------------------------------------------------------------------

def _load_nested_trainable(root, model_subdir, model_name, label_col, pred_col,
                            raw_ai_label, score_col=None, score_direction="none"):
    rows = []
    base = os.path.join(root, model_subdir, "inference_results")
    if not os.path.isdir(base):
        warnings.warn(f"[{model_name}] missing base dir: {base}")
        return pd.DataFrame()
    for train_ds in DATASETS:
        for train_llm in LLMS:
            train_dir = f"{train_ds}_{train_llm}"
            for test_ds in DATASETS:
                for test_llm in LLMS:
                    test_dir = f"tested_on_{test_ds}_{test_llm}"
                    path = os.path.join(base, train_dir, test_dir, "predictions.csv")
                    df = _safe_read_csv(path)
                    if df is None:
                        continue
                    if label_col not in df.columns or pred_col not in df.columns or "text" not in df.columns:
                        warnings.warn(f"[{model_name}] unexpected columns at {path}: {df.columns.tolist()}")
                        continue
                    true_c, pred_c = _canonicalize(
                        df[label_col].astype(int).values,
                        df[pred_col].astype(int).values,
                        raw_ai_label,
                    )
                    if score_col and score_col in df.columns:
                        score_c = _canonicalize_score(df[score_col].astype(float).values, score_direction)
                    else:
                        score_c = np.nan
                    out = pd.DataFrame({
                        "model": model_name,
                        "train_dataset": train_ds,
                        "train_llm": train_llm,
                        "test_dataset": test_ds,
                        "test_llm": test_llm,
                        "text": df["text"],
                        "true_label": true_c,
                        "predicted_label": pred_c,
                        "score": score_c,
                    })
                    rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_xlm_roberta(root):
    # XLM-R: raw 0=AI, 1=Human. Probability column NOT saved in
    # predictions.csv (the inference script computes it internally for its
    # own AUC but never writes it to df_test) -> no score, AUC will be NaN
    # for this model. Do not fabricate an AUC from hard labels.
    return _load_nested_trainable(
        root, "xlm_roberta", "XLM-RoBERTa",
        label_col="label", pred_col="predicted_label",
        raw_ai_label=0, score_col=None, score_direction="none",
    )


def load_indicbert(root):
    # Indic-BERT: raw 0=AI, 1=Human. Same "no saved probability" situation
    # as XLM-R (`all_probs.extend(probs[:, 1])` is computed but not written
    # to the predictions.csv schema this pipeline reads).
    return _load_nested_trainable(
        root, "indicBert", "Indic-BERT",
        label_col="label", pred_col="predicted_label",
        raw_ai_label=0, score_col=None, score_direction="none",
    )


def load_radar(root):
    # RADAR: raw 0=AI, 1=Human (`human_texts = label==1`, `ai_texts =
    # label==0` in the training loss; `preds = (probs>0.5)` maps prob>0.5 to
    # raw class 1 = Human). `prediction_probability` is therefore P(Human),
    # not P(AI) — canonical AI score = 1 - prediction_probability.
    return _load_nested_trainable(
        root, "RADAR", "RADAR",
        label_col="true_label", pred_col="predicted_label",
        raw_ai_label=0, score_col="prediction_probability", score_direction="human_prob",
    )


# ---------------------------------------------------------------------------
# Text-Fluoroscopy: flat "{train}_tested_on_{test}/predictions.csv" naming
# (per Note 3 — only these directories, ignore the nested duplicates)
# ---------------------------------------------------------------------------

def load_text_fluoroscopy(root):
    # Text-Fluoroscopy uses the OPPOSITE raw convention from the five
    # detectors above: raw 1 = AI/machine-generated (stated explicitly in
    # text_fluoroscopy_common.py; POSITIVE_IS_MACHINE=True and
    # machine_probs = probs_label1). So this detector's raw labels and its
    # predicted_prob_label1 column are ALREADY canonical (1=AI, higher
    # score=more AI) — no flip needed, unlike RADAR/CONDA/XLM-R/Indic-BERT.
    rows = []
    base_candidates = [
        os.path.join(root, "text_fluroscopy", "inference_results"),
        os.path.join(root, "text_fluoroscopy", "inference_results"),
    ]
    base = next((x for x in base_candidates if os.path.isdir(x)), base_candidates[0])
    if not os.path.isdir(base):
        warnings.warn(f"[Text-Fluoroscopy] missing base dir (checked both spellings): {base_candidates}")
        return pd.DataFrame()
    for train_ds in DATASETS:
        for train_llm in LLMS:
            for test_ds in DATASETS:
                for test_llm in LLMS:
                    dirname = f"{train_ds}_{train_llm}_tested_on_{test_ds}_{test_llm}"
                    path = os.path.join(base, dirname, "predictions.csv")
                    df = _safe_read_csv(path)
                    if df is None:
                        continue
                    needed = {"text", "true_label", "predicted_label"}
                    if not needed.issubset(df.columns):
                        warnings.warn(f"[Text-Fluoroscopy] unexpected columns at {path}: {df.columns.tolist()}")
                        continue
                    # text_fluoroscopy_common.py defines raw label 1=AI and
                    # exports predicted_prob_label1 = P(AI). Thus labels and
                    # score are already in canonical orientation.
                    true_c, pred_c = _canonicalize(
                        df["true_label"].astype(int).values,
                        df["predicted_label"].astype(int).values,
                        raw_ai_label=1,
                    )

                    if "predicted_prob_label1" in df.columns:
                        score_c = _canonicalize_score(
                            df["predicted_prob_label1"].astype(float).values,
                            "ai_native",
                        )
                    else:
                        score_c = np.nan
                    out = pd.DataFrame({
                        "model": "Text-Fluoroscopy",
                        "train_dataset": train_ds,
                        "train_llm": train_llm,
                        "test_dataset": test_ds,
                        "test_llm": test_llm,
                        "text": df["text"],
                        "true_label": true_c,
                        "predicted_label": pred_c,
                        "score": score_c,
                    })
                    rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# CONDA: "{train}_{train_llm}_{train_llm}/tested_on_{test}/predictions.csv"
# where the train-side directory name doubles the LLM, e.g.
# "hindisumm_gemini_gemini" (verified against the actual result-writer path,
# not the model-checkpoint path, which the CONDA code saves under a
# different, target-only naming scheme — see README).
# ---------------------------------------------------------------------------

def load_conda(root):
    # CONDA: raw 0=AI, 1=Human (`ai_recall = recall_score(..., pos_label=0)`
    # in its own inference code). Its softmaxed
    # `prediction_probability_class1` is P(Human), same situation as RADAR —
    # canonical AI score = 1 - prediction_probability_class1.
    rows = []
    base = os.path.join(root, "CONDA", "inference_results")
    if not os.path.isdir(base):
        warnings.warn(f"[CONDA] missing base dir: {base}")
        return pd.DataFrame()
    for train_ds in DATASETS:
        for train_llm in LLMS:
            train_dir = f"{train_ds}_{train_llm}_{train_llm}"
            for test_ds in DATASETS:
                for test_llm in LLMS:
                    test_dir = f"tested_on_{test_ds}_{test_llm}"
                    path = os.path.join(base, train_dir, test_dir, "predictions.csv")
                    df = _safe_read_csv(path)
                    if df is None:
                        continue
                    needed = {"text", "true_label", "predicted_label"}
                    if not needed.issubset(df.columns):
                        warnings.warn(f"[CONDA] unexpected columns at {path}: {df.columns.tolist()}")
                        continue
                    true_c, pred_c = _canonicalize(
                        df["true_label"].astype(int).values,
                        df["predicted_label"].astype(int).values,
                        raw_ai_label=0,
                    )
                    if "prediction_probability_class1" in df.columns:
                        score_c = _canonicalize_score(
                            df["prediction_probability_class1"].astype(float).values, "human_prob"
                        )
                    else:
                        score_c = np.nan
                    out = pd.DataFrame({
                        "model": "CONDA",
                        "train_dataset": train_ds,
                        "train_llm": train_llm,
                        "test_dataset": test_ds,
                        "test_llm": test_llm,
                        "text": df["text"],
                        "true_label": true_c,
                        "predicted_label": pred_c,
                        "score": score_c,
                    })
                    rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


LOADERS = {
    "XLM-RoBERTa": load_xlm_roberta,
    "Indic-BERT": load_indicbert,
    "RADAR": load_radar,
    "Text-Fluoroscopy": load_text_fluoroscopy,
    "CONDA": load_conda,
    "GPTZero": load_gptzero,
    "DetectGPT": load_detectgpt,
}

# All seven detectors participate in RQ2/RQ3. For GPTZero/DetectGPT,
# train_dataset/train_llm denote the source calibration pair (not a learned
# model checkpoint); the source-fitted threshold is frozen on target scores.
TRANSFER_CAPABLE_MODELS = set(LOADERS.keys())


def load_all(root, models=None):
    models = models or list(LOADERS.keys())
    frames = []
    for name in models:
        fn = LOADERS[name]
        print(f"Loading {name} ...")
        df = fn(root)
        if df is None or df.empty:
            warnings.warn(f"No data loaded for {name}.")
            continue
        # Drop EXACT full-row duplicates only (same model/cell/text/labels/
        # score) — this is safe because it can't discard disagreement. Rows
        # that share a cell/text but disagree on label/prediction are left
        # in place here and are instead resolved (with an audited drop, not
        # a silent "first wins") later in matching.dedupe_cell() at the
        # point where they'd actually affect a paired test.
        df = df.drop_duplicates(subset=["model", "train_dataset", "train_llm",
                                         "test_dataset", "test_llm", "text",
                                         "true_label", "predicted_label", "score"])
        print(f"  -> {len(df)} rows")
        frames.append(df)
    if not frames:
        raise RuntimeError("No data loaded for any model. Check --root path.")
    master = pd.concat(frames, ignore_index=True)
    master["correct"] = (master["true_label"] == master["predicted_label"]).astype(int)
    return master


def tag_setting(master: pd.DataFrame) -> pd.DataFrame:
    """Label each row with which RQ-relevant setting it belongs to."""
    def setting_row(r):
        if r["train_dataset"] == "zero_shot":
            return "zero_shot"
        if r["train_dataset"] == r["test_dataset"] and r["train_llm"] == r["test_llm"]:
            return "in_domain"
        if r["train_dataset"] == r["test_dataset"] and r["train_llm"] != r["test_llm"]:
            return "llm_cross_domain"
        if r["train_dataset"] != r["test_dataset"] and r["train_llm"] == r["test_llm"]:
            return "dataset_cross_domain"
        return "other_unseen_both"
    master = master.copy()
    master["setting"] = master.apply(setting_row, axis=1)
    return master
