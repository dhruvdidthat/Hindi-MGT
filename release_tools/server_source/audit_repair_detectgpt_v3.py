#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit DetectGPT labels, threshold calibration, and saved results without GPU inference.

This utility is deliberately non-destructive:
  * It never imports torch/transformers and never computes DetectGPT scores.
  * It reads existing llr score caches and saved prediction files.
  * The default mode writes an audit report only.
  * --repair-rq5 writes corrected RQ5 outputs into a NEW audit directory. It
    never overwrites the original DetectGPT CSVs.

Expected project layout (auto-detected, with multiple alternatives supported):
  CODE/training_models/detect_gpt_all_domain/inference_results/calibrated/
      score_cache/{train,test}_{dataset}_{llm}.csv
      detect_gpt_calibrated_all_results.csv
      train_{src_ds}_{src_llm}__test_{tgt_ds}_{tgt_llm}/predictions.csv
      coling/{src_ds}_{src_llm}_tested_on_coling_hindi/predictions.csv
  CODE/training_models/training_data/{dataset}/{dataset}_train_{llm}.csv
  CODE/training_models/testing_data/{dataset}/{dataset}_test_{llm}.csv
  CODE/COLING_2025_MGT_Task1/hindi_data.csv
  CODE/COLING_2025_MGT_Task1/detect_gpt_all_domain/inference_results/calibrated/
      score_cache/coling_hindi.csv
      coling/{source}_tested_on_coling_hindi/predictions.csv
  CODE/prompt_change/new_prompt_data/{dataset}_{llm}_new_prompt.csv
  CODE/prompt_change/detect_gpt/inference_results/calibrated/new_prompt/
      {src_ds}_{src_llm}_tested_on_{tgt_ds}_{tgt_llm}/predictions.csv

Install dependencies in the user's existing environment:
  pip install pandas numpy scikit-learn

Examples:
  python audit_repair_detectgpt_v3.py --project-root ~/MTP-2/HINDI_FINAL/Dhruv/CODE
  python audit_repair_detectgpt_v3.py --project-root ~/MTP-2/HINDI_FINAL/Dhruv/CODE --repair-rq5
  python audit_repair_detectgpt_v3.py --project-root ... --repair-rq5 --rq5-ai-label 0

Semantic defaults are based on the shared input files and their source readers:
  RQ1-RQ3 source truth raw AI label = 0 (the shared training/testing CSVs are
      read directly by XLM-R/GPTZero scripts, which treat 0 as AI and 1 as Human)
  RQ1-RQ3 original saved prediction AI label = 1 (the old DetectGPT script emits
      1 when llr >= threshold; this is separate from the source truth encoding)
  RQ4 COLING raw AI label = 0 (verified against model == 'human' metadata)
  RQ5 new-prompt raw AI label = auto (infer from unchanged Human-text overlap)

The README's DetectGPT 1=AI row describes the detector's saved prediction
encoding, but must not be applied to the common source `label` column unless
that source file is actually recoded. This utility keeps those encodings
separate and never rescales/recomputes llr values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, roc_curve
except Exception as exc:  # pragma: no cover - reported cleanly at runtime
    raise SystemExit("scikit-learn is required: pip install scikit-learn") from exc

DATASETS = ("hindisumm", "xquad")
LLMS_RQ123 = ("gemini", "llama3", "qwen3")
LLMS_RQ5 = ("gemini", "qwen3")
ZW_CHARS = "\u200b\u200c\u200d\ufeff"
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def norm_text(value: Any) -> str:
    """Conservative normalized join key; preserves punctuation and wording."""
    if not isinstance(value, str):
        if pd.isna(value):
            return ""
        value = str(value)
    value = unicodedata.normalize("NFC", value)
    value = value.translate(DEVANAGARI_DIGITS)
    for ch in ZW_CHARS:
        value = value.replace(ch, "")
    return " ".join(value.split()).strip()


def stable_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.is_file():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}", file=sys.stderr)
        return None


def unique_label_map(df: Optional[pd.DataFrame], label_col: str = "label") -> Tuple[Dict[str, int], int, int]:
    """Map normalized text -> unique binary label; return conflict/blank counts."""
    if df is None or "text" not in df.columns or label_col not in df.columns:
        return {}, 0, 0
    work = df[["text", label_col]].copy()
    work["key"] = work["text"].map(norm_text)
    work["label_num"] = pd.to_numeric(work[label_col], errors="coerce")
    work = work[work["key"].ne("") & work["label_num"].isin([0, 1])]
    groups = work.groupby("key")["label_num"]
    out: Dict[str, int] = {}
    conflicts = 0
    for key, vals in groups:
        vals_u = vals.astype(int).unique()
        if len(vals_u) == 1:
            out[key] = int(vals_u[0])
        else:
            conflicts += 1
    blank = int((df["text"].map(norm_text) == "").sum())
    return out, conflicts, blank


def key_score_map(df: Optional[pd.DataFrame], score_col: str = "llr") -> Tuple[Dict[str, float], int]:
    if df is None or "text" not in df.columns or score_col not in df.columns:
        return {}, 0
    work = df[["text", score_col]].copy()
    work["key"] = work["text"].map(norm_text)
    work["score_num"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work[work["key"].ne("") & np.isfinite(work["score_num"])]
    out: Dict[str, float] = {}
    conflicts = 0
    for key, vals in work.groupby("key")["score_num"]:
        unique = np.unique(vals.to_numpy(dtype=float))
        if len(unique) == 1:
            out[key] = float(unique[0])
        else:
            conflicts += 1
    return out, conflicts


def compare_maps(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, Any]:
    common = sorted(set(a) & set(b))
    if not common:
        return {"n_common": 0, "direct": None, "inverse": None, "mismatches": None}
    x = np.array([a[k] for k in common], dtype=int)
    y = np.array([b[k] for k in common], dtype=int)
    return {
        "n_common": len(common),
        "direct": float(np.mean(x == y)),
        "inverse": float(np.mean((1 - x) == y)),
        "mismatches": int(np.sum(x != y)),
    }


def find_named_files(root: Path, filename: str, max_results: int = 100) -> List[Path]:
    """Find named files while skipping common huge/noisy folders."""
    skip = {".git", ".venv", "venv", "__pycache__", "node_modules", ".cache"}
    found: List[Path] = []
    try:
        for p in root.rglob(filename):
            if any(part in skip for part in p.parts):
                continue
            if p.is_file():
                found.append(p)
                if len(found) >= max_results:
                    break
    except (PermissionError, OSError):
        pass
    return sorted(set(found))


def prefer_path(paths: Sequence[Path], terms: Sequence[str] = ()) -> Optional[Path]:
    if not paths:
        return None
    def rank(p: Path) -> Tuple[int, int, int]:
        s = str(p).replace("\\", "/").lower()
        term_score = sum(t.lower() in s for t in terms)
        # Prefer the project's known training_models tree when tie-breaking.
        known = int("/training_models/" in s)
        # Prefer shallower, more direct paths.
        return (term_score, known, -len(p.parts))
    return sorted(paths, key=rank, reverse=True)[0]


def choose_calibration_file(root: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.is_file() else None
    expected = [
        root / "training_models/detect_gpt_all_domain/inference_results/calibrated/detect_gpt_calibrated_all_results.csv",
        root / "detect_gpt_all_domain/inference_results/calibrated/detect_gpt_calibrated_all_results.csv",
        root / "training_models/detect_gpt/inference_results/calibrated/detect_gpt_calibrated_all_results.csv",
    ]
    for p in expected:
        if p.is_file():
            return p
    paths = find_named_files(root, "detect_gpt_calibrated_all_results.csv")
    paths = [p for p in paths if (p.parent / "score_cache").is_dir()]
    return prefer_path(paths, ("detect_gpt_all_domain", "calibrated"))


def discover_data_file(root: Path, dataset: str, split: str, llm: str) -> List[Path]:
    fname = f"{dataset}_{split}_{llm}.csv"
    candidates = find_named_files(root, fname)
    # Avoid detector output CSVs and score caches; keep raw input data only.
    filtered = []
    for p in candidates:
        s = str(p).replace("\\", "/").lower()
        if "score_cache" in s or "predictions.csv" in s or "inference_results" in s:
            continue
        if "perturb" in s:
            continue
        filtered.append(p)
    # Prioritize expected source roots.
    expected_parts = ("/training_models/training_data/" if split == "train" else "/training_models/testing_data/",)
    filtered.sort(key=lambda p: (any(x in str(p).replace("\\", "/").lower() for x in expected_parts), -len(p.parts)), reverse=True)
    return filtered


def choose_best_source(candidates: Sequence[Path], cache_df: Optional[pd.DataFrame]) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    cache_map, _, _ = unique_label_map(cache_df, "label")
    rows: List[Dict[str, Any]] = []
    scored: List[Tuple[Tuple[float, float, int], Path]] = []
    for p in candidates:
        df = read_csv(p)
        m, conflicts, blank = unique_label_map(df, "label")
        cmp = compare_maps(cache_map, m)
        coverage = cmp["n_common"] / max(1, len(cache_map))
        direct = cmp["direct"] if cmp["direct"] is not None else -1.0
        rows.append({
            "path": str(p), "rows": 0 if df is None else len(df),
            "unique_texts": len(m), "cache_overlap": cmp["n_common"],
            "coverage_of_cache": coverage, "direct_label_agreement": direct,
            "inverse_label_agreement": cmp["inverse"],
            "label_conflict_keys": conflicts, "blank_text_rows": blank,
            "sha256": stable_file_hash(p),
        })
        expected = int("/training_models/" in str(p).replace("\\", "/").lower())
        scored.append(((direct, coverage, expected), p))
    if not scored:
        return None, rows
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], rows


def threshold_from_scores(labels: np.ndarray, scores: np.ndarray, ai_label: int) -> Dict[str, float]:
    valid = np.isfinite(scores) & np.isin(labels, [0, 1])
    y = (labels[valid] == ai_label).astype(int)
    x = scores[valid].astype(float)
    if len(y) == 0 or len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return {"n": int(len(y)), "auc": math.nan, "threshold": math.nan, "youden_j": math.nan}
    fpr, tpr, thresholds = roc_curve(y, x)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, x)),
        "threshold": float(thresholds[idx]),
        "youden_j": float(j[idx]),
    }


def get_threshold_map(master: Optional[pd.DataFrame]) -> Tuple[Dict[Tuple[str, str], float], Dict[str, Any]]:
    out: Dict[Tuple[str, str], float] = {}
    issues: List[str] = []
    if master is None:
        return out, {"available": False, "issues": ["Calibration master CSV is missing/unreadable."]}
    required = {"train_dataset", "train_llm", "test_dataset", "test_llm", "threshold"}
    if not required.issubset(master.columns):
        return out, {"available": False, "issues": [f"Missing columns: {sorted(required - set(master.columns))}"]}
    in_domain = master[(master.train_dataset == master.test_dataset) & (master.train_llm == master.test_llm)]
    for (ds, llm), g in in_domain.groupby(["train_dataset", "train_llm"]):
        vals = pd.to_numeric(g.threshold, errors="coerce").dropna().unique()
        if len(vals) == 1:
            out[(str(ds), str(llm))] = float(vals[0])
        elif len(vals) == 0:
            issues.append(f"No finite threshold for {ds}-{llm}.")
        else:
            issues.append(f"Conflicting thresholds for {ds}-{llm}: {vals.tolist()}")
    return out, {"available": True, "in_domain_rows": int(len(in_domain)), "pairs_found": len(out), "issues": issues}


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray, scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else math.nan,
        "f1_ai": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)) if len(y_true) else math.nan,
        "ai_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)) if len(y_true) else math.nan,
        "human_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)) if len(y_true) else math.nan,
        "auroc_ai_higher_score": math.nan,
    }
    if scores is not None and len(y_true) and len(np.unique(y_true)) == 2:
        out["auroc_ai_higher_score"] = float(roc_auc_score(y_true, scores))
    return out


def resolve_paths(root: Path, calibration_arg: Optional[str]) -> Dict[str, Any]:
    calib_csv = choose_calibration_file(root, calibration_arg)
    calib_root = calib_csv.parent if calib_csv else None
    cache_root = calib_root / "score_cache" if calib_root else None
    train_root_candidates = [root / "training_models/training_data", root / "training_data"]
    test_root_candidates = [root / "training_models/testing_data", root / "testing_data"]
    train_root = next((p for p in train_root_candidates if p.is_dir()), None)
    test_root = next((p for p in test_root_candidates if p.is_dir()), None)

    new_prompt_candidates = [root / "prompt_change/new_prompt_data", root / "new_prompt_data"]
    new_prompt_root = next((p for p in new_prompt_candidates if p.is_dir()), None)
    if new_prompt_root is None:
        found = find_named_files(root, "hindisumm_gemini_new_prompt.csv")
        new_prompt_root = found[0].parent if found else None

    # RQ5 DetectGPT output path has varied between working-directory versions.
    rq5_candidates = [
        root / "prompt_change/detect_gpt/inference_results/calibrated/new_prompt",
        root / "prompt_change/detect_gpt_all_domain/inference_results/calibrated/new_prompt",
        root / "training_models/detect_gpt_all_domain/inference_results/calibrated/new_prompt",
        root / "prompt_change/detect_gpt/inference_results/new_prompt",
    ]
    rq5_root = next((p for p in rq5_candidates if p.is_dir()), None)
    if rq5_root is None:
        found = find_named_files(root, "detect_gpt_calibrated_new_prompt_results.csv")
        if found:
            rq5_root = found[0].parent

    coling_candidates = [
        root / "COLING_2025_MGT_Task1/hindi_data.csv",
        root / "coling/hindi_data.csv",
        root / "training_models/hindi_data.csv",
        root / "hindi_data.csv",
    ]
    coling_file = next((p for p in coling_candidates if p.is_file()), None)
    if coling_file is None:
        found = []
        for candidate in find_named_files(root, "hindi_data.csv"):
            candidate_df = read_csv(candidate)
            if candidate_df is not None and {"text", "label", "model"}.issubset(candidate_df.columns):
                found.append(candidate)
        coling_file = prefer_path(found, ("COLING_2025_MGT_Task1",))

    # RQ4 was run from CODE/COLING_2025_MGT_Task1, not from training_models.
    coling_calib_root_candidates = [
        root / "COLING_2025_MGT_Task1/detect_gpt_all_domain/inference_results/calibrated",
        root / "training_models/detect_gpt_all_domain/inference_results/calibrated",
        root / "detect_gpt_all_domain/inference_results/calibrated",
    ]
    rq4_calib_root = next((p for p in coling_calib_root_candidates if p.is_dir()), None)
    rq4_cache_root = rq4_calib_root / "score_cache" if rq4_calib_root else None

    coling_root_candidates = [
        (rq4_calib_root / "coling") if rq4_calib_root else None,
        (calib_root / "coling") if calib_root else None,
        root / "COLING_2025_MGT_Task1/detect_gpt_all_domain/inference_results/calibrated/coling",
        root / "training_models/detect_gpt_all_domain/inference_results/calibrated/coling",
        root / "detect_gpt_all_domain/inference_results/calibrated/coling",
    ]
    coling_root = next((p for p in coling_root_candidates if p is not None and p.is_dir()), None)

    return {
        "calibration_csv": calib_csv,
        "calibration_root": calib_root,
        "cache_root": cache_root,
        "train_root": train_root,
        "test_root": test_root,
        "new_prompt_root": new_prompt_root,
        "rq5_root": rq5_root,
        "coling_file": coling_file,
        "coling_root": coling_root,
        "rq4_calibration_root": rq4_calib_root,
        "rq4_cache_root": rq4_cache_root,
    }


def scan_rq123(root: Path, paths: Dict[str, Any], rq123_ai_label: int, rq123_pred_ai_label: int = 1) -> Dict[str, Any]:
    out: Dict[str, Any] = {"source_files": [], "threshold_checks": [], "prediction_checks": [], "issues": []}
    cache_root = paths["cache_root"]
    if cache_root is None or not cache_root.is_dir():
        out["issues"].append("RQ1-RQ3 score_cache folder not found.")
        return out
    master = read_csv(paths["calibration_csv"])
    thresholds, th_meta = get_threshold_map(master)
    out["threshold_master"] = th_meta
    out["thresholds"] = {f"{k[0]}-{k[1]}": v for k, v in thresholds.items()}

    cache_frames: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    for split in ("train", "test"):
        for ds in DATASETS:
            for llm in LLMS_RQ123:
                fname = f"{split}_{ds}_{llm}.csv"
                cache_file = cache_root / fname
                cache_df = read_csv(cache_file)
                if cache_df is None:
                    out["issues"].append(f"Missing score cache: {cache_file}")
                    continue
                cache_frames[(split, ds, llm)] = cache_df
                candidates = discover_data_file(root, ds, split, llm)
                best, candidate_info = choose_best_source(candidates, cache_df)
                cache_map, cache_conf, cache_blank = unique_label_map(cache_df, "label")
                best_df = read_csv(best)
                source_map, source_conf, source_blank = unique_label_map(best_df, "label")
                cmp = compare_maps(cache_map, source_map)
                out["source_files"].append({
                    "split": split, "dataset": ds, "llm": llm,
                    "cache_path": str(cache_file),
                    "best_matching_source_path": str(best) if best else None,
                    "cache_rows": len(cache_df), "cache_label_counts": dict(Counter(pd.to_numeric(cache_df.get("label"), errors="coerce").dropna().astype(int))) if "label" in cache_df else {},
                    "cache_unique_texts": len(cache_map), "source_candidates": candidate_info,
                    "best_direct_agreement": cmp["direct"], "best_inverse_agreement": cmp["inverse"],
                    "matched_texts": cmp["n_common"], "mismatches": cmp["mismatches"],
                    "cache_conflict_keys": cache_conf, "source_conflict_keys": source_conf,
                    "cache_blank_text_rows": cache_blank, "source_blank_text_rows": source_blank,
                    "declared_ai_raw_label": rq123_ai_label,
                })

    # Reproduce source thresholds from training score caches under AI_LABEL=1 and AI_LABEL=0.
    for ds in DATASETS:
        for llm in LLMS_RQ123:
            train = cache_frames.get(("train", ds, llm))
            saved_thr = thresholds.get((ds, llm))
            if train is None or not {"label", "llr"}.issubset(train.columns):
                out["threshold_checks"].append({"source": f"{ds}-{llm}", "status": "MISSING_TRAIN_CACHE_OR_COLUMNS"})
                continue
            labels = pd.to_numeric(train.label, errors="coerce").to_numpy(dtype=float)
            scores = pd.to_numeric(train.llr, errors="coerce").to_numpy(dtype=float)
            a1 = threshold_from_scores(labels, scores, 1)
            a0 = threshold_from_scores(labels, scores, 0)
            delta = abs(a1["threshold"] - saved_thr) if saved_thr is not None and np.isfinite(a1["threshold"]) else math.nan
            out["threshold_checks"].append({
                "source": f"{ds}-{llm}", "saved_threshold": saved_thr,
                "recomputed_if_ai_label_1": a1["threshold"], "train_auc_if_ai_label_1": a1["auc"], "train_youden_j_if_ai_label_1": a1["youden_j"],
                "recomputed_if_ai_label_0": a0["threshold"], "train_auc_if_ai_label_0": a0["auc"], "train_youden_j_if_ai_label_0": a0["youden_j"],
                "saved_matches_declared_ai_threshold": bool(np.isfinite(delta) and delta <= 1e-9),
                "threshold_abs_delta": delta,
                "interpretation": "Reproduction proves the numeric calibration procedure, not semantic label meaning.",
            })

    # Audit all RQ1-RQ3 (in-domain, one-axis LLM shift, one-axis dataset shift) predictions.
    for src_ds in DATASETS:
        for src_llm in LLMS_RQ123:
            for tgt_ds in DATASETS:
                for tgt_llm in LLMS_RQ123:
                    if src_ds != tgt_ds and src_llm != tgt_llm:
                        continue  # both-shift, not RQ1-RQ3
                    case = f"train_{src_ds}_{src_llm}__test_{tgt_ds}_{tgt_llm}"
                    pred_path = paths["calibration_root"] / case / "predictions.csv"
                    pred = read_csv(pred_path)
                    target_cache = cache_frames.get(("test", tgt_ds, tgt_llm))
                    if pred is None:
                        out["prediction_checks"].append({"case": case, "status": "MISSING_PREDICTIONS", "path": str(pred_path)})
                        continue
                    if not {"text", "label", "llr", "predicted_label"}.issubset(pred.columns):
                        out["prediction_checks"].append({"case": case, "status": "MISSING_REQUIRED_COLUMNS", "columns": list(pred.columns)})
                        continue
                    thr = thresholds.get((src_ds, src_llm))
                    llr = pd.to_numeric(pred.llr, errors="coerce").to_numpy(dtype=float)
                    raw_pred = pd.to_numeric(pred.predicted_label, errors="coerce").to_numpy(dtype=float)
                    mask = np.isfinite(llr) & np.isin(raw_pred, [0, 1])
                    expected = (np.where(llr[mask] >= thr, rq123_pred_ai_label, 1 - rq123_pred_ai_label).astype(int)
                                if thr is not None else np.array([], dtype=int))
                    rule_agree = float(np.mean(expected == raw_pred[mask])) if thr is not None and mask.any() else math.nan
                    pred_map, pred_conf, _ = unique_label_map(pred, "label")
                    cache_map, _, _ = unique_label_map(target_cache, "label") if target_cache is not None else ({}, 0, 0)
                    truth_cmp = compare_maps(pred_map, cache_map)
                    # Canonical labels: source truth is mapped from the shared
                    # input convention; saved old predictions retain their own
                    # DetectGPT output encoding (normally 1=AI).
                    raw_true = pd.to_numeric(pred.label, errors="coerce").to_numpy(dtype=float)
                    valid = np.isfinite(llr) & np.isin(raw_true, [0, 1]) & np.isin(raw_pred, [0, 1])
                    yt = (raw_true[valid] == rq123_ai_label).astype(int)
                    yp = (raw_pred[valid] == rq123_pred_ai_label).astype(int)
                    mm = metrics_binary(yt, yp, llr[valid]) if valid.any() else {}
                    out["prediction_checks"].append({
                        "case": case, "path": str(pred_path), "status": "READ",
                        "rows": len(pred), "raw_truth_counts": dict(Counter(pd.to_numeric(pred.label, errors="coerce").dropna().astype(int))),
                        "raw_pred_counts": dict(Counter(pd.to_numeric(pred.predicted_label, errors="coerce").dropna().astype(int))),
                        "threshold": thr, "prediction_rule_agreement": rule_agree,
                        "truth_vs_test_cache_direct": truth_cmp["direct"], "truth_vs_test_cache_inverse": truth_cmp["inverse"], "truth_cache_common": truth_cmp["n_common"],
                        "prediction_label_conflict_keys": pred_conf, **{f"canonical_{k}": v for k, v in mm.items()},
                    })
    return out


def find_coling_file(root: Path, preferred: Optional[Path]) -> Optional[Path]:
    if preferred and preferred.is_file():
        return preferred
    candidates = []
    for p in find_named_files(root, "hindi_data.csv"):
        df = read_csv(p)
        if df is not None and {"text", "label", "model"}.issubset(df.columns):
            candidates.append(p)
    return prefer_path(candidates, ("COLING_2025_MGT_Task1", "coling"))


def scan_rq4(root: Path, paths: Dict[str, Any], rq4_ai_label: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"metadata_check": {}, "prediction_checks": [], "issues": []}
    data_file = find_coling_file(root, paths["coling_file"])
    if data_file is None:
        out["issues"].append("Could not locate COLING hindi_data.csv with text,label,model columns.")
        return out
    df = read_csv(data_file)
    labels = pd.to_numeric(df.label, errors="coerce")
    models = df.model.astype(str).str.strip().str.lower()
    human_mask = models.eq("human")
    generated_mask = ~human_mask
    human_label_counts = dict(Counter(labels[human_mask & labels.isin([0, 1])].astype(int)))
    generated_label_counts = dict(Counter(labels[generated_mask & labels.isin([0, 1])].astype(int)))
    human_valid = labels[human_mask & labels.isin([0, 1])]
    generated_valid = labels[generated_mask & labels.isin([0, 1])]
    human_bad = int((human_valid == rq4_ai_label).sum())
    generated_expected = int((generated_valid == rq4_ai_label).sum())
    out["metadata_check"] = {
        "source_path": str(data_file), "rows": len(df), "raw_label_counts": dict(Counter(labels.dropna().astype(int))),
        "model_counts": dict(Counter(models)), "human_rows": int(human_mask.sum()),
        "human_raw_label_counts": human_label_counts, "nonhuman_model_rows": int(generated_mask.sum()),
        "nonhuman_raw_label_counts": generated_label_counts,
        "script_declared_ai_raw_label": rq4_ai_label,
        "human_rows_marked_ai": human_bad,
        "nonhuman_rows_marked_ai": generated_expected,
        "human_rows_with_valid_binary_label": int(len(human_valid)),
        "nonhuman_rows_with_valid_binary_label": int(len(generated_valid)),
        "label_mapping_supported_by_model_column": bool(
            len(human_valid) > 0 and len(generated_valid) > 0 and
            human_bad == 0 and generated_expected == len(generated_valid)
        ),
        "note": "The script's HINDI_AI_LABEL=0 is supported only when model=='human' rows are not label 0; the model column is the semantic evidence.",
    }
    source_map, source_conf, _ = unique_label_map(df, "label")
    canonical_source = {k: int(v == rq4_ai_label) for k, v in source_map.items()}
    calib_root = paths["calibration_root"]
    out_root = paths["coling_root"] or (calib_root / "coling" if calib_root else None)
    for ds in DATASETS:
        for llm in LLMS_RQ123:
            case = f"{ds}_{llm}_tested_on_coling_hindi"
            pred_path = (out_root / case / "predictions.csv") if out_root else None
            pred = read_csv(pred_path)
            if pred is None:
                out["prediction_checks"].append({"case": case, "status": "MISSING", "path": str(pred_path) if pred_path else None})
                continue
            if "text" not in pred.columns or "true_label" not in pred.columns or "predicted_label" not in pred.columns:
                out["prediction_checks"].append({"case": case, "status": "MISSING_REQUIRED_COLUMNS", "path": str(pred_path), "columns": list(pred.columns)})
                continue
            ptrue = pd.to_numeric(pred.get("true_label"), errors="coerce")
            ppred = pd.to_numeric(pred.get("predicted_label"), errors="coerce")
            pllr = pd.to_numeric(pred.get("llr"), errors="coerce") if "llr" in pred else pd.Series(np.nan, index=pred.index)
            work = pred.assign(_key=pred.text.map(norm_text), _true=ptrue)
            mapped = work[work._key.isin(canonical_source)].copy()
            agrees = (mapped._true.astype(int).to_numpy() == mapped._key.map(canonical_source).to_numpy(dtype=int)) if len(mapped) else np.array([], dtype=bool)
            out["prediction_checks"].append({
                "case": case, "status": "READ", "path": str(pred_path), "rows": len(pred),
                "saved_true_counts": dict(Counter(ptrue.dropna().astype(int))),
                "saved_pred_counts": dict(Counter(ppred.dropna().astype(int))),
                "llr_rows": int(pllr.notna().sum()), "source_overlap": len(mapped),
                "truth_matches_coling_metadata": float(np.mean(agrees)) if len(agrees) else None,
                "truth_mismatches": int((~agrees).sum()) if len(agrees) else None,
                "expected_high_score_prediction_ai": "predicted_label should be 1 when llr >= source threshold",
            })
    return out


def find_sibling_prompt_predictions(root: Path, detector: str, src_ds: str, src_llm: str, tgt_ds: str, tgt_llm: str) -> Optional[Path]:
    case = f"{src_ds}_{src_llm}_tested_on_{tgt_ds}_{tgt_llm}"
    if detector.lower() == "gptzero":
        bases = [root / "prompt_change/gptzero/inference_results_new_prompt", root / "prompt_change/gptzero/inference_results/calibrated/new_prompt"]
    else:
        bases = [root / "prompt_change/xlm_roberta/inference_results_new_prompt", root / "prompt_change/xlm_roberta/inference_results/new_prompt"]
    for b in bases:
        p = b / case / "predictions.csv"
        if p.is_file():
            return p
    # Fallback exact folder search.
    for p in find_named_files(root, "predictions.csv"):
        s = str(p).replace("\\", "/").lower()
        if detector.lower() in s and "prompt" in s and case.lower() in s:
            return p
    return None


def infer_rq5_ai_label(root: Path, paths: Dict[str, Any], rq123_ai_label: int, min_overlap: int = 15, agreement_required: float = 0.98) -> Dict[str, Any]:
    """Infer raw new-prompt AI label from unchanged known-Human overlap.

    Known Human examples come from DetectGPT's original test score cache, using
    the RQ1-RQ3 convention declared by the DetectGPT README/script. If overlap
    is insufficient, the routine reports that rather than inventing a label.
    """
    report: Dict[str, Any] = {"targets": [], "global_candidate_ai_label": None, "status": "UNRESOLVED", "evidence": []}
    cache_root = paths["cache_root"]
    new_root = paths["new_prompt_root"]
    if cache_root is None or new_root is None:
        report["evidence"].append("Missing score_cache or new_prompt_data root.")
        return report

    candidate_counts: Counter = Counter()
    all_matches = 0
    strict_targets = 0
    for ds in DATASETS:
        for llm in LLMS_RQ5:
            old_cache_path = cache_root / f"test_{ds}_{llm}.csv"
            new_path = new_root / f"{ds}_{llm}_new_prompt.csv"
            old_df, new_df = read_csv(old_cache_path), read_csv(new_path)
            if old_df is None or new_df is None or not {"text", "label"}.issubset(old_df.columns) or not {"text", "label"}.issubset(new_df.columns):
                report["targets"].append({"target": f"{ds}-{llm}", "status": "MISSING_INPUT", "old_cache": str(old_cache_path), "new_prompt": str(new_path)})
                continue
            old_labels, old_conf, _ = unique_label_map(old_df, "label")
            new_labels, new_conf, _ = unique_label_map(new_df, "label")
            old_human_label = 1 - rq123_ai_label
            human_keys = {k for k, v in old_labels.items() if v == old_human_label}
            matched = sorted(human_keys & set(new_labels))
            new_on_human = [new_labels[k] for k in matched]
            counts = Counter(new_on_human)
            if matched:
                hum_raw = counts.most_common(1)[0][0]
                agreement = counts[hum_raw] / len(matched)
                ai_candidate = 1 - int(hum_raw)
                candidate_counts[ai_candidate] += len(matched)
                all_matches += len(matched)
                passed = len(matched) >= min_overlap and agreement >= agreement_required
                if passed:
                    strict_targets += 1
            else:
                hum_raw, agreement, ai_candidate, passed = None, None, None, False
            # Cross-check shared new-prompt convention against GPTZero canonical outputs.
            sibling_path = find_sibling_prompt_predictions(root, "gptzero", ds, llm, ds, llm)
            sibling_check = None
            if sibling_path:
                sib = read_csv(sibling_path)
                if sib is not None and {"text", "true_label"}.issubset(sib.columns):
                    sib_map, _, _ = unique_label_map(sib, "true_label")
                    raw_new_map = new_labels
                    common = set(sib_map) & set(raw_new_map)
                    if common:
                        direct0 = float(np.mean([sib_map[k] == (raw_new_map[k] == 0) for k in common]))
                        direct1 = float(np.mean([sib_map[k] == (raw_new_map[k] == 1) for k in common]))
                        sibling_check = {"path": str(sibling_path), "n_common": len(common), "agreement_if_raw_ai_0": direct0, "agreement_if_raw_ai_1": direct1}
            report["targets"].append({
                "target": f"{ds}-{llm}", "status": "READ",
                "old_test_cache_path": str(old_cache_path), "new_prompt_path": str(new_path),
                "old_raw_label_counts": dict(Counter(pd.to_numeric(old_df.label, errors="coerce").dropna().astype(int))),
                "new_prompt_raw_label_counts": dict(Counter(pd.to_numeric(new_df.label, errors="coerce").dropna().astype(int))),
                "assumed_old_ai_raw_label": rq123_ai_label, "old_human_raw_label_used_for_match": old_human_label,
                "known_human_text_matches": len(matched), "new_prompt_labels_on_known_humans": dict(counts),
                "dominant_new_prompt_human_label": hum_raw, "homogeneity": agreement,
                "inferred_new_prompt_ai_label": ai_candidate, "target_overlap_pass": bool(passed),
                "old_cache_label_conflicts": old_conf, "new_prompt_label_conflicts": new_conf,
                "gptzero_cross_script_canonical_truth_check": sibling_check,
            })

    # Require consistent candidate across every target with enough matched known-Human examples.
    passing = [r for r in report["targets"] if r.get("target_overlap_pass")]
    # Add whether GPTZero cross-script evidence supports a raw new-prompt convention.
    sibling_values = []
    sibling_strengths = []
    for r in report["targets"]:
        sc = r.get("gptzero_cross_script_canonical_truth_check")
        if sc and sc.get("n_common", 0) > 0:
            a0, a1 = sc["agreement_if_raw_ai_0"], sc["agreement_if_raw_ai_1"]
            implied = 0 if a0 > a1 else 1 if a1 > a0 else None
            best = max(a0, a1)
            sibling_values.append(implied)
            sibling_strengths.append({"target": r.get("target"), "implied_ai_label": implied, "agreement": best, "n_common": sc["n_common"]})
    report["gptzero_implied_raw_ai_labels"] = sibling_values
    report["gptzero_sibling_cross_checks"] = sibling_strengths

    if passing:
        candidates = {r["inferred_new_prompt_ai_label"] for r in passing}
        if len(candidates) == 1:
            candidate = int(next(iter(candidates)))
            report["global_candidate_ai_label"] = candidate
            report["status"] = "SUPPORTED_BY_HUMAN_TEXT_OVERLAP" if len(passing) == 4 else "PARTIALLY_SUPPORTED_BY_HUMAN_TEXT_OVERLAP"
            report["targets_passing"] = len(passing)
            report["known_human_matches_total"] = sum(int(r["known_human_text_matches"]) for r in passing)
            if sibling_values and any(x is not None and x != candidate for x in sibling_values):
                report["status"] = "CONFLICTING_TARGET_MAPPINGS"
                report["global_candidate_ai_label"] = None
                report["evidence"].append(f"Human-overlap mapping {candidate} conflicts with GPTZero cross-script mapping(s) {sibling_values}.")
        else:
            report["status"] = "CONFLICTING_TARGET_MAPPINGS"
            report["evidence"].append(f"Different new-prompt AI labels inferred across targets: {sorted(candidates)}")
    else:
        # If exact unchanged-human overlap is sparse, accept a shared label
        # convention only when all four GPTZero in-domain outputs independently
        # map the new-prompt raw labels to the same canonical truth with >=99.9%
        # agreement. This is cross-script consistency, not external ground truth.
        valid_sibling = [x for x in sibling_strengths if x["implied_ai_label"] is not None and x["agreement"] >= 0.999]
        sibling_classes = {x["implied_ai_label"] for x in valid_sibling}
        if len(valid_sibling) == 4 and len(sibling_classes) == 1:
            report["global_candidate_ai_label"] = int(next(iter(sibling_classes)))
            report["status"] = "SUPPORTED_BY_GPTZERO_SHARED_LABEL_PIPELINE_ONLY"
            report["targets_passing"] = 0
            report["known_human_matches_total"] = all_matches
            report["evidence"].append("No sufficient unchanged-human overlap; raw label convention inferred from four in-domain GPTZero canonical-truth outputs. This is cross-script consistency, not an independent human-text validation.")
        else:
            report["status"] = "INSUFFICIENT_KNOWN_HUMAN_OVERLAP"
            report["evidence"].append(f"No target met both min_overlap={min_overlap} and homogeneity>={agreement_required:.1%}; four consistent GPTZero in-domain cross-checks were not available.")
    return report


def locate_rq5_predictions(root: Path, rq5_root: Optional[Path], ds: str, llm: str, src_ds: str, src_llm: str) -> Optional[Path]:
    case = f"{src_ds}_{src_llm}_tested_on_{ds}_{llm}"
    bases = [rq5_root] if rq5_root else []
    for b in bases:
        p = b / case / "predictions.csv"
        if p.is_file():
            return p
    # A few checked-in versions used a different parent path.
    for b in [root / "prompt_change/detect_gpt/inference_results/calibrated/new_prompt",
              root / "prompt_change/detect_gpt_all_domain/inference_results/calibrated/new_prompt",
              root / "training_models/detect_gpt_all_domain/inference_results/calibrated/new_prompt"]:
        p = b / case / "predictions.csv"
        if p.is_file():
            return p
    # Search exact case folder, detector-specific only.
    for p in find_named_files(root, "predictions.csv"):
        s = str(p).replace("\\", "/").lower()
        if case.lower() in s and "detect_gpt" in s and "prompt" in s:
            return p
    return None



def rebuild_rq123_from_cache(paths: Dict[str, Any], rq123_ai_label: int, outdir: Path) -> Dict[str, Any]:
    """Recalibrate and rebuild RQ1-RQ3 predictions from existing scores only."""
    cache_root = paths["cache_root"]
    if cache_root is None:
        return {"status": "FAILED", "reason": "score_cache not found", "cases_written": 0, "thresholds": {}}
    base = outdir / "rebuilt_from_score_cache" / "rq123"
    base.mkdir(parents=True, exist_ok=True)
    thresholds: Dict[Tuple[str, str], float] = {}
    threshold_rows: List[Dict[str, Any]] = []
    for ds in DATASETS:
        for llm in LLMS_RQ123:
            df = read_csv(cache_root / f"train_{ds}_{llm}.csv")
            if df is None or not {"label", "llr", "text"}.issubset(df.columns):
                return {"status": "FAILED", "reason": f"Missing or incomplete train score cache for {ds}-{llm}", "cases_written": 0, "thresholds": {}}
            labels = pd.to_numeric(df.label, errors="coerce").to_numpy(dtype=float)
            scores = pd.to_numeric(df.llr, errors="coerce").to_numpy(dtype=float)
            calc = threshold_from_scores(labels, scores, rq123_ai_label)
            if not np.isfinite(calc["threshold"]):
                return {"status": "FAILED", "reason": f"Cannot calibrate threshold for {ds}-{llm}; check classes/scores.", "cases_written": 0, "thresholds": {}}
            thresholds[(ds, llm)] = float(calc["threshold"])
            valid_labels = labels[np.isfinite(labels) & np.isin(labels, [0, 1])]
            threshold_rows.append({
                "train_dataset": ds, "train_llm": llm,
                "raw_ai_label": rq123_ai_label,
                "threshold": calc["threshold"], "train_youden_j": calc["youden_j"],
                "train_auroc_ai_higher_score": calc["auc"], "n_calibration": calc["n"],
                "raw_label_0_count": int(np.sum(valid_labels == 0)),
                "raw_label_1_count": int(np.sum(valid_labels == 1)),
                "ai_count": int(np.sum(valid_labels == rq123_ai_label)),
                "human_count": int(np.sum(valid_labels == (1-rq123_ai_label))),
            })
    pd.DataFrame(threshold_rows).to_csv(base / "source_thresholds_recomputed.csv", index=False)

    master_rows: List[Dict[str, Any]] = []
    for src_ds in DATASETS:
        for src_llm in LLMS_RQ123:
            thr = thresholds[(src_ds, src_llm)]
            for tgt_ds in DATASETS:
                for tgt_llm in LLMS_RQ123:
                    if src_ds != tgt_ds and src_llm != tgt_llm:
                        continue
                    test_df = read_csv(cache_root / f"test_{tgt_ds}_{tgt_llm}.csv")
                    if test_df is None or not {"text", "label", "llr"}.issubset(test_df.columns):
                        continue
                    raw_labels = pd.to_numeric(test_df.label, errors="coerce").to_numpy(dtype=float)
                    scores = pd.to_numeric(test_df.llr, errors="coerce").to_numpy(dtype=float)
                    valid = np.isfinite(raw_labels) & np.isin(raw_labels, [0, 1]) & np.isfinite(scores) & test_df.text.notna().to_numpy()
                    if not valid.any():
                        continue
                    raw = raw_labels[valid].astype(int)
                    y_true = (raw == rq123_ai_label).astype(int)
                    y_pred = (scores[valid] >= thr).astype(int)
                    met = metrics_binary(y_true, y_pred, scores[valid])
                    setting = "in_domain" if (src_ds == tgt_ds and src_llm == tgt_llm) else ("llm_cross" if src_ds == tgt_ds else "dataset_cross")
                    case = f"train_{src_ds}_{src_llm}__test_{tgt_ds}_{tgt_llm}"
                    case_dir = base / case
                    case_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({
                        "text": test_df.loc[valid, "text"].astype(str).to_numpy(),
                        "raw_label": raw,
                        "true_label": y_true,
                        "predicted_label": y_pred,
                        "llr": scores[valid],
                        "threshold": thr,
                    }).to_csv(case_dir / "predictions.csv", index=False)
                    row = {
                        "train_dataset": src_ds, "train_llm": src_llm,
                        "test_dataset": tgt_ds, "test_llm": tgt_llm,
                        "setting": setting, "raw_ai_label": rq123_ai_label,
                        "threshold": thr, **met,
                    }
                    pd.DataFrame([row]).to_csv(case_dir / "metrics.csv", index=False)
                    master_rows.append(row)
    if master_rows:
        pd.DataFrame(master_rows).to_csv(base / "rq123_recomputed_metrics.csv", index=False)
    return {"status": "COMPLETED", "cases_written": len(master_rows), "expected_cases": 24,
            "thresholds": {f"{k[0]}-{k[1]}": v for k,v in thresholds.items()}, "output_root": str(base)}


def align_rq4_scores_to_source(raw_df: pd.DataFrame, score_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, float]:
    """Align RQ4 scores to COLING rows without collapsing repeated texts.

    Prefer exact row-order alignment when normalized text sequences match.
    Otherwise use (normalized_text, occurrence_index), which keeps repeated
    identical texts as separate scored rows instead of rejecting/deduplicating
    them. Returns source rows with an llr column, alignment mode, and coverage.
    """
    if not {"text"}.issubset(raw_df.columns) or not {"text", "llr"}.issubset(score_df.columns):
        return pd.DataFrame(), "missing_columns", 0.0

    left = raw_df.copy().reset_index(drop=True)
    right = score_df[["text", "llr"]].copy().reset_index(drop=True)
    left["_key"] = left["text"].map(norm_text)
    right["_key"] = right["text"].map(norm_text)
    right["_llr"] = pd.to_numeric(right["llr"], errors="coerce")

    # Safest case: scoring dataframe retained source row order. This is how
    # the COLING scorer writes its cache/prediction CSVs.
    if len(left) == len(right) and np.array_equal(left["_key"].to_numpy(), right["_key"].to_numpy()):
        left["llr"] = right["_llr"].to_numpy(dtype=float)
        coverage = float(left["llr"].notna().sum() / max(1, len(left)))
        return left, "exact_row_order", coverage

    # Fallback: occurrence-index join preserves duplicate text rows in their
    # within-file order (e.g. repeated COLING article text).
    left["_occ"] = left.groupby("_key", sort=False).cumcount()
    right["_occ"] = right.groupby("_key", sort=False).cumcount()
    left = left[left["_key"].ne("")].copy()
    right = right[right["_key"].ne("")].copy()
    aligned = left.merge(right[["_key", "_occ", "_llr"]], on=["_key", "_occ"], how="left", validate="one_to_one", sort=False)
    aligned = aligned.sort_index() if aligned.index.is_monotonic_increasing else aligned
    aligned["llr"] = aligned["_llr"]
    coverage = float(aligned["llr"].notna().sum() / max(1, len(raw_df)))
    return aligned, "text_occurrence_order", coverage


def rebuild_rq4_from_cache(paths: Dict[str, Any], rq4_ai_label: int, thresholds: Dict[Tuple[str, str], float], outdir: Path) -> Dict[str, Any]:
    """Rebuild RQ4 from cached COLING llr scores / saved per-target llr; no inference."""
    data_file = paths.get("coling_file")
    if data_file is None or not data_file.is_file():
        return {"status": "FAILED", "reason": "COLING hindi_data.csv not found", "cases_written": 0}
    raw_df = read_csv(data_file)
    if raw_df is None or not {"text", "label", "model"}.issubset(raw_df.columns):
        return {"status": "FAILED", "reason": "COLING source must have text,label,model", "cases_written": 0}
    labels = pd.to_numeric(raw_df.label, errors="coerce")
    models = raw_df.model.astype(str).str.strip().str.lower()
    human = models.eq("human")
    generated = ~human
    human_valid = labels[human & labels.isin([0,1])]
    generated_valid = labels[generated & labels.isin([0,1])]
    mapping_ok = len(human_valid)>0 and len(generated_valid)>0 and (human_valid==1-rq4_ai_label).all() and (generated_valid==rq4_ai_label).all()
    if not mapping_ok:
        return {"status": "BLOCKED_LABEL_METADATA_CHECK", "reason": f"COLING model column does not fully support raw AI label={rq4_ai_label}", "cases_written": 0,
                "human_labels": dict(Counter(human_valid.astype(int))), "generated_labels": dict(Counter(generated_valid.astype(int)))}
    # RQ4 has one repeated normalized COLING text key in the source set.
    # Do NOT collapse it into a text->score dictionary: DetectGPT perturbations
    # are stochastic, so duplicate rows may have different valid llr scores.
    # Prefer the one-pass COLING score cache and align row-wise (or by duplicate
    # occurrence number); fall back to an existing RQ4 prediction file.
    candidate_frames: List[Tuple[str, pd.DataFrame]] = []
    cache_root = paths.get("rq4_cache_root") or paths.get("cache_root")
    if cache_root is not None:
        cache_candidate = read_csv(cache_root / "coling_hindi.csv")
        if cache_candidate is not None and {"text", "llr"}.issubset(cache_candidate.columns):
            candidate_frames.append((str(cache_root / "coling_hindi.csv"), cache_candidate))
    coling_root = paths.get("coling_root")
    if coling_root is not None:
        for ds in DATASETS:
            for llm in LLMS_RQ123:
                candidate_path = coling_root / f"{ds}_{llm}_tested_on_coling_hindi" / "predictions.csv"
                candidate = read_csv(candidate_path)
                if candidate is not None and {"text", "llr"}.issubset(candidate.columns):
                    candidate_frames.append((str(candidate_path), candidate))

    if not candidate_frames:
        return {"status": "FAILED", "reason": "Neither COLING score cache nor saved RQ4 llr outputs found", "cases_written": 0}

    aligned_candidates: List[Tuple[str, pd.DataFrame, str, float]] = []
    for source_path, candidate_df in candidate_frames:
        aligned_df, mode, coverage = align_rq4_scores_to_source(raw_df, candidate_df)
        aligned_candidates.append((source_path, aligned_df, mode, coverage))

    # The six RQ4 outputs should reuse the same one-time COLING scoring pass.
    # When multiple score sources cover the same source rows, require agreement.
    usable = [x for x in aligned_candidates if x[3] >= 0.98]
    if not usable:
        details = [{"path": x[0], "mode": x[2], "coverage": x[3]} for x in aligned_candidates]
        return {"status": "BLOCKED_LOW_SCORE_JOIN_COVERAGE", "reason": "No RQ4 score source aligned to at least 98% of COLING rows", "cases_written": 0, "candidate_alignments": details}

    chosen_path, chosen_aligned, chosen_mode, chosen_coverage = usable[0]
    chosen = chosen_aligned.copy()
    # Compare all other usable arrays where both have finite values. This catches
    # stale/mixed RQ4 prediction folders without assuming duplicate text scores
    # are identical across independent stochastic runs.
    for other_path, other_aligned, other_mode, other_coverage in usable[1:]:
        n = min(len(chosen), len(other_aligned))
        if n == len(chosen) == len(other_aligned):
            a = pd.to_numeric(chosen.llr, errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(other_aligned.llr, errors="coerce").to_numpy(dtype=float)
            both = np.isfinite(a) & np.isfinite(b)
            if both.any() and not np.allclose(a[both], b[both], rtol=1e-7, atol=1e-9):
                return {"status": "BLOCKED_INCONSISTENT_RQ4_SCORE_SOURCES", "reason": f"RQ4 score arrays disagree between {chosen_path} and {other_path}", "cases_written": 0, "chosen_alignment": chosen_mode, "other_alignment": other_mode}

    merged = chosen.copy()
    merged["_raw_label"] = pd.to_numeric(merged["label"], errors="coerce")
    merged["_model"] = merged["model"].astype(str)
    merged = merged[merged["_raw_label"].isin([0,1]) & np.isfinite(pd.to_numeric(merged["llr"], errors="coerce"))].copy()
    merged["llr"] = pd.to_numeric(merged["llr"], errors="coerce")
    if len(merged) / max(1, len(raw_df)) < 0.98:
        return {"status": "BLOCKED_LOW_SCORE_JOIN_COVERAGE", "reason": f"Only {len(merged)}/{len(raw_df)} COLING source rows have finite aligned scores", "cases_written": 0, "score_source": chosen_path, "alignment_mode": chosen_mode}
    base = outdir / "rebuilt_from_score_cache" / "rq4"
    base.mkdir(parents=True, exist_ok=True)
    rows=[]
    for ds in DATASETS:
        for llm in LLMS_RQ123:
            thr=thresholds.get((ds,llm))
            if thr is None:
                continue
            y_true=(merged._raw_label.astype(int).to_numpy()==rq4_ai_label).astype(int)
            scores=merged.llr.to_numpy(dtype=float)
            y_pred=(scores>=thr).astype(int)
            met=metrics_binary(y_true,y_pred,scores)
            case=f"{ds}_{llm}_tested_on_coling_hindi"
            case_dir=base/case; case_dir.mkdir(parents=True,exist_ok=True)
            pd.DataFrame({"text":merged.text.astype(str).to_numpy(),"raw_label":merged._raw_label.astype(int).to_numpy(),
                          "model":merged._model.to_numpy(),"true_label":y_true,"predicted_label":y_pred,
                          "llr":scores,"threshold":thr}).to_csv(case_dir/"predictions.csv",index=False)
            row={"Train Dataset":ds,"Train LLM":llm,"Test Dataset":"coling_hindi","Test LLM":"gpt-4o",
                 "Threshold":thr,"N Evaluated":len(merged),"RQ4_Raw_AI_Label":rq4_ai_label,**met}
            pd.DataFrame([row]).to_csv(case_dir/"metrics.csv",index=False); rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(base/"rq4_recomputed_metrics.csv",index=False)
    return {"status":"COMPLETED","cases_written":len(rows),"expected_cases":6,"output_root":str(base),
            "score_source":chosen_path,"alignment_mode":chosen_mode,"score_join_coverage":chosen_coverage,
            "duplicate_text_rows_preserved":int(merged["_key"].duplicated(keep=False).sum()) if "_key" in merged.columns else None}

def repair_rq5(root: Path, paths: Dict[str, Any], rq5_ai_label: int, thresholds: Dict[Tuple[str, str], float], outdir: Path, min_join_coverage: float = 0.98, allow_rethreshold: bool = False) -> Dict[str, Any]:
    """Write corrected canonical RQ5 outputs into a new directory."""
    new_root = paths["new_prompt_root"]
    if new_root is None:
        return {"status": "FAILED", "reason": "new_prompt_data root missing", "cases_written": 0}
    repair_root = outdir / "rq5_repaired"
    repair_root.mkdir(parents=True, exist_ok=True)
    master_rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    errors: List[str] = []

    for tgt_ds in DATASETS:
        for tgt_llm in LLMS_RQ5:
            new_path = new_root / f"{tgt_ds}_{tgt_llm}_new_prompt.csv"
            new_df = read_csv(new_path)
            if new_df is None:
                errors.append(f"Missing new-prompt source: {new_path}")
                continue
            src_map, src_conf, _ = unique_label_map(new_df, "label")
            if not src_map:
                errors.append(f"No unique source labels in {new_path}")
                continue
            for src_ds in DATASETS:
                for src_llm in LLMS_RQ5:
                    if src_ds != tgt_ds and src_llm != tgt_llm:
                        continue  # both shift excluded from Table 7
                    src_pair = (src_ds, src_llm)
                    thr = thresholds.get(src_pair)
                    pred_path = locate_rq5_predictions(root, paths["rq5_root"], tgt_ds, tgt_llm, src_ds, src_llm)
                    pred_df = read_csv(pred_path)
                    case = f"{src_ds}_{src_llm}_tested_on_{tgt_ds}_{tgt_llm}"
                    if thr is None:
                        errors.append(f"No source calibrated threshold for {src_pair}")
                        continue
                    if pred_df is None or not {"text", "llr"}.issubset(pred_df.columns):
                        errors.append(f"Missing RQ5 prediction/llr file for {case}: {pred_path}")
                        continue
                    # Check the threshold written into this result against the original
                    # calibrated source threshold before rebuilding any labels/metrics.
                    if "threshold" in pred_df.columns:
                        saved_thresholds = pd.to_numeric(pred_df.threshold, errors="coerce").dropna().unique()
                        threshold_mismatch = (len(saved_thresholds) != 1 or abs(float(saved_thresholds[0]) - float(thr)) > 1e-9)
                        if threshold_mismatch and not allow_rethreshold:
                            errors.append(f"Saved RQ5 threshold does not match calibration master for {case}: saved={saved_thresholds.tolist()}, master={thr}; not repaired")
                            manifest.append({"case": case, "status": "THRESHOLD_MISMATCH", "saved_thresholds": saved_thresholds.tolist(), "master_threshold": thr})
                            continue
                    else:
                        threshold_mismatch = False
                    # Ensure every scored text can be linked unambiguously to raw new-prompt label.
                    work = pred_df.copy()
                    work["_key"] = work.text.map(norm_text)
                    work["_llr"] = pd.to_numeric(work.llr, errors="coerce")
                    work = work[work._key.ne("") & np.isfinite(work._llr)]
                    n_before = len(work)
                    work["_raw_label"] = work._key.map(src_map)
                    joined = work.dropna(subset=["_raw_label"]).copy()
                    coverage = len(joined) / max(1, n_before)
                    if coverage < min_join_coverage:
                        errors.append(f"Join coverage {coverage:.2%} below {min_join_coverage:.2%} for {case}; not repaired")
                        manifest.append({"case": case, "status": "JOIN_COVERAGE_TOO_LOW", "rows_scored": n_before, "rows_joined": len(joined), "join_coverage": coverage, "source_label_conflict_keys": src_conf})
                        continue
                    if len(joined) == 0 or len(set(joined._raw_label.astype(int))) < 2:
                        errors.append(f"Missing one or both raw classes after join for {case}")
                        continue
                    y_true = (joined._raw_label.astype(int).to_numpy() == rq5_ai_label).astype(int)
                    scores = joined._llr.to_numpy(dtype=float)
                    y_pred = (scores >= thr).astype(int)  # canonical 1=AI; high llr=AI
                    saved_pred_agreement = None
                    saved_true_agreement = None
                    if "true_label" in joined.columns:
                        savedt = pd.to_numeric(joined.true_label, errors="coerce").to_numpy(dtype=float)
                        mt = np.isfinite(savedt) & np.isin(savedt, [0, 1])
                        if mt.any():
                            saved_true_agreement = float(np.mean(savedt[mt].astype(int) == y_true[mt]))
                    if "predicted_label" in joined.columns:
                        savedp = pd.to_numeric(joined.predicted_label, errors="coerce").to_numpy(dtype=float)
                        m = np.isfinite(savedp) & np.isin(savedp, [0, 1])
                        if m.any():
                            saved_pred_agreement = float(np.mean(savedp[m].astype(int) == y_pred[m]))
                    metrics = metrics_binary(y_true, y_pred, scores)
                    out_pred = pd.DataFrame({
                        "text": joined.text.astype(str).to_numpy(),
                        "true_label": y_true,
                        "predicted_label": y_pred,
                        "llr": scores,
                        "threshold": thr,
                        "raw_new_prompt_label": joined._raw_label.astype(int).to_numpy(),
                    })
                    case_dir = repair_root / case
                    case_dir.mkdir(parents=True, exist_ok=True)
                    out_pred.to_csv(case_dir / "predictions.csv", index=False)
                    metric_row = {
                        "Train Dataset": src_ds, "Train LLM": src_llm,
                        "Test Dataset": tgt_ds, "Test LLM": tgt_llm,
                        "Setting": "in_domain" if (src_ds == tgt_ds and src_llm == tgt_llm) else ("llm_cross" if src_ds == tgt_ds else "dataset_cross"),
                        "Threshold": thr, "N Evaluated": len(joined),
                        "Accuracy": metrics["accuracy"], "F1 Score": metrics["f1_ai"],
                        "AUROC": metrics["auroc_ai_higher_score"], "AI Recall": metrics["ai_recall"],
                        "Human Recall": metrics["human_recall"],
                        "RQ5_Raw_AI_Label": rq5_ai_label,
                        "Join_Coverage": coverage,
                        "Saved_Prediction_Rule_Agreement": saved_pred_agreement,
                        "Saved_True_Label_Agreement_After_Remap": saved_true_agreement,
                        "Saved_Threshold_Mismatch": bool(threshold_mismatch),
                    }
                    pd.DataFrame([metric_row]).to_csv(case_dir / "metrics.csv", index=False)
                    master_rows.append(metric_row)
                    manifest.append({"case": case, "status": "WRITTEN", "prediction_source": str(pred_path), "new_prompt_source": str(new_path), "rows_scored": n_before, "rows_joined": len(joined), "join_coverage": coverage, "threshold": thr, "saved_threshold_mismatch": bool(threshold_mismatch), "saved_true_label_agreement_after_remap": saved_true_agreement, "saved_prediction_rule_agreement": saved_pred_agreement, "output_dir": str(case_dir)})
    if master_rows:
        pd.DataFrame(master_rows).to_csv(repair_root / "detect_gpt_calibrated_new_prompt_results_REPAIRED.csv", index=False)
    pd.DataFrame(manifest).to_csv(repair_root / "repair_manifest.csv", index=False)
    return {"status": "COMPLETED_WITH_WARNINGS" if errors else "COMPLETED", "cases_written": len(master_rows), "expected_table7_cases": 12, "repair_root": str(repair_root), "errors": errors, "manifest_rows": len(manifest)}


def flatten_report(report: Dict[str, Any]) -> str:
    lines = [
        "# DetectGPT audit report", "",
        f"Generated: {report['generated_at']}",
        f"Project root: `{report['project_root']}`", "",
        "## Executive status", "",
    ]
    for k, v in report["executive_status"].items():
        lines.append(f"- **{k}:** {v}")
    lines += ["", "## Discovered paths", ""]
    for k, v in report["paths"].items():
        lines.append(f"- **{k}:** `{v}`")
    lines += ["", "## RQ1–RQ3: source/cache lineage", "",
              "| Split | Dataset | LLM | Cache rows | Best source file | Overlap | Direct agreement | Inverse agreement |", "|---|---|---|---:|---|---:|---:|---:|"]
    for r in report["rq123"].get("source_files", []):
        lines.append(f"| {r['split']} | {r['dataset']} | {r['llm']} | {r['cache_rows']} | `{r['best_matching_source_path']}` | {r['matched_texts']} | {fmt_pct(r['best_direct_agreement'])} | {fmt_pct(r['best_inverse_agreement'])} |")
    lines += ["", "### Threshold reproduction", "",
              "| Source | Saved threshold | Recomputed if raw AI=1 | Train AUC (AI=1) | Youden J (AI=1) | Recomputed if raw AI=0 | Saved matches declared mapping? |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in report["rq123"].get("threshold_checks", []):
        lines.append(f"| {r.get('source')} | {fmt_num(r.get('saved_threshold'))} | {fmt_num(r.get('recomputed_if_ai_label_1'))} | {fmt_num(r.get('train_auc_if_ai_label_1'))} | {fmt_num(r.get('train_youden_j_if_ai_label_1'))} | {fmt_num(r.get('recomputed_if_ai_label_0'))} | {r.get('saved_matches_declared_ai_threshold')} |")
    lines += ["", "### Prediction-rule checks", "",
              "| Case | Rows | Rule agreement | Truth vs test-cache direct | Threshold | Canonical AI AUC |", "|---|---:|---:|---:|---:|---:|"]
    for r in report["rq123"].get("prediction_checks", []):
        lines.append(f"| {r.get('case')} | {r.get('rows', '')} | {fmt_pct(r.get('prediction_rule_agreement'))} | {fmt_pct(r.get('truth_vs_test_cache_direct'))} | {fmt_num(r.get('threshold'))} | {fmt_num(r.get('canonical_auroc_ai_higher_score'))} |")
    lines += ["", "## RQ4: COLING metadata and saved output checks", ""]
    m = report["rq4"].get("metadata_check", {})
    if m:
        lines += [f"- Source file: `{m.get('source_path')}`",
                  f"- Model value counts: `{m.get('model_counts')}`",
                  f"- Raw label counts: `{m.get('raw_label_counts')}`",
                  f"- Human rows labelled as AI under HINDI_AI_LABEL={m.get('script_declared_ai_raw_label')}: **{m.get('human_rows_marked_ai')}**",
                  f"- Mapping supported by model metadata: **{m.get('label_mapping_supported_by_model_column')}**"]
    lines += ["", "| RQ4 source pair | Rows | Overlap with COLING data | Canonical truth agreement |", "|---|---:|---:|---:|"]
    for r in report["rq4"].get("prediction_checks", []):
        lines.append(f"| {r.get('case')} | {r.get('rows', '')} | {r.get('source_overlap', '')} | {fmt_pct(r.get('truth_matches_coling_metadata'))} |")
    lines += ["", "## RQ5: raw label inference from unchanged Human texts", "",
              f"- Inference status: **{report['rq5_label_inference'].get('status')}**",
              f"- Candidate raw AI label: **{report['rq5_label_inference'].get('global_candidate_ai_label')}**",
              f"- GPTZero cross-script implied raw AI label(s): `{report['rq5_label_inference'].get('gptzero_implied_raw_ai_labels')}`", "",
              "| Target | Known-human matches | New labels on matched Humans | Inferred raw AI label | Homogeneity | Passed? | GPTZero agreement if raw AI=0 |", "|---|---:|---|---:|---:|---|---:|"]
    for r in report["rq5_label_inference"].get("targets", []):
        sc = r.get("gptzero_cross_script_canonical_truth_check") or {}
        lines.append(f"| {r.get('target')} | {r.get('known_human_text_matches', '')} | `{r.get('new_prompt_labels_on_known_humans', {})}` | {r.get('inferred_new_prompt_ai_label', '')} | {fmt_pct(r.get('homogeneity'))} | {r.get('target_overlap_pass', False)} | {fmt_pct(sc.get('agreement_if_raw_ai_0'))} |")
    lines += ["", "## Issues / interpretation", ""]
    for s in report["interpretation"]:
        lines.append(f"- {s}")
    lines += ["", "## Important limitations", "",
              "- Label counts alone cannot identify which class is Human or AI.",
              "- A score-cache/source label match proves data lineage, not class semantics.",
              "- Reproducing a stored threshold proves the numeric calibration path, not that the class meaning was correct.",
              "- The audit does not recompute any DetectGPT score and does not require a GPU.",
              "- RQ5 repair, when enabled, uses the existing saved llr values and writes to a separate folder only."]
    return "\n".join(lines) + "\n"


def fmt_num(x: Any) -> str:
    try:
        f = float(x)
        return f"{f:.6f}" if math.isfinite(f) else "NA"
    except Exception:
        return "NA"


def fmt_pct(x: Any) -> str:
    try:
        f = float(x)
        return f"{f:.2%}" if math.isfinite(f) else "NA"
    except Exception:
        return "NA"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit/repair DetectGPT outputs without rerunning inference.")
    parser.add_argument("--project-root", required=True, help="CODE project root, e.g. ~/MTP-2/HINDI_FINAL/Dhruv/CODE")
    parser.add_argument("--calibration-csv", default=None, help="Optional explicit path to detect_gpt_calibrated_all_results.csv")
    parser.add_argument("--rq123-ai-label", type=int, choices=[0, 1], default=0, help="Raw AI label in shared RQ1-RQ3 source train/test CSVs (default 0, as used by XLM-R/GPTZero readers).")
    parser.add_argument("--rq123-pred-ai-label", type=int, choices=[0, 1], default=1, help="AI label in the OLD DetectGPT saved predicted_label column (default 1; independent of source truth encoding).")
    parser.add_argument("--rq4-ai-label", type=int, choices=[0, 1], default=0, help="Raw AI label in COLING hindi_data.csv (default 0 per DetectGPT COLING script).")
    parser.add_argument("--rq5-ai-label", choices=["auto", "0", "1"], default="auto", help="Raw AI label in new_prompt_data; auto uses unchanged-human overlap and refuses if ambiguous.")
    parser.add_argument("--min-human-overlap", type=int, default=15, help="Minimum known unchanged Human texts per target to infer RQ5 mapping (default 15).")
    parser.add_argument("--human-overlap-agreement", type=float, default=0.98, help="Required fraction of matched known Humans carrying one new-prompt label (default 0.98).")
    parser.add_argument("--repair-rq5", action="store_true", help="Write corrected RQ5 canonical predictions/metrics to a separate output folder if mapping is supported.")
    parser.add_argument("--rebuild-rq123", action="store_true", help="Recalibrate thresholds from existing source train llr caches and rebuild all RQ1-RQ3 outputs in a separate folder; no GPU inference.")
    parser.add_argument("--rebuild-rq4", action="store_true", help="Rebuild RQ4 from cached/saved COLING llr scores and source metadata in a separate folder; no GPU inference.")
    parser.add_argument("--rebuild-all-from-cache", action="store_true", help="Rebuild RQ1-RQ3 and RQ4, and attempt RQ5 repair, using existing cached/saved llr only. No original file is overwritten.")
    parser.add_argument("--outdir", default=None, help="Audit output directory (default: CODE/detectgpt_audit_TIMESTAMP).")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Project root does not exist: {root}")
    paths = resolve_paths(root, args.calibration_csv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else root / f"detectgpt_audit_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    rq123 = scan_rq123(root, paths, args.rq123_ai_label, args.rq123_pred_ai_label)
    rq4 = scan_rq4(root, paths, args.rq4_ai_label)
    rq5_inf = infer_rq5_ai_label(root, paths, args.rq123_ai_label, args.min_human_overlap, args.human_overlap_agreement)
    thresholds, th_meta = get_threshold_map(read_csv(paths["calibration_csv"]))

    rebuild_info: Dict[str, Any] = {}
    thresholds_for_targets = dict(thresholds)
    thresholds_recomputed = False
    if args.rebuild_rq123 or args.rebuild_all_from_cache:
        rq123_rebuild = rebuild_rq123_from_cache(paths, args.rq123_ai_label, outdir)
        rebuild_info["rq123"] = rq123_rebuild
        thresholds_recomputed = True
        if rq123_rebuild.get("status") == "COMPLETED":
            thresholds_for_targets = {}
            for key, value in rq123_rebuild.get("thresholds", {}).items():
                ds, llm = key.split("-", 1)
                thresholds_for_targets[(ds, llm)] = float(value)
        else:
            # Do not silently fall back to the old thresholds after a requested
            # recalibration failed; downstream RQ4/RQ5 rebuilds must be blocked.
            thresholds_for_targets = {}
    if args.rebuild_rq4 or args.rebuild_all_from_cache:
        rebuild_info["rq4"] = rebuild_rq4_from_cache(paths, args.rq4_ai_label, thresholds_for_targets, outdir)

    should_repair_rq5 = args.repair_rq5 or args.rebuild_all_from_cache
    if should_repair_rq5:
        if args.rq5_ai_label == "auto":
            chosen = rq5_inf.get("global_candidate_ai_label")
            if chosen is None:
                repair_info = {"status": "NOT_RUN_AMBIGUOUS_MAPPING", "reason": "No sufficiently strong unchanged-human overlap; inspect the audit rather than guessing.", "cases_written": 0}
            else:
                # Require at least two target overlap checks, unless GPTZero provides
                # consistent cross-script canonical-truth evidence for the same class.
                passing_count = int(rq5_inf.get("targets_passing", 0))
                sibling = [x for x in rq5_inf.get("gptzero_implied_raw_ai_labels", []) if x is not None]
                sibling_consensus = bool(len(sibling) >= 2 and all(x == chosen for x in sibling))
                if passing_count < 2 and not sibling_consensus:
                    repair_info = {"status": "BLOCKED_WEAK_MAPPING_EVIDENCE", "reason": f"Only {passing_count} targets have sufficient unchanged-human overlap and no two-target GPTZero cross-check.", "cases_written": 0}
                elif sibling and any(x != chosen for x in sibling):
                    repair_info = {"status": "BLOCKED_CONFLICTING_SIBLING_EVIDENCE", "reason": f"Human-overlap candidate {chosen} conflicts with GPTZero cross-script evidence {sibling}.", "cases_written": 0}
                else:
                    repair_info = repair_rq5(root, paths, int(chosen), thresholds_for_targets, outdir, allow_rethreshold=thresholds_recomputed)
        else:
            chosen = int(args.rq5_ai_label)
            repair_info = repair_rq5(root, paths, chosen, thresholds_for_targets, outdir, allow_rethreshold=thresholds_recomputed)
        report_repair = repair_info
    else:
        report_repair = {"status": "NOT_REQUESTED", "cases_written": 0}

    issues: List[str] = []
    if paths["calibration_csv"] is None:
        issues.append("Calibration master CSV not located; threshold reproduction and calibrated metrics cannot be fully checked.")
    if rq123.get("threshold_master", {}).get("pairs_found", 0) < 6:
        issues.append("Fewer than six source thresholds were found.")
    bad_th = [x for x in rq123.get("threshold_checks", []) if not x.get("saved_matches_declared_ai_threshold")]
    if bad_th:
        issues.append(f"{len(bad_th)} source thresholds did not reproduce under the declared RQ1-RQ3 raw AI label={args.rq123_ai_label}.")
    bad_sources = [x for x in rq123.get("source_files", []) if x.get("best_direct_agreement") is not None and x.get("best_direct_agreement") < 0.999]
    if bad_sources:
        issues.append(f"{len(bad_sources)} train/test caches did not directly match their best-matching source files. Inspect alternate candidates and paths.")
    low_rule = [x for x in rq123.get("prediction_checks", []) if x.get("prediction_rule_agreement") is not None and x.get("prediction_rule_agreement") < 0.999]
    if low_rule:
        issues.append(f"{len(low_rule)} RQ1-RQ3 prediction files do not fully follow llr >= threshold -> AI.")
    if not rq4.get("metadata_check", {}).get("label_mapping_supported_by_model_column", False):
        issues.append("RQ4 HINDI_AI_LABEL=0 was not independently confirmed by the COLING model column.")
    if rq5_inf.get("global_candidate_ai_label") is None:
        issues.append("RQ5 raw AI label could not be established confidently from unchanged-human overlaps; do not automatically flip labels.")
    else:
        issues.append(f"RQ5 unchanged-human overlap supports raw AI label={rq5_inf['global_candidate_ai_label']}; verify target-level coverage before adopting.")
    if report_repair.get("status") not in ("NOT_REQUESTED", "COMPLETED", "COMPLETED_WITH_WARNINGS"):
        issues.append(f"RQ5 repair status: {report_repair.get('status')}.")

    interpretation = [
        f"RQ1-RQ3 source ground truth is audited using raw AI label={args.rq123_ai_label}; the original saved DetectGPT prediction column is separately interpreted using raw AI label={args.rq123_pred_ai_label}. The source convention is based on the shared CSV readers (XLM-R/GPTZero) rather than the old DetectGPT output convention.",
        f"RQ4 is audited using raw AI label={args.rq4_ai_label}, as declared by the DetectGPT COLING script. The 'model' column is checked against the label values, then each saved canonical true_label is compared with the COLING source by normalized text.",
        "RQ5 raw AI label is inferred from unchanged human texts: original DetectGPT test-cache rows whose raw label is not the original AI label are treated as known Human; their matched new-prompt labels identify the new-prompt Human class. If overlap is too small or target mappings conflict, no automatic repair should occur.",
        "The audit distinguishes numeric pipeline reproduction from semantic label verification. Matching source/cache labels or reproducing thresholds does not, by itself, prove the labels mean AI/Human.",
        "DetectGPT llr direction remains higher=more AI-like. Do not negate llr or flip direction per dataset to improve AUROC. Low/negative-discrimination source calibrations under the verified AI-positive mapping should be reported as weak calibration, not hidden by choosing the opposite polarity.",
    ]

    def pstr(x: Optional[Path]) -> Optional[str]:
        return str(x) if x else None
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(root),
        "paths": {k: pstr(v) for k, v in paths.items()},
        "arguments": vars(args),
        "rq123": rq123,
        "rq4": rq4,
        "rq5_label_inference": rq5_inf,
        "rq5_repair": report_repair,
        "cache_rebuilds": rebuild_info,
        "interpretation": interpretation,
        "executive_status": {
            "RQ1-RQ3 source/cache lineage": "PASS" if not bad_sources else "REVIEW",
            "RQ1-RQ3 threshold reproduction": "PASS" if not bad_th else "REVIEW",
            "RQ1-RQ3 saved threshold decisions": "PASS" if not low_rule else "REVIEW",
            "RQ4 COLING label mapping": "SUPPORTED" if rq4.get("metadata_check", {}).get("label_mapping_supported_by_model_column", False) else "UNVERIFIED",
            "RQ5 raw label": rq5_inf.get("status"),
            "RQ5 repair": report_repair.get("status"),
        },
        "issues": issues,
    }
    (outdir / "detectgpt_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (outdir / "detectgpt_audit.md").write_text(flatten_report(report), encoding="utf-8")
    pd.DataFrame(rq123.get("source_files", [])).to_csv(outdir / "rq123_source_cache_lineage.csv", index=False)
    pd.DataFrame(rq123.get("threshold_checks", [])).to_csv(outdir / "rq123_threshold_reproduction.csv", index=False)
    pd.DataFrame(rq123.get("prediction_checks", [])).to_csv(outdir / "rq123_prediction_rule_audit.csv", index=False)
    pd.DataFrame(rq4.get("prediction_checks", [])).to_csv(outdir / "rq4_coling_prediction_audit.csv", index=False)
    pd.DataFrame(rq5_inf.get("targets", [])).to_csv(outdir / "rq5_human_overlap_audit.csv", index=False)
    print("\nDetectGPT audit complete.")
    print(f"Project root: {root}")
    print(f"Audit outputs: {outdir}")
    print(f"RQ1-RQ3 AI raw label assumed: {args.rq123_ai_label}")
    print(f"RQ4 COLING AI raw label assumed: {args.rq4_ai_label}")
    print(f"RQ5 inference: {rq5_inf.get('status')} | candidate AI raw label={rq5_inf.get('global_candidate_ai_label')}")
    print(f"RQ5 repair: {report_repair.get('status')} | cases written={report_repair.get('cases_written', 0)}")
    for task, detail in rebuild_info.items():
        print(f"{task.upper()} cache rebuild: {detail.get('status')} | cases written={detail.get('cases_written', 0)}")
    if issues:
        print("\nItems requiring attention:")
        for item in issues:
            print(f"  - {item}")
    print("\nNo original DetectGPT files were overwritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
