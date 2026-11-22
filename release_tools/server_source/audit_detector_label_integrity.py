#!/usr/bin/env python3
"""Non-destructive, one-run integrity audit for Hindi MGT detector outputs."""

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

DATASETS = ("hindisumm", "xquad")
LLMS = ("gemini", "llama3", "qwen3")
RQ5_LLMS = ("gemini", "qwen3")
ZW = "\u200b\u200c\u200d\ufeff"
DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def norm(x):
    if pd.isna(x):
        return ""
    s = unicodedata.normalize("NFC", str(x)).translate(DIGITS)
    for c in ZW:
        s = s.replace(c, "")
    return " ".join(s.split()).strip()


def read(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find(root, name, limit=20000):
    skip = {".git", ".venv", "venv", "__pycache__", "node_modules", ".cache"}
    out = []
    for p in root.rglob(name):
        if p.is_file() and not any(x in skip for x in p.parts):
            out.append(p)
            if len(out) >= limit:
                break
    return sorted(set(out))


def parse_case(path):
    pattern = (
        r"(hindisumm|xquad)_(gemini|llama3|qwen3)_tested_on_"
        r"(hindisumm|xquad)_(gemini|llama3|qwen3)"
    )
    for part in reversed(path.parts[:-1]):
        m = re.search(pattern, part.lower())
        if m:
            return m.groups()
    return None


def match_text_occurrences(source, predictions):
    """Match normalized text and occurrence number; never make a many-to-many join."""
    required_s = {"text", "label"}
    required_p = {"text", "true_label"}
    if not required_s.issubset(source.columns):
        return pd.DataFrame(), 0.0, "source_columns_missing"
    if not required_p.issubset(predictions.columns):
        return pd.DataFrame(), 0.0, "prediction_columns_missing"

    a = source.copy().reset_index(drop=True)
    b = predictions.copy().reset_index(drop=True)

    a["_key"] = a.text.map(norm)
    b["_key"] = b.text.map(norm)
    a["_occ"] = a.groupby("_key", sort=False).cumcount()
    b["_occ"] = b.groupby("_key", sort=False).cumcount()

    keep = ["_key", "_occ", "true_label"]
    for col in ("predicted_label", "predicted_prob_label1", "llr", "threshold"):
        if col in b.columns:
            keep.append(col)

    m = a.merge(
        b[keep],
        on=["_key", "_occ"],
        how="left",
        validate="one_to_one"
    )
    coverage = m.true_label.notna().mean() if len(m) else 0.0
    return m, float(coverage), "normalized_text_occurrence"


def compare_labels(a, b):
    x = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    valid = (
        np.isfinite(x) & np.isfinite(y)
        & np.isin(x, [0, 1]) & np.isin(y, [0, 1])
    )
    if not valid.any():
        return None, None, 0

    x = x[valid].astype(int)
    y = y[valid].astype(int)
    return float(np.mean(x == y)), float(np.mean(x == 1-y)), len(x)


def counts(df, col="label"):
    x = pd.to_numeric(df[col], errors="coerce")
    return int((x == 0).sum()), int((x == 1).sum())


def tf_predictions(root, phase):
    out = []
    for p in find(root, "predictions.csv"):
        path = str(p).lower().replace("\\", "/")
        if not any(x in path for x in (
            "text_fluoroscopy", "text_fluroscopy", "text-fluoro"
        )):
            continue

        case = parse_case(p)

        if phase == "rq123" and case and "new_prompt" not in path and "coling" not in path:
            out.append(p)
        elif phase == "rq4" and "coling" in path:
            out.append(p)
        elif phase == "rq5" and ("new_prompt" in path or "prompt_change" in path):
            out.append(p)

    return sorted(set(out))


def audit_tf_rq123(root, out):
    rows, evidence, training = [], [], []

    for pp in tf_predictions(root, "rq123"):
        case = parse_case(pp)
        if not case:
            continue

        src_ds, src_llm, tgt_ds, tgt_llm = case
        pred = read(pp)

        required = {
            "text", "true_label", "predicted_label", "predicted_prob_label1"
        }
        if pred is None or not required.issubset(pred.columns):
            rows.append({"prediction": str(pp), "status": "MISSING_COLUMNS"})
            continue

        sources = [
            p for p in find(root, f"{tgt_ds}_test_{tgt_llm}.csv", 100)
            if "text_fluoro" not in str(p).lower()
            and "text_fluroscopy" not in str(p).lower()
        ]

        for sp in sources:
            src = read(sp)
            if src is None or not {"text", "label"}.issubset(src.columns):
                continue

            m, coverage, mode = match_text_occurrences(src, pred)
            if m.empty:
                continue

            direct, inverse, n = compare_labels(m.label, m.true_label)

            prob = pd.to_numeric(m.get("predicted_prob_label1"), errors="coerce")
            yp = pd.to_numeric(m.get("predicted_label"), errors="coerce")
            valid = prob.notna() & yp.isin([0, 1])

            rule_gt = (
                float(np.mean(yp[valid].to_numpy(int) ==
                              (prob[valid].to_numpy(float) > .5).astype(int)))
                if valid.any() else None
            )
            rule_ge = (
                float(np.mean(yp[valid].to_numpy(int) ==
                              (prob[valid].to_numpy(float) >= .5).astype(int)))
                if valid.any() else None
            )

            s0, s1 = counts(src)
            p0, p1 = counts(pred, "true_label")

            status = "REVIEW"
            if coverage >= .999 and direct is not None and direct >= .999:
                status = "PASS_DIRECT"
            elif coverage >= .999 and inverse is not None and inverse >= .999:
                status = "PASS_INVERSE"

            rows.append({
                "case": pp.parent.name,
                "source_csv": str(sp),
                "prediction_csv": str(pp),
                "source_rows": len(src),
                "prediction_rows": len(pred),
                "source_raw0": s0,
                "source_raw1": s1,
                "prediction_truth_raw0": p0,
                "prediction_truth_raw1": p1,
                "join_mode": mode,
                "join_coverage": coverage,
                "direct_agreement": direct,
                "inverse_agreement": inverse,
                "n_matched": n,
                "pred_vs_prob_gt_0.5": rule_gt,
                "pred_vs_prob_ge_0.5": rule_ge,
                "source_sha256": digest(sp),
                "prediction_sha256": digest(pp),
                "status": status
            })

            if coverage >= .999:
                e = m.copy()
                e.insert(0, "case", pp.parent.name)
                e.insert(1, "source_csv", str(sp))
                evidence.append(e)

    for ds in DATASETS:
        for llm in LLMS:
            for p in find(root, f"{ds}_train_{llm}.csv", 100):
                if "training_data" not in str(p).lower():
                    continue
                d = read(p)
                if d is None or "label" not in d:
                    continue
                n0, n1 = counts(d)
                training.append({
                    "training_csv": str(p),
                    "rows": len(d),
                    "raw0_count": n0,
                    "raw1_count": n1,
                    "sha256": digest(p)
                })

    pd.DataFrame(rows).to_csv(out / "tf_rq123_source_comparison.csv", index=False)
    if evidence:
        pd.concat(evidence, ignore_index=True).to_csv(
            out / "tf_rq123_row_evidence.csv", index=False
        )
    pd.DataFrame(training).to_csv(out / "tf_training_label_counts.csv", index=False)

    return {
        "prediction_files": len(tf_predictions(root, "rq123")),
        "comparisons": len(rows),
        "direct_pass": sum(r.get("status") == "PASS_DIRECT" for r in rows),
        "inverse_pass": sum(r.get("status") == "PASS_INVERSE" for r in rows),
        "training_csvs": len(training)
    }


def audit_tf_rq4(root, out):
    rows = []
    dedicated = find(root, "hindi_data_perturb.csv", 100)
    official = [
        p for p in find(root, "hindi_data.csv", 100)
        if "coling" in str(p).lower()
    ]
    predictions = tf_predictions(root, "rq4")

    for dp in dedicated:
        d = read(dp)
        if d is None or not {"text", "label"}.issubset(d.columns):
            rows.append({"dedicated_file": str(dp), "status": "MISSING_COLUMNS"})
            continue

        for op in official:
            o = read(op)
            if o is None or not {"text", "label"}.issubset(o.columns):
                continue

            # Exact normalized text + occurrence alignment.
            a, b = o.copy(), d.copy()
            a["_key"], b["_key"] = a.text.map(norm), b.text.map(norm)
            a["_occ"] = a.groupby("_key", sort=False).cumcount()
            b["_occ"] = b.groupby("_key", sort=False).cumcount()
            m = a.merge(
                b[["_key", "_occ", "label"]].rename(columns={"label": "tf_label"}),
                on=["_key", "_occ"], how="inner", validate="one_to_one"
            )
            direct, inverse, n = compare_labels(m.label, m.tf_label) if len(m) else (None, None, 0)

            meta = {}
            if "model" in o.columns:
                model = o.model.astype(str).str.strip().str.lower()
                labels = pd.to_numeric(o.label, errors="coerce")
                human = labels[model.eq("human") & labels.isin([0, 1])]
                generated = labels[~model.eq("human") & labels.isin([0, 1])]
                meta = {
                    "official_human_n": len(human),
                    "official_human_raw0": int((human == 0).sum()),
                    "official_human_raw1": int((human == 1).sum()),
                    "official_generated_n": len(generated),
                    "official_generated_raw0": int((generated == 0).sum()),
                    "official_generated_raw1": int((generated == 1).sum()),
                    "metadata_supports_ai_raw0": bool(
                        len(human) and len(generated)
                        and (human == 1).all() and (generated == 0).all()
                    ),
                    "metadata_supports_ai_raw1": bool(
                        len(human) and len(generated)
                        and (human == 0).all() and (generated == 1).all()
                    )
                }

            rows.append({
                "dedicated_file": str(dp),
                "official_coling_file": str(op),
                "dedicated_rows": len(d),
                "official_rows": len(o),
                "matched_occurrences": n,
                "dedicated_coverage": n / max(1, len(d)),
                "direct_agreement": direct,
                "inverse_agreement": inverse,
                "dedicated_sha256": digest(dp),
                "official_sha256": digest(op),
                **meta
            })

    for pp in predictions:
        pred = read(pp)
        if pred is None or not {"text", "true_label"}.issubset(pred.columns):
            continue
        for dp in dedicated:
            src = read(dp)
            if src is None:
                continue
            m, coverage, _ = match_text_occurrences(src, pred)
            direct, inverse, n = compare_labels(m.label, m.true_label) if len(m) else (None, None, 0)
            rows.append({
                "phase": "saved_prediction_vs_dedicated_source",
                "prediction_file": str(pp),
                "dedicated_file": str(dp),
                "matched_n": n,
                "join_coverage": coverage,
                "direct_agreement": direct,
                "inverse_agreement": inverse
            })

    pd.DataFrame(rows).to_csv(out / "tf_rq4_coling_audit.csv", index=False)
    return {
        "dedicated_sources": len(dedicated),
        "official_sources": len(official),
        "prediction_files": len(predictions),
        "comparisons": len(rows)
    }


def audit_tf_rq5(root, out):
    rows = []
    new_sources = find(root, "*_new_prompt.csv", 500)
    predictions = tf_predictions(root, "rq5")

    for pp in predictions:
        case = parse_case(pp)
        if not case:
            continue
        target_ds, target_llm = case[2], case[3]
        pred = read(pp)
        if pred is None or not {"text", "true_label"}.issubset(pred.columns):
            continue

        for nf in new_sources:
            if nf.name.lower() != f"{target_ds}_{target_llm}_new_prompt.csv":
                continue
            src = read(nf)
            if src is None or not {"text", "label"}.issubset(src.columns):
                continue

            m, coverage, _ = match_text_occurrences(src, pred)
            direct, inverse, n = compare_labels(m.label, m.true_label) if len(m) else (None, None, 0)
            raw0, raw1 = counts(src)

            rows.append({
                "prediction_file": str(pp),
                "new_prompt_source": str(nf),
                "source_rows": len(src),
                "prediction_rows": len(pred),
                "matched_n": n,
                "join_coverage": coverage,
                "source_raw0": raw0,
                "source_raw1": raw1,
                "direct_agreement": direct,
                "inverse_agreement": inverse,
                "status": "PASS_DIRECT" if coverage >= .999 and direct is not None and direct >= .999
                          else ("PASS_INVERSE" if coverage >= .999 and inverse is not None and inverse >= .999
                                else "REVIEW")
            })

    pd.DataFrame(rows).to_csv(out / "tf_rq5_new_prompt_audit.csv", index=False)
    return {
        "new_prompt_sources": len(new_sources),
        "prediction_files": len(predictions),
        "comparisons": len(rows),
        "direct_pass": sum(r.get("status") == "PASS_DIRECT" for r in rows)
    }


def locate_detectgpt(root):
    rebuilt_files = find(root, "rq123_recomputed_metrics.csv", 100)
    rebuilt = rebuilt_files[0].parent if rebuilt_files else None

    cache_files = [
        p for p in find(root, "train_hindisumm_gemini.csv", 100)
        if "score_cache" in str(p).lower()
    ]
    cache = cache_files[0].parent if cache_files else None
    return rebuilt, cache


def audit_detectgpt(root, out):
    rebuilt, cache = locate_detectgpt(root)
    thresholds = {}
    threshold_rows, cache_rows, pred_rows = [], [], []

    if rebuilt:
        t = read(rebuilt / "source_thresholds_recomputed.csv")
        if t is not None:
            for _, r in t.iterrows():
                key = (str(r.train_dataset), str(r.train_llm))
                val = float(r.threshold)
                thresholds[key] = val
                threshold_rows.append({
                    "source": f"{key[0]}_{key[1]}",
                    "threshold": val,
                    "negative": val < 0,
                    "training_auc": float(r.train_auroc_ai_higher_score),
                    "youden_j": float(r.train_youden_j),
                    "n_calibration": int(r.n_calibration)
                })

    if cache:
        for ds in DATASETS:
            for llm in LLMS:
                p = cache / f"train_{ds}_{llm}.csv"
                d = read(p)
                if d is None or not {"label", "llr"}.issubset(d.columns):
                    cache_rows.append({"source": f"{ds}_{llm}", "status": "MISSING_CACHE"})
                    continue

                y = pd.to_numeric(d.label, errors="coerce").to_numpy(float)
                z = pd.to_numeric(d.llr, errors="coerce").to_numpy(float)
                valid = np.isfinite(y) & np.isin(y, [0, 1]) & np.isfinite(z)
                if not valid.any() or len(np.unique(y[valid])) < 2:
                    cache_rows.append({"source": f"{ds}_{llm}", "status": "INVALID_CACHE"})
                    continue

                # Source convention: raw 0 = AI; larger LLR = more AI-like.
                y_ai = (y[valid] == 0).astype(int)
                fpr, tpr, thr = roc_curve(y_ai, z[valid], pos_label=1)
                j = tpr - fpr
                ix = int(np.argmax(j))
                calc = float(thr[ix])
                saved = thresholds.get((ds, llm))

                cache_rows.append({
                    "source": f"{ds}_{llm}",
                    "cache": str(p),
                    "n": int(valid.sum()),
                    "raw0_ai_n": int((y[valid] == 0).sum()),
                    "raw1_human_n": int((y[valid] == 1).sum()),
                    "recomputed_threshold": calc,
                    "recomputed_youden_j": float(j[ix]),
                    "recomputed_auc": float(roc_auc_score(y_ai, z[valid])),
                    "v3_threshold": saved,
                    "matches_v3": bool(saved is not None and np.isclose(calc, saved, rtol=1e-8, atol=1e-10)),
                    "status": "PASS" if saved is not None and np.isclose(calc, saved, rtol=1e-8, atol=1e-10)
                              else "REVIEW"
                })

    if rebuilt:
        audit_root = rebuilt.parent.parent
        roots = [
            rebuilt,
            rebuilt.parent / "rq4",
            audit_root / "rq5_repaired"
        ]
        for folder in roots:
            if not folder.exists():
                continue
            for pp in folder.rglob("predictions.csv"):
                d = read(pp)
                required = {"true_label", "predicted_label", "llr", "threshold"}
                if d is None or not required.issubset(d.columns):
                    continue

                y = pd.to_numeric(d.true_label, errors="coerce")
                yp = pd.to_numeric(d.predicted_label, errors="coerce")
                z = pd.to_numeric(d.llr, errors="coerce")
                th = pd.to_numeric(d.threshold, errors="coerce")
                v = y.isin([0, 1]) & yp.isin([0, 1]) & z.notna() & th.notna()
                rule = (
                    float(np.mean(
                        yp[v].to_numpy(int) ==
                        (z[v].to_numpy(float) >= th[v].to_numpy(float)).astype(int)
                    )) if v.any() else None
                )
                pred_rows.append({
                    "file": str(pp),
                    "rows": len(d),
                    "valid_rows": int(v.sum()),
                    "unique_thresholds": int(th[v].nunique()),
                    "llr_min": float(z[v].min()) if v.any() else None,
                    "llr_max": float(z[v].max()) if v.any() else None,
                    "threshold_min": float(th[v].min()) if v.any() else None,
                    "threshold_max": float(th[v].max()) if v.any() else None,
                    "threshold_rule_agreement": rule,
                    "status": "PASS" if v.all() and rule is not None and rule >= .999 else "REVIEW"
                })

    pd.DataFrame(threshold_rows).to_csv(out / "detectgpt_v3_thresholds.csv", index=False)
    pd.DataFrame(cache_rows).to_csv(out / "detectgpt_threshold_independent_reproduction.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "detectgpt_rebuilt_prediction_audit.csv", index=False)

    return {
        "rebuilt_root": str(rebuilt) if rebuilt else None,
        "cache_root": str(cache) if cache else None,
        "thresholds_found": len(threshold_rows),
        "independent_passes": sum(r.get("status") == "PASS" for r in cache_rows),
        "prediction_files_checked": len(pred_rows),
        "negative_thresholds": sum(r.get("negative", False) for r in threshold_rows)
    }


def audit_gptzero(root, out):
    paths = find(root, "zero_shot_threshold_audit.csv", 100)
    p = paths[0] if paths else None
    d = read(p)
    if d is None or "model" not in d.columns:
        pd.DataFrame().to_csv(out / "gptzero_threshold_audit.csv", index=False)
        return {"threshold_file": str(p) if p else None, "thresholds_found": 0}

    g = d[d.model.astype(str).str.lower() == "gptzero"].copy()
    if "calibration_threshold" in g.columns:
        g["threshold_positive"] = pd.to_numeric(g.calibration_threshold, errors="coerce") > 0
        g["expected_rule"] = "AI if perplexity <= threshold; lower perplexity is AI"
    g.to_csv(out / "gptzero_threshold_audit.csv", index=False)
    return {
        "threshold_file": str(p),
        "thresholds_found": len(g),
        "all_positive": bool(g.threshold_positive.all()) if len(g) else None
    }


def write_tf_candidates(root, out):
    backup = out / "backup_original_textfluoro_predictions"
    candidate_root = out / "textfluoro_candidate_canonical_outputs"
    backup.mkdir(parents=True, exist_ok=True)
    manifest = []

    for phase in ("rq123", "rq4", "rq5"):
        for pp in tf_predictions(root, phase):
            d = read(pp)
            required = {"text", "true_label", "predicted_label", "predicted_prob_label1"}
            if d is None or not required.issubset(d.columns):
                continue

            rel = pp.relative_to(root)
            bp = backup / rel
            bp.parent.mkdir(parents=True, exist_ok=True)
            if not bp.exists():
                shutil.copy2(pp, bp)

            for ai_raw in (1, 0):
                x = d.copy()
                yt = pd.to_numeric(x.true_label, errors="coerce")
                yp = pd.to_numeric(x.predicted_label, errors="coerce")
                p1 = pd.to_numeric(x.predicted_prob_label1, errors="coerce")

                x["canonical_true_label_ai1"] = np.where(yt.isin([0, 1]), (yt == ai_raw).astype(int), np.nan)
                x["canonical_predicted_label_ai1"] = np.where(yp.isin([0, 1]), (yp == ai_raw).astype(int), np.nan)
                x["canonical_ai_score"] = p1 if ai_raw == 1 else 1 - p1
                x["assumed_raw_ai_label"] = ai_raw
                x["hypothesis"] = "github_raw1_ai" if ai_raw == 1 else "shared_raw0_ai"
                x["warning"] = "Candidate only; verify training and test source semantics before publication."

                dest = candidate_root / x.hypothesis.iloc[0] / phase / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                x.to_csv(dest, index=False)

                manifest.append({
                    "phase": phase,
                    "source": str(pp),
                    "backup": str(bp),
                    "candidate": str(dest),
                    "raw_ai_label_hypothesis": ai_raw,
                    "rows": len(x)
                })

    pd.DataFrame(manifest).to_csv(out / "candidate_manifest.csv", index=False)
    return {
        "backups": len(set(x["source"] for x in manifest)),
        "candidate_csvs": len(manifest),
        "hypotheses": ["GitHub: raw 1 = AI", "Alternative: raw 0 = AI"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, help="CODE project root")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Project root does not exist: {root}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / f"label_integrity_audit_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    print("Checking Text-Fluoroscopy, DetectGPT and GPTZero. No inference will run.")

    tf123 = audit_tf_rq123(root, out)
    tf4 = audit_tf_rq4(root, out)
    tf5 = audit_tf_rq5(root, out)
    dg = audit_detectgpt(root, out)
    gz = audit_gptzero(root, out)
    candidates = write_tf_candidates(root, out)

    summary = {
        "project_root": str(root),
        "audit_dir": str(out),
        "textfluoro_rq123": tf123,
        "textfluoro_rq4": tf4,
        "textfluoro_rq5": tf5,
        "detectgpt": dg,
        "gptzero": gz,
        "candidate_outputs": candidates
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Detector label-integrity audit",
        "",
        f"Project root: `{root}`",
        f"Audit folder: `{out}`",
        "",
        "## Text-Fluoroscopy",
        "",
        f"- RQ1-RQ3: {tf123['prediction_files']} prediction files; {tf123['comparisons']} source comparisons; {tf123['direct_pass']} direct passes; {tf123['inverse_pass']} inverse passes.",
        f"- Training CSVs checked: {tf123['training_csvs']}.",
        f"- RQ4: {tf4['dedicated_sources']} dedicated COLING files; {tf4['official_sources']} official COLING files; {tf4['comparisons']} comparisons.",
        f"- RQ5: {tf5['new_prompt_sources']} new-prompt sources; {tf5['prediction_files']} prediction files; {tf5['comparisons']} comparisons.",
        "",
        "GitHub documents Text-Fluoroscopy raw class 1 as AI. The training script feeds CSV labels directly to BCE training, so the server-side training CSV must use that same semantic mapping for the model's class 1 to mean AI. This audit compares the saved outputs against the exact server-side source files.",
        "",
        "Do not flip labels merely because class counts complement another detector. Investigate exact source paths, text matches and label agreement first.",
        "",
        "## DetectGPT",
        "",
        f"- Rebuilt thresholds found: {dg['thresholds_found']}.",
        f"- Independent threshold reproductions passing: {dg['independent_passes']}/6.",
        f"- Rebuilt prediction files checked: {dg['prediction_files_checked']}.",
        f"- Negative thresholds: {dg['negative_thresholds']}.",
        "",
        "Negative DetectGPT thresholds are valid on the signed standardized LLR scale. Higher LLR remains AI-like. GPTZero uses positive perplexity, where lower values indicate AI. The numerical threshold scales are not comparable.",
        "",
        "Only replace the paper threshold table if all six independent reproductions pass and the cache paths are the intended final training split.",
        "",
        "## GPTZero",
        "",
        f"- Threshold file: {gz['threshold_file']}",
        f"- Thresholds found: {gz['thresholds_found']}",
        f"- All positive: {gz.get('all_positive')}",
        "",
        "## Candidate files and backups",
        "",
        f"- Original Text-Fluoroscopy files backed up: {candidates['backups']}.",
        f"- Candidate files written: {candidates['candidate_csvs']}.",
        "",
        "The candidate files contain both label hypotheses. They are not approved final results until source semantics are established.",
        "",
        "## Evidence files",
        "",
        "- `tf_rq123_source_comparison.csv`",
        "- `tf_rq123_row_evidence.csv`",
        "- `tf_training_label_counts.csv`",
        "- `tf_rq4_coling_audit.csv`",
        "- `tf_rq5_new_prompt_audit.csv`",
        "- `detectgpt_v3_thresholds.csv`",
        "- `detectgpt_threshold_independent_reproduction.csv`",
        "- `detectgpt_rebuilt_prediction_audit.csv`",
        "- `gptzero_threshold_audit.csv`",
        "- `candidate_manifest.csv`",
        "",
        "No original file was overwritten. No GPU/model inference was performed."
    ]
    (out / "audit_report.md").write_text("\n".join(report), encoding="utf-8")

    print("\nAudit complete.")
    print(f"Output: {out}")
    print(f"Text-Fluoroscopy RQ1-RQ3 comparisons: {tf123['comparisons']}")
    print(f"Text-Fluoroscopy RQ4 comparisons: {tf4['comparisons']}")
    print(f"Text-Fluoroscopy RQ5 comparisons: {tf5['comparisons']}")
    print(f"DetectGPT threshold reproductions: {dg['independent_passes']}/6")
    print(f"Candidate CSVs: {candidates['candidate_csvs']}")
    print("Original inference files were NOT modified.")
    print("Read audit_report.md before choosing a label hypothesis.")


if __name__ == "__main__":
    main()
