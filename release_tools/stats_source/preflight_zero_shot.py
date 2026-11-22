#!/usr/bin/env python3
"""Read-only preflight for the GPTZero/DetectGPT RQ1-RQ5 integration."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

from loaders import (
    load_gptzero, load_detectgpt, load_zero_shot_rq4_all, load_zero_shot_rq5_all,
    set_detectgpt_rebuilt_root,
)


def summarize(name: str, frame: pd.DataFrame, expected: int) -> bool:
    if frame.empty:
        print(f"{name:22s} 0 cells (expected {expected}) [FAIL]")
        return False
    keys = ["train_dataset", "train_llm", "test_dataset", "test_llm"]
    cells = frame[keys].drop_duplicates().shape[0]
    bad_labels = not frame["true_label"].isin([0, 1]).all() or not frame["predicted_label"].isin([0, 1]).all()
    one_class = frame.groupby(keys)["true_label"].nunique().lt(2).sum()
    ok = cells == expected and not bad_labels and one_class == 0
    status = "PASS" if ok else "CHECK"
    print(f"{name:22s} {cells:2d} cells / {len(frame):6d} rows; invalid labels={bad_labels}; one-class cells={one_class} [{status}]")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CODE/training_models")
    ap.add_argument("--rq4-root", required=True, help="CODE/COLING_2025_MGT_Task1")
    ap.add_argument("--rq5-root", required=True, help="CODE/prompt_change")
    ap.add_argument("--detectgpt-rebuilt-root", default=None, help="Exact v3 detectgpt_audit_* directory")
    args = ap.parse_args()
    set_detectgpt_rebuilt_root(args.detectgpt_rebuilt_root)
    root, rq4, rq5 = map(lambda x: Path(x).expanduser().resolve(), (args.root, args.rq4_root, args.rq5_root))
    print("\nScore-only detector integration preflight (read-only)\n")
    try:
        rq4_all = load_zero_shot_rq4_all(root, rq4)
        rq5_all = load_zero_shot_rq5_all(root, rq5)
        frames = {
            "GPTZero RQ1-RQ3": load_gptzero(root),
            "DetectGPT RQ1-RQ3": load_detectgpt(root),
            "GPTZero RQ4": rq4_all.query("model == 'GPTZero'"),
            "DetectGPT RQ4": rq4_all.query("model == 'DetectGPT'"),
            "GPTZero RQ5": rq5_all.query("model == 'GPTZero'"),
            "DetectGPT RQ5": rq5_all.query("model == 'DetectGPT'"),
        }
    except Exception as exc:
        print(f"\n[FAIL] Loader exception: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    expectations = {
        "GPTZero RQ1-RQ3": 24, "DetectGPT RQ1-RQ3": 24,
        "GPTZero RQ4": 6, "DetectGPT RQ4": 6,
        "GPTZero RQ5": 12, "DetectGPT RQ5": 12,
    }
    ok = True
    for name, frame in frames.items():
        ok = summarize(name, frame, expectations[name]) and ok
    print("\nResult:", "ALL EXPECTED CELLS FOUND" if ok else "ONE OR MORE CELLS NEED ATTENTION")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
