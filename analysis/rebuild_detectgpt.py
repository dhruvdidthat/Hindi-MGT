"""Fit six source cutoffs and replay the 42 DetectGPT evaluations from stored LLRs."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/detectgpt"

def main():
    cuts = {}
    for p in sorted(CACHE.glob("train_*.csv.gz")):
        source = p.name.removeprefix("train_").removesuffix(".csv.gz")
        train = pd.read_csv(p, float_precision="round_trip")
        fpr, tpr, thresholds = roc_curve(train.label.eq(0), train.llr)
        cuts[source] = thresholds[np.argmax(tpr - fpr)]
    results = []
    for group in ("rq1_3", "rq4", "rq5"):
        expected = pd.read_csv(ROOT / "experiments" / group / "predictions.csv.gz", float_precision="round_trip")
        keys = ["train_dataset", "train_llm", "test_dataset", "test_llm"]
        for (sd, sl, td, tl), cell in expected[expected.model.eq("DetectGPT")].groupby(keys):
            source, target = f"{sd}_{sl}", f"{td}_{tl}"
            files = {"rq1_3": f"test_{target}", "rq4": "coling_hindi", "rq5": f"{source}_tested_on_{target}"}
            raw = pd.read_csv(CACHE / (files[group] + ".csv.gz"), float_precision="round_trip")
            raw["true_label"] = raw.label.eq(0).astype(int)
            raw["predicted_label"] = raw.llr.ge(cuts[source]).astype(int)
            raw = raw.rename(columns={"llr":"score"})
            columns = ["text_key", "true_label", "predicted_label", "score"]
            def ordered(frame):
                return frame[columns].sort_values(columns).reset_index(drop=True)
            pd.testing.assert_frame_equal(ordered(raw), ordered(cell), atol=1e-12, rtol=1e-12)
            raw[keys] = [sd, sl, td, tl]
            results.append(raw.assign(experiment=group, threshold=cuts[source]))
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    pd.concat(results).to_csv(out / "detectgpt_rebuilt.csv.gz", index=False)
    print("Verified", len(results), "DetectGPT evaluations", cuts)

if __name__ == "__main__":
    main()
