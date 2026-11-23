"""Recompute paper statistics from the distributed prediction tables."""
from pathlib import Path
import argparse
import pandas as pd
import pipeline as core
import pipeline_rq45 as transfer

ROOT = Path(__file__).resolve().parents[1]

def read_predictions(group):
    data = pd.read_csv(ROOT / "experiments" / group / "predictions.csv.gz", float_precision="round_trip")
    data["text"] = data.text_key  # Pairing identifier. Features were extracted before deidentification.
    data["correct"] = data.true_label.eq(data.predicted_label).astype(int)
    return data

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    core_data = read_predictions("rq1_3")
    core.run_rq1(core_data, out)
    core.run_rq2(core_data, out)
    core.run_rq3(core_data, out)
    core.run_error_overlap(core_data, out)
    core.run_regression(core_data, out)
    unseen = read_predictions("rq4")
    metrics = transfer.rq4_metrics(unseen, out)
    transfer.rq4_detector_comparison(unseen, out)
    transfer.rq4_training_dataset_comparison(unseen, out)
    transfer.rq4_friedman(metrics, out)
    transfer.make_rq4_plots(metrics, out)
    old = core_data[core_data.train_llm.isin(transfer.RQ5_LLMS) & core_data.test_llm.isin(transfer.RQ5_LLMS)]
    original = transfer.compute_cell_metric_table(old, prompt_name="original")
    changed = transfer.compute_cell_metric_table(read_predictions("rq5"), prompt_name="new")
    paired = transfer.match_rq5_original_and_new(original, changed)
    for name, frame in [("original_prompt_metrics", original), ("new_prompt_metrics", changed), ("original_vs_new_prompt", paired)]:
        frame.to_csv(out / f"rq5_{name}.csv", index=False)
    transfer.rq5_prompt_change_wilcoxon(paired, out)
    transfer.rq5_per_detector_wilcoxon(paired, out)
    transfer.rq5_largest_changes(paired, out)
    transfer.make_rq5_plots(paired, out)

if __name__ == "__main__":
    main()
