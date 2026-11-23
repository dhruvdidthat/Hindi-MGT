"""Convert one fresh, aligned inference export to paper label and score conventions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--truth", required=True, help="Column containing reference labels")
parser.add_argument("--truth-ai", type=int, choices=[0,1], required=True)
parser.add_argument("--prediction", help="Stored hard decision column")
parser.add_argument("--prediction-ai", type=int, choices=[0,1])
parser.add_argument("--score", help="Continuous score column")
parser.add_argument("--kind", choices=["p-human","p-ai","perplexity","llr"])
parser.add_argument("--threshold", type=float, help="Frozen source cutoff for perplexity or LLR")
args = parser.parse_args()
df = pd.read_csv(args.input, float_precision="round_trip")
def binary(name):
    values = pd.to_numeric(df[name], errors="raise")
    if not values.isin([0,1]).all():
        parser.error(f"{name} contains a missing or nonbinary label")
    return values
truth = binary(args.truth).eq(args.truth_ai)
if bool(args.score) != bool(args.kind):
    parser.error("Specify both --score and --kind")
score = np.nan
if args.score:
    raw = pd.to_numeric(df[args.score], errors="raise")
    if not np.isfinite(raw).all():
        parser.error("Review missing/infinite scores before converting a fresh export")
    if args.kind.startswith("p-") and not raw.between(0,1).all():
        parser.error("Probability outside [0,1]")
    score = {"p-human":1-raw, "p-ai":raw, "perplexity":-raw, "llr":raw}[args.kind]
if args.kind in ("perplexity","llr"):
    if args.threshold is None or not np.isfinite(args.threshold):
        parser.error("A finite source threshold is required")
    pred = raw.le(args.threshold) if args.kind=="perplexity" else raw.ge(args.threshold)
else:
    if not args.prediction or args.prediction_ai is None:
        parser.error("Specify the prediction column and its raw AI class")
    pred = binary(args.prediction).eq(args.prediction_ai)
out = df.copy()
out["true_label"], out["predicted_label"], out["score"] = truth.astype(int), pred.astype(int), score
out.to_csv(args.output, index=False, mode="x")
