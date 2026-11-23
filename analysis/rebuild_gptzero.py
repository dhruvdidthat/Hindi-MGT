"""Apply the frozen source thresholds to the saved mGPT test perplexities."""
from pathlib import Path
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parents[1]
rows = pd.read_csv(root / "data/gptzero_perplexities.csv.gz", float_precision="round_trip")
cutoffs = pd.read_csv(root / "data/source_thresholds.csv",float_precision="round_trip")
rows = rows.merge(cutoffs[cutoffs.model.eq("GPTZero")],on=["train_dataset","train_llm"],validate="many_to_one",how="left")
if rows.calibration_threshold.isna().any():
    raise ValueError("Missing source threshold")
rows["predicted_label"] = rows.perplexity.le(rows.calibration_threshold).astype(int)
rows["true_label"] = rows.raw_label.eq(0).astype(int)
keys = ["train_dataset","train_llm","test_dataset","test_llm"]
finite = rows.perplexity.where(np.isfinite(rows.perplexity))
maximum = finite.groupby([rows[k] for k in ["experiment",*keys]]).transform("max")
rows["score"] = -finite.fillna((10*maximum).fillna(1e6))
columns = [*keys,"text_key","true_label","predicted_label","score"]
def ordered(df):
    return df[columns].sort_values(columns).reset_index(drop=True)
for group, part in rows.groupby("experiment"):
    expected = pd.read_csv(root/"experiments"/group/"predictions.csv.gz",float_precision="round_trip")
    pd.testing.assert_frame_equal(ordered(part),ordered(expected[expected.model.eq("GPTZero")]))
out=root/"outputs"
out.mkdir(exist_ok=True)
rows.to_csv(out/"gptzero_rebuilt.csv.gz",index=False)
print("Verified GPTZero decisions and scores across RQ1–RQ5.")
