"""Compare normalized raw Text-Fluoroscopy predictions to all 42 evaluation exports."""
import pandas as pd
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp) / "normalized.csv.gz"
    subprocess.run([sys.executable,str(root/"analysis/normalize_export.py"),str(root/"data/text_fluoroscopy_exports.csv.gz"),str(output),
                    "--truth","true_label","--truth-ai","0","--prediction","predicted_label","--prediction-ai","0",
                    "--score","predicted_prob_label1","--kind","p-human"],check=True)
    raw = pd.read_csv(output)
keys = ["train_dataset","train_llm","test_dataset","test_llm","text_key","true_label","predicted_label","score"]
def ordered(frame):
    return frame[keys].sort_values(keys).reset_index(drop=True)
for group, part in raw.groupby("experiment"):
    final = pd.read_csv(root/"experiments"/group/"predictions.csv.gz")
    pd.testing.assert_frame_equal(ordered(part),ordered(final[final.model.eq("Text-Fluoroscopy")]))
print("Verified Text-Fluoroscopy labels and AI scores.")
