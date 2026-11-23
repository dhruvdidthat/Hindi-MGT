"""Compare replayed outputs with archival full-precision reference results."""
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
for reference in sorted((root / "analysis/reference").glob("*.csv")):
    expected = pd.read_csv(reference)
    actual = pd.read_csv(root / "outputs" / reference.name)[expected.columns]
    # Rank ties can appear in a different order across pandas versions.
    keys = list(expected.select_dtypes(include=["object"]).columns)
    if "table" in expected:
        keys.append("table")
    def order(frame):
        return frame.sort_values(keys, kind="stable").reset_index(drop=True) if keys else frame
    pd.testing.assert_frame_equal(order(expected), order(actual), check_dtype=False,
                                  atol=1e-8, rtol=1e-7, obj=reference.name)
    print(reference.name, len(actual), "rows verified")
