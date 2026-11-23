"""Create numeric Tables 4–8 with evidence for each AUROC."""
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "outputs"
keys = ["table","source_dataset","source_llm","target_dataset","target_llm","detector"]
summaries = pd.read_csv(root / "data/summary_auc.csv").set_index(keys,verify_integrity=True)
tables = []
for table, name in enumerate(["rq1_in_domain","rq2_llm_cross_domain","rq3_dataset_cross_domain","rq4_unseen_both","rq5_new_prompt"], 4):
    df = pd.read_csv(out / f"{name}_metrics.csv")
    for axis in ["dataset","llm"]:
        if axis in df:
            df["train_"+axis] = df["test_"+axis] = df[axis]
    df = df.rename(columns={"train_dataset":"source_dataset","train_llm":"source_llm","test_dataset":"target_dataset","test_llm":"target_llm","model":"detector","auc_score_col":"auc"})
    df["table"] = table
    df = df.set_index(keys,verify_integrity=True)
    df["auc_source"] = "prediction scores"
    missing = df.index[df.auc.isna()]
    df.loc[missing,"auc"] = summaries.loc[missing,"table_value_full_precision"]
    df.loc[missing,"auc_source"] = summaries.loc[missing,"archive_member"]
    tables.append(df[["auc","ai_recall","human_recall","auc_source"]])
result = pd.concat(tables).reset_index()
if len(result)!=294 or result.isna().any().any():
    raise ValueError("Incomplete performance table")
result.to_csv(out / "tables_4_to_8.csv",index=False)
