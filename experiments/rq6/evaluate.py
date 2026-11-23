"""Replay Table 9 with each evaluator's published positive class."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / "predictions.csv.gz")
rows = []
for (model, file), group in data.groupby(["detector", "file"]):
    ai = int(model == "grape")
    truth = 1 - group.label if ai else group.label
    pred = group.prediction
    rows.append(dict(detector=model, file=file, accuracy=accuracy_score(truth,pred),
                     f1_score=f1_score(truth,pred), roc_auc=roc_auc_score(truth, group.machine_prob),
                     ai_recall=recall_score(truth,pred,pos_label=ai),
                     human_recall=recall_score(truth,pred,pos_label=1-ai)))
out = folder.parents[1] / "outputs"
out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / "rq6_metrics.csv", index=False)
pd.DataFrame(rows).round(4).to_csv(out / "table9_printed.csv", index=False)
