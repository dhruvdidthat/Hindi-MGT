"""Render Figures 1–6. Run analysis/reproduce.py first."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

root = Path(__file__).resolve().parents[1]
out = root / "outputs"
dest = out / "figures"
dest.mkdir(exist_ok=True,parents=True)
def save(fig, number):
    fig.tight_layout()
    fig.savefig(dest / f"figure{number}.pdf", bbox_inches="tight")
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12,4))
for ax, ds in zip(axes, ["hindisumm","xquad"]):
    cumulative = ax.twinx()
    for generator in ["gemini","llama3","qwen3"]:
        frame = pd.read_csv(root / f"training_data/{ds}/{ds}_train_{generator}.csv")
        for label, name in [(0,generator),(1,"Human")]:
            if label == 1 and generator != "gemini":
                continue
            counts = frame.loc[frame.label.eq(label),"text"].str.split().str.len().dropna()
            _,_,bars = ax.hist(counts,bins=40,alpha=.25,label=name)
            cumulative.ecdf(counts,color=bars[0].get_facecolor()[:3])
    ax.set(title=ds,xlabel="Words",ylabel="Texts")
    cumulative.set_ylabel("Cumulative fraction")
    ax.legend()
save(fig, 1)

for number, name, group, metric in [
    (2,"rq2_llm_cross_domain_metrics","dataset","ai_recall"),
    (3,"rq3_dataset_cross_domain_metrics","llm","human_recall"),
]:
    frame = pd.read_csv(out / (name+".csv"))
    fig, ax = plt.subplots(figsize=(9,5))
    for i, (key, part) in enumerate(frame.groupby(group)):
        for j, (model, rows) in enumerate(part.groupby("model")):
            x = j + (i-.75)*.14
            values = rows[metric]
            ax.vlines(x,values.min(),values.max(),color=f"C{i}",label=key if j==0 else None)
            colors = rows.train_dataset.map({"hindisumm":"tab:blue","xquad":"tab:orange"}) if number==3 else f"C{i}"
            ax.scatter(np.repeat(x,len(rows)), values,c=colors,s=16)
    ax.set_xticks(range(frame.model.nunique()),sorted(frame.model.unique()),rotation=25,ha="right")
    ax.set_ylabel(metric.replace("_"," ").title())
    ax.legend(title=group)
    save(fig,number)

frame = pd.read_csv(root / "experiments/rq1_3/predictions.csv.gz")
features = ["ASL","AWL","ASW","PSW30","JUK"]
keys = ["model","train_dataset","train_llm","test_dataset","test_llm"]
groups = frame.groupby(keys)[features]
z = (frame[features]-groups.transform("mean"))/groups.transform("std",ddof=0)
labels = {0:"H",1:"AI"}
cases = frame.true_label.map(labels)+" → "+frame.predicted_label.map(labels)
bars = z.groupby(cases).mean().T[["H → H","AI → AI","AI → H","H → AI"]]
ax = bars.plot.bar(rot=0,color=["#0072B2","#E69F00","#009E73","#CC79A7"])
ax.axhline(0,color="black",lw=.6)
ax.set_ylabel("Mean standardized value")
save(ax.figure,4)

pairs = pd.read_csv(out / "error_overlap_pairwise.csv")
sym = pd.concat([pairs,pairs.rename(columns={"model_a":"model_b","model_b":"model_a"})])
matrix = sym.pivot_table(index="model_a",columns="model_b",values="jaccard")
fig,ax = plt.subplots(figsize=(8,7))
sns.heatmap(matrix,cmap="YlGnBu",annot=True,fmt=".2f",square=True,ax=ax,cbar_kws={"label":"Mean Jaccard overlap"})
ax.set(xlabel="Detector",ylabel="Detector",title="Pairwise error overlap")
save(fig,5)

coefs = pd.read_csv(out / "regression_coefficients.csv")
fig,ax = plt.subplots(figsize=(8,4))
for i,(outcome,part) in enumerate(coefs[coefs.term.str.endswith("_z")].groupby("outcome")):
    part = part.sort_values("term")
    y = np.arange(len(part))
    ax.errorbar(part.coef,y+i*.15,xerr=1.96*part.std_err,fmt="o",label=outcome)
ax.set_yticks(y,part.term.str.removesuffix("_z"))
ax.axvline(0,color="black",lw=.6)
ax.set_xlabel("Change in log odds per standard deviation")
ax.legend(fontsize=8)
save(fig,6)
