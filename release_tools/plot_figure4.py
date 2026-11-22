from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
F=["asl","awl","asw","psw30","juk"]
x=pd.read_csv(Path(__file__).with_name("figure4_inputs.csv.gz"))
norm_groups=['model','train_dataset','train_llm','test_dataset','test_llm']
x[F]=x.groupby(norm_groups,group_keys=False)[F].transform(lambda col:(col-col.mean())/col.std(ddof=0))
v=x.groupby('case')[F].mean().T[['H → H','AI → AI','AI → H','H → AI']];v.index=v.index.str.upper()
print(v,x.case.value_counts())
plt.rcParams.update({'pdf.fonttype':42,'font.size':10})
ax=v.plot.bar(figsize=(7,3.3),width=.80,color=['#0072B2','#E69F00','#009E73','#CC79A7'],edgecolor='#333333',linewidth=.5,rot=0)
ax.axhline(0,color='#333333',lw=.7);ax.set_ylabel('Mean standardized value');ax.set_xlabel('Linguistic feature');ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=4,frameon=False);plt.tight_layout();plt.savefig("figure4_local.pdf",bbox_inches='tight')
print("Wrote figure4_local.pdf")
