import argparse
import pandas as pd,numpy as np,zipfile,json
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description='Rebuild Figure 4 using per-inference-run normalization.')
parser.add_argument('--evidence-zip',required=True,help='Hindi MGT evidence ZIP from hindi_mgt_audit.py')
parser.add_argument('--stats-zip',required=True,help='Repaired statistical pipeline output ZIP')
parser.add_argument('--output',default='Figure_4_linguistic_profiles_local_normalization.pdf')
args=parser.parse_args()
z=zipfile.ZipFile(args.evidence_zip);run=zipfile.ZipFile(args.stats_zip)
def rd(p):return pd.read_csv(z.open('tables/'+p+'.tsv'),sep='\t',low_memory=False)
def rf(s):return pd.read_csv(run.open(next(n for n in run.namelist() if n.endswith('/'+s))))
manifest=json.loads(run.read(next(n for n in run.namelist() if n.endswith('/input_sha256.json'))))
F=['asl','awl','asw','psw30','juk']
feat=rd('mgt_stat_pipeline_updated/results_rq123/master_predictions_with_features.csv')[['__text_key']+F].drop_duplicates().set_index('__text_key');assert feat.index.is_unique
paths={'XLM-RoBERTa':'xlm_roberta','Indic-BERT':'indicBert','CONDA':'CONDA','RADAR':'RADAR','Text-Fluoroscopy':'text_fluroscopy','GPTZero':'gptzero'}
rows=[]
for rq,name in enumerate(['rq1_in_domain_metrics.csv','rq2_llm_cross_domain_metrics.csv','rq3_dataset_cross_domain_metrics.csv'],1):
 for r in rf(name).to_dict('records'):
  d=r.get('train_dataset',r.get('dataset'));e=r.get('test_dataset',d);l=r.get('train_llm',r.get('llm'));g=r.get('test_llm',r.get('llm'));m=r['model'];a=f'{d}_{l}';b=f'{e}_{g}'
  if m=='DetectGPT':p=f'detectgpt_audit_20261120_210343/rebuilt_from_score_cache/rq123/train_{a}__test_{b}/predictions.csv'
  else:
   sub=f'{a}/tested_on_{b}'
   if m=='GPTZero':sub=a if a==b else f'cross_train_{a}__test_{b}'
   if m=='CONDA':sub=f'{a}_{l}/tested_on_{b}'
   if m=='Text-Fluoroscopy':sub=f'{a}_tested_on_{b}'
   p=f'training_models/{paths[m]}/inference_results/{sub}/predictions.csv'
  assert p in manifest,p
  df=rd(p);ref=rd(f'training_models/testing_data/{e}/{e}_test_{g}.csv')
  keys=['__text_key','__occurrence'];df=df.merge(ref[keys+['label']].rename(columns={'label':'ref'}),on=keys,validate='one_to_one')
  y=1-df.ref;pred=df.predicted_label if m=='DetectGPT' else 1-df.predicted_label
  if m=='GPTZero':
   cuts=rd('training_models/gptzero/inference_results/gptzero_all_results.csv');cut=cuts.query('train_dataset==@d and train_llm==@l').threshold.unique();assert len(cut)==1;pred=(df.perplexity<=cut[0]).astype(int)
  assert len(df)==r['n'] and np.allclose([(pred[y==1]==1).mean(),(pred[y==0]==0).mean()],[r['ai_recall'],r['human_recall']]),(m,a,b)
  x=feat.loc[df.__text_key].reset_index(drop=True);x['case']=np.array(['H → H','H → AI','AI → H','AI → AI'])[2*y+pred]
  x['model']=m;x['train_dataset']=d;x['train_llm']=l;x['test_dataset']=e;x['test_llm']=g;rows.append(x)
norm_groups=['model','train_dataset','train_llm','test_dataset','test_llm']
x=pd.concat(rows,ignore_index=True);assert x[F].notna().all().all() and len(rows)==168 and len(x)==34412
group_sizes=x.groupby(norm_groups).size()
assert len(group_sizes)==168 and group_sizes.min()>1
x[F]=x.groupby(norm_groups,group_keys=False)[F].transform(lambda col:(col-col.mean())/col.std(ddof=0))
v=x.groupby('case')[F].mean().T[['H → H','AI → AI','AI → H','H → AI']];v.index=v.index.str.upper()
print(v,x.case.value_counts())
plt.rcParams.update({'pdf.fonttype':42,'font.size':10})
ax=v.plot.bar(figsize=(7,3.3),width=.80,color=['#0072B2','#E69F00','#009E73','#CC79A7'],edgecolor='#333333',linewidth=.5,rot=0)
ax.axhline(0,color='#333333',lw=.7);ax.set_ylabel('Mean standardized value');ax.set_xlabel('Linguistic feature');ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=4,frameon=False);plt.tight_layout();plt.savefig(args.output,bbox_inches='tight')
print(f'Wrote {args.output}')
