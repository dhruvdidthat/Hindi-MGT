#!/usr/bin/env python3
"""Repair the exact audited Hindi-MGT statistical package in a new sibling directory."""
import argparse
import csv
import hashlib
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

PAYLOAD = {'after': {'config.py': '3a577ce4b42dc8bbff4ab1782ea27a7914d76d9e973ea13b4ac0090c0424c561',
           'loaders.py': '44daf9a82f7cea515ab0a3e1f817cb53cf242cd2485416e67168d3f1748f9796',
           'pipeline_rq45.py': '9d58ea86816386ff83ea17b4e327c68b45bed34c7463aa3c3cf125e516618f18'},
 'before': {'config.py': 'd5b8ba459df944c5952a3e1f0df41942b30692be95af41c25317ae9bb6d016da',
            'linguistic_features.py': 'd40d40838ac1e9711ad2d9a81de265a39b885a4960902c2106092b8f840ecf9c',
            'loaders.py': '9dac4a38e8fd8edb57975bb8cf9db9499b71bd9f5c9a82ff032096669f33a2a3',
            'matching.py': 'a9024e80dd324ffde719443c6236b612aadbb788fb8180f7b6e0b098968a41f6',
            'pipeline.py': '50b57e760333c434783a7b67a8d9f9107af25b57cf087c3b69a806090108f358',
            'pipeline_rq45.py': '6688e54700f2ddffd0dffb00714064b779dd5732a15a41b6bf3117f8a69026be',
            'preflight_zero_shot.py': '3ccf4849fb008a5972c73b5d9b9816e0f403f7857882a032143562ab32de907e',
            'requirements.txt': 'd8443d2587e45a69806d9d9907a8953a8364971e881929b77ab4f8c6faf52dd0',
            'stats_tests.py': 'd6129d0f9bed887898ff801dbc42da2e16770a23b83ee7fbe3f638d9012d50e8'},
 'patch': {'config.py': [(44,
                          46,
                          '    "Text-Fluoroscopy": ModelSpec(0, False, "predicted_prob_label1", '
                          'True,\n'
                          '        "Audited server exports use raw 0=AI; class-1 probability is '
                          'P(Human).", "Convert both labels and use 1-p for the AI score."),\n')],
           'loaders.py': [(675, 681, '    # The audited server files encode human as class 1.\n'),
                          (703, 706, ''),
                          (709, 710, '                        raw_ai_label=0,\n'),
                          (715, 716, '                            "human_prob",\n'),
                          (822,
                           822,
                           '        df = df.loc[(df.train_dataset == df.test_dataset) | '
                           '(df.train_llm == df.test_llm)].copy()\n')],
           'pipeline_rq45.py': [(40,
                                 42,
                                 '    - The new-prompt XLM-R exporter already writes canonical '
                                 'AI=1 labels and P(AI) in predicted_prob_label1.\n'),
                                (350, 351, '        "raw_ai_label": 0,\n'),
                                (352, 353, '        "score_direction": "human_prob",\n'),
                                (861,
                                 863,
                                 '#   new-prompt export class 1 = AI; predicted_prob_label1 = '
                                 'P(AI)\n'),
                                (869,
                                 871,
                                 '#   server export class 0 = AI; predicted_prob_label1 = '
                                 'P(Human)\n'),
                                (879,
                                 881,
                                 '        "raw_ai_label": 1,\n'
                                 '        "score_col": "predicted_prob_label1",\n'),
                                (923, 924, '        "raw_ai_label": 0,\n'),
                                (925, 926, '        "score_direction": "human_prob",\n'),
                                (946,
                                 954,
                                 'def rq5_cases():\n'
                                 '    return {\n'
                                 '        (ds, llm, target_ds, target_llm)\n'
                                 '        for ds, llm, target_ds, target_llm in '
                                 'itertools.product(\n'
                                 '            DATASETS, RQ5_LLMS, DATASETS, RQ5_LLMS\n'
                                 '        )\n'
                                 '        if ds == target_ds or llm == target_llm\n'
                                 '    }\n'),
                                (955, 957, ''),
                                (958,
                                 993,
                                 'def check_rq5_coverage(frame):\n'
                                 '    keys = ["train_dataset", "train_llm", "test_dataset", '
                                 '"test_llm"]\n'
                                 '    expected = rq5_cases()\n'
                                 '    for model in RQ45_MODELS:\n'
                                 '        actual = set(frame.loc[frame.model == model, '
                                 'keys].itertuples(index=False, name=None))\n'
                                 '        if actual != expected:\n'
                                 '            raise ValueError(f"RQ5 {model}: '
                                 'missing={sorted(expected-actual)} '
                                 'extra={sorted(actual-expected)}")\n'),
                                (1040,
                                 1052,
                                 '        for train_ds, train_llm, test_ds, test_llm in '
                                 'sorted(rq5_cases()):\n'
                                 '            source = f"{train_ds}_{train_llm}"\n'
                                 '            target = f"{test_ds}_{test_llm}"\n'
                                 '            if cfg["layout"] == "nested":\n'
                                 '                prediction_path = base / source / '
                                 'f"tested_on_{target}" / "predictions.csv"\n'
                                 '            else:\n'
                                 '                prediction_path = base / '
                                 'f"{source}_tested_on_{target}" / "predictions.csv"\n'),
                                (1093, 1093, '    check_rq5_coverage(result)\n'),
                                (1167, 1167, '    check_rq5_coverage(master)\n'),
                                (1576,
                                 1576,
                                 '        "RQ5 uses the full archived prompt cohorts. Training '
                                 'overlap and different old/new cohort sizes limit prompt-only '
                                 'inference. Treat those tests as exploratory.",\n')]}}
VERIFIER = r'''"""Validate the audited server version and record the input files actually read."""
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from loaders import load_all, set_detectgpt_rebuilt_root
from matching import normalize_text_for_matching as key
from pipeline_rq45 import DATASETS, LLMS, RQ45_MODELS, load_rq4_coling, load_rq5_new_prompt, rq5_cases

root, out = map(Path, sys.argv[1:3])
training = root / "training_models"
audit = root / "detectgpt_audit_20261120_210343"
set_detectgpt_rebuilt_root(audit)
files, checks, failures = {}, [], []
read = pd.read_csv

def tracked(path, *args, **kwargs):
    frame = read(path, *args, **kwargs)
    p = Path(path).resolve()
    if p not in files:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        files[p] = h.hexdigest()
    return frame

def note(message, condition, **detail):
    checks.append(dict(check=message, passed=bool(condition), **detail))
    if not condition:
        failures.append(message)

def keys(frame, labels):
    return Counter(zip(frame.text.map(key), labels.astype(int)))

pd.read_csv = tracked
try:
    source_thresholds = {}
    for ds,llm in itertools.product(DATASETS,LLMS):
        fname = f"train_{ds}_{llm}.csv"
        cache = tracked(root / "COLING_2025_MGT_Task1/detect_gpt_all_domain/inference_results/calibrated/score_cache" / fname)
        src = tracked(training / "training_data" / ds / f"{ds}_train_{llm}.csv")
        valid = cache.loc[np.isfinite(cache.llr) & cache.label.isin([0,1])]
        fpr,tpr,cut = roc_curve(valid.label.eq(0),valid.llr)
        source_thresholds[ds,llm] = float(cut[np.argmax(tpr-fpr)])
        note(f"source calibration {ds}/{llm}", not (keys(valid,valid.label)-keys(src,src.label)),
             source_rows=len(src), scored_rows=len(valid), threshold=source_thresholds[ds,llm])
    original=load_all(str(training))
    coling=load_rq4_coling(root / "COLING_2025_MGT_Task1", calibration_root=training)
    prompt=load_rq5_new_prompt(root / "prompt_change", calibration_root=training)
    four=["train_dataset","train_llm","test_dataset","test_llm"]
    expected={c for c in itertools.product(DATASETS,LLMS,DATASETS,LLMS) if c[0]==c[2] or c[1]==c[3]}
    official=tracked(root / "COLING_2025_MGT_Task1/hindi_data.csv")
    note("COLING metadata",official.label.eq(official.model.str.strip().str.lower().eq("human").astype(int)).all())
    thresholds={}
    from matching import add_text_key, safe_paired_matrix
    from stats_tests import mcnemar_pairwise, holm_correction, cochrans_q_test
    test_p=[]
    for (ds,llm), cell in original.loc[(original.train_dataset==original.test_dataset) & (original.train_llm==original.test_llm)].groupby(["test_dataset","test_llm"]):
        pivot,truth,detail=safe_paired_matrix(add_text_key(cell),value_col="predicted_label")
        note(f"RQ1 pairing {ds}/{llm}",pivot.shape[1]==7 and len(pivot)==cell.loc[cell.model=="DetectGPT","text"].map(key).nunique() and detail["n_dropped_truth_mismatch"]==0,paired=len(pivot))
        if len(pivot) and pivot.shape[1]==7:
            q,df,pq=cochrans_q_test(pivot.eq(truth,axis=0).astype(int).values)
            for a,b in itertools.combinations(pivot.columns,2):
                test_p.append(mcnemar_pairwise(truth.values,pivot[a].values,pivot[b].values)[0])
            checks[-1]["cochran_p"]=float(pq)
    note("RQ1 computed pairwise family", len(test_p)==126 and np.isfinite(test_p).all(), comparisons=len(test_p),
         significant_global_holm=int(sum(np.asarray(holm_correction(test_p))<0.05)) if test_p else None)
    for group,frame,cases in [
        ("RQ1-3",original,expected),
        ("RQ4",coling,{(d,l,"coling_hindi","gpt4o") for d,l in itertools.product(DATASETS,LLMS)}),
        ("RQ5",prompt,rq5_cases())
    ]:
        refs={}
        for case, rows in frame.groupby(four):
            ds,llm,td,tl=case
            if (td,tl) not in refs:
                if group=="RQ4":
                    ref=official
                elif group=="RQ5":
                    ref=tracked(root / "prompt_change/new_prompt_data" / f"{td}_{tl}_new_prompt.csv")
                else:
                    ref=tracked(training / "testing_data" / td / f"{td}_test_{tl}.csv")
                refs[td,tl]=keys(ref,1-ref.label)
            for model,cell in rows.groupby("model"):
                got=keys(cell,cell.true_label)
                refkeys=refs[td,tl]
                missing=refkeys-got
                extra=got-refkeys
                allowed=Counter()
                if group=="RQ5" and model=="GPTZero" and td=="xquad":
                    # Use the record itself rather than assuming a hash representation.
                    input_rows=tracked(root / "prompt_change/new_prompt_data" / f"{td}_{tl}_new_prompt.csv")
                    found=input_rows.loc[input_rows.id.eq(18 if tl=="gemini" else 17) & input_rows.label.eq(1)]
                    # IDs are local to the generator file. Verify against the blank scored row.
                    scores=tracked(root/"prompt_change/gptzero/inference_results_new_prompt"/f"{ds}_{llm}_tested_on_{td}_{tl}"/"predictions.csv")
                    blanks=Counter(scores.loc[scores.perplexity.isna(),"text"].map(key))
                    allowed=Counter({(h,0):1 for h in found.text.map(key) if blanks[h]==1})
                note(f"{group} {case} {model}: reference truth and coverage",
                     not extra and missing==allowed and cell.predicted_label.isin([0,1]).all(),
                     rows=len(cell),missing=sum(missing.values()),extra=sum(extra.values()))
                if model in ("DetectGPT","GPTZero"):
                    cutoffs=cell.calibration_threshold.unique()
                    threshold=float(cutoffs[0])
                    pred=cell.score.ge(threshold if model=="DetectGPT" else -threshold).astype(int)
                    k=(model,ds,llm)
                    saved=thresholds.setdefault(k,threshold)
                    ok=len(cutoffs)==1 and threshold==saved and pred.eq(cell.predicted_label).all()
                    if model=="DetectGPT":
                        ok=ok and np.isclose(threshold,source_thresholds[ds,llm],atol=1e-12,rtol=0)
                    note(f"{group} {case} {model}: frozen decisions",ok,threshold=threshold)
        for model in RQ45_MODELS:
            seen=set(frame.loc[frame.model==model,four].itertuples(index=False,name=None))
            note(f"{group} {model}: case coverage",seen==cases,missing=sorted(cases-seen),extra=sorted(seen-cases))
finally:
    pd.read_csv=read
    (out/"input_sha256.json").write_text(json.dumps({str(p.relative_to(root)):h for p,h in sorted(files.items())},indent=2))
    (out/"verification.json").write_text(json.dumps(checks,indent=2))
if failures:
    raise SystemExit("Integrity checks failed; see verification.json: "+" | ".join(failures[:4]))
print(f"Passed {len(checks)} integrity checks.")
'''
VERSION = "wsdm27-stats-repair-1"
NOTES = """# Hindi MGT statistics repair 1

Use this directory for corrected statistical analysis of the audited server exports.
The source package and all training, inference and calibration data are preserved.
The JSON manifest identifies the input and repaired code. Each verification run records
the SHA-256 and relative path of each CSV consumed by the verification loaders.

* Text-Fluoroscopy: raw AI=0 in these server exports. Convert truth and hard predictions and use 1-P(class 1).
* New-prompt XLM-R: exported AI=1, use predicted_prob_label1 unchanged.
* CONDA: traverse source/tested_on_target/predictions.csv for new prompts.
* Restrict RQ1–3 to the 24 study keys and both RQ5 arms to 12 keys per model.
* Preserve the source-only GPTZero thresholds and the specified DetectGPT v3 rebuild.
* Mark full-cohort RQ5 as exploratory. No split, checkpoint, prediction or threshold was rewritten.

Publication status: candidate statistical implementation. A verified run establishes
the listed input, label, and coverage checks. Source data include training_models/training_data
and testing_data, COLING_2025_MGT_Task1/hindi_data.csv and prompt_change/new_prompt_data.
Per-detector prediction paths and versions actually consumed are in runs/*/input_sha256.json.
A changed SHA on the live server identifies a different artifact from a previous run.
Keep generated findings paired with the run directory and its run.json. The original
server package and its results are historical inputs, not the repaired output.

Research qualifications: XLM-R and Indic-BERT evaluate held-out test data during training.
Final-epoch saving alone does not prove whether the authors used these metrics for tuning.
CONDA uses unlabeled target training texts for adaptation. The archived RADAR training
script updates the second reward moment with mean(rewards)**2 and its log-probability
averaging includes padded positions. Resolve the checkpoint's training provenance before
claiming a standard PPO implementation. GPTZero decision replay was verified, but its
original threshold fitting was not independently reproduced from training perplexities.
Full-cohort prompt comparisons and repeated transfer cells do not establish an independent
held-out prompt effect. Check the resampling unit and correction family before reporting p-values.
 The seven-detector pipeline covers RQ1–5.
RQ5 held-out prompt pairing, RQ6 Nota AI checkpoint semantics, checkpoint selection
history and final manuscript reconciliation remain open. Only the source scripts are
versioned here; no research dataset or model has been approved for public release.
"""
def sha(b):
    return hashlib.sha256(b).hexdigest()

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path.cwd(),help="server CODE directory")
    p.add_argument("--verify",action="store_true",help="verify current data and record exact source paths")
    p.add_argument("--run",action="store_true",help="verify first, then run RQ1–5 in a fresh output directory")
    args=p.parse_args()
    root=args.root.resolve()
    src=root/"mgt_stat_pipeline_updated"
    dest=root/"mgt_stat_pipeline_repaired_v1"
    built={}
    for name,want in PAYLOAD["before"].items():
        data=(src/name).read_bytes()
        if sha(data)!=want:
            p.exit(2,f"Version mismatch: {src/name}. No changes made.\n")
        if name in PAYLOAD["patch"]:
            lines=data.decode().splitlines(True)
            for i,j,replacement in reversed(PAYLOAD["patch"][name]):
                lines[i:j]=[replacement]
            data="".join(lines).encode()
            if sha(data)!=PAYLOAD["after"][name]:
                raise RuntimeError(f"Repaired hash mismatch: {name}")
        if name.endswith(".py"):
            compile(data,name,"exec")
        built[name]=data
    built["changes.diff"]="".join(line for n in PAYLOAD["patch"] for line in difflib.unified_diff((src/n).read_text().splitlines(True),built[n].decode().splitlines(True),fromfile="original/"+n,tofile="repaired/"+n)).encode()
    built["verify_repair.py"]=VERIFIER.encode()
    built["REPAIR_NOTES.md"]=NOTES.encode()
    manifest=dict(version=VERSION,installer_sha256=sha(Path(__file__).read_bytes()),original_code=PAYLOAD["before"],
                  repaired_files={k:sha(v) for k,v in built.items()},source_package=src.name,
                  detectgpt_source="detectgpt_audit_20261120_210343",data_modifications=[])
    built["REPAIR_MANIFEST.json"]=(json.dumps(manifest,indent=2)+"\n").encode()
    if dest.exists():
        if not all((dest/k).is_file() and (dest/k).read_bytes()==v for k,v in built.items()):
            p.exit(2,f"Existing repaired directory differs; preserving {dest}\n")
    else:
        tmp=Path(tempfile.mkdtemp(prefix=".stats-repair-",dir=root))
        try:
            for k,v in built.items():
                (tmp/k).write_bytes(v)
            tmp.rename(dest)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
    print("Repaired statistical code:",dest,flush=True)
    if args.verify or args.run:
        runs=dest/"runs"
        runs.mkdir(exist_ok=True)
        out=Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_"),dir=runs))
        common=["--root",str(root/"training_models"),"--detectgpt-rebuilt-root",str(root/manifest["detectgpt_source"])]
        commands=[["verify_repair.py",str(root),str(out)]]
        if args.run:
            commands+=[["pipeline.py",*common,"--outdir",str(out/"rq123")],
                       ["pipeline_rq45.py",*common,"--rq4-root",str(root/"COLING_2025_MGT_Task1"),"--rq5-root",str(root/"prompt_change"),"--outdir",str(out/"rq45")]]
        record=dict(version=VERSION,python=sys.version,commands=commands,completed=[],status="running")
        env=dict(os.environ,MPLBACKEND="Agg",PYTHONUNBUFFERED="1")
        print("Run directory:",out,flush=True)
        try:
            for cmd in commands:
                write_json(out/"run.json",record)
                print("Running",cmd[0],flush=True)
                with (out/(cmd[0]+".log")).open("w") as log:
                    subprocess.run([sys.executable,*cmd],cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                record["completed"].append(cmd[0])
            if args.run:
                expected = {"rq123/rq1_in_domain_metrics.csv":42, "rq123/rq1_cochrans_q.csv":6,
                            "rq123/rq1_mcnemar_pairwise.csv":126, "rq123/rq2_llm_cross_domain_metrics.csv":84,
                            "rq123/rq3_dataset_cross_domain_metrics.csv":42, "rq45/rq4_unseen_both_metrics.csv":42,
                            "rq45/rq4_mcnemar_pairwise.csv":126, "rq45/rq5_original_prompt_metrics.csv":84,
                            "rq45/rq5_new_prompt_metrics.csv":84, "rq45/rq5_original_vs_new_prompt.csv":84}
                counts={}
                for rel, count in expected.items():
                    with (out/rel).open(newline="") as f:
                        counts[rel]=sum(1 for row in csv.DictReader(f))
                    if counts[rel]!=count:
                        raise RuntimeError(f"{rel}: expected {count} rows, found {counts[rel]}")
                record["output_row_counts"]=counts
            record["status"]="completed"
        except BaseException:
            record["status"]="failed_or_interrupted"
            raise
        finally:
            write_json(out/"run.json",record)
        print("Completed:",out)

if __name__=="__main__":
    main()
