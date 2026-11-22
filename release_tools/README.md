# Hindi MGT paper reproduction

“How Well Do Machine Generated Text Detectors Transfer to Hindi? Performance, Transfer, and Failure Analysis”

## Build the server workspace

Keep this complete release_tools directory together. From the original CODE root, in the existing research environment, run:

```sh
python3 release_tools/organize_hindi_mgt.py
python3 release_tools/organize_hindi_mgt.py --apply
cd Hindi_MGT_paper_release
python3 reproduction/repair_mgt_stats.py --root CODE --run
```

The first command checks the SHA-256 of each required server file. The second copies those files to a fresh directory and verifies each copy. Original paths are preserved under CODE because the archived loaders expect that layout. Other research work stays in the original server tree. An existing release blocks a second installation. If a checksum differs, compare the named file to the audit before selecting a newer version. The installer needs Python 3.10+, read access and enough free disk space for an independent copy. The statistics run additionally needs the original Python environment or the dependencies listed in stats_source/requirements.txt.

The statistical installer verifies its original source version and constructs mgt_stat_pipeline_repaired_v1 inside the copied CODE tree. Each execution records fresh results under that package's runs directory. Compare metric rows using detector, source and target keys. Results folders below contain reference exports, not new experiments.

| Question | Reference directory | Paper | Source |
|---|---|---|---|
| RQ1 in-domain | results/RQ1 | Table 4 | Six matched dataset/generator conditions |
| RQ2 generator transfer | results/RQ2 | Table 5, Figure 2 | Twelve conditions with dataset fixed |
| RQ3 dataset transfer | results/RQ3 | Table 6, Figure 3 | Six conditions with generator fixed |
| RQ4 unseen dataset and generator | results/RQ4 | Table 7 | Six source models per detector on COLING Hindi |
| RQ5 prompt robustness | results/RQ5 | Table 8 | Original/new prompt arms, Gemini and Qwen3 |
| RQ6 multilingual systems | results/RQ6 | Table 9 | Separate Grape and Nota AI inference_data/master_results.csv exports |

## Scores and labels

Use the repaired loaders when comparing outputs. Canonical AI is 1 and human is 0. Higher canonical scores indicate stronger AI evidence. Archived label meanings depend on detector, export and condition. In particular, server Text-Fluoroscopy uses raw AI=0 in the audited exports and new-prompt XLM-R uses AI=1. The DetectGPT v3 rebuilt exports already use canonical labels. Preserve raw outputs and apply conversions once. Use continuous scores for AUROC. When only hard labels are available, the performance table may require the detector's inference summary for the historical AUROC.

GPTZero uses mGPT perplexity with a 512-token limit. Classify at or below the frozen source cutoff as AI. DetectGPT uses the mGPT/mT5 perturbation score with higher LLR indicating AI. Preserve its source-specific cutoff, including negative values. Threshold fitting and inference are separate steps. Do not refit on target labels. RQ6 source metric column names identify the evaluators' own class conventions. Check the evaluator and checkpoint mapping before comparing positive-class F1.

## Figures and provenance

Run `python3 release_tools/plot_figure4.py` from the repository root with pandas and matplotlib installed. It writes figure4_local.pdf in the working directory. figure4_inputs.csv.gz contains outcome, five features and the grouping keys for each of 34,412 prediction appearances. Standardization uses the population standard deviation within each of 168 detector/source/target combinations. Bars average predictions within outcome. regenerate_figure4_local.py provides the independent route from the evidence and repaired-run ZIPs (see --help). results/shared contains overlap and regression results supporting Figures 5 and 6. Figure 1 describes the dataset and Figure 7 illustrates selected examples.

## Training, inference and scope

The repository root retains the existing public detector entry points and datasets. server_source mirrors study-specific scripts available from the archived server sources, including the COLING, prompt and multilingual implementations. Use the corresponding original working directory and inspect its input, output and device settings before running. Some scripts use local CUDA devices, external resources or model downloads. Credentials must be supplied privately. For historical experiment replay use the audited server files listed in server_inputs.json. The installer does not launch training or overwrite predictions. Keep fresh checkpoints and inference outputs in a separate experiment directory.

The manifest pins available inputs by content. A seed alone cannot establish that fresh training reproduces the historical checkpoints. Model revisions, software and CUDA versions, original checkpoint selection, stochastic perturbations and score caches matter. The archive supports outcome replay more strongly than fresh training. The full server statistics run and fresh training still need validation on the server. RQ6 is outside the repaired RQ1–5 statistical package. A complete independent rerun of every table and figure from a clean public clone has not been established.

## Changelog

- Added an additive, checksum-verified server installer to preserve historical experiments while isolating the paper inputs.
- Indexed reference results by RQ to map manuscript tables to the relevant evaluation and distinguish prompt arms.
- Included version-checked statistics repair and its expected starting sources to preserve per-export label interpretation.
- Included server implementation snapshots alongside public entry points to keep distinct implementations traceable. Removed credential literals from distribution copies where detected.
- Added compact Figure 4 inputs and a plotting entry point to make local normalization reproducible without model weights.
- Preserved the existing public training entry points because changes to optimizer behavior or label heads would define new experiments requiring fresh validation.
