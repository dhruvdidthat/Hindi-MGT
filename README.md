# How Well Do Machine Generated Text Detectors Transfer to Hindi?
## Performance, Transfer, and Failure Analysis

This repository accompanies the Hindi MGT study. It includes the HindiSumm and XQuad-Hindi training and test splits, the original detector implementations, the example-level evaluation outputs for all six research questions, and the code for the performance, statistical, and linguistic analyses.

## Reproduce the results on CPU

Run these commands from a clean clone of this revision using Python 3.12. A GPU and model checkpoints are required only for new model runs.

```bash
python3.12 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/replay.py
```

The command rebuilds the score-based decisions, checks the archived Text-Fluoroscopy conversions, runs the statistical analyses, generates tables and renders figures. Outputs are written under `outputs/`. Each verified comparison prints its filename and number of rows. The reference tables under `analysis/reference/` are used only in the final comparison. The dependency pins specify the tested replay environment. In particular, SciPy's default Wilcoxon calculation differs between versions in the presence of tied ranks or zeros.

| Study question | Distributed example-level inputs | Output |
|---|---|---|
| RQ1. In-domain detection | experiments/rq1_3 | Six matched source/target settings, Table 4, Cochran’s Q, paired McNemar comparisons with Holm correction, bootstrap intervals |
| RQ2. Transfer to a different LLM | experiments/rq1_3 | Twelve within-dataset generator transfers, Table 5, detector stability, Figure 2 |
| RQ3. Transfer to a different dataset | experiments/rq1_3 | Six transfers with generator fixed, Table 6 and Figure 3 |
| RQ4. Unseen dataset and generator | experiments/rq4 | Six source configurations evaluated on COLING 2025 Hindi/GPT-4o, Table 7, paired comparisons and Friedman tests |
| RQ5. Change of generation prompt | experiments/rq5 (new) and experiments/rq1_3 (original) | Twelve configurations per detector per arm, limited to Gemini and Qwen3, Table 8 and Wilcoxon comparisons |
| RQ6. External multilingual systems | experiments/rq6 | Nota AI and Grape, seven targets per system, Table 9 |

`outputs/tables_4_to_8.csv` identifies detector, source dataset and generator, target dataset and generator, AI recall, human recall and AUROC. When an archived prediction file lacks continuous scores, the 78 corresponding AUROCs come from the original inference summaries in `data/summary_auc.csv`. The output identifies the source summary filename. The remaining 804 numbers are computed from predictions. `outputs/table9_printed.csv` uses four decimal places as in the paper. Full precision is in `outputs/rq6_metrics.csv`.

## Labels, scores and corrections

We use **AI = 1 and human = 0** in the core analysis and orient continuous scores so that higher values indicate AI. AI recall is the fraction of AI examples detected. Human recall is the fraction of human examples recognized as human. The core analysis computes F1 with AI as the positive class. A model output labelled “class 1” refers to the training label index. Its meaning depends on the data and export convention.

| Detector | Interpretation of historical exports |
|---|---|
| XLM-R | Raw 0 denotes AI in the original conditions. New-prompt exports use class 1 for AI and supply its probability. |
| Indic-BERT | Raw 0 denotes AI. Hard predictions suffice for recall. AUROC needs ranking scores or an archived inference summary. |
| CONDA | Raw 0 denotes AI. Where available the class-1 score measures human probability. Target adaptation uses target texts. |
| RADAR | Raw 0 denotes AI. Where a human probability was recorded, its complement gives the AI score. |
| Text-Fluoroscopy | The published implementation names the probability column by class index. For the archived Text-Fluoroscopy outputs, raw class 0 denotes AI, so we reverse the stored class-1 probability for AUROC. We also map both stored labels to their semantic classes before reporting recall. |
| GPTZero | mGPT perplexity only. Inputs are truncated to 512 tokens. We predict AI when perplexity is at or below the source threshold. The ranking score is negative perplexity. |
| DetectGPT | mGPT/mT5 perturbation scoring. Higher LLR indicates stronger AI evidence. The calibrated paper prediction uses LLR at or above the frozen source cutoff. |

**DetectGPT.** The original implementation includes a fixed-zero decision rule and assumes class 1 denotes AI in its evaluation. The audited source-training score caches encode AI as 0. Fitting a threshold with the opposite positive class changes the ROC objective, and replacing a source-specific cutoff by zero can change the decision even when the LLR is identical. The correction reuses the measured LLRs. It creates new prediction tables from the existing examples. The corpus and perturbation draws are preserved.

Run `python analysis/rebuild_detectgpt.py`. For each of the six source pairs it reads `data/detectgpt/train_*.csv.gz`, converts raw label 0 to AI, computes the ROC for LLR and selects the first maximum of TPR minus FPR. It then applies that source cutoff to every associated target, including COLING and the changed prompts. The published score projections include the reference label aligned to each scored example. RQ5 scores were aligned to the new-prompt reference by normalized text and occurrence number. Each replay checks the score, reference class and decision against the released evaluation rows. Negative cutoffs are valid for this signed statistic.

**GPTZero.** Run `python analysis/rebuild_gptzero.py`. It joins saved perplexities to the source dataset and generator in `data/source_thresholds.csv`. The source training split supplies the Youden-J cutoff. The target labels are used for evaluation only. This route reproduces the frozen decisions and scores. Independently refitting those original thresholds requires the source-training perplexities. For a fresh fit, the detector's inference script computes them from the training texts.

**Text-Fluoroscopy and fresh exports.** Run `python analysis/check_text_fluoroscopy.py` to check all archived conditions. To convert a new CSV with the *same* raw convention, run the following command. Inputs must already contain aligned reference labels and model outputs. The converter creates a new file and rejects a pre-existing output.

```bash
python analysis/normalize_export.py predictions.csv canonical.csv \
  --truth true_label --truth-ai 0 --prediction predicted_label --prediction-ai 0 \
  --score predicted_prob_label1 --kind p-human
```

For an export where class 1 genuinely denotes AI, use the corresponding AI index and `--kind p-ai`. Supply the truth and prediction indices separately if the input split and output head use different encodings. Use the raw-label and probability columns from that particular version. Already canonical tables in `experiments/` can be used directly. For fresh GPTZero or DetectGPT score files, use `--score perplexity --kind perplexity` or `--score llr --kind llr` with `--threshold` set to the applicable source cutoff.

**Nota AI and Grape.** Run `python experiments/rq6/evaluate.py`. F1 uses human text as the positive class for Nota AI and AI text for Grape. Both evaluators compute binary F1 on index 1. Grape inverts the input reference labels after writing predictions and before computing metrics. The replay follows this sequence. Nota AI's evaluator directly compares its reference labels to the exported predictions. Its historical AUROC passes the exported score to ROC with raw class 1 as positive. Table 9 preserves that evaluator convention. This distinction matters when interpreting AUROC alongside the core canonical results.

## Data and prediction files

- `training_data/` and `testing_data/` contain six files per split, organized by dataset and generator (Gemini, Llama3 and Qwen3). The twelve original files matched the audited splits by SHA-256. The shared reference convention is raw 0 for AI.
- `perturbed_training_data/`, `perturbed_testing_data/` and `prompt_change/` retain the original detector inputs. The paper's RQ5 analysis selects the Gemini and Qwen3 conditions from the available files.
- `experiments/*/predictions.csv.gz` contains one row per detector prediction. Core keys identify detector and train/test dataset–generator pairs. The text key identifies a normalized text for pairing. Repeated predictions across runs remain separate observations. Scores can be missing when the detector recorded only a class decision.
- RQ1–RQ3 include precomputed Hindi linguistic features. `analysis/linguistic_features.py` defines sentence and word segmentation, syllable and conjunct counts, and the length normalization used for PSW30 and JUK. `analysis/matching.py` defines normalized text matching and handles duplicate and conflicting keys for paired comparisons.
- The standalone replay includes deidentified COLING prediction rows and cached scores. New COLING inference requires the original Hindi test text and any detector-specific perturbed variant.

## Statistical and linguistic analysis

`analysis/reproduce.py` runs the RQ1–RQ5 tests independently of figure rendering. The implementation is in `pipeline.py`, `pipeline_rq45.py` and `stats_tests.py`. Original and new prompt results are paired by detector, source and target configuration. RQ5 uses the archived cohorts. Their sizes can differ and some new-prompt texts overlap training. The comparison describes those cohorts.

`python figures/render.py` writes vector PDFs for Figures 1–6 after the analysis run. Figure 1 uses whitespace word counts, human examples from the Gemini training split for each dataset, and AI examples from each generator. It uses the audited shared split labels. The historical plotting notebook assigned the two raw indices oppositely, so compare the Figure 1 legend with the manuscript before claiming an exact visual reproduction. Figures 2 and 3 summarize transfer recall. Figure 4 standardizes each feature within each detector/source/target run and averages standardized observations by outcome. Figure 5 averages the paired error Jaccard values across in-domain conditions. Figure 6 uses regression coefficients and nominal 95% intervals from standard errors clustered by text. Figure 7 is a manually annotated qualitative illustration. Its original image is included in `figures/figure7_examples.png`. The plotting scripts reproduce the analyses with a fresh layout.

## Running detectors on new data

The seven original implementations under `detectors/` retain the code and hyperparameters of revision `2a642794`. Run their entry points from the repository root so relative dataset paths resolve. Use a separate environment with `python -m pip install -r requirements.txt`. Some entries specify a CUDA device index. Model downloads, memory requirements and any restricted model access depend on the detector.

| Detector directory | Training | Inference |
|---|---|---|
| detectors/xlmr | xlm_roberta_train.py | xlm_roberta_inference.py |
| detectors/indicbert | IndicBert_train.py | IndicBert_inference.py |
| detectors/conda | conda.py | Integrated in conda.py |
| detectors/radar | radar_train.py | radar_inference.py |
| detectors/text_fluoroscopy | text_fluoroscopy_train.py | text_fluoroscopy_inference.py |
| detectors/gptzero | Source calibration in gpt_zero.py | gpt_zero.py |
| detectors/detectgpt | Source calibration for the paper in analysis/rebuild_detectgpt.py | detect_gpt.py computes the original perturbation scores |

For example, run `python detectors/xlmr/xlm_roberta_train.py` followed by `python detectors/xlmr/xlm_roberta_inference.py`. For calibrated transfer scoring, see `experiments/rq1_3/detect_gpt_cross_domain.py` and `gpt_zero_cross_domain.py`. The RQ4 and RQ5 folders include the available dedicated inference implementations. Run these from the repository root with the input files and checkpoints specified in their configuration sections. Fresh model runs may write into their configured results roots. Keep fresh results separate from the distributed evaluation files.

The RQ6 implementations are under `experiments/rq6/nota_ai/` and `experiments/rq6/grape/`. For fresh external inference, run the relevant `inference_on_csv.py` or `ensemble_inference.py` in a working directory containing its `testing_data/` tree and required model files. Nota AI expects MGTDetectionModel.pt and its feature models. Grape requires the two LoRA adapters and the ensemble checkpoint specified in its script. Model weights are excluded from this distribution.

Exact replay here refers to evaluation from frozen scores and predictions. Fresh training and perturbation sampling also depend on checkpoint revisions, GPU kernels and software versions. The release verification exercises the CPU replay. A full independent GPU retraining of every detector is outstanding.
