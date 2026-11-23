# How Well Do Machine Generated Text Detectors Transfer to Hindi? Performance, Transfer, and Failure Analysis

This study asks whether a detector can still tell human-written Hindi from machine-generated Hindi when the dataset, generator, or prompt changes. This repository includes the data, detector implementations, saved predictions, and analysis code used to study those changes.

**To reproduce the paper’s results, start with the three steps below.** You can run them on a CPU. The repository already contains the detector predictions, so this part requires no GPU, model downloads, trained checkpoints, or access to the authors’ server. For training a detector or scoring new text, see [Running detectors on new data](#running-detectors-on-new-data).

## 1. Download the repository

Install Git and Python 3.12. In a terminal, run:

```bash
git clone https://github.com/dhruvdidthat/Hindi-MGT.git
cd Hindi-MGT
```

Run the remaining commands from this folder, which contains this README. If you already have the repository, open a terminal in that folder.

## 2. Install the analysis packages

A virtual environment keeps the packages for this study together. On **Linux or macOS**, run:

```bash
python3.12 -m venv .venv-analysis
source .venv-analysis/bin/activate
```

On **Windows PowerShell**, use:

```powershell
py -3.12 -m venv .venv-analysis
.\.venv-analysis\Scripts\Activate.ps1
```

Then, on either platform, run:

```bash
python -m pip install -r analysis/requirements.txt
```

Use the versions in this file for the paper analysis. For example, SciPy versions can give different Wilcoxon p-values when observations have tied ranks or zero differences.

## 3. Generate the tables and figures

```bash
python analysis/replay.py
```

This command rebuilds the GPTZero and DetectGPT decisions, checks the Text-Fluoroscopy label conversion, and uses the released predictions to calculate the results. It also evaluates Nota AI and Grape, draws the figures, and compares the numerical outputs with the study’s reference tables. The final check prints a filename and the number of rows verified for each successful comparison. A failed check stops the run and names the comparison to inspect.

| What you want to see | Where to look after the run |
|---|---|
| Tables 4–8, with one row per detector and condition | `outputs/tables_4_to_8.csv` |
| Table 9, rounded to four decimal places as in the paper | `outputs/table9_printed.csv` |
| Full-precision results for Nota AI and Grape | `outputs/rq6_metrics.csv` |
| Individual statistical tests | CSVs in `outputs/` with names beginning with `rq1_` through `rq5_` |
| Figures 1–6 | `outputs/figures/figure1.pdf` through `figure6.pdf` |
| Figure 7, the manually annotated examples | `figures/figure7_examples.png`, included with the repository |

For a first example, open `tables_4_to_8.csv` and filter for Table 4, source dataset `hindisumm`, source LLM `gemini`, and detector `DetectGPT`. That row describes a detector calibrated on HindiSumm/Gemini and evaluated on the corresponding test split. Its AUROC rounds to 0.37, its AI recall to 0.97, and its human recall to 0.08. The last two numbers mean that it recognized most AI examples but few human examples in that condition.

## Find an experiment

A **generator** writes the AI text. A **detector** predicts who wrote it. The **source** is the dataset and generator used to train or calibrate a detector. The **target** is the dataset and generator on which it is evaluated.

| Question | What changes at test time? | Prediction folder | Main results |
|---|---|---|---|
| RQ1 · In-domain | Same dataset and generator, evaluated on the test split. Six conditions per detector. | [rq1_3](experiments/rq1_3/) | Table 4, bootstrap intervals, Cochran’s Q, and paired McNemar tests with Holm correction |
| RQ2 · LLM transfer | The generator changes within the same dataset. Twelve conditions per detector. | [rq1_3](experiments/rq1_3/) | Table 5, detector stability, Figure 2 |
| RQ3 · Dataset transfer | The dataset changes while the generator stays the same. Six conditions per detector. | [rq1_3](experiments/rq1_3/) | Table 6 and Figure 3 |
| RQ4 · Unseen dataset and generator | Each of the six source configurations is tested on COLING 2025 Hindi with GPT-4o. | [rq4](experiments/rq4/) | Table 7, paired detector comparisons and Friedman tests |
| RQ5 · Prompt change | Compare the original and new prompts for Gemini and Qwen3. Each arm has twelve configurations per detector. | [Original](experiments/rq1_3/) and [new prompt](experiments/rq5/) | Table 8 and paired Wilcoxon tests |
| RQ6 · Multilingual detectors | Evaluate Nota AI and Grape on seven Hindi targets per system. | [rq6](experiments/rq6/) | Table 9 |

**Why are there only two Python scripts in `experiments/rq1_3/`?** The folder groups three experiments that use the same test splits. Its two Python files are additional scripts for applying GPTZero and DetectGPT across source and target conditions. The implementation for each of the seven detectors lives under [detectors/](detectors/). The shared [predictions.csv.gz](experiments/rq1_3/predictions.csv.gz) contains all seven detectors, with 4,916 prediction rows each across RQ1–RQ3. The analysis reads this file for all three questions. For example, [xlm_roberta_inference.py](detectors/xlmr/xlm_roberta_inference.py) is where to look for XLM-R inference.

## Understand the input data

- **Training and testing splits.** `training_data/` and `testing_data/` each contain six CSVs, one for each combination of HindiSumm or XQuad-Hindi with Gemini, Llama3 or Qwen3. The twelve files were checked against the study splits using SHA-256. In these shared input files, raw label 0 means AI and 1 means human.
- **Perturbed texts and prompts.** `perturbed_training_data/` and `perturbed_testing_data/` hold the perturbed versions used by detectors that require them. `prompt_change/` holds the prompt-change data. RQ5 selects the Gemini and Qwen3 conditions.
- **Saved predictions.** Each `experiments/*/predictions.csv.gz` is a compressed CSV. Python can read it directly with `pandas.read_csv`. A row records a detector’s prediction for one example. In the core experiments, the model, source and target columns identify the run. The text key identifies a normalized text for matching. The same text can appear more than once when several detectors or source configurations evaluate it.
- **COLING.** The replay includes deidentified prediction rows and saved scores for the Hindi test set. Scoring COLING texts afresh requires the original Hindi text and any perturbed version required by the chosen detector.

## How labels and scores become paper results

**AI recall** is the fraction of AI examples recognized as AI. **Human recall** is the fraction of human examples recognized as human. **Accuracy** counts correct predictions across both classes. **F1** combines precision and recall for the designated positive class. **AUROC** measures how well a continuous score ranks examples across the two classes.

For RQ1–RQ5, the analysis uses **AI = 1 and human = 0**, computes F1 with AI as positive, and orients continuous scores so that higher values indicate AI. A detector’s raw output may use a different convention. In particular, a column called “probability of class 1” takes its meaning from the class labels used by that implementation.

| Detector | How to interpret its saved output |
|---|---|
| XLM-R | Raw 0 means AI in the original conditions. In new-prompt exports, raw 1 means AI and the class-1 probability is already an AI score. |
| Indic-BERT | Raw 0 means AI. A saved class decision is sufficient to calculate recall. |
| CONDA | Raw 0 means AI. The class-1 probability, when available, measures human probability. CONDA uses target texts for adaptation. |
| RADAR | Raw 0 means AI. Where the export includes human probability, subtract it from 1 to obtain the AI score. |
| Text-Fluoroscopy | In the study exports, raw 0 means AI. Convert both the reference label and the predicted label. For AUROC use 1 minus the stored class-1 probability. |
| GPTZero | Uses mGPT perplexity. Lower perplexity gives stronger AI evidence. |
| DetectGPT | Uses mGPT/mT5 perturbation scores. Higher LLR gives stronger AI evidence. |

**Where AUROC comes from.** Some prediction files contain only the final class decision. For 78 AUROC entries, the analysis uses the recorded AUROC from the corresponding inference summary in `data/summary_auc.csv`. The `auc_source` column in the output identifies these summaries. The other 804 values across Tables 4–8 are calculated from saved predictions and scores. Files in `analysis/reference/` are used for the final numerical comparison.

### DetectGPT

The base scoring script includes a fixed cutoff of zero and an evaluation convention in which class 1 means AI. The saved source-training scores used in the paper label AI as 0. Both details matter. Using the opposite positive class changes which cutoff is selected. Using zero instead of the source cutoff can change a decision even when the text receives exactly the same score.

To reproduce the paper decisions, run `python analysis/rebuild_detectgpt.py`. For each of the six source combinations the script:

1. Reads the saved LLR scores from `data/detectgpt/train_*.csv.gz` and identifies raw label 0 as AI.
2. Chooses the first cutoff that maximizes the true-positive rate minus the false-positive rate on that source training split (Youden’s J).
3. Predicts AI when the test LLR is **at or above** the chosen cutoff. The same source cutoff applies to its transfer, COLING and new-prompt conditions. An LLR and its cutoff can both be negative.
4. Checks the reference class, score and decision against the released rows. Each scored example has its corresponding reference label. For RQ5, normalized text and occurrence number were used to match repeated texts correctly.

This procedure reuses the recorded perturbation scores. It reproduces the decisions without sampling new perturbations or generating new text.

### GPTZero

Inputs are truncated to **512 tokens**. GPTZero predicts AI when perplexity is **at or below** the source threshold. The analysis uses negative perplexity as the ranking score so that larger values indicate AI.

Run `python analysis/rebuild_gptzero.py` to combine each saved test perplexity with its source cutoff in `data/source_thresholds.csv`. Each cutoff was chosen with Youden’s J on the source training split. Target labels are used only to evaluate the predictions. To fit a threshold again you need the source-training perplexities. For a fresh fit, the detector script computes them from the training texts.

### Text-Fluoroscopy and other raw CSVs

Run `python analysis/check_text_fluoroscopy.py` to verify the label and probability conversions across the study conditions. Suppose a **new** Text-Fluoroscopy export uses the same convention, with AI stored as 0 and class-1 probability stored in `predicted_prob_label1`. The following command creates a separate analysis-ready CSV:

```bash
python analysis/normalize_export.py predictions.csv canonical.csv --truth true_label --truth-ai 0 --prediction predicted_label --prediction-ai 0 --score predicted_prob_label1 --kind p-human
```

The input must already pair each prediction with its correct reference label. The converter requires a new output filename. For another export, check the meaning of its columns first. The truth and prediction AI indices can be specified separately. Use `--kind p-ai` when the score is already an AI probability. For new GPTZero or DetectGPT scores, use `--score perplexity --kind perplexity` or `--score llr --kind llr`, together with `--threshold` set to that source’s cutoff. The core prediction tables under `experiments/` already use the analysis convention.

### Nota AI and Grape

Run `python experiments/rq6/evaluate.py` to generate Table 9. Both external evaluators compute binary F1 for index 1. **F1 uses human text as the positive class for Nota AI and AI text for Grape.** Grape inverts the input reference labels after writing predictions and before calculating metrics. Nota AI compares its reference labels directly with its predictions. Its AUROC uses the exported score with raw class 1 as positive. Table 9 follows each system’s evaluator.

## Statistical tests and linguistic features

To run the RQ1–RQ5 statistical analyses separately, use `python analysis/reproduce.py`. The methods are implemented in `analysis/pipeline.py`, `analysis/pipeline_rq45.py` and `analysis/stats_tests.py`. In RQ5, the pairing unit is a detector evaluated under the same source and target configuration with each prompt. The two sets of texts can differ in size, and some new-prompt texts overlap the training data. The comparison describes those evaluated sets.

The RQ1–RQ3 prediction file includes the linguistic features calculated for the texts. `analysis/linguistic_features.py` defines sentence and word segmentation, syllable and conjunct counts, and length normalization for PSW30 and JUK. `analysis/matching.py` defines how to match normalized texts and handle duplicate or conflicting keys in paired comparisons.

After the statistics run, `python figures/render.py` renders Figures 1–6:

| Figure | How to read it |
|---|---|
| 1 | Word-count distributions. Words are split on whitespace. For each dataset the human examples come from the Gemini training split, and the generated examples come from each generator. The renderer uses the verified split labels. The study notebook used the opposite raw-label assignment for this plot, which matters when comparing the legend with the manuscript. |
| 2 and 3 | Recall under generator transfer and dataset transfer, respectively. |
| 4 | Each feature is standardized within a single detector/source/target run, using that test set’s mean and standard deviation. The bars average those standardized values by classification outcome. Zero represents the local test-set mean. |
| 5 | Average Jaccard overlap between two detectors’ errors across in-domain conditions. |
| 6 | Regression coefficients with nominal 95% intervals based on standard errors clustered by text. |

The PDFs use the released plotting layout. Figure 7 is an illustration selected and annotated by the authors.

## Running detectors on new data

Training and inference use a separate set of dependencies. Create a separate Python environment and install `python -m pip install -r requirements.txt`. Run the detector scripts from the **repository root**, since they use relative paths for datasets and outputs. Some scripts specify a CUDA device index. Check the device setting, model access, available GPU memory and output directories before starting a new run.

| Detector folder | Training or calibration | Inference |
|---|---|---|
| [XLM-R](detectors/xlmr/) | `xlm_roberta_train.py` | `xlm_roberta_inference.py` |
| [Indic-BERT](detectors/indicbert/) | `IndicBert_train.py` | `IndicBert_inference.py` |
| [CONDA](detectors/conda/) | `conda.py` | Included in the same script |
| [RADAR](detectors/radar/) | `radar_train.py` | `radar_inference.py` |
| [Text-Fluoroscopy](detectors/text_fluoroscopy/) | `text_fluoroscopy_train.py` | `text_fluoroscopy_inference.py` |
| [GPTZero](detectors/gptzero/) | Source calibration in `gpt_zero.py` | `gpt_zero.py` |
| [DetectGPT](detectors/detectgpt/) | Source calibration for the paper in `analysis/rebuild_detectgpt.py` | `detect_gpt.py` computes perturbation scores |

For example, to train XLM-R and then run its inference script:

```bash
python detectors/xlmr/xlm_roberta_train.py
python detectors/xlmr/xlm_roberta_inference.py
```

For calibrated transfer scoring, the two dedicated scripts are in `experiments/rq1_3/`. The RQ4 and RQ5 folders contain the available scripts specific to COLING and prompt-change inference. Each script’s configuration identifies the input files, checkpoints and output paths it expects. Keep outputs from new runs in a separate results directory.

For fresh RQ6 inference, use `experiments/rq6/nota_ai/inference_on_csv.py` or `experiments/rq6/grape/ensemble_inference.py`. Run the chosen script from a working directory with the expected `testing_data/` tree and model files. Nota AI requires `MGTDetectionModel.pt` and its feature models. Grape requires the two LoRA adapters and the ensemble checkpoint named in its script. These weights are obtained separately.

The CPU workflow verifies evaluation from the released scores and predictions. Fresh GPU training and perturbation sampling also depend on model checkpoint versions, software, and GPU settings. Exact numerical agreement after retraining every detector has not been independently verified.