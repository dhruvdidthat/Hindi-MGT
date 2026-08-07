# Understanding the Challenges in Detecting Machine-Generated Hindi Text

This repository contains the implementation used in the paper "Understanding the Challenges in Detecting Machine-Generated Hindi Text".

## Repository Contents

| Script | Description |
|--------|-------------|
| `IndicBert_train.py` | Fine-tunes IndicBERT. |
| `IndicBert_inference.py` | Evaluates trained IndicBERT models on all train-test dataset/LLM combinations. |
| `radar_train.py` | Hindi adaptation of RADAR using XLM-RoBERTa and mT5 paraphrasing. |
| `radar_inference.py` | Evaluates trained RADAR models. |
| `text_fluroscopy_train.py` | Trains the Text Fluoroscopy detector using hidden-state KL features. |
| `text_fluroscopy_inference.py` | Runs inference for Text Fluoroscopy. |
| `conda.py` | Training and inference script for the ConDA domain adaptation method. |
| `gpt_zero.py` | GPTZero-style perplexity detector using mGPT with threshold learning (inferences on all the combinations). |
| `detect_gpt.py` | DetectGPT implementation using mGPT and mT5 perturbations (inferences on all the combinations). |

## Dataset Layout

```
training_data/
    hindisumm/
        hindisumm_train_gemini.csv
        hindisumm_train_llama3.csv
        hindisumm_train_qwen3.csv
    xquad/
        ...

testing_data/
    hindisumm/
    xquad/
```

Each CSV should contain:

- `text` : Hindi document
- `label`: `0 = AI`, `1 = Human`

## Environment

- Python 3.10+
- CUDA-enabled GPU recommended
- PyTorch + Hugging Face Transformers

## Installation

```bash
conda create -n hindi-mgt python=3.10
conda activate hindi-mgt
pip install -r requirements.txt
```

## Running Experiments

Train:
```bash
python IndicBert_train.py
python radar_train.py
python text_fluroscopy_train.py
python conda.py
```

Inference:
```bash
python IndicBert_inference.py
python radar_inference.py
python text_fluroscopy_inference.py
python gpt_zero.py
python detect_gpt.py
```

Outputs are written into each method's `trained_models/` and `inference_results/`
directories.

## Notes

- Random seed is fixed to **42** wherever applicable.
- Some scripts are configured to use CUDA device `cuda:1` by default. If your machine uses a different GPU index, update the device assignment in the script before running it.
- The code evaluates all combinations of:
  - Training datasets: HindiSumm, XQuAD-Hindi
  - Generator LLMs: Gemini, Llama 3, Qwen3
- Labels follow:
  - `0 = Machine-generated`
  - `1 = Human-written`
- Model checkpoints are stored separately for each dataset/LLM pair.
- Some models are downloaded automatically from Hugging Face, so internet access is required the first time you run the scripts.
- The repository is research-oriented and may require GPU memory depending on the model size.
