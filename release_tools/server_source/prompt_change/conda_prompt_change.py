#!/usr/bin/env python3
"""
ConDA inference script -- evaluates each trained model (CONDA/trained_models/{train_dataset}_{train_llm}/conda_model.pt)
against ALL 6 (dataset, llm) combinations of the prompt-shifted ("new_prompt") test data:

    hindisumm x {gemini, llama3, qwen3}
    xquad     x {gemini, llama3, qwen3}

i.e. 6 trained models x 6 test sets = 36 result rows.

Test data is read preferring:
    perturbed_new_prompt_data/{dataset}_{llm}_perturbed_new_prompt.csv
and falling back to:
    new_prompt_data/{dataset}_{llm}_new_prompt.csv
if the perturbed version isn't present (only 'text'/'label' columns are used either way).

Output layout:

    CONDA/
      inference_results_new_prompt/
        {train_dataset}_{train_llm}/
          tested_on_{test_dataset}_{test_llm}/
            predictions.csv
      conda_all_results_new_prompt.csv
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple
from itertools import product

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

SEED = 42
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Config -- adjust if your paths differ.
# ------------------------------------------------------------------
SAVE_MODEL_ROOT = "../training_models/CONDA/trained_models"                        # {train_dataset}_{train_llm}/conda_model.pt
PERTURBED_NEW_PROMPT_DIR = "perturbed_new_prompt_data"           # {dataset}_{llm}_perturbed_new_prompt.csv
RAW_NEW_PROMPT_DIR = "new_prompt_data"                           # {dataset}_{llm}_new_prompt.csv (fallback)
RESULTS_ROOT = os.path.join("CONDA", "inference_results_new_prompt")
CONSOLIDATED_RESULTS_CSV = os.path.join("CONDA", "conda_all_results_new_prompt.csv")
MAX_SEQ_LEN_DEFAULT = 256
EVAL_BATCH_SIZE = 16

DATASETS = ["hindisumm", "xquad"]
LLMS = ["gemini", "llama3", "qwen3"]


# ------------------------------------------------------------------
# Preprocessing / dataset classes (standalone copy, same as training script).
# ------------------------------------------------------------------
class PreProcess:
    def __init__(self, lowercase_norm=False, special_chars_norm=True, accented_norm=True, stopword_norm=False):
        self.lowercase_norm = lowercase_norm
        self.special_chars_norm = special_chars_norm
        self.accented_norm = accented_norm
        self.stopword_norm = stopword_norm
        self.normalizer = IndicNormalizerFactory().get_normalizer("hi")
        self.stopwords = set([
            'है', 'हैं', 'था', 'थे', 'थी', 'हो', 'होगा', 'होता', 'होती', 'के', 'का', 'की', 'को', 'में', 'से', 'पर', 'और', 'या', 'लेकिन'
        ])
    def normalize_hindi(self, text):
        return self.normalizer.normalize(text)
    def tokenize_hindi(self, text):
        return indic_tokenize.trivial_tokenize(text, lang='hi')
    def special_char_remove(self, text):
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    def accented_word_normalization(self, text):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    def stopword_remove(self, text):
        tokens = self.tokenize_hindi(text)
        return ' '.join(word for word in tokens if word not in self.stopwords)
    def fit(self, text):
        return str(text)


class EncodeEvalData(Dataset):
    def __init__(self, input_texts, tokenizer, max_sequence_length):
        self.input_texts = input_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
    def __len__(self):
        return len(self.input_texts)
    def __getitem__(self, index):
        text = self.input_texts[index]
        preprocessor = PreProcess(special_chars_norm=True, accented_norm=True, stopword_norm=True)
        text = preprocessor.fit(text)
        padded_sequences = self.tokenizer(text, padding='max_length', max_length=self.max_sequence_length, truncation=True)
        return (torch.tensor(padded_sequences['input_ids'], dtype=torch.long),
                torch.tensor(padded_sequences['attention_mask'], dtype=torch.long))


@dataclass
class SequenceClassifierOutputWithLastLayer(SequenceClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class XLMRobertaForContrastiveClassification(XLMRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.soft_max = nn.Softmax(dim=1)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        softmax_logits = self.soft_max(logits)
        if not return_dict:
            output = (softmax_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutputWithLastLayer(
            loss=loss,
            logits=softmax_logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def find_new_prompt_test_file(test_dataset, test_llm):
    """
    Prefer the perturbed_new_prompt_data file (schema-compatible with the standard
    ..._test_{llm}_perturb.csv files: has 'text' and 'label'); fall back to the raw
    new_prompt_data file if the perturbed one isn't present.
    """
    perturbed_path = os.path.join(PERTURBED_NEW_PROMPT_DIR, f"{test_dataset}_{test_llm}_perturbed_new_prompt.csv")
    if os.path.exists(perturbed_path):
        return perturbed_path
    raw_path = os.path.join(RAW_NEW_PROMPT_DIR, f"{test_dataset}_{test_llm}_new_prompt.csv")
    if os.path.exists(raw_path):
        return raw_path
    return None


def load_conda_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})

    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    xlm_model = XLMRobertaForContrastiveClassification.from_pretrained('xlm-roberta-base').to(device)
    xlm_model.load_state_dict(checkpoint["model_state_dict"])
    xlm_model.eval()

    return xlm_model, tokenizer, args


def run_inference(xlm_model, tokenizer, test_csv, max_seq_len, desc=""):
    df_test = pd.read_csv(test_csv).dropna(subset=['text'])
    pre = PreProcess(special_chars_norm=True, accented_norm=True, stopword_norm=True)
    df_test['text'] = df_test['text'].apply(pre.fit)

    texts = df_test['text'].tolist()
    eval_dataset = EncodeEvalData(texts, tokenizer, max_seq_len)
    eval_loader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    all_probs, all_preds = [], []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(eval_loader, desc=desc):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = xlm_model(input_ids, attention_mask=attention_mask)
            probs = outputs["logits"].cpu().numpy()
            preds = np.argmax(probs, axis=1).tolist()
            all_probs.extend(probs[:, 1].tolist())
            all_preds.extend(preds)

    y_true = df_test['label'].tolist() if 'label' in df_test.columns else [None] * len(df_test)

    if all(v is not None for v in y_true):
        acc = accuracy_score(y_true, all_preds)
        f1 = f1_score(y_true, all_preds)
        try:
            auroc = roc_auc_score(y_true, all_probs)
        except Exception:
            auroc = float('nan')
        ai_recall = recall_score(y_true, all_preds, pos_label=0)
        human_recall = recall_score(y_true, all_preds, pos_label=1)
    else:
        acc = f1 = auroc = ai_recall = human_recall = float('nan')

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "auroc": auroc,
        "AI_recall_label_0": ai_recall,
        "Human_recall_label_1": human_recall,
    }
    return metrics, df_test, all_preds, all_probs, y_true


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(CONSOLIDATED_RESULTS_CSV), exist_ok=True)

    header = ["train_dataset", "train_llm", "test_dataset", "test_llm",
              "accuracy", "f1_score", "auroc", "AI_recall_label_0", "Human_recall_label_1"]
    if not os.path.exists(CONSOLIDATED_RESULTS_CSV):
        pd.DataFrame(columns=header).to_csv(CONSOLIDATED_RESULTS_CSV, index=False)

    if not os.path.isdir(SAVE_MODEL_ROOT):
        raise FileNotFoundError(f"Trained model directory not found: {SAVE_MODEL_ROOT}")

    model_dirs = sorted([
        d for d in os.listdir(SAVE_MODEL_ROOT)
        if os.path.isdir(os.path.join(SAVE_MODEL_ROOT, d))
    ])
    if not model_dirs:
        raise FileNotFoundError(f"No trained model subdirectories found under {SAVE_MODEL_ROOT}")

    print(f"Found {len(model_dirs)} trained model checkpoint(s): {model_dirs}")

    all_test_combos = list(product(DATASETS, LLMS))  # all 6 (dataset, llm) pairs

    for model_dir_name in model_dirs:
        checkpoint_path = os.path.join(SAVE_MODEL_ROOT, model_dir_name, "conda_model.pt")
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Skipping {model_dir_name}: no conda_model.pt found inside.")
            continue

        # Folder name is treated as the authoritative train_dataset/train_llm label.
        parts = model_dir_name.split("_", 1)
        if len(parts) != 2:
            print(f"❌ Could not parse train_dataset/train_llm from folder name '{model_dir_name}'; skipping.")
            continue
        train_dataset, train_llm = parts
        if train_dataset not in DATASETS or train_llm not in LLMS:
            print(f"⚠️  Folder name '{model_dir_name}' does not match known dataset/LLM names; skipping.")
            continue

        try:
            xlm_model, tokenizer, args = load_conda_model(checkpoint_path)
        except Exception as e:
            print(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
            continue

        max_seq_len = args.get("max_sequence_length", MAX_SEQ_LEN_DEFAULT)

        for test_dataset, test_llm in all_test_combos:
            test_csv = find_new_prompt_test_file(test_dataset, test_llm)
            if test_csv is None:
                print(f"⚠️  No new-prompt test file found for ({test_dataset}, {test_llm}) "
                      f"[checked {PERTURBED_NEW_PROMPT_DIR} and {RAW_NEW_PROMPT_DIR}]; skipping.")
                continue

            desc = f"{model_dir_name} -> tested_on_{test_dataset}_{test_llm} (new_prompt)"
            print(f"\n🔎 {desc}\n   test file: {test_csv}")

            try:
                metrics, df_test, all_preds, all_probs, y_true = run_inference(
                    xlm_model, tokenizer, test_csv, max_seq_len, desc=desc
                )
            except Exception as e:
                print(f"❌ Inference failed for {desc}: {e}")
                continue

            print(f"   Acc={metrics['accuracy']:.4f}  F1={metrics['f1_score']:.4f}  "
                  f"AUROC={metrics['auroc']:.4f}  AI_R={metrics['AI_recall_label_0']:.4f}  "
                  f"Human_R={metrics['Human_recall_label_1']:.4f}")

            save_dir = os.path.join(RESULTS_ROOT, model_dir_name, f"tested_on_{test_dataset}_{test_llm}")
            os.makedirs(save_dir, exist_ok=True)
            pd.DataFrame({
                "text": df_test['text'].tolist(),
                "true_label": y_true,
                "predicted_label": all_preds,
                "prediction_probability_class1": all_probs
            }).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

            row = {
                "train_dataset": train_dataset,
                "train_llm": train_llm,
                "test_dataset": test_dataset,
                "test_llm": test_llm,
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
                "auroc": metrics["auroc"],
                "AI_recall_label_0": metrics["AI_recall_label_0"],
                "Human_recall_label_1": metrics["Human_recall_label_1"],
            }
            pd.DataFrame([row], columns=header).to_csv(CONSOLIDATED_RESULTS_CSV, mode='a', header=False, index=False)

        # Free GPU memory before moving to the next trained model.
        del xlm_model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n✅ Done. Consolidated results saved at: {CONSOLIDATED_RESULTS_CSV}")


if __name__ == "__main__":
    main()