"""
GPTZero inference script (modified to train all source->target combinations).
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, recall_score, roc_curve
)

# ---------------------- CONFIG ----------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

DATASETS = ["hindisumm", "xquad"]
LLMS = ["gemini", "llama3", "qwen3"]

TRAIN_ROOT = "training_data"
TEST_ROOT = "testing_data"
RESULTS_ROOT = "gptzero/inference_results"
MASTER_CSV = os.path.join(RESULTS_ROOT, "gptzero_all_results.csv")

os.makedirs(RESULTS_ROOT, exist_ok=True)

MODEL_NAME = "ai-forever/mGPT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ---------------------- PERPLEXITY ----------------------
@torch.no_grad()
def calculate_perplexity(text):
    try:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        outputs = model(**enc, labels=enc["input_ids"])
        loss = outputs.loss
        return torch.exp(loss).item() if loss is not None else np.inf
    except Exception:
        return np.inf

# ---------------------- THRESHOLD LEARNING ----------------------
def learn_threshold_from_training(df_train):
    """
    Learns optimal perplexity threshold using ROC (Youden's J statistic).
    Filters invalid (inf / nan) perplexities.

    Labels: 0 = AI, 1 = Human. Low perplexity -> AI-like, so we score with
    -perplexity and ask roc_curve to treat label 0 (AI) as the positive
    class. This part of the original code was already self-consistent.
    """

    # Keep only valid perplexity values
    df = df_train.replace([np.inf, -np.inf], np.nan).dropna(subset=["perplexity"])

    y_true = df["label"].values
    perplexities = df["perplexity"].values

    # Convert to AI-likelihood score (lower ppl = more AI-like)
    scores = -perplexities

    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=0)
    J = tpr - fpr
    best_idx = np.argmax(J)

    best_threshold = -thresholds[best_idx]
    return best_threshold


# ---------------------- CLASSIFIER ----------------------
def classify_by_threshold(ppl, threshold):
    # 0 = AI, 1 = Human
    return 0 if ppl <= threshold else 1

# ---------------------- METRICS ----------------------
def compute_metrics(y_true, y_pred, probs):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, probs)
    except:
        auroc = float("nan")
    ai_recall = recall_score(y_true, y_pred, pos_label=0)
    human_recall = recall_score(y_true, y_pred, pos_label=1)
    return acc, f1, auroc, ai_recall, human_recall

def append_to_master(row):
    df = pd.DataFrame([row])
    if os.path.exists(MASTER_CSV):
        df.to_csv(MASTER_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(MASTER_CSV, index=False)

# ---------------------- INFERENCE ----------------------
def run_gptzero_inference(dataset_name, llm_name):

    print(f"\n🚀 GPTZero | {dataset_name.upper()} + {llm_name.upper()}")

    # ---------- TRAINING ----------
    train_csv = f"{TRAIN_ROOT}/{dataset_name}/{dataset_name}_train_{llm_name}.csv"
    df_train = pd.read_csv(train_csv)

    train_ppl = []
    for text in tqdm(df_train["text"], desc="Training Perplexity"):
        train_ppl.append(calculate_perplexity(text))

    df_train["perplexity"] = train_ppl

    # Learn threshold
    threshold = learn_threshold_from_training(df_train)
    print(f"🔎 Learned threshold: {threshold:.2f}")

    # ---------- TESTING ----------
    test_csv = f"{TEST_ROOT}/{dataset_name}/{dataset_name}_test_{llm_name}.csv"
    df_test = pd.read_csv(test_csv)

    test_ppl = []
    preds = []

    for text in tqdm(df_test["text"], desc="Testing Perplexity"):
        ppl = calculate_perplexity(text)
        test_ppl.append(ppl)
        preds.append(classify_by_threshold(ppl, threshold))

    df_test["perplexity"] = test_ppl
    df_test["predicted_label"] = preds

    y_true = df_test["label"].values
    y_pred = preds

    test_ppl_arr = np.array(test_ppl, dtype=float)
    finite_ppl = test_ppl_arr[np.isfinite(test_ppl_arr)]
    fallback_max = finite_ppl.max() * 10 if finite_ppl.size > 0 else 1e6
    probs = np.where(np.isfinite(test_ppl_arr), test_ppl_arr, fallback_max)

    acc, f1, auroc, ai_recall, human_recall = compute_metrics(
        y_true, y_pred, probs
    )

    # ---------------------- SAVE ----------------------
    save_dir = f"{RESULTS_ROOT}/{dataset_name}_{llm_name}"
    os.makedirs(save_dir, exist_ok=True)

    df_test[["text", "label", "predicted_label", "perplexity"]].to_csv(
        f"{save_dir}/predictions.csv", index=False
    )

    result = {
        "train_dataset": dataset_name,
        "train_llm": llm_name,
        "test_dataset": dataset_name,
        "test_llm": llm_name,
        "threshold": threshold,
        "accuracy": acc,
        "f1_score": f1,
        "auroc": auroc,
        "AI_recall_label_0": ai_recall,
        "Human_recall_label_1": human_recall
    }

    append_to_master(result)

    print(f"✅ Done | Acc={acc:.3f}, AUC={auroc:.3f}, AI Recall={ai_recall:.3f}")

# ---------------------- CROSS-DOMAIN INFERENCE ----------------------
def run_gptzero_cross_inference(train_dataset, train_llm, test_dataset, test_llm):
    """
    Calibrates the threshold on (train_dataset, train_llm) TRAIN data and
    evaluates on (test_dataset, test_llm) TEST data.
      - RQ2 (LLM cross-domain):     train_dataset == test_dataset, train_llm != test_llm
      - RQ3 (dataset cross-domain): train_llm == test_llm, train_dataset != test_dataset
    """

    print(f"\n🔀 GPTZero CROSS | TRAIN: {train_dataset.upper()} + {train_llm.upper()} "
          f"-> TEST: {test_dataset.upper()} + {test_llm.upper()}")

    # ---------- TRAINING (threshold calibration on source) ----------
    train_csv = f"{TRAIN_ROOT}/{train_dataset}/{train_dataset}_train_{train_llm}.csv"
    df_train = pd.read_csv(train_csv)

    train_ppl = []
    for text in tqdm(df_train["text"], desc="Training Perplexity"):
        train_ppl.append(calculate_perplexity(text))

    df_train["perplexity"] = train_ppl

    # Learn threshold
    threshold = learn_threshold_from_training(df_train)
    print(f"🔎 Learned threshold: {threshold:.2f}")

    # ---------- TESTING (on target) ----------
    test_csv = f"{TEST_ROOT}/{test_dataset}/{test_dataset}_test_{test_llm}.csv"
    df_test = pd.read_csv(test_csv)

    test_ppl = []
    preds = []

    for text in tqdm(df_test["text"], desc="Testing Perplexity"):
        ppl = calculate_perplexity(text)
        test_ppl.append(ppl)
        preds.append(classify_by_threshold(ppl, threshold))

    df_test["perplexity"] = test_ppl
    df_test["predicted_label"] = preds

    y_true = df_test["label"].values
    y_pred = preds

    test_ppl_arr = np.array(test_ppl, dtype=float)
    finite_ppl = test_ppl_arr[np.isfinite(test_ppl_arr)]
    fallback_max = finite_ppl.max() * 10 if finite_ppl.size > 0 else 1e6
    probs = np.where(np.isfinite(test_ppl_arr), test_ppl_arr, fallback_max)

    acc, f1, auroc, ai_recall, human_recall = compute_metrics(
        y_true, y_pred, probs
    )

    # ---------------------- SAVE ----------------------
    save_dir = (f"{RESULTS_ROOT}/cross_train_{train_dataset}_{train_llm}"
                f"__test_{test_dataset}_{test_llm}")
    os.makedirs(save_dir, exist_ok=True)

    df_test[["text", "label", "predicted_label", "perplexity"]].to_csv(
        f"{save_dir}/predictions.csv", index=False
    )

    result = {
        "train_dataset": train_dataset,
        "train_llm": train_llm,
        "test_dataset": test_dataset,
        "test_llm": test_llm,
        "threshold": threshold,
        "accuracy": acc,
        "f1_score": f1,
        "auroc": auroc,
        "AI_recall_label_0": ai_recall,
        "Human_recall_label_1": human_recall
    }

    append_to_master(result)

    print(f"✅ Done | Acc={acc:.3f}, AUC={auroc:.3f}, AI Recall={ai_recall:.3f}, "
          f"Human Recall={human_recall:.3f}")

# ---------------------- MAIN ----------------------
if __name__ == "__main__":

    # ---- In-domain (unchanged) ----
    for dataset in DATASETS:
        for llm in LLMS:
            run_gptzero_inference(dataset, llm)

    # ---- RQ2: LLM cross-domain (dataset fixed, LLM changes) ----
    for dataset in DATASETS:
        for train_llm in LLMS:
            for test_llm in LLMS:
                if train_llm == test_llm:
                    continue
                run_gptzero_cross_inference(dataset, train_llm, dataset, test_llm)

    # ---- RQ3: Dataset cross-domain (LLM fixed, dataset changes) ----
    for llm in LLMS:
        for train_dataset in DATASETS:
            for test_dataset in DATASETS:
                if train_dataset == test_dataset:
                    continue
                run_gptzero_cross_inference(train_dataset, llm, test_dataset, llm)

    print("\n🎯 GPTZero inference with ROC-calibrated thresholds completed.")