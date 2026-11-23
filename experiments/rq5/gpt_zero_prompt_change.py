"""
GPTZero inference script - PROMPT-CHANGE evaluation (RQ5 / Table 7).

Evaluates each of the six pre-calibrated GPTZero detectors on each of the six
prompt-shifted test sets:

    hindisumm x {gemini, llama3, qwen3}
    xquad     x {gemini, llama3, qwen3}

Total: 6 calibrated thresholds x 6 test sets = 36 evaluations.

Test files:
    new_prompt_data/{dataset}_{llm}_new_prompt.csv      (columns: text, label)

NO calibration happens here. Each detector's perplexity threshold was learned
earlier (ROC / Youden's J on the TRAIN split of its dataset+LLM pair) and is
frozen below in CALIBRATED_THRESHOLDS. Nothing is fitted on the new-prompt data.

IMPORTANT LABEL CONVENTION
--------------------------
The test CSVs (and the original GPTZero implementation) use the raw convention:

    raw label 0 = AI / machine-generated
    raw label 1 = Human

For consistency with the downstream statistical pipeline (and the XLM-R
prompt-change script), this script converts everything it WRITES to the
canonical convention:

    true_label        1 = AI,  0 = Human
    predicted_label   1 = AI,  0 = Human

The decision rule is unchanged: perplexity <= frozen threshold => AI.
AUROC is calculated with AI as the positive semantic class, using
ai_score = -perplexity (lower perplexity = more AI-like). GPTZero produces no
probabilities, so none are fabricated; the raw perplexity and ai_score are
written instead.

Per evaluation, this script produces:

    <results_root>/
        <train_dataset>_<train_llm>_tested_on_<test_dataset>_<test_llm>/
            metrics.csv
            predictions.csv

And one master file:

    gptzero_all_results_new_prompt.csv
"""

import os
import gc
from itertools import product

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, recall_score
)

# ---------------------- CONFIG ----------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

DATASETS = ["hindisumm", "xquad"]
LLMS = ["gemini", "llama3", "qwen3"]

NEW_PROMPT_DIR = "prompt_change/new_prompt_data"
RESULTS_ROOT = "gptzero/inference_results_new_prompt"
MASTER_CSV = os.path.join(RESULTS_ROOT, "gptzero_all_results_new_prompt.csv")

# ---------------------- PRE-TRAINED CALIBRATIONS ----------------------
# Perplexity thresholds (label 0 = AI if perplexity <= threshold) learned on the
# TRAIN split of each (train_dataset, train_llm) pair. Copied at full precision
# from the provided GPTZero results table; every row of that table that shares a
# train pair carries the same threshold, so one value per pair is sufficient.
CALIBRATED_THRESHOLDS = {
    ("hindisumm", "gemini"): 3.4991369247436523,
    ("hindisumm", "llama3"): 2.9589478969573975,
    ("hindisumm", "qwen3"):  3.033813953399658,
    ("xquad",     "gemini"): 3.4861392974853516,
    ("xquad",     "llama3"): 3.1967599391937256,
    ("xquad",     "qwen3"):  2.8649966716766357,
}

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

# ---------------------- CLASSIFIER ----------------------
def classify_by_threshold(ppl, threshold):
    # Canonical output convention: 1 = AI, 0 = Human
    # (low perplexity -> AI-like; rule unchanged from the original script)
    return 1 if ppl <= threshold else 0

# ---------------------- METRICS ----------------------
def compute_metrics(y_true, y_pred, ai_score):
    """
    Canonical labels: 1 = AI, 0 = Human. AI is the positive semantic class
    for F1 and AUROC. ai_score: higher = more AI-like (here -perplexity).
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, ai_score)
    except ValueError:
        auroc = float("nan")
    ai_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    human_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return acc, f1, auroc, ai_recall, human_recall

# ---------------------- TEST-SET LOADING ----------------------
def load_new_prompt_test_set(test_dataset, test_llm):
    """
    Load one prompt-shifted test set:
        new_prompt_data/{dataset}_{llm}_new_prompt.csv   (columns: text, label)
    Labels in the file are RAW: 0 = AI, 1 = Human (converted to the canonical
    convention when results are computed/written).
    """
    test_path = os.path.join(
        NEW_PROMPT_DIR, f"{test_dataset}_{test_llm}_new_prompt.csv"
    )

    if not os.path.exists(test_path):
        print(f"WARNING: test file not found for "
              f"{test_dataset}+{test_llm}: {test_path}")
        return None

    df = pd.read_csv(test_path)

    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{test_path} is missing required columns: {sorted(missing)}"
        )

    # Remove rows with missing text (same as the other prompt-change scripts).
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{test_path}: labels must be 0/1 "
                         f"(0 = AI, 1 = Human); found {np.unique(labels)}")

    return df, texts, labels

# ---------------------- PERPLEXITY CACHE ----------------------
def compute_test_perplexities(texts, desc):
    """
    Perplexity depends only on the text and mGPT (no threshold, no training
    data), so it is computed ONCE per test set and then re-used for all six
    calibrated thresholds. Results are identical to recomputing it per case.
    """
    ppl = []
    for text in tqdm(texts, desc=desc):
        ppl.append(calculate_perplexity(text))
    ppl = np.array(ppl, dtype=float)

    n_bad = int((~np.isfinite(ppl)).sum())
    if n_bad:
        print(f"WARNING: {n_bad}/{len(ppl)} non-finite perplexities "
              f"(treated as Human, as in the original GPTZero script).")
    return ppl

# ---------------------- PER-CASE EVALUATION ----------------------
def evaluate_one_case(train_dataset, train_llm, test_dataset, test_llm, test_cache):
    """
    Apply the frozen threshold of (train_dataset, train_llm) to the
    prompt-shifted test set (test_dataset, test_llm).
    """

    print(f"\n🔀 GPTZero PROMPT-CHANGE | TRAIN: {train_dataset.upper()} + "
          f"{train_llm.upper()} -> TEST(new_prompt): "
          f"{test_dataset.upper()} + {test_llm.upper()}")

    # ---------- PRE-TRAINED CALIBRATION (no re-learning) ----------
    threshold = CALIBRATED_THRESHOLDS[(train_dataset, train_llm)]
    print(f"🔎 Using frozen threshold: {threshold:.6f}")

    # ---------- TESTING (cached perplexities) ----------
    df_test, texts, raw_labels, test_ppl_arr = test_cache

    # Raw CSV labels (0 = AI, 1 = Human) -> canonical (1 = AI, 0 = Human)
    true_labels = 1 - raw_labels

    predicted_labels = np.array(
        [classify_by_threshold(ppl, threshold) for ppl in test_ppl_arr],
        dtype=int,
    )

    # AI-likeliness score for AUROC: lower perplexity = more AI-like.
    # Non-finite perplexities get the same large fallback as before.
    finite_ppl = test_ppl_arr[np.isfinite(test_ppl_arr)]
    fallback_max = finite_ppl.max() * 10 if finite_ppl.size > 0 else 1e6
    ppl_for_score = np.where(np.isfinite(test_ppl_arr), test_ppl_arr, fallback_max)
    ai_score = -ppl_for_score

    acc, f1, auroc, ai_recall, human_recall = compute_metrics(
        true_labels, predicted_labels, ai_score
    )

    # ---------------------- SAVE ----------------------
    result_dir = os.path.join(
        RESULTS_ROOT,
        f"{train_dataset}_{train_llm}_tested_on_{test_dataset}_{test_llm}",
    )
    os.makedirs(result_dir, exist_ok=True)

    pd.DataFrame(
        {
            "text": texts,
            "true_label": true_labels,            # 1 = AI, 0 = Human
            "predicted_label": predicted_labels,  # 1 = AI, 0 = Human
            "perplexity": test_ppl_arr,
            "ai_score": -test_ppl_arr,            # higher = more AI-like
        }
    ).to_csv(os.path.join(result_dir, "predictions.csv"), index=False)

    # Same column names/order as the XLM-R metrics.csv; the frozen threshold
    # is appended as an extra last column for auditability.
    result = {
        "Train Dataset": train_dataset,
        "Train LLM": train_llm,
        "Test Dataset": test_dataset,
        "Test LLM": test_llm,
        "Accuracy": acc,
        "F1 Score": f1,
        "AUROC": auroc,
        "AI Recall": ai_recall,
        "Human Recall": human_recall,
        "Threshold": threshold,
    }

    pd.DataFrame([result]).to_csv(
        os.path.join(result_dir, "metrics.csv"), index=False
    )

    print(f"✅ Done | Acc={acc:.3f}, AUC={auroc:.3f}, AI Recall={ai_recall:.3f}, "
          f"Human Recall={human_recall:.3f}")

    return result

# ---------------------- MAIN ----------------------
def main():

    all_combinations = list(product(DATASETS, LLMS))

    # Every calibration used below must exist before any work is done.
    for combo in all_combinations:
        if combo not in CALIBRATED_THRESHOLDS:
            raise KeyError(f"No pre-trained threshold for {combo}")

    # ---- Load the six new-prompt test sets + their perplexities once ----
    test_cache = {}

    print("\n" + "=" * 75)
    print("Loading new-prompt test sets")
    print("=" * 75)

    for test_dataset, test_llm in all_combinations:

        loaded = load_new_prompt_test_set(test_dataset, test_llm)
        if loaded is None:
            continue

        df_test, texts, labels = loaded
        print(f"Loaded {test_dataset}+{test_llm}: {len(texts)} texts "
              f"(AI={(labels == 0).sum()}, Human={(labels == 1).sum()})")

        ppl = compute_test_perplexities(
            texts, desc=f"Perplexity {test_dataset}+{test_llm}"
        )
        test_cache[(test_dataset, test_llm)] = (df_test, texts, labels, ppl)

    print(f"\nCached {len(test_cache)}/{len(all_combinations)} "
          f"new-prompt test sets.")

    # ---- 6 calibrated thresholds x 6 test sets = 36 evaluations ----
    total_cases = len(all_combinations) ** 2
    case_num = 0

    for train_dataset, train_llm in all_combinations:
        for test_dataset, test_llm in all_combinations:

            case_num += 1
            print(f"\n[{case_num}/{total_cases}]")

            cached = test_cache.get((test_dataset, test_llm))
            if cached is None:
                print(f"Skipping: no test data for {test_dataset}+{test_llm}")
                continue

            try:
                evaluate_one_case(
                    train_dataset, train_llm, test_dataset, test_llm, cached
                )
            except Exception as exc:
                print(f"FAILED case {train_dataset}+{train_llm} -> "
                      f"{test_dataset}+{test_llm}: "
                      f"{type(exc).__name__}: {exc}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Rebuild master CSV from per-case metrics (no duplicate rows) ----
    all_metrics = []
    for train_dataset, train_llm in all_combinations:
        for test_dataset, test_llm in all_combinations:
            metrics_path = os.path.join(
                RESULTS_ROOT,
                f"{train_dataset}_{train_llm}_tested_on_"
                f"{test_dataset}_{test_llm}",
                "metrics.csv",
            )
            if os.path.exists(metrics_path):
                all_metrics.append(
                    pd.read_csv(metrics_path, float_precision="round_trip")
                )

    if all_metrics:
        master_df = pd.concat(all_metrics, ignore_index=True)
        master_df.to_csv(MASTER_CSV, index=False)
        print("\n" + "=" * 75)
        print("🎯 GPTZero prompt-change evaluations completed.")
        print(f"Evaluations written: {len(master_df)}/{total_cases}")
        print(f"Master results:\n  {MASTER_CSV}")
        print("=" * 75)
    else:
        print("\nNo evaluation results were generated.")


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    main()