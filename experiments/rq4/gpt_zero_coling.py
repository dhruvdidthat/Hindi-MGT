"""
GPTZero inference script - UNSEEN DATASET + UNSEEN LLM (RQ4 / Table 6).

Evaluates each of the six pre-calibrated GPTZero detectors on the Hindi subset
of the COLING 2025 multilingual MGT test set (GenAI Content Detection Task 1;
BBC news, machine text from GPT-4o), stored in a single file:

    hindi_data.csv
    columns: text, language, label, source, model, prompt_flag, prompt, domain,
             question, title, dialect, essay_title, category, grade, rating,
             testset_id, predicted_label

    hindisumm x {gemini, llama3, qwen3}  \\  six frozen perplexity thresholds
    xquad     x {gemini, llama3, qwen3}  /  -> 6 evaluations on the same file

NO calibration happens here. Each threshold was learned earlier (ROC / Youden's J
on the TRAIN split of its dataset+LLM pair) and is frozen in
CALIBRATED_THRESHOLDS. Nothing is fitted on the COLING data.

IMPORTANT LABEL CONVENTION
--------------------------
The COLING file's own label convention is set by COLING_AI_LABEL below and is
CHECKED against the file's `model` column at start-up (rows whose model is
"human" must carry the other label); the script stops if they disagree.

Everything this script WRITES uses the canonical convention (same as the XLM-R
and GPTZero prompt-change scripts):

    true_label        1 = AI,  0 = Human
    predicted_label   1 = AI,  0 = Human

Decision rule (unchanged): perplexity <= frozen threshold => AI.
AUROC uses AI as the positive class with ai_score = -perplexity. GPTZero gives
no probabilities, so none are fabricated.

The input file's own `predicted_label` column (if present) is IGNORED and not
copied, so it can never be confused with this script's predictions.

Per evaluation:

    <results_root>/<train_dataset>_<train_llm>_tested_on_coling_hindi_gpt4o/
        metrics.csv
        predictions.csv

And one master file:

    gptzero_all_results_coling.csv
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASETS = ["hindisumm", "xquad"]
LLMS = ["gemini", "llama3", "qwen3"]

COLING_CSV = "hindi_data.csv"
TEST_DATASET_NAME = "coling_2025_hindi"
TEST_LLM_NAME = "gpt-4o"
TEST_TAG = "coling_hindi_gpt4o"

# Label value that means "machine-generated" INSIDE hindi_data.csv.
# Verified against the `model` column at start-up (see verify_label_mapping).
COLING_AI_LABEL = 0

# If the file mixes languages, set this (e.g. "hi") to keep only those rows.
# Left as None, a multi-language file stops the script instead of being
# silently mixed.
LANGUAGE_FILTER = None

RESULTS_ROOT = "gptzero/inference_results_coling"
MASTER_CSV = os.path.join(RESULTS_ROOT, "gptzero_all_results_coling.csv")

# ---------------------- PRE-TRAINED CALIBRATIONS ----------------------
# Perplexity thresholds (AI if perplexity <= threshold) learned on the TRAIN
# split of each (train_dataset, train_llm) pair. Copied at full precision from
# the provided GPTZero results table.
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

# ---------------------- LABEL VERIFICATION ----------------------
def verify_label_mapping(df):
    """
    Print the label x model cross-tab and confirm COLING_AI_LABEL against the
    file itself: rows whose `model` is "human" must carry the non-AI label and
    all other rows the AI label. Stops on any inconsistency.
    """
    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError(
            f"{COLING_CSV}: labels must be 0/1; found "
            f"{sorted(df['label'].astype(str).unique())}"
        )

    if "model" not in df.columns:
        print("WARNING: no `model` column - label mapping could not be "
              f"verified from the data; assuming AI label = {COLING_AI_LABEL}.")
        return

    print("\nlabel x model cross-tab:")
    print(pd.crosstab(df["model"].astype(str), labels.astype(int)).to_string())

    is_human = df["model"].astype(str).str.strip().str.lower().eq("human")
    if not is_human.any() or is_human.all():
        print("WARNING: `model` has no human/machine split - label mapping "
              f"not verified; assuming AI label = {COLING_AI_LABEL}.")
        return

    human_labels = set(labels[is_human].astype(int).unique())
    ai_labels = set(labels[~is_human].astype(int).unique())
    if len(human_labels) != 1 or len(ai_labels) != 1 or human_labels == ai_labels:
        raise ValueError(
            "Label column is inconsistent with the `model` column "
            f"(human rows have labels {sorted(human_labels)}, other rows "
            f"{sorted(ai_labels)})."
        )
    inferred_ai = ai_labels.pop()
    if inferred_ai != COLING_AI_LABEL:
        raise ValueError(
            f"COLING_AI_LABEL={COLING_AI_LABEL} but the file marks machine "
            f"text with label {inferred_ai}. Fix COLING_AI_LABEL."
        )
    print(f"Label mapping verified: label {COLING_AI_LABEL} = AI (machine), "
          f"label {1 - COLING_AI_LABEL} = Human.")

# ---------------------- TEST-SET LOADING ----------------------
def load_coling_hindi():
    if not os.path.exists(COLING_CSV):
        raise FileNotFoundError(f"{COLING_CSV} not found")

    df = pd.read_csv(COLING_CSV)

    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{COLING_CSV} is missing required columns: "
                         f"{sorted(missing)}")

    if "language" in df.columns:
        langs = df["language"].astype(str).value_counts().to_dict()
        print(f"language counts: {langs}")
        if LANGUAGE_FILTER is not None:
            df = df[df["language"].astype(str) == LANGUAGE_FILTER]
            print(f"Filtered to language == {LANGUAGE_FILTER!r}: {len(df)} rows")
        elif len(langs) > 1:
            raise ValueError(
                f"{COLING_CSV} contains several languages {sorted(langs)}; "
                "set LANGUAGE_FILTER (e.g. 'hi') to select the Hindi rows."
            )

    n_before = len(df)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    if len(df) != n_before:
        print(f"Dropped {n_before - len(df)} rows with missing text.")

    verify_label_mapping(df)

    # Raw COLING label -> canonical (1 = AI, 0 = Human)
    true_labels = (df["label"].astype(int).to_numpy() == COLING_AI_LABEL).astype(int)
    texts = df["text"].astype(str).tolist()

    n_ai, n_h = int(true_labels.sum()), int((1 - true_labels).sum())
    print(f"Loaded {len(texts)} texts (AI={n_ai}, Human={n_h})")
    if (len(texts), n_ai, n_h) != (1199, 600, 599):
        print("NOTE: the paper describes 1,199 Hindi articles (599 human / "
              "600 GPT-4o); the counts above differ - check the file.")

    return df, texts, true_labels

# ---------------------- PERPLEXITY (computed once) ----------------------
def compute_test_perplexities(texts):
    """
    Perplexity depends only on the text and mGPT (no threshold, no training
    data), so it is computed ONCE and re-used for all six calibrated
    thresholds. Results are identical to recomputing it per case.
    """
    ppl = []
    for text in tqdm(texts, desc="Perplexity (COLING Hindi)"):
        ppl.append(calculate_perplexity(text))
    ppl = np.array(ppl, dtype=float)

    n_bad = int((~np.isfinite(ppl)).sum())
    if n_bad:
        print(f"WARNING: {n_bad}/{len(ppl)} non-finite perplexities "
              f"(classified Human, as in the original GPTZero script).")
    return ppl

# ---------------------- PER-CASE EVALUATION ----------------------
def evaluate_one_case(train_dataset, train_llm, test_cache):
    """
    Apply the frozen threshold of (train_dataset, train_llm) to the COLING
    Hindi test set.
    """

    print(f"\n🔀 GPTZero COLING | TRAIN: {train_dataset.upper()} + "
          f"{train_llm.upper()} -> TEST: {TEST_DATASET_NAME} ({TEST_LLM_NAME})")

    threshold = CALIBRATED_THRESHOLDS[(train_dataset, train_llm)]
    print(f"🔎 Using frozen threshold: {threshold:.6f}")

    df_test, texts, true_labels, test_ppl_arr = test_cache

    predicted_labels = np.array(
        [classify_by_threshold(ppl, threshold) for ppl in test_ppl_arr],
        dtype=int,
    )

    # AI-likeliness score for AUROC: lower perplexity = more AI-like.
    finite_ppl = test_ppl_arr[np.isfinite(test_ppl_arr)]
    fallback_max = finite_ppl.max() * 10 if finite_ppl.size > 0 else 1e6
    ppl_for_score = np.where(np.isfinite(test_ppl_arr), test_ppl_arr, fallback_max)
    ai_score = -ppl_for_score

    acc, f1, auroc, ai_recall, human_recall = compute_metrics(
        true_labels, predicted_labels, ai_score
    )

    # ---------------------- SAVE ----------------------
    result_dir = os.path.join(
        RESULTS_ROOT, f"{train_dataset}_{train_llm}_tested_on_{TEST_TAG}"
    )
    os.makedirs(result_dir, exist_ok=True)

    pred_df = pd.DataFrame(
        {
            "row_id": np.arange(len(texts)),
            "text": texts,
            "true_label": true_labels,            # 1 = AI, 0 = Human
            "predicted_label": predicted_labels,  # 1 = AI, 0 = Human
            "perplexity": test_ppl_arr,
            "ai_score": -test_ppl_arr,            # higher = more AI-like
        }
    )
    # Traceability columns from the source file (never its own predicted_label)
    for col in ("source", "model", "testset_id"):
        if col in df_test.columns:
            pred_df[col] = df_test[col].to_numpy()
    pred_df.to_csv(os.path.join(result_dir, "predictions.csv"), index=False)

    result = {
        "Train Dataset": train_dataset,
        "Train LLM": train_llm,
        "Test Dataset": TEST_DATASET_NAME,
        "Test LLM": TEST_LLM_NAME,
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

    for combo in all_combinations:
        if combo not in CALIBRATED_THRESHOLDS:
            raise KeyError(f"No pre-trained threshold for {combo}")

    print("\n" + "=" * 75)
    print(f"Loading {COLING_CSV}")
    print("=" * 75)

    df_test, texts, true_labels = load_coling_hindi()
    ppl = compute_test_perplexities(texts)
    test_cache = (df_test, texts, true_labels, ppl)

    # ---- six calibrated thresholds x one test file = 6 evaluations ----
    results = []
    for case_num, (train_dataset, train_llm) in enumerate(all_combinations, 1):
        print(f"\n[{case_num}/{len(all_combinations)}]")
        try:
            results.append(
                evaluate_one_case(train_dataset, train_llm, test_cache)
            )
        except Exception as exc:
            print(f"FAILED case {train_dataset}+{train_llm}: "
                  f"{type(exc).__name__}: {exc}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if results:
        pd.DataFrame(results).to_csv(MASTER_CSV, index=False)
        print("\n" + "=" * 75)
        print("🎯 GPTZero COLING evaluations completed.")
        print(f"Evaluations written: {len(results)}/{len(all_combinations)}")
        print(f"Master results:\n  {MASTER_CSV}")
        print("=" * 75)
    else:
        print("\nNo evaluation results were generated.")


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    main()