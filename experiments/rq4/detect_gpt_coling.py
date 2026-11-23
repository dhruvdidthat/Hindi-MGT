#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DetectGPT on the unseen dataset + unseen LLM (RQ4 / Table 6).

=======================================================================
WHAT THIS SCRIPT DOES
=======================================================================
Tests DetectGPT on the Hindi subset of the COLING 2025 multilingual MGT
test set (GenAI Content Detection Task 1; BBC news, GPT-4o) stored in a
single file, hindi_data.csv, using each of the SIX already-calibrated
thresholds (one per source pair: {hindisumm, xquad} x {gemini, llama3,
qwen3}). Nothing is re-calibrated here.

  - Scores z(x): UNCHANGED DetectGPT pipeline (same models, perturbation,
    hyperparameters, compute_llr). Each text is scored ONCE and cached.
  - Threshold epsilon: read from CALIBRATION_CSV (in-domain rows hold one
    threshold per (train_dataset, train_llm) pair).
  - Decision rule (unchanged): z >= epsilon  =>  AI.

Because the score does not depend on the source pair, AUROC is identical
for all six rows; only accuracy / F1 / AI recall / Human recall change
with the threshold.

hindi_data.csv columns used: text, label (and 'model' for an automatic
label-convention check). All other columns, including the file's own
'predicted_label', are ignored. Rows are NOT filtered by language: the file
is taken as-is.

Outputs use the canonical convention (1 = AI, 0 = Human):
    <COLING_ROOT>/<train_ds>_<train_llm>_tested_on_coling_hindi/
        metrics.csv
        predictions.csv
    detect_gpt_calibrated_coling_results.csv    (master, rebuilt each run)

=======================================================================
HYPERPARAMETERS vs. Mitchell et al. (2023) (UNCHANGED)
=======================================================================
  PCT_WORDS_MASKED = 0.15   MATCHES PAPER (Section 5.1)
  SPAN_LENGTH = 2           MATCHES PAPER (Section 5.1)
  N_PERTURB = 100           MATCHES PAPER (Section 4, Figure 8)
  MASK_TOP_P = 1.0, MASK_TEMPERATURE = 1.0   MATCHES PAPER
  MAX_TOKENS = 512          NOT a paper hyperparameter (truncation guard)
  CHUNK_SIZE, BUFFER_SIZE, LL_BATCH_SIZE   throughput knobs only
  Mask-filling model google/mt5-base        DEVIATION (paper: T5-3B/mT5-3B)
  Source model sberbank-ai/mGPT             DEVIATION (paper: English LMs)
"""

import os
import re
import gc
from itertools import product

import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, recall_score
)
from indicnlp.tokenize import indic_tokenize

# -------------------- SPEED SETUP --------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# -------------------- CONFIG --------------------
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
AUTOCAST_DTYPE = torch.bfloat16
USE_AUTOCAST = torch.cuda.is_available()

DATASETS = ["hindisumm", "xquad"]
LLMS = ["gemini", "llama3", "qwen3"]

HINDI_DATA_PATH = "hindi_data.csv"      # single COLING Hindi file

RESULTS_ROOT = "detect_gpt_all_domain/inference_results"
CALIB_ROOT = os.path.join(RESULTS_ROOT, "calibrated")
CACHE_ROOT = os.path.join(CALIB_ROOT, "score_cache")
CACHE_PATH = os.path.join(CACHE_ROOT, "coling_hindi.csv")

# Existing calibrated results (source of the pretrained thresholds).
CALIBRATION_CSV = os.path.join(CALIB_ROOT, "detect_gpt_calibrated_all_results.csv")

COLING_ROOT = os.path.join(CALIB_ROOT, "coling")
MASTER_RESULTS_CSV = os.path.join(
    COLING_ROOT, "detect_gpt_calibrated_coling_results.csv"
)

# Raw label meaning "AI" in hindi_data.csv. Verified automatically against
# the 'model' column (rows whose model is "human"); see check_label_convention().
HINDI_AI_LABEL = 0

MAX_TOKENS = 512          # implementation choice, not a paper hyperparameter

# --- Paper hyperparameters (Section 5.1) ---
N_PERTURB = 100
SPAN_LENGTH = 2
PCT_WORDS_MASKED = 0.15
MASK_TOP_P = 1.0
MASK_TEMPERATURE = 1.0

# --- Throughput knobs, NOT paper hyperparameters ---
CHUNK_SIZE = 25
BUFFER_SIZE = 1
LL_BATCH_SIZE = 16

BASE_MODEL_NAME = "sberbank-ai/mGPT"          # deviation from paper
MASK_MODEL_NAME = "google/mt5-base"           # deviation from paper

os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(CALIB_ROOT, exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(COLING_ROOT, exist_ok=True)

pattern = re.compile(r"<extra_id_\d+>")

# -------------------- LOAD MODELS --------------------
def _load_causal_lm(name):
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=AUTOCAST_DTYPE if USE_AUTOCAST else torch.float32,
            attn_implementation="sdpa",
        )
    except (ValueError, TypeError):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=AUTOCAST_DTYPE if USE_AUTOCAST else torch.float32,
        )
    return model.to(DEVICE).eval()


def _load_seq2seq_lm(name):
    try:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            name, torch_dtype=AUTOCAST_DTYPE if USE_AUTOCAST else torch.float32,
            attn_implementation="sdpa",
        )
    except (ValueError, TypeError):
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            name, torch_dtype=AUTOCAST_DTYPE if USE_AUTOCAST else torch.float32,
        )
    return model.to(DEVICE).eval()


base_model = _load_causal_lm(BASE_MODEL_NAME)
base_tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if base_tokenizer.pad_token_id is None:
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

mask_model = _load_seq2seq_lm(MASK_MODEL_NAME)
mask_tokenizer = transformers.AutoTokenizer.from_pretrained(MASK_MODEL_NAME, model_max_length=512)


# -------------------- UTILS (UNCHANGED) --------------------
def truncate_text(text):
    tokens = base_tokenizer.tokenize(text)
    tokens = tokens[:MAX_TOKENS]
    return base_tokenizer.convert_tokens_to_string(tokens)


def strip_newlines(text):
    return " ".join(text.split())


def get_ll_batch(texts, batch_size=LL_BATCH_SIZE):
    """
    Computes log p(x) for a list of texts under the base model, batched.
    Uses a manual per-sequence NLL computation (rather than the
    batch-averaged loss HF returns) so padding doesn't bias the result.
    """
    all_ll = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = base_tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(DEVICE)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
                out = base_model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
                logits = out.logits

        # Standard causal-LM shift: predict token t from tokens < t.
        shift_logits = logits[:, :-1, :].float()
        shift_labels = enc.input_ids[:, 1:]
        shift_mask = enc.attention_mask[:, 1:].float()

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_ll = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_ll = token_ll * shift_mask

        denom = shift_mask.sum(dim=1).clamp(min=1)
        seq_ll = (token_ll.sum(dim=1) / denom)
        all_ll.extend(seq_ll.tolist())
    return all_ll


def tokenize_and_mask(text):
    tokens = indic_tokenize.trivial_tokenize(text, lang="hi")
    mask_string = "<<<mask>>>"
    n_spans = int(PCT_WORDS_MASKED * len(tokens) / (SPAN_LENGTH + BUFFER_SIZE * 2))
    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, max(1, len(tokens) - SPAN_LENGTH))
        end = start + SPAN_LENGTH
        if mask_string not in tokens[max(0, start - BUFFER_SIZE):min(len(tokens), end + BUFFER_SIZE)]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    filled = 0
    for i, t in enumerate(tokens):
        if t == mask_string:
            tokens[i] = f"<extra_id_{filled}>"
            filled += 1
    return " ".join(tokens)


def perturb_texts(texts):
    outputs = []
    for i in range(0, len(texts), CHUNK_SIZE):
        batch = texts[i:i + CHUNK_SIZE]
        masked = [tokenize_and_mask(t) for t in batch]
        tokens = mask_tokenizer(
            masked,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE, enabled=USE_AUTOCAST):
                gen = mask_model.generate(
                    **tokens,
                    do_sample=True,
                    top_p=MASK_TOP_P,
                    temperature=MASK_TEMPERATURE,
                    max_length=512,
                )

        decoded = mask_tokenizer.batch_decode(gen, skip_special_tokens=False)
        for m, d in zip(masked, decoded):
            fills = pattern.split(d)[1:-1]
            toks = m.split()
            for j, f in enumerate(fills):
                tag = f"<extra_id_{j}>"
                if tag in toks:
                    toks[toks.index(tag)] = f.strip()
            outputs.append(" ".join(toks))
    return outputs


def compute_llr(text):
    """
    Normalized perturbation discrepancy:

        z(x) = (log p(x) - mean_i log p(x_tilde_i)) / std_i log p(x_tilde_i)

    Higher z => more likely machine-generated (paper Section 4, Alg. 1).
    """
    text = truncate_text(strip_newlines(text))
    perturbed = perturb_texts([text] * N_PERTURB)
    perturbed = [p for p in perturbed if p.strip()]
    if not perturbed:
        return None

    all_lls = get_ll_batch([text] + perturbed)
    ll_orig, ll_pert = all_lls[0], all_lls[1:]

    mu = np.mean(ll_pert)
    # ddof=0 (NumPy default) to match the official DetectGPT implementation
    sigma = np.std(ll_pert) if np.std(ll_pert) > 0 else 1.0
    return (ll_orig - mu) / sigma


# ===================== COLING (RQ4 / TABLE 6) INFERENCE LOGIC =====================

# -------------------- PRETRAINED THRESHOLDS --------------------
def load_calibrated_thresholds():
    """
    Reads the already-learned thresholds from CALIBRATION_CSV. The threshold
    depends only on the source (train_dataset, train_llm) pair, so it is taken
    from that pair's in-domain row. No re-calibration is performed.
    """
    if not os.path.exists(CALIBRATION_CSV):
        raise FileNotFoundError(
            f"Calibrated results CSV not found: {CALIBRATION_CSV}. "
            f"Set CALIBRATION_CSV to the file holding your learned thresholds."
        )

    df = pd.read_csv(CALIBRATION_CSV)
    df = df[(df["train_dataset"] == df["test_dataset"]) &
            (df["train_llm"] == df["test_llm"])]

    thresholds = {}
    for (ds, llm), grp in df.groupby(["train_dataset", "train_llm"]):
        vals = grp["threshold"].astype(float).unique()
        if len(vals) > 1:
            raise ValueError(
                f"Conflicting thresholds for {ds}+{llm} in {CALIBRATION_CSV}: "
                f"{vals.tolist()}. Remove duplicate/stale rows and rerun."
            )
        thresholds[(ds, llm)] = float(vals[0])

    print("\nLoaded pretrained thresholds:")
    for (ds, llm), thr in sorted(thresholds.items()):
        print(f"  {ds}+{llm}: {thr:.6f}")
    return thresholds


# -------------------- LABEL CONVENTION CHECK --------------------
def check_label_convention(df):
    """
    Uses the file's own 'model' column: rows whose model is "human" are
    human-written. If those rows are labelled AI under HINDI_AI_LABEL, the
    convention is inverted -> stop (labels are never flipped silently).
    """
    if "model" not in df.columns:
        print("  [label check skipped] no 'model' column in the file. "
              "Confirm HINDI_AI_LABEL manually.")
        return

    models = df["model"].astype(str).str.strip().str.lower()
    print(f"  model column values: {models.value_counts().to_dict()}")

    is_human = models == "human"
    n_human = int(is_human.sum())
    if n_human == 0:
        print("  [label check inconclusive] no rows with model == 'human'. "
              "Confirm HINDI_AI_LABEL manually.")
        return

    frac_ai = float((df.loc[is_human, "label"] == HINDI_AI_LABEL).mean())
    if frac_ai > 0.5:
        raise ValueError(
            f"Label convention looks INVERTED: {frac_ai:.0%} of {n_human} rows with "
            f"model == 'human' are labelled AI under HINDI_AI_LABEL={HINDI_AI_LABEL}. "
            f"Set HINDI_AI_LABEL = {1 - HINDI_AI_LABEL} and rerun."
        )
    if frac_ai > 0:
        print(f"  [label check warning] {frac_ai:.1%} of {n_human} human-model rows "
              f"are labelled AI.")
    else:
        print(f"  [label check OK] all {n_human} rows with model == 'human' are "
              f"labelled Human.")


# -------------------- SCORING WITH DISK CACHE --------------------
def load_or_score_coling():
    """
    Returns the COLING Hindi data with an 'llr' column (unchanged z-score).
    Scored once and cached so reruns reuse identical scores.
    """
    if os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH)
        print(f"Loaded cached scores: {CACHE_PATH} ({len(df)} rows)")
        check_label_convention(df)
        return df

    if not os.path.exists(HINDI_DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {HINDI_DATA_PATH}")

    df = pd.read_csv(HINDI_DATA_PATH)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{HINDI_DATA_PATH} is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    print(f"Rows: {len(df)} | label counts: {df['label'].value_counts().to_dict()}")
    if "language" in df.columns:
        print(f"language values: {df['language'].value_counts().to_dict()}")

    check_label_convention(df)   # fail fast, before the expensive scoring

    scores = []
    for text in tqdm(df["text"].astype(str).tolist(), desc="Scoring COLING Hindi"):
        scores.append(compute_llr(text))
    df["llr"] = scores
    df.to_csv(CACHE_PATH, index=False)
    return df


def classify_by_threshold(z, threshold):
    return 1 if z >= threshold else 0      # canonical: 1 = AI, 0 = Human


# -------------------- ONE EVALUATION --------------------
def evaluate_one_threshold(train_dataset, train_llm, threshold, df_scored):
    print("\n" + "=" * 75)
    print(f"THRESHOLD FROM {train_dataset}+{train_llm} (eps={threshold:.6f}) "
          f"-> TEST: COLING Hindi (GPT-4o)")
    print("=" * 75)

    valid = df_scored.dropna(subset=["llr"]).copy()
    n_dropped = len(df_scored) - len(valid)
    if n_dropped:
        print(f"Dropped {n_dropped} rows with no valid score.")

    # Canonical labels: 1 = AI, 0 = Human
    true_labels = (valid["label"].values == HINDI_AI_LABEL).astype(int)
    y_score = valid["llr"].values.astype(float)
    pred_labels = np.array([classify_by_threshold(z, threshold) for z in y_score])

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    try:
        auroc = roc_auc_score(true_labels, y_score)     # threshold-free
    except ValueError:
        auroc = float("nan")
    ai_recall = recall_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    human_recall = recall_score(true_labels, pred_labels, pos_label=0, zero_division=0)

    print(f"Accuracy:      {accuracy:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"AUROC:         {auroc:.4f}")
    print(f"AI Recall:     {ai_recall:.4f}")
    print(f"Human Recall:  {human_recall:.4f}")

    result_dir = os.path.join(
        COLING_ROOT, f"{train_dataset}_{train_llm}_tested_on_coling_hindi"
    )
    os.makedirs(result_dir, exist_ok=True)

    metrics_df = pd.DataFrame([{
        "Train Dataset": train_dataset,
        "Train LLM": train_llm,
        "Test Dataset": "coling_hindi",
        "Test LLM": "gpt-4o",
        "Threshold": threshold,
        "N Evaluated": len(valid),
        "Accuracy": accuracy,
        "F1 Score": f1,
        "AUROC": auroc,
        "AI Recall": ai_recall,
        "Human Recall": human_recall,
    }])
    metrics_df.to_csv(os.path.join(result_dir, "metrics.csv"), index=False)

    predictions_df = pd.DataFrame({
        "text": valid["text"].values,
        "true_label": true_labels,            # canonical: 1 = AI, 0 = Human
        "predicted_label": pred_labels,       # canonical: 1 = AI, 0 = Human
        "llr": y_score,
        "threshold": threshold,
    })
    predictions_df.to_csv(os.path.join(result_dir, "predictions.csv"), index=False)

    print(f"Results saved to:\n  {result_dir}")
    return metrics_df


# -------------------- MAIN --------------------
def main():
    thresholds = load_calibrated_thresholds()
    pairs = list(product(DATASETS, LLMS))

    print("\n" + "=" * 75)
    print("Loading / scoring COLING Hindi data")
    print("=" * 75)
    df_scored = load_or_score_coling()
    torch.cuda.empty_cache()
    gc.collect()

    for train_dataset, train_llm in pairs:
        if (train_dataset, train_llm) not in thresholds:
            print(f"\nSkipping {train_dataset}+{train_llm}: "
                  f"no calibrated threshold in {CALIBRATION_CSV}")
            continue
        evaluate_one_threshold(
            train_dataset, train_llm,
            thresholds[(train_dataset, train_llm)], df_scored,
        )

    # ---- Rebuild master CSV from all metrics.csv files (no duplicate rows on rerun) ----
    all_metrics = []
    for train_dataset, train_llm in pairs:
        metrics_path = os.path.join(
            COLING_ROOT,
            f"{train_dataset}_{train_llm}_tested_on_coling_hindi",
            "metrics.csv",
        )
        if os.path.exists(metrics_path):
            all_metrics.append(pd.read_csv(metrics_path))

    if all_metrics:
        master_df = pd.concat(all_metrics, ignore_index=True)
        master_df.to_csv(MASTER_RESULTS_CSV, index=False)
        print("\n" + "=" * 75)
        print("COLING evaluations completed.")
        print(f"Evaluations written: {len(master_df)}/{len(pairs)}")
        print(f"Master results:\n  {MASTER_RESULTS_CSV}")
        print("=" * 75)
    else:
        print("\nNo evaluation results were generated.")


if __name__ == "__main__":
    main()