#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DetectGPT inference script (zero-shot SCORE + training-calibrated THRESHOLD).

=======================================================================
WHAT CHANGED IN THIS VERSION (and ONLY this)
=======================================================================
The score z(x) = (log p(x) - mean_i log p(x~_i)) / std_i log p(x~_i) is
computed EXACTLY as before (same perturbation, same scoring, same
hyperparameters). The only change is the decision threshold epsilon
(Algorithm 1 in Mitchell et al. 2023 takes epsilon as an input):

  - Before: epsilon = 0, fixed a priori.
  - Now:    epsilon is learned from the TRAIN split of the source
            (dataset, LLM) pair by maximising Youden's J = TPR - FPR on the
            ROC curve (same procedure as the GPTZero script), then applied
            unchanged to the TEST split of the target (dataset, LLM) pair.

  * The score direction is fixed by the paper's hypothesis: higher z =>
    more likely machine-generated. The direction is NOT learned.
  * AUROC is computed on the raw score and is therefore unaffected by the
    threshold; only accuracy / F1 / AI recall / Human recall depend on it.
  * Cross-domain evaluation mirrors the GPTZero script:
      - in-domain
      - RQ2: LLM cross-domain     (dataset fixed, LLM changes)
      - RQ3: dataset cross-domain (LLM fixed, dataset changes)
  * Scores for each (split, dataset, LLM) are computed once and cached on
    disk, so each text is perturbed/scored only once across all runs.

Everything below that is not marked "# [CALIBRATION CHANGE]" is unchanged
from the previous script, including the hyperparameter notes.

=======================================================================
HYPERPARAMETERS vs. Mitchell et al. (2023) "DetectGPT: Zero-Shot
Machine-Generated Text Detection using Probability Curvature"
=======================================================================
See Section 5.1 "Hyperparameters" and Section 4 (Figure 8) of the paper.

  PCT_WORDS_MASKED = 0.15
      MATCHES PAPER. "Using BERT masked language modeling as inspiration,
      we use 15% as the mask rate." (Section 5.1).

  SPAN_LENGTH = 2
      MATCHES PAPER. Swept {2, 5, 10} on held-out XSum, 2 performed best.

  N_PERTURB = 100
      MATCHES PAPER. "We approximate the expectation in Eq. 1 with 100
      samples from T5" (Section 4); Figure 8 shows convergence ~100.

  MASK_TOP_P = 1.0, MASK_TEMPERATURE = 1.0
      MATCHES PAPER. Mask filler sampled directly with temperature 1.

  MAX_TOKENS = 512
      NOT a paper hyperparameter; practical truncation guard.

  CHUNK_SIZE, BUFFER_SIZE = 1, LL_BATCH_SIZE
      NOT paper hyperparameters; throughput/batching knobs.

  Mask-filling model: google/mt5-base
      DEVIATION FROM PAPER (paper: T5-3B / mT5-3B).

  Source model: sberbank-ai/mGPT
      DEVIATION FROM PAPER (paper: GPT-2/OPT/Neo/J/NeoX/GPT-3/J2).

=======================================================================
SPEED NOTES (target: NVIDIA L40)
=======================================================================
- Batched log-likelihood scoring of original + all perturbations.
- bf16 autocast, TF32 matmuls, cuDNN benchmark, inference_mode.
- These only change how fast numbers are computed, not what is computed.
"""

import os
import re
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, recall_score, roc_curve
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

TRAIN_ROOT = "training_data"                                  # [CALIBRATION CHANGE] added
TEST_ROOT = "testing_data"
RESULTS_ROOT = "detect_gpt_all_domain/inference_results"
# [CALIBRATION CHANGE] new output locations so the old master CSV (different
# columns) is not corrupted and old predictions.csv files are not overwritten.
CALIB_ROOT = os.path.join(RESULTS_ROOT, "calibrated")
CACHE_ROOT = os.path.join(CALIB_ROOT, "score_cache")
MASTER_CSV = os.path.join(CALIB_ROOT, "detect_gpt_calibrated_all_results.csv")

# [CALIBRATION CHANGE] Label convention of the CSVs. The previous DetectGPT
# script used 1 = AI, 0 = Human (the GPTZero script used the opposite).
# CONFIRM THIS MATCHES YOUR FILES; if your files are 0 = AI, set AI_LABEL = 0.
AI_LABEL = 1
HUMAN_LABEL = 1 - AI_LABEL

MAX_TOKENS = 512          # implementation choice, not a paper hyperparameter (see docstring)

# --- Paper hyperparameters (Section 5.1) ---
N_PERTURB = 100            # paper: 100 samples from T5
SPAN_LENGTH = 2            # paper: swept {2,5,10} on held-out XSum, 2 best
PCT_WORDS_MASKED = 0.15    # paper: "15% as the mask rate"
MASK_TOP_P = 1.0           # paper: no top-p truncation on mask filler
MASK_TEMPERATURE = 1.0     # paper: "sampling directly with temperature 1"

# --- Throughput knobs, NOT paper hyperparameters ---
CHUNK_SIZE = 25            # mask-fill generation batch size, tuned for L40 + mt5-base
BUFFER_SIZE = 1            # matches official reference implementation's default
LL_BATCH_SIZE = 16         # log-likelihood scoring batch size, tuned for L40 + mGPT

# [CALIBRATION CHANGE] DECISION_EPSILON = 0.0 is no longer used. The threshold
# epsilon is now learned per source (train_dataset, train_llm) pair; see
# learn_threshold_from_training().

BASE_MODEL_NAME = "sberbank-ai/mGPT"          # deviation from paper, see docstring
MASK_MODEL_NAME = "google/mt5-base"           # deviation from paper, see docstring

os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(CALIB_ROOT, exist_ok=True)        # [CALIBRATION CHANGE]
os.makedirs(CACHE_ROOT, exist_ok=True)        # [CALIBRATION CHANGE]

pattern = re.compile(r"<extra_id_\d+>")

# -------------------- LOAD MODELS --------------------
def _load_causal_lm(name):
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=AUTOCAST_DTYPE if USE_AUTOCAST else torch.float32,
            attn_implementation="sdpa",
        )
    except (ValueError, TypeError):
        # Fall back if this transformers/model combo doesn't support sdpa kwarg
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


# -------------------- UTILS --------------------
def truncate_text(text):
    tokens = base_tokenizer.tokenize(text)
    tokens = tokens[:MAX_TOKENS]
    return base_tokenizer.convert_tokens_to_string(tokens)


def strip_newlines(text):
    return " ".join(text.split())


def get_ll_batch(texts, batch_size=LL_BATCH_SIZE):
    """
    Computes log p(x) for a list of texts under the base model, batched.
    Replaces the previous one-text-at-a-time get_ll() loop, which was the
    main bottleneck once N_PERTURB was corrected to the paper's value of
    100. Uses a manual per-sequence NLL computation (rather than the
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

        # Mean log-likelihood per token, matching -out.loss.item() semantics
        # from the original per-example implementation.
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
    Original text and all N_PERTURB perturbations are scored together in
    one batched call to get_ll_batch for speed.
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


def append_master(row):
    df = pd.DataFrame([row])
    if os.path.exists(MASTER_CSV):
        df.to_csv(MASTER_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(MASTER_CSV, index=False)


# ===================== [CALIBRATION CHANGE] everything below =====================

# -------------------- SCORING WITH DISK CACHE --------------------
def load_or_score(split, dataset, llm):
    """
    Returns a dataframe with an 'llr' column (z-score) for the given split
    ('train' or 'test') of (dataset, llm). Scores are computed once with the
    unchanged compute_llr() and cached, so every later in-domain / cross-domain
    run reuses the identical scores for that split.
    """
    root = TRAIN_ROOT if split == "train" else TEST_ROOT
    path = f"{root}/{dataset}/{dataset}_{split}_{llm}.csv"
    cache = f"{CACHE_ROOT}/{split}_{dataset}_{llm}.csv"

    if os.path.exists(cache):
        return pd.read_csv(cache)
    if not os.path.exists(path):
        print(f"  [skip] missing file: {path}")
        return None

    df = pd.read_csv(path).dropna(subset=["text"]).reset_index(drop=True)
    scores = []
    for text in tqdm(df["text"].tolist(), desc=f"Scoring {split} {dataset}-{llm}"):
        scores.append(compute_llr(text))
    df["llr"] = scores
    df.to_csv(cache, index=False)
    return df


# -------------------- THRESHOLD LEARNING --------------------
def learn_threshold_from_training(df_train):
    """
    Learns epsilon from labeled TRAIN scores by maximising Youden's J
    (TPR - FPR) on the ROC curve.

    Direction is fixed by the DetectGPT hypothesis (higher z => AI), so the
    AI class is the positive class and the raw score is used directly (no
    negation, unlike GPTZero where low perplexity => AI).

    Returns (threshold, youden_j, train_auroc). Decision rule: z >= threshold
    => AI (identical to how sklearn's roc_curve defines a threshold).
    """
    df = df_train.replace([np.inf, -np.inf], np.nan).dropna(subset=["llr"])

    y_pos = (df["label"].values == AI_LABEL).astype(int)
    scores = df["llr"].values.astype(float)

    if len(np.unique(y_pos)) < 2:
        raise ValueError("Training split must contain both AI and Human samples.")

    fpr, tpr, thresholds = roc_curve(y_pos, scores)
    J = tpr - fpr
    best_idx = int(np.argmax(J))
    train_auc = roc_auc_score(y_pos, scores)
    return float(thresholds[best_idx]), float(J[best_idx]), float(train_auc)


def classify_by_threshold(z, threshold):
    return AI_LABEL if z >= threshold else HUMAN_LABEL


# -------------------- INFERENCE (in-domain AND cross-domain) --------------------
def run_detectgpt_inference(train_dataset, train_llm, test_dataset, test_llm):
    """
    Calibrates epsilon on TRAIN data of (train_dataset, train_llm) and
    evaluates on TEST data of (test_dataset, test_llm).
      - in-domain:                  train == test
      - RQ2 (LLM cross-domain):     same dataset, different LLM
      - RQ3 (dataset cross-domain): same LLM, different dataset
    """
    is_in_domain = (train_dataset == test_dataset and train_llm == test_llm)
    kind = "IN-DOMAIN" if is_in_domain else "CROSS"
    print(f"\nDetectGPT {kind} | TRAIN: {train_dataset.upper()} + {train_llm.upper()} "
          f"-> TEST: {test_dataset.upper()} + {test_llm.upper()}")

    # ---------- TRAINING (threshold calibration on source) ----------
    df_train = load_or_score("train", train_dataset, train_llm)
    if df_train is None:
        return
    threshold, youden_j, train_auc = learn_threshold_from_training(df_train)
    print(f"Learned threshold: {threshold:.4f} | train Youden J={youden_j:.3f} | "
          f"train AUROC={train_auc:.3f}")
    if youden_j <= 0:
        print("  WARNING: Youden J <= 0 on training data. The score has no usable "
              "signal in the paper's direction (higher z => AI) for this source, "
              "so the calibrated threshold is not meaningful.")

    # ---------- TESTING (on target) ----------
    df_test = load_or_score("test", test_dataset, test_llm)
    if df_test is None:
        return

    valid = df_test.dropna(subset=["llr"]).copy()
    y_true = valid["label"].values
    y_score = valid["llr"].values.astype(float)
    y_pred = np.array([classify_by_threshold(z, threshold) for z in y_score])
    valid["predicted_label"] = y_pred

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=AI_LABEL)

    # Paper-aligned primary metric: AUROC on the raw score (threshold-free).
    try:
        auroc = roc_auc_score((y_true == AI_LABEL).astype(int), y_score)
    except Exception:
        auroc = float("nan")

    ai_rec = recall_score(y_true, y_pred, pos_label=AI_LABEL)
    human_rec = recall_score(y_true, y_pred, pos_label=HUMAN_LABEL)

    # ---------- SAVE ----------
    tag = (f"train_{train_dataset}_{train_llm}__test_{test_dataset}_{test_llm}")
    save_dir = f"{CALIB_ROOT}/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    valid.to_csv(f"{save_dir}/predictions.csv", index=False)

    result = {
        "train_dataset": train_dataset,
        "train_llm": train_llm,
        "test_dataset": test_dataset,
        "test_llm": test_llm,
        "setting": "in_domain" if is_in_domain else "cross",
        "threshold": threshold,
        "train_youden_j": youden_j,
        "train_auroc": train_auc,
        "accuracy": acc,
        "f1_score": f1,
        "auroc": auroc,
        f"AI_recall_label_{AI_LABEL}": ai_rec,
        f"Human_recall_label_{HUMAN_LABEL}": human_rec,
    }
    append_master(result)
    torch.cuda.empty_cache()

    print(f"Done | Acc={acc:.3f}, AUC={auroc:.3f}, AI Recall={ai_rec:.3f}, "
          f"Human Recall={human_rec:.3f}")


# -------------------- MAIN --------------------
if __name__ == "__main__":

    # ---- In-domain ----
    for dataset in DATASETS:
        for llm in LLMS:
            run_detectgpt_inference(dataset, llm, dataset, llm)

    # ---- RQ2: LLM cross-domain (dataset fixed, LLM changes) ----
    for dataset in DATASETS:
        for train_llm in LLMS:
            for test_llm in LLMS:
                if train_llm == test_llm:
                    continue
                run_detectgpt_inference(dataset, train_llm, dataset, test_llm)

    # ---- RQ3: Dataset cross-domain (LLM fixed, dataset changes) ----
    for llm in LLMS:
        for train_dataset in DATASETS:
            for test_dataset in DATASETS:
                if train_dataset == test_dataset:
                    continue
                run_detectgpt_inference(train_dataset, llm, test_dataset, llm)

    print("\nDetectGPT inference with ROC-calibrated thresholds completed.")