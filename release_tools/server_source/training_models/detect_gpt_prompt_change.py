#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DetectGPT prompt-change inference (RQ5 / Table 7).

=======================================================================
WHAT THIS SCRIPT DOES
=======================================================================
Evaluates DetectGPT on the PROMPT-SHIFTED test sets (MGT regenerated with
the new prompt) using the thresholds that were ALREADY calibrated on the
original training data. Nothing is re-calibrated or re-trained here.

  - Scores z(x): computed with the UNCHANGED DetectGPT pipeline
    (same models, perturbation, hyperparameters, compute_llr).
  - Threshold epsilon: read from the existing calibrated results CSV
    (the in-domain rows hold one threshold per (train_dataset, train_llm)
    pair, learned from that pair's TRAIN split by Youden's J).
  - Decision rule (unchanged): z >= epsilon  =>  AI.

Per Section 5.6 of the paper, only Gemini 1.5 and Qwen3 are used
(Llama3 excluded). Every (trained pair) x (new-prompt test set) combination
is evaluated; the 'setting' column marks the Table 7 groups:
    in_domain       same dataset, same LLM
    llm_cross       same dataset, different LLM
    dataset_cross   same LLM, different dataset
    both_shift      dataset AND LLM differ (not in Table 7)

Test files:   new_prompt_data/{dataset}_{llm}_new_prompt.csv   (text, label)
Thresholds:   CALIBRATION_CSV (see config)

Label handling: outputs use the canonical convention (1 = AI, 0 = Human).
NEW_PROMPT_AI_LABEL says which raw label means AI in the new-prompt CSVs.
Because only the MGT is regenerated, the human texts should be identical to
those in the original test sets; the script uses that to verify the label
convention automatically and STOPS with an error if it looks inverted.

Each evaluation writes:
    <NEW_PROMPT_ROOT>/<train_ds>_<train_llm>_tested_on_<test_ds>_<test_llm>/
        metrics.csv
        predictions.csv
and one master file rebuilt from all metrics.csv files:
    detect_gpt_calibrated_new_prompt_results.csv

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
PROMPT_LLMS = ["gemini", "qwen3"]       # paper Sec. 5.6: Llama3 excluded from RQ5

TEST_ROOT = "testing_data"              # original test sets (used only for the label check)
NEW_PROMPT_DIR = "../prompt_change/new_prompt_data"      # prompt-shifted test sets
RESULTS_ROOT = "detect_gpt_all_domain/inference_results"
CALIB_ROOT = os.path.join(RESULTS_ROOT, "calibrated")
CACHE_ROOT = os.path.join(CALIB_ROOT, "score_cache")

# Existing calibrated results (source of the pretrained thresholds).
CALIBRATION_CSV = os.path.join(CALIB_ROOT, "detect_gpt_calibrated_all_results.csv")

NEW_PROMPT_ROOT = os.path.join(CALIB_ROOT, "new_prompt")
MASTER_RESULTS_CSV = os.path.join(
    NEW_PROMPT_ROOT, "detect_gpt_calibrated_new_prompt_results.csv"
)

# Label convention of the ORIGINAL testing_data CSVs for DetectGPT (README):
# 1 = AI, 0 = Human. Used only for the automatic label check below.
AI_LABEL = 1
HUMAN_LABEL = 1 - AI_LABEL

# Raw label meaning "AI" in the NEW-PROMPT CSVs (DetectGPT README convention).
# Verified automatically against the original human texts; see check_label_convention().
NEW_PROMPT_AI_LABEL = 1

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
os.makedirs(NEW_PROMPT_ROOT, exist_ok=True)

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


# ===================== PROMPT-CHANGE INFERENCE LOGIC =====================

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
def check_label_convention(df_new, dataset, llm):
    """
    Only the MGT is regenerated for the new prompt, so the human texts in the
    new-prompt CSV should be the same texts as the human texts in the original
    test set. If those texts carry the label meaning 'AI' under
    NEW_PROMPT_AI_LABEL, the label convention is inverted -> stop (we do not
    silently flip labels).
    """
    orig_path = f"{TEST_ROOT}/{dataset}/{dataset}_test_{llm}.csv"
    if not os.path.exists(orig_path):
        print(f"  [label check skipped] original test file not found: {orig_path}")
        return

    df_old = pd.read_csv(orig_path).dropna(subset=["text"])
    human_old = set(
        strip_newlines(str(t))
        for t in df_old.loc[df_old["label"] == HUMAN_LABEL, "text"]
    )

    new_norm = df_new["text"].map(lambda t: strip_newlines(str(t)))
    mask = new_norm.isin(human_old)
    n_match = int(mask.sum())

    if n_match == 0:
        print(f"  [label check inconclusive] no human texts of {dataset}+{llm} "
              f"found verbatim in the new-prompt file.")
        return

    frac_marked_ai = float((df_new.loc[mask, "label"] == NEW_PROMPT_AI_LABEL).mean())
    if frac_marked_ai > 0.5:
        raise ValueError(
            f"Label convention looks INVERTED for {dataset}+{llm}: "
            f"{frac_marked_ai:.0%} of {n_match} known human texts are labelled as AI "
            f"under NEW_PROMPT_AI_LABEL={NEW_PROMPT_AI_LABEL}. "
            f"Set NEW_PROMPT_AI_LABEL = {1 - NEW_PROMPT_AI_LABEL} and rerun."
        )
    if frac_marked_ai > 0:
        print(f"  [label check warning] {frac_marked_ai:.1%} of {n_match} matched "
              f"human texts are labelled AI for {dataset}+{llm}.")
    else:
        print(f"  [label check OK] {n_match} human texts match the original test set "
              f"and are labelled Human.")


# -------------------- SCORING WITH DISK CACHE --------------------
def load_or_score_new_prompt(dataset, llm):
    """
    Returns the new-prompt test set of (dataset, llm) with an 'llr' column
    (unchanged z-score). Scored once and cached, so reruns reuse the same scores.
    """
    path = f"{NEW_PROMPT_DIR}/{dataset}_{llm}_new_prompt.csv"
    cache = f"{CACHE_ROOT}/newprompt_{dataset}_{llm}.csv"

    if os.path.exists(cache):
        df = pd.read_csv(cache)
        check_label_convention(df, dataset, llm)
        return df

    if not os.path.exists(path):
        print(f"  [skip] new-prompt file not found for {dataset}+{llm}: {path}")
        return None

    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    check_label_convention(df, dataset, llm)   # fail fast, before the expensive scoring

    scores = []
    for text in tqdm(df["text"].astype(str).tolist(),
                     desc=f"Scoring new-prompt {dataset}-{llm}"):
        scores.append(compute_llr(text))
    df["llr"] = scores
    df.to_csv(cache, index=False)
    return df


def classify_by_threshold(z, threshold):
    return 1 if z >= threshold else 0      # canonical: 1 = AI, 0 = Human


def setting_name(train_dataset, train_llm, test_dataset, test_llm):
    same_ds = train_dataset == test_dataset
    same_llm = train_llm == test_llm
    if same_ds and same_llm:
        return "in_domain"
    if same_ds:
        return "llm_cross"
    if same_llm:
        return "dataset_cross"
    return "both_shift"


# -------------------- ONE EVALUATION --------------------
def evaluate_one_case(train_dataset, train_llm, test_dataset, test_llm,
                      threshold, df_test):
    print("\n" + "=" * 75)
    print(f"TRAIN(threshold)={train_dataset}+{train_llm} | "
          f"TEST(new_prompt)={test_dataset}+{test_llm} | eps={threshold:.6f}")
    print("=" * 75)

    valid = df_test.dropna(subset=["llr"]).copy()

    # Canonical labels: 1 = AI, 0 = Human
    true_labels = (valid["label"].values == NEW_PROMPT_AI_LABEL).astype(int)
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
        NEW_PROMPT_ROOT,
        f"{train_dataset}_{train_llm}_tested_on_{test_dataset}_{test_llm}",
    )
    os.makedirs(result_dir, exist_ok=True)

    metrics_df = pd.DataFrame([{
        "Train Dataset": train_dataset,
        "Train LLM": train_llm,
        "Test Dataset": test_dataset,
        "Test LLM": test_llm,
        "Setting": setting_name(train_dataset, train_llm, test_dataset, test_llm),
        "Threshold": threshold,
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

    combos = list(product(DATASETS, PROMPT_LLMS))

    # ---- Score each new-prompt test set once ----
    print("\n" + "=" * 75)
    print("Loading / scoring new-prompt test sets")
    print("=" * 75)

    scored = {}
    for test_dataset, test_llm in combos:
        df = load_or_score_new_prompt(test_dataset, test_llm)
        if df is not None:
            scored[(test_dataset, test_llm)] = df
            print(f"Ready {test_dataset}+{test_llm}: {len(df)} texts")
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nReady {len(scored)}/{len(combos)} new-prompt test sets.")

    # ---- Trained thresholds x new-prompt test sets ----
    total_cases = len(combos) ** 2
    case_num = 0

    for train_dataset, train_llm in combos:
        if (train_dataset, train_llm) not in thresholds:
            print(f"\nSkipping train {train_dataset}+{train_llm}: "
                  f"no calibrated threshold in {CALIBRATION_CSV}")
            continue
        threshold = thresholds[(train_dataset, train_llm)]

        for test_dataset, test_llm in combos:
            case_num += 1
            print("\n" + "#" * 80)
            print(f"[{case_num}/{total_cases}] TRAIN={train_dataset}+{train_llm} "
                  f"TEST={test_dataset}+{test_llm}")
            print("#" * 80)

            df_test = scored.get((test_dataset, test_llm))
            if df_test is None:
                print(f"Skipping: no new-prompt data for {test_dataset}+{test_llm}")
                continue

            try:
                evaluate_one_case(train_dataset, train_llm,
                                  test_dataset, test_llm,
                                  threshold, df_test)
            except Exception as exc:
                print(f"FAILED case {train_dataset}+{train_llm} -> "
                      f"{test_dataset}+{test_llm}: {type(exc).__name__}: {exc}")

    # ---- Rebuild master CSV from all metrics.csv files (no duplicate rows on rerun) ----
    all_metrics = []
    for train_dataset, train_llm in combos:
        for test_dataset, test_llm in combos:
            metrics_path = os.path.join(
                NEW_PROMPT_ROOT,
                f"{train_dataset}_{train_llm}_tested_on_{test_dataset}_{test_llm}",
                "metrics.csv",
            )
            if os.path.exists(metrics_path):
                all_metrics.append(pd.read_csv(metrics_path))

    if all_metrics:
        master_df = pd.concat(all_metrics, ignore_index=True)
        master_df.to_csv(MASTER_RESULTS_CSV, index=False)
        print("\n" + "=" * 75)
        print("All new-prompt evaluations completed.")
        print(f"Evaluations written: {len(master_df)}/{total_cases}")
        print(f"Master results:\n  {MASTER_RESULTS_CSV}")
        print("=" * 75)
    else:
        print("\nNo evaluation results were generated.")


if __name__ == "__main__":
    main()