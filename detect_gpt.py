#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DetectGPT inference script (pure zero-shot).

Label convention:
    0 = Human-written text
    1 = AI-generated / machine-generated text

This is the raw label convention used by the DetectGPT implementation.
"""

import os
import re
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
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

TEST_ROOT = "testing_data"
RESULTS_ROOT = "detect_gpt/inference_results"
MASTER_CSV = os.path.join(RESULTS_ROOT, "detect_gpt_all_results.csv")

MAX_TOKENS = 512          # implementation choice, not a paper hyperparameter (see docstring)

# --- Paper hyperparameters (Section 5.1) ---
N_PERTURB = 100            # paper: "10 samples from T5";
SPAN_LENGTH = 2            # paper: swept {2,5,10} on held-out XSum, 2 best
PCT_WORDS_MASKED = 0.15    # paper: "15% as the mask rate"
MASK_TOP_P = 1.0           # paper: no top-p truncation on mask filler
MASK_TEMPERATURE = 1.0     # paper: "sampling directly with temperature 1"

# --- Throughput knobs, NOT paper hyperparameters ---
CHUNK_SIZE = 25            # mask-fill generation batch size, tuned for L40 + mt5-base
BUFFER_SIZE = 1            # matches official reference implementation's default
LL_BATCH_SIZE = 16         # log-likelihood scoring batch size, tuned for L40 + mGPT

# Zero-shot decision threshold on the normalized perturbation discrepancy.
# score > DECISION_EPSILON  =>  predicted AI / machine-generated
# Fixed a priori at 0, never fit on any data - see prior correction notes.
DECISION_EPSILON = 0.0

BASE_MODEL_NAME = "sberbank-ai/mGPT"          # deviation from paper, see docstring
MASK_MODEL_NAME = "google/mt5-base"           # deviation from paper, see docstring

os.makedirs(RESULTS_ROOT, exist_ok=True)

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


# -------------------- INFERENCE --------------------
for dataset in DATASETS:
    for llm in LLMS:

        print(f"\nDetectGPT Inference | {dataset.upper()} + {llm.upper()}")

        test_path = f"{TEST_ROOT}/{dataset}/{dataset}_test_{llm}.csv"
        if not os.path.exists(test_path):
            continue

        df = pd.read_csv(test_path).dropna(subset=["text"])
        llr_scores = []

        for text in tqdm(df["text"].tolist()):
            llr_scores.append(compute_llr(text))

        df["llr"] = llr_scores

        valid = df.dropna(subset=["llr"])
        y_true = valid["label"].tolist()      # convention: 1 = AI, 0 = Human
        y_score = valid["llr"].tolist()        # higher score => more likely AI

        # Fixed, leakage-free zero-shot decision rule (epsilon = 0, never fit
        # on any data - see prior correction notes for why median/Youden
        # thresholding was removed).
        y_pred = [1 if x > DECISION_EPSILON else 0 for x in y_score]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Paper-aligned primary metric: AUROC on the raw score, no negation.
        try:
            auroc = roc_auc_score(y_true, y_score)
        except Exception:
            auroc = float("nan")

        # 1 = AI, 0 = Human
        ai_rec = recall_score(y_true, y_pred, pos_label=1)
        human_rec = recall_score(y_true, y_pred, pos_label=0)

        save_dir = f"{RESULTS_ROOT}/{dataset}_{llm}"
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(f"{save_dir}/predictions.csv", index=False)

        result = {
            "train_dataset": "NA",
            "train_llm": "NA",
            "test_dataset": dataset,
            "test_llm": llm,
            "accuracy": acc,
            "f1_score": f1,
            "auroc": auroc,
            "AI_recall_label_1": ai_rec,
            "Human_recall_label_0": human_rec,
        }

        append_master(result)
        torch.cuda.empty_cache()

print("\nDetectGPT inference complete.")