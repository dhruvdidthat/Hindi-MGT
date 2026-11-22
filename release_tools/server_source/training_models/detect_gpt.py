#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DetectGPT inference script (pure zero-shot).

=======================================================================
HYPERPARAMETERS vs. Mitchell et al. (2023) "DetectGPT: Zero-Shot
Machine-Generated Text Detection using Probability Curvature"
=======================================================================
See Section 5.1 "Hyperparameters" and Section 4 (Figure 8) of the paper.

  PCT_WORDS_MASKED = 0.15
      MATCHES PAPER. "Using BERT masked language modeling as inspiration,
      we use 15% as the mask rate." (Section 5.1). The previous value of
      0.25 was NOT from the paper; fixed here.

  SPAN_LENGTH = 2
      MATCHES PAPER. "We performed a small sweep over masked span lengths
      of {2, 5, 10} on a held-out set of XSum data, finding 2 to perform
      best. We use these settings for all experiments, without re-tuning."

  N_PERTURB = 100
      MATCHES PAPER. "We approximate the expectation in Eq. 1 with 100
      samples from T5" (Section 4). Figure 8 shows detection AUROC keeps
      improving as perturbations increase and converges around 100. The
      previous value of 3 was NOT from the paper (far too low for a
      reliable curvature estimate) and is fixed here. NOTE: this is the
      single biggest cost driver in the script; batching (below) is what
      makes 100 perturbations practical on an L40.

  MASK_TOP_P = 1.0, MASK_TEMPERATURE = 1.0
      MATCHES PAPER. "We do not tune the hyperparameters for the
      mask-filling model, sampling directly with temperature 1"
      (Section 5.1) - i.e. no nucleus/top-p truncation on the mask
      filler. (The top-p=0.96 / top-k=40 settings in Table 3 of the
      paper are for decoding the *source* model's generations in a
      separate robustness experiment, not for the mask-filling
      perturbation function - do not confuse the two.)

  MAX_TOKENS = 512
      NOT a paper hyperparameter. The paper does not fix a 512-token cap;
      it works with XSum/SQuAD/WritingPrompts passages of natural length
      and prompts with the first 30 tokens when generating machine text.
      Kept here as a practical truncation guard for arbitrary input text;
      state this as an implementation choice in your methods section, not
      as something drawn from the paper.

  CHUNK_SIZE (mask-fill generation batch size), BUFFER_SIZE = 1,
  LL_BATCH_SIZE (log-likelihood scoring batch size)
      NOT paper hyperparameters. These are throughput/batching knobs
      (BUFFER_SIZE=1 matches the official reference implementation's
      default spacing between masked spans, but is not discussed as a
      tunable "hyperparameter" in the paper text). Tuned here for L40
      throughput, not for paper fidelity - see SPEED NOTES below.

  Mask-filling model: google/mt5-base
      DEVIATION FROM PAPER. The paper uses T5-3B for its main (English)
      experiments and mT5-3B for the multilingual WMT experiment. This
      script uses mt5-base (much smaller) for speed/memory reasons given
      Hindi-language text. If you need closer paper fidelity, swap to
      "google/mt5-xl" (~3B) at a significant speed/memory cost.

  Source model: sberbank-ai/mGPT
      DEVIATION FROM PAPER. The paper evaluates GPT-2, OPT-2.7, GPT-Neo,
      GPT-J, GPT-NeoX, GPT-3, and Jurassic-2 Jumbo - none of which are
      appropriate for Hindi text. mGPT is used here as a multilingual
      substitute; this is a deliberate adaptation of the paper's method
      to a new language setting, not an error, but should be stated as
      such in your methods write-up.

=======================================================================
SPEED NOTES (target: NVIDIA L40)
=======================================================================
- log p(x) for the original text AND all N_PERTURB perturbations are
  now scored in a single batched forward pass (chunked by LL_BATCH_SIZE)
  instead of one text at a time. This is the main speedup, since it's
  what makes N_PERTURB=100 (the paper's value) tractable.
- bf16 autocast is used for both models. The L40 (Ada Lovelace) has
  native bf16 tensor core support, so this is close to "free" speed with
  negligible precision loss for log-likelihood scoring.
- TF32 matmuls and cuDNN benchmarking are enabled.
- torch.inference_mode() replaces torch.no_grad() (slightly faster, no
  autograd bookkeeping at all).
- Mask-fill generation batch size (CHUNK_SIZE) is raised since mT5-base
  is small and an L40 has ample headroom; if you switch to a larger
  mask-filling model, lower CHUNK_SIZE accordingly.
- All of the above only change *how fast* the numbers are computed, not
  *what* is computed - the perturbation/scoring logic is otherwise
  identical to the paper's Algorithm 1.
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
RESULTS_ROOT = "detect_gpt_2/inference_results"
MASTER_CSV = os.path.join(RESULTS_ROOT, "detect_gpt_all_results.csv")

MAX_TOKENS = 512          # implementation choice, not a paper hyperparameter (see docstring)

# --- Paper hyperparameters (Section 5.1) ---
N_PERTURB = 100            # paper: "10 samples from T5"
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