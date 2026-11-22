#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast-DetectGPT for Hindi - zero-shot, BLACK-BOX SURROGATE-MODEL setting
(mGPT sampling / mGPT scoring) evaluated on Gemini, Llama 3 and Qwen3 text.

Setting
-------
The texts under test are produced by Gemini / Llama 3 / Qwen3. mGPT is NOT
their source model, so this is the paper's *black-box* (surrogate-model)
setting, even though the sampling and scoring model are the same checkpoint
internally. It is NOT the source-model white-box setting of the paper.

Provenance (read before reusing)
--------------------------------
The criterion functions `get_samples`, `get_likelihood`,
`get_sampling_discrepancy` and `get_sampling_discrepancy_analytic` are
adapted from the official Fast-DetectGPT repository
(https://github.com/baoguangsheng/fast-detect-gpt, scripts/fast_detect_gpt.py),
Copyright (c) 2023 Bao Guangsheng, MIT License. The MIT notice is reproduced
at the bottom of this docstring and in the accompanying LICENSE file.
Paper: Bao et al., "Fast-DetectGPT: Efficient Zero-Shot Detection of
Machine-Generated Text via Conditional Probability Curvature", ICLR 2024.

Three layers (kept separate on purpose)
---------------------------------------
1. Official method : the criterion and its two estimators (analytic /
                      Monte Carlo with 10,000 samples).
2. Hindi adaptation : mGPT, 512-token project cap, CSV loader, output
                      schema, label convention, threshold policy.
3. Engineering      : inference_mode, SDPA attention, FP32 reductions,
                      per-row status tracking, run metadata.
                      None of these changes the target statistic.

Estimators
----------
  analytic     : exact expectation/variance over the vocabulary
                 (Appendix B). Deterministic. Default.
  monte_carlo  : official default path, N=10,000 independent token
                 alternatives per position, seeded per row.

Threshold policy
----------------
AUROC on the raw score (higher => AI) is the primary metric. Accuracy / F1 /
recalls are computed at a FIXED, pre-declared epsilon (default 0.0). They are
exploratory diagnostics only: epsilon=0 is not a paper-guaranteed operating
point and nothing is ever fitted on test labels.

Input CSV: columns `text`, `label`. Label convention is explicit (--ai-label,
default 1 = AI, 0 = Human) and must be verified against your files; the
pre-flight step prints label counts and one example per label.

-----------------------------------------------------------------------
MIT License - Copyright (c) 2023 Bao Guangsheng (upstream Fast-DetectGPT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-----------------------------------------------------------------------

Usage
-----
  smoke test : python fast_detect_gpt_hindi.py --smoke-test
  full run   : python fast_detect_gpt_hindi.py --estimator analytic
               python fast_detect_gpt_hindi.py --estimator monte_carlo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project configuration (defaults; all overridable from the command line)
# ---------------------------------------------------------------------------
DATASETS = ["hindisumm", "xquad"]
LLMS = ["gemini", "llama3", "qwen3"]
TEST_ROOT = "testing_data"
RESULTS_ROOT = "fast_detect_gpt/inference_results"
MASTER_NAME = "fast_detect_gpt_all_results.csv"

# One exact repo id, used consistently. Use the SAME id/revision in the Hindi
# DetectGPT runs (that script used the alias "sberbank-ai/mGPT").
SAMPLING_MODEL_NAME = "ai-forever/mGPT"
SCORING_MODEL_NAME = "ai-forever/mGPT"

MAX_TOKENS = 512          # project cap, NOT a Fast-DetectGPT hyperparameter
N_SAMPLES = 10_000        # official default for the Monte Carlo estimator
AI_LABEL = 1              # 1 = AI, 0 = Human (verify against your CSVs)
THRESHOLD = 0.0           # fixed exploratory epsilon; never fit on test data
SEED = 42

# Variance floor. sigma_tilde^2 is a SUM over tokens of per-token variances of
# log p under q; for any real passage it is >> 1. At or below this value the
# statistic is a 0/0 artefact (e.g. one-hot sampler or uniform scorer) and the
# row is reported as degenerate instead of receiving a fabricated score.
DEGENERATE_VARIANCE = 1e-6
VARIANCE_CLAMP = 1e-12    # roundoff guard applied before the square root


# ===========================================================================
# 1. CRITERION (official method) - independent of CSVs / dataframes
# ===========================================================================
class CriterionInputError(ValueError):
    """Raised for malformed logits/labels (a programming error, not data)."""


def check_criterion_inputs(logits_ref, logits_score, labels) -> None:
    for name, t in (("logits_ref", logits_ref), ("logits_score", logits_score)):
        if t.ndim != 3 or t.shape[0] != 1:
            raise CriterionInputError(
                f"{name} must have shape [1, T, V]; got {tuple(t.shape)}"
            )
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise CriterionInputError(
            f"labels must have shape [1, T]; got {tuple(labels.shape)}"
        )
    if logits_ref.shape != logits_score.shape:
        # The official code silently crops both to min(V). That does not make
        # token ids comparable across tokenizers, so we reject instead.
        raise CriterionInputError(
            "logits_ref and logits_score shapes differ "
            f"({tuple(logits_ref.shape)} vs {tuple(logits_score.shape)}); "
            "sampler and scorer must share sequence length and vocabulary."
        )
    if tuple(labels.shape) != tuple(logits_score.shape[:2]):
        raise CriterionInputError(
            f"labels {tuple(labels.shape)} do not align with logits "
            f"{tuple(logits_score.shape[:2])}"
        )
    if labels.shape[1] < 1:
        raise CriterionInputError("Need at least one scored token (T >= 1).")
    if labels.dtype not in (torch.int32, torch.int64):
        raise CriterionInputError(f"labels must be integer ids; got {labels.dtype}")
    vocab = logits_score.shape[-1]
    if int(labels.min()) < 0 or int(labels.max()) >= vocab:
        raise CriterionInputError("A label id is outside the vocabulary range.")
    if not torch.isfinite(logits_ref).all() or not torch.isfinite(logits_score).all():
        raise CriterionInputError("Non-finite values in logits.")


def shift_for_next_token(full_logits, input_ids):
    """Causal alignment used by the official driver.

    Position j of the returned logits predicts the token returned at position
    j of labels:  logits[:, :-1]  <->  input_ids[:, 1:].
    """
    return full_logits[:, :-1, :], input_ids[:, 1:]


# ---- Analytic estimator (Appendix B) -------------------------------------
def _analytic_core(logits_ref, logits_score, labels, valid_mask, reduction_dtype):
    """Per-row analytic criterion over [B, T, V] with an explicit valid-token
    mask [B, T] (True = real label position). Returns a [B] float tensor with
    NaN for degenerate / non-finite rows. Reductions run in `reduction_dtype`.
    """
    lr = logits_ref.to(reduction_dtype)
    ls = logits_score.to(reduction_dtype)
    lprobs_score = torch.log_softmax(ls, dim=-1)       # log p_theta(. | x_<j)
    probs_ref = torch.softmax(lr, dim=-1)              # q_phi(. | x_<j)

    log_likelihood = lprobs_score.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)  # mu_j  (Eq. 6)
    # Variance in centred form, sum_v q (l - mu)^2. Mathematically identical to
    # the official  sum_v q l^2 - mu^2  (Eq. 7) but free of catastrophic
    # cancellation; equivalence is covered by the unit tests.
    var_ref = (probs_ref * (lprobs_score - mean_ref.unsqueeze(-1)).square()).sum(dim=-1)

    m = valid_mask.to(reduction_dtype)
    # where() (not multiply) so garbage/inf at padded positions cannot leak
    # in via 0 * inf = nan.
    zero = torch.zeros_like(log_likelihood)
    numerator = torch.where(valid_mask, log_likelihood - mean_ref, zero).sum(dim=-1)
    variance_sum = torch.where(valid_mask, var_ref, zero).sum(dim=-1)

    bad = ~torch.isfinite(numerator) | ~torch.isfinite(variance_sum) \
        | (variance_sum <= DEGENERATE_VARIANCE)
    denom = variance_sum.clamp_min(VARIANCE_CLAMP).sqrt()
    score = numerator / denom
    return torch.where(bad, torch.full_like(score, float("nan")), score)


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels,
                                      reduction_dtype=torch.float32) -> float:
    """Official analytical criterion (one passage, batch size 1).

    d = [sum_j log p(x_j|x_<j) - sum_j mu_j] / sqrt(sum_j var_j)
    Returns NaN for degenerate variance (never a made-up value).
    """
    check_criterion_inputs(logits_ref, logits_score, labels)
    mask = torch.ones_like(labels, dtype=torch.bool)
    return float(_analytic_core(logits_ref, logits_score, labels, mask,
                                reduction_dtype)[0].item())


def get_sampling_discrepancy_analytic_batched(logits_ref, logits_score, labels,
                                              valid_mask,
                                              reduction_dtype=torch.float32):
    """Masked batched variant (padding positions never enter any sum).
    Provided for validation of the no-padding-contamination requirement; the
    experiment driver scores one passage at a time, exactly as upstream."""
    if logits_ref.ndim != 3 or logits_ref.shape != logits_score.shape:
        raise CriterionInputError("batched logits must be [B, T, V] and equal.")
    if labels.shape != logits_score.shape[:2] or valid_mask.shape != labels.shape:
        raise CriterionInputError("labels / valid_mask must be [B, T].")
    return _analytic_core(logits_ref, logits_score, labels, valid_mask, reduction_dtype)


# ---- Monte Carlo estimator (official default path) -----------------------
def get_samples(logits, labels, nsamples=N_SAMPLES):
    """Independent categorical alternatives at every position, all conditioned
    on the ORIGINAL prefix. Returns [1, T, nsamples]. (Official code.)"""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples


def get_likelihood(logits, labels):
    """Mean per-token log probability under the scoring model. (Official.)"""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)


def get_sampling_discrepancy(logits_ref, logits_score, labels,
                             nsamples=N_SAMPLES,
                             reduction_dtype=torch.float32) -> float:
    """Official Monte Carlo criterion. Randomness comes from the global torch
    RNG; the driver reseeds it per row for reproducibility."""
    check_criterion_inputs(logits_ref, logits_score, labels)
    if nsamples < 2:
        raise CriterionInputError("nsamples must be >= 2 to estimate a std.")
    lr = logits_ref.to(reduction_dtype)
    ls = logits_score.to(reduction_dtype)

    samples = get_samples(lr, labels, nsamples)
    log_likelihood_x = get_likelihood(ls, labels)
    log_likelihood_x_tilde = get_likelihood(ls, samples)

    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)          # unbiased, as official
    T = labels.shape[1]
    # Same degeneracy rule as the analytic path: sigma here is on the
    # mean-per-token scale, so sigma*T ~ sqrt(sum of variances).
    if (not torch.isfinite(sigma_tilde).all()
            or float(sigma_tilde.item()) * T <= math.sqrt(DEGENERATE_VARIANCE)):
        return float("nan")
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    value = float(discrepancy.item())
    return value if math.isfinite(value) else float("nan")


# ===========================================================================
# 2. MODELS AND PER-TEXT SCORING
# ===========================================================================
def check_tokenizer_compat(tok_a, tok_b) -> None:
    """Different sampler/scorer are accepted only if the FULL token-id ->
    token mapping and special ids are identical (vocab-size equality is not
    enough)."""
    if tok_a.get_vocab() != tok_b.get_vocab():
        raise ValueError("Sampling and scoring tokenizers have different "
                         "vocabularies; this pair is rejected.")
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        if getattr(tok_a, attr, None) != getattr(tok_b, attr, None):
            raise ValueError(f"Tokenizers disagree on {attr}; pair rejected.")


@dataclass
class Detector:
    scoring_model: object
    sampling_model: object
    scoring_tokenizer: object
    sampling_tokenizer: object
    same_model: bool
    device: torch.device
    max_tokens: int
    estimator: str            # "analytic" | "monte_carlo"
    n_samples: int
    reduction_dtype: torch.dtype

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _next_token_logits(self, model, input_ids):
        attention_mask = torch.ones_like(input_ids)     # single text, no padding
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False)
        return out.logits[:, :-1, :]

    def score(self, text, seed: Optional[int] = None) -> dict:
        """Score ONE passage. Always returns a dict (never raises for data
        problems) so every input row maps to exactly one output row."""
        res = dict(score=float("nan"), status="ok", n_tokens_full=0,
                   n_tokens_used=0, truncated=False, inference_seconds=0.0)
        if not isinstance(text, str) or not text.strip():
            res["status"] = "empty_text"
            return res
        try:
            # Official defaults: no whitespace/Unicode edits, tokenizer default
            # special-token behaviour. Full length is counted separately so
            # truncation is reported, not hidden.
            full = self.scoring_tokenizer(text, return_token_type_ids=False)
            res["n_tokens_full"] = len(full["input_ids"])
            enc = self.scoring_tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=self.max_tokens, return_token_type_ids=False,
            ).to(self.device)
            res["n_tokens_used"] = int(enc.input_ids.shape[1])
            res["truncated"] = res["n_tokens_full"] > res["n_tokens_used"]
            if res["n_tokens_used"] < 2:
                res["status"] = "too_short"
                return res
            labels = enc.input_ids[:, 1:]

            t0 = time.perf_counter()
            self._sync()
            logits_score = self._next_token_logits(self.scoring_model, enc.input_ids)
            if self.same_model:
                logits_ref = logits_score
            else:
                enc_ref = self.sampling_tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=self.max_tokens, return_token_type_ids=False,
                ).to(self.device)
                if not torch.equal(enc_ref.input_ids[:, 1:], labels):
                    res["status"] = "tokenizer_mismatch"
                    return res
                logits_ref = self._next_token_logits(self.sampling_model, enc_ref.input_ids)

            if self.estimator == "analytic":
                value = get_sampling_discrepancy_analytic(
                    logits_ref, logits_score, labels, self.reduction_dtype)
            else:
                if seed is not None:
                    torch.manual_seed(seed)
                value = get_sampling_discrepancy(
                    logits_ref, logits_score, labels, self.n_samples,
                    self.reduction_dtype)
            self._sync()
            res["inference_seconds"] = time.perf_counter() - t0

            if math.isfinite(value):
                res["score"] = value
            else:
                res["status"] = "degenerate_or_nonfinite"
        except torch.cuda.OutOfMemoryError:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            res["status"] = "oom"
        except Exception as exc:                         # keep row alignment
            res["status"] = f"error:{type(exc).__name__}"
            res["error_message"] = str(exc)[:200]
        return res


def _from_pretrained_causal_lm(name, dtype, revision):
    import transformers
    last = None
    for key in ("dtype", "torch_dtype"):                # name differs by version
        for attn in ("sdpa", None):
            kwargs = {key: dtype}
            if attn:
                kwargs["attn_implementation"] = attn
            if revision:
                kwargs["revision"] = revision
            try:
                model = transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs)
            except (TypeError, ValueError) as exc:
                last = exc
                continue
            # Older versions swallow unknown kwargs into the config; verify.
            if next(model.parameters()).dtype == dtype:
                return model, (attn or "default")
            del model
    raise RuntimeError(f"Could not load {name} in {dtype}: {last}")


def load_detector(args, device):
    import transformers
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    t0 = time.perf_counter()

    def load(name, revision):
        model, attn = _from_pretrained_causal_lm(name, dtype, revision)
        model.to(device).eval()
        tok = transformers.AutoTokenizer.from_pretrained(name, revision=revision) \
            if revision else transformers.AutoTokenizer.from_pretrained(name)
        return model, tok, attn

    s_model, s_tok, attn = load(args.scoring_model, args.scoring_revision)
    same = (args.sampling_model == args.scoring_model
            and args.sampling_revision == args.scoring_revision)
    if same:
        r_model, r_tok = s_model, s_tok
    else:
        r_model, r_tok, _ = load(args.sampling_model, args.sampling_revision)
        check_tokenizer_compat(s_tok, r_tok)
        if r_model.config.vocab_size != s_model.config.vocab_size:
            raise ValueError("Sampler/scorer logit dimensions differ; rejected.")

    probe = "नमस्ते दुनिया"
    with_sp = len(s_tok(probe)["input_ids"])
    no_sp = len(s_tok(probe, add_special_tokens=False)["input_ids"])
    info = dict(
        scoring_model=args.scoring_model,
        scoring_model_commit=getattr(s_model.config, "_commit_hash", None),
        sampling_model=args.sampling_model,
        sampling_model_commit=getattr(r_model.config, "_commit_hash", None),
        model_dtype=args.dtype, reduction_dtype="fp32",
        attention_implementation=attn,
        tokenizer_class=type(s_tok).__name__, tokenizer_is_fast=s_tok.is_fast,
        tokenizer_len=len(s_tok), model_vocab_size=s_model.config.vocab_size,
        bos_token_id=s_tok.bos_token_id, eos_token_id=s_tok.eos_token_id,
        pad_token_id=s_tok.pad_token_id,
        add_bos_token=getattr(s_tok, "add_bos_token", None),
        special_tokens_added_by_default=with_sp - no_sp,
        padding_side=s_tok.padding_side,
        model_load_seconds=time.perf_counter() - t0,
    )
    det = Detector(s_model, r_model, s_tok, r_tok, same, device, args.max_tokens,
                   args.estimator, args.n_samples, torch.float32)
    return det, info


# ===========================================================================
# 3. METRICS / EVALUATION (no model code)
# ===========================================================================
def compute_metrics(y_true, scores, ai_label: int, threshold: float,
                    n_boot: int = 1000, seed: int = 0) -> dict:
    """AUROC uses raw scores with AI as the positive class regardless of the
    0/1 convention. Hard-label metrics are at the FIXED `threshold`."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    y_ai = (y_true == ai_label).astype(int)
    n_ai, n_h = int(y_ai.sum()), int((1 - y_ai).sum())
    pred = (scores > threshold).astype(int)
    tp = int(((pred == 1) & (y_ai == 1)).sum()); fn = int(((pred == 0) & (y_ai == 1)).sum())
    fp = int(((pred == 1) & (y_ai == 0)).sum()); tn = int(((pred == 0) & (y_ai == 0)).sum())
    nan = float("nan")
    div = lambda a, b: a / b if b else nan
    out = dict(
        n_scored=len(y_true), n_ai=n_ai, n_human=n_h,
        accuracy=div(tp + tn, len(y_true)),
        f1_score=div(2 * tp, 2 * tp + fp + fn),
        precision=div(tp, tp + fp),
        ai_recall=div(tp, tp + fn), human_recall=div(tn, tn + fp),
        false_positive_rate=div(fp, fp + tn),
        tp=tp, fp=fp, tn=tn, fn=fn,
        auroc=nan, auroc_ci_low=nan, auroc_ci_high=nan,
    )
    if n_ai and n_h:
        from sklearn.metrics import roc_auc_score
        out["auroc"] = float(roc_auc_score(y_ai, scores))
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y_ai), len(y_ai))
            if y_ai[idx].min() == y_ai[idx].max():
                continue
            boots.append(roc_auc_score(y_ai[idx], scores[idx]))
        if boots:
            out["auroc_ci_low"], out["auroc_ci_high"] = map(
                float, np.percentile(boots, [2.5, 97.5]))
    return out


def evaluate_pair(df: pd.DataFrame, scorer: Callable, *, ai_label: int,
                  threshold: float, seed: int, show_progress: bool = True):
    """Score every row exactly once. Returns (predictions_df, metrics_dict).
    One output row per input row; failures are recorded in `status`."""
    if not {"text", "label"} <= set(df.columns):
        raise ValueError("CSV needs columns 'text' and 'label'.")
    labels = pd.to_numeric(df["label"], errors="coerce")
    bad_vals = set(labels.dropna().unique()) - {0, 1}
    if bad_vals:
        raise ValueError(f"Labels must be 0/1; found {sorted(bad_vals)}.")
    human_label = 1 - ai_label

    recs = []
    it = enumerate(df["text"].tolist())
    for pos, text in (tqdm(it, total=len(df)) if show_progress else it):
        recs.append(scorer(text, seed + pos))
    res = pd.DataFrame(recs)

    out = df.copy().reset_index(drop=True)
    out.insert(0, "row_id", np.arange(len(out)))
    out["discrepancy"] = res["score"].to_numpy()
    for col in ("status", "n_tokens_full", "n_tokens_used", "truncated",
                "inference_seconds"):
        out[col] = res[col].to_numpy()
    out["label_valid"] = labels.isin([0, 1]).to_numpy()
    scored = (out["status"] == "ok") & out["label_valid"] & out["discrepancy"].notna()
    out["predicted_label_at_epsilon"] = pd.array(
        [(ai_label if s > threshold else human_label) if ok else pd.NA
         for s, ok in zip(out["discrepancy"], scored)], dtype="Int64")

    m = compute_metrics(labels[scored.to_numpy()].to_numpy(),
                        out.loc[scored, "discrepancy"].to_numpy(),
                        ai_label, threshold, seed=seed)
    m.update(n_rows=len(out), n_skipped=int((~scored).sum()),
             n_truncated=int(out["truncated"].sum()),
             status_counts=json.dumps(out["status"].value_counts().to_dict()),
             inference_seconds=float(out["inference_seconds"].sum()))
    return out, m


def make_run_id(key: dict) -> str:
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def write_master(row: dict, path: str) -> None:
    """Upsert by run_id: re-running an identical configuration replaces its
    row instead of silently appending a duplicate."""
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        if "run_id" in old.columns:
            old = old[old["run_id"] != row["run_id"]]
        new = pd.concat([old, new], ignore_index=True)
    tmp = path + ".tmp"
    new.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ===========================================================================
# 4. DRIVER
# ===========================================================================
def preflight_labels(paths, ai_label):
    """Print label counts and one example per label so the AI/Human mapping
    can be verified by eye. Raises only for non-binary labels."""
    print("\nLabel pre-flight (AI_LABEL = %d):" % ai_label)
    for p in paths:
        df = pd.read_csv(p)
        lab = pd.to_numeric(df["label"], errors="coerce")
        vals = set(lab.dropna().unique())
        if not vals <= {0, 1}:
            raise ValueError(f"{p}: labels must be 0/1, found {sorted(vals)}")
        counts = lab.value_counts(dropna=False).to_dict()
        print(f"  {p}: rows={len(df)} counts={counts}")
        for v in sorted(vals):
            ex = df.loc[lab == v, "text"].dropna().astype(str).head(1).tolist()
            tag = "AI" if v == ai_label else "Human"
            print(f"     label {int(v)} (treated as {tag}): {ex[0][:80] if ex else ''!r}")


def get_args(argv=None):
    p = argparse.ArgumentParser(description="Hindi Fast-DetectGPT (black-box mGPT surrogate)")
    p.add_argument("--estimator", choices=["analytic", "monte_carlo"], default="analytic")
    p.add_argument("--n-samples", type=int, default=N_SAMPLES)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                   help="model weight dtype; reductions are always fp32")
    p.add_argument("--device", default="auto")
    p.add_argument("--ai-label", type=int, choices=[0, 1], default=AI_LABEL)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--scoring-model", default=SCORING_MODEL_NAME)
    p.add_argument("--sampling-model", default=SAMPLING_MODEL_NAME)
    p.add_argument("--scoring-revision", default=None)
    p.add_argument("--sampling-revision", default=None)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--llms", nargs="+", default=LLMS)
    p.add_argument("--test-root", default=TEST_ROOT)
    p.add_argument("--results-root", default=RESULTS_ROOT)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-n", type=int, default=10,
                   help="rows per file in smoke test (class-balanced)")
    return p.parse_args(argv)


def pick_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    return torch.device("cpu")


def main(argv=None) -> None:
    args = get_args(argv)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = pick_device(args.device)
    results_root = args.results_root + ("_smoke" if args.smoke_test else "")
    os.makedirs(results_root, exist_ok=True)
    master = os.path.join(results_root, MASTER_NAME)

    files = [(d, l, os.path.join(args.test_root, d, f"{d}_test_{l}.csv"))
             for d in args.datasets for l in args.llms]
    present = [(d, l, p) for d, l, p in files if os.path.exists(p)]
    for d, l, p in files:
        if not os.path.exists(p):
            print(f"[SKIP] missing: {p}")
    preflight_labels([p for _, _, p in present], args.ai_label)

    detector, info = load_detector(args, device)
    ai, hu = args.ai_label, 1 - args.ai_label
    run_t0 = time.perf_counter()
    env = dict(python=sys.version.split()[0], platform=platform.platform(),
               torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
               device=str(device),
               gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:
        pass
    print(f"\nFast-DetectGPT | black-box surrogate {args.sampling_model} / "
          f"{args.scoring_model} | estimator={args.estimator} | device={device}")

    for dataset, llm, path in present:
        print(f"\n{dataset.upper()} + {llm.upper()}")
        df = pd.read_csv(path)
        if args.smoke_test:                            # class-balanced head
            lab = pd.to_numeric(df["label"], errors="coerce")
            k = max(1, args.smoke_n // 2)
            df = pd.concat([df[lab == 1].head(k), df[lab == 0].head(k)]).sort_index()
        t0 = time.perf_counter()
        scorer = lambda text, seed, _d=detector: _d.score(text, seed)
        preds, m = evaluate_pair(df, scorer, ai_label=ai, threshold=args.threshold,
                                 seed=args.seed)
        wall = time.perf_counter() - t0

        out_dir = os.path.join(results_root, f"{dataset}_{llm}")
        os.makedirs(out_dir, exist_ok=True)
        name = "predictions.csv" if args.estimator == "analytic" \
            else f"predictions_{args.estimator}.csv"
        preds.to_csv(os.path.join(out_dir, name), index=False)

        key = dict(test_dataset=dataset, test_llm=llm, estimator=args.estimator,
                   n_samples=args.n_samples if args.estimator == "monte_carlo" else 0,
                   sampling_model=args.sampling_model, scoring_model=args.scoring_model,
                   sr=args.sampling_revision, cr=args.scoring_revision,
                   dtype=args.dtype, max_tokens=args.max_tokens,
                   threshold=args.threshold, seed=args.seed, ai_label=ai,
                   smoke_n=args.smoke_n if args.smoke_test else None)
        row = {  # legacy schema first (unchanged names), extras after
            "train_dataset": "NA", "train_llm": "NA",
            "test_dataset": dataset, "test_llm": llm,
            "accuracy": m["accuracy"], "f1_score": m["f1_score"], "auroc": m["auroc"],
            f"AI_recall_label_{ai}": m["ai_recall"],
            f"Human_recall_label_{hu}": m["human_recall"],
            "run_id": make_run_id(key), "estimator": args.estimator,
            "n_token_samples": key["n_samples"],
            "threshold": args.threshold,
            "threshold_policy": "fixed_exploratory_epsilon_not_fit_on_test",
            "auroc_ci_low": m["auroc_ci_low"], "auroc_ci_high": m["auroc_ci_high"],
            "precision": m["precision"], "false_positive_rate": m["false_positive_rate"],
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            "n_rows": m["n_rows"], "n_scored": m["n_scored"], "n_skipped": m["n_skipped"],
            "n_truncated": m["n_truncated"], "status_counts": m["status_counts"],
            "sampling_model": args.sampling_model, "scoring_model": args.scoring_model,
            "model_dtype": args.dtype, "max_tokens": args.max_tokens, "seed": args.seed,
            "setting": "black_box_surrogate", "smoke_test": args.smoke_test,
            "inference_seconds": m["inference_seconds"], "pair_wall_seconds": wall,
        }
        write_master(row, master)
        print(f"  n={m['n_scored']}/{m['n_rows']} AUROC={m['auroc']:.4f} "
              f"[{m['auroc_ci_low']:.3f},{m['auroc_ci_high']:.3f}] | at eps="
              f"{args.threshold}: ACC={m['accuracy']:.3f} AIR={m['ai_recall']:.3f} "
              f"HR={m['human_recall']:.3f} | inference {m['inference_seconds']:.1f}s")
        if m["auroc"] == m["auroc"] and m["auroc"] < 0.5:
            print("  NOTE: AUROC < 0.5 with AI as positive. Check --ai-label "
                  "(or the surrogate model genuinely inverts here).")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    meta = dict(info=info, env=env, args=vars(args),
                total_wall_seconds_excluding_model_load=time.perf_counter() - run_t0)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(results_root, f"run_metadata_{args.estimator}_{stamp}.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nDone. Master: {master}")


if __name__ == "__main__":
    main()