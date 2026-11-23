"""
Common utilities for the Hindi adaptation of Text Fluoroscopy
(Yu et al., EMNLP 2024) using IndicBERT (ai4bharat/indic-bert) as encoder.

Label convention used throughout (must match your CSV files):
    label = 1  ->  machine (LLM) generated text
    label = 0  ->  human written text
This matches the paper's AUROC definition: AUROC = P(a random machine text
scores higher than a random human text). If your CSVs use the opposite
convention, set POSITIVE_IS_MACHINE = False below instead of remapping data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

POSITIVE_IS_MACHINE = True  # label 1 == machine-generated

MODEL_NAME = "ai4bharat/indic-bert"


def load_tokenizer_and_mlm_model(model_name=MODEL_NAME, device="cpu"):
    """
    Loads the encoder as a Masked-LM model (not AutoModel) so that we have
    access to its vocabulary projection head, which the original Text
    Fluoroscopy method requires (it applies phi(.) — the LM's own next-token
    / masked-token head — to every layer, not just the last one).
    """
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    # keep_accents=True is AI4Bharat's own recommendation for Hindi/Indic
    # text so that vowel matras/diacritics are preserved by the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name, keep_accents=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()  # dropout must be OFF while extracting features
    return tokenizer, model


def get_vocab_head(model):
    """
    Returns a callable head(hidden_states) -> vocab logits, matching the
    model's own MLM head, so it can be applied to ANY layer's FULL TOKEN
    SEQUENCE (early-exit style), exactly as Text Fluoroscopy does with
    phi(.). Handles the common HF naming conventions.
    """
    if hasattr(model, "predictions"):   # AlbertForMaskedLM (IndicBERT)
        return model.predictions
    if hasattr(model, "cls"):           # BertForMaskedLM / similar
        return model.cls
    if hasattr(model, "lm_head"):       # causal LMs (e.g. gte-Qwen)
        return model.lm_head
    raise ValueError(
        "Could not locate an MLM/LM head on this model; Text Fluoroscopy "
        "requires projecting hidden states to vocabulary space."
    )


def last_valid_index(attention_mask_row):
    """Index of the last non-padding token for a single example."""
    return int(attention_mask_row.sum().item()) - 1


@torch.no_grad()
def encode_text(text, tokenizer, model, device, max_length=512):
    """
    Runs the encoder once and returns:
      hidden_states: tuple of (num_layers+1) tensors, each (seq_len, hidden_dim)
      last_idx: index of the last real (non-padding) token
    """
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(device)
    outputs = model(**inputs)
    hidden_states = tuple(h.squeeze(0) for h in outputs.hidden_states)  # (seq_len, hidden)
    last_idx = last_valid_index(inputs["attention_mask"][0])
    return hidden_states, last_idx


@torch.no_grad()
def select_layer_and_feature(hidden_states, last_idx, vocab_head):
    """
    Implements Text Fluoroscopy's core step (Section 3 of the paper),
    matching the official implementation's two SEPARATE stages:

      1. Layer selection: project EVERY token position of every layer
         into vocabulary space, then pick the interior layer M maximizing
         the sequence-averaged KL(q_N||q_j) + KL(q_0||q_j), j in {1,...,N-1}
         (reduction='batchmean' over token positions, as in the released
         code — NOT computed from a single token).
      2. Feature extraction: only AFTER M is chosen, take that layer's
         LAST-TOKEN hidden state as the classifier feature (paper's
         h_{t-1}^(M)).

    Logits for one layer at a time are computed and discarded immediately
    to keep peak memory bounded (projecting every token of every layer to
    vocab space is the expensive step).
    """
    num_layers = len(hidden_states)  # N+1 (embedding layer + N transformer layers)

    def layer_logits(h):
        # h: (seq_len, hidden_dim) -> (seq_len, vocab_size)
        return vocab_head(h.unsqueeze(0)).squeeze(0)

    first_logits = layer_logits(hidden_states[0])
    last_logits = layer_logits(hidden_states[-1])
    first_probs = F.softmax(first_logits, dim=-1)
    last_probs = F.softmax(last_logits, dim=-1)
    del first_logits, last_logits

    best_layer, best_score = None, None
    for j in range(1, num_layers - 1):  # interior layers only: J = {1,...,N-1}
        middle_logits = layer_logits(hidden_states[j])
        middle_log_probs = F.log_softmax(middle_logits, dim=-1)

        # F.kl_div(log_q_j, q_target, reduction="batchmean") averages
        # KL(q_target || q_j) over the sequence-position "batch" dim,
        # exactly like the official repo's per-token KL averaging.
        kl_last = F.kl_div(middle_log_probs, last_probs, reduction="batchmean")
        kl_first = F.kl_div(middle_log_probs, first_probs, reduction="batchmean")
        score = (kl_last + kl_first).item()

        del middle_logits, middle_log_probs

        if best_score is None or score > best_score:
            best_score, best_layer = score, j

    feature = hidden_states[best_layer][last_idx]
    return best_layer, feature


class BinaryClassifier(nn.Module):
    """
    Matches the paper's detector D: FC(1024) -> Tanh -> FC(512) -> Tanh -> FC(1).
    Outputs a raw logit — use BCEWithLogitsLoss for training and a sigmoid
    only at inference time, for numerical stability.
    """

    def __init__(self, input_size, hidden_sizes=(1024, 512), dropout_prob=0.4):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Dropout(dropout_prob), nn.Linear(prev_size, hidden_size), nn.Tanh()])
            prev_size = hidden_size
        self.dense = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, 1)

    def forward(self, x):
        x = self.dense(x)
        return self.classifier(x).squeeze(-1)  # raw logit, shape (batch,)