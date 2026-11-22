"""Shared detector conventions for the Hindi MGT statistical pipelines.

All statistical outputs are canonicalized to:
    true_label = 1 / predicted_label = 1 -> AI-generated
    true_label = 0 / predicted_label = 0 -> Human-written
    score: larger means more AI-like

Important: raw ground-truth encodings are distinct from old saved prediction
encodings. DetectGPT source CSVs use raw 0=AI, while its old prediction CSVs
used 1=AI. The cache-rebuilt DetectGPT outputs already contain canonical
labels and must NOT be flipped again.
"""
from dataclasses import dataclass
from typing import Optional

DATASETS = ["hindisumm", "xquad"]
LLMS = ["gemini", "llama3", "qwen3"]
RQ5_LLMS = ["gemini", "qwen3"]
TRAINABLE_MODELS = ["XLM-RoBERTa", "Indic-BERT", "CONDA", "RADAR", "Text-Fluoroscopy"]
ZERO_SHOT_MODELS = ["GPTZero", "DetectGPT"]
ALL_MODELS = TRAINABLE_MODELS + ZERO_SHOT_MODELS

@dataclass(frozen=True)
class ModelSpec:
    raw_ai_label: int
    raw_score_is_ai_probability: Optional[bool]
    score_column: Optional[str]
    source_verified: bool
    source_notes: str
    protocol_warnings: str = ""

MODEL_SPECS = {
    "XLM-RoBERTa": ModelSpec(0, False, None, True,
        "Shared training/testing CSV convention: raw 0=AI, 1=Human. Inference stores hard predictions but no per-example score.",
        "The supplied training code evaluates the held-out test split every epoch before saving the final model; describe this as test-informed model selection."),
    "Indic-BERT": ModelSpec(0, False, None, True,
        "Raw 0=AI, 1=Human. Prediction exports do not contain a usable per-example probability score.",
        "The supplied training code evaluates the test split every epoch; per-example AUC cannot be reconstructed from hard labels."),
    "CONDA": ModelSpec(0, False, "prediction_probability_class1", True,
        "Raw 0=AI, 1=Human; saved class-1 probability is P(Human), so canonical AI score is 1-p.",
        "Use the saved continuous class-1 probability only after inversion."),
    "RADAR": ModelSpec(0, False, "prediction_probability", True,
        "Raw 0=AI, 1=Human; saved sigmoid probability denotes Human, so canonical AI score is 1-p.",
        "Use the saved continuous probability only after inversion."),
    "Text-Fluoroscopy": ModelSpec(1, True, "predicted_prob_label1", True,
        "Raw 1=AI, 0=Human; predicted_prob_label1 is P(AI).", "No label or score inversion required."),
    "GPTZero": ModelSpec(0, False, "perplexity", True,
        "Raw shared source/test labels: 0=AI, 1=Human. Lower perplexity is more AI-like; canonical score is -perplexity.",
        "The source-training Youden-J threshold is read from gptzero_all_results.csv and applied to each target score; target labels are never used to choose it."),
    "DetectGPT": ModelSpec(0, False, "llr", True,
        "RQ1-RQ3 shared source/test labels are raw 0=AI, 1=Human. The v3 cache rebuild canonicalizes labels to 1=AI and preserves higher-LLR-is-AI direction.",
        "Do not reuse the old saved DetectGPT prediction columns or old thresholds. Use the v3 rebuilt_from_score_cache outputs; no inference rerun is needed."),
}

SEED = 42
BOOTSTRAPS = 2000
CLUSTER_BOOTSTRAPS = 1500
ALPHA = 0.05

def canonicalize_label(raw_label: int, model: str) -> int:
    return 1 if int(raw_label) == MODEL_SPECS[model].raw_ai_label else 0

# These entries are retained for report metadata and pipeline compatibility.
# The source-aware loaders implement the detector-specific path and threshold
# protocols rather than fitting thresholds on targets.
ZERO_SHOT_SPECS = {
    "GPTZero": {
        "subdir": "gptzero", "score_col": "perplexity",
        "label_cols": ("label", "true_label"),
        "raw_ai_label": 0, "score_direction": "lower_is_ai",
    },
    "DetectGPT": {
        "subdir": "detect_gpt_all_domain", "score_col": "llr",
        "label_cols": ("true_label",),
        "raw_ai_label": 1, "score_direction": "ai_native",
        "note": "Rebuilt output files are already canonical: true_label=1 AI, score=llr high AI.",
    },
}
