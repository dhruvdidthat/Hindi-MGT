#!/usr/bin/env python3
"""
Text-Fluoroscopy inference on all 6 x 6 new-prompt evaluations.

This script uses the corrected Hindi Text-Fluoroscopy train/common/inference
pipeline. It evaluates each of the six trained classifiers on each of the six
prompt-shifted test sets:

    hindisumm x {gemini, llama3, qwen3}
    xquad     x {gemini, llama3, qwen3}

Total:
    6 trained classifiers x 6 test sets = 36 evaluations.

Test files:
    new_prompt_data/{dataset}_{llm}_new_prompt.csv

Only `text` and `label` are used. Text-Fluoroscopy does not use a perturbed
text pair or a contrastive loss.

IMPORTANT:
The feature extraction here MUST be identical to training:
    IndicBERT MLM
        -> vocabulary-space KL for every token position
        -> select best interior layer
        -> last valid-token hidden state
        -> trained BinaryClassifier

The old implementation incorrectly:
    - used AutoModel instead of AutoModelForMaskedLM,
    - computed KL directly over hidden dimensions,
    - mean-pooled the selected layer,
    - used a different classifier definition/dropout.

All of those are removed here by importing the corrected implementation from
text_fluoroscopy_common.py.

Label convention:
    label = 1 -> machine-generated
    label = 0 -> human-written
"""

import os
import gc
from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)

from text_fluoroscopy_common import (
    MODEL_NAME,
    POSITIVE_IS_MACHINE,
    BinaryClassifier,
    encode_text,
    get_vocab_head,
    load_tokenizer_and_mlm_model,
    select_layer_and_feature,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

TRAINED_MODELS_ROOT = "text_fluoroscopy/trained_models"
NEW_PROMPT_DIR = "prompt_change/new_prompt_data"

RESULTS_ROOT = "text_fluoroscopy/inference_results_new_prompt"
MASTER_RESULTS_CSV = os.path.join(
    RESULTS_ROOT,
    "text_fluoroscopy_all_results_new_prompt.csv",
)

DATASETS = ["hindisumm", "xquad"]
LLMS = ["gemini", "llama3", "qwen3"]


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def build_test_features(
    test_dataset,
    test_llm,
    tokenizer,
    model,
    vocab_head,
    max_length=512,
    log_every=100,
):
    """
    Build features for one new-prompt test set using the exact common.py
    Text-Fluoroscopy feature pipeline.

    Returns:
        X_test: numpy array (N, hidden_size)
        y_test: numpy array
        texts: list[str]

    Returns None if the expected test file does not exist.
    """
    test_path = os.path.join(
        NEW_PROMPT_DIR,
        f"{test_dataset}_{test_llm}_new_prompt.csv",
    )

    if not os.path.exists(test_path):
        print(
            f"WARNING: test file not found for "
            f"{test_dataset}+{test_llm}: {test_path}"
        )
        return None

    df = pd.read_csv(test_path)

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{test_path} is missing required columns: {sorted(missing)}"
        )

    df = df.dropna(subset=["text"]).reset_index(drop=True)

    texts = df["text"].astype(str).tolist()
    labels = np.asarray(df["label"], dtype=np.float32)

    features = []

    print(
        f"\n--- Extracting features: "
        f"{test_dataset}+{test_llm} ---"
    )
    print(f"File: {test_path}")
    print(f"Texts: {len(texts)}")

    for i, text in enumerate(texts):
        hidden_states, last_idx = encode_text(
            text,
            tokenizer,
            model,
            device,
            max_length,
        )

        best_layer, feature = select_layer_and_feature(
            hidden_states,
            last_idx,
            vocab_head,
        )

        features.append(feature.cpu().numpy())

        if i % log_every == 0 or i == len(texts) - 1:
            print(
                f"  processed {i + 1}/{len(texts)} "
                f"(selected layer={best_layer})"
            )

    return np.stack(features), labels, texts


# ------------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------------

def load_classifier(model_path):
    """
    Load a classifier produced by text_fluoroscopy_train.py.

    Existing project checkpoints contain metadata in addition to the
    state_dict, so weights_only=False is used for these trusted,
    locally-generated checkpoints. Do not use this setting for untrusted
    checkpoint files.
    """
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"'model_state_dict' not found in checkpoint: {model_path}"
        )

    classifier = BinaryClassifier(
        input_size=int(checkpoint["input_size"]),
        dropout_prob=float(checkpoint.get("dropout_prob", 0.4)),
    ).to(device)

    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    return classifier, checkpoint


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def evaluate_predictions(labels, logits):
    """
    Evaluate raw classifier logits.

    Since BinaryClassifier outputs one raw logit:
        sigmoid(logit) = P(label == 1)

    With the project convention:
        label 1 = machine
        label 0 = human

    AUROC is explicitly computed with machine-generated text as the positive
    semantic class.
    """
    probs_label1 = 1.0 / (1.0 + np.exp(-logits))
    predicted_labels = (probs_label1 >= 0.5).astype(int)

    if POSITIVE_IS_MACHINE:
        machine_probs = probs_label1
        machine_labels = labels.astype(int)
        human_label = 0
        machine_label = 1
    else:
        machine_probs = 1.0 - probs_label1
        machine_labels = 1 - labels.astype(int)
        human_label = 1
        machine_label = 0

    accuracy = accuracy_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)
    auroc = roc_auc_score(machine_labels, machine_probs)

    human_recall = recall_score(
        labels,
        predicted_labels,
        pos_label=human_label,
        zero_division=0,
    )
    ai_recall = recall_score(
        labels,
        predicted_labels,
        pos_label=machine_label,
        zero_division=0,
    )

    return {
        "probs_label1": probs_label1,
        "predicted_labels": predicted_labels,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "AUROC": auroc,
        "Human Recall": human_recall,
        "AI Recall": ai_recall,
    }


# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------

def append_to_master_csv(metrics):
    os.makedirs(os.path.dirname(MASTER_RESULTS_CSV), exist_ok=True)

    if os.path.exists(MASTER_RESULTS_CSV):
        metrics.to_csv(
            MASTER_RESULTS_CSV,
            mode="a",
            header=False,
            index=False,
        )
    else:
        metrics.to_csv(
            MASTER_RESULTS_CSV,
            index=False,
        )


def evaluate_one_case(
    train_dataset,
    train_llm,
    test_dataset,
    test_llm,
    X_test,
    y_test,
    texts,
):
    print("\n" + "=" * 75)
    print(
        f"TRAIN={train_dataset}+{train_llm} | "
        f"TEST(new_prompt)={test_dataset}+{test_llm}"
    )
    print("=" * 75)

    model_path = os.path.join(
        TRAINED_MODELS_ROOT,
        f"{train_dataset}_{train_llm}",
        "model.pt",
    )

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    classifier, checkpoint = load_classifier(model_path)

    expected_input_size = int(checkpoint["input_size"])
    if X_test.shape[1] != expected_input_size:
        raise ValueError(
            f"Feature dimension mismatch for {model_path}: "
            f"checkpoint expects {expected_input_size}, "
            f"but test features have {X_test.shape[1]}."
        )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        logits = (
            classifier(X_test_tensor)
            .detach()
            .cpu()
            .numpy()
        )

    metrics_values = evaluate_predictions(y_test, logits)

    print(f"Accuracy:      {metrics_values['Accuracy']:.4f}")
    print(f"F1 Score:      {metrics_values['F1 Score']:.4f}")
    print(f"AUROC:         {metrics_values['AUROC']:.4f}")
    print(f"Human Recall:  {metrics_values['Human Recall']:.4f}")
    print(f"AI Recall:     {metrics_values['AI Recall']:.4f}")

    print(
        "\nClassification Report:\n",
        classification_report(
            y_test,
            metrics_values["predicted_labels"],
            target_names=["Human", "AI"]
            if POSITIVE_IS_MACHINE
            else ["AI", "Human"],
            zero_division=0,
        ),
    )

    result_dir = os.path.join(
        RESULTS_ROOT,
        f"{train_dataset}_{train_llm}_tested_on_"
        f"{test_dataset}_{test_llm}",
    )
    os.makedirs(result_dir, exist_ok=True)

    metrics_df = pd.DataFrame(
        [
            {
                "Train Dataset": train_dataset,
                "Train LLM": train_llm,
                "Test Dataset": test_dataset,
                "Test LLM": test_llm,
                "Accuracy": metrics_values["Accuracy"],
                "F1 Score": metrics_values["F1 Score"],
                "AUROC": metrics_values["AUROC"],
                "AI Recall": metrics_values["AI Recall"],
                "Human Recall": metrics_values["Human Recall"],
            }
        ]
    )

    metrics_df.to_csv(
        os.path.join(result_dir, "metrics.csv"),
        index=False,
    )
    append_to_master_csv(metrics_df)

    predictions_df = pd.DataFrame(
        {
            "text": texts,
            "true_label": y_test.astype(int),
            "predicted_label": metrics_values["predicted_labels"],
            "predicted_prob_label1": metrics_values["probs_label1"],
        }
    )

    predictions_df.to_csv(
        os.path.join(result_dir, "predictions.csv"),
        index=False,
    )

    print(f"Results saved to: {result_dir}")

    del classifier
    del X_test_tensor

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    all_combinations = list(product(DATASETS, LLMS))

    print("Loading IndicBERT MLM model...")
    tokenizer, model = load_tokenizer_and_mlm_model(
        MODEL_NAME,
        device,
    )
    vocab_head = get_vocab_head(model)

    # --------------------------------------------------------------
    # Compute each test set's Text-Fluoroscopy features exactly once.
    # They do not depend on which classifier is later applied.
    # --------------------------------------------------------------
    test_feature_cache = {}

    for test_dataset, test_llm in all_combinations:
        features = build_test_features(
            test_dataset,
            test_llm,
            tokenizer,
            model,
            vocab_head,
        )

        if features is not None:
            test_feature_cache[(test_dataset, test_llm)] = features

    print(
        f"\nCached {len(test_feature_cache)}/{len(all_combinations)} "
        f"new-prompt test sets."
    )

    # --------------------------------------------------------------
    # 6 trained classifiers x 6 test sets = 36 evaluations.
    # --------------------------------------------------------------
    total_cases = len(all_combinations) ** 2
    case_num = 0

    for train_dataset, train_llm in all_combinations:
        for test_dataset, test_llm in all_combinations:
            case_num += 1

            print(
                f"\n[{case_num}/{total_cases}] "
                f"TRAIN={train_dataset}+{train_llm} "
                f"TEST={test_dataset}+{test_llm}"
            )

            cached = test_feature_cache.get(
                (test_dataset, test_llm)
            )

            if cached is None:
                print(
                    f"Skipping: no test features for "
                    f"{test_dataset}+{test_llm}"
                )
                continue

            X_test, y_test, texts = cached

            try:
                evaluate_one_case(
                    train_dataset,
                    train_llm,
                    test_dataset,
                    test_llm,
                    X_test,
                    y_test,
                    texts,
                )
            except Exception as exc:
                print(
                    f"FAILED case "
                    f"{train_dataset}+{train_llm} -> "
                    f"{test_dataset}+{test_llm}: "
                    f"{type(exc).__name__}: {exc}"
                )

    print(
        "\nAll new-prompt evaluations completed."
        f"\nMaster results: {MASTER_RESULTS_CSV}"
    )


if __name__ == "__main__":
    main()