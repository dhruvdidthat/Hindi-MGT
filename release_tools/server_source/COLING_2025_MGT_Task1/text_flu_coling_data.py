#!/usr/bin/env python3
"""
Text-Fluoroscopy inference on the fixed COLING external test set.

This script is the COLING-data counterpart of the corrected Hindi
Text-Fluoroscopy train/inference pipeline.

Important:
- It does NOT train anything.
- It loads the six already-trained classifiers:
      text_fluoroscopy/trained_models/{dataset}_{llm}/model.pt
  for:
      hindisumm × {gemini, llama3, qwen3}
      xquad     × {gemini, llama3, qwen3}
- It evaluates all six classifiers on:
      ./perturbed_coling_data/hindi_data_perturb.csv
- Text-Fluoroscopy uses only the original `text` column. `text_perturb`,
  if present, is intentionally ignored because Text-Fluoroscopy does not
  use contrastive/perturbed text pairs.
- COLING features are computed ONCE and reused for all six classifiers.
- Feature extraction MUST use the same common.py implementation used during
  training:
      1. IndicBERT MLM vocabulary-space KL layer selection over ALL tokens.
      2. Select the best interior layer.
      3. Extract that layer's last valid-token representation.
- The classifier is the exact BinaryClassifier definition from common.py.
- Label convention:
      label = 1 -> machine-generated
      label = 0 -> human-written
"""

import os
import gc

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

# Use the same GPU family as the corrected inference script.
# Change to "cuda:1" if that is the GPU you want to reserve for this job.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

# Adjust these only if your project directory layout differs.
TRAINED_MODELS_ROOT = "../training_models/text_fluoroscopy/trained_models"
COLING_TEST_FILE = "./perturbed_coling_data/hindi_data_perturb.csv"

RESULTS_ROOT = "text_fluoroscopy/inference_results_coling"
MASTER_RESULTS_DIR = "Text_fluoroscopy_results"
MASTER_RESULTS_CSV = os.path.join(
    MASTER_RESULTS_DIR, "text_fluoroscopy_coling_results.csv"
)

RUNS = [
    ("hindisumm", "gemini"),
    ("hindisumm", "llama3"),
    ("hindisumm", "qwen3"),
    ("xquad", "gemini"),
    ("xquad", "llama3"),
    ("xquad", "qwen3"),
]


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def build_coling_features(tokenizer, model, vocab_head, max_length=512):
    """
    Extract the exact same Text-Fluoroscopy features used during training.

    For each text:
      IndicBERT
        -> hidden states
        -> vocabulary-space KL for every token position
        -> best interior layer
        -> last valid-token hidden representation

    Returns:
        X_test: numpy array of shape (N, hidden_size)
        y_test: numpy array of labels
        texts: list[str]
    """
    if not os.path.exists(COLING_TEST_FILE):
        raise FileNotFoundError(
            f"COLING test file not found: {COLING_TEST_FILE}"
        )

    df = pd.read_csv(COLING_TEST_FILE)

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"COLING file is missing required columns: {sorted(missing)}"
        )

    df = df.dropna(subset=["text"]).reset_index(drop=True)

    texts = df["text"].astype(str).tolist()
    labels = np.asarray(df["label"], dtype=np.float32)

    features = []

    print(
        f"\nExtracting Text-Fluoroscopy features from "
        f"{COLING_TEST_FILE} ({len(texts)} texts)..."
    )

    for i, text in enumerate(texts):
        hidden_states, last_idx = encode_text(
            text,
            tokenizer,
            model,
            device,
            max_length=max_length,
        )

        best_layer, feature = select_layer_and_feature(
            hidden_states,
            last_idx,
            vocab_head,
        )

        features.append(feature.cpu().numpy())

        if i % 100 == 0 or i == len(texts) - 1:
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
    Load the classifier saved by text_fluoroscopy_train.py.

    weights_only=False is intentional because the existing checkpoints
    contain metadata such as NumPy/Python scalar values in addition to the
    state_dict. Only use this for trusted checkpoints produced by this
    project.
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

    input_size = int(checkpoint["input_size"])
    dropout_prob = float(checkpoint.get("dropout_prob", 0.4))

    classifier = BinaryClassifier(
        input_size=input_size,
        dropout_prob=dropout_prob,
    ).to(device)

    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    return classifier, checkpoint


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def evaluate_predictions(labels, logits):
    """
    Convert classifier logits to label-1 probabilities and compute metrics.

    Common convention:
        1 = machine
        0 = human

    AUROC is explicitly computed as:
        P(machine score > human score)
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
# Evaluation
# ------------------------------------------------------------------

def append_master_result(row):
    os.makedirs(MASTER_RESULTS_DIR, exist_ok=True)

    row_df = pd.DataFrame([row])

    if os.path.exists(MASTER_RESULTS_CSV):
        row_df.to_csv(
            MASTER_RESULTS_CSV,
            mode="a",
            header=False,
            index=False,
        )
    else:
        row_df.to_csv(MASTER_RESULTS_CSV, index=False)


def evaluate_one_case(
    train_dataset,
    train_llm,
    X_test,
    y_test,
    texts,
):
    print("\n" + "=" * 70)
    print(
        f"Evaluating TRAIN={train_dataset}+{train_llm} "
        f"-> COLING test set"
    )
    print("=" * 70)

    model_path = os.path.join(
        TRAINED_MODELS_ROOT,
        f"{train_dataset}_{train_llm}",
        "model.pt",
    )

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    classifier, checkpoint = load_classifier(model_path)

    expected_input_size = int(checkpoint["input_size"])
    if X_test.shape[1] != expected_input_size:
        raise ValueError(
            f"Feature dimension mismatch for {model_path}: "
            f"checkpoint expects {expected_input_size}, "
            f"but COLING features have {X_test.shape[1]} dimensions."
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

    metrics = evaluate_predictions(y_test, logits)

    print(f"Accuracy:      {metrics['Accuracy']:.4f}")
    print(f"F1 Score:      {metrics['F1 Score']:.4f}")
    print(f"AUROC:         {metrics['AUROC']:.4f}")
    print(f"Human Recall:  {metrics['Human Recall']:.4f}")
    print(f"AI Recall:     {metrics['AI Recall']:.4f}")

    print(
        "\nClassification Report:\n",
        classification_report(
            y_test,
            metrics["predicted_labels"],
            target_names=["Human", "AI"]
            if POSITIVE_IS_MACHINE
            else ["AI", "Human"],
            zero_division=0,
        ),
    )

    result_dir = os.path.join(
        RESULTS_ROOT,
        f"{train_dataset}_{train_llm}",
        "tested_on_coling_hindi_data",
    )
    os.makedirs(result_dir, exist_ok=True)

    predictions_df = pd.DataFrame(
        {
            "text": texts,
            "true_label": y_test.astype(int),
            "predicted_label": metrics["predicted_labels"],
            "predicted_prob_label1": metrics["probs_label1"],
        }
    )

    predictions_df.to_csv(
        os.path.join(result_dir, "predictions.csv"),
        index=False,
    )

    metrics_row = {
        "Source": f"{train_dataset}_{train_llm}",
        "Target": "COLING_hindi_data",
        "Test_Set": "hindi_data_perturb.csv",
        "Case_Type": "external",
        "F1": metrics["F1 Score"],
        "Accuracy": metrics["Accuracy"],
        "Human_Recall": metrics["Human Recall"],
        "AI_Recall": metrics["AI Recall"],
        "AUCROC": metrics["AUROC"],
    }

    pd.DataFrame([metrics_row]).to_csv(
        os.path.join(result_dir, "metrics.csv"),
        index=False,
    )

    append_master_result(metrics_row)

    print(f"Results saved to: {result_dir}")

    del classifier
    del X_test_tensor

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics_row


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(MASTER_RESULTS_DIR, exist_ok=True)

    print("Loading IndicBERT MLM model...")
    tokenizer, model = load_tokenizer_and_mlm_model(
        MODEL_NAME,
        device,
    )
    vocab_head = get_vocab_head(model)

    # Compute COLING features exactly once because they are independent
    # of which trained classifier is subsequently evaluated.
    X_test, y_test, texts = build_coling_features(
        tokenizer,
        model,
        vocab_head,
    )

    print(
        f"\nCOLING feature matrix: {X_test.shape}"
    )

    for case_num, (dataset, llm) in enumerate(RUNS, start=1):
        print(
            f"\n[{case_num}/{len(RUNS)}] "
            f"Running {dataset}+{llm}"
        )

        try:
            evaluate_one_case(
                dataset,
                llm,
                X_test,
                y_test,
                texts,
            )
        except Exception as exc:
            print(
                f"FAILED {dataset}+{llm}: {type(exc).__name__}: {exc}"
            )

    print(
        "\nAll COLING evaluations completed."
        f"\nMaster results: {MASTER_RESULTS_CSV}"
    )


if __name__ == "__main__":
    main()
