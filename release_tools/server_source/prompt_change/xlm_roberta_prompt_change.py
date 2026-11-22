#!/usr/bin/env python3
"""
XLM-RoBERTa inference on all 6 x 6 new-prompt evaluations.

Evaluates each of the six trained XLM-R classifiers on each of the six
prompt-shifted test sets:

    hindisumm x {gemini, llama3, qwen3}
    xquad     x {gemini, llama3, qwen3}

Total:
    6 trained classifiers x 6 test sets = 36 evaluations.

Test files:
    new_prompt_data/{dataset}_{llm}_new_prompt.csv

Each test CSV must contain:
    text
    label

IMPORTANT LABEL CONVENTION
--------------------------
The original XLM-R training/inference implementation uses:

    raw model label 0 = AI / machine-generated
    raw model label 1 = Human

For consistency with the downstream statistical pipeline, this inference
script converts outputs to the canonical convention:

    true_label     1 = AI
                   0 = Human

    predicted_label
                   1 = AI
                   0 = Human

Therefore:

    canonical AI probability = raw XLM-R probability for class 0
    canonical Human probability = raw XLM-R probability for class 1

AUROC is calculated with AI as the positive semantic class.

Per evaluation, this script produces:

    <results_root>/
        <train_dataset>_<train_llm>_tested_on_<test_dataset>_<test_llm>/
            metrics.csv
            predictions.csv

And one master file:

    xlm_roberta_all_results_new_prompt.csv
"""

import os
import gc
import random
from itertools import product

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)

from tqdm import tqdm


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------

# Keep the same GPU selection as the original XLM-R inference script.
device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DATASETS = [
    "hindisumm",
    "xquad",
]

LLMS = [
    "gemini",
    "llama3",
    "qwen3",
]


# Trained XLM-R checkpoints
TRAINED_MODELS_ROOT = "../training_models/xlm_roberta/trained_models"

# Prompt-shifted test data
NEW_PROMPT_DIR = "new_prompt_data"

# Results
RESULTS_ROOT = "xlm_roberta/inference_results_new_prompt"

MASTER_RESULTS_CSV = os.path.join(
    RESULTS_ROOT,
    "xlm_roberta_all_results_new_prompt.csv",
)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class ArticleDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_len=512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):

        text = str(self.texts[item])
        label = int(self.labels[item])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }


# ------------------------------------------------------------------
# Test-set loading
# ------------------------------------------------------------------

def load_new_prompt_test_set(
    test_dataset,
    test_llm,
):
    """
    Load one prompt-shifted test set.

    Expected file:

        new_prompt_data/{dataset}_{llm}_new_prompt.csv

    Expected columns:

        text
        label

    Original XLM-R label convention:

        0 = AI
        1 = Human
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

    required_columns = {
        "text",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{test_path} is missing required columns: "
            f"{sorted(missing)}"
        )

    # Match Text-Fluoroscopy behavior: remove rows with missing text.
    df = (
        df
        .dropna(subset=["text"])
        .reset_index(drop=True)
    )

    texts = (
        df["text"]
        .astype(str)
        .tolist()
    )

    raw_labels = (
        df["label"]
        .astype(int)
        .to_numpy()
    )

    return df, texts, raw_labels


# ------------------------------------------------------------------
# XLM-R inference
# ------------------------------------------------------------------

def run_model_inference(
    model,
    tokenizer,
    texts,
    raw_labels,
    batch_size=16,
    max_len=512,
):
    """
    Run XLM-R inference.

    Original XLM-R label convention:

        raw label 0 = AI
        raw label 1 = Human

    Output convention used by this script:

        canonical label 1 = AI
        canonical label 0 = Human

    Therefore:

        canonical_predicted_label = 1 - raw_predicted_label

        AI probability = raw probability for class 0
        Human probability = raw probability for class 1
    """

    dataset = ArticleDataset(
        texts=texts,
        labels=raw_labels,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    raw_predictions = []
    raw_probabilities_class0 = []
    raw_probabilities_class1 = []
    raw_true_labels = []

    model.eval()

    with torch.no_grad():

        for batch in tqdm(
            dataloader,
            desc="XLM-R inference",
        ):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            )

            raw_preds = torch.argmax(
                outputs.logits,
                dim=1,
            )

            raw_predictions.extend(
                raw_preds.cpu().numpy().tolist()
            )

            raw_probabilities_class0.extend(
                probabilities[:, 0]
                .cpu()
                .numpy()
                .tolist()
            )

            raw_probabilities_class1.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
                .tolist()
            )

            raw_true_labels.extend(
                labels.cpu().numpy().tolist()
            )

    raw_predictions = np.asarray(
        raw_predictions,
        dtype=int,
    )

    raw_true_labels = np.asarray(
        raw_true_labels,
        dtype=int,
    )

    raw_probabilities_class0 = np.asarray(
        raw_probabilities_class0,
        dtype=float,
    )

    raw_probabilities_class1 = np.asarray(
        raw_probabilities_class1,
        dtype=float,
    )

    # --------------------------------------------------------------
    # Convert to canonical convention:
    #
    #   1 = AI
    #   0 = Human
    #
    # Original XLM-R:
    #
    #   0 = AI
    #   1 = Human
    # --------------------------------------------------------------

    predicted_labels = 1 - raw_predictions

    true_labels = 1 - raw_true_labels

    # Canonical label 1 is AI.
    predicted_prob_label1 = raw_probabilities_class0

    predicted_prob_label0 = raw_probabilities_class1

    return {
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
        "predicted_prob_label1": predicted_prob_label1,
        "predicted_prob_label0": predicted_prob_label0,

        # Keep raw values available for debugging/auditing.
        "raw_true_labels": raw_true_labels,
        "raw_predicted_labels": raw_predictions,
        "raw_prob_class0": raw_probabilities_class0,
        "raw_prob_class1": raw_probabilities_class1,
    }


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def evaluate_predictions(
    true_labels,
    predicted_labels,
    predicted_prob_label1,
):
    """
    Calculate metrics using canonical labels:

        1 = AI
        0 = Human

    predicted_prob_label1 is therefore:

        P(AI)

    AI is explicitly treated as the positive semantic class for AUROC.
    """

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    # AI = positive semantic class.
    try:
        auroc = roc_auc_score(
            true_labels,
            predicted_prob_label1,
        )
    except ValueError:
        auroc = float("nan")

    ai_recall = recall_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    human_recall = recall_score(
        true_labels,
        predicted_labels,
        pos_label=0,
        zero_division=0,
    )

    return {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "AUROC": auroc,
        "AI Recall": ai_recall,
        "Human Recall": human_recall,
    }


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_xlm_roberta_model(
    model_path,
):
    """
    Load a trained XLM-R sequence-classification model.

    The model directory is expected to contain the standard Hugging Face
    files generated during training.
    """

    print(
        f"\nLoading model from:\n"
        f"  {model_path}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model


# ------------------------------------------------------------------
# Per-case evaluation
# ------------------------------------------------------------------

def evaluate_one_case(
    train_dataset,
    train_llm,
    test_dataset,
    test_llm,
    test_cache,
):
    """
    Evaluate one:

        train_dataset + train_llm
                    ->
        test_dataset + test_llm

    combination.
    """

    print("\n" + "=" * 75)

    print(
        f"TRAIN={train_dataset}+{train_llm} | "
        f"TEST(new_prompt)={test_dataset}+{test_llm}"
    )

    print("=" * 75)

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model_path = os.path.join(
        TRAINED_MODELS_ROOT,
        f"{train_dataset}_{train_llm}",
    )

    if not os.path.isdir(model_path):

        print(
            f"Model directory not found:\n"
            f"  {model_path}"
        )

        return None

    tokenizer, model = load_xlm_roberta_model(
        model_path
    )

    # --------------------------------------------------------------
    # Cached test data
    # --------------------------------------------------------------

    df_test, texts, raw_labels = test_cache

    # --------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------

    inference = run_model_inference(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        raw_labels=raw_labels,
        batch_size=16,
        max_len=512,
    )

    true_labels = inference["true_labels"]
    predicted_labels = inference["predicted_labels"]
    predicted_prob_label1 = inference[
        "predicted_prob_label1"
    ]

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    metrics = evaluate_predictions(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        predicted_prob_label1=predicted_prob_label1,
    )

    print(
        f"Accuracy:      {metrics['Accuracy']:.4f}"
    )

    print(
        f"F1 Score:      {metrics['F1 Score']:.4f}"
    )

    print(
        f"AUROC:         {metrics['AUROC']:.4f}"
    )

    print(
        f"AI Recall:     {metrics['AI Recall']:.4f}"
    )

    print(
        f"Human Recall:  {metrics['Human Recall']:.4f}"
    )

    # --------------------------------------------------------------
    # Classification report
    # --------------------------------------------------------------

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=[0, 1],
            target_names=[
                "Human",
                "AI",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------------
    # Result directory
    # --------------------------------------------------------------

    result_dir = os.path.join(
        RESULTS_ROOT,
        f"{train_dataset}_{train_llm}_tested_on_"
        f"{test_dataset}_{test_llm}",
    )

    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Metrics CSV
    # --------------------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "Train Dataset": train_dataset,
                "Train LLM": train_llm,
                "Test Dataset": test_dataset,
                "Test LLM": test_llm,
                "Accuracy": metrics["Accuracy"],
                "F1 Score": metrics["F1 Score"],
                "AUROC": metrics["AUROC"],
                "AI Recall": metrics["AI Recall"],
                "Human Recall": metrics["Human Recall"],
            }
        ]
    )

    metrics_path = os.path.join(
        result_dir,
        "metrics.csv",
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Predictions CSV
    # --------------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "text": texts,

            # Canonical:
            #   1 = AI
            #   0 = Human
            "true_label": true_labels,

            # Canonical:
            #   1 = AI
            #   0 = Human
            "predicted_label": predicted_labels,

            # P(AI)
            "predicted_prob_label1": predicted_prob_label1,

            # P(Human)
            "predicted_prob_label0": inference[
                "predicted_prob_label0"
            ],
        }
    )

    predictions_path = os.path.join(
        result_dir,
        "predictions.csv",
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print(
        f"\nResults saved to:\n"
        f"  {result_dir}"
    )

    # --------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------

    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics_df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    os.makedirs(
        RESULTS_ROOT,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Six train combinations
    #
    # hindisumm + gemini
    # hindisumm + llama3
    # hindisumm + qwen3
    # xquad + gemini
    # xquad + llama3
    # xquad + qwen3
    # --------------------------------------------------------------

    all_combinations = list(
        product(
            DATASETS,
            LLMS,
        )
    )

    print(
        "\nTrain/test combinations:"
    )

    for dataset, llm in all_combinations:
        print(
            f"  {dataset}+{llm}"
        )

    # --------------------------------------------------------------
    # Load all six new-prompt test sets once.
    # --------------------------------------------------------------

    test_cache = {}

    print(
        "\n" + "=" * 75
    )

    print(
        "Loading new-prompt test sets"
    )

    print(
        "=" * 75
    )

    for test_dataset, test_llm in all_combinations:

        loaded = load_new_prompt_test_set(
            test_dataset,
            test_llm,
        )

        if loaded is not None:

            test_cache[
                (test_dataset, test_llm)
            ] = loaded

            _, texts, labels = loaded

            print(
                f"Loaded "
                f"{test_dataset}+{test_llm}: "
                f"{len(texts)} texts"
            )

    print(
        f"\nCached "
        f"{len(test_cache)}/{len(all_combinations)} "
        f"new-prompt test sets."
    )

    # --------------------------------------------------------------
    # 6 trained models x 6 test sets = 36 evaluations.
    # --------------------------------------------------------------

    total_cases = (
        len(all_combinations) ** 2
    )

    case_num = 0

    for train_dataset, train_llm in all_combinations:

        for test_dataset, test_llm in all_combinations:

            case_num += 1

            print(
                "\n" + "#" * 80
            )

            print(
                f"[{case_num}/{total_cases}] "
                f"TRAIN={train_dataset}+{train_llm} "
                f"TEST={test_dataset}+{test_llm}"
            )

            print(
                "#" * 80
            )

            cached = test_cache.get(
                (
                    test_dataset,
                    test_llm,
                )
            )

            if cached is None:

                print(
                    f"Skipping: no test data for "
                    f"{test_dataset}+{test_llm}"
                )

                continue

            try:

                evaluate_one_case(
                    train_dataset=train_dataset,
                    train_llm=train_llm,
                    test_dataset=test_dataset,
                    test_llm=test_llm,
                    test_cache=cached,
                )

            except Exception as exc:

                print(
                    f"FAILED case "
                    f"{train_dataset}+{train_llm} -> "
                    f"{test_dataset}+{test_llm}: "
                    f"{type(exc).__name__}: {exc}"
                )

    # --------------------------------------------------------------
    # Rebuild master CSV from all metrics.csv files.
    #
    # This is safer than repeatedly appending because rerunning the
    # script cannot accidentally duplicate rows in the master file.
    # --------------------------------------------------------------

    all_metrics = []

    for train_dataset, train_llm in all_combinations:

        for test_dataset, test_llm in all_combinations:

            result_dir = os.path.join(
                RESULTS_ROOT,
                f"{train_dataset}_{train_llm}_tested_on_"
                f"{test_dataset}_{test_llm}",
            )

            metrics_path = os.path.join(
                result_dir,
                "metrics.csv",
            )

            if os.path.exists(metrics_path):

                metrics_df = pd.read_csv(
                    metrics_path
                )

                all_metrics.append(
                    metrics_df
                )

    if all_metrics:

        master_df = pd.concat(
            all_metrics,
            ignore_index=True,
        )

        # Ensure deterministic ordering.
        master_df = master_df[
            [
                "Train Dataset",
                "Train LLM",
                "Test Dataset",
                "Test LLM",
                "Accuracy",
                "F1 Score",
                "AUROC",
                "AI Recall",
                "Human Recall",
            ]
        ]

        master_df.to_csv(
            MASTER_RESULTS_CSV,
            index=False,
        )

        print(
            "\n" + "=" * 75
        )

        print(
            "All new-prompt evaluations completed."
        )

        print(
            f"Evaluations written: "
            f"{len(master_df)}/{total_cases}"
        )

        print(
            f"Master results:\n"
            f"  {MASTER_RESULTS_CSV}"
        )

        print(
            "=" * 75
        )

    else:

        print(
            "\nNo evaluation results were generated."
        )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()