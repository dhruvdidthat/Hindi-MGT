"""
Inference script for the Hindi Text Fluoroscopy adaptation.
See text_fluoroscopy_train_fixed.py for the list of fixes applied — the
feature extraction here (vocab-space KL layer selection + last-token
pooling) must exactly mirror what was used at training time.

Label convention:
    1 = machine-generated / AI text
    0 = human-written text

"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, classification_report

from text_fluoroscopy_common import (
    MODEL_NAME,
    POSITIVE_IS_MACHINE,
    load_tokenizer_and_mlm_model,
    get_vocab_head,
    encode_text,
    select_layer_and_feature,
    BinaryClassifier,
)

device = "cuda:1" if torch.cuda.is_available() else "cpu"

MASTER_CSV_PATH = "text_fluoroscopy/inference_results/text_fluoroscopy_all_results.csv"


def build_features(texts, tokenizer, model, vocab_head, max_length=512, log_every=100):
    feats = []
    for i, text in enumerate(texts):
        hidden_states, last_idx = encode_text(text, tokenizer, model, device, max_length)
        _, feature = select_layer_and_feature(hidden_states, last_idx, vocab_head)
        feats.append(feature.cpu().numpy())
        if i % log_every == 0:
            print(f"  processed {i}/{len(texts)}")
    return np.stack(feats)


def append_to_master_csv(df):
    os.makedirs(os.path.dirname(MASTER_CSV_PATH), exist_ok=True)
    if os.path.exists(MASTER_CSV_PATH):
        df.to_csv(MASTER_CSV_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(MASTER_CSV_PATH, index=False)


def run_inference(train_dataset, train_llm, test_dataset, test_llm, tokenizer, model, vocab_head):
    print(f"\n=== Evaluating: TRAIN={train_dataset}+{train_llm} | TEST={test_dataset}+{test_llm} ===")

    save_dir = f"text_fluoroscopy/trained_models/{train_dataset}_{train_llm}"
    model_path = os.path.join(save_dir, "model.pt")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    test_path = f"testing_data/{test_dataset}/{test_dataset}_test_{test_llm}.csv"
    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}")
        return

    test_df = pd.read_csv(test_path).dropna(subset=["text"])
    texts = test_df["text"].tolist()
    labels = np.array(test_df["label"].tolist(), dtype=np.float32)

    X_test = build_features(texts, tokenizer, model, vocab_head)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    classifier = BinaryClassifier(
        checkpoint["input_size"], dropout_prob=checkpoint.get("dropout_prob", 0.4)
    ).to(device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    with torch.no_grad():
        logits = classifier(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
    probs_label1 = 1 / (1 + np.exp(-logits))  # P(label == 1)

    # AUROC must be P(machine-generated scores higher). Flip if label 1 == human.
    machine_probs = probs_label1 if POSITIVE_IS_MACHINE else (1 - probs_label1)
    machine_labels = labels if POSITIVE_IS_MACHINE else (1 - labels)

    y_pred = (probs_label1 > 0.5).astype(int)

    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    rocauc = roc_auc_score(machine_labels, machine_probs)
    human_pos_label = 0 if POSITIVE_IS_MACHINE else 1
    ai_pos_label = 1 if POSITIVE_IS_MACHINE else 0
    human_recall = recall_score(labels, y_pred, pos_label=human_pos_label)
    ai_recall = recall_score(labels, y_pred, pos_label=ai_pos_label)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC (P=machine-generated): {rocauc:.4f}")
    print(f"Human Recall: {human_recall:.4f}")
    print(f"AI Recall: {ai_recall:.4f}")
    print(
        "\nClassification Report:\n",
        classification_report(labels, y_pred, target_names=["Human", "AI"] if POSITIVE_IS_MACHINE else ["AI", "Human"]),
    )

    result_dir = f"text_fluoroscopy/inference_results/{train_dataset}_{train_llm}_tested_on_{test_dataset}_{test_llm}"
    os.makedirs(result_dir, exist_ok=True)

    metrics = pd.DataFrame(
        [
            {
                "Train Dataset": train_dataset,
                "Train LLM": train_llm,
                "Test Dataset": test_dataset,
                "Test LLM": test_llm,
                "Accuracy": acc,
                "F1 Score": f1,
                "AUROC": rocauc,
                "AI Recall": ai_recall,
                "Human Recall": human_recall,
            }
        ]
    )
    metrics.to_csv(os.path.join(result_dir, "metrics.csv"), index=False)
    append_to_master_csv(metrics)

    preds_df = pd.DataFrame(
        {
            "text": texts,
            "true_label": labels,
            "predicted_label": y_pred,
            "predicted_prob_label1": probs_label1,
        }
    )
    preds_df.to_csv(os.path.join(result_dir, "predictions.csv"), index=False)
    print(f"Results saved to {result_dir}")


def main():
    tokenizer, model = load_tokenizer_and_mlm_model(MODEL_NAME, device)
    vocab_head = get_vocab_head(model)

    datasets = ["hindisumm", "xquad"]
    llms = ["gemini", "llama3", "qwen3"]

    for train_dataset in datasets:
        for train_llm in llms:
            run_inference(train_dataset, train_llm, train_dataset, train_llm, tokenizer, model, vocab_head)
            for test_llm in [l for l in llms if l != train_llm]:
                run_inference(train_dataset, train_llm, train_dataset, test_llm, tokenizer, model, vocab_head)
            other_dataset = [d for d in datasets if d != train_dataset][0]
            run_inference(train_dataset, train_llm, other_dataset, train_llm, tokenizer, model, vocab_head)

    cross_domain_cases = [
        ("hindisumm", "xquad", "llama3"),
        ("hindisumm", "xquad", "qwen3"),
        ("xquad", "hindisumm", "llama3"),
        ("xquad", "hindisumm", "qwen3"),
    ]
    for train_dataset, test_dataset, llm in cross_domain_cases:
        run_inference(train_dataset, llm, test_dataset, llm, tokenizer, model, vocab_head)


if __name__ == "__main__":
    main()
