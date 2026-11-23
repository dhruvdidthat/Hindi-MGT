"""
Hindi adaptation of Text Fluoroscopy (Yu et al., EMNLP 2024).
Trains one detector per (dataset, generator LLM) pair.

Expects CSVs at training_data/<dataset>/<dataset>_train_<llm>.csv with
columns "text" and "label" (label: 1 = machine-generated, 0 = human;
see POSITIVE_IS_MACHINE in text_fluoroscopy_common.py).
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from text_fluoroscopy_common import (
    MODEL_NAME,
    load_tokenizer_and_mlm_model,
    get_vocab_head,
    encode_text,
    select_layer_and_feature,
    BinaryClassifier,
)

SEED = 42
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Official-implementation hyperparameters (paper's Section 4.1 + released code)
LEARNING_RATE = 0.003
BATCH_SIZE = 16
NUM_EPOCHS = 10
DROPOUT_PROB = 0.4
VAL_FRACTION = 1 / 9  # 8:1 train:val split overall


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_features(texts, tokenizer, model, vocab_head, max_length=512, log_every=100):
    """Runs the encoder + layer-selection step for every text, returns a
    (N, hidden_dim) feature matrix built from the dynamically-selected layer."""
    feats = []
    for i, text in enumerate(texts):
        hidden_states, last_idx = encode_text(text, tokenizer, model, device, max_length)
        _, feature = select_layer_and_feature(hidden_states, last_idx, vocab_head)
        feats.append(feature.cpu().numpy())
        if i % log_every == 0:
            print(f"  processed {i}/{len(texts)}")
    return np.stack(feats)


def train_one_detector(dataset_name, llm, tokenizer, model, vocab_head):
    print(f"\n=== Training detector: dataset={dataset_name}, llm={llm} ===")

    train_path = f"training_data/{dataset_name}/{dataset_name}_train_{llm}.csv"
    save_dir = f"text_fluoroscopy/trained_models/{dataset_name}_{llm}"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "model.pt")

    df = pd.read_csv(train_path).dropna(subset=["text"])
    texts_all = df["text"].tolist()
    labels_all = df["label"].tolist()

    # 8:1 train/validation split (paper protocol), stratified on label
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts_all, labels_all, test_size=VAL_FRACTION, random_state=SEED, stratify=labels_all
    )
    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    print("Extracting train features...")
    X_train = build_features(train_texts, tokenizer, model, vocab_head)
    y_train = np.array(train_labels, dtype=np.float32)

    print("Extracting val features...")
    X_val = build_features(val_texts, tokenizer, model, vocab_head)
    y_val = np.array(val_labels, dtype=np.float32)

    classifier = BinaryClassifier(X_train.shape[1], dropout_prob=DROPOUT_PROB).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        classifier.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)

        # ---- validation: select the checkpoint, don't just watch loss ----
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(X_val_t).cpu().numpy()
        val_probs = 1 / (1 + np.exp(-val_logits))
        val_auc = roc_auc_score(y_val, val_probs)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  train_loss={epoch_loss:.4f}  val_AUROC={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

    torch.save(
        {
            "model_state_dict": best_state,
            "best_val_auc": best_val_auc,
            "input_size": X_train.shape[1],
            "dropout_prob": DROPOUT_PROB,
        },
        model_save_path,
    )
    print(f"Saved BEST checkpoint (val_AUROC={best_val_auc:.4f}) to {model_save_path}")


def main():
    set_seed(SEED)
    tokenizer, model = load_tokenizer_and_mlm_model(MODEL_NAME, device)
    vocab_head = get_vocab_head(model)

    dataset_list = ["hindisumm", "xquad"]
    llm_list = ["gemini", "llama3", "qwen3"]

    for dataset_name in dataset_list:
        for llm in llm_list:
            train_one_detector(dataset_name, llm, tokenizer, model, vocab_head)


if __name__ == "__main__":
    main()
