import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# CONFIG
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

INPUT_ROOT = "testing_data"
OUTPUT_ROOT = "inference_data"
MASTER_CSV = "master_results.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
ID_COLUMN = "id"  # optional

BATCH_SIZE = 8
MAX_LENGTH = 256

# Subtask A labels
id2label = {0: "human", 1: "machine"}
label2id = {"human": 0, "machine": 1}

# Base models
LLAMA_BASE = "meta-llama/Llama-3.2-1B"
QWEN_BASE = "Qwen/Qwen2.5-1.5B-Instruct"

# LoRA adapters
LLAMA_ADAPTER = "./meta-llama_local/Llama-3.2-1B/subtaskA/0/checkpoint-37918"
QWEN_ADAPTER = "./Qwen_local/Qwen2.5-1.5B/subtaskA/0/checkpoint-37918"

# Ensemble head checkpoint (your trained linear layer)
ENSEMBLE_CKPT = "./ensemble/mono/checkpoints/best_checkpoint.pth"


# -----------------------------
# Dataset for inference
# -----------------------------
class EnsembleInferenceDataset(Dataset):
    def __init__(self, texts, tokenizer1, tokenizer3, ids=None, max_length=256):
        self.texts = texts
        self.tokenizer1 = tokenizer1
        self.tokenizer3 = tokenizer3
        self.ids = ids
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        inputs_model1 = self.tokenizer1(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs_model3 = self.tokenizer3(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs_model1 = {k: v.squeeze(0) for k, v in inputs_model1.items()}
        inputs_model3 = {k: v.squeeze(0) for k, v in inputs_model3.items()}

        out = {
            "input_model1": inputs_model1,
            "input_model3": inputs_model3,
        }

        if self.ids is not None:
            out["id"] = self.ids[idx]

        return out


def collate_fn(batch):
    input_model1 = {
        k: torch.stack([b["input_model1"][k] for b in batch]) for k in batch[0]["input_model1"]
    }
    input_model3 = {
        k: torch.stack([b["input_model3"][k] for b in batch]) for k in batch[0]["input_model3"]
    }

    result = {
        "input_model1": input_model1,
        "input_model3": input_model3,
    }

    if "id" in batch[0]:
        result["id"] = [b["id"] for b in batch]

    return result


# -----------------------------
# Ensemble Model (binary output)
# -----------------------------
class EnsembleModel(nn.Module):
    """
    Subtask A:
    model1 logits: [B,2]
    model3 logits: [B,2]
    concat: [B,4]
    linear: [B,1] -> sigmoid for machine prob
    """

    def __init__(self, model1, model3):
        super().__init__()
        self.model1_model = model1
        self.model3_model = model3

        # Freeze base models
        for p in self.model1_model.parameters():
            p.requires_grad = False
        for p in self.model3_model.parameters():
            p.requires_grad = False

        self.linear = nn.Linear(4, 1)

    def forward(self, input_model1, input_model3):
        device = next(self.linear.parameters()).device

        input_model1 = {k: v.to(device) for k, v in input_model1.items()}
        input_model3 = {k: v.to(device) for k, v in input_model3.items()}

        logits1 = self.model1_model(**input_model1).logits  # [B,2]
        logits3 = self.model3_model(**input_model3).logits  # [B,2]

        concat = torch.cat([logits1, logits3], dim=1)        # [B,4]
        out = self.linear(concat).squeeze(1)                # [B]
        return out


# -----------------------------
# Load Models
# -----------------------------
def load_ensemble_system():
    print("\n🔄 Loading base models + adapters...")

    # --- LLaMA ---
    model1 = AutoModelForSequenceClassification.from_pretrained(
        LLAMA_BASE,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model1 = PeftModel.from_pretrained(model1, LLAMA_ADAPTER)

    tok1 = AutoTokenizer.from_pretrained(LLAMA_BASE)
    tok1.pad_token = tok1.eos_token
    model1.config.pad_token_id = model1.config.eos_token_id

    # --- Qwen ---
    model3 = AutoModelForSequenceClassification.from_pretrained(
        QWEN_BASE,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
    )
    model3 = PeftModel.from_pretrained(model3, QWEN_ADAPTER)

    tok3 = AutoTokenizer.from_pretrained(QWEN_BASE, trust_remote_code=True)
    tok3.pad_token = tok3.eos_token
    model3.config.pad_token_id = model3.config.eos_token_id

    # --- Ensemble ---
    ensemble = EnsembleModel(model1, model3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble.to(device)
    ensemble.eval()

    # Load ensemble checkpoint (linear head)
    print("✅ Loading ensemble head:", ENSEMBLE_CKPT)
    state = torch.load(ENSEMBLE_CKPT, map_location=device)
    ensemble.load_state_dict(state, strict=True)

    return ensemble, tok1, tok3, device


# -----------------------------
# Run inference on 1 CSV
# -----------------------------
@torch.no_grad()
def infer_on_csv(file_path, ensemble, tok1, tok3, device):
    print(f"\n📄 Processing: {file_path}")
    df = pd.read_csv(file_path)

    if TEXT_COLUMN not in df.columns:
        print(f"❌ Missing '{TEXT_COLUMN}' column. Skipping.")
        return None

    texts = df[TEXT_COLUMN].astype(str).tolist()

    ids = None
    if ID_COLUMN in df.columns:
        ids = df[ID_COLUMN].tolist()

    dataset = EnsembleInferenceDataset(texts, tok1, tok3, ids=ids, max_length=MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    preds = []
    probs = []

    for batch in tqdm(loader, desc="Inference", leave=False):
        out = ensemble(batch["input_model1"], batch["input_model3"])  # [B]
        prob_machine = torch.sigmoid(out).detach().cpu().numpy()
        pred_machine = (prob_machine > 0.5).astype(int)

        probs.extend(prob_machine.tolist())
        preds.extend(pred_machine.tolist())

    df["prediction"] = preds
    df["machine_prob"] = probs

    # Save output CSV with mirrored structure
    rel_path = file_path.replace(INPUT_ROOT, OUTPUT_ROOT)
    os.makedirs(os.path.dirname(rel_path), exist_ok=True)
    df.to_csv(rel_path, index=False)
    print(f"💾 Saved → {rel_path}")

    # Metrics if label exists
    if LABEL_COLUMN in df.columns:
        df[LABEL_COLUMN] = 1 - df[LABEL_COLUMN]
        y_true = df[LABEL_COLUMN].tolist()
        y_pred = preds
        y_prob = probs

        # safety for AUC if single class exists
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        ai_recall = recall_score(y_true, y_pred, pos_label=1)    # machine recall
        human_recall = recall_score(y_true, y_pred, pos_label=0) # human recall

        return {
            "file": file_path,
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc": auc,
            "machine_recall(label=1)": ai_recall,
            "human_recall(label=0)": human_recall,
        }

    # no label → return None metrics but still keep output saved
    return {
        "file": file_path,
        "accuracy": None,
        "f1_score": None,
        "roc_auc": None,
        "machine_recall(label=1)": None,
        "human_recall(label=0)": None,
    }


# -----------------------------
# MAIN
# -----------------------------
def main():
    ensemble, tok1, tok3, device = load_ensemble_system()
    all_results = []

    for root, _, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(root, file)
                metrics = infer_on_csv(path, ensemble, tok1, tok3, device)
                if metrics:
                    all_results.append(metrics)

    master_df = pd.DataFrame(all_results)
    master_df.to_csv(MASTER_CSV, index=False)

    print("\n📊 Master Results Saved →", MASTER_CSV)
    print("\n✅ ALL FILES PROCESSED!\n")


if __name__ == "__main__":
    main()
