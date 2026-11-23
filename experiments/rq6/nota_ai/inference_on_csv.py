import os
import os
import torch
import pandas as pd
import langid
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from transformers import AutoTokenizer
from utils import LLMFeatureExtractor
from model import MGTDetectionModel
import warnings

warnings.filterwarnings("ignore")

HF_TOKEN = os.environ.get("HF_TOKEN")
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

INPUT_ROOT = "testing_data"
OUTPUT_ROOT = "inference_data"
MASTER_CSV = "master_results.csv"


# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
def get_models():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    unseen_model = MGTDetectionModel()
    unseen_model.load_state_dict(torch.load("MGTDetectionModel.pt", map_location="cuda:0"))
    unseen_model = unseen_model.eval().to("cuda:0")

    return {"unseen": unseen_model}, tokenizer


# ---------------------------------------------------
# Inference on a single text
# ---------------------------------------------------
def run_inference_on_text(text, extractor, model_dict):
    detected_language, _ = langid.classify(text)
    seen_lang = ["en", "ru", "de", "zh", "ar", "bg", "id", "ur", "it"]

    # SEEN languages disabled
    # if detected_language in seen_lang:
    #     return None, None

    # Extract features
    input_dict = extractor.feature_extract(text)
    input_tensors = {k: v.to("cuda:0") for k, v in input_dict.items()}

    with torch.no_grad():
        outputs = model_dict["unseen"](**input_tensors)

    logits = outputs.logits[0]
    pred = logits.argmax().item()
    pred = 1 - pred  # Flip: 
    score = torch.softmax(logits, dim=-1)[1].item()  # probability Machine label

    label = 1 if pred == 1 else 0  # 0=Machine, 1=Human
    return label, score


# ---------------------------------------------------
# Compute metrics
# ---------------------------------------------------
def compute_metrics(true, pred, prob):
    accuracy = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    auc = roc_auc_score(true, prob)

    # Recalls
    ai_recall = recall_score(true, pred, pos_label=0)
    human_recall = recall_score(true, pred, pos_label=1)

    return accuracy, f1, auc, ai_recall, human_recall


# ---------------------------------------------------
# Process a single CSV File
# ---------------------------------------------------
def process_csv(file_path, extractor, model_dict, tokenizer):
    print(f"\n📄 Processing: {file_path}")

    df = pd.read_csv(file_path)
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        print(f"❌ Missing required columns in {file_path}")
        return None

    predictions = []
    probabilities = []

    for idx, row in df.iterrows():
        text = str(row[TEXT_COLUMN])
        pred, prob = run_inference_on_text(text, extractor, model_dict)
        predictions.append(pred)
        probabilities.append(prob)

        if idx % 10 == 0:
            print(f"  Processed {idx} rows...")

    df["prediction"] = predictions
    df["machine_prob"] = probabilities

    # Output path mirroring structure
    rel_path = file_path.replace(INPUT_ROOT, OUTPUT_ROOT)
    os.makedirs(os.path.dirname(rel_path), exist_ok=True)
    df.to_csv(rel_path, index=False)

    print(f"💾 Saved → {rel_path}")

    # Compute metrics
    y_true = df[LABEL_COLUMN].tolist()
    y_pred = predictions
    y_prob = probabilities

    accuracy, f1, auc, ai_recall, human_recall = compute_metrics(y_true, y_pred, y_prob)

    return {
        "file": file_path,
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": auc,
        "ai_recall(label=0)": ai_recall,
        "human_recall(label=1)": human_recall,
    }


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    print("\n🔍 Loading feature extractor + models ...")
    extractor = LLMFeatureExtractor(HF_TOKEN)
    model_dict, tokenizer = get_models()

    all_results = []

    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                metrics = process_csv(file_path, extractor, model_dict, tokenizer)
                if metrics:
                    all_results.append(metrics)

    # Save master CSV
    master_df = pd.DataFrame(all_results)
    master_df.to_csv(MASTER_CSV, index=False)

    print("\n📊 Master Results Saved →", MASTER_CSV)
    print("\n✅ ALL TASKS COMPLETED!\n")


if __name__ == "__main__":
    main()
