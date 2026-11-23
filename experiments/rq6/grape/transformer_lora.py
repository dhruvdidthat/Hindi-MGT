import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# from huggingface_hub import login
# login(token="your token")  # ✅ keep as-is (original logic)

from datasets import Dataset, load_dataset
import pandas as pd
import evaluate
import numpy as np
np.random.seed(3407)
import time
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig
)
from scipy.special import softmax
import logging

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel
)

# ============================================================
# ✅ REPLACED ARGPARSE WITH VARIABLES (ONLY CHANGE REQUESTED)
# ============================================================

DEV_TEST_FILE_PATH = "./multilingual_devtest_text_id_only.jsonl"   # <-- change if needed
SUBTASK = "A"                                         # "A" or "B"
MODEL_NAME = "Qwen/Qwen2.5-1.5B"                       # base model to train/test
PREDICTION_FILE_PATH = "./predictions"                # unused in your script now but kept

RANDOM_SEED = 0
set_seed(RANDOM_SEED)

# ============================================================

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](
        examples["text"],
        padding='max_length',
        max_length=256,
        truncation=True
    )

def get_data(dev_test_path, random_seed):
    """
    Load train, validation, and test datasets using the datasets library.
    """
    print("Loading datasets...")

    ds = load_dataset("Jinyan1/COLING_2025_MGT_multingual")

    train_df = ds['train']
    dev_df = ds['dev']

    tmp = train_df.train_test_split(shuffle=True, seed=random_seed, test_size=0.1)
    train_df = tmp['train']
    val_df = tmp['test']

    dev_test_df = pd.read_json(dev_test_path, lines=True)
    print("Data loaded and split successfully.")

    return train_df, val_df, dev_df, dev_test_df

def compute_metrics(eval_pred):
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    results = {}
    results.update(f1_metric.compute(predictions=predictions, references=labels, average="micro"))
    return results

def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):
    train_dataset = train_df
    valid_dataset = valid_df

    tokenizer = AutoTokenizer.from_pretrained(model)

    # LoRA config (unchanged)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.1,
        bias='none',
        task_type='SEQ_CLS'
    )

    # load model (unchanged)
    model = AutoModelForSequenceClassification.from_pretrained(
        model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    ).bfloat16()

    model = get_peft_model(model, lora_config)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # tokenize data
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save best model
    best_model_path = checkpoints_path + '/best/'
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    trainer.save_model(best_model_path)

def test(test_df, model, tokenizer, id2label, label2id):
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    test_dataset = test_df
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(tokenized_test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)

    return results, preds

def prediction(test_df, model, tokenizer, id2label, label2id):
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    test_dataset = Dataset.from_pandas(test_df)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(tokenized_test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    return preds

# ============================================================
# ✅ MAIN FLOW (UNCHANGED LOGIC)
# ============================================================
if __name__ == "__main__":

    dev_test_path = DEV_TEST_FILE_PATH
    model_name = MODEL_NAME
    subtask = SUBTASK
    prediction_path = PREDICTION_FILE_PATH
    random_seed = RANDOM_SEED

    if subtask == "A":
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}
    elif subtask == "B":
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
        label2id = {'human': 0, 'chatGPT': 1, 'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    else:
        logging.error(f"Wrong subtask: {subtask}. It should be A or B")
        raise ValueError(f"Wrong subtask: {subtask}. It should be A or B")

    print("Loading data .... \n")
    train_df, valid_df, test_df, dev_test_df = get_data(dev_test_path, random_seed)

    start = time.time()
    fine_tune(train_df, valid_df, f"{model_name}/subtask{subtask}/{random_seed}", id2label, label2id, model_name)
    end = time.time()
    print(f"\n✅ Training finished in {end - start:.2f} seconds\n")

    # ============================================================
    # Original placeholders kept, but now auto-filled correctly ✅
    # ============================================================
    base_model_path = model_name
    adapter_path = f"{model_name}/subtask{subtask}/{random_seed}/best/"

    print("Loading base model + adapter...")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    ).bfloat16()

    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    results, predictions_out = test(test_df, model, tokenizer, id2label, label2id)
    logging.info(results)
    print("\n✅ TEST RESULTS:")
    print(results)

    print("\nstart to infer on dev_test")
    prediction_dev_test = prediction(dev_test_df, model, tokenizer, id2label, label2id)
    prediction_dev_test_df = pd.DataFrame({'testset_id': dev_test_df['testset_id'], 'label': prediction_dev_test})

    out_json_path = "./dev_test_predictions.jsonl"
    prediction_dev_test_df.to_json(out_json_path, lines=True, orient='records')
    print(f"\n✅ Saved dev_test predictions at: {out_json_path}")
