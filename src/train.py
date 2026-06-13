"""
Training script for Kaggle or local GPU environment.

Training for submission should be run on Kaggle, not GitHub Actions.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import wandb
import yaml
from datasets import Dataset
from huggingface_hub import login
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--data", default="data/processed/prepared_dataset.csv")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open("configs/id2label.json", "r", encoding="utf-8") as f:
        id2label_raw = json.load(f)

    with open("configs/label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)

    id2label = {int(k): v for k, v in id2label_raw.items()}

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    model_name = config["model_name"]
    text_column = config["text_column"]
    training_cfg = config["training"]

    df = pd.read_csv(args.data)

    train_df, eval_df = train_test_split(
        df,
        test_size=training_cfg["test_size"],
        random_state=training_cfg["random_state"],
        stratify=df["label_id"],
    )

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch[text_column], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column("label_id", "labels")
    eval_dataset = eval_dataset.rename_column("label_id", "labels")

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    run_name = f"run-{training_cfg['version']}"

    wandb.init(
        project=config["project_name"],
        name=run_name,
        config={
            "model": model_name,
            "epochs": training_cfg["epochs"],
            "batch_size": training_cfg["batch_size"],
            "learning_rate": training_cfg["learning_rate"],
            "version": training_cfg["version"],
            "platform": "Kaggle",
        },
    )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=training_cfg["epochs"],
        per_device_train_batch_size=training_cfg["batch_size"],
        per_device_eval_batch_size=training_cfg["batch_size"],
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=run_name,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    hf_repo_name = config["hf_repo_name"]
    if hf_repo_name and hf_repo_name != "your-hf-username/your-model-repo":
        model.push_to_hub(hf_repo_name)
        tokenizer.push_to_hub(hf_repo_name)
        wandb.run.summary["huggingface_model"] = f"https://huggingface.co/{hf_repo_name}"

    wandb.finish()


if __name__ == "__main__":
    main()
