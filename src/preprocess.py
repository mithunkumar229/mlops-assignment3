"""
Data preparation script.

Example:
python src/preprocess.py \
  --input data/raw/dataset.csv \
  --output data/processed/prepared_dataset.csv \
  --text-column text \
  --label-column label
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def clean_text(value: str) -> str:
    """Basic text cleaning for text-classification datasets."""
    value = str(value).lower().strip()
    value = re.sub(r"http\S+|www\S+", "", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def preprocess(input_path: str, output_path: str, text_column: str, label_column: str) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    required_columns = {text_column, label_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Raw rows: {len(df)}")
    print("Class distribution before cleaning:")
    print(df[label_column].value_counts(dropna=False))

    df = df[[text_column, label_column]].copy()
    df = df.dropna(subset=[text_column, label_column])
    df[text_column] = df[text_column].apply(clean_text)
    df = df[df[text_column].str.len() > 0]
    df = df.drop_duplicates(subset=[text_column, label_column])

    labels = sorted(df[label_column].astype(str).unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    df[label_column] = df[label_column].astype(str)
    df["label_id"] = df[label_column].map(label2id)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    with open(config_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)

    with open(config_dir / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)

    print(f"Prepared rows: {len(df)}")
    print("Class distribution after cleaning:")
    print(df[label_column].value_counts())
    print(f"Saved prepared dataset to: {output_file}")
    print("Saved label mappings to configs/id2label.json and configs/label2id.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV file")
    parser.add_argument("--output", required=True, help="Path to cleaned CSV file")
    parser.add_argument("--text-column", default="text", help="Text column name")
    parser.add_argument("--label-column", default="label", help="Label column name")
    args = parser.parse_args()

    preprocess(args.input, args.output, args.text_column, args.label_column)


if __name__ == "__main__":
    main()
