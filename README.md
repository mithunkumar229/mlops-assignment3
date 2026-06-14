# End-to-End MLOps Pipeline

Docker · GitHub Actions · Kaggle · Weights & Biases · Hugging Face

## Project Overview

This repository contains a complete MLOps pipeline for fine-tuning a compact Hugging Face model, tracking experiments on Weights & Biases, packaging inference with Docker, and automating validation/inference using GitHub Actions.

## Repository Structure

```text
mlops-assignment3/
├── .github/workflows/
│   ├── ci.yml
│   └── inference.yml
├── configs/
│   └── train_config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── kaggle_training_template.ipynb
├── reports/
│   ├── report_template.md
│   └── screenshots/
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   └── inference.py
├── tests/
│   └── test_inference.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

```bash
python src/preprocess.py \
  --input data/raw/dataset.csv \
  --output data/processed/prepared_dataset.csv \
  --label-column label \
  --text-column text
```

This script creates:

```text
data/processed/prepared_dataset.csv
configs/id2label.json
configs/label2id.json
```

Only the label mapping files should be committed. Do not commit large datasets.

## Training

Training must be done on Kaggle with GPU enabled.

Use:

```text
notebooks/kaggle_training_template.ipynb
```

Required Kaggle Secrets:

```text
WANDB_API_KEY
HF_TOKEN
```

Run at least two versions with different hyperparameters, for example:

| Version | Epochs | Batch Size | Learning Rate |
|---|---:|---:|---:|
| v1 | 3 | 16 | 3e-5 |
| v2 | 4 | 16 | 2e-5 |

## Inference Locally

```bash
export HF_MODEL_NAME="your-hf-username/your-model-repo"
export INPUT_TEXT="Sample text to classify"
python src/inference.py
```

## Docker

Build:

```bash
docker build --build-arg HF_MODEL_NAME=your-hf-username/your-model-repo -t mlops-a3-inference:latest .
```

Run:

```bash
docker run --rm \
  -e HF_TOKEN=<your_token_optional_if_public_model> \
  -e INPUT_TEXT="Sample text" \
  mlops-a3-inference:latest
```

Push:

```bash
docker tag mlops-a3-inference:latest your-dockerhub/mlops-a3-inference:latest
docker push your-dockerhub/mlops-a3-inference:latest
```

## GitHub Actions

Two workflows are included:

1. `ci.yml`  
   Runs linting on push to `develop` and pull request to `main`.

2. `inference.yml`  
   Manually runs inference using `workflow_dispatch`.

Add these GitHub repository secrets:

```text
HF_TOKEN
WANDB_API_KEY
```

## Required Public Links for Final Report

- GitHub Repository:
- Kaggle Notebook v1:
- Kaggle Notebook v2:
- Hugging Face Model:
- Docker Image:
- W&B Dashboard:

## Team Contributions

| Student Name | Roll Number | Contribution |
|---|---|---|
| Student 1 |  | Repository setup, CI/CD |
| Student 2 |  | Data preparation |
| Student 3 |  | Model training and W&B |
| Student 4 |  | Docker and report |
```



