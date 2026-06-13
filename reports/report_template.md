# End-to-End MLOps Pipeline Report

## 1. Student Details and Contributions

| Name | Roll Number | Contribution |
|---|---|---|
| Student 1 |  |  |
| Student 2 |  |  |
| Student 3 |  |  |
| Student 4 |  |  |

## 2. Live Project Links

| Component | Link |
|---|---|
| GitHub Repository |  |
| Kaggle Notebook v1 |  |
| Kaggle Notebook v2 |  |
| Hugging Face Model |  |
| Docker Image |  |
| W&B Dashboard |  |

## 3. Git Repository Setup

Repository was created as a public GitHub repository with README, LICENSE, `.gitignore`, source folders, GitHub Actions workflows, and branch protection.

Add screenshot here:

```text
reports/screenshots/github-collaborators.png
reports/screenshots/branch-protection.png
```

## 4. Data Cleaning and Normalisation

Dataset selected:

Cleaning decisions:

- Removed missing text and label rows.
- Lowercased text.
- Removed URLs and punctuation.
- Removed duplicate text-label pairs.
- Encoded labels into numeric IDs.
- Saved `id2label.json` and `label2id.json`.

Class distribution before and after cleaning:

| Class | Raw Count | Cleaned Count |
|---|---:|---:|
|  |  |  |

## 5. Model Selection Rationale

Model selected:

Model card reference:

Justification, 100–150 words:

> We selected this model because...

## 6. Experiment Comparison

| Metric | Version 1 | Version 2 |
|---|---:|---:|
| Epochs |  |  |
| Batch Size |  |  |
| Learning Rate |  |  |
| Validation Loss |  |  |
| Accuracy |  |  |
| F1 Score |  |  |

Best version:

Reason:

## 7. W&B Tracking

Add screenshot:

```text
reports/screenshots/wandb-runs-comparison.png
```

## 8. Docker Design

Docker choices:

- Used `python:3.11-slim`.
- Installed only inference dependencies.
- Used `ARG HF_MODEL_NAME` to make model configurable at build time.
- Inference accepts `INPUT_TEXT` through environment variable.

Successful Docker run log:

```text
Paste Docker output here.
```

## 9. GitHub Actions

CI workflow:

Inference workflow:

Successful inference run screenshot or badge:

```text
reports/screenshots/github-actions-inference.png
```

Actions log:

```text
Paste successful inference workflow log here.
```

## 10. Challenges and Learnings

Challenges:

Learnings:

What we would do differently:
