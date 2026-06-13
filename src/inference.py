"""
Inference script.

Environment variables:
HF_MODEL_NAME: Hugging Face model repo name
INPUT_TEXT: text to classify
HF_TOKEN: optional token, needed only for private models
"""

import os
import sys

from transformers import pipeline


def main() -> None:
    model_name = os.getenv("HF_MODEL_NAME", "your-hf-username/your-model-repo")
    input_text = os.getenv("INPUT_TEXT")

    if not input_text:
        input_text = "This is a sample input for classification."

    if model_name == "your-hf-username/your-model-repo":
        print("ERROR: Please set HF_MODEL_NAME to your public Hugging Face model repo.")
        sys.exit(1)

    hf_token = os.getenv("HF_TOKEN")

    classifier = pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        token=hf_token,
    )

    result = classifier(input_text)
    print("Input:", input_text)
    print("Prediction:", result)


if __name__ == "__main__":
    main()
