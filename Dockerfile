FROM python:3.11-slim

ARG HF_MODEL_NAME=your-hf-username/your-model-repo
ENV HF_MODEL_NAME=${HF_MODEL_NAME}
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir transformers torch huggingface-hub

COPY src/ ./src/

CMD ["python", "src/inference.py"]
