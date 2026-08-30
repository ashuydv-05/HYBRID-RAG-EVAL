# Backend Dockerfile - FastAPI & LangGraph RAG Service
FROM python:3.12-slim

WORKDIR /app

# Prevent Python from writing .pyc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, data, and scripts
COPY src/ ./src/
COPY data/ ./data/
COPY script/ ./script/

# Create model cache directories with write permissions for non-root containers (e.g. Hugging Face Spaces)
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch /app/logs && \
    chmod -R 777 /app

EXPOSE 7860
EXPOSE 8000

# Run FastAPI backend with dynamic cloud PORT (defaults to 7860 on Hugging Face Spaces or 8000 locally)
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
