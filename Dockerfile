# arXiv Research Assistant - Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Install system dependencies for docling/rapidocr
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and install dependencies with uv
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml --no-cache

# Copy application code
COPY src/ ./src/
COPY script/ ./script/

# Create logs directory and set permissions
RUN mkdir -p /app/log && chmod +x script/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["script/docker-entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
