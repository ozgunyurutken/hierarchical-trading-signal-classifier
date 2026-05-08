FROM python:3.11-slim

WORKDIR /project

# System dependencies for compiled wheels (xgboost, lightgbm, hmmlearn, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project source
COPY src/ /project/src/
COPY app/ /project/app/
COPY config.yaml /project/config.yaml

# Pre-computed features + trained models needed for date-mode predictions.
# Bundled into the image so the demo runs offline.
COPY data/processed/btc_aligned.csv /project/data/processed/
COPY data/processed/btc_features_stage1.csv /project/data/processed/
COPY data/processed/btc_features_stage3.csv /project/data/processed/
COPY data/processed/btc_features_macro.csv /project/data/processed/

ENV PYTHONPATH=/project
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
