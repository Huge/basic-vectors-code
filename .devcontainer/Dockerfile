FROM python:3.9-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends
    build-essential
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir sentence-transformers numpy scikit-learn