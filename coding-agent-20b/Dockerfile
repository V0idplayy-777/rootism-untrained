FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-dev.txt pyproject.toml ./
RUN pip install -r requirements.txt
COPY . .
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

FROM base AS train
CMD ["python", "training/train_production.py", "--help"]

FROM base AS serve
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://127.0.0.1:8000/health || exit 1
CMD ["python", "-m", "inference.server", "--checkpoint", "checkpoints/final.pt", "--port", "8000"]

FROM base AS eval
CMD ["python", "evaluation/production_evaluate.py", "--help"]
