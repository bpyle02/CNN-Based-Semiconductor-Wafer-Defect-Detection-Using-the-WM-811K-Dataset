# syntax=docker/dockerfile:1.4
# Multi-stage Docker build for wafer defect detection
# Stage 1: Base image with CUDA
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04 as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Development (with dev tools)
FROM base as development

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Copy project files first so the editable install has the package tree
COPY . .

# Editable install with full dev extras (packaging defined in pyproject.toml)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[dev]"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8501 8000

# Default: Run training
CMD ["python", "train.py", "--model", "all", "--epochs", "5", "--device", "cuda"]

# Stage 3: Production (minimal image)
# Aliased as both `prod` and `production` so CI (`--target prod`) and legacy
# callers (`--target production`) both resolve.
FROM base as prod

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# Copy only files needed for the production serving stack
COPY pyproject.toml setup.py /app/
COPY train.py /app/
COPY src /app/src
# config.yaml is required; if missing the build fails loudly instead of silently
# producing a broken image. Keep a committed config.yaml at the repo root.
COPY config.yaml /app/config.yaml

# Production install: core + server extras only (FastAPI, uvicorn)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[server]"

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Health check (device-agnostic: verifies torch imports and model loading)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; from src.models import WaferCNN; print('healthy')" || exit 1

# Run inference server
CMD ["python", "-m", "src.inference.server"]

# Back-compat alias for callers that still reference `production`.
FROM prod as production

# Stage 4: Jupyter (for interactive development)
FROM development as jupyter

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install jupyter jupyterlab ipywidgets

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
