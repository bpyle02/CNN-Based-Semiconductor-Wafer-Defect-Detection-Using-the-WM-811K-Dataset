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

RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8501 8000

# Default: Run training
CMD ["python", "train.py", "--model", "all", "--epochs", "5", "--device", "cuda"]

# Stage 3: Production (minimal image)
FROM base as production

RUN pip install --upgrade pip

# Copy requirements and install (production only)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY train.py setup.py /app/
COPY src /app/src
COPY config.yaml /app/ 2>/dev/null || true

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Health check (device-agnostic: verifies torch imports and model loading)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; from src.models import WaferCNN; print('healthy')" || exit 1

# Run inference server
CMD ["python", "-m", "src.inference.server"]

# Stage 4: Jupyter (for interactive development)
FROM development as jupyter

RUN pip install jupyter jupyterlab ipywidgets

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
