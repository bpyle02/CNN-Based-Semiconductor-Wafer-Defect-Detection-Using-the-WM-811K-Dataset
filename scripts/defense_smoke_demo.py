"""Defense smoke demo for the wafer-defect repository.

This script exercises the inference stack without requiring an external server.
If FastAPI is available, it loads a temporary CNN checkpoint, runs prediction on
a synthetic wafer-style image, requests a Grad-CAM overlay, and writes the
result to ``docs/generated/defense_demo_gradcam.png``.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MISSING_DEPENDENCY = None

try:
    import numpy as np
    import torch
    from PIL import Image
    from src.models import WaferCNN
except ImportError as exc:
    np = None
    torch = None
    Image = None
    WaferCNN = None
    MISSING_DEPENDENCY = exc

try:
    from fastapi.testclient import TestClient
    from src.inference.server import create_app

    FASTAPI_AVAILABLE = True
except ImportError:
    TestClient = None
    create_app = None
    FASTAPI_AVAILABLE = False


def create_synthetic_wafer(size: int = 96) -> np.ndarray:
    """Create a simple synthetic wafer-style image for smoke testing."""
    image = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2.0
    radius = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    max_radius = np.max(radius)
    image = np.clip(1.0 - radius / max_radius, 0.0, 1.0)
    image[(yy - center) ** 2 + (xx - center) ** 2 < (size * 0.10) ** 2] = 1.0
    return image


def image_to_base64(image_array: np.ndarray) -> str:
    """Encode a grayscale image to base64 PNG."""
    image = Image.fromarray((np.clip(image_array, 0.0, 1.0) * 255).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def ensure_checkpoint(checkpoint_path: Path) -> Path:
    """Create a temporary CNN checkpoint if one is not supplied."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        if WaferCNN is None or torch is None:
            raise RuntimeError("Torch-based model creation is unavailable.")
        model = WaferCNN(num_classes=9)
        torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def run_demo(checkpoint_path: Path) -> int:
    """Run the smoke demo using FastAPI's TestClient."""
    if not FASTAPI_AVAILABLE:
        logger.warning("fastapi is not installed. Install requirements.txt to run the demo.")
        return 1

    checkpoint_path = ensure_checkpoint(checkpoint_path)
    app = create_app(device="cpu")
    client = TestClient(app)

    load_response = client.post(
        "/load_model",
        json={
            "model_type": "cnn",
            "checkpoint_path": str(checkpoint_path),
        },
    )
    load_response.raise_for_status()
    model_name = load_response.json()["model_name"]

    image_array = create_synthetic_wafer()
    predict_response = client.post(
        "/predict",
        json={
            "image_base64": image_to_base64(image_array),
            "model_name": model_name,
            "return_gradcam": True,
        },
    )
    predict_response.raise_for_status()
    payload = predict_response.json()

    output_dir = Path("docs/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    gradcam_path = output_dir / "defense_demo_gradcam.png"
    gradcam_bytes = base64.b64decode(payload["gradcam_base64"])
    gradcam_path.write_bytes(gradcam_bytes)

    logger.info("Defense smoke demo completed.")
    logger.info(f"Loaded model: {model_name}")
    logger.info(f"Predicted class: {payload['class_name']}")
    logger.info(f"Confidence: {payload['confidence']:.4f}")
    logger.info(f"Inference time (ms): {payload['inference_ms']:.2f}")
    logger.info(f"GradCAM written to: {gradcam_path}")
    return 0


def main() -> int:
    if MISSING_DEPENDENCY is not None:
        logger.warning(
            "Missing runtime dependency for defense_smoke_demo.py: "
            f"{MISSING_DEPENDENCY}. Use the same Python environment that "
            'successfully runs `pytest -q` after `python -m pip install -e ".[dev]"`.'
        )
        return 1

    parser = argparse.ArgumentParser(description="Run a defense smoke demo.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/defense_demo_cnn.pth"),
        help="Optional checkpoint path. A temporary CNN checkpoint is created if missing.",
    )
    args = parser.parse_args()
    return run_demo(args.checkpoint)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    raise SystemExit(main())
