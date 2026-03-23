#!/usr/bin/env python
"""
Test script for the FastAPI inference server.

Demonstrates:
- Creating a server instance
- Loading models
- Making predictions on synthetic images
- Error handling

Run this after starting the server:
    python inference_server.py --port 8000

Or test the server directly without running it externally:
    python test_inference_server.py
"""

import base64
import io
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None

from src.models import WaferCNN

if FASTAPI_AVAILABLE:
    from src.inference.server import create_app, ModelServer, ModelType
else:
    create_app = None
    ModelServer = None
    ModelType = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="fastapi is not installed",
)


def create_synthetic_image(size: tuple = (96, 96)) -> np.ndarray:
    """
    Create a synthetic wafer map image for testing.

    Args:
        size: Image size (height, width)

    Returns:
        Numpy array with values in [0, 1]
    """
    # Create a simple pattern: center dot with radial gradient
    h, w = size
    center = (h // 2, w // 2)

    image = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            image[y, x] = max(0, 1 - (dist / max_dist))

    return image


def image_to_base64(image_array: np.ndarray) -> str:
    """
    Convert numpy array to base64-encoded PNG.

    Args:
        image_array: Image as numpy array [0, 1]

    Returns:
        Base64-encoded PNG string
    """
    # Convert to PIL Image
    pil_image = Image.fromarray((image_array * 255).astype(np.uint8))

    # Encode to PNG
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)

    # Encode to base64
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded


def test_server_direct() -> None:
    """Test server directly using TestClient."""
    logger.info("Testing inference server (direct mode)...")

    # Create app
    app = create_app(device="cpu")
    client = TestClient(app)

    # Test 1: Health check
    logger.info("\n1. Testing /health endpoint...")
    response = client.get("/health")
    assert response.status_code == 200
    health = response.json()
    assert health["status"] == "healthy"
    logger.info(f"   Status: {health['status']}")
    logger.info(f"   Model loaded: {health['model_loaded']}")
    logger.info(f"   Device: {health['device']}")

    # Test 2: List models
    logger.info("\n2. Testing /models endpoint...")
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    logger.info(f"   Available architectures: {models['available_architectures']}")
    logger.info(f"   Current model: {models['current_model']}")

    # Test 3: Predict without model (should fail)
    logger.info("\n3. Testing /predict without loaded model (should fail)...")
    image = create_synthetic_image()
    image_b64 = image_to_base64(image)
    response = client.post(
        "/predict",
        json={"image_base64": image_b64}
    )
    assert response.status_code == 503
    logger.info(f"   Expected 503: {response.json()['detail']}")

    # Test 4: Load model
    logger.info("\n4. Testing /load_model endpoint...")

    # Create a temporary model for testing
    model = WaferCNN(num_classes=9)
    model_path = Path("checkpoints")
    model_path.mkdir(exist_ok=True)
    checkpoint_path = model_path / "test_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"   Created test model: {checkpoint_path}")

    try:
        response = client.post(
            "/load_model",
            json={
                "model_type": "cnn",
                "checkpoint_path": str(checkpoint_path)
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"]
        logger.info(f"   Model loaded: {result['model_name']}")
        logger.info(f"   Total parameters: {result['total_parameters']:,}")
        logger.info(f"   Trainable parameters: {result['trainable_parameters']:,}")

        # Test 5: Get model info
        logger.info("\n5. Testing /model_info endpoint...")
        response = client.get("/model_info")
        assert response.status_code == 200
        info = response.json()
        logger.info(f"   Architecture: {info['architecture']}")
        logger.info(f"   Input size: {info['input_size']}")
        logger.info(f"   Number of classes: {info['num_classes']}")
        logger.info(f"   Classes: {info['classes']}")

        # Test 6: Predict with base64 image
        logger.info("\n6. Testing /predict endpoint...")
        image = create_synthetic_image()
        image_b64 = image_to_base64(image)
        response = client.post(
            "/predict",
            json={"image_base64": image_b64}
        )
        assert response.status_code == 200
        prediction = response.json()
        logger.info(f"   Predicted class: {prediction['class_name']}")
        logger.info(f"   Confidence: {prediction['confidence']:.4f}")
        logger.info(f"   Inference time: {prediction['inference_ms']:.2f}ms")
        logger.info(f"   All probabilities: {json.dumps(prediction['probabilities'], indent=6)}")

        # Test 7: Predict with GradCAM enabled
        logger.info("\n7. Testing /predict endpoint with GradCAM...")
        response = client.post(
            "/predict",
            json={
                "image_base64": image_b64,
                "model_name": result["model_name"],
                "return_gradcam": True,
            }
        )
        assert response.status_code == 200
        prediction = response.json()
        assert prediction["gradcam_base64"] is not None
        base64.b64decode(prediction["gradcam_base64"])

        # Test 8: Predict with file upload
        logger.info("\n8. Testing /predict_file endpoint...")
        image = create_synthetic_image()
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)

        response = client.post(
            f"/predict_file?model_name={result['model_name']}&return_gradcam=true",
            files={"file": ("test.png", buffer, "image/png")}
        )
        assert response.status_code == 200
        prediction = response.json()
        logger.info(f"   Predicted class: {prediction['class_name']}")
        logger.info(f"   Confidence: {prediction['confidence']:.4f}")
        assert prediction["gradcam_base64"] is not None
        base64.b64decode(prediction["gradcam_base64"])

        # Test 9: Model name mismatch
        logger.info("\n9. Testing error handling (model mismatch)...")
        response = client.post(
            "/predict",
            json={"image_base64": image_b64, "model_name": "wrong-model"}
        )
        assert response.status_code == 400

        # Test 10: Invalid base64
        logger.info("\n10. Testing error handling (invalid base64)...")
        response = client.post(
            "/predict",
            json={"image_base64": "not-valid-base64!!!"}
        )
        assert response.status_code in [400, 422]  # Validation error
        logger.info(f"   Expected validation error (status {response.status_code})")

        logger.info("\n" + "="*60)
        logger.info("All tests passed!")
        logger.info("="*60)

    finally:
        # Cleanup
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            if model_path.exists() and not any(model_path.iterdir()):
                model_path.rmdir()


def test_model_server_direct() -> None:
    """Test ModelServer directly."""
    logger.info("Testing ModelServer class directly...")

    # Create server
    server = ModelServer(device="cpu")

    # Create and save a test model
    model = WaferCNN(num_classes=9)
    model_path = Path("checkpoints")
    model_path.mkdir(exist_ok=True)
    checkpoint_path = model_path / "test_model.pth"
    torch.save(model.state_dict(), checkpoint_path)

    try:
        # Load model
        logger.info("\n1. Loading model...")
        info = server.load_model(ModelType.CNN, str(checkpoint_path))
        logger.info(f"   Model: {info['model_name']}")
        logger.info(f"   Parameters: {info['total_parameters']:,}")

        # Predict on synthetic image
        logger.info("\n2. Running inference on synthetic image...")
        image = create_synthetic_image()
        class_id, probs, inference_time = server.predict(image)
        logger.info(f"   Predicted class: {class_id}")
        logger.info(f"   Confidence: {probs[class_id]:.4f}")
        logger.info(f"   Inference time: {inference_time:.2f}ms")

        logger.info("\nModelServer test passed!")

    finally:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if model_path.exists() and not any(model_path.iterdir()):
            model_path.rmdir()


if __name__ == "__main__":
    import sys

    logger.info("=" * 60)
    logger.info("Inference Server Test Suite")
    logger.info("=" * 60)

    if not FASTAPI_AVAILABLE:
        logger.error("fastapi not installed. Skipping inference server tests.")
        sys.exit(1)

    try:
        test_model_server_direct()
        test_server_direct()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        sys.exit(1)
