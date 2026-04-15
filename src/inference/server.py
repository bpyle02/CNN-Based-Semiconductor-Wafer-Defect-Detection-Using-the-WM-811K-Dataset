"""
FastAPI-based real-time inference server for wafer defect classification.

Provides REST endpoints for model serving with support for:
- Image-based predictions (base64 or file upload)
- Multi-model serving with dynamic loading and model registry
- Health checks and model introspection
- Comprehensive request validation and error handling
- CORS support for web clients
- Performance metrics tracking

References:
    [141] (2021). "Deploying ML Models in Semiconductor Manufacturing"
"""

import base64
import io
import logging
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from src.data.dataset import KNOWN_CLASSES
from src.data.preprocessing import get_imagenet_normalize
from src.exceptions import InferenceError, ModelError
from src.inference.gradcam import GradCAM
from src.model_registry import resolve_trusted_checkpoint_path, verify_checkpoint
from src.models import WaferCNN, get_efficientnet_b0, get_resnet18

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Available model architectures."""

    CNN = "cnn"
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"


class PredictionRequest(BaseModel):
    """Request schema for base64-encoded image prediction."""

    image_base64: str = Field(..., description="Base64-encoded image (PNG/JPEG)")
    model_id: Optional[str] = Field(
        None,
        description="Optional ID of the model to use. If not provided, the default model is used.",
    )
    model_name: Optional[str] = Field(
        None,
        description="Backward-compatible alias for model_id.",
    )
    return_gradcam: bool = Field(
        False, description="If true, include a base64-encoded GradCAM overlay in the response."
    )

    @field_validator("image_base64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate base64 string format."""
        if not v:
            raise ValueError("image_base64 cannot be empty")
        if len(v) > 10_000_000:  # ~10 MB limit
            raise ValueError("image_base64 exceeds 10 MB limit")
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
        return v


class BatchPredictionRequest(BaseModel):
    """Request schema for batch base64-encoded image prediction."""

    images: List[str] = Field(..., description="List of base64-encoded images")
    model_id: Optional[str] = None
    model_name: Optional[str] = None

    @field_validator("images")
    @classmethod
    def validate_images(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("images must contain at least one item")
        for value in values:
            try:
                base64.b64decode(value, validate=True)
            except Exception as exc:
                raise ValueError(f"Invalid base64 encoding: {exc}") from exc
        return values


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    class_name: str = Field(..., description="Predicted defect class name")
    class_id: int = Field(..., description="Integer class ID (0-8)")
    confidence: float = Field(..., description="Softmax probability of predicted class")
    probabilities: Dict[str, float] = Field(
        ..., description="Softmax probabilities for all classes"
    )
    model_id: str = Field(..., description="Model ID used for prediction")
    model_name: str = Field(..., description="Backward-compatible alias for model_id")
    input_shape: Tuple[int, int] = Field(
        ..., description="Image shape (height, width) after validation"
    )
    inference_ms: float = Field(..., description="Inference time in milliseconds")
    gradcam_base64: Optional[str] = Field(None, description="Base64-encoded GradCAM activation map")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictionResponse]
    total_time_ms: float


class LoadModelRequest(BaseModel):
    """Request schema for loading a model."""

    model_type: ModelType = Field(..., description="Model architecture type")
    checkpoint_path: str = Field(..., description="Path to model checkpoint (.pth file)")
    model_id: Optional[str] = Field(None, description="User-defined ID for this model")

    @field_validator("checkpoint_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate checkpoint path exists and is safe to load."""
        p = Path(v).resolve()
        if not p.exists():
            raise ValueError("Checkpoint not found")
        if not p.suffix == ".pth":
            raise ValueError(f"Expected .pth file, got {p.suffix}")
        if p.is_symlink():
            raise ValueError("Symlinked checkpoint paths are not allowed")
        return str(p)


class ModelInfo(BaseModel):
    """Model information schema."""

    id: str
    architecture: str
    device: str
    num_parameters: int
    is_default: bool


class ServerMetrics(BaseModel):
    """Server performance metrics."""

    total_requests: int
    avg_latency_ms: float
    uptime_seconds: float
    loaded_models_count: int


class ModelServer:
    """
    Model server managing multiple models and inference execution.
    """

    def __init__(
        self,
        device: str = "cpu",
        allowed_checkpoint_dirs: Optional[List[Union[Path, str]]] = None,
    ) -> None:
        self.device = torch.device(device)
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.default_model_id: Optional[str] = None
        self.allowed_checkpoint_dirs = tuple(allowed_checkpoint_dirs or ())

        # Metrics
        self.start_time = time.time()
        self.request_count = 0
        self.latency_history: deque[float] = deque(maxlen=100)

        logger.info(f"ModelServer initialized on device: {self.device}")

    def load_model(
        self, model_type: ModelType, checkpoint_path: str, model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            resolved_path = resolve_trusted_checkpoint_path(
                checkpoint_path,
                allowed_roots=self.allowed_checkpoint_dirs or None,
            )

            if not verify_checkpoint(resolved_path):
                raise ModelError(f"Checkpoint integrity verification failed for {resolved_path}")

            checkpoint = torch.load(str(resolved_path), map_location=self.device, weights_only=True)

            if model_type == ModelType.CNN:
                model = WaferCNN(num_classes=9)
            elif model_type == ModelType.RESNET:
                model = get_resnet18(num_classes=9, pretrained=False)
            elif model_type == ModelType.EFFICIENTNET:
                model = get_efficientnet_b0(num_classes=9, pretrained=False)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                for key in ("model_state_dict", "state_dict", "global_model"):
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break

            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            m_id = model_id or resolved_path.stem
            self.models[m_id] = model
            self.model_metadata[m_id] = {
                "architecture": model_type.value,
                "num_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "path": str(resolved_path),
                "input_size": [96, 96],
                "num_classes": len(KNOWN_CLASSES),
                "classes": list(KNOWN_CLASSES),
            }

            if self.default_model_id is None:
                self.default_model_id = m_id

            logger.info(f"Loaded model '{m_id}' ({model_type.value})")
            return {
                "model_id": m_id,
                "model_name": m_id,
                "total_parameters": self.model_metadata[m_id]["num_params"],
                "trainable_parameters": self.model_metadata[m_id]["trainable_params"],
                **self.model_metadata[m_id],
            }

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {str(e)}")
            raise ModelError(f"Model loading failed: {str(e)}") from e

    def get_model(self, model_id: Optional[str] = None) -> Tuple[str, torch.nn.Module]:
        m_id = model_id or self.default_model_id
        if not m_id or m_id not in self.models:
            raise InferenceError(f"Model ID '{m_id}' not found or no models loaded.")
        return m_id, self.models[m_id]

    def _prepare_base_image(self, image_array: np.ndarray) -> np.ndarray:
        """Convert arbitrary input image to normalized 96x96 grayscale."""
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2)

        h, w = image_array.shape
        image_array = image_array.astype(np.float32)
        max_val = float(np.max(image_array))

        if max_val <= 1.0:
            normalized = image_array
        elif max_val <= 2.0:
            normalized = image_array / 2.0
        else:
            normalized = image_array / 255.0

        normalized = np.clip(normalized, 0.0, 1.0)
        image = Image.fromarray((normalized * 255.0).astype(np.uint8))
        image = image.resize((96, 96), Image.Resampling.BILINEAR)
        return np.array(image, dtype=np.float32) / 255.0

    def _predict_internal(
        self, image_array: np.ndarray, model_id: Optional[str] = None
    ) -> Tuple[int, np.ndarray, float, str]:
        m_id, model = self.get_model(model_id)

        start_t = time.time()
        base_image = self._prepare_base_image(image_array)
        img_stack = np.stack([base_image] * 3, axis=0)
        tensor = torch.from_numpy(img_stack).unsqueeze(0).to(self.device)

        # Apply normalization if needed
        arch = self.model_metadata[m_id]["architecture"]
        if arch in ["resnet", "efficientnet"]:
            norm = get_imagenet_normalize()
            tensor = norm(tensor)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            class_id = int(np.argmax(probs))

        latency = (time.time() - start_t) * 1000
        self.request_count += 1
        self.latency_history.append(latency)

        return class_id, probs, latency, m_id

    def predict(
        self, image_array: np.ndarray, model_id: Optional[str] = None
    ) -> Tuple[int, np.ndarray, float]:
        class_id, probs, latency, _ = self._predict_internal(image_array, model_id)
        return class_id, probs, latency

    def predict_batch(
        self, image_arrays: List[np.ndarray], model_id: Optional[str] = None
    ) -> Tuple[List[int], List[np.ndarray], float, str]:
        m_id, model = self.get_model(model_id)
        start_t = time.time()

        tensors = []
        for img in image_arrays:
            base = self._prepare_base_image(img)
            tensors.append(torch.from_numpy(np.stack([base] * 3, axis=0)).unsqueeze(0))

        batch_tensor = torch.cat(tensors, dim=0).to(self.device)
        arch = self.model_metadata[m_id]["architecture"]
        if arch in ["resnet", "efficientnet"]:
            batch_tensor = get_imagenet_normalize()(batch_tensor)

        with torch.no_grad():
            logits = model(batch_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            class_ids = np.argmax(probs, axis=1).tolist()

        latency = (time.time() - start_t) * 1000
        self.request_count += len(image_arrays)
        self.latency_history.append(latency / len(image_arrays))

        return class_ids, list(probs), latency, m_id

    def generate_gradcam(
        self,
        image_array: np.ndarray,
        model_id: Optional[str] = None,
        target_class: Optional[int] = None,
    ) -> Tuple[str, int]:
        m_id, model = self.get_model(model_id)
        base_image = self._prepare_base_image(image_array)

        # Find last conv layer
        target_layer = None
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

        if not target_layer:
            raise InferenceError("No conv layer found for GradCAM")

        gradcam = GradCAM(model, target_layer)
        img_stack = np.stack([base_image] * 3, axis=0)
        tensor = torch.from_numpy(img_stack).unsqueeze(0).to(self.device)

        try:
            heatmap, resolved_class = gradcam.generate(
                tensor, target_class=target_class, device=str(self.device)
            )

            # Encode overlay
            base_rgb = np.stack([base_image] * 3, axis=-1)
            heatmap_rgb = np.stack([heatmap, 0.35 * heatmap, np.zeros_like(heatmap)], axis=-1)
            overlay = np.clip(0.6 * base_rgb + 0.75 * heatmap_rgb, 0, 1)

            img = Image.fromarray((overlay * 255.0).astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8"), resolved_class
        finally:
            gradcam.remove_hooks()


def create_app(
    device: str = "cpu",
    allowed_checkpoint_dirs: Optional[List[Union[Path, str]]] = None,
    model_checkpoint: Optional[str] = None,
    model_type: Optional["ModelType"] = None,
) -> FastAPI:
    app = FastAPI(title="Wafer Defect Detection Server", version="2.0.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    server = ModelServer(device=device, allowed_checkpoint_dirs=allowed_checkpoint_dirs)
    if model_checkpoint and model_type:
        server.load_model(model_type, model_checkpoint)

    def resolve_request_model_id(
        model_id: Optional[str], model_name: Optional[str]
    ) -> Optional[str]:
        return model_id or model_name

    def prediction_response_from_probs(
        *,
        class_id: int,
        probs: np.ndarray,
        latency: float,
        model_name: str,
        gradcam: Optional[str] = None,
    ) -> PredictionResponse:
        return PredictionResponse(
            class_name=KNOWN_CLASSES[class_id],
            class_id=class_id,
            confidence=float(probs[class_id]),
            probabilities={KNOWN_CLASSES[i]: float(p) for i, p in enumerate(probs)},
            model_id=model_name,
            model_name=model_name,
            input_shape=(96, 96),
            inference_ms=latency,
            gradcam_base64=gradcam,
        )

    @app.get("/health", tags=["System"])
    async def health_check():
        return {
            "status": "healthy",
            "model_loaded": server.default_model_id is not None,
            "models_loaded": list(server.models.keys()),
            "default_model": server.default_model_id,
            "device": str(server.device),
        }

    @app.get("/models", tags=["Models"])
    async def list_models():
        return {
            "available_architectures": [member.value for member in ModelType],
            "loaded_models": list(server.models.keys()),
            "current_model": server.default_model_id,
        }

    @app.get("/model_info", tags=["Models"])
    async def model_info(model_name: Optional[str] = Query(default=None)):
        resolved_model_id = resolve_request_model_id(model_name, None)
        m_id, _ = server.get_model(resolved_model_id)
        metadata = server.model_metadata[m_id]
        return {
            "model_name": m_id,
            "architecture": metadata["architecture"],
            "input_size": metadata["input_size"],
            "num_classes": metadata["num_classes"],
            "classes": metadata["classes"],
        }

    @app.get("/metrics", response_model=ServerMetrics, tags=["System"])
    async def get_metrics():
        avg_lat = (
            sum(server.latency_history) / len(server.latency_history)
            if server.latency_history
            else 0.0
        )
        return ServerMetrics(
            total_requests=server.request_count,
            avg_latency_ms=avg_lat,
            uptime_seconds=time.time() - server.start_time,
            loaded_models_count=len(server.models),
        )

    @app.post("/load_model", tags=["Models"])
    async def load_model(request: LoadModelRequest):
        try:
            info = server.load_model(request.model_type, request.checkpoint_path, request.model_id)
            return {"success": True, **info, "info": info}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
    def predict(request: PredictionRequest):
        try:
            img_data = base64.b64decode(request.image_base64)
            img = Image.open(io.BytesIO(img_data))
            requested_model = resolve_request_model_id(request.model_id, request.model_name)
            class_id, probs, latency, m_id = server._predict_internal(
                np.array(img), requested_model
            )

            gradcam = None
            if request.return_gradcam:
                gradcam, _ = server.generate_gradcam(np.array(img), m_id, class_id)
            return prediction_response_from_probs(
                class_id=class_id,
                probs=probs,
                latency=latency,
                model_name=m_id,
                gradcam=gradcam,
            )
        except InferenceError as e:
            status_code = 503 if not server.models else 400
            raise HTTPException(status_code=status_code, detail=str(e))
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
    def predict_batch(request: BatchPredictionRequest):
        try:
            images = [
                np.array(Image.open(io.BytesIO(base64.b64decode(b64)))) for b64 in request.images
            ]
            requested_model = resolve_request_model_id(request.model_id, request.model_name)
            class_ids, probs_list, total_latency, m_id = server.predict_batch(
                images, requested_model
            )

            preds = []
            for i, (cid, probs) in enumerate(zip(class_ids, probs_list)):
                preds.append(
                    prediction_response_from_probs(
                        class_id=cid,
                        probs=probs,
                        latency=total_latency / len(images),
                        model_name=m_id,
                    )
                )
            return BatchPredictionResponse(predictions=preds, total_time_ms=total_latency)
        except InferenceError as e:
            status_code = 503 if not server.models else 400
            raise HTTPException(status_code=status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/predict_file", response_model=PredictionResponse, tags=["Inference"])
    async def predict_file(
        file: UploadFile = File(...),
        model_name: Optional[str] = Query(default=None),
        return_gradcam: bool = Query(default=False),
    ):
        try:
            image_bytes = await file.read()
            img = Image.open(io.BytesIO(image_bytes))
            class_id, probs, latency, m_id = server._predict_internal(np.array(img), model_name)
            gradcam = None
            if return_gradcam:
                gradcam, _ = server.generate_gradcam(np.array(img), m_id, class_id)
            return prediction_response_from_probs(
                class_id=class_id,
                probs=probs,
                latency=latency,
                model_name=m_id,
                gradcam=gradcam,
            )
        except InferenceError as e:
            status_code = 503 if not server.models else 400
            raise HTTPException(status_code=status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app
