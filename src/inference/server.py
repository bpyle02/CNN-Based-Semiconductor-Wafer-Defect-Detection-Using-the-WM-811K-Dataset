"""
FastAPI-based real-time inference server for wafer defect classification.

Provides REST endpoints for model serving with support for:
- Image-based predictions (base64 or file upload)
- Multi-model serving with dynamic loading
- Health checks and model introspection
- Comprehensive request validation and error handling
- CORS support for web clients
"""

import base64
import io
import logging
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Any
from enum import Enum
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.exceptions import InferenceError, ModelError
from src.inference.gradcam import GradCAM
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.model_registry import resolve_trusted_checkpoint_path, verify_checkpoint
from src.data.dataset import KNOWN_CLASSES
from src.data.preprocessing import get_imagenet_normalize

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Available model architectures."""
    CNN = "cnn"
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"


class PredictionRequest(BaseModel):
    """Request schema for base64-encoded image prediction."""

    image_base64: str = Field(
        ...,
        description="Base64-encoded image (PNG/JPEG)"
    )
    model_name: Optional[str] = Field(
        None,
        description="Optional name of the active model. If provided, it must match the currently loaded model."
    )
    return_gradcam: bool = Field(
        False,
        description="If true, include a base64-encoded GradCAM overlay in the response."
    )

    @field_validator('image_base64')
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

    MAX_IMAGES: ClassVar[int] = 128

    images: List[str] = Field(..., description="List of base64-encoded images")
    model_name: Optional[str] = Field(
        None,
        description="Optional name of the active model. If provided, it must match the currently loaded model.",
    )

    @field_validator("images")
    @classmethod
    def validate_images(cls, images: List[str]) -> List[str]:
        if not images:
            raise ValueError("images cannot be empty")
        if len(images) > cls.MAX_IMAGES:
            raise ValueError(f"images exceeds the maximum batch size of {cls.MAX_IMAGES}")
        for idx, image in enumerate(images):
            if not image:
                raise ValueError(f"images[{idx}] cannot be empty")
            if len(image) > 10_000_000:
                raise ValueError(f"images[{idx}] exceeds 10 MB limit")
            try:
                base64.b64decode(image, validate=True)
            except Exception as exc:
                raise ValueError(f"Invalid base64 encoding at index {idx}: {exc}") from exc
        return images


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    class_name: str = Field(
        ...,
        description="Predicted defect class name"
    )
    class_id: int = Field(
        ...,
        description="Integer class ID (0-8)"
    )
    confidence: float = Field(
        ...,
        description="Softmax probability of predicted class"
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Softmax probabilities for all classes"
    )
    model_name: str = Field(
        ...,
        description="Model used for prediction"
    )
    input_shape: Tuple[int, int] = Field(
        ...,
        description="Image shape (height, width) after validation"
    )
    inference_ms: float = Field(
        ...,
        description="Inference time in milliseconds"
    )
    gradcam_base64: Optional[str] = Field(
        None,
        description="Base64-encoded GradCAM activation map (if requested)"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    total_time_ms: float


class LoadModelRequest(BaseModel):
    """Request schema for loading a model."""

    model_type: ModelType = Field(
        ...,
        description="Model architecture type"
    )
    checkpoint_path: str = Field(
        ...,
        description="Path to model checkpoint (.pth file)"
    )

    @field_validator('checkpoint_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate checkpoint path exists and is safe to load."""
        p = Path(v).resolve()
        if not p.exists():
            raise ValueError("Checkpoint not found")
        if not p.suffix == '.pth':
            raise ValueError(f"Expected .pth file, got {p.suffix}")
        if p.is_symlink():
            raise ValueError("Symlinked checkpoint paths are not allowed")
        return str(p)


class ModelInfo(BaseModel):
    """Model information schema."""

    name: str
    architecture: str
    device: str
    num_parameters: int
    num_trainable_parameters: int
    input_size: Tuple[int, int]
    num_classes: int
    classes: List[str]


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_loaded: bool
    current_model: Optional[str] = None
    device: str
    torch_version: str


class ModelServer:
    """
    Model server wrapper for managing model lifecycle and inference.

    Handles:
    - Model loading and switching
    - Input preprocessing and validation
    - Inference execution
    - Output post-processing
    """

    def __init__(
        self,
        device: str = "cpu",
        allowed_checkpoint_dirs: Optional[List[Path | str]] = None,
    ) -> None:
        """
        Initialize the model server.

        Args:
            device: Device to load models on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model: Optional[torch.nn.Module] = None
        self.model_name: Optional[str] = None
        self.model_type: Optional[ModelType] = None
        self.allowed_checkpoint_dirs = tuple(allowed_checkpoint_dirs or ())
        logger.info(f"ModelServer initialized on device: {self.device}")

    def load_model(
        self,
        model_type: ModelType,
        checkpoint_path: str
    ) -> Dict[str, Any]:
        """
        Load a model checkpoint.

        Args:
            model_type: Type of model to load
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with model info

        Raises:
            ModelError: If checkpoint loading fails
        """
        try:
            resolved_path = resolve_trusted_checkpoint_path(
                checkpoint_path,
                allowed_roots=self.allowed_checkpoint_dirs or None,
            )

            # Verify checkpoint integrity before loading
            if not verify_checkpoint(resolved_path):
                raise ModelError(
                    f"Checkpoint integrity verification failed for {resolved_path}"
                )

            # PyTorch security: use weights_only=True
            checkpoint = torch.load(
                str(resolved_path),
                map_location=self.device,
                weights_only=True
            )

            # Infer model type and initialize architecture
            if model_type == ModelType.CNN:
                model = WaferCNN(num_classes=9)
            elif model_type == ModelType.RESNET:
                model = get_resnet18(num_classes=9)
            elif model_type == ModelType.EFFICIENTNET:
                model = get_efficientnet_b0(num_classes=9)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Load state dict from the common checkpoint formats used in this repo.
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                for key in ('model_state_dict', 'state_dict', 'global_model'):
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
                else:
                    if not checkpoint or not all(
                        isinstance(value, torch.Tensor) for value in checkpoint.values()
                    ):
                        raise ValueError(
                            "Unsupported checkpoint format. Expected a raw state_dict "
                            "or one of: model_state_dict, state_dict, global_model."
                        )

            model.load_state_dict(state_dict)

            model.to(self.device)
            model.eval()

            self.model = model
            self.model_name = resolved_path.stem
            self.model_type = model_type

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            info = {
                'model_name': self.model_name,
                'model_type': model_type.value,
                'device': str(self.device),
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'checkpoint_path': str(resolved_path),
            }

            logger.info(f"Loaded model: {self.model_name} ({model_type.value})")
            return info

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {str(e)}")
            raise ModelError(f"Model loading failed: {str(e)}") from e

    def _prepare_base_image(self, image_array: np.ndarray) -> np.ndarray:
        """Convert arbitrary input image to normalized 96x96 grayscale."""
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2)
        elif len(image_array.shape) != 2:
            raise ValueError(
                f"Expected 2D or 3D image, got shape {image_array.shape}"
            )

        h, w = image_array.shape
        if h < 32 or w < 32 or h > 512 or w > 512:
            raise ValueError(
                f"Image size {h}x{w} out of acceptable range [32-512]"
            )

        image_array = image_array.astype(np.float32)
        max_value = float(np.max(image_array))

        if max_value <= 1.0:
            normalized = image_array
        elif max_value <= 2.0:
            normalized = image_array / 2.0
        else:
            normalized = image_array / 255.0

        normalized = np.clip(normalized, 0.0, 1.0)
        image = Image.fromarray((normalized * 255.0).astype(np.uint8))
        image = image.resize((96, 96), Image.BILINEAR)
        return np.array(image, dtype=np.float32) / 255.0

    def _tensor_from_base_image(self, base_image: np.ndarray) -> torch.Tensor:
        """Convert normalized grayscale image to model input tensor."""
        image_array = np.stack([base_image] * 3, axis=0)
        tensor = torch.from_numpy(image_array).unsqueeze(0).to(self.device)

        if self.model_type in [ModelType.RESNET, ModelType.EFFICIENTNET]:
            imagenet_norm = get_imagenet_normalize()
            tensor = imagenet_norm(tensor)

        return tensor

    def _find_last_conv_layer(self) -> torch.nn.Module:
        """Return the last convolutional layer for GradCAM."""
        if self.model is None:
            raise InferenceError("No model loaded. Call load_model() first.")

        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module

        raise InferenceError("Unable to find a convolutional layer for GradCAM.")

    def _encode_gradcam_overlay(
        self,
        base_image: np.ndarray,
        heatmap: np.ndarray,
    ) -> str:
        """Create a base64-encoded PNG overlay from a heatmap."""
        base_rgb = np.stack([base_image] * 3, axis=-1)
        heatmap = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
        heatmap_rgb = np.stack(
            [heatmap, 0.35 * heatmap, np.zeros_like(heatmap)],
            axis=-1,
        )
        overlay = np.clip(0.60 * base_rgb + 0.75 * heatmap_rgb, 0.0, 1.0)
        image = Image.fromarray((overlay * 255.0).astype(np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def preprocess_image(self, image_array: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image_array: Input image as numpy array (H, W) or (H, W, C)

        Returns:
            Preprocessed tensor (1, 3, 96, 96)

        Raises:
            ValueError: If image validation fails
        """
        base_image = self._prepare_base_image(image_array)
        return self._tensor_from_base_image(base_image)

    def predict(
        self,
        image_array: np.ndarray
    ) -> Tuple[int, np.ndarray, float]:
        """
        Run inference on preprocessed image.

        Args:
            image_array: Input image as numpy array (H, W) or (H, W, C)

        Returns:
            Tuple of (predicted_class_id, probabilities, inference_time_ms)

        Raises:
            InferenceError: If no model is loaded or inference fails
        """
        if self.model is None:
            raise InferenceError("No model loaded. Call load_model() first.")

        try:
            tensor = self.preprocess_image(image_array)

            with torch.no_grad():
                start_time = time.time()
                logits = self.model(tensor)
                inference_time = (time.time() - start_time) * 1000  # ms

                probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
                predicted_class_id = int(np.argmax(probabilities))

            return predicted_class_id, probabilities, inference_time

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise InferenceError(f"Inference failed: {str(e)}") from e

    def predict_batch(
        self,
        image_arrays: List[np.ndarray]
    ) -> Tuple[List[int], List[np.ndarray], float]:
        """Run inference on a batch of images."""
        if self.model is None:
            raise InferenceError("No model loaded.")

        tensors = []
        for img in image_arrays:
            tensors.append(self.preprocess_image(img))
        
        batch_tensor = torch.cat(tensors, dim=0).to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            logits = self.model(batch_tensor)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            probs = F.softmax(logits, dim=1).cpu().numpy()
        class_ids = np.argmax(probs, axis=1).tolist()

        return class_ids, list(probs), inference_time

    def decode_image_base64(self, image_base64: str) -> np.ndarray:
        """Decode a base64 PNG/JPEG payload into a float32 numpy array."""
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        return np.array(image, dtype=np.float32)

    def _build_prediction_response(
        self,
        class_id: int,
        probabilities: np.ndarray,
        inference_time: float,
        gradcam_base64: Optional[str] = None,
    ) -> PredictionResponse:
        """Build the canonical prediction payload from model outputs."""
        return PredictionResponse(
            class_name=KNOWN_CLASSES[class_id],
            class_id=class_id,
            confidence=float(probabilities[class_id]),
            probabilities={
                KNOWN_CLASSES[i]: float(p)
                for i, p in enumerate(probabilities)
            },
            model_name=self.model_name or "unknown",
            input_shape=(96, 96),
            inference_ms=inference_time,
            gradcam_base64=gradcam_base64,
        )

    def generate_gradcam(
        self,
        image_array: np.ndarray,
        target_class: Optional[int] = None,
    ) -> Tuple[str, int]:
        """Generate a base64-encoded GradCAM overlay for an input image."""
        if self.model is None:
            raise InferenceError("No model loaded. Call load_model() first.")

        base_image = self._prepare_base_image(image_array)
        tensor = self._tensor_from_base_image(base_image)
        target_layer = self._find_last_conv_layer()
        gradcam = GradCAM(self.model, target_layer)

        try:
            heatmap, resolved_class = gradcam.generate(
                tensor,
                target_class=target_class,
                device=str(self.device),
            )
        except Exception as e:
            logger.error(f"GradCAM generation failed: {str(e)}")
            raise InferenceError(f"GradCAM generation failed: {str(e)}") from e
        finally:
            gradcam.remove_hooks()

        return self._encode_gradcam_overlay(base_image, heatmap), resolved_class


def create_app(
    device: str = "cpu",
    model_checkpoint: Optional[str] = None,
    model_type: Optional[ModelType] = None,
    allowed_checkpoint_dirs: Optional[List[Path | str]] = None,
) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        device: Device to use ('cpu' or 'cuda')
        model_checkpoint: Optional path to checkpoint to load on startup
        model_type: Type of model to load (required if model_checkpoint is set)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Wafer Defect Detection Server",
        description="Real-time inference server for CNN-based wafer defect classification",
        version="1.0.0",
    )

    # Add CORS middleware (permissive for research use; restrict origins in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Initialize model server
    server = ModelServer(
        device=device,
        allowed_checkpoint_dirs=allowed_checkpoint_dirs,
    )

    # Load initial model if provided
    if model_checkpoint and model_type:
        try:
            server.load_model(model_type, model_checkpoint)
        except Exception as e:
            logger.warning(f"Failed to load initial model: {e}")

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health check endpoint",
        tags=["System"]
    )
    async def health_check() -> HealthResponse:
        """
        Check server health and model status.

        Returns:
            Health status with model information
        """
        return HealthResponse(
            status="healthy",
            model_loaded=server.model is not None,
            current_model=server.model_name,
            device=str(server.device),
            torch_version=torch.__version__,
        )

    @app.get(
        "/models",
        response_model=Dict[str, Any],
        summary="List available models",
        tags=["Models"]
    )
    async def list_models() -> Dict[str, Any]:
        """
        List available model architectures.

        Returns:
            Dictionary with available model types
        """
        return {
            "available_architectures": [m.value for m in ModelType],
            "current_model": server.model_name or "none",
        }

    @app.post(
        "/load_model",
        response_model=Dict[str, Any],
        summary="Load a model checkpoint",
        tags=["Models"]
    )
    async def load_model_endpoint(request: LoadModelRequest) -> Dict[str, Any]:
        """
        Load a model checkpoint from disk.

        Args:
            request: LoadModelRequest with model_type and checkpoint_path

        Returns:
            Model information after loading

        Raises:
            HTTPException: If loading fails
        """
        try:
            info = server.load_model(request.model_type, request.checkpoint_path)
            return {
                "success": True,
                "message": f"Model loaded successfully",
                **info,
            }
        except ModelError as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load model: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load model: {str(e)}"
            )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Predict on base64-encoded image",
        tags=["Inference"]
    )
    def predict_base64(request: PredictionRequest) -> PredictionResponse:
        """
        Run inference on a base64-encoded image.
        Uses synchronous 'def' so FastAPI runs it in a threadpool,
        preventing CPU-bound PyTorch calls from blocking the event loop.
        """
        if server.model is None:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. POST to /load_model first."
            )

        if request.model_name is not None and request.model_name != server.model_name:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Requested model '{request.model_name}' is not currently loaded. "
                    f"Active model: '{server.model_name}'."
                ),
            )

        try:
            # Decode base64 image
            image_array = server.decode_image_base64(request.image_base64)

            # Run inference
            class_id, probabilities, inference_time = server.predict(image_array)

            gradcam_base64 = None
            if request.return_gradcam:
                gradcam_base64, _ = server.generate_gradcam(
                    image_array,
                    target_class=class_id,
                )

            response = server._build_prediction_response(
                class_id=class_id,
                probabilities=probabilities,
                inference_time=inference_time,
                gradcam_base64=gradcam_base64,
            )

            logger.info(
                f"Prediction: {response.class_name} (confidence: {response.confidence:.4f}, "
                f"inference: {inference_time:.2f}ms)"
            )
            return response

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except InferenceError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/predict_batch",
        response_model=BatchPredictionResponse,
        summary="Predict on batch of base64-encoded images",
        tags=["Inference"]
    )
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Run batch inference on multiple base64-encoded images."""
        if server.model is None:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. POST to /load_model first.",
            )

        if request.model_name is not None and request.model_name != server.model_name:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Requested model '{request.model_name}' is not currently loaded. "
                    f"Active model: '{server.model_name}'."
                ),
            )

        try:
            image_arrays = [server.decode_image_base64(img_b64) for img_b64 in request.images]

            class_ids, probs_list, total_inference_time = server.predict_batch(image_arrays)

            predictions = []
            per_image_ms = total_inference_time / len(image_arrays)
            for cid, probs in zip(class_ids, probs_list):
                predictions.append(
                    server._build_prediction_response(
                        class_id=cid,
                        probabilities=probs,
                        inference_time=per_image_ms,
                    )
                )

            return BatchPredictionResponse(
                predictions=predictions,
                total_time_ms=total_inference_time
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except InferenceError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/predict_file",
        response_model=PredictionResponse,
        summary="Predict on uploaded image file",
        tags=["Inference"]
    )
    async def predict_file(
        file: UploadFile = File(...),
        model_name: Optional[str] = Query(
            None,
            description="Optional name of the active model. If provided, it must match the currently loaded model."
        ),
        return_gradcam: bool = Query(
            False,
            description="If true, include a base64-encoded GradCAM overlay in the response."
        ),
    ) -> PredictionResponse:
        """
        Run inference on an uploaded image file.
        Uses 'async def' because 'await file.read()' is an I/O operation.
        """
        if server.model is None:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. POST to /load_model first."
            )

        if model_name is not None and model_name != server.model_name:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Requested model '{model_name}' is not currently loaded. "
                    f"Active model: '{server.model_name}'."
                ),
            )

        try:
            # Validate file type
            if file.content_type not in [
                "image/png",
                "image/jpeg",
                "image/jpg",
                "image/bmp"
            ]:
                raise HTTPException(
                    status_code=400,
                    detail="File must be PNG, JPEG, or BMP"
                )

            # Read and convert image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image, dtype=np.float32)

            # Run inference
            class_id, probabilities, inference_time = server.predict(image_array)

            # Build response
            class_name = KNOWN_CLASSES[class_id]
            confidence = float(probabilities[class_id])
            gradcam_base64 = None
            if return_gradcam:
                gradcam_base64, _ = server.generate_gradcam(
                    image_array,
                    target_class=class_id,
                )

            response = PredictionResponse(
                class_name=class_name,
                class_id=class_id,
                confidence=confidence,
                probabilities={
                    KNOWN_CLASSES[i]: float(p)
                    for i, p in enumerate(probabilities)
                },
                model_name=server.model_name or "unknown",
                input_shape=(96, 96),
                inference_ms=inference_time,
                gradcam_base64=gradcam_base64,
            )

            logger.info(
                f"Prediction: {class_name} (confidence: {confidence:.4f}, "
                f"inference: {inference_time:.2f}ms)"
            )
            return response

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except InferenceError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/model_info",
        response_model=Optional[ModelInfo],
        summary="Get current model information",
        tags=["Models"]
    )
    async def get_model_info() -> Optional[ModelInfo]:
        """
        Get detailed information about currently loaded model.

        Returns:
            ModelInfo with architecture details, or None if no model loaded
        """
        if server.model is None:
            return None

        total_params = sum(p.numel() for p in server.model.parameters())
        trainable_params = sum(
            p.numel() for p in server.model.parameters() if p.requires_grad
        )

        return ModelInfo(
            name=server.model_name or "unknown",
            architecture=server.model_type.value if server.model_type else "unknown",
            device=str(server.device),
            num_parameters=total_params,
            num_trainable_parameters=trainable_params,
            input_size=(96, 96),
            num_classes=9,
            classes=list(KNOWN_CLASSES),
        )

    return app
