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
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.data.dataset import KNOWN_CLASSES
from src.data.preprocessing import get_image_transforms, get_imagenet_normalize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        description="Model name to use for prediction. If None, uses currently loaded model."
    )
    return_gradcam: bool = Field(
        False,
        description="Whether to return GradCAM activation map (base64-encoded)"
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
            base64.b64decode(v)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
        return v


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
        """Validate checkpoint path exists."""
        p = Path(v)
        if not p.exists():
            raise ValueError(f"Checkpoint not found: {v}")
        if not p.suffix == '.pth':
            raise ValueError(f"Expected .pth file, got {p.suffix}")
        return v


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

    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize the model server.

        Args:
            device: Device to load models on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model: Optional[torch.nn.Module] = None
        self.model_name: Optional[str] = None
        self.model_type: Optional[ModelType] = None
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
            RuntimeError: If checkpoint loading fails
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Infer model type and initialize architecture
            if model_type == ModelType.CNN:
                model = WaferCNN(num_classes=9)
            elif model_type == ModelType.RESNET:
                model = get_resnet18(num_classes=9)
            elif model_type == ModelType.EFFICIENTNET:
                model = get_efficientnet_b0(num_classes=9)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Load state dict (handle dict vs model weights)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()

            self.model = model
            self.model_name = Path(checkpoint_path).stem
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
                'checkpoint_path': checkpoint_path,
            }

            logger.info(f"Loaded model: {self.model_name} ({model_type.value})")
            return info

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

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
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2)
        elif len(image_array.shape) != 2:
            raise ValueError(
                f"Expected 2D or 3D image, got shape {image_array.shape}"
            )

        # Validate size (should be ~96x96)
        h, w = image_array.shape
        if h < 32 or w < 32 or h > 512 or w > 512:
            raise ValueError(
                f"Image size {h}x{w} out of acceptable range [32-512]"
            )

        # Resize to 96x96
        image = Image.fromarray((image_array * 255).astype(np.uint8))
        image = image.resize((96, 96), Image.BILINEAR)
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Normalize: divide by 2.0 (as in preprocessing)
        image_array = image_array / 2.0

        # Stack to 3 channels
        image_array = np.stack([image_array] * 3, axis=0)

        # Convert to tensor
        tensor = torch.from_numpy(image_array).unsqueeze(0).to(self.device)

        # Apply ImageNet normalization only for pretrained models
        if self.model_type in [ModelType.RESNET, ModelType.EFFICIENTNET]:
            imagenet_norm = get_imagenet_normalize()
            tensor = imagenet_norm(tensor)

        return tensor

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
            RuntimeError: If no model is loaded or inference fails
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        try:
            tensor = self.preprocess_image(image_array)

            with torch.no_grad():
                import time
                start_time = time.time()
                logits = self.model(tensor)
                inference_time = (time.time() - start_time) * 1000  # ms

                probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
                predicted_class_id = int(np.argmax(probabilities))

            return predicted_class_id, probabilities, inference_time

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}")


def create_app(
    device: str = "cpu",
    model_checkpoint: Optional[str] = None,
    model_type: Optional[ModelType] = None
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

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize model server
    server = ModelServer(device=device)

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
    async def predict_base64(request: PredictionRequest) -> PredictionResponse:
        """
        Run inference on a base64-encoded image.

        Args:
            request: PredictionRequest with base64 image data

        Returns:
            PredictionResponse with class, confidence, and probabilities

        Raises:
            HTTPException: If no model loaded or prediction fails
        """
        if server.model is None:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. POST to /load_model first."
            )

        try:
            # Decode base64 image
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image, dtype=np.float32)

            # Run inference
            class_id, probabilities, inference_time = server.predict(image_array)

            # Build response
            class_name = KNOWN_CLASSES[class_id]
            confidence = float(probabilities[class_id])

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
            )

            logger.info(
                f"Prediction: {class_name} (confidence: {confidence:.4f}, "
                f"inference: {inference_time:.2f}ms)"
            )
            return response

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/predict_file",
        response_model=PredictionResponse,
        summary="Predict on uploaded image file",
        tags=["Inference"]
    )
    async def predict_file(file: UploadFile = File(...)) -> PredictionResponse:
        """
        Run inference on an uploaded image file.

        Supports PNG, JPEG, BMP formats.

        Args:
            file: Uploaded image file

        Returns:
            PredictionResponse with class, confidence, and probabilities

        Raises:
            HTTPException: If no model loaded or prediction fails
        """
        if server.model is None:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. POST to /load_model first."
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
            )

            logger.info(
                f"Prediction: {class_name} (confidence: {confidence:.4f}, "
                f"inference: {inference_time:.2f}ms)"
            )
            return response

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
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
