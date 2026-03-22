#!/usr/bin/env python
"""
CLI entry point for the real-time wafer defect detection inference server.

Usage:
    # Start server on port 8000 with CPU
    python inference_server.py --port 8000

    # Start server on GPU with a specific model
    python inference_server.py --model checkpoints/best_cnn.pth --model-type cnn --device cuda --port 8000

    # Start server with different host binding
    python inference_server.py --host 0.0.0.0 --port 8000

    # View API documentation
    Open http://localhost:8000/docs in a web browser after starting

Example requests:
    # Health check
    curl http://localhost:8000/health

    # Load a model
    curl -X POST http://localhost:8000/load_model \
      -H "Content-Type: application/json" \
      -d '{"model_type": "cnn", "checkpoint_path": "checkpoints/best_cnn.pth"}'

    # Predict on uploaded file
    curl -X POST http://localhost:8000/predict_file \
      -F "file=@test_image.png"

    # Predict on base64-encoded image
    cat image.png | base64 | jq -R '{image_base64: .}' | curl -X POST \
      http://localhost:8000/predict -d @- -H "Content-Type: application/json"
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from src.inference.server import create_app, ModelType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> int:
    """
    Parse command-line arguments and start the inference server.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Real-time wafer defect detection inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 for external access)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference (default: cpu)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint to load on startup (.pth file)"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["cnn", "resnet", "efficientnet"],
        help="Model architecture type (required if --model is specified)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development only)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.model is not None:
        if args.model_type is None:
            parser.error("--model-type is required when --model is specified")
        if not Path(args.model).exists():
            parser.error(f"Model checkpoint not found: {args.model}")

    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                args.device = "cpu"
        except ImportError:
            logger.warning("PyTorch not available, using CPU")
            args.device = "cpu"

    # Create FastAPI app
    logger.info(f"Creating FastAPI application...")
    model_type = ModelType(args.model_type) if args.model_type else None
    app = create_app(
        device=args.device,
        model_checkpoint=args.model,
        model_type=model_type
    )

    # Start server
    logger.info(f"Starting inference server on {args.host}:{args.port}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Workers: {args.workers}")
    if args.model:
        logger.info(f"Loaded model: {args.model} ({args.model_type})")
    logger.info(f"API documentation: http://{args.host}:{args.port}/docs")
    logger.info(f"Alternative UI: http://{args.host}:{args.port}/redoc")

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level=args.log_level,
        )
        return 0
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Server failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
