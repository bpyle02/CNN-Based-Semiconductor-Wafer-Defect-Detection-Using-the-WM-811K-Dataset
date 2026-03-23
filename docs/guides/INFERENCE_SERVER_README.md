# Real-Time Inference Server

Production-ready FastAPI server for wafer defect classification with TorchServe integration.

## Features

- **Multi-model serving**: Support for CNN, ResNet-18, and EfficientNet-B0 architectures
- **Dynamic model loading**: Load checkpoints at runtime via REST API
- **Flexible input**: Base64-encoded images or file uploads
- **Comprehensive validation**: Input size, format, and error handling
- **CORS support**: Web client integration
- **Async endpoints**: High-concurrency inference
- **Production-ready**: Logging, health checks, model introspection

## Quick Start

### Installation

Ensure dependencies are installed:

```bash
pip install fastapi uvicorn python-multipart torch torchvision
```

### Start Server

```bash
# Default (CPU, port 8000)
python inference_server.py

# With GPU
python inference_server.py --device cuda --port 8000

# Load model on startup
python inference_server.py \
  --model checkpoints/best_cnn.pth \
  --model-type cnn \
  --device cuda \
  --port 8000

# Bind to all interfaces (for remote access)
python inference_server.py --host 0.0.0.0 --port 8000
```

### View API Documentation

Open browser and navigate to: `http://localhost:8000/docs` (Swagger UI)

Alternative: `http://localhost:8000/redoc` (ReDoc)

## Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "current_model": "best_cnn",
  "device": "cpu",
  "torch_version": "1.13.0"
}
```

### List Models

```bash
GET /models
```

Response:
```json
{
  "available_architectures": ["cnn", "resnet", "efficientnet"],
  "current_model": "best_cnn"
}
```

### Load Model

```bash
POST /load_model
Content-Type: application/json

{
  "model_type": "cnn",
  "checkpoint_path": "checkpoints/best_cnn.pth"
}
```

Response:
```json
{
  "success": true,
  "message": "Model loaded successfully",
  "model_name": "best_cnn",
  "model_type": "cnn",
  "device": "cpu",
  "total_parameters": 1208233,
  "trainable_parameters": 1208233,
  "checkpoint_path": "checkpoints/best_cnn.pth"
}
```

### Predict (Base64 Image)

```bash
POST /predict
Content-Type: application/json

{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "model_name": null,
  "return_gradcam": false
}
```

Response:
```json
{
  "class_name": "Center",
  "class_id": 0,
  "confidence": 0.8234,
  "probabilities": {
    "Center": 0.8234,
    "Donut": 0.0512,
    "Edge-Loc": 0.0321,
    "Edge-Ring": 0.0278,
    "Loc": 0.0198,
    "Near-full": 0.0156,
    "Random": 0.0142,
    "Scratch": 0.0128,
    "none": 0.0031
  },
  "model_name": "best_cnn",
  "input_shape": [96, 96],
  "inference_ms": 45.23,
  "gradcam_base64": null
}
```

### Predict (File Upload)

```bash
POST /predict_file
Content-Type: multipart/form-data

file: <binary PNG/JPEG/BMP file>
```

Response: Same as `/predict` endpoint

### Get Model Info

```bash
GET /model_info
```

Response:
```json
{
  "name": "best_cnn",
  "architecture": "cnn",
  "device": "cpu",
  "num_parameters": 1208233,
  "num_trainable_parameters": 1208233,
  "input_size": [96, 96],
  "num_classes": 9,
  "classes": [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none"
  ]
}
```

## Usage Examples

### Python Client

```python
import requests
import base64
from pathlib import Path

# Load server URL
BASE_URL = "http://localhost:8000"

# Load model
response = requests.post(
    f"{BASE_URL}/load_model",
    json={
        "model_type": "cnn",
        "checkpoint_path": "checkpoints/best_cnn.pth"
    }
)
print(response.json())

# Predict on file
with open("wafer_image.png", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/predict_file", files=files)
    prediction = response.json()
    print(f"Predicted: {prediction['class_name']} ({prediction['confidence']:.2%})")

# Predict on base64
image_path = Path("wafer_image.png")
with open(image_path, "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post(
    f"{BASE_URL}/predict",
    json={"image_base64": image_data}
)
prediction = response.json()
print(f"All probabilities: {prediction['probabilities']}")
```

### cURL

```bash
# Load model
curl -X POST http://localhost:8000/load_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "cnn",
    "checkpoint_path": "checkpoints/best_cnn.pth"
  }'

# Health check
curl http://localhost:8000/health

# Predict on file
curl -X POST http://localhost:8000/predict_file \
  -F "file=@wafer_image.png"

# Get model info
curl http://localhost:8000/model_info
```

### JavaScript/Node.js

```javascript
// Load model
const loadModelResponse = await fetch('http://localhost:8000/load_model', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model_type: 'cnn',
    checkpoint_path: 'checkpoints/best_cnn.pth'
  })
});
const loadResult = await loadModelResponse.json();
console.log('Model loaded:', loadResult.model_name);

// Predict on file
const formData = new FormData();
formData.append('file', imageFile);  // from input[type=file]
const predictResponse = await fetch('http://localhost:8000/predict_file', {
  method: 'POST',
  body: formData
});
const prediction = await predictResponse.json();
console.log(`Predicted: ${prediction.class_name} (${(prediction.confidence * 100).toFixed(1)}%)`);

// Predict on base64
const imageBase64 = await fileToBase64(imageFile);
const predictResponse = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image_base64: imageBase64 })
});
const prediction = await predictResponse.json();
console.log('Probabilities:', prediction.probabilities);
```

## CLI Options

```
--host HOST              Host to bind to (default: 127.0.0.1)
--port PORT              Port number (default: 8000)
--device {cpu,cuda}      Device for inference (default: cpu)
--model MODEL            Checkpoint path to load on startup
--model-type {cnn,resnet,efficientnet}
                         Model architecture (required with --model)
--workers WORKERS        Number of worker processes (default: 1)
--reload                 Auto-reload on code changes (dev only)
--log-level {debug,info,warning,error}
                         Logging level (default: info)
--help                   Show help message
```

## Architecture

### ModelServer Class

Low-level model management:

```python
from src.inference.server import ModelServer, ModelType

server = ModelServer(device="cpu")
server.load_model(ModelType.CNN, "checkpoints/best_cnn.pth")

class_id, probabilities, inference_ms = server.predict(image_array)
print(f"Predicted: {class_id}, Confidence: {probabilities[class_id]:.4f}")
```

### create_app Function

FastAPI application factory:

```python
from src.inference.server import create_app, ModelType

app = create_app(
    device="cpu",
    model_checkpoint="checkpoints/best_cnn.pth",
    model_type=ModelType.CNN
)

# Use with uvicorn or test with TestClient
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.get("/health")
```

## Input Validation

### Image Requirements

- **Format**: PNG, JPEG, BMP
- **Size**: 32-512 pixels (auto-resized to 96x96)
- **Channels**: Grayscale or RGB (converted to grayscale if needed)
- **Base64 limit**: 10 MB

### Error Responses

| Status | Scenario | Example |
|--------|----------|---------|
| 200 | Success | Valid prediction response |
| 400 | Invalid request | Malformed JSON, bad image data |
| 422 | Validation error | Invalid base64, missing fields |
| 503 | No model loaded | Need to POST to /load_model first |
| 500 | Server error | Exception during inference |

## Performance

Typical inference times (CPU, CNN model):

- **Single image**: 30-60ms
- **Batch of 10**: 80-120ms total
- **Memory usage**: ~300MB (model + 100 concurrent requests)

GPU (CUDA):

- **Single image**: 5-15ms
- **Batch of 10**: 20-40ms total
- **Memory usage**: ~600MB GPU, 200MB CPU

## Testing

Run the comprehensive test suite:

```bash
python test_inference_server.py
```

Tests:

- ModelServer direct usage
- Endpoint health checks
- Model loading and switching
- Prediction with base64 and file upload
- Error handling and validation
- Parameter counting

## Production Deployment

### With Gunicorn (multiple workers)

```bash
gunicorn \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile - \
  'src.inference.server:create_app()'
```

### With Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "src.inference.server:create_app", "--host", "0.0.0.0", "--port", "8000"]
```

### Health Check

For container orchestration:

```bash
curl --fail http://localhost:8000/health || exit 1
```

## TorchServe Integration

The `ModelServer` class is compatible with TorchServe's model format. To export:

```python
import torch
from src.models import WaferCNN

model = WaferCNN(num_classes=9)
# ... train model ...
torch.save(model.state_dict(), "model.pth")
```

Load in TorchServe:

```bash
torch-model-archiver \
  --model-name wafer_cnn \
  --version 1.0 \
  --model-file src/models/cnn.py \
  --serialized-file model.pth \
  --handler handler.py

torchserve --start --model-store model_store
```

## Troubleshooting

### "No module named 'fastapi'"

```bash
pip install fastapi uvicorn python-multipart
```

### "CUDA not available"

Server automatically falls back to CPU. Explicitly use `--device cpu` if needed.

### "No model loaded" (503 error)

Load a model first:

```bash
curl -X POST http://localhost:8000/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_type": "cnn", "checkpoint_path": "checkpoints/best_cnn.pth"}'
```

### Slow inference

- Check device with `/health` endpoint
- Ensure model is loaded with `/model_info`
- Profile with `--log-level debug`

## API Versioning

Future versions:

- `/v2/predict` - Batch predictions
- `/v2/models` - Model registry
- `/v2/explain` - GradCAM explanations
- Streaming responses for large batches

## Related Files

- `src/inference/server.py` - FastAPI server implementation
- `inference_server.py` - CLI entry point
- `test_inference_server.py` - Comprehensive test suite
- `src/models/` - Model architectures (CNN, ResNet, EfficientNet)
- `src/data/preprocessing.py` - Image preprocessing logic

## License

This inference server is part of the wafer defect detection project. See LICENSE file for details.
