# Inference Server Quick Start

## Start Server (30 seconds)

```bash
# 1. Install dependencies (if not already done)
pip install fastapi uvicorn python-multipart

# 2. Start server
python inference_server.py --port 8000

# 3. Open API docs in browser
# Navigate to: http://localhost:8000/docs
```

## Make Predictions (from another terminal)

### Option A: Load model + Predict (cURL)

```bash
# Load model
curl -X POST http://localhost:8000/load_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "cnn",
    "checkpoint_path": "checkpoints/best_cnn.pth"
  }'

# Predict on file
curl -X POST http://localhost:8000/predict_file \
  -F "file=@path/to/wafer_image.png"
```

### Option B: Python Client

```python
import requests

# Load model
requests.post('http://localhost:8000/load_model', json={
    'model_type': 'cnn',
    'checkpoint_path': 'checkpoints/best_cnn.pth'
})

# Predict on file
with open('wafer_image.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict_file',
        files={'file': f}
    )

prediction = response.json()
print(f"Predicted: {prediction['class_name']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### Option C: File Upload via Web UI

1. Open http://localhost:8000/docs
2. Click "Try it out" on `/predict_file` endpoint
3. Upload your image file
4. See results in response body

## Available Endpoints

| Endpoint | Purpose |
|----------|---------|
| GET /health | Check server status |
| GET /models | List available architectures |
| GET /model_info | Get loaded model details |
| POST /load_model | Load checkpoint |
| POST /predict | Predict on base64 image |
| POST /predict_file | Predict on uploaded file |

## Class Names (Predicted Values)

```
0: Center
1: Donut
2: Edge-Loc
3: Edge-Ring
4: Loc
5: Near-full
6: Random
7: Scratch
8: none
```

## Response Example

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
  "inference_ms": 45.23
}
```

## Common Commands

```bash
# Start with GPU
python inference_server.py --device cuda --port 8000

# Start and load model immediately
python inference_server.py \
  --model checkpoints/best_cnn.pth \
  --model-type cnn

# Start on external IP (for remote access)
python inference_server.py --host 0.0.0.0 --port 8000

# Run tests
python test_inference_server.py

# View help
python inference_server.py --help
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError: fastapi" | `pip install fastapi uvicorn python-multipart` |
| "No model loaded" (503 error) | POST to `/load_model` endpoint first |
| "CUDA not available" | Use `--device cpu` or check GPU drivers |
| Server won't start | Check port 8000 is not in use: `lsof -i :8000` |

## Files

- **Server code**: `src/inference/server.py`
- **CLI**: `inference_server.py`
- **Tests**: `test_inference_server.py`
- **User docs**: `INFERENCE_SERVER_README.md`
- **Architecture**: `docs/INFERENCE_SERVER_ARCHITECTURE.md`
- **Summary**: `IMPLEMENTATION_SUMMARY_15.md`

## Next Steps

1. **Production deployment**: See Docker section in `INFERENCE_SERVER_README.md`
2. **Custom preprocessing**: Extend `ModelServer.preprocess_image()`
3. **Batch predictions**: Implement `/predict_batch` endpoint
4. **GradCAM**: Enable `return_gradcam=true` in `/predict` (when implemented)
5. **Model registry**: Build persistent model list (future enhancement)

## Performance

**Typical latency (CPU):**
- Load model: 1-2 seconds
- Predict image: 30-100ms
- All 9 class probabilities returned

**Concurrent requests:**
- 1 worker: ~10 images/sec
- 4 workers: ~40 images/sec
- GPU: 100+ images/sec

## Support

- Full API documentation: http://localhost:8000/docs
- Architecture details: See `docs/INFERENCE_SERVER_ARCHITECTURE.md`
- Examples: See `INFERENCE_SERVER_README.md`
- Tests: Run `python test_inference_server.py`
