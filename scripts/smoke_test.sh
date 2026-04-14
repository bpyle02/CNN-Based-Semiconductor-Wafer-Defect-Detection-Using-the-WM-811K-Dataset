#!/usr/bin/env bash
# Quick end-to-end smoke test — ~30s on CPU.
#
# Goal: catch regressions that full pytest might miss (missing model class,
# broken CLI arg parsing, runtime import cycle, config schema drift) without
# requiring the real WM-811K dataset or a GPU.
#
# Runs on every PR via .github/workflows/ci.yml. Exits non-zero on any
# failure so CI blocks the merge.

set -euo pipefail

echo "=== smoke test start ==="
START=$(date +%s)

echo
echo "[1/5] Import check (src/)"
python -c "
from src.models import WaferCNN, WaferResNet, WaferEfficientNet
from src.training.trainer import train_model
from src.data.dataset import WaferMapDataset, KNOWN_CLASSES
from src.analysis.evaluate import evaluate_model
from src.inference.gradcam import GradCAM
print('  all imports OK')
print(f'  known classes: {len(KNOWN_CLASSES)} classes')
"

echo
echo "[2/5] CLI help renders"
python train.py --help > /dev/null
echo "  train.py --help OK"

echo
echo "[3/5] Doctor reports healthy env"
python scripts/doctor.py --json | python -c "
import json, sys
payload = json.load(sys.stdin)
status = payload.get('status', '?')
print(f'  doctor status: {status}')
if status == 'error':
    print('  FAIL: doctor reports error')
    sys.exit(1)
"

echo
echo "[4/5] One-epoch train on synthetic data"
python train.py --model cnn --epochs 1 --batch-size 32 --device cpu --seed 42 --synthetic > /tmp/smoke_train.log 2>&1 || {
  echo "  FAIL: see log tail below"
  tail -40 /tmp/smoke_train.log
  exit 1
}
echo "  1-epoch train OK"

echo
echo "[5/5] Test suite (fast subset)"
pytest -q tests/unit --maxfail=3 -x

END=$(date +%s)
echo
echo "=== smoke test passed in $((END - START))s ==="
