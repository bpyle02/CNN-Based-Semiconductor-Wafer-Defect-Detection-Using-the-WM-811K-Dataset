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
echo "[1/4] Import check (src/)"
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
echo "[2/4] CLI help renders"
python train.py --help > /dev/null
echo "  train.py --help OK"

echo
echo "[3/4] Doctor reports healthy env"
python scripts/doctor.py --json | python -c "
import json, sys
payload = json.load(sys.stdin)
status = payload.get('status', '?')
print(f'  doctor status: {status}')
if status == 'error':
    print('  FAIL: doctor reports error')
    sys.exit(1)
"

# Intentionally no dataset-dependent training step here — `--synthetic` in
# train.py still requires data/LSWMD_new.pkl, which CI doesn't have. For a
# full-dataset 1-epoch smoke on a dev machine, run `make smoke` instead.

echo
echo "[4/4] Test suite (fast subset)"
pytest -q tests/unit --maxfail=3 -x

END=$(date +%s)
echo
echo "=== smoke test passed in $((END - START))s ==="
