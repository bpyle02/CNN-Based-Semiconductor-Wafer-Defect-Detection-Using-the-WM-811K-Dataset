#!/bin/bash
# Wait for training to complete and finalize submission

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================="
echo "Waiting for training to complete..."
echo "======================================================================="

# Poll for completion (check every 30 seconds)
MAX_WAIT=7200  # 2 hours max
ELAPSED=0
POLL_INTERVAL=30

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Try to read latest output from train.py
    if [ -f "train_output.log" ]; then
        if tail -5 train_output.log | grep -q "RESULTS SUMMARY"; then
            echo "✓ Training complete!"
            break
        fi
    fi

    echo "Still training... (${ELAPSED}s elapsed)"
    sleep $POLL_INTERVAL
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "✗ Timeout waiting for training (exceeded 2 hours)"
    exit 1
fi

echo ""
echo "======================================================================="
echo "Extracting results and updating report..."
echo "======================================================================="

python scripts/extract_and_update_report.py < train_output.log

echo ""
echo "======================================================================="
echo "Compiling LaTeX documents..."
echo "======================================================================="

cd docs
pdflatex -interaction=nonstopmode wafer_defect_detection_report.tex
pdflatex -interaction=nonstopmode wafer_defect_detection_report.tex
pdflatex -interaction=nonstopmode presentation.tex

echo ""
echo "======================================================================="
echo "Submission ready!"
echo "======================================================================="
ls -lh wafer_defect_detection_report.pdf presentation.pdf

exit 0
