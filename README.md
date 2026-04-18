# CNN-Based Semiconductor Wafer Defect Detection Using the WM-811K Dataset
Penn State World Campus AI 570 Group Project

## Project Overview
This project implements a high-performance Hybrid CNN model to detect and classify defects in semiconductor wafer maps using the WM-811K dataset. It combines deep feature extraction (ResNet-18) with raw geometric measurements (Area, Eccentricity, etc.) to achieve superior accuracy on difficult minority classes like "Scratches."

## Key Features
- **Hybrid Architecture:** CNN + Geometric feature fusion.
- **Advanced Augmentation:** Uses **Albumentations** (ElasticTransform, GridDistortion) for industrial-grade robustness.
- **Explainability:** **Captum** integration for official PyTorch model attribution (LayerGradCam).
- **MPS Optimization:** Fully optimized for MPS-accelerated hardware using **multi-threaded data loading**.
- **Imbalance Handling:** Focal Loss and WeightedRandomSampling to manage 989x class imbalance.

## Performance
- **Test Accuracy:** 95.96%
- **Recall (Scratch):** 88%

## Usage
1. **Environment Setup:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train & Benchmark:**
   ```bash
   python src/training/train.py
   ```
3. **Evaluate & Explain:**
   ```bash
   python src/analysis/evaluate.py
   ```
   *Generates Normalized Confusion Matrix and Captum Attribution Plots.*
