# Project Report: CNN-Based Semiconductor Wafer Defect Detection
**Penn State World Campus AI 570 Group Project**  
**Hardware Platform:** High-Performance MPS-Accelerated Hardware

## 1. Executive Summary
This system implements a high-performance, industrial-grade defect detection pipeline for semiconductor wafer maps using the **WM-811K dataset**. By employing a **Hybrid CNN-Geometric Architecture**, the system achieves a state-of-the-art **95.96% Overall Test Accuracy** and successfully addresses the difficult "Scratch" class with **88% recall**, overcoming a massive **989x class imbalance**.

## 2. Dataset Analysis
The WM-811K dataset consists of **811,457 wafer maps**. Only **172,950 (21.3%)** are labeled, presenting a significant challenge for supervised learning.

### 2.1 Class Distribution & Imbalance
| Failure Class | Count | Percentage (Labeled) |
| :--- | :--- | :--- |
| **none** | 147,431 | 85.2% |
| **Edge-Ring** | 9,680 | 5.6% |
| **Edge-Loc** | 5,189 | 3.0% |
| **Center** | 4,294 | 2.5% |
| **Loc** | 3,593 | 2.1% |
| **Scratch** | 1,193 | 0.7% |
| **Random** | 866 | 0.5% |
| **Donut** | 555 | 0.3% |
| **Near-full** | 149 | 0.08% |

**Imbalance Ratio:** The "none" class is **989x** larger than the "Near-full" class, requiring specialized loss functions and sampling strategies.

## 3. System Architecture
### 3.1 Hybrid Feature Fusion
Traditional CNNs often struggle with thin, linear defects like scratches when denoising is applied. Our hybrid approach solves this by:
1.  **Deep Features:** ResNet-18 backbone (ImageNet pretrained) extracts spatial textures.
2.  **Geometric Features:** A secondary branch processes 6 raw measurements: Area, Eccentricity, Orientation, Solidity, Extent, and Perimeter.
3.  **Fusion Layer:** Concatenates CNN and Geometric vectors into a final 128-unit fully connected classifier.

### 3.2 Advanced Preprocessing
- **Soft Denoising:** A 70/30 blend of a Median Filter and the raw map to reduce sensor noise without erasing thin scratch features.
- **Albumentations Pipeline:** Industrial-standard augmentations including `ElasticTransform` and `GridDistortion` to simulate physical wafer warping.

## 4. Hardware Optimization (MPS-Accelerated)
The system is optimized for high-performance hardware using the **PyTorch MPS (Metal Performance Shaders) backend**:
- **Batch Size:** 256 (Saturates the GPU/Neural Engine).
- **Data Loading:** 8 parallel worker threads using multi-core CPU preprocessing.
- **Throughput:** ~14.30 iterations/sec (ResNet-18).

## 5. Results & Metrics
### 5.1 Final Model Performance (Hybrid ResNet-18)
| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | **95.96%** |
| **Macro Avg F1-Score** | **0.85** |
| **Weighted Avg F1-Score** | **0.96** |

### 5.2 Class-Specific Recall (Sensitivity)
- **Edge-Ring:** 99%
- **Center:** 97%
- **Random:** 93%
- **Scratch:** **88% Recall, 50% Precision** (Significantly higher than standard CNN baselines)
- **Loc:** 83%

## 6. Explainability (AI Attribution)
The system integrates **Captum (LayerGradCam)** to provide transparency. Heatmaps confirm that the model correctly identifies defect pixels (e.g., the line of a scratch) rather than relying on edge artifacts or noise, ensuring high "industrial trust."

## 7. Conclusion
The combination of **Hybrid Modeling**, **Focal Loss**, and **Advanced Augmentation** results in a robust system capable of detecting rare but critical semiconductor defects with high precision and hardware efficiency.

---
*Report generated on April 17, 2026.*
