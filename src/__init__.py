"""
CNN-Based Semiconductor Wafer Defect Detection Package

A comprehensive deep learning pipeline for classifying defects in semiconductor
wafer maps using the WM-811K dataset. Implements three architectures: custom CNN,
ResNet-18, and EfficientNet-B0 with transfer learning and interpretability via GradCAM.

Modules:
    data: Dataset loading, preprocessing, and augmentation
    models: Custom CNN and pretrained architecture implementations
    training: Training loop, configuration, and scheduling
    analysis: Evaluation metrics and visualization utilities
    inference: GradCAM interpretability and prediction
"""

__version__ = "0.2.1"
__author__ = "Team: Anindita Paul, Brandon Pyle, Anand Rajan, Brett Rettura"

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
