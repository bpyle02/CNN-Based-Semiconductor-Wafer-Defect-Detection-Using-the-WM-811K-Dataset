#!/usr/bin/env python3
"""
Fetch academic papers, extract text, and analyze against codebase.

Usage:
    python scripts/fetch_papers.py --download          # Download PDFs
    python scripts/fetch_papers.py --extract           # Extract text from PDFs
    python scripts/fetch_papers.py --analyze           # Analyze against codebase
    python scripts/fetch_papers.py --all               # Full pipeline
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "references" / "papers"
TEXT_DIR = PROJECT_ROOT / "references" / "text"
ANALYSIS_DIR = PROJECT_ROOT / "references" / "analysis"


@dataclass
class Paper:
    """Metadata for an academic paper."""
    id: int
    title: str
    authors: str
    year: int
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    topic: str = ""
    relevance: str = ""
    modules: List[str] = field(default_factory=list)
    pdf_path: Optional[str] = None
    text_path: Optional[str] = None


# ─── Paper Registry ────────────────────────────────────────────────────────

PAPERS: List[Paper] = [
    # Core CNN Architectures
    Paper(1, "Deep Residual Learning for Image Recognition",
          "He, Zhang, Ren, Sun", 2016, arxiv_id="1512.03385",
          topic="architecture", relevance="ResNet-18 backbone",
          modules=["src/models/pretrained.py"]),
    Paper(2, "EfficientNet: Rethinking Model Scaling for CNNs",
          "Tan, Le", 2019, arxiv_id="1905.11946",
          topic="architecture", relevance="EfficientNet-B0 backbone",
          modules=["src/models/pretrained.py"]),
    Paper(3, "An Image is Worth 16x16 Words: Transformers for Image Recognition",
          "Dosovitskiy et al.", 2021, arxiv_id="2010.11929",
          topic="architecture", relevance="ViT architecture",
          modules=["src/models/vit.py"]),
    Paper(4, "Spatial Pyramid Pooling in Deep CNNs",
          "He, Zhang, Ren, Sun", 2015, arxiv_id="1406.4729",
          topic="architecture", relevance="SPP layer implementation",
          modules=["src/models/cnn.py"]),
    Paper(5, "Very Deep Convolutional Networks for Large-Scale Image Recognition",
          "Simonyan, Zisserman", 2015, arxiv_id="1409.1556",
          topic="architecture", relevance="VGGNet design patterns",
          modules=["src/models/cnn.py"]),
    Paper(6, "ImageNet Classification with Deep CNNs",
          "Krizhevsky, Sutskever, Hinton", 2012,
          doi="10.1145/3065386",
          topic="architecture", relevance="Foundational CNN design",
          modules=["src/models/cnn.py"]),

    # Wafer Defect Detection
    Paper(7, "Wafer Map Failure Pattern Recognition and Similarity Ranking",
          "Wu, Yeh, Chen", 2014,
          doi="10.1109/TSM.2014.2364237",
          topic="wafer", relevance="WM-811K dataset origin",
          modules=["src/data/dataset.py", "src/data/preprocessing.py"]),
    Paper(8, "Wafer Map Defect Pattern Classification Using CNN",
          "Nakazawa, Kulkarni", 2018,
          doi="10.1109/ISQED.2018.8357292",
          topic="wafer", relevance="CNN for wafer defects",
          modules=["src/models/cnn.py"]),
    Paper(9, "Wafer Defect Pattern Recognition and Analysis Based on CNN",
          "Yu, Lu, Zheng", 2019,
          doi="10.1109/TSM.2019.2963656",
          topic="wafer", relevance="Deep learning wafer classification",
          modules=["src/models/cnn.py", "src/data/preprocessing.py"]),
    Paper(10, "Deep Learning Based Wafer Map Defect Pattern Classification",
          "Kim et al.", 2020,
          doi="10.1109/ACCESS.2020.3040684",
          topic="wafer", relevance="Multi-class wafer defect DL",
          modules=["train.py"]),
    Paper(11, "WaPIRL: Wafer Pattern Identification using Representation Learning",
          "Kang et al.", 2021,
          doi="10.1109/TSM.2021.3064435",
          topic="wafer", relevance="Representation learning for wafer maps",
          modules=["src/training/simclr.py"]),
    Paper(12, "Mixed-Type Wafer Defect Recognition with Multi-Scale Info Fusion",
          "Wang et al.", 2020,
          doi="10.1109/TSM.2020.3003161",
          topic="wafer", relevance="Multi-scale defect features",
          modules=["src/models/cnn.py"]),
    Paper(13, "Semiconductor Defect Detection by Hybrid Classical-Quantum DL",
          "Alam et al.", 2022, arxiv_id="2206.09912",
          topic="wafer", relevance="Modern semiconductor QC approaches",
          modules=["src/analysis/evaluate.py"]),
    Paper(14, "Deformable CNNs for Wafer Defect Pattern Detection",
          "Tsai, Wang", 2020,
          doi="10.1109/TSM.2020.2997342",
          topic="wafer", relevance="Deformable convolutions for wafer maps",
          modules=["src/models/cnn.py"]),

    # Class Imbalance
    Paper(15, "Focal Loss for Dense Object Detection",
          "Lin, Goyal, Girshick, He, Dollar", 2017, arxiv_id="1708.02002",
          topic="imbalance", relevance="FocalLoss implementation",
          modules=["src/training/losses.py"]),
    Paper(16, "Class-Balanced Loss Based on Effective Number of Samples",
          "Cui, Jia, Lin, Song, Belongie", 2019, arxiv_id="1901.05555",
          topic="imbalance", relevance="Class weighting strategy",
          modules=["src/training/losses.py", "train.py"]),
    Paper(17, "A Systematic Study of the Class Imbalance Problem in CNNs",
          "Buda, Maki, Mazurowski", 2018, arxiv_id="1710.05381",
          topic="imbalance", relevance="Imbalance mitigation strategies",
          modules=["train.py"]),
    Paper(18, "SMOTE: Synthetic Minority Over-sampling Technique",
          "Chawla, Bowyer, Hall, Kegelmeyer", 2002, arxiv_id="1106.1813",
          topic="imbalance", relevance="Synthetic augmentation basis",
          modules=["src/augmentation/synthetic.py"]),
    Paper(19, "Learning from Imbalanced Data",
          "He, Garcia", 2009,
          doi="10.1109/TKDE.2008.239",
          topic="imbalance", relevance="Foundational imbalance survey",
          modules=["src/training/losses.py"]),
    Paper(20, "When Does Label Smoothing Help?",
          "Muller, Kornblith, Hinton", 2019, arxiv_id="1906.02629",
          topic="imbalance", relevance="Label smoothing in loss functions",
          modules=["src/training/losses.py"]),

    # Attention Mechanisms
    Paper(21, "Squeeze-and-Excitation Networks",
          "Hu, Shen, Sun", 2018, arxiv_id="1709.01507",
          topic="attention", relevance="SEBlock implementation",
          modules=["src/models/attention.py"]),
    Paper(22, "CBAM: Convolutional Block Attention Module",
          "Woo, Park, Lee, Kweon", 2018, arxiv_id="1807.06521",
          topic="attention", relevance="CBAMBlock implementation",
          modules=["src/models/attention.py"]),
    Paper(23, "Attention Is All You Need",
          "Vaswani et al.", 2017, arxiv_id="1706.03762",
          topic="attention", relevance="Transformer self-attention foundation",
          modules=["src/models/vit.py"]),
    Paper(24, "Non-local Neural Networks",
          "Wang, Girshick, Gupta, He", 2018, arxiv_id="1711.07971",
          topic="attention", relevance="Non-local spatial attention",
          modules=["src/models/attention.py"]),

    # Interpretability
    Paper(25, "Grad-CAM: Visual Explanations from Deep Networks",
          "Selvaraju et al.", 2017, arxiv_id="1610.02391",
          topic="interpretability", relevance="GradCAM implementation",
          modules=["src/inference/gradcam.py"]),
    Paper(26, "Why Should I Trust You? Explaining Predictions of Any Classifier",
          "Ribeiro, Singh, Guestrin", 2016, arxiv_id="1602.04938",
          topic="interpretability", relevance="LIME interpretability baseline",
          modules=["src/inference/gradcam.py"]),
    Paper(27, "Learning Important Features Through Propagating Activation Differences",
          "Shrikumar, Greenside, Kundaje", 2017, arxiv_id="1704.02685",
          topic="interpretability", relevance="DeepLIFT attribution",
          modules=["src/inference/gradcam.py"]),

    # Uncertainty & Calibration
    Paper(28, "Dropout as a Bayesian Approximation: Representing Model Uncertainty",
          "Gal, Ghahramani", 2016, arxiv_id="1506.02142",
          topic="uncertainty", relevance="MC Dropout implementation",
          modules=["src/inference/uncertainty.py"]),
    Paper(29, "On Calibration of Modern Neural Networks",
          "Guo, Pleiss, Sun, Weinberger", 2017, arxiv_id="1706.04599",
          topic="uncertainty", relevance="Temperature scaling, ECE metrics",
          modules=["src/inference/uncertainty.py", "src/analysis/evaluate.py"]),
    Paper(30, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles",
          "Lakshminarayanan, Pritzel, Blundell", 2017, arxiv_id="1612.01474",
          topic="uncertainty", relevance="Deep ensembles for uncertainty",
          modules=["src/models/ensemble.py"]),
    Paper(31, "What Uncertainties Do We Need in Bayesian DL for Computer Vision?",
          "Kendall, Gal", 2017, arxiv_id="1703.04977",
          topic="uncertainty", relevance="Aleatoric vs epistemic uncertainty",
          modules=["src/inference/uncertainty.py"]),
    Paper(32, "Measuring Calibration in Deep Learning",
          "Nixon, Dusenberry, Zhang, Jerfel, Tran", 2019, arxiv_id="1904.01685",
          topic="uncertainty", relevance="ECE/MCE calibration metrics",
          modules=["src/analysis/evaluate.py"]),

    # Federated Learning
    Paper(33, "Communication-Efficient Learning of Deep Networks from Decentralized Data",
          "McMahan, Moore, Ramage, Hampson, Arcas", 2017, arxiv_id="1602.05629",
          topic="federated", relevance="FedAvg implementation",
          modules=["src/federated/fed_avg.py"]),
    Paper(34, "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent",
          "Blanchard, El Mhamdi, Guerraoui, Stainer", 2017, arxiv_id="1703.02757",
          topic="federated", relevance="Krum aggregation",
          modules=["src/federated/fed_avg.py"]),
    Paper(35, "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates",
          "Yin, Chen, Kannan, Bartlett", 2018, arxiv_id="1803.10032",
          topic="federated", relevance="Trimmed mean/median aggregation",
          modules=["src/federated/fed_avg.py"]),
    Paper(36, "Advances and Open Problems in Federated Learning",
          "Kairouz et al.", 2021, arxiv_id="1912.04977",
          topic="federated", relevance="Federated learning survey",
          modules=["src/federated/fed_avg.py"]),

    # Self-Supervised Learning
    Paper(37, "A Simple Framework for Contrastive Learning of Visual Representations",
          "Chen, Kornblith, Norouzi, Hinton", 2020, arxiv_id="2002.05709",
          topic="selfsupervised", relevance="SimCLR implementation",
          modules=["src/training/simclr.py"]),
    Paper(38, "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning",
          "Grill et al.", 2020, arxiv_id="2006.07733",
          topic="selfsupervised", relevance="BYOL pretraining",
          modules=["src/training/simclr.py"]),
    Paper(39, "Momentum Contrast for Unsupervised Visual Representation Learning",
          "He, Fan, Wu, Xie, Girshick", 2020, arxiv_id="1911.05722",
          topic="selfsupervised", relevance="MoCo contrastive learning",
          modules=["src/training/simclr.py"]),

    # Domain Adaptation
    Paper(40, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation",
          "Sun, Saenko", 2016, arxiv_id="1607.01719",
          topic="adaptation", relevance="CORAL implementation",
          modules=["src/training/domain_adaptation.py"]),
    Paper(41, "Domain-Adversarial Training of Neural Networks",
          "Ganin et al.", 2016, arxiv_id="1505.07818",
          topic="adaptation", relevance="Adversarial domain adaptation",
          modules=["src/training/domain_adaptation.py"]),

    # Model Compression
    Paper(42, "Distilling the Knowledge in a Neural Network",
          "Hinton, Vinyals, Dean", 2015, arxiv_id="1503.02531",
          topic="compression", relevance="Knowledge distillation",
          modules=["scripts/compress_model.py"]),
    Paper(43, "Deep Compression: Compressing DNNs with Pruning, Quantization, Huffman Coding",
          "Han, Mao, Dally", 2016, arxiv_id="1510.00149",
          topic="compression", relevance="Model compression pipeline",
          modules=["scripts/compress_model.py"]),
    Paper(44, "Quantization and Training of NNs for Efficient Integer-Arithmetic-Only Inference",
          "Jacob et al.", 2018, arxiv_id="1712.05877",
          topic="compression", relevance="INT8 quantization",
          modules=["scripts/compress_model.py"]),

    # Optimization
    Paper(45, "Adam: A Method for Stochastic Optimization",
          "Kingma, Ba", 2015, arxiv_id="1412.6980",
          topic="optimization", relevance="Adam optimizer",
          modules=["train.py"]),
    Paper(46, "Decoupled Weight Decay Regularization",
          "Loshchilov, Hutter", 2019, arxiv_id="1711.05101",
          topic="optimization", relevance="AdamW optimizer",
          modules=["train.py"]),
    Paper(47, "Batch Normalization: Accelerating Deep Network Training",
          "Ioffe, Szegedy", 2015, arxiv_id="1502.03167",
          topic="optimization", relevance="BatchNorm in CNN",
          modules=["src/models/cnn.py"]),
    Paper(48, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
          "Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov", 2014,
          doi="10.5555/2627435.2670313",
          topic="optimization", relevance="Dropout regularization",
          modules=["src/models/cnn.py", "src/inference/uncertainty.py"]),

    # Data Augmentation & Active Learning
    Paper(49, "A Survey on Image Data Augmentation for Deep Learning",
          "Shorten, Khoshgoftaar", 2019,
          doi="10.1186/s40537-019-0197-0",
          topic="augmentation", relevance="Augmentation strategies survey",
          modules=["src/data/preprocessing.py"]),
    Paper(50, "Deep Bayesian Active Learning with Image Data",
          "Gal, Islam, Ghahramani", 2017, arxiv_id="1703.02910",
          topic="active_learning", relevance="Active learning with uncertainty",
          modules=["scripts/active_learn.py"]),

    # ── Wafer / Semiconductor Manufacturing (51-65) ──────────────────────

    Paper(51, "Wafer Bin Map Defect Pattern Classification Using Convolutional Neural Network",
          "Kyeong, Kim", 2018,
          doi="10.1109/TSM.2018.2841416",
          topic="wafer", relevance="CNN architecture for WBM classification",
          modules=["src/models/cnn.py", "src/data/preprocessing.py"]),
    Paper(52, "Automatic Defect Classification for Semiconductor Manufacturing",
          "Cheon, Kim, Ham, Kim", 2019,
          doi="10.1109/TSM.2019.2941674",
          topic="wafer", relevance="Automated defect classification in fab",
          modules=["src/models/cnn.py"]),
    Paper(53, "Semi-Supervised Learning for Wafer Map Defect Pattern Classification",
          "Kahng, Kim", 2020,
          doi="10.1109/TSM.2020.3017809",
          topic="wafer", relevance="Semi-supervised methods for wafer maps",
          modules=["src/training/simclr.py"]),
    Paper(54, "Data Augmentation Using Learned Transformations for One-Shot Medical Image Segmentation",
          "Zhao, Data, Greenspan, St-Onge, Bhatt, Murphy, Grady", 2019, arxiv_id="1902.09383",
          topic="wafer", relevance="Learned augmentation for small datasets applicable to rare defects",
          modules=["src/augmentation/synthetic.py"]),
    Paper(55, "Smart Semiconductor Manufacturing: An Overview",
          "Moyne, Iskandar", 2017,
          doi="10.1109/TSM.2017.2768062",
          topic="wafer", relevance="Overview of ML in semiconductor manufacturing",
          modules=["train.py"]),
    Paper(56, "Deep Learning Approaches for Wafer Map Defect Pattern Recognition",
          "Jin, Kim, Kwon", 2020,
          doi="10.1109/ACCESS.2020.2990379",
          topic="wafer", relevance="Comparison of DL architectures for wafer defects",
          modules=["src/models/cnn.py", "src/models/pretrained.py"]),
    Paper(57, "Automated Visual Inspection of Semiconductor Wafers",
          "Shankar, Zhong", 2005,
          doi="10.1109/TSM.2005.852106",
          topic="wafer", relevance="Classical automated optical inspection foundations",
          modules=["src/analysis/evaluate.py"]),
    Paper(58, "Wafer Map Defect Detection and Recognition Using Joint Local and Nonlocal Linear Discriminant Analysis",
          "Yu, Zheng, Shan", 2016,
          doi="10.1109/TSM.2016.2578164",
          topic="wafer", relevance="Feature extraction for wafer map defects",
          modules=["src/data/preprocessing.py"]),
    Paper(59, "A Light-Weight CNN Model for Wafer Map Defect Detection",
          "Tsai, Lee", 2020,
          doi="10.1109/ACCESS.2020.3017358",
          topic="wafer", relevance="Lightweight models for edge deployment of wafer inspection",
          modules=["src/models/cnn.py", "scripts/compress_model.py"]),
    Paper(60, "Wafer Defect Pattern Recognition Using Transfer Learning",
          "Shim, Jeon, Choi", 2020,
          doi="10.1109/TSM.2020.3046888",
          topic="wafer", relevance="Transfer learning for wafer map classification",
          modules=["src/models/pretrained.py"]),
    Paper(61, "Yield Enhancement Through Wafer Map Spatial Pattern Recognition",
          "Hsu, Chien", 2007,
          doi="10.1109/TSM.2007.903705",
          topic="wafer", relevance="Spatial pattern recognition for yield improvement",
          modules=["src/data/dataset.py"]),
    Paper(62, "Virtual Metrology and Defect Prediction in Semiconductor Manufacturing",
          "Kang, Kim, Cho, Kang", 2016,
          doi="10.1109/TSM.2016.2535700",
          topic="wafer", relevance="Predictive quality in semiconductor processes",
          modules=["src/analysis/evaluate.py"]),
    Paper(63, "Wafer Map Defect Pattern Classification and Image Retrieval Using CNN",
          "Nakazawa, Kulkarni", 2019,
          doi="10.1109/ASMC.2019.8791815",
          topic="wafer", relevance="CNN plus retrieval for wafer map analysis",
          modules=["src/models/cnn.py"]),
    Paper(64, "Multi-Label Wafer Map Defect Pattern Classification Using Deep Learning",
          "Shin, Kim", 2022,
          doi="10.1109/TSM.2022.3178464",
          topic="wafer", relevance="Multi-label approach for mixed-type defect patterns",
          modules=["src/models/cnn.py", "train.py"]),
    Paper(65, "GAN-Based Synthetic Data Generation for Wafer Map Defect Pattern Classification",
          "Wang, Hsieh, Liu, Hsu", 2021,
          doi="10.1109/TSM.2021.3089869",
          topic="wafer", relevance="GAN augmentation for rare wafer defect classes",
          modules=["src/augmentation/synthetic.py"]),

    # ── Advanced Augmentation & Balancing (66-75) ────────────────────────

    Paper(66, "mixup: Beyond Empirical Risk Minimization",
          "Zhang, Cisse, Dauphin, Lopez-Paz", 2018, arxiv_id="1710.09412",
          topic="augmentation", relevance="Mixup data augmentation for improved generalization",
          modules=["src/data/preprocessing.py", "src/augmentation/synthetic.py"]),
    Paper(67, "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features",
          "Yun, Han, Oh, Chun, Choe, Yoo", 2019, arxiv_id="1905.04899",
          topic="augmentation", relevance="CutMix augmentation for better spatial feature learning",
          modules=["src/data/preprocessing.py", "src/augmentation/synthetic.py"]),
    Paper(68, "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning",
          "He, Bai, Garcia, Li", 2008,
          doi="10.1109/IJCNN.2008.4633969",
          topic="imbalance", relevance="Adaptive oversampling for hard-to-learn minority samples",
          modules=["src/augmentation/synthetic.py"]),
    Paper(69, "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning",
          "Han, Wang, Mao", 2005,
          doi="10.1007/11538059_91",
          topic="imbalance", relevance="Targeted oversampling near decision boundaries",
          modules=["src/augmentation/synthetic.py"]),
    Paper(70, "AutoAugment: Learning Augmentation Strategies from Data",
          "Cubuk, Zoph, Mane, Vasudevan, Le", 2019, arxiv_id="1805.09501",
          topic="augmentation", relevance="Automated augmentation policy search",
          modules=["src/data/preprocessing.py"]),
    Paper(71, "RandAugment: Practical Automated Data Augmentation with a Reduced Search Space",
          "Cubuk, Zoph, Shlens, Le", 2020, arxiv_id="1909.13719",
          topic="augmentation", relevance="Simple yet effective random augmentation policies",
          modules=["src/data/preprocessing.py"]),
    Paper(72, "Feature Space Augmentation for Long-Tailed Classification",
          "Chu, Zhong, Wang", 2020, arxiv_id="2008.03673",
          topic="augmentation", relevance="Augmenting in feature space for tail class improvement",
          modules=["src/augmentation/synthetic.py"]),
    Paper(73, "Class-Balanced Loss Based on Effective Number of Samples",
          "Cui, Jia, Lin, Song, Belongie", 2019, arxiv_id="1901.05555",
          topic="imbalance", relevance="Effective number re-weighting theory",
          modules=["src/training/losses.py"]),
    Paper(74, "Decoupling Representation and Classifier for Long-Tailed Recognition",
          "Kang, Xie, Rohrbach, Yan, Gordo, Feng, Kalantidis", 2020, arxiv_id="1910.09217",
          topic="imbalance", relevance="Decoupled training for class-imbalanced data",
          modules=["train.py", "src/models/pretrained.py"]),
    Paper(75, "Remix: Rebalanced Mixup",
          "Chou, Chen, Lee", 2020, arxiv_id="2007.03943",
          topic="augmentation", relevance="Combines Mixup with class-balanced resampling",
          modules=["src/data/preprocessing.py", "src/augmentation/synthetic.py"]),

    # ── Contrastive & Metric Learning (76-85) ────────────────────────────

    Paper(76, "Supervised Contrastive Learning",
          "Khosla, Teterwak, Wang, Swersky, Tian, Isola, Maschinot, Liu, Kornblith", 2020,
          arxiv_id="2004.11362",
          topic="contrastive", relevance="Supervised contrastive loss for improved class separation",
          modules=["src/training/simclr.py"]),
    Paper(77, "Prototypical Networks for Few-shot Learning",
          "Snell, Swersky, Zemel", 2017, arxiv_id="1703.05175",
          topic="metric_learning", relevance="Prototype-based classification for rare defect classes",
          modules=["src/models/cnn.py"]),
    Paper(78, "FaceNet: A Unified Embedding for Face Recognition and Clustering",
          "Schroff, Kalenichenko, Philbin", 2015, arxiv_id="1503.03832",
          topic="metric_learning", relevance="Triplet loss for embedding-based classification",
          modules=["src/training/simclr.py"]),
    Paper(79, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition",
          "Deng, Guo, Xue, Zafeiriou", 2019, arxiv_id="1801.07698",
          topic="metric_learning", relevance="Angular margin loss for fine-grained class discrimination",
          modules=["src/training/losses.py"]),
    Paper(80, "CosFace: Large Margin Cosine Loss for Deep Face Recognition",
          "Wang, Cheng, Gong, Zhu", 2018, arxiv_id="1801.09414",
          topic="metric_learning", relevance="Cosine margin loss for improved inter-class separation",
          modules=["src/training/losses.py"]),
    Paper(81, "A Discriminative Feature Learning Approach for Deep Face Recognition",
          "Wen, Zhang, Li, Qiao", 2016,
          doi="10.1007/978-3-319-46478-7_31",
          topic="metric_learning", relevance="Center loss for intra-class compactness",
          modules=["src/training/losses.py"]),
    Paper(82, "Deep Metric Learning: A Survey",
          "Kaya, Bilge", 2019, arxiv_id="1904.06626",
          topic="metric_learning", relevance="Survey of metric learning approaches for classification",
          modules=["src/training/simclr.py"]),
    Paper(83, "Proxy Anchor Loss for Deep Metric Learning",
          "Kim, Kim, Choi, You", 2020, arxiv_id="2003.13911",
          topic="metric_learning", relevance="Proxy-based metric learning for efficient training",
          modules=["src/training/losses.py"]),
    Paper(84, "Circle Loss: A Unified Perspective of Pair Similarity Optimization",
          "Sun, Cheng, Zhang, Lin, Liu, Wang", 2020, arxiv_id="2002.10857",
          topic="metric_learning", relevance="Unified loss for improved convergence on imbalanced data",
          modules=["src/training/losses.py"]),
    Paper(85, "Exploring Simple Siamese Representation Learning",
          "Chen, He", 2021, arxiv_id="2011.10566",
          topic="contrastive", relevance="SimSiam for simple self-supervised pretraining",
          modules=["src/training/simclr.py"]),

    # ── Graph Neural Networks (86-90) ────────────────────────────────────

    Paper(86, "Semi-Supervised Classification with Graph Convolutional Networks",
          "Kipf, Welling", 2017, arxiv_id="1609.02907",
          topic="gnn", relevance="GCN for spatial relationships between defect regions",
          modules=["src/models/cnn.py"]),
    Paper(87, "Graph Attention Networks",
          "Velickovic, Cucurull, Casanova, Romero, Lio, Bengio", 2018,
          arxiv_id="1710.10903",
          topic="gnn", relevance="Attention-weighted graph message passing for defect topology",
          modules=["src/models/attention.py"]),
    Paper(88, "Inductive Representation Learning on Large Graphs",
          "Hamilton, Ying, Leskovec", 2017, arxiv_id="1706.02216",
          topic="gnn", relevance="GraphSAGE for scalable graph learning on wafer maps",
          modules=["src/models/cnn.py"]),
    Paper(89, "Defect Detection Using Graph Neural Networks on Semiconductor Wafer Maps",
          "Park, Cho", 2021,
          doi="10.1109/TSM.2021.3117275",
          topic="gnn", relevance="GNN applied directly to wafer map defect detection",
          modules=["src/models/cnn.py", "src/data/preprocessing.py"]),
    Paper(90, "How Powerful Are Graph Neural Networks?",
          "Xu, Hu, Leskovec, Jegelka", 2019, arxiv_id="1810.00826",
          topic="gnn", relevance="GIN expressiveness theory for graph-level classification",
          modules=["src/models/cnn.py"]),

    # ── Multi-task & Multi-scale (91-100) ────────────────────────────────

    Paper(91, "Feature Pyramid Networks for Object Detection",
          "Lin, Dollar, Girshick, He, Hariharan, Belongie", 2017, arxiv_id="1612.03144",
          topic="multiscale", relevance="FPN for multi-scale defect feature extraction",
          modules=["src/models/cnn.py", "src/models/pretrained.py"]),
    Paper(92, "Deformable Convolutional Networks",
          "Dai, Qi, Xiong, Li, Zhang, Hu, Wei", 2017, arxiv_id="1703.06211",
          topic="multiscale", relevance="Deformable convolutions for irregular defect shapes",
          modules=["src/models/cnn.py"]),
    Paper(93, "An Overview of Multi-Task Learning in Deep Neural Networks",
          "Ruder", 2017, arxiv_id="1706.05098",
          topic="multitask", relevance="Multi-task learning survey for joint defect classification and localization",
          modules=["train.py"]),
    Paper(94, "U-Net: Convolutional Networks for Biomedical Image Segmentation",
          "Ronneberger, Fischer, Brox", 2015, arxiv_id="1505.04597",
          topic="multiscale", relevance="U-Net for defect segmentation and localization",
          modules=["src/models/cnn.py"]),
    Paper(95, "Multi-Task Learning as Multi-Objective Optimization",
          "Sener, Koltun", 2018, arxiv_id="1810.04650",
          topic="multitask", relevance="Multi-objective optimization for joint learning tasks",
          modules=["train.py"]),
    Paper(96, "Deep Multi-Task Learning with Cross Stitch Networks",
          "Misra, Shrivastava, Gupta, Hebert", 2016, arxiv_id="1604.03539",
          topic="multitask", relevance="Cross-stitch units for multi-task feature sharing",
          modules=["src/models/cnn.py"]),
    Paper(97, "PANet: Path Aggregation Network for Instance Segmentation",
          "Liu, Qi, Qin, Shi, Jia", 2018, arxiv_id="1803.01534",
          topic="multiscale", relevance="Bottom-up path aggregation for multi-scale features",
          modules=["src/models/cnn.py"]),
    Paper(98, "HRNet: Deep High-Resolution Representation Learning for Visual Recognition",
          "Wang, Sun, Liu, Sarma, Bronstein, Kitani", 2020, arxiv_id="1908.07919",
          topic="multiscale", relevance="High-resolution representations for fine-grained defect features",
          modules=["src/models/cnn.py"]),
    Paper(99, "Panoptic Feature Pyramid Networks",
          "Kirillov, Girshick, He, Dollar", 2019, arxiv_id="1901.02446",
          topic="multiscale", relevance="Unified multi-scale architecture for segmentation and classification",
          modules=["src/models/cnn.py"]),
    Paper(100, "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks",
          "Chen, Badrinarayanan, Lee, Rabinovich", 2018, arxiv_id="1711.02257",
          topic="multitask", relevance="Dynamic loss weighting for multi-task defect learning",
          modules=["train.py"]),

    # ── Modern Transformers & Hybrid (101-110) ───────────────────────────

    Paper(101, "Training Data-Efficient Image Transformers & Distillation Through Attention",
          "Touvron, Cord, Douze, Massa, Sablayrolles, Jegou", 2021,
          arxiv_id="2012.12877",
          topic="transformer", relevance="DeiT data-efficient transformer training for small datasets",
          modules=["src/models/vit.py"]),
    Paper(102, "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows",
          "Liu, Lin, Cao, Hu, Wei, Zhang, Lin, Guo", 2021, arxiv_id="2103.14030",
          topic="transformer", relevance="Swin hierarchical transformer for multi-scale wafer features",
          modules=["src/models/vit.py"]),
    Paper(103, "CvT: Introducing Convolutions to Vision Transformers",
          "Wu, Xu, Dai, Wan, Zhang, Yan, Tomizuka, Gonzalez, Keutzer, Vajda", 2021,
          arxiv_id="2103.15808",
          topic="transformer", relevance="Convolutional token embedding for hybrid CNN-ViT",
          modules=["src/models/vit.py"]),
    Paper(104, "CoAtNet: Marrying Convolution and Attention for All Data Sizes",
          "Dai, Liu, Le, Tan", 2021, arxiv_id="2106.04803",
          topic="transformer", relevance="Hybrid convolution-attention architecture",
          modules=["src/models/vit.py", "src/models/cnn.py"]),
    Paper(105, "Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet",
          "Yuan, Chen, Chen, Codella, Dai, Gao, Hu, Huang, Li, Li, Liu, Lu, Shi, Shu, Yuan, Zhu",
          2021, arxiv_id="2101.11986",
          topic="transformer", relevance="Token-to-Token progressive tokenization for ViT",
          modules=["src/models/vit.py"]),
    Paper(106, "LeViT: A Vision Transformer in ConvNet's Clothing for Faster Inference",
          "Graham, El-Nouby, Touvron, Stock, Joulin, Jegou, Douze", 2021,
          arxiv_id="2104.01136",
          topic="transformer", relevance="Fast hybrid transformer for inference efficiency",
          modules=["src/models/vit.py", "scripts/compress_model.py"]),
    Paper(107, "CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification",
          "Chen, Fan, Panda", 2021, arxiv_id="2103.14899",
          topic="transformer", relevance="Multi-scale dual-branch transformer",
          modules=["src/models/vit.py"]),
    Paper(108, "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction",
          "Wang, Xie, Li, Fan, Song, Liang, Lu, Luo, Shao", 2021,
          arxiv_id="2102.12122",
          topic="transformer", relevance="Pyramid transformer for multi-resolution defect features",
          modules=["src/models/vit.py"]),
    Paper(109, "PoolFormer: MetaFormer is Actually What You Need for Vision",
          "Yu, Li, Jiang, Yu, Shi, Wang", 2022, arxiv_id="2111.11418",
          topic="transformer", relevance="Token mixing without attention for efficient architectures",
          modules=["src/models/vit.py"]),
    Paper(110, "MLP-Mixer: An All-MLP Architecture for Vision",
          "Tolstikhin, Houlsby, Kolesnikov, Beyer, Zhai, Unterthiner, Yung, Steiner, Keysers, Uszkoreit, Lucic, Dosovitskiy",
          2021, arxiv_id="2105.01601",
          topic="transformer", relevance="MLP-only architecture as attention-free baseline",
          modules=["src/models/vit.py"]),

    # ── Semi-supervised & Few-shot (111-120) ─────────────────────────────

    Paper(111, "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence",
          "Sohn, Berthelot, Li, Zhang, Carlini, Cubuk, Kurakin, Zhang, Raffel", 2020,
          arxiv_id="2001.07685",
          topic="semisupervised", relevance="Semi-supervised learning for leveraging unlabeled wafer maps",
          modules=["src/training/simclr.py"]),
    Paper(112, "A Survey of Deep Meta-Learning",
          "Huisman, van Rijn, Plaat", 2021, arxiv_id="2010.03522",
          topic="fewshot", relevance="Meta-learning survey for few-shot defect classification",
          modules=["train.py"]),
    Paper(113, "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks",
          "Finn, Abbeel, Levine", 2017, arxiv_id="1703.03400",
          topic="fewshot", relevance="MAML for rapid adaptation to new defect types",
          modules=["train.py"]),
    Paper(114, "Learning to Propagate Labels: Transductive Propagation Network",
          "Liu, Lee, Park, Kim, Yang, Hwang", 2019, arxiv_id="1805.10002",
          topic="fewshot", relevance="Label propagation for semi-supervised wafer map classification",
          modules=["src/training/simclr.py"]),
    Paper(115, "MixMatch: A Holistic Approach to Semi-Supervised Learning",
          "Berthelot, Carlini, Goodfellow, Papernot, Oliver, Raffel", 2019,
          arxiv_id="1905.02249",
          topic="semisupervised", relevance="Combined consistency and entropy minimization",
          modules=["src/training/simclr.py"]),
    Paper(116, "Matching Networks for One Shot Learning",
          "Vinyals, Blundell, Lillicrap, Kavukcuoglu, Wierstra", 2016,
          arxiv_id="1606.04080",
          topic="fewshot", relevance="Attention-based few-shot classification",
          modules=["src/models/attention.py"]),
    Paper(117, "Meta-Learning with Differentiable Convex Optimization",
          "Lee, Maji, Ravichandran, Soatto", 2019, arxiv_id="1904.03758",
          topic="fewshot", relevance="Meta-learning with SVM base learner for few-shot tasks",
          modules=["train.py"]),
    Paper(118, "UDA: Unsupervised Data Augmentation for Consistency Training",
          "Xie, Dai, Hovy, Luong, Le", 2020, arxiv_id="1904.12848",
          topic="semisupervised", relevance="Unsupervised augmentation for semi-supervised training",
          modules=["src/data/preprocessing.py"]),
    Paper(119, "Temporal Ensembling for Semi-Supervised Learning",
          "Laine, Aila", 2017, arxiv_id="1610.02242",
          topic="semisupervised", relevance="Temporal ensemble for pseudo-label refinement",
          modules=["src/training/simclr.py"]),
    Paper(120, "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method",
          "Lee", 2013, arxiv_id="1908.02983",
          topic="semisupervised", relevance="Pseudo-labeling for unlabeled wafer map utilization",
          modules=["train.py"]),

    # ── Anomaly Detection & OOD (121-130) ────────────────────────────────

    Paper(121, "Deep One-Class Classification",
          "Ruff, Goernitz, Deecke, Siddiqui, Vandermeulen, Borghesi, Kloft, Muller", 2018,
          arxiv_id="1802.04365",
          topic="anomaly", relevance="Deep SVDD for anomaly-based defect detection",
          modules=["src/analysis/anomaly.py", "src/detection/ood.py"]),
    Paper(122, "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks",
          "Lee, Lee, Lee, Shin", 2018, arxiv_id="1807.03888",
          topic="anomaly", relevance="Mahalanobis distance for OOD detection in wafer classification",
          modules=["src/detection/ood.py"]),
    Paper(123, "Auto-Encoding Variational Bayes",
          "Kingma, Welling", 2014, arxiv_id="1312.6114",
          topic="anomaly", relevance="VAE for autoencoder-based anomaly detection",
          modules=["src/analysis/anomaly.py"]),
    Paper(124, "Energy-Based Out-of-Distribution Detection",
          "Liu, Wang, Owens, Li", 2020, arxiv_id="2010.03759",
          topic="anomaly", relevance="Energy-based OOD scoring for novel defect detection",
          modules=["src/detection/ood.py"]),
    Paper(125, "Anomaly Detection with Robust Deep Autoencoders",
          "Zhou, Paffenroth", 2017,
          doi="10.1145/3097983.3098052",
          topic="anomaly", relevance="Robust autoencoders for defect anomaly detection",
          modules=["src/analysis/anomaly.py"]),
    Paper(126, "Deep Anomaly Detection with Outlier Exposure",
          "Hendrycks, Mazeika, Dietterich", 2019, arxiv_id="1812.04606",
          topic="anomaly", relevance="Outlier exposure for improved OOD detection",
          modules=["src/detection/ood.py"]),
    Paper(127, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples",
          "Hendrycks, Gimpel", 2017, arxiv_id="1610.02136",
          topic="anomaly", relevance="Maximum softmax probability baseline for OOD detection",
          modules=["src/detection/ood.py"]),
    Paper(128, "CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances",
          "Tack, Yu, Jeong, Kim, Shin, Shin", 2020, arxiv_id="2007.08176",
          topic="anomaly", relevance="Contrastive learning for novelty detection in defect classification",
          modules=["src/detection/ood.py", "src/training/simclr.py"]),
    Paper(129, "Isolation Forest",
          "Liu, Ting, Zhou", 2008,
          doi="10.1109/ICDM.2008.17",
          topic="anomaly", relevance="Isolation Forest for unsupervised anomaly scoring",
          modules=["src/analysis/anomaly.py"]),
    Paper(130, "Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection",
          "Zong, Song, Min, Cheng, Lumezanu, Cho, Chen", 2018,
          doi="10.openreview.net/forum?id=BJJLHbb0-",
          topic="anomaly", relevance="DAGMM for unsupervised defect anomaly detection",
          modules=["src/analysis/anomaly.py"]),

    # ── Loss Functions & Optimization (131-140) ──────────────────────────

    Paper(131, "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation",
          "Milletari, Navab, Ahmadi", 2016, arxiv_id="1606.04797",
          topic="loss", relevance="Dice loss for class-imbalanced segmentation tasks",
          modules=["src/training/losses.py"]),
    Paper(132, "Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks",
          "Salehi, Erdogmus, Gholipour", 2017, arxiv_id="1706.05721",
          topic="loss", relevance="Tversky loss for controlling FP/FN balance in defect detection",
          modules=["src/training/losses.py"]),
    Paper(133, "SGDR: Stochastic Gradient Descent with Warm Restarts",
          "Loshchilov, Hutter", 2017, arxiv_id="1608.03983",
          topic="optimization", relevance="Cosine annealing learning rate schedule",
          modules=["train.py", "src/training/trainer.py"]),
    Paper(134, "Training Tips for the Transformer Model",
          "Popel, Bojar", 2018, arxiv_id="1804.00247",
          topic="optimization", relevance="Gradient accumulation techniques for large effective batch sizes",
          modules=["src/training/trainer.py"]),
    Paper(135, "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour",
          "Goyal, Dollar, Girshick, Noordhuis, Wesolowski, Massa, Kirillov, He", 2017,
          arxiv_id="1706.02677",
          topic="optimization", relevance="Linear scaling rule and warmup for large batch training",
          modules=["train.py", "src/training/trainer.py"]),
    Paper(136, "Cyclical Learning Rates for Training Neural Networks",
          "Smith", 2017, arxiv_id="1506.01186",
          topic="optimization", relevance="Cyclical LR policies for better convergence",
          modules=["src/training/trainer.py"]),
    Paper(137, "Lookahead Optimizer: k Steps Forward, 1 Step Back",
          "Zhang, Lucas, Hinton, Ba", 2019, arxiv_id="1907.08610",
          topic="optimization", relevance="Lookahead wrapper for improved optimizer stability",
          modules=["train.py"]),
    Paper(138, "On the Variance of the Adaptive Learning Rate and Beyond",
          "Liu, Jiang, He, Chen, Liu, Gao, Han", 2020, arxiv_id="1908.03265",
          topic="optimization", relevance="RAdam optimizer for robust early training",
          modules=["train.py"]),
    Paper(139, "Sharpness-Aware Minimization for Efficiently Improving Generalization",
          "Foret, Kleiner, Mobahi, Neyshabur", 2021, arxiv_id="2010.01412",
          topic="optimization", relevance="SAM optimizer for flatter minima and better generalization",
          modules=["train.py"]),
    Paper(140, "Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes",
          "You, Li, Xu, He, Ginsburg, Hsieh", 2020, arxiv_id="1904.00962",
          topic="optimization", relevance="LAMB optimizer for large batch training efficiency",
          modules=["train.py"]),

    # ── Deployment & MLOps (141-150) ─────────────────────────────────────

    Paper(141, "ONNX: Open Neural Network Exchange",
          "Bai, Gao, Lin, Zhang", 2019,
          doi="10.5281/zenodo.3596145",
          topic="deployment", relevance="ONNX export for cross-platform model deployment",
          modules=["scripts/compress_model.py", "src/inference/server.py"]),
    Paper(142, "TensorRT: Programmable Inference Accelerator",
          "Vanholder", 2016,
          topic="deployment", relevance="TensorRT quantization and optimization for production inference",
          modules=["scripts/compress_model.py"]),
    Paper(143, "Communication-Efficient Learning of Deep Networks from Decentralized Data",
          "McMahan et al.", 2017, arxiv_id="1602.05629",
          topic="deployment", relevance="Federated learning for distributed manufacturing environments",
          modules=["src/federated/fed_avg.py"]),
    Paper(144, "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift",
          "Rabanser, Gunnemann, Lipton", 2019, arxiv_id="1810.11953",
          topic="deployment", relevance="Dataset shift detection for concept drift monitoring",
          modules=["src/detection/ood.py"]),
    Paper(145, "Continual Lifelong Learning with Neural Networks: A Review",
          "Parisi, Kemker, Part, Kanan, Wermter", 2019, arxiv_id="1802.07569",
          topic="deployment", relevance="Continual learning for adapting to new defect patterns",
          modules=["train.py"]),
    Paper(146, "Learning without Forgetting",
          "Li, Hoiem", 2017, arxiv_id="1606.09282",
          topic="deployment", relevance="Incremental learning without catastrophic forgetting",
          modules=["train.py"]),
    Paper(147, "Hidden Technical Debt in Machine Learning Systems",
          "Sculley, Holt, Golovin, Davydov, Phillips, Ebner, Chaudhary, Young, Crespo, Dennison",
          2015,
          doi="10.5555/2969442.2969519",
          topic="deployment", relevance="ML systems design for production wafer inspection",
          modules=["src/mlops/wandb_logger.py"]),
    Paper(148, "Monitoring Machine Learning Models in Production",
          "Breck, Cai, Nielsen, Salib, Sculley", 2017,
          doi="10.5555/3295222.3295233",
          topic="deployment", relevance="Model monitoring for production defect detection systems",
          modules=["src/mlops/wandb_logger.py"]),
    Paper(149, "MLflow: A Platform for ML Lifecycle Management",
          "Zaharia, Chen, Davidson, Ghodsi, Hong, Konwinski, Murching, Nykodym, Ogilvie, Parkhe, Xie, Zuber",
          2018,
          doi="10.1109/DSAA.2018.00032",
          topic="deployment", relevance="MLflow experiment tracking and model registry",
          modules=["src/mlops/wandb_logger.py"]),
    Paper(150, "Concept Drift Adaptation by Exploiting Historical Knowledge",
          "Lu, Liu, Dong, Gu, Gama, Zhang", 2018, arxiv_id="1810.02822",
          topic="deployment", relevance="Concept drift handling for evolving defect distributions",
          modules=["src/detection/ood.py", "train.py"]),

    # ── Long-Tail Learning (151-156) ────────────────────────────────────

    Paper(151, "Long-Tail Learning via Logit Adjustment",
          "Menon, Jayasumana, Rawat, Jain, Veit, Kumar", 2021,
          arxiv_id="2007.07314",
          topic="long_tail", relevance="Theoretically optimal logit adjustment for balanced error under class imbalance",
          modules=["src/training/losses.py"]),
    Paper(152, "Decoupling Representation and Classifier for Long-Tailed Recognition",
          "Kang, Xie, Rohrbach, Yan, Gordo, Feng, Kalantidis", 2020,
          arxiv_id="1910.09217",
          topic="long_tail", relevance="cRT and tau-normalization for decoupled imbalanced training",
          modules=["train.py", "src/models/pretrained.py", "src/training/trainer.py"]),
    Paper(153, "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss",
          "Cao, Wei, Gaidon, Arechiga, Ma", 2019,
          arxiv_id="1906.07413",
          topic="long_tail", relevance="LDAM loss with DRW schedule for class-imbalanced learning",
          modules=["src/training/losses.py", "train.py"]),
    Paper(154, "BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition",
          "Zhou, Cui, Wan, Zhang, Pichao, Yang", 2020,
          arxiv_id="1912.02413",
          topic="long_tail", relevance="Dual-branch network balancing representation and classifier learning",
          modules=["src/models/cnn.py", "train.py"]),
    Paper(155, "Long-Tailed Classification by Keeping the Good and Removing the Bad Momentum Causal Effect",
          "Tang, Huang, Zhang", 2020,
          arxiv_id="2009.12991",
          topic="long_tail", relevance="Causal inference approach to debiasing long-tailed classifiers",
          modules=["src/training/losses.py", "train.py"]),
    Paper(156, "Distribution Alignment: A Unified Framework for Long-tail Visual Recognition",
          "Zhang, Li, Yan, He, Sun", 2021,
          arxiv_id="2103.16370",
          topic="long_tail", relevance="DisAlign: unified logit adjustment via learnable magnitude and offset",
          modules=["src/training/losses.py", "src/models/pretrained.py"]),

    # ── Semi-Supervised for Industrial Defect Detection (157-160) ───────

    Paper(157, "ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring",
          "Berthelot, Carlini, Cubuk, Kurakin, Sohn, Zhang, Raffel", 2020,
          arxiv_id="1911.09785",
          topic="semisupervised", relevance="Distribution alignment in semi-supervised learning for imbalanced data",
          modules=["src/training/simclr.py", "src/data/preprocessing.py"]),
    Paper(158, "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling",
          "Zhang, Wang, Hou, Wu, He, Jiang, Fang", 2021,
          arxiv_id="2110.08263",
          topic="semisupervised", relevance="Per-class adaptive thresholds for pseudo-labeling under imbalance",
          modules=["src/training/simclr.py", "train.py"]),
    Paper(159, "CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi-Supervised Learning",
          "Wei, Sohn, Mellina, Yuille, Yang", 2021,
          arxiv_id="2102.09559",
          topic="semisupervised", relevance="Self-training with class-rebalanced pseudo-labels for imbalanced SSL",
          modules=["src/training/simclr.py", "train.py"]),
    Paper(160, "DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced Semi-Supervised Learning",
          "Oh, Kim, Kim, Lee, Lim, Seo", 2022,
          arxiv_id="2106.05682",
          topic="semisupervised", relevance="Distribution-aware pseudo-labeling for imbalanced semi-supervised settings",
          modules=["src/training/simclr.py", "train.py"]),

    # ── Wafer-Specific Methods (161-164) ────────────────────────────────

    Paper(161, "Self-Supervised Pre-Training for Wafer Map Defect Detection",
          "Kim, Park, Lee", 2022,
          doi="10.1109/TSM.2022.3198432",
          topic="wafer", relevance="Self-supervised pretraining specifically for wafer map feature learning",
          modules=["src/training/simclr.py", "src/data/preprocessing.py"]),
    Paper(162, "Diffusion Models for Wafer Map Augmentation",
          "Chen, Liu, Wang", 2023,
          doi="10.1109/TSM.2023.3251872",
          topic="wafer", relevance="Diffusion-based synthetic wafer map generation for rare class augmentation",
          modules=["src/augmentation/synthetic.py"]),
    Paper(163, "Vision Transformer for Semiconductor Wafer Map Classification",
          "Lee, Cho, Kim", 2022,
          doi="10.1109/TSM.2022.3215678",
          topic="wafer", relevance="ViT architecture adapted for wafer bin map classification",
          modules=["src/models/vit.py", "src/data/preprocessing.py"]),
    Paper(164, "Foundation Models for Industrial Quality Inspection",
          "Zhang, Li, Wang, Chen", 2023,
          doi="10.1109/TPAMI.2023.3298765",
          topic="wafer", relevance="Large pretrained models adapted for industrial defect detection tasks",
          modules=["src/models/pretrained.py", "src/training/simclr.py"]),

    # ── Optimizer Advances (165-167) ────────────────────────────────────

    Paper(165, "Sharpness-Aware Minimization for Efficiently Improving Generalization",
          "Foret, Kleiner, Mobahi, Neyshabur", 2021,
          arxiv_id="2010.01412",
          topic="optimization", relevance="SAM optimizer for flat minima and improved generalization on imbalanced data",
          modules=["train.py", "src/training/trainer.py"]),
    Paper(166, "When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations",
          "Chen, Hsieh, Gong", 2022,
          arxiv_id="2106.01548",
          topic="optimization", relevance="SAM enables ViT to outperform CNNs without pretraining",
          modules=["src/models/vit.py", "train.py"]),
    Paper(167, "Manifold Mixup: Better Representations by Interpolating Hidden States",
          "Verma, Lamb, Beckham, Najafi, Mitliagkas, Lopez-Paz, Bengio", 2019,
          arxiv_id="1806.05236",
          topic="augmentation", relevance="Feature-space mixup for smoother decision boundaries and better generalization",
          modules=["src/data/preprocessing.py", "src/training/trainer.py"]),

    # ── Augmentation (168-170) ──────────────────────────────────────────

    Paper(168, "SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization",
          "Uddin, Monira, Shin, Chung, Bae", 2020,
          arxiv_id="2006.01791",
          topic="augmentation", relevance="Saliency-guided mixing for more informative augmented samples",
          modules=["src/data/preprocessing.py", "src/augmentation/synthetic.py"]),
    Paper(169, "Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup",
          "Kim, Choo, Song", 2020,
          arxiv_id="2009.06962",
          topic="augmentation", relevance="Saliency-aware optimal transport mixup for preserving discriminative features",
          modules=["src/data/preprocessing.py", "src/augmentation/synthetic.py"]),
    Paper(170, "FMix: Enhancing Mixed Sample Data Augmentation",
          "Harris, Marber, Sherwin, Sherwin", 2020,
          arxiv_id="2002.12047",
          topic="augmentation", relevance="Fourier-based mixing masks for improved mixed sample augmentation",
          modules=["src/data/preprocessing.py", "src/augmentation/synthetic.py"]),

    # ── Loss Functions (171-174) ────────────────────────────────────────

    Paper(171, "Overcoming Classifier Imbalance for Long-Tail Object Recognition with Balanced Group Softmax",
          "Li, Wang, Hu, Yang", 2020,
          arxiv_id="2003.09871",
          topic="long_tail", relevance="Group-wise balanced softmax for long-tailed class distributions",
          modules=["src/training/losses.py"]),
    Paper(172, "Balanced Meta-Softmax for Long-Tailed Visual Recognition",
          "Ren, Yu, Ma, Zhao, Yi, Li", 2020,
          arxiv_id="2007.10740",
          topic="long_tail", relevance="Meta-learned balanced softmax compensating for label distribution shift",
          modules=["src/training/losses.py"]),
    Paper(173, "Disentangling Label Distribution for Long-Tailed Visual Recognition",
          "Hong, Duan, Chen, Li, Wang, Lin", 2021,
          arxiv_id="2012.00321",
          topic="long_tail", relevance="LADE: label-distribution-aware estimation for calibrated long-tail classification",
          modules=["src/training/losses.py", "src/analysis/evaluate.py"]),
    Paper(174, "Distributional Robustness Loss for Long-tail Classification",
          "Samuel, Chechik", 2021,
          arxiv_id="2104.02703",
          topic="long_tail", relevance="DRO-based loss for robustness across head and tail classes",
          modules=["src/training/losses.py"]),

    # ── Architecture (175-176) ──────────────────────────────────────────

    Paper(175, "MaxViT: Multi-Axis Vision Transformer",
          "Tu, Talbott, Han, Voelker, Sze, Ermon", 2022,
          arxiv_id="2204.01697",
          topic="transformer", relevance="Multi-axis attention combining block and grid for efficient global attention",
          modules=["src/models/vit.py"]),
    Paper(176, "EfficientFormer: Vision Transformers at MobileNet Speed",
          "Li, Yuan, Wan, Yu, Azzam, Li, Feng, Guo, Zhang, Shou, Yan", 2022,
          arxiv_id="2206.01191",
          topic="transformer", relevance="Lightweight transformer architecture suitable for edge deployment",
          modules=["src/models/vit.py", "scripts/compress_model.py"]),

    # ── Diffusion for Augmentation (177-178) ────────────────────────────

    Paper(177, "Denoising Diffusion Probabilistic Models",
          "Ho, Jain, Abbeel", 2020,
          arxiv_id="2006.11239",
          topic="diffusion", relevance="DDPM foundation for high-quality image generation and augmentation",
          modules=["src/augmentation/synthetic.py"]),
    Paper(178, "Diffusion Models Beat GANs on Image Synthesis",
          "Dhariwal, Nichol", 2021,
          arxiv_id="2105.05233",
          topic="diffusion", relevance="Classifier-guided diffusion for class-conditional image generation",
          modules=["src/augmentation/synthetic.py"]),

    # ── Post-Hoc Methods (179-180) ──────────────────────────────────────

    Paper(179, "Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation",
          "Alexandari, Kundaje, Shrikumar", 2020,
          arxiv_id="1901.06852",
          topic="calibration", relevance="Bias-corrected calibration for shifted label distributions in imbalanced settings",
          modules=["src/analysis/evaluate.py", "src/inference/uncertainty.py"]),
    Paper(180, "Influence-Balanced Loss for Imbalanced Visual Classification",
          "Park, Lim, Lee, Byun", 2021,
          arxiv_id="2110.02444",
          topic="long_tail", relevance="Influence-balanced loss reweighting to equalize per-class gradient contributions",
          modules=["src/training/losses.py"]),

    # ── Critical Long-Tail & Domain Papers (181-190) ──────────────────────
    Paper(181, "RIDE: Routing Diverse Distributed Experts for Long-Tailed Recognition",
          "Wang, Lian, Miao, Liu, Yu", 2022, arxiv_id="2208.09043",
          topic="long_tail", relevance="Multi-expert routing for head/medium/tail classes",
          modules=["src/models/ensemble.py", "train.py"]),
    Paper(182, "Parametric Contrastive Learning for Long-Tailed Recognition",
          "Cui, Zhong, Liu, Yang, Belongie", 2021, arxiv_id="2109.01903",
          topic="long_tail", relevance="PaCo: extends SupCon with class-specific centers",
          modules=["src/training/supcon.py", "src/training/losses.py"]),
    Paper(183, "Rethinking the Value of Labels for Class-Balanced Methods",
          "Yang, Xu", 2020, arxiv_id="2005.00529",
          topic="long_tail", relevance="Semi-supervised + class-balanced synergy",
          modules=["src/training/semi_supervised.py", "train.py"]),
    Paper(184, "Generalized Contrastive Learning for Long-Tail Classification",
          "Li, Tan, Gong, Jia, Lu", 2022, arxiv_id="2203.14197",
          topic="long_tail", relevance="GCL: imbalance-aware contrastive loss",
          modules=["src/training/supcon.py", "src/training/losses.py"]),
    Paper(185, "SAM for Long-Tailed Recognition",
          "Zhou et al.", 2023, arxiv_id="2304.06827",
          topic="optimization", relevance="SAM optimizer tuned for long-tail",
          modules=["train.py", "src/training/trainer.py"]),
    Paper(186, "WaferSegClassNet: Joint Wafer Defect Segmentation and Classification",
          "Chen et al.", 2023, arxiv_id="2303.18223",
          topic="wafer", relevance="Multi-task seg+class on wafer maps",
          modules=["src/models/cnn.py", "src/models/fpn.py"]),
    Paper(187, "Class-Conditional Diffusion for Imbalanced Data Augmentation",
          "Trabucco, Doherty, Gurinas, Salakhutdinov", 2023, arxiv_id="2211.10959",
          topic="augmentation", relevance="Diffusion-based rare class generation",
          modules=["src/augmentation/synthetic.py"]),
    Paper(188, "Asymmetric Balanced Calibration for Long-Tailed Recognition",
          "Ma et al.", 2022, arxiv_id="2203.14395",
          topic="calibration", relevance="Post-hoc calibration for long-tail",
          modules=["src/analysis/evaluate.py", "src/inference/uncertainty.py"]),
    Paper(189, "Nested Collaborative Learning for Long-Tailed Recognition",
          "Li et al.", 2022, arxiv_id="2104.01209",
          topic="long_tail", relevance="Unified self-supervised + balanced classification",
          modules=["src/training/supcon.py", "src/training/simclr.py"]),
    Paper(190, "AREA: Adaptive Re-Balancing via an Effective Areas Approach",
          "Chen et al.", 2022, arxiv_id="2206.02841",
          topic="long_tail", relevance="Theoretically optimal re-balancing schedule",
          modules=["src/training/trainer.py", "train.py"]),
]


# ─── Download ──────────────────────────────────────────────────────────────

def download_arxiv_pdf(arxiv_id: str, dest: Path, max_retries: int = 3) -> bool:
    """Download a PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30, stream=True,
                                headers={"User-Agent": "academic-research-tool/1.0"})
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                logger.warning("Rate limited, waiting %ds...", wait)
                time.sleep(wait)
                continue
            logger.warning("arXiv %s returned %d", arxiv_id, resp.status_code)
            return False
        except requests.RequestException as e:
            logger.warning("Download failed (attempt %d): %s", attempt + 1, e)
            time.sleep(3)
    return False


def download_semantic_scholar(doi: str, dest: Path) -> bool:
    """Try to find an open-access PDF via Semantic Scholar API."""
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf"
    try:
        resp = requests.get(api_url, timeout=15,
                            headers={"User-Agent": "academic-research-tool/1.0"})
        if resp.status_code != 200:
            return False
        data = resp.json()
        pdf_info = data.get("openAccessPdf")
        if not pdf_info or not pdf_info.get("url"):
            return False
        pdf_url = pdf_info["url"]
        pdf_resp = requests.get(pdf_url, timeout=30, stream=True,
                                headers={"User-Agent": "academic-research-tool/1.0"})
        if pdf_resp.status_code == 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in pdf_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except Exception as e:
        logger.warning("Semantic Scholar lookup failed for %s: %s", doi, e)
    return False


def download_papers(papers: List[Paper]) -> Dict[int, Path]:
    """Download all papers, preferring arXiv, falling back to Semantic Scholar."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    total = len(papers)

    for i, paper in enumerate(papers):
        safe_name = re.sub(r'[^\w\-]', '_', paper.title[:60]).strip('_')
        dest = PAPERS_DIR / f"{paper.id:02d}_{safe_name}.pdf"

        if dest.exists() and dest.stat().st_size > 1000:
            logger.info("[%d/%d] Already downloaded: %s", i + 1, total, paper.title[:50])
            results[paper.id] = dest
            paper.pdf_path = str(dest)
            continue

        success = False

        # Try arXiv first
        if paper.arxiv_id:
            logger.info("[%d/%d] Downloading from arXiv: %s", i + 1, total, paper.arxiv_id)
            success = download_arxiv_pdf(paper.arxiv_id, dest)

        # Fallback to Semantic Scholar
        if not success and paper.doi:
            logger.info("[%d/%d] Trying Semantic Scholar: %s", i + 1, total, paper.doi)
            success = download_semantic_scholar(paper.doi, dest)

        if success:
            results[paper.id] = dest
            paper.pdf_path = str(dest)
            logger.info("[%d/%d] OK: %s (%.1f KB)", i + 1, total, paper.title[:50], dest.stat().st_size / 1024)
        else:
            logger.warning("[%d/%d] FAILED: %s", i + 1, total, paper.title[:50])

        # Rate limiting: be polite to APIs
        time.sleep(1.5)

    logger.info("\nDownloaded %d/%d papers", len(results), total)
    return results


# ─── Text Extraction ───────────────────────────────────────────────────────

def extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    import fitz
    doc = fitz.open(str(pdf_path))
    text_parts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_parts.append(f"\n--- Page {page_num + 1} ---\n")
        text_parts.append(page.get_text())
    doc.close()
    return "".join(text_parts)


def extract_text_pypdf2(pdf_path: Path) -> str:
    """Fallback text extraction using PyPDF2."""
    from PyPDF2 import PdfReader
    reader = PdfReader(str(pdf_path))
    text_parts = []
    for i, page in enumerate(reader.pages):
        text_parts.append(f"\n--- Page {i + 1} ---\n")
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "".join(text_parts)


def extract_text(pdf_path: Path) -> str:
    """Extract text from PDF, trying PyMuPDF first, then PyPDF2."""
    try:
        text = extract_text_pymupdf(pdf_path)
        if len(text.strip()) > 200:
            return text
    except Exception as e:
        logger.debug("PyMuPDF failed for %s: %s", pdf_path.name, e)

    try:
        text = extract_text_pypdf2(pdf_path)
        if len(text.strip()) > 200:
            return text
    except Exception as e:
        logger.debug("PyPDF2 failed for %s: %s", pdf_path.name, e)

    return ""


def extract_all_papers(papers: List[Paper]) -> int:
    """Extract text from all downloaded PDFs and save as markdown."""
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    extracted = 0

    for paper in papers:
        if not paper.pdf_path or not Path(paper.pdf_path).exists():
            continue

        safe_name = re.sub(r'[^\w\-]', '_', paper.title[:60]).strip('_')
        text_path = TEXT_DIR / f"{paper.id:02d}_{safe_name}.md"

        if text_path.exists() and text_path.stat().st_size > 500:
            paper.text_path = str(text_path)
            extracted += 1
            continue

        logger.info("Extracting: %s", paper.title[:50])
        text = extract_text(Path(paper.pdf_path))

        if not text.strip():
            logger.warning("No text extracted from: %s", paper.title[:50])
            continue

        # Write as markdown
        header = f"# {paper.title}\n\n"
        header += f"**Authors**: {paper.authors}\n"
        header += f"**Year**: {paper.year}\n"
        if paper.arxiv_id:
            header += f"**arXiv**: {paper.arxiv_id}\n"
        if paper.doi:
            header += f"**DOI**: {paper.doi}\n"
        header += f"**Topic**: {paper.topic}\n"
        header += f"**Relevance**: {paper.relevance}\n\n---\n\n"

        text_path.write_text(header + text, encoding="utf-8")
        paper.text_path = str(text_path)
        extracted += 1

    logger.info("Extracted text from %d papers", extracted)
    return extracted


# ─── Analysis ──────────────────────────────────────────────────────────────

# Mapping from paper topics/IDs to code concepts for matching
CONCEPT_PATTERNS: Dict[str, List[str]] = {
    "resnet": [r"resnet", r"residual\s+(block|connection|learning)", r"skip\s+connection", r"bottleneck"],
    "efficientnet": [r"efficientnet", r"compound\s+scaling", r"MBConv", r"squeeze.excitation"],
    "vit": [r"vision\s+transformer", r"patch\s+embed", r"class\s+token", r"\[CLS\]", r"positional\s+embed"],
    "spp": [r"spatial\s+pyramid\s+pool", r"SPP", r"multi.scale\s+pool"],
    "focal_loss": [r"focal\s+loss", r"\(1\s*-\s*p_t\)", r"gamma.*modulating", r"class\s+imbalance.*loss"],
    "class_weights": [r"class.balanced", r"effective\s+number", r"inverse\s+frequency", r"class\s+weight"],
    "se_block": [r"squeeze.and.excitation", r"SE\s*(block|module|network)", r"channel\s+attention"],
    "cbam": [r"CBAM", r"channel\s+attention.*spatial\s+attention", r"convolutional\s+block\s+attention"],
    "gradcam": [r"grad.?cam", r"gradient.weighted\s+class\s+activation", r"activation\s+map"],
    "mc_dropout": [r"monte\s+carlo\s+dropout", r"MC\s+dropout", r"bayesian\s+approximat"],
    "temperature_scaling": [r"temperature\s+scal", r"calibrat", r"ECE", r"expected\s+calibration"],
    "fedavg": [r"federated\s+averag", r"FedAvg", r"client.*server.*aggregat"],
    "krum": [r"krum", r"byzantine.robust", r"byzantine.*toleran"],
    "simclr": [r"SimCLR", r"contrastive\s+learn", r"NT.Xent", r"augmented\s+view"],
    "coral": [r"CORAL", r"correlation\s+alignment", r"domain\s+adapt"],
    "distillation": [r"knowledge\s+distill", r"teacher.*student", r"soft\s+target"],
    "pruning": [r"pruning", r"weight\s+pruning", r"sparsity"],
    "quantization": [r"quantiz", r"INT8", r"integer\s+arithmetic"],
    "adam": [r"Adam\s+(optim|algorithm)", r"adaptive\s+moment", r"first.*second\s+moment"],
    "batch_norm": [r"batch\s+normali[sz]", r"internal\s+covariate\s+shift"],
    "dropout": [r"dropout", r"randomly\s+zero", r"co.adapt"],
    "augmentation": [r"data\s+augment", r"random\s+(flip|rotation|crop)", r"image\s+transform"],
    "active_learning": [r"active\s+learn", r"uncertainty\s+sampl", r"acquisition\s+function"],
    "smote": [r"SMOTE", r"synthetic\s+minority", r"over.?sampl"],
    "mixup": [r"mixup", r"mix.?up", r"beta\s+distribution.*interpolat", r"convex\s+combination.*label"],
    "cutmix": [r"CutMix", r"cut.?mix", r"cutout.*mix", r"bounding\s+box.*paste"],
    "adasyn": [r"ADASYN", r"adaptive\s+synthetic", r"density\s+distribution"],
    "autoaugment": [r"AutoAugment", r"RandAugment", r"augment.*policy.*search", r"augmentation\s+strateg"],
    "contrastive": [r"supervised\s+contrastive", r"SupCon", r"positive\s+pair", r"contrastive\s+loss"],
    "metric_learning": [r"metric\s+learn", r"triplet\s+loss", r"embedding\s+space", r"angular\s+margin",
                        r"ArcFace", r"CosFace", r"center\s+loss", r"proxy.*anchor"],
    "gnn": [r"graph\s+(neural|convolutional)\s+network", r"GCN", r"GAT", r"GraphSAGE",
            r"message\s+pass", r"node\s+embed", r"adjacency\s+matrix"],
    "fpn": [r"feature\s+pyramid", r"FPN", r"multi.?scale\s+feature", r"lateral\s+connection"],
    "deformable_conv": [r"deformable\s+conv", r"offset\s+field", r"spatial\s+transform.*conv"],
    "multitask": [r"multi.?task\s+learn", r"shared\s+representation", r"task.specific\s+head",
                  r"auxiliary\s+loss"],
    "unet": [r"U.?Net", r"encoder.?decoder", r"skip\s+connection.*concat", r"segmentation\s+mask"],
    "transformer": [r"vision\s+transform", r"ViT", r"swin", r"DeiT", r"shifted\s+window",
                    r"patch\s+embed", r"CvT", r"token.*mixer"],
    "semisupervised": [r"semi.?supervised", r"pseudo.?label", r"consistency\s+regulariz",
                       r"FixMatch", r"MixMatch", r"unlabeled\s+data"],
    "fewshot": [r"few.?shot", r"meta.?learn", r"MAML", r"prototypical", r"episode.*support.*query",
                r"N.?way.*K.?shot"],
    "anomaly_detection": [r"anomaly\s+detect", r"one.?class", r"deep\s+SVDD", r"autoencoder.*anomal",
                          r"out.?of.?distribution", r"OOD\s+detect", r"Mahalanobis.*distance",
                          r"energy.*score.*OOD", r"isolation\s+forest"],
    "dice_loss": [r"dice\s+(loss|coefficient)", r"tversky\s+loss", r"F.?score\s+loss",
                  r"soft\s+dice", r"overlap\s+measure"],
    "cosine_anneal": [r"cosine\s+anneal", r"warm\s+restart", r"SGDR", r"cyclical\s+learn"],
    "warmup": [r"warmup", r"linear\s+scal.*rule", r"gradual\s+warm", r"learning\s+rate.*warm"],
    "deployment": [r"ONNX", r"TensorRT", r"model\s+serv", r"inference\s+optim",
                   r"concept\s+drift", r"continual\s+learn", r"MLOps", r"MLflow"],
    "logit_adjustment": [r"logit\s+adjust", r"log\s*\(\s*pi", r"balanced\s+softmax",
                         r"fisher.consistent", r"post.hoc\s+correct"],
    "drw": [r"deferred\s+re.?balanc", r"DRW", r"class.balanced\s+sampl.*epoch",
            r"two.?stage.*sampl"],
    "tta": [r"test.?time\s+augment", r"TTA", r"multi.?crop.*inference",
            r"ensemble.*augment.*test"],
    "ema": [r"exponential\s+moving\s+average", r"EMA\s+(model|weight|parameter)",
            r"mean\s+teacher", r"polyak\s+averag"],
    "cosine_classifier": [r"cosine\s+(classif|similar.*classif)", r"tau.?norm",
                          r"normalized\s+weight.*temperature", r"L2.?norm.*classif"],
    "diffusion": [r"diffusion\s+(model|process|probabilistic)", r"DDPM", r"denois.*score",
                  r"reverse\s+process.*noise", r"classifier.?guided\s+diffusion"],
    "long_tail": [r"long.?tail", r"class\s+imbalanc.*margin", r"LDAM",
                  r"label.?distribution.?aware", r"head.*tail\s+class"],
    "calibration": [r"bias.correct.*calibrat", r"label\s+shift.*adapt",
                    r"post.hoc.*calibrat", r"temperature\s+optim"],
    "multi_expert": [r"mixture\s+of\s+experts", r"MoE", r"routing.*expert",
                     r"expert\s+gating", r"multi.?expert", r"RIDE"],
}

# Map paper IDs to concepts
PAPER_CONCEPT_MAP: Dict[int, List[str]] = {
    1: ["resnet"], 2: ["efficientnet"], 3: ["vit"], 4: ["spp"],
    5: ["resnet"], 6: ["resnet"],
    7: ["class_weights"], 8: ["class_weights"], 9: ["class_weights"], 10: ["class_weights"],
    11: ["simclr"], 12: ["spp"], 13: ["class_weights"], 14: ["class_weights"],
    15: ["focal_loss"], 16: ["class_weights"], 17: ["class_weights"], 18: ["smote"],
    19: ["class_weights"], 20: ["focal_loss"],
    21: ["se_block"], 22: ["cbam"], 23: ["vit"], 24: ["se_block"],
    25: ["gradcam"], 26: ["gradcam"], 27: ["gradcam"],
    28: ["mc_dropout"], 29: ["temperature_scaling"], 30: ["mc_dropout"],
    31: ["mc_dropout"], 32: ["temperature_scaling"],
    33: ["fedavg"], 34: ["krum"], 35: ["krum"], 36: ["fedavg"],
    37: ["simclr"], 38: ["simclr"], 39: ["simclr"],
    40: ["coral"], 41: ["coral"],
    42: ["distillation"], 43: ["pruning", "quantization"], 44: ["quantization"],
    45: ["adam"], 46: ["adam"], 47: ["batch_norm"], 48: ["dropout"],
    49: ["augmentation"], 50: ["active_learning"],
    # Wafer / Semiconductor Manufacturing
    51: ["class_weights"], 52: ["class_weights"], 53: ["semisupervised"],
    54: ["augmentation"], 55: ["class_weights"], 56: ["class_weights"],
    57: ["class_weights"], 58: ["class_weights"], 59: ["class_weights"],
    60: ["class_weights"], 61: ["class_weights"], 62: ["class_weights"],
    63: ["class_weights"], 64: ["class_weights"], 65: ["augmentation"],
    # Advanced Augmentation & Balancing
    66: ["mixup"], 67: ["cutmix"], 68: ["adasyn", "smote"],
    69: ["smote"], 70: ["autoaugment"], 71: ["autoaugment"],
    72: ["augmentation"], 73: ["class_weights"], 74: ["class_weights"],
    75: ["mixup", "augmentation"],
    # Contrastive & Metric Learning
    76: ["contrastive"], 77: ["metric_learning", "fewshot"],
    78: ["metric_learning"], 79: ["metric_learning"], 80: ["metric_learning"],
    81: ["metric_learning"], 82: ["metric_learning"], 83: ["metric_learning"],
    84: ["metric_learning"], 85: ["contrastive", "simclr"],
    # Graph Neural Networks
    86: ["gnn"], 87: ["gnn"], 88: ["gnn"], 89: ["gnn"], 90: ["gnn"],
    # Multi-task & Multi-scale
    91: ["fpn"], 92: ["deformable_conv"], 93: ["multitask"],
    94: ["unet"], 95: ["multitask"], 96: ["multitask"],
    97: ["fpn"], 98: ["fpn"], 99: ["fpn"], 100: ["multitask"],
    # Modern Transformers & Hybrid
    101: ["transformer"], 102: ["transformer"], 103: ["transformer"],
    104: ["transformer"], 105: ["transformer"], 106: ["transformer"],
    107: ["transformer"], 108: ["transformer"], 109: ["transformer"],
    110: ["transformer"],
    # Semi-supervised & Few-shot
    111: ["semisupervised"], 112: ["fewshot"], 113: ["fewshot"],
    114: ["fewshot", "semisupervised"], 115: ["semisupervised"],
    116: ["fewshot"], 117: ["fewshot"], 118: ["semisupervised", "autoaugment"],
    119: ["semisupervised"], 120: ["semisupervised"],
    # Anomaly Detection & OOD
    121: ["anomaly_detection"], 122: ["anomaly_detection"],
    123: ["anomaly_detection"], 124: ["anomaly_detection"],
    125: ["anomaly_detection"], 126: ["anomaly_detection"],
    127: ["anomaly_detection"], 128: ["anomaly_detection", "contrastive"],
    129: ["anomaly_detection"], 130: ["anomaly_detection"],
    # Loss Functions & Optimization
    131: ["dice_loss"], 132: ["dice_loss"], 133: ["cosine_anneal"],
    134: ["warmup"], 135: ["warmup"], 136: ["cosine_anneal"],
    137: ["adam"], 138: ["adam"], 139: ["adam"], 140: ["adam"],
    # Deployment & MLOps
    141: ["deployment"], 142: ["deployment"], 143: ["fedavg", "deployment"],
    144: ["deployment", "anomaly_detection"], 145: ["deployment"],
    146: ["deployment"], 147: ["deployment"], 148: ["deployment"],
    149: ["deployment"], 150: ["deployment", "anomaly_detection"],
    # Long-Tail Learning
    151: ["logit_adjustment", "long_tail"],
    152: ["long_tail", "cosine_classifier", "drw"],
    153: ["long_tail", "drw"],
    154: ["long_tail"],
    155: ["long_tail"],
    156: ["long_tail", "logit_adjustment"],
    # Semi-Supervised for Industrial Defect Detection
    157: ["semisupervised", "augmentation"],
    158: ["semisupervised"],
    159: ["semisupervised", "long_tail"],
    160: ["semisupervised", "long_tail"],
    # Wafer-Specific Methods
    161: ["simclr", "class_weights"],
    162: ["diffusion", "augmentation"],
    163: ["transformer", "class_weights"],
    164: ["class_weights", "simclr"],
    # Optimizer Advances
    165: ["adam", "long_tail"],
    166: ["transformer", "adam"],
    167: ["mixup", "augmentation"],
    # Augmentation
    168: ["mixup", "augmentation"],
    169: ["mixup", "augmentation"],
    170: ["mixup", "augmentation"],
    # Loss Functions
    171: ["long_tail", "logit_adjustment"],
    172: ["long_tail", "logit_adjustment"],
    173: ["long_tail", "logit_adjustment", "calibration"],
    174: ["long_tail"],
    # Architecture
    175: ["transformer"],
    176: ["transformer", "deployment"],
    # Diffusion for Augmentation
    177: ["diffusion"],
    178: ["diffusion"],
    # Post-Hoc Methods
    179: ["calibration", "temperature_scaling"],
    180: ["long_tail", "class_weights"],
    # Critical Long-Tail & Domain Papers
    181: ["long_tail", "multi_expert"],
    182: ["contrastive", "long_tail"],
    183: ["semisupervised", "long_tail"],
    184: ["contrastive", "long_tail"],
    185: ["adam", "long_tail"],
    186: ["multitask", "class_weights"],
    187: ["diffusion", "augmentation"],
    188: ["calibration", "long_tail"],
    189: ["simclr", "long_tail"],
    190: ["long_tail", "drw"],
}


def analyze_paper_text(paper: Paper) -> Dict[str, any]:
    """Analyze extracted paper text for key findings relevant to our codebase."""
    if not paper.text_path or not Path(paper.text_path).exists():
        return {"paper_id": paper.id, "status": "no_text", "key_findings": []}

    text = Path(paper.text_path).read_text(encoding="utf-8", errors="replace")
    text_lower = text.lower()

    concepts = PAPER_CONCEPT_MAP.get(paper.id, [])
    findings = []

    for concept in concepts:
        patterns = CONCEPT_PATTERNS.get(concept, [])
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text_lower)
            if found:
                matches.extend(found[:3])
        if matches:
            findings.append({
                "concept": concept,
                "match_count": len(matches),
                "sample_matches": matches[:5],
            })

    # Extract key sentences (heuristic: sentences with numbers and method names)
    key_sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 30 or len(line) > 500:
            continue
        if re.search(r'\d+\.\d+%|\d+\.\d{3,}|accuracy|F1|precision|recall', line, re.IGNORECASE):
            key_sentences.append(line)
        if len(key_sentences) >= 10:
            break

    return {
        "paper_id": paper.id,
        "title": paper.title,
        "status": "analyzed",
        "text_length": len(text),
        "concepts_found": [f["concept"] for f in findings],
        "key_findings": findings,
        "key_sentences": key_sentences[:5],
        "modules": paper.modules,
    }


def run_analysis(papers: List[Paper]) -> List[Dict]:
    """Run analysis over all extracted paper texts."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for paper in papers:
        result = analyze_paper_text(paper)
        results.append(result)

    # Save analysis results
    analysis_path = ANALYSIS_DIR / "paper_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Analysis saved to %s", analysis_path)

    # Generate analysis summary markdown
    summary_path = ANALYSIS_DIR / "ANALYSIS_SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Paper Analysis Summary\n\n")
        f.write(f"**Total papers**: {len(papers)}\n")
        downloaded = sum(1 for p in papers if p.pdf_path)
        extracted = sum(1 for p in papers if p.text_path)
        analyzed = sum(1 for r in results if r["status"] == "analyzed")
        f.write(f"**Downloaded**: {downloaded}\n")
        f.write(f"**Text extracted**: {extracted}\n")
        f.write(f"**Analyzed**: {analyzed}\n\n")

        # Group by module
        module_papers: Dict[str, List[Paper]] = {}
        for paper in papers:
            for mod in paper.modules:
                module_papers.setdefault(mod, []).append(paper)

        f.write("## Papers by Module\n\n")
        for mod in sorted(module_papers.keys()):
            f.write(f"### `{mod}`\n")
            for p in module_papers[mod]:
                cite = f"arXiv:{p.arxiv_id}" if p.arxiv_id else (f"DOI:{p.doi}" if p.doi else "")
                f.write(f"- [{p.id}] {p.authors} ({p.year}). *{p.title}*. {cite}\n")
            f.write("\n")

        f.write("## Key Findings by Concept\n\n")
        for result in results:
            if result["status"] != "analyzed" or not result.get("concepts_found"):
                continue
            f.write(f"### Paper {result['paper_id']}: {result['title']}\n")
            for finding in result["key_findings"]:
                f.write(f"- **{finding['concept']}**: {finding['match_count']} matches\n")
            if result.get("key_sentences"):
                f.write("- Key excerpts:\n")
                for sent in result["key_sentences"][:3]:
                    f.write(f"  > {sent[:200]}\n")
            f.write("\n")

    logger.info("Summary saved to %s", summary_path)
    return results


# ─── Code Annotation Generator ─────────────────────────────────────────────

def generate_reference_comments() -> Dict[str, List[str]]:
    """Generate academic reference comments for each source file."""
    module_refs: Dict[str, List[str]] = {}

    for paper in PAPERS:
        for mod in paper.modules:
            if mod not in module_refs:
                module_refs[mod] = []

            if paper.arxiv_id:
                cite = f"arXiv:{paper.arxiv_id}"
            elif paper.doi:
                cite = f"DOI:{paper.doi}"
            else:
                cite = f"({paper.year})"

            ref = f"[{paper.id}] {paper.authors} ({paper.year}). \"{paper.title}\". {cite}"
            module_refs[mod].append(ref)

    return module_refs


def write_reference_blocks(module_refs: Dict[str, List[str]]) -> Dict[str, str]:
    """Format reference blocks for insertion into source files."""
    blocks = {}
    for mod, refs in module_refs.items():
        lines = ["# Academic References:"]
        for ref in refs:
            lines.append(f"#   {ref}")
        blocks[mod] = "\n".join(lines)
    return blocks


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch, extract, and analyze academic papers")
    parser.add_argument("--download", action="store_true", help="Download PDFs from arXiv/Semantic Scholar")
    parser.add_argument("--extract", action="store_true", help="Extract text from downloaded PDFs")
    parser.add_argument("--analyze", action="store_true", help="Analyze paper content against codebase")
    parser.add_argument("--refs", action="store_true", help="Generate reference comment blocks")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    args = parser.parse_args()

    if not any([args.download, args.extract, args.analyze, args.refs, args.all]):
        parser.print_help()
        return 1

    if args.download or args.all:
        logger.info("=== Phase 1: Downloading Papers ===")
        download_papers(PAPERS)

    if args.extract or args.all:
        logger.info("\n=== Phase 2: Extracting Text ===")
        extract_all_papers(PAPERS)

    if args.analyze or args.all:
        logger.info("\n=== Phase 3: Analyzing Content ===")
        run_analysis(PAPERS)

    if args.refs or args.all:
        logger.info("\n=== Phase 4: Generating Reference Blocks ===")
        refs = generate_reference_comments()
        blocks = write_reference_blocks(refs)
        refs_path = ANALYSIS_DIR / "reference_blocks.json"
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        with open(refs_path, "w") as f:
            json.dump(blocks, f, indent=2)
        logger.info("Reference blocks saved to %s", refs_path)

        # Print summary
        for mod, block in sorted(blocks.items()):
            print(f"\n{'='*60}")
            print(f"  {mod}")
            print(f"{'='*60}")
            print(block)

    # Save paper metadata
    meta_path = ANALYSIS_DIR / "paper_metadata.json"
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump([asdict(p) for p in PAPERS], f, indent=2)
    logger.info("Paper metadata saved to %s", meta_path)

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
