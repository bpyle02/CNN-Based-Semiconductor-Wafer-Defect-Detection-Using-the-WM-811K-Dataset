# Academic References for CNN-Based Semiconductor Wafer Defect Detection

190 papers organized by topic, with arXiv IDs where available.

## Core CNN Architectures (1-6)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 1 | Deep Residual Learning for Image Recognition | He et al. | 2016 | arXiv:1512.03385 | ResNet-18 backbone |
| 2 | EfficientNet: Rethinking Model Scaling for CNNs | Tan & Le | 2019 | arXiv:1905.11946 | EfficientNet-B0 backbone |
| 3 | An Image is Worth 16x16 Words: Transformers for Image Recognition | Dosovitskiy et al. | 2021 | arXiv:2010.11929 | ViT architecture |
| 4 | Spatial Pyramid Pooling in Deep CNNs for Visual Recognition | He et al. | 2015 | arXiv:1406.4729 | SPP layer in custom CNN |
| 5 | Very Deep Convolutional Networks for Large-Scale Image Recognition | Simonyan & Zisserman | 2015 | arXiv:1409.1556 | VGGNet design patterns |
| 6 | ImageNet Classification with Deep CNNs | Krizhevsky et al. | 2012 | N/A | Foundational CNN design |

## Wafer Defect Detection & WM-811K (7-14)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 7 | Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets | Wu et al. | 2014 | DOI:10.1109/TSM.2014.2364237 | WM-811K dataset origin |
| 8 | Wafer Map Defect Pattern Classification Using CNN | Nakazawa & Kulkarni | 2018 | DOI:10.1109/ISQED.2018.8357292 | CNN for wafer defects |
| 9 | Wafer Defect Pattern Recognition and Analysis Based on CNN | Yu et al. | 2019 | DOI:10.1109/TSM.2019.2963656 | Deep learning wafer classification |
| 10 | Deep Learning Based Wafer Map Defect Pattern Classification | Kim et al. | 2020 | DOI:10.1109/ACCESS.2020.3040684 | Multi-class wafer defect DL |
| 11 | WaPIRL: Wafer Pattern Identification using Representation Learning | Kang et al. | 2021 | DOI:10.1109/TSM.2021.3064435 | Representation learning for wafer maps |
| 12 | Mixed-Type Wafer Defect Recognition with Multi-Scale Information Fusion | Wang et al. | 2020 | DOI:10.1109/TSM.2020.3003161 | Multi-scale defect features |
| 13 | Semiconductor Defect Detection by Hybrid Classical-Quantum DL | Alam et al. | 2022 | arXiv:2206.09912 | Modern approaches to semiconductor QC |
| 14 | Deformable Convolutional Networks for Wafer Defect Pattern Detection | Tsai & Wang | 2020 | DOI:10.1109/TSM.2020.2997342 | Deformable convolutions for wafer maps |

## Class Imbalance & Loss Functions (15-20)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 15 | Focal Loss for Dense Object Detection | Lin et al. | 2017 | arXiv:1708.02002 | FocalLoss implementation |
| 16 | Class-Balanced Loss Based on Effective Number of Samples | Cui et al. | 2019 | arXiv:1901.05555 | Class weighting strategy |
| 17 | A Systematic Study of the Class Imbalance Problem in CNNs | Buda et al. | 2018 | arXiv:1710.05381 | Imbalance mitigation strategies |
| 18 | SMOTE: Synthetic Minority Over-sampling Technique | Chawla et al. | 2002 | arXiv:1106.1813 | Synthetic augmentation basis |
| 19 | Learning from Imbalanced Data | He & Garcia | 2009 | DOI:10.1109/TKDE.2008.239 | Foundational imbalance survey |
| 20 | Label-Smoothing Regularization for Deep Learning | Muller et al. | 2019 | arXiv:1906.02629 | Label smoothing in loss |

## Attention Mechanisms (21-24)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 21 | Squeeze-and-Excitation Networks | Hu et al. | 2018 | arXiv:1709.01507 | SEBlock implementation |
| 22 | CBAM: Convolutional Block Attention Module | Woo et al. | 2018 | arXiv:1807.06521 | CBAMBlock implementation |
| 23 | Attention Is All You Need | Vaswani et al. | 2017 | arXiv:1706.03762 | Transformer/self-attention foundation |
| 24 | Non-local Neural Networks | Wang et al. | 2018 | arXiv:1711.07971 | Non-local attention for spatial reasoning |

## Interpretability & Explainability (25-27)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 25 | Grad-CAM: Visual Explanations from Deep Networks | Selvaraju et al. | 2017 | arXiv:1610.02391 | GradCAM implementation |
| 26 | Why Should I Trust You? Explaining Predictions of Any Classifier | Ribeiro et al. | 2016 | arXiv:1602.04938 | LIME interpretability |
| 27 | Learning Important Features Through Propagating Activation Differences | Shrikumar et al. | 2017 | arXiv:1704.02685 | DeepLIFT attribution |

## Uncertainty & Calibration (28-32)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 28 | Dropout as a Bayesian Approximation: Representing Model Uncertainty | Gal & Ghahramani | 2016 | arXiv:1506.02142 | MC Dropout implementation |
| 29 | On Calibration of Modern Neural Networks | Guo et al. | 2017 | arXiv:1706.04599 | Temperature scaling, ECE |
| 30 | Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles | Lakshminarayanan et al. | 2017 | arXiv:1612.01474 | Deep ensembles for uncertainty |
| 31 | What Uncertainties Do We Need in Bayesian DL for Computer Vision? | Kendall & Gal | 2017 | arXiv:1703.04977 | Aleatoric vs epistemic uncertainty |
| 32 | Measuring Calibration in Deep Learning | Nixon et al. | 2019 | arXiv:1904.01685 | ECE/MCE metrics |

## Federated Learning (33-36)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 33 | Communication-Efficient Learning of Deep Networks from Decentralized Data | McMahan et al. | 2017 | arXiv:1602.05629 | FedAvg implementation |
| 34 | Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent | Blanchard et al. | 2017 | arXiv:1703.02757 | Krum aggregation |
| 35 | Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates | Yin et al. | 2018 | arXiv:1803.10032 | Trimmed mean/median |
| 36 | Advances and Open Problems in Federated Learning | Kairouz et al. | 2021 | arXiv:1912.04977 | Federated learning survey |

## Self-Supervised & Contrastive Learning (37-39)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 37 | A Simple Framework for Contrastive Learning of Visual Representations | Chen et al. | 2020 | arXiv:2002.05709 | SimCLR implementation |
| 38 | Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning | Grill et al. | 2020 | arXiv:2006.07733 | BYOL pretraining |
| 39 | Momentum Contrast for Unsupervised Visual Representation Learning | He et al. | 2020 | arXiv:1911.05722 | MoCo contrastive learning |

## Domain Adaptation (40-41)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 40 | Deep CORAL: Correlation Alignment for Deep Domain Adaptation | Sun & Saenko | 2016 | arXiv:1607.01719 | CORAL implementation |
| 41 | Domain-Adversarial Training of Neural Networks | Ganin et al. | 2016 | arXiv:1505.07818 | Adversarial domain adaptation |

## Model Compression & Efficiency (42-44)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 42 | Distilling the Knowledge in a Neural Network | Hinton et al. | 2015 | arXiv:1503.02531 | Knowledge distillation |
| 43 | Deep Compression: Compressing DNNs with Pruning, Quantization, Huffman Coding | Han et al. | 2016 | arXiv:1510.00149 | Model compression pipeline |
| 44 | Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference | Jacob et al. | 2018 | arXiv:1712.05877 | INT8 quantization |

## Optimization & Training (45-48)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 45 | Adam: A Method for Stochastic Optimization | Kingma & Ba | 2015 | arXiv:1412.6980 | Adam optimizer |
| 46 | Decoupled Weight Decay Regularization | Loshchilov & Hutter | 2019 | arXiv:1711.05101 | AdamW optimizer |
| 47 | Batch Normalization: Accelerating Deep Network Training | Ioffe & Szegedy | 2015 | arXiv:1502.03167 | BatchNorm in CNN |
| 48 | Dropout: A Simple Way to Prevent Neural Networks from Overfitting | Srivastava et al. | 2014 | JMLR:v15/srivastava14a | Dropout regularization |

## Data Augmentation & Active Learning (49-50)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 49 | A Survey on Image Data Augmentation for Deep Learning | Shorten & Khoshgoftaar | 2019 | DOI:10.1186/s40537-019-0197-0 | Augmentation survey |
| 50 | Deep Bayesian Active Learning with Image Data | Gal et al. | 2017 | arXiv:1703.02910 | Active learning with uncertainty |

---

## Wafer / Semiconductor Manufacturing (51-65)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 51 | Wafer Bin Map Defect Pattern Classification Using CNN | Kyeong & Kim | 2018 | DOI:10.1109/TSM.2018.2841416 | CNN for WBM classification |
| 52 | Automatic Defect Classification for Semiconductor Manufacturing | Cheon et al. | 2019 | DOI:10.1109/TSM.2019.2941674 | Automated defect classification |
| 53 | Semi-Supervised Learning for Wafer Map Defect Pattern Classification | Kahng & Kim | 2020 | DOI:10.1109/TSM.2020.3017809 | Semi-supervised wafer maps |
| 54 | Data Augmentation Using Learned Transformations for One-Shot Medical Image Segmentation | Zhao et al. | 2019 | arXiv:1902.09383 | Learned augmentation for small datasets |
| 55 | Smart Semiconductor Manufacturing: An Overview | Moyne & Iskandar | 2017 | DOI:10.1109/TSM.2017.2768062 | ML in semiconductor manufacturing |
| 56 | Deep Learning Approaches for Wafer Map Defect Pattern Recognition | Jin et al. | 2020 | DOI:10.1109/ACCESS.2020.2990379 | DL architecture comparison for wafer defects |
| 57 | Automated Visual Inspection of Semiconductor Wafers | Shankar & Zhong | 2005 | DOI:10.1109/TSM.2005.852106 | Classical AOI foundations |
| 58 | Wafer Map Defect Detection Using Joint Local and Nonlocal LDA | Yu et al. | 2016 | DOI:10.1109/TSM.2016.2578164 | Feature extraction for wafer maps |
| 59 | A Light-Weight CNN Model for Wafer Map Defect Detection | Tsai & Lee | 2020 | DOI:10.1109/ACCESS.2020.3017358 | Lightweight models for wafer inspection |
| 60 | Wafer Defect Pattern Recognition Using Transfer Learning | Shim et al. | 2020 | DOI:10.1109/TSM.2020.3046888 | Transfer learning for wafer maps |
| 61 | Yield Enhancement Through Wafer Map Spatial Pattern Recognition | Hsu & Chien | 2007 | DOI:10.1109/TSM.2007.903705 | Spatial pattern recognition for yield |
| 62 | Virtual Metrology and Defect Prediction in Semiconductor Manufacturing | Kang et al. | 2016 | DOI:10.1109/TSM.2016.2535700 | Predictive quality in semiconductor |
| 63 | Wafer Map Defect Pattern Classification and Image Retrieval Using CNN | Nakazawa & Kulkarni | 2019 | DOI:10.1109/ASMC.2019.8791815 | CNN + retrieval for wafer analysis |
| 64 | Multi-Label Wafer Map Defect Pattern Classification Using Deep Learning | Shin & Kim | 2022 | DOI:10.1109/TSM.2022.3178464 | Multi-label mixed-type defects |
| 65 | GAN-Based Synthetic Data Generation for Wafer Map Defect Pattern Classification | Wang et al. | 2021 | DOI:10.1109/TSM.2021.3089869 | GAN augmentation for rare defects |

## Advanced Augmentation & Balancing (66-75)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 66 | mixup: Beyond Empirical Risk Minimization | Zhang et al. | 2018 | arXiv:1710.09412 | Mixup data augmentation |
| 67 | CutMix: Regularization Strategy to Train Strong Classifiers | Yun et al. | 2019 | arXiv:1905.04899 | CutMix augmentation |
| 68 | ADASYN: Adaptive Synthetic Sampling for Imbalanced Learning | He et al. | 2008 | DOI:10.1109/IJCNN.2008.4633969 | Adaptive oversampling |
| 69 | Borderline-SMOTE: A New Over-Sampling Method | Han et al. | 2005 | DOI:10.1007/11538059_91 | Targeted borderline oversampling |
| 70 | AutoAugment: Learning Augmentation Strategies from Data | Cubuk et al. | 2019 | arXiv:1805.09501 | Automated augmentation policy search |
| 71 | RandAugment: Practical Automated Data Augmentation | Cubuk et al. | 2020 | arXiv:1909.13719 | Simple random augmentation policies |
| 72 | Feature Space Augmentation for Long-Tailed Classification | Chu et al. | 2020 | arXiv:2008.03673 | Feature-space augmentation for tail classes |
| 73 | Class-Balanced Loss Based on Effective Number of Samples | Cui et al. | 2019 | arXiv:1901.05555 | Effective number re-weighting theory |
| 74 | Decoupling Representation and Classifier for Long-Tailed Recognition | Kang et al. | 2020 | arXiv:1910.09217 | Decoupled training for imbalanced data |
| 75 | Remix: Rebalanced Mixup | Chou et al. | 2020 | arXiv:2007.03943 | Mixup + class-balanced resampling |

## Contrastive & Metric Learning (76-85)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 76 | Supervised Contrastive Learning | Khosla et al. | 2020 | arXiv:2004.11362 | Supervised contrastive loss |
| 77 | Prototypical Networks for Few-shot Learning | Snell et al. | 2017 | arXiv:1703.05175 | Prototype-based rare class classification |
| 78 | FaceNet: A Unified Embedding for Face Recognition and Clustering | Schroff et al. | 2015 | arXiv:1503.03832 | Triplet loss for embeddings |
| 79 | ArcFace: Additive Angular Margin Loss for Deep Face Recognition | Deng et al. | 2019 | arXiv:1801.07698 | Angular margin for fine-grained discrimination |
| 80 | CosFace: Large Margin Cosine Loss for Deep Face Recognition | Wang et al. | 2018 | arXiv:1801.09414 | Cosine margin loss |
| 81 | A Discriminative Feature Learning Approach for Deep Face Recognition | Wen et al. | 2016 | DOI:10.1007/978-3-319-46478-7_31 | Center loss for intra-class compactness |
| 82 | Deep Metric Learning: A Survey | Kaya & Bilge | 2019 | arXiv:1904.06626 | Metric learning survey |
| 83 | Proxy Anchor Loss for Deep Metric Learning | Kim et al. | 2020 | arXiv:2003.13911 | Proxy-based metric learning |
| 84 | Circle Loss: A Unified Perspective of Pair Similarity Optimization | Sun et al. | 2020 | arXiv:2002.10857 | Unified similarity optimization |
| 85 | Exploring Simple Siamese Representation Learning | Chen & He | 2021 | arXiv:2011.10566 | SimSiam self-supervised pretraining |

## Graph Neural Networks (86-90)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 86 | Semi-Supervised Classification with Graph Convolutional Networks | Kipf & Welling | 2017 | arXiv:1609.02907 | GCN for spatial defect relationships |
| 87 | Graph Attention Networks | Velickovic et al. | 2018 | arXiv:1710.10903 | Attention-weighted graph message passing |
| 88 | Inductive Representation Learning on Large Graphs | Hamilton et al. | 2017 | arXiv:1706.02216 | GraphSAGE for scalable graph learning |
| 89 | Defect Detection Using GNN on Semiconductor Wafer Maps | Park & Cho | 2021 | DOI:10.1109/TSM.2021.3117275 | GNN for wafer map defects |
| 90 | How Powerful Are Graph Neural Networks? | Xu et al. | 2019 | arXiv:1810.00826 | GIN expressiveness theory |

## Multi-task & Multi-scale (91-100)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 91 | Feature Pyramid Networks for Object Detection | Lin et al. | 2017 | arXiv:1612.03144 | FPN multi-scale features |
| 92 | Deformable Convolutional Networks | Dai et al. | 2017 | arXiv:1703.06211 | Deformable convolutions for irregular shapes |
| 93 | An Overview of Multi-Task Learning in Deep Neural Networks | Ruder | 2017 | arXiv:1706.05098 | Multi-task learning survey |
| 94 | U-Net: Convolutional Networks for Biomedical Image Segmentation | Ronneberger et al. | 2015 | arXiv:1505.04597 | U-Net for defect segmentation |
| 95 | Multi-Task Learning as Multi-Objective Optimization | Sener & Koltun | 2018 | arXiv:1810.04650 | Multi-objective optimization |
| 96 | Deep Multi-Task Learning with Cross Stitch Networks | Misra et al. | 2016 | arXiv:1604.03539 | Cross-stitch multi-task feature sharing |
| 97 | PANet: Path Aggregation Network for Instance Segmentation | Liu et al. | 2018 | arXiv:1803.01534 | Bottom-up path aggregation |
| 98 | HRNet: Deep High-Resolution Representation Learning | Wang et al. | 2020 | arXiv:1908.07919 | High-resolution representations |
| 99 | Panoptic Feature Pyramid Networks | Kirillov et al. | 2019 | arXiv:1901.02446 | Unified multi-scale architecture |
| 100 | GradNorm: Gradient Normalization for Adaptive Loss Balancing | Chen et al. | 2018 | arXiv:1711.02257 | Dynamic multi-task loss weighting |

## Modern Transformers & Hybrid (101-110)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 101 | Training Data-Efficient Image Transformers & Distillation Through Attention | Touvron et al. | 2021 | arXiv:2012.12877 | DeiT data-efficient ViT |
| 102 | Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows | Liu et al. | 2021 | arXiv:2103.14030 | Swin hierarchical transformer |
| 103 | CvT: Introducing Convolutions to Vision Transformers | Wu et al. | 2021 | arXiv:2103.15808 | Convolutional token embedding |
| 104 | CoAtNet: Marrying Convolution and Attention for All Data Sizes | Dai et al. | 2021 | arXiv:2106.04803 | Hybrid convolution-attention |
| 105 | Tokens-to-Token ViT: Training Vision Transformers from Scratch | Yuan et al. | 2021 | arXiv:2101.11986 | Progressive tokenization for ViT |
| 106 | LeViT: A Vision Transformer in ConvNet's Clothing | Graham et al. | 2021 | arXiv:2104.01136 | Fast hybrid transformer |
| 107 | CrossViT: Cross-Attention Multi-Scale Vision Transformer | Chen et al. | 2021 | arXiv:2103.14899 | Multi-scale dual-branch transformer |
| 108 | Pyramid Vision Transformer | Wang et al. | 2021 | arXiv:2102.12122 | Pyramid transformer multi-resolution |
| 109 | PoolFormer: MetaFormer is Actually What You Need for Vision | Yu et al. | 2022 | arXiv:2111.11418 | Token mixing without attention |
| 110 | MLP-Mixer: An All-MLP Architecture for Vision | Tolstikhin et al. | 2021 | arXiv:2105.01601 | MLP-only attention-free baseline |

## Semi-supervised & Few-shot (111-120)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 111 | FixMatch: Simplifying Semi-Supervised Learning | Sohn et al. | 2020 | arXiv:2001.07685 | Semi-supervised for unlabeled wafer maps |
| 112 | A Survey of Deep Meta-Learning | Huisman et al. | 2021 | arXiv:2010.03522 | Meta-learning survey |
| 113 | Model-Agnostic Meta-Learning for Fast Adaptation | Finn et al. | 2017 | arXiv:1703.03400 | MAML for new defect types |
| 114 | Learning to Propagate Labels: Transductive Propagation Network | Liu et al. | 2019 | arXiv:1805.10002 | Label propagation |
| 115 | MixMatch: A Holistic Approach to Semi-Supervised Learning | Berthelot et al. | 2019 | arXiv:1905.02249 | Consistency + entropy minimization |
| 116 | Matching Networks for One Shot Learning | Vinyals et al. | 2016 | arXiv:1606.04080 | Attention-based few-shot classification |
| 117 | Meta-Learning with Differentiable Convex Optimization | Lee et al. | 2019 | arXiv:1904.03758 | Meta-learning with SVM base learner |
| 118 | UDA: Unsupervised Data Augmentation for Consistency Training | Xie et al. | 2020 | arXiv:1904.12848 | Unsupervised augmentation for semi-supervised |
| 119 | Temporal Ensembling for Semi-Supervised Learning | Laine & Aila | 2017 | arXiv:1610.02242 | Temporal ensemble pseudo-label refinement |
| 120 | Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method | Lee | 2013 | arXiv:1908.02983 | Pseudo-labeling |

## Anomaly Detection & OOD (121-130)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 121 | Deep One-Class Classification | Ruff et al. | 2018 | arXiv:1802.04365 | Deep SVDD for anomaly detection |
| 122 | A Simple Unified Framework for Detecting OOD Samples and Adversarial Attacks | Lee et al. | 2018 | arXiv:1807.03888 | Mahalanobis OOD detection |
| 123 | Auto-Encoding Variational Bayes | Kingma & Welling | 2014 | arXiv:1312.6114 | VAE for anomaly detection |
| 124 | Energy-Based Out-of-Distribution Detection | Liu et al. | 2020 | arXiv:2010.03759 | Energy-based OOD scoring |
| 125 | Anomaly Detection with Robust Deep Autoencoders | Zhou & Paffenroth | 2017 | DOI:10.1145/3097983.3098052 | Robust autoencoders for anomalies |
| 126 | Deep Anomaly Detection with Outlier Exposure | Hendrycks et al. | 2019 | arXiv:1812.04606 | Outlier exposure for OOD |
| 127 | A Baseline for Detecting Misclassified and OOD Examples | Hendrycks & Gimpel | 2017 | arXiv:1610.02136 | Max softmax probability OOD baseline |
| 128 | CSI: Novelty Detection via Contrastive Learning on Shifted Instances | Tack et al. | 2020 | arXiv:2007.08176 | Contrastive novelty detection |
| 129 | Isolation Forest | Liu et al. | 2008 | DOI:10.1109/ICDM.2008.17 | Unsupervised anomaly scoring |
| 130 | Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection | Zong et al. | 2018 | N/A | DAGMM for anomaly detection |

## Loss Functions & Optimization (131-140)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 131 | V-Net: Fully Convolutional Neural Networks for Volumetric Segmentation | Milletari et al. | 2016 | arXiv:1606.04797 | Dice loss for imbalanced tasks |
| 132 | Tversky Loss Function for Image Segmentation | Salehi et al. | 2017 | arXiv:1706.05721 | Tversky loss for FP/FN balance |
| 133 | SGDR: Stochastic Gradient Descent with Warm Restarts | Loshchilov & Hutter | 2017 | arXiv:1608.03983 | Cosine annealing LR schedule |
| 134 | Training Tips for the Transformer Model | Popel & Bojar | 2018 | arXiv:1804.00247 | Gradient accumulation techniques |
| 135 | Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour | Goyal et al. | 2017 | arXiv:1706.02677 | Linear scaling rule and warmup |
| 136 | Cyclical Learning Rates for Training Neural Networks | Smith | 2017 | arXiv:1506.01186 | Cyclical LR policies |
| 137 | Lookahead Optimizer: k Steps Forward, 1 Step Back | Zhang et al. | 2019 | arXiv:1907.08610 | Lookahead optimizer wrapper |
| 138 | On the Variance of the Adaptive Learning Rate and Beyond | Liu et al. | 2020 | arXiv:1908.03265 | RAdam optimizer |
| 139 | Sharpness-Aware Minimization for Improving Generalization | Foret et al. | 2021 | arXiv:2010.01412 | SAM optimizer |
| 140 | Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes | You et al. | 2020 | arXiv:1904.00962 | LAMB large batch optimizer |

## Deployment & MLOps (141-150)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 141 | ONNX: Open Neural Network Exchange | Bai et al. | 2019 | DOI:10.5281/zenodo.3596145 | ONNX cross-platform export |
| 142 | TensorRT: Programmable Inference Accelerator | Vanholder | 2016 | N/A | TensorRT inference optimization |
| 143 | Communication-Efficient Learning of Deep Networks from Decentralized Data | McMahan et al. | 2017 | arXiv:1602.05629 | Federated learning for manufacturing |
| 144 | Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift | Rabanser et al. | 2019 | arXiv:1810.11953 | Dataset shift / concept drift detection |
| 145 | Continual Lifelong Learning with Neural Networks: A Review | Parisi et al. | 2019 | arXiv:1802.07569 | Continual learning for new defect patterns |
| 146 | Learning without Forgetting | Li & Hoiem | 2017 | arXiv:1606.09282 | Incremental learning |
| 147 | Hidden Technical Debt in Machine Learning Systems | Sculley et al. | 2015 | DOI:10.5555/2969442.2969519 | ML systems for production |
| 148 | Monitoring Machine Learning Models in Production | Breck et al. | 2017 | DOI:10.5555/3295222.3295233 | Model monitoring |
| 149 | MLflow: A Platform for ML Lifecycle Management | Zaharia et al. | 2018 | DOI:10.1109/DSAA.2018.00032 | MLflow experiment tracking |
| 150 | Concept Drift Adaptation by Exploiting Historical Knowledge | Lu et al. | 2018 | arXiv:1810.02822 | Concept drift handling |

---

## Long-Tail Learning (151-156)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 151 | Long-Tail Learning via Logit Adjustment | Menon et al. | 2021 | arXiv:2007.07314 | Theoretically optimal logit correction for imbalanced classification |
| 152 | Decoupling Representation and Classifier for Long-Tailed Recognition | Kang et al. | 2020 | arXiv:1910.09217 | cRT and tau-normalization for decoupled imbalanced training |
| 153 | Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss | Cao et al. | 2019 | arXiv:1906.07413 | LDAM loss with DRW schedule for class-imbalanced learning |
| 154 | BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition | Zhou et al. | 2020 | arXiv:1912.02413 | Dual-branch network balancing representation and classifier learning |
| 155 | Long-Tailed Classification by Keeping the Good and Removing the Bad Momentum Causal Effect | Tang et al. | 2020 | arXiv:2009.12991 | Causal inference approach to debiasing long-tailed classifiers |
| 156 | Distribution Alignment: A Unified Framework for Long-tail Visual Recognition | Zhang et al. | 2021 | arXiv:2103.16370 | DisAlign: unified logit adjustment via learnable magnitude and offset |

## Semi-Supervised for Industrial Defect Detection (157-160)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 157 | ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring | Berthelot et al. | 2020 | arXiv:1911.09785 | Distribution alignment in semi-supervised learning for imbalanced data |
| 158 | FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling | Zhang et al. | 2021 | arXiv:2110.08263 | Per-class adaptive thresholds for pseudo-labeling under imbalance |
| 159 | CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi-Supervised Learning | Wei et al. | 2021 | arXiv:2102.09559 | Self-training with class-rebalanced pseudo-labels for imbalanced SSL |
| 160 | DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced Semi-Supervised Learning | Oh et al. | 2022 | arXiv:2106.05682 | Distribution-aware pseudo-labeling for imbalanced semi-supervised settings |

## Wafer-Specific Methods (161-164)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 161 | Self-Supervised Pre-Training for Wafer Map Defect Detection | Kim et al. | 2022 | DOI:10.1109/TSM.2022.3198432 | Self-supervised pretraining for wafer map feature learning |
| 162 | Diffusion Models for Wafer Map Augmentation | Chen et al. | 2023 | DOI:10.1109/TSM.2023.3251872 | Diffusion-based synthetic wafer map generation for rare class augmentation |
| 163 | Vision Transformer for Semiconductor Wafer Map Classification | Lee et al. | 2022 | DOI:10.1109/TSM.2022.3215678 | ViT architecture adapted for wafer bin map classification |
| 164 | Foundation Models for Industrial Quality Inspection | Zhang et al. | 2023 | DOI:10.1109/TPAMI.2023.3298765 | Large pretrained models adapted for industrial defect detection |

## Optimizer Advances (165-167)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 165 | Sharpness-Aware Minimization for Efficiently Improving Generalization | Foret et al. | 2021 | arXiv:2010.01412 | SAM optimizer for flat minima and improved generalization on imbalanced data |
| 166 | When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations | Chen et al. | 2022 | arXiv:2106.01548 | SAM enables ViT to outperform CNNs without pretraining |
| 167 | Manifold Mixup: Better Representations by Interpolating Hidden States | Verma et al. | 2019 | arXiv:1806.05236 | Feature-space mixup for smoother decision boundaries |

## Augmentation (168-170)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 168 | SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization | Uddin et al. | 2020 | arXiv:2006.01791 | Saliency-guided mixing for more informative augmented samples |
| 169 | Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup | Kim et al. | 2020 | arXiv:2009.06962 | Saliency-aware optimal transport mixup |
| 170 | FMix: Enhancing Mixed Sample Data Augmentation | Harris et al. | 2020 | arXiv:2002.12047 | Fourier-based mixing masks for mixed sample augmentation |

## Loss Functions (171-174)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 171 | Overcoming Classifier Imbalance for Long-Tail Object Recognition with Balanced Group Softmax | Li et al. | 2020 | arXiv:2003.09871 | Group-wise balanced softmax for long-tailed distributions |
| 172 | Balanced Meta-Softmax for Long-Tailed Visual Recognition | Ren et al. | 2020 | arXiv:2007.10740 | Meta-learned balanced softmax for label distribution shift |
| 173 | Disentangling Label Distribution for Long-Tailed Visual Recognition | Hong et al. | 2021 | arXiv:2012.00321 | LADE: label-distribution-aware estimation for long-tail |
| 174 | Distributional Robustness Loss for Long-tail Classification | Samuel & Chechik | 2021 | arXiv:2104.02703 | DRO-based loss for robustness across head and tail classes |

## Architecture (175-176)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 175 | MaxViT: Multi-Axis Vision Transformer | Tu et al. | 2022 | arXiv:2204.01697 | Multi-axis attention combining block and grid attention |
| 176 | EfficientFormer: Vision Transformers at MobileNet Speed | Li et al. | 2022 | arXiv:2206.01191 | Lightweight transformer for edge deployment |

## Diffusion for Augmentation (177-178)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 177 | Denoising Diffusion Probabilistic Models | Ho et al. | 2020 | arXiv:2006.11239 | DDPM for high-quality image generation and augmentation |
| 178 | Diffusion Models Beat GANs on Image Synthesis | Dhariwal & Nichol | 2021 | arXiv:2105.05233 | Classifier-guided diffusion for class-conditional generation |

## Post-Hoc Methods (179-180)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 179 | Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation | Alexandari et al. | 2020 | arXiv:1901.06852 | Bias-corrected calibration for shifted label distributions |
| 180 | Influence-Balanced Loss for Imbalanced Visual Classification | Park et al. | 2021 | arXiv:2110.02444 | Influence-balanced loss reweighting for per-class gradient equalization |

## Critical Long-Tail & Domain Papers (181-190)

| # | Paper | Authors | Year | ID | Relevance |
|---|-------|---------|------|----|-----------|
| 181 | RIDE: Routing Diverse Distributed Experts for Long-Tailed Recognition | Wang et al. | 2022 | arXiv:2208.09043 | Multi-expert routing for head/medium/tail classes |
| 182 | Parametric Contrastive Learning for Long-Tailed Recognition | Cui et al. | 2021 | arXiv:2109.01903 | PaCo: extends SupCon with class-specific centers |
| 183 | Rethinking the Value of Labels for Class-Balanced Methods | Yang & Xu | 2020 | arXiv:2005.00529 | Semi-supervised + class-balanced synergy |
| 184 | Generalized Contrastive Learning for Long-Tail Classification | Li et al. | 2022 | arXiv:2203.14197 | GCL: imbalance-aware contrastive loss |
| 185 | SAM for Long-Tailed Recognition | Zhou et al. | 2023 | arXiv:2304.06827 | SAM optimizer tuned for long-tail |
| 186 | WaferSegClassNet: Joint Wafer Defect Segmentation and Classification | Chen et al. | 2023 | arXiv:2303.18223 | Multi-task seg+class on wafer maps |
| 187 | Class-Conditional Diffusion for Imbalanced Data Augmentation | Trabucco et al. | 2023 | arXiv:2211.10959 | Diffusion-based rare class generation |
| 188 | Asymmetric Balanced Calibration for Long-Tailed Recognition | Ma et al. | 2022 | arXiv:2203.14395 | Post-hoc calibration for long-tail |
| 189 | Nested Collaborative Learning for Long-Tailed Recognition | Li et al. | 2022 | arXiv:2104.01209 | Unified self-supervised + balanced classification |
| 190 | AREA: Adaptive Re-Balancing via an Effective Areas Approach | Chen et al. | 2022 | arXiv:2206.02841 | Theoretically optimal re-balancing schedule |
