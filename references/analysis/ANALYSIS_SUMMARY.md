# Paper Analysis Summary

**Total papers**: 150
**Downloaded**: 0
**Text extracted**: 0
**Analyzed**: 0

## Papers by Module

### `scripts/active_learn.py`
- [50] Gal, Islam, Ghahramani (2017). *Deep Bayesian Active Learning with Image Data*. arXiv:1703.02910

### `scripts/compress_model.py`
- [42] Hinton, Vinyals, Dean (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531
- [43] Han, Mao, Dally (2016). *Deep Compression: Compressing DNNs with Pruning, Quantization, Huffman Coding*. arXiv:1510.00149
- [44] Jacob et al. (2018). *Quantization and Training of NNs for Efficient Integer-Arithmetic-Only Inference*. arXiv:1712.05877
- [59] Tsai, Lee (2020). *A Light-Weight CNN Model for Wafer Map Defect Detection*. DOI:10.1109/ACCESS.2020.3017358
- [106] Graham, El-Nouby, Touvron, Stock, Joulin, Jegou, Douze (2021). *LeViT: A Vision Transformer in ConvNet's Clothing for Faster Inference*. arXiv:2104.01136
- [141] Bai, Gao, Lin, Zhang (2019). *ONNX: Open Neural Network Exchange*. DOI:10.5281/zenodo.3596145
- [142] Vanholder (2016). *TensorRT: Programmable Inference Accelerator*. 

### `src/analysis/anomaly.py`
- [121] Ruff, Goernitz, Deecke, Siddiqui, Vandermeulen, Borghesi, Kloft, Muller (2018). *Deep One-Class Classification*. arXiv:1802.04365
- [123] Kingma, Welling (2014). *Auto-Encoding Variational Bayes*. arXiv:1312.6114
- [125] Zhou, Paffenroth (2017). *Anomaly Detection with Robust Deep Autoencoders*. DOI:10.1145/3097983.3098052
- [129] Liu, Ting, Zhou (2008). *Isolation Forest*. DOI:10.1109/ICDM.2008.17
- [130] Zong, Song, Min, Cheng, Lumezanu, Cho, Chen (2018). *Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection*. DOI:10.openreview.net/forum?id=BJJLHbb0-

### `src/analysis/evaluate.py`
- [13] Alam et al. (2022). *Semiconductor Defect Detection by Hybrid Classical-Quantum DL*. arXiv:2206.09912
- [29] Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural Networks*. arXiv:1706.04599
- [32] Nixon, Dusenberry, Zhang, Jerfel, Tran (2019). *Measuring Calibration in Deep Learning*. arXiv:1904.01685
- [57] Shankar, Zhong (2005). *Automated Visual Inspection of Semiconductor Wafers*. DOI:10.1109/TSM.2005.852106
- [62] Kang, Kim, Cho, Kang (2016). *Virtual Metrology and Defect Prediction in Semiconductor Manufacturing*. DOI:10.1109/TSM.2016.2535700

### `src/augmentation/synthetic.py`
- [18] Chawla, Bowyer, Hall, Kegelmeyer (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. arXiv:1106.1813
- [54] Zhao, Data, Greenspan, St-Onge, Bhatt, Murphy, Grady (2019). *Data Augmentation Using Learned Transformations for One-Shot Medical Image Segmentation*. arXiv:1902.09383
- [65] Wang, Hsieh, Liu, Hsu (2021). *GAN-Based Synthetic Data Generation for Wafer Map Defect Pattern Classification*. DOI:10.1109/TSM.2021.3089869
- [66] Zhang, Cisse, Dauphin, Lopez-Paz (2018). *mixup: Beyond Empirical Risk Minimization*. arXiv:1710.09412
- [67] Yun, Han, Oh, Chun, Choe, Yoo (2019). *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*. arXiv:1905.04899
- [68] He, Bai, Garcia, Li (2008). *ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning*. DOI:10.1109/IJCNN.2008.4633969
- [69] Han, Wang, Mao (2005). *Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning*. DOI:10.1007/11538059_91
- [72] Chu, Zhong, Wang (2020). *Feature Space Augmentation for Long-Tailed Classification*. arXiv:2008.03673
- [75] Chou, Chen, Lee (2020). *Remix: Rebalanced Mixup*. arXiv:2007.03943

### `src/data/dataset.py`
- [7] Wu, Yeh, Chen (2014). *Wafer Map Failure Pattern Recognition and Similarity Ranking*. DOI:10.1109/TSM.2014.2364237
- [61] Hsu, Chien (2007). *Yield Enhancement Through Wafer Map Spatial Pattern Recognition*. DOI:10.1109/TSM.2007.903705

### `src/data/preprocessing.py`
- [7] Wu, Yeh, Chen (2014). *Wafer Map Failure Pattern Recognition and Similarity Ranking*. DOI:10.1109/TSM.2014.2364237
- [9] Yu, Lu, Zheng (2019). *Wafer Defect Pattern Recognition and Analysis Based on CNN*. DOI:10.1109/TSM.2019.2963656
- [49] Shorten, Khoshgoftaar (2019). *A Survey on Image Data Augmentation for Deep Learning*. DOI:10.1186/s40537-019-0197-0
- [51] Kyeong, Kim (2018). *Wafer Bin Map Defect Pattern Classification Using Convolutional Neural Network*. DOI:10.1109/TSM.2018.2841416
- [58] Yu, Zheng, Shan (2016). *Wafer Map Defect Detection and Recognition Using Joint Local and Nonlocal Linear Discriminant Analysis*. DOI:10.1109/TSM.2016.2578164
- [66] Zhang, Cisse, Dauphin, Lopez-Paz (2018). *mixup: Beyond Empirical Risk Minimization*. arXiv:1710.09412
- [67] Yun, Han, Oh, Chun, Choe, Yoo (2019). *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features*. arXiv:1905.04899
- [70] Cubuk, Zoph, Mane, Vasudevan, Le (2019). *AutoAugment: Learning Augmentation Strategies from Data*. arXiv:1805.09501
- [71] Cubuk, Zoph, Shlens, Le (2020). *RandAugment: Practical Automated Data Augmentation with a Reduced Search Space*. arXiv:1909.13719
- [75] Chou, Chen, Lee (2020). *Remix: Rebalanced Mixup*. arXiv:2007.03943
- [89] Park, Cho (2021). *Defect Detection Using Graph Neural Networks on Semiconductor Wafer Maps*. DOI:10.1109/TSM.2021.3117275
- [118] Xie, Dai, Hovy, Luong, Le (2020). *UDA: Unsupervised Data Augmentation for Consistency Training*. arXiv:1904.12848

### `src/detection/ood.py`
- [121] Ruff, Goernitz, Deecke, Siddiqui, Vandermeulen, Borghesi, Kloft, Muller (2018). *Deep One-Class Classification*. arXiv:1802.04365
- [122] Lee, Lee, Lee, Shin (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks*. arXiv:1807.03888
- [124] Liu, Wang, Owens, Li (2020). *Energy-Based Out-of-Distribution Detection*. arXiv:2010.03759
- [126] Hendrycks, Mazeika, Dietterich (2019). *Deep Anomaly Detection with Outlier Exposure*. arXiv:1812.04606
- [127] Hendrycks, Gimpel (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples*. arXiv:1610.02136
- [128] Tack, Yu, Jeong, Kim, Shin, Shin (2020). *CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances*. arXiv:2007.08176
- [144] Rabanser, Gunnemann, Lipton (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*. arXiv:1810.11953
- [150] Lu, Liu, Dong, Gu, Gama, Zhang (2018). *Concept Drift Adaptation by Exploiting Historical Knowledge*. arXiv:1810.02822

### `src/federated/fed_avg.py`
- [33] McMahan, Moore, Ramage, Hampson, Arcas (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. arXiv:1602.05629
- [34] Blanchard, El Mhamdi, Guerraoui, Stainer (2017). *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent*. arXiv:1703.02757
- [35] Yin, Chen, Kannan, Bartlett (2018). *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates*. arXiv:1803.10032
- [36] Kairouz et al. (2021). *Advances and Open Problems in Federated Learning*. arXiv:1912.04977
- [143] McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. arXiv:1602.05629

### `src/inference/gradcam.py`
- [25] Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. arXiv:1610.02391
- [26] Ribeiro, Singh, Guestrin (2016). *Why Should I Trust You? Explaining Predictions of Any Classifier*. arXiv:1602.04938
- [27] Shrikumar, Greenside, Kundaje (2017). *Learning Important Features Through Propagating Activation Differences*. arXiv:1704.02685

### `src/inference/server.py`
- [141] Bai, Gao, Lin, Zhang (2019). *ONNX: Open Neural Network Exchange*. DOI:10.5281/zenodo.3596145

### `src/inference/uncertainty.py`
- [28] Gal, Ghahramani (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty*. arXiv:1506.02142
- [29] Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural Networks*. arXiv:1706.04599
- [31] Kendall, Gal (2017). *What Uncertainties Do We Need in Bayesian DL for Computer Vision?*. arXiv:1703.04977
- [48] Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. DOI:10.5555/2627435.2670313

### `src/mlops/wandb_logger.py`
- [147] Sculley, Holt, Golovin, Davydov, Phillips, Ebner, Chaudhary, Young, Crespo, Dennison (2015). *Hidden Technical Debt in Machine Learning Systems*. DOI:10.5555/2969442.2969519
- [148] Breck, Cai, Nielsen, Salib, Sculley (2017). *Monitoring Machine Learning Models in Production*. DOI:10.5555/3295222.3295233
- [149] Zaharia, Chen, Davidson, Ghodsi, Hong, Konwinski, Murching, Nykodym, Ogilvie, Parkhe, Xie, Zuber (2018). *MLflow: A Platform for ML Lifecycle Management*. DOI:10.1109/DSAA.2018.00032

### `src/models/attention.py`
- [21] Hu, Shen, Sun (2018). *Squeeze-and-Excitation Networks*. arXiv:1709.01507
- [22] Woo, Park, Lee, Kweon (2018). *CBAM: Convolutional Block Attention Module*. arXiv:1807.06521
- [24] Wang, Girshick, Gupta, He (2018). *Non-local Neural Networks*. arXiv:1711.07971
- [87] Velickovic, Cucurull, Casanova, Romero, Lio, Bengio (2018). *Graph Attention Networks*. arXiv:1710.10903
- [116] Vinyals, Blundell, Lillicrap, Kavukcuoglu, Wierstra (2016). *Matching Networks for One Shot Learning*. arXiv:1606.04080

### `src/models/cnn.py`
- [4] He, Zhang, Ren, Sun (2015). *Spatial Pyramid Pooling in Deep CNNs*. arXiv:1406.4729
- [5] Simonyan, Zisserman (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556
- [6] Krizhevsky, Sutskever, Hinton (2012). *ImageNet Classification with Deep CNNs*. DOI:10.1145/3065386
- [8] Nakazawa, Kulkarni (2018). *Wafer Map Defect Pattern Classification Using CNN*. DOI:10.1109/ISQED.2018.8357292
- [9] Yu, Lu, Zheng (2019). *Wafer Defect Pattern Recognition and Analysis Based on CNN*. DOI:10.1109/TSM.2019.2963656
- [12] Wang et al. (2020). *Mixed-Type Wafer Defect Recognition with Multi-Scale Info Fusion*. DOI:10.1109/TSM.2020.3003161
- [14] Tsai, Wang (2020). *Deformable CNNs for Wafer Defect Pattern Detection*. DOI:10.1109/TSM.2020.2997342
- [47] Ioffe, Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training*. arXiv:1502.03167
- [48] Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. DOI:10.5555/2627435.2670313
- [51] Kyeong, Kim (2018). *Wafer Bin Map Defect Pattern Classification Using Convolutional Neural Network*. DOI:10.1109/TSM.2018.2841416
- [52] Cheon, Kim, Ham, Kim (2019). *Automatic Defect Classification for Semiconductor Manufacturing*. DOI:10.1109/TSM.2019.2941674
- [56] Jin, Kim, Kwon (2020). *Deep Learning Approaches for Wafer Map Defect Pattern Recognition*. DOI:10.1109/ACCESS.2020.2990379
- [59] Tsai, Lee (2020). *A Light-Weight CNN Model for Wafer Map Defect Detection*. DOI:10.1109/ACCESS.2020.3017358
- [63] Nakazawa, Kulkarni (2019). *Wafer Map Defect Pattern Classification and Image Retrieval Using CNN*. DOI:10.1109/ASMC.2019.8791815
- [64] Shin, Kim (2022). *Multi-Label Wafer Map Defect Pattern Classification Using Deep Learning*. DOI:10.1109/TSM.2022.3178464
- [77] Snell, Swersky, Zemel (2017). *Prototypical Networks for Few-shot Learning*. arXiv:1703.05175
- [86] Kipf, Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. arXiv:1609.02907
- [88] Hamilton, Ying, Leskovec (2017). *Inductive Representation Learning on Large Graphs*. arXiv:1706.02216
- [89] Park, Cho (2021). *Defect Detection Using Graph Neural Networks on Semiconductor Wafer Maps*. DOI:10.1109/TSM.2021.3117275
- [90] Xu, Hu, Leskovec, Jegelka (2019). *How Powerful Are Graph Neural Networks?*. arXiv:1810.00826
- [91] Lin, Dollar, Girshick, He, Hariharan, Belongie (2017). *Feature Pyramid Networks for Object Detection*. arXiv:1612.03144
- [92] Dai, Qi, Xiong, Li, Zhang, Hu, Wei (2017). *Deformable Convolutional Networks*. arXiv:1703.06211
- [94] Ronneberger, Fischer, Brox (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. arXiv:1505.04597
- [96] Misra, Shrivastava, Gupta, Hebert (2016). *Deep Multi-Task Learning with Cross Stitch Networks*. arXiv:1604.03539
- [97] Liu, Qi, Qin, Shi, Jia (2018). *PANet: Path Aggregation Network for Instance Segmentation*. arXiv:1803.01534
- [98] Wang, Sun, Liu, Sarma, Bronstein, Kitani (2020). *HRNet: Deep High-Resolution Representation Learning for Visual Recognition*. arXiv:1908.07919
- [99] Kirillov, Girshick, He, Dollar (2019). *Panoptic Feature Pyramid Networks*. arXiv:1901.02446
- [104] Dai, Liu, Le, Tan (2021). *CoAtNet: Marrying Convolution and Attention for All Data Sizes*. arXiv:2106.04803

### `src/models/ensemble.py`
- [30] Lakshminarayanan, Pritzel, Blundell (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. arXiv:1612.01474

### `src/models/pretrained.py`
- [1] He, Zhang, Ren, Sun (2016). *Deep Residual Learning for Image Recognition*. arXiv:1512.03385
- [2] Tan, Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. arXiv:1905.11946
- [56] Jin, Kim, Kwon (2020). *Deep Learning Approaches for Wafer Map Defect Pattern Recognition*. DOI:10.1109/ACCESS.2020.2990379
- [60] Shim, Jeon, Choi (2020). *Wafer Defect Pattern Recognition Using Transfer Learning*. DOI:10.1109/TSM.2020.3046888
- [74] Kang, Xie, Rohrbach, Yan, Gordo, Feng, Kalantidis (2020). *Decoupling Representation and Classifier for Long-Tailed Recognition*. arXiv:1910.09217
- [91] Lin, Dollar, Girshick, He, Hariharan, Belongie (2017). *Feature Pyramid Networks for Object Detection*. arXiv:1612.03144

### `src/models/vit.py`
- [3] Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition*. arXiv:2010.11929
- [23] Vaswani et al. (2017). *Attention Is All You Need*. arXiv:1706.03762
- [101] Touvron, Cord, Douze, Massa, Sablayrolles, Jegou (2021). *Training Data-Efficient Image Transformers & Distillation Through Attention*. arXiv:2012.12877
- [102] Liu, Lin, Cao, Hu, Wei, Zhang, Lin, Guo (2021). *Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows*. arXiv:2103.14030
- [103] Wu, Xu, Dai, Wan, Zhang, Yan, Tomizuka, Gonzalez, Keutzer, Vajda (2021). *CvT: Introducing Convolutions to Vision Transformers*. arXiv:2103.15808
- [104] Dai, Liu, Le, Tan (2021). *CoAtNet: Marrying Convolution and Attention for All Data Sizes*. arXiv:2106.04803
- [105] Yuan, Chen, Chen, Codella, Dai, Gao, Hu, Huang, Li, Li, Liu, Lu, Shi, Shu, Yuan, Zhu (2021). *Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet*. arXiv:2101.11986
- [106] Graham, El-Nouby, Touvron, Stock, Joulin, Jegou, Douze (2021). *LeViT: A Vision Transformer in ConvNet's Clothing for Faster Inference*. arXiv:2104.01136
- [107] Chen, Fan, Panda (2021). *CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification*. arXiv:2103.14899
- [108] Wang, Xie, Li, Fan, Song, Liang, Lu, Luo, Shao (2021). *Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction*. arXiv:2102.12122
- [109] Yu, Li, Jiang, Yu, Shi, Wang (2022). *PoolFormer: MetaFormer is Actually What You Need for Vision*. arXiv:2111.11418
- [110] Tolstikhin, Houlsby, Kolesnikov, Beyer, Zhai, Unterthiner, Yung, Steiner, Keysers, Uszkoreit, Lucic, Dosovitskiy (2021). *MLP-Mixer: An All-MLP Architecture for Vision*. arXiv:2105.01601

### `src/training/domain_adaptation.py`
- [40] Sun, Saenko (2016). *Deep CORAL: Correlation Alignment for Deep Domain Adaptation*. arXiv:1607.01719
- [41] Ganin et al. (2016). *Domain-Adversarial Training of Neural Networks*. arXiv:1505.07818

### `src/training/losses.py`
- [15] Lin, Goyal, Girshick, He, Dollar (2017). *Focal Loss for Dense Object Detection*. arXiv:1708.02002
- [16] Cui, Jia, Lin, Song, Belongie (2019). *Class-Balanced Loss Based on Effective Number of Samples*. arXiv:1901.05555
- [19] He, Garcia (2009). *Learning from Imbalanced Data*. DOI:10.1109/TKDE.2008.239
- [20] Muller, Kornblith, Hinton (2019). *When Does Label Smoothing Help?*. arXiv:1906.02629
- [73] Cui, Jia, Lin, Song, Belongie (2019). *Class-Balanced Loss Based on Effective Number of Samples*. arXiv:1901.05555
- [79] Deng, Guo, Xue, Zafeiriou (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. arXiv:1801.07698
- [80] Wang, Cheng, Gong, Zhu (2018). *CosFace: Large Margin Cosine Loss for Deep Face Recognition*. arXiv:1801.09414
- [81] Wen, Zhang, Li, Qiao (2016). *A Discriminative Feature Learning Approach for Deep Face Recognition*. DOI:10.1007/978-3-319-46478-7_31
- [83] Kim, Kim, Choi, You (2020). *Proxy Anchor Loss for Deep Metric Learning*. arXiv:2003.13911
- [84] Sun, Cheng, Zhang, Lin, Liu, Wang (2020). *Circle Loss: A Unified Perspective of Pair Similarity Optimization*. arXiv:2002.10857
- [131] Milletari, Navab, Ahmadi (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. arXiv:1606.04797
- [132] Salehi, Erdogmus, Gholipour (2017). *Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks*. arXiv:1706.05721

### `src/training/simclr.py`
- [11] Kang et al. (2021). *WaPIRL: Wafer Pattern Identification using Representation Learning*. DOI:10.1109/TSM.2021.3064435
- [37] Chen, Kornblith, Norouzi, Hinton (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. arXiv:2002.05709
- [38] Grill et al. (2020). *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning*. arXiv:2006.07733
- [39] He, Fan, Wu, Xie, Girshick (2020). *Momentum Contrast for Unsupervised Visual Representation Learning*. arXiv:1911.05722
- [53] Kahng, Kim (2020). *Semi-Supervised Learning for Wafer Map Defect Pattern Classification*. DOI:10.1109/TSM.2020.3017809
- [76] Khosla, Teterwak, Wang, Swersky, Tian, Isola, Maschinot, Liu, Kornblith (2020). *Supervised Contrastive Learning*. arXiv:2004.11362
- [78] Schroff, Kalenichenko, Philbin (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering*. arXiv:1503.03832
- [82] Kaya, Bilge (2019). *Deep Metric Learning: A Survey*. arXiv:1904.06626
- [85] Chen, He (2021). *Exploring Simple Siamese Representation Learning*. arXiv:2011.10566
- [111] Sohn, Berthelot, Li, Zhang, Carlini, Cubuk, Kurakin, Zhang, Raffel (2020). *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*. arXiv:2001.07685
- [114] Liu, Lee, Park, Kim, Yang, Hwang (2019). *Learning to Propagate Labels: Transductive Propagation Network*. arXiv:1805.10002
- [115] Berthelot, Carlini, Goodfellow, Papernot, Oliver, Raffel (2019). *MixMatch: A Holistic Approach to Semi-Supervised Learning*. arXiv:1905.02249
- [119] Laine, Aila (2017). *Temporal Ensembling for Semi-Supervised Learning*. arXiv:1610.02242
- [128] Tack, Yu, Jeong, Kim, Shin, Shin (2020). *CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances*. arXiv:2007.08176

### `src/training/trainer.py`
- [133] Loshchilov, Hutter (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. arXiv:1608.03983
- [134] Popel, Bojar (2018). *Training Tips for the Transformer Model*. arXiv:1804.00247
- [135] Goyal, Dollar, Girshick, Noordhuis, Wesolowski, Massa, Kirillov, He (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:1706.02677
- [136] Smith (2017). *Cyclical Learning Rates for Training Neural Networks*. arXiv:1506.01186

### `train.py`
- [10] Kim et al. (2020). *Deep Learning Based Wafer Map Defect Pattern Classification*. DOI:10.1109/ACCESS.2020.3040684
- [16] Cui, Jia, Lin, Song, Belongie (2019). *Class-Balanced Loss Based on Effective Number of Samples*. arXiv:1901.05555
- [17] Buda, Maki, Mazurowski (2018). *A Systematic Study of the Class Imbalance Problem in CNNs*. arXiv:1710.05381
- [45] Kingma, Ba (2015). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980
- [46] Loshchilov, Hutter (2019). *Decoupled Weight Decay Regularization*. arXiv:1711.05101
- [55] Moyne, Iskandar (2017). *Smart Semiconductor Manufacturing: An Overview*. DOI:10.1109/TSM.2017.2768062
- [64] Shin, Kim (2022). *Multi-Label Wafer Map Defect Pattern Classification Using Deep Learning*. DOI:10.1109/TSM.2022.3178464
- [74] Kang, Xie, Rohrbach, Yan, Gordo, Feng, Kalantidis (2020). *Decoupling Representation and Classifier for Long-Tailed Recognition*. arXiv:1910.09217
- [93] Ruder (2017). *An Overview of Multi-Task Learning in Deep Neural Networks*. arXiv:1706.05098
- [95] Sener, Koltun (2018). *Multi-Task Learning as Multi-Objective Optimization*. arXiv:1810.04650
- [100] Chen, Badrinarayanan, Lee, Rabinovich (2018). *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*. arXiv:1711.02257
- [112] Huisman, van Rijn, Plaat (2021). *A Survey of Deep Meta-Learning*. arXiv:2010.03522
- [113] Finn, Abbeel, Levine (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*. arXiv:1703.03400
- [117] Lee, Maji, Ravichandran, Soatto (2019). *Meta-Learning with Differentiable Convex Optimization*. arXiv:1904.03758
- [120] Lee (2013). *Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method*. arXiv:1908.02983
- [133] Loshchilov, Hutter (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. arXiv:1608.03983
- [135] Goyal, Dollar, Girshick, Noordhuis, Wesolowski, Massa, Kirillov, He (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:1706.02677
- [137] Zhang, Lucas, Hinton, Ba (2019). *Lookahead Optimizer: k Steps Forward, 1 Step Back*. arXiv:1907.08610
- [138] Liu, Jiang, He, Chen, Liu, Gao, Han (2020). *On the Variance of the Adaptive Learning Rate and Beyond*. arXiv:1908.03265
- [139] Foret, Kleiner, Mobahi, Neyshabur (2021). *Sharpness-Aware Minimization for Efficiently Improving Generalization*. arXiv:2010.01412
- [140] You, Li, Xu, He, Ginsburg, Hsieh (2020). *Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes*. arXiv:1904.00962
- [145] Parisi, Kemker, Part, Kanan, Wermter (2019). *Continual Lifelong Learning with Neural Networks: A Review*. arXiv:1802.07569
- [146] Li, Hoiem (2017). *Learning without Forgetting*. arXiv:1606.09282
- [150] Lu, Liu, Dong, Gu, Gama, Zhang (2018). *Concept Drift Adaptation by Exploiting Historical Knowledge*. arXiv:1810.02822

## Key Findings by Concept

