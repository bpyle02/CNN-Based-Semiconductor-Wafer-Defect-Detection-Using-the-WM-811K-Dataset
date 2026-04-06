# Parametric Contrastive Learning for Long-Tailed Recognition

**Authors**: Cui, Zhong, Liu, Yang, Belongie
**Year**: 2021
**arXiv**: 2109.01903
**Topic**: long_tail
**Relevance**: PaCo: extends SupCon with class-specific centers

---


--- Page 1 ---
Robust ﬁne-tuning of zero-shot models
Mitchell Wortsman∗†
Gabriel Ilharco∗†
Jong Wook Kim§
Mike Li‡
Simon Kornblith⋄
Rebecca Roelofs⋄
Raphael Gontijo-Lopes⋄
Hannaneh Hajishirzi†◦
Ali Farhadi⋆†
Hongseok Namkoong⋆‡
Ludwig Schmidt†△
Abstract
Large pre-trained models such as CLIP or ALIGN oﬀer consistent accuracy across a range of data
distributions when performing zero-shot inference (i.e., without ﬁne-tuning on a speciﬁc dataset). Although
existing ﬁne-tuning methods substantially improve accuracy on a given target distribution, they often
reduce robustness to distribution shifts. We address this tension by introducing a simple and eﬀective
method for improving robustness while ﬁne-tuning: ensembling the weights of the zero-shot and ﬁne-tuned
models (WiSE-FT). Compared to standard ﬁne-tuning, WiSE-FT provides large accuracy improvements
under distribution shift, while preserving high accuracy on the target distribution. On ImageNet and ﬁve
derived distribution shifts, WiSE-FT improves accuracy under distribution shift by 4 to 6 percentage
points (pp) over prior work while increasing ImageNet accuracy by 1.6 pp. WiSE-FT achieves similarly
large robustness gains (2 to 23 pp) on a diverse set of six further distribution shifts, and accuracy gains of
0.8 to 3.3 pp compared to standard ﬁne-tuning on seven commonly used transfer learning datasets. These
improvements come at no additional computational cost during ﬁne-tuning or inference.
1
Introduction
A foundational goal of machine learning is to develop models that work reliably across a broad range of
data distributions. Over the past few years, researchers have proposed a variety of distribution shifts on
which current algorithmic approaches to enhance robustness yield little to no gains [97, 70]. While these
negative results highlight the diﬃculty of learning robust models, large pre-trained models such as CLIP
[82], ALIGN [45] and BASIC [77] have recently demonstrated unprecedented robustness to these challenging
distribution shifts. The success of these models points towards pre-training on large, heterogeneous datasets
as a promising direction for increasing robustness. However, an important caveat is that these robustness
improvements are largest in the zero-shot setting, i.e., when the model performs inference without ﬁne-tuning
on a speciﬁc target distribution.
In a concrete application, a zero-shot model can be ﬁne-tuned on extra application-speciﬁc data, which often
yields large performance gains on the target distribution. However, in the experiments of Radford et al. [82]
and Pham et al. [77], ﬁne-tuning comes at the cost of robustness: across several natural distribution shifts,
the accuracy of their ﬁne-tuned models is lower than that of the original zero-shot model. This leads to a
natural question:
Can zero-shot models be ﬁne-tuned without reducing accuracy under distribution shift?
As pre-trained models are becoming a cornerstone of machine learning, techniques for ﬁne-tuning them on
downstream applications are increasingly important. Indeed, the question of robustly ﬁne-tuning pre-trained
∗⋆These authors contributed equally.
†University of Washington §OpenAI ‡Columbia University ⋄Google Research, Brain Team
◦Allen Institute for Artiﬁcial Intelligence △Toyota Research Institute
Code provided at https://github.com/mlfoundations/wise-ft.
1
arXiv:2109.01903v3  [cs.CV]  21 Jun 2022

--- Page 2 ---
Accuracy on the reference distribution (e.g., ImageNet)
Accuracy on the distribution shifts
Models trained on reference
distribution train set
Zero-shot CLIP models
Eﬀective
robustness
Fine-tuned
CLIP
Schematic: ﬁne-tuning CLIP on the reference distribution leads to
higher accuracy on the reference distribution but less robustness
Accuracy on the reference distribution (e.g., ImageNet)
Accuracy on the distribution shifts
Models trained on reference
distribution train set
Zero-shot CLIP models
■Weight-space ensemble for α ∈[0, 1]:
θα = (1 −α) · θzero-shot + α · θﬁne-tuned
θzero-shot
θﬁne-tuned
Schematic: our method, WiSE-FT leads to
better accuracy on the distribution shifts without
decreasing accuracy on the reference distribution
Varyi
nga
mix
in
ga
c
o
e
f
f
i
c
i
e
n
t
a
α
55
60
65
70
75
80
85
ImageNet (top-1, %)
30
35
40
45
50
55
60
65
70
75
Avg. accuracy on 5 distribution shifts
Real data: our method
75 76 77 78 79 80 81 82 83
84
85
86
87
ImageNet (top-1, %)
67
68
69
70
71
72
73
74
75
76
77
Avg. accuracy on 5 distribution shifts
+1.6 pp ImageNet
+4.5 pp Distribution shifts
Real data: our method (zoomed-in)
CLIP zero-shot models
Linear ﬁt (CLIP zero-shot models)
CLIP ﬁne-tuned end-to-end
CLIP ﬁne-tuned with a linear classiﬁer
(prior work)
Weight-space ensemble (end-to-end)
Weight-space ensemble (linear classiﬁer)
Weight-space ensemble with α = 0.5
Standard ImageNet models
Linear ﬁt (standard ImageNet models)
y = x
Figure 1: (Top left) Zero-shot CLIP models exhibit moderate accuracy on the reference distribution (x-axis,
the target for ﬁne-tuning) and high eﬀective robustness (accuracy on the distribution shifts beyond the baseline
models). In contrast, standard ﬁne-tuning—either end-to-end or with a linear classiﬁer (ﬁnal layer)—attains
higher accuracy on the reference distribution but less eﬀective robustness. (Top right) Our method linearly
interpolates between the zero-shot and ﬁne-tuned models with a mixing coeﬃcient α ∈[0, 1]. (Bottom) On
ﬁve distribution shifts derived from ImageNet (ImageNetV2, ImageNet-R, ImageNet Sketch, ObjectNet, and
ImageNet-A), WiSE-FT improves average accuracy relative to both the zero-shot and ﬁne-tuned models
while maintaining or improving accuracy on ImageNet.
models has recently also been raised as an open problem by several authors [3, 9, 82, 77]. Andreassen et al. [3]
explored several ﬁne-tuning approaches but found that none yielded models with improved robustness at high
accuracy. Furthermore, Taori et al. [97] demonstrated that no current algorithmic robustness interventions
provide consistent gains across the distribution shifts where zero-shot models excel.
In this paper, we conduct an empirical investigation to understand and improve ﬁne-tuning of zero-shot
models from a distributional robustness perspective. We begin by measuring how diﬀerent ﬁne-tuning
approaches (last-layer vs. end-to-end ﬁne-tuning, hyperparameter changes, etc.) aﬀect the accuracy under
distribution shift of the resulting ﬁne-tuned models. Our empirical analysis uncovers two key issues in the
standard ﬁne-tuning process. First, the robustness of ﬁne-tuned models varies substantially under even
small changes in hyperparameters, but the best hyperparameters cannot be inferred from accuracy on the
target distribution alone. Second, more aggressive ﬁne-tuning (e.g., using a larger learning rate) yields larger
accuracy improvements on the target distribution, but can also reduce accuracy under distribution shift by a
large amount.
Motivated by the above concerns, we propose a robust way of ﬁne-tuning zero-shot models that addresses the
aforementioned trade-oﬀand achieves the best of both worlds: increased performance under distribution shift
2

--- Page 3 ---
while maintaining or even improving accuracy on the target distribution relative to standard ﬁne-tuning. In
addition, our method simpliﬁes the choice of hyperparameters in the ﬁne-tuning process.
Our method (Figure 1) has two steps: ﬁrst, we ﬁne-tune the zero-shot model on the target distribution.
Second, we combine the original zero-shot and ﬁne-tuned models by linearly interpolating between their
weights, which we refer to as weight-space ensembling. Interpolating model parameters is a classical idea in
convex optimization dating back decades (e.g., see [84, 79]). Here, we empirically study model interpolation
for non-convex models from the perspective of distributional robustness. Interestingly, linear interpolation in
weight-space still succeeds despite the non-linearity in the activation functions of the neural networks.
Weight-space ensembles for ﬁne-tuning (WiSE-FT) substantially improve accuracy under distribution shift
compared to prior work while maintaining high performance on the target distribution. Concretely, on
ImageNet [17] and ﬁve of the natural distribution shifts studied by Radford et al. [82], WiSE-FT applied
to standard end-to-end ﬁne-tuning improves accuracy under distribution shift by 4 to 6 percentage points
(pp) over prior work while maintaining or improving the ImageNet accuracy of the ﬁne-tuned CLIP model.
Relative to the zero-shot model, WiSE-FT improves accuracy under distribution shift by 1 to 9 pp. Moreover,
WiSE-FT improves over a range of alternative approaches such as regularization and evaluating at various
points throughout ﬁne-tuning. These robustness gains come at no additional computational cost during
ﬁne-tuning or inference.
While our investigation centers around CLIP, we observe similar trends for other zero-shot models including
ALIGN [45], BASIC [77], and a ViT model pre-trained on JFT [21]. For instance, WiSE-FT improves the
ImageNet accuracy of a ﬁne-tuned BASIC-L model by 0.4 pp, while improving average accuracy under
distribution shift by 2 to 11 pp.
To understand the robustness gains of WiSE-FT, we ﬁrst study WiSE-FT when ﬁne-tuning a linear classiﬁer
(last layer) as it is more amenable to analysis. In this linear case, our procedure is equivalent to ensembling
the outputs of two models, and experiments point towards the complementarity of model predictions as a key
property. For end-to-end ﬁne-tuning, we connect our observations to earlier work on the phenomenology of
deep learning. Neyshabur et al. [73] found that end-to-end ﬁne-tuning the same model twice yielded two
diﬀerent solutions that were connected via a linear path in weight-space along which error remains low, known
as linear mode connectivity [25]. Our observations suggest a similar phenomenon along the path generated by
WiSE-FT, but the exact shape of the loss landscape and connection between error on the target and shifted
distributions are still open problems.
In addition to the aforementioned ImageNet distribution shifts, WiSE-FT consistently improves robustness
on a diverse set of six additional distribution shifts including: (i) geographic shifts in satellite imagery and
wildlife recognition (WILDS-FMoW, WILDS-iWildCam) [49, 13, 6], (ii) reproductions of the popular image
classiﬁcation dataset CIFAR-10 with a distribution shift (CIFAR-10.1 and CIFAR-10.2) [83, 62], and (iii)
datasets with distribution shift induced by temporal perturbations in videos (ImageNet-Vid-Robust and
YTBB-Robust) [88]. Beyond the robustness perspective, WiSE-FT also improves accuracy compared to
standard ﬁne-tuning, reducing the relative error rate by 4-49% on a range of seven datasets: ImageNet,
CIFAR-10, CIFAR-100 [54], Describable Textures [14], Food-101 [10], SUN397 [103], and Stanford Cars [53].
Even when ﬁne-tuning data is scarce, reﬂecting many application scenarios, we ﬁnd that WiSE-FT improves
performance.
Overall, WiSE-FT is simple, universally applicable in the problems we studied, and can be implemented in a
few lines of code. Hence we encourage its adoption for ﬁne-tuning zero-shot models.
2
Background and experimental setup
Our experiments compare the performance of zero-shot models, corresponding ﬁne-tuned models, and models
produced by WiSE-FT. To measure robustness, we contrast model accuracy on two related but diﬀerent
3

--- Page 4 ---
ImageNet (Deng et al.)
ImageNetV2 (Recht et al.)
ImageNet-R (Hendrycks et al.)
ImageNet Sketch (Wang et al.)
ObjectNet (Barbu et al.)
ImageNet-A (Hendrycks et al.)
Figure 2: Samples of the class lemon, from the reference distribution ImageNet [17] and the derived
distribution shifts considered in our main experiments: ImageNet-V2 [83], ImageNet-R [37], ImageNet Sketch
[100], ObjectNet [4], and ImageNet-A [38].
distributions, a reference distribution Dref which is the target for ﬁne-tuning, and shifted distribution Dshift.1
We assume both distributions have test sets for evaluation, and Dref has an associated training set Str
ref
which is typically used for training or ﬁne-tuning. The goal for a model is to achieve both high accuracy
and consistent performance on the two distributions Dref and Dshift. This is a natural goal as humans often
achieve similar accuracy across the distribution shifts in our study [89].
For a model f, we let Accref(f) and Accshift(f) refer to classiﬁcation accuracy on the reference and shifted
test sets, respectively. We consider k-way image classiﬁcation, where xi is an image with corresponding label
yi ∈{1, ..., k}. The outputs of f are k-dimensional vectors of non-normalized class scores.
Distribution shifts. Taori et al. [97] categorized distribution shifts into two broad categories: (i) synthetic,
e.g., ℓ∞-adversarial examples or artiﬁcial changes in image contrast, brightness, etc. [35, 8, 7, 29, 2]; and (ii)
natural, where samples are not perturbed after acquisition and changes in data distributions arise through
naturally occurring variations in lighting, geographic location, crowdsourcing process, image styles, etc.
[97, 83, 37, 38, 49]. Following Radford et al. [81], our focus here is on natural distribution shifts as they are
more representative of the real world when no active adversary is present. Speciﬁcally, we present our key
results for ﬁve natural distribution shifts derived from ImageNet (i.e., Str
ref is ImageNet):
• ImageNet-V2 (IN-V2) [83], a reproduction of the ImageNet test set with distribution shift
• ImageNet-R (IN-R) [37], renditions (e.g., sculptures, paintings) for 200 ImageNet classes
• ImageNet Sketch (IN-Sketch) [100], which contains sketches instead of natural images
• ObjectNet [4], a test set of objects in various scenes with 113 classes overlapping with ImageNet
• ImageNet-A (IN-A) [38], a test set of natural images misclassiﬁed by a ResNet-50 [34] for 200 ImageNet
classes.
Figure 2 illustrates the ﬁve distribution shifts.
Eﬀective robustness and scatter plots. To compare the robustness of models with diﬀerent accuracies on
the reference distribution, we follow the eﬀective robustness framework introduced by Taori et al. [97]. Eﬀective
robustness quantiﬁes robustness as accuracy beyond a baseline trained only on the reference distribution.
1Dref and Dshift are sometimes referred to as in-distribution (ID) and out-of-distribution (OOD). In this work, we include
evaluations of zero-shot models, which are not trained on data from the reference distribution, so referring to Dref would be
imprecise. For clarity, we avoid the ID/OOD terminology.
4

--- Page 5 ---
A useful tool for studying (eﬀective) robustness are scatter plots that illustrate model performance under
distribution shift [83, 97]. These scatter plots display accuracy on the reference distribution on the x-axis and
accuracy under distribution shift on the y-axis, i.e., a model f is shown as a point (Accref(f), Accshift(f)).
Figure 1 exempliﬁes these scatter plots with both schematics and real data. For the distribution shifts
we study, accuracy on the reference distribution is a reliable predictor of accuracy under distribution shift
[97, 70]. In other words, there exists a function β : [0, 1] →[0, 1] such that Accshift(f) approximately equals
β(Accref(f)) for models f trained on the train set Str
ref. Eﬀective robustness [97] is accuracy beyond this
baseline, deﬁned formally as ρ(f) = Accshift(f) −β(Accref(f)).
In the corresponding scatter plots, eﬀective robustness is vertical movement above expected accuracy under
distribution shift (Figure 1, top). Eﬀective robustness thereby disentangles accuracy changes on the reference
distribution from the eﬀect of robustness interventions. When we say that a model is robust to distribution
shift, we mean that eﬀective robustness is positive. Taori et al. [97] observed that no algorithmic robustness
intervention consistently achieves substantial eﬀective robustness across the distribution shifts in Figure 2—the
ﬁrst method to do so was zero-shot CLIP. Empirically, when applying logit (or probit) axis scaling, models
trained on the reference distribution approximately lie on a linear trend [97, 70]. As in Taori et al. [97], we
apply logit axis scaling and show 95% Clopper-Pearson conﬁdence intervals for the accuracies of select points.
Zero-shot models and CLIP. We primarily explore CLIP models [82], although we also investigate
other zero-shot models including ALIGN [45], BASIC [77] and a ViT model pre-trained on JFT [21].
Zero-shot models exhibit eﬀective robustness and lie on a qualitatively diﬀerent linear trend (Figure 1).
CLIP-like models are pre-trained using image-caption pairs from the web. Given a set of image-caption
pairs {(x1, s1)..., (xB, sB)}, CLIP-like models train an image-encoder g and text-encoder h such that the
similarity ⟨g(xi), h(si)⟩is maximized relative to unaligned pairs. CLIP-like models perform zero-shot k-way
classiﬁcation given an image x and class names C = {c1, ..., ck} by matching x with potential captions.
For instance, using caption si = “a photo of a {ci}” for each class i, the zero-shot model predicts the class
via arg maxj ⟨g(x), h(sj)⟩.2
Equivalently, one can construct Wzero-shot ∈Rd×k with columns h(sj) and
compute outputs f(x) = g(x)⊤Wzero-shot. Unless explicitly mentioned, our experiments use the CLIP model
ViT-L/14@336px, although all CLIP models are displayed in our scatter plots (additional details provided in
Appendix D.1).
3
Weight-space ensembles for ﬁne-tuning
This section describes and motivates our proposed method, WiSE-FT, which consists of two simple steps.
First, we ﬁne-tune the zero-shot model on application-speciﬁc data. Second, we combine the original zero-shot
and ﬁne-tuned models by linearly interpolating between their weights, also referred to as weight-space
ensembling. WiSE-FT can be implemented in a few lines of PyTorch, and we provide example code in
Appendix A.
The zero-shot model excels under distribution shift while standard ﬁne-tuning achieves high accuracy on
the reference distribution. Our motivation is to combine these two models into one that achieves the best of
both worlds. Weight-space ensembles are a natural choice as they ensemble without extra computational cost.
Moreover, previous work has suggested that interpolation in weight space may improve performance when
models share part of their optimization trajectory [43, 73].
Step 1: Standard ﬁne-tuning. As in Section 2, we let Str
ref denote the dataset used for ﬁne-tuning
and g denote the image encoder used by CLIP. We are now explicit in writing g(x, Venc) where x is an
input image and Venc are the parameters of the encoder g. Standard ﬁne-tuning considers the model
f(x, θ) = g (x, Venc)⊤Wclassiﬁer where Wclassiﬁer ∈Rd×k is the classiﬁcation head and θ = [Venc, Wclassiﬁer]
2For improved accuracy, the embedding of a few candidate captions are averaged, e.g., s(1)
i
=“a photo of a {ci}” and
s(2)
i
=“a picture of a {ci}” (referred to as prompt ensembling [82]).
5

--- Page 6 ---
are the parameters of f. We then solve arg minθ
nP
(xi,yi)∈Str
ref ℓ(f(xi, θ), yi) + λR(θ)
o
where ℓis the cross-
entropy loss and R is a regularization term (e.g., weight decay). We consider the two most common variants
of ﬁne-tuning: end-to-end, where all values of θ are modiﬁed, and ﬁne-tuning only a linear classiﬁer, where
Venc is ﬁxed at the value learned during pre-training. Appendices D.2 and D.3 provide additional details.
Step 2: Weight-space ensembling. For a mixing coeﬃcient α ∈[0, 1], we consider the weight-space
ensemble between the zero-shot model with parameters θ0 and the model obtained via standard ﬁne-tuning
with parameters θ1. The predictions of the weight-space ensemble wse are given by
wse(x, α) = f(x, (1 −α) · θ0 + α · θ1) ,
(1)
i.e., we use the element-wise weighted average of the zero-shot and ﬁned-tuned parameters. When ﬁne-tuning
only the linear classiﬁer, weight-space ensembling is equivalent to the traditional output-space ensemble
[20, 11, 26] (1 −α) · f(x, θ0) + α · f(x, θ1) since Equation 1 decomposes as (1 −α) · g(x, Venc)⊤Wzero-shot +
α · g(x, Venc)⊤Wclassiﬁer.
As neural networks are non-linear with respect to their parameters, ensembling all layers—as we do when
end-to-end ﬁne-tuning—typically fails, achieving no better accuracy than a randomly initialized neural
network [25]. However, as similarly observed by previous work where part of the optimization trajectory
is shared [43, 25, 73], we ﬁnd that the zero-shot and ﬁne-tuned models are connected by a linear path in
weight-space along which accuracy remains high (explored further in Section 5.2).
Remarkably, as we show in Section 4, WiSE-FT improves accuracy under distribution shift while maintaining
high performance on the reference distribution relative to ﬁne-tuned models. These improvements come
without any additional computational cost as a single set of weights is used.
4
Results
This section presents our key experimental ﬁndings. First, we show that WiSE-FT boosts the accuracy of a
ﬁne-tuned CLIP model on ﬁve ImageNet distribution shifts studied by Radford et al. [82], while maintaining
or improving ImageNet accuracy. Next, we present additional experiments, including more distribution shifts,
the eﬀect of hyperparameters, accuracy improvements on the reference distribution, and experiments in
the low-data regime. Finally, we demonstrate that our ﬁndings are more broadly applicable by exploring
WiSE-FT for BASIC [77], ALIGN [45], and a ViT-H/14 [21] model pre-trained on JFT-300M [93].
Main results: ImageNet and associated distribution shifts. As illustrated in Figure 1, when the
mixing coeﬃcient α varies from 0 to 1, wse(·, α) is able to simultaneously improve accuracy on both the
reference and shifted distributions. A breakdown for each dataset is shown in Appendix C.1. Table 1 presents
our main results on ImageNet and ﬁve derived distribution shifts. WiSE-FT (end-to-end, α=0.5) outperforms
numerous strong models in both average accuracy under distribution shift and the average accuracy on the
reference and shifted distributions. While future work may lead to more sophisticated strategies for choosing
the mixing coeﬃcient α, α=0.5 yields close to optimal performance across a range of experiments. Hence,
we recommend α=0.5 when no domain knowledge is available. Appendix B further explores the eﬀect of α.
Moreover, results for twelve additional backbones are shown in Appendix C.
Robustness on additional distribution shifts. Beyond the ﬁve distribution shifts derived from ImageNet,
WiSE-FT consistently improves robustness on a diverse set of further distributions shifts including geographic
shifts in satellite imagery and wildlife recognition (WILDS-FMoW [49, 13], WILDS-iWildCam [49, 6]),
reproductions of the popular image classiﬁcation dataset CIFAR-10 [54] with a distribution shift (CIFAR-10.1
[83] and CIFAR-10.2 [62]), and datasets with distribution shift induced by temporal perturbations in videos
(ImageNet-Vid-Robust and YTBB-Robust [89]). Concretely, WiSE-FT (α=0.5) improves performance under
distribution shift by 3.5, 6.2, 1.7, 2.1, 9.0 and 23.2 pp relative to the ﬁne-tuned solution while decreasing
performance on the reference distribution by at most 0.3 pp (accuracy on the reference distribution often
improves). In contrast to the ImageNet distribution shifts, the zero-shot model initially achieves less than
6

--- Page 7 ---
Distribution shifts
Avg
Avg
IN (reference)
IN-V2
IN-R
IN-Sketch
ObjectNet*
IN-A
shifts
ref., shifts
CLIP ViT-L/14@336px
Zero-shot [82]
76.2
70.1
88.9
60.2
70.0
77.2
73.3
74.8
Fine-tuned LC [82]
85.4
75.9
84.2
57.4
66.2
75.3
71.8
78.6
Zero-shot (PyTorch)
76.6
70.5
89.0
60.9
69.1
77.7
73.4
75.0
Fine-tuned LC (ours)
85.2
75.8
85.3
58.7
67.2
76.1
72.6
78.9
Fine-tuned E2E (ours)
86.2
76.8
79.8
57.9
63.3
65.4
68.6
77.4
WiSE-FT (ours)
LC, α=0.5
83.7
76.3
89.6
63.0
70.7
79.7
75.9
79.8
LC, optimal α
85.3
76.9
89.8
63.0
70.7
79.7
75.9
80.2
E2E, α=0.5
86.8
79.5
89.4
64.7
71.1
79.9
76.9
81.8
E2E, optimal α
87.1
79.5
90.3
65.0
72.1
81.0
77.4
81.9
Table 1: Accuracy of various methods on ImageNet and derived distribution shifts for CLIP ViT-L/14@336px
[82]. E2E: end-to-end; LC: linear classiﬁer. Avg shifts displays the mean performance among the ﬁve
distribution shifts, while Avg reference, shifts shows the average of ImageNet (reference) and Avg shifts. For
optimal α, we choose the single mixing coeﬃcient that maximizes the column. Results for additional models
are provided in Appendix C.7.
30% accuracy on the WILDS distribution shifts, and WiSE-FT provides improvements regardless. Appendix
C.2 (Figure 9 and Table 6) includes more detailed results.
Hyperparameter variation and alternatives. As illustrated by Figure 3, moderate changes in standard
hyperparameters such as the learning rate or the number of epochs can substantially aﬀect performance
under distribution shift. Moreover, these performance diﬀerences cannot be detected reliably from model
performance on reference data alone. For instance, while training for 10 epochs with learning rate 3 · 10−5
and 3 · 10−6 lead to a small accuracy diﬀerence on ImageNet (0.3 pp), accuracy under distribution shift varies
by as much as 8 pp.
Furthermore, tuning hyperparameters on ImageNet data can also reduce robustness. For instance, while
moving from small to moderate learning rates (10−7 to 3 · 10−5) improves performance on ImageNet by 5 pp,
it also deteriorates accuracy under distribution shift by 8 pp.
WiSE-FT addresses this brittleness of hyperparameter tuning: even when using a learning rate 3 · 10−5 where
standard ﬁne-tuning leads to low robustness, applying WiSE-FT removes the trade-oﬀbetween accuracy
on the reference and shifted distributions. The models which can be achieved by varying α are as good or
better than those achievable by other hyperparameter conﬁgurations. Then, instead of searching over a wide
range of hyperparameters, only α needs to be considered. Moreover, evaluating diﬀerent values of α does not
require training new models.
There is no hyperparameter in Figure 3 which can be varied to match or exceed the optimal curve produced
by WiSE-FT. In our experiments, this frontier is reached only through methods that average model weights,
either using WiSE-FT or with a more sophisticated averaging scheme: keeping an exponential moving average
of all model iterates (EMA, [95]). Comparisons with EMA are detailed in Appendix C.3.2.
Additional comparisons are also presented in Appendix C.3, including distillation, additional regularization,
and CoOp [112]. Finally, Appendix C.4 recreates Figure 3 with stronger data augmentation and ﬁnds similar
trends.
Accuracy gains on reference distributions. Beyond robustness to distribution shift, Table 2 demon-
strates that WiSE-FT also improves accuracy after ﬁne-tuning on seven datasets. When ﬁne-tuning end-to-end
∗Although this table considers ImageNet class names, ObjectNet provides alternative class names which can improve the
performance of zero-shot CLIP by 2.3 percentage points (Appendix D.4).
7

--- Page 8 ---
65
70
75
80
ImageNet (top-1, %)
50
55
60
65
Avg. accuracy on 5 distribution shifts
2
epochs
4
10
Hyperparameter:
Fix learning rate, vary number of epochs
LR = 1 · 10−7
LR = 1 · 10−6
LR = 3 · 10−6
LR = 1 · 10−5
LR = 2 · 10−5
LR = 3 · 10−5
65
70
75
80
ImageNet (top-1, %)
50
55
60
65
Avg. accuracy on 5 distribution shifts
1e-07
learning rate
1e-06
3e-06
1e-05
2e-05
3e-05
Hyperparameter:
Fix number of epochs, vary learning rate
Epochs = 2
Epochs = 4
Epochs = 10
65
70
75
80
ImageNet (top-1, %)
50
55
60
65
Avg. accuracy on 5 distribution shifts
Varying LR, number of epochs,
and regularization coeﬃcient.
Hyperparameter: optimizer and regularization
AdamW
SGD
Adam no decay
Adam regularize to zero-shot
65
70
75
80
ImageNet (top-1, %)
50
55
60
65
Avg. accuracy on 5 distribution shifts
Iteration 250
1000
2500
Hyperparameter: terminating training early
Evaluation along optimization trajectory
Hyperparameter conﬁg (completed training)
Select early termination solutions
65
70
75
80
ImageNet (top-1, %)
50
55
60
65
Avg. accuracy on 5 distribution shifts
Weight-space ensembles (varied hyperparameters)
Weight-space ensemble
Hyperparameter conﬁg
CLIP zero-shot models
Linear ﬁt (CLIP zero-shot models)
Weight-space ensemble (end-to-end)
Figure 3: The robustness of ﬁne-tuned models varies substantially under even small changes in hyperparameters.
Applying WiSE-FT addresses this brittleness and can remove the trade-oﬀbetween accuracy on the reference
and shifted distributions. Results shown for CLIP ViT-B/16 ﬁne-tuned with cosine-annealing learning rate
schedule and all models in the top left and top middle plots are ﬁne-tuned with AdamW [61]. Moreover,
regularize to zero-shot appends the regularizer λ∥θ −θ0∥2
2 to the ﬁne-tuning objective, where θ0 are the
parameters of the zero-shot model.
on ImageNet, CIFAR-10, CIFAR-100, Describable Textures, Food-101, SUN397, and Stanford Cars, WiSE-FT
reduces relative error by 4 to 49%. Even though standard ﬁne-tuning directly optimizes for high accuracy
on the reference distribution, WiSE-FT achieves better performance. Appendix C.5 includes more details,
including explorations in the low-data regime.
Beyond CLIP. Figure 4 illustrates that WiSE-FT is generally applicable to zero-shot models beyond CLIP,
and beyond models pre-trained contrastively with image-text pairs. First, we interpolate between the weights
of the zero-shot and ﬁne-tuned BASIC-L model [77], ﬁnding that α=0.5 improves average accuracy on ﬁve
distribution shifts derived from ImageNet by over 7 pp while improving ImageNet accuracy by 0.4 pp relative
to the ﬁne-tuned BASIC-L model (a per-dataset breakdown is provided in Figure 24 and Table 12 of the
Appendix). As in Pham et al. [77], the model is ﬁne-tuned using a contrastive loss and half of the ImageNet
training data. WiSE-FT provides improvements on both reference and shifted distributions, despite these
experimental diﬀerences.
Next, we consider the application of WiSE-FT to a ViT-H/14 model [21] pre-trained on JFT-300M [93],
where the zero-shot classiﬁer is constructed by manually identifying a class correspondence (details provided
in Section C.7.2). WiSE-FT improves performance under distribution shift over both the zero-shot and
ﬁne-tuned models. When α=0.8, WiSE-FT outperforms the ﬁne-tuned model by 2.2 pp on distribution shifts,
while maintaining ImageNet performance within 0.2 pp of the ﬁne-tuned model. This result demonstrates that
WiSE-FT can be successfully applied even to models which do not use contrastive image-text pre-training.
8

--- Page 9 ---
ImageNet
CIFAR10
CIFAR100
Cars
DTD
SUN397
Food101
Standard ﬁne-tuning
86.2
98.6
92.2
91.6
81.9
80.7
94.4
WiSE-FT (α=0.5)
86.8 (+0.6)
99.3 (+0.7)
93.3 (+1.1)
93.3 (+1.7) 84.6 (+2.8) 83.2 (+2.5) 96.1 (+1.6)
WiSE-FT (opt. α)
87.1 (+0.9)
99.5 (+0.8)
93.4 (+1.2)
93.6 (+2.0) 85.2 (+3.3) 83.3 (+2.6) 96.2 (+1.8)
Table 2: Beyond robustness, WiSE-FT can improve accuracy after ﬁne-tuning on several datasets.
85
86
87
88
ImageNet (top-1, %)
76
78
80
82
84
Avg. accuracy on 5 distribution shifts
BASIC-L
70
75
80
85
ImageNet (top-1, %)
64
66
68
70
72
Avg. accuracy on 5 distribution shifts
ViT-H/14 (JFT)
75
80
85
ImageNet (top-1, %)
74
75
76
77
78
79
Avg. accuracy on 5 distribution shifts
ALIGN
Weight-space ensemble (end-to-end)
Weight-space ensemble with α = 0.5
BASIC-L zero-shot
BASIC-L ﬁne-tuned end-to-end
ViT-H/14 (JFT) zero-shot
ViT-H/14 (JFT) ﬁne-tuned end-to-end
ALIGN zero-shot
ALIGN ﬁne-tuned end-to-end
Figure 4: WiSE-FT applied to BASIC-L [77], a ViT-H/14 [21] model pre-trained on JFT-300M [93] and
ALIGN [45].
Finally, we apply WiSE-FT to the ALIGN model of Jia et al. [45], which is similar to CLIP but is pre-trained
with a diﬀerent dataset, ﬁnding similar trends.
5
Discussion
This section further analyzes the empirical phenomena we have observed so far. We begin with the case where
only the ﬁnal linear layer is ﬁne-tuned and predictions from the weight-space ensemble can be factored into
the outputs of the zero-shot and ﬁne-tuned model. Next, we connect our observations regarding end-to-end
ﬁne-tuning with earlier work on the phenomenology of deep learning.
5.1
Zero-shot and ﬁne-tuned models are complementary
In this section, we ﬁnd that the zero-shot and ﬁne-tuned models have diverse predictions, both on reference and
shifted distributions. Moreover, while the ﬁne-tuned models are more conﬁdent on the reference distribution,
the reverse is true under distribution shift.
Zero-shot and ﬁne-tuned models are diverse.
In certain cases, ensemble accuracy is correlated with
diversity among the constituents [57, 30]. If two models make coincident mistakes, so will their ensemble,
and no beneﬁt will be gained from combining them. Here, we explore two measures of diversity: prediction
diversity, which measures the fraction of examples for which two classiﬁers disagree but one is correct; and
Centered Kernel Alignment Complement, the complement of CKA [51]. Additional diversity measures and
details are provided in Appendix E. In Figure 5 (left), we show that the zero-shot and ﬁne-tuned models are
diverse both on the reference and shifted distributions, despite sharing the same backbone. As a point of
comparison, we include avg. diversity measures between two linear classiﬁers ﬁne-tuned with random splits
on half of ImageNet,3 denoted in orange in Figure 5.
3Two linear classiﬁers ﬁne-tuned on the same data converge to similar solutions, resulting in negligible diversity. As a stronger
baseline, we ﬁne-tune classiﬁers on diﬀerent subsets of ImageNet, with half of the data.
9

--- Page 10 ---
ImageNet
ImageNetV2
ObjectNet
ImageNet Sketch
ImageNet-A
ImageNet-R
0.00
0.05
0.10
0.15
Prediction Diversity
Div. between zero-shot and linear classiﬁer
Div. between two linear classiﬁers
ImageNet
ImageNetV2
ObjectNet
ImageNet Sketch
ImageNet-A
ImageNet-R
0.0
0.2
0.4
0.6
CKA Complement
0.4
0.5
Zero-shot overrides
0.4
0.5
Zero-shot is overriden
ImageNet
ImageNetV2
ObjectNet
ImageNet Sketch
ImageNet-A
ImageNet-R
ImageNet
ImageNetV2
ObjectNet
ImageNet Sketch
ImageNet-A
ImageNet-R
0.0
0.5
1.0
δzero-shot −δlinear
Figure 5: (Left) Zero-shot and ﬁne-tuned models exhibit diversity in their predictions. (Middle) On most
distribution shifts, the zero-shot model overrides the linear classiﬁer more than it is overridden. The reverse
is true for ImageNet (reference). (Right) Similarly, zero-shot models are more conﬁdent under distribution
shift, while the reverse is true on the reference distribution. The margin δf measures the average diﬀerence
between the largest and second largest unormalized output for classiﬁer f
Models are more conﬁdent where they excel. In order for the ensemble model to be eﬀective, it should
leverage each model’s expertise based on which distribution the data is from. Here, we empirically show
that this occurs on a number of datasets we consider. First, we examine the cases where the models being
ensembled disagree. We say the zero-shot model overrides the ﬁne-tuned model if their predictions disagree
and the zero-shot prediction matches that of the weight-space ensemble. Similarly, if models disagree and the
linear classiﬁer prediction matches the ensemble, we say the zero-shot is overridden. Figure 5 (middle) shows
the fraction of samples where the zero-shot model overrides and is overridden by the ﬁne-tuned linear classiﬁer
for α=0.5. Other than ImageNetV2, which was collected to closely reproduce ImageNet, the zero-shot model
overrides the linear classiﬁer more than it is overridden on the distribution shifts.
Additionally, we are interested in measuring model conﬁdence. Recall that we are ensembling quantities
before a softmax is applied, so we avoid criteria that use probability vectors, e.g., Guo et al. [33]. Instead, we
consider the margin δ between the largest and second largest output of each classiﬁer. Figure 5 (right) shows
that the zero-shot model is more conﬁdent in its predictions under distribution shift, while the reverse is true
on the reference distribution.
5.2
An error landscape perspective
We now turn to empirical phenomena we observe when weight-space ensembling all layers in the network.
Speciﬁcally, this section formalizes our observations and details related phenomena. Recall that the weight-
space ensemble of θ0 and θ1 is given by f(x, (1 −α) · θ0 + α · θ1) (Equation 1).
For a distribution D and model f, let AccD,f(θ) denote the expected accuracy of f evaluated with parameters
θ on distribution D.
Observation 1: As illustrated in Figure 6, on ImageNet and the ﬁve associated distribution shifts we
consider
AccD,f((1 −α) · θ0 + α · θ1) ≥(1 −α) · AccD,f(θ0) + α · AccD,f(θ1)
(2)
for all α ∈[0, 1].
Note that equation 2 uses the baseline of linearly interpolating between the accuracies of the two endpoints,
which is always achievable by using weights θ1 with probability α and using model θ0 otherwise. In the case
where the accuracy of both endpoints are similar, Equation 2 is equivalent to the deﬁnition of Linear Mode
Connectivity of Frankle et al. [25].
To assist in contextualizing Observation 1, we review related phenomena. Neural networks are nonlinear,
hence weight-space ensembles only achieve good performance in exceptional cases—interpolating the weights
of two networks trained from a random initialization results in no better accuracy than a random classiﬁer
10

--- Page 11 ---
0.0
0.25
0.5
0.75
1.0
α
76
78
80
82
84
86
Accuracy (top-1, %)
ImageNet
0.0
0.25
0.5
0.75
1.0
α
70
72
74
76
78
80
ImageNetV2
0.0
0.25
0.5
0.75
1.0
α
78
80
82
84
86
88
90
92
ImageNet-R
0.0
0.25
0.5
0.75
1.0
α
56
58
60
62
64
66
ImageNet Sketch
0.0
0.25
0.5
0.75
1.0
α
62
64
66
68
70
72
ObjectNet
0.0
0.25
0.5
0.75
1.0
α
64
66
68
70
72
74
76
78
80
ImageNet-A
Acc ((1 −α) · θ0 + α · θ1) [Linearly interpolating the weights]
(1 −α) · Acc (θ0) + α · Acc (θ1) [Linear endpoint accuracy interpolation baseline]
max {Acc (θ0) , Acc (θ1)}
Figure 6: On ImageNet and the main distribution shifts we consider, linearly interpolating between the
weights of θ0 and θ1 exceeds the baseline of linearly interpolating the accuracies of the two models for all
α (Observation 1). Moreover, there exists an α for which WiSE-FT outperforms both the zero-shot and
ﬁne-tuned models (Observation 2).
[25]. Linear mode connectivity has been observed by Frankle et al. [25]; Izmailov et al. [43] when part
of the training trajectory is shared, and by Neyshabur et al. [73] when two models are ﬁne-tuned with a
shared initialization. In particular, the observations of Neyshabur et al. [73] may elucidate why weight-space
ensembles attain high accuracy in the setting we consider, as they suggest that ﬁne-tuning remains in a region
where solutions are connected by a linear path along which error remains low. Instead of considering the
weight-space ensemble of two ﬁne-tuned models, we consider the weight-space ensemble of the pre-trained and
ﬁne-tuned models. This is only possible for a pre-trained model capable of zero-shot inference such as CLIP.
Observation 2: As illustrated by Figure 6, on ImageNet and the ﬁve associated distribution shifts we
consider, weight-space ensembling (end-to-end) may outperform both the zero-shot and ﬁne-tuned models,
i.e., there exists an α for which AccD,f ((1 −α) · θ0 + α · θ1) ≥max {AccD,f (θ0) , AccD,f (θ1)}.
We are not the ﬁrst to observe that when interpolating between models, the accuracy of models along the path
may exceed that of either endpoint [43, 73, 102]. Neyshabur et al. [73] conjecture that interpolation could
produce solutions closer to the true center of a basin. In contrast to Neyshabur et al. [73], we interpolate
between models which observe diﬀerent data.
6
Related work
Robustness. Understanding how models perform under distribution shift remains an important goal, as
real world models may encounter data from new environments [80, 98]. Previous work has studied model
behavior under synthetic [35, 99, 65, 29, 23, 2] and natural distribution shift [37, 49, 100, 4, 38]. Interventions
used for synthetic shifts do not typically provide robustness to many natural distribution shifts [97]. In
contrast, accuracy on the reference distribution is often a reliable predictor for accuracy under distribution
shift [106, 69, 97, 94, 70].
On the other hand, D’Amour et al. [16] show that accuracy under certain
distribution shifts cannot be reliably inferred from accuracy on the reference distribution. We observe a
similar phenomenon when ﬁne-tuning with diﬀerent hyperparameters (Section 4, Figure 3).
Pre-training and transfer learning. Pre-training on large amounts of data is a powerful technique for
building high-performing machine learning systems [90, 21, 50, 107, 81, 12]. One increasingly popular class
of vision models are those pre-trained with auxiliary language supervision, which can be used for zero-shot
inference [18, 86, 111, 82, 45, 77, 109]. When pre-trained models are adapted to a speciﬁc distribution through
standard ﬁne-tuning, eﬀective robustness deteriorates at convergence [3]. In natural language processing,
previous work proposed stable ﬁne-tuning methods that incur computational overhead [46, 113], alleviating
problems such as representational collapse [1]. More generally, a variety of methods have attempted to
mitigate catastrophic forgetting [67]. Kirkpatrick et al. [48]; Zenke et al. [108] explored weighted quadratic
11

--- Page 12 ---
regularization for sequential learning. Xuhong et al. [105] showed that, for ﬁne-tuning, the simple quadratic
regularization explored in Section 4 performs best, while Lubana et al. [63] explored the connection between
quadratic regularization and interpolation. Andreassen et al. [3] found that many approaches from continual
learning do not provide robustness to multiple natural distribution shifts. Finally, Li et al. [59] investigate
the eﬀect of ﬁne-tuning hyperparameters on performance.
Traditional (output-space) ensembles. Traditional ensemble methods, which we refer to as output-space
ensembles, combine the predictions (outputs) of many classiﬁers [20, 5, 11, 27, 58, 26]. Typically, output-space
ensembles outperform individual classiﬁers and provide uncertainty estimates under distribution shift that
are more callibrated than baselines [58, 75, 92]. In contrast to these works, we consider the ensemble of two
models which have observed diﬀerent data. Output-space ensembles require more computational resources as
they require a separate pass through each model. Compared to an ensemble of 15 models trained on the
same dataset, Mustafa et al. [72] ﬁnd an improvement of 0.8–1.6 pp under distribution shift (on ImageNetV2,
ImageNet-R, ObjectNet, and ImageNet-A) by ensembling a similar number of models pre-trained on diﬀerent
datasets. In contrast, we see an improvement of 2–15 pp from ensembling two models. Moreover, as we
ensemble in weight-space, no extra compute is required compared to a single model.
Weight-space ensembles. Weight-space ensembles linearly interpolate between the weights of diﬀerent
models [25, 64, 32, 95]. For example, Izmailov et al. [43] average checkpoints saved throughout training for
improved performance. Indeed, averaging the weights along the training trajectory is a central method in
optimization [84, 78, 74]. For instance, Zhang et al. [110] propose optimizing with a set of fast and slow
weights, where every k steps, these two sets of weights are averaged and a new trajectory begins. Here, we
revisit these techniques from a distributional robustness perspective and consider the weight-space ensemble
of models which have observed diﬀerent data.
Concurrent and subsequent work. Topics including robust ﬁne-tuning, ensembles for improved robust-
ness, and interpolating the weights of ﬁne-tuned models are studied in concurrent and subsequent work.
Kumar et al. [55] observe that ﬁne-tuning end-to-end often results in higher accuracy on the reference
distribution but lower accuracy under distribution shift, compared to linear classiﬁer ﬁne-tuning. To address
this, Kumar et al. [55] ﬁrst ﬁne-tune a linear classiﬁer and use this as the initialization for end-to-end
ﬁne-tuning. We consider ﬁne-tuning zero-shot models, and so we begin with a classiﬁer (i.e., the zero-shot
classiﬁer) which we are using as the initialization for end-to-end ﬁne-tuning. In a separate work, Kumar et al.
[56] ﬁnd that calibrated output-space ensembles can be used to mitigate accuracy trade-oﬀs. In Figures 10
and 25 of the Appendix, we observe that it is possible to mitigate accuracy trade-oﬀs with output-space
ensembles even without calibration.
Hewitt et al. [40] explore the application of output-space ensembles and distillation to mitigate accuracy
trade-oﬀs which arise in ﬁne-tuning models for natural language generation. Hewitt et al. [40] observe
that output-space ensembles mainly outperform distillation, which we observe for a separate domain in
Figure 13 of the Appendix. Gontijo-Lopes et al. [31] explore output-space ensembles of models across
hyper-parameters, architectures, frameworks, and datasets. They ﬁnd that specializing in subdomains of data
leads to high ensemble performance. Finally, Matena and Raﬀel [66] introduce a method of combining models
in weight-space that goes beyond linear interpolation with a single mixing-coeﬃcient as employed in WiSE-FT.
Speciﬁcally, Matena and Raﬀel [66] employ Fisher information as a measure of per-parameter importance.
While their experiments do not examine accuracy under distribution shift, their goal of combining diﬀering
expertise into one shared model is well aligned with ours.
7
Limitations, impact, and conclusion
Limitations. While we expect our ﬁndings to be more broadly applicable to other domains such as natural
language processing, our investigation here is limited to image classiﬁcation. Exploring ﬁne-tuning for object
detection and natural language processing are interesting directions for future work. Moreover, although the
12

--- Page 13 ---
interpolation parameter setting α=0.5 provides good overall performance, we leave the question of ﬁnding
the optimal α for speciﬁc target distributions to future work.
Impact. Radford et al. [82] and Brown et al. [12] extensively discuss the broader impact of large zero-shot
models and identify potential causes of harm including model biases and potential malicious uses such as
surveillance systems. WiSE-FT is a ﬁne-tuning method that builds on such models, and thus may perpetuate
their negative impact.
Conclusion. WiSE-FT can substantially improve performance under distribution shift with minimal or no
loss in accuracy on the target distribution compared to standard ﬁne-tuning. We view WiSE-FT as a ﬁrst
step towards more sophisticated ﬁne-tuning schemes and anticipate that future work will continue to leverage
the robustness of zero-shot models for building more reliable neural networks.
Acknowledgements
We thank Anders Andreassen, Tim Dettmers, Jesse Dodge, Katie Everett, Samir Gadre, Ari Holtzman, Sewon
Min, Mohammad Norouzi, Nam Pho, Ben Poole, Sarah Pratt, Alec Radford, Jon Shlens, and Rohan Taori
for helpful discussions and draft feedback, Hyak at UW for computing support, Rosanne Liu for fostering the
collaboration, and Basil Mustafa for providing an earlier version of the mapping between JFT and ImageNet
classes. This work is in part supported by NSF IIS 1652052, IIS 17303166, DARPA N66001-19-2-4031,
DARPA W911NF-15-1-0543 and gifts from Allen Institute for Artiﬁcial Intelligence.
References
[1] Armen Aghajanyan, Akshat Shrivastava, Anchit Gupta, Naman Goyal, Luke Zettlemoyer, and Sonal Gupta.
Better ﬁne-tuning by reducing representational collapse. In International Conference on Learning Representations
(ICLR), 2021. https://openreview.net/forum?id=OQ08SN70M1V.
[2] Michael A Alcorn, Qi Li, Zhitao Gong, Chengfei Wang, Long Mai, Wei-Shinn Ku, and Anh Nguyen. Strike
(with) a pose: Neural networks are easily fooled by strange poses of familiar objects. In Conference on Computer
Vision and Pattern Recognition (CVPR), 2019. https://arxiv.org/abs/1811.11553.
[3] Anders Andreassen, Yasaman Bahri, Behnam Neyshabur, and Rebecca Roelofs.
The evolution of out-of-
distribution robustness throughout ﬁne-tuning, 2021. https://arxiv.org/abs/2106.15831.
[4] Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum,
and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition
models. In Advances in Neural Information Processing Systems (NeurIPS), 2019. URL https://proceedings.
neurips.cc/paper/2019/file/97af07a14cacba681feacf3012730892-Paper.pdf.
[5] Eric Bauer and Ron Kohavi. An empirical comparison of voting classiﬁcation algorithms: Bagging, boosting,
and variants. Machine learning, 1999. https://link.springer.com/article/10.1023/A:1007515423169.
[6] Sara Beery, Arushi Agarwal, Elijah Cole, and Vighnesh Birodkar. The iwildcam 2021 competition dataset. In
Conference on Computer Vision and Pattern Recognition (CVPR) FGVC8 Workshop, 2021. https://arxiv.
org/abs/2105.03494.
[7] Battista Biggio and Fabio Roli. Wild patterns: Ten years after the rise of adversarial machine learning. Pattern
Recognition, 2018. https://arxiv.org/abs/1712.03141.
[8] Battista Biggio, Igino Corona, Davide Maiorca, Blaine Nelson, Nedim ˇSrndi´c, Pavel Laskov, Giorgio Giacinto,
and Fabio Roli. Evasion attacks against machine learning at test time. In Joint European conference on machine
learning and knowledge discovery in databases, 2013. https://arxiv.org/abs/1708.06131.
[9] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S
Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al.
On the opportunities and risks of
foundation models, 2021. https://arxiv.org/abs/2108.07258.
13

--- Page 14 ---
[10] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with
random forests. In European Conference on Computer Vision (ECCV), 2014. https://data.vision.ee.ethz.
ch/cvl/datasets_extra/food-101/.
[11] Leo Breiman. Bagging predictors. Machine learning, 1996. https://link.springer.com/article/10.1007/
BF00058655.
[12] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, et al.
Language models are few-shot learners. In Advances in Neural Information Processing Systems (NeurIPS), 2020.
https://arxiv.org/abs/2005.14165.
[13] Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional map of the world. In Conference
on Computer Vision and Pattern Recognition (CVPR), 2018. https://arxiv.org/abs/1711.07846.
[14] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi.
Describing
textures in the wild. In Conference on Computer Vision and Pattern Recognition (CVPR), 2014. https:
//arxiv.org/abs/1311.3618.
[15] Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certiﬁed adversarial robustness via randomized smoothing. In
International Conference on Machine Learning (ICML), 2019. https://arxiv.org/abs/1902.02918.
[16] Alexander D’Amour, Katherine Heller, Dan Moldovan, Ben Adlam, Babak Alipanahi, Alex Beutel, Christina
Chen, Jonathan Deaton, Jacob Eisenstein, Matthew D Hoﬀman, et al. Underspeciﬁcation presents challenges
for credibility in modern machine learning, 2020. https://arxiv.org/abs/2011.03395.
[17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical
image database. In Conference on Computer Vision and Pattern Recognition, 2009. https://ieeexplore.ieee.
org/document/5206848.
[18] Karan Desai and Justin Johnson. Virtex: Learning visual representations from textual annotations. In Conference
on Computer Vision and Pattern Recognition (CVPR), 2021. https://arxiv.org/abs/2006.06666.
[19] Terrance DeVries and Graham W Taylor. Improved regularization of convolutional neural networks with cutout,
2017. https://arxiv.org/abs/1708.04552.
[20] Thomas G Dietterich. Ensemble methods in machine learning. In International workshop on multiple classiﬁer
systems, 2000. https://link.springer.com/chapter/10.1007/3-540-45014-9_1.
[21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An
image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on
Learning Representations (ICLR), 2021. https://arxiv.org/abs/2010.11929.
[22] Logan Engstrom, Brandon Tran, Dimitris Tsipras, Ludwig Schmidt, and Aleksander Madry. Exploring the
landscape of spatial robustness. In International Conference on Machine Learning (ICML), 2019. https:
//arxiv.org/abs/1712.02779.
[23] Kevin Eykholt, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Chaowei Xiao, Atul Prakash, Tadayoshi
Kohno, and Dawn Song. Robust physical-world attacks on deep learning visual classiﬁcation. In Conference on
Computer Vision and Pattern Recognition (CVPR), 2018. https://arxiv.org/abs/1707.08945.
[24] Stanislav Fort, Gintare Karolina Dziugaite, Mansheej Paul, Sepideh Kharaghani, Daniel M Roy, and Surya
Ganguli. Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time
evolution of the neural tangent kernel. In Advances in Neural Information Processing Systems (NeurIPS), 2020.
https://arxiv.org/abs/2010.15110.
[25] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity
and the lottery ticket hypothesis. In International Conference on Machine Learning (ICML), 2020. https:
//arxiv.org/abs/1912.05671.
14

--- Page 15 ---
[26] Yoav Freund and Robert E Schapire. A decision-theoretic generalization of on-line learning and an application
to boosting. Journal of Computer and System Sciences, 1997. https://www.sciencedirect.com/science/
article/pii/S002200009791504X.
[27] Jerome Friedman, Trevor Hastie, Robert Tibshirani, et al. The elements of statistical learning. Springer series
in statistics New York, 2001.
[28] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A Wichmann, and Wieland Brendel.
Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In
International Conference on Learning Representations (ICLR), 2018. https://arxiv.org/abs/1811.12231.
[29] Robert Geirhos, Carlos R Medina Temme, Jonas Rauber, Heiko H Sch¨utt, Matthias Bethge, and Felix A
Wichmann. Generalisation in humans and deep neural networks. In Advances in Neural Information Processing
Systems (NeurIPS), 2018. https://arxiv.org/abs/1808.08750.
[30] Raphael Gontijo-Lopes, Yann Dauphin, and Ekin D Cubuk. No one representation to rule them all: Overlapping
features of training methods, 2021. https://arxiv.org/abs/2007.01434.
[31] Raphael Gontijo-Lopes, Yann Dauphin, and Ekin D. Cubuk. No one representation to rule them all: overlapping
features of training methods, 2021. https://arxiv.org/abs/2110.12899.
[32] Ian J Goodfellow, Oriol Vinyals, and Andrew M Saxe. Qualitatively characterizing neural network optimization
problems. In International Conference on Learning Representations (ICLR), 2014. https://arxiv.org/abs/
1412.6544.
[33] Chuan Guo, GeoﬀPleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In
International Conference on Machine Learning (ICML), 2017. https://arxiv.org/abs/1706.04599.
[34] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In
Conference on Computer Vision and Pattern Recognition (CVPR), 2016. https://arxiv.org/abs/1512.03385.
[35] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and
perturbations. International Conference on Learning Representations (ICLR), 2019. https://arxiv.org/abs/
1903.12261.
[36] Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, and Balaji Lakshminarayanan.
AugMix: A simple data processing method to improve robustness and uncertainty. In International Conference
on Learning Representations (ICLR), 2020. https://arxiv.org/abs/1912.02781.
[37] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai,
Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. The many faces
of robustness: A critical analysis of out-of-distribution generalization. International Conference on Computer
Vision (ICCV), 2021. https://arxiv.org/abs/2006.16241.
[38] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples.
Conference on Computer Vision and Pattern Recognition (CVPR), 2021. https://arxiv.org/abs/1907.07174.
[39] Matteo Hessel, David Budden, Fabio Viola, Mihaela Rosca, Eren Sezener, and Tom Hennigan. Optax: composable
gradient transformation and optimisation, in jax!, 2020. URL http://github.com/deepmind/optax.
[40] John Hewitt, Xiang Lisa Li, Sang Michael Xie, Benjamin Newman, and Percy Liang. Ensembles and cocktails:
Robust ﬁnetuning for natural language generation. In NeurIPS 2021 Workshop on Distribution Shifts, 2021.
https://openreview.net/forum?id=qXucB21w1C3.
[41] Geoﬀrey Hinton, Oriol Vinyals, and JeﬀDean. Distilling the knowledge in a neural network. In Advances in
Neural Information Processing Systems (NeurIPS) Deep Learning Workshop, 2015. https://arxiv.org/abs/
1503.02531.
[42] Tin Kam Ho. The random subspace method for constructing decision forests. IEEE transactions on pattern
analysis and machine intelligence, 1998. https://ieeexplore.ieee.org/document/709601.
15

--- Page 16 ---
[43] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging
weights leads to wider optima and better generalization. In Conference on Uncertainty in Artiﬁcial Intelligence
(UAI), 2018. https://arxiv.org/abs/1803.05407.
[44] Arthur Jacot, Franck Gabriel, and Cl´ement Hongler. Neural tangent kernel: Convergence and generalization in
neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2018. https://arxiv.org/
abs/1806.07572.
[45] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V Le, Yunhsuan Sung, Zhen
Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision.
In International Conference on Machine Learning (ICML), 2021. https://arxiv.org/abs/2102.05918.
[46] Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Tuo Zhao. Smart: Robust and
eﬃcient ﬁne-tuning for pre-trained natural language models through principled regularized optimization. In
Association for Computational Linguistics (ACL), 2019. https://arxiv.org/abs/1911.03437.
[47] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,
2014.
[48] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu,
Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic
forgetting in neural networks. Proceedings of the national academy of sciences (PNAS), 2017. https://arxiv.
org/abs/1612.00796.
[49] Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani,
Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness,
Wei Guo, Berton A. Earnshaw, Imran S. Haque, Sara Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson,
Sergey Levine, Chelsea Finn, and Percy Liang. WILDS: A benchmark of in-the-wild distribution shifts. In
International Conference on Machine Learning (ICML), 2021. https://arxiv.org/abs/2012.07421.
[50] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil
Houlsby. Big transfer (bit): General visual representation learning. In European Conference on Computer Vision
(ECCV), 2020. https://arxiv.org/abs/1912.11370.
[51] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoﬀrey Hinton.
Similarity of neural network
representations revisited. In International Conference on Machine Learning (ICML), 2019. https://arxiv.
org/abs/1905.00414.
[52] Simon Kornblith, Jonathon Shlens, and Quoc V Le. Do better imagenet models transfer better? In Conference
on Computer Vision and Pattern Recognition (CVPR), 2019. https://arxiv.org/abs/1805.08974.
[53] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for ﬁne-grained categorization.
In International Conference on Computer Vision (ICCV) Workshops, 2013. https://ieeexplore.ieee.org/
document/6755945.
[54] Alex Krizhevsky, Geoﬀrey Hinton, et al. Learning multiple layers of features from tiny images, 2009. https:
//www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf.
[55] Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. Fine-tuning distorts pretrained
features and underperforms out-of-distribution, 2021. https://openreview.net/forum?id=UYneFzXSJWh.
[56] Ananya Kumar, Aditi Raghunathan, Tengyu Ma, and Percy Liang. Calibrated ensembles: A simple way
to mitigate ID-OOD accuracy tradeoﬀs. In NeurIPS 2021 Workshop on Distribution Shifts, 2021. https:
//openreview.net/forum?id=dmDE-9e9F_x.
[57] Ludmila I Kuncheva and Christopher J Whitaker. Measures of diversity in classiﬁer ensembles and their
relationship with the ensemble accuracy. Machine learning, 2003. https://doi.org/10.1023/A:1022859003006.
[58] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty
estimation using deep ensembles. In Advances in Neural Information Processing Systems (NeurIPS), 2017.
https://arxiv.org/abs/1612.01474.
16

--- Page 17 ---
[59] Hao Li, Pratik Chaudhari, Hao Yang, Michael Lam, Avinash Ravichandran, Rahul Bhotika, and Stefano Soatto.
Rethinking the hyperparameters for ﬁne-tuning. In International Conference on Learning Representations
(ICLR), 2020. https://arxiv.org/abs/2002.11770.
[60] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. In International
Conference on Learning Representations (ICLR), 2016. https://arxiv.org/abs/1608.03983.
[61] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on
Learning Representations (ICLR), 2019. https://openreview.net/forum?id=Bkg6RiCqY7.
[62] Shangyun Lu, Bradley Nott, Aaron Olson, Alberto Todeschini, Hossein Vahabi, Yair Carmon, and Ludwig
Schmidt. Harder or diﬀerent? a closer look at distribution shift in dataset reproduction. In International
Conference on Machine Learning (ICML) Workshop on Uncertainty and Robustness in Deep Learning, 2020.
http://www.gatsby.ucl.ac.uk/~balaji/udl2020/accepted-papers/UDL2020-paper-101.pdf.
[63] Ekdeep Singh Lubana, Puja Trivedi, Danai Koutra, and Robert P. Dick. How do quadratic regularizers prevent
catastrophic forgetting: The role of interpolation, 2021. https://arxiv.org/abs/2102.02805.
[64] James Lucas, Juhan Bae, Michael R Zhang, Stanislav Fort, Richard Zemel, and Roger Grosse. Analyzing
monotonic linear interpolation in neural network loss landscapes. In International Conference on Machine
Learning (ICML), 2021. https://arxiv.org/abs/2104.11044.
[65] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep
learning models resistant to adversarial attacks. In International Conference on Learning Representations
(ICLR), 2017. https://arxiv.org/abs/1706.06083.
[66] Michael Matena and Colin Raﬀel. Merging models with ﬁsher-weighted averaging, 2021. https://arxiv.org/
abs/2111.09832.
[67] Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks: The sequential
learning problem. Psychology of Learning and Motivation, 1989. https://www.sciencedirect.com/science/
article/pii/S0079742108605368.
[68] Mary L McHugh. Interrater reliability: the kappa statistic. Biochemia medica, 2012.
[69] John Miller, Karl Krauth, Benjamin Recht, and Ludwig Schmidt. The eﬀect of natural distribution shift
on question answering models.
In International Conference on Machine Learning (ICML), 2020.
https:
//arxiv.org/abs/2004.14444.
[70] John P Miller, Rohan Taori, Aditi Raghunathan, Shiori Sagawa, Pang Wei Koh, Vaishaal Shankar, Percy
Liang, Yair Carmon, and Ludwig Schmidt. Accuracy on the line: on the strong correlation between out-of-
distribution and in-distribution generalization. In International Conference on Machine Learning (ICML), 2021.
https://arxiv.org/abs/2107.04649.
[71] Rafael M¨uller, Simon Kornblith, and Geoﬀrey Hinton. When does label smoothing help? In Advances in Neural
Information Processing Systems (NeurIPS), 2019. https://arxiv.org/abs/1906.02629.
[72] Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Andr´e Susano Pinto, Daniel Keysers, and Neil Houlsby. Deep
ensembles for low-data transfer learning, 2020. https://arxiv.org/abs/2010.06866.
[73] Behnam Neyshabur, Hanie Sedghi, and Chiyuan Zhang. What is being transferred in transfer learning? In
Advances in Neural Information Processing Systems (NeurIPS), 2020. https://arxiv.org/abs/2008.11687.
[74] Alex Nichol, Joshua Achiam, and John Schulman. On ﬁrst-order meta-learning algorithms, 2018. https:
//arxiv.org/abs/1803.02999.
[75] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua V Dillon,
Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty? evaluating predictive
uncertainty under dataset shift. In Advances in Neural Information Processing Systems (NeurIPS), 2019.
https://arxiv.org/abs/1906.02530.
17

--- Page 18 ---
[76] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen,
Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep
learning library. In Advances in Neural Information Processing Systems (NeurIPS), 2019. https://arxiv.org/
abs/1912.01703.
[77] Hieu Pham, Zihang Dai, Golnaz Ghiasi, Hanxiao Liu, Adams Wei Yu, Minh-Thang Luong, Mingxing Tan, and
Quoc V. Le. Combined scaling for zero-shot transfer learning, 2021. https://arxiv.org/abs/2111.10050.
[78] Boris T Polyak and Anatoli B Juditsky. Acceleration of stochastic approximation by averaging. SIAM journal on
control and optimization, 1992. https://epubs.siam.org/doi/abs/10.1137/0330046?journalCode=sjcodc.
[79] Boris Teodorovich Polyak. New method of stochastic approximation type. Automation and remote control, 1990.
[80] Joaquin Qui˜nonero-Candela, Masashi Sugiyama, Neil D Lawrence, and Anton Schwaighofer. Dataset shift in
machine learning. Mit Press, 2009.
[81] Alec Radford, Jeﬀrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language Models are
Unsupervised Multitask Learners, 2019. https://openai.com/blog/better-language-models/.
[82] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable
visual models from natural language supervision. In International Conference on Machine Learning (ICML),
2021. https://arxiv.org/abs/2103.00020.
[83] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do ImageNet classiﬁers generalize
to ImageNet? In International Conference on Machine Learning (ICML), 2019. https://arxiv.org/abs/1902.
10811.
[84] David Ruppert. Eﬃcient estimations from a slowly convergent robbins-monro process, 1988. https://ecommons.
cornell.edu/handle/1813/8664.
[85] Hadi Salman, Greg Yang, Jerry Li, Pengchuan Zhang, Huan Zhang, Ilya Razenshteyn, and Sebastien Bubeck.
Provably robust deep learning via adversarially trained smoothed classiﬁers. In Advances in Neural Information
Processing Systems (NeurIPS), 2019. https://arxiv.org/abs/1906.04584.
[86] Mert Bulent Sariyildiz, Julien Perez, and Diane Larlus. Learning visual representations with caption annotations.
In European Conference on Computer Vision (ECCV), 2020. https://arxiv.org/abs/2008.01392.
[87] Ali Shafahi, Mahyar Najibi, Amin Ghiasi, Zheng Xu, John Dickerson, Christoph Studer, Larry S Davis, Gavin
Taylor, and Tom Goldstein. Adversarial training for free! In Advances in Neural Information Processing Systems
(NeurIPS), 2019. https://arxiv.org/abs/1904.12843.
[88] Vaishaal Shankar, Achal Dave, Rebecca Roelofs, Deva Ramanan, Benjamin Recht, and Ludwig Schmidt. Do
image classiﬁers generalize across time?, 2019. https://arxiv.org/abs/1906.02168.
[89] Vaishaal Shankar, Rebecca Roelofs, Horia Mania, Alex Fang, Benjamin Recht, and Ludwig Schmidt. Evaluating
machine accuracy on imagenet.
In International Conference on Machine Learning (ICML), 2020.
http:
//proceedings.mlr.press/v119/shankar20c/shankar20c.pdf.
[90] Ali Sharif Razavian, Hossein Azizpour, Josephine Sullivan, and Stefan Carlsson. Cnn features oﬀ-the-shelf: an
astounding baseline for recognition. In Proceedings of the IEEE conference on computer vision and pattern
recognition workshops, 2014. https://arxiv.org/abs/1403.6382.
[91] David B Skalak et al. The sources of increased accuracy for two proposed boosting algorithms. In American
Association for Artiﬁcial Intelligence (AAAI), Integrating Multiple Learned Models Workshop, 1996. https:
//citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.40.2269&rep=rep1&type=pdf.
[92] Asa Cooper Stickland and Iain Murray. Diverse ensembles improve calibration. In International Conference
on Machine Learning (ICML) Workshop on Uncertainty and Robustness in Deep Learning, 2020.
https:
//arxiv.org/abs/2007.04206.
18

--- Page 19 ---
[93] Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable eﬀectiveness
of data in deep learning era. In International Conference on Computer Vision (ICCV), 2017. https://arxiv.
org/abs/1707.02968.
[94] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James
Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al.
Scalability in perception for autonomous driving:
Waymo open dataset. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020. https:
//arxiv.org/abs/1912.04838.
[95] Christian Szegedy, Vincent Vanhoucke, Sergey Ioﬀe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception
architecture for computer vision. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
https://arxiv.org/abs/1512.00567.
[96] Mingxing Tan and Quoc Le. Eﬃcientnet: Rethinking model scaling for convolutional neural networks. In
International Conference on Machine Learning (ICML), 2019. https://proceedings.mlr.press/v97/tan19a/
tan19a.pdf.
[97] Rohan Taori, Achal Dave, Vaishaal Shankar, Nicholas Carlini, Benjamin Recht, and Ludwig Schmidt. Measuring
robustness to natural distribution shifts in image classiﬁcation. In Advances in Neural Information Processing
Systems (NeurIPS), 2020. https://arxiv.org/abs/2007.00644.
[98] Antonio Torralba and Alexei A Efros. Unbiased look at dataset bias. In Conference on Computer Vision
and Pattern Recognition (CVPR), 2011. https://people.csail.mit.edu/torralba/publications/datasets_
cvpr11.pdf.
[99] Florian Tram`er, Alexey Kurakin, Nicolas Papernot, Ian Goodfellow, Dan Boneh, and Patrick McDaniel. Ensemble
adversarial training: Attacks and defenses. In International Conference on Learning Representations (ICLR),
2017. https://arxiv.org/abs/1705.07204.
[100] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by
penalizing local predictive power. In Advances in Neural Information Processing Systems (NeurIPS), 2019.
https://arxiv.org/abs/1905.13549.
[101] Ross Wightman. Pytorch image models. https://github.com/rwightman/pytorch-image-models, 2019.
[102] Mitchell Wortsman, Maxwell C Horton, Carlos Guestrin, Ali Farhadi, and Mohammad Rastegari. Learning
neural network subspaces. In International Conference on Machine Learning (ICML), 2021. https://arxiv.
org/abs/2102.10472.
[103] Jianxiong Xiao, Krista A Ehinger, James Hays, Antonio Torralba, and Aude Oliva. Sun database: Exploring a
large collection of scene categories. International Journal of Computer Vision, 2016. https://link.springer.
com/article/10.1007/s11263-014-0748-y.
[104] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. Self-training with noisy student improves
imagenet classiﬁcation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2020. https:
//arxiv.org/abs/1911.04252.
[105] LI Xuhong, Yves Grandvalet, and Franck Davoine. Explicit inductive bias for transfer learning with convolutional
networks. In International Conference on Machine Learning (ICML), 2018. https://arxiv.org/abs/1802.
01483.
[106] Chhavi Yadav and L´eon Bottou. Cold case: The lost mnist digits. In Advances in Neural Information Processing
Systems (NeurIPS), 2019. https://arxiv.org/abs/1905.10498.
[107] I Zeki Yalniz, Herv´e J´egou, Kan Chen, Manohar Paluri, and Dhruv Mahajan. Billion-scale semi-supervised
learning for image classiﬁcation, 2019. https://arxiv.org/abs/1905.00546.
[108] Friedemann Zenke, Ben Poole, and Surya Ganguli.
Continual learning through synaptic intelligence.
In
International Conference on Machine Learning (ICML), 2017. https://arxiv.org/abs/1703.04200.
19

--- Page 20 ---
[109] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas
Beyer. Lit: Zero-shot transfer with locked-image text tuning, 2021. https://arxiv.org/abs/2111.07991.
[110] Michael R Zhang, James Lucas, Geoﬀrey Hinton, and Jimmy Ba. Lookahead optimizer: k steps forward, 1 step
back. In Advances in Neural Information Processing Systems (NeurIPS), 2019. https://arxiv.org/abs/1907.
08610.
[111] Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D Manning, and Curtis P Langlotz. Contrastive learning
of medical visual representations from paired images and text, 2020. https://arxiv.org/abs/2010.00747.
[112] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models,
2021. https://arxiv.org/abs/2109.01134.
[113] Chen Zhu, Yu Cheng, Zhe Gan, Siqi Sun, Tom Goldstein, and Jingjing Liu. Freelb: Enhanced adversarial
training for natural language understanding. In International Conference on Learning Representations (ICLR),
2020. https://arxiv.org/abs/1909.11764.
20

--- Page 21 ---
A
Pseudocode for WiSE-FT
Algorithm 1 Pytorch pseudocode for WiSE-FT
def wse(model, zeroshot_checkpoint, finetuned_checkpoint, alpha):
# load state dicts from checkpoints
theta_0 = torch.load(zeroshot_checkpoint)["state_dict"]
theta_1 = torch.load(finetuned_checkpoint)["state_dict"]
# make sure checkpoints are compatible
assert set(theta_0.keys()) == set(theta_1.keys())
# interpolate between all weights in the checkpoints
theta = {
key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
for key in theta_0.keys()
}
# update the model (in-place) according to the new weights
model.load_state_dict(theta)
def wise_ft(model, dataset, zeroshot_checkpoint, alpha, hparams):
# load the zero-shot weights
theta_0 = torch.load(zeroshot_checkpoint)["state_dict"]
model.load_state_dict(theta_0)
# standard fine-tuning
finetuned_checkpoint = finetune(model, dataset, hparams)
# perform weight-space ensembling (in-place)
wse(model, zeroshot_checkpoint, finetuned_checkpoint, alpha)
B
Mixing coeﬃcient
Table 3 compares the performance of WiSE-FT using a ﬁxed mixing coeﬃcient α=0.5 with the ﬁxed optimal
mixing coeﬃcient. On ImageNet and the ﬁve derived distribution shifts, the average performance of the
optimal α is 0 to 0.4 percentage points better than that of α=0.5. Due to its simplicity and eﬀectiveness, we
recommend using α=0.5 when no domain knowledge is available. Finding the optimal value of the mixing
coeﬃcient for any distribution is an interesting question for future work. Unlike other hyperparameters, no
re-training is required to test diﬀerent α, so tuning is relatively cheap.
C
Additional experiments
This section supplements the results of Section 4. First, in Section C.1 we provide a breakdown of Figure 1
for each distribution shift. Next, in Section C.2 we provide eﬀective robustness scatter plots for six additional
distribution shifts, ﬁnding WiSE-FT to provide consistent improvements under distribution shift without any
loss in performance on the reference distribution. Section C.3 compares WiSE-FT with additional alternatives
including distillation and CoOp [112]. Beyond robustness, Section C.5 demonstrates that WiSE-FT can
provide accuracy improvements on reference data, with a focus on the low-data regime. Section C.6 showcases
that the accuracy improvements under distribution shift are not isolated to large models, ﬁnding similar
trends across scales of pre-training computes. Section C.7 explores the application of WiSE-FT for additional
models such as ALIGN [45], a ViT-H/14 model pre-trained on JFT [21] and BASIC [77]. Finally, Section C.8
ensembles zero-shot CLIP with an independently trained classiﬁer.
C.1
Breakdown of CLIP experiments on ImageNet
In contrast to Figures 1 and 4, where our key experimental results for ImageNet and ﬁve derived distribution
shifts are averaged, we now display the results separately for each distribution shift. Results are provided in
Figures 7, 8.
21

--- Page 22 ---
Distribution shifts
Avg
Avg
IN (ref.)
IN-V2
IN-R
IN-Sketch
ObjectNet
IN-A
shifts
ref., shifts
ViT-B/16, end-to-end
0.9
0.4
1.4
0.2
0.4
2.4
0.5
0.0
ViT-B/16, linear classiﬁer
1.8
0.6
1.2
0.1
0.2
0.6
0.1
0.2
ViT-L/14@336, end-to-end
0.3
0.0
0.9
0.3
1.0
1.1
0.5
0.1
ViT-L/14@336, linear classiﬁer
1.6
0.6
0.2
0.0
0.0
0.0
0.0
0.4
Table 3: Diﬀerence in performance (percentage points) between WiSE-FT using the optimal mixing coeﬃcient
and a ﬁxed value of α=0.5 for CLIP ViT-B/16 and ViT-L/14@336. For each cell in the table, the optimal
mixing coeﬃcient α is chosen individually such that the corresponding metric is maximized. Results for all
mixing coeﬃcients are available in Tables 4 and 5. Avg shifts displays the mean performance among the ﬁve
distribution shifts, while Avg reference, shifts shows the average of ImageNet (reference) and Avg shifts.
To assist in contextualizing the results, the scatter plots we display also show a wide range of machine learning
models from a comprehensive testbed of evaluations [97, 70], including: models trained on Str
D (standard
training); models trained on additional data and ﬁne-tuned using Str
D (trained with more data); and models
trained using various existing robustness interventions, e.g. special data augmentation [19, 22, 28, 36] or
adversarially robust models [65, 15, 85, 87].
Additionally, Tables 4 and 5 show the performance of WiSE-FT for various values of the mixing coeﬃcient α
on ImageNet and ﬁve derived distribution shifts, for CLIP ViT-L/14@336 and the ViT-B/16 model.
75
77
79
81
83
85
87
ImageNet (top-1, %)
65
67
69
71
73
75
77
79
ImageNetV2 (top-1, %)
94
95
96
97
98
ImageNet (class-subsampled) (top-1, %)
70
72
74
76
78
80
82
84
86
88
90
ImageNet-R (top-1, %)
74
79
84
ImageNet (top-1, %)
52
57
62
67
ImageNet Sketch (top-1, %)
80
85
90
ImageNet (class-subsampled) (top-1, %)
55
60
65
70
ObjectNet (top-1, %)
94
96
98
ImageNet (class-subsampled) (top-1, %)
64
66
68
70
72
74
76
78
80
ImageNet-A (top-1, %)
CLIP zero-shot models
Linear ﬁt (CLIP zero-shot models)
CLIP ﬁne-tuned end-to-end
CLIP ﬁne-tuned with a linear classiﬁer
(prior work)
Weight-space ensemble (end-to-end)
Weight-space ensemble (linear classiﬁer)
Weight-space ensemble with α = 0.5
Standard ImageNet models
Linear ﬁt (standard ImageNet models)
Trained with more data
Existing robustness interventions
y = x
Figure 7: A per-dataset breakdown of the key experimental results (Figure 1). WiSE-FT improves accuracy
on ImageNet and ﬁve derived distribution shifts. Standard ImageNet models, models trained with more data,
and existing robustness interventions are from the testbed of Taori et al. [97].
22

--- Page 23 ---
60
65
70
75
80
85
ImageNet (top-1, %)
50
55
60
65
70
75
80
ImageNetV2 (top-1, %)
60 65 70 75
80
85
90
95
ImageNet (class-subsampled) (top-1, %)
30
35
40
45
50
55
60
65
70
75
80
85
90
ImageNet-R (top-1, %)
50
55
60
65
70
75
80
85
ImageNet (top-1, %)
20
25
30
35
40
45
50
55
60
65
ImageNet Sketch (top-1, %)
60
65
70
75
80
85
90
ImageNet (class-subsampled) (top-1, %)
40
45
50
55
60
65
70
ObjectNet (top-1, %)
80
85
90
95
ImageNet (class-subsampled) (top-1, %)
30
35
40
45
50
55
60
65
70
75
80
ImageNet-A (top-1, %)
CLIP zero-shot models
Linear ﬁt (CLIP zero-shot models)
CLIP ﬁne-tuned end-to-end
CLIP ﬁne-tuned with a linear classiﬁer
(prior work)
Weight-space ensemble (end-to-end)
Weight-space ensemble (linear classiﬁer)
Standard ImageNet models
Linear ﬁt (standard ImageNet models)
Trained with more data
Existing robustness interventions
y = x
Figure 8: A zoomed-out version of Figure 7. WiSE-FT improves accuracy on ImageNet and ﬁve derived
distribution shifts. Standard ImageNet models, models trained with more data, and existing robustness
interventions are from the testbed of Taori et al. [97].
23

--- Page 24 ---
Distribution shifts
Avg
Avg
IN (ref.)
IN-V2
IN-R
IN-Sketch
ObjectNet
IN-A
shifts
ref., shifts
WiSE-FT, end-to-end
α=0.00
76.6
70.5
89.0
60.9
68.5
77.6
73.3
74.9
α=0.05
78.7
72.6
89.6
62.2
69.5
79.0
74.6
76.7
α=0.10
80.4
74.2
89.9
63.1
70.4
79.8
75.5
78.0
α=0.15
81.9
75.4
90.1
63.8
71.1
80.4
76.2
79.1
α=0.20
83.2
76.5
90.3
64.3
71.6
80.8
76.7
80.0
α=0.25
84.2
77.5
90.3
64.6
72.1
81.0
77.1
80.7
α=0.30
85.1
78.3
90.3
64.9
72.1
81.0
77.3
81.2
α=0.35
85.7
78.7
90.1
65.0
72.0
81.0
77.4
81.6
α=0.40
86.2
79.2
89.9
65.0
71.9
80.7
77.3
81.8
α=0.45
86.6
79.4
89.6
64.9
71.6
80.6
77.2
81.9
α=0.50
86.8
79.5
89.4
64.7
71.1
79.9
76.9
81.8
α=0.55
87.0
79.3
88.9
64.5
70.7
79.1
76.5
81.8
α=0.60
87.1
79.2
88.5
64.1
70.1
78.2
76.0
81.5
α=0.65
87.1
79.3
87.8
63.6
69.6
77.4
75.5
81.3
α=0.70
87.1
79.1
87.0
63.1
68.9
76.5
74.9
81.0
α=0.75
87.0
78.8
86.1
62.5
68.1
75.2
74.1
80.5
α=0.80
86.9
78.4
85.1
61.7
67.4
73.8
73.3
80.1
α=0.85
86.8
78.0
84.0
61.0
66.4
72.0
72.3
79.5
α=0.90
86.7
77.6
82.8
60.0
65.5
69.9
71.2
79.0
α=0.95
86.5
77.2
81.3
59.0
64.3
67.7
69.9
78.2
α=1.00
86.2
76.8
79.8
57.9
63.3
65.4
68.6
77.4
WiSE-FT, linear classiﬁer
α=0.00
76.6
70.5
89.0
60.9
69.1
77.7
73.4
75.0
α=0.05
77.6
71.3
89.2
61.3
69.3
78.3
73.9
75.8
α=0.10
78.4
72.1
89.4
61.7
69.6
78.8
74.3
76.3
α=0.15
79.3
72.8
89.5
62.1
70.0
79.0
74.7
77.0
α=0.20
80.0
73.5
89.6
62.4
70.3
79.3
75.0
77.5
α=0.25
80.8
74.1
89.7
62.6
70.5
79.5
75.3
78.0
α=0.30
81.5
74.8
89.7
62.8
70.7
79.5
75.5
78.5
α=0.35
82.1
75.4
89.8
62.9
70.7
79.6
75.7
78.9
α=0.40
82.7
75.8
89.7
63.0
70.7
79.6
75.8
79.2
α=0.45
83.2
76.1
89.7
63.0
70.7
79.6
75.8
79.5
α=0.50
83.7
76.3
89.6
63.0
70.7
79.7
75.9
79.8
α=0.55
84.1
76.5
89.5
62.9
70.5
79.6
75.8
79.9
α=0.60
84.4
76.7
89.3
62.7
70.3
79.5
75.7
80.1
α=0.65
84.7
76.8
89.1
62.6
70.1
79.4
75.6
80.2
α=0.70
85.0
76.9
88.9
62.3
69.9
79.1
75.4
80.2
α=0.75
85.1
76.8
88.4
61.9
69.7
78.9
75.1
80.1
α=0.80
85.3
76.9
87.9
61.4
69.3
78.5
74.8
80.0
α=0.85
85.3
76.7
87.4
60.9
68.8
78.1
74.4
79.8
α=0.90
85.3
76.4
86.8
60.3
68.4
77.3
73.8
79.5
α=0.95
85.3
76.2
86.1
59.5
67.7
76.8
73.3
79.3
α=1.00
85.2
75.8
85.3
58.7
67.2
76.1
72.6
78.9
Table 4:
WiSE-FT accuracy on the reference and shifted distributions for various values of the mixing
coeﬃcient α. Results shown for CLIP ViT-L/14@336. Note that α=0.0 corresponds to the zero-shot model,
while α = 1.0 corresponds to standard ﬁne-tuning. Avg shifts displays the mean performance among the ﬁve
distribution shifts, while Avg reference, shifts shows the average of ImageNet (reference) and Avg shifts.
24

--- Page 25 ---
Distribution shifts
Avg
Avg
IN (ref.)
IN-V2
IN-R
IN-Sketch
ObjectNet
IN-A
shifts
ref., shifts
WiSE-FT, end-to-end
α=0.00
68.3
61.9
77.6
48.2
53.0
49.8
58.1
63.2
α=0.05
70.7
64.0
78.6
49.6
54.5
51.5
59.6
65.2
α=0.10
72.9
65.7
79.4
50.8
55.7
52.5
60.8
66.8
α=0.15
74.8
67.2
79.9
51.7
56.6
53.5
61.8
68.3
α=0.20
76.4
68.7
80.1
52.5
57.1
54.2
62.5
69.5
α=0.25
77.8
69.9
80.1
53.1
57.4
54.6
63.0
70.4
α=0.30
78.9
70.6
80.1
53.6
57.5
54.6
63.3
71.1
α=0.35
79.7
71.5
79.9
53.9
57.6
54.3
63.4
71.5
α=0.40
80.5
72.1
79.6
54.1
57.7
53.8
63.5
72.0
α=0.45
81.2
72.4
79.3
54.0
57.5
53.2
63.3
72.2
α=0.50
81.7
72.8
78.7
53.9
57.3
52.2
63.0
72.3
α=0.55
82.1
73.0
78.0
53.8
56.6
51.4
62.6
72.3
α=0.60
82.4
72.9
77.2
53.4
56.2
50.0
61.9
72.2
α=0.65
82.6
73.1
76.3
53.0
55.5
48.9
61.4
72.0
α=0.70
82.6
73.2
75.2
52.4
55.0
47.4
60.6
71.6
α=0.75
82.6
73.1
73.9
51.8
54.3
46.0
59.8
71.2
α=0.80
82.5
72.8
72.7
51.0
53.5
44.6
58.9
70.7
α=0.85
82.3
72.4
71.1
50.0
52.7
42.9
57.8
70.0
α=0.90
82.1
72.0
69.5
48.9
51.7
40.9
56.6
69.3
α=0.95
81.7
71.5
67.7
47.6
50.7
38.8
55.3
68.5
α=1.00
81.3
70.9
65.6
46.3
49.6
36.7
53.8
67.5
WiSE-FT, linear classiﬁer
α=0.00
68.4
62.6
77.6
48.2
53.8
50.0
58.4
63.4
α=0.05
69.9
63.7
77.9
48.9
54.2
50.6
59.1
64.5
α=0.10
71.3
64.8
78.2
49.5
54.7
51.0
59.6
65.5
α=0.15
72.5
65.8
78.4
50.0
55.1
51.1
60.1
66.3
α=0.20
73.6
66.6
78.4
50.5
55.3
51.5
60.5
67.0
α=0.25
74.7
67.4
78.4
50.8
55.3
51.8
60.7
67.7
α=0.30
75.6
68.0
78.3
51.1
55.4
51.7
60.9
68.2
α=0.35
76.4
68.8
78.2
51.3
55.5
51.6
61.1
68.8
α=0.40
77.1
69.0
77.8
51.3
55.5
51.4
61.0
69.0
α=0.45
77.7
69.4
77.6
51.3
55.4
51.3
61.0
69.3
α=0.50
78.2
69.9
77.2
51.2
55.3
51.2
61.0
69.6
α=0.55
78.6
70.1
76.7
51.0
55.0
50.9
60.7
69.7
α=0.60
79.0
70.2
76.1
50.8
54.7
50.5
60.5
69.8
α=0.65
79.3
70.4
75.7
50.4
54.5
50.1
60.2
69.8
α=0.70
79.6
70.4
75.2
50.1
54.2
49.9
60.0
69.8
α=0.75
79.7
70.4
74.6
49.7
53.9
49.5
59.6
69.7
α=0.80
79.8
70.5
73.9
49.3
53.6
49.0
59.3
69.5
α=0.85
79.9
70.4
73.2
48.7
53.3
48.6
58.8
69.3
α=0.90
80.0
70.3
72.4
48.1
52.8
47.8
58.3
69.2
α=0.95
79.9
70.1
71.7
47.5
52.6
46.9
57.8
68.8
α=1.00
79.9
69.8
70.8
46.9
52.1
46.4
57.2
68.6
Table 5:
WiSE-FT accuracy on the reference and shifted distributions for various values of the mixing
coeﬃcient α. Results shown for CLIP ViT-B/16. Note that α=0.0 corresponds to the zero-shot model, while
α = 1.0 corresponds to standard ﬁne-tuning. Avg shifts displays the mean performance among the ﬁve
distribution shifts, while Avg reference, shifts shows the average of ImageNet (reference) and Avg shifts.
25

--- Page 26 ---
90
92
94
96
98
ImageNet (class-subsampled) (top-1, %)
55
60
65
70
75
80
85
90
95
Imagenet-Vid-Robust (pm-0, %)
+9.0pp OOD
-0.1pp ID
Imagenet-Vid-Robust
CLIP zero-shot models
Linear ﬁt (CLIP zero-shot models)
CLIP ﬁne-tuned end-to-end
CLIP ﬁne-tuned with a linear classiﬁer
Weight-space ensemble (end-to-end)
Weight-space ensemble (linear classiﬁer)
Weight-space ensemble with α = 0.5
Standard ImageNet models
Linear ﬁt (standard ImageNet models)
Trained with more data
Existing robustness interventions
y = x
90
92
94
96
98
ImageNet (class-subsampled) (top-1, %)
40
45
50
55
60
65
70
75
80
85
90
95
YTBB-Robust (pm-0, %)
+23.2pp OOD
-0.1pp ID
YTBB-Robust
85 87 89 91 93
95
97
99
CIFAR10 (top-1, %)
75
80
85
90
95
CIFAR-10.1 (top-1, %)
Relative to LC:
+2.0pp OOD
+1.2pp ID
Relative to E2E:
+1.7pp OOD
+0.7pp ID
CIFAR-10.1
75
80
85
90
95
CIFAR10 (top-1, %)
65
70
75
80
85
90
95
CIFAR-10.2 (top-1, %)
Relative to LC:
+2.1pp OOD
+1.2pp ID
Relative to E2E:
+2.1pp OOD
+0.7pp ID
CIFAR-10.2
20
25
30
35
40
45 50 55
60
65
70
75
In-distribution test accuracy
15
20
25
30
35
40
45
50
OOD worst region accuracy
Relative to LC:
+14.2pp OOD
+16.9pp ID
Relative to E2E:
+3.5pp OOD
-0.3pp ID
WILDS-FMoW
10
15
20
25
30
35
40
45
50
55
ID test macro F1
10
15
20
25
30
35
40
45
OOD test macro F1
Relative to LC:
+4.5pp OOD
+1.0pp ID
Relative to E2E:
+6.2pp OOD
+3.6pp ID
WILDS-iWildCam
Figure 9: WiSE-FT improves accuracy under distribution shift relative to standard ﬁne-tuning on ImageNet-
Vid-Robust, YTBB-Robust [88], CIFAR-10.1 [83], CIFAR-10.2 [62], WILDS-FMoW [49, 13], and WILDS-
iWildCam [49, 6].
C.2
Robustness on additional distribution shifts
Figure 9 displays the eﬀective robustness scatter plots for the six additional distribution shifts discussed in
Section 4 (analogous results provided in Table 6).
Concretely, we consider: (i) ImageNet-Vid-Robust and YTBB-Robust, datasets with distribution shift induced
by temporal perturbations in videos [88]; (ii) CIFAR-10.1 [83] and CIFAR-10.2 [62], reproductions of the
popular image classiﬁcation dataset CIFAR-10 [54] with a distribution shift; (iii) WILDS-FMoW, a satellite
image recognition task where the test set has a geographic and temporal distribution shift [49, 13]; (iv)
WILDS-iWildCam, a wildlife recognition task where the test set has a geographic distribution shift [49, 6].
C.3
Comparison with alternative methods
We now extend Section 4 and compare WiSE-FT to additional methods of ﬁne-tuning. We begin with
contrasting the weight-space and output-space ensemble. Next, we show the that varying the decay parameter
of an exponential moving average also moves along the curve produced by WiSE-FT. Finally, we compare
with additional methods when ﬁne-tuning only a linear classiﬁer including distillation and various forms of
regularization.
26

--- Page 27 ---
Zero-shot
Fine-tuned
WiSE-FT, α=0.5
WiSE-FT, optimal α
ImageNet-Vid-Robust (pm-0)
95.9
86.5
95.5
96.5
YTBBRobust (pm-0)
95.8
66.5
89.7
96.0
CIFAR-10.1 (top-1)
92.5
95.9
97.6
98.0
CIFAR-10.2 (top-1)
88.8
91.3
93.4
94.4
WILDS-FMoW: ID test (accuracy)
28.0
73.3
73.0
74.8
WILDS-FMoW: OOD worst region accuracy
23.8
46.0
49.5
49.7
WILDS-iWildCam: ID test macro F1
15.1
52.1
55.8
55.8
WILDS-iWildCam: OOD test macro F1
15.5
39.9
46.1
46.4
Table 6: WiSE-FT improves results on ImageNet-Vid-Robust, YTBB-Robust [88], CIFAR-10.1 [83], CIFAR-
10.2 [62], WILDS-FMoW [49, 13], and WILDS-iWildCam [49, 6]. Reported numbers are percentages. This is
the corresponding table for Figure 9. This table displays results for ﬁne-tuning only a linear classiﬁer for
ImageNet-Vid-Robust and YTBBRobust and end-to-end ﬁne-tuning for the remainder.
65
70
75
80
ImageNet (top-1, %)
50
55
60
65
Avg. accuracy on 5 distribution shifts
CLIP zero-shot
Linear ﬁt (CLIP zero-shot)
Weight-space ensemble (end-to-end)
Output-space ensemble (end-to-end)
Figure 10: Comparing the weight-space ensemble f(x, (1 −α) · θ0 + α · θ1) with the output-space ensemble
(1−α)f(x, θ0)+α·f(x, θ1) when ﬁne-tuning end-to-end with learning rate 3·10−5. Note that the output-space
ensemble requires 2x compute.
C.3.1
Output-space ensembles
Figure 10 compares the weight-space ensemble f(x, (1 −α) · θ0 + α · θ1) with the output-space ensemble
(1 −α)f(x, θ0) + α · f(x, θ1). Both exhibit a favorable trend, though the output-space ensemble requires
twice as much compute. Section F further explores the relation between the weight-space and output-space
ensemble.
27

--- Page 28 ---
60
65
70
75
80
ImageNet Accuracy (top-1, %)
42
44
46
48
50
52
54
Avg. accuracy on 5 distribution shifts
Linear ﬁt (CLIP zero-shot models)
WiSE-FT Curves
EMA throughout ﬁne-tuning
CLIP zero-shot models
Standard ﬁne-tuning (No EMA)
Fine-tuned with EMA (decay 0.99)
Fine-tuned with EMA (decay 0.999)
Fine-tuned with EMA (decay 0.9999)
Fine-tuned with EMA (decay 0.99999)
Fine-tuned with EMA (decay 0.999999)
Figure 11: Results for the debiased variant of EMA described in Appendix C.3.2. EMA improves accuracy on
both ImageNet and on the distribution shifts, and further applying WiSE-FT to EMA solutions can improve
robustness. The solutions with no EMA, decay 0.99, and decay 0.999 are overlapping in the plot, as are the
solutions with decay 0.99999 and 0.999999.
60
65
70
75
80
ImageNet Accuracy (top-1, %)
42
44
46
48
50
52
54
Avg. accuracy on 5 distribution shifts
60
65
70
75
80
ImageNet Accuracy (top-1, %)
42
44
46
48
50
52
54
Avg. accuracy on 5 distribution shifts
Linear ﬁt (CLIP zero-shot models)
WiSE-FT Curves
EMA throughout ﬁne-tuning
CLIP zero-shot models
Fine-tuned with EMA (decay 0.9999)
Fine-tuned with EMA (decay 0.99992)
Fine-tuned with EMA (decay 0.99994)
Fine-tuned with EMA (decay 0.99997)
Fine-tuned with EMA (decay 0.99999)
Standard ﬁne-tuning (No EMA)
Figure 12: Results for the variant of EMA biased towards the initialization, described in Appendix C.3.2.
Varying the EMA decay β moves along the curve produced by WiSE-FT. Applying WiSE-FT to EMA
solutions moves further along the curve produced by WiSE-FT.
C.3.2
Comparison to exponential moving averages
Weight-averaging along the trajectory can improve the performance of models. For instance, Szegedy et al.
[95] use a running average of the model parameters for their Inception-v2 model. The exponential moving
average (EMA) is a standard technique for keeping a running average of model parameters and is implemented
in libraries such as Optax [39] and Pytorch ImageNet Models [101].
This section explores two variants of EMA for model parameters θ ∈Rn. The ﬁrst variant is a debiased
EMA, where debiasing is done as in Kingma and Ba [47] (Algorithm 1). For each iteration t ∈{1, ..., T}
let θt ∈Rn be the model parameters at step t and let µt ∈Rn be the EMA at step t. For t = 0, µ0 ←0,
otherwise µt ←β · µt−1 + (1 −β) · θt where β is a decay hyperparameter. The ﬁnal debiased EMA is given
by µT /(1 −βT ). Results for various decay hyperparameters are illustrated by Figure 11.
Next, we explore a variant of EMA that is biased towards the initialization θ0. As before, µt ←β · µt−1 +
(1 −β) · θt. However µ0 is now initialized to be θ0, instead of zeros. Moreover, at the end of ﬁne-tuning we
use the biased estimate µT . Results for this variant are illustrated by Figure 12.
Section 4 (Figure 3) showed that decreasing learning rate, training epochs, or early stopping leads to solutions
that lie below the curve produced by WiSE-FT. On the other hand, using an exponential moving average
28

--- Page 29 ---
74 75 76 77 78 79 80 81 82
83
84
85
86
ImageNet (to-1, %)
70
71
72
73
74
75
76
Avg. accuracy on 5 distribution shifts
Comparison to random interpolation
Random interpolation
74 75 76 77 78 79 80 81 82
83
84
85
86
ImageNet (to-1, %)
70
71
72
73
74
75
76
Avg. accuracy on 5 distribution shifts
Comparison to softmax output ensemble
Softmax output ensemble
74 75 76 77 78 79 80 81 82
83
84
85
86
ImageNet (to-1, %)
70
71
72
73
74
75
76
Avg. accuracy on 5 distribution shifts
Comparison to linear classiﬁer with regularization
Weight decay
Label smoothing
No regularization
L1 regularization
74 75 76 77 78 79 80 81 82
83
84
85
86
ImageNet (to-1, %)
70
71
72
73
74
75
76
Avg. accuracy on 5 distribution shifts
Comparison to distillation
Distillation
74 75 76 77 78 79 80 81 82
83
84
85
86
ImageNet (to-1, %)
70
71
72
73
74
75
76
Avg. accuracy on 5 distribution shifts
Comparison to regularize to zero-shot
Regularize to zero-shot
74 75 76 77 78 79 80 81 82
83
84
85
86
ImageNet (to-1, %)
70
71
72
73
74
75
76
Avg. accuracy on 5 distribution shifts
Comparison to warmstart linear classiﬁer
Weight decay
Label smoothing
No regularization
L1 regularization
CLIP zero-shot models
Linear ﬁt (CLIP zero-shot models)
CLIP ﬁne-tuned with a linear classiﬁer
Weight-space ensemble (linear classiﬁer)
Figure 13: Accuracy on the reference and shifted distributions of WiSE-FT and the alternatives described in
Section C.3.3.
(EMA) and varying the EMA decay β can move along or slightly outside or along the curve produced by
WiSE-FT. For instance, solutions using the second EMA variant follow the WiSE-FT curve. Indeed, applying
WiSE-FT with mixing coeﬃcient 1 −βT to the debiased EMA variant exactly recovers the second EMA
variant described above. Moreover, further applying WiSE-FT to EMA solutions (i.e., interpolating the
weights of the zero-shot model with the EMA solution) can lead to additional robustness. We also evaluate
EMA along the ﬁne-tuning trajectory, ﬁnding improved performance under distribution shift for the variant
biased towards the initialization. For the debiased EMA, each model along the trajectory is debiased by
1/(1 −βt). As shown in Figures 11,12, evaluations along the trajectory underperform solutions generated by
applying WiSE-FT.
C.3.3
Additional comparisons when ﬁne-tuning a linear classiﬁer
We compare against several additional alternatives when ﬁne-tuning only a linear classiﬁer. As this setting
is computationally cheaper compared to end-to-end, it allows for comprehensive experimentation. Many of
the examined approaches exhibit a concave trend in eﬀective robustness plots, although WiSE-FT matches
methods requiring more compute or oﬀers better performance (Figure 13).
Random interpolation. This method uses either the zero-shot or ﬁne-tuned linear classiﬁer depending on
a (biased) coin ﬂip. For hyperparameter α ∈[0, 1] outputs are computed as (1 −ξ) · f(x, θ0) + ξ · f(x, θ1)
where ξ is a Bernoulli(α) random variable. For this method and all others with a hyperparameter α ∈[0, 1]
we evaluate models for α ∈{0, 0.05, 0.1, ..., 1}.
29

--- Page 30 ---
Ensembling softmax outputs. Instead of ensembling in weight space, this method combines softmax
probabilities assigned by the zero-shot and ﬁne-tuned linear classiﬁer. Concretely, for hyperparameter
α ∈[0, 1] outputs are computed as (1 −α) · softmax(f(x, θ0)) + α · softmax(f(x, θ1)). This method performs
comparably to weight-space ensembling but requires slightly more compute.
Linear classiﬁer with various regularizers. We explore ﬁne-tuning linear classiﬁers with four regulariza-
tion strategies: no regularization, weight decay, L1 regularization, and label smoothing [71]. Linear classiﬁers
are trained with mini-batch optimization, using the AdamW optimizer [61, 76] with a cosine-annealing
learning rate schedule [60]. This method is signiﬁcantly faster and less memory-intensive than the L-BFGS
implementation used by Radford et al. [82] at ImageNet scale with similar accuracy. Additional details on
hyperparameters and more analyses are provided in Appendix D.3.
Two variants of this method are shown in Figure 13, one for which the the linear classiﬁer is initialized
randomly and another for which the linear classiﬁer is initialized with the zero-shot weights (denoted
warmstart). If the convex problem is solved then the initialization does not play a role. However we are using
mini-batch optimization and, in certain cases, terminating training before an optimum is reached.
Distillation. Network distillation [41] trains one network to match the outputs of another. We use this
technique to ﬁne-tune while matching the outputs of the zero-shot model with weights θ0. For a hyperparameter
α ∈[0, 1] and cross-entropy loss ℓwe ﬁne-tune θ according to the minimization objective
X
(xi,yi)∈Str
D
(1 −α) · ℓ(f(xi, θ), yi) + α · ℓ(f(xi, θ), f(xi, θ0)) .
(3)
Regularization towards zero-shot. We train a linear classiﬁer with an additional regularization term
which penalizes movement from the zero-shot classiﬁer’s weights. For a hyperparameter λ ∈{1 · 10−8, 5 ·
10−8, 1 · 107, ..., 5 · 102} we add the regularization term λ ∥W −Wzero-shot∥2
F where W is the linear classiﬁer
being ﬁne-tuned. In most cases this method performs slightly worse than distillation.
Finally, Figure 14 and Table 7 demonstrate that WiSE-FT achieves better accuracy than the recently
proposed CoOp method [112] on ImageNet and four derived distribution shifts. Instead of ﬁne-tuning network
parameters, CoOp instead learns continuous embedding for the language prompts. We note that CoOp and
WiSE-FT could be used in conjunction in future work. We compare with the ViT-B/16 section in Table 7 of
Zhou et al. [112]. For comparison we use the same CLIP model as CoOp and also train only on 16 images
per class. When end-to-end ﬁne-tuning we use 10 epochs and learning rate 10−5.
C.4
Changes in data augmentation
In the majority of our experiments we follow Radford et al. [82] in using minimal data augmentation. However,
Figure 14 recreates Figure 3 with the default ImageNet train augmentation used in PyTorch ImageNet Models
[101], which includes random cropping, horizontal ﬂipping and color jitter. As shown in Figure 14, we ﬁnd
similar trends with this stronger data augmentation. Further investigating the eﬀect of data augmentation
remains an interesting direction for future work.
ImageNet (IN)
INV2
IN-R
IN-A
IN Sketch
CoOp [112]
71.73
64.56
75.28
49.93
47.89
WiSE-FT (linear classiﬁere, α = 0.5)
73.02
65.19
77.63
49.81
49.09
WiSE-FT (end-to-end, α = 0.5)
72.38
65.29
78.47
51.07
49.72
Table 7: Comparing WiSE-FT with CoOp [112]. Both methods ﬁne-tune the ViT-B/16 CLIP model on 16
examples per class of ImageNet. Also see Figure 14.
30

--- Page 31 ---
65
70
ImageNet (top-1, %)
55
60
65
ImageNetV2 (top-1, %)
65
70
ImageNet (top-1, %)
65
70
75
80
ImageNet-R (top-1, %)
65
70
ImageNet (top-1, %)
35
40
45
50
ImageNet Sketch (top-1, %)
65
70
ImageNet (top-1, %)
35
40
45
50
ImageNet-A (top-1, %)
Linear ﬁt (CLIP zero-shot models)
Weight-space ensemble (linear classiﬁer)
Weight-space ensemble (end-to-end)
CoOp
CLIP zero-shot models
Weight-space ensemble with α = 0.5
CLIP ﬁne-tuned with a linear classiﬁer
CLIP ﬁne-tuned end-to-end
Figure 14: Comparing WiSE-FT with CoOp [112]. Both methods ﬁne-tune the ViT-B/16 CLIP model on 16
examples per class of ImageNet.
C.5
Accuracy improvements on reference datasets
Beyond robustness, Figure 16 demonstrates that WiSE-FT can provide accuracy improvements on ImageNet
and a number of datasets considered by Kornblith et al. [52]: CIFAR-10, CIFAR-100 [54], Describable
Textures [14], Food-101 [10], SUN397 [103], and Stanford Cars [53]. This is surprising as standard ﬁne-tuning
optimizes for low error on the reference distribution. Figure 16 supplements Table 2 by providing accuracy
information for all mixing coeﬃcients α.
In many application-speciﬁc scenarios, only a small amount of data is available for ﬁne-tuning. Accordingly,
we examine the performance of WiSE-FT when only k examples per class are used for ﬁne-tuning on the
seven aforementioned datasets (k = {1, 5, 10, 25, 50}). In contrast with Figure 16, we now ﬁne-tune only
the linear classiﬁer allowing for comprehensive experiments. Average results are shown in Figure 17, while
Figures 18 and 19 provide a breakdown for all datasets.
31

--- Page 32 ---
65
70
75
80
ImageNet (top-1, %)
50
55
60
65
Avg. accuracy on 5 distribution shifts
2
epochs
4
10
Hyperparameter:
Fix learning rate, vary number of epochs
LR = 1 · 10−7
LR = 1 · 10−6
LR = 3 · 10−6
LR = 1 · 10−5
LR = 2 · 10−5
LR = 3 · 10−5
65
70
75
80
ImageNet (top-1, %)
50
55
60
65
Avg. accuracy on 5 distribution shifts
1e-07
learning rate
1e-06 3e-06
1e-05
2e-05
Hyperparameter:
Fix number of epochs, vary learning rate
Epochs = 2
Epochs = 4
Epochs = 10
Epochs = 20
65
70
75
80
ImageNet (top-1, %)
50
55
60
65
Avg. accuracy on 5 distribution shifts
Iteration 250
1000
2500
Hyperparameter: terminating training early
Evaluation along optimization trajectory
Hyperparameter conﬁg (completed training)
Select early termination solutions
65
70
75
80
ImageNet (top-1, %)
50
55
60
65
Avg. accuracy on 5 distribution shifts
Weight-space ensembles (varied hyperparameters)
Weight-space ensemble
Hyperparameter conﬁg
CLIP zero-shot models
Linear ﬁt (CLIP zero-shot models)
Weight-space ensemble (end-to-end)
Figure 14. The robustness of ﬁne-tuned
models varies substantially under even
small changes in hyperparameters. Ap-
plying WiSE-FT addresses this brit-
tleness and can remove the trade-oﬀ
between accuracy on the reference and
shifted distributions. Results shown for
CLIP ViT-B/16 ﬁne-tuned with cosine-
annealing learning rate schedule and
ImageNet data augmentation from Py-
torch ImageNet Models [101].
0.0
0.25
0.5
0.75
1.0
α
75
80
85
Top-1 accuracy, %
+ 0.6pp
ImageNet
0.0
0.25
0.5
0.75
1.0
α
95
96
97
98
99
Top-1 accuracy, %
+ 0.7pp
CIFAR10
0.0
0.25
0.5
0.75
1.0
α
75
80
85
90
Top-1 accuracy, %
+ 1.1pp
CIFAR100
0.0
0.25
0.5
0.75
1.0
α
75
80
85
90
Top-1 accuracy, %
+ 1.7pp
Cars
0.0
0.25
0.5
0.75
1.0
α
50
55
60
65
70
75
80
85
Top-1 accuracy, %
+ 2.8pp
DTD
0.0
0.25
0.5
0.75
1.0
α
70
75
80
Top-1 accuracy, %
+ 2.5pp
SUN397
0.0
0.25
0.5
0.75
1.0
α
94
95
96
Top-1 accuracy, %
+ 1.6pp
Food101
Weight-space ensemble
CLIP ﬁne-tuned end-to-end
CLIP zero-shot
Figure 16: The accuracy of WiSE-FT (end-to-end) with mixing coeﬃcient α on ImageNet and a number of
datasets considered by Kornblith et al. [52]: CIFAR-10, CIFAR-100 [54], Describable Textures [14], Food-101
[10], SUN397 [103], and Stanford Cars [53].
32

--- Page 33 ---
1
5
10
25
50
Train samples per class
−5
0
5
10
Percentage points
Accuracy gain over
the zero-shot model
1
5
10
25
50
Train samples per class
0
10
20
Accuracy gain over
the linear classiﬁer
1
5
10
25
50
Train samples per class
−2
0
2
4
Accuracy gain over the
model that performs best on average
α = 0.0 (CLIP zero-shot)
α = 0.25
α = 0.5
α = 0.75
α = 1.0 (ﬁne-tuned linear classiﬁer)
Optimal α
Figure 17: WiSE-FT can improve accuracy over the linear classiﬁer and zero-shot model in the low data
regime. On the x-axis we consider k = {1, 5, 10, 25, 50} examples per class for ﬁne-tuning. On the y-axis we
display accuracy improvements of WiSE-FT averaged over seven datasets [17, 54, 14, 10, 103, 53]. For k = 1,
the zero-shot model outperforms the ﬁne-tuned linear classiﬁer, and ensembles closer to the zero-shot model
(small α) yield high performance. When more data is available, the reverse is true, and higher values of α
improve performance. Figures 18 and 19 display a breakdown for all datasets.
33

--- Page 34 ---
103
104
105
106
−10
−5
0
5
10
Accuracy gain over
zero-shot model
ImageNet
102
103
104
−10
−5
0
5
10
CIFAR-100
103
104
−10
−5
0
5
10
SUN397
103
−10
−5
0
5
10
15
Stanford Cars
103
104
105
106
−10
−5
0
5
10
15
20
25
30
Accuracy gain over
linear classiﬁer
102
103
104
−10
−5
0
5
10
15
20
25
103
104
−10
−5
0
5
10
15
20
25
103
−15
−10
−5
0
5
10
15
20
25
103
104
105
106
Number of train examples
−3
−2
−1
0
1
2
3
4
Accuracy gain over
best model
102
103
104
Number of train examples
−3
−2
−1
0
1
2
3
4
Optimal α
α = 0 (CLIP zero-shot)
α = 0.25
α = 0.5
α = 0.75
α = 1.0 (ﬁne-tuned linear classiﬁer)
103
104
Number of train examples
−3
−2
−1
0
1
2
3
4
5
6
103
Number of train examples
−3
−2
−1
0
1
2
3
4
5
Figure 18: WiSE-FT improves accuracy over the linear classiﬁer and zero-shot model in the low data regime.
On the x-axis we consider k = {1, 5, 10, 25, 50} examples per class and the full training set. On the y-axis we
consider the accuracy improvement of WiSE-FT over the (top) zero-shot model, (middle) ﬁne-tuned linear
classiﬁer, and (bottom) best of the zero-shot and ﬁne-tuned linear classiﬁer.
34

--- Page 35 ---
101
102
103
104
−3
−2
−1
0
1
2
3
Accuracy gain over
zero-shot model
CIFAR-10
102
103
−10
−5
0
5
10
15
20
25
30
Describable Textures
102
103
104
−3
−2
−1
0
1
2
Food-101
101
102
103
104
0
5
10
15
20
25
Accuracy gain over
linear classiﬁer
102
103
−30
−25
−20
−15
−10
−5
0
5
10
15
20
102
103
104
0
5
10
15
20
25
30
101
102
103
104
Number of train examples
−2
−1
0
1
Accuracy gain over
best model
102
103
Number of train examples
−4
−3
−2
−1
0
1
2
3
4
5
6
7
Optimal α
α = 0 (CLIP zero-shot)
α = 0.25
α = 0.5
α = 0.75
α = 1.0 (ﬁne-tuned linear classiﬁer)
102
103
104
Number of train examples
−1
0
1
Figure 19: WiSE-FT improves accuracy over the linear classiﬁer and zero-shot model in the low data regime.
On the x-axis we consider k = {1, 5, 10, 25, 50} examples per class and the full training set. On the y-axis we
consider the accuracy improvement of WiSE-FT over the (top) zero-shot model, (middle) ﬁne-tuned linear
classiﬁer, and (bottom) best of the zero-shot and ﬁne-tuned linear classiﬁer.
35

--- Page 36 ---
104
3 · 104
105
GPU hours estimate for training model
0
2
4
6
8
Avg. accuracy gain (pp)
104
3 · 104
105
GPU hours estimate for training model
0
2
4
6
8
Avg. accuracy on 5 distribution shifts
104
3 · 104
105
GPU hours estimate for training model
0
2
4
6
8
Decreasing at most 0 pp ID
Decreasing at most 0.1 pp ID
Decreasing at most 1.0 pp ID
Figure 20: WiSE-FT provides beneﬁts for all CLIP models. Accuracy under distribution shift can be improved
relative to the linear classiﬁer with less than ϵ ∈{0, 0.1, 1} percentage points (pp) loss in accuracy on the
reference distribution, across orders of magnitude of training compute. The CLIP model RN50x64 requires
the most GPU hours to train.
65
67
69
71
73
75
77
79
81
ImageNet (top-1, %)
60
62
64
66
68
70
72
74
ImageNetV2 (top-1, %)
90
91
92
93
94
95
ImageNet (class-subsampled) (top-1, %)
60
62
64
66
68
70
72
74
76
78
80
ImageNet-R (top-1, %)
65
70
75
80
ImageNet (top-1, %)
40
45
50
55
ImageNet Sketch (top-1, %)
70
75
80
85
ImageNet (class-subsampled) (top-1, %)
40
45
50
55
ObjectNet (top-1, %)
90
92
94
96
ImageNet (class-subsampled) (top-1, %)
30
32
34
36
38
40
42
44
46
48
50
52
54
56
ImageNet-A (top-1, %)
CLIP zero-shot models
Linear ﬁt (CLIP zero-shot models)
CLIP ﬁne-tuned end-to-end
CLIP ﬁne-tuned with a linear classiﬁer
(prior work)
Weight-space ensemble (end-to-end)
Weight-space ensemble (linear classiﬁer)
Weight-space ensemble with α = 0.5
Standard ImageNet models
Linear ﬁt (standard ImageNet models)
Trained with more data
Existing robustness interventions
y = x
Figure 21: WiSE-FT improves accuracy on the reference and shifted distributions for numerous distribution
shifts with a smaller CLIP ViT-B/16 model.
C.6
Robustness across scales of pre-training compute
The strong correlation between standard test accuracy and accuracy under distribution shift holds from low
to high performing models. This oﬀers the opportunity to explore robustness for smaller, easy to run models.
Our exploration began with the lowest accuracy CLIP models and similar trends held at scale. Figure 20
shows improved accuracy under distribution shift with minimal loss on reference performance across orders of
magnitude of pre-training compute with WiSE-FT when ﬁne-tuning a linear classiﬁer. Moreover, in Figure 21
we recreate the experimental results for ImageNet and ﬁve associated distribution shifts with a smaller CLIP
ViT-B/16 model, ﬁnding similar trends. Recall that unless otherwise mentioned our experiments use the
larger CLIP model (ViT-L/14@336px).
36

--- Page 37 ---
Distribution shifts
Avg
Avg
IN (ref.)
IN-V2
IN-R
IN-Sketch
ObjectNet
IN-A
shifts
ref., shifts
CLIP ViT-B/16 [82]
Zero-shot
68.3
61.9
77.6
48.2
53.0
49.8
58.1
63.2
Standard ﬁne-tuning
81.3
70.9
65.6
46.3
49.6
36.7
53.8
67.5
WiSE-FT (α=0.5)
81.7
72.8
78.7
53.9
57.3
52.2
63.0
72.3
WiSE-FT (opt. α)
82.6
73.2
80.1
54.1
57.7
54.6
63.5
72.3
CLIP ViT-L/14@336px [82]
Zero-shot
76.6
70.5
89.0
60.9
68.5
77.6
73.3
74.9
Standard ﬁne-tuning
86.2
76.8
79.8
57.9
63.3
65.4
68.6
77.4
WiSE-FT (α=0.5)
86.8
79.5
89.4
64.7
71.1
79.9
76.9
81.8
WiSE-FT (opt. α)
87.1
79.5
90.3
65.0
72.1
81.0
77.4
81.9
ALIGN [45]
Zero-shot
76.4
70.1
92.1
67.9
67.2
75.9
74.6
75.5
Standard ﬁne-tuning
88.2
80.1
88.5
69.1
61.0
76.3
75.0
81.6
WiSE-FT (α=0.5)
86.3
79.2
93.0
71.1
67.8
81.0
78.4
82.3
WiSE-FT (opt. α)
88.3
80.4
93.3
71.1
68.6
81.0
78.4
82.8
JFT pre-trained ViT-H [21]
Zero-shot
72.9
66.1
85.9
57.0
59.2
58.4
65.3
69.1
Standard ﬁne-tuning
85.4
77.6
84.9
62.8
63.1
60.8
69.8
77.6
WiSE-FT (α=0.5)
82.9
75.4
89.3
63.8
65.8
66.2
72.1
77.5
WiSE-FT (opt. α)
85.4
77.8
89.3
64.5
66.0
66.6
72.5
78.6
BASIC-M [77]
Zero-shot
81.4
74.1
90.6
67.4
73.5
66.7
74.5
78.0
Standard ﬁne-tuning
86.2
77.8
84.9
64.3
75.3
63.7
73.2
79.7
WiSE-FT (α=0.5)
85.6
78.5
90.2
68.6
78.0
71.1
77.3
81.4
WiSE-FT (opt. α)
86.2
78.6
91.1
68.8
78.0
71.4
77.4
81.4
BASIC-L [77]
Zero-shot
85.6
80.5
95.7
76.2
82.3
85.7
84.1
84.8
Standard ﬁne-tuning
87.5
79.8
84.3
68.0
77.4
72.1
76.3
81.9
WiSE-FT (α=0.5)
87.9
81.6
94.5
73.6
84.1
83.2
83.4
85.7
WiSE-FT (opt. α)
87.9
82.1
96.0
76.5
84.9
86.5
85.0
86.2
Table 8:
WiSE-FT accuracy on ImageNet and derived distribution shifts for various models ﬁne-tuned
end-to-end. Avg shifts displays the mean performance among the ﬁve distribution shifts, while Avg reference,
shifts shows the average of ImageNet (reference) and Avg shifts. For optimal α, we choose the single mixing
coeﬃcient that maximizes the column.
C.7
WiSE-FT and additional models
Table 8 summarizes the results for the main models we study, CLIP, ALIGN, BASIC and a ViT model
pre-trained on JFT. Details are provided in the subsequent sections.
C.7.1
ALIGN
In addition to CLIP, we show WiSE-FT to be eﬀective for an additional zero-shot model, ALIGN [45]. Results
are shown in Figure 22 and Table 9. End-to-end ﬁne-tuning is performed using AdamW, which we found to
perform slightly better than SGD + momentum. The model is ﬁne-tuned for 40,000 steps with a batch size
of 512, a maximum learning rate of 5 × 10−6, and weight decay of 0.1. The learning rate schedule consisted
of 500 steps of linear warmup followed by cosine decay. The linear classiﬁer is trained using L-BFGS and no
label smoothing. All models are evaluated on 360 × 360 pixel crops obtained by taking the central 87.5%
square region of the test set images. For end-to-end ﬁne-tuning, we take 299 × 299 pixel Inception-style
random crops from the original ImageNet images during training; for linear classiﬁer training, we use the same
37

--- Page 38 ---
72
74
76
78
80
82
84
86
88
ImageNet (top-1, %)
64
66
68
70
72
74
76
78
80
ImageNetV2 (top-1, %)
72
74
76
78
80
82
84
86
88
ImageNet (top-1, %)
80
82
84
86
88
90
92
ImageNet-R (top-1, %)
72
74
76
78
80
82
84
86
88
ImageNet (top-1, %)
58
60
62
64
66
68
70
72
ImageNet Sketch (top-1, %)
72
74
76
78
80
82
84
86
88
ImageNet (top-1, %)
56
58
60
62
64
66
68
ObjectNet (top-1, %)
72
74
76
78
80
82
84
86
88
ImageNet (top-1, %)
66
68
70
72
74
76
78
80
ImageNet-A (top-1, %)
Weight-space ensemble (linear classiﬁer)
Weight-space ensemble (end-to-end)
Varying the L2 regularization coeﬃcient
ALIGN zero-shot
ALIGN ﬁne-tuned with a linear classiﬁer
ALIGN ﬁne-tuned end-to-end
Weight-space ensemble with α = 0.5
Figure 22: WiSE-FT applied to ALIGN [45]. We also show the eﬀect of varying the L2 regularization strength
for linear classiﬁer ﬁne-tuning.
preprocessing as at evaluation time. The weights of the zero-shot model are calibrated using temperature
scaling on the ImageNet training set before performing WiSE-FT.
38

--- Page 39 ---
Distribution shifts
Avg
Avg
IN (reference)
IN-V2
IN-R
IN-Sketch
ObjectNet
IN-A
shifts
reference, shifts
WiSE-FT, end-to-end
α=0.00
76.4
70.1
92.1
67.9
67.2
75.9
74.6
75.5
α=0.05
77.9
71.6
92.5
68.5
67.8
76.9
75.5
76.7
α=0.10
79.2
73.0
92.7
69.0
68.2
77.9
76.2
77.7
α=0.15
80.5
74.3
92.9
69.5
68.5
78.6
76.8
78.7
α=0.20
81.6
75.4
93.0
70.0
68.6
79.2
77.2
79.4
α=0.25
82.7
76.3
93.2
70.3
68.6
79.8
77.6
80.2
α=0.30
83.5
77.1
93.2
70.5
68.6
80.1
77.9
80.7
α=0.35
84.4
77.8
93.3
70.7
68.6
80.3
78.1
81.2
α=0.40
85.2
78.3
93.3
70.8
68.3
80.6
78.3
81.8
α=0.45
85.8
78.8
93.2
71.0
68.1
80.8
78.4
82.1
α=0.50
86.3
79.2
93.0
71.1
67.8
81.0
78.4
82.3
α=0.55
86.7
79.6
92.8
71.1
67.3
81.0
78.4
82.6
α=0.60
87.1
79.7
92.6
71.1
66.8
80.8
78.2
82.7
α=0.65
87.5
80.0
92.3
71.0
66.3
80.6
78.0
82.8
α=0.70
87.7
80.2
92.0
70.9
65.8
80.4
77.9
82.8
α=0.75
87.9
80.4
91.5
70.7
65.1
79.9
77.5
82.7
α=0.80
88.0
80.3
91.1
70.5
64.3
79.4
77.1
82.5
α=0.85
88.2
80.4
90.5
70.2
63.5
78.9
76.7
82.5
α=0.90
88.3
80.4
89.9
69.9
62.8
78.2
76.2
82.2
α=0.95
88.3
80.3
89.2
69.5
61.8
77.3
75.6
81.9
α=1.00
88.2
80.1
88.5
69.1
61.0
76.3
75.0
81.6
WiSE-FT, linear classiﬁer
α=0.00
76.4
70.1
92.1
68.0
67.2
75.8
74.6
75.5
α=0.05
77.5
71.1
92.3
68.3
67.4
76.3
75.1
76.3
α=0.10
78.6
72.0
92.3
68.6
67.6
76.5
75.4
77.0
α=0.15
79.5
73.0
92.4
69.0
67.7
76.9
75.8
77.7
α=0.20
80.3
73.5
92.4
69.1
67.8
77.3
76.0
78.2
α=0.25
81.1
74.2
92.4
69.2
67.8
77.3
76.2
78.7
α=0.30
81.8
74.6
92.4
69.2
67.8
77.5
76.3
79.0
α=0.35
82.4
75.1
92.4
69.1
67.8
77.6
76.4
79.4
α=0.40
82.9
75.5
92.2
69.0
67.7
77.8
76.4
79.7
α=0.45
83.4
75.8
92.2
68.9
67.4
77.7
76.4
79.9
α=0.50
83.7
76.1
91.9
68.8
67.3
77.6
76.3
80.0
α=0.55
84.1
76.0
91.8
68.6
67.1
77.4
76.2
80.2
α=0.60
84.5
76.3
91.6
68.5
66.8
77.0
76.0
80.2
α=0.65
84.7
76.4
91.3
68.2
66.4
76.9
75.8
80.2
α=0.70
84.9
76.4
91.0
68.0
66.2
76.5
75.6
80.2
α=0.75
85.1
76.4
90.6
67.6
65.9
76.2
75.3
80.2
α=0.80
85.2
76.4
90.2
67.3
65.5
75.9
75.1
80.2
α=0.85
85.2
76.5
89.7
66.8
65.0
75.3
74.7
80.0
α=0.90
85.2
76.3
89.2
66.3
64.4
74.9
74.2
79.7
α=0.95
85.2
76.0
88.6
65.7
63.8
74.4
73.7
79.5
α=1.00
85.1
75.7
87.8
65.1
63.2
73.7
73.1
79.1
Table 9:
WiSE-FT accuracy on the reference and shifted distributions for various values of the mixing
coeﬃcient α. Results shown for ALIGN, ﬁne-tuned end-to-end (top) and with a linear classiﬁer (bottom).
Note that α=0.0 corresponds to the zero-shot model, while α = 1.0 corresponds to standard ﬁne-tuning. Avg
shifts displays the mean performance among the ﬁve distribution shifts, while Avg reference, shifts shows the
average of ImageNet (reference) and Avg shifts.
39

--- Page 40 ---
68
70
72
74
76
78
80
82
84
ImageNet (top-1, %)
58
60
62
64
66
68
70
72
74
76
78
ImageNetV2 (top-1, %)
68
70
72
74
76
78
80
82
84
ImageNet (top-1, %)
78
80
82
84
86
88
ImageNet-R (top-1, %)
68
70
72
74
76
78
80
82
84
ImageNet (top-1, %)
50
52
54
56
58
60
62
64
ImageNet Sketch (top-1, %)
68
70
72
74
76
78
80
82
84
ImageNet (top-1, %)
52
54
56
58
60
62
64
66
ObjectNet (top-1, %)
68
70
72
74
76
78
80
82
84
ImageNet (top-1, %)
50
52
54
56
58
60
62
64
66
ImageNet-A (top-1, %)
Weight-space ensemble (linear classiﬁer)
Varying the L2 regularization (linear classiﬁer)
Weight-space ensemble (end-to-end)
ViT-H/14 (JFT) zero-shot
ViT-H/14 (JFT) ﬁne-tuned with a linear classiﬁer
ViT-H/14 (JFT) ﬁne-tuned end-to-end
Weight-space ensemble with α = 0.5
Figure 23: WiSE-FT applied to ViT-H/14 [21] pre-trained on JFT. We also show the eﬀect of varying the L2
regularization strength for linear classiﬁer ﬁne-tuning.
C.7.2
JFT pre-training
We also investigate whether WiSE-FT can provide gains for models trained using a standard image classiﬁcation
objective on the JFT-300M dataset [93]. Results are shown in Figure 23 and Table 10. For 973/1000 ImageNet
classes, we were able to manually identify a corresponding class from the 18K classes in JFT. We use this
mapping between ImageNet and JFT classes to obtain zero-shot ImageNet weights from the ﬁnal layer
weights of the pre-trained ViT-H/14 model from Dosovitskiy et al. [21]. We also train a linear classiﬁer on the
ﬁxed penultimate layer of the same ViT-H/14 model using L-BFGS without label smoothing with softmax
cross-entropy loss, and ﬁne-tune end-to-end using AdamW with maximum learning rate 5 · 10−6 and weight
decay 0.1 for 20k iterations at batch size 512 with sigmoid cross-entropy loss. As for CLIP models, our
learning rate schedule consists of 500 steps of linear warmup followed by cosine decay. All ViT-H/14 models
are trained and evaluated on 224 × 224 pixel images. For fair evaluation, we prevent ﬁne-tuned solutions
from predicting the 27 classes with no plausible corresponding JFT class at all points on the WiSE-FT curve
but still include these points in the denominator when computing accuracy.
40

--- Page 41 ---
Distribution shifts
Avg
Avg
IN (ref.)
IN-V2
IN-R
IN-Sketch
ObjectNet
IN-A
shifts
ref., shifts
WiSE-FT, edn-to-end
α=0.00
72.9
66.1
85.9
57.0
59.2
58.4
65.3
69.1
α=0.05
74.1
67.3
86.4
57.9
60.4
59.9
66.4
70.2
α=0.10
75.3
68.3
86.9
58.8
61.2
60.8
67.2
71.2
α=0.15
76.5
69.5
87.4
59.7
62.2
61.7
68.1
72.3
α=0.20
77.5
70.8
87.9
60.5
63.0
62.8
69.0
73.2
α=0.25
78.5
71.6
88.3
61.2
63.8
63.5
69.7
74.1
α=0.30
79.6
72.5
88.6
61.8
64.4
64.3
70.3
74.9
α=0.35
80.6
73.5
88.9
62.3
64.9
64.8
70.9
75.8
α=0.40
81.5
74.1
89.1
62.8
65.3
65.4
71.3
76.4
α=0.45
82.2
74.8
89.2
63.3
65.6
65.8
71.7
77.0
α=0.50
82.9
75.4
89.3
63.8
65.8
66.2
72.1
77.5
α=0.55
83.4
75.9
89.3
64.0
66.0
66.3
72.3
77.8
α=0.60
83.9
76.4
89.3
64.3
66.0
66.6
72.5
78.2
α=0.65
84.3
76.8
89.1
64.5
65.9
66.4
72.5
78.4
α=0.70
84.7
77.1
88.9
64.5
65.8
66.0
72.5
78.6
α=0.75
84.9
77.4
88.5
64.5
65.6
65.3
72.3
78.6
α=0.80
85.2
77.6
88.1
64.4
65.2
64.8
72.0
78.6
α=0.85
85.3
77.8
87.5
64.1
64.7
63.8
71.6
78.4
α=0.90
85.4
77.8
86.8
63.7
64.4
63.2
71.2
78.3
α=0.95
85.4
77.8
85.9
63.3
63.9
62.2
70.6
78.0
α=1.00
85.4
77.6
84.9
62.8
63.1
60.8
69.8
77.6
WiSE-FT, linear classiﬁer
α=0.00
72.9
66.1
85.9
57.0
59.2
58.4
65.3
69.1
α=0.05
74.0
67.3
86.3
57.5
60.3
59.2
66.1
70.0
α=0.10
75.1
68.3
86.7
58.1
61.2
60.1
66.9
71.0
α=0.15
76.1
69.1
87.0
58.5
61.8
60.8
67.4
71.8
α=0.20
77.1
70.0
87.3
59.0
62.4
61.1
68.0
72.5
α=0.25
78.0
71.0
87.5
59.5
63.0
61.6
68.5
73.2
α=0.30
78.8
71.7
87.7
59.8
63.3
61.9
68.9
73.8
α=0.35
79.6
72.2
87.8
60.1
63.6
62.2
69.2
74.4
α=0.40
80.3
72.9
87.9
60.4
63.6
62.3
69.4
74.8
α=0.45
80.9
73.4
88.0
60.5
63.8
62.5
69.6
75.2
α=0.50
81.5
73.8
88.0
60.7
63.9
62.5
69.8
75.7
α=0.55
81.9
74.1
88.0
60.8
63.7
62.5
69.8
75.8
α=0.60
82.4
74.4
87.9
60.8
63.5
62.4
69.8
76.1
α=0.65
82.8
74.7
87.8
60.7
63.2
62.3
69.7
76.2
α=0.70
83.1
75.0
87.6
60.7
63.0
62.0
69.7
76.4
α=0.75
83.4
75.2
87.4
60.5
62.7
61.8
69.5
76.5
α=0.80
83.6
75.4
87.1
60.2
62.4
61.4
69.3
76.4
α=0.85
83.7
75.4
86.7
59.8
61.9
60.7
68.9
76.3
α=0.90
83.9
75.4
86.3
59.4
61.4
60.3
68.6
76.2
α=0.95
84.0
75.3
85.7
58.9
61.0
59.4
68.1
76.0
α=1.00
84.0
75.1
85.1
58.3
60.4
58.8
67.5
75.8
Table 10:
WiSE-FT accuracy on the reference and shifted distributions for various values of the mixing
coeﬃcient α. Results shown for ViT-H/14 pre-trained on JFT-300M, ﬁne-tuned end-to-end (top) and with a
linear classiﬁer (bottom). Note that α=0.0 corresponds to the zero-shot model, while α = 1.0 corresponds to
standard ﬁne-tuning. Avg shifts displays the mean performance among the ﬁve distribution shifts, while Avg
reference, shifts shows the average of ImageNet (reference) and Avg shifts.
41

--- Page 42 ---
C.7.3
BASIC
We apply WiSE-FT to BASIC [77], ﬁne-tuning both the image and text encoder with a contrastive loss on
half of the ImageNet training data, as in Pham et al. [77]. Results are shown in Figure 24 and Tables 11 and
12.
85
86
87
88
ImageNet (top-1, %)
79
80
81
82
ImageNetV2 (top-1, %)
85
86
87
88
ImageNet (top-1, %)
80
82
84
86
88
90
92
94
96
ImageNet-R (top-1, %)
85
86
87
88
ImageNet (top-1, %)
65
67
69
71
73
75
77
ImageNet Sketch (top-1, %)
85
86
87
88
ImageNet (top-1, %)
76
78
80
82
84
ObjectNet (top-1, %)
85
86
87
88
ImageNet (top-1, %)
70
72
74
76
78
80
82
84
86
88
ImageNet-A (top-1, %)
Weight-space ensemble (end-to-end)
BASIC-L zero-shot
BASIC-L ﬁne-tuned end-to-end
Weight-space ensemble with α = 0.5
Figure 24: WiSE-FT improves accuracy relative to the ﬁne-tuned model on ImageNet and ﬁve derived
distribution shifts for BASIC-L [77] using ImageNet class names to construct the zero-shot classiﬁer.
42

--- Page 43 ---
Distribution shifts
Avg
Avg
IN (ref.)
IN-V2
IN-R
IN-Sketch
ObjectNet
IN-A
shifts
ref., shifts
α=0.00
81.4
74.1
90.6
67.4
73.5
66.7
74.5
78.0
α=0.05
82.2
75.0
90.8
67.9
74.6
67.8
75.2
78.7
α=0.10
82.8
75.9
90.9
68.2
75.4
68.5
75.8
79.3
α=0.15
83.3
76.4
91.0
68.4
76.2
69.3
76.3
79.8
α=0.20
83.8
76.8
91.0
68.6
76.9
70.0
76.7
80.2
α=0.25
84.1
77.1
91.1
68.7
77.4
70.5
77.0
80.5
α=0.30
84.5
77.4
91.0
68.8
77.7
70.8
77.1
80.8
α=0.35
84.9
77.9
90.8
68.8
77.8
71.3
77.3
81.1
α=0.40
85.2
78.1
90.7
68.7
77.9
71.3
77.3
81.2
α=0.45
85.4
78.3
90.5
68.7
78.0
71.4
77.4
81.4
α=0.50
85.6
78.5
90.2
68.6
78.0
71.1
77.3
81.4
α=0.55
85.8
78.5
89.9
68.4
78.0
70.6
77.1
81.4
α=0.60
85.9
78.4
89.5
68.1
78.0
70.5
76.9
81.4
α=0.65
86.0
78.5
89.1
67.7
77.8
70.3
76.7
81.3
α=0.70
86.1
78.5
88.8
67.3
77.6
69.7
76.4
81.2
α=0.75
86.2
78.6
88.4
67.0
77.3
69.2
76.1
81.2
α=0.80
86.2
78.5
87.8
66.6
77.1
68.3
75.7
81.0
α=0.85
86.2
78.5
87.2
66.0
76.7
67.5
75.2
80.7
α=0.90
86.2
78.4
86.5
65.5
76.2
66.4
74.6
80.4
α=0.95
86.2
78.2
85.7
65.0
75.8
65.3
74.0
80.1
α=1.00
86.2
77.8
84.9
64.3
75.3
63.7
73.2
79.7
Table 11:
WiSE-FT accuracy on the reference and shifted distributions for various values of the mixing
coeﬃcient α. Results shown for BASIC-M using ImageNet class names. Note that α=0.0 corresponds to the
zero-shot model, while α = 1.0 corresponds to standard ﬁne-tuning. Avg shifts displays the mean performance
among the ﬁve distribution shifts, while Avg reference, shifts shows the average of ImageNet (reference) and
Avg shifts.
43

--- Page 44 ---
Distribution shifts
Avg
Avg
IN (ref.)
IN-V2
IN-R
IN-Sketch
ObjectNet
IN-A
shifts
ref., shifts
α=0.00
85.6
80.5
95.7
76.2
82.3
85.7
84.1
84.8
α=0.05
86.4
81.2
95.8
76.5
83.6
86.0
84.6
85.5
α=0.10
86.9
81.7
96.0
76.5
84.3
86.5
85.0
86.0
α=0.15
87.3
81.9
96.0
76.4
84.6
86.3
85.0
86.2
α=0.20
87.5
82.1
95.9
76.1
84.8
86.1
85.0
86.2
α=0.25
87.6
82.1
95.7
75.8
84.9
86.0
84.9
86.2
α=0.30
87.7
82.1
95.6
75.4
84.9
85.7
84.7
86.2
α=0.35
87.8
82.0
95.4
75.0
84.9
84.9
84.4
86.1
α=0.40
87.8
81.8
95.1
74.5
84.7
84.5
84.1
85.9
α=0.45
87.8
81.6
94.9
74.0
84.5
83.8
83.8
85.8
α=0.50
87.9
81.6
94.5
73.6
84.1
83.2
83.4
85.7
α=0.55
87.8
81.4
94.1
73.1
83.9
82.6
83.0
85.4
α=0.60
87.9
81.3
93.6
72.7
83.6
82.0
82.6
85.2
α=0.65
87.9
81.3
93.0
72.3
83.2
81.3
82.2
85.1
α=0.70
87.8
81.2
92.3
71.8
82.7
80.5
81.7
84.8
α=0.75
87.8
81.0
91.5
71.4
82.0
79.6
81.1
84.4
α=0.80
87.9
81.0
90.4
70.7
81.3
78.5
80.4
84.2
α=0.85
87.8
80.8
89.1
70.1
80.6
77.5
79.6
83.7
α=0.90
87.7
80.6
87.7
69.5
79.6
76.1
78.7
83.2
α=0.95
87.5
80.3
86.1
68.8
78.5
74.5
77.6
82.5
α=1.00
87.5
79.8
84.3
68.0
77.4
72.1
76.3
81.9
Table 12:
WiSE-FT accuracy on the reference and shifted distributions for various values of the mixing
coeﬃcient α. Results shown for BASIC-L using ImageNet class names. Note that α=0.0 corresponds to the
zero-shot model, while α = 1.0 corresponds to standard ﬁne-tuning. Avg shifts displays the mean performance
among the ﬁve distribution shifts, while Avg reference, shifts shows the average of ImageNet (reference) and
Avg shifts.
44

--- Page 45 ---
80
85
90
Avg. in-distribution accuracy
60
65
70
75
80
Avg. accuracy on 5 distribution shifts
Comparable IN-trained model
80
85
90
Avg. in-distribution accuracy
60
65
70
75
80
Avg. accuracy on 5 distribution shifts
Strong IN-trained model
CLIP zero-shot
Linear ﬁt (CLIP zero-shot)
CLIP ﬁne-tuned end-to-end
Weight-space ensemble (end-to-end)
NS EﬃcientNet-B6
Ensemble with NS EﬃcientNet-B6
NS EﬃcientNet-L2
Ensemble with NS EﬃcientNet-L2
Figure 25: Ensembling with a zero-shot model improves accuracy under distribution shift of an
independently trained model. (Left) Output-space ensembling with an independently trained model
(NoisyStudent EﬃcientNet-B6) with comparable performance to the end-to-end ﬁne-tuned model on the
reference distribution. (Right) Output-space ensembling with an independently trained model with strong
performance on the reference distribution (NoisyStudent EﬃcientNet-L2). Results averaged over the ﬁve
distribution shifts as in Figure 1.
C.8
Ensembling zero-shot CLIP with independently trained models
So far we have shown that a zero-shot model can be used to improve performance under distribution shift
of the derived ﬁne-tuned model. Here, we investigate whether this improvement is speciﬁc to ﬁne-tuned
models. On the contrary, we ﬁnd that the performance under distribution shift of independently trained
models improves when ensembling with robust models. Note that in the general case where the models
being ensembled have diﬀerent architectures, we are unable to perform weight-space ensembling; instead, we
ensemble the outputs of each model. This increases the computational cost of inference, in contrast to the
results shown in Section 4.
Concretely, we ensemble zero-shot CLIP with two Noisy Student EﬃcientNet models [104, 96]: (i) EﬃcientNet-
B6 (Figure 25, left), with performance on the reference distribution comparable to the end-to-end ﬁne-tuned
CLIP model; and (ii) EﬃcientNet-L2 (Figure 25, right), the strongest model available on PyTorch ImageNet
Models [101]. In both cases, we observe substantial improvements from ensembling—13.6 pp and 6.9 pp in
average accuracy under distribution shift without reducing performance on the reference dataset. Further
results are shown in Table 13.
D
Experimental details
D.1
CLIP zero-shot
This section extends Section 2 with more details on inference with the CLIP zero-shot model. First, in
all settings we use the CLIP model ViT-L/14@336px, except when explicitly mentioned otherwise. Second,
CLIP learns a temperature parameter which is factored into the learned weight matrix Wzero-shot described
in Section 2. Finally, to construct Wzero-shot we ensemble the 80 prompts provided by CLIP at https:
//github.com/openai/CLIP. However, we manually engineer prompts for ﬁve datasets: WILDS-FMoW,
WILDS-iWildCam, Stanford Cars, Describable Textures and Food-101, which are found in the code.
D.2
End-to-end ﬁne-tuning
Two important experimental details for end-to-end ﬁne-tuning are as follows:
45

--- Page 46 ---
Distribution shifts
Avg
Avg
IN (reference) IN-V2 IN-R IN-Sketch ObjectNet IN-A shifts reference, shifts
CLIP
End-to-end ﬁne-tuned
86.2
76.8
79.8
57.9
63.3
65.4
68.6
77.4
WSE (α=0.75)
87.0
78.8
86.1
62.5
68.1
75.2
74.1
80.5
WSE (α=0.5)
86.8
79.5
89.4
64.7
71.1
79.9
76.9
81.8
WSE (α=0.4)
86.2
79.2
89.9
65.0
71.9
80.7
77.3
81.8
WSE (optimal α)
87.1
79.5
90.3
65.0
72.1
81.0
77.6
82.3
NS EﬃcientNet-B6
No ensemble
86.5
77.7
65.6
47.8
58.3
62.3
62.3
74.4
OSE (α=0.75)
87.0
78.8
86.4
56.7
66.5
75.9
72.9
80.0
OSE (α=0.5)
86.2
78.7
89.2
63.8
69.3
78.6
75.9
81.1
OSE (α=0.4)
84.3
77.2
89.5
63.8
69.7
79.0
75.8
80.0
OSE (optimal α)
87.1
79.3
89.7
63.8
69.7
79.3
76.4
81.8
NS EﬃcientNet-L2
No ensemble
88.3
80.8
74.6
47.6
69.8
84.7
71.5
79.9
OSE (α=0.75)
88.6
81.6
88.0
53.4
72.2
87.1
76.5
82.5
OSE (α=0.5)
87.4
80.6
90.2
63.4
73.1
86.5
78.8
83.1
OSE (α=0.4)
85.2
78.5
90.5
63.9
72.6
86.0
78.3
81.8
OSE (optimal α)
88.6
81.7
90.5
63.9
73.1
87.1
79.3
83.9
Table 13: Accuracy of various independently trained models ensembled with CLIP on ImageNet and derived
distribution shifts. OSE denotes output-space ensembling. Avg shifts displays the mean performance among
the ﬁve distribution shifts, while Avg reference, shifts shows the average of ImageNet (reference) and Avg
shifts.
• We initialize the ﬁnal classiﬁcation layer with the zero-shot classiﬁer used by CLIP. We scale the zero-
shot classiﬁer weights by the temperature parameter of the pre-trained CLIP model at initialization,
and do not include a temperature parameter during ﬁne-tuning.
• As the zero-shot classiﬁer expects the outputs of the image-encoder g to be normalized, we continue to
normalize the outputs of g during ﬁne-tuning.
When ﬁne-tuning end-to-end, unless otherwise mentioned, we use the AdamW optimizer [61, 76] and choose
the largest batch size such that the model ﬁts into 8 GPUs (512 for ViT-B/16). Unless otherwise mentioned,
we use the default PyTorch AdamW hyperparameters β1 = 0.9, β2 = 0.999, ϵ = 10−8, weight decay of 0.1
and a cosine-annealing learning rate schedule [60] with 500 warm-up steps. Unless otherwise mentioned we
use a learning rate of 3 × 10−5, gradient clipping at global norm 1 and ﬁne-tune for a total of 10 epochs.
Additionally, unless otherwise mentioned we use the same data augmentations as [82], randomly cropping a
square from resized images with the largest dimension being 336 pixels for ViT-L/14@336px and 224 for the
remaining models.
D.3
Fine-tuning a linear classiﬁer
This section extends the description of linear classiﬁer training from Appendix C.3 with details on hyperpa-
rameters and additional analyses. In each of the four regularization strategies—no regularization, weight
decay, L1 regularization, and label smoothing—we run 64 hyperparameter conﬁgurations. For each trial,
mini-batch size is drawn uniformly from {64, 128, 256} and learning rate is set to 10−β with β chosen uniformly
at random from the range [0, 6]. Hyperparameters for each regularization strategy are as follows: (i) The
weight decay coeﬃcient is set to 10−λ where λ is chosen uniformly at random from [0, 4] for each trial; (ii)
The L1 regularization coeﬃcient is set to 10−λ where λ is chosen uniformly at random from [4, 8] for each
trial; (iii) The label smoothing [71] coeﬃcient λ is chosen uniformly at random from [0, 0.25] for each trial.
The linear classiﬁer used for ensembling attains the best performance in-distribution. The hyperparameters
46

--- Page 47 ---
75
80
85
90
ImageNet (class-subsampled) (top-1, %)
64
66
68
70
72
ObjectNet (top-1, %)
75
80
85
90
ImageNet (class-subsampled) (top-1, %)
64
66
68
70
72
ObjectNet (top-1, %)
CLIP zero-shot
Linear ﬁt (CLIP zero-shot)
CLIP ﬁne-tuned with a linear classiﬁer
Figure 26: Eﬀective robustness scatter plots for ObjectNet, with and without adapting to class shift. Left:
Using ImageNet class names to construct the zero-shot classiﬁer. Right: Using ObjectNet class names to
construct the zero-shot classiﬁer.
from this trial are then used in the distillation and regularization experiments described in Appendix C.3. In
the low-data regime (Section C.5), this process is repeated for each k and dataset.
When training linear classiﬁers with k images per class as in Section C.5 the maximum number of epochs T is
scaled approximately inversely proportional to the amount of data removed (e.g., with half the data we train
for twice as many epochs so the number of iterations is consistent). To choose the T we use default PyTorch
AdamW hyperparameters (learning rate 0.001, weight decay 0.01) and double the number of epochs until
performance saturates. For each random hyperparameter run we choose the epochs uniformly from {1, ..., T}.
D.4
ObjectNet
The zero-shot models in Table 1 use the ImageNet class names instead of the ObjectNet class names. However,
this adaptation to class shift improves performance by 2.3% [82]. Out of the ﬁve datasets used for the
majority of the experiments in Section 3, ObjectNet is the only dataset for which this is possible. In Figure
26 we compare weight-space ensembles with and without adaptation to class shift.
E
Diversity measures
Let S = {(x(i), y(i)), 1 ≤i ≤N} be a classiﬁcation set with input data x(i) and labels y(i) ∈{1, ..., C}, where
C is the number of classes. A classiﬁer f is a function that maps inputs x to logits f(x) ∈RC, yielding
predictions ˆy = arg max1≤c≤C f(x)c. We consider measures of diversity M(f, g, S) between two classiﬁers f
and g and the dataset S. For simplicity, ˆy(i)
f
is used to denote the predictions from classiﬁer f given inputs
x(i) (and similarly for g).
Prediction Diversity (PD). One of the most intuitive ways to measure diversity between pairs of classiﬁers
is to compute the fraction of samples where they disagree while one is correct [42, 91]. Formally, the prediction
diversity PD is deﬁned as:
PD(f, g, S) = 1
N
X
1≤i≤N
1 [df ∨dg] ,
(4)
where
df =

ˆy(i)
f
= y(i) ∧ˆy(i)
g
̸= y(i)
.
(5)
dg =

ˆy(i)
f
̸= y(i) ∧ˆy(i)
g
= y(i)
.
(6)
47

--- Page 48 ---
Cohen’s Kappa Complement (CC). Cohen’s kappa coeﬃcient is a measure of agreement between two
annotators [68]. Here, we use it’s complement as a diversity measure between two classiﬁers:
CC(f, g, S) = 1 −po −pe
1 −pe
= 1 −po
1 −pe
,
(7)
where pe is the expected agreement between the classiﬁers and po is the empirical probability of agreement.
Formally, if nf,k is the number of samples where classiﬁer f predicted label k (i.e. nf,k = P
1≤i≤N 1[ˆyi
f = k]),
then:
pe =
1
N 2
X
1≤c≤C
nf,cng,c,
po = 1
N
X
1≤i≤N
1[ˆyi
f = ˆyi
g]
(8)
KL Divergence (KL). The Kullback-Leibler divergence measures how diﬀerent a probability distribution
is from another. Let p(i)
f
= softmax
 f(x(i))

for a classiﬁer f, and let p(i)
f,c be the probability assigned to
class c. We consider the average KL-divergence over all samples as a diversity measure:
KL(f, g, S) = 1
N
X
1≤i≤N
X
1≤c≤C
p(i)
f,c log
 
p(i)
f,c
p(i)
g,c
!
.
(9)
Centered Kernel Alignment Complement (CKAC). CKA is a similarity measure that compares two
diﬀerent sets of high-dimensional representations [51]. It is commonly used for comparing representations of
two neural networks, or determining correspondences between two hidden layers of the same network. CKA
measures the agreement between two matrices containing the pair-wise similarities of all samples in a dataset,
where each matrix is constructed according to the representations of a model. More formally, let S ∈RN×d
denote the d-dimensional features for all samples in a dataset S, pre-processed to center the columns. For
two models f and g yielding similarity matrices Sf and Sg, CKA is deﬁned as:
CKA(f, g, S) =
||S⊤
g Sf||2
F
||S⊤
f Sf||F ||S⊤
g Sg||F
,
(10)
where ||S||F denotes the Frobenius norm of the matrix S. Larger CKA values indicate larger similarities
between the representations of the two models, and thus, smaller diversity. We deﬁne the diversity measure
CKAC as:
CKAC = 1 −CKA.
(11)
Note that CKAC is computationally expensive to compute for large datasets.
For this reason, in our
experiments with distributions larger than 10,000 samples, we randomly sample 10,000 to compute this
measure.
Diversity across diﬀerent architectures We extend Figure 5 to show results for all combinations of
diversity measures, datasets, and CLIP models. Similarly to before, the baselines compares models with
the same encoder, with two linear classiﬁers trained on diﬀerent subsets of ImageNet with half of the data.
Results are shown in Figures 27-30.
48

--- Page 49 ---
0.0
0.1
0.2
PD
ImageNet
0.0
0.1
0.2
ImageNetV2
0.0
0.1
ObjectNet
RN50
ViT-B/32
RN101
RN50x4
ViT-B/16
RN50x16
RN50x64
ViT-L/14
ViT-L/14@336px
0.0
0.1
0.2
PD
ImageNet Sketch
RN50
ViT-B/32
RN101
RN50x4
ViT-B/16
RN50x16
RN50x64
ViT-L/14
ViT-L/14@336px
0.0
0.1
0.2
ImageNet-A
RN50
ViT-B/32
RN101
RN50x4
ViT-B/16
RN50x16
RN50x64
ViT-L/14
ViT-L/14@336px
0.0
0.1
0.2
ImageNet-R
Diversity between zero-shot and linear classiﬁer
Diversity between two linear classiﬁers
Prediction Diversity (PD) across diﬀerent models and datasets
Figure 27: Prediction Diversity (PD) for multiple datasets and CLIP models (Equation 4).
0.0
0.2
CC
ImageNet
0.0
0.2
0.4
ImageNetV2
0.0
0.2
0.4
ObjectNet
RN50
ViT-B/32
RN101
RN50x4
ViT-B/16
RN50x16
RN50x64
ViT-L/14
ViT-L/14@336px
0.00
0.25
0.50
CC
ImageNet Sketch
RN50
ViT-B/32
RN101
RN50x4
ViT-B/16
RN50x16
RN50x64
ViT-L/14
ViT-L/14@336px
0.0
0.2
0.4
ImageNet-A
RN50
ViT-B/32
RN101
RN50x4
ViT-B/16
RN50x16
RN50x64
ViT-L/14
ViT-L/14@336px
0.0
0.2
0.4
ImageNet-R
Diversity between zero-shot and linear classiﬁer
Diversity between two linear classiﬁers
Cohen’s Kappa Complement (CC) across diﬀerent models and datasets
Figure 28: Cohen’s Kappa Complement (CC) for multiple datasets and CLIP models (Equation 7).
49

--- Page 50 ---
0.0000
0.0005
0.0010
KL
ImageNet
0.0000
0.0005
0.0010
ImageNetV2
0.0000
0.0025
0.0050
ObjectNet
RN50
ViT-B/32
RN101
RN50x4
ViT-B/16
RN50x16
RN50x64
ViT-L/14
ViT-L/14@336px
0.0000
0.0005
0.0010
KL
ImageNet Sketch
RN50
ViT-B/32
RN101
RN50x4
ViT-B/16
RN50x16
RN50x64
ViT-L/14
ViT-L/14@336px
0.000
0.002
0.004
ImageNet-A
RN50
ViT-B/32
RN101
RN50x4
ViT-B/16
RN50x16
RN50x64
ViT-L/14
ViT-L/14@336px
0.000
0.002
0.004
ImageNet-R
Diversity between zero-shot and linear classiﬁer
Diversity between two linear classiﬁers
Average Kullback-Leibler divergence (KL) across diﬀerent models and datasets
Figure 29: Average KL Divergence (KL) for multiple datasets and CLIP models (Equation 9).
0.00
0.25
0.50
CKAC
ImageNet
0.00
0.25
0.50
ImageNetV2
0.0
0.2
0.4
ObjectNet
RN50
ViT-B/32
RN101
RN50x4
ViT-B/16
RN50x16
RN50x64
ViT-L/14
ViT-L/14@336px
0.0
0.5
CKAC
ImageNet Sketch
RN50
ViT-B/32
RN101
RN50x4
ViT-B/16
RN50x16
RN50x64
ViT-L/14
ViT-L/14@336px
0.00
0.25
0.50
ImageNet-A
RN50
ViT-B/32
RN101
RN50x4
ViT-B/16
RN50x16
RN50x64
ViT-L/14
ViT-L/14@336px
0.0
0.5
ImageNet-R
Diversity between zero-shot and linear classiﬁer
Diversity between two linear classiﬁers
Centered Kernel Alignment Complement (CKAC) across diﬀerent models and datasets
Figure 30: Central Kernel Alignment Complement (CKAC) for multiple datasets and CLIP models
(Equation 11).
50

--- Page 51 ---
F
When do weight-space ensembles approximate output-space en-
sembles?
In practice we observe a diﬀerence between weight-space and output-space ensembling. However, it is worth
noting that these two methods of ensembling are not as diﬀerent as they initially appear. In certain regimes
a weight-space ensemble approximates the corresponding output-space ensemble—for instance, when training
is well approximated by a linear expansion, referred to as the NTK regime [44]. Fort et al. [24] ﬁnd that a
linear expansion becomes more accurate in the later phase of neural network training, a phase which closely
resembles ﬁne-tuning.
Consider the set Θ = {(1 −α)θ0 + αθ1 : α ∈[0, 1]} consisting of all θ which lie on the linear path between θ0
and θ1.
Proposition 1. When f(θ) = f(θ0) + ∇f(θ0)⊤(θ −θ0) for all θ ∈Θ, the weight- and output-space ensemble
of θ0 and θ1 are equivalent.
Proof. We may begin with the weight-space ensemble and retrieve the output-space ensemble
f((1 −α)θ0 + αθ1)
(12)
= f(θ0) + ∇f(θ0)⊤((1 −α)θ0 + αθ1 −θ0)
(13)
= f(θ0) + α∇f(θ0)⊤(θ1 −θ0)
(14)
= f(θ0) + α∇f(θ0)⊤(θ1 −θ0) + αf(θ0) −αf(θ0)
(15)
= (1 −α)f(θ0) + α

f(θ0) + ∇f(θ0)⊤(θ1 −θ0)

(16)
= (1 −α)f(θ0) + αf(θ1)
(17)
where the ﬁrst and ﬁnal line follow by the linearity assumption.
51
