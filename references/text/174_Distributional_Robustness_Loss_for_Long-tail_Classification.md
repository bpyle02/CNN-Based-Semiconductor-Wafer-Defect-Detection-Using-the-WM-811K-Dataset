# Distributional Robustness Loss for Long-tail Classification

**Authors**: Samuel, Chechik
**Year**: 2021
**arXiv**: 2104.02703
**Topic**: long_tail
**Relevance**: DRO-based loss for robustness across head and tail classes

---


--- Page 1 ---
Adversarial Robustness under Long-Tailed Distribution
Tong Wu1,5, Ziwei Liu2, Qingqiu Huang3, Yu Wang4, Dahua Lin1,5,6,7
1The Chinese University of Hong Kong, 2S-Lab, Nanyang Technological University, 3Huawei,
4Tsinghua University, 5SenseTime-CUHK Joint Lab, 6Centre of Perceptual and Interactive Intelligence
7Shanghai AI Laboratory
{wt020,dhlin,hq016}@ie.cuhk.edu.hk, ziwei.liu@ntu.edu.sg, yu-wang@mail.tsinghua.edu.cn
Abstract
Adversarial robustness has attracted extensive studies
recently by revealing the vulnerability and intrinsic char-
acteristics of deep networks. However, existing works on
adversarial robustness mainly focus on balanced datasets,
while real-world data usually exhibits a long-tailed distri-
bution. To push adversarial robustness towards more real-
istic scenarios, in this work we investigate the adversarial
vulnerability as well as defense under long-tailed distribu-
tions. In particular, we ﬁrst reveal the negative impacts
induced by imbalanced data on both recognition perfor-
mance and adversarial robustness, uncovering the intrinsic
challenges of this problem. We then perform a systematic
study on existing long-tailed recognition methods in con-
junction with the adversarial training framework. Several
valuable observations are obtained: 1) natural accuracy is
relatively easy to improve, 2) fake gain of robust accuracy
exists under unreliable evaluation, and 3) boundary error
limits the promotion of robustness. Inspired by these obser-
vations, we propose a clean yet effective framework, RoBal,
which consists of two dedicated modules, a scale-invariant
classiﬁer and data re-balancing via both margin engineer-
ing at training stage and boundary adjustment during in-
ference. Extensive experiments demonstrate the superiority
of our approach over other state-of-the-art defense meth-
ods. To our best knowledge, we are the ﬁrst to tackle adver-
sarial robustness under long-tailed distributions, which we
believe would be a signiﬁcant step towards real-world ro-
bustness. Our code is available at: https://github.
com/wutong16/Adversarial_Long-Tail.
1. Introduction
Despite the great progress on a variety of computer vi-
sion tasks, deep neural networks are found to be vulnerable
to minor adversarial perturbations [39], i.e., easily misled
to make incorrect predictions. The existence of adversarial
examples reveals a non-negligible security risk to modern
Natural 
Accuracy
Robust 
Accuracy
Baseline
Our
Defense
Long-tailed Recognition
Classes
Balanced CIFAR-10
Long-tailed CIFAR-10
AT-trained A_nat
AT-trained A_rob
Plainly-trained A_nat
Accuracy
Sample number
Figure 1. Upper: A long-tailed data distribution induces decreas-
ing natural and robust accuracy from head to tail and a magniﬁed
“sacriﬁce” of natural accuracy especially to tail classes when ad-
versarial training is applied. Lower: Evaluation results on two
metric dimensions, including a number of long-tailed recognition
methods combined with adversarial training, several state-of-the-
art defense methods, and our RoBal in a region with trade-off.
computer vision models, with extensive efforts devoted to
improving adversarial robustness.
Existing adversarial robustness research mainly focuses
on balanced datasets such as CIFAR and ImageNet [13].
Nevertheless, real-world data usually exhibit a long-tailed
distribution [41, 12], which brings challenges not only to the
recognition tasks themselves but also to robustness against
adversarial attacks.
The former has been attracting in-
creasing attention recently, with a number of algorithms
arXiv:2104.02703v3  [cs.CV]  17 Aug 2021

--- Page 2 ---
[25, 17, 53, 7, 2, 46, 43] proposed to tackle the issue; On
the other hand, the latter remains largely unexplored.
To cast light on the challenges of adversarial robustness
in long-tailed recognition (LT), we ﬁrst perform an intu-
itive comparison between networks trained on the balanced
and long-tailed versions of CIFAR-10, respectively. Apart
from normally trained models, we also adopt the adver-
sarial training (AT) framework [26], which is one of the
most effective and widely used defense methods, to provide
the basic adversarial robustness for the networks. Per-class
classiﬁcation recalls are evaluated on clean images and im-
ages permuted by PGD attack [26], denoted by natural ac-
curacy Anat and robust accuracy Arob, respectively. Anat
is evaluated on both plain models and AT-trained models,
while Arob is performed only on the latter. Results are vi-
sualized in Fig. 1. There are three main observations from
the comparison: 1) Anat on plain models drops from head
to tail, which is exactly what traditional long-tailed recogni-
tion aims to solve. 2) A similar decreasing tendency reason-
ably occurs in Arob. 3) It is worth noting that Anat drops
more signiﬁcantly at the tail when adversarial training is
applied, indicating that the well-known “sacriﬁce” of the
natural accuracy induced by adversarial training is further
magniﬁed for tail classes under a long-tailed distribution.
To form a better understanding of the problem, the re-
lationship between natural and robust accuracy can be con-
nected by boundary error Rbdy [52] as:
Arob = Anat −Rbdy,
(1)
where Rbdy represents how likely the features of clean and
correctly predicted inputs are close to the ϵ-extension of the
decision boundary. It represents the gap between the two
forms of accuracy and indicates the vulnerability of samples
against adversarial attacks.
Hence, to achieve improvement on both recognition per-
formance and adversarial robustness, a natural idea is to
raise Anat while keeping a small value of Rbdy. Speciﬁ-
cally, on the one hand, we are able to address the issue of
imbalance in data distribution via re-balancing strategies,
thus we conduct a systematic study of currently widely used
long-tailed recognition approaches to explore the proper
combinations of these methods and the adversarial train-
ing framework. On the other hand, we would analyze why
a normalized embedding space promotes model resistance
against attacks, and then a scale-invariant classiﬁer is in-
troduced to replace the ﬁnal linear layer. The idea of data
re-balancing is then well aligned with the cosine classiﬁer
by the cooperation of class-aware and pair-aware margins
during training and boundary adjustment at inference.
Note that the imbalance in data distribution and the de-
ﬁciency in sample numbers are two issues induced simulta-
neously when we turn to long-tailed datasets instead of the
artiﬁcially balanced ones. Although the importance of data
scale in adversarial robustness has been widely studied [34],
we mainly focus on the problem of imbalance in this paper.
We study the effect of them separately in Sec 5, verifying
that eliminating prediction priors is crucial to reducing the
vulnerability of tail classes under attack.
Our contributions are as follows: 1) To our best knowl-
edge, we are the ﬁrst to tackle adversarial robustness under
long-tailed distribution, which we believe would be a signif-
icant step towards real-world robustness. 2) We conduct a
systematic study on existing long-tailed recognition meth-
ods and their adoption into the adversarial training proce-
dure. Important insights are gained based on experimental
observations. 3) We further develop a clean yet effective
approach, RoBal, that achieves state-of-the-art performance
on both natural and robust accuracy.
2. Related Works
Long-Tailed Recognition. To tackle the long-tailed recog-
nition problem, traditional re-balancing approaches include
re-sampling [5, 16, 13, 37, 1] and re-weighting [7, 1]. How-
ever, these methods may suffer from the issue of under-
representing major classes and over-ﬁtting minor ones. To
mitigate these negative impacts, more ﬂexible usages of
the basic methods were proposed, such as decoupled train-
ing [17, 53] and deferred re-balancing schedule [2], respec-
tively, and they are proved to be more effective. Further,
recently proposed approaches address class-speciﬁc prop-
erties by perspectives like margin [2], bias [31, 28], tem-
perature [49] or weight scale [17, 20], and some of these
methods can be either adopted to the whole training process
or in a post-processing manner. Another trend of works fo-
cuses on sample-speciﬁc properties via hard example min-
ing [23] or sample-aware re-weighting strategies leverag-
ing meta-learning [32, 15, 38]. Besides, several recent ap-
proaches propose to transfer knowledge from head to tail
through memory module [25], inter-class feature transfer-
ring [24], and “major-to-minor” translation [21]. In this
paper, we would revisit and summarize a number of these
methods and explore their effective combination with ad-
versarial training in Sec. 3.
Adversarial Robustness.
Plenty of adversarial defense
methods have been proposed to tackle the problem of adver-
sarial vulnerability. Among them, adversarial training [26]
is one of the most effective and reliable strategies. Improve-
ments have been made based on the AT framework via the-
oretical analysis and loss function examination [52, 44, 8].
Many efforts have also been devoted to exploring differ-
ent training mechanisms such as metric learning [27], self-
supervised learning [14], and semi-supervised learning [4].
Since AT is of the high computational cost and time con-
sumption, another line of works [36, 45, 51] was proposed
to accelerate the training procedure. Besides, some general
strategies were revealed to be critical to the robustness per-
formance such as label smoothing [35], early stop [33], dif-

--- Page 3 ---
Table 1. A systematic study of current LT strategies combined with AT framework, detailed explanations are to be included in Sec. A.3.
Green, red, and blue denote impressive Anat, unreliable evaluation of Arob under PGD attack, and the smallest Rbdy, respectively.
Stage
Methods
Formulation
Clean
PGD
AA
Gap
Train
Vanilla FC
gi = W T
i f(x)
62.33
29.30
28.15
34.18
Vanilla Cos
gi = f
W T
i ef(x)
56.59
29.38
27.23
29.36
Class-aware margin [2]
gi = W T
i f(x) −1{i = y} · δi
63.24
29.81
28.70
34.54
Cosine with margin [42]
gi = f
W T
i ef(x) −1{i = y} · m
58.47
31.73
28.45
30.02
Class-aware temperature [50]
gi = W T
i f(x) · (ni/nmax)γ
73.11
30.12
28.62
44.49
Class-aware bias [28, 31]
gi = W T
i f(x) + τ log(ni)
74.46
32.45
30.55
43.91
Hard-exmaple mining [23]
r(y) = (1 −py)γ, applyed with BCE loss
62.07
30.73
28.12
33.95
Re-sampling [37]
rs(i) ∝1/ni
58.62
25.06
24.25
34.37
Re-weighting [7]
r(y) = (1 −β)/(1 −βn
y )
64.33
34.53
29.01
35.32
Fine-tune
One-epoch re-sampling [17]
hi = W ′T
i f(x), W ′
i re-trained with RS
70.88
29.81
28.59
42.29
One-epoch re-weighting [2, 7]
hi = W ′T
i f(x), W ′
i ﬁne-tuned with RW
71.72
32.34
28.25
43.47
Learnable classiﬁer scale [17]
hi = si · W T
i f(x), where si is learnable
69.63
28.81
27.99
41.64
Inference
Classiﬁer re-scaling [50, 20]
hi = (Wi/nτ
i )T f(x)
73.84
39.05
28.23
45.61
Classiﬁer normalization [17]
hi = (Wi/ ∥Wi∥τ)T f(x)
72.70
36.57
29.77
42.93
Class-aware bias [28]
hi = W T
i f(x) −τ log(ni)
74.25
31.95
30.45
43.80
Feature disentangling [40]
hi = W T
i (f(x) −α cos(f(x), d) · d)
71.16
32.69
30.48
40.68
ferent activation functions [48], batch normalization [47],
and embedding space [30]. Most recently, Pang et al. [29]
and Gowal et al. [11] made systematic studies on the effect
of basic training settings and some other choices, including
model size, data, loss, and activation functions, respectively.
Our method is also built on the AT framework, while we fo-
cus on the long-tailed training data distribution to explore
how it affects the accuracy and ways for improvement.
3. Long-tailed Recognition with Defense
In this section, we ﬁrst brieﬂy introduce the adversar-
ial training (AT) framework. Then we conduct a system-
atic study on some popular long-tailed recognition (LT)
strategies and explore their proper combination with the AT
framework, where the effectiveness is evaluated by Anat,
Arob, and Rbdy. We further reveal a fake increase of ro-
bustness that could be induced under unreliable evaluation.
Finally, valuable knowledge from the study is summarized
and inspires us to develop our method in Sec. 4.
3.1. Adversarial Training Preliminaries
Adversarial training, as one of the most effective defense
methods, is adopted as the basic framework to maintain ba-
sic robustness in this paper. The standard AT and its variants
can be formulated as a mini-max problem:
min
θ
E(x,y)∼D [LT (θ; x + δ, y)] ,
where δ = argmax
δ∈B(ϵ)
LA(θ; x + δ, y).
(2)
The inner optimization aims to ﬁnd effective adversarial ex-
amples by maximizing LA, and the outer optimization up-
dates network parameters to minimize the training loss LT .
The standard AT proposed by Madry et al. [26] uses Cross-
Entropy loss(CE) for both LA and LT , while we would ex-
plore the effects of different choices in this paper.
3.2. Revisiting Long-tailed Recognition Methods
Preliminaries. The LT methods, who could be naturally
combined with the AT framework, can be categorized into
three phases based on different applying stages: training,
ﬁne-tuning, and inference, as summarized in Table 1. The
evaluation metrics include the accuracy of the clean images
and permuted images under PGD-20 [26] and Auto-Attack
(AA) [6]. Details are to be introduced in Sec. 5. We also
report the gap between clean accuracy and Auto-Attack ac-
curacy for a better view of boundary error.
Notiﬁcations. Suppose there are C classes in total with
ni, i ∈{1, 2, ..., C} samples for class i. We denote f(x)
as the deep feature extracted from image x and W
=
[W1, ..., WC] as the classiﬁer weight vectors. And the nor-
malized weight vectors and features are denoted as f
Wi =

--- Page 4 ---
Wi/ ∥Wi∥and ef(x) = f/ ∥f∥.
Training Stage. Methods applied to training stage include
class-aware re-sampling [37, 25, 17, 53], and several cost-
sensitive learning approaches. We denote the sampling fre-
quency for class i as rs(i) in Table 1. Cost-sensitive learn-
ing methods usually modify the loss function by introduc-
ing class-speciﬁc parameters like weight (CB [7]), margin
(LDAM [2]), bias (LA-train [28], Balanced Softmax [31]),
and temperature (CDT-train [49]). A hard example examin-
ing method (Focal [23]) is also included here although it is
applied with binary cross entropy loss. A general loss func-
tion for the cost-sensitive methods above with CE loss can
be formulated as:
L′
CE(W; f(x), y) = −rw(y) · log(
ezy
P
i ezi ),
where zi = gi(Wi, f(x)),
(3)
where g(W, f(x)) denotes the logit before softmax, and
rw(y) is a re-weighting factor for class y. The widely used
linear classiﬁer would have gi(Wi, f(x)) = W T
i f(x) + bi.
Different methods on training would have different g func-
tions, which are listed in Table 1 ( bi is omit for simplicity).
L′CE can be adopted to AT procedure in three modes:
replacing the CE in LA, LT , or both of them, where LA
and LT would affect the optimization of the adversarial ex-
amples and network parameter updating, respectively. We
empirically observed that modifying LT has a more con-
spicuous inﬂuence on the results. Thus results reported here
are conducted with the second mode. We present a more de-
tailed study in the supplementary material Sec. B.
Fine-tuning Stage. Fine-tuning based methods propose to
re-train [17] or ﬁne-tune the classiﬁer via data re-balancing
techniques with the backbone frozen, which take advan-
tage of the idea of decoupling the learning of representa-
tion and classiﬁer. We empirically ﬁnd out that one-epoch
of ﬁne-tuning with class-aware sampling or re-weighting
would remarkably raise Anat while more steps make little
difference. A similar conclusion is drawn when only weight
scales si are learned at this stage (LWS [17]).
Inference Stage. LT methods applied at the inference stage
based on a vanilla trained model would usually conduct
a different forwarding process from the training stage to
address shifted data distributions from train-set to test-set.
Speciﬁcally, we denote h(W, f(x)) as the logit producing
function on inference, and the prediction is performed by:
argmax
i∈[C]
hi(Wi, f(x)),
(4)
We consider four post processing methods in this paper
including classiﬁer normalization (τ-norm [17]), classiﬁer
re-scaling based on sample numbers (CDT-post [50, 20]),
feature disentangling (TDE [40]), and logit adjustment (LA-
post [28]), as shown in Table 1.
3.3. Analysis and Takeaways
Here we summarize some of the key observations and
knowledge revealed by the empirical study, including the
effectiveness of LT methods on natural accuracy, the chal-
lenge to robustness evaluation reliability induced by some
LT strategies, and the importance of boundary error for ro-
bust accuracy.
Natural accuracy is easy to improve. According to the
results in Table 1, a number of LT strategies are proved
effective in improving Anat, as marked in green. We can
either apply a cost-sensitive loss in outer minimization LT
during the whole training stage, or leverage ﬁne-tuning and
post-processing to boost the performance with little extra
computational cost. This indicates that both re-balancing
strategies applied during the training process and boundary
adjustment at inference time positively impact Anat, which
inspires the development of our method in Sec. 4.
Fake improvement exists for robust accuracy evalua-
tion. It is noticeable that methods based on classiﬁer nor-
malization [17] and re-scaling [50, 20] achieve impressive
robust accuracy under PGD-attack, as marked in red, while
the AA evaluated results remain ordinary. This is due to
the sensitivity of PGD attack to both logits stretching and
compressing, which is worth attention.
Consider a uniformly re-scaled classiﬁer W ′
i = Wi/10κ
at inference time, where logit scales and κ are negatively
correlated. As shown in Fig. 2, Anat is not effected since
the re-scaling operation by 10−κ > 0 does not change the
ordering; AA evaluated Arob is also invariant to scaling
which can be seen as a reliable evaluation of robustness;
while PGD robustness exhibits a minimum at κ ≈0 and
increases as κ leaves zero on both sides.
The reason lies in the updating of PGD attack, which is
based on the gradient produced by the inner CE loss:
∇f(x)LCE(W; f(x), y) = −∇f(x)zy +
X
i
pi∇f(x)zi
=
X
i̸=y
pi(Wi −Wy).
(5)
The vectors Wy to Wi with i ̸= y are weighted by softmax
produced prediction conﬁdence, pi, which is effective by
focusing more on easily confused classes. But the sensitiv-
ity of softmax to both the absolute and relative values of its
components leads to two kinds of fail cases of PGD.
1) Gradient vanishing. The false sense of security due
to gradient vanishing is not new knowledge [3, 6]: a cor-
rectly predicted clean image with y = argmaxi zi would
gain py ≈1 and pi ≈0(i ̸= y) when all logits are scaled
up, leading to zero gradient as in Eqn. 5. We calculate the
ratio of zero in gradient at pixel level during the PGD up-
dating following [6], and it converges to the same level as
Anat rather than 100% in their paper (Fig. 2).

--- Page 5 ---
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
-4 -3.5 -3 -2.5 -2 -1.5 -1 -0.5
0
0.5
1
1.5
2
2.5
3
Percent
Accuracy
kappa
Robustness evaluation with varing logit scale
PGD
Clean
AutoAttack
Zero_grad
weight operation based
LT methods
gradient vanishing
direction averaging
Figure 2. Anat and Arob under AA are invariant to logit scaling,
while Arob under PGD is easily over estimated when using weight
operation based LT methods. The zero in gradients occurs on orig-
inally correctly predicted images with logit stretching.
2) Direction averaging. On the other side, compressed
logits lead to averaged softmax outputs, where
pi
≈
1/C, and the updating direction becomes ∇f(x)LCE =
1
C
P
i Wi −Wy = W −Wy, pointing from Wy to the av-
eraged weight vector W. It only depends on y and fails
to take sample-speciﬁc properties into consideration; and
since it is ﬁxed throughout the inner maximization proce-
dure of Eqn. 2, the iterative attack actually degenerates to be
single-step. Consequently, the attack is weakened to some
extent so that Arob gradually converges to a ﬁxed value as
κ goes up, leading to fake gain of performance that weight
operation based methods suffer from.
Boundary error matters for robust accuracy. When eras-
ing the false sense of security, the reliable Arob under AA in
Table 1 seems not signiﬁcantly affected as Anat raises. In
fact, a huge gap between natural and robust accuracy is ob-
served, and it continuously widens as the former improves,
which can be reﬂected by the term Rbdy1 in Eqn. 1. The
phenomenon exposes the importance of controlling Rbdy
in order to improve both Anat and Arob, which is an is-
sue that many LT methods do not promise to solve. How-
ever, we found that cosine classiﬁer based methods could
beneﬁt from a relatively smaller Rbdy compared with lin-
ear classiﬁers. One evidence is that in Table 1, the models
trained with a cosine classiﬁers exhibit the lowest gap be-
tween Anat and Arob under AA, marked in blue in the last
column. We would analyze the reason behind it in the next
section and how to take advantage of its good property.
4. Methodology
Earlier discoveries cast light on two key factors of solv-
ing this challenging problem: 1) a proper feature and clas-
siﬁer embedding helps to achieve a lower boundary er-
ror Rbdy, and 2) the combination of long-tailed recogni-
tion (LT) methods with adversarial training (AT) framework
1The case that a wrong prediction being “corrected” after the attack
rarely happens, so Rbdy can be basically reﬂected by Anat −Arob.
Count
Count
Scaling Ratio
Feature Norm Scaling Ratio after Attack
Classifier L2-norm
Class Index
Figure 3. Left: the feature norm distribution shifts after adver-
sarial attack, the successfully attacked samples statistically have a
larger scaling factor than those that stay robust; Right: the classi-
ﬁer weight norm roughly decreases towards the tail classes.
would beneﬁt Anat. Hence, we propose a clean yet effec-
tive approach which consists of two components, i.e. scale-
invariant classiﬁer and two-stage re-balancing, to achieve
Robust and Balanced predictions, namely RoBal.
4.1. Scale-invariant Classiﬁer
In a basic classiﬁcation task using a standard linear clas-
siﬁer, the predicted logit of class i can be represented as:
W T
i f(x) + bi = ||Wi|| · ||f(x)|| cos θi + bi,
(6)
where we can see that the prediction depends on three fac-
tors: 1) the scale of weight vector ||Wi|| and feature vector
||f(x)||; 2) the angle cos θi between them; and 3) the bias
of the classiﬁer bi. In this section we would focus on the
ﬁrst factor to show the importance of being scale-invariant.
Firstly, the decomposition above indicates that the pre-
diction of a sample can be changed by simply scaling its
norm in the feature space.
We consider this to be one
of the schemes adversarial examples use to confuse the
model, leading to different feature norm distributions be-
tween “successfully attacked” and “robust” samples.
To
be speciﬁc, the originally correctly predicted images can be
separated into two groups by whether the attack is success-
ful. We then calculate the scaling ratio ∥f(x + δ)∥/ ∥f(x)∥
between each attacked and clean input pair. We could ob-
serve different distributions of the two groups in Fig. 3,
where a successful attack is more likely to happen with a
relatively higher scaling ratio.
Besides the feature embedding, the scales of the weight
vectors ||Wi|| in a linear classiﬁer would also induce prob-
lems to the long-tail scenario. Speciﬁcally, they usually de-
crease towards the tail classes as shown in Fig. 3, which is
also observed in previous works [17, 20]. Different weight

--- Page 6 ---
Biased boundary
Adjusted boundary
Class j
Class i
Class j
Class i
(a)
(b)
Class j
Class i
𝒎𝟎
𝒎𝒊
(c)
𝒎𝒊𝒋=  𝒎𝒋𝒊= 𝟎
(d)
Class j
Class i
Higher
deviation
Lower
deviation
𝜽𝒊
𝒄𝒐𝒔𝜽𝒊= 𝒄𝒐𝒔𝜽𝒋
𝒎𝟎
𝒎𝒋
𝒎𝒊𝒋
𝒎𝒋𝒊
𝜽𝒋
Corrected
Attacked via
norm scaling
Bias + margins
Boundary
adjustment
Figure 4. (a) and (b): biased decision boundary is induced by im-
balanced weight norms while boundary adjustment helps correct
some mistakes; feature norm scaling can result in a successful at-
tack; (c): margin construction with hyper-parameters ignored for
simplicity; (d): boundary adjustment at inference stage.
scales result in biased decision boundaries (Fig. 4) and hurt
recognition performance. This could be alleviated by ad-
justing the class-speciﬁc bias. But it still suffers from a high
adversarial risk even after boundary adjustment considering
the varying feature norm, as shown in Fig. 4
Based on the observation and analysis, a natural idea is
to remove the inﬂuence of scales from both the features and
weights. Therefore, a scale-invariant classiﬁer, e.g., cosine
classiﬁer that limits the vectors to a hyper-sphere [30], could
be a proper choice. The beneﬁt of it reducing Rbdy has been
revealed in Table 1. And we would further introduce how
to address the issue of imbalance via its combination with
re-balancing strategies.
4.2. Two-stage Re-balancing
Based on the normalized features and classiﬁer weights,
we further consider the problem of long-tailed data distri-
bution. Recall the knowledge we gain from Sec. 3, where
signiﬁcant natural accuracy gain via effective elimination
of imbalance can happen either during training or simply
at inference time.
Accordingly, we propose a two-stage
re-balancing framework in this section, which exactly fo-
cuses on the other two factors in Eqn. 6, namely cos θi and
bi: 1) margins introduced to the cosine classiﬁer at train-
ing stage promote a more compact representation learning,
where both class-aware and pair-aware margin engineering
would boost network performance in our long-tailed sce-
nario; 2) boundary adjustment at inference stage further
tackles the issue of higher variance and deviation in tail
classes. Different cases of the compensation and cooper-
ation between them is explored in ablation study in Sec. 5.
Class-aware Margin on Training. A straight forward and
widely used idea to take class imbalance into consideration
is to assign class-speciﬁc bias in the CE loss during train-
ing. Following Ren et al. [31] and Menon et al. [28], we
adopt the form of bi = τb log(ni), and the modiﬁed CE
Loss becomes:
L0 = −log( ezy+by
P
i ezi+bi ) = log(1 +
X
i̸=y
ezi−zy+τb log( ni
ny )),
(7)
where τb is a hyper-parameter controlling the bias value cal-
culation. However, on considering the formulation in the
manner of margin, we ﬁnd that the margin from the ground
truth class y to class i, namely τb log(ni/ny), would be-
come negative when ny > ni, leading to less discriminat-
ing representation and classiﬁer learning of head classes.
To deal with the issue, we further add a class-aware margin
term together with the pre-deﬁned bi, which assigns a larger
margin value to the head class in compensation:
mi = τm
s log
ni
nmin
+ m0.
(8)
Here the ﬁrst term would increase along with with ni while
achieving its lowest at zero when ni = nmin, and τm is the
hyper-parameter to control the trend; the second term m0 >
0 is a uniform margin for all classes, as is a commonly used
strategy for cosine classiﬁer based networks [42]; s repre-
sents a temperature here to expand the value range of the
cosine outputs, which helps to present a more clearly for-
mulated loss function as below:
L1 = −log
 
es(cos θy−my)+by
es(cos θy−my)+by + P
i̸=y es cos θi+bi
!
= log(1 +
X
i̸=y
es(cos θi−cos θy+myi)),
(9)
where cos θi = f
W T
i ef(x), and
myi = τb
s log( ni
ny
) + τm
s log( ny
nmin
) + m0
= (τb −τm)
s
log( ni
ny
) + τm
s log( ni
nmin
) + m0.
(10)
Notice that we here adopt f
Wi = Wi/(||Wi|| + γ) here,
which is slightly different from Sec. 3 while we empirically
ﬁnd it able to produce slightly better performance. The ﬁrst
line of formulation in Eqn. 10 constructs the margin be-
tween ground truth class y and a negative class i: a com-
position of a pair-aware margin log(ni/ny) scaled by τb, a
class-aware margin log(ni/nmin) scaled by τm, and a uni-
form m0, as shown in Fig. 4. While The second line reveals
a more direct relationship to the data distribution: ny occurs
only in the ﬁrst term to assign a larger margin to tail classes
when τb −τm > 0. It encourages a more compact and
discriminating learning on them and especially beneﬁts the
imbalance learning process. We would show how each term
effects the training in the ablation study.
Class-speciﬁc Bias on Inference. With a normalized clas-
siﬁer, the decision boundary is naturally unbiased at infer-
ence time. However, the sparse data distribution in the tail

--- Page 7 ---
Table 2. Experimental results on CIFAR-10-LT and CIFAR-100-LT with WideResnet-34-10.
Dataset
CIFAR-10-LT
CIFAR-100-LT
Methods
Clean
FGSM PGD
MIM
CW
AA
Clean
FGSM PGD
MIM
CW
AA
Plain
77.16
7.01
0.00
0.00
0.00
0.00
62.29
3.94
0.00
0.00
0.00
0.00
AT [26]
62.33
33.57
29.30
30.02
30.31
28.15
48.96
21.06
17.26
17.80
17.65
16.26
TRADES [52]
54.29
32.80
30.20
30.53
29.58
28.94
43.71
23.06
21.13
21.42
19.49
18.68
HE [30]
58.47
35.17
31.73
32.26
29.96
28.45
48.63
23.06
19.56
20.28
19.20
17.60
MMA [8]
61.51
36.40
29.29
30.38
29.59
25.91
54.98
19.65
13.52
13.98
12.35
14.54
AVmixup [22]
66.97
33.90
28.40
29.67
26.43
24.39
52.45
23.28
19.04
20.78
14.82
12.60
RoBal-N
75.52
40.04
33.50
34.57
33.68
31.72
51.63
22.81
19.01
19.50
19.42
18.16
RoBal-R
74.51
40.55
33.87
34.92
34.12
32.04
50.38
23.59
19.48
20.13
20.16
18.69
leads to higher uncertainty [19] and feature deviation [50],
while head classes would beneﬁt from a more compact fea-
ture embedding and concise classiﬁer learning. As a result,
when using a uniform margin m0 with τb = τm = 0 dur-
ing training, we can still observe an obvious decrease in the
recall of the tail classes. Thus a post processing strategy to
adjust the cosine boundary is still needed, as can be formu-
lated as a dual process to the pre-deﬁned bi during training
in Eqn. 8. τp is introduced here and the inference becomes:
argmax
i∈[C]
s · cos θi −τp log(
ni
P
j nj
)
(11)
Actually, when class-speciﬁc margins are added along with
the uniform one, the dependence on boundary adjustment is
eliminated, as to be explored in Sec. 5.
Regularization Term. Finally, inspired by the some of the
well-known AT variants [18, 52, 8], an additional regular-
ization term between the paired features or logits produced
by the clean and perturbed images would further promote
the robustness performance. Different kinds of regulariza-
tion terms can be easily adopted into the training frame-
work via modiﬁcation on the loss function, and we follow
Zhang et al. [52] to take advantage of a KL-divergence term,
and the overall loss function become:
L = L1(x+δ, y)+α·KL( f
W T
i ef(x+δ), f
W T
i ef(x)) (12)
where δ is the perturbation generated by inner maximiza-
tion guided by a plain cross-entropy loss, performed on the
direct outputs of the cosine operation without margins.
5. Experiments
Datasets. We conduct experiments on the long-tailed ver-
sions of CIFAR-10 and CIFAR-100 following [7]. Imbal-
ance Ratio (IR) in the main experiments is set as 50 and 10,
respectively. Experimental results with various IRs are also
provided in Table 4.
Evaluation Metrics. On evaluating model robustness, the
allowed l∞norm-bounded perturbation is ϵ = 8/255. At-
tacks conducted include the single-step attack FGSM [10]
and several iterative attacks including PGD, MIM, and
C&W performed for 20 steps with a step size of 2/255. We
also use the recently proposed Auto Attack (AA) [6] which
is an ensemble of different attacks and is parameter-free.
Comparison Methods. We compare our method with sev-
eral state-of-the-art defense methods besides the standard
AT [26], including TRADES [52], MMA [8], HE [30],
and AVmixup [22], among which AVmixup [22] is re-
implemented and the others are evaluated with the ofﬁcially
released code.
Implementation details on our network
training, hyper-parameter setting, and attacking algorithms
are included in the supplementary material.
5.1. Comparison Results
The comparison with other defense methods is reported
in Table 2. Since a trade-off between Natural and Robust
accuracy usually exists, we report our results with different
emphasis, denoted by RoBal-N and RoBal-R, respectively.
The setting of hyper-parameters to control the trade-off can
be found in Sec. A. On CIFAR-10-LT, our method signif-
icantly outperforms all the compared ones on both Anat
of clean images and Arob under ﬁve different attacks. On
CIFAR-100-LT, TRADES [52] and RoBal achieve compa-
rable results on robust accuracy.
However, they signiﬁ-
cantly sacriﬁce the performance on Anat, while our method
also consistently boosts Anat compared with AT baseline.
AVmixup [22] and MMA [8] (on CIFAR-100-LT) achieve
decent Anat while suffering from the poor robust perfor-
mance under AA. It is noted that the overall improvement in
CIFAR-100-LT is less signiﬁcant than CIFAR-10-LT, pos-
sibly due to the smaller imbalance ratio or the increase of
class number; thus we also provide preliminary results on
ImageNet-LT [25] with 1000 classes in Sec. B.4.
5.2. Ablation Study
Here we explore the effect of hyper-parameters during
the two-stage re-balancing. We mainly discuss three terms
according to Eqn. 10, namely m0, τb −τm, and τm. We
change the critical variable while ﬁx the others for analysis
as shown in Table 3. Several interesting observations in-
clude: 1) a proper m0 > 0 leads to higher Arob yet lower

--- Page 8 ---
0.0
0.2
0.4
0.6
0.8
1.0
1
2
3
4
5
6
7
8
9
10
Recall
Class Index
Per-class Recall on Clean Images
Full
Sml-Bal
LT_Base
LT_Our
0.0
0.2
0.4
0.6
0.8
1.0
1
2
3
4
5
6
7
8
9
10
Recall
Class Index
Per-class Recall on Attacked Images
Full
Sml-Bal
LT_Base
LT_Our
Figure 5. Results under different data scales and distributions.
Anat at both stages; 2) a higher τb −τm promotes natural
accuracy at the end of training stage, yet a peak is observed
for robustness; 3) a larger τm helps reduce the boundary er-
ror Rbdy, while it could hurt Anat if set too large. Please
refer to the supplementary for more experimental results.
Table 3. Effect of hyper-parameters. We ﬁx τm = τb = 0 in block
1, m0 = 0.1, τm = 0 in block 2, and m0 = 0.1, τb −τm = 1.2
in block 3. ** denotes the key metric that is worth noting.
End of training stage
Inference with τp∗
m0
Clean
AA
Clean**
AA
0
63.51
28.47
75.98
30.46
0.1
63.17
29.13
75.51
31.24
0.2
60.6
29.04
74.83
32.04
0.3
56.59
28.63
71.31
31.28
τb −τm
Clean**
AA
Clean
AA
0
63.51
28.47
75.51
31.24
0.5
68.58
30.55
75.77
30.94
1
73.93
31.52
75.42
31.61
1.5
74.67
30.95
74.67
30.95
τm
Clean
AA
Training gap**
-0.3
74.88
30.66
44.22
0
75.08
31.79
43.29
0.3
74.51
32.04
42.47
0.6
71.84
31.41
40.43
5.3. Further Analysis
Effect of Data Scale. Since a long-tailed dataset differs
from the full dataset by both class-wise sample numbers
and the overall data scale, we conduct a comparison with
1) the original full dataset (Full) and 2) a dataset with the
same number of samples as the long-tailed version but is
uniformly distributed (Sml-Bal). From the per-class recall
shown in Fig. 5, we can see that: (1) Performance of Sml-
Bal are uniformly lower than that of the full dataset. (2)
Table 4. Experimental results on CIFAR-10-LT with different IRs
IR
Methods
Clean
PGD
MIM
CW
AA
100
AT
56.72
27.27
27.87
27.80
25.97
TRADES
45.67
26.97
27.29
26.53
25.93
Our
68.07
30.35
31.25
30.55
28.97
50
AT
62.33
29.30
29.72
30.31
28.15
TRADES
54.29
30.20
30.53
29.58
28.94
Our
73.93
34.24
35.37
34.58
32.70
20
AT
74.09
33.59
34.=65
34.27
32.02
TRADES
65.17
34.65
35.29
34.07
33.06
Our
78.49
39.17
40.44
39.38
37.58
10
AT
79.45
37.11
37.93
38.31
35.51
TRADES
72.92
39.15
39.95
38.41
37.33
Our
81.20
40.22
41.75
40.91
38.90
The basic AT framework produces an apparent decrease in
recall from head to tail on both clean and attacked images.
Speciﬁcally, head classes gain even higher robustness than
balanced baselines, indicating that the intrinsic prediction
bias raise their resistance to attack. (3) Our RoBal (LT-Our)
applied to the long-tailed dataset efﬁciently re-balances the
per-class recall compared with the baseline (LT-base).
Effect of Imbalance Ratio.
We also constructed long-
tailed datasets with different imbalanced ratios (IR) follow-
ing [7] to evaluate the performance of AT, TRADES [52],
and our methods. As shown in Table 4, our method outper-
forms the baseline and TRADES [52] remarkably on both
natural accuracy and robust accuracies over different IRs.
6. Conclusion
In this paper, 1) we ﬁrst reveal the negative impacts in-
duced by long-tailed data distribution on both recognition
performance and adversarial robustness, uncovering the in-
trinsic challenges of this problem. 2) Then, a systematic
study on existing long-tailed recognition approaches and
their combination with the adversarial training framework
contributes several valuable observations. 3) Finally, in-
spired by them, we propose a clean yet effective framework
that beneﬁts from the norm-invariant property of cosine
classiﬁer and a two-stage re-balancing framework, which
outperforms existing state-of-the-art defense methods. To
our best knowledge, we are the ﬁrst to tackle adversarial
robustness under long-tailed distribution, which we believe
would be a signiﬁcant step towards real-world robustness.
Acknowledgements.
This research was conducted in col-
laboration with SenseTime. This work is supported by GRF
14203518, ITS/431/18FX, CUHK Agreement TS1712093,
NTU NAP and A*STAR through the Industry Alignment
Fund - Industry Collaboration Projects Grant, and the
Shanghai Committee of Science and Technology, China
(Grant No. 20DZ1100800).

--- Page 9 ---
A. Implementation Details of Experiments
A.1. Training Details and Hyper-parameter Setting
We adopt the WideResNet-34-10 as the model architec-
ture. The initial learning rate is set as 0.1 with a decay factor
of 10 at 60 and 75 epochs, totally 80 epochs. We use the last
epoch for evaluation without early-stop for all the methods.
We use the SGD momentum optimizer with weight decay
set as 2×10−4. We use a batch size of 64 for all the experi-
ments in the main paper. The adversarial training is applied
with the maximal permutation of 8/255 and a step size of
2/255 (0.031 and 0.0078 are used for implementation). The
number of iterations in the inner maximization is set as 5,
and a study on the effect of PGD steps in AT is reported
in Sec. B.2. There are multiple hyper-parameters involved,
where those that control margins or boundary adjustment
are the most critical. Speciﬁcally, we adopt m0 = 0.1 for
CIFAR-10-LT and m0 ∈{0.2, 0.3} for CIFAR-100-LT for
different emphasis (i.e., the trade-off between natural and
robust accuracy). τb −τm = 1.2 in Eqn.10 would basically
produce a good result via training stage re-balancing, while
τb −τm = 0 with τp = 1.5 would also work well based on
pure boundary adjustment at inference time. The optimal
value of τp relies mainly on τb −τm. The ablation study
includes detailed comparisons. Other hyper-parameters are
less sensitive and have relatively small impact on the per-
formance, where we adopt s = 10, γ ∈{1/32, 1/16}, and
we set α = 6, 3 in Eqn.12 for CIFAR-10-LT and CIFAR-
100-LT, respectively.
A.2. Code References
For the defense methods we compare with, we leverage
the ofﬁcially released code for them if available, including
TRADES [52] 2, MMA [8] 3, Free [36] 4, and HE [30] 5.
AVmixup [22] are re-implement according to the paper.
For the attacks used for evaluation, we refer to sev-
eral ofﬁcially released code bases and the original papers
for the implementation, including FGSM [10], PGD [26],
MIM [9], C&W [3], and Auto Attack [6] 6.
For the long-tailed recognition methods in Table 1, we
also refer to the ofﬁcial code of them if available.
A.3. Implementation Details of Table 1
In Sec.3.2 of the paper, we revisit and formulate a num-
ber of long-tailed recognition methods. We would report the
hyper-parameters selected for them when combining with
adversarial training framework in our implementation in Ta-
2https://github.com/yaodongyu/TRADES
3https://github.com/BorealisAI/mma_training
4https
:
/
/
github
.
com
/
mahyarnajibi
/
FreeAdversarialTraining
5https://github.com/ShawnXYang/AT_HE
6https://github.com/fra31/auto-attack
ble 1, where we choose the optimal values by searching the
hyper-parameters with a step size of 1 or 0.1.
B. Extensive Experiments
B.1. Loss Functions in Adversarial Training
In Sec.3, a modiﬁed loss function L′CE can be adopted
to AT procedure in three modes: replacing the CE in LA,
LT , or both of them.
We study the effect of the three
modes in Table S6. It can be observed that: 1) replacing
CE in LA of the inner maximization would slightly beneﬁt
the natural accuracy with re-weighting [7], class-aware tem-
perature [50], and bias [28, 31], while re-weighting would
hurt robustness in this scenario; class-aware margin [2] is
beneﬁcial to robust accuracy but hurts the natural accuracy
slightly; 2) replacing CE in LT of the outer minimization or
both LA and LT would result in a signiﬁcantly higher natu-
ral accuracy with class-aware temperature and bias, and the
robust accuracy also raises to some extent.
B.2. Effect of PGD Steps during Training
We use an iteration number of 5 with the step size set as
2/255, approximately 0.0078, for the adversarial training
procedure. We adopt this setting for an acceptable balanc-
ing of natural and robust accuracy of the baseline. We study
the effect of PGD iterations and step sizes in Table S7. As
the iteration number increases, the natural accuracy is im-
proved along with the decline of robust accuracy. Espe-
cially for CIFAR-10-LT that when we change from 5 steps
to 7 steps, there is a sharp decrease in clean accuracy. As a
result, we choose a 5-step PGD for the adversarial training
framework in the paper.
B.3. Intrinsic Properties among Classes
Apart from the distribution of sample numbers, different
intrinsic properties and the confusion cross categories are
also non-negligible factors that lead to varying performance
among classes. As could be seen in Fig.1, when trained
on balanced CIFAR-10, the difference in Anat is relatively
minor, while it reveals the disparity of difﬁculty and vul-
nerability among classes, leading to a signiﬁcant variance
in Arob. Speciﬁcally, Class 2, 3, and 4 demonstrate signiﬁ-
cantly lower robust accuracy compared with others.
To study this phenomenon, we train a network on the bal-
anced CIFAR-10 and visualize the latent space via t-SNE in
Fig. S6. It shows that the classes with lower Anat, such as
Class 3, obviously have less concentrated and partially over-
lapped distributions, making them easier to be attacked. It
can also be observed that Class 2, 3, and 4 have clearly more
dispersed distributions under the attack, which is consistent
with their low Arob. While under the long-tailed distribu-
tion, Class 3 beneﬁts from the advantage of sample num-
bers over Class 4-9. Therefore, its accuracy becomes even

--- Page 10 ---
Table S5. Hyper-parameters selected for LT methods used in Table 1, where we choose the optimal values by searching the hyper-
parameters with a step of 1 or 0.1. * denotes that we use CB-Focal.
Stage
Methods
Formulation
Hyper-parameters
Training
Vanilla FC
gi = W T
i f(x)
-
Vanilla Cos
gi = f
W T
i ef(x)
temperature s = 16
Class-aware margin [2]
gi = W T
i f(x) −1{i = y} · δi
δmax = 0.5, δ ∝n−1/4
Cosine with margin [42, 30]
gi = f
W T
i ef(x) −1{i = y} · m
m = 0.2, s = 10
Class-aware temperature [50]
gi = W T
i f(x) · (ni/nmax)γ
γ = 0.3
Class-aware bias [28, 31]
gi = Wi
T f(x) + τ log(ni)
τ = 1
Hard-example mining [23]
r(y) = (1 −py)γ, applyed with BCE loss
γ = 2
Re-sampling [37]
rs(i) ∝1/ni
-
Re-weighting* [7]
r(y) = (1 −β)/(1 −βn
y )
β = 0.9999, γ = 2
Fine-tuning
One-epoch re-sampling [17]
hi = W ′T
i f(x), W ′
i re-trained with RS
-
One-epoch re-weighting [2, 7]
hi = W ′T
i f(x), W ′
i ﬁne-tuned with RW
β = 0.9999, γ = 2
Learnable classiﬁer scale [17]
hi = si · W T
i f(x), where si is learnable
-
Inference
Classiﬁer re-scaling [50, 20]
hi = (Wi/nτ
i )T f(x)
τ = 0.3
Classiﬁer normalization [17]
hi = (Wi/ ∥Wi∥τ)T f(x)
τ = 2
Class-aware bias [28]
hi = W T
i f(x) −τ log(ni)
τ = 1
Feature disentangling [40]
hi = W T
i (f(x) −α cos(f(x), d) · d)
α = 0.1
Table S6. Different loss function applications in adversarial train-
ing. Inner, outer, or both denote to replace Cross-Entropy loss
(CE) in the inner maximization of LA, outer minimization of LT ,
or both of them of Eqn.2 in the paper, respectively. A batch size
of 128 is used here different from the main paper, which does not
affect the relative comparison among them.
Method
Apply
Clean
PGD
AA
CE
both
62.29
28.14
26.78
Class-aware margin [2]
inner
61.27
28.22
28.23
outer
60.70
28.04
26.75
both
60.79
28.13
26.97
Re-weighting [7]
inner
66.77
22.15
21.07
outer
62.76
32.76
27.77
both
62.78
33.32
27.94
Class-aware temperature [50]
inner
63.98
26.89
25.96
outer
72.93
30.71
29.45
both
72.70
28.26
27.21
Class-aware bias [28, 31]
inner
64.09
27.27
27.31
outer
71.33
29.25
27.82
both
73.00
29.67
28.28
higher than the original uniform distribution with the help
of the induced prediction bias. A joint analysis of the effect
by both intrinsic properties and the distribution of sample
numbers among classes would be an interesting direction in
the future.
Table S7. Effect of different iteration numbers and step size in the
inner maximum of the adversarial training procedure.
Adversarial Training
CIFAR-10-LT
CIFAR-100-LT
Iterations
Step size
Clean
PGD
Clean
PGD
1
0.031
64.94
25.39
47.96
14.23
3
0.010
64.03
26.44
47.33
15.32
5
0.0078
62.29
28.14
46.16
15.91
7
0.0078
58.92
29.70
45.23
16.82
10
0.0078
57.61
29.27
45.31
17.40
B.4. Experiments on ImageNet-LT
We also evaluate our method on the more complicated
ImageNet-LT [25] to encourage the exploration of real-
world robustness. Due to the high resolution and large data
scale, we adopt the standard single-step adversarial training
(FGSM) and Fast adversarial training [45]. We use ResNet-
50 as the backbone with ϵ = 2/255 and 4/255 follow-
ing [36, 45]. The preliminary results are shown in Table S8.
Experimental results validate the effectiveness of our ap-
proach over the baseline. The relatively lower performance
on ImageNet-LT compared to CIFAR also indicates that ad-
versarial defense on the 1000-class ImageNet-LT is a more
challenging problem, which is worth further exploration by
the community.

--- Page 11 ---
Clean image features
Attacked image features
Figure S6. Latent space visualization before and after the attack.
Table S8. Adversarial robustness results on ImageNet-LT.
Method
ϵ
CLEAN
FGSM
PGD-20
FAST-AT
2 / 255
11.36
8.23
7.16
FAST-Our
15.45
11.51
10.31
FGSM-AT
2 / 255
25.64
15.32
14.59
FGSM-Our
30.02
18.50
17.67
FAST-AT
4 / 255
7.20
4.52
3.76
FAST-Our
10.76
7.28
6.13
FGSM-AT
4 / 255
21.94
10.88
9.45
FGSM-Our
25.88
13.49
11.87
C. Adversarial Attacks
Fast Gradient Sign Method (FGSM) [10] is a single-step
attack that generates adversarial examples through a permu-
tation along the gradient of the loss function with respect to
the clean image as:
xadv = x + ϵ · sign(∇xLCE(xadv
t
, y)).
(13)
Projected Gradient Descent (PGD) [26] starts from an ini-
tialization point that is uniformly sampled from the allowed
ϵ −ball centered at the clean image, and it extends FGSM
by iteratively applying multiple small steps of permutation
updating with respect to the current gradient as:
xadv
t+1 = clipx,ϵ(xadv
t
+ η · sign(∇xLCE(xadv
t
, y))). (14)
Momentum Iterative gradient-based Methods (MIM)
[9] integrates the momentum into BIM with a decay factor
µ,
gt+1 = µ · gt +
∇xLCE(xadv
t
, y)
∇xLCE(xadv
t
, y)

1
,
(15)
and the permuted image is updated by:
xadv
t+1 = clipx,ϵ(xadv
t
+ η · sign(gt+1))).
(16)
Carlini & Wagner (C&W) [3] is another powerful attack
based on optimization, where an auxiliary variable ω is in-
duced and an adversarial example constrained by l2 norm is
represented by x′ = 1
2(tanh ω +1). It can be optimized by:
argmin
ω
{c · f(x′) + ∥x′ −x∥2
2},
(17)
where
f(x′) = max(max
i̸=y Z(x′) −Z(x′)y, −κ),
(18)
and here κ controls the conﬁdence of the adversarial exam-
ples. It can also be extended to other lp threat model by
solving c · f(x + δ) + ||δ||p in an iterative manner.
Auto Attack [6] is a combination of multiple attacks
that forms a parameter-free and computationally af-
fordable ensemble of attacks to evaluate adversarial
robustness.
The standard attacks includes four selected
attacks: APGDCE, targeted version of APGD-DLR and
FAB, and Square Attack.
Here we use the ﬁrst two in
our evaluation, because since the attack is applied in a
curriculum manner, we empirically observe that after
targeted APGD-DLR, basically few adversarial examples
are further explored by the last two attacks. So the change
in the tested results of robust accuracy is quite small
while the evaluation time can be signiﬁcantly shortened.
References
[1] Mateusz Buda, Atsuto Maki, and Maciej A Mazurowski. A
systematic study of the class imbalance problem in convo-
lutional neural networks. Neural Networks, 106:249–259,
2018. 2
[2] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga,
and Tengyu Ma. Learning imbalanced datasets with label-
distribution-aware margin loss. In Advances in Neural Infor-
mation Processing Systems (NIPS), pages 1565–1576, 2019.
2, 3, 4, 9, 10
[3] Nicholas Carlini and David Wagner. Towards evaluating the
robustness of neural networks. In IEEE Symposium on Se-
curity and Privacy (S & P), pages 39–57. IEEE, 2017. 4, 9,
11
[4] Yair Carmon, Aditi Raghunathan, Ludwig Schmidt, John C
Duchi, and Percy S Liang. Unlabeled data improves adver-
sarial robustness. In Advances in Neural Information Pro-
cessing Systems (NIPS), pages 11192–11203, 2019. 2
[5] Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and
W Philip Kegelmeyer.
Smote: synthetic minority over-
sampling technique.
Journal of artiﬁcial intelligence re-
search, 16:321–357, 2002. 2
[6] Francesco Croce and Matthias Hein.
Reliable evalua-
tion of adversarial robustness with an ensemble of diverse
parameter-free attacks. In International Conference on Ma-
chine Learning (ICML), 2020. 3, 4, 7, 9, 11
[7] Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge
Belongie. Class-balanced loss based on effective number of

--- Page 12 ---
samples. Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition (CVPR), 2019. 2, 3, 4, 7, 8,
9, 10
[8] Gavin Weiguang Ding, Yash Sharma, Kry Yik Chau Lui, and
Ruitong Huang. MMA training: Direct input space margin
maximization through adversarial training. In International
Conference on Learning Representations (ICLR), 2020. 2, 7,
9
[9] Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun
Zhu, Xiaolin Hu, and Jianguo Li. Boosting adversarial at-
tacks with momentum. In Proceedings of the IEEE Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
pages 9185–9193, 2018. 9, 11
[10] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy.
Explaining and harnessing adversarial examples. 2015. 7, 9,
11
[11] Sven Gowal, Chongli Qin, Jonathan Uesato, Timothy Mann,
and Pushmeet Kohli. Uncovering the limits of adversarial
training against norm-bounded adversarial examples, 2020.
3
[12] Agrim Gupta, Piotr Dollar, and Ross Girshick.
LVIS: A
dataset for large vocabulary instance segmentation. In Pro-
ceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 2019. 1
[13] Haibo He and Edwardo A Garcia. Learning from imbalanced
data. IEEE Transactions on Knowledge and Data Engineer-
ing, 21(9):1263–1284, 2009. 1, 2
[14] Dan Hendrycks, Kimin Lee, and Mantas Mazeika. Using
pre-training can improve model robustness and uncertainty.
In International Conference on Machine Learning (ICML),
pages 2712–2721, 2019. 2
[15] Muhammad Abdullah Jamal, Matthew Brown, Ming-Hsuan
Yang, Liqiang Wang, and Boqing Gong. Rethinking class-
balanced methods for long-tailed visual recognition from
a domain adaptation perspective.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 7610–7619, 2020. 2
[16] Nathalie Japkowicz and Shaju Stephen.
The class imbal-
ance problem: A systematic study. Intelligent Data Analysis,
6(5):429–449, 2002. 2
[17] Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan,
Albert Gordo, Jiashi Feng, and Yannis Kalantidis. Decou-
pling representation and classiﬁer for long-tailed recogni-
tion. In Eighth International Conference on Learning Rep-
resentations (ICLR), 2020. 2, 3, 4, 5, 10
[18] Harini Kannan, Alexey Kurakin, and Ian Goodfellow. Adver-
sarial logit pairing. arXiv preprint arXiv:1803.06373, 2018.
7
[19] Salman Khan, Munawar Hayat, Syed Waqas Zamir, Jianbing
Shen, and Ling Shao. Striking the right balance with uncer-
tainty. Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition (CVPR), 2019. 7
[20] Byungju Kim and Junmo Kim. Adjusting decision boundary
for class imbalanced learning. IEEE Access, pages 81674–
81685, 2020. 2, 3, 4, 5, 10
[21] Jaehyung Kim, Jongheon Jeong, and Jinwoo Shin. M2m:
Imbalanced classiﬁcation via major-to-minor translation. In
Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), 2020. 2
[22] Saehyung Lee, Hyungyu Lee, and Sungroh Yoon. Adver-
sarial vertex mixup: Toward better adversarially robust gen-
eralization.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pages
272–281, 2020. 7, 9
[23] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and
Piotr Dollar.
Focal loss for dense object detection.
Pro-
ceedings of the IEEE International Conference on Computer
Vision (ICCV), 2017. 2, 3, 4, 10
[24] Jialun Liu, Yifan Sun, Chuchu Han, Zhaopeng Dou, and
Wenhui Li. Deep representation learning on long-tailed data:
A learnable embedding augmentation perspective. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 2970–2979, 2020. 2
[25] Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang,
Boqing Gong, and Stella X. Yu.
Large-scale long-tailed
recognition in an open world. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), 2019. 2, 4, 7, 10
[26] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
Dimitris Tsipras, and Adrian Vladu. Towards deep learning
models resistant to adversarial attacks. In International Con-
ference on Learning Representations (ICLR), 2018. 2, 3, 7,
9, 11
[27] Chengzhi Mao, Ziyuan Zhong, Junfeng Yang, Carl Vondrick,
and Baishakhi Ray. Metric learning for adversarial robust-
ness. In Advances in Neural Information Processing Systems
(NIPS), pages 480–491, 2019. 2
[28] Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh
Rawat, Himanshu Jain, Andreas Veit, and Sanjiv Kumar.
Long-tail learning via logit adjustment, 2020. 2, 3, 4, 6, 9,
10
[29] Tianyu Pang, Xiao Yang, Yinpeng Dong, Hang Su, and Jun
Zhu. Bag of tricks for adversarial training, 2020. 3
[30] Tianyu Pang, Xiao Yang, Yinpeng Dong, Kun Xu, Hang Su,
and Jun Zhu. Boosting adversarial training with hypersphere
embedding. 2020. 3, 6, 7, 9, 10
[31] Jiawei Ren, Cunjun Yu, Shunan Sheng, Xiao Ma, Haiyu
Zhao, Shuai Yi, and Hongsheng Li. Balanced meta-softmax
for long-tailed visual recognition. Advances in Neural In-
formation Processing Systems (NIPS), 2020. 2, 3, 4, 6, 9,
10
[32] Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urta-
sun. Learning to reweight examples for robust deep learning.
arXiv preprint arXiv:1803.09050, 2018. 2
[33] Leslie Rice, Eric Wong, and J Zico Kolter. Overﬁtting in
adversarially robust deep learning. International Conference
on Machine Learning (ICML), 2020. 2
[34] Ludwig Schmidt, Shibani Santurkar, Dimitris Tsipras, Kunal
Talwar, and Aleksander Madry. Adversarially robust gener-
alization requires more data. In Advances in Neural Infor-
mation Processing Systems (NIPS), pages 5014–5026, 2018.
2
[35] Ali Shafahi, Amin Ghiasi, Furong Huang, and Tom Gold-
stein. Label smoothing and logit squeezing: A replacement
for adversarial training?, 2019. 2

--- Page 13 ---
[36] Ali Shafahi, Mahyar Najibi, Mohammad Amin Ghiasi,
Zheng Xu, John Dickerson, Christoph Studer, Larry S Davis,
Gavin Taylor, and Tom Goldstein. Adversarial training for
free!
In Advances in Neural Information Processing Sys-
tems (NIPS), pages 3358–3369, 2019. 2, 9, 10
[37] Li Shen, Zhouchen Lin, and Qingming Huang. Relay back-
propagation for effective learning of deep convolutional neu-
ral networks. In Proceedings of the European Conference on
Computer Vision (ECCV), pages 467–482. Springer, 2016. 2,
3, 4, 10
[38] Jun Shu, Qi Xie, Lixuan Yi, Qian Zhao, Sanping Zhou,
Zongben Xu, and Deyu Meng. Meta-weight-net: Learning
an explicit mapping for sample weighting. In Advances in
Neural Information Processing Systems (NIPS), 2019. 2
[39] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan
Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus.
Intriguing properties of neural networks.
arXiv preprint
arXiv:1312.6199, 2013. 1
[40] Kaihua Tang, Jianqiang Huang, and Hanwang Zhang. Long-
tailed classiﬁcation by keeping the good and removing the
bad momentum causal effect. In Advances in Neural Infor-
mation Processing Systems (NIPS), 2020. 3, 4, 10
[41] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui,
Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and
Serge Belongie. The inaturalist species classiﬁcation and de-
tection dataset. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 8769–8778,
2018. 1
[42] Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong
Gong, Jingchao Zhou, Zhifeng Li, and Wei Liu. Cosface:
Large margin cosine loss for deep face recognition. In Pro-
ceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), pages 5265–5274, 2018. 3, 6,
10
[43] Jiaqi Wang,
Wenwei
Zhang,
Yuhang
Zang,
Yuhang
Cao, Jiangmiao Pang, Tao Gong, Kai Chen, Ziwei Liu,
Chen Change Loy, and Dahua Lin. Seesaw loss for long-
tailed instance segmentation. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), 2021. 2
[44] Yisen Wang, Difan Zou, Jinfeng Yi, James Bailey, Xingjun
Ma, and Quanquan Gu. Improving adversarial robustness
requires revisiting misclassiﬁed examples. In International
Conference on Learning Representations (ICLR), 2020. 2
[45] Eric Wong, Leslie Rice, and J Zico Kolter. Fast is better
than free: Revisiting adversarial training. In International
Conference on Learning Representations (ICLR), 2019. 2,
10
[46] Tong Wu, Qingqiu Huang, Ziwei Liu, Yu Wang, and Dahua
Lin. Distribution-balanced loss for multi-label classiﬁcation
in long-tailed datasets. In European Conference on Com-
puter Vision (ECCV), 2020. 2
[47] Cihang Xie, Mingxing Tan, Boqing Gong, Jiang Wang,
Alan L Yuille, and Quoc V Le. Adversarial examples im-
prove image recognition. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 819–828, 2020. 3
[48] Cihang Xie, Mingxing Tan, Boqing Gong, Alan Yuille,
and Quoc V Le.
Smooth adversarial training.
arXiv
preprin:2006.14536, 2020. 3
[49] Cihang Xie and Alan Yuille.
Intriguing properties of ad-
versarial training at scale. In International Conference on
Learning Representations (ICLR), 2019. 2, 4
[50] Han-Jia Ye, Hong-You Chen, De-Chuan Zhan, and Wei-Lun
Chao. Identifying and compensating for feature deviation in
imbalanced deep learning. arXiv preprin:2001.01385, 2020.
3, 4, 7, 9, 10
[51] Dinghuai Zhang, Tianyuan Zhang, Yiping Lu, Zhanxing
Zhu, and Bin Dong. You only propagate once: Accelerating
adversarial training via maximal principle. In Advances in
Neural Information Processing Systems (NIPS), pages 227–
238, 2019. 2
[52] Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric Xing, Lau-
rent El Ghaoui, and Michael Jordan. Theoretically principled
trade-off between robustness and accuracy. In International
Conference on Machine Learning (ICLR), pages 7472–7482,
2019. 2, 7, 8, 9
[53] Boyan Zhou, Quan Cui, Xiu-Shen Wei, and Zhao-Min Chen.
BBN: Bilateral-branch network with cumulative learning for
long-tailed visual recognition. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 1–8, 2020. 2, 4
