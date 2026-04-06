# DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced Semi-Supervised Learning

**Authors**: Oh, Kim, Kim, Lee, Lim, Seo
**Year**: 2022
**arXiv**: 2106.05682
**Topic**: semisupervised
**Relevance**: Distribution-aware pseudo-labeling for imbalanced semi-supervised settings

---


--- Page 1 ---
DASO: Distribution-Aware Semantics-Oriented Pseudo-label
for Imbalanced Semi-Supervised Learning
Youngtaek Oh1
Dong-Jin Kim2
In So Kweon1
1KAIST, South Korea.
2UC Berkeley / ICSI, CA.
1{youngtaek.oh, iskweon}@kaist.ac.kr
2djkim93@berkeley.edu
Abstract
The capability of the traditional semi-supervised learn-
ing (SSL) methods is far from real-world application due
to severely biased pseudo-labels caused by (1) class imbal-
ance and (2) class distribution mismatch between labeled
and unlabeled data.
This paper addresses such a rela-
tively under-explored problem. First, we propose a general
pseudo-labeling framework that class-adaptively blends the
semantic pseudo-label from a similarity-based classifier to
the linear one from the linear classifier, after making the
observation that both types of pseudo-labels have comple-
mentary properties in terms of bias. We further introduce
a novel semantic alignment loss to establish balanced fea-
ture representation to reduce the biased predictions from the
classifier. We term the whole framework as Distribution-
Aware Semantics-Oriented (DASO) Pseudo-label. We con-
duct extensive experiments in a wide range of imbalanced
benchmarks: CIFAR10/100-LT, STL10-LT, and large-scale
long-tailed Semi-Aves with open-set class, and demonstrate
that, the proposed DASO framework reliably improves SSL
learners with unlabeled data especially when both (1) class
imbalance and (2) distribution mismatch dominate.
1. Introduction
Semi-supervised learning (SSL) [7] has shown to be
promising for leveraging unlabeled data to reduce the cost
of constructing labeled data [4,5,36,40,58] and even boost
the performance at scale [29, 49, 69, 70].
The common
approach of these algorithms is to produce pseudo-labels
for unlabeled data based on model’s predictions and uti-
lize them for regularizing model training [29, 38, 58]. Al-
though adopted in a variety of tasks, these algorithms often
assume class-balanced data, while many real-world datasets
exhibit long-tailed distributions [3, 18, 31, 32]. With class-
imbalanced data, the class distribution of pseudo-labels
from unlabeled data becomes severely biased to the ma-
jority classes due to confirmation bias [2].
Such biased
pseudo-labels can further bias the model during training.
Labeled
Unlabeled
DASO PL
SSL dataset
Linear PL
Semantic PL
Eqn. (2)
Eqn. (3)
Linear
Classifier
Similarity
Classifier
DASO
Blending
Head
Tail
Head
Tail
Rel. PL size
Lin. PL
Label
Head
Tail
1
Rel. PL size
Sem. PL
Label
Head
Tail
1
Rel. PL size
DASO
Label
Head
Tail
1
Pseudo-label(PL)generation
PL Blending
Class size
Class size
Figure 1. Glimpse of the DASO framework. DASO reduces the
overall bias in pseudo-labels (PL) from unlabeled data by blend-
ing two complementary PLs from different classifiers. Note that
bias is conceptually illustrated as relative PL size (Rel. PL size),
meaning that pseudo-label size is normalized by actual label size.
Many methods of handling class-imbalanced labels have
been proposed in the supervised learning community, but
little interest has been made in re-balancing pseudo-labels
in SSL. Recent studies have explored this imbalanced SSL
setting, where as a reference to the class distribution of unla-
beled data, it is often assumed that it is the same as the class
distribution of labels [33,66], or a separate distribution esti-
mate is required [33]. However, the actual class distribution
of unlabeled data is unknown without the labels. For ex-
ample, unlabeled data may have large class distribution gap
from labeled data, including many samples in novel classes
not defined in the label set [60]. As we elaborate in Sec. 4,
the bias of pseudo-labels also depends on such class distri-
bution mismatch between labeled and unlabeled data, and
using inaccurate estimates or wrong assumptions about the
unlabeled data cannot be helpful under imbalanced SSL.
In this work, we present a new imbalanced SSL method
specifically tailored for alleviating the bias in pseudo-labels
under class-imbalanced data, while discarding the common
assumption that the class distribution of unlabeled data is
the same with the label distribution. To this end, as shown
in Fig. 1, we observe that semantic pseudo-labels [22] ob-
tained from a similarity-based classifier [57] are biased to-
wards minority classes as opposed to linear classifier-based
1

--- Page 2 ---
pseudo-labels [38, 58] being biased towards head classes.
As illustrated in Sec. 3.2, we draw the key inspiration from
those complementary properties of two different types of
pseudo-labels to develop a new pseudo-labeling scheme.
In this regard, we introduce a generic imbalanced SSL
framework termed Distribution-Aware Semantics-Oriented
(DASO) Pseudo-label in Sec. 3.3. Building upon the exist-
ing SSL learner, we propose to blend the linear and seman-
tic pseudo-labels in different proportions for each class to
reduce the overall bias. This blending strategy can provide
a more balanced supervision than simply using either of the
pseudo-label. The primary novelty comes from the schedul-
ing of the weights for mixing the pseudo-labels. Specifi-
cally, we dynamically adjust the relative weights of seman-
tic pseudo-labels to be blended so that linear pseudo-labels
are less biased according to the current class distribution of
pseudo-labels. By virtue of such mechanism, without re-
sorting to any class priors for the unlabeled data, DASO
reliably brings performance gain even with substantial class
distribution mismatch between labeled and unlabeled data.
We further propose a simple yet effective semantic align-
ment loss to establish balanced feature representation via
balanced class prototypes, which is the extension of the
consistency regularization framework in [58, 68] onto fea-
ture space. We align the unlabeled data onto each of the
similar prototypes, by consistently assigning two different
views of an unlabeled sample in feature space to the same
prototype. These enhanced feature representations not only
help linear classifier produce less biased predictions, but can
also be reused for semantic pseudo-labels from similarity-
based classifier. We validate the semantic alignment loss is
useful under imbalanced SSL, especially helpful for DASO.
The efficacy of DASO is extensively justified with the
imbalanced versions of benchmarks: CIFAR-10/100 [35]
and STL-10 [12] in Sec. 4. We even test DASO with large-
scale long-tailed Semi-Aves [60] with open-set classes in
unlabeled data, closely related to real-world scenarios. As
such, DASO consistently benefits under various distribu-
tions of unlabeled data and degrees of imbalance, demon-
strating to be a truly generic framework that works well on
top of diverse frameworks such as existing SSL learners and
even other re-balancing frameworks for labels and SSL.
The key contributions in our work can be summa-
rized as follows: (1) We propose a novel pseudo-labeling
framework, DASO, for debiasing pseudo-labels by class-
adaptively blending two complementary types of pseudo-
labels observing current class distribution of pseudo-labels.
(2) DASO introduces semantic alignment loss to further al-
leviate the bias from high-quality feature representation, by
aligning each unlabeled example to the similar prototype.
(3) DASO readily integrates with other frameworks to show
significant performance improvements under diverse imbal-
anced SSL setup, including the most practical scenario.
2. Related Work
Class-imbalanced learning. Datasets that well capture the
dynamic nature of real-world exhibit class-imbalanced, or
long-tailed distributions [21,63]. Learning on such datasets
has been a great challenge to deep neural networks, since
they cannot generalize well to the rare classes [3]. Con-
ventional approaches to combat the imbalance include data
re-sampling [1,8,34], cost-sensitive re-weighting [6,14,47],
and decoupling the representation and the classifier [27,74].
Recently, learning expert models across classes [64, 67]
and re-balancing with the data distribution in loss compu-
tation phase [25, 43,51] are also shown to be effective. On
the other hand, [42, 71] leveraged unlabeled data for class-
imbalanced learning. Unlike all the aforementioned meth-
ods, we focus on alleviating the bias of pseudo-labels in
semi-supervised learning due to class imbalanced labels and
distribution mismatch between labeled and unlabeled data.
Semi-supervised learning (SSL). SSL aims to learn from
both labeled and unlabeled data. For unlabeled data, SSL
generates targets (e.g., pseudo-labels) from model predic-
tions via pseudo-labeling [29, 38], consistency regulariza-
tion [44, 61], and combinations of them [4, 5, 30, 36] un-
der cluster assumption [7].
However, pseudo-labels can
be biased with class-imbalanced data [33], which harm
the model when utilized. Some works deal with such is-
sue via loss re-weighting [26, 29, 39], optimization [33],
data re-sampling [66], and meta-learning sample impor-
tance [52,53]. However, class distribution of unlabeled data
either unknown or different from the labeled one can also
exacerbate the bias, limiting the applicability of such meth-
ods. In this aspect, we devise a new pseudo-labeling method
that handles such challenging but practical scenarios.
3. Proposed Method
3.1. Preliminaries
Problem setup. We consider K-class semi-supervised im-
age classification that leverages both labeled data X
=
{(xn, yn)}N
n=1 and unlabeled data U = {um}M
m=1 to train
a model f. Note that the model f = f cls
ϕ ◦f enc
θ
consists
of a feature encoder f enc
θ
followed by a linear classifier f cls
ϕ ,
where θ and ϕ are the set of parameters of f enc
θ
and f cls
ϕ .
The input image x is paired with the label y to learn Lcls
(e.g., cross-entropy) from the prediction f(x). For the un-
labeled data, a pseudo-label1 ˆp ∈RK is assigned to learn
the unsupervised loss Lu = Φu (ˆp, f(u)), where Φu can
be implemented via entropy [19] or consistency regulariza-
tion [37,61], depending on the SSL learner.
For FixMatch [58] as an example, the pseudo-label ˆp =
OneHot

argmaxk p(w)
k

with p(w) = f (Aw(u)) provides
1In this work, we assume it includes both one-hot form and soft form
cases: Σk ˆpk = 1 where ˆpk ∈[0, 1].
2

--- Page 3 ---
{C0, C1, C2}
{C3, C4, C5, C6}
{C7, C8, C9}
Class index
0.0
0.2
0.4
0.6
0.8
1.0
Recall
FixMatch (avg: 0.68)
USADTM (avg: 0.74)
DASO (avg: 0.79)
(a) Recall of pseudo-labels.
{C0, C1, C2}
{C3, C4, C5, C6}
{C7, C8, C9}
Class index
0.0
0.2
0.4
0.6
0.8
1.0
Precision
FixMatch (avg: 0.84)
USADTM (avg: 0.57)
DASO (avg: 0.76)
(b) Precision of pseudo-labels.
{C0, C1, C2}
{C3, C4, C5, C6}
{C7, C8, C9}
Class index
0
20
40
60
80
100
Test Top1 Accuracy (%)
FixMatch (avg: 68.6%)
USADTM (avg: 72.3%)
DASO (avg: 76.3%)
(c) Class-wise test accuracy.
Figure 2. Analysis on recall and precision of pseudo-labels and the corresponding test accuracy. Note that the class index from x-axis is
sorted by the class size; C0 and C9 are the head and tail classes, respectively. Although USADTM [22] improves the recall of minority
classes, the precision of those classes is significantly reduced. In contrast, DASO improves the recall of minority classes while sustaining
the precision, which leads to higher test accuracy of those classes. More analyses with various SSL methods are provided in Appendix E.1.
the target for the prediction p(s) = f (As(u)) with some
confident ones to the cross-entropy loss H as follows:
  \lab el {e q n
:
fix
m
atch
}
 \
P
h
i
 _u (\hat
 {p},\,p^{{(s)}}) = \mathbbm {1}\left (\max _k p^{{(w)}}_k \geq \tau \right )\,\cH \left (\hat {p},\,p^{{(s)}} \right ), 
(1)
where Aw and As correspond to weak augmentation (e.g.,
random flip and crop) and advanced augmentation (e.g.,
RandAugment [13] followed by Cutout [17]), respectively.
Imbalanced semi-supervised learning. Let us denote Nk
and Mk as the number of labeled and unlabeled examples
respectively in class k. The degree of imbalance for each
data is characterized by the imbalance ratio, γl or γu, where
we assume γl = maxk Nk
mink Nk ≫1 under imbalanced SSL. γu
is specified in the same way using the actual labels with-
out access during training. It is worth noting that the class
distribution of U (e.g., γu) may be either similar to X, or
significantly divergent in practice, and such varying distri-
butions greatly affect the SSL performances with the same
X as shown in Table 3. In this regard, our goal is to produce
debiased pseudo-labels with class-imbalanced data, while
maintaining the performances of SSL algorithms with vari-
ous, but still unknown class distribution of unlabeled data.
3.2. Motivation
Linear and semantic pseudo-label. Pseudo-labeling based
on linear classifier (i.e., fc layer), which has been widely
adopted by pseudo-label-based algorithms [10,30–32] espe-
cially for SSL [38,58], can produce biased pseudo-labels to-
wards majority classes with class-imbalanced data. We ab-
breviate this type of pseudo-labels as linear pseudo-labels.
Instead, pseudo-labels can be obtained from similarity-
based classifier [15, 54] by measuring the similarity of a
given representation (e.g., prototypes [57]) to an unlabeled
sample in feature space, which we call simply semantic
pseudo-labels. As note, similarity-based classifier has been
widely adopted for reducing biased predictions [27,41,50].
In SSL, USADTM [22] utilizes semantic pseudo-labeling
method. As following, we conduct a simple experiment to
explore each aspect of linear and semantic pseudo-labels.
Trade-offs between linear and semantic pseudo-label.
As shown in Fig. 2, we compare FixMatch [58] and US-
ADTM [22] using linear and semantic pseudo-label respec-
tively, under imbalanced SSL setup. From Figs. 2a and 2b,
the linear pseudo-labels from FixMatch achieve high recall
in majority classes while low recall but high precision in the
minorities, suggesting that actual minority class examples
are biased towards head classes. In contrast, for semantic
pseudo-labels from USADTM, the actual majorities are bi-
ased towards minority classes. This is because the precision
of tail classes has decreased significantly in Fig. 2b, while
the recall has increased in sacrifice of the recall from head
classes in Fig. 2a. Comparing the test accuracy from Fig. 2c,
USADTM shows relatively increased overall test accuracy
compared to FixMatch by virtue of more abundant minor-
ity pseudo-labels, while losing the accuracy on the head. In
other words, the overall increase in accuracy is limited when
only using semantic pseudo-labels. We provide two lessons
from the simple experiment in Fig. 2, as summarized by:
1. Semantic pseudo-labels are reversely biased towards the
tail side, which lead to the limited accuracy gain.
2. The linear and semantic pseudo-labels have the comple-
mentary properties useful for reducing the overall bias.
These empirical findings motivate us to exploit the linear
and semantic pseudo-labels differently in different classes
for debiasing. For example, as the linear pseudo-label for
a sample u points to the majorities, more semantic pseudo-
label component should contribute to the final pseudo-label
to prevent the false positives towards the head, and the vice
versa when the linear pseudo-label predicts u as minority.
We also present the result of our solution, DASO, in
Fig. 2, where the recall of the final pseudo-label has in-
creased but the overall pseudo-labels are still not biased
towards the minority classes, unlike USADTM. Thanks
to such unbiased pseudo-labels between the head and tail
classes obtained by properly blending two pseudo-labels,
the overall test accuracy also increased a lot from Fig. 2c.
3

--- Page 4 ---
3.3. DASO Pseudo-label Framework
We propose DASO, a generic framework for imbalanced
SSL with two novel contributions as (1) distribution-aware
blending for the linear and semantic pseudo-labels and (2)
semantic alignment loss, which are described as follows.
Framework overview. Without loss of generality, we con-
sider DASO built on top of FixMatch [58] for convenience
in notations, while DASO can easily integrate with other
SSL learners as shown in Tables 1 and 3. First, the linear
and semantic pseudo-label, ˆp and q(w) are produced with a
feature z(w) = f enc
θ (Aw(u)) from the linear and similarity-
based classifier, respectively. Then the final pseudo-label ˆp′
is obtained from the distribution-aware blending process us-
ing ˆp and q(w), and it provides the target to Lu = Φu(ˆp′, p)
instead of linear pseudo-label in the existing SSL learner.
In case of FixMatch, the prediction of u corresponds to
p = p(s) = f(As(u)). For the semantic alignment loss, the
semantic pseudo-label q(w) provides the target for q(s) to
the cross-entropy, where q(s) is the result of the similarity-
based classifier with z(s) = f enc
θ (As(u)). Note that we de-
note q(w) as ˆq for simplicity, unless confusion arises.
Balanced prototype generation. To execute a similarity-
based classifier for obtaining the semantic pseudo-label, we
first build a set of class prototypes C = {ck}K
k=1 from X,
similar to [22]. In detail, we build a dictionary of memory
queue Q = {Qk}K
k=1 where each key corresponds to the
class and Qk denotes a memory queue for class k with the
fixed size |Qk|. The class prototype ck for every class k
is efficiently calculated by averaging the feature points in
the queue Qk, where we update Qk for all k at every step
by pushing new features from labeled data in the batch and
discarding the most old ones when Qk is full.
The prototype representation can also be imbalanced us-
ing class-imbalanced labeled data. To prevent such biased
prototypes, we additionally propose balancing the proto-
types compared to [22] in two ways. First, instead of the
size of Qk in proportional to the class frequency, we fix the
size of Qk for all k to the same amount as L. By averag-
ing the same number of features from each class, we can
compensate for the prototypes especially for the minority
classes, with earlier samples remaining in Qk. Secondly,
we adopt momentum encoder f enc
θ′ when extracting the fea-
tures for prototype generation inspired by [23]. Note that
f enc
θ′ has the same architecture with f enc
θ , but θ′ is the expo-
nential moving average (EMA) of θ with momentum ratio
ρ, i.e., θ′ ←ρθ′+(1−ρ)θ. This stabilizes the movement of
each prototype in feature space across iteration by slowing
the pace of network parameter updates. We will verify the
effectiveness of balanced prototypes in Table 7.
Linear and semantic pseudo-label generation. We obtain
the linear pseudo-label ˆp using the linear classifier followed
by softmax activation: ˆp = σ(f cls
ϕ (z(w))). The semantic
pseudo-label ˆq is obtained from the similarity-based classi-
fier that measures the per-class similarity of a query feature
point z of either z(w) or z(s) to the balanced prototypes C:
  \ label { eq n :sim_pl } q = \sigma \left (\text {sim}(z, \mC ) \mathbin {/} T_\text {proto}\right ), 
(2)
where sim(·, ·) corresponds to cosine similarity, and Tproto is
a temperature hyper-parameter for the classifier. Note that
ˆp is biased towards head classes while ˆq is the vice versa.
Distribution-aware blending. To obtain class-specific un-
biased pseudo-label ˆp′, the semantic pseudo-label ˆq should
be exploited differently across the class. To this end, we
propose a novel blending method for pseudo-labels, where
we increase the exposure of the component of ˆq when ˆp is
more biased to the head classes. Formally, we blend them
with a set of distribution-aware weights υ = {υk}K
k=1 to
reduce the bias that might occur when using either ˆp or ˆq:
 \ l ab e l {e qn : mix up} \hat {p}^{\prime } = (1-\upsilon _{k^{\prime }})\, \hat {p} + \upsilon _{k^{\prime }} \hat {q}, 
(3)
where k′ is the class prediction from ˆp, and each υk is de-
rived as υk =
1
maxk ˆm
1/Tdist
k

ˆm1/Tdist
k

. Note that ˆm is the
normalized class distribution of the current pseudo-labels,
which is the accumulation of ˆp′ over a few previous itera-
tions and Tdist is a hyper-parameter that intercedes the op-
timal trade-offs between ˆp and ˆq. Overall, in terms of the
linear pseudo-label, the minority pseudo-labels will remain
as minority, while pseudo-labels predicted as majority will
be likely to recover the original classes thanks to large υk′.
Note that we dynamically adjust the set of weights υ that
determines relative intensity of ˆq in Eq. (3), based on the
current bias of pseudo-labels ˆm. This makes DASO flex-
ible to various distributions of U without resorting to any
pre-defined distribution. For example, even under the same
prediction of ˆp for a head class, more ˆq is blended when
the current model is more biased. Similarly, a concurrent
work [65] accumulates predictions for adaptive debiasing.
Semantic alignment loss. To establish more balanced fea-
ture representations, we propose new semantic alignment
loss for regularizing the feature encoder f enc
θ . It extends the
consistency training framework with two asymmetric aug-
mentations Aw and As like [58, 68] onto feature space. In
high-level, we align each unlabeled sample u to the most
similar prototype used in the similarity-based classifier, by
imposing consistent assignment for two augmented views
Aw(u) and As(u) to the same ck in feature space. Note ˆq is
reused to provide the target for q(s) with the cross-entropy
loss H:
  \lab e l
 
{eq n:ali
gnment} \cL _{\text {align}} = \cH \left (\hat {q},\, q^{(s)} \right ), 
(4)
where q(s) is from the similarity-based classifier by passing
through z(s) = f enc
θ (As(u)) to Eq. (2). Since Lalign re-
lates unlabeled data to the label space through consistently
assigning to C constructed from labeled features, such en-
hanced representation can implicitly guide the classifier f cls
ϕ
4

--- Page 5 ---
to produce less biased predictions in general, where we val-
idate the efficacy of Lalign in Secs. 4.4 and 4.5 respectively.
Total objective. DASO is a generic framework that can
easily couple with other SSL algorithms with the modified
pseudo-label, where the final DASO objective is as below:
  \la b el { e qn:t o tal} \cL _{\text {DASO}} = \cL _{\text {cls}} + \lambda _u \cL _{u} + \lambda _{\text {align}} \cL _{\text {align}}, 
(5)
where both Lcls and Lu with λu come from the base SSL
learner, and Lalign is newly introduced from DASO. Note
that Lu takes the proposed blended pseudo-label in Eq. (3)
instead of the original linear pseudo-label of the learner. We
emphasize that DASO is also applicable to traditional SSL
algorithms for performance gain without Lalign due to the
absence of As in the algorithm, as validated in Table 3.
4. Experiments
4.1. Experimental Setup
To ensure reproducibility2, all the settings of DASO and
other baseline methods are clarified in Appendix C.3.
Datasets. We conduct SSL experiments with various sce-
narios where the class distribution of unlabeled data is not
just limited to the class distribution of labeled data. To ac-
commodate such conditions, we adopt CIFAR-10/100 [35]
and STL-10 [12] typically adopted in SSL literature [58].
We make the imbalanced versions by exponentially decreas-
ing the amount of samples per class [14]. Following [33],
we denote the head class size as N1 (M1), and the imbal-
ance ratio as γl (γu) for the labeled (unlabeled) data respec-
tively. Note that γl and γu can vary independently, and we
specify ‘LT’ for those imbalanced variants. We also con-
sider Semi-Aves benchmark [60] for practical setup, which
is the large-scale collection of bird species with natural
long-tailed distribution. Its unlabeled data also show long-
tailed distribution, and include large portion of examples
in broader categories compared to samples in labeled data
(e.g., open-set). For more details, see Appendix C.1.
Baseline methods. We consider Supervised baseline, learn-
ing cross-entropy with only labeled data. For using unla-
beled data, we mainly adopt FixMatch [58] for its simplic-
ity and powerful performances. To extensively validate our
proposed method in terms of re-balancing, we mainly com-
pare it with the following re-balancing algorithms on top of
FixMatch. Note that the results with other baseline SSL al-
gorithms are provided in Table 3 and the Appendix D.3. We
consider logit adjustment (LA) [43] for balancing labels.
Note that LA can also be applied to SSL methods for re-
balancing using labels. For re-balancing in unlabeled data
similar to our framework, DARP [33] and CReST [66] are
compared. We also experiment with the recently proposed
ABC [39] that performs single unified re-balancing using
both labeled and unlabeled data simultaneously.
2Code is available at: https://github.com/ytaek-oh/daso.
Training and evaluation. We have re-implemented all the
baseline methods using PyTorch [48] and conducted exper-
iments under the same codebase for fair comparison, as
suggested by [45].
We train Wide ResNet-28-2 [72] on
CIFAR10/100-LT and STL10-LT as a backbone. For train-
ing Semi-Aves, we fine-tune the ResNet-34 [24] pre-trained
on ImageNet [16]. To evaluate, we use the EMA network
with the parameters updating every steps, following [5,33].
As note, the class score is measured via learned linear clas-
sifier at inference time. We measure the top-1 accuracy on
the test data every epoch and finally obtain the median of
the accuracy values during the last 20 evaluations [5]. When
reporting the results, we compute the mean and standard de-
viation of three independent runs.
4.2. Results on CIFAR10/100-LT and STL10-LT.
As the main results, we first consider the case when the
distribution of labeled data and unlabeled data is the same
(e.g., γ = γl = γu) in Table 1, which is the ideal case for
SSL. In Table 2, we relax such assumption and test imbal-
anced SSL methods under practical yet challenging scenar-
ios with diverse unlabeled data distributions (e.g., γl ̸= γu).
In case of γl = γu. We compare the proposed DASO with
several baseline methods, with or without class re-balancing
in Table 1. For Supervised case, even if Logit Adjustment
(LA) [43] is applied, the performances are rather limited
compared to even na¨ıve SSL method (i.e., FixMatch [58]).
We then compare imbalanced SSL methods: DARP [33]
and CReST+ [66] with the proposed DASO on FixMatch.
Remarkably, DASO shows comparable or even better re-
sults in most setups with significant gains compared to base-
line FixMatch, although DARP and CReST+ even push the
predictions of unlabeled data to the label distribution using
the assumption γl = γu (i.e., distribution alignment [4]).
This verifies the efficacy of DASO for debiasing pseudo-
labels, even without resorting to the label distribution.
To validate DASO can reliably benefit from re-balancing
labels for debiasing pseudo-labels, we further compare im-
balanced SSL methods on label re-balancing FixMatch via
LA [43] (noted as FixMatch + LA). The results show DASO
performs the best in most of the setups. It is noticeable that
LA with DASO always improves performances compared
to both FixMatch w/ DASO and FixMatch + LA cases.
Finally, we consider ABC [39] in the bottom of Ta-
ble 1. It jointly trains the SSL learner and the auxiliary
balanced classifier (ABC) using both labeled and unlabeled
data with linear pseudo-labels, while the ABC is opted for
evaluation. We find that training ABC can readily be ex-
tended by just replacing the linear pseudo-label for ABC
with DASO pseudo-label (3). Finally, DASO can be signif-
icantly pushed by combining with ABC [39] (i.e., 13% gain
upon FixMatch for CIFAR-10). It verifies the flexibility of
DASO on any baselines regardless of re-balancing methods.
5

--- Page 6 ---
CIFAR10-LT
CIFAR100-LT
γ = γl = γu = 100
γ = γl = γu = 150
γ = γl = γu = 10
γ = γl = γu = 20
Algorithm
N1 = 500
N1 = 1500
N1 = 500
N1 = 1500
N1 = 50
N1 = 150
N1 = 50
N1 = 150
M1 = 4000
M1 = 3000
M1 = 4000
M1 = 3000
M1 = 400
M1 = 300
M1 = 400
M1 = 300
Supervised
47.3 ±0.95
61.9 ±0.41
44.2 ±0.33
58.2 ±0.29
29.6 ±0.57
46.9 ±0.22
25.1 ±1.14
41.2 ±0.15
w/ LA [43]
53.3 ±0.44
70.6 ±0.21
49.5 ±0.40
67.1 ±0.78
30.2 ±0.44
48.7 ±0.89
26.5 ±1.31
44.1 ±0.42
FixMatch [58]
67.8 ±1.13
77.5 ±1.32
62.9 ±0.36
72.4 ±1.03
45.2 ±0.55
56.5 ±0.06
40.0 ±0.96
50.7 ±0.25
w/ DARP [33]
74.5 ±0.78
77.8 ±0.63
67.2 ±0.32
73.6 ±0.73
49.4 ±0.20
58.1 ±0.44
43.4 ±0.87
52.2 ±0.66
w/ CReST+ [66]
76.3 ±0.86
78.1 ±0.42
67.5 ±0.45
73.7 ±0.34
44.5 ±0.94
57.4 ±0.18
40.1 ±1.28
52.1 ±0.21
w/ DASO (Ours)
76.0 ±0.37
79.1 ±0.75
70.1 ±1.81
75.1 ±0.77
49.8 ±0.24
59.2 ±0.35
43.6 ±0.09
52.9 ±0.42
FixMatch + LA [43]
75.3 ±2.45
82.0 ±0.36
67.0 ±2.49
78.0 ±0.91
47.3 ±0.42
58.6 ±0.36
41.4 ±0.93
53.4 ±0.32
w/ DARP [33]
76.6 ±0.92
80.8 ±0.62
68.2 ±0.94
76.7 ±1.13
50.5 ±0.78
59.9 ±0.32
44.4 ±0.65
53.8 ±0.43
w/ CReST+ [66]
76.7 ±1.13
81.1 ±0.57
70.9 ±1.18
77.9 ±0.71
44.0 ±0.21
57.1 ±0.55
40.6 ±0.55
52.3 ±0.20
w/ DASO (Ours)
77.9 ±0.88
82.5 ±0.08
70.1 ±1.68
79.0 ±2.23
50.7 ±0.51
60.6 ±0.71
44.1 ±0.61
55.1 ±0.72
FixMatch + ABC [39]
78.9 ±0.82
83.8 ±0.36
66.5 ±0.78
80.1 ±0.45
47.5 ±0.18
59.1 ±0 .21
41.6 ±0.83
53.7 ±0.55
w/ DASO (Ours)
80.1 ±1.16
83.4 ±0.31
70.6 ±0.80
80.4 ±0.56
50.2 ±0.62
60.0 ±0.32
44.5 ±0.25
55.3 ±0.53
Table 1. Comparison of accuracy (%) with combinations of re-balancing methods on CIFAR10/100-LT under γl = γu setup. Our DASO
consistently improves the performance over all the baselines without or with re-balancing, even with ABC [39] designed for imbalanced
SSL. We indicate the best results for each division as bold. More results including new baseline methods are provided in Appendix D.1.
CIFAR10-LT (γl ̸= γu)
STL10-LT (γu = N/A)
γu = 1 (uniform)
γu = 1/100 (reversed)
γl = 10
γl = 20
Algorithm
N1 = 500
N1 = 1500
N1 = 500
N1 = 1500
N1 = 150
N1 = 450
N1 = 150
N1 = 450
M1 = 4000
M1 = 3000
M1 = 4000
M1 = 3000
M = 100k
M = 100k
M = 100k
M = 100k
FixMatch [58]
73.0 ±3.81
81.5 ±1.15
62.5 ±0.94
71.8 ±1.70
56.1 ±2.32
72.4 ±0.71
47.6 ±4.87
64.0 ±2.27
w/ DARP [33]
82.5 ±0.75
84.6 ±0.34
70.1 ±0.22
80.0 ±0.93
66.9 ±1.66
75.6 ±0.45
59.9 ±2.17
72.3 ±0.60
w/ CReST [66]
83.2 ±1.67
87.1 ±0.28
70.7 ±2.02
80.8 ±0.39
61.7 ±2.51
71.6 ±1.17
57.1 ±3.67
68.6 ±0.88
w/ CReST+ [66]
82.2 ±1.53
86.4 ±0.42
62.9 ±1.39
72.9 ±2.00
61.2 ±1.27
71.5 ±0.96
56.0 ±3.19
68.5 ±1.88
w/ DASO (Ours)
86.6 ±0.84
88.8 ±0.59
71.0 ±0.95
80.3 ±0.65
70.0 ±1.19
78.4 ±0.80
65.7 ±1.78
75.3 ±0.44
Table 2. Comparison of accuracy (%) for imbalanced SSL methods on CIFAR10-LT and STL10-LT under γl ̸= γu setup. For CIFAR10-LT,
γl is fixed to 100, and γu is unknown for STL10-LT. Our DASO consistently shows significant gains on FixMatch [58] without resorting
to any class prior under diverse class distribution mismatches between labeled and unlabeled data. We indicate the best results as bold.
In case of γl ̸= γu. The class distribution of unlabeled data
could be either unknown or arguably different from that of
the labeled data in real-world (e.g., γl ̸= γu). To simulate
such scenarios, for CIFAR10-LT, we consider two extreme
cases for the class distribution of unlabeled data: uniform
(γu = 1) and flipped long-tail (γu = 1/100) with respect to
the labeled data. For STL10-LT, since we cannot control the
size and imbalance of unlabeled data due to unknown labels,
we instead set γl ∈{10, 20} with the whole fixed unlabeled
data. Table 2 summarizes the results of imbalanced SSL
methods under the setups. Note that more comparisons of
SSL methods with different re-balancing techniques (i.e.,
LA [43] and ABC [39]) are presented in Appendix D.2.
Surprisingly, DASO outperforms other baselines by sig-
nificant margins in most cases. For example, DASO shows
13.6% and 18.1% of absolute gain from FixMatch upon
CIFAR-10 (γu = 1) and STL-10 (γl = 20), respectively.
Though DARP [33] estimates the distribution of unlabeled
data in advance as the prior, the estimation accuracy de-
creases as using less labels for training. Under γl ̸= γu, we
evaluate both CReST with self-training only and CReST+
with progressive distribution alignment [66]. Clearly, re-
sorting to the label distributions as the prior for unlabeled
data in CReST+ rather harms the accuracy compared to
CReST, since the assumption of γl = γu is violated. In
particular, when the class distribution of unlabeled data is
completely inverted (γu = 1/100), the accuracy loss be-
comes more severe, resulting in little gain over FixMatch.
By virtue of debiased pseudo-labels from DASO, the
abundant minority-class unlabeled samples are correctly
used despite class-imbalanced labels. Consequently, the re-
sults confirm that conditioning on a certain distribution for
unlabeled data (e.g., γu = γl) is undesirable in imbalanced
SSL, and DASO greatly reduces the bias in presence of dis-
tribution mismatch, even without access to the distribution.
DASO on other SSL learner. To verify DASO is a generic
pseudo-labeling framework, we evaluate DASO based on
other SSL algorithms including MeanTeacher [61], Mix-
Match [5], and ReMixMatch [4] in Table 3. As note, Mean-
Teacher and MixMatch only perform pseudo-label blend-
ing (3) without semantic alignment loss (4) due to the ab-
sence of As. For CIFAR10-LT, we set γl = 100 and for
6

--- Page 7 ---
C10-LT
C100-LT
STL10-LT
N1 = 1500
N1 = 150
N1 = 450
M1 = 3000
M1 = 300
M = 100k
Algorithm
γu = 100
γu = 1
γu = 10
γu : N/A
Mean Teacher [61]
68.6 ±0.88
46.4 ±0.98
52.1 ±0.09
54.6 ±1.17
w/ DASO (Ours)
70.7 ±0.59
87.6 ±0.27
52.5 ±0.37
78.4 ±0.80
MixMatch [5]
65.7 ±0.23
35.7 ±0.69
54.2 ±0.47
52.7 ±1.42
w/ DASO (Ours)
70.9 ±1.91
73.4 ±2.05
55.6 ±0.49
68.4 ±0.71
ReMixMatch [4]
77.0 ±0.55
60.4 ±0.70
61.5 ±0.57
71.9 ±0.86
w/ DASO (Ours)
80.2 ±0.68
90.5 ±0.35
62.1 ±0.69
80.9 ±0.55
Table 3. Comparison of accuracy (%) from DASO upon other
SSL methods: MeanTeacher [61], MixMatch [5], and ReMix-
Match [4]. DASO improves the performances in all the setups.
Benchmark
Semi-Aves
U = Uin
U = Uin + Uout
Method
Last Top1
Med20 Top1
Last Top1
Med20 Top1
Supervised
41.7 ±0.32
41.7 ±0.32
41.7 ±0.32
41.7 ±0.32
FixMatch [58]
53.8 ±0.17
53.8 ±0.13
45.7 ±0.89
46.1 ±0.50
w/ DARP [33]
52.3 ±0.48
52.1 ±0.48
46.3 ±0.70
46.4 ±0.61
w/ CReST [66]
52.1 ±0.36
52.2 ±0.27
43.6 ±0.69
43.6 ±0.68
w/ CReST+ [66]
53.9 ±0.38
53.8 ±0.38
45.1 ±1.09
45.2 ±1.00
w/ DASO (Ours)
54.5 ±0.08
54.6 ±0.12
47.9 ±0.41
47.9 ±0.38
Table 4.
Comparison of accuracy (%) on Semi-Aves bench-
mark [60]. DASO shows the best performance among state-of-
the-art imbalanced SSL methods. Moreover, DASO still performs
well in presence of massive open-set class examples Uout.
Lalign
C10
STL10
FixMatch
✗
68.25
55.53
DASO
✗
70.98
61.64
FixMatch
✓
73.15
58.51
DASO
✓
75.97
70.21
Table 5.
Ablation study on
pseudo-label blending and se-
mantic alignment loss Lalign.
C10
STL10
υk = 0
73.15
58.51
υk = 1
72.35
62.60
υk = 0.5
72.96
64.21
DASO
75.97
70.21
Table 6. Ablation study on the
pseudo-label blending strategy
with Lalign applied.
bal.
EMA
C10
STL10
✗
✗
74.98
68.54
✓
✗
74.54
70.01
✗
✓
75.01
69.49
✓
✓
75.97
70.21
Table 7.
Ablation study on
balancing prototypes and using
EMA encoder on DASO.
C10
STL10
Tdist = 0.3
73.97
70.21
Tdist = 0.5
74.47
68.35
Tdist = 1.0
74.82
65.96
Tdist = 1.5
75.97
64.54
Table 8. Ablation study on Tdist
for DASO. We select Tdist by 1.5
and 0.3 each.
CIFAR100-LT and STL10-LT, we set γl = 10. We observe
that DASO greatly improves the performances for all the se-
tups, and notably, it achieves 2.05× accuracy compared to
MixMatch and brings 29.1% absolute gain in ReMixMatch
on CIFAR10-LT under γu = 1. This implies that DASO
noticeably helps SSL algorithms in general to benefit from
unlabeled data under imbalanced SSL setup. As note, we
show the comparison of imbalanced SSL methods built on
other SSL learner (e.g., ReMixMatch [4]) in Appendix D.3.
4.3. Results on Large-Scale Semi-Aves
We test DASO on a realistic Semi-Aves benchmark [60].
Both labeled data (X) and unlabeled data (U) show long-
tailed distributions, while U contains large open-set exam-
ples (Uout) that do not belong to any of the classes in X. The
results are shown in Table 4. We report both cases: U = Uin
and U = Uin + Uout, where Uin contains examples that share
the class of X. We measure the performances by top-1 accu-
racy, reporting the one in the final (Last Top1) and the me-
dian values in last 20 epochs (Med20 Top1), following [45].
More details on this dataset can be found in Appendix C.1.
In case of U = Uin. As it has the distribution gap between
X and U, baseline DARP [33] and CReST [66] with inade-
quate class prior from X show only a slight gain or even
unsatisfactory performances compared to FixMatch [58].
In contrary, DASO shows the best performance among the
baselines with favorable improvements upon FixMatch.
In case of U = Uin + Uout. Since U contains large amount
of open-set class examples, performance drop is observed
consistently across all baselines, as similar observations are
made in [9, 20, 46]. Among the baselines, DASO shows
the best performance with favorable gain. The results sug-
gest that DARP [33] is slightly helpful when both Uin and
Uout are considered altogether for optimization. Concerning
CReST and CReST+ [66] with self-training, due to noisy
predictions from Uout for constructing datasets for the next
generation, they rather performs poorly than FixMatch. As
such, DASO has superiority in the challenging but practi-
cal scenario of long-tailed distributions, even in presence of
large amount of open-set examples. To understand this, we
further provide the analyses on the confidence plots with or
without DASO using each of Uin and Uout in Appendix E.5.
4.4. Ablation Study
We conduct ablation studies to understand why DASO
reliably provides improvements to baseline methods. To ac-
commodate both γl = γu and γl ̸= γu cases, we consider
FixMatch on CIFAR10-LT with N1 = 500, γ = 100 (noted
as C10) and STL10-LT with N1 = 150, γl = 10 (noted as
STL10) respectively to evaluate each aspect of DASO.
Component analysis. Table 5 studies the two major com-
ponents of DASO: distribution-aware pseudo-label blend-
ing and the semantic alignment loss.
From the table,
both blending mechanism and Lalign provides significant
gain over FixMatch. For example, the blending and Lalign
achieve about 6% and 3% absolute gain, respectively, and
combining both shows 15.7% gain in total on STL10. The
results confirm that both class-adaptively blending linear
and semantic pseudo-labels and the semantic alignment loss
are important for reducing bias under imbalanced SSL.
7

--- Page 8 ---
0k
50k
100k
150k
200k
250k
Training Steps
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
Recall of Pseudo-Label
0k
50k
100k
150k
200k
250k
Training Steps
10
20
30
40
50
60
70
80
Test Accuracy (%)
FixMatch (minority)
FixMatch (overall)
w/ DASO (minority)
w/ DASO (overall)
Figure 3. Train curves for the recall of pseudo-labels (left) and the
test accuracy (right) on CIFAR10-LT. DASO significantly reme-
dies the bias of pseudo-labels on minority classes, and such unbi-
ased pseudo-labels lead to large gains on the test accuracy.
Effect of pseudo-label blending. Table 6 studies the differ-
ent way of pseudo-label blending on DASO with constant
weights. Due to the bias in the pseudo-labels, using either
linear (υk = 0) or semantic (υk = 1) pseudo-label leads to
a marginal gain. In addition, blending them with the same
ratio (υk = 0.5) shows the lower performance compared to
our final DASO, which demonstrates that distribution-aware
class-adaptive blending is crucial for imbalanced SSL.
Effect of balanced prototype. Table 7 studies the differ-
ent design choices of DASO in prototype generation: bal-
anced prototypes (noted as bal.) with EMA encoder (noted
as EMA). When generating class prototypes, using class-
imbalanced queue without EMA encoder leads to worse
performance. In contrary, DASO with both balanced queue
using EMA encoder shows the best performance, showing
that both correspond to the valid components for the bal-
anced prototypes from imbalanced labeled data.
Ablation study on Tdist. In Table 8, we study the effect
of the temperature hyper-parameter Tdist to compute the
weights for pseudo-label blending described in Eq. (3). We
empirically find that, for CIFAR-10 and STL-10, Tdist = 1.5
and Tdist = 0.3 show the best performance respectively.
4.5. Detailed Analysis
In this section, we qualitatively analyze how DASO im-
proves the performance under imbalanced SSL setup. We
consider FixMatch [58] without and with DASO trained on
CIFAR10-LT with γ = 100 and N1 = 500. Note that Ap-
pendix E includes analyses in more various setups.
Unbiased pseudo-label improves test accuracy. We visu-
alize the train curves for the recall of pseudo-labels and the
test accuracy values in Fig. 3. We denote those for the mi-
norities (e.g., last 20% classes) as dashed lines. From the
left of Fig. 3, DASO significantly raises the final recall for
the tail classes, which is 3× compared to that of FixMatch.
From the right, both minority and overall test accuracy val-
ues in final greatly improved by virtue of the less biased
pseudo-labels towards the head classes, which are nearly
3× and 9% compared to those of FixMatch, respectively.
FixMatch
C6
FixMatch w/ DASO (Ours)
Tail 
clusters
C6
C0
C1
C2
C3
C4
C5
C6
C7
C8
C9
C0
C1
C2
C3
C4
C5
C6
C7
C8
C9
Figure 4. Comparison of t-SNE visualization of unlabeled data
from FixMatch (left) and FixMatch w/ DASO (right). Learning
with DASO helps the model to establish tail-class clusters in fea-
ture space, which can further reduce the biases from the classifier.
Tail-class clusters are better identified. To verify the effi-
cacy of reducing the bias, we present t-SNE [62] visualiza-
tions of the encoders’ outputs on U from FixMatch and w/
DASO respectively. As shown in Fig. 4, tail class examples
(e.g., C8 and C9) from FixMatch are scattered to the major-
ity classes. From the right, however, the clusters of tail are
clearly recognized as indicated. In addition, the separability
of C6 is improved. Thanks to such well identified tail-class
clusters from DASO, the actual minority unlabeled exam-
ples are correctly leveraged to learn the unbiased model.
5. Discussion
Conclusion.
We proposed a novel distribution-aware
semantics-oriented (DASO) pseudo-label for imbalanced
semi-supervised learning. DASO adaptively blends the lin-
ear and semantic pseudo-labels within each class to mitigate
the overall bias across the class. Moreover, we introduced
balanced prototypes and semantic alignment loss. From ex-
tensive experiments, we showed the efficacy of DASO on
various challenging and realistic setups, especially when
class imbalance and class distribution mismatch dominate.
Potential societal impact. The proposed solution can con-
tribute to solving various social problems attributed to im-
balance in real-world, such as gender, racial or religious
bias, by improving the fairness of classifiers using unlabeled
data. Also, our method can contribute to the active learning
research [11, 28, 55], which can also suffer from the bias.
However, the proposed algorithm should be carefully con-
sidered as it can be used to raise other fairness issues such
as over-balance or discrimination against minorities.
Limitations. This study focused on alleviating the bias of
pseudo-labels, treating unlabeled data as truly unlabeled.
DASO modulates the debiased pseudo-labels by introduc-
ing a hyper-parameter Tdist, which is effective and efficient
than estimating the class distribution of unlabeled data.
However, Tdist can be highly dependent on each data and
distribution.
As mentioned in [45], tuning such hyper-
parameter is not straightforward under label-scarce setting,
which is the common concern in SSL literature.
8

--- Page 9 ---
Acknowledgements
This research was supported by the National Research
Foundation of Korea (NRF)’s program of developing and
demonstrating innovative products based on public demand
funded by the Korean government (Ministry of Science and
ICT (MSIT)) (No. NRF-2021M3E8A2100445).
References
[1] Shin Ando and Chun Yuan Huang.
Deep over-sampling
framework for classifying imbalanced data. In Joint Euro-
pean Conference on Machine Learning and Knowledge Dis-
covery in Databases, pages 770–785, 2017. 2
[2] Eric Arazo, Diego Ortego, Paul Albert, Noel E O’Connor,
and Kevin McGuinness. Pseudo-labeling and confirmation
bias in deep semi-supervised learning. In International Joint
Conference on Neural Networks (IJCNN), pages 1–8, 2020.
1
[3] Samy Bengio. Sharing representations for long tail computer
vision problems. In ACM International Conference on Mul-
timodal Interaction, pages 1–1, 2015. 1, 2
[4] David Berthelot, Nicholas Carlini, Ekin D. Cubuk, Alex Ku-
rakin, Kihyuk Sohn, Han Zhang, and Colin Raffel. Remix-
match: Semi-supervised learning with distribution matching
and augmentation anchoring. In International Conference on
Learning Representations (ICLR), 2020. 1, 2, 5, 6, 7, 17, 19,
24
[5] David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas
Papernot, Avital Oliver, and Colin A Raffel.
Mixmatch:
A holistic approach to semi-supervised learning.
In Ad-
vances in Neural Information Processing Systems (NIPS),
volume 32, pages 5049–5059, 2019. 1, 2, 5, 6, 7, 17, 21
[6] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga,
and Tengyu Ma. Learning imbalanced datasets with label-
distribution-aware margin loss. In Advances in Neural Infor-
mation Processing Systems (NIPS), volume 32, pages 1567–
1578, 2019. 2, 15, 16, 18
[7] Olivier Chapelle, Bernhard Scholkopf, and Alexander Zien.
Semi-supervised learning.
IEEE Transactions on Neural
Networks, 20(3):542–542, 2009. 1, 2
[8] Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and
W Philip Kegelmeyer.
Smote: synthetic minority over-
sampling technique.
Journal of artificial intelligence re-
search, 16:321–357, 2002. 2
[9] Yanbei Chen, Xiatian Zhu, Wei Li, and Shaogang Gong.
Semi-supervised learning under class distribution mismatch.
In AAAI Conference on Artificial Intelligence (AAAI), vol-
ume 34, pages 3569–3576, 2020. 7
[10] Jae Won Cho, Dong-Jin Kim, Jinsoo Choi, Yunjae Jung, and
In So Kweon. Dealing with missing modalities in the visual
question answer-difference prediction task through knowl-
edge distillation. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, 2021. 3
[11] Jae Won Cho, Dong-Jin Kim, Yunjae Jung, and In So
Kweon. Mcdal: Maximum classifier discrepancy for active
learning. IEEE transactions on neural networks and learn-
ing systems, 2022. 8
[12] Adam Coates, Andrew Ng, and Honglak Lee. An analysis of
single-layer networks in unsupervised feature learning. In In-
ternational Conference on Artificial Intelligence and Statis-
tics (AISTATS), volume 15, pages 215–223, 2011. 2, 5, 15
[13] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V
Le. Randaugment: Practical automated data augmentation
with a reduced search space. In Advances in Neural Infor-
mation Processing Systems (NIPS), 2020. 3, 13, 17
[14] Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge
Belongie. Class-balanced loss based on effective number of
samples. In IEEE Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 9268–9277, 2019. 2, 5, 15,
16, 18
[15] Piew Datta and Dennis Kibler. Symbolic nearest mean clas-
sifiers. In AAAI Conference on Artificial Intelligence (AAAI),
pages 82–87, 1997. 3
[16] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In IEEE Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 248–255, 2009. 5, 15
[17] Terrance DeVries and Graham W Taylor. Improved regular-
ization of convolutional neural networks with cutout. arXiv
preprint arXiv:1708.04552, 2017. 3, 13, 17
[18] Qi Dong, Shaogang Gong, and Xiatian Zhu.
Imbalanced
deep learning by minority class incremental rectification.
IEEE Transactions on Pattern Analysis and Machine Intel-
ligence (TPAMI), 41(6):1367–1381, 2018. 1
[19] Yves Grandvalet and Yoshua Bengio.
Semi-supervised
learning by entropy minimization. In Advances in Neural
Information Processing Systems (NIPS), volume 17, pages
281–296, 2005. 2
[20] Lan-Zhe Guo, Zhen-Yu Zhang, Yuan Jiang, Yu-Feng Li,
and Zhi-Hua Zhou. Safe deep semi-supervised learning for
unseen-class unlabeled data. In International Conference on
Machine Learning (ICML), pages 3897–3906, 2020. 7
[21] Agrim Gupta, Piotr Dollar, and Ross Girshick.
Lvis: A
dataset for large vocabulary instance segmentation. In IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 5356–5364, 2019. 2
[22] Tao Han, Junyu Gao, Yuan Yuan, and Qi Wang. Unsuper-
vised semantic aggregation and deformable template match-
ing for semi-supervised learning.
In Advances in Neural
Information Processing Systems (NIPS), volume 33, pages
9972–9982, 2020. 1, 3, 4, 17, 18, 21
[23] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross
Girshick. Momentum contrast for unsupervised visual repre-
sentation learning. In IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), pages 9729–9738, 2020. 4
[24] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition.
In IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 770–778, 2016. 5, 15
[25] Youngkyu Hong, Seungju Han, Kwanghee Choi, Seokjun
Seo, Beomsu Kim, and Buru Chang. Disentangling label dis-
tribution for long-tailed visual recognition. In IEEE Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
pages 6626–6636, June 2021. 2
9

--- Page 10 ---
[26] Minsung Hyun, Jisoo Jeong, and Nojun Kwak.
Class-
imbalanced semi-supervised learning.
arXiv preprint
arXiv:2002.06815, 2020. 2
[27] Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan,
Albert Gordo, Jiashi Feng, and Yannis Kalantidis. Decou-
pling representation and classifier for long-tailed recogni-
tion. In International Conference on Learning Representa-
tions (ICLR), 2020. 2, 3, 16, 18
[28] Dong-Jin Kim, Jae Won Cho, Jinsoo Choi, Yunjae Jung, and
In So Kweon. Single-modal entropy based active learning for
visual question answering. In British Machine Vision Con-
ference (BMVC), 2021. 8
[29] Dong-Jin Kim, Jinsoo Choi, Tae-Hyun Oh, and In So
Kweon. Image captioning with very scarce supervised data:
Adversarial semi-supervised learning approach. In Confer-
ence on Empirical Methods in Natural Language Processing
(EMNLP), 2019. 1, 2
[30] Dong-Jin Kim, Jinsoo Choi, Tae-Hyun Oh, Youngjin Yoon,
and In So Kweon. Disjoint multi-task learning between het-
erogeneous human-centric tasks. In IEEE Winter Conference
on Applications of Computer Vision (WACV). IEEE, 2018. 2,
3
[31] Dong-Jin Kim, Xiao Sun, Jinsoo Choi, Stephen Lin, and
In So Kweon. Detecting human-object interactions with ac-
tion co-occurrence priors. In European Conference on Com-
puter Vision (ECCV), 2020. 1, 3
[32] Dong-Jin Kim, Xiao Sun, Jinsoo Choi, Stephen Lin, and
In So Kweon.
Acp++: Action co-occurrence priors for
human-object interaction detection. IEEE Transactions on
Image Processing (TIP), 30:9150–9163, 2021. 1, 3
[33] Jaehyung Kim, Youngbum Hur, Sejun Park, Eunho Yang,
Sung Ju Hwang, and Jinwoo Shin. Distribution aligning re-
finery of pseudo-label for imbalanced semi-supervised learn-
ing. In Advances in Neural Information Processing Systems
(NIPS), 2020. 1, 2, 5, 6, 7, 15, 17, 18, 19
[34] Jaehyung Kim, Jongheon Jeong, and Jinwoo Shin. M2m:
Imbalanced classification via major-to-minor translation. In
IEEE Conference on Computer Vision and Pattern Recogni-
tion (CVPR), pages 13896–13905, 2020. 2
[35] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple
layers of features from tiny images. Technical report, 2009.
2, 5, 15
[36] Chia-Wen Kuo, Chih-Yao Ma, Jia-Bin Huang, and Zsolt
Kira.
Featmatch: Feature-based augmentation for semi-
supervised learning. In European Conference on Computer
Vision (ECCV), volume 18, pages 479–495, 2020. 1, 2
[37] Samuli Laine and Timo Aila. Temporal ensembling for semi-
supervised learning. In International Conference on Learn-
ing Representations (ICLR), 2016. 2
[38] Dong-Hyun Lee.
Pseudo-label:
The simple and effi-
cient semi-supervised learning method for deep neural net-
works. In Workshop on challenges in representation learn-
ing, ICML, 2013. 1, 2, 3, 16, 17, 18
[39] Hyuck Lee, Seungjae Shin, and Heeyoung Kim.
Abc:
Auxiliary balanced classifier for class-imbalanced semi-
supervised learning. arXiv preprint arXiv:2110.10368, 2021.
2, 5, 6, 18, 19, 22
[40] Junnan Li, Caiming Xiong, and Steven Hoi. Comatch: Semi-
supervised learning with contrastive graph regularization. In
IEEE International Conference on Computer Vision (ICCV),
2021. 1
[41] Junnan Li, Caiming Xiong, and Steven Hoi. Mopro: We-
bly supervised learning with momentum prototypes. In In-
ternational Conference on Learning Representations (ICLR),
2021. 3
[42] Yunru Liu, Tingran Gao, and Haizhao Yang.
Selectnet:
Learning to sample from the wild for imbalanced data train-
ing. In Mathematical and Scientific Machine Learning, vol-
ume 107, pages 193–206, 2020. 2
[43] Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh
Rawat, Himanshu Jain, Andreas Veit, and Sanjiv Kumar.
Long-tail learning via logit adjustment.
In International
Conference on Learning Representations (ICLR), 2021. 2,
5, 6, 16, 18, 19, 20
[44] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and
Shin Ishii.
Virtual adversarial training: a regularization
method for supervised and semi-supervised learning. IEEE
Transactions on Pattern Analysis and Machine Intelligence
(TPAMI), 41(8):1979–1993, 2018. 2
[45] Avital Oliver, Augustus Odena, Colin A Raffel, Ekin Dogus
Cubuk, and Ian Goodfellow.
Realistic evaluation of deep
semi-supervised learning algorithms. In Advances in Neural
Information Processing Systems (NIPS), volume 31, pages
3235–3246, 2018. 5, 7, 8
[46] Jongjin Park, Sukmin Yun, Jongheon Jeong, and Jin-
woo Shin.
Opencos: Contrastive semi-supervised learn-
ing for handling open-set unlabeled data.
arXiv preprint
arXiv:2107.08943, 2021. 7
[47] Seulki Park, Jongin Lim, Younghan Jeon, and Jin Young
Choi. Influence-balanced loss for imbalanced visual clas-
sification. In IEEE International Conference on Computer
Vision (ICCV), pages 735–744, 2021. 2
[48] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer,
James Bradbury, Gregory Chanan, Trevor Killeen, Zeming
Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison,
Andreas Kopf, Edward Yang, Zachary DeVito, Martin Rai-
son, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner,
Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An
imperative style, high-performance deep learning library. In
Advances in Neural Information Processing Systems (NIPS),
volume 32, page 8026–8037, 2019. 5
[49] Hieu Pham, Qizhe Xie, Zihang Dai, and Quoc V Le. Meta
pseudo labels. In IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), pages 11557–11568, 2021. 1
[50] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg
Sperl, and Christoph H Lampert. icarl: Incremental classifier
and representation learning. In Proceedings of the IEEE con-
ference on Computer Vision and Pattern Recognition, pages
2001–2010, 2017. 3
[51] Jiawei Ren, Cunjun Yu, shunan sheng, Xiao Ma, Haiyu
Zhao, Shuai Yi, and hongsheng Li. Balanced meta-softmax
for long-tailed visual recognition.
In Advances in Neural
Information Processing Systems (NIPS), volume 33, pages
4175–4186, 2020. 2
10

--- Page 11 ---
[52] Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urta-
sun. Learning to reweight examples for robust deep learning.
In International Conference on Machine Learning (ICML),
pages 4334–4343. PMLR, 2018. 2
[53] Zhongzheng Ren, Raymond Yeh, and Alexander Schwing.
Not all unlabeled data are equal: Learning to weight data in
semi-supervised learning. In Advances in Neural Informa-
tion Processing Systems (NIPS), volume 33, pages 21786–
21797, 2020. 2
[54] Ruslan Salakhutdinov and Geoff Hinton. Learning a non-
linear embedding by preserving class neighbourhood struc-
ture. In Artificial Intelligence and Statistics, pages 412–419.
PMLR, 2007. 3
[55] Inkyu Shin, Dong-Jin Kim, Jae Won Cho, Sanghyun Woo,
KwanYong Park, and In So Kweon. Labor: Labeling only
if required for domain adaptive semantic segmentation. In
IEEE International Conference on Computer Vision (ICCV),
2021. 8
[56] Leslie N Smith and Adam Conovaloff. Building one-shot
semi-supervised (boss) learning up to fully supervised per-
formance. arXiv preprint arXiv:2006.09363, 2020. 17, 18
[57] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypi-
cal networks for few-shot learning. In Advances in Neural
Information Processing Systems (NIPS), volume 30, pages
4077–4087, 2017. 1, 3
[58] Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao
Zhang, Nicholas Carlini, Ekin D Cubuk, Alex Kurakin, Han
Zhang, and Colin Raffel.
Fixmatch: Simplifying semi-
supervised learning with consistency and confidence. In Ad-
vances in Neural Information Processing Systems (NIPS),
2020. 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24
[59] Jong-Chyi Su, Zezhou Cheng, and Subhransu Maji. A real-
istic evaluation of semi-supervised learning for fine-grained
classification. In IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 2021. 15
[60] Jong-Chyi Su and Subhransu Maji.
The semi-supervised
inaturalist-aves challenge at fgvc7 workshop, 2021. 1, 2, 5,
7, 15, 23, 24
[61] Antti Tarvainen and Harri Valpola. Mean teachers are better
role models: Weight-averaged consistency targets improve
semi-supervised deep learning results. In Advances in Neural
Information Processing Systems (NIPS), volume 30, pages
1195–1204, 2017. 2, 6, 7, 17, 21, 22
[62] Laurens Van der Maaten and Geoffrey Hinton.
Visualiz-
ing data using t-sne. Journal of machine learning research,
9(11), 2008. 8, 23
[63] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui,
Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and
Serge Belongie. The inaturalist species classification and de-
tection dataset. In IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), pages 8769–8778, 2018. 2, 15
[64] Xudong Wang, Long Lian, Zhongqi Miao, Ziwei Liu,
and Stella Yu. Long-tailed recognition by routing diverse
distribution-aware experts. In International Conference on
Learning Representations (ICLR), 2021. 2
[65] Xudong Wang, Zhirong Wu, Long Lian, and Stella X Yu.
Debiased learning from naturally imbalanced pseudo-labels
for zero-shot and semi-supervised learning. arXiv preprint
arXiv:2201.01490, 2022. 4
[66] Chen Wei, Kihyuk Sohn, Clayton Mellina, Alan Yuille, and
Fan Yang. Crest: A class-rebalancing self-training frame-
work for imbalanced semi-supervised learning.
In IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), 2021. 1, 2, 5, 6, 7, 18, 19, 20
[67] Liuyu Xiang, Guiguang Ding, and Jungong Han. Learning
from multiple experts: Self-paced knowledge distillation for
long-tailed classification. In European Conference on Com-
puter Vision (ECCV), pages 247–263, 2020. 2
[68] Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong,
and Quoc V Le. Unsupervised data augmentation for consis-
tency training. In Advances in Neural Information Process-
ing Systems (NIPS), 2020. 2, 4
[69] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V
Le. Self-training with noisy student improves imagenet clas-
sification. In IEEE Conference on Computer Vision and Pat-
tern Recognition (CVPR), pages 10687–10698, 2020. 1
[70] I Zeki Yalniz, Herv´e J´egou, Kan Chen, Manohar Paluri,
and Dhruv Mahajan. Billion-scale semi-supervised learning
for image classification. arXiv preprint arXiv:1905.00546,
2019. 1
[71] Yuzhe Yang and Zhi Xu. Rethinking the value of labels for
improving class-imbalanced learning. In Advances in Neural
Information Processing Systems (NIPS), 2020. 2
[72] Sergey Zagoruyko and Nikos Komodakis. Wide residual net-
works. In British Machine Vision Conference (BMVC), pages
87.1–87.12, 2016. 5, 15
[73] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and
David Lopez-Paz. mixup: Beyond empirical risk minimiza-
tion. In International Conference on Learning Representa-
tions (ICLR), 2018. 17
[74] Boyan Zhou, Quan Cui, Xiu-Shen Wei, and Zhao-Min Chen.
Bbn: Bilateral-branch network with cumulative learning for
long-tailed visual recognition. In IEEE Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 9719–
9728, 2020. 2
11

--- Page 12 ---
Supplementary Materials for
DASO: Distribution-Aware Semantics-Oriented Pseudo-Label
for Imbalanced Semi-Supervised Learning
Contents
A. Notations
13
B. Algorithm
14
C. Detailed Experimental Setup
15
C.1. Benchmarks
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15
C.2. Training Details
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15
C.3. Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15
D. Additional Experiments
18
D.1. Comprehensive Comparison with More Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
18
D.2. DASO with Label Re-Balancing when γl ̸= γu . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19
D.3. Comparison based on ReMixMatch . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19
D.4. Results on Test-Time Logit Adjustment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20
D.5. More Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20
E. Detailed Analysis
20
E.1. Recall and Precision Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20
E.2. Confusion Matrix on Test Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22
E.3. Train Curves for Recall and Accuracy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23
E.4. Further Comparison of Feature Representations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23
E.5. Confidence Analysis from Out-of-class Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23
F. Overall Framework
24
12

--- Page 13 ---
A. Notations
In this section, we clarify all the notations with corresponding descriptions introduced in this work.
Notation
Description
DASO
Distribution-Aware Semantic-Oriented (Pseudo-label)
SSL
Semi-Supervised Learning.
K
The number of classes in the labeled data.
X, U
Labeled data and unlabeled data.
N, M
Total number of examples in labeled data and unlabeled data.
Nk, Mk
Number of examples in class k for labeled data and unlabeled data.
γl, γu
Imbalance ratio for labeled data and unlabeled data.
ˆm
Empirical pseudo-label distribution in probability form; ˆm ∈[0, 1]K.
σ(·)
Softmax activation.
H(y, p)
Cross-entropy between the target y and prediction p.
sim(·, ·)
Cosine similarity.
f
A classification model; a feature encoder f enc
θ
followed by a linear classifier f cls
ϕ .
f enc
θ′
An EMA encoder (momentum encoder).
ρ
Decay ratio for the momentum encoder.
Q
A dictionary of memory queue; {Qk}K
k=1.
L
The maximum queue size for the balanced memory queue.
C
A set of class prototypes; {ck}K
k=1.
Tproto
A temperature factor for the similarity-based classifier.
Tdist
A temperature factor for the empirical pseudo-label distribution.
ˆp, q(w) or ˆq
A linear pseudo-label and semantic pseudo-label.
υ
Class-specific mixup factor for the linear and semantic pseudo-label; {υk}K
k=1.
ˆp′
A blended pseudo-label.
PseudoLabel(·)
Pseudo-labeler specified by an SSL algorithm.
Φu(·, ·)
A regularizer for U, specified by an SSL algorithm.
λu
The loss weight for Lu.
Lalign
Semantic alignment loss.
λalign
The loss weight for Lalign.
P
Pre-train steps for applying pseudo-label blending and Lalign.
Aw
A set of weak augmentations; horizontal flip and/or crop.
As
A set of strong augmentations; RandAugment [13] followed by Cutout [17].
µ
Unlabeled batch ratio; multiplied to the labeled batch size B.
Table 9. Notations and their descriptions used throughout this work.
13

--- Page 14 ---
B. Algorithm
Algorithm 1 summarizes the blending procedure for the linear and semantic pseudo-labels based on the empirical pseudo-
label distribution, and Algorithm 2 represents the whole DASO framework built upon a typical SSL algorithm where the
regularizer for the SSL algorithm corresponds to Φu.
Algorithm 1 Distribution-aware pseudo-label blending,
ˆp′ ←Blend (ˆp, ˆq, Tdist).
Input: Linear pseudo-label ˆp ∈[0, 1]K, semantic pseudo-label ˆq ∈[0, 1]K,
Temperature factor for the pseudo-label distribution Tdist.
Require: Empirical pseudo-label distribution ˆm = { ˆmk}K
k=1.
Output: Blended pseudo-label ˆp′ ∈[0, 1]K.
for k = 1 to K do
υk ←ˆm1/Tdist
k
{Temperature scaling for empirical pseudo-label distribution.}
υk ←υk/ maxk υk {Normalization for blending.}
end for
k′ ←argmaxk ˆpk
{Class prediction of the linear pseudo-label.}
ˆp′ ←(1 −υk′) ˆp + υk′ ˆq
{Pseudo-label blending.}
Algorithm 2 Distribution-Aware Semantic-Oriented (DASO) Pseudo-label framework.
Input: A batch of labeled data XB = {(xb, yb)}B
b=1 and unlabeled data UB = {ub}µB
b=1.
Network for feature encoder f enc
θ , momentum encoder f enc
θ′ , and linear classifier f cls
ϕ .
Dictionary of memory queue Q = {Qk}K
k=1, Momentum decay ratio ρ.
Maximum queue size L, temperature factor for the similarity-based classifier Tproto,
Pre-train steps for pseudo-label blending P, current training step t.
Require: A set of weak augmentations Aw and strong augmentations As.
{Balanced Prototype Generation.}
Enqueue z(l) into Qk, where z(l) = f enc
θ′ (x) and k ←y,
∀(x, y) ∈XB.
Dequeue the earliest elements from Qk s.t. |Qk| = L,
∀k ∈{1, . . . , K}.
ck ←
1
|Qk|
P
zi∈Qk zi, ∀k ∈{1, . . . , K},
{A set of balanced prototypes C = {ck}K
k=1.}
{Pseudo-label generation.}
for u in UB do
z(w) ←f enc
θ
(Aw(u)),
z(s) ←f enc
θ
(As(u))
{feature extraction}
ˆp ←σ

f cls
ϕ
 z(w)
,
q(w) ←σ
 sim
 z(w), C

/ Tproto

ˆp′ ←Blend
 ˆp, q(w), Tdist

if t ≥P else ˆp
{Blend pseudo-labels after P train steps.}
end for
{Compute losses.}
Lcls ←E(x,y)∈XB [H (y, σ (f(x)))]
Lalign ←Eu∈UB

1 (t ≥P) · H
 q(w), q(s)
where q(s) ←σ
 sim
 z(s), C

/ Tproto

.
Lu ←Eu∈UB

Φu
 ˆp′, p(s)
where p(s) ←f cls
ϕ
 z(s)
.
LDASO ←Lcls + λuLu + λalignLalign
{Update parameters.}
Update θ and ϕ to minimize LDASO via SGD optimizer.
θ′ ←ρθ′ + (1 −ρ)θ
{Update the parameters of momentum encoder.}
t ←t + 1
14

--- Page 15 ---
C. Detailed Experimental Setup
C.1. Benchmarks
In this work, we evaluate both cases of (i) labeled data and unlabeled data shares the same class distribution (e.g., γl = γu),
and (ii) the class distribution of unlabeled data can be different from the labeled data in various degree (e.g., γl ̸= γu).
CIFAR-10 and CIFAR-100. CIFAR benchmarks [35] originally have the same number of examples per class; 5000 and 500
examples in 32 × 32 sized image for CIFAR-10 and CIFAR-100, respectively. We use the head class size N1 and imbalance
ratio of labels γl to craft the synthetically long-tailed variants across the level of imbalance and total amount of labels,
following the protocol from [33]. The number of examples other than the head class is calculated by Nk = N1 · γ
−k−1
K−1
l
as
proposed by [14]. Note that each Nk, the number of examples in class k is sorted in a descending order (i.e., N1 ≥· · · ≥NK).
Similarly, the number of examples per class for the unlabeled data can be determined by: Mk = M1 ·γ
−k−1
K−1
u
using the labels,
and the true labels are thrown away before training. We call those variants as CIFAR10/100-LT, which consist of labeled and
unlabeled splits. We measure the performance on the test data, which have 10k examples in total for both data.
STL-10. To generate STL10-LT: a long-tailed variant of STL-10 [12], we follow the same process as explained in above.
Besides the 5k labeled examples, STL-10 contains additional 100k unlabeled examples from a similar but broader distribution
compared to the labeled data. Since the information about the class distribution of the unlabeled data is not known, we only
construct the imbalanced labeled data and use the whole 100k unlabeled examples for training.
Semi-Aves. We also consider Semi-Aves benchmark [60] for more realistic scenarios. Semi-Aves includes 1k species of
birds sampled from the iNaturalist-2018 [63] with long-tailed class distribution. Moreover, only 200 species are considered
in-class, and the other 800 species correspond to the out-of-class (i.e., novel, open-set) categories for the unlabeled data. For
in-class examples, about 4k examples are labeled (X), and the other 27k examples are unlabeled (Uin). Note that the class
distribution of labeled data does not match that of Uin (γl ̸= γu), as illustrated in [60]. The out-of-class unlabeled data (Uout)
have 122k examples in total. Semi-Aves benchmark provides 2k images and 8k images (i.e., 10 images and 40 images per
class) for the validation and test data, respectively. We combine the labeled training data and validation data, 6k in total, for
the labeled training data in our experiments, following [59]. As note, we do not make any distinction between Uin and Uout
when learning on the whole unlabeled data (U = Uin + Uout).
C.2. Training Details
CIFAR10/100-LT and STL10-LT. Following the training protocol in [33], we train a Wide ResNet-28-2 [72] with 1.5M
parameters for 250k iterations. We set the batch size of the labeled data as 64, and the network is optimized via Nesterov
SGD with momentum 0.9 and weight decay 5e-4. For the methods with using only labels, the base learning rate is set to 0.1
with linear warm-up applied during the first 2.5% of the total train steps, and it decays after 80% and 90% of the training
phase by a factor of 100, respectively, following [6]. For SSL methods, we set the base learning rate as 0.03, which is
fixed during the training. For the exponential moving average (EMA) network parameters for evaluation, the decay ratio ρ
is set to 0.999. We further clarify the details for each method, such as hyper-parameters in Appendix C.3. We measure the
performance every 500 iterations (e.g., considered as 1 epoch), and report the median value in last 20 evaluations.
Semi-Aves. We train ResNet-34 [24] with 21.3M parameters pre-trained on ImageNet [16]. For the Supervised method, we
train for 90 epochs of the labeled data, while we train 90 epochs of unlabeled data for SSL methods, using SGD optimizer
with momentum 0.9. The base learning rate is set to 0.1 and 0.04 for the Supervised and SSL method each, with the linear
warm-up for the first 5 epochs and it decays after 30 and 60 epochs, by a factor of 10. We set the labeled batch size as 256.
All training images are randomly cropped and re-scaled to 224 × 224 size with random horizontal flip. The EMA decay ratio
is ρ = 0.9. The hyper-parameters of the individual method is described in Appendix C.3.
C.3. Implementation Details
DASO. Tdist, for scaling the empirical pseudo-label distribution, is chosen out of {0.3, 0.5, 1.0, 1.5}.
Specifically, for
CIFAR10-LT, Tdist = 1.5 in case of γl = γu, while Tdist = 0.3 in the case of γl ̸= γu. For the other hyper-parameters,
Tproto = 0.05, L = 256, and λalign = 1, which are kept unchanged during experiments. The ablation study for those param-
eters is provided in Appendix D.5. We start applying DASO with Lalign after a few pre-training steps P = 5000 to avoid
unconfident predictions in the early stage of training. For empirical pseudo-label distribution ˆm, we accumulate the class
predictions of the final pseudo-labels ˆp′ every 100 iterations on CIFAR10/100-LT and STL10-LT. For Semi-Aves, we set
P = 20 epochs and update ˆm every epoch. For the EMA decay ratio ρ for prototype generation, we simply use the same
15

--- Page 16 ---
parameter of the one for evaluation. Table 10 summarizes the training details of DASO.
parameter
CIFAR10-LT
CIFAT100-LT
STL10-LT
Semi-Aves
lr
0.03
0.04
B
64
256
µ
2
5
SGD momentum
0.9
0.9
Nesterov
True
True
weight decay
5e-4
3e-4
L
256
256
ρ
0.999
0.9
Tproto
0.05
0.05
λalign
1.0
1.0
P
5000 steps
20 epochs
Tdist
{1.5, 0.3}
0.3
0.3
0.5
Table 10. A complete list of training details for DASO framework.
Supverised. The only labeled data is trained via standard cross-entropy loss H. The training protocol and hyper-parameters
(total iterations, learning rate, optimizer, and etc.) are described in Appendix C.2.
Re-weighting with the Effective Number of Samples [14]. The per-class weights are applied to the cross-entropy loss
based on the effective number of samples.
  E _ { N _k}
 =  \frac {1 - \beta ^{N_k}} {1 - \beta }, 
(6)
where Nk corresponds to the number of samples in class k, and then the weight for class k is set to be proportional to the
inverse of the effective number ENk. β is a hyper-parameter, which is set to 0.999 during the experiments.
LDAM-DRW [6]. Decision boundary of the classifier takes up more margin in rare classes, using LDAM loss:
  \la b e l {
eqn: ldam
} \c L _{ L D
AM} =  -\ ,  \log  \ ,
 
\ fra
c
 { e^{ z_{y_k} - \Delta _{y_k} } } { e^{ z_{y_k} - \Delta _{y_k} } + \sum _{j \neq {y_k}} e^{z_j} },\; \textnormal {where } \Delta _k \propto \frac {1} {N^{1/4}_k}. 
(7)
Then it adopts deferred re-weighting scheme (DRW) to apply re-balancing algorithm in later stage of training. Following
DRW scheme, we apply re-weighting objective Eq. (7) after 200k iterations.
cRT [27]. After training the entire network under imbalanced distribution, the classifier is re-trained with the parameters of
the feature encoder fixed for a balanced objective. We first train a model with cross-entropy loss. In classifier re-training
phase, we simply re-weight the cross-entropy loss with the weights based on the effective number of samples [14] for 100k
iterations. The learning rate schedule under re-training phase is proportionally adjusted.
Logit Adjustment (LA) [43]. Logits are adjusted by enforcing a large margin for the minority classes compared to the
majority ones in either two ways: post-hoc adjustment or logit-adjusted cross-entropy, based on the class frequency of labels.
In this work, we adopt the latter strategy. Before measuring cross-entropy for the labeled data, each logit is adjusted by:
  p _k  \ lef tarrow p_k + \tau \log {n}_k, 
(8)
where p = f(x) and nk denotes the class label frequency value in class k. τ = 1 is a temperature scaling factor.
PseudoLabel [38]. The one-hot pseudo-label ˆp from p = f(u) regularizes the unlabeled example. Only the predictions with
the highest probability value above a certain threshold τ contribute to the regularizer. We set τ to 0.95.
  \Phi  _ u (
\
hat
 
{p } ,
\
, p) =  \ mathbbm {1}\left (\max _k p_k \geq \tau \right )\,\cH \left (\hat {p},\,p \right ), 
(9)
where ˆp = OneHot(argmaxk pk). We set the loss weight λu = 1 and apply linear ramp-up with the ratio of 0.4; λu linearly
increases starting from 0 and attains the maximum value (λu = 1) at 40% of the total iterations.
16

--- Page 17 ---
MeanTeacher [61]. The momentum encoder f EMA = f cls
ϕ′ ◦f enc
θ′
generates the target for the prediction of unlabeled data,
where ϕ′ and θ′ are the momentum-updating network parameters of linear classifier and feature encoder, respectively.
  \Phi  _ u (\hat { p},\, p ) = \ no r m  {\sigma ( \ha t  {p}) - \sigma (p)}^2,\;\textnormal {where}\; \hat {p} = f^{\text {EMA}}(\cA _w(u))\; \textnormal {and}\; p=f(\cA _w(u)). 
(10)
We set the EMA decay ratio ρ = 0.999. λu is set to 50, applying the linear ramp-up with the ratio of 0.4.
MixMatch [5]. Pseudo-label is produced from the multiple augmentations of the same image with entropy regularization.
Then the model learns mixup [73] images and (pseudo-) labels over the whole labeled and unlabeled data. We use the number
of augmentations as 2, temperature scaling factor as 0.5, and the sampling hyper-parameter for mixup regularization α as 0.5.
We also apply linear ramp-up strategy for λu, where it attains its maximum value 100 with the ratio of 0.016.
ReMixMatch [4]. It adds up two techniques of Augmentation Anchoring and Distribution Alignment over MixMatch [5].
We use the advanced augmentation as RandAugment [13] followed by Cutout [17]. Considering the computational cost, we
set the number of advanced augmentations as µ = 2. For the others, we set the temperature scaling factor for pseudo-labels
as 0.5, and α as 0.75. The weights for pre-mixup loss and rotation loss are both set to 0.5. For λu, the linear ramp-up ratio
is set to 0.016 with λu = 1.5. We apply weak augmentations for convenience for the labeled data, instead of advanced
augmentation.
FixMatch [58]. One-hot pseudo-labels are generated from weakly augmented images as the same with PseudoLabel [38],
then they provide the targets for the predictions from strong augmentations of the same images to the cross-entropy loss H:
  \Phi  _u(\ h a
t
 {p
}
,\,p
^
{ {
(
s
)
})  = \m
athbbm {1}\left (\max _k p^{{(w)}}_k \geq \tau \right )\,\cH \left (\hat {p},\,p^{{(s)}} \right ), 
(11)
where ˆp = OneHot

argmaxk p(w)
k

with p(w) = f (Aw(u)) and p(s) = f (As(u)). We use RandAugment [13] for the
advanced augmentation. For fair comparisons to ReMixMatch [4], we use the unlabeled batch ratio µ as 2. For the other
hyper-parameters, λu is set to 1 without applying linear ramp-up strategy.
USADTM [22]. It combines unsupervised semantic aggregation (USA); a clustering objective in unlabeled data and de-
formable template matching (DTM); assigning a semantic pseudo-label to each unlabeled example solely from feature-space.
The semantic pseudo-label is determined by the agreement of two different distance measure from a sample to each class
prototypes constructed from the labeled data. In our experiments, we use the loss weight for the mutual information loss
α = 0.1 and τ = 0.85 for the confidence threshold, following [22]. We note that [22] keeps some confident unlabeled
examples to treat them as labeled examples to enforce cross-entropy loss due to the limited labels (i.e., 4 labels per class).
This would also help generally in imbalanced SSL, but we do not adopt this strategy in our experiments in order to fairly
comparing with other SSL methods focusing on the aspect of pseudo-labeling method.
BOSS [56]. This originally proposes to apply three techniques altogether on FixMatch [58] to achieve state-of-the-art per-
formance on CIFAR-10 benchmark under one label per class: prototype (single-example per class) refining, pseudo-label
re-balancing, and self-training iterations. We only adopt pseudo-label re-balancing method from the original paper for fairly
comparing under imbalanced SSL. Pseudo-label re-balancing includes adjusting loss weights and confidence thresholds
based on the class distribution of predicted pseudo-labels on top of the FixMatch loss:
  \lab el {e q n
:
sup
_
boss
}
 \P
h
i
 _ u(\
h
a
t { p},\,
p^{{(s)}}) = \mathbbm {1}\left (\max _k p^{{(w)}}_k \geq \tau _k \right )\,\frac {1} {Z \cdot \hat {c}_k} \cH \left (\hat {p},\,p^{{(s)}} \right ), 
(12)
where τk is the class-dependent confidence threshold defined as:
  \ t a u  
_
k =
\ta
u - \De
l
ta \cdot \left (1 - \frac {\hat {c}_k} {\max _k \hat {c}_k} \right ), 
(13)
and ˆck is the number of predicted pseudo-labels in the current batch for class k. We fix ∆= 0.25 during the experiments.
Note that the scale of Φu is adjusted by a factor of Z to consistently maintain the relative scale of λu.
DARP [33]. The class distribution of the predicted pseudo-labels is explicitly adjusted to the given class priors via solving
a convex optimization problem. In our experiments, we use the class prior as the class label frequency in case of γl = γu
for CIFAR10-LT and CIFAR100-LT, and in case of Semi-Aves benchmark. In other cases, i.e., γl ̸= γu, we estimate the
distribution of the unlabeled data (e.g., Mk) using held-out validation set, following [33]. We start applying DARP at 100k
iterations of training with refining pseudo-labels every 10 steps. We use α = 2.0 for removing the noisy entries.
17

--- Page 18 ---
Method type
CIFAR10-LT
CIFAR100-LT
STL10-LT
γ = γl = γu = 100
γ = γl = γu = 10
γl = 10, γu: unknown
Algorithm
SSL
LB
PB
N1 = 500
N1 = 1500
N1 = 50
N1 = 150
N1 = 150
N1 = 450
M1 = 4000
M1 = 3000
M1 = 400
M1 = 300
M = 100k
M = 100k
Supervised
47.3 ±0.95
61.9 ±0.41
29.6 ±0.57
46.9 ±0.22
40.2 ±1.80
60.4 ±1.91
w/ LDAM-DRW [6]
✓
50.1 ±1.55
65.7 ±1.49
28.4 ±0.32
46.2 ±0.46
41.8 ±3.05
62.1 ±1.39
w/ cRT [27]
✓
49.5 ±1.05
65.8 ±0.47
30.1 ±0.50
48.0 ±0.43
40.8 ±1.95
61.6 ±1.83
w/ LA [43]
✓
53.3 ±0.44
70.6 ±0.21
30.2 ±0.44
48.7 ±0.89
42.8 ±1.78
63.1 ±1.13
PseudoLabel [38]
✓
47.8 ±1.06
63.4 ±0.81
30.7 ±0.18
47.8 ±0.40
42.3 ±0.83
60.4 ±1.11
USADTM [22]
✓
72.9 ±0.74
73.3 ±0.39
48.7 ±1.00
58.2 ±0.79
68.9 ±1.83
77.1 ±0.74
FixMatch [58]
✓
67.8 ±1.13
77.5 ±1.32
45.2 ±0.55
56.5 ±0.06
56.1 ±2.32
72.4 ±0.71
w/ CB re-weight [14]
✓
✓
72.2 ±1.28
80.9 ±1.52
46.0 ±0.27
58.3 ±0.46
58.9 ±2.79
74.7 ±0.55
w/ LA [43]
✓
✓
75.3 ±2.45
82.0 ±0.36
47.3 ±0.42
58.6 ±0.36
63.4 ±2.99
75.9 ±1.25
w/ BOSS [56]
✓
✓
70.3 ±0.87
76.5 ±0.66
50.0 ±0.39
59.3 ±0.22
66.4 ±2.09
76.0 ±0.85
w/ DARP [33]
✓
✓
74.5 ±0.78
77.8 ±0.63
49.4 ±0.20
58.1 ±0.44
66.9 ±1.66
75.6 ±0.45
w/ CReST [66]
✓
✓
73.4 ±3.10
76.6 ±1.23
44.3 ±0.77
57.1 ±0.58
61.7 ±2.51
71.6 ±1.17
w/ CReST+ [66]
✓
✓
76.3 ±0.86
78.1 ±0.42
44.5 ±0.94
57.1 ±0.65
61.2 ±1.27
71.5 ±0.96
w/ DASO (Ours)
✓
✓
76.0 ±0.37
79.1 ±0.75
49.8 ±0.24
59.2 ±0.35
70.0 ±1.19
78.4 ±0.80
w/ CB re-weight + DASO (Ours)
✓
✓
✓
77.3 ±0.86
81.2 ±0.77
50.3 ±0.18
60.1 ±0.12
70.2 ±1.05
77.8 ±0.58
w/ LA + DASO (Ours)
✓
✓
✓
77.9 ±0.88
82.5 ±0.08
50.7 ±0.51
60.6 ±0.71
71.3 ±1.81
79.0 ±0.58
Table 11. Comparison of accuracy (%) with different methods and their combinations on CIFAR10-LT, CIFAR100-LT, and STL10-LT
under different label sizes with class imbalance. SSL denotes semi-supervised learning. LB and PB correspond to re-balancing for labels
and pseudo-labels, respectively. Our DASO shows consistent performance gain over the baseline FixMatch [58], and adding label re-
balancing to our method shows the best performance among the baselines. CIFAR10/100-LT benchmarks represent the γl = γu setup, and
STL10-LT corresponds to γl ̸= γu setup. We indicate the best results in bold and the second-best results with underlined.
CReST [66]. Self-training is adopted where a SSL algorithm is iteratively re-trained with adding some acceptable pseudo-
labeled samples to the labeled data. The relative ratio of pseudo-labeled samples that will be added to the labeled set in
next generation for each class k is defined as: µk = (NK+1−k/N1)α, where Nk is the label size for class k, suggesting that
minority-class pseudo-labels are more likely to be added. In CReST+, it adds the progressive distribution alignment (PDA) to
the CReST method. To fairly compare with other baselines with 250k of the maximum iterations in total, we divide the whole
iterations to 5 generations, where each generation trains 50k iterations for CIFAR10/100-LT and STL10-LT. For Semi-Aves,
we divide the whole 90 epochs to 3 generations of 30 epochs. For CIFAR10/100-LT and STL10-LT, we set α = 1/3 and
tmin = 0.5, and α = 0.7 and tmin = 0.5 for Semi-Aves respectively similar to [66].
ABC [39]. It trains an auxiliary balanced classifier (ABC) built upon a whole SSL learner (e.g., FixMatch [58]). In particular,
ABC shares the feature extractor with the existing pipeline, and learns the re-weighted versions of both cross-entropy with
labels and consistency regularization from unlabeled data. The re-weight mechanism is performed by the balanced batch
of labeled data and unlabeled data, where the batched images corresponding to each labels and predicted pseudo-labels are
dropped with a probability sampled from Bernoulli distribution. Here, the parameter for Bernoulli is inversely proportional
to the class frequency of the labels and pseudo-labels respectively. The ABC classifier is opted during inference.
D. Additional Experiments
D.1. Comprehensive Comparison with More Baselines
Experiments from the main paper evaluated DASO and other baseline methods specifically designed for re-balancing the
biased pseudo-labels under class-imbalanced labels and distribution mismatch between X and U. In Table 11, we introduce
more diverse baseline methods for comparisons across different benchmarks including both γl = γu and γl ̸= γu cases. As
following, we term SSL methods as SSL, label re-balancing methods as LB, and the re-balancing methods for pseudo-labels
as PB from Table 11. We consider LDAM-DRW [6], classifier re-training (cRT) [27], and class re-weighting with effective
number of samples (CB re-weight) [14] for LB, respectively. For SSL methods, we additionally introduce PseudoLabel [38]
and USADTM [22]. We further consider BOSS [56] as PB. The implementation details on those methods are explained
in Appendix C.3. Note that we extensively compare PB methods based on other than FixMatch in Table 13.
18

--- Page 19 ---
We observe in Table 11 that applying LB improves the performance for Supervised and semi-supervised (SSL, PB) learn-
ing methods in general. This suggests that the bias of pseudo-label can be reduced by LB methods. In particular, the
performance of DASO can be further pushed by additionally applying LB methods, as noted from CB re-weight + DASO and
LA + DASO. This verifies that DASO is complementary to the existing LB methods, where the source for the performance
improvement of DASO itself comes from the ability to truly alleviate the bias of pseudo-labels, not just re-balancing the
labels.
D.2. DASO with Label Re-Balancing when γl ̸= γu
We further evaluate DASO combined with other re-balancing techniques: LA [43] and ABC [39], when the class distribu-
tion of unlabeled data significantly differs from the labeled data (e.g., γl ̸= γu). In this setup, we conduct experiments with
STL10-LT, as shown in Table 12.
STL10-LT (M = 100k)
γl = 10
γl = 20
Algorithm
N1 = 150
N1 = 450
N1 = 150
N1 = 450
FixMatch [58]
56.1 ±2.32
72.4 ±0.71
47.6 ±4.87
64.0 ±2.27
w/ DASO (Ours)
70.0 ±1.19
78.4 ±0.80
65.7 ±1.78
75.3 ±0.44
FixMatch w/ LA [43]
64.4 ±1.35
75.9 ±1.25
51.5 ±3.23
67.4 ±1.04
w/ DASO (Ours)
71.7 ±1.09
79.0 ±0.58
65.6 ±1.43
75.8 ±0.81
FixMatch + ABC [39]
66.3 ±1.00
77.1 ±0.56
59.3 ±2.66
73.0 ±0.91
w/ DASO (Ours)
69.6 ±0.94
77.9 ±0.89
64.5 ±2.81
74.7 ±0.16
Table 12. Comparison of accuracy (%) with the combination of various re-balancing methods on γl ̸= γu setup. DASO somewhat obtains
performance gain when even combined with either LA [43] or ABC [39] on FixMatch. We indicate the best results as bold.
We observe that both LA [43] and ABC [39], are beneficial upon baseline FixMatch. Moreover, the performance can be
further pushed when DASO is applied on top of those methods. However, the performances show marginal improvements
compared to the FixMatch w/ DASO. This opens a new challenge that calls for the design of an unified re-balancing approach
of labels and unlabeled data, which can also well address the potentially unknown unlabeled data.
D.3. Comparison based on ReMixMatch
To verify the efficacy of DASO as a generic framework, we further compare the pseudo-label re-balancing (PB) methods
based on ReMixMatch [4]. In particular, we provide the results as the same way when DASO is integrated with FixMatch [58]
from the main paper. Table 13 shows the results. We compare each method on CIFAR10/100-LT and STL10-LT, varying the
imbalance ratio while the amount of labels used are fixed by N1. Note that for CIFAR benchmarks, γ = γl = γu.
CIFAR10-LT
CIFAR100-LT
STL10-LT
N1 = 500, M1 = 4000.
N1 = 50, M1 = 400.
N1 = 150.
Algorithm
γ = 100
γ = 150
γ = 10
γ = 20
γl = 10
γl = 20
ReMixMatch [4]
70.9 ±2.37
64.7 ±0.95
52.3 ±0.91
46.5 ±0.30
54.4 ±2.15
46.5 ±1.93
w/ DARP [33]
72.2 ±2.72
65.7 ±1.20
52.8 ±0.65
47.0 ±0.17
61.2 ±2.62
59.5 ±2.56
w/ CReST+ [66]
75.6 ±1.60
65.9 ±2.20
49.9 ±0.80
44.5 ±1.04
64.1 ±1.68
49.2 ±0.90
w/ DASO (Ours)
76.8 ±0.81
68.5 ±0.98
53.6 ±0.81
47.8 ±0.69
75.0 ±0.95
68.5 ±5.14
Table 13. Comparison of accuracy (%) with various pseudo-label re-balancing (PB) methods upon different baseline SSL learner, ReMix-
Match [4]. DASO outperforms all the other methods by a significant margin, which is consistent with the results when the baseline SSL
learner was FixMatch from the main paper. We indicate the best results as bold.
As can be seen, DASO achieves the best results among the baselines for comparison. From CIFAR benchmarks (e.g.,
γl = γu), DASO outperforms both DARP [33] and CReST+ [66] that leverages the assumption of γl = γu explicitly; for
example, they utilize the actual class distribution of unlabeled data. As note, while CReST+ is beneficial for ReMixMatch
when trained on CIFAR10-LT, but it performs worse in CIFAR100-LT results. This might come from the limited amount of
labels and the repeated training with re-initializing models via self-training. For STL10-LT cases, the improvements from
19

--- Page 20 ---
both DARP and CReST+ can be limited due to the mismatch of class distributions between the labeled data and unlabeled
data. In contrary, DASO significantly surpasses the other methods without the access to the class distribution of either labels
or unlabeled data. To summarize, DASO can improve typical baseline SSL methods under imbalanced data in general.
D.4. Results on Test-Time Logit Adjustment
In the main paper, we have considered Logit Adjustment (LA) [43] as applying logit-adjusted cross-entropy loss during
training. This point is also explained in Appendix C.3. On the other hand, we also consider adjusting the logits during
inference also present in [43]; we denote this type of LA as LA (inf). In Table 14, we report the results obtained from LA [43]
by this strategy when the class distribution of labeled data and unlabeled data are identical (γ = γl = γu).
CIFAR10-LT
CIFAR100-LT
N1 = 1500, M1 = 3000
N1 = 50, M1 = 300
Algorithm
γ = 100
γ = 150
γ = 10
γ = 20
FixMatch [58]
77.5 ±1.32
72.4 ±1.03
56.5 ±0.06
50.7 ±0.25
FixMatch w/ LA [43]
82.0 ±0.36
78.0 ±0.91
58.6 ±0.36
53.4 ±0.32
FixMatch w/ LA + CReST+ [66]
81.1 ±0.57
77.9 ±0.71
57.1 ±0.55
52.3 ±0.20
FixMatch w/ LA + DASO (Ours)
82.5 ±0.08
79.0 ±2.23
60.6 ±0.71
55.1 ±0.72
FixMatch w/ LA (inf) [43]
82.8 ±1.43
79.2 ±1.15
58.7 ±0.63
53.3 ±0.43
FixMatch w/ LA (inf) + CReST+ [66]
82.9 ±0.24
80.3 ±0.56
57.8 ±0.47
53.3 ±0.83
FixMatch w/ LA (inf) + DASO (Ours)
84.5 ±0.55
81.8 ±0.83
60.5 ±0.49
55.2 ±0.47
Table 14. Comparison of accuracy (%) with different strategies of applying Logit Adjustment (LA) [43]: either train-time (noted as LA)
or during inference (noted as LA (inf)). We observe large gains compared to baseline FixMatch when LA is applied during inference.
D.5. More Ablation Study
We conduct several ablation studies on the hyper-parameters in DASO framework. As the same with the ablation study
conducted from the main paper, we consider FixMatch [58] with DASO on CIFAR10-LT with N1 = 500, γ = 100 (denoted
as C10) and STL10-LT with N1 = 150, γl = 10 (denoted as STL10) respectively. Table 15 compares different values of the
queue size L for constructing the balanced prototypes. Table 16 tests different temperature factor Tproto for the similarity-
based classifier. Finally, Table 17 shows the effect of different loss weights λalign for the semantic alignment loss. We shaded
rows that correspond to the hyper-parameter of the complete DASO framework. We also indicate the best results in bold.
C10
STL10
FixMatch
68.25
55.53
L = 128
73.77
69.17
L = 256
75.97
70.21
L = 512
75.03
69.96
L = 1024
74.36
69.64
L = 2048
73.50
69.99
Table 15. Ablation study on L, the bal-
anced queue size.
C10
STL10
FixMatch
68.25
55.53
Tproto = 0.02
73.84
68.19
Tproto = 0.05
75.97
70.21
Tproto = 0.2
70.53
66.62
Tproto = 0.5
52.36
60.92
Tproto = 1.0
46.47
57.40
Table 16. Ablation study on Tproto for
semantic pseudo-label.
C10
STL10
FixMatch
68.25
55.53
λalign = 0
70.98
61.64
λalign = 0.5
73.78
69.01
λalign = 1
75.97
70.21
λalign = 1.5
74.59
71.51
λalign = 2
74.57
71.12
Table 17.
Ablation study on λalign,
which is a weight for Lalign.
As note, we do not tune the hyper-parameters above (L, Tproto, λalign) depending on different benchmarks across different
imbalance ratio. For example, in STL10-LT case, using λalign value higher than 1 seems effective, but the result of 70.21%
obtained from λalign = 1 already performs well.
E. Detailed Analysis
E.1. Recall and Precision Analysis
E.1.1
Detailed comparison for linear pseudo-label and semantic pseudo-label methods
We first take a closer look at the bias of pseudo-labels of each method by analyzing per-class recall and precision. We then
compare the class-wise test accuracy of each model to evaluate the capability for each class, as done in the main paper. Fig. 5
20

--- Page 21 ---
provides the comparison of FixMatch w/ DASO (ours) and USADTM [22] over FixMatch [58] trained on CIFAR10-LT.
C0
C3
C6
C9
Class index
0.2
0.4
0.6
0.8
1.0
Recall
FixMatch (avg: 0.68)
USADTM (avg: 0.74)
DASO (avg: 0.79)
(a) Recall of pseudo-labels
C0
C3
C6
C9
Class index
0.0
0.2
0.4
0.6
0.8
1.0
Precision
FixMatch (avg: 0.84)
USADTM (avg: 0.57)
DASO (avg: 0.76)
(b) Precision of pseudo-labels
C0
C3
C6
C9
Class index
20
40
60
80
100
Test Top1 Accuracy (%)
FixMatch (avg: 68.6%)
USADTM (avg: 72.3%)
DASO (avg: 76.3%)
(c) Class-wise test accuracy
Figure 5. Analysis of bias in pseudo-labels and test accuracy. We consider FixMatch [58] for linear pseudo-labels, USADTM [22] for
semantic pseudo-labels, and the proposed FixMatch w/ DASO trained on CIFAR10-LT with N1 = 500 with γl = γu = 100.
Compared to the linear pseudo-labels, the recall of semantic pseudo-labels on minority classes significantly increased
in Fig. 5a. However, their precision values are degraded on the minorities, which means that the semantic pseudo-labels have
the bias towards the minorities, leading to performance drop on the majority classes.
In contrary, the pseudo-labels generated from our DASO maintain high precision while the recall on the minority classes
increased, encouraging high performance on both of majority and minority classes. From the analyses, pseudo-labels from
DASO find the trade-off between linear and semantic pseudo-labels with respect to the bias that performs well on test data.
Since DASO also aims to keep the prediction of majority classes, the test accuracy drop on the head classes is well addressed.
Note that Fig. 6 shows the same analysis on the models trained on CIFAR100-LT.
C0
C20
C40
C60
C80
C99
Class index
0.0
0.2
0.4
0.6
0.8
Recall
FixMatch (avg: 0.4)
USADTM (avg: 0.44)
DASO (avg: 0.47)
(a) Recall of pseudo-labels
C0
C20
C40
C60
C80
C99
Class index
0.0
0.2
0.4
0.6
0.8
Precision
FixMatch (avg: 0.43)
USADTM (avg: 0.42)
DASO (avg: 0.48)
(b) Precision of pseudo-labels
C0
C20
C40
C60
C80
C99
Class index
20
40
60
80
Test Top1 Accuracy (%)
FixMatch (avg: 44.6%)
USADTM (avg: 48.4%)
DASO (avg: 49.3%)
(c) Class-wise test accuracy
Figure 6. Analysis of bias in pseudo-labels. We consider FixMatch [58] for linear pseudo-labels, USADTM [22] for semantic pseudo-
labels, and the proposed FixMatch w/ DASO trained on CIFAR100-LT with N1 = 50 with γl = γu = 10.
E.1.2
DASO with class distribution mismatch on traditional SSL learner
We present the analyses of bias in pseudo-labels for the other classic SSL algorithms: MeanTeacher [61] and MixMatch [5]
in Figs. 7 and 8, respectively, in case of uniform distribution of unlabeled data; i.e., γu = 1. In such a case, class distribution
mismatch (i.e., γl ̸= γu) can damage the accuracy of the model.
From the recall curves in Figs. 7a and 8a and the precision curves in Figs. 7b and 8b, the pseudo-labels of the baseline SSL
learners are severely biased towards the head classes, since most of the minority class examples are collapsed to the majority
class ones. The unlabeled data with γu = 1 rather significantly accelerated the bias, to the point where the precision curve is
completely reversed; precision values in the majority classes significantly degraded, compared to the recall curve. Thereby,
the model rarely predicts some of the minority class examples for the test dataset in Figs. 7c and 8c.
In contrast, we demonstrate that DASO can even completely mitigate such a devastating bias, by just coupling the lin-
ear pseudo-labels with the semantic pseudo-labels obtained from the similarity-based classifier. In this case, the semantic
alignment loss Lalign is not applied, due to the absence of advanced augmentation As for MeanTeacher and MixMatch.
Surprisingly, in MeanTeacher (MT) with DASO, the recall and precision values become uniform, resulting in a uniform per-
class test accuracy in Fig. 7c. When combined with MixMatch [5], DASO also recovers the minority-class pseudo-labels
significantly. In final, the averaged test accuracy can be more than doubled (i.e., 37.3% →77.2%), as shown in Fig. 8c.
21

--- Page 22 ---
C0
C3
C6
C9
Class index
0.0
0.2
0.4
0.6
0.8
1.0
Recall
MT (avg: 0.45)
 w/ DASO (avg: 0.86)
(a) Recall of pseudo-labels
C0
C3
C6
C9
Class index
0.4
0.6
0.8
1.0
Precision
MT (avg: 0.71)
 w/ DASO (avg: 0.87)
(b) Precision of pseudo-labels
C0
C3
C6
C9
Class index
0
20
40
60
80
100
Test Top1 Accuracy (%)
MT (avg: 45.4%)
 w/ DASO (avg: 86.5%)
(c) Class-wise test accuracy
Figure 7. Analysis of bias in pseudo-labels and test accuracy. We consider MeanTeacher (MT) [61], and the proposed DASO applied to
MT (MT w/ DASO) trained on CIFAR10-LT with N1 = 1500 with γl = 100 and γu = 1.
C0
C3
C6
C9
Class index
0.0
0.2
0.4
0.6
0.8
1.0
Recall
MM (avg: 0.35)
MM w/ DASO (avg: 0.64)
(a) Recall of pseudo-labels
C0
C3
C6
C9
Class index
0.0
0.2
0.4
0.6
0.8
1.0
Precision
MM (avg: 0.68)
MM w/ DASO (avg: 0.75)
(b) Precision of pseudo-labels
C0
C3
C6
C9
Class index
0
20
40
60
80
100
Test Top1 Accuracy (%)
MM (avg: 37.3%)
MM w/ DASO (avg: 77.2%)
(c) Class-wise test accuracy
Figure 8. Analysis of bias in pseudo-labels and test accuracy. We consider MixMatch (MM) [61], and the proposed DASO applied to MM
(MM w/ DASO) trained on CIFAR10-LT with N1 = 1500 with γl = 100 and γu = 1.
As such, DASO helps alleviate the bias in pseudo-labels, even when the class distributions between labeled and unlabeled
data substantially differ, without accessing the knowledge about the underlying distribution of unlabeled data.
C0
C1
C2
C3
C4
C5
C6
C7
C8
C9
Predicted label
C0
C1
C2
C3
C4
C5
C6
C7
C8
C9
True label
0.98 0.0 0.01 0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
1.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.06 0.0 0.88 0.02 0.02 0.01 0.01 0.0
0.0
0.0
0.04 0.0 0.04 0.82 0.02 0.05 0.01 0.0
0.0
0.0
0.02 0.0 0.04 0.03 0.87 0.0 0.01 0.02 0.0
0.0
0.02 0.0 0.06 0.27 0.03 0.58 0.01 0.02 0.0
0.0
0.03 0.0
0.1 0.12 0.01 0.0 0.74 0.0
0.0
0.0
0.07 0.0 0.06 0.13 0.09 0.02 0.0 0.62 0.0
0.0
0.56 0.13 0.02 0.02 0.0
0.0
0.0
0.0 0.27 0.0
0.17 0.7
0.0 0.01 0.0
0.0
0.0
0.0
0.0
0.1
FixMatch (Avg Acc.: 68.6%)
C0
C1
C2
C3
C4
C5
C6
C7
C8
C9
Predicted label
0.98 0.0 0.01 0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
1.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.0
0.06 0.0 0.87 0.02 0.02 0.01 0.0
0.0 0.01 0.0
0.03 0.0 0.04 0.83 0.03 0.06 0.0
0.0
0.0
0.0
0.01 0.0 0.03 0.03 0.88 0.01 0.01 0.03 0.0
0.0
0.02 0.0 0.04 0.24 0.03 0.64 0.0 0.03 0.0
0.0
0.03 0.01 0.09 0.11 0.03 0.0 0.73 0.0
0.0
0.0
0.06 0.0 0.05 0.11 0.06 0.05 0.0 0.65 0.0
0.0
0.33 0.13 0.01 0.01 0.0
0.0
0.0
0.0 0.52 0.0
0.08 0.36 0.0
0.0
0.0
0.0
0.0
0.0
0.0 0.55
w/ DASO (Avg Acc.: 76.5%)
C0
C1
C2
C3
C4
C5
C6
C7
C8
C9
Predicted label
0.96 0.0 0.01 0.01 0.0
0.0
0.0
0.0 0.02 0.0
0.0 0.99 0.0
0.0
0.0
0.0
0.0
0.0
0.0 0.01
0.05 0.0 0.86 0.02 0.02 0.01 0.02 0.01 0.0
0.0
0.03 0.01 0.04 0.78 0.02 0.07 0.03 0.02 0.0
0.0
0.01 0.0 0.03 0.03 0.84 0.0 0.02 0.07 0.0
0.0
0.02 0.0 0.04 0.21 0.03 0.64 0.01 0.04 0.0
0.0
0.02 0.0 0.06 0.06 0.01 0.01 0.83 0.0 0.01 0.0
0.04 0.0 0.03 0.09 0.05 0.04 0.01 0.72 0.01 0.01
0.14 0.08 0.01 0.0
0.0
0.0
0.0
0.0 0.74 0.02
0.05 0.21 0.0
0.0
0.0
0.0
0.0
0.0 0.01 0.72
w/ ABC + DASO (Avg Acc.: 80.8%)
0.0
0.2
0.4
0.6
0.8
1.0
Figure 9. Analysis of predictions from test data via confusion matrix. All the methods are trained on CIFAR10-LT with γ = 100 and
N1 = 500 upon the same fixed random seed. DASO greatly recovers the predictions on the actual minority class examples in test data.
E.2. Confusion Matrix on Test Data
We compare the confusion matrices of the predictions from the test data. From the baseline FixMatch [58], we further
apply our DASO on both FixMatch and FixMatch w/ ABC [39]. As shown in Fig. 9, the predictions on the tail classes (e.g.,
C8 and C9) in FixMatch are severely biased towards the majority classes (e.g., C1). This limits the overall performance,
which is carried by the non-minority classes (68.6%). On the other hand, from the center of Fig. 9, DASO significantly
alleviates the bias towards the head classes observing C8 and C9 classes, while the performances on the other classes are well
maintained. When DASO is integrated with ABC [39] in the right figure, the accuracy values are further improved.
22

--- Page 23 ---
E.3. Train Curves for Recall and Accuracy
We compare the train curves of recall and test accuracy values from FixMatch [58] and FixMatch w/ DASO (Ours) trained
on CIFAR10/100-LT respectively in Figs. 10a and 10b. Here, we plot those from majority classes (e.g., first 20% classes) and
minority classes (e.g., last 20% classes), in addition to the overall values. From both CIFAR10/100-LT benchmarks, DASO
significantly improves the recall and test accuracy values on the minority classes, while relatively maintaining those from the
majority classes. This verifies the efficacy of DASO that specifically handles the biased minority classes in unlabeled data.
0k
50k
100k
150k
200k
250k
Training Steps
0.2
0.4
0.6
0.8
1.0
Recall of Pseudo-Label
0k
50k
100k
150k
200k
250k
Training Steps
20
40
60
80
100
Test Accuracy (%)
FixMatch (minority)
w/ DASO (minority)
FixMatch (majority)
w/ DASO (majority)
FixMatch (overall)
w/ DASO (overall)
(a) Train curves on CIFAR10-LT
0k
50k
100k
150k
200k
250k
Training Steps
0.1
0.2
0.3
0.4
0.5
0.6
0.7
Recall of Pseudo-Label
0k
50k
100k
150k
200k
250k
Training Steps
10
20
30
40
50
60
70
Test Accuracy (%)
FixMatch (minority)
w/ DASO (minority)
FixMatch (majority)
w/ DASO (majority)
FixMatch (overall)
w/ DASO (overall)
(b) Train curves on CIFAR100-LT
Figure 10. Train curves for the recall and test accuracy values obtained from FixMatch and FixMatch w/ DASO (Ours). The training details
are consistent from the main paper. DASO well reduces the biases on the tail classes, while preserving those from the head classes.
E.4. Further Comparison of Feature Representations
To verify the efficacy of the proposed semantic alignment loss (Lalign), we further visualize the t-SNE [62] of the feature
encoder outputs from FixMatch w/ Lalign in the center of Fig. 11. Compared to FixMatch, applying Lalign without the
class-adaptive pseudo-label blending can already cluster the minority classes (e.g., C6, C8, and C9) in the center of the
figure. However, those indicated clusters lie nearby the head-class clusters (e.g., C0 and C1), where the classifier can still be
confused. In that sense, the complete DASO from the right figure further improves the separability of the tail classes from
the head classes. This demonstrates that while applying the semantic alignment loss Lalign could be helpful for the minority
classes, both class-adaptive pseudo-label blending and Lalign are the essential components for our DASO framework.
FixMatch
C6
FixMatch w/ 
align
C8
C9
C6
FixMatch w/ DASO (Ours)
C8, C9
C6
C0
C1
C2
C3
C4
C5
C6
C7
C8
C9
C0
C1
C2
C3
C4
C5
C6
C7
C8
C9
C0
C1
C2
C3
C4
C5
C6
C7
C8
C9
Figure 11. Comparison of t-SNE [62] visualizations of feature representations. We additionally compare the model trained with FixMatch
w/ Lalign between the original FixMatch [58] and FixMatch w/ DASO (Ours). Note that both of the semantic alignment loss Lalign and our
class-adaptive pseudo-label blending contribute to alleviating the bias in pseudo-labels in perspective of feature representation.
E.5. Confidence Analysis from Out-of-class Examples
To investigate the efficacy of DASO pseudo-label, we analyze the confidence of predictions of unlabeled data after training
model with U = Uin + Uout under Semi-Aves benchmark [60]. Fig. 12 visualizes the histograms of entropy values obtained
from either FixMatch [58] or FixMatch w/ DASO, respectively. Note that since both models do not explicitly learn how to
distinguish in-class and out-of-class categories at all, those samples cannot be completely separated in confidence plot.
23

--- Page 24 ---
0
1
2
3
4
5
Entropy
0
1000
2000
3000
4000
5000
6000
7000
8000
Num. samples
Uout
Uin
(a) FixMatch [58]
0
1
2
3
4
5
Entropy
0
500
1000
1500
2000
2500
3000
3500
4000
4500
Num. samples
Uout
Uin
(b) FixMatch w/ DASO (Ours)
Figure 12. Comparisons of DASO and FixMatch [58] on the distribution of entropy values from the predictions of samples in Uin and Uout
of Semi-Aves benchmark [60], respectively. We observe that examples Uin relatively remain in low-entropy (e.g., high-confidence) area,
while those in Uout are well pushed towards the high-entropy (e.g., low-confidence) area from DASO (ours).
FixMatch w/ DASO, which learned the blending of linear and semantic pseudo-labels can be effective in that the out-
of-class examples in Uout are further pushed towards the low-confidence region (i.e., higher entropy) compared to the in-
class unlabeled examples in Uin. For example, about 8k out-of-class examples correspond to the most confident samples
in Fig. 12a, while they reduced to 4k with DASO in Fig. 12b. We suppose DASO has the implicit ability to push more
examples corresponding to out-of-class that can cause degradation, towards the low-confident area. This point implies the
potential application of DASO towards an open-set SSL scenario, where SSL algorithms also observe unlabeled data in a
broader class distribution compared to the labels, and learning without harmful out-of-class examples would be important.
F. Overall Framework
Feature
Encoder
Linear
Classifier
Cross
Entropy
Feature
Encoder
Feature
Encoder
Linear
Classifier
Similarity
Classifier
Similarity
Classifier
Linear
Classifier
𝑧𝑧
𝑧𝑧(𝑤𝑤)
𝑧𝑧(𝑠𝑠)
Cross
Entropy
DASO
PL Blend
Unsup.
Loss
Labeled data
𝑥𝑥
Unlabeled data
𝑢𝑢
Weakly-
augmented
𝒜𝒜𝑤𝑤𝑥𝑥
Strongly-
augmented
𝒜𝒜𝑠𝑠𝑢𝑢
Weakly-
augmented
𝒜𝒜𝑤𝑤𝑢𝑢
Prediction 𝑝𝑝(𝑠𝑠)
Prediction 𝑞𝑞(𝑠𝑠)
Semantic PL ො𝑞𝑞
Linear PL ̂𝑝𝑝
Prediction
Semantic
Alignment Loss
ℒalign
Supervised
Loss ℒcls
Label 𝑦𝑦
DASO PL ̂𝑝𝑝′
Unsupervised Loss 
ℒ𝑢𝑢
𝑓𝑓𝜃𝜃
enc
𝑓𝑓𝜃𝜃
enc
𝑓𝑓𝜃𝜃
enc
𝑓𝑓𝜙𝜙
cls
𝑓𝑓𝜙𝜙
cls
𝑓𝑓𝜙𝜙
cls
sg
sg
Figure 13. Overall framework of DASO including the blending of pseudo-labels (DASO PL Blend) and the semantic alignment loss
(Lalign). As explained in Sec. 3.3 of the main paper, ‘balanced prototypes’ for executing the similarity-based classifier are generated from
EMA features of labeled data, which is omitted in this figure. Two main components of DASO framework (blending of pseudo-labels and
semantic alignment loss) can easily integrate with typical semi-supervised learning algorithms such as FixMatch [58] and ReMixMatch [4]
for debiasing pseudo-labels. Note that ‘sg’ means stop-gradient operation.
24
