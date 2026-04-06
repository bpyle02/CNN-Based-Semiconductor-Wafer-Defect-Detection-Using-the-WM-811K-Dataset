# Circle Loss: A Unified Perspective of Pair Similarity Optimization

**Authors**: Sun, Cheng, Zhang, Lin, Liu, Wang
**Year**: 2020
**arXiv**: 2002.10857
**Topic**: metric_learning
**Relevance**: Unified loss for improved convergence on imbalanced data

---


--- Page 1 ---
Circle Loss: A Uniﬁed Perspective of Pair Similarity Optimization
Yifan Sun1∗, Changmao Cheng1∗, Yuhan Zhang2∗, Chi Zhang1, Liang Zheng3, Zhongdao Wang4, Yichen Wei1†
1MEGVII Technology 2Beihang University 3Australian National University 4Tsinghua University
{peter, chengchangmao, zhangchi, weiyichen}@megvii.com
Abstract
This paper provides a pair similarity optimization view-
point on deep feature learning, aiming to maximize the
within-class similarity sp and minimize the between-class
similarity sn.
We ﬁnd a majority of loss functions, in-
cluding the triplet loss and the softmax cross-entropy loss,
embed sn and sp into similarity pairs and seek to reduce
(sn −sp). Such an optimization manner is inﬂexible, be-
cause the penalty strength on every single similarity score
is restricted to be equal. Our intuition is that if a similarity
score deviates far from the optimum, it should be empha-
sized. To this end, we simply re-weight each similarity to
highlight the less-optimized similarity scores. It results in
a Circle loss, which is named due to its circular decision
boundary. The Circle loss has a uniﬁed formula for two
elemental deep feature learning paradigms, i.e., learning
with class-level labels and pair-wise labels. Analytically,
we show that the Circle loss offers a more ﬂexible optimiza-
tion approach towards a more deﬁnite convergence target,
compared with the loss functions optimizing (sn −sp). Ex-
perimentally, we demonstrate the superiority of the Circle
loss on a variety of deep feature learning tasks. On face
recognition, person re-identiﬁcation, as well as several ﬁne-
grained image retrieval datasets, the achieved performance
is on par with the state of the art.
1. Introduction
This paper holds a similarity optimization view towards
two elemental deep feature learning paradigms, i.e., learn-
ing from data with class-level labels and from data with
pair-wise labels. The former employs a classiﬁcation loss
function (e.g., softmax cross-entropy loss [25, 16, 36]) to
optimize the similarity between samples and weight vec-
tors. The latter leverages a metric loss function (e.g., triplet
loss [9, 22]) to optimize the similarity between samples. In
our interpretation, there is no intrinsic difference between
these two learning approaches. They both seek to minimize
∗Equal contribution.
†Corresponding author.
𝑠!
A
C
𝑠"
B
T
T’
𝑠!
A
C
𝑠"
B
T
0
1
1
0
1
1
T’
(a)
(b)
Figure 1: Comparison between the popular optimization
manner of reducing (sn−sp) and the proposed optimization
manner of reducing (αnsn −αpsp). (a) Reducing (sn −sp)
is prone to inﬂexible optimization (A, B and C all have
equal gradients with respect to sn and sp), as well as am-
biguous convergence status (both T and T ′ on the decision
boundary are acceptable). (b) With (αnsn −αpsp), the Cir-
cle loss dynamically adjusts its gradients on sp and sn, and
thus beneﬁts from a ﬂexible optimization process. For A, it
emphasizes on increasing sp; for B, it emphasizes on reduc-
ing sn. Moreover, it favors a speciﬁed point T on the circu-
lar decision boundary for convergence, setting up a deﬁnite
convergence target.
between-class similarity sn, as well as to maximize within-
class similarity sp.
From this viewpoint, we ﬁnd that many popular loss
functions (e.g., triplet loss [9, 22], softmax cross-entropy
loss and its variants [25, 16, 36, 29, 32, 2]) share a similar
optimization pattern. They all embed sn and sp into sim-
ilarity pairs and seek to reduce (sn −sp). In (sn −sp),
increasing sp is equivalent to reducing sn. We argue that
this symmetric optimization manner is prone to the follow-
ing two problems.
• Lack of ﬂexibility for optimization.
The penalty
strength on sn and sp is restricted to be equal. Given the
speciﬁed loss functions, the gradients with respect to sn
and sp are of same amplitudes (as detailed in Section 2).
In some corner cases, e.g., sp is small and sn already ap-
proaches 0 (“A” in Fig. 1 (a)), it keeps on penalizing sn
with a large gradient. It is inefﬁcient and irrational.
arXiv:2002.10857v2  [cs.CV]  15 Jun 2020

--- Page 2 ---
• Ambiguous convergence status. Optimizing (sn−sp)
usually leads to a decision boundary of sp −sn = m (m
is the margin). This decision boundary allows ambiguity
(e.g., “T” and “T ′” in Fig. 1 (a)) for convergence. For ex-
ample, T has {sn, sp} = {0.2, 0.5} and T ′ has {s′
n, s′
p} =
{0.4, 0.7}. They both obtain the margin m = 0.3. However,
comparing them against each other, we ﬁnd the gap between
s′
n and sp is only 0.1. Consequently, the ambiguous conver-
gence compromises the separability of the feature space.
With these insights, we reach an intuition that different
similarity scores should have different penalty strength. If
a similarity score deviates far from the optimum, it should
receive a strong penalty. Otherwise, if a similarity score
already approaches the optimum, it should be optimized
mildly.
To this end, we ﬁrst generalize (sn −sp) into
(αnsn −αpsp), where αn and αp are independent weight-
ing factors, allowing sn and sp to learn at different paces.
We then implement αn and αp as linear functions w.r.t. sn
and sp respectively, to make the learning pace adaptive to
the optimization status: The farther a similarity score de-
viates from the optimum, the larger the weighting factor
will be. Such optimization results in the decision boundary
αnsn −αpsp = m, yielding a circle shape in the (sn, sp)
space, so we name the proposed loss function Circle loss.
Being simple, Circle loss intrinsically reshapes the char-
acteristics of the deep feature learning from the following
three aspects:
First, a uniﬁed loss function. From the uniﬁed simi-
larity pair optimization perspective, we propose a uniﬁed
loss function for two elemental learning paradigms, learn-
ing with class-level labels and with pair-wise labels.
Second, ﬂexible optimization.
During training, the
gradient back-propagated to sn (sp) will be ampliﬁed by
αn (αp). Those less-optimized similarity scores will have
larger weighting factors and consequentially get larger gra-
dients. As shown in Fig. 1 (b), the optimization on A, B
and C are different to each other.
Third, deﬁnite convergence status. On the circular de-
cision boundary, Circle loss favors a speciﬁed convergence
status (“T” in Fig. 1 (b)), as to be demonstrated in Sec-
tion 3.3. Correspondingly, it sets up a deﬁnite optimization
target and beneﬁts the separability.
The main contributions of this paper are summarized as
follows:
• We propose Circle loss, a simple loss function for deep
feature learning. By re-weighting each similarity score
under supervision, Circle loss beneﬁts the deep feature
learning with ﬂexible optimization and deﬁnite conver-
gence target.
• We present Circle loss with compatibility to both class-
level labels and pair-wise labels. Circle loss degener-
ates to triplet loss or softmax cross-entropy loss with
slight modiﬁcations.
• We conduct extensive experiments on a variety of deep
feature learning tasks, e.g. face recognition, person re-
identiﬁcation, car image retrieval and so on. On all
these tasks, we demonstrate the superiority of Circle
loss with performance on par with the state of the art.
2. A Uniﬁed Perspective
Deep feature learning aims to maximize the within-class
similarity sp, as well as to minimize the between-class sim-
ilarity sn. Under the cosine similarity metric, for example,
we expect sp →1 and sn →0.
To this end, learning with class-level labels and learn-
ing with pair-wise labels are two elemental paradigms.
They are conventionally considered separately and signif-
icantly differ from each other w.r.t to the loss functions.
Given class-level labels, the ﬁrst one basically learns to
classify each training sample to its target class with a clas-
siﬁcation loss, e.g. L2-Softmax [21], Large-margin Soft-
max [15], Angular Softmax [16], NormFace [30], AM-
Softmax [29], CosFace [32], ArcFace [2]. These methods
are also known as proxy-based learning, as they optimize
the similarity between samples and a set of proxies rep-
resenting each class. In contrast, given pair-wise labels,
the second one directly learns pair-wise similarity (i.e., the
similarity between samples) in the feature space and thus
requires no proxies, e.g., constrastive loss [5, 1], triplet
loss [9, 22], Lifted-Structure loss [19], N-pair loss [24], His-
togram loss [27], Angular loss [33], Margin based loss [38],
Multi-Similarity loss [34] and so on.
This paper views both learning approaches from a uni-
ﬁed perspective, with no preference for either proxy-based
or pair-wise similarity. Given a single sample x in the fea-
ture space, let us assume that there are K within-class sim-
ilarity scores and L between-class similarity scores associ-
ated with x. We denote these similarity scores as {si
p} (i =
1, 2, · · · , K) and {sj
n} (j = 1, 2, · · · , L), respectively.
To minimize each sj
n as well as to maximize si
p, (∀i ∈
{1, 2, · · · , K}, ∀j ∈{1, 2, · · · , L}), we propose a uniﬁed
loss function by:
Luni = log
h
1 +
K
X
i=1
L
X
j=1
exp(γ(sj
n −si
p + m))
i
= log
h
1 +
L
X
j=1
exp(γ(sj
n + m))
K
X
i=1
exp(γ(−si
p))
i
,
(1)
in which γ is a scale factor and m is a margin for better
similarity separation.
Eq. 1 is intuitive. It iterates through every similarity pair
to reduce (sj
n −si
p). We note that it degenerates to triplet
loss or classiﬁcation loss, through slight modiﬁcations.
Given class-level labels, we calculate the similarity
scores between x and weight vectors wi (i = 1, 2, · · · , N)

--- Page 3 ---
𝑑𝐿
𝑑𝑠$
𝑑𝐿
𝑑𝑠%
𝑠$
𝑠%
𝑠$
𝑠%
𝑠$
𝑠%
𝑠$
𝑠%
𝑠$
𝑠%
𝑠$
𝑠%
𝑑𝐿
𝑑𝑠$
𝑑𝐿
𝑑𝑠%
𝑑𝐿
𝑑𝑠$
𝑑𝐿
𝑑𝑠%
(a) Triplet loss
(b) AMSoftmax loss
(c) Circle loss
A
A
B
B
B
A
B
A
B
A
B
A
Figure 2: The gradients of the loss functions. (a) Triplet loss. (b) AM-Softmax loss. (c) The proposed Circle loss. Both
triplet loss and AM-Softmax loss present the lack of ﬂexibility for optimization. The gradients with respect to sp (left) and sn
(right) are restricted to equal and undergo a sudden decrease upon convergence (the similarity pair B). For example, at A, the
within-class similarity score sp already approaches 1, and still incurs a large gradient. Moreover, the decision boundaries are
parallel to sp = sn, which allows ambiguous convergence. In contrast, the proposed Circle loss assigns different gradients
to the similarity scores, depending on their distances to the optimum. For A (both sn and sp are large), Circle loss lays
emphasis on optimizing sn. For B, since sn signiﬁcantly decreases, Circle loss reduces its gradient and thus enforces a
moderated penalty. Circle loss has a circular decision boundary, and promotes accurate convergence status.
(N is the number of training classes) in the classiﬁcation
layer. Speciﬁcally, we get (N −1) between-class simi-
larity scores by: sj
n = w⊺
j x/(∥wj∥∥x∥) (wj is the j-th
non-target weight vector).
Additionally, we get a single
within-class similarity score (with the superscript omitted)
sp = w⊺
yx/(∥wy∥∥x∥). With these prerequisite, Eq. 1 de-
generates to AM-Softmax [29, 32], an important variant of
Softmax loss (i.e., softmax cross-entropy loss):
Lam = log
h
1 +
N−1
X
j=1
exp(γ(sj
n + m)) exp(−γsp)
i
= −log
exp(γ(sp −m))
exp(γ(sp −m)) + PN−1
j=1 exp(γsj
n)
.
(2)
Moreover, with m = 0, Eq. 2 further degenerates to
Normface [30]. By replacing the cosine similarity with the
inner product and setting γ = 1, it ﬁnally degenerates to
Softmax loss.
Given pair-wise labels, we calculate the similarity
scores between x and the other features in the mini-
batch. Speciﬁcally, sj
n = (xj
n)⊺x/(∥xj
n∥∥x∥) (xj
n is the
j-th sample in the negative sample set N) and si
p
=
(xi
p)⊺x/(∥xi
p∥∥x∥) (xi
p is the i-th sample in the positive
sample set P). Correspondingly, K = |P|, L = |N|. Eq. 1
degenerates to triplet loss with hard mining [22, 8]:
Ltri =
lim
γ→+∞
1
γ Luni
=
lim
γ→+∞
1
γ log
h
1 +
K
X
i=1
L
X
j=1
exp(γ(sj
n −si
p + m))
i
= max

sj
n −si
p + m

+.
(3)
Speciﬁcally, we note that in Eq. 3, the “P exp(·)” op-
eration is utilized by Lifted-Structure loss [19], N-pair
loss [24], Multi-Similarity loss [34] and etc., to conduct
“soft” hard mining among samples. Enlarging γ gradually
reinforces the mining intensity and when γ →+∞, it re-
sults in the canonical hard mining in [22, 8].
Gradient analysis. Eq. 2 and Eq. 3 show triplet loss,
Softmax loss and its several variants can be interpreted as
speciﬁc cases of Eq. 1. In another word, they all optimize
(sn −sp). Under the toy scenario where there are only a
single sp and sn, we visualize the gradients of triplet loss
and AM-Softmax loss in Fig. 2 (a) and (b), from which we
draw the following observations:
• First, before the loss reaches its decision boundary
(upon which the gradients vanish), the gradients with
respect to both sp and sn are the same to each other.
The status A has {sn, sp} = {0.8, 0.8}, indicating
good within-class compactness. However, A still re-
ceives a large gradient with respect to sp. It leads to a
lack of ﬂexibility during optimization.
• Second, the gradients stay (roughly) constant before
convergence and undergo a sudden decrease upon con-
vergence.
The status B lies closer to the decision
boundary and is better optimized, compared with A.
However, the loss functions (both triplet loss and AM-
Softmax loss) enforce an approximately equal penalty
on A and B. It is another evidence of inﬂexibility.
• Third, the decision boundaries (the white dashed lines)
are parallel to sn −sp = m. Any two points (e.g., T
and T ′ in Fig. 1) on this boundary have an equal sim-
ilarity gap of m, and are thus of equal difﬁculties to
achieve. In another word, loss functions minimizing
(sn −sp +m) lay no preference on T or T ′ for conver-
gence, and are prone to ambiguous convergence. Ex-

--- Page 4 ---
perimental evidence of this problem is to be accessed
in Section 4.6.
These problems originate from the optimization manner
of minimizing (sn −sp), in which reducing sn is equivalent
to increasing sp. In the following Section 3, we will trans-
fer such an optimization manner into a more general one to
facilitate higher ﬂexibility.
3. A New Loss Function
3.1. Self-paced Weighting
We consider to enhance the optimization ﬂexibility by
allowing each similarity score to learn at its own pace, de-
pending on its current optimization status. We ﬁrst neglect
the margin item m in Eq. 1 and transfer the uniﬁed loss
function into the proposed Circle loss by:
Lcircle = log
h
1 +
K
X
i=1
L
X
j=1
exp
 γ(αj
nsj
n −αi
psi
p)
i
= log
h
1 +
L
X
j=1
exp(γαj
nsj
n)
K
X
i=1
exp(−γαi
psi
p),
i
(4)
in which αj
n and αi
p are non-negative weighting factors.
Eq. 4 is derived from Eq. 1 by generalizing (sj
n−si
p) into
(αj
nsj
n−αi
psi
p). During training, the gradient with respect to
(αj
nsj
n −αi
psi
p) is to be multiplied with αj
n (αi
p) when back-
propagated to sj
n (si
p). When a similarity score deviates far
from its optimum (i.e., On for sj
n and Op for si
p), it should
get a large weighting factor so as to get effective update
with large gradient. To this end, we deﬁne αi
p and αj
n in a
self-paced manner:
(
αi
p = [Op −si
p]+,
αj
n = [sj
n −On]+,
(5)
in which [·]+ is the “cut-off at zero” operation to ensure αi
p
and αj
n are non-negative.
Discussions. Re-scaling the cosine similarity under su-
pervision is a common practice in modern classiﬁcation
losses [21, 30, 29, 32, 39, 40]. Conventionally, all the sim-
ilarity scores share an equal scale factor γ. The equal re-
scaling is natural when we consider the softmax value in a
classiﬁcation loss function as the probability of a sample be-
longing to a certain class. In contrast, Circle loss multiplies
each similarity score with an independent weighting factor
before re-scaling. It thus gets rid of the constraint of equal
re-scaling and allows more ﬂexible optimization. Besides
the beneﬁts of better optimization, another signiﬁcance of
such a re-weighting (or re-scaling) strategy is involved with
the underlying interpretation. Circle loss abandons the in-
terpretation of classifying a sample to its target class with
a large probability. Instead, it holds a similarity pair opti-
mization perspective, which is compatible with two learning
paradigms.
3.2. Within-class and Between-class Margins
In loss functions optimizing (sn −sp), adding a margin
m reinforces the optimization [15, 16, 29, 32]. Since sn
and −sp are in symmetric positions, a positive margin on
sn is equivalent to a negative margin on sp. It thus only
requires a single margin m. In Circle loss, sn and sp are
in asymmetric positions. Naturally, it requires respective
margins for sn and sp, which is formulated by:
Lcircle = log

1 +
L
X
j=1
exp(γαj
n(sj
n −∆n))
K
X
i=1
exp(−γαi
p(si
p −∆p))

(6)
in which ∆n and ∆p are the between-class and within-class
margins, respectively.
Basically, Circle loss in Eq. 6 expects si
p > ∆p and
sj
n < ∆n. We further analyze the settings of ∆n and ∆p
by deriving the decision boundary. For simplicity, we con-
sider the case of binary classiﬁcation, in which the decision
boundary is achieved at αn(sn −∆n) −αp(sp −∆p) = 0.
Combined with Eq. 5, the decision boundary is given by:
(sn −On + ∆n
2
)2 + (sp −Op + ∆p
2
)2 = C
(7)
in which C =
 (On −∆n)2 + (Op −∆p)2
/4.
Eq. 7 shows that the decision boundary is the arc of a
circle, as shown in Fig. 1 (b). The center of the circle is
at sn = (On + ∆n)/2, sp = (Op + ∆p)/2, and its radius
equals
√
C.
There are ﬁve hyper-parameters for Circle loss, i.e., Op,
On in Eq. 5 and γ, ∆p, ∆n in Eq. 6. We reduce the hyper-
parameters by setting Op = 1+m, On = −m, ∆p = 1−m,
and ∆n = m. Consequently, the decision boundary in Eq. 7
is reduced to:
(sn −0)2 + (sp −1)2 = 2m2.
(8)
With the decision boundary deﬁned in Eq. 8, we have
another intuitive interpretation of Circle loss. It aims to op-
timize sp →1 and sn →0. The parameter m controls
the radius of the decision boundary and can be viewed as
a relaxation factor. In another word, Circle loss expects
si
p > 1 −m and sj
n < m.
Hence there are only two hyper-parameters, i.e., the scale
factor γ and the relaxation margin m. We will experimen-
tally analyze the impacts of m and γ in Section 4.5.
3.3. The Advantages of Circle Loss
The gradients of Circle loss with respect to sj
n and si
p are
derived as follows:
∂Lcircle
∂sj
n
= Z
exp
 γ((sj
n)2 −m2)

PL
l=1 exp
 γ((sln)2 −m2)
 γ(sj
n + m),
(9)
and
∂Lcircle
∂sip
= Z
exp
 γ((si
p −1)2 −m2)

PK
k=1 exp
 γ((skp −1)2 −m2)
 γ(si
p−1−m), (10)

--- Page 5 ---
in both of which Z = 1 −exp(−Lcircle).
Under the toy scenario of binary classiﬁcation (or only
a single sn and sp), we visualize the gradients under dif-
ferent settings of m in Fig. 2 (c), from which we draw the
following three observations:
• Balanced optimization on sn and sp. We recall that the
loss functions minimizing (sn −sp) always have equal gra-
dients on sp and sn and is inﬂexible. In contrast, Circle loss
presents dynamic penalty strength. Among a speciﬁed sim-
ilarity pair {sn, sp}, if sp is better optimized in comparison
to sn (e.g., A = {0.8, 0.8} in Fig. 2 (c)), Circle loss assigns
a larger gradient to sn (and vice versa), so as to decrease
sn with higher superiority. The experimental evidence of
balanced optimization is to be accessed in Section 4.6.
• Gradually-attenuated gradients. At the start of train-
ing, the similarity scores deviate far from the optimum and
gain large gradients (e.g., “A” in Fig. 2 (c)). As the train-
ing gradually approaches the convergence, the gradients on
the similarity scores correspondingly decays (e.g., “B” in
Fig. 2 (c)), elaborating mild optimization. Experimental re-
sult in Section 4.5 shows that the learning effect is robust
to various settings of γ (in Eq. 6), which we attribute to the
automatically-attenuated gradients.
• A (more) deﬁnite convergence target. Circle loss has
a circular decision boundary and favors T rather than T ′
(Fig. 1) for convergence. It is because T has the smallest
gap between sp and sn, compared with all the other points
on the decision boundary. In another word, T ′ has a larger
gap between sp and sn and is inherently more difﬁcult to
maintain. In contrast, losses that minimize (sn −sp) have
a homogeneous decision boundary, that is, every point on
the decision boundary is of the same difﬁculty to reach. Ex-
perimentally, we observe that Circle loss leads to a more
concentrated similarity distribution after convergence, as to
be detailed in Section 4.6 and Fig. 5.
4. Experiments
We comprehensively evaluate the effectiveness of Circle
loss under two elemental learning approaches, i.e., learn-
ing with class-level labels and learning with pair-wise la-
bels. For the former approach, we evaluate our method on
face recognition (Section 4.2) and person re-identiﬁcation
(Section 4.3) tasks.
For the latter approach, we use the
ﬁne-grained image retrieval datasets (Section 4.4), which
are relatively small and encourage learning with pair-wise
labels. We show that Circle loss is competent under both
settings. Section 4.5 analyzes the impact of the two hyper-
parameters, i.e., the scale factor γ in Eq. 6 and the relaxation
factor m in Eq. 8. We show that Circle loss is robust un-
der reasonable settings. Finally, Section 4.6 experimentally
conﬁrms the characteristics of Circle loss.
4.1. Settings
Face recognition.
We use the popular dataset MS-
Celeb-1M [4] for training. The native MS-Celeb-1M data
is noisy and has a long-tailed data distribution. We clean
the dirty samples and exclude few tail identities (≤3 im-
ages per identity). It results in 3.6M images and 79.9K
identities. For evaluation, we adopt MegaFace Challenge
1 (MF1) [12], IJB-C [17], LFW [10], YTF [37] and CFP-
FP [23] datasets and the ofﬁcial evaluation protocols are
used.
We also polish the probe set and 1M distractors
on MF1 for more reliable evaluation, following [2]. For
data pre-processing, we resize the aligned face images to
112 × 112 and linearly normalize the pixel values of RGB
images to [−1, 1] [36, 15, 32]. We only augment the train-
ing samples by random horizontal ﬂip. We choose the pop-
ular residual networks [6] as our backbones. All the models
are trained with 182k iterations. The learning rate is started
with 0.1 and reduced by 10× at 50%, 70% and 90% of to-
tal iterations respectively. The default hyper-parameters of
our method are γ = 256 and m = 0.25 if not speciﬁed.
For all the model inference, we extract the 512-D feature
embeddings and use cosine distance as the metric.
Person re-identiﬁcation.
Person re-identiﬁcation (re-
ID) aims to spot the appearance of the same person in dif-
ferent observations. We evaluate our method on two pop-
ular datasets, i.e., Market-1501 [41] and MSMT17 [35].
Market-1501 contains 1,501 identities, 12,936 training im-
ages and 19,732 gallery images captured with 6 cameras.
MSMT17 contains 4,101 identities, 126,411 images cap-
tured with 15 cameras and presents a long-tailed sample
distribution. We adopt two network structures, i.e. a global
feature learning model backboned on ResNet50 and a part-
feature model named MGN [31]. We use MGN with consid-
eration of its competitive performance and relatively con-
cise structure. The original MGN uses a Sofmax loss on
each part feature branch for training. Our implementation
concatenates all the part features into a single feature vec-
tor for simplicity. For Circle loss, we set γ = 128 and
m = 0.25.
Fine-grained image retrieval.
We use three datasets
for evaluation on ﬁne-grained image retrieval, i.e. CUB-
200-2011 [28], Cars196 [14] and Stanford Online Prod-
ucts [19]. CARS-196 contains 16, 183 images which belong
to 196 class of cars. The ﬁrst 98 classes are used for train-
ing and the last 98 classes are used for testing. CUB-200-
2010 has 200 different class of birds. We use the ﬁrst 100
class with 5, 864 images for training and the last 100 class
with 5, 924 images for testing. SOP is a large dataset that
consists of 120, 053 images belonging to 22, 634 classes of
online products. The training set contains 11, 318 class in-
cludes 59, 551 images and the rest 11, 316 class includes
60, 499 images are for testing. The experimental setup fol-
lows [19]. We use BN-Inception [11] as the backbone to

--- Page 6 ---
Table 1:
Face identiﬁcation and veriﬁcation results on
MFC1 dataset. “Rank 1” denotes rank-1 identiﬁcation ac-
curacy. “Veri.” denotes veriﬁcation TAR (True Accepted
Rate) at 1e-6 FAR (False Accepted Rate) with 1M dis-
tractors. “R34” and “R100” denote using ResNet34 and
ResNet100 backbones, respectively.
Loss function
Rank 1 (%)
Veri. (%)
R34
R100
R34
R100
Softmax
92.36
95.04
92.72
95.16
NormFace [30]
92.62
95.27
92.91
95.37
AM-Softmax [29, 32]
97.54
98.31
97.64
98.55
ArcFace [2]
97.68
98.36
97.70
98.58
CircleLoss (ours)
97.81
98.50
98.12
98.73
Table 2: Face veriﬁcation accuracy (%) on LFW, YTF and
CFP-FP with ResNet34 backbone.
Loss function
LFW [10]
YTF [37]
CFP-FP [23]
Softmax
99.18
96.19
95.01
NormFace [30]
99.25
96.03
95.34
AM-Softmax [29, 32]
99.63
96.31
95.78
ArcFace [2]
99.68
96.34
95.84
CircleLoss(ours)
99.73
96.38
96.02
Table 3: Comparison of TARs on the IJB-C 1:1 veriﬁcation
task.
Loss function
TAR@FAR (%)
1e-3
1e-4
1e-5
ResNet34, AM-Softmax [29, 32]
95.87
92.14
81.86
ResNet34, ArcFace [2]
95.94
92.28
84.23
ResNet34, CircleLoss(ours)
96.04
93.44
86.78
ResNet100, AM-Softmax [29, 32]
95.93
93.19
88.87
ResNet100, ArcFace [2]
96.01
93.25
89.10
ResNet100, CircleLoss(ours)
96.29
93.95
89.60
learn 512-D embeddings.
We adopt P-K sampling trat-
egy [8] to construct mini-batch with P = 16 and K = 5.
For Circle loss, we set γ = 80 and m = 0.4.
4.2. Face Recognition
For face recognition task, we compare Circle loss
against several popular classiﬁcation loss functions, i.e.,
vanilla Softmax, NormFace [30], AM-Softmax [29] (or
CosFace [32]), ArcFace [2].
Following the original pa-
pers [29, 2], we set γ = 64, m = 0.35 for AM-Softmax
and γ = 64, m = 0.5 for ArcFace.
We report the identiﬁcation and veriﬁcation results on
MegaFace Challenge 1 dataset (MFC1) in Table 1. Circle
loss marginally outperforms the counterparts under differ-
Table 4: Evaluation of Circle loss on re-ID task. We report
R-1 accuracy (%) and mAP (%).
Method
Market-1501
MSMT17
R-1
mAP
R-1
mAP
PCB [26] (Softmax)
93.8
81.6
68.2
40.4
MGN [31] (Softmax+Triplet)
95.7
86.9
-
-
JDGL [42]
94.8
86.0
77.2
52.3
ResNet50 + AM-Softmax
92.4
83.8
75.6
49.3
ResNet50 + CircleLoss(ours)
94.2
84.9
76.3
50.2
MGN + AM-Softmax
95.3
86.6
76.5
51.8
MGN + CircleLoss(ours)
96.1
87.4
76.9
52.1
ent backbones. For example, with ResNet34 as the back-
bone, Circle loss surpasses the most competitive one (Ar-
cFace) by +0.13% at rank-1 accuracy. With ResNet100 as
the backbone, while ArcFace achieves a high rank-1 accu-
racy of 98.36%, Circle loss still outperforms it by +0.14%.
The same observations also hold for the veriﬁcation metric.
Table
2
summarizes
face
veriﬁcation
results
on
LFW [10], YTF [37] and CFP-FP [23]. We note that perfor-
mance on these datasets is already near saturation. Specif-
ically, ArcFace is higher than AM-Softmax by +0.05%,
+0.03%, +0.07% on three datasets, respectively.
Circle
loss remains the best one, surpassing ArcFace by +0.05%,
+0.06% and +0.18%, respectively.
We further compare Circle loss with AM-Softmax
and ArcFace on IJB-C 1:1 veriﬁcation task in Table 3.
Under both ResNet34 and ResNet100 backbones, Cir-
cle loss presents considerable superiority.
For example,
with ResNet34, Circle loss signiﬁcantly surpasses Arc-
Face by +1.16% and +2.55% on “TAR@FAR=1e-4” and
“TAR@FAR=1e-5”, respectively.
4.3. Person Re-identiﬁcation
We evaluate Circle loss on re-ID task in Table 4.
MGN [31] is one of the state-of-the-art methods and is
featured for learning multi-granularity part-level features.
Originally, it uses both Softmax loss and triplet loss to fa-
cilitate joint optimization. Our implementation of “MGN
(ResNet50) + AM-Softmax” and “MGN (ResNet50)+ Cir-
cle loss” only use a single loss function for simplicity.
We make three observations from Table 4.
First, we
ﬁnd that Circle loss can achieve competitive re-ID accu-
racy against state of the art.
We note that “JDGL” is
slightly higher than “MGN + Circle loss” on MSMT17 [35].
JDGL [42] uses a generative model to augment the training
data, and signiﬁcantly improves re-ID over the long-tailed
dataset. Second, comparing Circle loss with AM-Softmax,
we observe the superiority of Circle loss, which is consis-
tent with the experimental results on the face recognition
task. Third, comparing “ResNet50 + Circle loss” against

--- Page 7 ---
Table 5: Comparison of R@K(%) on three ﬁne-grained image retrieval datasets. Superscript denotes embedding size.
Loss function
CUB-200-2011 [28]
Cars196 [14]
Stanford Online Products [19]
R@1
R@2
R@4
R@8
R@1
R@2
R@4
R@8
R@1
R@10
R@102
R@103
LiftedStruct64 [19]
43.6
56.6
68.6
79.6
53.0
65.7
76.0
84.3
62.5
80.8
91.9
97.4
HDC384 [18]
53.6
65.7
77.0
85.6
73.7
83.2
89.5
93.8
69.5
84.4
92.8
97.7
HTL512 [3]
57.1
68.8
78.7
86.5
81.4
88.0
92.7
95.7
74.8
88.3
94.8
98.4
ABIER512 [20]
57.5
71.5
79.8
87.4
82.0
89.0
93.2
96.1
74.2
86.9
94.0
97.8
ABE512 [13]
60.6
71.5
79.8
87.4
85.2
90.5
94.0
96.1
76.3
88.4
94.8
98.2
Multi-Simi512 [34]
65.7
77.0
86.3
91.2
84.1
90.4
94.0
96.5
78.2
90.5
96.0
98.7
CircleLoss512
66.7
77.4
86.2
91.2
83.4
89.8
94.1
96.5
78.3
90.5
96.1
98.6
(a) scale factor 𝛾
(b) relaxation factor m
Rank-1 accuracy (%) on MFC1
Rank-1 accuracy (%) on MFC1
Figure 3: Impact of two hyper-parameters. In (a), Circle
loss presents high robustness on various settings of scale
factor γ. In (b), Circle loss surpasses the best performance
of both AM-Softmax and ArcFace within a large range of
relaxation factor m.
“MGN + Circle loss”, we ﬁnd that part-level features bring
incremental improvement to Circle loss.
It implies that
Circle loss is compatible with the part-model specially de-
signed for re-ID.
4.4. Fine-grained Image Retrieval
We evaluate the compatibility of Circle loss to pair-wise
labeled data on three ﬁne-grained image retrieval datasets,
i.e., CUB-200-2011, Cars196, and Standford Online Prod-
ucts. On these datasets, majority methods [19, 18, 3, 20,
13, 34] adopt the encouraged setting of learning with pair-
wise labels. We compare Circle loss against these state-
of-the-art methods in Table 5.
We observe that Circle
loss achieves competitive performance, on all of the three
datasets. Among the competing methods, LiftedStruct [19]
and Multi-Simi [34] are specially designed with elaborate
hard mining strategies for learning with pair-wise labels.
HDC [18], ABIER [20] and ABE [13] beneﬁt from model
ensemble. In contrast, the proposed Circle loss achieves
performance on par with the state of the art, without any
bells and whistles.
Figure 4: The change of sp and sn values during training.
We linearly lengthen the curves within the ﬁrst 2k iterations
to highlight the initial training process (in the green zone).
During the early training stage, Circle loss rapidly increases
sp, because sp deviates far from the optimum at the initial-
ization and thus attracts higher optimization priority.
4.5. Impact of the Hyper-parameters
We analyze the impact of two hyper-parameters, i.e., the
scale factor γ in Eq. 6 and the relaxation factor m in Eq. 8
on face recognition tasks.
The scale factor γ determines the largest scale of each
similarity score. The concept of the scale factor is critical in
a lot of variants of Softmax loss. We experimentally eval-
uate its impact on Circle loss and make a comparison with
several other loss functions involving scale factors. We vary
γ from 32 to 1024 for both AM-Softmax and Circle loss.
For ArcFace, we only set γ to 32, 64 and 128, as it becomes
unstable with larger γ in our implementation. The results
are visualized in Fig. 3. Compared with AM-Softmax and
ArcFace, Circle loss exhibits high robustness on γ. The
main reason for the robustness of Circle loss on γ is the au-
tomatic attenuation of gradients. As the similarity scores
approach the optimum during training, the weighting fac-
tors gradually decrease. Consequentially, the gradients au-
tomatically decay, leading to a moderated optimization.
The relaxation factor m determines the radius of the
circular decision boundary. We vary m from −0.2 to 0.3

--- Page 8 ---
𝑠𝑠𝑛𝑛
𝑠𝑠𝑝𝑝
𝑠𝑠𝑝𝑝
𝑠𝑠𝑝𝑝
𝑠𝑠𝑛𝑛
𝑠𝑠𝑛𝑛
(a) AMSoftmax (m=0.35)
(b) Circle loss  (m=0.325)
(c) Circle loss  (m=0.25)
Figure 5: Visualization of the similarity distribution after convergence. The blue dots mark the similarity pairs crossing
the decision boundary during the whole training process. The green dots mark the similarity pairs after convergence. (a)
AM-Softmax seeks to minimize (sn −sp). During training, the similarity pairs cross the decision boundary through a wide
passage. After convergence, the similarity pairs scatter in a relatively large region in the (sn, sp) space. In (b) and (c), Circle
loss has a circular decision boundary. The similarity pairs cross the decision boundary through a narrow passage and gather
into a relatively concentrated region.
(with 0.05 as the interval) and visualize the results in Fig. 3
(b). It is observed that under all the settings from −0.05 to
0.25, Circle loss surpasses the best performance of Arcface,
as well as AM-Softmax, presenting a considerable degree
of robustness.
4.6. Investigation of the Characteristics
Analysis of the optimization process.
To intuitively
understand the learning process, we show the change of sn
and sp during the whole training process in Fig. 4, from
which we draw two observations:
First, at the initialization, all the sn and sp scores are
small. It is because randomized features are prone to be
far away from each other in the high dimensional feature
space [40, 7]. Correspondingly, sp get signiﬁcantly larger
weights (compared with sn), and the optimization on sp
dominates the training, incurring a fast increase in similar-
ity values in Fig. 4. This phenomenon evidences that Circle
loss maintains a ﬂexible and balanced optimization.
Second, at the end of the training, Circle loss achieves
both better within-class compactness and between-class dis-
crepancy (on the training set), compared with AM-Softmax.
Because Circle loss achieves higher performance on the
testing set, we believe that it indicates better optimization.
Analysis of the convergence.
We analyze the conver-
gence status of Circle loss in Fig. 5. We investigate two
issues: how the similarity pairs consisted of sn and sp cross
the decision boundary during training and how they are dis-
tributed in the (sn, sp) space after convergence. The results
are shown in Fig. 5. In Fig. 5 (a), AM-Softmax loss adopts
the optimal setting of m = 0.35. In Fig. 5 (b), Circle loss
adopts a compromised setting of m = 0.325. The decision
boundaries of (a) and (b) are tangent to each other, allowing
an intuitive comparison. In Fig. 5 (c), Circle loss adopts its
optimal setting of m = 0.25. Comparing Fig. 5 (b) and (c)
against Fig. 5 (a), we ﬁnd that Circle loss presents a rela-
tively narrower passage on the decision boundary, as well
as a more concentrated distribution for convergence (espe-
cially when m = 0.25). It indicates that Circle loss fa-
cilitates more consistent convergence for all the similarity
pairs, compared with AM-Softmax loss. This phenomenon
conﬁrms that Circle loss has a more deﬁnite convergence
target, which promotes the separability in the feature space.
5. Conclusion
This paper provides two insights into the optimization
process for deep feature learning.
First, a majority of
loss functions, including the triplet loss and popular clas-
siﬁcation losses, conduct optimization by embedding the
between-class and within-class similarity into similarity
pairs. Second, within a similarity pair under supervision,
each similarity score favors different penalty strength, de-
pending on its distance to the optimum.
These insights
result in Circle loss, which allows the similarity scores to
learn at different paces. The Circle loss beneﬁts deep fea-
ture learning with high ﬂexibility in optimization and a
more deﬁnite convergence target. It has a uniﬁed formula
for two elemental learning approaches, i.e., learning with
class-level labels and learning with pair-wise labels. On
a variety of deep feature learning tasks, e.g., face recog-
nition, person re-identiﬁcation, and ﬁne-grained image re-
trieval, the Circle loss achieves performance on par with the
state of the art.

--- Page 9 ---
References
[1] S. Chopra, R. Hadsell, and Y. LeCun. Learning a similarity
metric discriminatively, with application to face veriﬁcation.
2005 IEEE Computer Society Conference on Computer Vi-
sion and Pattern Recognition (CVPR’05), 1:539–546 vol. 1,
2005. 2
[2] J. Deng, J. Guo, N. Xue, and S. Zafeiriou. Arcface: Additive
angular margin loss for deep face recognition. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition, 2019. 1, 2, 5, 6
[3] W. Ge. Deep metric learning with hierarchical triplet loss.
In The European Conference on Computer Vision (ECCV),
September 2018. 7
[4] Y. Guo, L. Zhang, Y. Hu, X. He, and J. Gao. Ms-celeb-1m:
A dataset and benchmark for large-scale face recognition. In
European Conference on Computer Vision, 2016. 5
[5] R. Hadsell, S. Chopra, and Y. LeCun. Dimensionality reduc-
tion by learning an invariant mapping. In IEEE Computer
Society Conference on Computer Vision and Pattern Recog-
nition (CVPR), volume 2, pages 1735–1742. IEEE, 2006. 2
[6] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning
for image recognition. In CVPR, 2016. 5
[7] L. He, Z. Wang, Y. Li, and S. Wang. Softmax dissection: To-
wards understanding intra- and inter-clas objective for em-
bedding learning. CoRR, abs/1908.01281, 2019. 8
[8] A. Hermans, L. Beyer, and B. Leibe.
In defense of the
triplet loss for person re-identiﬁcation.
arXiv preprint
arXiv:1703.07737, 2017. 3, 5
[9] E. Hoffer and N. Ailon. Deep metric learning using triplet
network.
In International Workshop on Similarity-Based
Pattern Recognition, pages 84–92. Springer, 2015. 1, 2
[10] G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller.
Labeled faces in the wild: A database for studying face
recognition in unconstrained environments. Technical Re-
port 07-49, University of Massachusetts, Amherst, October
2007. 5, 6
[11] S. Ioffe and C. Szegedy. Batch normalization: Accelerating
deep network training by reducing internal covariate shift.
arXiv preprint arXiv:1502.03167, 2015. 5
[12] I. Kemelmacher-Shlizerman, S. M. Seitz, D. Miller, and
E. Brossard. The megaface benchmark: 1 million faces for
recognition at scale. In Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, pages 4873–
4882, 2016. 5
[13] W. Kim, B. Goyal, K. Chawla, J. Lee, and K. Kwon.
Attention-based ensemble for deep metric learning. In The
European Conference on Computer Vision (ECCV), Septem-
ber 2018. 7
[14] J. Krause, M. Stark, J. Deng, and L. Fei-Fei. 3d object rep-
resentations for ﬁne-grained categorization. In Proceedings
of the IEEE International Conference on Computer Vision
Workshops, pages 554–561, 2013. 5, 7
[15] W. Liu, Y. Wen, Z. Yu, M. Li, B. Raj, and L. Song.
Sphereface: Deep hypersphere embedding for face recog-
nition. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 212–220, 2017. 2, 4, 5
[16] W. Liu, Y. Wen, Z. Yu, and M. Yang. Large-margin softmax
loss for convolutional neural networks. In ICML, 2016. 1, 2,
4
[17] B. Maze, J. Adams, J. A. Duncan, N. Kalka, T. Miller,
C. Otto, A. K. Jain, W. T. Niggel, J. Anderson, J. Cheney,
et al. Iarpa janus benchmark-c: Face dataset and protocol. In
2018 International Conference on Biometrics (ICB), pages
158–165. IEEE, 2018. 5
[18] H. Oh Song, S. Jegelka, V. Rathod, and K. Murphy. Deep
metric learning via facility location. In The IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), July
2017. 7
[19] H. Oh Song, Y. Xiang, S. Jegelka, and S. Savarese. Deep
metric learning via lifted structured feature embedding. In
Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pages 4004–4012, 2016. 2, 3, 5,
7
[20] M. Opitz, G. Waltner, H. Possegger, and H. Bischof. Deep
metric learning with bier: Boosting independent embeddings
robustly. IEEE Transactions on Pattern Analysis and Ma-
chine Intelligence, pages 1–1, 2018. 7
[21] R. Ranjan, C. D. Castillo, and R. Chellappa. L2-constrained
softmax loss for discriminative face veriﬁcation.
arXiv
preprint arXiv:1703.09507, 2017. 2, 4
[22] F. Schroff, D. Kalenichenko, and J. Philbin.
Facenet: A
uniﬁed embedding for face recognition and clustering. In
Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 815–823, 2015. 1, 2, 3
[23] S. Sengupta, J.-C. Chen, C. Castillo, V. M. Patel, R. Chel-
lappa, and D. W. Jacobs. Frontal to proﬁle face veriﬁcation
in the wild. In 2016 IEEE Winter Conference on Applica-
tions of Computer Vision (WACV), pages 1–9. IEEE, 2016.
5, 6
[24] K. Sohn. Improved deep metric learning with multi-class
n-pair loss objective. In NIPS, 2016. 2, 3
[25] Y. Sun, X. Wang, and X. Tang. Deep learning face repre-
sentation from predicting 10,000 classes. In Proceedings of
the IEEE conference on computer vision and pattern recog-
nition, pages 1891–1898, 2014. 1
[26] Y. Sun, L. Zheng, Y. Yang, Q. Tian, and S. Wang. Beyond
part models: Person retrieval with reﬁned part pooling (and a
strong convolutional baseline). In The European Conference
on Computer Vision (ECCV), September 2018. 6
[27] E. Ustinova and V. S. Lempitsky. Learning deep embeddings
with histogram loss. In NIPS, 2016. 2
[28] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie.
The Caltech-UCSD Birds-200-2011 Dataset. Technical Re-
port CNS-TR-2011-001, California Institute of Technology,
2011. 5, 7
[29] F. Wang, J. Cheng, W. Liu, and H. Liu. Additive margin
softmax for face veriﬁcation. IEEE Signal Processing Let-
ters, 25(7):926–930, 2018. 1, 2, 3, 4, 6
[30] F. Wang, X. Xiang, J. Cheng, and A. L. Yuille. Normface:
L2 hypersphere embedding for face veriﬁcation. In Proceed-
ings of the 25th ACM international conference on Multime-
dia, pages 1041–1049. ACM, 2017. 2, 3, 4, 6

--- Page 10 ---
[31] G. Wang, Y. Yuan, X. Chen, J. Li, and X. Zhou. Learning
discriminative features with multiple granularities for person
re-identiﬁcation. 2018 ACM Multimedia Conference on Mul-
timedia Conference - MM 18, 2018. 5, 6
[32] H. Wang, Y. Wang, Z. Zhou, X. Ji, D. Gong, J. Zhou, Z. Li,
and W. Liu. Cosface: Large margin cosine loss for deep face
recognition. In The IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), 2018. 1, 2, 3, 4, 5, 6
[33] J. J. Wang, F. Zhou, S. Wen, X. Liu, and Y. Lin. Deep metric
learning with angular loss. 2017 IEEE International Confer-
ence on Computer Vision (ICCV), pages 2612–2620, 2017.
2
[34] X. Wang, X. Han, W. Huang, D. Dong, and M. R. Scott.
Multi-similarity loss with general pair weighting for deep
metric learning.
In Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, pages 5022–
5030, 2019. 2, 3, 7
[35] L. Wei, S. Zhang, W. Gao, and Q. Tian. Person transfer gan
to bridge domain gap for person re-identiﬁcation.
In The
IEEE Conference on Computer Vision and Pattern Recogni-
tion (CVPR), June 2018. 5, 6
[36] Y. Wen, K. Zhang, Z. Li, and Y. Qiao.
A discrimina-
tive feature learning approach for deep face recognition. In
European conference on computer vision, pages 499–515.
Springer, 2016. 1, 5
[37] L. Wolf, T. Hassner, and I. Maoz. Face recognition in un-
constrained videos with matched background similarity. In
CVPR, 2011. 5, 6
[38] C.-Y. Wu, R. Manmatha, A. J. Smola, and P. Krahenbuhl.
Sampling matters in deep embedding learning. In Proceed-
ings of the IEEE International Conference on Computer Vi-
sion, pages 2840–2848, 2017. 2
[39] X. Zhang, F. X. Yu, S. Karaman, W. Zhang, and S.-F. Chang.
Heated-up softmax embedding.
ArXiv, abs/1809.04157,
2018. 4
[40] X. Zhang, R. Zhao, Y. Qiao, X. Wang, and H. Li. Adacos:
Adaptively scaling cosine logits for effectively learning deep
face representations. In CVPR, 2019. 4, 8
[41] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, and Q. Tian.
Scalable person re-identiﬁcation: A benchmark. In The IEEE
International Conference on Computer Vision (ICCV), De-
cember 2015. 5
[42] Z. Zheng, X. Yang, Z. Yu, L. Zheng, Y. Yang, and J. Kautz.
Joint discriminative and generative learning for person re-
identiﬁcation. In The IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), June 2019. 6
