# Proxy Anchor Loss for Deep Metric Learning

**Authors**: Kim, Kim, Choi, You
**Year**: 2020
**arXiv**: 2003.13911
**Topic**: metric_learning
**Relevance**: Proxy-based metric learning for efficient training

---


--- Page 1 ---
Proxy Anchor Loss for Deep Metric Learning
Sungyeon Kim
Dongwon Kim
Minsu Cho
Suha Kwak
POSTECH, Pohang, Korea
{tjddus9597, kdwon, mscho, suha.kwak}@postech.ac.kr
Abstract
Existing metric learning losses can be categorized into
two classes: pair-based and proxy-based losses. The former
class can leverage ﬁne-grained semantic relations between
data points, but slows convergence in general due to its high
training complexity. In contrast, the latter class enables fast
and reliable convergence, but cannot consider the rich data-
to-data relations. This paper presents a new proxy-based
loss that takes advantages of both pair- and proxy-based
methods and overcomes their limitations. Thanks to the use
of proxies, our loss boosts the speed of convergence and is
robust against noisy labels and outliers. At the same time,
it allows embedding vectors of data to interact with each
other through its gradients to exploit data-to-data relations.
Our method is evaluated on four public benchmarks, where
a standard network trained with our loss achieves state-of-
the-art performance and most quickly converges.
1. Introduction
Learning a semantic distance metric has been a crucial
step for many applications such as content-based image
retrieval [14, 21, 27, 29], face veriﬁcation [18, 25], per-
son re-identiﬁcation [3, 38], few-shot learning [24, 26, 30],
and representation learning [14, 33, 41]. Following their
great success in visual recognition, deep neural networks
have been employed recently for metric learning. The net-
works are trained to project data onto an embedding space
in which semantically similar data (e.g., images of the same
class) are closely grouped together. Such a quality of the
embedding space is given mainly by loss functions used for
training the networks, and most of the losses are categorized
into two classes: pair-based and proxy-based.
The pair-based losses are built upon pairwise distances
between data in the embedding space. A seminal example
is Contrastive loss [4, 9], which aims to minimize the dis-
tance between a pair of data if their class labels are identical
and to separate them otherwise. Recent pair-based losses
consider a group of pairwise distances to handle relations
between more than two data [14, 25, 27, 29, 32, 34, 35, 39].
Method
Proxy-Anchor (Ours)
MS [34]
Proxy-NCA [21]
Semi-Hard Triplet [25]
N-Pair [27]
Time Per Epoch
27.10s
28.43s
27.41s
29.97s
28.41s
Training Time (Min)
R@1
0.3
0.5
0.4
0.6
0.7
0.8
0.9
20
40
60
80
100
Figure 1. Accuracy in Recall@1 versus training time on the Cars-
196 [17] dataset. Note that all methods were trained with batch
size of 150 on a single Titan Xp GPU. Our loss enables to achieve
the highest accuracy, and converge faster than the baselines in
terms of both the number of epochs and the actual training time.
These losses provide rich supervisory signals for training
embedding networks by comparing data to data and exam-
ining ﬁne-grained relations between them, i.e., data-to-data
relations. However, since they take a tuple of data as a unit
input, the losses cause prohibitively high training complex-
ity1, O(M 2) or O(M 3) where M is the number of training
data, thus slow convergence. Furthermore, some tuples do
not contribute to training or even degrade the quality of the
learned embedding space. To resolve these issues, learning
with the pair-based losses often entails tuple sampling tech-
niques [10, 25, 37, 40], which however have to be tuned by
hand and may increase the risk of overﬁtting.
The proxy-based losses resolve the above complexity is-
sue by introducing proxies [1, 21, 23]. A proxy is a repre-
sentative of a subset of training data and learned as a part
of the network parameters. Existing losses in this category
consider each data point as an anchor, associate it with prox-
ies instead of other images, and encourage the anchor to be
close to proxies of the same class and far apart from those
of different classes. Proxy-based losses reduce the training
1The training complexity indicates the amount of computation required
to address the entire training dataset [1, 6, 10, 23, 35].
arXiv:2003.13911v1  [cs.CV]  31 Mar 2020

--- Page 2 ---
(b) N-pair
(d) Proxy-NCA
E
(e) Ours
(c) Lifted Structure
(a) Triplet
Figure 2. Comparison between popular metric learning losses and ours. Small nodes are embedding vectors of data in a batch, and black
ones indicate proxies; their different shapes represent distinct classes. The associations deﬁned by the losses are expressed by edges, and
thicker edges get larger gradients. Also, embedding vectors associated with the anchor are colored in red if they are of the same class of the
anchor (i.e., positive) and in blue otherwise (i.e., negative). (a) Triplet loss [25, 32] associates each anchor with a positive and a negative
data point without considering their hardness. (b) N-pair loss [27] and (c) Lifted Structure loss [29] reﬂect hardness of data, but do not
utilize all data in the batch. (d) Proxy-NCA loss [21] cannot exploit data-to-data relations since it associates each data point only with
proxies. (e) Our loss handles entire data in the batch, and associates them with each proxy with consideration of their relative hardness
determined by data-to-data relations. See the text for more details.
complexity and enable faster convergence since the num-
ber of proxies is substantially smaller than that of training
data in general. Further, these losses tend to be more robust
against label noises and outliers. However, since they asso-
ciate each data point only with proxies, proxy-based losses
can leverage only data-to-proxy relations, which are impov-
erished compared to the rich data-to-data relations available
for pair-based losses.
In this paper, we propose a novel proxy-based loss called
Proxy-Anchor loss, which takes good points of both proxy-
based and pair-based losses while correcting their defects.
Unlike the existing proxy-based losses, the proposed loss
utilizes each proxy as an anchor and associates it with all
data in a batch. Speciﬁcally, for each proxy, the loss aims
to pull data of the same class close to the proxy and to push
others away in the embedding space. Due to the use of prox-
ies, our loss boosts the speed of convergence with no hyper-
parameter for tuple sampling, and is robust against noisy la-
bels and outliers. At the same time, it can take data-to-data
relations into account like pair-based losses; this property is
given by associating all data in a batch with each proxy so
that the gradients with respect to a data point are weighted
by its relative proximity to the proxy (i.e., relative hard-
ness) affected by the other data in the batch. Thanks to the
above advantages, a standard embedding network trained
with our loss achieves state-of-the-art accuracy and most
quickly converges as shown in Figure 1. The contribution
of this paper is three-fold:
• We propose a novel metric learning loss that takes ad-
vantages of both pair-based and proxy-based methods; it
leverages rich data-to-data relations and enables fast and
reliable convergence.
• A standard embedding network trained with our loss
achieves state-of-the-art performance on the four public
benchmarks for metric learning [17, 19, 29, 36].
• Our loss speeds up convergence greatly without careful
data sampling; its convergence is even faster than those
of Proxy-NCA [21] and Multi-Similarity loss [34].
2. Related Work
In this section, we categorize metric learning losses into
two classes, pair-based and proxy-based losses, then review
relevant methods for each category.
2.1. Pair-based Losses
Contrastive loss [2, 4, 9] and Triplet loss [25, 32] are
seminal examples of loss functions for deep metric learning.
Contrastive loss takes a pair of embedding vectors as input,
and aims to pull them together if they are of the same class
and push them apart otherwise. Triplet loss considers a data
point as an anchor, associates it with a positive and a neg-
ative data point, and constrains the distance of the anchor-
positive pair to be smaller than that of the anchor-negative
pair in the embedding space as illustrated in Figure 2(a).
Recent pair-based losses aim to leverage higher order re-
lations between data and reﬂect their hardness for further
enhancement. As generalizations of Triplet loss, N-pair
loss [27] and Lifted Structure loss [29] associate an anchor
with a single positive and multiple negative data points, and
pull the positive to the anchor and push the negatives away
from the anchor while considering their hardness. As shown
in Figure 2(b) and 2(c), however, these losses do not utilize
entire data in a batch since they sample the same number
of data per negative class, thus may drop informative ex-
amples during training. In contrast, Ranked List loss [35]
takes into account all positive and negative data in a batch
and aims to separate the positive and negative sets. Multi-
Similarity loss [34] also considers every pair of data in a
batch, and assigns a weight to each pair according to three
complementary types of similarity to focus more on useful
pairs for improving performance and convergence speed.

--- Page 3 ---
Pair-based losses enjoy rich and ﬁne-grained data-to-
data relations as they examine tuples (i.e., data pairs or their
combinations) during training. However, since the number
of tuples increases polynomially with the number of train-
ing data, their training complexity is prohibitively high and
convergence is slow. In addition, a large amount of tuples
are not effective and sometimes even degrade the quality
of the learned embedding space [25, 37]. To address this
issue, most pair-based losses entail tuple sampling tech-
niques [10, 25, 37, 40] to select and utilize tuples that will
contribute to training. However, these techniques involve
hyperparameters that have to be tuned carefully, and may
increase the risk of overﬁtting since they rely mostly on lo-
cal pairwise relations within a batch. Another way to alle-
viating the complexity issue is to assign larger weights to
more useful pairs during training as in [34], which however
also incorporates a sampling technique.
Our loss resolves this complexity issue by adopting prox-
ies, which enables faster and more reliable convergence
compared to pair-based losses. Furthermore, it demands no
additional hyperparameter for tuple sampling.
2.2. Proxy-based Losses
Proxy-based metric learning is a relatively new approach
that can address the complexity issue of the pair-based
losses. A proxy means a representative of a subset of train-
ing data and is estimated as a part of the embedding net-
work parameters. The common idea of the methods in this
category is to infer a small set of proxies that capture the
global structure of an embedding space and relate each data
point with the proxies instead of the other data points dur-
ing training. Since the number of proxies is signiﬁcantly
smaller than that of training data, the training complexity
can be reduced substantially.
The ﬁrst proxy-based loss is Proxy-NCA [21], which
is an approximation of Neighborhood Component Analy-
sis (NCA) [8] using proxies. In its standard setting, Proxy-
NCA loss assigns a single proxy for each class, associates a
data point with proxies, and encourages the positive pair to
be close and negative pairs to be far apart, as illustrated in
Figure 2(d). SoftTriple loss [23], an extension of SoftMax
loss for classiﬁcation, is similar to Proxy-NCA yet assigns
multiple proxies to each class to reﬂect intra-class variance.
Manifold Proxy loss [1] is an extension of N-pair loss us-
ing proxies, and improves the performance by adopting a
manifold-aware distance instead of Euclidean distance to
measure the semantic distance in the embedding space.
Using proxies in these losses helps improve training con-
vergence greatly, but has an inherent limitation as a side
effect: Since each data point is associated only with prox-
ies, the rich data-to-data relations that are available for the
pair-based methods are not accessible anymore. Our loss
can overcome this limitation since its gradients reﬂect rela-
tive hardness of data and allow their embedding vectors to
interact with each other during training.
3. Our Method
We propose a new metric learning loss called Proxy-
Anchor loss to overcome the inherent limitations of the pre-
vious methods. The loss employs proxies that enable fast
and reliable convergence as in proxy-based losses. Also,
although it is built upon data-proxy relations, our loss can
utilize data-to-data relations during training like pair-based
losses since it enables embedding vectors of data points to
be affected by each other through its gradients. This prop-
erty of our loss improves the quality of the learned embed-
ding space substantially.
In this section, we ﬁrst review Proxy-NCA loss [21],
a representative proxy-based loss, for comparison to our
Proxy-Anchor loss. We then describe our Proxy-Anchor
loss in detail and analyze its training complexity.
3.1. Review of Proxy-NCA Loss
In the standard setting, Proxy-NCA loss [21] assigns a
proxy to each class so that the number of proxies is the same
with that of class labels. Given an input data point as an
anchor, the proxy of the same class of the input is regarded
as positive and the other proxies are negative. Let x denote
the embedding vector of the input, p+ be the positive proxy,
and p−be a negative proxy. The loss is then given by
ℓ(X) =
X
x∈X
−log
es(x,p+)
X
p−∈P −
es(x,p−)
(1)
=
X
x∈X
n
−s(x, p+) + LSE
p−∈P −s(x, p−)
o
,
(2)
where X is a batch of embedding vectors, P −is the set
of negative proxies, and s(·, ·) denotes the cosine similarity
between two vectors. In addition, LSE in Eq. (2) means the
Log-Sum-Exp function, a smooth approximation to the max
function. The gradient of Proxy-NCA loss with respect to
s(x, p) is given by
∂ℓ(X)
∂s(x, p) =









−1,
if p = p+,
es(x,p)
X
p−∈P −
es(x,p−) ,
otherwise.
(3)
Eq. (3) shows that minimizing the loss encourages x and
p+ to be close to each other, and x and p−to be far away.
In particular, x and p+ are pulled together by the constant
power, while x and p−closer to each other (i.e., harder neg-
ative) are more strongly pushed away.
Proxy-NCA loss enables fast convergence thanks to its
low training complexity, O(MC) where M is the number

--- Page 4 ---
of training data and C is that of classes, which is substan-
tially lower than O(M 2) or O(M 3) of pair-based losses
since C ≪M; refer to Section 3.3 for details. Also, prox-
ies are robust against outliers and noisy labels since they are
trained to represent groups of data. However, since the loss
associates each embedding vector only with proxies, it can-
not exploit ﬁne-grained data-to-data relations. This draw-
back limits the capability of embedding networks trained
with Proxy-NCA loss.
3.2. Proxy-Anchor Loss
Our Proxy-Anchor loss is designed to overcome the lim-
itation of Proxy-NCA while keeping the low training com-
plexity. The main idea is to take each proxy as an anchor
and associate it with entire data in a batch, as illustrated in
Figure 2(e), so that the data interact with each other through
the proxy anchor during training. Our loss assigns a proxy
for each class following the standard proxy assignment set-
ting of Proxy-NCA, and is formulated as
ℓ(X) =
1
|P +|
X
p∈P +
log

1 +
X
x∈X+
p
e−α(s(x,p)−δ)

+ 1
|P|
X
p∈P
log

1 +
X
x∈X−
p
eα(s(x,p)+δ)

,
(4)
where δ > 0 is a margin, α > 0 is a scaling factor, P
indicates the set of all proxies, and P + denotes the set of
positive proxies of data in the batch. Also, for each proxy
p, a batch of embedding vectors X is divided into two sets:
X+
p , the set of positive embedding vectors of p, and X−
p =
X −X+
p . The proposed loss can be rewritten in an easier-
to-interpret form as
ℓ(X) =
1
|P +|
X
p∈P +

Softplus

LSE
x∈X+
p
−α(s(x, p) −δ)

+ 1
|P|
X
p∈P

Softplus

LSE
x∈X−
p
α(s(x, p) + δ)

,
(5)
where Softplus(z) = log (1 + ez), ∀z ∈R, and is a smooth
approximation of ReLU.
How it works: Regarding Log-Sum-Exp as the max func-
tion, it is easy to notice that the loss aims to pull p and its
most dissimilar positive example (i.e., hardest positive ex-
ample) together, and to push p and its most similar nega-
tive example (i.e., hardest negative example) apart. Due to
the nature of Log-Sum-Exp, the loss in practice pulls and
pushes all embedding vectors in the batch, but with differ-
ent degrees of strength that are determined by their relative
hardness. This characteristic is demonstrated by the gradi-
ent of our loss with respect to s(x, p), which is given by
∂ℓ(X)
∂s(x, p) =



















1
|P +|
−α h+
p (x)
1 +
X
x′∈X+
p
h+
p (x′)
,
∀x ∈X+
p ,
1
|P|
α h−
p (x)
1 +
X
x′∈X−
p
h−
p (x′)
,
∀x ∈X−
p ,
(6)
where h+
p (x) = e−α(s(x,p)−δ) and h−
p (x) = eα(s(x,p)+δ)
are positive and negative hardness metrics for embedding
vector x given proxy p, respectively; h+
p (x) is large when
the positive embedding vector x is far from p, and h−
p (x) is
large when the negative embedding vector x is close to p.
The scaling parameter α and margin δ control the relative
hardness of data points, and in consequence, determine how
strongly pull or push their embedding vectors.
As shown in the above equations, the gradient for s(x, p)
is affected by not only x but also other embedding vectors
in the batch; the gradient becomes larger when x is harder
than the others. In this way, our loss enables embedding
vectors in the batch to interact with each other and reﬂects
their relative hardness through the gradients, which helps
enhance the quality of the learned embedding space.
Comparison to Proxy-NCA: The key difference and ad-
vantage of Proxy-Anchor over Proxy-NCA is the active
consideration of relative hardness based on data-to-data re-
lations. This property enables Proxy-Anchor loss to pro-
vide richer supervisory signals to embedding networks dur-
ing training. The gradients of the two losses demonstrate
this clearly. In Proxy-NCA loss, the scale of the gradient
is constant for every positive example and that of a nega-
tive example is calculated by taking only few proxies into
account as shown in Eq. (3).
In particular, the constant
gradient scale for positive examples damages the ﬂexibility
and generalizability of embedding networks [37]. In con-
trast, Proxy-Anchor loss determines the scale of the gradi-
ent by taking relative hardness into consideration for both
positive and negative examples as shown in Eq. (6). This
feature of our loss allows the embedding network to con-
sider data-to-data relations that are ignored in Proxy-NCA
and observe much larger area of the embedding space dur-
ing training than Proxy-NCA. Figure 3 illustrates these dif-
ferences between the two losses in terms of handling the
relative hardness of embedding vectors. In addition, unlike
Proxy-Anchor loss, the margin imposed in our loss leads to
intra-class compactness and inter-class separability, result-
ing in a more discriminative embedding space.
3.3. Training Complexity Analysis
Let M, C, B, and U denote the numbers of training sam-
ples, classes, batches per epoch, and proxies held by each

--- Page 5 ---
Case of Positive Examples
3
E
3
W
3
E
3
W
3
E
3
W
3
3
W
E
(a) Proxy-NCA
(b) Proxy-Anchor
Case of Negative Examples
3
E
3
W
3
E
3
W
3
W
E
3
3
E
3
W
(c) Proxy-NCA
(d) Proxy-Anchor
Figure 3. Differences between Proxy-NCA and Proxy-Anchor in handling proxies and embedding vectors during training. Each proxy
is colored in black and three different colors indicate distinct classes. The associations deﬁned by the losses are expressed by edges, and
thicker edges get larger gradients. (a) Gradients of Proxy-NCA loss with respect to positive examples have the same scale regardless of
their hardness. (b) Proxy-Anchor loss dynamically determines gradient scales regarding relative hardness of all positive examples so as
to pull harder positives more strongly. (c) In Proxy-NCA, each negative example is pushed only by a small number of proxies without
considering the distribution of embedding vectors in ﬁne details. (d) Proxy-Anchor loss considers the distribution of embedding vectors in
more details as it has all negative examples affect each other in their gradients.
class, respectively. U is 1 thus ignored in most of proxy-
based losses including ours, but is nontrivial for those man-
aging multiple proxies per class such as SoftTriple loss [23].
Table 1 compares the training complexity of our loss
with those of popular pair- and proxy-based losses. The
complexity of our loss is O(MC) since it compares every
proxy with all positive or all negative examples in a batch.
More speciﬁcally, in Eq. (4), the complexity of the ﬁrst sum-
mation is O(MC) and that of the second summation is also
O(MC), hence the total training complexity is O(MC).
The complexity of Proxy-NCA [21] is also O(MC) since
each data point is associated with one positive proxy and
C−1 negative proxies as can be seen in Eq. (2). On the
other hand, SoftTriple loss [23], a modiﬁcation of SoftMax
using multiple proxies per class, associates each data point
with U positive proxies and U(C−1) negative proxies. The
total training complexity of this loss is thus O(MCU 2). In
conclusion, the complexity of our loss is the same with or
even lower than that of other proxy-based losses.
The training complexity of pair-based losses is higher
than that of proxy-based ones. Since Contrastive loss [2,
4, 9] takes a pair of data as input, its training complexity
is O(M 2). On the other hand, Triplet loss that examines
triplets of data has complexity O(M 3), which can be re-
duced by triplet mining strategies. For example, semi-hard
mining [25] reduces the complexity to O(M 3/B2) by se-
lecting negative pairs that are located within a neighbor-
hood of anchor but sufﬁciently far from it. Similarly, Smart
mining [10] lowers the complexity to O(M 2) by sampling
Type
Loss
Training Complexity
Proxy
Proxy-Anchor (Ours)
O(MC)
Proxy-NCA [21]
O(MC)
SoftTriple [23]
O(MCU 2)
Pair
Contrastive [2, 4, 9]
O(M 2)
Triplet (Semi-Hard) [25]
O(M 3/B2)
Triplet (Smart) [10]
O(M 2)
N-pair [27]
O(M 3)
Lifted Structure [29]
O(M 3)
Table 1. Comparison of training complexities.
hard triplets using an approximated nearest neighbor index.
However, even with these techniques, the training complex-
ity of Triplet loss is still high. Like Triplet loss, N-pair
loss [27] and Lifted Structure loss [29] that compare each
positive pair of data to multiple negative pairs also have
complexity O(M 3). The training complexity of these losses
becomes prohibitively high as the number of training data
M increases, which slows down the speed of convergence
as demonstrated in Figure 1.
4. Experiments
In this section, our method is evaluated and compared to
current state-of-the-art on the four benchmark datasets for
deep metric learning [17, 19, 29, 36]. We also investigate
the effect of hyperparameters and embedding dimensional-
ity of our loss to demonstrate its robustness.

--- Page 6 ---
Recall@K
CUB-200-2011
Cars-196
1
2
4
8
1
2
4
8
Clustering64 [28]
BN
48.2
61.4
71.8
81.9
58.1
70.6
80.3
87.8
Proxy-NCA64 [21]
BN
49.2
61.9
67.9
72.4
73.2
82.4
86.4
87.8
Smart Mining64 [10]
G
49.8
62.3
74.1
83.3
64.7
76.2
84.2
90.2
MS64 [34]
BN
57.4
69.8
80.0
87.8
77.3
85.3
90.5
94.2
SoftTriple64 [23]
BN
60.1
71.9
81.2
88.5
78.6
86.6
91.8
95.4
Proxy-Anchor64
BN
61.7
73.0
81.8
88.8
78.8
87.0
92.2
95.5
Margin128 [37]
R50
63.6
74.4
83.1
90.0
79.6
86.5
91.9
95.1
HDC384 [40]
G
53.6
65.7
77.0
85.6
73.7
83.2
89.5
93.8
A-BIER512 [22]
G
57.5
68.7
78.3
86.2
82.0
89.0
93.2
96.1
ABE512 [15]
G
60.6
71.5
79.8
87.4
85.2
90.5
94.0
96.1
HTL512 [7]
BN
57.1
68.8
78.7
86.5
81.4
88.0
92.7
95.7
RLL-H512 [35]
BN
57.4
69.7
79.2
86.9
74.0
83.6
90.1
94.1
MS512 [34]
BN
65.7
77.0
86.3
91.2
84.1
90.4
94.0
96.5
SoftTriple512 [23]
BN
65.4
76.4
84.5
90.4
84.5
90.7
94.5
96.9
Proxy-Anchor512
BN
68.4
79.2
86.8
91.6
86.1
91.7
95.0
97.3
†Contra+HORDE512 [13]
BN
66.3
76.7
84.7
90.6
83.9
90.3
94.1
96.3
†Proxy-Anchor512
BN
71.1
80.4
87.4
92.5
88.3
93.1
95.7
97.5
Table 2.
Recall@K (%) on the CUB-200-2011 and Cars-196 datasets. Superscripts denote embedding sizes and † indicates models
using larger input images. Backbone networks of the models are denoted by abbreviations: G–GoogleNet [31], BN–Inception with batch
normalization [12], R50–ResNet50 [11].
4.1. Datasets
We employ CUB-200-2011 [36], Cars-196 [17], Stan-
ford Online Product (SOP) [29] and In-shop Clothes Re-
trieval (In-Shop) [19] datasets for evaluation. For CUB-
200-2011, we use 5,864 images of its ﬁrst 100 classes for
training and 5,924 images of the other classes for testing.
For Cars-196, 8,054 images of its ﬁrst 98 classes are used
for training and 8,131 images of the other classes are kept
for testing. For SOP, we follow the standard dataset split
in [29] using 59,551 images of 11,318 classes for training
and 60,502 images of the rest classes for testing. Also for
In-Shop, we follow the setting in [19] using 25,882 images
of the ﬁrst 3,997 classes for training and 28,760 images of
the other classes for testing; the test set is further partitioned
into a query set with 14,218 images of 3,985 classes and a
gallery set with 12,612 images of 3,985 classes.
4.2. Implementation Details
Embedding network: For a fair comparison to previous
work, the Inception network with batch normalization [12]
pre-trained for ImageNet classiﬁcation [5] is adopted as our
embedding network. We change the size of its last fully
connected layer according to the dimensionality of embed-
ding vectors, and L2-normalize the ﬁnal output.
Training: In every experiment, we employ AdamW opti-
mizer [20], which has the same update step of Adam [16]
yet decays the weight separately. Our model is trained for
40 epochs with initial learning rate 10−4 on the CUB-200-
2011 and Cars-196, and for 60 epochs with initial learning
rate 6 · 10−4 on the SOP and In-shop. The learning rate for
proxies is scaled up 100 times for faster convergence. Input
batches are randomly sampled during training.
Proxy setting: We assign a single proxy for each semantic
class following Proxy-NCA [21]. The proxies are initialized
using a normal distribution to ensure that they are uniformly
distributed on the unit hypersphere.
Image setting: Input images are augmented by random
cropping and horizontal ﬂipping during training while they
are center-cropped in testing. The default size of cropped
images is 224×224 as in most of previous work, but for
comparison to HORDE [13], we also implement models
trained and tested with 256×256 cropped images.
Hyperparameter setting: α and δ in Eq. (4) is set to 32
and 10−1, respectively, for all experiments.
4.3. Comparison to Other Methods
We demonstrate the superiority of our Proxy-Anchor
loss quantitatively by evaluating its image retrieval perfor-
mance on the four benchmark datasets. For a fair compar-
ison to previous work, the accuracy of our model is mea-
sured in three different settings: 64/128 embedding dimen-
sion with the default image size (224×224), 512 embedding
dimension with the default image size, and 512 embedding
dimension with the larger image size (256×256).
Results on the CUB-200-2011 and Cars-196 datasets are
summarized in Table 2.
Our model outperforms all the
previous arts including ensemble methods [15, 22] in all
the three settings. In particular, on the challenging CUB-
200-2011 dataset, it improves the previous best score by a
large margin, 2.7% in Recall@1. As reported in Table 3,

--- Page 7 ---
Recall@K
1
10
100
1000
Clustering64 [28]
67.0
83.7
93.2
-
Proxy-NCA64 [21]
73.7
-
-
-
MS64 [34]
74.1
87.8
94.7
98.2
SoftTriple64 [23]
76.3
89.1
95.3
-
Proxy-Anchor64
76.5
89.0
95.1
98.2
Margin128 [37]
72.7
86.2
93.8
98.0
HDC384 [40]
69.5
84.4
92.8
97.7
A-BIER512 [22]
74.2
86.9
94.0
97.8
ABE512 [15]
76.3
88.4
94.8
98.2
HTL512 [7]
74.8
88.3
94.8
98.4
RLL-H512 [35]
76.1
89.1
95.4
-
MS512 [34]
78.2
90.5
96.0
98.7
SoftTriple512 [23]
78.3
90.3
95.9
-
Proxy-Anchor512
79.1
90.8
96.2
98.7
†Contra+HORDE512 [13]
80.1
91.3
96.2
98.7
†Proxy-Anchor512
80.3
91.4
96.4
98.7
Table 3. Recall@K (%) on the SOP. Superscripts denote embed-
ding sizes and † indicates models using larger input images.
Recall@K
1
10
20
40
HDC384 [40]
62.1
84.9
89.0
92.3
HTL128 [7]
80.9
94.3
95.8
97.4
MS128 [34]
88.0
97.2
98.1
98.7
Proxy-Anchor128
90.8
97.9
98.5
99.0
FashionNet4096 [19]
53.0
73.0
76.0
79.0
A-BIER512 [22]
83.1
95.1
96.9
97.8
ABE512 [15]
87.3
96.7
97.9
98.5
MS512 [34]
89.7
97.9
98.5
99.1
Proxy-Anchor512
91.5
98.1
98.8
99.1
†Contra+HORDE512 [13]
90.4
97.8
98.4
98.9
†Proxy-Anchor512
92.6
98.3
98.9
99.3
Table 4.
Recall@K (%) on the In-Shop. Superscripts denote
embedding sizes and † indicates models using larger input images.
our model also achieves state-of-the-art performance on the
SOP dataset. It outperforms previous models in all the cases
except for Recall@10 and Recall@100 with 64 dimensional
embedding, but even in these cases it achieves the second
best.
Finally, on the In-Shop dataset, it attains the best
scores in all the three settings as shown in Table 4.
For all the datasets, our model with the larger crop size
and 512 dimensional embedding achieves the state-of-the-
art performance. Also note that our model with the low em-
bedding dimension often outperforms existing models with
the high embedding dimension, which suggests that our loss
allows to learn a more compact yet effective embedding
space. Last, but not least, our loss boosts the convergence
speed greatly as summarized in Figure 1.
4.4. Qualitative Results
To further demonstrate the superiority of our loss, we
present qualitative retrieval results of our model on the four
Query
Top-4 Retrievals
(a)
(b)
(d)
(c)
Figure 4. Qualitative results on the CUB-200-2011 (a), Cars-196
(b), SOP (c) and In-shop (d). For each query image (leftmost), top-
4 retrievals are presented. The results with red boundary are fail-
ure cases, which are however substantially similar to their query
images in terms of appearance.
datasets. As can be seen in Figure 4, intra-class appearance
variation is signiﬁcantly large in these datasets in particular
by pose variation and background clutter in the CUB200-
2011, distinct object colors in the Cars-196, and view-point
changes in the SOP and In-Shop datasets. Even with these
challenges, the embedding network trained with our loss
performs retrieval robustly.
4.5. Impact of Hyperparameters
Batch size: To investigate the effect of batch size on the
performance of our loss, we examine Recall@1 of our loss
while varying batch size on the four benchmark datasets.
The result of the analysis is summarized in Table 5 and 6,
where one can observe that larger batch sizes improve per-
formance since our loss can consider a larger number of ex-
amples and their relations within each batch. On the other
hand, performance is slightly reduced when the batch size
is small since it is difﬁcult to determine the relative hard-
ness in this setting. On the datasets with a large number of
images and classes, i.e., SOP and In-shop, our loss needs
to utilize more examples to fully leverage the relations be-

--- Page 8 ---
Batch size
Recall@1
CUB-200-2011
Cars-196
30
65.9
84.6
60
67.0
86.2
90
68.4
86.2
120
68.5
86.3
150
68.6
86.4
180
69.0
86.2
Table 5. Accuracy of our model in Recall@1 versus batch size on
the CUB-200-2011 and Cars-196.
Batch size
Recall@1
SOP
In-shop
30
76.0
91.3
60
78.0
91.3
90
78.5
91.5
120
78.9
91.7
150
79.1
91.9
300
79.3
92.0
600
79.3
91.7
Table 6. Accuracy of our model in Recall@1 versus batch size on
the SOP and In-shop.
32
64
128
256
512
1024
Embedding Dimension
72.5
75.0
77.5
80.0
82.5
85.0
R@1
Proxy-Anchor
MS
Figure 5. Accuracy in Recall@1 versus embedding dimension on
the Cars-196.
tween data points. Our loss achieves the best performance
when the batch size is equal to or larger than 300.
Embedding dimension: The dimension of embedding vec-
tors is a crucial factor that controls the trade-off between
speed and accuracy in image retrieval systems. We thus
investigate the effect of embedding dimensions on the re-
trieval accuracy in our Proxy-Anchor loss. We test our loss
with embedding dimensions varying from 64 to 1,024 fol-
lowing the experiment in [34], and further examine that with
32 embedding dimension. The result of analysis is quan-
tiﬁed in Figure 5, in which the retrieval performance of
our loss is compared with that of MS loss [34]. The per-
formance of our loss is fairly stable when the dimension
is equal to or larger than 128. Moreover, our loss outper-
forms MS loss in all embedding dimensions, and more im-
portantly, its accuracy does not degrade even with the very
high dimensional embedding unlike MS loss.
4
8
16
32
64
60
70
80
90
0
0.1
0.2
0.3
0.4
64.48
65.56
66.68
68.24
69.09
76.4
77.94
77.73
79.15
79.14
84.11
83.66
83.51
83.58
84.52
86.29
86.1
85.66
85.71
85.13
83.66 86.67
86.26
85.51
86.35
δ
R@1
α
64
87
Figure 6. Accuracy in Recall@1 versus δ and α on the Cars-196.
α and δ of our loss: We also investigate the effect of
the two hyperparameters α and δ of our loss on the Cars-
196 dataset. The results of our analysis are summarized
in Figure 6, in which we examine Recall@1 of Proxy-
Anchor by varying the values of the hyperparameters α ∈
{4, 8, 16, 32, 64} and δ ∈{0, 0.1, 0.2, 0.3, 0.4}. The results
suggest that when α is greater than 16, the accuracy of our
model is high and stable, thus insensitive to the hyperpa-
rameter setting. Our loss outperforms current state-of-the-
art with any α greater than 16. In addition, increasing δ
improves performance although its effect is relatively small
when α is large. Note that our hyperparameter setting re-
ported in Section 4.2 is not the best, although it outperforms
all existing methods on the dataset, as we did not tune the
hyperparameters to optimize the test accuracy.
5. Conclusion
We have proposed a novel metric learning loss that takes
advantages of both proxy- and pair-based losses.
Like
proxy-based losses, it enables fast and reliable convergence,
and like pair-based losses, it can leverage rich data-to-
data relations during training. As a result, our model has
achieved state-of-the-art performance on the four public
benchmark datasets, and at the same time, converged most
quickly with no careful data sampling technique. In the fu-
ture, we will explore extensions of our loss for deep hashing
networks to improve its computational efﬁciency in testing
as well as that in training.
Acknowledgement: This work was supported by IITP grant,
Basic Science Research Program, and R&D program for Ad-
vanced Integrated-intelligence for IDentiﬁcation through the NRF
funded by the Ministry of Science, ICT (No.2019-0-01906
Artiﬁcial Intelligence Graduate School Program (POSTECH),
NRF-2018R1C1B6001223,
NRF-2018R1A5A1060031,
NRF-
2018M3E3A1057306, NRF-2017R1E1A1A01077999).

--- Page 9 ---
References
[1] Nicolas Aziere and Sinisa Todorovic. Ensemble deep mani-
fold similarity learning using hard proxies. In Proc. IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), 2019.
[2] Jane Bromley,
Isabelle Guyon,
Yann Lecun,
Eduard
Sckinger, and Roopak Shah. Signature veriﬁcation using a
”siamese” time delay neural network. In Proc. Neural Infor-
mation Processing Systems (NeurIPS), 1994.
[3] Weihua Chen, Xiaotang Chen, Jianguo Zhang, and Kaiqi
Huang. Beyond triplet loss: A deep quadruplet network for
person re-identiﬁcation. In Proc. IEEE Conference on Com-
puter Vision and Pattern Recognition (CVPR), 2017.
[4] S. Chopra, R. Hadsell, and Y. LeCun. Learning a similarity
metric discriminatively, with application to face veriﬁcation.
In Proc. IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2005.
[5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. ImageNet: a large-scale hierarchical image
database. In Proc. IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 2009.
[6] Thanh-Toan Do, Toan Tran, Ian Reid, Vijay Kumar, Tuan
Hoang, and Gustavo Carneiro. A theoretically sound upper
bound on the triplet loss for improving the efﬁciency of deep
distance metric learning. In Proc. IEEE Conference on Com-
puter Vision and Pattern Recognition (CVPR), 2019.
[7] Weifeng Ge, Weilin Huang, Dengke Dong, and Matthew R.
Scott.
Deep metric learning with hierarchical triplet loss.
In Proc. European Conference on Computer Vision (ECCV),
2018.
[8] Jacob Goldberger, Geoffrey E Hinton, Sam T Roweis, and
Ruslan R Salakhutdinov. Neighbourhood components anal-
ysis.
In Proc. Neural Information Processing Systems
(NeurIPS), 2005.
[9] R. Hadsell, S. Chopra, and Y. LeCun. Dimensionality reduc-
tion by learning an invariant mapping. In Proc. IEEE Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
2006.
[10] Ben Harwood, Vijay Kumar B G, Gustavo Carneiro, Ian
Reid, and Tom Drummond. Smart mining for deep metric
learning. In Proc. IEEE International Conference on Com-
puter Vision (ICCV), 2017.
[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In Proc. IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), June 2016.
[12] Sergey Ioffe and Christian Szegedy. Batch normalization:
Accelerating deep network training by reducing internal co-
variate shift. In Proc. International Conference on Machine
Learning (ICML), 2015.
[13] Pierre Jacob, David Picard, Aymeric Histace, and Edouard
Klein. Metric learning with horde: High-order regularizer
for deep embeddings. In Proc. IEEE International Confer-
ence on Computer Vision (ICCV), 2019.
[14] Sungyeon Kim, Minkyo Seo, Ivan Laptev, Minsu Cho, and
Suha Kwak. Deep metric learning beyond binary supervi-
sion. In Proc. IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 2019.
[15] Wonsik Kim, Bhavya Goyal, Kunal Chawla, Jungmin Lee,
and Keunjoo Kwon. Attention-based ensemble for deep met-
ric learning. In Proc. European Conference on Computer
Vision (ECCV), 2018.
[16] Diederik P. Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. In Proc. International Conference
on Learning Representations (ICLR), 2015.
[17] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei.
3d object representations for ﬁne-grained categorization. In
Proceedings of the IEEE International Conference on Com-
puter Vision Workshops, pages 554–561, 2013.
[18] Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha
Raj, and Le Song. Sphereface: Deep hypersphere embedding
for face recognition. In Proc. IEEE Conference on Computer
Vision and Pattern Recognition (CVPR), 2017.
[19] Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou
Tang. Deepfashion: Powering robust clothes recognition and
retrieval with rich annotations. In Proc. IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2016.
[20] Ilya Loshchilov and Frank Hutter. Decoupled weight decay
regularization. In Proc. International Conference on Learn-
ing Representations (ICLR), 2019.
[21] Yair Movshovitz-Attias, Alexander Toshev, Thomas K Le-
ung, Sergey Ioffe, and Saurabh Singh. No fuss distance met-
ric learning using proxies. In Proc. IEEE International Con-
ference on Computer Vision (ICCV), 2017.
[22] Michael Opitz, Georg Waltner, Horst Possegger, and Horst
Bischof.
Deep metric learning with bier: Boosting inde-
pendent embeddings robustly. IEEE Transactions on Pattern
Analysis and Machine Intelligence (TPAMI), 2018.
[23] Qi Qian, Lei Shang, Baigui Sun, Juhua Hu, Hao Li, and Rong
Jin. Softtriple loss: Deep metric learning without triplet sam-
pling. In Proc. IEEE International Conference on Computer
Vision (ICCV), 2019.
[24] Limeng Qiao, Yemin Shi, Jia Li, Yaowei Wang, Tiejun
Huang, and Yonghong Tian.
Transductive episodic-wise
adaptive metric for few-shot learning. In Proc. IEEE Inter-
national Conference on Computer Vision (ICCV), 2019.
[25] Florian Schroff, Dmitry Kalenichenko, and James Philbin.
FaceNet: A uniﬁed embedding for face recognition and clus-
tering. In Proc. IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 2015.
[26] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypi-
cal networks for few-shot learning. In Advances in Neural
Information Processing Systems, pages 4077–4087, 2017.
[27] Kihyuk Sohn. Improved deep metric learning with multi-
class n-pair loss objective. In Proc. Neural Information Pro-
cessing Systems (NeurIPS), 2016.
[28] Hyun Oh Song, Stefanie Jegelka, Vivek Rathod, and Kevin
Murphy. Deep metric learning via facility location. In Proc.
IEEE Conference on Computer Vision and Pattern Recogni-
tion (CVPR), 2017.
[29] Hyun Oh Song, Yu Xiang, Stefanie Jegelka, and Silvio
Savarese. Deep metric learning via lifted structured feature
embedding. In Proc. IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), 2016.
[30] Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS
Torr, and Timothy M Hospedales. Learning to compare: Re-
lation network for few-shot learning. In Proc. IEEE Confer-
ence on Computer Vision and Pattern Recognition (CVPR),

--- Page 10 ---
2018.
[31] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet,
Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent
Vanhoucke, and Andrew Rabinovich.
Going deeper with
convolutions. In Proc. IEEE Conference on Computer Vi-
sion and Pattern Recognition (CVPR), 2015.
[32] Jiang Wang, Yang Song, T. Leung, C. Rosenberg, Jingbin
Wang, J. Philbin, Bo Chen, and Ying Wu. Learning ﬁne-
grained image similarity with deep ranking. In Proc. IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), 2014.
[33] Xiaolong Wang and Abhinav Gupta. Unsupervised learning
of visual representations using videos. In Proc. IEEE Inter-
national Conference on Computer Vision (ICCV), 2015.
[34] Xun Wang, Xintong Han, Weilin Huang, Dengke Dong, and
Matthew R Scott.
Multi-similarity loss with general pair
weighting for deep metric learning. In Proc. IEEE Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
2019.
[35] Xinshao Wang, Yang Hua, Elyor Kodirov, Guosheng Hu,
Romain Garnier, and Neil M Robertson. Ranked list loss for
deep metric learning. In Proc. IEEE Conference on Com-
puter Vision and Pattern Recognition (CVPR), 2019.
[36] P. Welinder, S. Branson, T. Mita, C. Wah, F. Schroff, S. Be-
longie, and P. Perona. Caltech-UCSD Birds 200. Technical
Report CNS-TR-2010-001, California Institute of Technol-
ogy, 2010.
[37] Chao-Yuan Wu, R. Manmatha, Alexander J. Smola, and
Philipp Krahenbuhl. Sampling matters in deep embedding
learning. In Proc. IEEE International Conference on Com-
puter Vision (ICCV), 2017.
[38] Tong Xiao, Shuang Li, Bochao Wang, Liang Lin, and Xiao-
gang Wang. Joint detection and identiﬁcation feature learn-
ing for person search. In Proc. IEEE Conference on Com-
puter Vision and Pattern Recognition (CVPR), 2017.
[39] Baosheng Yu and Dacheng Tao. Deep metric learning with
tuplet margin loss. In Proc. IEEE International Conference
on Computer Vision (ICCV), 2019.
[40] Yuhui Yuan, Kuiyuan Yang, and Chao Zhang. Hard-aware
deeply cascaded embedding. In Proc. IEEE International
Conference on Computer Vision (ICCV), 2017.
[41] Sergey Zagoruyko and Nikos Komodakis. Learning to com-
pare image patches via convolutional neural networks. In
Proc. IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2015.

--- Page 11 ---
Network
Image Size
CUB-200-2011
Cars-196
224 × 224
R@1
R@2
R@4
R@8
R@1
R@2
R@4
R@8
GoogleNet
63.8
74.4
83.6
90.4
84.3
90.4
94.1
96.7
Inception-BN
68.4
79.2
86.8
91.6
86.1
91.7
95.0
97.3
ResNet-50
69.7
80.0
87.0
92.4
87.7
92.9
95.8
97.9
ResNet-101
70.8
81.0
88.1
93.0
87.9
93.0
96.1
97.9
Inception-BN
256 × 256
71.1
80.4
87.4
92.5
88.3
93.1
95.7
97.5
324 × 324
74.0
82.9
88.9
93.2
91.1
94.9
96.9
98.3
448 × 448
77.3
85.6
91.1
94.2
92.9
96.1
97.7
98.7
Table 7. Comparison both different backbone networks and different sizes of images on the CUB-200-2011 and Cars-196 datasets.
A. Appendix
This appendix presents additional experimental results
omitted from the main paper due to the space limit. Sec-
tion A.1 analyzes the impact of the backbone networks and
the size of input images in our framework. Finally, Sec-
tion A.2 provides t-SNE visualization of the learned embed-
ding space and more qualitative results of image retrieval on
the four benchmark datasets [17, 19, 29, 36].
A.1. Impact of Backbone Network & Image Size
Existing methods in deep metric learning have adopted
various kinds of backbone networks. In this section, we
compare the performance of our loss using popular net-
work architectures as backbone networks on the CUB-200-
2011 [36] and Cars-196 [17]. For all experiments, we use
512 dimensional embedding and ﬁx hyperparameters α and
δ to 32 and 10−1, respectively. In addition to the Incep-
tion with batch normalization (Inception-BN) used in the
main paper, we adopt GoogleNet, ResNet-50, and ResNet-
101 as embedding networks trained with our loss. The re-
sults are summarized in Table 7, where a more powerful
architecture achieves a better score in general. Note that
our method with GoogleNet backbone outperforms existing
models based on the same backbone, except for ABE [15],
an ensemble model, on the Cars-196 dataset. Furthermore,
when using ResNet-50 and ResNet-101 as backbone net-
works, our model outperforms all the previous methods by
large margins.
The main paper showed that the large image size con-
tributed signiﬁcantly to performance improvement, and we
further investigate the performance at larger image size set-
tings. We evaluate our method with Inception-BN back-
bone while varying the sizes of input images: {224 ×
224, 256 × 256, 324 × 324, 448 × 448}. Table 7 also shows
that the accuracy improves consistently as the sizes of the
input images increase. Even our model with images size of
448 × 448 has 8.9% improvement over the default images
size (224 × 224) in Recall@1. Increasing the image size
decreases the allowable batch size, but with enough GPU
memory, using a larger image size is the most effective way
to improve performance than using a powerful architecture.
A.2. Additional Qualitative Results
More qualitative examples for image retrieval on the
CUB-200-2011 and Cars-196 are shown in Figure 7 and
Figure 8, respectively.
The results of our model are
compared with those of model trained with Proxy-NCA
loss [21] using the same backbone network. The overall
results indicate that our model learned a higher quality em-
bedding space than the baseline. In the examples in the
3rd and 4th rows of Figure 7, both models retrieved birds
with a similar appearance to the query, but only our model
produces accurate results. Also, the example in the ﬁrst
row of Figure 8 shows successful retrievals despite different
view-point changes and colors. Figures 9 and 10 compare
the qualitative results of the SOP and In-shop datasets. As
shown in the 2nd, 4th, and 5th rows of Figure 9, our model
successfully retrieved the same object even with extreme
view-point changes. Also, in the 4th row of Figure 10, the
baseline is confused with a short dress of similar pattern,
whereas our model retrieves the long dress exactly.
Finally, Figures 11, 12, 13 and 14 show t-SNE visual-
izations of the embedding spaces learned by our loss on the
test splits of the four benchmark datasets. These 2D vi-
sualizations are generated by mapping each image onto a
location of a square grid using Jonker-Volgenant algorithm.
The results demonstrate that all data points in the embed-
ding space have relevant nearest neighbors, which suggest
that our model learns a semantic similarity that can be gen-
eralized even in the test set.

--- Page 12 ---
Query
Proxy-NCA
Proxy-Anchor (Ours)
Figure 7. Qualitative results on the CUB-200-2011 comparing with Proxy-NCA loss. For each query image (leftmost), top-4 retrievals are
presented. The result with red boundary is a failure case.
Query
Proxy-Anchor (Ours)
Proxy-NCA
Figure 8. Qualitative results on the Cars-196 comparing with Proxy-NCA loss. For each query image (leftmost), top-4 retrievals are
presented. The result with red boundary is a failure case.

--- Page 13 ---
Query
Proxy-Anchor (Ours)
Proxy-NCA
Figure 9. Qualitative results on the SOP comparing with Proxy-NCA loss. For each query image (leftmost), top-4 retrievals are presented.
The result with red boundary is a failure case.
Query
Proxy-Anchor (Ours)
Proxy-NCA
Figure 10. Qualitative results on the In-shop comparing with Proxy-NCA loss. For each query image (leftmost), top-4 retrievals are
presented. The result with red boundary is a failure case.

--- Page 14 ---
Figure 11. t-SNE visualization of our embedding space learned on the test split of CUB-200-2011 dataset in a grid.

--- Page 15 ---
Figure 12. t-SNE visualization of our embedding space learned on the test split of Cars-196 dataset in a grid.

--- Page 16 ---
Figure 13. t-SNE visualization of our embedding space learned on the test split of SOP dataset in a grid.

--- Page 17 ---
Figure 14. t-SNE visualization of our embedding space learned on the test split of In-shop dataset in a grid.
