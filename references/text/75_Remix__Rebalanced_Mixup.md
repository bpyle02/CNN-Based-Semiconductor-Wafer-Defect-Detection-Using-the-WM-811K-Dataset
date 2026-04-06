# Remix: Rebalanced Mixup

**Authors**: Chou, Chen, Lee
**Year**: 2020
**arXiv**: 2007.03943
**Topic**: augmentation
**Relevance**: Combines Mixup with class-balanced resampling

---


--- Page 1 ---
Remix: Rebalanced Mixup
Hsin-Ping Chou1, Shih-Chieh Chang1, Jia-Yu Pan2, Wei Wei2, Da-Cheng Juan2
1Department of Computer Science, National Tsing-Hua University, Hsinchu, Taiwan
2Google Research, Mountain View, CA, USA
Abstract. Deep image classiﬁers often perform poorly when training
data are heavily class-imbalanced. In this work, we propose a new reg-
ularization technique, Remix, that relaxes Mixup’s formulation and en-
ables the mixing factors of features and labels to be disentangled. Specif-
ically, when mixing two samples, while features are mixed in the same
fashion as Mixup, Remix assigns the label in favor of the minority class
by providing a disproportionately higher weight to the minority class.
By doing so, the classiﬁer learns to push the decision boundaries to-
wards the majority classes and balance the generalization error between
majority and minority classes. We have studied the state-of-the art reg-
ularization techniques such as Mixup, Manifold Mixup and CutMix un-
der class-imbalanced regime, and shown that the proposed Remix sig-
niﬁcantly outperforms these state-of-the-arts and several re-weighting
and re-sampling techniques, on the imbalanced datasets constructed by
CIFAR-10, CIFAR-100, and CINIC-10. We have also evaluated Remix on
a real-world large-scale imbalanced dataset, iNaturalist 2018. The exper-
imental results conﬁrmed that Remix provides consistent and signiﬁcant
improvements over previous methods.
Keywords: imbalanced data, Mixup, regularization, image classiﬁca-
tion
1
Introduction
Deep neural networks have made notable breakthroughs in many ﬁelds such as
computer vision [35,27,20], natural language processing [18,19,9] and reinforce-
ment learning [28]. Aside from delicately-designed algorithms and architectures,
training data is one of the critical factors that aﬀects the performance of neural
models. In general, training data needs to be carefully labeled and designed in a
way to achieve a balanced distribution among classes. However, a common prob-
lem in practice is that certain classes may have a signiﬁcantly larger presence in
the training set than other classes, making the distribution skewed. Such a sce-
nario is referred to as data imbalance. Data imbalance may bias neural networks
toward the majority classes when making inferences.
Many previous works have been proposed to mitigate this issue for training
neural network models. Most of the existing works can be split into two cat-
egories: re-weighting and re-sampling. Re-weighting focuses on tuning the cost
(or loss) for diﬀerent classes. Re-sampling focuses on reconstructing a balanced
arXiv:2007.03943v3  [cs.CV]  19 Nov 2020

--- Page 2 ---
2
Chou et al.
Fig. 1: Assume the label “butterﬂy” belongs to the minority class and the label
“yellow plant” belongs to the majority class. All linear combinations of the two
images are on the dashed line. For a mixed image that is 70% “butterﬂy“ and
30% “yellow plant“ like the middle one, Mixup will assign the label to be 70%
“butterﬂy“ and 30% “yellow plant“. However, Remix would assign a label that
is in favor of the minority class, e.g., 100% “butterﬂy“. For more details on how
exactly Remix assigns labels for the mix images, please refer to Section 3 and 4.
dataset by either over-sampling the minority classes or under-sampling the ma-
jority classes. Both re-weighting and re-sampling have some disadvantages when
used for deep neural networks. Re-weighting tends to make optimization diﬃcult
under extreme imbalance. Furthermore, it has been shown that re-weighting is
not eﬀective when no regularization is applied [4]. Re-sampling is very useful
in general especially for over-sampling techniques like SMOTE [2]. However, it
is hard to integrate into modern deep neural networks where feature extraction
and classiﬁcation are performed in an end-to-end fashion while over-sampling
is done subsequent to feature extraction. This issue is particularly diﬃcult to
overcome when training with large-scale datasets.
In order to come up with a solution that is convenient to incorporate for
large-scale datasets, we focus on regularization techniques which normally intro-
duce little extra costs. Despite the recent success in regularizations [10,37,32,36],
these advanced techniques are often designed for balanced data and evaluated
on commonly used datasets (e.g., CIFAR, ImageNet ILSVRC 2012) while real-
world datasets tend to have a long-tailed distribution of labels [14,24]. As a
result, our motivation is to make the commonly used regularization techniques
such as Mixup [37], Manifold Mixup [32] and CutMix [36] perform better in the
real-world imbalanced scenario.
The key idea of Mixup is to train a neural model with mixed samples virtually
created via the convex combinations of pairs of features and labels. Speciﬁcally,
Mixup assumes that linear interpolations of feature vectors should come with
linear interpolations of the associated labels using the same mixing factor λ.
We observe that this assumption works poorly when the class distribution is
imbalanced. In this work, we propose Remix that relaxes the constraint of using
the same mixing factor and allows the mixing factors of the features and labels
to be diﬀerent when constructing the virtual mixed samples. Fig. 1 illustrates
the diﬀerence between Remix and Mixup. Note that the mixing factor of labels is

--- Page 3 ---
Rebalanced Mixup
3
selected in a way to provide a better trade-oﬀbetween the majority and minority
classes.
This work brings the following contributions. (a) We propose Remix, a com-
putationally cheap regularization technique to improve the model generalization
when training with imbalanced data. (b) The proposed Remix can be applied to
all Mixup-based regularizations and can be easily used with existing solutions
against data imbalance to achieve better performance. (c) We evaluate Remix
extensively on various imbalanced settings and conﬁrm that Remix is general
and eﬀective for diﬀerent scenarios.
2
Related Works
Re-Weighting Re-weighting (cost-sensitive learning) focuses on tuning cost or
loss to redeﬁne the importance of each class or sample [6,34,15,3]. In particular,
early works [11,4] study on how re-weighting aﬀects the decision boundary in
the binary classiﬁcation case. The naive practice of dealing with an imbalanced
dataset is weighted by the inverse class frequency or by the inverse square root
of class frequency. Motivated by the observation that each sample might cover a
small neighboring region rather than just a single point, Cui et al. [7] introduced
the concept of “eﬀective number” of a class, which takes the class overlapping
into consideration for re-weighting. In general, re-weighting methods perform
poorly when the classes are extremely imbalanced, where the performance of
majority classes is signiﬁcantly compromised. As an example, Cao et al. [5] show
that re-weighting can perform even worse than vanilla training in the extreme
setting.
Re-Sampling Re-sampling methods can be summarized into two categories:
over-sampling the minority classes [25,16] and under-sampling the majority classes.
Both of these methods have drawbacks. Over-sampling the minority classes may
cause over-ﬁtting to these samples, and under-sampling majority samples dis-
cards data and information, which is wasteful when the imbalance is extreme. For
over-sampling, instead of sampling from the same group of data, augmentation
techniques are applied to create synthetic samples. Classical methods include
SMOTE [2] and ADASYN [12]. The key idea of such methods is to ﬁnd the
k nearest neighbors of a given sample and use the interpolation to create new
samples.
Alternative Training Objectives Novel objectives are also proposed to ﬁght
against class imbalance. For example, Focal Loss [21] identiﬁes the class imbal-
ance in object detection task, and the authors proposed to add a modulating
term to cross entropy in order to focus on hard negative examples. Although Fo-
cal Loss brought signiﬁcant improvements in object detection tasks, this method
is known to be less eﬀective for large-scale imbalanced image classiﬁcation [5].
Another important work for designing an alternative objective for class imbal-
ance is the Label-Distribution-Aware Margin Loss [5], which is motivated by the

--- Page 4 ---
4
Chou et al.
recent progress on various margin-based losses [23,22]. Cao et al. [5] derived a
theoretical formulation to support their proposed method that encourages mi-
nority classes to have larger margins and encourage majority classes to have
smaller margins.
Other Types Two competitive state-of-the arts [38,17] focus on the represen-
tation learning and classiﬁer learning of a CNN. Kang et al. [17] found that it
is possible to achieve strong long-tailed recognition ability by adjusting only the
classiﬁer. Zhou et al. [38] proposed a Bilateral-Branch Network to take care both
representation and classiﬁer learning.
Mixup-based Regularization Mixup [37] is a regularization technique that
proposed to train with interpolations of samples. Despite its simplicity, it works
surprisingly well for improving generalization of deep neural networks. Mixup
inspires several follow-up works like Manifold Mixup [32], RICAP [29] and Cut-
Mix [36] that have shown signiﬁcant improvement over Mixup. Mixup also shed
lights upon other learning tasks such as semi-supervised learning [33,1], adver-
sarial defense [31] and neural network calibration [30].
3
Preliminaries
3.1
Mixup
Mixup [37] was proposed as a regularization technique for improving the gen-
eralization of deep neural networks. The general idea of Mixup is to generate
mixed sample ˜xMU and ˜y by linearly combining an arbitrary sample pair (xi,yi;
xj, yj) in a dataset D. In Eq. 1, this mixing process is done by using a mixing
factor λ which is sampled from the beta distribution.
˜xMU = λxi + (1 −λ)xj
(1)
˜y = λyi + (1 −λ)yj
(2)
3.2
Manifold Mixup
Instead of mixing samples in the feature space, Manifold Mixup [37] performs
the linear combination in the embedding space. This is achieved by randomly
performing the linear combination at an eligible layer k and conducting Mixup
on (gk(xi), yi) and (gk(xj), yj) where gk(xi) denotes a forward pass until layer
k. As a result, the mixed representations which we denoted as ˜xMM can be
thought of ”mixed samples” that is forwarded from layer k to the output layer.
Conducting the interpolations in deeper hidden layers which captures higher-
level information provides more training signals than Mixup and thus further
improve the generalization.

--- Page 5 ---
Rebalanced Mixup
5
˜xMM = λgk(xi) + (1 −λ)gk(xj)
(3)
˜y = λyi + (1 −λ)yj
(4)
3.3
CutMix
Inspired by Mixup and Cutout [10], rather than mixing samples on the entire
input feature space like Mixup does, CutMix [36] works by masking out a patch
of it B when generating the synthesized samples where patch B is a masking
box with width rw = W
√
1 −λ and height rh = H
√
1 −λ randomly sampled
across the image. Here W and H are the original width and height of the image,
respectively. The generated block makes sure that the proportion of the image
being masked out is equal to
rwrh
W H = 1 −λ. A image level mask M is then
generated based on B with elements equal to 0 when it is blocked by B and 1
otherwise. CutMix is deﬁned in a way similar to Mixup and Manifold Mixup in
Eq. 6. Here ⊙is element-wise multiplication and we denote M to be generated
by a random process that involves W, H, and λ using a mapping f(·)
˜xCM = M ⊙xi + (1 −M) ⊙xj
(5)
˜y = λyi + (1 −λ)yj
(6)
M ∼f(·|λ, W, H)
(7)
4
Remix
We observe that both Mixup, Manifold Mixup, and CutMix use the same mixing
factor λ for mixing samples in both feature space and label space. We argue
that it does not make sense under the imbalanced data regime and propose to
disentangle the mixing factors. After relaxing the mixing factors, we are able to
assign a higher weight to the minority class so that we can create labels that are
in favor to the minority class. Before we further introduce our method. We ﬁrst
show the formulation of Remix as below:
˜xRM = λxxi + (1 −λx)xj
(8)
˜yRM = λyyi + (1 −λy)yj
(9)
The above formulation is in a more general form compares to other Mixup-
based methods. In fact, ˜xRM can be generated based on Mixup, Manifold Mixup,
and CutMix according to Eq. 1, Eq. 3 and Eq. 5 respectively. Here we use Mixup
for the above formulation as an example. Note that Eq. 9 relaxes the mixing
factors which are otherwise tightly coupled in the original Mixup’s formulation.
Mixup, Manifold Mixup, and Cutmix are a special case when λy = λx. Again,

--- Page 6 ---
6
Chou et al.
(a) τ=0
(b) 0 < τ < 1
(c) τ=1
Fig. 2: A simple illustration of how the hyper-parameter τ aﬀects the boundary.
Blue and red dots are majority and minority samples in the feature space. The
dashed line represents all the possible mixed samples and the solid black line
represents the decision boundary. When τ is set to 0, mixed samples are linearly
mixed and labelled as the original Mixup, Manifold Mixup, and Cutmix algo-
rithms. But when τ is set to a value larger than zero, then part of the mixed
samples on the red dashed line will be labelled as the minority class. In the most
extreme case for ﬁghting against data imbalance, τ is set to 1 where all mixed
samples are labelled as the minority class.
λx is sampled from the beta distribution and we deﬁne the exact form of λy as
in Eq.10.
λy =





0,
ni/nj ≥κ and λ < τ;
1,
ni/nj ≤1/κ and 1 −λ < τ;
λ,
otherwise
(10)
Here ni and nj denote the number of samples in the corresponding classiﬁca-
tion class from sample i and sample j. For example, if yi = 1 and yj = 10, ni and
nj would be the number of samples for class 1 and 10, which are the class that
these two samples represent. κ and τ are two hyper-parameters in our method.
To understand what Eq. 10 is about, we ﬁrst deﬁne the κ-majority below.
Deﬁnition 1. κ-Majority. A sample (xi, yi), is considered to be κ-majority than
sample (xj, yj), if ni/nj ≥κ where ni and nj represent the number of samples
that belong to class yi and class yj, respectively.
The general idea in Eq. 10 shows when exactly Remix assigns the synthesized
labels in favor to the minority class. When xi is κ-majority to xj and the other
condition is met, λy is set to 0 which makes the the synthesized labels 100%
contributed by the minority class yj. Conversely, when xj is κ-majority to xi
along with other conditions, λy to 1 which makes the the synthesized labels 100%
contributed by the minority class yi.
The reason behind this choice of making the synthesized samples to be labeled
as the minority class is to move the decision boundary towards the majority class.

--- Page 7 ---
Rebalanced Mixup
7
This is aligned with the consensus in the community of imbalanced classiﬁcation
problem. In [5], the authors gave a rather theoretical analysis illustrating that
how exactly the decision boundary should be pushed towards the majority classes
by using a margin loss. Because pushing the decision boundary towards too much
may hurt the performance of the majority class. As a result, we don’t want
the synthesized labels to be always pointing to the minority class whenever
mixing a majority class sample and a minority class sample. To achieve that we
have introduced another condition in Eq. 10 controlled by parameter τ that is
conditioned on λ. In both conditions, extreme cases will be rejected and λy will be
set to λ when λx is smaller than τ. The geometric interpretation of this condition
can be visualized in Fig. 5. Here we see that when τ is set to 0, our approach will
degenerate to the base method, which can be Mixup, Manifold Mixup, or Cutmix
depending on the choice of design. When τ is set to a value that is larger than 0,
synthesized samples ˜xRM that are close to the minority classes will be labelled
as the minority class, thus beneﬁting the imbalanced classiﬁcation problems. To
summarize, τ controls the extent that the minority samples would dominate the
label of the synthesized samples. When the conditions are not met, or in other
words, when there is no trade-oﬀto be made, we will just use the base method
to generate the labels. This can be illustrated in the last condition in Eq. 10,
when none of i or j can claim to be κ-majority over each other and in this case
λy is set to λx.
Attentive readers may realize that using the same τ for various pairs of a
majority class and a minority class implies that we want to enforce the same
trade-oﬀfor those pairs. One may wonder why not introduce τij for each pair
of classes? This is because the trade-oﬀfor multi-class problem is intractable to
ﬁnd. Hence, instead of deﬁning τij for each pair of classes, we use a single τ for
all pairs of classes. Despite its simplicity, using a single τ is suﬃcient to achieve
a better trade-oﬀ.
Remix might look similar to SMOTE [2] and ADASYN [12] at ﬁrst glance,
but they are very diﬀerent in two perspectives. First, the interpolation of Remix
can be conducted with any two given samples while SMOTE and ADASYN
rely on the knowledge of a sample’s same-class neighbors before conducting
the interpolation. Moreover, rather than only focusing on creating new data
points, Remix also pays attention to labelling the mixed data which is not an
issue to SMOTE since the interpolation is conducted between same-class data
points. Secondly, Remix follows Mixup to train the classiﬁer only on the mixed
samples while SMOTE and ADASYN train the classiﬁer on both original data
and synthetic data.
To give a straightforward explanation of why Remix would beneﬁt learning
with imbalance datasets, consider a mixed example ˜x between a majority class
and a minority class, the mixed sample includes features of both classes yet
we mark it as the minority class more. This force the neural network model
to learn that when there are features of a majority class and a minority class
appearing in the sample, it should more likely to consider it as the minority
class. This means that the classiﬁer is being less strict to the minority class.

--- Page 8 ---
8
Chou et al.
Algorithm 1 Remix
Require: Dataset D = {(xi, yi)}n
i=1. A model with parameter θ
1: Initialize the model parameters θ randomly
2: while θ is not converged do
3:
{(xi, yi), (xj, yj)}M
m=1 ←SamplePairs(D, M)
4:
λx ∼Beta(α, α)
5:
for m = 1 to M do
6:
˜xRM ←RemixImage(xi, xj, λx) according to Eq.8
7:
λy ←LabelMixingFactor(λx, ni, nk, τ, κ) according to Eq.10
8:
˜yRM ←RemixLabel(yi, yj, λy) according to Eq.9
9:
end for
10:
L(θ) ←
1
M
P
(˜x,˜y) L((˜x, ˜y); θ)
11:
θ ←θ −δ∇θL(θ)
12: end while
(a) Long-tail: ρ = 100
(b) Step: ρ = 10, µ = 0.5
(c) iNaturalist 2018
Fig. 3: Histograms of three imbalanced class distributions. (a) and (b) are syn-
thesized from long-tail and step imbalance, respectively. (c) represents the class
distribution of iNaturalist 2018 dataset.
Please see Section 4.7 for qualitative analysis. Note that Remix method is a
relaxation technique, and thus may be integrated with other techniques. In the
following experiments, besides showing the results of the pure Remix method,
we also show that the Remix method can work together with the re-weighting or
the re-sampling techniques. Algorithm 1 shows the pseudo-code of the proposed
Remix method.
5
Experiments
5.1
Datasets
We compare the proposed Remix with state-of-the-art methods ﬁghting against
class imbalance on the following datasets: (a) artiﬁcially created imbalanced
datasets using CIFAR-10, CIFAR-100 and CINIC-10 datasets, and (b) iNatural-
ist 2018, a real-world and large-scale imbalanced dataset.
Imbalanced CIFAR The original CIFAR-10 and CIFAR-100 datasets both
contain 50,000 training images and 10,000 validation images of size 32×32, with

--- Page 9 ---
Rebalanced Mixup
9
10 and 100 classes, respectively. We follow [5] to construct class-imbalanced
datasets from the CIFAR-10 and CIFAR-100 datasets with two common im-
balance types “long-tailed imbalance” and “step imbalance”. The validation set
is kept balanced as original. Fig. 3(a)(b) illustrates the two imbalance types.
For a dataset with long-tailed imbalance, the class sizes (number of samples in
the class) of the dataset follow an exponential decay. For a dataset with step
imbalance, a parameter µ is used to denotes the fraction of minority classes. µ
is set to 0.5 [5] for all of the experiments. The parameter ρ of the constructed
datasets denotes the imbalance ratio between the number of samples from the
most frequent and the least frequent classes, i.e., ρ = maxi{ni}/ minj{nj}.
Imbalanced CINIC The CINIC-10 dataset [8] is compiled by combining CIFAR-
10 images with images downsampled from the ImageNet database. It contains
270,000 images and it splits train, validation, and test set equally. We only use
the oﬃcial split of training and validation. Using the CINIC-10 dataset helps
us compare diﬀerent methods better because it has 9000 training data per class
which allows us to conduct extensive experiments with various imbalance ratios
while making sure each class still preserves a certain number of data. This helps
us focus more on the imbalance between classes rather than solving a few-shot
classiﬁcation problem for the minority classes.
iNaturalist 2018 The iNaturalist species classiﬁcation dataset [13] is a real-
world large-scale imbalanced dataset which has 437,513 training images of 8,142
classes. The dataset features many visually similar species which are extremely
diﬃcult to accurately classify without expert knowledge. We adopt the oﬃcial
training and validation splits for our experiments where the training datasets
have a long-tailed label distribution and the validation set is designed to have a
balanced distribution.
CIFAR and CINIC-10 For fair comparisons, we ported the oﬃcial code of [5]
into our codebase. We follow [7,5] use ResNet-32 for all CIFAR experiments and
we use ResNet-18 for all CINIC experiments. We train 300 epochs and decay
the learning rate 0.01 at 150, 225 epoch. We use stochastic gradient descent
(SGD) with momentum 0.9 and weight decay 0.0002. For non-LDAM methods,
We train it for 300 epochs with mini-batch size 128. We decay our learning rate
by 0.1 at 150, 250 epoch. All CIFAR and CINIC experiment results are mean
over 5 runs. Standard data augmentation is applied, which is the combination
of random crop, random horizontal ﬂip and normalization. If DRW or DRS are
used for the training, we use re-weighting or re-sampling at the second learning
rate decay. We set τ = 0.5 and κ = 3 for all experiments. The choice of these two
parameters can be found with simple grid search. Despite that one might want to
use carefully-tuned parameters for diﬀerent imbalance scenarios, we empirically
found that setting τ = 0.5 and κ = 3 is able to provide consistent improvements
over the previous state-of-the-arts.

--- Page 10 ---
10
Chou et al.
Table 1: Top-1 accuracy of on imbalanced CIFAR-10 and CIFAR-100.
† denotes the results from the original paper.
Dataset
Imbalanced CIFAR-10
Imbalanced CIFAR-100
Imbalance Type
long-tailed
step
long-tailed
step
Imbalance Ratio
100
10
100
10
100
10
100
10
ERM
71.86 86.22 64.17 84.02 40.12 56.77 40.13 54.74
Focal [21] †
70.18 86.66 63.91 83.64 38.41 55.78 38.57 53.27
RW Focal [7] †
74.57
87.1
60.27 83.46 36.02 57.99 19.75 50.02
DRS [5] †
74.50 86.72 72.03 85.17 40.33 57.26 41.35 56.79
DRW [5] †
74.86 86.88 71.60 85.51 40.66 57.32 41.14 57.22
LDAM [5] †
73.35 86.96 66.58 85.00 39.60 56.91 39.58 56.27
LDAM-DRW [5]
76.57
86.7
75.94 86.52 42.64 57.18 45.40 57.09
Mixup [37]
73.09 88.00 65.80 85.20 40.83 58.37 39.64 54.46
Remix
75.36 88.15 68.98 86.34 41.94 59.36 39.96 57.06
BBN [38] †
79.82 88.32
–
–
42.56 59.12
–
–
Remix-LDAM-DRW 79.33 86.78 77.81 86.46 45.02 59.47 45.32 56.59
Remix-LDAM-DRS 79.45 87.16 78.00 86.91 45.66 59.21 45.74 56.19
Remix-RS
76.23 87.70 67.28 86.63 41.13 58.62 39.74 56.09
Remix-RW
75.1
87.91 68.74 86.38 33.51 57.65 17.42 54.45
Remix-DRS
79.53 88.85 77.46 88.16 46.53 60.52 47.25 60.76
Remix-DRW
79.76 89.02 77.86 88.34 46.77 61.23 46.78 60.44
iNaturalist 2018 We use ResNet-50 as the backbone network across all exper-
iments for iNaturalist 2018. Each image is ﬁrst resized to 256 × 256, and then a
224 × 224 crop is randomly sampled from an image or its horizontal ﬂip. Then
color jittering and lighting are applied. Follow [7,5], we train the network for 90
epochs with an initial learning rate of 0.1 and mini-batch size 256. We anneal
the learning rate at epoch 30 and 60. For the longer training schedule, we train
the network for 200 epochs with an initial learning rate of 0.1 and anneal the
learning rate at epoch 75 and 150. Using the longer training schedule is necessary
for Mixup-based regularizations to converge [37,36]. We set τ = 0.5 and κ = 3.
Baseline Methods for Comparison We compare our methods with vanilla
training, state-of-the-art techniques and their combinations. (1) Empirical risk
minimization (ERM): Standard training with no anti-imbalance techniques in-
volved. (2) Focal: Use focal loss instead of cross entropy. (3) Re-weighting (RW):
Re-weight each sample by the eﬀective number which is deﬁned as En = (1 −
βn)/(1 −β), where β = (N −1)/N. (4) Re-sampling (RS): Each example is
sampled with probability proportional to the inverse of eﬀective number. (5)
Deferred re-weighting and deferred re-sampling (DRW, DRS): A deferred train-
ing procedure which ﬁrst trains using ERM before annealing the learning rate,
and then deploys RW or RS (6) LDAM: A label-distribution-aware margin loss

--- Page 11 ---
Rebalanced Mixup
11
Table 2: Top-1 accuracy on imbalanced CINIC-10 using ResNet-18.
Imbalance Type
long-tailed
step
Imbalance Ratio
200
100
50
10
200
100
50
10
ERM
56.16 61.82 72.34 77.06 51.64 55.64 68.35 74.16
RS [7]
53.71 59.11 71.28 75.99 50.65 53.82 65.54 71.33
RW [7]
54.84 60.87 72.62 76.88 50.47 55.91 69.24 74.81
DRW [5]
59.66 63.14 73.56 77.88 54.41 57.87 68.76 72.85
DRS [5]
57.98 62.16 73.14 77.39 52.67 57.41 69.52 75.89
Mixup [37]
57.93 62.06 74.55 79.28 53.47 56.91 69.74 75.59
Remix
58.86 63.21 75.07 79.02 54.22 57.57 70.21 76.37
LDAM-DRW [5] 60.80 65.51 74.94 77.90 54.93 61.17 72.26 76.12
Remix-DRS
61.64 65.95 75.34 79.17 60.12 66.53 75.47 79.86
Remix-DRW
62.95 67.76 75.49 79.43 62.82 67.56 76.55 79.36
which considers the trade-oﬀbetween the class margins. (7) Mixup: Each batch
of training samples are generated according to [37] (8) BBN [38] and LWS [17]:
Two state-of-the-arts methods. We directly copy the results from the original
paper.
5.2
Results on Imbalanced CIFAR and CINIC
In Table 1 and Table 2, we compare the previous state-of-the-arts with the pure
Remix method and a variety of the Remix-integrated methods. Speciﬁcally, we
ﬁrst integrate Remix with basic re-weighting and re-sampling techniques with
respect to the eﬀective number [7], and we use the deferred version of them [5].
We also experiment with a variety that integrate our method with the LDAM
loss [5]. We observe that Remix works particularly well with re-weighting and re-
sampling. Among all the methods that we experiment with, the best performance
was achieved by the method that integrates Remix with either the deferred re-
sampling method (Remix-DRS) or the deferred re-weighting method (Remix-
DRW).
Regarding the reason why Remix-DRS and Remix-DRW achieve the best
performance, we provide an intuitive explanation as the following: We believe
that the improvement of our Remix method comes from the imbalance-aware
mixing equation for labels (Eq.10), particularly when λy is set to either 0 or 1.
The more often the conditions are satisﬁed to make λy be either 0 or 1, the more
opportunity is given to the learning algorithm to adjust for the data imbalance.
To increase the chance that those conditions are satisﬁed, we need to have more
pairs of training samples where each pair is consisted of one sample of a majority
class and one sample of a minority class. Since, by the deﬁnition of a minority
class, there are not many samples from a minority class, the chance of forming
such pairs of training data may not be very high.

--- Page 12 ---
12
Chou et al.
Table 3: Validation errors on iNatu-
ralist 2018 using ResNet-50.
Loss
Schedule Top-1 Top-5
ERM
SGD
40.19 18.99
RW Focal
SGD
38.88 18.97
Mixup
SGD
39.69 17.88
Remix
SGD
38.69 17.70
LDAM
DRW
32.89 15.20
BBN [38] †
—
30.38
—
LWS (200 epoch) [17] †
—
30.5
—
LDAM (200 epoch)
DRW
31.42 14.68
Remix (200 epoch)
DRS
29.26 12.55
Remix (200 epoch)
DRW
29.51 12.73
Table 4: Top-1 accuracy of ResNet-
18 trained with imbalanced CIFAR-
10 with imbalance ratio ρ=100
Imbalance Type long-tailed step
Mixup-DRW
81.09
76.13
Remix-DRW
81.60
79.35
Mixup-DRS
80.40
75.58
Remix-DRS
81.11
79.35
When the re-sampling method is used with Remix, the re-sampling method
increases the probability of having data of minority classes in the training batches.
With more samples from minority classes in the training batches, the chance of
forming sample pairs that satisﬁed the conditions is increased, thus allows Remix
to provide better trade-oﬀon the data imbalance among classes.
Likewise, although using Remix with re-weighting does not directly increase
the probability of pairs that satisfy the conditions, the weights assigned to the
minority classes will amplify the eﬀect when a case of majority-minority sample
pair is encountered, and thus, will also guide the classiﬁer to have better trade-oﬀ
on the data imbalance.
On the other hand, using the LDAM loss doesn’t further improve the perfor-
mance. We suspect that the trade-oﬀLDAM intends to make is competing with
the trade-oﬀRemix intends to make. Therefore, for the rest of the experiments,
we will focus more on the methods where DRS and DRW are integrated with
Remix.
5.3
Results on iNaturalist 2018
In Table 3, we present the results in similar order as Table 2. The results show
the same trends to the results on CIFAR and CINIC. Again, our proposed
method, Remix, outperforms the original Mixup. Also, state-of-the-art results
are achieved when Remix is used with re-weighting or re-sampling techniques.
The model’s performance is signiﬁcantly better than the previous state-of-the-
arts and outperforms the baseline (ERM) by a large margin. The improvement
is signiﬁcant and more importantly, the training cost remains almost the same
in terms of training time.

--- Page 13 ---
Rebalanced Mixup
13
5.4
Ablation Studies
When it comes to addressing the issue of data imbalance, one common approach
is to simply combine the re-sampling or re-weighting method with Mixup. In
Table 4, we show the comparison of Mixup and Remix when they are integrated
with the re-sampling technique (DRS) or the re-weighting technique (DRW).
Note that our methods still outperform Mixup-based methods. The results in
Table 4 imply that, while Remix can be considered as over-sampling the minority
classes in the label space, the performance gain from Remix does not completely
overlap with the gains from the re-weighting or re-sampling techniques.
We also observe that the improvement is more signiﬁcant on datasets with
step imbalance, than it is on datasets with long-tailed imbalance. In the long-
tailed setting, the distribution of the class sizes makes it less likely to have pairs
of data samples that satisfy the conditions of Eq.10 to make λy either 0 or 1. On
the other hand, the conditions of Eq.10 are relatively more likely to be satisﬁed,
on a dataset with the step imbalance. In other words, there is less room for
Remix to unleash its power on a dataset with long-tailed setting.
The proposed Remix method is general and can be applied with other Mixup-
based regularizations, such as Manifold Mixup and CutMix. In Table 5, we show
that, the performance of Manifold Mixup and CutMix increases when they em-
ploy the Remix regularization, signiﬁcantly outperforming the performance of
the vanilla Manifold Mixup or CutMix.
Moreover, we also observe that when the imbalance ratio is not very extreme
(ρ=10), using Mixup or Remix doesn’t produce much diﬀerence. However, when
the imbalance is extreme (e.g., ρ=100), employing our proposed method is sig-
niﬁcantly better than the vanilla version of Manifold Mixup or CutMix.
Table 5: Top-1 accuracy of ResNet-32 on imbalanced CIFAR-10 and CIFAR-100.
Dataset
Imbalanced CIFAR-10
Imbalanced CIFAR-100
Imbalance Type
long-tailed
step
long-tailed
step
Imbalance Ratio
100
10
100
10
100
10
100
10
ERM
71.86 86.22 64.17 84.02 40.12 56.77
40.13
54.74
Mixup [37]
73.09 88.00 65.80 85.20 40.83 58.37
39.64
54.46
Remix
75.36 88.15 68.98 86.34 41.94 59.36 39.96 57.06
Manifold Mixup [32] 73.47 87.78 66.13 85.22 41.19 58.55
39.52
53.72
Remix-MM
77.07 88.70 69.78 87.39 44.12 60.76 40.22 58.01
CutMix [36]
75.04 88.30 67.97 86.35 41.86 59.47
40.23
56.59
Remix-CM
76.59 87.96 69.61 87.59 43.55 60.15 40.336 57.78

--- Page 14 ---
14
Chou et al.
Fig. 4: A visualization of the decision boundary of the models learned by diﬀerent
methods. Remix creates tighter margin for the majority class, and compensates
the eﬀect from the data imbalance.
5.5
Qualitative Analysis
To further demonstrate the eﬀect of Remix we present the results of Remix
on the commonly used datasets, namely, “two blobs”, “two moons” , and “two
circles” datasets from scikit-learn [26]. In Fig.4, the original balanced datasets
are shown at the rightmost column. The three columns at the left show the
created imbalanced datasets with the imbalance ratio ρ=10. The majority class
is plotted in black and the minority class is plotted in white. The results in Fig.4
show that Remix creates tighter margin for the majority class. In all three cases,
we observe that even though our regularization sacriﬁces some training accuracy
for the majority class (some black dots are misclassiﬁed), but it actually provides
a better decision boundary for the minority class.
6
Conclusions and Future Work
In this paper, we redesigned the Mixup-based regularizations for imbalanced
data, called Remix. It relaxes the mixing factor which results in pushing the
decision boundaries towards majority classes. Our method is easy to implement,
end-to-end trainable and computation eﬃcient which are critical for training on
large-scale imbalanced datasets. We also show that it can be easily used with
existing techniques to achieve superior performance. Despite the ease of use and
eﬀectiveness, the current analysis is rather intuitive. Our future work is to dig
into the mechanism and hopefully provide more in-depth analysis and theoretical
guarantees which may shed lights on a more ideal form of Remix as the current
one involves two hyper-parameters.

--- Page 15 ---
Rebalanced Mixup
15
Supplementary Materials
Questions for Using Remix
In this section, we highlight things that one may wonder when using Remix.
We have some observations on these questions but did not explore in suﬃcient
depth.
How should τ and κ be chosen? Simple grid search is used to ﬁnd proper
values for them. We ﬁrst ﬁx τ to 0 and search κ. We have done this for both
step imbalance and long-tail imbalance. It is rather easy to tune κ for step
imbalance since we are dealing with ”binary” imbalance. As long as the value
can diﬀerentiate the minority from majority, Remix would function as we expect.
After ﬁnding the value for κ, we then ﬁx kappa and search for τ. Note that we
have used the same hyper-parameter setting for most experiments because we
want to show that Remix is not very sensitive to the hyper-parameters. Below
is a very simple ablation studies on τ. It is evaluated on CIFAR-10 with ρ=100
with Remix-DRW. To our surprise, the hyper-parameter τ isn’t as sensitive as
we expected. However, based on our studies on toy dataset two blobs two moon
and two circles with a MLP, we found that τ is sensitive for the exact position
of the decision boundary. As a results, we think that the reason it is not that
sensitive when used with ResNets is because ResNets are highly non-linear. The
τ
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
Accuracy 76.1 76.4 76.4 77.2 77.4 77.3 77.5 77.8 77.5 77.6
How does the original hyper-parameter α from Mixup aﬀects the per-
formance of Remix? Remix is greatly based on Mixup. As a result, in order to
remix to work, one has to ﬁrst make sure that Mixup is able to improve the per-
formance using a certain value for α. And since we use commonly used datasets
which Mixup was also evaluated on, we were able to use the same α as them,
which are 1.0 for CIFAR, CINIC and 0.4 for iNaturalist 2018 (Since iNaturalist
2018 images are in the same size as ImageNet, we use the same value)
Does the amount of times that Remix falls into the ﬁrst and second
condition in eq.10 matter? Since the third condition makes Remix degener-
ate to Mixup, one may wonder if the number of times that it doesn’t degenerate
matter. Our intuition is that since Remix works fairly well with re-weighting and
re-sampling, it matters. For example, when Remix is used with re-sampling, the
chance that a majority sample and minority sample are used to create a mixed
sample is greatly increased and thus increase the number of times that Remix
falls into the ﬁrst and second condition in eq.10.

--- Page 16 ---
16
Chou et al.
(a) ERM
(b) Remix
Fig. 5: To show that the improvement in overall accuracy is coming from the
improvement on minority classes, we show the confusion matrices on CIFAR-10.
References
1. Berthelot, D., Carlini, N., Goodfellow, I.G., Papernot, N., Oliver, A., Raﬀel, C.:
Mixmatch: A holistic approach to semi-supervised learning. In: NeurIPS (2019)
2. Bowyer, K.W., Chawla, N.V., Hall, L.O., Kegelmeyer, W.P.: Smote: Synthetic
minority over-sampling technique. J. Artif. Intell. Res. 16, 321–357 (2002)
3. Bul`o, S.R., Neuhold, G., Kontschieder, P.: Loss max-pooling for semantic image
segmentation. 2017 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) pp. 7082–7091 (2017)
4. Byrd, J., Lipton, Z.C.: What is the eﬀect of importance weighting in deep learning?
In: ICML (2018)
5. Cao, K., Wei, C., Gaidon, A., Arechiga, N., Ma, T.: Learning imbalanced datasets
with label-distribution-aware margin loss. In: Advances in Neural Information Pro-
cessing Systems (2019)
6. Chung, Y.A., Lin, H.T., Yang, S.W.: Cost-aware pre-training for multiclass cost-
sensitive deep learning. In: IJCAI (2015)
7. Cui, Y., Jia, M., Lin, T.Y., Song, Y., Belongie, S.: Class-balanced loss based on
eﬀective number of samples. In: CVPR (2019)
8. Darlow, L.N., Crowley, E., Antoniou, A., Storkey, A.J.: Cinic-10 is not imagenet
or cifar-10. ArXiv abs/1810.03505 (2018)
9. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidi-
rectional transformers for language understanding. In: NAACL-HLT (2019)
10. DeVries, T., Taylor, G.W.: Improved regularization of convolutional neural net-
works with cutout. arXiv preprint arXiv:1708.04552 (2017)
11. Elkan, C.: The foundations of cost-sensitive learning. In: IJCAI (2001)
12. He, H., Bai, Y., Garcia, E.A., Li, S.: Adasyn: Adaptive synthetic sampling approach
for imbalanced learning. 2008 IEEE International Joint Conference on Neural Net-
works (IEEE World Congress on Computational Intelligence) pp. 1322–1328 (2008)
13. Horn, G.V., Aodha, O.M., Song, Y., Cui, Y., Sun, C., Shepard, A., Adam, H., Per-
ona, P., Belongie, S.J.: The inaturalist species classiﬁcation and detection dataset.
2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition pp.
8769–8778 (2017)

--- Page 17 ---
Rebalanced Mixup
17
14. Horn, G.V., Perona, P.: The devil is in the tails: Fine-grained classiﬁcation in the
wild. ArXiv abs/1709.01450 (2017)
15. Huang, C., Li, Y., Loy, C.C., Tang, X.: Learning deep representation for imbalanced
classiﬁcation. 2016 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR) pp. 5375–5384 (2016)
16. Jaehyung Kim, Jongheon Jeong, J.S.: Imbalanced classiﬁcation via adversarial
minority over-sampling. OpenReview (2019), https://openreview.net/pdf?id=
HJxaC1rKDS
17. Kang, B., Xie, S., Rohrbach, M., Yan, Z., Gordo, A., Feng, J., Kalantidis, Y.:
Decoupling representation and classiﬁer for long-tailed recognition. International
Conference on Learning Representations (2020), https://openreview.net/forum?
id=r1Ddp1-Rb
18. Lample, G., Ott, M., Conneau, A., Denoyer, L., Ranzato, M.: Phrase-based &
neural unsupervised machine translation. In: EMNLP (2018)
19. Lan, Z.Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., Soricut, R.: Al-
bert: A lite bert for self-supervised learning of language representations. ArXiv
abs/1909.11942 (2019)
20. Li, Z., Dekel, T., Cole, F., Tucker, R., Snavely, N., Liu, C., Freeman, W.T.: Learning
the depths of moving people by watching frozen people. In: CVPR (2019)
21. Lin, T.Y., Goyal, P., Girshick, R.B., He, K., Doll´ar, P.: Focal loss for dense object
detection. IEEE transactions on pattern analysis and machine intelligence (2017)
22. Liu, W., Wen, Y., Yu, Z., Li, M., Raj, B., Song, L.: Sphereface: Deep hypersphere
embedding for face recognition. 2017 IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) pp. 6738–6746 (2017)
23. Liu, W., Wen, Y., Yu, Z., Yang, M.: Large-margin softmax loss for convolutional
neural networks. In: ICML (2016)
24. Liu, Z., Miao, Z., Zhan, X., Wang, J., Gong, B., Yu, S.X.: Large-scale long-tailed
recognition in an open world. In: CVPR (2019)
25. Mullick, S.S., Datta, S., Das, S.: Generative adversarial minority oversampling. In:
The IEEE International Conference on Computer Vision (ICCV) (October 2019)
26. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O.,
Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A.,
Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, E.: Scikit-learn: Machine
learning in Python. Journal of Machine Learning Research 12, 2825–2830 (2011)
27. Shaham, T.R., Dekel, T., Michaeli, T.: Singan: Learning a generative model from a
single natural image. In: The IEEE International Conference on Computer Vision
(ICCV) (October 2019)
28. van Steenkiste, S., Greﬀ, K., Schmidhuber, J.: A perspective on objects and sys-
tematic generalization in model-based rl. ArXiv abs/1906.01035 (2019)
29. Takahashi, R., Matsubara, T., Uehara, K.: Ricap: Random image cropping and
patching data augmentation for deep cnns. In: Proceedings of The 10th Asian
Conference on Machine Learning (2018)
30. Thulasidasan, S., Chennupati, G., Bilmes, J.A., Bhattacharya, T., Michalak, S.E.:
On mixup training: Improved calibration and predictive uncertainty for deep neural
networks. In: NeurIPS (2019)
31. Tianyu Pang, K.X., Zhu, J.: Mixup inference: Better exploiting mixup to defend
adversarial attacks. In: ICLR (2020)
32. Verma, V., Lamb, A., Beckham, C., Najaﬁ, A., Mitliagkas, I., Lopez-Paz, D., Ben-
gio, Y.: Manifold mixup: Better representations by interpolating hidden states. In:
Chaudhuri, K., Salakhutdinov, R. (eds.) Proceedings of the 36th International

--- Page 18 ---
18
Chou et al.
Conference on Machine Learning. Proceedings of Machine Learning Research,
vol. 97, pp. 6438–6447. PMLR, Long Beach, California, USA (09–15 Jun 2019),
http://proceedings.mlr.press/v97/verma19a.html
33. Verma, V., Lamb, A., Kannala, J., Bengio, Y., Lopez-Paz, D.: Interpolation con-
sistency training for semi-supervised learning. In: IJCAI (2019)
34. Wang, S., Liu, W., Wu, J., Cao, L., Meng, Q., Kennedy, P.J.: Training deep neural
networks on imbalanced data sets. 2016 International Joint Conference on Neural
Networks (IJCNN) pp. 4368–4374 (2016)
35. Wang, T.C., Liu, M.Y., Zhu, J.Y., Liu, G., Tao, A., Kautz, J., Catanzaro, B.:
Video-to-video synthesis. In: NeurIPS (2018)
36. Yun, S., Han, D., Oh, S.J., Chun, S., Choe, J., Yoo, Y.: Cutmix: Regularization
strategy to train strong classiﬁers with localizable features. In: International Con-
ference on Computer Vision (ICCV) (2019)
37. Zhang, H., Ciss´e, M., Dauphin, Y., Lopez-Paz, D.: mixup: Beyond empirical
risk minimization. International Conference on Learning Representations (2018),
https://openreview.net/forum?id=r1Ddp1-Rb
38. Zhou, B., Cui, Q., Wei, X.S., Chen, Z.M.: Bbn: Bilateral-branch network with
cumulative learning for long-tailed visual recognition. In: IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR) (June 2020)
