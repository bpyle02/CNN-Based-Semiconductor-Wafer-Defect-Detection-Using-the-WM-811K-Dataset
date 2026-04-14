# Balanced Meta-Softmax for Long-Tailed Visual Recognition

**Authors**: Ren, Yu, Ma, Zhao, Yi, Li
**Year**: 2020
**arXiv**: 2007.10740
**Topic**: long_tail
**Relevance**: Meta-learned balanced softmax compensating for label distribution shift

---


--- Page 1 ---
Balanced Meta-Softmax
for Long-Tailed Visual Recognition
Jiawei Ren1, Cunjun Yu1, Shunan Sheng1,2, Xiao Ma1,3,
Haiyu Zhao1*, Shuai Yi1, Hongsheng Li4
1 SenseTime Research
2 Nanyang Technological University
3 National University of Singapore
4 Multimedia Laboratory, The Chinese University of Hong Kong
{renjiawei, zhaohaiyu, yishuai}@sensetime.com
cunjun.yu@gmail.com
shen0152@e.ntu.edu.sg
xiao-ma@comp.nus.edu.sg
hsli@ee.cuhk.edu.hk
Abstract
Deep classiﬁers have achieved great success in visual recognition. However, real-
world data is long-tailed by nature, leading to the mismatch between training and
testing distributions. In this paper, we show that the Softmax function, though used
in most classiﬁcation tasks, gives a biased gradient estimation under the long-tailed
setup. This paper presents Balanced Softmax, an elegant unbiased extension of
Softmax, to accommodate the label distribution shift between training and testing.
Theoretically, we derive the generalization bound for multiclass Softmax regression
and show our loss minimizes the bound. In addition, we introduce Balanced Meta-
Softmax, applying a complementary Meta Sampler to estimate the optimal class
sample rate and further improve long-tailed learning. In our experiments, we
demonstrate that Balanced Meta-Softmax outperforms state-of-the-art long-tailed
classiﬁcation solutions on both visual recognition and instance segmentation tasks.†
1
Introduction
Most real-world data comes with a long-tailed nature: a few high-frequency classes (or head classes)
contributes to most of the observations, while a large number of low-frequency classes (or tail classes)
are under-represented in data. Taking an instance segmentation dataset, LVIS [9], for example, the
number of instances in banana class can be thousands of times more than that of a bait class. In
practice, the number of samples per class generally decreases from head to tail classes exponentially.
Under the power law, the tails can be undesirably heavy. A model that minimizes empirical risk on
long-tailed training datasets often underperforms on a class-balanced test dataset. As datasets are
scaling up nowadays, the long-tailed nature poses critical difﬁculties to many vision tasks, e.g., visual
recognition and instance segmentation.
An intuitive solution to long-tailed task is to re-balance the data distribution. Most state-of-the-art
(SOTA) methods use the class-balanced sampling or loss re-weighting to “simulate" a balanced
training set [3, 36]. However, they may under-represent the head class or have gradient issues during
optimization. Cao et al. [4] introduced Label-Distribution-Aware Margin Loss (LDAM), from the
perspective of the generalization error bound. Given fewer training samples, a tail class should have
a higher generalization error bound during optimization. Nevertheless, LDAM is derived from the
hinge loss, under a binary classiﬁcation setup and is not suitable for multi-class classiﬁcation.
*Corresponding author
†Code available at https://github.com/jiawei-ren/BalancedMetaSoftmax
34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada.
arXiv:2007.10740v3  [cs.LG]  22 Nov 2020

--- Page 2 ---
We propose Balanced Meta-Softmax (BALMS) for long-tailed visual recognition. We ﬁrst show that
the Softmax function is intrinsically biased under the long-tailed scenario. We derive a Balanced
Softmax function from the probabilistic perspective that explicitly models the test-time label distribu-
tion shift. Theoretically, we found that optimizing for the Balanced Softmax cross-entropy loss is
equivalent to minimizing the generalization error bound. Balanced Softmax generally improves long-
tailed classiﬁcation performance on datasets with moderate imbalance ratios, e.g., CIFAR-10-LT [21]
with a maximum imbalance factor of 200. However, for datasets with an extremely large imbalance
factor, e.g., LVIS [9] with an imbalance factor of 26,148, the optimization process becomes difﬁcult.
Complementary to the loss function, we introduce the Meta Sampler, which learns to re-sample for
achieving high validation accuracy by meta-learning. The combination of Balanced Softmax and
Meta Sampler could efﬁciently address long-tailed classiﬁcation tasks with high imbalance factors.
We evaluate BALMS on both long-tailed image classiﬁcation and instance segmentation on ﬁve
commonly used datasets: CIFAR-10-LT [21], CIFAR-100-LT [21], ImageNet-LT [26], Places-LT [39]
and LVIS [9]. On all datasets, BALMS outperforms state-of-the-art methods. In particular, BALMS
outperforms all SOTA methods on LVIS, with an extremely high imbalanced factor, by a large margin.
We summarize our contributions as follows: 1) we theoretically analyze the incapability of Softmax
function in long-tailed tasks; 2) we introduce Balanced Softmax function that explicitly considers
the label distribution shift during optimization; 3) we present Meta Sampler, a meta-learning based
re-sampling strategy for long-tailed learning.
2
Related Works
Data Re-Balancing. Pioneer works focus on re-balancing during training. Speciﬁcally, re-sampling
strategies [22, 5, 10, 12, 31, 2, 1] try to restore the true distributions from the imbalanced training data.
Re-weighting, i.e., cost-sensitive learning [36, 13, 14, 28], assigns a cost weight to the loss of each
class. However, it is argued that over-sampling inherently overﬁts the tail classes and under-sampling
under-represents head classes’ rich variations. Meanwhile, re-weighting tends to cause unstable
training especially when the class imbalance is severe because there would be abnormally large
gradients when the weights are very large.
Loss Function Engineering. Tan et al. [35] point out that randomly dropping some scores of tail
classes in the Softmax function can effectively help, by balancing the positive gradients and negative
gradients ﬂowing through the score outputs. Cao et al. [4] show that the generalization error bound
could be minimized by increasing the margins of tail classes. Hayat et al. [11] modify the loss
function based on Bayesian uncertainty. Li et al. [23] propose two novel loss functions to balance
the gradient ﬂow. Khan et al. [19] jointly learn the model parameters and the class-dependent loss
function parameters. Ye et al. [37] force a large margin for minority classes to prevent feature
deviation. We progress this line of works by introducing probabilistic insights that also bring
empirical improvements. We show in this paper that an ideal loss function should be unbiased under
the long-tailed scenarios.
Meta-Learning. Many approaches [15, 30, 32] have been proposed to tackle the long-tailed issue
with meta-learning. Many of them [15, 30] focus on optimizing the weight-per-sample as a learnable
parameter, which appears as a hyper-parameter in the sample-based re-weight approach. This group
of methods requires a clean and unbiased dataset as a meta set, i.e., development set, which is usually
a ﬁxed subset of the training images and use bi-level optimization to estimate the weight parameter.
Decoupled Training. Kang et al. [18] point out that decoupled training, a simple yet effective
solution, could signiﬁcantly improve the generalization issue on long-tailed datasets. The classiﬁer
is the only under-performed component when training in imbalanced datasets. However, in our
experiments, we found this technique is not adequate for datasets with extremely high imbalance
factors, e.g., LVIS [9]. Interestingly in our experiments, we observed that decoupled training is
complementary to our proposed BALMS, and combining them results in additional improvements.
3
Balanced Meta-Softmax
The major challenge for long-tailed visual recognition is the mismatch between the imbalanced train-
ing data distribution and the balanced metrics, e.g., mean Average Precision (mAP), that encourage
2

--- Page 3 ---
minimizing error on a balanced test set. Let X = {xi, yi}, i ∈{1, · · · , n} be the balanced test set,
where xi denotes a data point and yi denotes its label. Let k be the number of classes, nj be the
number of samples in class j, where Pk
j=1 nj = n. Similarly, we denote the long-tailed training set
as ˆ
X = {ˆxi, ˆyi}, i ∈{1, . . . , n}. Normally, we have ∀i, p(ˆyi) ̸= p(yi). Speciﬁcally, for a tail class j,
p(ˆyj) ≪p(yj), which makes the generalization under long-tailed scenarios extremely challenging.
We introduce Balanced Meta-Softmax (BALMS) for long-tailed visual recognition. It has two
components: 1) a Balanced Softmax function that accommodates the label distribution shift between
training and testing; 2) a Meta Sampler that learns to re-sample training set by meta-learning. We
denote a feature extractor function as f and a linear classiﬁer’s weight as θ.
3.1
Balanced Softmax
Label Distribution Shift. We begin by revisiting the multi-class Softmax regression, where we are
generally interested in estimating the conditional probability p(y|x), which can be modeled as a
multinomial distribution φ:
φ = φ1{y=1}
1
φ1{y=2}
2
· · · φ1{y=k}
k
;
φj =
eηj
Pk
i=1 eηi ;
k
X
j=1
φj = 1
(1)
where 1(·) is the indicator function and Softmax function maps a model’s class-j output ηj = θT
j f(x)
to the conditional probability φj.
From the Bayesian inference’s perspective, φj can also be interpreted as:
φj = p(y = j|x) = p(x|y = j)p(y = j)
p(x)
(2)
where p(y = j) is in particular interest under the class-imbalanced setting. Assuming that all
instances in the training dataset and the test dataset are generated from the same process p(x|y = j),
there could still be a discrepancy between training and testing given different label distribution
p(y = j) and evidence p(x). With a slight abuse of the notation, we re-deﬁne φ to be the conditional
distribution on the balanced test set and deﬁne ˆφ to be the conditional probability on the imbalanced
training set. As a result, standard Softmax provides a biased estimation for φ.
Balanced Softmax. To eliminate the discrepancy between the posterior distributions of training and
testing, we introduce Balanced Softmax. We use the same model outputs η to parameterize two
conditional probabilities: φ for testing and ˆφ for training.
Theorem 1. Assume φ to be the desired conditional probability of the balanced dataset, with the form
φj = p(y = j|x) = p(x|y=j)
p(x)
1
k, and ˆφ to be the desired conditional probability of the imbalanced
training set, with the form ˆφj = ˆp(y = j|x) = p(x|y=j)
ˆp(x)
nj
Pk
i=1 ni . If φ is expressed by the standard
Softmax function of model output η, then ˆφ can be expressed as
ˆφj =
njeηj
Pk
i=1 nieηi .
(3)
We use the exponential family parameterization to prove Theorem 1. The proof can be found in the
supplementary materials. Theorem 1 essentially shows that applying the following Balanced Softmax
function can naturally accommodate the label distribution shifts between the training and test sets.
We deﬁne the Balanced Softmax function as
ˆl(θ) = −log(ˆφy) = −log
 
nyeηy
Pk
i=1 nieηi
!
.
(4)
We further investigate the improvement brought by the Balanced Softmax in the following sections.
Many vision tasks, e.g., instance segmentation, might use multiple binary logistic regressions instead
of a multi-class Softmax regression. By virtue of Bayes’ theorem, a similar strategy can be applied to
the multiple binary logistic regressions. The detailed derivation is left in the supplementary materials.
3

--- Page 4 ---
Generalization Error Bound. Generalization error bound gives the upper bound of a model’s test
error, given its training error. With dramatically fewer training samples, the tail classes have much
higher generalization bounds than the head classes, which make good classiﬁcation performance on
tail classes unlikely. In this section, we show that optimizing Eqn. 4 is equivalent to minimizing the
generalization upper bound.
Margin theory provides a bound based on the margins [17]. Margin bounds usually negatively
correlate to the magnitude of the margin, i.e., a larger margin leads to lower generalization error.
Consequently, given a constraint on the sum of margins of all classes, there would be a trade-off
between minority classes and majority classes [4].
Locating such an optimal margin for multi-class classiﬁcation is non-trivial. The bound investigated
in [4] was established for binary classiﬁcation using hinge loss. Here, we try to develop the margin
bound for the multi-class Softmax regression. Given the previously deﬁned φ and ˆφ, we derive ˆl(θ)
by minimizing the margin bound. Margin bound commonly bounds the 0-1 error:
err0,1 = Pr

θT
y f(x) < max
i̸=y θT
i f(x)

.
(5)
However, directly using the 0-1 error as the loss function is not ideal for optimization. Instead,
negative log likelihood (NLL) is generally considered more suitable. With continuous relaxation of
Eqn. 5, we have
err(t) = Pr[t < log(1 +
X
i̸=y
eθT
i f(x)−θT
y f(x))] = Pr [ly(θ) > t] ,
(6)
where t ≥0 is any threshold, and ly(θ) is the standard negative log-likelihood with Softmax, i.e.,
the cross-entropy loss. This new error is still a counter, but describes how likely the test loss will be
larger than a given threshold. Naturally, we deﬁne our margin for class j to be
γj = t −
max
(x,y)∈Sj lj(θ).
(7)
where Sj is the set of all class j samples. If we force a large margin γj during training, i.e., force the
training loss to be much lower than t, then err(t) will be reduced. The Theorem 2 in [17] can then
be directly generalized as
Theorem 2. Let t ≥0 be any threshold, for all γj > 0, with probability at least 1 −δ, we have
errbal(t) ≲1
k
k
X
j=1
 1
γj
s
C
nj
+ log n
√nj

;
γ∗
j =
βn−1/4
j
Pk
i=1 n−1/4
i
,
(8)
where errbal(t) is the error on the balanced test set, ≲is used to hide constant terms and C is some
measure on complexity. With a constraint on Pk
j=1 γj = β, Cauchy-Schwarz inequality gives us the
optimal γ∗
j .
The optimal γ∗suggests that we need larger γ for the classes with fewer samples. In other words, to
achieve the optimal generalization ability, we need to focus on minimizing the training loss of the tail
classes. To enforce the optimal margin, for each class j, the desired training loss ˆl∗
j(θ) is
ˆl∗
j(θ) = lj(θ) + γ∗
j ,
where
lj(θ) = −log(φj),
(9)
Corollary 2.1. ˆl∗
j(θ) = lj(θ) + γ∗
j = lj(θ) +
βn−1/4
j
Pk
i=1 n−1/4
i
can be approximated by ˆlj(θ) when:
ˆlj(θ) = −log(ˆφj);
ˆφj =
eηj−log γ∗
j
Pk
i=1 eηi−log γ∗
i =
n
1
4
j eηj
Pk
i=1 n
1
4
i eηi
(10)
We provide a sketch of proof to the corollary in supplementary materials. Notice that compared
to Eqn. 4, we have an additional constant 1/4. We empirically ﬁnd that setting 1/4 to 1 leads to
the optimal results, which may suggest that Eqn. 8 is not necessarily tight. To this point, the label
distribution shift and generalization bound of multi-class Softmax regression lead us to the same loss
form: Eqn. 4.
4

--- Page 5 ---
3.2
Meta Sampler
Re-sampling. Although Balanced Softmax accommodates the label distribution shift, the optimiza-
tion process is still challenging when given large datasets with extremely imbalanced data distribution.
For example, in LVIS, the bait class may appear only once when the banana class appears thousands
of times, making the bait class difﬁcult to contribute to the model training due to low sample rate.
Re-sampling is usually adopted to alleviate this issue, by increasing the number of minority class
samples in each training batch. Recent works [34, 3] show that the global minimum of the Softmax
regression is independent of the mini-batch sampling process. Our visualization in the supplemen-
tary material conﬁrms this ﬁnding. As a result, a suitable re-sampling strategy could simplify the
optimization landscape of Balanced Softmax under extremely imbalanced data distribution.
Over-balance. Class-balanced sampler (CBS) is a common re-sampling strategy. CBS balances the
number of samples for each class in a mini-batch. It effectively helps to re-train the linear classiﬁer
in the decoupled training setup [18]. However, in our experiments, we ﬁnd that naively combining
CBS with Balanced Softmax may worsen the performance.
We ﬁrst theoretically analyze the cause of the performance drop. When the linear classiﬁer’s weight
θj for class j converges, i.e., PB
s=1
∂L(s)
∂θj
= 0, we should have:
B
X
s=1
∂L(s)
∂θj
=
B/k
X
s=1
f(x(s)
y=j)(1 −ˆφ(s)
j ) −
k
X
i̸=j
B/k
X
s=1
f(x(s)
y=i)ˆφ(s)
j
= 0,
(11)
where B is the batch size and k is the number of classes. Samples per class have been ensured to be
B/k by CBS. We notice that ˆφj, the output of Balanced Softmax, casts a varying, minority-favored
effect to the importance of each class.
We use an extreme case to demonstrate the effect. When the classiﬁcation loss converges to 0, the
conditional probability of the correct class ˆφy is expected to be close to 1. For any positive sample
x+ and negative sample x−of class j, we have ˆφj(x+) ≈φj(x+) and ˆφj(x−) ≈nj
ni φj(x−), when
ˆφy →1. Eqn. 11 can be rewritten as
1
n2
j
E(x+,y=j)∼Dtrain[f(x+)(1 −φj)] −
k
X
i̸=j
1
n2
i
E(x−,y=i)∼Dtrain[f(x−)φj] ≈0
(12)
where Dtrain is the training set. The formal derivation of Eqn. 12 is in the supplementary materials.
Compared to the inverse loss weight, i.e., 1/nj for class j, combining Balanced Softmax with CBS
leads to the over-balance problem, i.e., 1/n2
j for class j, which deviates from the optimal distribution.
Although re-sampling does not affect the global minimum, an over-balanced, tail class dominated
optimization process may lead to local minimums that favor the minority classes. Moreover, Balanced
Softmax’s effect in the optimization process is dependent on the model’s output, which makes
hand-crafting a re-sampling strategy infeasible.
Meta Sampler. To cope with CBS’s over-balance issue, we introduce Meta Sampler, a learnable
version of CBS based on meta-learning, which explicitly learns the optimal sample rate. We ﬁrst
deﬁne the empirical loss by sampling from dataset D as LD(θ) = E(x,y)∼D[l(θ)] for standard
Softmax, and ˆLD(θ) = E(x,y)∼D[ˆl(θ)] for Balanced Softmax, where ˆl(θ) is deﬁned previously in
Eqn. 4.
To estimate the optimal sample rates for different classes, we adopt a bi-level meta-learning strategy:
we update the parameter ψ of sample distribution πψ in the inner loop and update the classiﬁer
parameters θ in the outer loop,
π∗
ψ = arg min
ψ LDmeta(θ∗(πψ))
s.t.
θ∗(πψ) = arg min
θ
ˆLDq(x,y;πψ)(θ),
(13)
where πj
ψ = p(y = j; ψ) is the sample rate for class j, Dq(x,y;πψ) is the training set with class sample
distribution πψ, and Dmeta is a meta set we introduce to supervise the inner loop optimization. We
create the meta set by class-balanced sampling from the training set Dtrain. Empirically, we found it
sufﬁcient for inner loop optimization. An intuition to this bi-level optimization strategy is that: we
5

--- Page 6 ---
want to learn best sample distribution parameter ψ such that the network, parameterized by θ, outputs
best performance on meta dataset Dmeta when trained by samples from πψ.
We ﬁrst compute the per-instance sample rate ρi = πc(i)
ψ
/ Pn
i=1 πc(i)
ψ
, where c(i) denotes the label
class for instance i and n is total number of training samples, and sample a training batch Bψ from a
parameterized multi-nomial distribution ρ. Then we optimize the model in a meta-learning setup by
1. sample a mini-batch Bψ given distribution πψ and perform one step gradient descent to get
a surrogate model parameterized by ˜θ by ˜θ ←θ −∇θ ˆLBψ(θ).
2. compute the LDmeta(˜θ) of the surrogate model on the meta dataset Dmeta and optimize the
sample distribution parameter by ψ ←ψ −∇ψLDmeta(˜θ) with the standard cross-entropy
loss with Softmax.
3. update the model parameter θ ←θ −∇θ ˆLBψ(θ) with Balanced Softmax.
However, sampling from a discrete distribution is not differentiable by nature. To allow end-to-end
training for the sampling process, when forming the mini-batch Bψ, we apply the Gumbel-Softmax
reparameterization trick [16]. A detailed explanation can be found in the supplementary materials.
4
Experiments
4.1
Exprimental Setup
Datasets. We perform experiments on long-tailed image classiﬁcation datasets, including CIFAR-10-
LT [21], CIFAR-100-LT [21], ImageNet-LT [26] and Places-LT [39] and one long-tailed instance
segmentation dataset, LVIS [9]. We deﬁne the imbalance factor of a dataset as the number of training
instances in the largest class divided by that of the smallest. Details of datasets are in Table 1.
Dataset
#Classes
Imbalance Factor
CIFAR-10-LT [21]
10
10-200
CIFAR-100-LT [21]
100
10-200
ImageNet-LT [26]
1,000
256
Places-LT [39]
365
996
LVIS [9]
1,230
26,148
Table 1: Details of long-tailed datatsets. For
both CIFAR-10 and CIFAR-100, we report re-
sults with different imbalance factors.
Evaluation Setup.
For classiﬁcation tasks, af-
ter training on the long-tailed dataset, we eval-
uate the models on the corresponding balanced
test/validation dataset and report top-1 accuracy.
We also report accuracy on three splits of the set
of classes: Many-shot (more than 100 images),
Medium-shot (20 ∼100 images), and Few-shot
(less than 20 images). Notice that results on small
datasets, i.e., CIFAR-LT 10/100, tend to show large
variances, we report the mean and standard error
under 3 repetitive experiments. We show details
of long-tailed dataset generation in supplementary
materials. For LVIS, we use ofﬁcial training and
testing splits. Average Precision (AP) in COCO style [24] for both bounding box and instance mask
are reported. Our implementation details can be found in the supplementary materials.
0
20
40
60
80
100
0.000
0.005
0.010
0.015
0.020
0.025
Softmax
EQL
Balanced Softmax
0
20
40
60
80
100
0.000
0.005
0.010
0.015
0.020
0.025
0
20
40
60
80
100
0.000
0.005
0.010
0.015
0.020
0.025
Imbalance Factor = 10
Imbalance Factor = 100
Imbalance Factor = 200
Figure 1: Experiment on CIFAR-100-LT. x-axis is the class labels with decreasing training samples
and y-axis is the marginal likelihood p(y) on the test set. We use end-to-end training for the
experiment. Balanced Softmax is more stable under a high imbalance factor compared to the Softmax
baseline and the SOTA method, Equalization Loss (EQL).
6

--- Page 7 ---
Dataset
CIFAR-10-LT
CIFAR-100-LT
Imbalance Factor
200
100
10
200
100
10
End-to-end training
Softmax
71.2 ± 0.3
77.4 ± 0.8
90.0 ± 0.2
41.0 ± 0.3
45.3 ± 0.3
61.9 ± 0.1
CBW
72.5 ± 0.2
78.6 ± 0.6
90.1 ± 0.2
36.7 ± 0.2
42.3 ± 0.8
61.4 ± 0.3
CBS
68.3 ± 0.3
77.8 ± 2.2
90.2 ± 0.2
37.8 ± 0.1
42.6 ± 0.4
61.2 ± 0.3
Focal Loss [25]
71.8 ± 2.1
77.1 ± 0.2
90.3 ± 0.2
40.2 ± 0.5
43.8 ± 0.1
60.0 ± 0.6
Class Balanced Loss [6]
72.6 ± 1.8
78.2 ± 1.1
89.9 ± 0.3
39.9 ± 0.1
44.6 ± 0.4
59.8 ± 1.1
LDAM Loss [4]
73.6 ± 0.1
78.9 ± 0.9
90.3 ± 0.1
41.3 ± 0.4
46.1 ± 0.1
62.1 ± 0.3
Equalization Loss [35]
74.6 ± 0.1
78.5 ± 0.1
90.2 ± 0.2
43.3 ± 0.1
47.4 ± 0.2
60.5 ± 0.6
Decoupled training
cRT [18]
76.6 ± 0.2
82.0 ± 0.2
91.0 ± 0.0
44.5 ± 0.1
50.0 ± 0.2
63.3 ± 0.1
LWS [18]
78.1 ± 0.0
83.7 ± 0.0
91.1 ± 0.0
45.3 ± 0.1
50.5 ± 0.1
63.4 ± 0.1
BALMS
81.5 ± 0.0
84.9 ± 0.1
91.3 ± 0.1
45.5 ± 0.1
50.8 ± 0.0
63.0 ± 0.1
Table 2: Top 1 accuracy for CIFAR-10/100-LT. Softmax: the standard cross-entropy loss with
Softmax. CBW: class-balanced weighting. CBS: class-balanced sampling. LDAM Loss: LDAM
loss without DRW. Results of Focal Loss, Class Balanced Loss, LDAM Loss and Equalization Loss
are reproduced with optimal hyper-parameters reported in their original papers. BALMS generally
outperforms SOTA methods, especially when the imbalance factor is high. Note that for all compared
methods, we reproduce higher accuracy than reported in original papers. Comparison with their
originally reported results is provided in the supplmentary materials.
4.2
Long-Tailed Image Classiﬁcation
We present the results for long-tailed image classiﬁcation in Table 2 and Table 3. On all datasets,
BALMS achieves SOTA performance compared with all end-to-end training and decoupled training
methods. In particular, we notice that BALMS demonstrates a clear advantage under two cases: 1)
When the imbalance factor is high. For example, on CIFAR-10 with an imbalance factor of 200,
BALMS is higher than the SOTA method, LWS [18], by 3.4%. 2) When the dataset is large. BALMS
achieves comparable performance with cRT on ImageNet-LT, which is a relatively small dataset, but
it signiﬁcantly outperforms cRT on a larger dataset, Places-LT.
In addition, we study the robustness of the proposed Balanced Softmax compared to standard Softmax
and SOTA loss function for long-tailed problems, EQL [35]. We visualize the marginal likelihood
p(y), i.e., the sum of scores on each class, on the test set with different losses given different
imbalance factors in Fig. 1. Balanced Softmax clearly gives a more balanced likelihood under
different imbalance factors. Moreover, we show Meta Sampler’s effect on p(y) in Fig. 2. Compared
to CBS, Meta Sampler signiﬁcantly relieves the over-balance issue.
0
2
4
6
8
0.04
0.06
0.08
0.10
0.12
0.14
0.16
0.18
0.20
0.22
Softmax
CBS
BS
BS + CBS
BS + Meta Sampler
0
20
40
60
80
100
0.000
0.004
0.008
0.012
0.016
0.020
CIFAR-10-LT
CIFAR-100-LT
Figure 2: Visualization of p(y) on test set with Meta Sampler and CBS. x-axis is the class labels with
decreasing training samples and y-axis is the marginal likelihood p(y) on the test set. The result is on
CIFAR-10/100-LT with imbalance factor 200. We use decoupled training for the experiment. BS:
Balanced Softmax. BS + CBS shows a clear bias towards the tail classes, especially on CIFAR-100-LT.
Compared to BS + CBS, BS + Meta Sampler effectively alleviates the over-balance problem.
7

--- Page 8 ---
Dataset
ImageNet-LT
Places-LT
Accuracy
Many
Medium
Few
Overall
Many
Medium
Few
Overall
End-to-end training
Lifted Loss [33]
35.8
30.4
17.9
30.8
41.1
35.4
24
35.2
Focal Loss [25]
36.4
29.9
16
30.5
41.1
34.8
22.4
34.6
Range Loss [38]
35.8
30.3
17.6
30.7
41.1
35.4
23.2
35.1
OLTR [26]
43.2
35.1
18.5
35.6
44.7
37.0
25.3
35.9
Equalization Loss [35]
-
-
-
36.4
-
-
-
-
Decoupled training
cRT [18]
-
-
-
41.8
42.0
37.6
24.9
36.7
LWS [18]
-
-
-
41.4
40.6
39.1
28.6
37.6
BALMS
50.3
39.5
25.3
41.8
41.2
39.8
31.6
38.7
Table 3: Top 1 Accuracy on ImageNet-LT and Places-LT. We present results with ResNet-10 [26] for
ImageNet-LT and ImageNet pre-trained ResNet-152 for Places-LT. Baseline results are taken from
original papers. BALMS generally outperforms the SOTA models.
Method
APm
APf
APc
APr
APb
Softmax
23.7
27.3
24.0
13.6
24.0
Sigmoid
23.6
27.3
24.0
12.7
24.0
Focal Loss [25]
23.4
27.5
23.5
12.8
23.8
Class Balanced Loss [6]
23.3
27.3
23.8
11.4
23.9
LDAM [4]
24.1
26.3
25.3
14.6
24.5
LWS [18]
23.8
26.8
24.4
14.4
24.1
Equalization Loss [35]
25.2
26.6
27.3
14.6
25.7
Balanced Softmax†
26.3
28.8
27.3
16.2
27.0
BALMS
27.0
27.5
28.9
19.6
27.6
Table 4: Results for LVIS dataset. APm denotes Average Precision of masks. APb denotes Average
Precision of bounding box. APf, APc and APr denote Average Precision of masks on frequent classes,
common classes and rare classes. †: the multiple binary logistic regression variant of Balanced
Softmax, more details in the supplementary material. BALMS signiﬁcantly outperforms SOTA
models given high imbalance factor in LVIS. All compared methods are reproduced with higher AP
than reported in the original papers.
4.3
Long-Tailed Instance Segmentation
LVIS dataset is one of the most challenging datasets in the vision community. As suggested in Tabel 1,
the dataset has a much higher imbalance factor compared to the rest (26148 vs. less than 1000)
and contains many very few-shot classes. Compared to the image classiﬁcation datasets, which are
relatively small and have lower imbalance factors, the LVIS dataset gives a more reliable evaluation
of the performance of long-tailed learning methods.
Since one image might contain multiple instances from several categories, we hereby use Meta
Reweighter, a re-weighting version of Meta Sampler, instead of Meta Sampler. As shown in Table 4,
BALMS achieves the best results among all the approaches and outperform others by a large margin,
especially in rare classes, where BALMS achieves an average precision of 19.6 while the best of
the rest is 14.6. The results suggest that with the Balanced Softmax function and learnable Meta
Reweighter, BALMS is able to give more balanced gradients and tackles the extremely imbalanced
long-tailed tasks.
In particular, LVIS is composed of images of complex daily scenes with natural long-tailed categories.
To this end, we believe BALMS is applicable to real-world long-tailed visual recognition challenges.
4.4
Component Analysis
We conduct an extensive component analysis on CIFAR-10/100-LT dataset to further understand the
effect of each proposed component of BALMS. The results are presented in Table 5.
8

--- Page 9 ---
Dataset
CIFAR-10-LT
CIFAR-100-LT
Imbalance Factor
200
100
10
200
100
10
End-to-end training
(1) Softmax
71.2 ± 0.3
77.4 ± 0.8
90.0 ± 0.2
41.0 ± 0.3
45.3 ± 0.3
61.9 ± 0.1
(2) Balanced Softmax 1
4
71.6 ± 0.7
78.4 ± 0.9
90.5 ± 0.1
41.9 ± 0.2
46.4 ± 0.7
62.6 ± 0.3
(3) Balanced Softmax
79.0 ± 0.8
83.1 ± 0.4
90.9 ± 0.4
45.9 ± 0.3
50.3 ± 0.3
63.1 ± 0.2
Decoupled training
(4) Balanced Softmax 1
4 +DT
72.2 ± 0.1
79.1 ± 0.2
90.2 ± 0.0
42.3 ± 0.0
46.1 ± 0.1
62.5 ± 0.1
(5) Balanced Softmax 1
4 +DT+MS
76.2 ± 0.4
81.4 ± 0.1
91.0 ± 0.1
44.1 ± 0.2
49.2 ± 0.1
62.8 ± 0.2
(6) Balanced Softmax+DT
78.6 ± 0.1
83.7 ± 0.1
91.2 ± 0.0
45.1 ± 0.0
50.4 ± 0.0
63.4 ± 0.0
(7) Balanced Softmax+CBS+DT
80.6 ± 0.1
84.8 ± 0.0
91.2 ± 0.1
42.0 ± 0.0
47.4 ± 0.2
62.3 ± 0.0
(8) DT+MS
73.6 ± 0.2
79.9 ± 0.4
90.9 ± 0.1
44.2 ± 0.1
49.2 ± 0.1
63.0 ± 0.0
(9) Balanced Softmax+DT+MR
79.2 ± 0.0
84.1 ± 0.0
91.2 ± 0.1
45.3 ± 0.3
50.8 ± 0.0
63.5 ± 0.1
(10) BALMS
81.5 ± 0.0
84.9 ± 0.1
91.3 ± 0.1
45.5 ± 0.1
50.8 ± 0.0
63.0 ± 0.1
Table 5: Component Analysis on CIFAR-10/100-LT. CBS: class-balanced sampling. DT: decoupled
training without CBS. MS: Meta Sampler. MR: Meta Reweighter. Balanced Softmax 1
4: the loss
variant in Eqn. 10. Balanced Softmax and Meta Sampler both contribute to the ﬁnal performance.
Balanced Softmax. Comparing (1), (2) with (3), and (5), (8) with (10), we observe that Balanced
Softmax gives a clear improvement to the overall performance, under both end-to-end training and
decoupled training setup. It successfully accommodates the distribution shift between training and
testing. In particular, we observe that Balanced Softmax 1
4, which we derive in Eqn. 10, cannot yield
ideal results, compared to our proposed Balanced Softmax in Eqn. 4.
Meta-Sampler. From (6), (7), (9) and (10), we observe that Meta-Sampler generally improves the
performance, when compared with no Meta-Sampler, and variants of Meta-Sampler. We notice that
the performance gain is larger with a higher imbalance factor, which is consistent with our observation
in LVIS experiments. In (9) and (10), Meta-Sampler generally outperforms the Meta-Reweighter and
suggests the discrete sampling process gives a more efﬁcient optimization process. Comparing (7)
and (10), we can see Meta-Sampler addresses the over-balancing issue discussed in Section 3.2.
Decoupled Training. Comparing (2) with (4) and (3) with (6), decoupled training scheme and
Balanced Softmax are two orthogonal components and we can beneﬁt from both at the same time.
5
Conclusion
We have introduced BALMS for long-tail visual recognition tasks. BALMS tackles the distribution
shift between training and testing, combining meta-learning with generalization error bound theory: it
optimizes a Balanced Softmax function which theoretically minimizes the generalization error bound;
it improves the optimization in large long-tailed datasets by learning an effective Meta Sampler.
BALMS generally outperforms SOTA methods on 4 image classiﬁcation datasets and 1 instance
segmentation dataset by a large margin, especially when the imbalance factor is high.
However, Meta Sampler is computationally expensive in practice and the optimization on large
datasets is slow. In addition, the Balanced Softmax function only approximately guarantees a
generalization error bound. Future work may extend the current framework to a wider range of tasks,
e.g., machine translation, and correspondingly design tighter bounds and computationally efﬁcient
meta-learning algorithms.
6
Acknowledgements
This work is supported in part by the General Research Fund through the Research Grants Council
of Hong Kong under grants (Nos. CUHK14208417, CUHK14207319), in part by the Hong Kong
Innovation and Technology Support Program (No. ITS/312/18FX).
9

--- Page 10 ---
Broader Impact
Due to the Zipﬁan distribution of categories in real life, algorithms, and models with exceptional
performance on research benchmarks may not remain powerful in the real world. BALMS, as a
light-weight method, only adds minimal computational cost during training and is compatible with
most of the existing works for visual recognition. As a result, BALMS could be beneﬁcial to bridge
the gap between research benchmarks and industrial applications for visual recognition.
However, there can be some potential negative effects. As BALMS empowers deep classiﬁers with
stronger recognition capability on long-tailed distribution, the application of such a classiﬁcation
algorithm can be further extended to more real-life scenarios. We should be cautious about the misuse
of the method proposed. Depending on the scenario, it might cause negative effects on democratic
privacy.
References
[1] Ricardo Barandela, E Rangel, Jose Salvador Sanchez, and Francesc J Ferri. Restricted decon-
tamination for the imbalanced training sample problem. Iberoamerican Congress on Pattern
Recognition, 21(9):1263–1284, 2009.
[2] Mateusz Buda, Atsuto Maki, and Maciej A Mazurowski. A systematic study of the class
imbalance problem in convolutional neural networks. Neural Networks, 106:249–259, 2018.
[3] Jonathon Byrd and Zachary Lipton. What is the effect of importance weighting in deep learning?
In International Conference on Machine Learning, pages 872–881, 2019.
[4] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. Learning imbalanced
datasets with label-distribution-aware margin loss. In H. Wallach, H. Larochelle, A. Beygelzimer,
F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing
Systems 32, pages 1567–1578. Curran Associates, Inc., 2019.
[5] Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer. Smote:
Synthetic minority over-sampling technique. Journal of Artiﬁcial Intelligence Research, 16:
321–357, 2002.
[6] Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge J. Belongie. Class-balanced loss
based on effective number of samples. 2019 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), pages 9260–9269, 2019.
[7] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale
Hierarchical Image Database. In CVPR09, 2009.
[8] Edward Grefenstette, Brandon Amos, Denis Yarats, Phu Mon Htut, Artem Molchanov, Franziska
Meier, Douwe Kiela, Kyunghyun Cho, and Soumith Chintala. Generalized inner loop meta-
learning. arXiv preprint arXiv:1910.01727, 2019.
[9] Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A dataset for large vocabulary instance seg-
mentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
2019.
[10] Hui Han, Wen-Yuan Wang, and Bing-Huan Mao. Borderline-smote: a new over-sampling
method in imbalanced data sets learning. International Conference on Intelligent Computing,
16:321–357, 2005.
[11] Munawar Hayat, Salman Khan, Syed Waqas Zamir, Jianbing Shen, and Ling Shao. Gaussian
afﬁnity for max-margin class imbalanced learning. In The IEEE International Conference on
Computer Vision (ICCV), October 2019.
[12] H. He and E. A. Garcia. Learning from imbalanced data. IEEE Transactions on Knowledge and
Data Engineering, 21(9):1263–1284, 2009.
10

--- Page 11 ---
[13] Chen Huang, Yining Li, Chen Change Loy, and Xiaoou Tang. Learning deep representation
for imbalanced classiﬁcation. In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 5375–5384, 2016.
[14] Chen Huang, Yining Li, Change Loy Chen, and Xiaoou Tang. Deep imbalanced learning for
face recognition and attribute prediction. IEEE transactions on pattern analysis and machine
intelligence, 2019.
[15] Muhammad Abdullah Jamal, Matthew Brown, Ming-Hsuan Yang, Liqiang Wang, and Boqing
Gong. Rethinking class-balanced methods for long-tailed visual recognition from a domain
adaptation perspective. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), June 2020.
[16] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparametrization with gumbel-softmax. In
Proceedings International Conference on Learning Representations 2017, April 2017.
[17] Sham M Kakade, Karthik Sridharan, and Ambuj Tewari. On the complexity of linear prediction:
Risk bounds, margin bounds, and regularization. In Advances in neural information processing
systems, pages 793–800, 2009.
[18] Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng,
and Yannis Kalantidis. Decoupling representation and classiﬁer for long-tailed recognition.
International Conference on Learning Representations, abs/1910.09217, 2020.
[19] Salman H Khan, Munawar Hayat, Mohammed Bennamoun, Ferdous A Sohel, and Roberto
Togneri. Cost-sensitive learning of deep feature representations from imbalanced data. IEEE
transactions on neural networks and learning systems, 29(8):3573–3587, 2017.
[20] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. International
Conference on Learning Representations, 2015.
[21] Alex Krizhevsky. Learning multiple layers of features from tiny images. University of Toronto,
05 2012.
[22] Miroslav Kubat and Stan Matwin. Addressing the curse of imbalanced training sets: One-sided
selection. In In Proceedings of the Fourteenth International Conference on Machine Learning,
pages 179–186. Morgan Kaufmann, 1997.
[23] Buyu Li, Yu Liu, and Xiaogang Wang. Gradient harmonized single-stage detector. In AAAI
Conference on Artiﬁcial Intelligence, 2019.
[24] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European
conference on computer vision, pages 740–755. Springer, 2014.
[25] Tsung-Yi Lin, Priya Goyal, Ross B. Girshick, Kaiming He, and Piotr Dollár. Focal loss for
dense object detection. 2017 IEEE International Conference on Computer Vision (ICCV), pages
2999–3007, 2017.
[26] Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X. Yu.
Large-scale long-tailed recognition in an open world. 2019 IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), pages 2532–2541, 2019.
[27] I. Loshchilov and F. Hutter. Sgdr: Stochastic gradient descent with warm restarts. In Interna-
tional Conference on Learning Representations, April 2017.
[28] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed rep-
resentations of words and phrases and their compositionality. In C. J. C. Burges, L. Bottou,
M. Welling, Z. Ghahramani, and K. Q. Weinberger, editors, Advances in Neural Information
Processing Systems 26, pages 3111–3119. Curran Associates, Inc., 2013.
11

--- Page 12 ---
[29] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas
Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy,
Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-
performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-
Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32,
pages 8024–8035. Curran Associates, Inc., 2019.
[30] Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urtasun. Learning to reweight examples
for robust deep learning. In ICML, 2018.
[31] Li Shen, Zhouchen Lin, and Qingming Huang. Relay backpropagation for effective learning
of deep convolutional neural networks. In European conference on computer vision, pages
467–482. Springer, 2016.
[32] Jun Shu, Qi Xie, Lixuan Yi, Qian Zhao, Sanping Zhou, Zongben Xu, and Deyu Meng. Meta-
weight-net: Learning an explicit mapping for sample weighting.
In Advances in Neural
Information Processing Systems, pages 1917–1928, 2019.
[33] Hyun Oh Song, Yu Xiang, Stefanie Jegelka, and Silvio Savarese. Deep metric learning via lifted
structured feature embedding. In Computer Vision and Pattern Recognition (CVPR), 2016.
[34] Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The
implicit bias of gradient descent on separable data. The Journal of Machine Learning Research,
19(1):2822–2878, 2018.
[35] Jingru Tan, Changbao Wang, Buyu Li, Quanquan Li, Wanli Ouyang, Changqing Yin, and Junjie
Yan. Equalization loss for long-tailed object recognition. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.
[36] Yu-Xiong Wang, Deva Ramanan, and Martial Hebert. Learning to model the tail. In Advances
in Neural Information Processing Systems, pages 7029–7039, 2017.
[37] Han-Jia Ye, Hong-You Chen, De-Chuan Zhan, and Wei-Lun Chao. Identifying and compen-
sating for feature deviation in imbalanced deep learning. arXiv preprint arXiv:2001.01385,
2020.
[38] X. Zhang, Z. Fang, Y. Wen, Z. Li, and Y. Qiao. Range loss for deep face recognition with
long-tailed training data. In 2017 IEEE International Conference on Computer Vision (ICCV),
pages 5419–5428, 2017.
[39] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A
10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 2017.
12

--- Page 13 ---
Appendix A
Proofs and Derivations
A.1
Proof to Theorem 1
The exponential family parameterization of the multinomial distribution gives us the standard Softmax
function as the canonical response function
φj =
eηj
Pk
i=1 eηi
(14)
and also the canonical link function
ηj = log( φj
φk
)
(15)
We begin by adding a term −log(φj/ˆφj) to both sides of Eqn. 15,
ηj −log φj
ˆφj
= log( φj
φk
) −log(φj
ˆφj
) = log(
ˆφj
φk
)
(16)
Subsequently,
φke
ηj−log
φj
ˆ
φj = ˆφj
(17)
φk
k
X
i=1
e
ηi−log φi
ˆ
φi =
k
X
i=1
ˆφi = 1
(18)
φk = 1/
k
X
i=1
e
ηi−log φi
ˆ
φi
(19)
Substitute Eqn. 19 back to Eqn. 17, we have
ˆφj = φke
ηj−log
φj
ˆ
φj =
e
ηj−log
φj
ˆ
φj
Pk
i=1 e
ηi−log φi
ˆ
φi
(20)
Recall that
φj = p(y = j|x) = p(x|y = j)
p(x)
1
k ;
ˆφj = ˆp(y = j|x) = p(x|y = j)
ˆp(x)
nj
n
(21)
then
log φj
ˆφj
= log n
knj
+ log ˆp(x)
p(x)
(22)
Finally, bring Eqn. 22 back to Eqn. 20
ˆφj =
e
ηj−log
n
knj −log ˆ
p(x)
p(x)
Pk
i=1 eηi−log
n
kni −log ˆ
p(x)
p(x)
=
njeηj
Pk
i=1 nieηi
(23)
A.2
Derivation for the Multiple Binary Logistic Regression variant
Deﬁnition. Multiple Binary Logisitic Regression uses k binary logistic regression to do multi-class
classiﬁcation. Same as Softmax regression, the predicted label is the class with the maximum model
output,
ypred = arg max
j (ηj).
(24)
The only difference is that φj is expressed by a logistic function of ηj
φj =
eηj
1 + eηj
(25)
13

--- Page 14 ---
and the loss function sums up binary classiﬁcation loss on all classes
l(θ) =
k
X
j=1
−log ˜φj
(26)
where
˜φj =
φj,
if y = j
1 −φj,
otherwise
(27)
Setup. By the virtue of Bayes’ theorem, φj and 1 −φj can be decomposed as
φj = p(x|y = j)p(y = j)
p(x)
;
1 −φj = p(x|y ̸= j)p(y ̸= j)
p(x)
(28)
and for ˆφ and 1 −ˆφ,
ˆφj = p(x|y = j)ˆp(y = j)
ˆp(x)
;
1 −ˆφj = p(x|y ̸= j)ˆp(y ̸= j)
ˆp(x)
(29)
Derivation. Again, we introduce the exponential family parameterization and have the following
link function for φj
ηj = log
φj
1 −φj
(30)
Bring the decomposition Eqn. 28 and Eqn.29 into the link function above
ηj = log(
ˆφj
1 −ˆφj
· φj
ˆφj
· 1 −ˆφj
1 −φj
)
(31)
ηj = log(
ˆφj
1 −ˆφj
· p(x|y = j)p(y = j)/p(x)
p(x|y = j)ˆp(y = j)/ˆp(x) · p(x|y ̸= j)ˆp(y ̸= j)/ˆp(x)
p(x|y ̸= j)p(y ̸= j)/p(x))
(32)
Simplify the above equation
ηj = log(
ˆφj
1 −ˆφj
· p(y = j)
ˆp(y = j) · ˆp(y ̸= j)
p(y ̸= j))
(33)
Substitute the nj in to the equation above
ηj = log(
ˆφj
1 −ˆφj
· n/k
nj
· n −nj
n −n/k )
(34)
Then
ηj −log(n/k
nj
· n −nj
n −n/k ) = log(
ˆφj
1 −ˆφj
)
(35)
Finally, we have
ˆφj =
e
ηj−log( n/k
nj ·
n−nj
n−n/k )
1 + e
ηj−log( n/k
nj ·
n−nj
n−n/k )
(36)
Remark. A careful implementation should be made for instance segmentation tasks. As discussed in
[35], suppressing background samples’ gradient leads to a large number of false positives. Therefore,
we restrict our loss to foreground samples, while applying the standard Sigmoid function to back-
ground samples, and ignore the constant
n/k
n−n/k to avoid penalizing the background class. Please
refer to our code for the above-mentioned implementation details.
14

--- Page 15 ---
A.3
Proof to Theorem 2
Setup. Firstly, we deﬁne f as,
f(x) := −l(θ) + t
(37)
where l(θ) and t is previously deﬁned in the main paper.
Let errj(t) be the 0-1 loss on example from class j
errj(t) =
Pr
(x,y)∈Sj[f(x) < 0] =
Pr
(x,y)∈Sj[l(θ) > t]
(38)
and errγ,j(t) be the 0-1 margin loss on example from class j
errγ,j(t) =
Pr
(x,y)∈Sj[f(x) < γj] =
Pr
(x,y)∈Sj[l(θ) + γj > t]
(39)
Let ˆ
errγ,j(t) denote the empirical variant of errγ,j(t).
Proof. For any δ > 0 and with probability at least 1 −δ, for all γj > 0, and f ∈F, Theorem 2 in
[17] directly gives us
errj(t) ≤ˆ
errγ,j(t) + 4
γj
ˆRj(F) +
s
log(log2
4B
γj )
nj
+
s
log(1/δ)
2nj
(40)
where sup(x,y)∈S |l(θ) −t| ≤B and ˆRj(F) denotes the empirical Rademacher complexity of
function family F. By applying [4]’s analysis on the empirical Rademacher complexity and union
bound over all classes, we have the generalization error bound for the loss on a balanced test set
errbal(t) ≤1
k
k
X
j=1

ˆ
errγ,j(t) + 4
γj
s
C(F)
nj
+ ϵj(γj)

(41)
where
ϵj(γj) ≜
s
log(log2
4B
γj )
nj
+
s
log(1/δ)
2nj
(42)
is a low-order term of nj. To minimize the generalization error bound Eqn. 40, we essentially need
to minimize
k
X
j=1
4
γj
s
C(F)
nj
(43)
By constraining the sum of γ as Pk
j=1 γj = β, we can directly apply Cauchy-Schwarz inequality to
solve the optimal γ
γ∗
j =
βn−1/4
j
Pk
i=1 n−1/4
i
.
(44)
A.4
Proof to Corollary 2.1
Preliminary. Notice that ˆl∗
j(θ) = lj(θ) + γ∗
j can not be achieved for all class j, since −log ˆφ∗
j =
−log φj + γ∗
j and γ∗
j > 0 implies
ˆφ∗
j < φj;
k
X
j=1
ˆφ∗
j <
k
X
j=1
φj = 1
(45)
The equation above contradicts the deﬁnition that the sum of ˆφ∗should be exactly equal to 1. To
solve the contradiction, we introduce a term γbase > 0, such that
−log ˆφ∗
j = −log φj −γbase + γ∗
j ;
k
X
j=1
ˆφ∗
j = 1
(46)
15

--- Page 16 ---
To justify the new term γbase, we recall the deﬁnition of error
errγ,j(t) =
Pr
(x,y)∈Sj[l(θ) + γj > t];
errbal(t) =
Pr
(x,y)∈Sbal[l(θ) > t]
(47)
If we tweak the threshold t with the term γbase
errγ,j(t + γbase) =
Pr
(x,y)∈Sj[l(θ) + γj > t + γbase] =
Pr
(x,y)∈Sj[(l(θ) −γbase) + γj > t]
(48)
errbal(t + γbase) =
Pr
(x,y)∈Sbal[l(θ) > t + γbase] =
Pr
(x,y)∈Sbal[(l(θ) −γbase) > t]
(49)
As γ∗is not a function of t, the value of γ∗will not be affected by the tweak. Thus, instead of looking
for ˆl∗
j(θ) = lj(θ) + γ∗
j that minimizes the generalization bound for errbal(t), we are in fact looking
for ˆl∗
j(θ) = (lj(θ) −γbase) + γ∗
j that minimizes generalization bound for errbal(t + γbase)
Proof. In this section, we show that ˆlj in the corollary is an approximation of ˆl∗
j.
ˆlj(θ) −(lj(θ) −γbase) = log φj −log ˆφj + γbase
(50)
= log
eηj
Pk
i=1 eηi −log
eηj−log γ∗
j
Pk
i=1 eηi−log γ∗
i + γbase
(51)
= log
eηj
Pk
i=1 eηi −log
eηj
Pk
i=1 eηi−log γ∗
i +log γ∗
j + γbase
(52)
= log
k
X
i=1
eηi−log γ∗
i +log γ∗
j −log
k
X
i=1
eηi + γbase
(53)
= (
k
X
i=1
eηi−log γ∗
i +log γ∗
j −
k
X
i=1
eηi)/α + γbase
(Mean-Value Theorem)
(54)
= (γ∗
j
k
X
i=1
1
γ∗
i
eηi −
k
X
i=1
eηi)/α + γbase
(55)
≥(γ∗
j
β (
k
X
i=1
e
1
2 ηi)2 −
k
X
i=1
eηi)/α + γbase
(Cauchy-Schwarz Inequality)
(56)
= (γ∗
j
λ
β
k
X
i=1
eηi −
k
X
i=1
eηi)/α + γbase
(1 ≤λ ≤k)
(57)
≈γ∗
j
(let β = 1, γbase = 1)
(58)
(59)
where α =
d
dx log(x′) for some x′ in between Pk
i=1 eηi−log γ∗
i +log γ∗
j and Pk
i=1 eηi, λ is close to 1
when the model converges. Although the approximation holds under some constraints, we show that
it approximately minimizes the generalization bound derived in the last section.
16

--- Page 17 ---
A.5
Derivation for Eqn.12
Gradient for positive samples:
∂ˆl(s)
y=j(θ)
∂θj
=
∂−log ˆφ(s)
j
∂θj
(60)
=
∂−log
e
θT
j f(x(s))+log nj
Pn
i=1 eθT
i f(x(s))+log ni
∂θj
(61)
= −∂θT
j f(x(s)) + log nj
∂θj
+ ∂log Pn
i=1 eθT
i f(x(s))+log ni
∂θj
(62)
= −f(x(s)) + f(x(s))
eθT
j f(x(s))+log nj
Pn
i=1 eθT
i f(x(s))+log ni
(63)
= −f(x(s)) + f(x(s))ˆφ(s)
j
(64)
= f(x(s))(ˆφ(s)
j
−1)
(65)
Gradient for negative samples:
∂ˆl(s)
y̸=j(θ)
∂θj
= ∂−log ˆφ(s)
y
∂θj
(66)
=
∂−log
eθT
y f(x(s))+log ny
Pn
i=1 eθT
i f(x(s))+log ni
∂θj
(67)
= −∂θT
y f(x(s)) + log ny
∂θj
+ ∂log Pn
i=1 eθT
i f(x(s))+log ni
∂θj
(68)
= f(x(s))
eθT
j f(x(s))+log nj
Pn
i=1 eθT
i f(x(s))+log ni
(69)
= f(x(s))ˆφ(s)
j
(70)
Overall gradients on the training dataset:
n
X
s=1
l(s)(θ) =
nj
X
s=1
l(s)
y=j(θ) +
k
X
i̸=j
ni
X
s=1
l(s)
y=i(θ)
(71)
=
nj
X
s=1
f(x(s))(ˆφ(s)
j
−1) +
k
X
i̸=j
ni
X
s=1
f(x(s))ˆφ(s)
j
(72)
With Class-Balanced Sampling (CBS), number of samples in each class is equalized and therefore
changed from ni and nj to B/k
B
X
s=1
l(s)(θ) =
B/k
X
s=1
f(x(s))(ˆφ(s)
j
−1) +
k
X
i̸=j
B/k
X
s=1
f(x(s))ˆφ(s)
j
(73)
Set the overall gradient of a training batch to be zero gives
B/k
X
s=1
f(x(s))(1 −ˆφ(s)
j ) −
k
X
i̸=j
B/k
X
s=1
f(x(s))ˆφ(s)
j
= 0
(74)
17

--- Page 18 ---
We can also rewrite the equation using empirical expectation
1
nj
E(x+,y=j)∼Dtrain[f(x+)(1 −ˆφj)] −
k
X
i̸=j
1
ni
E(x−,y=i)∼Dtrain[f(x−)ˆφj] = 0
(75)
Then we make the following approximation when the training loss is close to 0, i.e., ˆφy →1
lim
ˆφy→1
nyeηy
nyeηy + Pk
i̸=y nieηi = 1
(76)
lim
ˆφy→1
1
1 + Pk
i̸=y
ni
ny eηi−ηy = 1
(77)
lim
ˆφy→1
k
X
i̸=y
ni
ny
eηi−ηy = 0
(78)
lim
ˆφy→1
k
X
i̸=y
eηi−ηy = 0
(79)
for positive samples:
lim
ˆφy=j→1
ˆφj/φj =
lim
ˆφy=j→1
nyeηy
nyeηy + Pk
i̸=y nieηi /
eηy
eηy + Pk
i̸=y eηi
(80)
=
lim
ˆφy=j→1
nyeηy
eηy
·
eηy + Pk
i̸=y eηi
nyeηy + Pk
i̸=y nieηi
(81)
=
lim
ˆφy=j→1
ny · 1
ny
·
1 + Pk
i̸=y eηi−ηy
1 + Pk
i̸=y
ni
ny eηi−ηy
(82)
=
lim
ˆφy=j→1
ny · 1
ny
· 1 + 0
1 + 0
(83)
= 1
(84)
for negative samples:
lim
ˆφy̸=j→1
ˆφj/φj =
lim
ˆφy̸=j→1
njeηj
nyeηy + Pk
i̸=y nieηi /
eηj
eηy + Pk
i̸=y eηi
(85)
=
lim
ˆφy̸=j→1
njeηj
eηj
·
eηy + Pk
i̸=y eηi
nyeηy + Pk
i̸=y nieηi
(86)
=
lim
ˆφy̸=j→1
nj · 1
ny
·
1 + Pk
i̸=y eηi−ηy
1 + Pk
i̸=y
ni
ny eηi−ηy
(87)
=
lim
ˆφy̸=j→1
nj · 1
ny
· 1 + 0
1 + 0
(88)
= nj/ny
(89)
Therefore, when ˆφy →1, Eqn.75 can be expanded as
1
nj
E(x+,y=j)∼Dtrain[f(x+)(1 −φj)] −
k
X
i̸=j
1
ni
E(x−,y=i)∼Dtrain[f(x−)φj
nj
ni
] ≈0
(90)
That is
1
n2
j
E(x+,y=j)∼Dtrain[f(x+)(1 −φj)] −
k
X
i̸=j
1
n2
i
E(x−,y=i)∼Dtrain[f(x−)φj] ≈0
(91)
18

--- Page 19 ---
Appendix B
Detailed Description for Meta Sampler and Meta Reweighter
B.1
Meta Sampler
To estimate the optimal sample rate, we ﬁrst make the sampler differentiable. Normally, class-
balanced samplers take following steps:
1. Deﬁne a class sample distribution π = π1{y=1}
1
π1{y=2}
2
. . . π1{y=k}
k
.
2. Assign πj to all instance-label pairs (x, y = j) and normalize over the dataset, to give the
instance sample distribution ρ = ρ1{i=1}
1
ρ1{i=2}
2
. . . ρ1{i=n}
n
.
3. Draw discrete image indexes from ρ to form a batch with a size b.
4. Augment the images and feed images into a model.
The steps where discrete sampling and image augmentation happen are usually not differentiable. We
propose a simple yet effective method to back-propagate the gradient directly from the loss to the
learnable sample rates.
Firstly, we use the Straight-through Gumbel Estimator [16] to approximate the gradient through the
multinomial sampling:
sj =
((log ρj + gj)/τ)
Pn
i=1 exp((log(ρi + gi)/τ))
(92)
where s is the sample result, g is i.i.d. samples drawn from Gumbel(0, 1) and τ is the temperature
coefﬁcient. Straight-through means that we use argmax to discretize s to (0,1) during forward and use
∇s during backward. Gumbel-Softmax re-parameterization is commonly found to have less variance
in gradient estimation than score functions [16].
Then, we use an external memory to connect sampler with loss. We use the Straight-through Gumbel
Estimator to draw b discrete samples from ρ, we denote as sb×n. sb×n is matrix of a n-dimensional
one-hot vectors, representing b selected images. Concretely, for the i-th sample, if the Gumbel
Estimator gives a sampling result to be c-th image, we have s(i) to be
s(i)
j
=
1,
if j = c
0,
otherwise
(93)
We save this matrix into an external memory during data preparation. After obtaining the classiﬁcation
loss l(θ), which is the i-th loss in the batch computed from the c-th sample, we re-weight the loss by
˜l(i)(θ) = l(i)(θ) · s(i)
c
(94)
Notice that the re-weight will not change the loss value, it only connects sampling results with the
classiﬁcation loss in the computation graph. By doing so, the gradient from the loss can directly
reach the learnable sample rate π.
B.2
Meta Reweighter
Since one image might contain multiple instances from several categories, we use Meta Reweighter,
rather than Meta Sampler on the LVIS dataset. Speciﬁcally, we assign the loss weight for instance
i to be ρi = πj, where π is a learnable class weight and j is the class label of instance i. Next, we
perform similar bi-level optimization as in Meta Sampler, where we re-weight the loss of an instance
by its loss weight ρi instead of a discrete 0-1 sampling result si.
Appendix C
Implementation Details
C.1
Hardware
We use Intel Xeon Gold 6148 CPU @ 2.40GHz with Nvidia V100 GPU for model training. We take
a single GPU to train models on CIFAR-10-LT, CIRFAR-100-LT, ImageNet-LT and Places-LT, and 8
GPUs to train models on LVIS.
19

--- Page 20 ---
C.2
Software
We implement our proposed algorithm with PyTorch-1.3.0 [29] for all experiments. Second-order
derivatives are computed with Higher [8] library.
C.3
Training details
Decoupled Training. Through the paper, we refer to decoupled training as training the last linear
classiﬁer on a ﬁxed feature extractor obtained from instance-balanced training.
Meta Sampler/Reweighter. We apply Meta Sampler/Reweighter only when decoupled training to
save computational costs. We start them at the beginning of the decoupled training with no deferment.
CIFAR-10-LT and CIFAR-100-LT. All experiments use ResNet-32 as backbone like [6]. We use
Nesterov SGD with momentum 0.9 and weight-decay 0.0005 for training. We use a total mini-batch
size of 512 images on a single GPU. The learning rate increased from 0.05 to 0.1 in the ﬁrst 800
iterations. Cosine scheduler [27] is applied afterward, with a minimum learning rate of 0. Our
augmentation follows [35]. In testing, the image size is 32x32. In end-to-end training, the model
is trained for 13K iterations. In decoupled training experiments, we ﬁx the Softmax model, i.e.,
the instance-balanced baseline model obtained from the previous end-to-end training, as the feature
extractor. And the classiﬁer is trained for 2K iterations. For Meta Sampler and Meta Reweighter,
we use Adam[20] with betas (0.9, 0.99) and weight decay 0. The learning rate is set to 0.01 with no
warm-up strategy or scheduler applied. The meta-set is formed by randomly sampling 512 images
from the training set with replacement, using Class-Balanced Sampling.
ImageNet-LT and Places-LT. We follow the setup in [18] for decoupled classiﬁer retraining. We
ﬁrst train a base model without any bells and whistles following Kang et al. [18] for these two
datasets. For ImageNet-LT, the model is trained for 90 epochs from scratch. For Places-LT, we
choose ResNet-152 as the backbone network pre-trained on the full ImageNet-2012 dataset and train
it on Places-LT following Kang et al [18]. For both datasets, we use SGD optimizer with momentum
0.9, batch size 512, cosine learning rate schedule [27] decaying from 0.2 to 0 and image resolution
224 × 224.
After obtaining the base model, we retrain the last linear classiﬁer. For Meta Sampler, we use
Adam[20] with betas (0.9, 0.99) and weight decay 0. The learning rate is set to 0.01 with no warm-up
strategy and is kept unchanged during the training process. The meta-set is formed by randomly
sampling 512 images from the training set with replacement, using Class-Balanced Sampling. For
ImageNet-LT, we use SGD optimizer with momentum 0.9, batch size 512, cosine learning rate
schedule decaying from 0.2 to 0 for 10 epochs. For Places-LT, we use SGD optimizer with momentum
0.9, batch size 128, cosine learning rate schedule decaying from 0.01 to 0 for 10 epochs.
For the training process, we resize the image to 224 × 224. During testing, we ﬁrst resize the image
to 256 × 256 and do center-crop to obtain an image of 224 × 224.
LVIS. We use the off-the-shelf model Mask R-CNN with the backbone network ResNet-50 for LVIS.
The backbone network is pre-trained on ImageNet. We follow the setup (including Repeat Factor
Sampling) from the original dataset paper [9] for two baseline models (Softmax and Sigmoid). We
use an SGD optimizer with 0.9 momentum, 0.01 initial learning rate, and 0.0001 weight decay. The
model is trained for 90k iterations with 8 images per mini-batch. The learning rate is dropped by a
factor of 10 at both 60k iterations and 80k iterations.
Methods other than baselines are trained under the decoupled training scheme, with the above-
mentioned models as the base model. Slightly different from the decoupled training for classiﬁcation
tasks [18], we ﬁne-tune the bounding box classiﬁer (one fully connected layer) instead of retraining
it from scratch. This signiﬁcantly saves the training time. We use an SGD optimizer with 0.9
momentum, 0.02 initial learning rate, and 0.0001 weight decay. The model is trained for 22k
iterations with 8 images per mini-batch. The learning rate is dropped by a factor of 10 at both 11k
iterations and 18k iterations.
For our method with a Meta Reweighter, we use Adam optimizer with 0.001 for the Meta Reweighter
and train the Meta Reweighter together with the model. The learning rate is kept unchanged during
the training process.
20

--- Page 21 ---
We apply scale jitter and random ﬂip at training time (sampling image scale for the shorter side from
640, 672, 704, 736, 768, 800). For testing, images are resized to a shorter image edge of 800 pixels;
no test-time augmentation is used.
C.4
Meta-learned sample rates with Softmax and Balanced Softmax
Figure 3 demonstrates that compared with standard Softmax function, Meta Sampler learns a more
balanced sample rates with our proposed Balanced Softmax. The sample rates for all the classes are
initialized with 0.5 and are constrained in the range of (0,1).
The blue bar represents the learned sample rates with standard Softmax. The sample rates of tail
classes approach 1 while the sample rates of head classes approach 0. Such an extreme divergence in
sample rates could potentially pose challenges to the meta-learning optimization process. A very low
optimal learning rate may also not be numerically stable.
With Balanced Softmax, we can see that Meta Sampler produces a more balanced distribution of
sample rates. After convergence, the sample rates for Softmax has a variance of 0.13. Balanced
Softmax signiﬁcantly reduces the variance to 0.03.
class 1
class 20
class 40
class 60
class 80 class 100
0.0
0.2
0.4
0.6
0.8
1.0
Softmax
Balanced Softmax
Figure 3: Learned sample rates with Meta-Sampler when training with Softmax and Balanced
Softmax. The experiment is on CIFAR-100-LT with imbalanced factor 200. The X-axis denotes
classes with a decreasing number of training samples. Y-axis denotes sample rates for different
classes. Balanced Softmax gives a smoother distribution compared to Softmax.
Appendix D
More Details Regarding Datasets
D.1
Basic information
We hereby provide more details about datasets mentioned in the paper in Table 6
All the datasets are publicly available for downloading, we provide the download link as follows:
ImageNet, CIFAR-10 and CIFAR-100, Places365, and LVIS.
D.2
Long-tailed datasets generation
CIFAR10-LT and CIFAR100-LT. We generated the long-tailed version of CIFAR-10 and CIFAR-
100 following Cui et al. [6]. For both the original CIFAR-10 and CIFAR-100, they contain 50000
training images and 10000 test images at a size of 32 × 32 uniformly distributed in 10 classes and
100 classes. The long-tailed version is created by randomly reducing training samples. In particular,
21

--- Page 22 ---
Dataset
#Classes
Imbalance Factor
#Train Instances
Head Class Size
Tail Class Size
CIFAR-10-LT [21]
10
10-200
50,000 – 11,203
5,000
500-25
CIFAR-100-LT [21]
100
10-200
50,000 – 9,502
500
50-2
ImageNet-LT [26]
1,000
256
115,846
1280
5
Places-LT [39]
365
996
62,500
4,980
5
LVIS [9]
1,230
26,148
693,958
26,148
1
Table 6: Details of long-tailed datatsets. Notice that for both CIFAR-10-LT and CIFAR-100-LT, the
number of tail class varies with different imbalance factors.
Dataset
CIFAR-10-LT
CIFAR-100-LT
Imbalance Factor
200
100
10
200
100
10
Focal Loss∗[25]
65.29
70.38
86.66
35.62
38.41
55.78
Class Balanced Loss∗[6]
68.89
74.57
87.49
36.23
39.60
57.99
L2RW∗[30]
66.51
74.16
85.19
33.38
40.23
53.73
LDAM† [4]
-
73.35
86.96
-
39.6
56.91
LDAM-DRW† [4]
-
77.03
88.16
-
42.04
58.71
Meta-Weight-Net∗[32]
68.89
75.21
87.84
37.91
42.09
58.46
Equalization Loss‡ [35]
-
-
-
43.38
-
-
BALMS
81.5
84.9
91.3
45.5
50.8
63.0
Table 7: Comparisons with reported SOTA results on Top 1 accuracy for CIFAR-LT. * indicates
results reported in [32]. † indicates results reported in [4]. ‡ indicates results reported in [35].
the number of samples in the y-th class is nyµy, where ny is the original number of training samples
in the class and µ ∈(0, 1). By varying µ, we generate three training sets with the imbalance factors
of 200, 100, and 10. The test set is kept unchanged and balance.
ImageNet-LT. We use the long-tailed version of ImageNet from Liu et al. [26]. It is created by
ﬁrstly sampling the class sizes from a Pareto distribution with the power value α = 6, followed by
sampling the corresponding number of images for each class. The ImageNet-LT dataset has 115,846
training images in 1,000 classes, and its imbalance factor is 256 as shown in Table 6. The original
ImageNet [7] validation set is used as the test set, which contains 50 images for each class.
Places-LT. In a similar spirit to the long-tailed ImageNet, a long-tailed version of the Places-365
dataset is generated using the same strategy as above. It contains 62,500 training images from 365
classes with an imbalance factor 996. In the test set, there are 100 test images for each class.
LVIS. We use ofﬁcial training and validation split from LVIS [9]. No modiﬁcation is made.
Appendix E
Comparisons with Reported SOTA Results on CIFAR-LT
We used our reproduced results on CIFAR-LT in the empirical analysis section in the paper since
prior works chose different baselines and cannot be fairly compared with. Table 7 compares our
method with more results originally reported in corresponding papers.
22

--- Page 23 ---
Appendix F
More Visualizations and Analysis
F.1
Visualization and analysis on the feature space of Balanced Softmax
Recent work [18] shows that instance-balanced training results in the best feature space in practice.
In this section, we use t-SNE to visualize the feature space created by Balanced Softmax. The result
is shown in Fig. 4. The following pattern can be observed: CBS and Balanced Softmax tend to have
a more concentrated center area compared to the Softmax baseline. This indicates that the Softmax
baseline’s features are more suitable for the classiﬁcation task than Balanced Softmax and CBS’s.
Further empirical analysis in Table 8 advocates the claim.
Feature Training
Classiﬁer Training
Accuracy
Softmax
Softmax
69.53
Softmax+CBS
Softmax
57.06
Balanced Softmax
Softmax
65.75
Softmax
Softmax+CBS
76.59
Softmax+CBS
Softmax+CBS
63.96
Balanced Softmax
Softmax+CBS
75.35
Softmax
Balanced Softmax
78.53
Softmax+CBS
Balanced Softmax
68.24
Balanced Softmax
Balanced Softmax
77.04
Table 8: Comparison of decoupled training results with features from Softmax and Balanced Softmax.
The experiment is on CIFAR-10-LT with imbalanced factor 200. The Softmax pretrained features
generally outperform the Balanced Softmax pretrained features.
Softmax
Softmax+CBS
Balanced Softmax
Figure 4: t-SNE visualization of the feature space created by different methods. The experiment is
on CIFAR-10-LT with imbalanced factor 200. The 10 colors represent the 10 classes. Compared to
Softmax, Softmax+CBS and Balanced Softmax have a more concentrated center area, making them
less suitable for classiﬁcation.
23

--- Page 24 ---
F.2
Visualization of re-sampling’s effect towards training
We use a two-dimensional, three-way classiﬁcation example to demonstrate re-sampling’s effect on
training a one-layer linear classiﬁer either with standard Softamx or with Balanced Softmax. The
result, shown in Figure 5, conﬁrms that the linear classiﬁer’s solution is unaffected by re-sampling.
Meanwhile, different re-sampling strategies have different effects on the optimization process, where
CBS causes the over-balance problem to Balanced Softmax’s optimization.
Method
Iterations
Softmax
Softmax
+ CBS
Balanced Softmax
Balanced Softmax
+ CBS
1000
5000
25000
125000
Figure 5: Visualization of decision boundaries over iterations with different training setups. We
create an imbalanced, two-dimensional, dummy dataset of three classes: red, yellow and blue. The
red point represents 10000 red samples, the yellow point represents 100 yellow samples and the blue
point represents 1 blue sample. Background shading shows the decision surface. Both Softmax and
Softmax+CBS converge to symmetric decision boundaries, and Softmax+CBS converges faster than
Softmax. Note that symmetric decision boundaries do not optimize for the generalization error bound
on an imbalanced dataset [4]. Both Balanced Softmax and Balanced Softmax+CBS converge to a
better solution: they successfully push the decision boundary from the minority class toward the
majority class. Compared to Balanced Softmax, Balanced Softmax+CBS shows the over-balance
problem: its optimization is dominated by the minority class.
24
