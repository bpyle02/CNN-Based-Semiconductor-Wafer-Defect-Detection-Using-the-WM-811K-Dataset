# Disentangling Label Distribution for Long-Tailed Visual Recognition

**Authors**: Hong, Duan, Chen, Li, Wang, Lin
**Year**: 2021
**arXiv**: 2012.00321
**Topic**: long_tail
**Relevance**: LADE: label-distribution-aware estimation for calibrated long-tail classification

---


--- Page 1 ---
Disentangling Label Distribution for Long-tailed Visual Recognition
Youngkyu Hong* Seungju Han* Kwanghee Choi* Seokjun Seo
Beomsu Kim
Buru Chang†
Hyperconnect
{youngkyu.hong,seungju.han,kwanghee.choi,seokjun.seo,beomsu.kim,buru.chang}@hpcnt.com
Abstract
The current evaluation protocol of long-tailed visual
recognition trains the classiﬁcation model on the long-
tailed source label distribution and evaluates its perfor-
mance on the uniform target label distribution. Such pro-
tocol has questionable practicality since the target may
also be long-tailed.
Therefore, we formulate long-tailed
visual recognition as a label shift problem where the tar-
get and source label distributions are different.
One of
the signiﬁcant hurdles in dealing with the label shift prob-
lem is the entanglement between the source label distri-
bution and the model prediction.
In this paper, we fo-
cus on disentangling the source label distribution from the
model prediction.
We ﬁrst introduce a simple but over-
looked baseline method that matches the target label dis-
tribution by post-processing the model prediction trained
by the cross-entropy loss and the Softmax function.
Al-
though this method surpasses state-of-the-art methods on
benchmark datasets, it can be further improved by di-
rectly disentangling the source label distribution from the
model prediction in the training phase. Thus, we propose
a novel method, LAbel distribution DisEntangling (LADE)
loss based on the optimal bound of Donsker-Varadhan rep-
resentation. LADE achieves state-of-the-art performance
on benchmark datasets such as CIFAR-100-LT, Places-LT,
ImageNet-LT, and iNaturalist 2018. Moreover, LADE out-
performs existing methods on various shifted target label
distributions, showing the general adaptability of our pro-
posed method.
1. Introduction
Based on large-scale datasets such as ImageNet [50],
COCO [34], and Places [64], deep neural networks have
achieved signiﬁcant progress in various visual recogni-
tion tasks, including classiﬁcation [22, 52], object detec-
tion [47, 18], and segmentation [48]. In contrast to these
relatively balanced datasets, real-world data often exhibit
*Equal contribution. Author ordering determined by coin ﬂip.
†Corresponding author.
Cross-entropy
𝒑𝒔(𝒚)
𝒑𝒕(𝒚)
𝒑𝒔(𝒚|𝒙)
Training Phase
Inference Phase
Discrepancy
𝒑𝒔(𝒚|𝒙)
v
Entanglement
𝒑𝒔(𝒚)
𝒑𝒕(𝒚)
𝒑𝒔(𝒚|𝒙)
𝒑(𝒙|𝒚)
𝒑(𝒙)
Disentanglement
Injection
𝒑𝒕(𝒚|𝒙)
𝒑(𝒙|𝒚)
𝒑(𝒙)
Injection
Adaptation
LADE (Ours)
Training Phase
Inference Phase
v
Figure 1: A comparison between the cross-entropy loss and
our proposed LADE loss in long-tailed visual recognition.
After training with the cross-entropy loss, the model predic-
tion gets entangled with the source label distribution ps(y),
which causes a discrepancy with the target label distribu-
tion pt(y) during the inference phase. Our proposed LADE
disentangles ps(y) from the model prediction so that it can
adapt to the arbitrary target probability by injecting pt(y).
long-tailed distribution where head (major) classes occupy
most of the data, while tail (minor) classes have a hand-
ful of samples [58, 38]. Unfortunately, the performance of
state-of-the-art classiﬁcation models degrades on datasets
following the long-tailed distribution [7, 21, 60].
To tackle this problem, many long-tailed visual recogni-
tion methods [7, 21, 25, 8, 51, 62, 9] have been proposed.
These methods compare their effectiveness by (1) training
on the long-tailed source label distribution ps(y) and (2)
evaluating on the uniform target label distribution pt(y).
However, we argue that this evaluation protocol is often im-
practical as it is natural to assume that pt(y) could be the
arbitrary distribution such as uniform distribution [50] and
long-tailed distribution [17, 10].
From this perspective, we are motivated to explore a new
method that adapts the model to the arbitrary pt(y). In this
paper, we borrow the concept of the label distribution shift
problems [16, 36, 56] to the long-tailed visual recognition
1
arXiv:2012.00321v2  [cs.CV]  20 Mar 2021

--- Page 2 ---
0
20
40
60
80
100
0.0
0.2
0.4
0.6
0.8
1.0
(Average) Probability
Avg. Prob.
ps(y)
pt(y)
Class index (Head to tail)
(a) Cross-entropy
0
20
40
60
80
100
0.0
0.2
0.4
0.6
0.8
1.0
(Average) Probability
Avg. Prob.
ps(y)
pt(y)
Class index (Head to tail)
(b) LADE
Figure 2: The average probability for each class calculated
on the balanced test set.
The model is ResNet-32 [22]
trained on CIFAR-100-LT [9], which has uniform target
label distribution pt(y) while the source dataset has long-
tailed distribution ps(y). (a) depicts the average probabil-
ity when trained with the cross-entropy loss, which shows
average probability correlates with ps(y), resulting in the
discrepancy with pt(y). (b) depicts the average probabil-
ity when trained and inferred with LADE, which shows the
ability of LADE on adapting to pt(y).
task.
However, it is problematic to directly use the model pre-
diction p(y|x; θ) which is ﬁtted to the source probability
ps(y|x), as the target probability pt(y|x) is shifted from
ps(y) to pt(y) (Figure 1). Figure 2a shows the entanglement
between the model prediction and ps(y) when the model
is trained by the cross-entropy (CE) loss and the Softmax
function. To alleviate this problem, we focus on disentan-
gling ps(y) from the model outputs so that the shifted target
label distribution pt(y) can be injected to estimate the target
probability.
We shed light on a simple yet strong baseline,
called Post-Compensated Softmax (PC Softmax) that post-
processes the model prediction to disentangle ps(y) from
p(y|x; θ) and then incorporate pt(y) to the disentangled
model output probability.
Despite the simplicity of the
method, PC Softmax outperforms state-of-the-art methods
in long-tailed visual recognition (will be described in Sec-
tion 4). Although this observation demonstrates the effec-
tiveness of the disentanglement in the inference phase, PC
Softmax can be further improved by directly disentangling
ps(y) in the training phase.
Thus, we propose a novel method, LAbel distribution
DisEntangling (LADE) loss. LADE utilizes the Donsker-
Varadhan (DV) representation [15] to directly disentangle
ps(y) from p(y|x; θ). Figure 2b shows that LADE disentan-
gles ps(y) from p(y|x; θ). We claim that the disentangle-
ment in the training phase shows even better performance
on adapting to arbitrary target label distributions.
We conduct several experiments to compare our pro-
posed method with existing long-tailed visual recognition
methods, and show that LADE achieves state-of-the-art
performance on benchmark datasets such as CIFAR-100-
LT [30], Places-LT [38], ImageNet-LT [38], and iNaturalist
2018 [57]. Moreover, we demonstrate that the classiﬁcation
model trained with LADE can cope with arbitrary pt(y) by
evaluating the performance on datasets with various shifted
pt(y). We further show that our proposed LADE can also
be effective in terms of conﬁdence calibration. Our contri-
butions in this paper are summarized as follows:
• We introduce a simple yet strong baseline method, PC
Softmax, which outperforms state-of-the-art methods
in long-tailed visual recognition benchmark datasets.
• We propose a novel loss called LADE that directly dis-
entangles the source label distribution in the training
phase so that the model effectively adapts to arbitrary
target label distributions.
• We show that LADE achieves state-of-the-art perfor-
mance in long-tailed visual recognition on various tar-
get label distributions.
2. Related work
2.1. Long-tailed visual recognition
Most long-tailed visual recognition methods can be di-
vided into two strategies: modifying the data sampler to
balance the class frequency during optimization [7, 21, 25,
8, 51], and modifying the class-wise weights of the classiﬁ-
cation loss to increase the importance of tail classes in terms
of empirical risk minimization [1, 65, 37, 40, 25, 21, 7].
Both strategies suffer from under-representation of head
classes or memorization of tail classes [62, 9]. To overcome
these problems, [23, 28, 9, 63] introduce strategies for pre-
venting deteriorated representation learning caused by re-
balancing. [60, 38] utilize knowledge from head classes
to learn tail classes. [29, 59] augment tail class samples
while preserving the diversity of dataset. [24] applies do-
main adaptation on learning tail class representation.
Recent approaches introduce advanced re-balancing
methods for better accommodation of tail class samples.
[13] calculates the effective number of samples per each
class to re-balance the loss. [9] enforces a greater margin
from the decision boundary for tail classes. [55] disentan-
gles feature learning from the confounding effect in mo-
mentum by backdoor adjustment.
2.2. Label distribution shift
In this paper, we cope with long-tailed visual recogni-
tion as one of the label distribution shift problems.
We
summarize recently proposed studies in label distribution
shift problems that assume ps(y) ̸= pt(y) and ps(x|y) =
pt(x|y). [36] estimates the degree of label shift using a
black box predictor. [16] extends the work of [36] with
2

--- Page 3 ---
the expectation-maximization algorithm. [2] introduces a
domain adaptation to handle label distribution shifts by es-
timating importance weights. [46] adjusts the logits before
applying the Softmax function by the frequency of each
class considering the uniform target label distribution.
2.3. Donsker-Varadhan representation
Donsker-Varadhan (DV) representation [15] is the dual
variational representation of Kullback-Leibler (KL) diver-
gence [32]. It is proven that the optimal bound of the DV
representation is the log-likelihood ratio of two distributions
of the KL divergence [3, 4].
The usefulness of the DV
representation has been broadly shown in the area includ-
ing mutual information estimation [4, 35, 45] or generative
models [4, 44]. However, [54, 12, 41] have pointed out the
instability of directly using the DV representation. To avoid
the issue, we use the regularized DV representation from
[12] to approximate network logits as the log-likelihood ra-
tio log(p(x|y)/p(x)). To the best of our knowledge, this is
the ﬁrst attempt to utilize the optimal bound inside the DV
representation in the long-tailed visual recognition.
3. Method
3.1. Preliminaries
We start by revisiting the most common loss for train-
ing the Softmax regression (also known as the multinomial
logistic regression) model [5], namely the CE loss:
p(y|x; θ) =
efθ(x)[y]
P
c efθ(x)[c]
(1)
LCE(fθ(x), y) = −log(p(y|x; θ)),
(2)
where x is the input image and y is the target label, ps(x, y)
and pt(x, y) are the source (train) and target (test) data dis-
tributions, and fθ(x)[y] is the logit of class y of the model.
The Softmax regression model estimates ps(y|x) and works
well when the source and the target label distribution are
the same, i.e. ps(y) = pt(y).
However, we focus on the label shift problem [16, 36]
where the target label distribution is shifted from the source
label distribution, i.e.
ps(x|y) = pt(x|y) but ps(y) ̸=
pt(y). Since the model prediction estimates ps(y|x), it can-
not be used to predict the shifted distribution. This is due to
the strong coupling between ps(y|x) and ps(y), as justiﬁed
from the Bayes’ rule:
ps(y|x) = ps(y)ps(x|y)
ps(x)
=
ps(y)ps(x|y)
P
c ps(c)ps(x|c).
(3)
3.2. PC Softmax: Post-Compensated Softmax
The straightforward way to handle the label distribution
shift is by replacing ps(y) with pt(y). We introduce a post-
compensation (PC) strategy that modiﬁes the logit in the
inference phase:
Deﬁnition 3.1 (Post-Compensation Strategy) The
Post-
Compensation strategy modiﬁes model logits as follows:
f P C
θ
(x)[y] = fθ(x)[y] −log ps(y) + log pt(y)
(4)
where ps(y) is the distribution which the model logits are
entangled with, and pt(y) is the target distribution that the
model tries to incorporate with.
Note that the concept of the PC strategy is not entirely new
since [40, 7, 26] previously covered as a different form of
multiplying pt(y)/ps(y) to the output probability. How-
ever, our PC strategy does not violate the categorical prob-
ability assumption, i.e. P
c pt(y = c|x) = 1.
We apply the PC strategy to the Softmax regression
model, which we call Post-Compensated Softmax (PC Soft-
max). For the Softmax regression model, the PC strategy is
the proper adjustment for estimating target data distribution.
Theorem 1 (Post-Compensated Softmax).
Let ps(x, y)
and pt(x, y) be the source and target data distributions, re-
spectively. If fθ(x)[y] is the logit of class y from the Soft-
max regression model estimating ps(y|x), then the estima-
tion of pt(y|x) is formulated as:
pt(y|x; θ) =
pt(y)
ps(y) · efθ(x)[y]
P
c
pt(c)
ps(c) · efθ(x)[c]
(5)
=
e(fθ(x)[y]−log ps(y)+log pt(y))
P
c e(fθ(x)[c]−log ps(c)+log pt(c))
(6)
=
ef P C
θ
(x)[y]
P
c ef P C
θ
(x)[c] .
(7)
Proof. See the Supplementary Material.
We emphasize that PC Softmax becomes a strong base-
line that surpasses previous state-of-the-art long-tailed vi-
sual recognition methods. However, recent literature does
not consider this as a baseline.
PC Softmax can also be viewed as an extension of Bal-
anced Softmax [46], which modiﬁes the Softmax function
to accommodate the uniform target label distribution in the
training phase. In contrast, PC Softmax modiﬁes the model
logits in the inference phase to match the arbitrary target
label distribution pt(y).
3.3. LADER: LAbel distribution DisEntangling
Regularizer
Performance gain from the PC strategy shows the efﬁ-
cacy of disentangling the source label distribution. How-
ever, the PC strategy does not involve the disentanglement
in the training phase, which we claim as the ingredient for
better adaptability to arbitrary target label distributions. To
achieve this, we design a new modeling objective that works
3

--- Page 4 ---
as a substitute for ps(y|x). We derive the new objective in
two steps: (1) detaching ps(y) from ps(y|x), which results
in ps(x|y)/ps(x), and (2) replacing ps(y) in ps(x) with the
uniform prior pu(y), i.e. pu(y = c) = 1/C, where C is the
total number of classes.
Finally, the modeling objective for the model logits is:
fθ(x)[y] = log pu(x|y)
pu(x) .
(8)
We utilize the optimal form of the regularized Donsker-
Varadhan (DV) representation [15, 3, 4, 12] to model the
log-likelihood ratio above explicitly.
Theorem 2 (Optimal form of the regularized DV represen-
tation). Let P, Q be arbitrary distributions with supp(P) ⊆
supp(Q). Suppose for every function T : Ω→R on some
domain Ω, the function T that minimizes the regularized
DV representation is the log-likelihood ratio of P and Q:
log dP
dQ = arg max
T :Ω→R
(EP[T] −log(EQ[eT ])
−λ(log(EQ[eT ]))2),
(9)
for any λ ∈R+ when the expectations are ﬁnite.
Proof. See Subsection 7.2 from [12].
By plugging P = pu(x|y) and Q = pu(x) into Equa-
tion 9 and choosing the function family of T : Ω→R to
be parametrized by the logits of the deep neural network,
the optimal fθ(x)[y] approaches to the target objective in
Equation 8:
log pu(x|y)
pu(x) ≥arg max
fθ
(Ex∼pu(x|y)[fθ(x)[y]]
−log Ex∼pu(x)[efθ(x)[y])]
−λ(log(Ex∼pu(x)[efθ(x)[y])]))2).
(10)
Since the exact estimations of the expectation with re-
spect to pu(x|y) and pu(x) are intractable, we use the
Monte Carlo approximation [49] using a single batch:
Ex∼pu(x|c)[fθ(x)[c]] ≈1
Nc
N
X
i=1
1yi=c · fθ(xi)[c] (11)
Ex∼pu(x)[efθ(x)[c]] = E(x,y)∼ps(x,y)[pu(y)
ps(y) efθ(x)[c]] (12)
≈1
N
N
X
i=1
pu(yi)
ps(yi) · efθ(xi)[c], (13)
where xi and yi are i-th sample and label, respectively, N is
the total number of samples, and Nc is the number of sam-
ples for class c. In Equation 12, importance sampling [27]
is used to approximate the expectation with respect to pu(x)
using samples from ps(x):
pu(x)
ps(x) =
P
c pu(x|c)pu(c)
P
c ps(x|c)ps(c) = pu(y)
ps(y) ,
(14)
for the sample label pair (x, y) ∼ps(x, y), where we as-
sume ps(x|c) = 0 for c ̸= y.
Finally, we derive a novel loss that regularizes the logits
to approach Equation 8 by applying Equation 11, 12, and
13 to Equation 10:
Deﬁnition 3.2 (LADER) For a single batch of sample-
label pairs (xi, yi) with i = 1, ..., N, LAbel distribution
DisEntangling Regularizer (LADER) is deﬁned as follows:
LLADERc = −1
Nc
N
X
i=1
1yi=c · fθ(xi)[c]
+ log( 1
N
N
X
i=1
pu(yi)
ps(yi) · efθ(xi)[c])
+ λ(log( 1
N
N
X
i=1
pu(yi)
ps(yi) · efθ(xi)[c]))2
(15)
LLADER =
X
c∈S
αc · LLADERc,
(16)
with nonnegative hyperparameters λ, α1, . . . , αC, where C
is total number of classes, Nc is the number of samples of
class c and S is the set of classes existing inside the batch.
Empirically, we ﬁnd out that regularizing the head classes
more strongly than the tail classes is more effective. Thus,
we apply αc = ps(y = c) as the weight for the regularizer
of class c, LLADERc in Equation 16.
3.4. Deriving the conditional probability from dis-
entangled logits
LADER regularizes the logits to be log(pu(x|y)/pu(x))
to ensure the logits are explicitly disentangled from the
source label distribution ps(y). To estimate the conditional
probability pt(y|x) of the arbitrary data distribution pt(x, y)
from the regularized logits, we use the modiﬁed Softmax
function derived from the Bayes’ rule with the assumption
of pt(x|y) = pu(x|y):
pt(y|x; θ) =
pt(y)pt(x|y; θ)
P
c pt(c)pt(x|c; θ)
=
pt(y)pu(x|y; θ)
P
c pt(c)pu(x|c; θ) =
pt(y) · efθ(x)[y]
P
c pt(c) · efθ(x)[c] .
(17)
Similar to this, we can estimate ps(y|x) by swapping
pt(y) of Equation 17 with ps(y), so that ps(y|x; θ) can be
optimized by the CE loss. Thus, we can combine LADER
with the CE loss as our ﬁnal loss for training:
4

--- Page 5 ---
Deﬁnition 3.3 (LADE) LAbel distribution DisEntangling
(LADE) loss is deﬁned as follows:
LLADE−CE(fθ(x), y) = −log(ps(y|x; θ))
(18)
= −log(
ps(y) · efθ(x)[y]
P
c ps(c) · efθ(x)[c] )
(19)
LLADE(fθ(x), y) = LLADE−CE(fθ(x), y)
+α · LLADER(fθ(x), y),
(20)
where α is a nonnegative hyperparameter, which deter-
mines the regularization strength of LLADER.
Note that Balanced Softmax [46] is equivalent to LADE
with α = 0, but LADE is derived from an entirely different
perspective of directly regularizing the logits. Furthermore,
Balanced Softmax only covers the uniform target label dis-
tribution, while our method is designed to cover arbitrary
target label distributions without re-training.
In the inference phase, we inject the target label distribu-
tion as in Equation 17.
4. Experiments
We compare PC Softmax and LADE with current state-
of-the-art methods. First, we evaluate the performance on
the uniform target label distribution, which is the prevalent
evaluation scheme of long-tailed visual recognition. Then,
we assess the performance on variously shifted target label
distributions. Finally, we conduct further analysis to show
that LADE successfully disentangles the source label dis-
tribution and improves conﬁdence calibration. We provide
source codes1 of LADE for the reproduction of the experi-
ments conducted in this paper. Details of the hyperparam-
eter tuning process and the results of the ablation test are
reported in the Supplementary Material.
4.1. Experimental setup
Long-tailed dataset
We follow the common evaluation
protocol [38, 9, 63, 55] in long-tailed visual recognition,
which trains classiﬁcation models on the long-tailed source
label distribution and evaluates their performance on the
uniform target label distribution. We use four benchmark
datasets with at least 100 classes to simulate the real-world
long-tailed data distribution: CIFAR-100-LT [9], Places-
LT [38], ImageNet-LT [38], and iNaturalist 2018 [57]. We
deﬁne the imbalance ratio as Nmax/Nmin, where N is the
number of samples in each class. CIFAR-100-LT has three
variants with controllable data imbalance ratios 10, 50, and
100. The details of datasets are summarized in Table 1.
1https://github.com/hyperconnect/LADE
Table 1:
The details of the training set of long-tailed
datasets.
Dataset
# of classes
# of samples
Imbalance ratio
CIFAR-100-LT
100
50K
{10, 50, 100}
Places-LT
365
62.5K
996
ImageNet-LT
1K
186K
256
iNaturalist 2018
8K
437K
500
Comparison with other methods.
We compare PC Soft-
max and LADE with three categories of methods:
• Baseline methods.
For our baseline, we use Softmax
(Equation 1), Focal loss (Focal) [33], OLTR [38], CB-
Focal [13], and LDAM [9].
• Two-stage training. To demonstrate our method’s efﬁ-
ciency and effectiveness, we compare our method with
two-staged state-of-the-art methods that employ a ﬁne-
tuning strategy. LDAM+DRW [9] applies a ﬁne-tuning
step with loss re-weighting. Decouple [28] re-balances
the classiﬁer during the ﬁne-tuning stage.
• Other state-of-the-art methods.
BBN [63], Causal
Norm [55], and Balanced Softmax [46] are recently pro-
posed state-of-the-art methods on long-tail visual recog-
nition. BBN uses an extra additional network branch to
deal with an imbalanced training set. Causal Norm uti-
lizes backdoor adjustment to remove indirect causal effect
caused by imbalanced source label distribution.
Evaluation Protocol.
We report evaluation results us-
ing top-1 accuracy. Following [38], for ImageNet-LT and
Places-LT, we categorize the classes into three groups de-
pending on the number of samples of each class and further
report each group’s evaluation results. The three groups are
deﬁned as follows: Many covering classes with > 100 im-
ages, Medium covering classes with ≥20 and ≤100 im-
ages, Few covering classes with < 20 images.
4.2. Results on balanced test label distribution
Evaluation
results
on
CIFAR-100-LT,
Places-LT,
ImageNet-LT, and iNaturalist 2018 are shown in Table 2,
3, 4, and 5, respectively. All the datasets have a uniform
target label distribution.
PC Softmax shows comparable
or better results than the previous state-of-the-art results
on benchmark datasets.
This result is quite surprising
considering the simplicity of PC Softmax. Our proposed
method, LADE, achieves even better performance in long-
tailed visual recognition on all four benchmark datasets,
advancing the state-of-the-art even further.
5

--- Page 6 ---
Table 2: Top-1 accuracy on CIFAR-100-LT with differ-
ent imbalance ratios. Rows with † denote results directly
borrowed from [55]. We use the same backbone network
with [55].
Dataset
CIFAR-100 LT
Imbalance ratio
100
50
10
Focal Loss†
38.4
44.3
55.8
LDAM†
42.0
46.6
58.7
BBN†
42.6
47.0
59.1
Causal Norm†
44.1
50.3
59.6
Balanced Softmax
45.1
49.9
61.6
Softmax
41.0
45.5
59.0
PC Softmax
45.3
49.5
61.2
LADE
45.4
50.5
61.7
Table 3: The performances on Places-LT [38], starting from
an ImageNet pre-trained ResNet-152. Rows with † denote
results directly borrowed from [28].
Method
Many
Medium
Few
All
Focal Loss†
41.1
34.8
22.4
34.6
OLTR†
44.7
37.0
25.3
35.9
Decouple-τ-norm†
37.8
40.7
31.8
37.9
Decouple-LWS†
40.6
39.1
28.6
37.6
Causal Norm
23.8
35.8
40.4
32.4
Balanced Softmax
42.0
39.3
30.5
38.6
Softmax
46.4
27.9
12.5
31.5
PC Softmax
43.0
39.1
29.6
38.7
LADE
42.8
39.0
31.2
38.8
CIFAR-100-LT
Table 2 shows the evaluation results on
CIFAR-100-LT. As shown in the table, in CIFAR-100-LT,
LADE outperforms all the baselines over all the imbalance
ratios.
PC Softmax also shows better performance than
other methods except for Balanced Softmax and our pro-
posed LADE.
Places-LT
We further evaluate PC Softmax and LADE
on Places-LT, and Table 3 shows the experimental re-
sults. LADE achieves a new state-of-the-art of 38.8% top-
1 overall accuracy, without using a two-stage training as
in Decouple-τ-norm [28]. PC Softmax shows yet another
promising result by surpassing the previous state-of-the-art,
while Softmax offers poor results. This result is quite im-
pressive since both models are the same, and the only dif-
ference occurs in the inference phase.
ImageNet-LT
We conduct experiments on ImageNet-LT
to demonstrate the effectiveness of LADE in the large-scale
dataset. We observe that the model is under-ﬁtting at 90
epochs when using LADE. Previous works [28, 63] train
model for longer epochs to deal with under-ﬁtting. Thus, we
Table 4: The performances on ImageNet-LT [38]. Rows
with § denote results directly borrowed from [55].
Method
Many
Medium
Few
All
90 epochs
Focal Loss§
64.3
37.1
8.2
43.7
OLTR§
51.0
40.8
20.8
41.9
Decouple-cRT§
61.8
46.2
27.4
49.6
Decouple-τ-norm§
59.1
46.9
30.7
49.4
Decouple-LWS§
60.2
47.2
30.3
49.9
Causal Norm§
62.7
48.8
31.6
51.8
Balanced Softmax
62.2
48.8
29.8
51.4
Softmax
65.1
35.7
6.6
43.1
PC Softmax
60.4
46.7
23.8
48.9
LADE
62.3
49.3
31.2
51.9
180 epochs
Causal Norm
65.2
47.7
29.8
52.0
Balanced Softmax
63.6
48.4
32.9
52.1
Softmax
68.1
41.9
14.4
48.2
PC Softmax
63.9
49.1
34.3
52.8
LADE
65.1
48.9
33.4
53.0
Table 5: Top-1 accuracy over all classes on iNaturalist 2018.
Rows with † denote results directly borrowed from [28] and
⋆denotes the result directly borrowed from [63].
Method
Top-1 Accuracy
CB-Focal†
61.1
LDAM†
64.6
LDAM+DRW†
68.0
Decouple-τ-norm†
69.3
Decouple-LWS†
69.5
BBN⋆
69.6
Causal Norm
63.9
Balanced Softmax
69.8
Softmax
65.0
PC Softmax
69.3
LADE
70.0
also report the evaluation results at both 90 and 180 epochs,
respectively, and this is different from [55] where they train
the baseline methods during 90 epochs. Table 4 presents the
performance of our method on ImageNet-LT. LADE yields
53.0% top-1 overall accuracy with 180 epochs, which is
better than the previous state-of-the-art, including Causal
Norm trained with 180 epochs. PC Softmax also shows a
favorable result, 52.8%, where it also outperforms the pre-
vious state-of-the-art-results. Besides, LADE still achieves
the best result compared to the methods when LADE and
the methods are trained for 90 epochs.
iNaturalist 2018
To show the scalability of LADE on a
large-scale dataset, we evaluate our methods in the real-
world long-tailed dataset, iNaturalist 2018. Since iNatural-
ist 2018 does not contain a validation set, we train the model
6

--- Page 7 ---
Table 6: Top-1 accuracy over all classes on test time shifted ImageNet-LT. All models are trained for 180 epochs.
Dataset
Forward
Uniform
Backward
Imbalance ratio
50
25
10
5
2
1
2
5
10
25
50
Causal Norm
64.1
62.5
60.1
57.8
54.6
52.0
49.3
45.8
43.4
40.4
38.4
Balanced Softmax
62.5
60.9
58.8
57.0
54.4
52.1
49.6
46.5
44.1
41.4
39.7
Softmax
66.3
63.9
60.4
57.1
52.3
48.2
44.2
38.9
35.0
30.5
27.9
PC Causal Norm
66.7
64.3
60.9
58.1
54.6
52.0
49.8
47.9
47.0
46.7
46.7
PC Balanced Softmax
65.5
63.1
59.9
57.3
54.3
52.1
50.2
48.8
48.3
48.5
49.0
PC Softmax
66.6
63.9
60.6
58.1
55.0
52.8
51.0
49.3
48.8
48.5
49.0
LADE
67.4
64.8
61.3
58.6
55.2
53.0
51.2
49.8
49.2
49.3
50.0
with LADE for 200 epochs and report the test accuracy at
200 epochs, following [28]. Table 5 shows the top-1 accu-
racy over all classes on iNaturalist 2018. LADE reaches
the best accuracy 70.0% among the other methods, even
without any branch structure of ﬁne-tuning as BBN [63],
nor a two-stage training scheme as [28]. Further, LADE
surpasses PC Softmax with a large gap, +0.7%, where PC
Softmax still shows a competitive result compared to other
methods. This result indicates that PC Softmax is effec-
tive for small datasets but performance worsens for larger
datasets, while LADE scales well on large datasets as well.
4.3. Results on variously shifted test label distribu-
tions
Test sets are rarely well-balanced in real-world scenar-
ios. To simulate and compare the performance of various
state-of-the-art methods in the wild, we propose a more re-
alistic evaluation protocol. We ﬁrst train the model on the
long-tailed source label distribution. Then we examine the
performance on a range of target label distributions, from
distributions that resemble the source label distributions to
radically different distributions, similar to [2, 36].
We choose the ImageNet-LT from Section 4.2 as the
source dataset. The test set of ImageNet-LT is uniformly
distributed, and each class has 50 samples; hence the max-
imum imbalance ratio is 50. Similar to constructing the
CIFAR-LT training dataset [13, 9], we additionally design
two types of test datasets.
Let us assume ImageNet-LT
classes are sorted by descending values of the number of
samples per class. Then the shifted test dataset is deﬁned as
follows: (1) Forward. nj = N · µ(j−1)/C. As the imbal-
ance ratio increases, it becomes similar to the source label
distribution. (2) Backward. nj = N · µ(C−j)/C. The or-
der is ﬂipped so that it gets more different as the imbalance
ratio increases. Here µ is the imbalance ratio, N is the num-
ber of samples per class in the original ImageNet-LT test set
(= 50), C is the number of classes, j is the class index and
1 ≤j ≤C, and nj is the number of samples in class j for
the shifted test set.
We compare LADE with Softmax, Balanced Softmax,
and Causal Norm on this evaluation protocol. For a fair
comparison, we apply our PC strategy to state-of-the-art
methods: log pt(y) −log pu(y) is added to the logits be-
fore applying the softmax function for Balanced Softmax,
and Causal Norm, as both methods target the uninformative
prior. Theorem 1 is used for the Softmax instead.
Table 6 shows the top-1 accuracy in the range of test
datasets between the Forward- and Backward-type datasets.
Our PC strategy shows consistent performance gain, which
indicates the beneﬁts of plug-and-play target label distribu-
tions. Moreover, LADE outperforms all the other methods
in every imbalance settings, and the performance gap be-
tween LADE and PC Softmax gets wider as the dataset gets
more imbalanced. These results demonstrate the general
adaptability of our proposed method on real-world scenar-
ios, where the target label distribution is variously shifted.
4.4. Further analysis
Visualization of the logit values
In this subsection, we
visualize the logit values of each class in order to demon-
strate the effect of LADE. By disentangling the source la-
bel distribution with LADE as described in Section 3.3, the
logit value fθ(x)[y] should converge to log C for the posi-
tive samples:
fθ(x)[y] = log pu(x|y)
pu(x) = log pu(y|x)
pu(y)
= log C,
(21)
where we assume perfectly separable case, i.e. pu(y|x) = 1
for (x, y) ∼pu(x, y) and C is the number of classes.
Figure 3 shows how the logits are distributed for each
class. The hyperparameter α represents the regularization
strength for LADER. As α increases, the logit values grad-
ually converge to the theoretical value y = log C = log 100
(dotted line in the ﬁgure), reconﬁrming Theorem 2. This re-
sult indicates that LADER successfully regularizes the logit
values as we intended.
Conﬁdence calibration
Previous literature claims that
the conﬁdence of the neural network classiﬁer does not rep-
resent its true accuracy [20]. We regard a classiﬁer is well-
7

--- Page 8 ---
0
25
50
75
100
  
15
10
5
0
5
10
15
20
25
PC Softmax
0
25
50
75
100
LADE with  = 0.0
0
25
50
75
100
LADE with  = 0.001
0
25
50
75
100
LADE with  = 0.01
0
25
50
75
100
LADE with  = 0.1
Positive samples
Negative samples
Logit size
Class index (Head to tail)
Figure 3: ResNet-32 model logits correspond to each class, where the model is trained on CIFAR-100-LT with an imbalance
ratio of 100. For each class c, positive samples denote the sample corresponds to class c, and negative samples denote the
other. The colored area denotes the variance of logit values, while the line indicates the mean.
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
ECE: 0.1082
Accuracy: 0.5200
Causal Norm
Ideal
Outputs
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
ECE: 0.0615
Accuracy: 0.5213
Balanced Softmax
Ideal
Outputs
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
ECE: 0.0567
Accuracy: 0.5276
PC Softmax
Ideal
Outputs
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
ECE: 0.0346
Accuracy: 0.5301
LADE
Ideal
Outputs
Confidence
Figure 4: Reliability diagrams of ResNeXt-50-32x4d [61] on ImageNet-LT. The average conﬁdence of the model trained by
LADE nearly matches its accuracy.
calibrated when its predictive probability, maxy p(y|x),
represents the true probability [20].
For example, when
a calibrated classiﬁer predicts the label y with predictive
probability 0.75, it has a 75% chance of being correct. It
will be catastrophic if we cannot trust the conﬁdence of the
neural network in the domain of medical diagnosis [11] and
self-driving car [6].
[42] suspect the endlessly growing logit values induced
by a combination of naive CE loss and the Softmax function
as the culprit of over-conﬁdence. Since LADER regular-
izes the logit size, we expect that using LADE prevents the
model from being over-conﬁdent. Through experiments, we
observe that LADE improves calibration. Figure 4 shows
the reliability diagrams with 20-bins. Using expected cal-
ibration error (ECE) [43], we quantitatively measure the
miscalibration rate of the model trained on ImageNet-LT.
We compare our LADE with PC Softmax and the current
state-of-the-art methods, Causal Norm [55], and Balanced
Softmax [46]. Results show that LADE produces a more
calibrated classiﬁer than other methods, with the ECE of
0.0346, which conﬁrms our expectation.
5. Conclusion
In this paper, we suggest that disentangling the source
label distribution from the model prediction is useful for
long-tailed visual recognition. To disentangle the source la-
bel distribution, we ﬁrst introduce a simple yet strong base-
line, called PC Softmax, that matches the target label dis-
tribution by post-processing the model prediction trained
by the cross-entropy loss and the Softmax function. We
further propose a novel loss, LADE, that directly disentan-
gles the source label distribution in the training phase based
on the optimal bound of Donsker-Varadhan representation.
Experiment results demonstrate that PC Softmax and our
proposed LADE outperform state-of-the-art long-tailed vi-
sual recognition methods on real-world benchmark datasets.
Furthermore, LADE achieves state-of-the-art performance
on various shifted target label distributions. Lastly, further
experiments show that our proposed LADE is also effective
in terms of conﬁdence calibration. We plan to extend our
research to other vision domain problems that suffer from
long-tailed distributions, such as object detection and seg-
mentation.
8

--- Page 9 ---
References
[1] Rehan Akbani, Stephen Kwek, and Nathalie Japkowicz. Ap-
plying support vector machines to imbalanced datasets. In
European conference on machine learning, pages 39–50.
Springer, 2004. 2
[2] Kamyar Azizzadenesheli, Anqi Liu, Fanny Yang, and An-
imashree Anandkumar.
Regularized learning for domain
adaptation under label shifts. In International Conference
on Learning Representations, 2019. 3, 7
[3] Arindam Banerjee. On bayesian bounds. In Proceedings
of the 23rd international conference on Machine learning,
pages 81–88, 2006. 3, 4
[4] Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajesh-
war, Sherjil Ozair, Yoshua Bengio, Aaron Courville, and De-
von Hjelm. Mutual information neural estimation. In Inter-
national Conference on Machine Learning, pages 531–540,
2018. 3, 4
[5] Christopher M Bishop.
Pattern recognition and machine
learning. springer, 2006. 3
[6] Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski,
Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D
Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, et al.
End to end learning for self-driving cars.
arXiv preprint
arXiv:1604.07316, 2016. 8
[7] Mateusz Buda, Atsuto Maki, and Maciej A Mazurowski. A
systematic study of the class imbalance problem in convo-
lutional neural networks. Neural Networks, 106:249–259,
2018. 1, 2, 3
[8] Jonathon Byrd and Zachary Lipton. What is the effect of
importance weighting in deep learning?
In International
Conference on Machine Learning, pages 872–881, 2019. 1,
2
[9] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga,
and Tengyu Ma. Learning imbalanced datasets with label-
distribution-aware margin loss. In Advances in Neural Infor-
mation Processing Systems, pages 1567–1578, 2019. 1, 2, 5,
7
[10] Jo˜ao Carreira, Eric Noland, Andras Banki-Horvath, Chloe
Hillier, and Andrew Zisserman. A short note about kinetics-
600. CoRR, abs/1808.01340, 2018. 1
[11] Rich Caruana, Yin Lou, Johannes Gehrke, Paul Koch, Marc
Sturm, and Noemie Elhadad. Intelligible models for health-
care: Predicting pneumonia risk and hospital 30-day read-
mission. In Proceedings of the 21th ACM SIGKDD interna-
tional conference on knowledge discovery and data mining,
pages 1721–1730, 2015. 8
[12] Kwanghee Choi and Siyeong Lee. Regularized mutual infor-
mation neural estimation. arXiv preprint arXiv:2011.07932,
2020. 3, 4, 13
[13] Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge
Belongie. Class-balanced loss based on effective number of
samples. In Proceedings of the IEEE Conference on Com-
puter Vision and Pattern Recognition, pages 9268–9277,
2019. 2, 5, 7
[14] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In 2009 IEEE conference on computer vision and
pattern recognition, pages 248–255. Ieee, 2009. 12
[15] MD Donsker and SRS Varadhan. Large deviations for sta-
tionary gaussian processes. Communications in Mathemati-
cal Physics, 97(1-2):187–210, 1985. 2, 3, 4
[16] Saurabh Garg, Yifan Wu, Sivaraman Balakrishnan, and
Zachary C. Lipton.
A uniﬁed view of label shift estima-
tion. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Had-
sell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Ad-
vances in Neural Information Processing Systems 33: An-
nual Conference on Neural Information Processing Systems
2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. 1,
2, 3
[17] Krzysztof J. Geras, Stacey Wolfson, S. Gene Kim, Linda
Moy, and Kyunghyun Cho. High-resolution breast cancer
screening with multi-view deep convolutional neural net-
works. CoRR, abs/1703.07047, 2017. 1
[18] Ross Girshick. Fast r-cnn. In Proceedings of the IEEE inter-
national conference on computer vision, pages 1440–1448,
2015. 1
[19] Priya Goyal, Piotr Doll´ar, Ross Girshick, Pieter Noord-
huis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch,
Yangqing Jia, and Kaiming He.
Accurate, large mini-
batch sgd: Training imagenet in 1 hour.
arXiv preprint
arXiv:1706.02677, 2017. 12
[20] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger.
On calibration of modern neural networks. In Proceedings
of the 34th International Conference on Machine Learning-
Volume 70, pages 1321–1330, 2017. 7, 8
[21] Haibo He and Edwardo A Garcia. Learning from imbalanced
data. IEEE Transactions on knowledge and data engineer-
ing, 21(9):1263–1284, 2009. 1, 2
[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 770–778, 2016. 1, 2, 12, 15
[23] Chen Huang, Yining Li, Chen Change Loy, and Xiaoou
Tang. Learning deep representation for imbalanced classiﬁ-
cation. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 5375–5384, 2016. 2
[24] Muhammad Abdullah Jamal, Matthew Brown, Ming-Hsuan
Yang, Liqiang Wang, and Boqing Gong. Rethinking class-
balanced methods for long-tailed visual recognition from
a domain adaptation perspective.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 7610–7619, 2020. 2
[25] Nathalie Japkowicz and Shaju Stephen.
The class imbal-
ance problem: A systematic study. Intelligent data analysis,
6(5):429–449, 2002. 1, 2
[26] Justin M Johnson and Taghi M Khoshgoftaar.
Survey on
deep learning with class imbalance. Journal of Big Data,
6(1):27, 2019. 3
[27] Herman Kahn. Use of different monte carlo sampling tech-
niques. 1955. 4
[28] Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan,
Albert Gordo, Jiashi Feng, and Yannis Kalantidis. Decou-
pling representation and classiﬁer for long-tailed recogni-
9

--- Page 10 ---
tion. In International Conference on Learning Representa-
tions, 2019. 2, 5, 6, 7, 12
[29] Jaehyung Kim, Jongheon Jeong, and Jinwoo Shin. M2m:
Imbalanced classiﬁcation via major-to-minor translation. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 13896–13905, 2020. 2
[30] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple
layers of features from tiny images. 2009. 2, 12
[31] Meelis Kull, Miquel Perello Nieto, Markus K¨angsepp,
Telmo Silva Filho, Hao Song, and Peter Flach. Beyond tem-
perature scaling: Obtaining well-calibrated multi-class prob-
abilities with dirichlet calibration. In Advances in Neural
Information Processing Systems, volume 32. Curran Asso-
ciates, Inc., 2019. 13
[32] Solomon Kullback and Richard A Leibler.
On informa-
tion and sufﬁciency. The annals of mathematical statistics,
22(1):79–86, 1951. 3
[33] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and
Piotr Doll´ar. Focal loss for dense object detection. In Pro-
ceedings of the IEEE international conference on computer
vision, pages 2980–2988, 2017. 5
[34] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence
Zitnick. Microsoft coco: Common objects in context. In
European conference on computer vision, pages 740–755.
Springer, 2014. 1
[35] Xiao Lin, Indranil Sur, Samuel A Nastase, Ajay Di-
vakaran, Uri Hasson, and Mohamed R Amer.
Data-
efﬁcient mutual information neural estimator. arXiv preprint
arXiv:1905.03319, 2019. 3
[36] Zachary C. Lipton, Yu-Xiang Wang, and Alexander J.
Smola. Detecting and correcting for label shift with black
box predictors. In Jennifer G. Dy and Andreas Krause, ed-
itors, Proceedings of the 35th International Conference on
Machine Learning, ICML 2018, Stockholmsm¨assan, Stock-
holm, Sweden, July 10-15, 2018, volume 80 of Proceedings
of Machine Learning Research, pages 3128–3136. PMLR,
2018. 1, 2, 3, 7
[37] Xu-Ying Liu and Zhi-Hua Zhou. The inﬂuence of class im-
balance on cost-sensitive learning: An empirical study. In
Sixth International Conference on Data Mining (ICDM’06),
pages 970–974. IEEE, 2006. 2
[38] Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang,
Boqing Gong, and Stella X Yu.
Large-scale long-tailed
recognition in an open world. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition,
pages 2537–2546, 2019. 1, 2, 5, 6, 12
[39] Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient
descent with warm restarts. In 5th International Conference
on Learning Representations, ICLR 2017, Toulon, France,
April 24-26, 2017, Conference Track Proceedings. OpenRe-
view.net, 2017. 12
[40] Dragos Margineantu. When does imbalanced data require
more than cost-sensitive learning.
In Proceedings of the
AAAI’2000 Workshop on Learning from Imbalanced Data
Sets, pages 47–50, 2000. 2, 3
[41] David McAllester and Karl Stratos. Formal limitations on the
measurement of mutual information. In International Con-
ference on Artiﬁcial Intelligence and Statistics, pages 875–
884, 2020. 3
[42] Rafael M¨uller, Simon Kornblith, and Geoffrey E Hinton.
When does label smoothing help?
In Advances in Neural
Information Processing Systems, pages 4694–4703, 2019. 8
[43] Mahdi Pakdaman Naeini, Gregory F Cooper, and Milos
Hauskrecht.
Obtaining well calibrated probabilities using
bayesian binning.
In Proceedings of the... AAAI Confer-
ence on Artiﬁcial Intelligence. AAAI Conference on Artiﬁcial
Intelligence, volume 2015, page 2901. NIH Public Access,
2015. 8
[44] Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-
gan: Training generative neural samplers using variational
divergence minimization. In Advances in neural information
processing systems, pages 271–279, 2016. 3
[45] Ben Poole, Sherjil Ozair, A¨aron van den Oord, Alex Alemi,
and George Tucker. On variational bounds of mutual infor-
mation. In ICML, 2019. 3
[46] Jiawei Ren, Cunjun Yu, shunan sheng, Xiao Ma, Haiyu
Zhao, Shuai Yi, and hongsheng Li. Balanced meta-softmax
for long-tailed visual recognition. In H. Larochelle, M. Ran-
zato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances
in Neural Information Processing Systems, volume 33, pages
4175–4186. Curran Associates, Inc., 2020. 3, 5, 8
[47] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
Faster r-cnn: Towards real-time object detection with region
proposal networks. In Advances in neural information pro-
cessing systems, pages 91–99, 2015. 1
[48] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-
net: Convolutional networks for biomedical image segmen-
tation. In International Conference on Medical image com-
puting and computer-assisted intervention, pages 234–241.
Springer, 2015. 1
[49] Reuven Y Rubinstein and Dirk P Kroese. Simulation and the
Monte Carlo method, volume 10. John Wiley & Sons, 2016.
4
[50] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, San-
jeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,
Aditya Khosla, Michael Bernstein, et al.
Imagenet large
scale visual recognition challenge. International journal of
computer vision, 115(3):211–252, 2015. 1
[51] Li Shen, Zhouchen Lin, and Qingming Huang. Relay back-
propagation for effective learning of deep convolutional neu-
ral networks. In European conference on computer vision,
pages 467–482. Springer, 2016. 1, 2
[52] Karen Simonyan and Andrew Zisserman. Very deep con-
volutional networks for large-scale image recognition.
In
Yoshua Bengio and Yann LeCun, editors, 3rd International
Conference on Learning Representations, ICLR 2015, San
Diego, CA, USA, May 7-9, 2015, Conference Track Proceed-
ings, 2015. 1
[53] Jasper Snoek, Yaniv Ovadia, Emily Fertig, Balaji Lakshmi-
narayanan, Sebastian Nowozin, D. Sculley, Joshua V. Dil-
lon, Jie Ren, and Zachary Nado. Can you trust your model’s
uncertainty? evaluating predictive uncertainty under dataset
10

--- Page 11 ---
shift. In Advances in Neural Information Processing Systems
32: Annual Conference on Neural Information Processing
Systems 2019, NeurIPS 2019, December 8-14, 2019, Van-
couver, BC, Canada, pages 13969–13980, 2019. 13
[54] Jiaming Song and Stefano Ermon. Understanding the limi-
tations of variational mutual information estimators. In In-
ternational Conference on Learning Representations, 2019.
3
[55] Kaihua Tang, Jianqiang Huang, and Hanwang Zhang. Long-
tailed classiﬁcation by keeping the good and removing the
bad momentum causal effect. Advances in Neural Informa-
tion Processing Systems, 33, 2020. 2, 5, 6, 8, 12
[56] Junjiao Tian, Yen-Cheng Liu, Nathaniel Glaser, Yen-Chang
Hsu, and Zsolt Kira. Posterior re-calibration for imbalanced
datasets. Advances in Neural Information Processing Sys-
tems, 33, 2020. 1
[57] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui,
Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and
Serge Belongie. The inaturalist species classiﬁcation and de-
tection dataset. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 8769–8778,
2018. 2, 5, 12
[58] Grant Van Horn and Pietro Perona.
The devil is in the
tails: Fine-grained classiﬁcation in the wild. arXiv preprint
arXiv:1709.01450, 2017. 1
[59] Xinyue Wang, Yilin Lyu, and Liping Jing. Deep generative
model for robust imbalance classiﬁcation. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 14124–14133, 2020. 2
[60] Yu-Xiong Wang, Deva Ramanan, and Martial Hebert. Learn-
ing to model the tail. In Advances in Neural Information
Processing Systems, pages 7029–7039, 2017. 1, 2
[61] Saining Xie, Ross Girshick, Piotr Doll´ar, Zhuowen Tu, and
Kaiming He. Aggregated residual transformations for deep
neural networks. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 1492–1500,
2017. 8, 12
[62] Yuzhe Yang and Zhi Xu. Rethinking the value of labels for
improving class-imbalanced learning. Advances in Neural
Information Processing Systems, 33, 2020. 1, 2
[63] Boyan Zhou, Quan Cui, Xiu-Shen Wei, and Zhao-Min
Chen. Bbn: Bilateral-branch network with cumulative learn-
ing for long-tailed visual recognition.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 9719–9728, 2020. 2, 5, 6, 7
[64] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva,
and Antonio Torralba. Places: A 10 million image database
for scene recognition. IEEE transactions on pattern analysis
and machine intelligence, 40(6):1452–1464, 2017. 1, 12
[65] Zhi-Hua Zhou and Xu-Ying Liu. Training cost-sensitive neu-
ral networks with methods addressing the class imbalance
problem. IEEE Transactions on knowledge and data engi-
neering, 18(1):63–77, 2005. 2
11

--- Page 12 ---
Appendix
6.1. Proof to Theorem 1
Assume pt(y|x) to be the target conditional probability
and ps(y|x) to be the source conditional probability. We
start with ps(y|x) formulated with logits fθ(x)[y]:
ps(y|x) =
efθ(x)[y]
P
c efθ(x)[c] .
(22)
By applying the log function on both sides,
fθ(x)[y] = log ps(y|x) + Cx
= log

ps(y)ps(x|y)
P
c ps(c)ps(x|c)

+ Cx
= log(ps(y)ps(x|y)) + C′
x
= log(ps(y)pt(x|y)) + C′
x
= log(pt(y)pt(x|y)) + log ps(y)
−log pt(y) + C′
x,
(23)
where Cx and C′
x can be regarded as constants for a ﬁxed x
as follows:
Cx = log
 X
c
efθ(x)[c]
!
,
(24)
C′
x = Cx −log
 X
c
ps(c)ps(x|c)
!
.
(25)
Let us derive the post-compensated logit f P C
θ
(Deﬁnition
3.1) from fθ:
log(pt(y)pt(x|y))
= fθ(x)[y] −log ps(y) + log pt(y) −C′
x
= f P C
θ
(x)[y] −C′
x.
(26)
Re-calculating the Softmax function yields:
ef P C
θ
(x)[y]
P
c ef P C
θ
(x)[c] =
ef P C
θ
(x)[y]−C′
x
P
c ef P C
θ
(x)[c]−C′
x
=
pt(y)pt(x|y)
P
c pt(c)pt(x|c)
= pt(y|x),
(27)
which ends the proof.
6.2. Implementation details
For all the experiments over multiple datasets, we use
the SGD optimizer with momentum γ = 0.9 and weight
decay 5 · 10−4 to optimize the network if not speciﬁed. We
use the same random seed throughout the whole experiment
for a fair comparison. For image classiﬁcation on CIFAR-
100-LT and ImageNet-LT, we follow most of the details
from [55], and on Places-LT and iNaturalist 2018, we fol-
low [28]. All the models are trained on 4 GPUs, except
CIFAR-100-LT, where we use 1 GPU. We ﬁnd the optimal
hyperparameters based on a grid search with the validation
set. However, as the iNaturalist 2018 dataset does not con-
tain the validation set, we use the same λ and α searched
on the ImageNet-LT dataset since it has a similar number
of classes and samples compared to the iNaturalist 2018
dataset. Detailed experiment settings for LADE are sum-
marized in Table 7.
Table 7: Experimental settings on four benchmark datasets
when using LADE. IB stands for the imbalance ratio.
Dataset
λ
α
Batch size
CIFAR-100-LT (IB 10)
0.01
0.01
256
CIFAR-100-LT (IB 50)
0.01
0.01
256
CIFAR-100-LT (IB 100)
0.01
0.1
256
Places-LT
0.1
0.005
128
ImageNet-LT
0.5
0.05
256
iNaturalist 2018
0.5
0.05
256
CIFAR-100-LT [30]
On the CIFAR-100-LT dataset, we
use ResNet-32 [22] as the backbone network for all the ex-
periments, following the implementation of [55]. We train
for 200 epochs and apply the linear warm-up learning rate
schedule [19] to the ﬁrst ﬁve epochs. The learning rate is
initialized as 0.2, and it is decayed at the 120th and 160th
epoch by 0.01.
Places-LT [64]
We use ResNet-152 [22] as the back-
bone network with pretraining on the ImageNet-2012 [14]
dataset. We use 0.05 and 0.001 for the initial learning rate
of the classiﬁer and the feature extractor. We train for 30
epochs with a learning rate decay of 0.1 every 10 epochs.
ImageNet-LT [14]
On the ImageNet-LT dataset, we uti-
lize ResNeXt-50-32x4d [61] as the backbone network for
all the experiments. We use the cosine learning rate sched-
ule [39] decaying from 0.05 to 0.0 during 180 epochs.
iNaturalist 2018 [57]
For the iNaturalist 2018 dataset, we
use ResNet-50 [22] as the backbone network for all experi-
ments. We use cosine learning rate scheduling [39] decay-
ing from 0.1 to 0.0 during 200 epochs, following [28].
Data Pre-processing
We follow [38] for the details on
image preprocessing. For the training set, images are re-
sized to 256 × 256 and randomly cropped to 224 × 224.
After cropping, we augment images with random horizon-
tal ﬂip with probability p = 0.5 and apply random color
12

--- Page 13 ---
jitter. For validation and test set, images are center cropped
to 224 × 224 without any augmentation.
6.3. Ablation study
To verify the effectiveness of the regularizer term for DV
representation (Equation 9) and LADER (Equation 16), we
conduct an ablation test. Table 8 shows how the top-1 ac-
curacy changes when removing the regularizer term for the
DV representation (λ = 0) or removing LADER (α = 0),
respectively.
Table 8: Ablation study for LADE on the long-tailed bench-
mark datasets. LADE (Ours) shows the best evaluation per-
formance, and λ = 0 and α = 0 denote the performance
with the same settings except for the DV representation reg-
ularization or LADER, respectively.
Dataset
LADE (Ours)
λ = 0
α = 0
CIFAR-100-LT (IB 10)
61.7
61.5
61.6
CIFAR-100-LT (IB 50)
50.5
49.5
49.9
CIFAR-100-LT (IB 100)
45.4
45.2
45.1
Places-LT
38.8
38.5
38.6
ImageNet-LT
53.0
47.0
52.1
iNaturalist 2018
70.0
58.3
69.8
[12] introduces λ to control the instability induced from
directly using the DV representation. The model suffers
a severe performance drop on ImageNet-LT and iNatural-
ist 2018 when the regularizer term for DV representation is
not used (λ = 0). α represents the regularization strength
of LADER on logits, as mentioned in Section 4.4. With-
out LADER (α = 0), performance degradation is observed,
demonstrating the efﬁcacy of LADER.
6.4. Additional results on variously shifted test label
distributions
In Section 4.3, we show that our LADE achieves state-
of-the-art performance on variously shifted test label dis-
tribution with ImageNet-LT, which is the large-scale long-
tailed dataset. We further conduct experiments on the small-
scale dataset, CIFAR-100-LT, to ensure the consistent ef-
fectiveness of our LADE loss. For the training set, we use
CIFAR-100-LT with an imbalance ratio of 50. The shifted
test set is constructed by the same setting in Section 4.3. As
shown in Table 9, LADE outperforms all the other methods,
which is consistent with the results on ImageNet-LT (Table
6). We can also reconﬁrm the effectiveness of the PC strat-
egy. These results from CIFAR-100-LT and ImageNet-LT
imply that our PC strategy and LADE work well on both
small-scale and large-scale datasets.
6.5. Additional conﬁdence calibration results
We report the additional results of LADE against other
methods in the perspective of conﬁdence calibration, using
the same datasets from the section above, CIFAR-100-LT
with an imbalance ratio of 50 for the small-scale dataset
and ImageNet-LT for the large-scale dataset. Following [53,
31], we estimate the quality of calibration on two datasets
with four metrics:
• Expected Calibration Error
ECE = 1
N
M
X
m=1
|Bm| · |acc(Bm) −conf(Bm)|,
(28)
• Classwise Expected Calibration Error
Classwise-ECE
= 1
C
C
X
j=1
M
X
m=1
|Bm,j| · |acc(Bm,j) −conf(Bm,j)|
(29)
• Brier Score
Brier =
N
X
i=1
C
X
c=1
(p(yi = c|xi; θ) −1(yi = c))2,
(30)
• Negative Log Likelihood
NLL = −
N
X
i=1
log p(yi|xi; θ),
(31)
where N is the total number of test samples (xi, yi),
C is the total number of classes, M(= 20) is the to-
tal number of bins, each bin Bm is the set of in-
dices of test samples where m−1
M
< p(yi|xi; θ) ≤
m
M ,
|Bm| is the total number of samples inside the bin Bm,
acc(Bm) =
1
|Bm|
P
i∈Bm 1(arg maxyj p(yj|xi; θ) = yi),
and conf(Bm) =
1
|Bm|
P
i∈Bm p(yi|xi; θ). The bin Bm,j
is the set of indices of test samples where the class for the
samples is j, and the other deﬁnitions |Bm,j|, acc(Bm,j)
and conf(Bm,j) are exactly same as the above.
Table 10 and 11 summarize the calibration results on
CIFAR-100-LT and ImageNet-LT datasets, respectively.
For all the evaluation metrics, LADE shows better overall
calibration results than baseline methods. These observa-
tions demonstrate that our proposed LADE is effective in
terms of calibration on both small-scale (CIFAR-100-LT)
and large-scale (ImageNet-LT) datasets.
13

--- Page 14 ---
Table 9: Top-1 accuracy over all classes on test time shifted CIFAR-100-LT with imbalance ratio of 50.
Dataset
Forward
Uniform
Backward
Imbalance ratio
50
25
10
5
2
1
2
5
10
25
50
Causal Norm
63.7
61.6
58.7
55.9
51.5
48.1
44.7
41.2
38.3
35.6
33.6
Balanced Softmax
59.6
58.5
56.9
54.8
52.2
49.9
47.5
45.1
42.7
40.9
39.9
Softmax
65.9
63.4
59.7
55.6
50.1
45.5
40.8
35.2
30.5
26.8
23.9
PC Causal Norm
66.1
62.9
58.8
55.6
51.2
48.1
45.7
44.2
43.4
44.3
44.9
PC Balanced Softmax
65.9
63.1
59.5
56.3
52.2
49.9
47.9
46.9
46.4
47.3
48.4
PC Softmax
66.0
63.2
59.2
55.9
52.4
49.5
47.5
46.7
46.2
47.4
49.0
LADE
67.4
64.7
60.2
56.3
52.8
50.5
48.2
47.4
46.6
48.1
49.4
Table 10: Conﬁdence calibration results on CIFAR-100-LT with imbalance ratio of 50.
Method
Accuracy
ECE
Classwise ECE
Brier
NLL
Causal Norm
48.1
0.150
0.00483
0.689
2.13
Balanced Softmax
49.9
0.168
0.00461
0.673
2.07
Softmax
45.5
0.249
0.00680
0.769
2.50
PC Softmax
49.5
0.174
0.00472
0.678
2.10
LADE
50.5
0.148
0.00434
0.658
2.02
Table 11: Conﬁdence calibration results on ImageNet-LT.
Method
Accuracy
ECE
Classwise ECE
Brier
NLL
Causal Norm
52.0
0.108
0.000461
0.634
2.42
Balanced Softmax
52.1
0.061
0.000406
0.621
2.20
Softmax
48.2
0.140
0.000603
0.688
2.47
PC Softmax
52.8
0.057
0.000411
0.615
2.17
LADE
53.0
0.035
0.000406
0.611
2.18
14

--- Page 15 ---
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
ECE: 0.1504
Accuracy: 0.4808
Causal Norm
Ideal
Outputs
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
ECE: 0.1680
Accuracy: 0.4989
Balanced Softmax
Ideal
Outputs
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
ECE: 0.1740
Accuracy: 0.4944
PC Softmax
Ideal
Outputs
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
ECE: 0.1479
Accuracy: 0.5054
LADE
Ideal
Outputs
Confidence
Figure 6: Reliability diagrams of ResNet-32 [22] on CIFAR-100-LT with imbalance ratio of 50.
15
