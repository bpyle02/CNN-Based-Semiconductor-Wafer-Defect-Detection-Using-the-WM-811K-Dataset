# Long-Tail Learning via Logit Adjustment

**Authors**: Menon, Jayasumana, Rawat, Jain, Veit, Kumar
**Year**: 2021
**arXiv**: 2007.07314
**Topic**: long_tail
**Relevance**: Theoretically optimal logit adjustment for balanced error under class imbalance

---


--- Page 1 ---
Long-Tail Learning via Logit Adjustment
Aditya Krishna Menon
Sadeep Jayasumana
Ankit Singh Rawat
Himanshu Jain
Andreas Veit
Sanjiv Kumar
Google Research, New York
{adityakmenon,sadeep,ankitsrawat,himj,aveit,sanjivk}@google.com
July 13, 2021
Abstract
Real-world classiﬁcation problems typically exhibit an imbalanced or long-tailed label dis-
tribution, wherein many labels are associated with only a few samples. This poses a challenge
for generalisation on such labels, and also makes naïve learning biased towards dominant labels.
In this paper, we present two simple modiﬁcations of standard softmax cross-entropy training
to cope with these challenges. Our techniques revisit the classic idea of logit adjustment based
on the label frequencies, either applied post-hoc to a trained model, or enforced in the loss
during training. Such adjustment encourages a large relative margin between logits of rare
versus dominant labels. These techniques unify and generalise several recent proposals in the
literature, while possessing ﬁrmer statistical grounding and empirical performance. A reference
implementation of our methods is available at:
https://github.com/google-research/google-research/tree/master/logit_adjustment.
1
Introduction
Real-world classiﬁcation problems typically exhibit a long-tailed label distribution, wherein most
labels are associated with only a few samples [Van Horn and Perona, 2017, Buda et al., 2017, Liu
et al., 2019]. Owing to this paucity of samples, generalisation on such labels is challenging; moreover,
naïve learning on such data is susceptible to an undesirable bias towards dominant labels. This
problem has been widely studied in the literature on learning under class imbalance [Cardie and
Howe, 1997, Chawla et al., 2002, Qiao and Liu, 2009, He and Garcia, 2009, Wallace et al., 2011] and
the related problem of cost-sensitive learning [Elkan, 2001, Zadrozny and Elkan, 2001, Masnadi-Shirazi
and Vasconcelos, 2010, Dmochowski et al., 2010].
Recently, long-tail learning has received renewed interest in the context of neural networks. Two
active strands of work involve post-hoc normalisation of the classiﬁcation weights [Zhang et al., 2019,
Kim and Kim, 2019, Kang et al., 2020, Ye et al., 2020], and modiﬁcation of the underlying loss to
account for varying class penalties [Zhang et al., 2017, Cui et al., 2019, Cao et al., 2019, Tan et al.,
2020]. Each of these strands is intuitive, and has proven empirically successful. However, they are
not without limitation: e.g., weight normalisation crucially relies on the weight norms being smaller
for rare classes; however, this assumption is sensitive to the choice of optimiser (see §2). On the other
hand, loss modiﬁcation sacriﬁces the consistency that underpins the softmax cross-entropy (see §5.2).
Consequently, existing techniques may result in suboptimal solutions even in simple settings (§6.1).
In this paper, we present two simple modiﬁcations of softmax cross-entropy training that unify
several recent proposals, and overcome their limitations. Our techniques revisit the classic idea of
logit adjustment based on label frequencies [Provost, 2000, Zhou and Liu, 2006, Collell et al., 2016],
applied either post-hoc on a trained model, or as a modiﬁcation of the training loss. Conceptually,
logit adjustment encourages a large relative margin between a pair of rare and dominant labels.
1
arXiv:2007.07314v2  [cs.LG]  9 Jul 2021

--- Page 2 ---
Method
Procedure
Consistent?
Reference
Weight normalisation
Post-hoc weight scaling
×
[Kang et al., 2020]
Adaptive margin
Softmax with rare +ve upweighting
×
[Cao et al., 2019]
Equalised margin
Softmax with rare -ve downweighting
×
[Tan et al., 2020]
Logit-adjusted threshold
Post-hoc logit translation
✓
This paper (cf. (9))
Logit-adjusted loss
Softmax with logit translation
✓
This paper (cf. (10))
Table 1: Comparison of approaches to long-tail learning. Weight normalisation re-scales the classiﬁ-
cation weights; by contrast, we add per-label oﬀsets to the logits.
Margin approaches uniformly
increase the margin between a rare positive and all negatives [Cao et al., 2019], or decrease the margin
between all positives and a rare negative [Tan et al., 2020] to prevent suppression of rare labels’
gradients. By contrast, we increase the margin between a rare positive and a dominant negative.
This has a ﬁrm statistical grounding: unlike recent techniques, it is consistent for minimising the
balanced error (cf. (2)), a common metric in long-tail settings which averages the per-class errors.
This grounding translates into strong empirical performance on real-world datasets.
In summary, our contributions are: (i) we present two realisations of logit adjustment for long-tail
learning, applied either post-hoc (§4.2) or during training (§5.2) (ii) we establish that logit adjustment
overcomes limitations in recent proposals (see Table 1), and in particular is Fisher consistent for
minimising the balanced error (cf. (2)); (iii) we conﬁrm the eﬃcacy of the proposed techniques on
real-world datasets (§6). In the course of our analysis, we also present a general version of the softmax
cross-entropy with a pairwise label margin (11), which oﬀers ﬂexibility in controlling the relative
contribution of labels to the overall loss.
2
Problem setup and related work
Consider a multiclass classiﬁcation problem with instances X and labels Y = [L]
.= {1, 2, . . . , L}.
Given a sample S = {(xn, yn)}N
n=1 ∼PN, for unknown distribution P over X × Y, our goal is to
learn a scorer f : X →RL that minimises the misclassiﬁcation error Px,y

y /∈argmaxy′∈Y fy′(x)

.
Typically, one minimises a surrogate loss ℓ: Y × RL →R, such as the softmax cross-entropy,
ℓ(y, f(x)) = log
h X
y′∈[L] efy′(x)i
−fy(x) = log
h
1 +
X
y′̸=y efy′(x)−fy(x)i
.
(1)
For py(x) ∝efy(x), we may view p(x)
.= [p1(x), . . . , pL(x)] ∈∆|Y| as an estimate of P(y | x).
The setting of learning under class imbalance or long-tail learning is where the distribution P(y) is
highly skewed, so that many (rare or “tail”) labels have a very low probability of occurrence. Here,
the misclassiﬁcation error is not a suitable measure of performance: a trivial predictor which classiﬁes
every instance to the majority label will attain a low misclassiﬁcation error. To cope with this, a
natural alternative is the balanced error [Chan and Stolfo, 1998, Brodersen et al., 2010, Menon et al.,
2013], which averages each of the per-class error rates:
BER(f)
.= 1
L
X
y∈[L] Px|y

y /∈argmaxy′∈Y fy′(x)

.
(2)
This can be seen as implicitly using a balanced class-probability function Pbal(y | x) ∝1
L · P(x | y), as
opposed to the native P(y | x) ∝P(y) · P(x | y) that is employed in the misclassiﬁcation error.
Broadly, extant approaches to coping with class imbalance (see also Table 2) modify:
(i) the inputs to a model, for example by over- or under-sampling [Kubat and Matwin, 1997, Chawla
et al., 2002, Wallace et al., 2011, Mikolov et al., 2013, Mahajan et al., 2018, Yin et al., 2018]
2

--- Page 3 ---
(ii) the outputs of a model, for example by post-hoc correction of the decision threshold [Fawcett and
Provost, 1996, Collell et al., 2016] or weights [Kim and Kim, 2019, Kang et al., 2020]
(iii) the internals of a model, for example by modifying the loss function [Xie and Manski, 1989,
Morik et al., 1999, Cui et al., 2019, Zhang et al., 2017, Cao et al., 2019, Tan et al., 2020]
Family
Method
Reference
Post-hoc correction
Modify threshold
[Fawcett and Provost, 1996, Provost, 2000, Mal-
oof, 2003, King and Zeng, 2001, Collell et al.,
2016]
Normalise weights
[Zhang et al., 2019, Kim and Kim, 2019, Kang
et al., 2020]
Data modiﬁcation
Under-sampling
[Kubat and Matwin, 1997, Wallace et al., 2011]
Over-sampling
[Chawla et al., 2002]
Feature transfer
[Yin et al., 2018]
Loss weighting
Loss balancing
[Xie and Manski, 1989, Morik et al., 1999,
Menon et al., 2013]
Volume weighting
[Cui et al., 2019]
Average top-k loss
[Fan et al., 2017]
Domain adaptation
[Jamal et al., 2020]
Margin modiﬁcation
Cost-sensitive SVM
[Masnadi-Shirazi and Vasconcelos, 2010, Iran-
mehr et al., 2019]
Range loss
[Zhang et al., 2017]
Label-aware margin
[Cao et al., 2019]
Equalised negatives
[Tan et al., 2020]
Table 2: Summary of diﬀerent approaches to learning under class imbalance.
One may easily combine approaches from the ﬁrst stream with those from the latter two. Consequently,
we focus on the latter two in this work, and describe some representative recent examples from each.
Post-hoc weight normalisation. Suppose fy(x) = w⊤
y Φ(x) for classiﬁcation weights wy ∈RD
and representations Φ: X →RD, as learned by a neural network. (We may add per-label bias terms
to fy by adding a constant feature to Φ.) A fruitful avenue of exploration involves decoupling of
representation and classiﬁer learning [Zhang et al., 2019]. Concretely, we ﬁrst learn {wy, Φ} via
standard training on the long-tailed training sample S, and then predict for x ∈X
argmaxy∈[L] w⊤
y Φ(x)/ντ
y = argmaxy∈[L] fy(x)/ντ
y ,
(3)
for τ > 0, where νy = P(y) in Kim and Kim [2019], Ye et al. [2020] and νy = ∥wy∥2 in Kang et al.
[2020]. Further to the above, one may also enforce ∥wy∥2 = 1 during training [Kim and Kim, 2019].
Intuitively, either choice of νy upweights the contribution of rare labels through weight normalisation.
The choice νy = ∥wy∥2 is motivated by the observations that ∥wy∥2 tends to correlate with P(y).
Loss modiﬁcation. A classic means of coping with class imbalance is to balance the loss, wherein
ℓ(y, f(x)) is weighted by P(y)−1 [Xie and Manski, 1989, Morik et al., 1999]: for example,
ℓ(y, f(x)) =
1
P(y) · log
h
1 +
X
y′̸=y efy′(x)−fy(x)i
.
(4)
While intuitive, balancing has minimal eﬀect in separable settings: solutions that achieve zero training
loss will necessarily remain optimal even under weighting [Byrd and Lipton, 2019]. Intuitively, one
would like instead to shift the separator closer to a dominant class. Li et al. [2002], Wu et al. [2008],
Masnadi-Shirazi and Vasconcelos [2010], Iranmehr et al. [2019], Gottlieb et al. [2020] thus proposed
3

--- Page 4 ---
0
2
4
6
8
Class y
8
9
10
11
12
13
||wy||
Momentum
Adam
(a) CIFAR-10-LT.
0
20
40
60
80
100
Class y
4
5
6
7
||wy||
Momentum
Adam
(b) CIFAR-100-LT.
Figure 1: Mean and standard deviation over 5 runs of per-class weight norms for a ResNet-32 under
momentum and Adam optimisers. We use long-tailed (“LT”) versions of CIFAR-10 and CIFAR-100,
and sort classes in descending order of frequency; the ﬁrst class is 100 times more likely to appear
than the last class. Both optimisers yield solutions with comparable balanced error. However, the
weight norms have incompatible trends: under momentum, the norms are strongly correlated with
class frequency, while with Adam, the norms are anti-correlated or independent of the class frequency.
Consequently, weight normalisation under Adam is ineﬀective for combatting class imbalance.
to add per-class margins into the hinge loss. [Cao et al., 2019] proposed to add a per-class margin
into the softmax cross-entropy:
ℓ(y, f(x)) = log
h
1 +
X
y′̸=y eδy · efy′(x)−fy(x)i
,
(5)
where δy ∝P(y)−1/4. This upweights rare “positive” labels y, which enforces a larger margin between
a rare positive y and any “negative” y′ ̸= y. Separately, Tan et al. [2020] proposed
ℓ(y, f(x)) = log
h
1 +
X
y′̸=y eδy′ · efy′(x)−fy(x)i
,
(6)
where δy′ ≤0 is an non-decreasing transform of P(y′). The motivation is that, in the original softmax
cross-entropy without {δy′}, a rare label often receives a strong inhibitory gradient signal as it
disproportionately appear as a negative for dominant labels. See also Liu et al. [2016, 2017], Wang
et al. [2018], Khan et al. [2019] for similar weighting of negatives in the softmax.
Limitations of existing approaches. Each of the above methods are intuitive, and have shown
strong empirical performance. However, a closer analysis identiﬁes some subtle limitations.
Limitations of weight normalisation. Post-hoc weight normalisation with νy = ∥wy∥2 per Kang et al.
[2020] is motivated by the observation that the weight norm ∥wy∥2 tends to correlate with P(y).
However, we now show this assumption is highly dependent on the choice of optimiser.
We consider long-tailed versions of CIFAR-10 and CIFAR-100, wherein the ﬁrst class is 100 times
more likely to appear than the last class. (See §6.2 for more details on these datasets.) We optimise
a ResNet-32 using both SGD with momentum and Adam optimisers. Figure 1 conﬁrms that under
SGD, ∥wy∥2 and the class priors P(y) are correlated. However, with Adam, the norms are either
anti-correlated or independent of the class priors. This marked diﬀerence may be understood in light
of recent study of the implicit bias of optimisers [Soudry et al., 2018]; cf. Appendix F. One may hope
to side-step this by simply using νy = P(y); unfortunately, even this choice has limitations (see §4.2).
Limitations of loss modiﬁcation. Enforcing a per-label margin per (5) and (6) is intuitive, as it
allows for shifting the decision boundary away from rare classes. However, when doing so, it is
important to ensure Fisher consistency [Lin, 2004] (or classiﬁcation calibration [Bartlett et al., 2006])
of the resulting loss for the balanced error. That is, the minimiser of the expected loss (equally, the
empirical risk in the inﬁnite sample limit) should result in a minimal balanced error. Unfortunately,
both (5) and (6) are not consistent in this sense, even for binary problems; see §5.2, §6.1 for details.
4

--- Page 5 ---
3
Logit adjustment for long-tail learning: a statistical view
The above suggests there is scope for improving performance on long-tail problems, both in terms of
post-hoc correction and loss modiﬁcation. We now show how a statistical perspective on the problem
suggests simple procedures of each type, both of which overcome the limitations discussed above.
Recall that our goal is to minimise the balanced error (2). A natural question is: what is the best
possible or Bayes-optimal scorer for this problem, i.e., f ∗∈argminf : X→RL BER(f). Evidently, such
an f ∗must depend on the (unknown) underlying distribution P(x, y). Indeed, we have [Menon et al.,
2013], [Collell et al., 2016, Theorem 1]
argmaxy∈[L] f ∗
y (x) = argmaxy∈[L] Pbal(y | x) = argmaxy∈[L] P(x | y),
(7)
where Pbal is the balanced class-probability as per §2. In words, the Bayes-optimal prediction is the
label under which the given instance x ∈X is most likely. Consequently, for ﬁxed class-conditionals
P(x | y), varying the class priors P(y) arbitrarily will not aﬀect the optimal scorers. This is intuitively
desirable: the balanced error is agnostic to the level of imbalance in the label distribution.
To further probe (7), suppose the underlying class-probabilities P(y | x) ∝exp(s∗
y(x)), for (unknown)
scorer s∗: X →RL. Since by deﬁnition Pbal(y | x) ∝P(y | x)/P(y), (7) becomes
argmaxy∈[L] Pbal(y | x) = argmaxy∈[L] exp(s∗
y(x))/P(y) = argmaxy∈[L] s∗
y(x) −ln P(y),
(8)
i.e., we translate the (unknown) distributional scores or logits based on the class priors. This simple
fact immediately suggests two means of optimising for the balanced error:
(i) train a model to estimate the standard P(y | x) (e.g., by minimising the standard softmax-cross
entropy on the long-tailed data), and then explicitly modify its logits post-hoc as per (8)
(ii) train a model to estimate the balanced Pbal(y | x), whose logits are implicitly modiﬁed as per (8)
Such logit adjustment techniques — which have been a classic approach to class-imbalance [Provost,
2000] — neatly align with the post-hoc and loss modiﬁcation streams discussed in §2. However, unlike
most previous techniques from these streams, logit adjustment is endowed with a clear statistical
grounding: by construction, the optimal solution under such adjustment coincides with the Bayes-
optimal solution (7) for the balanced error, i.e., it is Fisher consistent for minimising the balanced
error. We shall demonstrate this translates into superior empirical performance (§6). Note also that
logit adjustment may be easily extended to cover performance measures beyond the balanced error,
e.g., with distinct costs for errors on dominant and rare classes; we leave a detailed study and contrast
to existing cost-sensitive approaches [Iranmehr et al., 2019, Gottlieb et al., 2020] to future work.
We now study each of the techniques (i) and (ii) in turn.
4
Post-hoc logit adjustment
We now detail to perform post-hoc logit adjustment on a classiﬁer trained on long-tailed data. We
further show this bears similarity to recent weight normalisation schemes, but has a subtle advantage.
4.1
The post-hoc logit adjustment procedure
Given a sample S ∼PN of long-tailed data, suppose we learn a neural network with logits fy(x) =
w⊤
y Φ(x). Given these, one typically predicts the label argmaxy∈[L] fy(x). When trained with the
softmax cross-entropy, one may view py(x) ∝exp(fy(x)) as an approximation of the underlying
P(y | x), and so this equivalently predicts the label with highest estimated class-probability.
In post-hoc logit adjustment, we propose to instead predict, for suitable τ > 0:
argmaxy∈[L] exp(w⊤
y Φ(x))/πτ
y = argmaxy∈[L] fy(x) −τ · log πy,
(9)
5

--- Page 6 ---
where π ∈∆Y are estimates of the class priors P(y), e.g., the empirical class frequencies on the
training sample S. Eﬀectively, (9) adds a label-dependent oﬀset to each of the logits. When τ = 1,
this can be seen as applying (8) with a plugin estimate of P(y | x), i.e., py(x) ∝exp(w⊤
y Φ(x)). When
τ ̸= 1, this can be seen as applying (8) to temperature scaled estimates ¯py(x) ∝exp(τ −1 ·w⊤
y Φ(x)). To
unpack this, recall that (8) justiﬁes post-hoc logit thresholding given access to the true probabilities
P(y | x). In principle, the outputs of a suﬃciently high-capacity neural network aim to mimic these
probabilities. In practice, these estimates are often uncalibrated [Guo et al., 2017]. One may thus
need to ﬁrst calibrate the probabilities before applying logit adjustment. Temperature scaling is one
means of doing so, and is often used in the context of distillation [Hinton et al., 2015].
One may treat τ as a tuning parameter to be chosen based on some measure of holdout calibration,
e.g., the expected calibration error [Murphy and Winkler, 1987, Guo et al., 2017], probabilistic
sharpness [Gneiting et al., 2007, Kuleshov et al., 2018], or a proper scoring rule such as the log-loss or
squared error [Gneiting and Raftery, 2007]. One may alternately ﬁx τ = 1 and aim to learn inherently
calibrated probabilities, e.g., via label smoothing [Szegedy et al., 2016, Müller et al., 2019].
4.2
Comparison to existing post-hoc techniques
Post-hoc logit adjustment with τ = 1 is not a new idea in the class imbalance literature. Indeed, this
is a standard technique when creating stratiﬁed samples [King and Zeng, 2001], and when training
binary classiﬁers [Fawcett and Provost, 1996, Provost, 2000, Maloof, 2003]. In multiclass settings,
this has been explored in Zhou and Liu [2006], Collell et al. [2016]. However, τ ̸= 1 is important in
practical usage of neural networks, owing to their lack of calibration. Further, we now explicate that
post-hoc logit adjustment has an important advantage over recent post-hoc weight normalisation
techniques.
Recall that weight normalisation involves learning a scorer fy(x) = w⊤
y Φ(x), and then post-hoc
normalising the weights via wy/ντ
y for τ > 0. We demonstrated in §2 that using νy = ∥wy∥2 may be
ineﬀective when using adaptive optimisers. However, even with νy = πy, there is a subtle contrast to
post-hoc logit adjustment: while the former performs a multiplicative update to the logits, the latter
performs an additive update. The two techniques may thus yield diﬀerent orderings over labels, since
wT
1 Φ(x)
π1
< wT
2 Φ(x)
π2
< · · · < wT
LΦ(x)
πL
̸=⇒
̸⇐=
ewT
1 Φ(x)
π1
< ewT
2 Φ(x)
π2
< · · · < ewT
LΦ(x)
πL
.
Weight normalisation is thus not consistent for minimising the balanced error, unlike logit adjustment.
Indeed, if a rare label y has negative score w⊤
y Φ(x) < 0, and there is another label with positive score,
then it is impossible for the weight normalisation to give y the highest score. By contrast, under
logit adjustment, wT
y Φ(x) −ln πy will be lower for dominant classes, regardless of the original sign.
5
The logit adjusted softmax cross-entropy
We now show how to directly bake logit adjustment into the softmax cross-entropy. We show that
this approach has an intuitive relation to existing loss modiﬁcation techniques.
5.1
The logit adjusted loss
From §3, the second approach to optimising for the balanced error is to directly model Pbal(y | x) ∝
P(y | x)/P(y). To do so, consider the following logit adjusted softmax cross-entropy loss for τ > 0:
ℓ(y, f(x)) = −log
efy(x)+τ·log πy
P
y′∈[L] efy′(x)+τ·log πy′ = log

1 +
X
y′̸=y
πy′
πy
τ
· e(fy′(x)−fy(x))

.
(10)
6

--- Page 7 ---
Given a scorer that minimises the above, we now predict argmaxy∈[L]fy(x) as usual.
Compared to the standard softmax cross-entropy (1), the above applies a label-dependent oﬀset
to each logit. Compared to (9), we directly enforce the class prior oﬀset while learning the logits,
rather than doing this post-hoc. The two approaches have a deeper connection: observe that (10) is
equivalent to using a scorer of the form gy(x) = fy(x) + τ · log πy. We thus have argmaxy∈[L]fy(x) =
argmaxy∈[L]gy(x) −τ · log πy. Consequently, one can equivalently view learning with this loss as
learning a standard scorer g(x), and post-hoc adjusting its logits to make a prediction. For convex
objectives, we thus do not expect any diﬀerence between the solutions of the two approaches. For
non-convex objectives, as encountered in neural networks, the bias endowed by adding τ · log πy to
the logits is however likely to result in a diﬀerent local minima.
For more insight into the loss, consider the following pairwise margin loss
ℓ(y, f(x)) = αy · log
h
1 +
X
y′̸=y e∆yy′ · e(fy′(x)−fy(x))i
,
(11)
for label weights αy > 0, and pairwise label margins ∆yy′ representing the desired gap between
scores for y and y′. For τ = 1, our logit adjusted loss (10) corresponds to (11) with αy = 1 and
∆yy′ = log
 πy′
πy

. This demands a larger margin between rare positive (πy ∼0) and dominant
negative (πy′ ∼1) labels, so that scores for dominant classes do not overwhelm those for rare ones.
5.2
Comparison to existing loss modiﬁcation techniques
A cursory inspection of (5), (6) reveals a striking similarity to our logit adjusted softmax cross-
entropy (10). The balanced loss (4) also bears similarity, except that the weighting is performed
outside the logarithm. Each of these losses are special cases of the pairwise margin loss (11) enforcing
uniform margins that only consider the positive or negative label, unlike our approach.
For example, αy =
1
πy and ∆yy′ = 0 yields the balanced loss (4). This does not explicitly enforce
a margin between the labels, which is undesirable for separable problems [Byrd and Lipton, 2019].
When αy = 1, the choice ∆yy′ = π−1/4
y
yields (5). Finally, ∆yy′ = log F(πy′) yields (6), where
F : [0, 1] →(0, 1] is some non-decreasing function, e.g., F(z) = zτ for τ > 0. These losses thus either
consider the frequency of the positive y or negative y′, but not both simultaneously.
The above choices of α and ∆are all intuitively plausible. However, §3 indicates that our loss in (10)
has a ﬁrm statistical grounding: it ensures Fisher consistency for the balanced error.
Theorem 1. For any δ ∈RL
+, the pairwise loss in (11) is Fisher consistent with weights and margins
αy = δy/πy
∆yy′ = log
 δy′/δy

.
Observe that when δy = πy, we immediately deduce that the logit-adjusted loss of (10) is consistent.
Similarly, δy = 1 recovers the classic result that the balanced loss is consistent. While the above
is only a suﬃcient condition, it turns out that in the binary case, one may neatly encapsulate a
necessary and suﬃcient condition for consistency that rules out other choices; see Appendix B.1.
This suggests that existing proposals may thus underperform with respect to the balanced error in
certain settings, as veriﬁed empirically in §6.1.
5.3
Discussion and extensions
One may be tempted to combine the logit adjusted loss in (10) with the post-hoc adjustment of (9).
However, following §3, such an approach would not be statistically coherent. Indeed, minimising
a logit adjusted loss encourages the model to estimate the balanced class-probabilities Pbal(y | x).
Applying post-hoc adjustment will distort these probabilities, and is thus expected to be harmful.
7

--- Page 8 ---
More broadly, however, there is value in combining logit adjustment with other techniques. For
example, Theorem 1 implies that it is sensible to combine logit adjustment with loss weighting;
e.g., one may pick ∆yy′ = τ · log (πy′/πy), and αy = πτ−1
y
. This is similar to Cao et al. [2019], who
found beneﬁts in combining weighting with their loss. One may also generalise the formulation in
Theorem 1, and employ ∆yy′ = τ1 · log πy −τ2 · log πy′, where τ1, τ2 are constants. This interpolates
between the logit adjusted loss (τ1 = τ2) and a version of the equalised margin loss (τ1 = 0).
Cao et al. [2019, Theorem 2] provides a rigorous generalisation bound for the adaptive margin loss
under the assumption of separable training data with binary labels. The inconsistency of the loss
with respect to the balanced error concerns the more general scenario of non-separable multiclass
data data, which may occur, e.g., owing to label noise or limitation in model capacity. We shall
subsequently demonstrate that encouraging consistency can lead to gains in practical settings. We
shall further see that combining the ∆implicit in this loss with our proposed ∆can lead to further
gains, indicating a potentially complementary nature of the losses.
Interestingly, for τ = −1, a similar loss to (10) has been considered in the context of negative sampling
for scalability [Yi et al., 2019]: here, one samples a small subset of negatives based on the class
priors π, and applies logit correction to obtain an unbiased estimate of the unsampled loss function
based on all the negatives [Bengio and Senecal, 2008]. Losses of the general form (11) have also been
explored for structured prediction [Zhang, 2004, Pletscher et al., 2010, Hazan and Urtasun, 2010].
6
Experimental results
We now present experiments conﬁrming our main claims: (i) on simple binary problems, existing
weight normalisation and loss modiﬁcation techniques may not converge to the optimal solution
(§6.1); (ii) on real-world datasets, our post-hoc logit adjustment outperforms weight normalisation,
and one can obtain further gains via our logit adjusted softmax cross-entropy (§6.2).
6.1
Results on synthetic dataset
We consider a binary classiﬁcation task, wherein samples from class y ∈{±1} are drawn from a 2D
Gaussian with isotropic covariance and means µy = y · (+1, +1). We introduce class imbalance by
setting P(y = +1) = 5%. The Bayes-optimal classiﬁer for the balanced error is (see Appendix G)
f ∗(x) = +1 ⇐⇒P(x | y = +1) > P(x | y = −1) ⇐⇒(µ1 −µ−1)⊤x > 0,
(12)
i.e., it is a linear separator passing through the origin. We compare this separator against those found
by several margin losses based on (11): standard ERM (∆yy′ = 0), the adaptive loss [Cao et al., 2019]
(∆yy′ = π−1/4
y
), an instantiation of the equalised loss [Tan et al., 2020] (∆yy′ = log πy′), and our
logit adjusted loss (∆yy′ = log
πy′
πy ). For each loss, we train an aﬃne classiﬁer on a sample of 10, 000
instances, and evaluate the balanced error on a test set of 10, 000 samples over 100 independent trials.
Figure 2 conﬁrms that the logit adjusted margin loss attains a balanced error close to that of the
Bayes-optimal, which is visually reﬂected by its learned separator closely matching that in (12). This
is in line with our claim of the logit adjusted margin loss being consistent for the balanced error,
unlike other approaches. Figure 2 also compares post-hoc weight normalisation and logit adjustment
for varying scaling parameter τ (c.f. (3), (9)). Logit adjustment is seen to approach the performance
of the Bayes predictor; any weight normalisation is however seen to hamper performance. This
veriﬁes the consistency of logit adjustment, and inconsistency of weight normalisation (§4.2).
8

--- Page 9 ---
ERM
Adaptive
Equalised
Logit
adjusted Bayes
7.5
10.0
12.5
15.0
17.5
20.0
Balanced error
4
2
0
2
4
x1
4
2
0
2
4
x2
ERM
Adaptive
Equalised
Logit
adjusted
Bayes
0.0
0.5
1.0
1.5
2.0
Scaling ( )
10
20
30
40
50
Balanced error
Weight normalisation
Logit adjustment
Bayes
Figure 2: Results on synthetic binary classiﬁcation problem. Our logit adjusted loss tracks the
Bayes-optimal solution and separator (left & middle panel). Post-hoc logit adjustment matches the
Bayes performance with suitable scaling (right panel); however, any weight normalisation fails.
Method
CIFAR-10-LT
CIFAR-100-LT
ImageNet-LT
iNaturalist
ERM
27.16
61.64
53.11
38.66
Weight normalisation (τ = 1) [Kang et al., 2020]
24.02
58.89
52.00
48.05
Weight normalisation (τ = τ ∗) [Kang et al., 2020]
21.50
58.76
49.37
34.10⋆
Adaptive [Cao et al., 2019]
26.65†
60.40†
52.15
35.42†
Equalised [Tan et al., 2020]
26.02
57.26
54.02
38.37
Logit adjustment post-hoc (τ = 1)
22.60
58.24
49.66
33.98
Logit adjustment post-hoc (τ = τ ∗)
19.08
57.90
49.56
33.80
Logit adjustment loss (τ = 1)
22.33
56.11
48.89
33.64
Table 3: Test set balanced error (averaged over 5 trials) on real-world datasets. We use a ResNet-32
for the CIFAR datasets, and ResNet-50 for the ImageNet and iNaturalist datasets. Here, †, ⋆are
numbers for “LDAM + SGD” from Cao et al. [2019, Table 2, 3] and “τ-normalised” from Kang et al.
[2020, Table 3, 7]. Here, τ = τ ∗refers to using the best possible value of tuning parameter τ. See
Figure 3 for plots as a function of τ, and the “Discussion” subsection for further extensions.
6.2
Results on real-world datasets
We present results on the CIFAR-10, CIFAR-100, ImageNet and iNaturalist 2018 datasets. Following
prior work, we create “long-tailed versions” of the CIFAR datasets by suitably downsampling
examples per label following the Exp proﬁle of Cui et al. [2019], Cao et al. [2019] with imbalance
ratio ρ = maxy P(y)/miny P(y) = 100. Similarly, we use the long-tailed version of ImageNet produced
by Liu et al. [2019]. We employ a ResNet-32 for CIFAR, and a ResNet-50 for ImageNet and iNaturalist.
All models are trained using SGD with momentum; see Appendix D for more details. See also
Appendix E.1 for results on CIFAR under the Step proﬁle considered in the literature.
Baselines. We consider: (i) empirical risk minimisation (ERM) on the long-tailed data, (ii) post-hoc
weight normalisation [Kang et al., 2020] per (3) (using νy = ∥wy∥2 and τ = 1) applied to ERM,
(iii) the adaptive margin loss [Cao et al., 2019] per (5), and (iv) the equalised loss [Tan et al.,
2020] per (6), with δy′ = F(πy′) for the threshold-based F of Tan et al. [2020]. Cao et al. [2019]
demonstrated superior performance of their adaptive margin loss against several other baselines, such
as the balanced loss of (4), and that of Cui et al. [2019]. Where possible, we report numbers for the
baselines (which use the same setup as above) from the respective papers. See also our concluding
discussion about extensions to such methods that improve performance.
We compare the above methods against our proposed post-hoc logit adjustment (9), and logit adjusted
loss (10). For post-hoc logit adjustment, we ﬁx the scalar τ = 1; we analyse the eﬀect of tuning this
in Figure 3. We do not perform any further tuning of our logit adjustment techniques.
Results and analysis. Table 3 summarises our results, which demonstrate our proposed logit
adjustment techniques consistently outperform existing methods.
Indeed, while weight normalisation
9

--- Page 10 ---
0
1
2
3
4
Scaling parameter ( )
20
22
24
26
28
Balanced error
Logit adjustment
Weight normalisation
(a) CIFAR-10.
0.0
0.5
1.0
1.5
2.0
Scaling parameter ( )
58
60
62
64
66
68
70
Balanced error
Logit adjustment
Weight normalisation
(b) CIFAR-100.
0.0
0.5
1.0
1.5
2.0
Scaling parameter ( )
33
34
35
36
37
38
39
40
Balanced error
Logit adjustment
Weight normalisation
(c) iNaturalist.
Figure 3: Comparison of balanced error for post-hoc correction techniques when varying scaling
parameter τ (c.f. (3), (9)). Post-hoc logit adjustment consistently outperforms weight normalisation.
0
1
2
3
4
5
6
7
8
9
Class index
0.0
0.1
0.2
0.3
0.4
0.5
0.6
Error on class
ERM
Adaptive
Equalised
Logit
adjusted
(a) CIFAR-10.
0
1
2
3
4
5
6
7
8
9
Class group
0.0
0.2
0.4
0.6
0.8
Error on group
ERM
Adaptive
Equalised
Logit
adjusted
(b) CIFAR-100.
0
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
Class group
0.0
0.1
0.2
0.3
0.4
0.5
0.6
Error on group
ERM
Adaptive
Equalised
Logit
adjusted
(c) iNaturalist.
Figure 4: Per-class error rates of loss modiﬁcation techniques. For (b) and (c), we aggregate the
classes into 10 groups. ERM displays a strong bias towards dominant classes (lower indices). Our
proposed logit adjusted softmax loss achieves signiﬁcant gains on rare classes (higher indices).
oﬀers gains over ERM, these are improved signiﬁcantly by post-hoc logit adjustment (e.g., 8% relative
reduction on CIFAR-10). Similarly loss correction techniques are generally outperformed by our logit
adjusted softmax cross-entropy (e.g., 6% relative reduction on iNaturalist).
Figure 3 studies the eﬀect of tuning the scaling parameter τ > 0 aﬀorded by post-hoc weight
normalisation (using νy = ∥wy∥2) and post-hoc logit adjustment. Even without any scaling, post-hoc
logit adjustment generally oﬀers superior performance to the best result from weight normalisation
(cf. Table 3); with scaling, this is further improved. See Appendix E.4 for a plot on ImageNet-LT.
Figure 4 breaks down the per-class accuracies on CIFAR-10, CIFAR-100, and iNaturalist. On the
latter two datasets, for ease of visualisation, we aggregate the classes into ten groups based on
their frequency-sorted order (so that, e.g., group 0 comprises the top
L
10 most frequent classes). As
expected, dominant classes generally see a lower error rate with all methods. However, the logit
adjusted loss is seen to systematically improve performance over ERM, particularly on rare classes.
While our logit adjustment techniques perform similarly, there is a slight advantage to the loss
function version. Nonetheless, the strong performance of post-hoc logit adjustment corroborates the
ability to decouple representation and classiﬁer learning in long-tail settings [Zhang et al., 2019].
Discussion and extensions Table 3 shows the advantage of logit adjustment over recent post-hoc
and loss modiﬁcation proposals, under standard setups from the literature. We believe further
improvements are possible by fusing complementary ideas, and remark on four such options.
First, one may use a more complex base architecture; our choices are standard in the literature,
but, e.g., Kang et al. [2020] found gains on ImageNet-LT by employing a ResNet-152, with further
gains from training it for 200 as opposed to the customary 90 epochs. Table 4 conﬁrms that logit
adjustment similarly beneﬁts from this choice. For example, on iNaturalist, we obtain an improved
balanced error of 31.15% for the logit adjusted loss. When training for more (200) epochs per the
suggestion of Kang et al. [2020], this further improves to 30.12%.
Second, one may combine together the ∆’s for various special cases of the pairwise margin loss.
10

--- Page 11 ---
ImageNet-LT
iNaturalist
Method
ResNet-50
ResNet-152
ResNet-50
ResNet-152
ResNet-152
90 epochs
90 epochs
200 epochs
ERM
53.11
53.30
38.66
35.88
34.38
Weight normalisation (τ = 1) [Kang et al., 2020]
52.00
51.49
48.05
45.17
45.33
Weight normalisation (τ = τ ∗) [Kang et al., 2020]
49.37
48.97
34.10
31.85
30.34
Adaptive [Cao et al., 2019]
52.15
53.34
35.42
31.18
29.46
Equalised [Tan et al., 2020]
54.02
51.38
38.37
35.86
34.53
Logit adjustment post-hoc (τ = 1)
49.66
49.25
33.98
31.46
30.15
Logit adjustment post-hoc (τ = τ ∗)
49.56
49.15
33.80
31.08
29.74
Logit adjustment loss (τ = 1)
48.89
47.86
33.64
31.15
30.12
Logit adjustment plus adaptive loss (τ = 1)
51.25
50.46
31.56
29.22
28.02
Table 4: Test set balanced error (averaged over 5 trials) on real-world datasets with more complex base
architectures. Employing a ResNet-152 is seen to systematically improve all methods’ performance,
with logit adjustment remaining superior to existing approaches. The ﬁnal row reports the results of
combining logit adjustment with the adaptive margin loss of Cao et al. [2019], which yields further
gains on iNaturalist.
Indeed, we ﬁnd that combining our relative margin with the adaptive margin of Cao et al. [2019]
— i.e., using the pairwise margin loss with ∆yy′ = log
πy′
πy +
1
π1/4
y
— results in a top-1 accuracy of
31.56% on iNaturalist. When using a ResNet-152, this further improves to 29.22% when trained for
90 epochs, and 28.02% when trained for 200 epochs. While such a combination is nominally heuristic,
we believe there is scope to formally study such schemes, e.g., in terms of induced generalisation
performance.
Third, Cao et al. [2019] observed that their loss beneﬁts from a deferred reweighting scheme (DRW),
wherein the model begins training as normal, and then applies class-weighting after a ﬁxed number
of epochs. On CIFAR-10-LT and CIFAR-100-LT, this achieves 22.97% and 57.96% error respectively;
both are outperformed by our vanilla logit adjusted loss. On iNaturalist with a ResNet-50, this
achieves an error of 32.0%, outperforming our 33.6%. (Note that our simple combination of the
relative and adaptive margins outperforms these reported numbers of DRW.) However, given the
strong improvement of our loss over that in Cao et al. [2019] when both methods use SGD, we expect
that employing DRW (which applies to any loss) may be similarly beneﬁcial for our method.
Fourth, per §2, one may perform data augmentation; e.g., see Tan et al. [2020, Section 6]. While
further exploring such variants are of empirical interest, we hope to have illustrated the conceptual
and empirical value of logit adjustment, and leave this for future work.
11

--- Page 12 ---
References
Peter L. Bartlett, Michael I. Jordan, and Jon D. McAuliﬀe. Convexity, classiﬁcation, and risk bounds.
Journal of the American Statistical Association, 101(473):138–156, 2006.
Y. Bengio and J. S. Senecal. Adaptive importance sampling to accelerate training of a neural
probabilistic language model. Trans. Neur. Netw., 19(4):713–722, April 2008. ISSN 1045-9227.
Kay H. Brodersen, Cheng Soon Ong, Klaas E. Stephan, and Joachim M. Buhmann. The balanced
accuracy and its posterior distribution. In Proceedings of the International Conference on Pattern
Recognition (ICPR), pages 3121–3124, Aug 2010.
Mateusz Buda, Atsuto Maki, and Maciej A. Mazurowski. A systematic study of the class imbalance
problem in convolutional neural networks. arXiv:1710.05381 [cs, stat], October 2017.
Jonathon Byrd and Zachary Chase Lipton. What is the eﬀect of importance weighting in deep
learning? In Proceedings of the 36th International Conference on Machine Learning, ICML 2019,
9-15 June 2019, Long Beach, California, USA, pages 872–881, 2019.
Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. Learning imbalanced datasets
with label-distribution-aware margin loss. In Advances in Neural Information Processing Systems,
2019.
Claire Cardie and Nicholas Howe. Improving minority class prediction using case-speciﬁc feature
weights. In Proceedings of the International Conference on Machine Learning (ICML), 1997.
Philip K. Chan and Salvatore J. Stolfo. Learning with non-uniform class and cost distributions:
Eﬀects and a distributed multi-classiﬁer approach. In KDD-98 Workshop on Distributed Data
Mining, 1998.
Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer. SMOTE:
Synthetic minority over-sampling technique. Journal of Artiﬁcial Intelligence Research (JAIR), 16:
321–357, 2002.
Guillem Collell, Drazen Prelec, and Kaustubh R. Patil. Reviving threshold-moving: a simple plug-in
bagging ensemble for binary and multiclass imbalanced data. CoRR, abs/1606.08698, 2016.
Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. Class-balanced loss based on
eﬀective number of samples. In CVPR, 2019.
Jacek P. Dmochowski, Paul Sajda, and Lucas C. Parra. Maximum likelihood in cost-sensitive learning:
Model speciﬁcation, approximations, and upper bounds. Journal of Machine Learning Research,
11:3313–3332, 2010.
Charles Elkan. The foundations of cost-sensitive learning. In Proceedings of the International Joint
Conference on Artiﬁcial Intelligence (IJCAI), 2001.
Yanbo Fan, Siwei Lyu, Yiming Ying, and Baogang Hu. Learning with average top-k loss. In I. Guyon,
U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors,
Advances in Neural Information Processing Systems 30, pages 497–505. Curran Associates, Inc.,
2017.
Tom Fawcett and Foster Provost. Combining data mining and machine learning for eﬀective user
proﬁling. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery
and Data Mining (KDD), pages 8–13. AAAI Press, 1996.
Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation.
Journal of the American Statistical Association, 102(477):359–378, 2007.
12

--- Page 13 ---
Tilmann Gneiting, Fadoua Balabdaoui, and Adrian E. Raftery. Probabilistic forecasts, calibration
and sharpness. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(2):
243–268, 2007.
Lee-Ad Gottlieb, Eran Kaufman, and Aryeh Kontorovich. Apportioned margin approach for cost
sensitive large margin classiﬁers, 2020.
Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew
Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1
hour. arXiv preprint arXiv:1706.02677, 2017.
Chuan Guo, GeoﬀPleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural
networks. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017,
Sydney, NSW, Australia, 6-11 August 2017, pages 1321–1330, 2017.
Tamir Hazan and Raquel Urtasun. Approximated structured prediction for learning large scale
graphical models. CoRR, abs/1006.2899, 2010. URL http://arxiv.org/abs/1006.2899.
Haibo He and Edwardo A. Garcia. Learning from imbalanced data. IEEE Transactions on Knowledge
and Data Engineering, 21(9):1263–1284, 2009.
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In 2016 IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
Geoﬀrey E. Hinton, Oriol Vinyals, and Jeﬀrey Dean. Distilling the knowledge in a neural network.
CoRR, abs/1503.02531, 2015.
Arya Iranmehr, Hamed Masnadi-Shirazi, and Nuno Vasconcelos. Cost-sensitive support vector
machines. Neurocomputing, 343:50–64, 2019.
Muhammad Abdullah Jamal, Matthew Brown, Ming-Hsuan Yang, Liqiang Wang, and Boqing Gong.
Rethinking class-balanced methods for long-tailed visual recognition from a domain adaptation
perspective, 2020.
Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, and Yan-
nis Kalantidis. Decoupling representation and classiﬁer for long-tailed recognition. In Eighth
International Conference on Learning Representations (ICLR), 2020.
S. Khan, M. Hayat, S. W. Zamir, J. Shen, and L. Shao. Striking the right balance with uncertainty.
In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages
103–112, 2019.
Byungju Kim and Junmo Kim. Adjusting decision boundary for class imbalanced learning, 2019.
Gary King and Langche Zeng. Logistic regression in rare events data. Political Analysis, 9(2):137–163,
2001.
Vladimir Koltchinskii, Dmitriy Panchenko, and Fernando Lozano. Some new bounds on the gen-
eralization error of combined classiﬁers. In T. K. Leen, T. G. Dietterich, and V. Tresp, editors,
Advances in Neural Information Processing Systems 13, pages 245–251. MIT Press, 2001.
Miroslav Kubat and Stan Matwin. Addressing the curse of imbalanced training sets: One-sided
selection. In Proceedings of the International Conference on Machine Learning (ICML), 1997.
Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon. Accurate uncertainties for deep learning
using calibrated regression. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th
International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning
Research, pages 2796–2804, Stockholmsmässan, Stockholm Sweden, 10–15 Jul 2018. PMLR.
13

--- Page 14 ---
Yaoyong Li, Hugo Zaragoza, Ralf Herbrich, John Shawe-Taylor, and Jaz S. Kandola. The perceptron
algorithm with uneven margins. In Proceedings of the Nineteenth International Conference on
Machine Learning, ICML ’02, page 379–386, San Francisco, CA, USA, 2002. Morgan Kaufmann
Publishers Inc. ISBN 1558608737.
Yi Lin. A note on margin-based loss functions in classiﬁcation. Statistics & Probability Letters, 68
(1):73 – 82, 2004. ISSN 0167-7152.
Weiyang Liu, Yandong Wen, Zhiding Yu, and Meng Yang. Large-margin softmax loss for convolutional
neural networks. In Proceedings of the 33rd International Conference on International Conference
on Machine Learning - Volume 48, ICML’16, page 507–516. JMLR.org, 2016.
Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, and Le Song. Sphereface: Deep
hypersphere embedding for face recognition. In 2017 IEEE Conference on Computer Vision and
Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 6738–6746, 2017.
Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X. Yu. Large-scale
long-tailed recognition in an open world. In IEEE Conference on Computer Vision and Pattern
Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 2537–2546. Computer
Vision Foundation / IEEE, 2019.
Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li,
Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised
pretraining. In Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss, editors,
Computer Vision – ECCV 2018, pages 185–201, Cham, 2018. Springer International Publishing.
ISBN 978-3-030-01216-8.
Marcus A. Maloof. Learning when data sets are imbalanced and when costs are unequal and unknown.
In ICML 2003 Workshop on Learning from Imbalanced Datasets, 2003.
Hamed Masnadi-Shirazi and Nuno Vasconcelos. Risk minimization, probability elicitation, and cost-
sensitive SVMs. In Proceedings of the 27th International Conference on International Conference
on Machine Learning, ICML’10, page 759–766, Madison, WI, USA, 2010. Omnipress. ISBN
9781605589077.
Aditya Krishna Menon, Harikrishna Narasimhan, Shivani Agarwal, and Sanjay Chawla. On the
statistical consistency of algorithms for binary classiﬁcation under class imbalance. In Proceedings
of the 30th International Conference on Machine Learning, ICML 2013, Atlanta, GA, USA, 16-21
June 2013, pages 603–611, 2013.
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeﬀrey Dean. Distributed representa-
tions of words and phrases and their compositionality. In Proceedings of the 26th International
Conference on Neural Information Processing Systems, NIPS’13, page 3111–3119, Red Hook, NY,
USA, 2013. Curran Associates Inc.
Katharina Morik, Peter Brockhausen, and Thorsten Joachims. Combining statistical learning with
a knowledge-based approach - a case study in intensive care monitoring. In Proceedings of the
Sixteenth International Conference on Machine Learning (ICML), pages 268–277, San Francisco,
CA, USA, 1999. Morgan Kaufmann Publishers Inc. ISBN 1-55860-612-2.
Rafael Müller, Simon Kornblith, and Geoﬀrey E. Hinton. When does label smoothing help? In
Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information
Processing Systems 2019, NeurIPS 2019, 8-14 December 2019, Vancouver, BC, Canada, pages
4696–4705, 2019.
Allan H. Murphy and Robert L. Winkler. A general framework for forecast veriﬁcation. Monthly
Weather Review, 115(7):1330–1338, 1987.
14

--- Page 15 ---
Patrick Pletscher, Cheng Soon Ong, and Joachim M. Buhmann. Entropy and margin maximization
for structured output learning. In José Luis Balcázar, Francesco Bonchi, Aristides Gionis, and
Michèle Sebag, editors, Machine Learning and Knowledge Discovery in Databases, pages 83–98,
Berlin, Heidelberg, 2010. Springer Berlin Heidelberg.
Foster Provost. Machine learning from imbalanced data sets 101. In Proceedings of the AAAI-2000
Workshop on Imbalanced Data Sets, 2000.
Xingye Qiao and Yufeng Liu. Adaptive weighted learning for unbalanced multicategory classiﬁcation.
Biometrics, 65(1):159–168, 2009.
Mark D. Reid and Robert C. Williamson. Composite binary losses. Journal of Machine Learning
Research, 11:2387–2422, 2010.
Daniel Soudry, Elad Hoﬀer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit
bias of gradient descent on separable data. J. Mach. Learn. Res., 19(1):2822–2878, January 2018.
ISSN 1532-4435.
Christian Szegedy, Vincent Vanhoucke, Sergey Ioﬀe, Jonathon Shlens, and Zbigniew Wojna. Rethink-
ing the inception architecture for computer vision. In 2016 IEEE Conference on Computer Vision
and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pages 2818–2826,
2016.
Jingru Tan, Changbao Wang, Buyu Li, Quanquan Li, Wanli Ouyang, Changqing Yin, and Junjie
Yan. Equalization loss for long-tailed object recognition, 2020.
Keiji Tatsumi and Tetsuzo Tanino. Support vector machines maximizing geometric margins for
multi-class classiﬁcation. TOP, 22(3):815–840, 2014.
Keiji Tatsumi, Masashi Akao, Ryo Kawachi, and Tetsuzo Tanino.
Performance evaluation of
multiobjective multiclass support vector machines maximizing geometric margins. Numerical
Algebra, Control & Optimization, 1:151, 2011. ISSN 2155-3289.
Grant Van Horn and Pietro Perona. The devil is in the tails: Fine-grained classiﬁcation in the wild.
arXiv preprint arXiv:1709.01450, 2017.
B.C. Wallace, K.Small, C.E. Brodley, and T.A. Trikalinos. Class imbalance, redux. In Proc. ICDM,
2011.
F. Wang, J. Cheng, W. Liu, and H. Liu. Additive margin softmax for face veriﬁcation. IEEE Signal
Processing Letters, 25(7):926–930, 2018.
Shan-Hung Wu, Keng-Pei Lin, Chung-Min Chen, and Ming-Syan Chen. Asymmetric support vector
machines: Low false-positive learning under the user tolerance. In Proceedings of the 14th ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’08, page
749–757, New York, NY, USA, 2008. Association for Computing Machinery. ISBN 9781605581934.
Yu Xie and Charles F. Manski. The logit model and response-based samples. Sociological Methods &
Research, 17(3):283–302, 1989.
Han-Jia Ye, Hong-You Chen, De-Chuan Zhan, and Wei-Lun Chao. Identifying and compensating for
feature deviation in imbalanced deep learning, 2020.
Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe
Zhao, Li Wei, and Ed Chi.
Sampling-bias-corrected neural modeling for large corpus item
recommendations. In Proceedings of the 13th ACM Conference on Recommender Systems, RecSys
’19, page 269–277, New York, NY, USA, 2019. Association for Computing Machinery. ISBN
9781450362436.
15

--- Page 16 ---
Xi Yin, Xiang Yu, Kihyuk Sohn, Xiaoming Liu, and Manmohan Chandraker. Feature transfer
learning for deep face recognition with long-tail data. CoRR, abs/1803.09014, 2018.
Bianca Zadrozny and Charles Elkan. Learning and making decisions when costs and probabilities
are both unknown. In Proceedings of the Seventh ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining, KDD ’01, page 204–213, New York, NY, USA, 2001.
Association for Computing Machinery. ISBN 158113391X.
Junjie Zhang, Lingqiao Liu, Peng Wang, and Chunhua Shen. To balance or not to balance: A
simple-yet-eﬀective approach for learning with long-tailed distributions, 2019.
Tong Zhang. Class-size independent generalization analsysis of some discriminative multi-category
classiﬁcation methods. In Proceedings of the 17th International Conference on Neural Information
Processing Systems, NIPS’04, page 1625–1632, Cambridge, MA, USA, 2004. MIT Press.
X. Zhang, Z. Fang, Y. Wen, Z. Li, and Y. Qiao. Range loss for deep face recognition with long-
tailed training data. In 2017 IEEE International Conference on Computer Vision (ICCV), pages
5419–5428, 2017.
Zhi-Hua Zhou and Xu-Ying Liu. Training cost-sensitive neural networks with methods addressing
the class imbalance problem. IEEE Transactions on Knowledge and Data Engineering (TKDE),
18(1), 2006.
16

--- Page 17 ---
Supplementary material for “Long tail learning via logit
adjustment”
A
Proofs of results in body
Proof of Theorem 1. Denote ηy(x) = P(y | x). Suppose we employ a margin ∆yy′ = log
δy′
δy . Then,
the loss is
ℓ(y, f(x)) = −log
δy · efy(x)
P
y′∈[L] δy′ · efy′(x) = −log
efy(x)+log δy
P
y′∈[L] efy′(x)+log δy′ .
Consequently, under constant weights αy = 1, the Bayes-optimal score will satisfy f ∗
y (x) + log δy =
log ηy(x), or f ∗
y (x) = log ηy(x)
δy .
Now suppose we use generic weights α ∈RL
+. The risk under this loss is
Ex,y [ℓα(y, f(x))] =
X
y∈[L]
πy · Ex|y=y [ℓα(y, f(x))]
=
X
y∈[L]
πy · Ex|y=y [ℓα(y, f(x))]
=
X
y∈[L]
πy · αy · Ex|y=y [ℓ(y, f(x))]
∝
X
y∈[L]
¯πy · Ex|y=y [ℓ(y, f(x))] ,
where ¯πy ∝πy · αy. Consequently, learning with the weighted loss is equivalent to learning with
the original loss, on a distribution with modiﬁed base-rates ¯π. Under such a distribution, we have
class-conditional distribution
¯ηy(x) = ¯P(y | x) = P(x | y) · ¯πy
¯P(x)
= ηy(x) · ¯πy
πy
· P(x)
¯P(x) ∝ηy(x) · αy.
Consequently, suppose αy = δy
πy . Then, f ∗
y (x) = log ¯ηy(x)
δy
= log ηy(x)
πy
+ C(x), where C(x) does not
depend on y. Consequently, argmaxy∈[L] f ∗
y (x) = argmaxy∈[L]
ηy(x)
πy , which is the Bayes-optimal
prediction for the balanced error.
In sum, a consistent family can be obtained by choosing any set of constants δy > 0 and setting
αy = δy
πy
∆yy′ = log δy′
δy
.
17

--- Page 18 ---
B
On the consistency of binary margin-based losses
It is instructive to study the pairwise margin loss (11) in the binary case. Endowing the loss with a
temperature parameter γ > 0, we get1
ℓ(+1, f) = ω+1
γ
· log(1 + eγ·δ+1 · e−γ·f)
ℓ(−1, f) = ω−1
γ
· log(1 + eγ·δ−1 · eγ·f)
(13)
for constants ω±1, γ > 0 and δ±1 ∈R. Here, we have used δ+1 = ∆+1,−1 and δ−1 = ∆−1,+1 for
simplicity. The choice ω±1 = 1, δ±1 = 0 recovers the temperature scaled binary logistic loss. Evidently,
as γ →+∞, these converge to weighted hinge losses with variable margins, i.e.,
ℓ(+1, f) = ω+1 · [δ+1 −f]+
ℓ(−1, f) = ω−1 · [δ−1 + f]+.
We study two properties of this family losses. First, under what conditions are the losses Fisher
consistent for the balanced error? We shall show that in fact there is a simple condition characterising
this. Second, do the losses preserve properness of the original binary logistic loss? We shall show
that this is always the case, but that the losses involve fundamentally diﬀerent approximations.
B.1
Consistency of the binary pairwise margin loss
Given a loss ℓ, its Bayes optimal solution is f ∗∈argminf : X→R E [ℓ(y, f(x))]. For consistency with
respect to the balanced error in the binary case, we require this optimal solution f ∗to satisfy
f ∗(x) > 0 ⇐⇒η(x) > π, where η(x)
.= P(y = 1 | x) and π
.= P(y = 1) [Menon et al., 2013]. This is
equivalent to a simple condition on the weights ω and margins δ of the pairwise margin loss.
Lemma 2. The losses in (13) are consistent for the balanced error iﬀ
ω+1
ω−1
· σ(γ · δ+1)
σ(γ · δ−1) = 1 −π
π
,
where σ(z) = (1 + exp(z))−1.
Proof of Lemma 2. Denote η(x)
.= P(y = +1 | x), and π
.= P(y = +1). From Lemma 3 below,
the pairwise margin loss is proper composite with invertible link function Ψ: [0, 1] →R ∪{±∞}.
Consequently, since by deﬁnition the Bayes-optimal score for a proper composite loss is f ∗(x) =
Ψ(η(x)) [Reid and Williamson, 2010], to have consistency for the balanced error, from (14), (15), we
require
Ψ−1(0) = π ⇐⇒
1
1 −ℓ′(+1,0)
ℓ′(−1,0)
= π
⇐⇒1 −ℓ′(+1, 0)
ℓ′(−1, 0) = 1
π
⇐⇒−ℓ′(+1, 0)
ℓ′(−1, 0) = 1 −π
π
⇐⇒ω+1
ω−1
· σ(γ · δ+1)
σ(γ · δ−1) = 1 −π
π
.
1Compared to the multiclass case, we assume here a scalar score f ∈R. This is equivalent to constraining that
P
y∈[L] fy = 0 for the multiclass case.
18

--- Page 19 ---
From the above, some admissible parameter choices include:
• ω+1 = 1
π, ω−1 =
1
1−π, δ±1 = 1; i.e., the standard weighted loss with a constant margin
• ω±1 = 1, δ+1 = 1
γ · log 1−π
π , δ−1 = 1
γ · log
π
1−π; i.e., the unweighted loss with a margin biased
towards the rare class, as per our logit adjustment procedure
The second example above is unusual in that it requires scaling the margin with the temperature;
consequently, the margin disappears as γ →+∞. Other combinations are of course possible, but
note that one cannot arbitrarily choose parameters and hope for consistency in general. Indeed, some
inadmissible choices are naïve applications of the margin modiﬁcation or weighting, e.g.,
• ω+1 = 1
π, ω−1 =
1
1−π, δ+1 = 1
γ · log 1−π
π , δ−1 = 1
γ · log
π
1−π; i.e., combining both weighting and
margin modiﬁcation
• ω±1 = 1, δ+1 = 1
γ · (1 −π), δ−1 = 1
γ · π; i.e., speciﬁc margin modiﬁcation
Note further that the choices of Cao et al. [2019], Tan et al. [2020] do not meet the requirements of
Lemma 2.
We make two ﬁnal remarks.
First, the above only considers consistency of the result of loss
minimisation. For any choice of weights and margins, we may apply suitable post-hoc correction
to the predictions to account for any bias in the optimal scores. Second, as γ →+∞, any constant
margins δ±1 > 0 will have no eﬀect on the consistency condition, since σ(γ · δ±1) →1. The condition
will be wholly determined by the weights ω±1. For example, we may choose ω+1 = 1
π, ω−1 =
1
1−π,
δ+1 = 1, and δ−1 =
π
1−π; the resulting loss will not be consistent for ﬁnite γ, but will become so in
the limit γ →+∞. For more discussion on this particular loss, see Appendix C.
B.2
Properness of the pairwise margin loss
In the above, we appealed to the pairwise margin loss being proper composite, in the sense of Reid
and Williamson [2010]. Intuitively, this speciﬁes that the loss has Bayes-optimal score of the form
f ∗(x) = Ψ(η(x)), where Ψ is some invertible function, and η(x) = P(y = 1 | x). We have the following
general result about properness of any member of the pairwise margin family.
Lemma 3. The losses in (13) are proper composite, with link function
Ψ(p) = 1
γ · log


a · b
q
−c

±
sa · b
q
−c
2
+ 4 · a
q

−log 2,
where a = ω+1
ω−1 · eγ·δ+1
eγ·δ−1 , b = eγ·δ−1, c = eγ·δ+1, and q = 1−p
p .
Proof of Lemma 3. The above family of losses is proper composite iﬀthe function
Ψ−1(f) =
1
1 −ℓ′(+1,f)
ℓ′(−1,f)
(14)
is invertible [Reid and Williamson, 2010, Corollary 12]. We have
ℓ′(+1, f) = −ω+1 ·
eγ·δ+1 · e−γ·f
1 + eγ·δ+1 · e−γ·f
ℓ′(−1, f) = +ω−1 ·
eγ·δ−1 · eγ·f
1 + eγ·δ−1 · eγ·f .
(15)
19

--- Page 20 ---
The invertibility of Ψ−1 is immediate. To compute the link function Ψ, note that
p =
1
1 −ℓ′(+1,f)
ℓ′(−1,f)
⇐⇒1
p = 1 −ℓ′(+1, f)
ℓ′(−1, f)
⇐⇒−ℓ′(+1, f)
ℓ′(−1, f) = 1 −p
p
⇐⇒ω+1
ω−1
·
eγ·δ+1 · e−γ·f
1 + eγ·δ+1 · e−γ·f · 1 + eγ·δ−1 · eγ·f
eγ·δ−1 · eγ·f
= 1 −p
p
⇐⇒ω+1
ω−1
· eγ·δ+1
eγ·δ−1 ·
1
eγ·f + eγ·δ+1 · 1 + eγ·δ−1 · eγ·f
eγ·f
= 1 −p
p
⇐⇒a · 1 + b · g
g2 + c · g = q,
where a = ω+1
ω−1 · eγ·δ+1
eγ·δ−1 , b = eγ·δ−1, c = eγ·δ+1, g = eγ·f, and q = 1−p
p . Thus,
a · 1 + b · g
g2 + c · g = q ⇐⇒g2 + c · g
1 + b · g = a
q
⇐⇒g2 +

c −a · b
q

· g −a
q = 0
⇐⇒g =

a·b
q −c

±
r
a·b
q −c
2
+ 4 · a
q
2
.
As a sanity check, suppose a = b = c = γ = 1. This corresponds to the standard logistic loss. Then,
Ψ(p) = log

1
q −1

±
r
1
q −1
2
+ 4 · 1
q
2
= log
p
1 −p,
which is the standard logit function.
Figure 5 and 6 compares the link functions for a few diﬀerent settings:
• the balanced loss, where ω+1 = 1
π, ω−1 =
1
1−π, and δ±1 = 1
• an unequal margin loss, where ω±1 = 1, δ+1 = 1
γ · log 1−π
π , and δ−1 = 1
γ · log
π
1−π
• a balanced + margin loss, where ω+1 = 1
π, ω−1 =
1
1−π, δ+1 = 1, and δ−1 =
π
1−π.
The property Ψ−1(0) = π for π = P(y = 1) holds for the ﬁrst two choices with any γ > 0, and the
third choice as γ →+∞. This indicates the Fisher consistency of these losses for the balanced error.
However, the precise way this is achieved is strikingly diﬀerent in each case. In particular, each loss
implicitly involves a fundamentally diﬀerent link function.
To better understand the eﬀect of parameter choices, Figure 7 illustrates the conditional Bayes risk
curves, i.e.,
L(p) = p · ℓ(+1, Ψ(p)) + (1 −p) · ℓ(+1, Ψ(p)).
We remark here that for the balanced error, this function takes the form L(p) = p·Jp < πK+(1−p)·Jp >
πK, i.e., it is a “tent shaped” concave function with a maximum at p = π.
For ease of comparison, we normalise this curves to have a maximum of 1. Figure 7 shows that
simply applying unequal margins does not aﬀect the underlying conditional Bayes risk compared
to the standard log-loss; thus, the change here is purely in terms of the link function. By contrast,
either balancing the loss or applying a combination of weighting and margin modiﬁcation results in a
closer approximation to the conditional Bayes risk curve for the cost-sensitive loss with cost π.
20

--- Page 21 ---
0.0
0.2
0.4
0.6
0.8
1.0
p
4
2
0
2
4
(p)
Balanced
Unequal margin
Balanced + margin
0.0
0.2
0.4
0.6
0.8
1.0
p
1.5
1.0
0.5
0.0
0.5
1.0
1.5
(p)
Balanced
Unequal margin
Balanced + margin
Figure 5: Comparison of link functions for various losses assuming π = 0.2, with γ = 1 (left) and
γ = 8 (right). The balanced loss uses ωy =
1
πy . The unequal margin loss uses δy = 1
γ · log 1−π
π . The
balanced + margin loss uses δ−1 =
π
1−π, δ+1 = 1, ω+1 = 1
π.
4
2
0
2
4
v
0.0
0.2
0.4
0.6
0.8
1.0
1(v)
Balanced
Unequal margin
Balanced + margin
4
2
0
2
4
v
0.0
0.2
0.4
0.6
0.8
1.0
1(v)
Balanced
Unequal margin
Balanced + margin
Figure 6: Comparison of link functions for various losses assuming π = 0.2, with γ = 1 (left) and
γ = 8 (right). The balanced loss uses ωy =
1
πy . The unequal margin loss uses δy = 1
γ · log 1−πy
πy . The
balanced + margin loss uses δ−1 =
π
1−π, δ+1 = 1, ω+1 = 1
π.
0.0
0.2
0.4
0.6
0.8
1.0
p
0.2
0.4
0.6
0.8
1.0
L(p)
Balanced
Unequal margin
Balanced + margin
0.0
0.2
0.4
0.6
0.8
1.0
p
0.0
0.2
0.4
0.6
0.8
1.0
L(p)
Balanced
Unequal margin
Balanced + margin
Figure 7: Comparison of conditional Bayes risk functions for various losses assuming π = 0.2, with
γ = 1 (left) and γ = 8 (right). The balanced loss uses ωy =
1
πy . The unequal margin loss uses
δy = 1
γ · log 1−πy
πy . The ﬁrst balanced + margin loss uses δ−1 = π, δ+1 = 1, ω+1 = 1
π. The second
balanced + margin loss uses δ−1 =
π
1−π, δ+1 = 1, ω+1 = 1
π.
21

--- Page 22 ---
C
Relation to cost-sensitive SVMs
We recapitulate the analysis of Masnadi-Shirazi and Vasconcelos [2010] in our notation. Consider a
binary cost-sensitive learning problem with cost parameter c ∈(0, 1). The Bayes-optimal classiﬁer for
this task corresponds to f ∗(x) = Jη(x) > cK. The case c = 0.5 is the standard classiﬁcation problem.
Suppose we wish to design a weighted, variable margin SVM for this task, i.e.,
ℓ(+1, f) = ω+1 · [δ+1 −f]+
ℓ(−1, f) = ω−1 · [δ−1 + f]+
where ω±1, δ±1 ≥0. The conditional risk for this loss is
L(η, f) = η · ℓ(+1, f) + (1 −η) · ℓ(−1, f)
=





(1 −η) · ω−1 · (δ−1 + f)
if f > δ+1
η · ω+1 · (δ+1 −f) + (1 −η) · ω−1 · (δ−1 + f)
if f ∈[−δ−1, δ+1]
η · ω+1 · (δ+1 −f)
if f < −δ−1.
As this is a piecewise linear function, which is decreasing for f < −δ−1 and increasing for f > δ+1,
the only possible minimum is at {δ+1, −δ−1}. To ensure consistency, we seek the minimum to be δ+1
iﬀη > c. Observe that
L(η, δ+1) < L(η, −δ−1) ⇐⇒(1 −η) · ω−1 < η · ω+1
⇐⇒
η
1 −η > ω−1
ω+1
⇐⇒η >
ω−1
ω−1 + ω+1
.
Consequently, we must have
ω−1
ω−1 + ω+1
= c ⇐⇒ω+1
ω−1
= 1 −c
c
.
Observe here that the margin terms δ±1 do not appear in the consistency condition: thus, as long as
the weights are suitably chosen, any choice of margin terms will result in a consistent loss.
However, the margins do inﬂuence the form conditional Bayes risk: this is
L(η) =
(
(1 −η) · ω−1 · (δ−1 + δ+1)
if η > c
η · ω−1 · (δ−1 + δ+1)
if η < c.
For the purposes of normalisation, it is natural to require this function to attain a maximum at 1.
This corresponds to choosing
δ−1 + δ+1 = 1
c ·
1
ω+1
.
In the class-imbalance setting, c = π, and so we require
ω+1
ω−1
= 1 −π
π
δ−1 + δ+1 = 1
π ·
1
ω+1
for consistency and normalisation respectively. This gives two degrees of freedom: the choice of ω+1
(which determines ω−1), and then the choice of δ+1 (which determines δ−1). For example, we could
pick ω+1 = 1
π, ω−1 =
1
1−π, δ+1 = 1, δ−1 =
π
1−π.
22

--- Page 23 ---
To relate this to Masnadi-Shirazi and Vasconcelos [2010], the latter considered separate costs C−1, C+1
for a false positive and false negative respectively. With this, they suggested to use Masnadi-Shirazi
and Vasconcelos [2010, Equation 34]
ℓ(+1, f) = d ·
he
d −f
i
+
ℓ(−1, f) = a ·
 b
a + f

+
with δ+1 = e
d = 1, d = ω+1 = C+1, a = ω−1 = 2C−1 −1, and δ−1 =
b
a = 1
a. The constraints
C1 ≥2C−1 −1 and C−1 ≥1 are also enforced.
Under this setup, the cost ratio is
C−1
C−1+C+1 . In the class-imbalance setting, we have
C−1
C−1+C+1 = π, and
so C+1 = 1−π
π ·C−1. By the consistency condition, we have C+1 = ω+1 = 1−π
π ·ω−1 = 1−π
π ·(2C−1−1).
Thus, we must set C−1 = 1, and so C+1 = 1−π
π . Thus, we obtain the parameters ω+1 = 1−π
π , ω−1 = 1,
δ+1 = 1, δ−1 =
π
1−π. By rescaling the weights, we obtain ω+1 = 1
π, ω−1 =
1
1−π, δ+1 = 1, δ−1 =
π
1−π.
Observe that this is exactly one of the losses considered in Appendix B.1.
D
Experimental setup
Intending a fair comparison, we use the same setup for all the methods for each dataset. All networks
are trained with SGD with a momentum value of 0.9. Unless otherwise speciﬁed, linear learning rate
warm-up is used in the ﬁrst 5 epochs to reach the base learning rate, and a weight decay of 10−4 is
used. Other dataset speciﬁc details are given below.
CIFAR-10 and CIFAR-100: We use a CIFAR ResNet-32 model trained for 200 epochs. The base
learning rate is set to 0.1, which is decayed by 0.1 at the 160th epoch and again at the 180th epoch.
Mini-batches of 128 images are used.
We also use the standard CIFAR data augmentation procedure used in previous works such as Cao
et al. [2019], He et al. [2016], where 4 pixels are padded on each size and a random 32 × 32 crop is
taken. Images are horizontally ﬂipped with a probability of 0.5.
ImageNet: We use a ResNet-50 model trained for 90 epochs. The base learning rate is 0.4, with
cosine learning rate decay. We use a batch size of 512 and the standard data augmentation comprising
of random cropping and ﬂipping as described in Goyal et al. [2017]. Following Kang et al. [2020], we
use a weight decay of 5 × 10−4 on this dataset.
iNaturalist: We again use a ResNet-50 and train it for 90 epochs with a base learning rate of 0.4
and cosine learning rate decay. The data augmentation procedure is the same as the one used in
ImageNet experiment above. We use a batch size of 512.
E
Additional experiments
We present here additional experiments:
(i) we present results for CIFAR-10 and CIFAR-100 on the Step proﬁle [Cao et al., 2019] with
ρ = 100
(ii) we further verﬁy that weight norms may not correlate with class priors under Adam
(iii) we include the results of post-hoc correction, and a breakdown of per-class errors, on ImageNet-LT
23

--- Page 24 ---
E.1
Results on CIFAR-LT with Step-100 proﬁle
Table 5 summarises results on the Step-100 proﬁle. Here, with τ = 1, weight normalisation slightly
outperforms logit adjustment. However, with τ > 1, logit adjustment is again found to be superior
(54.80); see Figure 8.
Method
CIFAR-10-LT
CIFAR-100-LT
ERM
36.54
60.23
Weight normalisation (τ = 1)
30.86
55.19
Adaptive
34.61
58.86
Equalised
31.42
57.82
Logit adjustment post-hoc (τ = 1)
28.66
55.82
Logit adjustment (loss)
27.57
55.52
Table 5: Test set balanced error (averaged over 5 trials) on CIFAR-10-LT and CIFAR-100-LT under
the Step-100 proﬁle; lower is better. On CIFAR-100-LT, weight normalisation edges out logit
adjustment. See Figure 8 for a demonstrated that tuned versions of the same outperfom weight
normalisation.
0
1
2
3
4
Scaling parameter ( )
20
25
30
35
40
45
50
Balanced error
Logit adjustment
Weight normalisation
0.0
0.5
1.0
1.5
2.0
Scaling parameter ( )
55
60
65
70
75
Balanced error
Logit adjustment
Weight normalisation
Figure 8: Post-hoc adjustment on Step-100 proﬁle, CIFAR-10 and CIFAR-100. Logit adjustment
outperforms weight normalisation with suitable tuning.
E.2
Per-class errors on ImageNet-LT
Figure 9 breaks down the per-class accuracies on ImageNet-LT. As before, the logit adjustment
procedure shows signiﬁcant gains on rarer classes.
E.3
Post-hoc correction on ImageNet-LT
Figure 10 compares post-hoc correction techniques as the scaling parameter τ is varied on ImageNet-LT.
As before, logit adjustment with suitable tuning is seen to be competitive with weight normalisation.
E.4
Per-group errors
Following Liu et al. [2019], Kang et al. [2020], we additionally report errors on a per-group basis,
where we construct three groups of classes: “Many”, comprising those with at least 100 training
24

--- Page 25 ---
0
1
2
3
4
5
6
7
8
9
Class group
0.0
0.2
0.4
0.6
0.8
Error on group
Method
ERM
Adaptive
Equalised
Logit
adjusted
Figure 9: Comparison of per-class balanced error on ImageNet-LT. Classes are sorted in order of
frequency, and bucketed into 10 groups.
0.0
0.5
1.0
1.5
2.0
Scaling parameter ( )
48
50
52
54
56
58
Balanced error
Logit adjustment
Weight normalisation
Figure 10: Post-hoc correction on ImageNet.
examples; “Medium”, comprising those with at least 20 and at most 100 training examples; and “Few”,
comprising those with at most 20 training examples. This is a coarser level of granularity than the
grouping employed in the previous section, and the body. Figure 11 shows that the logit adjustment
procedure shows consistent gains over all three groups.
F
Does weight normalisation increase margins?
Suppose that one uses SGD with a momentum, and ﬁnds solutions where ∥wy∥2 tracks the class
priors. One intuition behind normalisation of weights is that, drawing inspiration from the binary
case, this ought to increase the classiﬁcation margins for tail classes.
Unfortunately, this intuition is not necessarily borne out. Consider a scorer fy(x) = wT
y Φ(x), where
wy ∈Rd and Φ: X →Rd. The functional margin for an example (x, y) is [Koltchinskii et al., 2001]
γf(x, y)
.= wT
y Φ(x) −max
y′̸=y wT
y′Φ(x).
(16)
This generalises the classical binary margin, wherein by convention Y = {±1}, w−1 = −w1, and
γf(x, y)
.= y · wT
1 Φ(x) = 1
2 ·
 wT
y Φ(x) −wT
−yΦ(x)

,
(17)
which agrees with (16) upto scaling. One may also deﬁne the geometric margin in the binary case to
25

--- Page 26 ---
Many
Medium
Class group
0.0
0.1
0.2
0.3
0.4
0.5
0.6
Error on group
ERM
Adaptive
Equalised
Logit
adjusted
(a) CIFAR-10-LT
Many
Medium
Few
Class group
0.0
0.2
0.4
0.6
0.8
Error on group
ERM
Adaptive
Equalised
Logit
adjusted
(b) CIFAR-100-LT.
Many
Medium
Few
Class group
0.0
0.1
0.2
0.3
0.4
0.5
0.6
Error on group
ERM
Adaptive
Equalised
Logit
adjusted
(c) ImageNet-LT.
Many
Medium
Few
Class group
0.0
0.1
0.2
0.3
0.4
Error on group
ERM
Adaptive
Equalised
Logit
adjusted
(d) iNaturalist.
Figure 11: Comparison of per-group errors. We construct three groups of classes: “Many”, comprising
those with at least 100 training examples; “Medium”, comprising those with at least 20 and at most
100 training examples; and “Few”, comprising those with at most 20 training examples.
be the distance of (x, y) from its classiﬁer:
γg,b(x)
.= |w1 · Φ(x)|
∥w1∥2
.
(18)
Clearly, γg,b(x) = |γf(x,y)|
∥w1∥2 , and so for ﬁxed functional margin, one may increase the geometric margin
by minimising ∥w1∥2. However, the same is not necessarily true in the multiclass setting, since
here the functional and geometric margins do not generally align [Tatsumi et al., 2011, Tatsumi
and Tanino, 2014]. In particular, controlling each ∥wy∥2 does not necessarily control the geometric
margin.
G
Bayes-optimal classiﬁer under Gaussian class-conditionals
Suppose
P(x | y) =
1
√
2πσ · exp

−∥x −µy∥2
2σ2

for suitable µy and σ. Then,
P(x | y = +1) > P(x | y = −1) ⇐⇒exp

−∥x −µ+1∥2
2σ2

> exp

−∥x −µ−1∥2
2σ2

⇐⇒∥x −µ+1∥2
2σ2
< ∥x −µ−1∥2
2σ2
⇐⇒∥x −µ+1∥2 < ∥x −µ−1∥2
⇐⇒2 · (µ+1 −µ−1)Tx > ∥µ+1∥2 −∥µ−1∥2.
Now use the fact that in our setting, ∥µ+1∥2 = ∥µ−1∥2.
26

--- Page 27 ---
We remark also that the class-probability function is
P(y = +1 | x) = P(x | y = +1) · P(y = +1)
P(x)
= P(x | y = +1) · P(y = +1)
P
y′ P(x | y′) · P(y′)
=
1
1 + P(x|y=−1)·P(y=−1)
P(x|y=+1)·P(y=+1)
.
Now,
P(x | y = −1)
P(x | y = +1) = exp
∥x −µ+1∥2 −∥x −µ−1∥2
2σ2

= exp
∥µ+1∥2 −∥µ−1∥2 −2 · (µ+1 −µ−1)Tx
2σ2

= exp
−(µ+1 −µ−1)Tx
σ2

.
Thus,
P(y = +1 | x) =
1
1 + exp(−wT
∗x + b∗),
where w∗=
1
σ2 ·(µ+1−µ−1), and b∗= log P(y=−1)
P(y=+1). This implies that a sigmoid model for P(y = +1 | x),
as employed by logistic regression, is well-speciﬁed for the problem. Further, the bias term b∗is seen
to take the form of the log-odds of the class-priors per (8), as expected.
27
