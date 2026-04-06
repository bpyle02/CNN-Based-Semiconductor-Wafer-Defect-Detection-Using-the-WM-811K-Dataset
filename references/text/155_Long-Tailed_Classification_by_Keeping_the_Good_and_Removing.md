# Long-Tailed Classification by Keeping the Good and Removing the Bad Momentum Causal Effect

**Authors**: Tang, Huang, Zhang
**Year**: 2020
**arXiv**: 2009.12991
**Topic**: long_tail
**Relevance**: Causal inference approach to debiasing long-tailed classifiers

---


--- Page 1 ---
Long-Tailed Classification by Keeping the Good and
Removing the Bad Momentum Causal Effect
Kaihua Tang1,
Jianqiang Huang1,2∗,
Hanwang Zhang1
1Nanyang Technological University,
2Damo Academy, Alibaba Group
kaihua001@e.ntu.edu.sg,
jianqiang.jqh@gmail.com,
hanwangzhang@ntu.edu.sg
Abstract
As the class size grows, maintaining a balanced dataset across many classes is
challenging because the data are long-tailed in nature; it is even impossible when the
sample-of-interest co-exists with each other in one collectable unit, e.g., multiple
visual instances in one image. Therefore, long-tailed classification is the key
to deep learning at scale. However, existing methods are mainly based on re-
weighting/re-sampling heuristics that lack a fundamental theory. In this paper,
we establish a causal inference framework, which not only unravels the whys of
previous methods, but also derives a new principled solution. Specifically, our
theory shows that the SGD momentum is essentially a confounder in long-tailed
classification. On one hand, it has a harmful causal effect that misleads the tail
prediction biased towards the head. On the other hand, its induced mediation also
benefits the representation learning and head prediction. Our framework elegantly
disentangles the paradoxical effects of the momentum, by pursuing the direct
causal effect caused by an input sample. In particular, we use causal intervention in
training, and counterfactual reasoning in inference, to remove the “bad” while keep
the “good”. We achieve new state-of-the-arts on three long-tailed visual recognition
benchmarks2: Long-tailed CIFAR-10/-100, ImageNet-LT for image classification
and LVIS for instance segmentation.
1
Introduction
Over the years, we have witnessed the fast development of computer vision techniques [1–3],
stemming from large and balanced datasets such as ImageNet [4] and MS-COCO [5]. Along with the
growth of the digital data created by us, the crux of making a large-scale dataset is no longer about
where to collect, but how to balance. However, the cost of expanding them to a larger class vocabulary
with balanced data is not linear — but exponential — as the data will be inevitably long-tailed by
Zipf’s law [6]. Specifically, a single sample increased for one data-poor tail class will result in more
samples from the data-rich head. Sometimes, even worse, re-balancing the class is impossible. For
example, in instance segmentation [7], if we target at increasing the images of tail class instances like
“remote controller”, we have to bring in more head instances like “sofa” and “TV” simultaneously in
every newly added image [8].
Therefore, long-tailed classification is indispensable for training deep models at scale. Recent
work [9–11] starts to fill in the performance gap between class-balanced and long-tailed datasets,
while new long-tailed benchmarks are springing up such as Long-tailed CIFAR-10/-100 [12, 10],
ImageNet-LT [9] for image classification and LVIS [7] for object detection and instance segmentation.
Despite the vigorous development of this field, we find that the fundamental theory is still missing. We
conjecture that it is mainly due to the paradoxical effects of long tail. On one hand, it is bad because
∗Corresponding author.
2Our code is available on https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch
34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada.
arXiv:2009.12991v5  [cs.CV]  26 Oct 2025

--- Page 2 ---
M
X
D
Y
M: Momentum
X: Feature
D: Projection on Head
Y: Prediction
(a) The Proposed Causal Graph
(c) Relative Change of Accuracy from 𝜇=0.98
0
10
20
30
40
50
60
70
0.78 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.96 0.98
Relative Change (%)
Momentum Decay Ratio 𝜇
All
Many
Medium
Few
(b) Mean magnitude of 𝑥for each class 𝑖
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
0
44
88
132
176
220
264
308
352
396
440
484
528
572
616
660
704
748
792
836
880
924
968
Magnitude
Class Index
Figure 1: (a) The proposed causal graph explaining the causal effect of momentum. See Section 3 for
details. (b) The mean magnitudes of feature vectors for each class i after training with momentum
µ = 0.9, where i is ranking from head to tail. (c) The relative change of the performance on the basis
of µ = 0.98 shows that the few-shot tail is more vulnerable to the momentum.
the classification is severely biased towards the data-rich head. On the other hand, it is good because
the long-tailed distribution essentially encodes the natural inter-dependencies of classes — “TV” is
indeed a good context for “controller” — any disrespect of it will hurt the feature representation
learning [10], e.g., re-weighting [13, 14] or re-sampling [15, 16] inevitably causes under-fitting to the
head or over-fitting to the tail.
Inspired by the above paradox, latest studies [10, 11] show promising results in disentangling the
“good” from the “bad”, by the naïve two-stage separation of imbalanced feature learning and balanced
classifier training. However, such disentanglement does not explain the whys and wherefores of the
paradox, leaving critical questions unanswered: given that the re-balancing causes under-fitting/over-
fitting, why is the re-balanced classifier good but the re-balanced feature learning bad? The two-stage
design clearly defies the end-to-end merit that we used to believe since the deep learning era; but why
does the two-stage training significantly outperform the end-to-end one in long-tailed classification?
In this paper, we propose a causal framework that not only fundamentally explains the previous
methods [15–17, 9, 11, 10], but also provides a principled solution to further improve long-tailed
classification. The proposed causal graph of this framework is given in Figure 1 (a). We find that
the momentum M in any SGD optimizer [18, 19] (also called betas in Adam optimizer [20]), which
is indispensable for stabilizing gradients, is a confounder who is the common cause of the sample
feature X (via M →X) and the classification logits Y (via M →D →Y ). In particular, D denotes
the X’s projection on the head feature direction that eventually deviates X. We will justify the
graph later in Section 3. Here, Figure 1 (b&c) sheds some light on how the momentum affects the
feature X and the prediction Y . From the causal graph, we may revisit the “bad” long-tailed bias in
a causal view: the backdoor [21] path X ←M →D →Y causes the spurious correlation even if
X has nothing to do with the predicted Y , e.g., misclassifying a tail sample to the head. Also, the
mediation [22] path X →D →Y mixes up the pure contribution made by X →Y . For the “good”
bias, X →D →Y respects the inter-relationships of the semantic concepts in classification, that is,
the head class knowledge contributes a reliable evidence to filter out wrong predictions. For example,
if a rare sample is closer to the head class “TV” and “sofa”, it is more likely to be a living room
object (e.g., “remote controller”) but not an outdoor one (e.g., “car”).
Based on the graph that explains the paradox of the “bad” and “good”, we propose a principled
solution for long-tailed classification. It is a natural derivation of pursuing the direct causal effect along
X →Y by removing the momentum effect. Thanks to causal inference [23], we can elegantly keep
the “good” while remove the “bad”. First, to learn the model parameters, we apply de-confounded
training with causal intervention: while it removes the “bad” by backdoor adjustment [21] who
cuts off the backdoor confounding path X ←M →D →Y , it keeps the “good” by retaining the
mediation X →D →Y . Second, we calculate the direct causal effect of X →Y as the final
prediction logits. It disentangles the “good” from the “bad” in a counterfactual world, where the bad
effect is considered as the Y ’s indirect effect when X is zero but D retains the value when X = x. In
contrast to the prevailing two-stage design [11] that requires unbiased re-training in the 2nd stage, our
solution is one-stage and re-training free. Interestingly, as discussed in Section 4.4, we show that why
the re-training is inevitable in their method and why ours can avoid it with even better performance.
2

--- Page 3 ---
On image classification benchmarks Long-tailed CIFAR-10/-100 [12, 10] and ImageNet-LT [9], we
outperform previous state-of-the-arts [10, 11] on all splits and settings, showing that the performance
gain is not merely from catering to the long tail or a specific imbalanced distribution. In object detec-
tion and instance segmentation benchmark LVIS [7], our method also has a significant advantage over
the former winner [17] of LVIS 2019 challenge. We achieve 3.5% and 3.1% absolute improvements
on mask AP and box AP using the same Cascade Mask R-CNN with R101-FPN backbone [24].
2
Related Work
Re-Balanced Training. The most widely-used solution for long-tailed classification is arguably
to re-balance the contribution of each class in the training phase. It can be either achieved by re-
sampling [25, 26, 15, 16, 27] or re-weighting [13, 14, 12, 17]. However, they inevitably cause the
under-fitting/over-fitting problem to head/tail classes. Besides, relying on the accessibility of data
distribution also limits their application scope, e.g., not applicable in online and streaming data.
Hard Example Mining. The instance-level re-weighting [28–30] is also a practical solution. Instead
of hacking the prior distribution of classes, focusing on the hard samples also alleviates the long-tailed
issue, e.g., using meta-learning to find the conditional weights for each samples [31], enhancing the
samples of hard categories by group softmax [32].
Transfer Learning/Two-Stage Approach. Recent work shows a new trend of addressing the
long-tailed problem by transferring the knowledge from head to tail. The sharing bilateral-branch
network [10], the two-stage training [11], the dynamic curriculum learning [33] and the transferring
memory features [9] / head distributions [34] are all shown to be effective in long-tailed recognition,
yet, they either significantly increase the parameters or require a complicated training strategy.
Causal Inference. Causal inference [23, 35] has been widely adopted in psychology, politics and
epidemiology for years [36–38]. It doesn’t just serve as an interpretation framework, but also provides
solutions to achieve the desired objectives by pursing causal effect. Recently, causal inference has
also attracted increasing attention in computer vision society [39–44] for removing the dataset bias in
domain-specific applications, e.g., using pure direct effect to capture the spurious bias in VQA [41]
and NWGM for Captioning [42]. Compared to them, our method offers a fundamental framework for
general long-tailed visual recognition.
3
A Causal View on Momentum Effect
To systematically study the long-tailed classification and how momentum affects the prediction,
we construct a causal graph [23, 22] in Figure 1 (a) with four variables: momentum (M), object
feature (X), projection on head direction (D), and model prediction (Y ). The causal graph is a
directed acyclic graph used to indicate how variables of interest {M, X, D, Y } interacting with
each other through causal links. The nodes M and D constitute a confounder and a mediator,
respectively. A confounder is a variable that influences both correlated and independent variables,
creating a spurious statistical correlation. Considering a causal graph exercise ←age →cancer,
the elder people spend more time on physical exercise after retirement and they are also easier to
get cancer due to the elder age, so the confounder age creates a spurious correlation that more
physical exercise will increase the chance of getting cancer. The example of a mediator would be
drug →placebo →cure, where mediator placebo is the side effect of taking drug that prevents
us from getting the direct effect of drug →cure.
Before we delve into the rationale of our causal graph, let’s take a brief review on the SGD with
momentum [19]. Without loss of generality, we adopt the Pytorch implementation [45]:
vt = µ · vt−1
| {z }
momentum
+gt,
θt = θt−1 −lr · vt,
(1)
where the notations in the t-th iteration are: model parameters θt, gradient gt, velocity vt, momentum
decay ratio µ, and learning rate lr. Other versions of SGD [18, 19] only change the position of some
hyper-parameters and we can easily prove them equivalent with each other. The use of momentum
considerably dampens the oscillations caused by each single sample. In our causal graph, momentum
M is the overall effect of µ · vT −1 at the convergence t = T, which is the exponential moving
average of the gradient over all past samples with decay rate µ. Eq. (1) shows that, given fixed
3

--- Page 4 ---
hyper-parameters µ and lr, each sample M = m is a function of the model initialization and the
mini-batch sampling strategy, that is, M has infinite samples.
In a balanced dataset, the momentum is equally contributed by every class. However, when the
dataset is long-tailed, it will be dominated by the head samples, emerging the following causal links:
(a) Decompose the gradient velocity
(b) Decompose the biased feature vector
𝒗𝒕
𝒈𝒕
𝝁⋅𝒗𝒕−𝟏
𝒙
ሷ𝒙
𝒅
Figure 2: Based on Assumption 1, the feature vector x
can be decomposed into a discriminative feature ¨x and a
projection on head direction d
M →X. This link says that the backbone
parameters used to generate feature vec-
tors X, are trained under the effect of M.
This is obvious from Eq. (1) and can be
illustrated in Figure 1 (b), where we vi-
sualize how the magnitudes of X change
from head to tail.
(M, X) →D. This link denotes that the
momentum also causes feature vector X
deviates to the head direction D, which
is also determined by M. In a long-tailed dataset, few head classes possess most of the training
samples, who have less variance than the data-poor but class-rich tail, so the moving averaged
momentum will thus point to a stable head direction. Specifically, as shown in Figure 2, we can
decompose any feature vector x into x = ¨x + d, where D = d = ˆdcos(x, ˆd)∥x∥. In particular, the
head direction ˆd is given in Assumption 1, whose validity is detailed in Appendix A.
Assumption 1 The head direction ˆd is the unit vector of the exponential moving average features
with decay rate µ like momentum, i.e., ˆd = xT /∥xT ∥, where xt = µ · xt−1 + xt and T is the
number of the total training iterations.
Note that Assumption 1 says that the head direction is exactly determined by the sample moving
average in the dataset, which does not need the accessibility of the class statistics at all. In particular,
as we show in Appendix A, when the dataset is balanced, Assumption 1 also holds but suggests that
X →Y is naturally not affected by M.
X →D →Y & X →Y. These links indicate that the effect of X can be disentangled into an indirect
(mediation) and a direct effect. Thanks to the above orthogonal decomposition: x = ¨x + d, the
indirect effect is affected by d while the direct effect is affected by ¨x, and they together determine
the total effect. As shown in Figure 4, when we change the scale parameter α of d, the performance
of the tail classes monotonically increases with α, which inspires us to remove the mediation effect
of D in Section 4.2.
4
The Proposed Solution
Based on the proposed causal graph in Figure 1 (a), we can delineate our goal for long-tailed
classification: the pursuit of the direct causal effect along X →Y . In causal inference, it is defined
as Total Direct Effect (TDE) [46, 22]:
arg max
i∈C
TDE(Yi) = [Yd = i|do(X = x)] −[Yd = i|do(X = x0)],
(2)
M
D
Y
M
D
Y
𝒙
𝒙𝟎
𝒙
Figure 3: The TDE inference (Eq. (2))
for the long-tailed classification after de-
confounded training.
Subtracted left:
[Yd = i|do(X = x)], minus right: [Yd =
i|do(X = x0)].
where x0 denotes a null input (0 in this paper). We
define the causal effect as the prediction logits Yi for
the i-th class. Subscript d denotes that the mediator D
always takes the value d in the deconfounded causal
graph model of Figure 1 (a) with do(X = x), where
the do-operator denotes the causal intervention [23] that
modifies the graph by M ̸→X. Thus, Eq. (2) shows an
important principle in long-tailed classification: before
we calculate the final TDE (Section 4.2), we need to
first perform de-confounded training (Section 4.1) to
estimate the “modified” causal graph parameters.
We’d like to highlight that Eq. (2) removes the “bad”
while keeps the “good” in a reconcilable way. First, in
4

--- Page 5 ---
training, the do-operator removes the “bad” confounder bias while keeps the “good” mediator bias,
because the do-operator retains the mediation path. Second, in inference, the mediator value d is
imposed in both terms to keep the “good” of the mediator bias (towards head) in logit prediction; it
also removes its “bad” by subtracting the second term: the prediction when the input X is null (x0)
but the mediator D is still the value d when X had been x. Note that such a counterfactual minus
elegantly characterizes the “bad” mediation bias, just like how we capture the tricky placebo effect:
we cheat the patient to take a placebo drug, setting the direct drug effect drug →cure to zero; thus,
any cure observed must be purely due to the non-zero placebo effect drug →placebo →cure.
4.1
De-confounded Training
The model for the proposed causal graph is optimized under the causal intervention do(X = x), which
aims to preserve the “good” feature learning from the momentum and cut off its “bad” confounding
effect. We apply the backdoor adjustment [21] to derive the de-confounded model:
P(Y = i|do(X = x)) =
X
m
P(Y = i|X = x, M = m)P(M = m)
(3)
=
X
m
P(Y = i, X = x|M = m)P(M = m)
P(X = x|M = m)
.
(4)
As there are infinite number of M = m, it is prohibitively to achieve the above backdoor adjust-
ment. Fortunately, the Inverse Probability Weighting [23] formulation in Eq. (4) provides us a new
perspective in approximating the infinite sampling (i, x)|m. For a finite dataset, no matter how many
m there are, we can only observe one (i, x) given one m. In such cases, the number of m values
that Eq. (4) would encounter is equal to the number of samples (i, x) available, not to the number of
possible m values, which is prohibitive. In fact, thanks to the backdoor adjustment, which connects
the equivalence between the originally confounded model P and the deconfounded model P with
do(X), we can collect samples from the former, that act as though they were drawn from the latter.
Therefore, Eq. (4) can be approximated as
P(Y = i|do(X = x)) ≈1
K
K
X
k=1
eP(Y = i, X = xk|M = m),
(5)
where eP is the inverse weighted probability and we will drop M = m in the rest of the paper for
notation simplicity and bear in mind that x still depends on m. In particular, compared to the vanilla
trick, we apply a multi-head strategy [47] to equally divide the channel (or dimensions) of weights
and features into K groups, which can be considered as K times more fine-grained sampling.
We model eP in Eq. (5) as the softmax activated probability of the energy-based model [48]:
eP(Y = i, X = xk) ∝E(i, xk; wk
i ) = τ f(i, xk; wk
i )
g(i, xk; wk
i ) ,
(6)
where τ is a positive scaling factor akin to the inverse temperature in Gibbs distribution. Recall
Assumption 1 that xk = ¨xk + dk. The numerator, i.e., the unnormalized effect, can be implemented
as logits f(i, xk; wk
i ) = (wk
i )⊤(¨xk + dk) = (wk
i )⊤xk, and the denominator is a normalization
term (or propensity score [49]) that only balances the magnitude of the variables: g(i, xk; wk
i ) =
∥xk∥· ∥wk
i ∥+ γ∥xk∥, where the first term is a class-specific energy and the second term is a
class-agnostic baseline energy.
Putting the above all together, the logit calculation for P(Y = i|do(X = x)) can be formulated as:
[Y = i|do(X = x)] = τ
K
K
X
k=1
(wk
i )⊤(¨xk + dk)
(∥wk
i ∥+ γ)∥xk∥= τ
K
K
X
k=1
(wk
i )⊤xk
(∥wk
i ∥+ γ)∥xk∥.
(7)
Interestingly, this model also explains the effectiveness of normalized classifiers like cosine classi-
fier [50, 51]. We will further discuss it in Section 4.4.
5

--- Page 6 ---
Methods
Two-stage
Re-balancing (do(D))
De-confound (do(X))
Direct Effect
Cosine [50, 51]
-
-
✔
-
LDAM [12]
-
✔
✔
CDE
OLTR [9]
✔
✔
-
NDE
BBN [10]
✔
✔
-
NDE
Decouple [11]
✔
✔
-
NDE
EQL [17]
-
✔
-
-
Our method
-
-
✔
TDE
Table 1: Revisiting the previous state-of-the-arts in our causal graph. CDE: Controlled Direct Effect.
NDE: Natural Direct Effect. TDE: Total Direct Effect.
4.2
Total Direct Effect Inference
After the de-confounded training, the causal graph is now ready for inference. The TDE of X →Y
in Eq. (2) can thus be depicted as in Figure 3. By applying the counterfactual consistency rule [52],
we have [Yd = i|do(X = x)] = [Y = i|do(X = x)]. This indicates that we can use Eq. (7) to
calculate the first term of Eq. (2). Thanks to Assumption 1, we can disentangle x by x = ¨x + d,
where d = ∥d∥· ˆd = cos(x, ˆd)∥x∥· ˆd. Therefore, we have [Yd = i|do(X = x0)] that replaces the
¨x in Eq. (7) with zero vector, just like “cheating” the model with a null input but keeping everything
else unchanged. Overall, the final TDE calculation for Eq. (2) is
TDE(Yi) = τ
K
K
X
k=1
 
(wk
i )⊤xk
(∥wk
i ∥+ γ)∥xk∥−α · cos(xk, ˆd
k) · (wk
i )⊤ˆd
k
∥wk
i ∥+ γ
!
,
(8)
where α controls the trade-off between the indirect and direct effect as shown in Figure 4.
4.3
Background-Exempted Inference
(a) Accuracy for different TDE parameter 𝛼
0
10
20
30
40
50
60
70
0
0.5
1
1.5
2
2.5
3
3.5
4
4.5
5
Accuracy (%)
TDE Parameter 
All
Many
Medium
Few
𝛼
Figure 4: The influence of parameter α in
Eq. (8) on ImageNet-LT val set [9] shows
how D controls the head/tail preference.
Some classification tasks need a special “background”
class to filter out samples belonging to none of the
classes of interest, e.g., object detection and instance
segmentation use the background class to remove non-
object regions [3, 24], and recommender systems as-
sume that the majority of the items are irrelevant to a
user [53]. In such tasks, most of the training samples
are background and hence the background class is a
good head class, whose effect should be kept and thus
exempted from the TDE calculation. To this end, we
propose a background-exempted inference that partic-
ular uses the original inference (total effect) for back-
ground class. The inference can be formulated as:
arg max
i∈C
 (1 −p0) ·
qi
1−q0
i ̸= 0
p0
i = 0 ,
(9)
where i = 0 is the background class, pi = P(Y =
i|do(X = x)) is the de-confounded probability that
we defined in Section 4.1, qi is the softmax activated
probability of the original TDE(Yi) in Eq. (8). Note that Eq. (9) adds up to 1 from i = 0 to C.
4.4
Revisiting Two-stage Training
The proposed framework also theoretically explains the previous state-of-the-arts as shown in Table 1.
Please see Appendix B for the detailed revisit for each method.
Two-stage Re-balancing. Naïve re-balanced training fails to retain a natural mediation D that
respects the inter-dependencies among classes. Therefore, the two-stage training is adopted by most
of the re-balancing methods: imbalanced pre-training the backbone with natural D and then balanced
re-training a fair classifier with the fixed backbone for feature representation. Later, we will show
that the second stage re-balancing essentially plays a counterfactual role, which reveals the reason
why the stage-2 is indispensable.
6

--- Page 7 ---
Methods
Many-shot
Medium-shot
Few-shot
Overall
Focal Loss† [28]
64.3
37.1
8.2
43.7
OLTR† [9]
51.0
40.8
20.8
41.9
Decouple-OLTR† [9, 11]
59.9
45.8
27.6
48.7
Decouple-Joint [11]
65.9
37.5
7.7
44.4
Decouple-NCM [11]
56.6
45.3
28.1
47.3
Decouple-cRT [11]
61.8
46.2
27.4
49.6
Decouple-τ-norm [11]
59.1
46.9
30.7
49.4
Decouple-LWS [11]
60.2
47.2
30.3
49.9
Baseline
66.1
38.4
8.9
45.0
Cosine† [50, 51]
67.3
41.3
14.0
47.6
Capsule† [9, 54]
67.1
40.0
11.2
46.5
(Ours) De-confound
67.9
42.7
14.7
48.6
(Ours) Cosine-TDE
61.8
47.1
30.4
50.5
(Ours) Capsule-TDE
62.3
46.9
30.6
50.6
(Ours) De-confound-TDE
62.7
48.8
31.6
51.8
Table 2: The performances on ImageNet-LT test set [9]. All models were using the ResNeXt-50
backbone. The superscript † denotes being re-implemented by our framework and hyper-parameters.
De-confounded Training. Technically, the proposed de-confounded training in Eq. (7) is the multi-
head classifier with normalization. The normalized classifier, like cosine classifier, has already been
embraced by various methods [50, 51, 9, 11] based on empirical practice. However, as we will show
in Table 2, without the guidance of our causal graph, their normalizations perform worse than the
proposed de-confounded model. For example, methods like decouple [11] only applies normalization
in the 2nd stage balanced classifier training, and hence its feature learning is not de-confounded.
Direct Effect. The one-stage re-weighting/re-sampling training methods, like LDAM [12], can
be interpreted as calculating Controlled Direct Effect (CDE) [23]: CDE(Yi) = [Y = i|do(X =
x), do(D = d0)] −[Y = i|do(X = x0), do(D = d0)], where x0 is a dummy vector and d0 is
a constant vector. CDE performs a physical intervention — re-balancing — on the training data
by setting the bias D to a constant. Note that the second term of CDE is a constant that does not
affect the classification. However, CDE removes the “bad” at the cost of hurting the “good” during
representation learning, as D is no longer a natural mediation generated by X.
The two-stage methods [10, 11] are essentially Natural Direct Effect (NDE), where the stage-2
re-balanced training is actually an intervention on D that forces the direction ˆd do not head to any
class. Therefore, when attached with the stage-1 imbalanced pre-trained features, the balanced
classifier calculates the NDE: NDE(Yi) = [Yd0 = i|do(X = x)] −[Yd0 = i|do(X = x0)], where
x0 and d0 are dummy vectors, because the stage-2 balanced classifier forces the logits to nullify
any class-specific momentum direction; do(X = x) as stage-1 backbone is frozen and M ̸→X;
the second term can be omitted as it is a class-agnostic constant. Besides that their stage-1 training
is still confounded, as we will show in experiments, our TDE is better than NDE because the latter
completely removes the entire effect of D by setting D = d0, which is however sometimes good,
e.g., mis-classifying “warthog” as the head-class “pig” is better than “car”; TDE admits the effect by
keeping D = d as a baseline and further compares the fine-grained difference via the direct effect,
e.g., by admitting that “warthog” does look like “pig”, TDE finds out that the tusk is the key difference
between “warthog” and “pig”, and that is why our method can focus on more discriminative regions
in Figure 5.
5
Experiments
The proposed method was evaluated on three long-tailed benchmarks: Long-tailed CIFAR-10/-100,
ImageNet-LT for image classification and LVIS for object detection and instance segmentation. The
consistent improvements across different tasks demonstrate our broad application domain.
Datasets and Protocols. We followed [12, 10] to collect the long-tailed versions of CIFAR-10/-100
with controllable degrees of data imbalance ratio ( Nmax
Nmin , where N is number of samples in each
category), which controls the distribution of training sets. ImageNet-LT [9] is a long-tailed subset of
ImageNet dataset [4]. It consists of 1k classes over 186k images, where 116k/20k/50k for train/val/test
sets, respectively. In train set, the number of images per class is ranged from 1,280 to 5, which
7

--- Page 8 ---
Baseline
Decouple-LWS
Our Method
Input
vulture
green lizard
harp
grey fox
Many-shot Classes
house finch
brown bear
meerkat
alligator lizard
Medium-shot Classes
Few-shot Classes
warthog
bighorn
proboscis monkey
kimono
Figure 5: The visualized activation maps of the linear classifier baseline, Decouple-LWS [11] and the
proposed method on ImageNet-LT using the Grad-CAM [55].
Dataset
Long-tailed CIFAR-100
Long-tailed CIFAR-10
Imbalance ratio
100
50
10
100
50
10
Focal Loss [28]
38.4
44.3
55.8
70.4
76.7
86.7
Mixup [56]
39.5
45.0
58.0
73.1
77.8
87.1
Class-balanced Loss [13]
39.6
45.2
58.0
74.6
79.3
87.1
LDAM [12]
42.0
46.6
58.7
77.0
81.0
88.2
BBN [10]
42.6
47.0
59.1
79.8
82.2
88.3
(Ours) De-confound
40.5
46.2
58.9
71.7
77.8
86.8
(Ours) De-confound-TDE
44.1
50.3
59.6
80.6
83.6
88.5
Table 3: Top-1 accuracy on Long-tailed CIFAR-10/-100 with different imbalance ratios. All models
are using the same ResNet-32 backbone. We further adopted the same warm-up scheduler from
BBN [10] for fair comparisons.
imitates the long-tailed distribution that commonly exists in the real world. The test and val sets were
balanced and reported on four splits: Many-shot containing classes with > 100 images, Medium-shot
including classes with ≥20 & ≤100 images, Few-shot covering classes with < 20 images, and
Overall for all classes. LVIS [7] is a large vocabulary instance segmentation dataset with 1,230/1,203
categories in V0.5/V1.0, respectively. It contains a 57k/100k train set (V0.5/V1.0) under a significant
long-tailed distribution, and relatively balanced 5k/20k val set (V0.5/V1.0) and 20k test set.
Evaluation. For Long-tailed CIFAR-10/-100 [12, 10], we evaluated Top-1 accuracy under three
different imbalance ratios: 100/50/10. For ImageNet-LT [9], the evaluation results were reported
as the percentage of accuracy on four splits. For LVIS [7], the evaluation metrics are standard
segmentation mask AP calculated across IoU threshold 0.5 to 0.95 for all classes. These classes can
also be categorized by the frequency and independently reported as APr, APc, APf: subscripts r, c, f
stand for rare (appeared in < 10 images), common (appeared in 11 −100 images), and frequent
(appeared in > 100 images). Since we can use the LVIS to detect bounding boxes, the detection
results were reported as APbbox.
Implementation Details. For image classification on ImageNet-LT, we used ResNeXt-50-32x4d [2]
as our backbone for all experiments. All models were trained by using SGD optimizer with momentum
µ = 0.9 and batch size 512. The learning rate was decayed by a cosine scheduler [57] from 0.2
to 0.0 in 90 epochs. Hyper-parameters were chosen by the performances on ImageNet-LT val set,
and we set K = 2, τ = 16, γ = 1/32, α = 3.0. For Long-tailed CIFAR-10/-100, we changed
the backbone to ResNet-32 and the training scheduler to warm-up scheduler like BBN [10] for fair
comparisons. All parameters except for α are inherited from ImageNet-LT, which was set to 1.0/1.5
for CIFAR-10/-100 respectively. For instance segmentation and object detection on LVIS, we chose
Cascade Mask R-CNN framework [24] implemented by [58]. The optimizer was also SGD with
momentum µ = 0.9 and we used batch size 16 for a R101-FPN backbone. The models were trained
in 20 epochs with learning rate starting at 0.02 and decaying by the factor of 0.1 at the 16-th and
19-th epochs. We selected the top 300 predicted boxes following [7, 17]. The hyper-parameters
on LVIS were directly adopted from the ImageNet-LT, except for α = 1.5. The main difference
8

--- Page 9 ---
Methods
LVIS Version
AP
AP50
AP75
APr
APc
APf
APbbox
Focal Loss† [28]
V0.5
21.1
32.1
22.6
3.2
21.1
28.3
22.6
(2019 Winner) EQL [17]
V0.5
24.9
37.9
26.7
10.3
27.3
27.8
27.9
Baseline
V0.5
22.6
33.5
24.4
2.5
23.0
30.2
24.3
Cosine† [50, 51]
V0.5
25.0
37.7
27.0
9.3
25.5
30.8
27.1
Capsule† [9, 54]
V0.5
25.4
37.8
27.4
8.5
26.4
31.0
27.1
(Ours) De-confound
V0.5
25.7
38.5
27.8
11.4
26.1
30.9
27.7
(Ours) Cosine-TDE
V0.5
28.1
42.6
30.2
20.8
28.7
30.3
30.6
(Ours) Capsule-TDE
V0.5
28.4
42.1
30.8
21.1
29.7
29.6
30.4
(Ours) De-confound-TDE
V0.5
28.4
43.0
30.6
22.1
29.0
30.3
31.0
Baseline
V1.0
21.8
32.7
23.2
1.1
20.9
31.9
23.9
(Ours) De-confound
V1.0
23.5
34.8
25.0
5.2
22.7
32.3
25.8
(Ours) De-confound-TDE
V1.0
27.1
40.1
28.7
16.0
26.9
32.1
30.0
Table 4: All models are using the same Cascade Mask R-CNN framework [24] with R101-FPN
backbone [59]. The reported results are evaluated on LVIS val set [7].
between image classification and object detection/instance segmentation is that the latter includes a
background class i = 0, which is a head class used to make a binary decision between foreground and
background. As we discussed in Section. 4.3, the Background-Exempted Inference should be used to
retain the good background bias. The comparison between with and without Background-Exempted
Inference is given in Appendix C.
Ablation studies. To study the effectiveness of the proposed de-confounded training and TDE
inference, we tested a variety of ablation models: 1) the linear classifier baseline (no biased term); 2)
the cosine classifier [50, 51]; 3) the capsule classifier [9], where x is normalized by the non-linear
function from [54]; 4) the proposed de-confounded model with normal softmax inference; 5) different
versions of the TDE. As reported in Table (2,4), the de-confound TDE achieves the best performance
under all settings. The TDE inference improves all three normalized models, because the cosine and
capsule classifiers can be considered as approximations to the proposed de-confounded model. To
show that the mediation effect removed by TDE indeed controls the preference towards head direction,
we changed the parameter α as shown in Figure 4, resulting the smooth increasing/decreasing of the
performances on tail/head classes, respectively.
Comparisons with State-of-The-Art Methods. The previous state-of-the-art results on ImageNet-
LT are achieved by the two-stage re-balanced training [11] that decouples the backbone and classifier.
However, as we discussed in Section 4.4, this kind of approaches are less effective or efficient. In
Long-tailed CIFAR-10/-100, we outperform the previous methods [13, 12, 10] in all imbalance ratios,
which proves that the proposed method can automatically adapt to different data distributions. In
LVIS dataset, after a simple adaptation, we beat the champion EQL [17] of LVIS Challenge 2019 in
Table 4. All reported results in Table 4 are using the same Cascade Mask R-CNN framework [24] and
R101-FPN backbone [59] for fair comparison. The EQL results were copied from [17], which were
trained by 16 GPUs and 32 batch size while the proposed method only used 8 GPUs and half of the
batch size. We didn’t compare the EQL results on the final challenge test server, because they claimed
to exploit external dataset and other tricks like ensemble to win the challenge. Note that EQL is also a
re-balanced method, having the same problems as [11]. We also visualized the activation maps using
Grad-CAM [55] in Figure 5. The linear classifier baseline and decouple-LWS [11] usually activate
the entire objects and some context regions to make a prediction. Meanwhile, the de-confound TDE
only focuses on the direct effect, i.e., the most discriminative regions, so it usually activates on a
more compact area, which is less likely to be biased towards its similar head classes. For example, to
classify a “kimono”, the proposed method only focuses on the discriminative feature rather than the
entire body, which is similar to some other clothes like “dress”.
6
Conclusions
In this work, we first proposed a causal framework to pinpoint the causal effect of momentum in
the long-tailed classification, which not only theoretically explains the previous methods, but also
provides an elegant one-stage training solution to extract the unbiased direct effect of each instance.
The detailed implementation consists of de-confounded training and total direct effect inference,
which is simple, adaptive, and agnostic to the prior statistics of the class distribution. We achieved
the new stage-of-the-arts of various tasks on both ImageNet-LT and LVIS benchmarks. As moving
forward, we are going to 1) further validate our theory in a wider spectrum of application domains
and 2) seek better feature disentanglement algorithms for more precise counterfactual effects.
9

--- Page 10 ---
Broader Impact
The positive impacts of this work are two-fold: 1) it improves the fairness of the classifier, which
prevents the potential discrimination of deep models, e.g., an unfair AI could blindly cater to the
majority, causing gender, racial or religious discrimination; 2) it allows the larger vocabulary datasets
to be easily collected without a compulsory class-balancing pre-processing, e.g., to train autonomous
vehicles, by using the proposed method, we don’t need collecting as many ambulance images as
normal van images do. The negative impacts could also happen when the proposed long-tailed
classification technique falls into the wrong hands, e.g., it can be used to identify the minority groups
for malicious purposes. Therefore, it’s our duty to make sure that the long-tailed classification
technique is used for the right purpose.
A
Additional Explanations of Assumption 1
To better understand the (M, X) →D and Assumption 1, let’s take a simple example. Given a
learnable parameter θ ∈R2, and its gradients of instances for class A, B approximate to (1, 1) and
(-1, 1) respectively. If each of these two classes has 50 samples, the mean gradient would be (0, 1),
which is the optimal gradient direction shared by both A and B. The momentum will thus accelerate
on this direction that optimizes the model to fairly discriminate two classes. However, if there are 99
samples from class A and only 1 sample from class B (long-tailed dataset), the mean gradient would
be (0.98, 1). In this case, the momentum direction now approximates to the class A (head) gradients,
encouraging the backbone parameters to generate head-like feature vectors, i.e., creating an unfair
deviation towards the head.
(a) Magnitude of 𝑤𝑖for each class 𝑖
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
0
50
100
150
200
250
300
350
400
450
500
550
600
650
700
750
800
850
900
950
Magnitude
Class Index
Figure 6:
The magnitudes of classifier
weights ∥wi∥for each class after training
with momentum µ = 0.9, where i is rank-
ing by the number of training samples in a
descending order.
Since the momentum in SGD [45, 18, 19] usually dom-
inates the gradient velocity, the effect of such a devi-
ation is not trivial, which will eventually create the
head projection D on all feature vectors generated by
the backbone. It’s worth noting that although there
are non-linear activation layers in the backbone, due
to the central limit theorem [60], the overall effect of
these deviated parameters is still following the normal
distribution, which means we can use the moving av-
eraged feature to approximate this head direction, i.e.,
the Assumption 1 in the original paper.
In addition, even in a balanced dataset, the Assumption
1 still holds. Considering the above example, the mean
gradient is (0, 1) for balanced A and B, which is not
biased towards either direction: (1, 1) or (-1, 1). In
other word, the D still exists for the balanced dataset,
but the cos(x, ˆd) should be almost the same for all
classes. Therefore, the M →D →Y won’t cause
any preference in the balanced dataset, which naturally
allows X →Y free from the effect of M. It’s also intu-
itively easy to understand, because when the dataset is
balanced, the mean feature only represents the common
patterns shared by all classes, e.g., the D in a balanced
face recognition dataset is the mean face, which would be a contour of human head that not biased
towards any specific face categories.
B
Revisiting Previous Methods in Long-Tailed Classification
In this section, we will revisit the previous state-of-the-arts in two aspects: the normalized classifiers
and the re-balancing strategies.
Normalized Classifiers. The normalized classifiers [50, 51, 11, 9] have already been widely adopted
in long-tailed classification based on empirical practice. As we discussed in the Section 4, the
correctly applied normalized classifiers are approximations of the proposed de-confounded training.
10

--- Page 11 ---
Methods
BG-Exempted
AP
AP50
AP75
APr
APc
APf
APbbox
De-confound
✗
25.7
38.5
27.8
11.4
26.1
30.9
27.7
De-confound-TDE
False
23.4
35.7
24.9
13.1
23.6
27.1
24.8
De-confound-TDE
True
28.4
43.0
30.6
22.1
29.0
30.3
31.0
Table 5: The results of the proposed TDE with/without Background-Exempted Inference on LVIS [7]
V0.5 val set. The Cascade Mask R-CNN framework [24] with R101-FPN backbone [59] is used.
However, without the guidance of the proposed causal framework, most of them are not utilized in a
proper way. We define the general normalized classifier as the following equation:
arg max
i∈C
P(Y = i|X = x) =
ezi
PC
c=1 ezc ,
where zi = τ
K
K
X
k=1
(wk
i )⊤xk
N(xk, wk
i ).
(10)
Since in most of the previous methods, K is set to 1, so we slightly abuse the notation to omit the
superscript k for simplicity.
The cosine classifier [50, 51] is defined based on the cosine similarity, which has N(x, wi) =
∥x∥· ∥wi∥. It is commonly used in the tasks like few-shot learning [61]. In Table 2,3 of original
paper, we have proved its effectiveness in the long-tailed classification. The capsule classifier is
proposed by Liu et al. [9] as the replacement of vanilla cosine classifier in OLTR. It changes the
l2 norm of x into the squashing non-linear function proposed in Capsule Network [54], which
allows the normalized x having a magnitude range from 0 to 1, representing the probability of x
in its direction. The final normalization term can thus be defined as N(x, wi) = (∥x∥+ 1) · ∥wi∥.
However, the OLTR [9] doesn’t use it to de-confound the visual feature. Instead, its x is the joint
embedding of the feature vector and an attentive memory vector. The Decouple [11] also invents
two different types of normalized classifiers: τ-norm classifier and Learnable Weight Scaling (LWS)
classifier. They empirically found that the l2 norm of wi is not uniform in the long-tailed dataset,
and has a positive correlation with the number of training samples for class i, as shown in Figure 6.
Therefore, their normalized classifiers only normalize the wi: the τ-norm classifier is defined as
N(x, wi) = ∥wi∥τ, τ ∈[0, 1] while LWS is N(x, wi) = gi, where gi is a learnable parameter. Yet,
these decouple classifiers fail to de-confound the M →X for two reasons: 1) they don’t considering
the confounding effect on x; 2) they only apply the normalized classifiers on the 2nd stage when the
backbone has already been frozen.
Re-balancing Strategies. Both OLTR [9] and Decouple [11] adopt the same class-aware sampler in
their 2nd stage training, which forces each class to contribute the same number of samples regardless
of the size. To dynamically combine the two training stages, the BBN [10] utilizes a bilateral-branch
design to smoothly transfer the sampling strategy from the imbalanced branch to the re-balancing
branch, where two branches share the same set of parameters but learn from different sampling
strategies, which has the same spirit as two-stage design in OLTR [9] and Decouple [11]. As to the
EQL [17], since the re-sampling is complicated in the object detection and instance segmentation
tasks, where objects from different classes co-exist in one image, they choose the re-weighted loss to
balance the contributions of different classes.
C
Background-Exempted Inference
The results with and without Background-Exempted Inference are reported in Table 5. As we can
see, the Background-Exempted strategy successfully prevents the TDE from hurting the foreground-
background selection. It is the key to apply TDE in tasks like object detection and instance segmen-
tation that include one or more legitimately biased head categories, i.e., this strategy allows us to
conduct TDE on a selected subset of categories.
D
The Difference Between Re-balancing NDE and The Proposed TDE
In this section, we will further discuss the relationship between two-stage re-balancing NDE and the
proposed TDE. As we discussed in Section 4.3 of original paper, the 2nd-stage re-balanced classifier
essentially calculates the NDE(Yi) = [Yd′ = i|do(X = x)] −[Yd′ = i|do(X = x′)], where the
11

--- Page 12 ---
Class A       0
Class B
Training (imbalanced data)
Class A       0
Class B
Training (imbalanced data)
Class A       0
Class B
Testing (balanced data)
Class A       0
Class B
Testing (balanced data)
Mismatch
(a) Baseline
(d) The Proposed TDE
Class A       0
Class B
Training (re-balanced data)
Class A       0
Class B
Testing (balanced data)
Bad Model
(b) One-stage Re-balancing
Training Samples
Re-balanced Samples
Testing Samples
(c) Two-stage Re-balancing
Class A       0
Class B
1st Stage Training (imbalanced data)
Class A       0
Class B
2nd Stage Training (re-balanced data)
Class A       0
Class B
Testing (balanced data)
Figure 7: A simple one-dimensional binary classification example of conventional classifier, one-
/two-stage re-balancing classifiers, and the proposed TDE.
second term can be omitted because x′ is a dummy vector and the moving averaged d′ in a balanced
set won’t point to any specific classes, so it is actually a constant offset. Therefore, the crux of
understanding the NDE would be why the 2nd-stage re-balanced training equals to the first term
[Yd′ = i|do(X = x)]. It is because when the backbone is frozen, it breaks the dependency between
M →X, which is a straightforward implementation of causal intervention do(X = x). The original
OLTR [9] violates this intervention by fine-tuning the backbone parameters in the 2nd stage, and it
thus performs much worse than the Decouple-OLTR in the Table 2 of original paper, which freezes
the backbone parameters. Meanwhile, the balanced re-sampling also brings a fair d′ as we discussed
in the third paragraph of Section A.
To better illustrate both the similarity and the difference between re-balancing NDE and the proposed
TDE, we constructed a one-dimensional binary classification example for conventional classifier,
one-/two-stage re-balancing classifiers, and the proposed TDE in Figure 7, where the gaussian
distribution curve represents the feature distribution generated by the backbone, and the 0 point
is the classifier’s decision boundary. The conventional classifier and one-stage re-balancing are
fundamentally problematic, because they either cause the mismatching in the inference or learn
a bad backbone model. In the meantime, both two-stage re-balancing and the proposed TDE are
able to correctly remove the bias by proper adjustments. The 2nd-stage re-balanced training (NDE)
fixes the backbone parameters do(X = x) learnt from 1st-stage imbalanced training, i.e., the frozen
curve in the image, and then re-samples an artificially balanced data distribution to create a fair d′.
The overall re-balancing NDE can be considered as subtracting a bias offset from original decision
boundary. Meanwhile, the proposed TDE removes the bias effect (head projection) from feature
vectors. Both two types of adjustments can properly remove the head bias in this example. That’s why
TDE and NDE should be theoretically identical in the long-tailed classification scenario. However,
the 2nd-stage re-balancing NDE has two disadvantages: 1) its adjustment requires an additional
training stage to fine-tune the classifier weights, which relies on the accessibility of data distribution;
2) if non-linear modules are applied to the feature vectors, e.g., a global context layer that conducts
interactions among all objects {xj} in an image, the NDE can only remove a linear approximation of
this non-linear activated head bias, while the TDE would be able to maintain the natural interactions
of features in both original logit term and the subtracted counterfactual term. It explains why
the Decouple-OLTR in Table 2 of original paper doesn’t perform as good as Decouple-τ-norm or
Decouple-LWS, because OLTR involves non-linear interactions between feature vectors and memory
vectors, so a linear adjustment on classifier’s decision boundary cannot completely remove the head
bias.
E
Additional Ablation Studies
The hyper-parameters used in original paper are selected according to the performances on ImageNet-
LT val set as shown in Table 6. To further study the multi-head strategy on different normalized
12

--- Page 13 ---
K
τ
γ
α
Many-shot
Medium-shot
Few-shot
Overall
1
16.0
1/32.0
✗
69.8
42.8
14.9
49.4
4
16.0
1/32.0
✗
69.0
42.3
13.1
48.6
2
8.0
1/32.0
✗
69.5
31.3
1.6
42.0
2
32.0
1/32.0
✗
68.6
41.3
13.0
47.9
2
16.0
1/16.0
✗
69.3
44.0
14.2
49.7
2
16.0
1/64.0
✗
69.9
43.3
14.7
49.6
2
16.0
1/32.0
✗
69.5
43.9
15.2
49.8
2
16.0
1/32.0
2.5
66.2
49.8
29.4
53.3
2
16.0
1/32.0
3.0
64.5
50.0
32.6
53.3
2
16.0
1/32.0
3.5
62.5
49.9
36.0
52.9
Table 6: Hyper-parameters selection based on performances of ImageNet-LT val set, where ✗for α
means that TDE inference is not included. The backbone we used here is ResNeXt-50-32x4d.
Methods
#heads K
Many-shot
Medium-shot
Few-shot
Overall
Cosine† [50, 51]
1
67.3
41.3
14.0
47.6
Cosine† [50, 51]
2
67.5
42.1
14.1
48.1
Capsule† [9, 54]
1
67.1
40.0
11.2
46.5
Capsule† [9, 54]
2
67.7
41.3
12.6
47.6
(Ours) De-confound
1
67.3
41.8
15.0
47.9
(Ours) De-confound
2
67.9
42.7
14.7
48.6
(Ours) Cosine-TDE
1
61.8
47.1
30.4
50.5
(Ours) Cosine-TDE
2
63.0
47.3
31.0
51.1
(Ours) Capsule-TDE
1
62.3
46.9
30.6
50.6
(Ours) Capsule-TDE
2
62.4
47.9
31.5
51.2
(Ours) De-confound-TDE
1
62.5
47.8
32.8
51.4
(Ours) De-confound-TDE
2
62.7
48.8
31.6
51.8
Table 7: The performances of cosine classifier [50, 51] and capsule classifier [9, 54] under different
number of head K on ImageNet-LT test set. Other hyper-parameters are fixed.
Methods
Backbone
Many-shot
Medium-shot
Few-shot
Overall
Baseline
ResNeXt-50
66.1
38.4
8.9
45.0
De-confound
ResNeXt-50
67.9
42.7
14.7
48.6
De-confound-TDE
ResNeXt-50
62.7
48.8
31.6
51.8
Baseline
ResNeXt-101
68.7
42.5
11.8
48.4
De-confound
ResNeXt-101
68.9
44.3
16.5
50.0
De-confound-TDE
ResNeXt-101
64.7
50.0
33.0
53.3
Table 8: The performances of the proposed method under different backbones in ImageNet-LT test
set.
Methods
Backbone
AP
AP50
AP75
APr
APc
APf
APbbox
Baseline
R101-FPN
22.6
33.5
24.4
2.5
23.0
30.2
24.3
De-confound
R101-FPN
25.7
38.5
27.8
11.4
26.1
30.9
27.7
De-confound-TDE
R101-FPN
28.4
43.0
30.6
22.1
29.0
30.3
31.0
Baseline
X101-FPN
26.4
39.5
28.4
7.4
28.1
32.0
28.5
De-confound
X101-FPN
28.4
41.9
30.6
13.3
29.5
32.9
30.5
De-confound-TDE
X101-FPN
30.4
45.1
32.9
21.1
31.8
32.3
33.1
Table 9: The performances of the proposed method under different backbones in LVIS V0.5 val set.
13

--- Page 14 ---
Methods
AP
AP50
AP75
APr
APc
APf
Baseline
19.4
29.8
20.6
3.9
21.9
30.8
De-confound
20.8
31.8
22.1
7.4
22.7
31.2
De-confound-TDE
23.0
35.2
24.1
12.7
24.5
30.7
Table 10: The single model performances of the proposed method on LVIS V0.5 evaluation test
server [62].
classifiers, we tested the K = 2 on cosine classifier [50, 51] and capsule classifier [9, 54] in Table 7.
It proves that the advantage of the proposed de-confounded model doesn’t come from larger K, and
the multi-head fine-grained sampling can generally improves the de-confounded training, no matter
what kind of normalization function we choose.
As shown in Table 8,9, we tested the proposed method on different backbones. After equipped
with ResNeXt-101-32x4d and ResNeXt-101-64x4d [2] for ImageNet-LT [9] and LVIS [7] V0.5,
respectively, the proposed method gains additional improvements. In ImageNet-LT dataset, we
changed some hyper-parameters (K = 4, γ = 1/64.0) and increased the training epochs to 120,
because of the significantly increased number of model parameters. The hyper-parameters for LVIS
are still the same as original paper.
We also reported the performances of the proposed method on LVIS V0.5 evaluation test server [62]
in Table 10, where we used ResNeXt-101-64x4d backbone and the original hyper-parameters. It’s
worth noting that these are single model performances, which neither exploited external dataset nor
utilized any model enhancement tricks.
References
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition.
In CVPR, 2016.
[2] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transforma-
tions for deep neural networks. In CVPR, 2017.
[3] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection
with region proposal networks. In Advances in neural information processing systems, pages 91–99, 2015.
[4] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge.
International journal of computer vision, 115(3):211–252, 2015.
[5] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár,
and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, pages 740–755. Springer,
2014.
[6] William J Reed. The pareto, zipf and other power laws. Economics letters, 74(1):15–19, 2001.
[7] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation.
In CVPR, pages 5356–5364, 2019.
[8] Tong Wu, Qingqiu Huang, Ziwei Liu, Yu Wang, and Dahua Lin. Distribution-balanced loss for multi-label
classification in long-tailed datasets. In ECCV, 2020.
[9] Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X Yu. Large-scale
long-tailed recognition in an open world. In CVPR, 2019.
[10] Boyan Zhou, Quan Cui, Xiu-Shen Wei, and Zhao-Min Chen. Bbn: Bilateral-branch network with
cumulative learning for long-tailed visual recognition. In CVPR, 2020.
[11] Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, and Yannis
Kalantidis. Decoupling representation and classifier for long-tailed recognition. In ICLR, 2020.
[12] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. Learning imbalanced datasets
with label-distribution-aware margin loss. In Advances in Neural Information Processing Systems, pages
1567–1578, 2019.
14

--- Page 15 ---
[13] Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. Class-balanced loss based on
effective number of samples. In CVPR, pages 9268–9277, 2019.
[14] Salman H Khan, Munawar Hayat, Mohammed Bennamoun, Ferdous A Sohel, and Roberto Togneri.
Cost-sensitive learning of deep feature representations from imbalanced data. IEEE transactions on neural
networks and learning systems, 2017.
[15] Li Shen, Zhouchen Lin, and Qingming Huang. Relay backpropagation for effective learning of deep
convolutional neural networks. In ECCV, pages 467–482. Springer, 2016.
[16] Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin
Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In ECCV,
2018.
[17] Jingru Tan, Changbao Wang, Buyu Li, Quanquan Li, Wanli Ouyang, Changqing Yin, and Junjie Yan.
Equalization loss for long-tailed object recognition. In CVPR, 2020.
[18] Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and
momentum in deep learning. In ICML, pages 1139–1147, 2013.
[19] Ning Qian. On the momentum term in gradient descent learning algorithms. Neural networks, 12(1):145–
151, 1999.
[20] Diederik P Kingma and Jimmy Ba.
Adam: A method for stochastic optimization.
arXiv preprint
arXiv:1412.6980, 2014.
[21] Judea Pearl. Causal diagrams for empirical research. Biometrika, 82(4):669–688, 1995.
[22] Judea Pearl. Direct and indirect effects. In Proceedings of the 17th conference on uncertainty in artificial
intelligence. Morgan Kaufmann Publishers Inc., 2001.
[23] Judea Pearl, Madelyn Glymour, and Nicholas P Jewell. Causal inference in statistics: A primer. John
Wiley & Sons, 2016.
[24] Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: Delving into high quality object detection. In CVPR,
2018.
[25] Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. Smote: synthetic
minority over-sampling technique. Journal of artificial intelligence research, 16:321–357, 2002.
[26] Chris Drummond, Robert C Holte, et al. Class imbalance and cost sensitivity: why under-sampling beats
over-sampling. In Workshop on learning from imbalanced datasets II, volume 11, pages 1–8. Citeseer,
2003.
[27] Xinting Hu, Yi Jiang, Kaihua Tang, Jingyuan Chen, Chunyan Miao, and Hanwang Zhang. Learning to
segment the tail. In CVPR, 2020.
[28] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object
detection. In ICCV, 2017.
[29] Jun Shu, Qi Xie, Lixuan Yi, Qian Zhao, Sanping Zhou, Zongben Xu, and Deyu Meng. Meta-weight-net:
Learning an explicit mapping for sample weighting. In Advances in Neural Information Processing
Systems, 2019.
[30] Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urtasun. Learning to reweight examples for robust
deep learning. arXiv preprint arXiv:1803.09050, 2018.
[31] Muhammad Abdullah Jamal, Matthew Brown, Ming-Hsuan Yang, Liqiang Wang, and Boqing Gong.
Rethinking class-balanced methods for long-tailed visual recognition from a domain adaptation perspective.
In CVPR, 2020.
[32] Yu Li, Tao Wang, Bingyi Kang, Sheng Tang, Chunfeng Wang, Jintao Li, and Jiashi Feng. Overcoming
classifier imbalance for long-tail object detection with balanced group softmax. In CVPR, 2020.
[33] Yiru Wang, Weihao Gan, Jie Yang, Wei Wu, and Junjie Yan. Dynamic curriculum learning for imbalanced
data classification. In ICCV, 2019.
[34] Jialun Liu, Yifan Sun, Chuchu Han, Zhaopeng Dou, and Wenhui Li. Deep representation learning on
long-tailed data: A learnable embedding augmentation perspective. In CVPR, 2020.
15

--- Page 16 ---
[35] Judea Pearl and Dana Mackenzie. The Book of Why: The New Science of Cause and Effect. Basic Books,
2018.
[36] David P MacKinnon, Amanda J Fairchild, and Matthew S Fritz. Mediation analysis. Annu. Rev. Psychol.,
2007.
[37] Luke Keele. The statistics of causal inference: A view from political methodology. Political Analysis,
2015.
[38] Lorenzo Richiardi, Rino Bellocco, and Daniela Zugna. Mediation analysis in epidemiology: methods,
interpretation and bias. International journal of epidemiology, 2013.
[39] Kaihua Tang, Yulei Niu, Jianqiang Huang, Jiaxin Shi, and Hanwang Zhang. Unbiased scene graph
generation from biased training. In CVPR, 2020.
[40] Jiaxin Qi, Yulei Niu, Jianqiang Huang, and Hanwang Zhang. Two causal principles for improving visual
dialog. In CVPR, 2020.
[41] Yulei Niu, Kaihua Tang, Hanwang Zhang, Zhiwu Lu, Xian-Sheng Hua, and Ji-Rong Wen. Counterfactual
vqa: A cause-effect look at language bias. arXiv preprint arXiv:2006.04315, 2020.
[42] Xu Yang, Hanwang Zhang, and Jianfei Cai. Deconfounded image captioning: A causal retrospect. arXiv
preprint arXiv:2003.03923, 2020.
[43] Dong Zhang, Hanwang Zhang, Jinhui Tang, Xiansheng Hua, and Qianru Sun. Causal intervention for
weakly-supervised semantic segmentation. In NeurIPS, 2020.
[44] Zhongqi Yue, Hanwang Zhang, Qianru Sun, and Xian-Sheng Hua. Interventional few-shot learning. In
NeurIPS, 2020.
[45] SGD implementation in PyTorch. https://pytorch.org/docs/stable/_modules/torch/optim/
sgd.html.
[46] Tyler J VanderWeele. A three-way decomposition of a total effect into direct, indirect, and interactive
effects. Epidemiology (Cambridge, Mass.), 2013.
[47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing
systems, pages 5998–6008, 2017.
[48] Yann LeCun, Sumit Chopra, Raia Hadsell, M Ranzato, and F Huang. A tutorial on energy-based learning.
Predicting structured data, 2006.
[49] Peter C Austin. An introduction to propensity score methods for reducing the effects of confounding in
observational studies. Multivariate behavioral research, 2011.
[50] Spyros Gidaris and Nikos Komodakis. Dynamic few-shot visual learning without forgetting. In CVPR,
2018.
[51] Hang Qi, Matthew Brown, and David G Lowe. Low-shot learning with imprinted weights. In CVPR, 2018.
[52] Judea Pearl. On the consistency rule in causal inference: axiom, definition, assumption, or theorem?
Epidemiology, 21(6):872–875, 2010.
[53] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. Bpr: Bayesian
personalized ranking from implicit feedback. In UAI, 2009.
[54] Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. Dynamic routing between capsules. In Advances in
neural information processing systems, pages 3856–3866, 2017.
[55] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and
Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In ICCV,
2017.
[56] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk
minimization. In ICLR, 2018.
[57] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint
arXiv:1608.03983, 2016.
16

--- Page 17 ---
[58] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng,
Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu
Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change
Loy, and Dahua Lin. MMDetection: Open mmlab detection toolbox and benchmark. arXiv preprint
arXiv:1906.07155, 2019.
[59] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature
pyramid networks for object detection. In CVPR, pages 2117–2125, 2017.
[60] Douglas C Montgomery and George C Runger. Applied statistics and probability for engineers. John
Wiley & Sons, 2010.
[61] Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, and Jia-Bin Huang. A closer look at
few-shot classification. In ICLR, 2019.
[62] LVIS v0.5 Evaluation Server. https://evalai.cloudcv.org/web/challenges/challenge-page/
473/overview.
17
