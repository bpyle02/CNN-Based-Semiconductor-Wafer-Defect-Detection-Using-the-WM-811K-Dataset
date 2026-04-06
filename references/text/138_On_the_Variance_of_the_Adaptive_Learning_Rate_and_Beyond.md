# On the Variance of the Adaptive Learning Rate and Beyond

**Authors**: Liu, Jiang, He, Chen, Liu, Gao, Han
**Year**: 2020
**arXiv**: 1908.03265
**Topic**: optimization
**Relevance**: RAdam optimizer for robust early training

---


--- Page 1 ---
Published as a conference paper at ICLR 2020
ON THE VARIANCE OF THE ADAPTIVE LEARNING
RATE AND BEYOND
Liyuan Liu ∗
University of Illinois, Urbana-Champaign
ll2@illinois
Haoming Jiang †
Georgia Tech
jianghm@gatech.edu
Pengcheng He, Weizhu Chen
Microsoft Dynamics 365 AI
{penhe,wzchen}@microsoft.com
Xiaodong Liu, Jianfeng Gao
Microsoft Research
{xiaodl,jfgao}@microsoft.com
Jiawei Han
University of Illinois, Urbana-Champaign
hanj@illinois
ABSTRACT
The learning rate warmup heuristic achieves remarkable success in stabilizing
training, accelerating convergence and improving generalization for adaptive
stochastic optimization algorithms like RMSprop and Adam. Pursuing the theory
behind warmup, we identify a problem of the adaptive learning rate – its vari-
ance is problematically large in the early stage, and presume warmup works as a
variance reduction technique. We provide both empirical and theoretical evidence
to verify our hypothesis. We further propose Rectiﬁed Adam (RAdam), a novel
variant of Adam, by introducing a term to rectify the variance of the adaptive
learning rate. Experimental results on image classiﬁcation, language modeling,
and neural machine translation verify our intuition and demonstrate the efﬁcacy
and robustness of RAdam.1
1
INTRODUCTION
Adam-eps
Adam-2k
Adam-vanilla
RAdam
Adam-warmup
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
0
10k
20k
30k
40k
50k
60k
70k
Training loss
Overlapped
Figure 1: Training loss v.s. # of
iterations of Transformers on the
De-En IWSLT’14 dataset.
Fast and stable optimization algorithms are what generations
of researchers have been pursuing (Gauss, 1823; Cauchy,
1847). Remarkably, stochastic gradient-based optimization,
such as stochastic gradient descent (SGD), has witnessed
tremendous success in many ﬁelds of science and engineering
despite its simplicity. Recently, many efforts have been made
to accelerate optimization by applying adaptive learning rate.
In particular, Adagrad (Duchi et al., 2010) and its variants, e.g.,
RMSprop (Hinton et al., 2012), Adam (Kingma & Ba, 2014),
Adadelta (Zeiler, 2012) and Nadam (Dozat, 2016), stand out
due to their fast convergence, and have been considered as the
optimizer of choice in many applications.
However, it has been observed that these optimization methods may converge to bad/suspicious
local optima, and have to resort to a warmup heuristic – using a small learning rate in the ﬁrst
few epochs of training to mitigate such problem (Vaswani et al., 2017; Popel & Bojar, 2018). For
example, when training typical Transformers based neural machine translation models on the De-En
IWSLT’14 dataset, removing the warmup stage increases the training loss from 3 to around 10, as
shown in Figure 1. Similar phenomena are observed in other scenarios like BERT (a bidirectional
transformer language model) pre-training (Devlin et al., 2019).
Due to the lack of the theoretical underpinnings, there is neither guarantee that warmup would bring
consistent improvements for various machine learning settings nor guidance on how we should
∗Work was done during an internship at Microsoft Dynamics 365 AI.
†Work was done during an internship at Microsoft Dynamics 365 AI.
1All implementations are available at: https://github.com/LiyuanLucasLiu/RAdam.
1
arXiv:1908.03265v4  [cs.LG]  26 Oct 2021

--- Page 2 ---
Published as a conference paper at ICLR 2020
conduct warmup. Thus, researchers typically use different settings in different applications and
have to take a trial-and-error approach, which can be tedious and time-consuming.
In this paper, we conduct both empirical and theoretical analysis of the convergence issue to identify
its origin. We show that its root cause is: the adaptive learning rate has undesirably large variance in
the early stage of model training, due to the limited amount of training samples being used. Thus,
to reduce such variance, it is better to use smaller learning rates in the ﬁrst few epochs of training,
which justiﬁes the warmup heuristic.
Inspired by our analysis results, we propose a new variant of Adam, called Rectiﬁed Adam (RAdam),
which explicitly rectiﬁes the variance of the adaptive learning rate based on derivations. We conduct
extensive experiments on language modeling, image classiﬁcation, and neural machine translation.
RAdam brings consistent improvement over the vanilla Adam, which veriﬁes the variance issue
generally exists on various tasks across different network architectures.
In summary, our main contributions are two-fold:
• We identify the variance issue of the adaptive learning rate and present a theoretical justiﬁcation
for the warmup heuristic. We show that the convergence issue is due to the undesirably large
variance of the adaptive learning rate in the early stage of model training.
• We propose a new variant of Adam (i.e., RAdam), which not only explicitly rectiﬁes the variance
and is theoretically sound, but also compares favorably with the heuristic warmup.
2
PRELIMINARIES AND MOTIVATIONS
Generic adaptive methods. Algorithm 1 is a generic framework (all operations are element-wise).
It describes various popular stochastic gradient descent algorithms (Reddi et al., 2018). Speciﬁcally,
different optimization algorithms can be speciﬁed by different choices of φ(.) and ψ(.), where φ(.)
speciﬁes how the momentum at time step t is calculated, and ψ(.) how the adaptive learning rate at
t is calculated. For example, in the Adam algorithm, we have:
φ(g1, · · · , gt) = (1 −β1) Pt
i=1 βt−i
1
gi
1 −βt
1
and
ψ(g1, · · · , gt) =
s
1 −βt
2
(1 −β2) Pt
i=1 βt−i
2
g2
i
. (1)
For numerical stability, the function ψ(.) in Equation 1 is usually calculated as bψ(g1, · · · , gt) =
√
1−βt
2
ϵ+√
(1−β2) Pt
i=1 βt−i
2
g2
i
, where ϵ is a relatively small / negligible value (e.g., 1 × 10−8).
Algorithm 1: Generic adaptive optimization method setup. All operations are element-wise.
Input: {αt}T
t=1: step size, {φt, ψt}T
t=1: function to calculate momentum and adaptive rate,
θ0: initial parameter, f(θ): stochastic objective function.
Output: θT : resulting parameters
1 while t = 1 to T do
2
gt ←∇θft(θt−1) (Calculate gradients w.r.t. stochastic objective at timestep t)
3
mt ←φt(g1, · · · , gt) (Calculate momentum)
4
lt ←ψt(g1, · · · , gt) (Calculate adaptive learning rate)
5
θt ←θt−1 −αtmtlt (Update parameters)
6 return θT
Learning rate warmup. Instead of setting the learning rate αt as a constant or in a decreasing
order, a learning rate warmup strategy sets αt as smaller values in the ﬁrst few steps, thus not
satisfying ∀t αt+1 ≤αt. For example, linear warmup sets αt = t α0 when t < Tw. Warmup has
been demonstrated to be beneﬁcial in many deep learning applications. For example, in the NMT
experiments in Figure 1, the training loss convergences around 10 when warmup is not applied
(Adam-vanilla), and it surprisingly decreases to below 3 after applying warmup (Adam-warmup).
To further analyze this phenomenon, we visualize the histogram of the absolute value of gradients
on a log scale in Figure 2. We observe that, without applying warmup, the gradient distribution
is distorted to have a mass center in relatively small values within 10 updates. Such gradient dis-
tortion means that the vanilla Adam is trapped in bad/suspicious local optima after the ﬁrst few
2

--- Page 3 ---
Published as a conference paper at ICLR 2020
Iteration
Adam with warmup
Adam without warmup
6.76×10'
9.38×10'
Iteration
Iteration
4.08×10'
4.08×10'
Iteration
< -./0
-.1'
-.1/
-.2
< -./0
-.1'
-.1/
-.2
< -./0
-.1'
-.1/
-.2
< -./0
-.1'
-.1/
-.2
-.3
1
10
25
50
75
100
5
1
10
25
50
75
100
5
1
40K
70k
1
40K
70k
The distribution is distorted within 10 updates. 
Figure 2: The absolute gradient histogram of the Transformers on the De-En IWSLT’ 14 dataset
during the training (stacked along the y-axis). X-axis is absolute value in the log scale and the
height is the frequency. Without warmup, the gradient distribution is distorted in the ﬁrst 10 steps.
   
   
   
  
   
   
   
  
   
   
   
  
Adam-2k
5.72 × 106
RAdam
6.82 × 106
Adam-eps
5.42 × 106
10−20
𝑒−16
𝑒−12
𝑒−8
< 𝑒−20
𝑒−16
𝑒−12
𝑒−8
< 𝑒−20
𝑒−16
𝑒−12
𝑒−8
Iteration
Iteration
Iteration
< 𝑒−20
1
40K
70k
1
40K
70k
1
40K
70k
Figure 3: The histogram of the absolute value of gradients (on a log scale) during the training of
Transformers on the De-En IWSLT’ 14 dataset. using Adam-2k, RAdam and Adam-eps.
updates. Warmup essentially reduces the impact of these problematic updates to avoid the conver-
gence problem. In the following sections, we focus our analysis on learning rate warmup for the
Adam algorithm, while it can be applied to other algorithms that use similar adaptive learning rate
(ψ(.)) designs, e.g., RMSprop (Hinton et al., 2012) and Nadam (Dozat, 2016).
3
VARIANCE OF THE ADAPTIVE LEARNING RATE
In this section, we ﬁrst introduce empirical evidence, then analyze the variance of the adaptive
learning rate to support our hypothesis – Due to the lack of samples in the early stage, the adaptive
learning rate has an undesirably large variance, which leads to suspicious/bad local optima.
To convey our intuition, we begin with a special case. When t = 1, we have ψ(g1) =
p
1/g2
1.
We view {g1, · · · , gt} as i.i.d. Gaussian random variables following N(0, σ2)2. Therefore, 1/g2
1
is subject to the scaled inverse chi-squared distribution, Scale-inv-X 2(1, 1/σ2), and Var[
p
1/g2
1]
is divergent. It means that the adaptive ratio can be undesirably large in the ﬁrst stage of learning.
Meanwhile, setting a small learning rate at the early stage can reduce the variance (Var[αx] =
α2 Var[x]), thus alleviating this problem. Therefore, we suggest it is the unbounded variance of the
adaptive learning rate in the early stage that causes the problematic updates.
3.1
WARMUP AS VARIANCE REDUCTION
In this section, we design a set of controlled experiments to verify our hypothesis. Particularly, we
design two variants of Adam that reducing the variance of the adaptive learning rate: Adam-2k and
Adam-eps. We compare them to vanilla Adam with and without warmup on the IWSLT’14 German
to English translation dataset (Cettolo et al., 2014).
In order to reduce the variance of the adaptive learning rate (ψ(.)), Adam-2k only updates ψ(.) in the
ﬁrst two thousand iterations, while the momentum (φ(.)) and parameters (θ) are ﬁxed3; other than
this, it follows the original Adam algorithm. To make comparison with other methods, its iterations
are indexed from -1999 instead of 1. In Figure 1, we observe that, after getting these additional
two thousand samples for estimating the adaptive learning rate, Adam-2k avoids the convergence
problem of the vanilla-Adam. Also, comparing Figure 2 and Figure 3, getting large enough samples
prevents the gradient distribution from being distorted. These observations verify our hypothesis
that the lack of sufﬁcient data samples in the early stage is the root cause of the convergence issue.
2The mean zero normal assumption is valid at the beginning of the training, since weights are sampled from
normal distributions with mean zero (Balduzzi et al., 2017), further analysis is conducted in Section 5.3.
3Different from Gotmare et al. (2019), all parameters and ﬁrst moments are frozen in the ﬁrst 2000 iterations.
3

--- Page 4 ---
Published as a conference paper at ICLR 2020
Another straightforward way to reduce the variance is to increase the value of ϵ in bψ(g1, · · · , gt) =
√
1−βt
2
ϵ+√
(1−β2) Pt
i=1 βt−i
2
g2
i
. Actually, if we assume bψ(.) is subject to the uniform distribution, its vari-
ance equals to
1
12ϵ2 . Therefore, we design Adam-eps, which uses a non-negligibly large ϵ = 10−4,
while ϵ = 10−8 for vanilla Adam. Its performance is summarized in Figure 1. We observe that it
does not suffer from the serious convergence problem of vanilla-Adam. This further demonstrates
that the convergence problem can be alleviated by reducing the variance of the adaptive learning
rate, and also explains why tuning ϵ is important in practice (Liu et al., 2019). Besides, similar to
Adam-2k, it prevents the gradient distribution from being distorted (as shown in Figure 3). However,
as in Figure 1, it produces a much worse performance comparing to Adam-2k and Adam-warmup.
We conjecture that this is because large ϵ induces a large bias into the adaptive learning rate and
slows down the optimization process. Thus, we need a more principled and rigorous way to con-
trol the variance of the adaptive learning rate. In the next subsection, we will present a theoretical
analysis of the variance of the adaptive learning rate.
3.2
ANALYSIS OF ADAPTIVE LEARNING RATE VARIANCE
As mentioned before, Adam uses the exponential moving average to calculate the adaptive learning
rate. For gradients {g1, · · · , gt}, their exponential moving average has a larger variance than their
simple average. Also, in the early stage (t is small), the difference of the exponential weights of
{g1, · · · , gt} is relatively small (up to 1 −βt−1
2
). Therefore, for ease of analysis, we approximate
the distribution of the exponential moving average as the distribution of the simple average (Nau,
2014), i.e., p(ψ(.)) = p(
r
1−βt
2
(1−β2) Pt
i=1 βt−i
2
g2
i ) ≈p(
q
t
Pt
i=1 g2
i ). Since gi ∼N(0, σ2), we have
t
Pt
i=1 g2
i ∼Scale-inv-X 2(t, 1
σ2 ). Therefore, we assume
1−βt
2
(1−β2) Pt
i=1 βt−i
2
g2
i also subjects to a scaled
inverse chi-square distribution with ρ degrees of freedom (further analysis on this approximation is
conducted in Section 5.3). Based on this assumption, we can calculate Var[ψ2(.)] and the PDF of
ψ2(.). Now, we proceed to the analysis of its square root variance, i.e., Var[ψ(.)], and show how the
variance changes with ρ (which corresponds to number of used training samples).
Theorem 1. If ψ2(.) ∼Scale-inv-X 2(ρ, 1
σ2 ), Var[ψ(.)] monotonically decreases as ρ increases.
Proof. For ∀ρ > 4, we have:
Var[ψ(.)] = E[ψ2(.)] −E[ψ(.)]2 = τ 2(
ρ
ρ −2 −ρ 22ρ−5
π
B(ρ −1
2
, ρ −1
2
)2),
(2)
where B(.) is the beta function. By analyzing the derivative of Var[ψ(.)], we know it monotonically
decreases as ρ increases. The detailed derivation is elaborated in the Appendix A.
Theorem 1 gives a qualitative analysis of the variance of the adaptive learning rate. It shows that,
due to the lack of used training samples in the early stage, Var[ψ(.)] is larger than the late stage
(Figure 8). To rigorously constraint the variance, we perform a quantiﬁed analysis on Var[ψ(.)] by
estimating the degree of freedoms ρ.
4
RECTIFIED ADAPTIVE LEARNING RATE
In the previous section, Equation 2 gives the analytic form of Var[ψ(.)], where ρ is the degree of
freedoms. Here, we ﬁrst give an estimation of ρ based on t to conduct a quantiﬁed analysis for
Var[ψ(g1, · · · , gt)], then we describe the design of the learning rate rectiﬁcation, and compare it to
the heuristic warmup strategies.
4.1
ESTIMATION OF ρ
The exponential moving average (EMA) can be interpreted as an approximation to the simple mov-
ing average (SMA) in real application (Nau, 2014), i.e.,
p
 
(1 −β2) Pt
i=1 βt−i
2
g2
i
1 −βt
2
!
≈p
 Pf(t,β2)
i=1
g2
t+1−i
f(t, β2)
!
.
(3)
4

--- Page 5 ---
Published as a conference paper at ICLR 2020
Algorithm 2: Rectiﬁed Adam. All operations are element-wise.
Input: {αt}T
t=1: step size, {β1, β2}: decay rate to calculate moving average and moving 2nd
moment, θ0: initial parameter, ft(θ): stochastic objective function.
Output: θt: resulting parameters
1 m0, v0 ←0, 0 (Initialize moving 1st and 2nd moment)
2 ρ∞←2/(1 −β2) −1 (Compute the maximum length of the approximated SMA)
3 while t = {1, · · · , T} do
4
gt ←∇θft(θt−1) (Calculate gradients w.r.t. stochastic objective at timestep t)
5
vt ←β2vt−1 + (1 −β2)g2
t (Update exponential moving 2nd moment)
6
mt ←β1mt−1 + (1 −β1)gt (Update exponential moving 1st moment)
7
c
mt ←mt/(1 −βt
1) (Compute bias-corrected moving average)
8
ρt ←ρ∞−2tβt
2/(1 −βt
2)(Compute the length of the approximated SMA)
9
if the variance is tractable, i.e., ρt > 4 then
10
lt ←
p
(1 −βt
2)/vt (Compute adaptive learning rate)
11
rt ←
q
(ρt−4)(ρt−2)ρ∞
(ρ∞−4)(ρ∞−2)ρt (Compute the variance rectiﬁcation term)
12
θt ←θt−1 −αtrt c
mtlt (Update parameters with adaptive momentum)
13
else
14
θt ←θt−1 −αt c
mt (Update parameters with un-adapted momentum)
15 return θT
where f(t, β2) is the length of the SMA which allows the SMA to have the same “center of mass”
with the EMA. In other words, f(t, β2) satisﬁes:
(1 −β2) Pt
i=1 βt−i
2
· i
1 −βt
2
=
Pf(t,β2)
i=1
(t + 1 −i)
f(t, β2)
.
(4)
By solving Equation 4, we have: f(t, β2) =
2
1−β2 −1 −
2tβt
2
1−βt
2 .
In the previous section,
we assume:
1−βt
2
(1−β2) Pt
i=1 βt−i
2
g2
i
∼Scale-inv-X 2(ρ, 1
σ2 ). Here, since gi ∼N(0, σ2), we have
Pf(t,β2)
i=1
g2
t+1−i
f(t,β2)
∼Scale-inv-X 2(f(t, β2), 1
σ2 ). Thus, Equation 3 views Scale-inv-X 2(f(t, β2), 1
σ2 )
as an approximation to Scale-inv-X 2(ρ, 1
σ2 ). Therefore, we treat f(t, β2) as an estimation of ρ. For
ease of notation, we mark f(t, β2) as ρt. Also, we refer
2
1−β2 −1 as ρ∞(maximum length of the
approximated SMA), due to the inequality f(t, β2) ≤limt→∞f(t, β2) =
2
1−β2 −1.
4.2
VARIANCE ESTIMATION AND RECTIFICATION
Based on previous estimations, we have Var[ψ(.)] = τ 2(
ρt
ρt−2 −ρt 22ρt−5
π
B( ρt−1
2
, ρt−1
2
)2). The
value of this function in the early stage is signiﬁcantly larger than the late stage (as analyzed later, it
decays roughly at the speed of O( 1
ρt )). For example, the variance at ρt = 5 is over 100 times larger
than the variance at ρt = 500. Additionally, based on Theorem 1, we know minρt Var[ψ(.)] =
Var[ψ(.)]|ρt=ρ∞and mark this minimal value as Cvar. In order to ensure that the adaptive learning
rate (ψ(.)) has consistent variance, we rectify the variance at the t-th timestamp as below,
Var[rt ψ(g1, · · · , gt)] = Cvar
where
rt =
p
Cvar/Var[ψ(g1, · · · , gt)].
Although we have the analytic form of Var[ψ(.)] (i.e., Equation 2), it is not numerically stable.
Therefore, we use the ﬁrst-order approximation to calculate the rectiﬁcation term. Speciﬁcally, by
approximating
p
ψ2(.) to the ﬁrst order (Wolter, 2007),
p
ψ2(.) ≈
p
E[ψ2(.)] +
1
2
p
E[ψ2(.)]
(ψ2(.) −E[ψ2(.)])
and
Var[ψ(.)] ≈Var[ψ2(.)]
4 E[ψ2(.)] .
Since ψ2(.) ∼Scale-inv-X 2(ρt, 1
σ2 ), we have:
Var[ψ(.)] ≈ρt/[2(ρt −2)(ρt −4)σ2].
(5)
In Section 5.3, we conduct simulation experiments to examine Equation 5 and ﬁnd that it is a reliable
approximation. Based on Equation 5, we know that Var[
p
ψ(.)] decreases approximately at the
5

--- Page 6 ---
Published as a conference paper at ICLR 2020
40
42
44
46
48
50
52
54
56
58
60
0
0.5M
1M
1.5M
2M
2.5M
3M
3.5M
4M
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
10 11 12 13
110
120
130
140
150
160
170
10k 12k 14k 16k 18k 20k 22k
8k
Training PPL
Test PPL
Gradient updates
Iterations over training set
RAdam
Adam
36.92
35.70
Figure 4: Language modeling (LSTMs) on the One Billion Word.
Table 1: Image Classiﬁcation
Method
Acc.
CIFAR10
SGD
91.51
Adam
90.54
RAdam
91.38
ImageNet
SGD
69.86
Adam
66.54
RAdam
67.62
80
82
84
86
88
90
92
0
20
40
60
80
100
120
140
160
Adam
SGD
RAdam
Iteration over entire dataset
Test accuracy
46
48
50
52
54
56
58
60
62
64
66
68
70
0
10
20
30
40
50
60
70
80
90
1.2
1.3
1.4
1.5
1.6
1.7
1.8
1.9
2
2.1
2.2
2.3
2.4
0
10
20
30
40
50
60
70
80
90
0
20
40
60
80
100
120
140
160
0
0.05
0.1
0.15
0.2
0.25
0.3
0.35
0.4
0.45
0.5
0.55
Training loss
Iteration over entire dataset
Iteration over entire dataset
Iteration over entire dataset
Test accuracy
Training loss
ImageNet
CIFAR10
Figure 5: Training of ResNet-18 on the ImageNet and ResNet-20 on the CIFAR10 dataset.
speed of O( 1
ρt ). With this approximation, we can calculate the rectiﬁcation term as:
rt =
s
(ρt −4)(ρt −2)ρ∞
(ρ∞−4)(ρ∞−2)ρt
.
Applying our rectiﬁcation term to Adam, we come up with a new variant of Adam, Rectiﬁed Adam
(RAdam), as summarized in Algorithm 2. Speciﬁcally, when the length of the approximated SMA is
less or equal than 4, the variance of the adaptive learning rate is intractable and the adaptive learning
rate is inactivated. Otherwise, we calculate the variance rectiﬁcation term and update parameters
with the adaptive learning rate. It is worth mentioning that, if β2 ≤0.6, we have ρ∞≤4 and
RAdam is degenerated to SGD with momentum.
4.3
IN COMPARISON WITH WARMUP AND OTHER STABILIZATION TECHNIQUES
Different from the analysis in this paper, warmup is originally proposed to handle training with very
large batches for SGD (Goyal et al., 2017; Gotmare et al., 2019; Bernstein et al., 2018; Xiao et al.,
2017). We notice that rt has a similar form to the heuristic linear warmup, which can be viewed as
setting the rectiﬁcation term as min(t,Tw)
Tw
. It veriﬁes our intuition that warmup works as a variance
reduction technique. RAdam deactivates the adaptive learning rate when its variance is divergent,
thus avoiding undesired instability in the ﬁrst few updates. Besides, our method does not require an
additional hyperparameter (i.e., Tw) and can automatically adapt to different moving average rules.
Here, we identify and address an underlying issue of adaptive optimization methods independent
of (neural) model architectures. Thus, the proposed rectiﬁcation term is orthogonal to other train-
ing stabilization techniques such as gradient clipping (Bengio et al., 2013), smoothing the adaptive
learning rate (i.e., increasing ϵ, applying geometric mean ﬁlter (Chen et al., 2018), or adding range
constraints (Luo et al., 2019)), initialization (Balduzzi et al., 2017; Zhang et al., 2019) and normal-
ization (Ba et al., 2016; Ioffe & Szegedy, 2015). Indeed, these techniques can be combined with the
proposed variance rectiﬁcation method.
5
EXPERIMENTS
We evaluate RAdam on several benchmarks: One Billion Word for language modeling; Cifar10
and ImageNet for image classiﬁcation; IWSLT’14 De-En/EN-DE and WMT’16 EN-De for neural
machine translation. Following Loshchilov & Hutter (2018), we decouple weight decays in the
vanilla Adam, Adam with warmup and RAdam in our experiments. Details are in Appendix B.
6

--- Page 7 ---
Published as a conference paper at ICLR 2020
78
80
82
84
86
88
90
92
0
20
40
60
80
100
120
140
160
180
0
0.05
0.1
0.15
0.2
0.25
0.3
0.35
0.4
0.45
0.5
0.55
0
20
40
60
80
100
120
140
160
0
0.05
0.1
0.15
0.2
0.25
0.3
0.35
0.4
0.45
0.5
0.55
0
20
40
60
80
100
120
140
160
lr = 0.1
lr = 0.03
lr = 0.01
lr = 0.003
SGD
RAdam
Adam
Test accuracy
Training loss
0
0.05
0.1
0.15
0.2
0.25
0.3
0.35
0.4
0.45
0.5
0.55
0
20
40
60
80
100
120
140
160
78
80
82
84
86
88
90
92
0
20
40
60
80
100
120
140
160
78
80
82
84
86
88
90
92
0
20
40
60
80
100
120
140
160
Diﬀerent learning 
rates lead to similar 
performance.
Sensitive to the choice 
of the learning rate.
X-axis is the 
epoch #.  
Figure 6: Performance of RAdam, Adam and SGD with different learning rates on CIFAR10.
87
87.5
88
88.5
89
89.5
90
90.5
91
91.5
0
20
40
60
80
100
120
140
160
lr = 0.1
lr = 0.03
lr = 0.01
lr = 0.003
Test accuracy
Training loss
200
Comparing to RAdam, heuristic linear warmup needs to tune the warmup length to get the similar performance.
1000
0
0.02
0.04
0.06
0.08
0.1
0.12
0.14
0.16
0.18
0.2
0.22
0
20
40
60
80
100
120
140
160
87
87.5
88
88.5
89
89.5
90
90.5
91
91.5
0
20
40
60
80
100
120
140
160
0
0.02
0.04
0.06
0.08
0.1
0.12
0.14
0.16
0.18
0.2
0.22
0
20
40
60
80
100
120
140
160
87
87.5
88
88.5
89
89.5
90
90.5
91
91.5
0
20
40
60
80
100
120
140
160
0
0.02
0.04
0.06
0.08
0.1
0.12
0.14
0.16
0.18
0.2
0.22
0
20
40
60
80
100
120
140
160
87
87.5
88
88.5
89
89.5
90
90.5
91
91.5
0
20
40
60
80
100
120
140
160
0
0.02
0.04
0.06
0.08
0.1
0.12
0.14
0.16
0.18
0.2
0.22
0
20
40
60
80
100
120
140
160
87
87.5
88
88.5
89
89.5
90
90.5
91
91.5
0
20
40
60
80
100
120
140
160
0
0.02
0.04
0.06
0.08
0.1
0.12
0.14
0.16
0.18
0.2
0.22
0
20
40
60
80
100
120
140
160
RAdam
length:     100
500
Adam with warmup
X-axis is the epoch #
Figure 7: Performance of RAdam, Adam with warmup on CIFAR10 with different learning rates.
5.1
COMPARING TO VANILLA ADAM
As analyzed before, the adaptive learning rate has undesirably large variance in the early stage
of training and leads to suspicious/bad local optima on NMT. One question we are interested in
is: whether such an issue widely exits in other similar tasks and applications. Thus, we conduct
a set of experiments with two classical tasks of NLP and CV, i.e., language modeling and image
classiﬁcation. RAdam not only results in consistent improvements over the vanilla Adam, but also
demonstrates its robustness to the change of learning rates. It veriﬁes that the variance issue exists
in various machine learning applications, and has a big impact on the model behavior.
Performance Comparison.
The performances on language modeling (i.e., One Billion
Word (Chelba et al., 2013)) and image classiﬁcation (i.e., CIFAR10 (Krizhevsky et al., 2009) and
ImageNet (Deng et al., 2009)) are presented in Figure 4, 5. The results show that RAdam out-
performs Adam in all three datasets. As shown in Figure 4, although the rectiﬁcation term makes
RAdam slower than the vanilla Adam in the ﬁrst few epochs, it allows RAdam to converge faster
after that. In other words, by reducing the variance of the adaptive learning rate in the early stage, it
gets both faster convergence and better performance, which veriﬁes the impact of the variance issue.
We also observe that RAdam obtains consistent improvements over Adam on image classiﬁcation.
It is worth noting that, on both ImageNet and CIFAR10, although RAdam fails to outperform SGD
in terms of test accuracy, it results in a better training performance (e.g., the training accuracy of
SGD, Adam, and RAdam on ImageNet are 69.57, 69.12 and 70.30 respectively).
Robustness to Learning Rate Change. Besides performance improvements, RAdam also improves
the robustness of model training. We use different initial learning rates, conduct experiments with
ResNet-20 on the CIFAR10 datasets, and summarize their performance in Figure 6. For learning
rates within a broad range (i.e., {0.1, 0.03, 0.01, 0.003}), RAdam achieves consistent model perfor-
mances (their test accuracy curves highly overlap with each other), while Adam and SGD are shown
to be more sensitive to the learning rate. The observation can be interpreted that by rectifying the
variance of the adaptive learning rate, RAdam improves the robustness of model training and can
adapt to different learning rates of a broader range.
7

--- Page 8 ---
Published as a conference paper at ICLR 2020
Table 2: BLEU score on Neural Machine Translation.
Method
IWSLT’14 DE-EN
IWSLT’14 EN-DE
WMT’16 EN-DE
Adam with warmup
34.66 ± 0.014
28.56 ± 0.067
27.03
RAdam
34.76 ± 0.003
28.48 ± 0.054
27.27
5.2
COMPARING TO HEURISTIC WARMUP
To examine the effectiveness of RAdam, we ﬁrst conduct comparisons on neural machine transla-
tion, on which the state-of-the-art employs Adam with the linear warmup. Speciﬁcally, we conduct
experiments on three datasets, i.e., IWSLT’14 De-En, IWSLT’14 En-De, and WMT’16 En-De. Due
to the limited size of the IWSLT’14 dataset, we conduct experiments using 5 different random seeds
and report their mean and standard derivation. As discussed before, the vanilla Adam algorithm
leads to suspicious/bad local optima (i.e., converges to a training perplexity around 500), and needs
a learning rate warmup stage to stabilize the training.
We summarize the performance obtained with the heuristic warmup and our proposed rectiﬁcation
term in Table 2 and visualize the training curve of IWSLT De-En in Figure 1. With a consistent
adaptive learning rate variance, our proposed method achieves similar performance to that of previ-
ous state-of-the-art warmup heuristics. It veriﬁes our intuition that the problematic updates of Adam
are indeed caused by the undesirably large variance in the early stage.
Moreover, we applied Adam with warmup on the CIFAR10 dataset. Its best accuracy on the test
set is 91.29, which is similar to RAdam (91.38). However, we found that RAdam requires less hy-
perparameter tuning. Speciﬁcally, we visualize their learning curves in Figure 7. For some warmup
steps, Adam with warmup is relatively more sensitive to the choice of the learning rate. RAdam,
at the same time, is not only more robust, but also can automatically control the warmup behav-
ior (i.e., without requiring the length of warmup). For example, when setting the learning rate as
0.1, Adam with 100 steps of warmup fails to get satisfying performance and only results in an ac-
curacy of 90.13; RAdam successfully gets an accuracy of 91.06, with the original setting of the
moving average calculation (i.e., β1 = 0.9, β2 = 0.999). We conjecture the reason is due to the fact
that RAdam, which is based on a rigorous variance analysis, explicitly avoids the extreme situation
where the variance is divergent, and rectiﬁes the variance to be consistent in other situations.
5.3
SIMULATED VERIFICATION
In Sections 3 and 4, we approximate Var[
q
t/ Pt
i=1 g2
i ] to the ﬁrst order, and assume ψ2(.) =
1−βt
2
(1−β2) Pt
i=1 βt−i
2
g2
i subjects to a scaled inverse chi-square distribution (this assumption covers the
approximation from EMA to SMA). Here, we examine these two approximations using simulations.
First Order Approximation of Var[
q
t/ Pt
i=1 g2
i ]. To compare Equations 5 and 2, we assume
τ = 1 and plot their values and difference for ν = {5, · · · , 500} in Figure 8. The curve of the
analytic form and the ﬁrst-order approximation highly overlap, and their difference is much smaller
than their value. This result veriﬁes that our ﬁrst-order approximation is very accurate.
Scaled Inverse Chi-Square Distribution Assumption. In this paper, we assume gi accords to a
Normal distribution with a zero mean. We also assume ψ2(.) accords to the scaled inverse chi-square
distribution to derive the variance of Var[ψ(.)], based on the similarity between the exponential
moving average and simple moving average. Here, we empirically verify this assumption.
Speciﬁcally, since gi in the optimization problem may not be zero-mean, we assume its expectation
is µ and sample gi from N(µ, 1). Then, based on these samples, we calculate the variance of the
original adaptive learning rate and the proposed rectiﬁed adaptive learning rate, i.e., Var[ 1
b
vt ] and
Var[ rt
b
vt ] respectively. We set β2 to 0.999, the number of sampled trajectories to 5000, the number
of iterations to 6000, and summarize the simulation results in Figure 9. Across all six settings with
different µ, the adaptive learning rate has a larger variance in the ﬁrst stage and the rectiﬁed adaptive
learning rate has relative consistent variance. This veriﬁes the reliability of our assumption.
8

--- Page 9 ---
Published as a conference paper at ICLR 2020
0
200
400
10−5
10−4
10−3
10−2
10−1
100
Diﬀerence
Analytic
First Order Approx.
Figure 8: The value of Equation 2,
Equation 5 and their difference (abso-
lute difference). The x-axis is ρ and
the y-axis is the variance (log scale).
0
2500
5000
10−3
10−2
10−1
0
2500
5000
10−3
10−2
10−1
0
2500
5000
10−3
10−2
10−1
Var[ 1
vt]
Var[ct
vt]
µ = 0
µ = 0.001
µ = 0.01
0
2500
5000
10−3
10−2
10−1
0
2500
5000
10−4
10−3
10−2
0
2500
5000
10−7
10−6
10−5
µ = 0.1
µ = 1
µ = 10
Figure 9: The simulation of Var[ 1
vt ] and Var[ ct
vt ]. The x-axis
is iteration # (from 5), the y-axis is the variance (log scale).
6
CONCLUSION
In this paper, we explore the underlying principle of the effectiveness of the warmup heuristic used
for adaptive optimization algorithms. Speciﬁcally, we identify that, due to the limited amount of
samples in the early stage of model training, the adaptive learning rate has an undesirably large
variance and can cause the model to converge to suspicious/bad local optima. We provide both
empirical and theoretical evidence to support our hypothesis, and further propose a new variant
of Adam, whose adaptive learning rate is rectiﬁed so as to have a consistent variance. Empirical
results demonstrate the effectiveness of our proposed method. In future work, we plan to replace the
rectiﬁcation strategy by sharing the second moment estimation across similar parameters.
ACKNOWLEDGE
We thank Zeyuan Allen-Zhu for valuable discussions and comments, Microsoft Research Technol-
ogy Engineering team for setting up GPU machines. Research was sponsored in part by DARPA
No. W911NF-17-C-0099 and FA8750-19-2-1004, National Science Foundation IIS 16-18481, IIS
17-04532, and IIS-17-41317, and DTRA HDTRA11810026.
REFERENCES
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint
arXiv:1607.06450, 2016.
David Balduzzi, Marcus Frean, Lennox Leary, JP Lewis, Kurt Wan-Duo Ma, and Brian McWilliams.
The shattered gradients problem: If resnets are the answer, then what is the question? In ICML,
2017.
Yoshua Bengio, Nicolas Boulanger-Lewandowski, and Razvan Pascanu. Advances in optimizing
recurrent networks. In 2013 IEEE International Conference on Acoustics, Speech and Signal
Processing, pp. 8624–8628. IEEE, 2013.
Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Anima Anandkumar. signsgd:
Compressed optimisation for non-convex problems. In ICML, 2018.
Augustin Cauchy.
M´ethode g´en´erale pour la r´esolution des systemes d’´equations simultan´ees.
Comp. Rend. Sci. Paris, 25(1847):536–538, 1847.
Mauro Cettolo, Jan Niehues, Sebastian St¨uker, Luisa Bentivogli, and Marcello Federico. Report on
the 11th iwslt evaluation campaign, iwslt 2014. In Proceedings of the International Workshop on
Spoken Language Translation,, 2014.
Ciprian Chelba, Tomas Mikolov, Michael Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and
Tony Robinson. One billion word benchmark for measuring progress in statistical language mod-
eling. In INTERSPEECH, 2013.
9

--- Page 10 ---
Published as a conference paper at ICLR 2020
Jinghui Chen, Dongruo Zhou, Yiqi Tang, Ziyan Yang, and Quanquan Gu.
Closing the gener-
alization gap of adaptive gradient methods in training deep neural networks.
arXiv preprint
arXiv:1806.06763, 2018.
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In ICML, 2009.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. In NAACL-HLT, 2019.
Timothy Dozat. Incorporating nesterov momentum into adam. 2016.
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and
stochastic optimization. In COLT, 2010.
Carl-Friedrich Gauss. Theoria combinationis observationum erroribus minimis obnoxiae. Commen-
tationes Societatis Regiae Scientiarum Gottingensis Recentiores, 1823.
Akhilesh Gotmare, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. A closer look at
deep learning heuristics: Learning rate restarts, warmup and distillation. In ICLR, 2019.
Priya Goyal, Piotr Doll´ar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, An-
drew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet
in 1 hour. arXiv preprint arXiv:1706.02677, 2017.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog-
nition. In CVPR, 2016.
Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky. Neural networks for machine learning
lecture 6a overview of mini-batch gradient descent. Cited on, 2012.
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by
reducing internal covariate shift. In ICML, 2015.
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2014.
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.
Technical report, Citeseer, 2009.
Liyuan Liu, Xiang Ren, Jingbo Shang, Jian Peng, and Jiawei Han. Efﬁcient contextualized repre-
sentation: Language model pruning for sequence labeling. EMNLP, 2018.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining
approach. arXiv preprint arXiv:1907.11692, 2019.
Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in adam. In ICLR, 2018.
Liangchen Luo, Yuanhao Xiong, Yan Liu, and Xu Sun. Adaptive gradient methods with dynamic
bound of learning rate. In ICLR, 2019.
Robert Nau. Forecasting with moving averages. 2014.
Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier,
and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. In NAACL, 2019.
Martin Popel and Ondˇrej Bojar. Training tips for the transformer model. The Prague Bulletin of
Mathematical Linguistics, 110(1):43–70, 2018.
Sashank J Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of adam and beyond. In
ICLR, 2018.
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking
the inception architecture for computer vision. In CVPR, 2016.
10

--- Page 11 ---
Published as a conference paper at ICLR 2020
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, 2017.
Kirk M Wolter. Taylor series methods. In Introduction to variance estimation. 2007.
Lin Xiao, Adams Wei Yu, Qihang Lin, and Weizhu Chen. Dscovr: Randomized primal-dual block
coordinate algorithms for asynchronous distributed optimization. J. Mach. Learn. Res., 2017.
Matthew D Zeiler. Adadelta: an adaptive learning rate method. arXiv preprint arXiv:1212.5701,
2012.
Hongyi Zhang, Yann N Dauphin, and Tengyu Ma. Fixup initialization: Residual learning without
normalization. In ICLR, 2019.
11

--- Page 12 ---
Published as a conference paper at ICLR 2020
A
PROOF OF THEOREM 1
For ease of notation, we refer ψ2(.) as x and
1
σ2 as τ 2. Thus, x ∼Scale-inv-X 2(ρ, τ 2) and:
p(x) = (τ 2ρ/2)ρ/2
Γ(ρ/2)
exp[ −ρτ 2
2x ]
x1+ρ/2
and
E[x] =
ρ
(ρ −2)σ2 (∀ρ > 2)
(6)
where Γ(.) is the gamma function. Therefore, we have:
E[√x] =
Z ∞
0
√x p(x) dx = τ√ρ Γ((ρ −1)/2)
√
2 Γ(ρ/2)
(∀ρ > 4).
(7)
Based on Equation 6 and 7, for ∀ρ > 4, we have:
Var[ψ(.)] = Var[√x] = E[x] −E[√x]2 = τ 2(
ρ
ρ −2 −ρ 22ρ−5
π
B(ρ −1
2
, ρ −1
2
)2),
(8)
where B(.) is the beta function. To prove the monotonic property of Var[ψ(.)], we need to show:
Lemma 1. for t ≥4, ∂
∂t(
t
t−2 −t 22t−5
π
B( t−1
2 , t−1
2 )2) < 0
Proof. The target inequality can be re-wrote as
∂
∂t(
t
t −2 −t 22t−5
π
B(t −1
2
, t −1
2
)2)
=
−2
(t −2)2 −22t−5
π
B(t −1
2
, t −1
2
)2 −t 22t−5 ln 4
π
B(t −1
2
, t −1
2
)2
−2t 22t−5
π
B(t −1
2
, t −1
2
)2(Ψ(t −1
2
) −Ψ(t −1)),

Ψ(x) = Γ′(x)
Γ(x)

< 0
This inequality is equivalent to:
64π
(t −2)24tB( t−1
2 , t−1
2 )2 + 1 + t ln 4 + 2tΨ(t −1
2
)
> 2tΨ(t −1)
(i)
= t[Ψ(t −1
2
) + Ψ( t
2) + ln 4],
where (i) is derived from Legendre duplication formula. Simplify the above inequality, we get:
64π
(t −2)24tB( t−1
2 , t−1
2 )2 + 1 + tΨ(t −1
2
) −tΨ( t
2) > 0,
We only need to show
64π
(t −2)24tB( t−1
2 , t−1
2 )2 + 1 + tΨ(t −1
2
) −tΨ( t
2)
≥
64π
(t −2)24tB( t−1
2 , t−1
2 )2 + 2 + t(ln(t/2) −1/(t/2 −0.5)) −t ln(t/2)
=
64π
(t −2)24tB( t−1
2 , t−1
2 )2 −
2
t −1
>
64π
(t −2)24tB( t−1
2 , t−1
2 )2 −
2
t −2 ≥0,
where the ﬁrst inequality is from ln(x) −1/(2x) > Ψ(x) > ln(x + 0.5) −1/x.
Therefore, we only need to show
32π ≥(t −2)4tB(t −1
2
, t −1
2
)2,
which is equivalent to
(t −2)4tB(t −1
2
, t −1
2
)2 = (t −2)4t Γ( t−1
2 )4
Γ(t −1)2
(i)
= (t −2)4t Γ( t−1
2 )2
Γ(t/2)2 42−tπ = 16π(t −2)Γ( t−1
2 )2
Γ(t/2)2 ≤32π,
12

--- Page 13 ---
Published as a conference paper at ICLR 2020
where (i) is from Legendre duplication formula.
So we only need to show
(t −2)Γ( t−1
2 )2
Γ(t/2)2 ≤2
(9)
Using Gautschi’s inequality ( Γ(x+1)
Γ(x+s) < (x + 1)1−s), we have
(t −2)Γ( t−1
2 )2
Γ(t/2)2 ≤(t −2)(t −1
2
)−1 = 2(t −2)
t −1
< 2
(10)
B
IMPLEMENTATION DETAILS
B.1
LANGUAGE MODELING
Our implementation is based on the previous work (Liu et al., 2018). Speciﬁcally, we use two-layer
LSTMs with 2048 hidden states with adaptive softmax to conduct experiments on the one billion
words dataset. Word embedding (random initialized) of 300 dimensions is used as the input and the
adaptive softmax is incorporated with a default setting (cut-offs are set to [4000, 40000, 200000]).
Additionally, as pre-processing, we replace all tokens occurring equal or less than 3 times with as
UNK. Dropout is applied to each layer with a ratio of 0.1, gradients are clipped at 5.0. We use the
default hyper-parameters to update moving averages, i.e.β1 = 0.9 and β2 = 0.999. The learning
rate is set to start from 0.001, and decayed at the start of 10th epochs. LSTMs are unrolled for 20
steps without resetting the LSTM states and the batch size is set to 128. All models are trained on
one NVIDIA Tesla V100 GPU.
B.2
IMAGEINE CLASSIFICATION
We use the default ResNet architectures (He et al., 2016) in a public pytorch re-implementation4.
Speciﬁcally, we use 20-layer ResNet (9 Basic Blocks) for CIFAR-10 and 18-layer ResNet (8 Basic
Blocks) for ImageNet. Batch size is 128 for CIFAR-10 and 256 for ImageNet. The model is trained
for 186 epoches and the learning rate decays at the 81-th and the 122-th epoches by 0.1 on CIFAR-
10, while the model is trained for 90 epoches and the learning rate decays at the 31-th and the 61-th
epoch by 0.1 on ImageNet. For Adam and RAdam, we set β1 = 0.9, β2 = 0.999. For SGD, we
set the momentum factor as 0.9. The weight decay rate is 10−4. Random cropping and random
horizontal ﬂipping are applied to training data.
B.3
NEURAL MACHINE TRANSLATION
Our experiments are based on the default Transformers (Vaswani et al., 2017) implementation from
the fairseq package (Ott et al., 2019). Speciﬁcally, we use word embedding with 512 dimensions
and 6-layer encoder / decoder with 4 head and 1024 feedforward dimensions on the IWSLT14’
dataset; use word embedding with 512 dimension and 6-layer encoder/decoder with 8 heads and
2048 feedforward dimensions on the WMT14’ dataset. Label smoothed cross entropy is used as
the objective function with an uncertainty = 0.1 (Szegedy et al., 2016). We use linear learning rate
decay starting from 3e−4, and the checkpoints of the last 20 epoches are averaged before evaluation.
As to the wamrup strategy, we use a linear warmup for Adam in the ﬁrst 4000 updates, and set β2
to satisfy ν = 4000 (β2 = 0.9995). In the IWSLT’14 dataset, we conduct training on one NVIDIA
Tesla V100 GPU, set maximum batch size as 4000, apply dropout with a ratio 0.3, using weight
decay of 0.0001 and clip the gradient norm at 25. In the WMT’16 dataset, we conduct training on
four NVIDIA Quadro R8000 GPUs and set maximum batch size as 8196.
C
DOWNGRADING TO SGDM
As a byproduct determined by math derivations, we degenerated RAdam to SGD with momentum
in the ﬁrst several updates. Although this stage only contains several gradient updates, these up-
4https://github.com/bearpaw/pytorch-classification
13

--- Page 14 ---
Published as a conference paper at ICLR 2020
dates could be quite damaging (e.g., in our Figure 2, the gradient distribution is distorted within 10
gradient updates). Intuitively, updates with divergent adaptive learning rate variance could be more
damaging than the ones with converged variance, as divergent variance implies more instability. As
a case study, we performed experiments on the CIFAR10 dataset. Five-run average results are sum-
marized in Table 3. The optimizer fails to get an equally reliably model when changing the ﬁrst
4 updates to Adam, yet the inﬂuence of switching is less deleterious when we change 5-8 updates
instead. This result veriﬁes our intuition and is in agreement with our theory — the ﬁrst few updates
could be more damaging than later updates. By saying that, we still want to emphasize that this part
(downgrading to SGDM) is only a minor part of our algorithm design whereas our main focus is on
the mechanism of warmup and the derivation of the rectiﬁcation term.
Table 3: Performance on CIFAR10 (lr = 0.1).
1-4 steps
5-8 steps
8+ steps
test
acc
train
loss
train
error
RAdam
RAdam
RAdam
91.08
0.021
0.74
Adam (w. divergent var.)
RAdam
RAdam
89.98
0.060
2.12
SGD
Adam (w. convergent var.)
RAdam
90.29
0.038
1.23
14
