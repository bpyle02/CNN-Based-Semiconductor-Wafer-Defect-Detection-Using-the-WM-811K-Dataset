# Dropout as a Bayesian Approximation: Representing Model Uncertainty

**Authors**: Gal, Ghahramani
**Year**: 2016
**arXiv**: 1506.02142
**Topic**: uncertainty
**Relevance**: MC Dropout implementation

---


--- Page 1 ---
Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning
Yarin Gal
YG279@CAM.AC.UK
Zoubin Ghahramani
ZG201@CAM.AC.UK
University of Cambridge
Abstract
Deep learning tools have gained tremendous at-
tention in applied machine learning.
However
such tools for regression and classiﬁcation do
not capture model uncertainty.
In compari-
son, Bayesian models offer a mathematically
grounded framework to reason about model un-
certainty, but usually come with a prohibitive
computational cost. In this paper we develop a
new theoretical framework casting dropout train-
ing in deep neural networks (NNs) as approxi-
mate Bayesian inference in deep Gaussian pro-
cesses.
A direct result of this theory gives us
tools to model uncertainty with dropout NNs –
extracting information from existing models that
has been thrown away so far.
This mitigates
the problem of representing uncertainty in deep
learning without sacriﬁcing either computational
complexity or test accuracy. We perform an ex-
tensive study of the properties of dropout’s un-
certainty. Various network architectures and non-
linearities are assessed on tasks of regression
and classiﬁcation, using MNIST as an example.
We show a considerable improvement in predic-
tive log-likelihood and RMSE compared to ex-
isting state-of-the-art methods, and ﬁnish by us-
ing dropout’s uncertainty in deep reinforcement
learning.
1. Introduction
Deep learning has attracted tremendous attention from re-
searchers in ﬁelds such as physics, biology, and manufac-
turing, to name a few (Baldi et al., 2014; Anjos et al., 2015;
Bergmann et al., 2014).
Tools such as neural networks
(NNs), dropout, convolutional neural networks (convnets),
and others are used extensively. However, these are ﬁelds in
which representing model uncertainty is of crucial impor-
tance (Krzywinski & Altman, 2013; Ghahramani, 2015).
Proceedings of the 33 rd International Conference on Machine
Learning, New York, NY, USA, 2016. JMLR: W&CP volume
48. Copyright 2016 by the author(s).
With the recent shift in many of these ﬁelds towards the use
of Bayesian uncertainty (Herzog & Ostwald, 2013; Traﬁ-
mow & Marks, 2015; Nuzzo, 2014), new needs arise from
deep learning tools.
Standard deep learning tools for regression and classiﬁca-
tion do not capture model uncertainty. In classiﬁcation,
predictive probabilities obtained at the end of the pipeline
(the softmax output) are often erroneously interpreted as
model conﬁdence. A model can be uncertain in its predic-
tions even with a high softmax output (ﬁg. 1). Passing a
point estimate of a function (solid line 1a) through a soft-
max (solid line 1b) results in extrapolations with unjustiﬁed
high conﬁdence for points far from the training data. x∗for
example would be classiﬁed as class 1 with probability 1.
However, passing the distribution (shaded area 1a) through
a softmax (shaded area 1b) better reﬂects classiﬁcation un-
certainty far from the training data.
Model uncertainty is indispensable for the deep learning
practitioner as well. With model conﬁdence at hand we can
treat uncertain inputs and special cases explicitly. For ex-
ample, in the case of classiﬁcation, a model might return a
result with high uncertainty. In this case we might decide
to pass the input to a human for classiﬁcation. This can
happen in a post ofﬁce, sorting letters according to their zip
code, or in a nuclear power plant with a system responsi-
ble for critical infrastructure (Linda et al., 2009). Uncer-
tainty is important in reinforcement learning (RL) as well
(Szepesv´ari, 2010). With uncertainty information an agent
can decide when to exploit and when to explore its envi-
ronment. Recent advances in RL have made use of NNs for
Q-value function approximation. These are functions that
estimate the quality of different actions an agent can take.
Epsilon greedy search is often used where the agent selects
its best action with some probability and explores other-
wise. With uncertainty estimates over the agent’s Q-value
function, techniques such as Thompson sampling (Thomp-
son, 1933) can be used to learn much faster.
Bayesian probability theory offers us mathematically
grounded tools to reason about model uncertainty, but these
usually come with a prohibitive computational cost. It is
perhaps surprising then that it is possible to cast recent
arXiv:1506.02142v6  [stat.ML]  4 Oct 2016

--- Page 2 ---
Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
(a) Arbitrary function f(x) as a function of data x (softmax input)
(b) σ(f(x)) as a function of data x (softmax output)
Figure 1. A sketch of softmax input and output for an idealised binary classiﬁcation problem. Training data is given between the
dashed grey lines. Function point estimate is shown with a solid line. Function uncertainty is shown with a shaded area. Marked with a
dashed red line is a point x∗far from the training data. Ignoring function uncertainty, point x∗is classiﬁed as class 1 with probability 1.
deep learning tools as Bayesian models – without chang-
ing either the models or the optimisation. We show that
the use of dropout (and its variants) in NNs can be inter-
preted as a Bayesian approximation of a well known prob-
abilistic model: the Gaussian process (GP) (Rasmussen &
Williams, 2006). Dropout is used in many models in deep
learning as a way to avoid over-ﬁtting (Srivastava et al.,
2014), and our interpretation suggests that dropout approx-
imately integrates over the models’ weights. We develop
tools for representing model uncertainty of existing dropout
NNs – extracting information that has been thrown away so
far. This mitigates the problem of representing model un-
certainty in deep learning without sacriﬁcing either compu-
tational complexity or test accuracy.
In this paper we give a complete theoretical treatment of
the link between Gaussian processes and dropout, and de-
velop the tools necessary to represent uncertainty in deep
learning. We perform an extensive exploratory assessment
of the properties of the uncertainty obtained from dropout
NNs and convnets on the tasks of regression and classiﬁ-
cation. We compare the uncertainty obtained from differ-
ent model architectures and non-linearities in regression,
and show that model uncertainty is indispensable for clas-
siﬁcation tasks, using MNIST as a concrete example. We
then show a considerable improvement in predictive log-
likelihood and RMSE compared to existing state-of-the-
art methods. Lastly we give a quantitative assessment of
model uncertainty in the setting of reinforcement learning,
on a practical task similar to that used in deep reinforce-
ment learning (Mnih et al., 2015).1
2. Related Research
It has long been known that inﬁnitely wide (single hid-
den layer) NNs with distributions placed over their weights
converge to Gaussian processes (Neal, 1995; Williams,
1997). This known relation is through a limit argument that
does not allow us to translate properties from the Gaus-
sian process to ﬁnite NNs easily. Finite NNs with distri-
1Code and demos are available at http://yarin.co.
butions placed over the weights have been studied exten-
sively as Bayesian neural networks (Neal, 1995; MacKay,
1992). These offer robustness to over-ﬁtting as well, but
with challenging inference and additional computational
costs. Variational inference has been applied to these mod-
els, but with limited success (Hinton & Van Camp, 1993;
Barber & Bishop, 1998; Graves, 2011). Recent advances
in variational inference introduced new techniques into
the ﬁeld such as sampling-based variational inference and
stochastic variational inference (Blei et al., 2012; Kingma
& Welling, 2013; Rezende et al., 2014; Titsias & L´azaro-
Gredilla, 2014; Hoffman et al., 2013). These have been
used to obtain new approximations for Bayesian neural
networks that perform as well as dropout (Blundell et al.,
2015).
However these models come with a prohibitive
computational cost. To represent uncertainty, the number
of parameters in these models is doubled for the same net-
work size. Further, they require more time to converge and
do not improve on existing techniques. Given that good un-
certainty estimates can be cheaply obtained from common
dropout models, this might result in unnecessary additional
computation. An alternative approach to variational infer-
ence makes use of expectation propagation (Hern´andez-
Lobato & Adams, 2015) and has improved considerably
in RMSE and uncertainty estimation on VI approaches
such as (Graves, 2011).
In the results section we com-
pare dropout to these approaches and show a signiﬁcant
improvement in both RMSE and uncertainty estimation.
3. Dropout as a Bayesian Approximation
We show that a neural network with arbitrary depth and
non-linearities, with dropout applied before every weight
layer, is mathematically equivalent to an approximation
to the probabilistic deep Gaussian process (Damianou &
Lawrence, 2013) (marginalised over its covariance function
parameters). We would like to stress that no simplifying as-
sumptions are made on the use of dropout in the literature,
and that the results derived are applicable to any network
architecture that makes use of dropout exactly as it appears
in practical applications.
Furthermore, our results carry
to other variants of dropout as well (such as drop-connect

--- Page 3 ---
Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
(Wan et al., 2013), multiplicative Gaussian noise (Srivas-
tava et al., 2014), etc.). We show that the dropout objec-
tive, in effect, minimises the Kullback–Leibler divergence
between an approximate distribution and the posterior of
a deep Gaussian process (marginalised over its ﬁnite rank
covariance function parameters). Due to space constraints
we refer the reader to the appendix for an in depth review
of dropout, Gaussian processes, and variational inference
(section 2), as well as the main derivation for dropout and
its variations (section 3). The results are summarised here
and in the next section we obtain uncertainty estimates for
dropout NNs.
Let by be the output of a NN model with L layers and a loss
function E(·, ·) such as the softmax loss or the Euclidean
loss (square loss). We denote by Wi the NN’s weight ma-
trices of dimensions Ki × Ki−1, and by bi the bias vec-
tors of dimensions Ki for each layer i = 1, ..., L. We de-
note by yi the observed output corresponding to input xi
for 1 ≤i ≤N data points, and the input and output sets
as X, Y. During NN optimisation a regularisation term is
often added. We often use L2 regularisation weighted by
some weight decay λ, resulting in a minimisation objective
(often referred to as cost),
Ldropout := 1
N
N
X
i=1
E(yi, byi) + λ
L
X
i=1
 ||Wi||2
2 + ||bi||2
2

.
(1)
With dropout, we sample binary variables for every input
point and for every network unit in each layer (apart from
the last one). Each binary variable takes value 1 with prob-
ability pi for layer i. A unit is dropped (i.e. its value is set
to zero) for a given input if its corresponding binary vari-
able takes value 0. We use the same values in the backward
pass propagating the derivatives to the parameters.
In comparison to the non-probabilistic NN, the deep Gaus-
sian process is a powerful tool in statistics that allows us to
model distributions over functions. Assume we are given a
covariance function of the form
K(x, y) =
Z
p(w)p(b)σ(wT x + b)σ(wT y + b)dwdb
with some element-wise non-linearity σ(·) and distribu-
tions p(w), p(b). In sections 3 and 4 in the appendix we
show that a deep Gaussian process with L layers and co-
variance function K(x, y) can be approximated by placing
a variational distribution over each component of a spec-
tral decomposition of the GPs’ covariance functions. This
spectral decomposition maps each layer of the deep GP to
a layer of explicitly represented hidden units, as will be
brieﬂy explained next.
Let Wi be a (now random) matrix of dimensions Ki ×
Ki−1 for each layer i, and write ω = {Wi}L
i=1. A priori,
we let each row of Wi distribute according to the p(w)
above. In addition, assume vectors mi of dimensions Ki
for each GP layer. The predictive probability of the deep
GP model (integrated w.r.t. the ﬁnite rank covariance func-
tion parameters ω) given some precision parameter τ > 0
can be parametrised as
p(y|x, X, Y) =
Z
p(y|x, ω)p(ω|X, Y)dω
(2)
p(y|x, ω) = N
 y; by(x, ω), τ −1ID

by
 x, ω = {W1, ...,WL}

=
r
1
KL
WLσ

...
r
1
K1
W2σ
 W1x + m1

...

The posterior distribution p(ω|X, Y) in eq. (2) is in-
tractable. We use q(ω), a distribution over matrices whose
columns are randomly set to zero, to approximate the in-
tractable posterior. We deﬁne q(ω) as:
Wi = Mi · diag([zi,j]Ki
j=1)
zi,j ∼Bernoulli(pi) for i = 1, ..., L, j = 1, ..., Ki−1
given some probabilities pi and matrices Mi as variational
parameters. The binary variable zi,j = 0 corresponds then
to unit j in layer i −1 being dropped out as an input to
layer i. The variational distribution q(ω) is highly multi-
modal, inducing strong joint correlations over the rows of
the matrices Wi (which correspond to the frequencies in
the sparse spectrum GP approximation).
We minimise the KL divergence between the approximate
posterior q(ω) above and the posterior of the full deep GP,
p(ω|X, Y). This KL is our minimisation objective
−
Z
q(ω) log p(Y|X, ω)dω + KL(q(ω)||p(ω)).
(3)
We rewrite the ﬁrst term as a sum
−
N
X
n=1
Z
q(ω) log p(yn|xn, ω)dω
and approximate each term in the sum by Monte Carlo in-
tegration with a single sample bωn ∼q(ω) to get an unbi-
ased estimate −log p(yn|xn, bωn). We further approximate
the second term in eq. (3) and obtain PL
i=1
  pil2
2 ||Mi||2
2 +
l2
2 ||mi||2
2

with prior length-scale l (see section 4.2 in the
appendix). Given model precision τ we scale the result by
the constant 1/τN to obtain the objective:
LGP-MC ∝1
N
N
X
n=1
−log p(yn|xn, bωn)
τ
(4)
+
L
X
i=1
 pil2
2τN ||Mi||2
2 +
l2
2τN ||mi||2
2

.

--- Page 4 ---
Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
Setting
E(yn, by(xn, bωn)) = −log p(yn|xn, bωn)/τ
we recover eq. (1) for an appropriate setting of the pre-
cision hyper-parameter τ and length-scale l. The sampled
bωn result in realisations from the Bernoulli distribution zn
i,j
equivalent to the binary variables in the dropout case2.
4. Obtaining Model Uncertainty
We next derive results extending on the above showing that
model uncertainty can be obtained from dropout NN mod-
els.
Following section 2.3 in the appendix, our approximate
predictive distribution is given by
q(y∗|x∗) =
Z
p(y∗|x∗, ω)q(ω)dω
(5)
where ω = {Wi}L
i=1 is our set of random variables for a
model with L layers.
We will perform moment-matching and estimate the ﬁrst
two moments of the predictive distribution empirically.
More speciﬁcally, we sample T sets of vectors of realisa-
tions from the Bernoulli distribution {zt
1, ..., zt
L}T
t=1 with
zt
i = [zt
i,j]Ki
j=1, giving {Wt
1, ..., Wt
L}T
t=1. We estimate
Eq(y∗|x∗)(y∗) ≈1
T
T
X
t=1
by∗(x∗, Wt
1, ..., Wt
L)
(6)
following proposition C in the appendix. We refer to this
Monte Carlo estimate as MC dropout.
In practice this
is equivalent to performing T stochastic forward passes
through the network and averaging the results.
This result has been presented in the literature before as
model averaging. We have given a new derivation for this
result which allows us to derive mathematically grounded
uncertainty estimates as well. Srivastava et al. (2014, sec-
tion 7.5) have reasoned empirically that MC dropout can
be approximated by averaging the weights of the network
(multiplying each Wi by pi at test time, referred to as stan-
dard dropout).
We estimate the second raw moment in the same way:
Eq(y∗|x∗)
 (y∗)T (y∗)

≈τ −1ID
+ 1
T
T
X
t=1
by∗(x∗, Wt
1, ..., Wt
L)T by∗(x∗, Wt
1, ..., Wt
L)
following proposition D in the appendix. To obtain the
model’s predictive variance we have:
Varq(y∗|x∗)
 y∗
≈τ −1ID
2In the appendix (section 4.1) we extend this derivation to
classiﬁcation. E(·) is deﬁned as softmax loss and τ is set to 1.
+ 1
T
T
X
t=1
by∗(x∗, Wt
1, ..., Wt
L)T by∗(x∗, Wt
1, ..., Wt
L)
−Eq(y∗|x∗)(y∗)T Eq(y∗|x∗)(y∗)
which equals the sample variance of T stochastic forward
passes through the NN plus the inverse model precision.
Note that y∗is a row vector thus the sum is over the outer-
products. Given the weight-decay λ (and our prior length-
scale l) we can ﬁnd the model precision from the identity
τ = pl2
2Nλ.
(7)
We can estimate our predictive log-likelihood by Monte
Carlo integration of eq. (2). This is an estimate of how
well the model ﬁts the mean and uncertainty (see section
4.4 in the appendix). For regression this is given by:
log p(y∗|x∗, X, Y) ≈logsumexp

−1
2τ||y −byt||2

−log T −1
2 log 2π −1
2 log τ −1
(8)
with a log-sum-exp of T terms and byt stochastic forward
passes through the network.
Our predictive distribution q(y∗|x∗) is expected to be
highly multi-modal, and the above approximations only
give a glimpse into its properties. This is because the ap-
proximating variational distribution placed on each weight
matrix column is bi-modal, and as a result the joint dis-
tribution over each layer’s weights is multi-modal (section
3.2 in the appendix).
Note that the dropout NN model itself is not changed.
To estimate the predictive mean and predictive uncertainty
we simply collect the results of stochastic forward passes
through the model. As a result, this information can be
used with existing NN models trained with dropout. Fur-
thermore, the forward passes can be done concurrently, re-
sulting in constant running time identical to that of standard
dropout.
5. Experiments
We next perform an extensive assessment of the properties
of the uncertainty estimates obtained from dropout NNs
and convnets on the tasks of regression and classiﬁcation.
We compare the uncertainty obtained from different model
architectures and non-linearities, both on tasks of extrap-
olation, and show that model uncertainty is important for
classiﬁcation tasks using MNIST (LeCun & Cortes, 1998)
as an example. We then show that using dropout’s uncer-
tainty we can obtain a considerable improvement in predic-
tive log-likelihood and RMSE compared to existing state-
of-the-art methods. We ﬁnish with an example use of the

--- Page 5 ---
Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
(a) Standard dropout with weight averaging
(b) Gaussian process with SE covariance function
(c) MC dropout with ReLU non-linearities
(d) MC dropout with TanH non-linearities
Figure 2. Predictive mean and uncertainties on the Mauna Loa CO2 concentrations dataset, for various models. In red is the
observed function (left of the dashed blue line); in blue is the predictive mean plus/minus two standard deviations (8 for ﬁg. 2d).
Different shades of blue represent half a standard deviation. Marked with a dashed red line is a point far away from the data: standard
dropout conﬁdently predicts an insensible value for the point; the other models predict insensible values as well but with the additional
information that the models are uncertain about their predictions.
model’s uncertainty in a Bayesian pipeline.
We give a
quantitative assessment of the model’s performance in the
setting of reinforcement learning on a task similar to that
used in deep reinforcement learning (Mnih et al., 2015).
Using the results from the previous section, we begin by
qualitatively evaluating the dropout NN uncertainty on two
regression tasks. We use two regression datasets and model
scalar functions which are easy to visualise. These are tasks
one would often come across in real-world data analysis.
We use a subset of the atmospheric CO2 concentrations
dataset derived from in situ air samples collected at Mauna
Loa Observatory, Hawaii (Keeling et al., 2004) (referred to
as CO2) to evaluate model extrapolation. In the appendix
(section D.1) we give further results on a second dataset,
the reconstructed solar irradiance dataset (Lean, 2004), to
assess model interpolation. The datasets are fairly small,
with each dataset consisting of about 200 data points. We
centred and normalised both datasets.
5.1. Model Uncertainty in Regression Tasks
We trained several models on the CO2 dataset. We use NNs
with either 4 or 5 hidden layers and 1024 hidden units. We
use either ReLU non-linearities or TanH non-linearities in
each network, and use dropout probabilities of either 0.1 or
0.2. Exact experiment set-up is given in section E.1 in the
appendix.
Extrapolation results are shown in ﬁgure 2. The model is
trained on the training data (left of the dashed blue line),
and tested on the entire dataset.
Fig. 2a shows the re-
sults for standard dropout (i.e. with weight averaging and
without assessing model uncertainty) for the 5 layer ReLU
model. Fig. 2b shows the results obtained from a Gaussian
process with a squared exponential covariance function for
comparison. Fig. 2c shows the results of the same network
as in ﬁg. 2a, but with MC dropout used to evaluate the pre-
dictive mean and uncertainty for the training and test sets.
Lastly, ﬁg. 2d shows the same using the TanH network with
5 layers (plotted with 8 times the standard deviation for vi-
sualisation purposes). The shades of blue represent model
uncertainty: each colour gradient represents half a standard
deviation (in total, predictive mean plus/minus 2 standard
deviations are shown, representing 95% conﬁdence). Not
plotted are the models with 4 layers as these converge to
the same results.
Extrapolating the observed data, none of the models can
capture the periodicity (although with a suitable covariance
function the GP will capture it well). The standard dropout
NN model (ﬁg. 2a) predicts value 0 for point x∗(marked
with a dashed red line) with high conﬁdence, even though
it is clearly not a sensible prediction. The GP model repre-
sents this by increasing its predictive uncertainty – in effect
declaring that the predictive value might be 0 but the model
is uncertain. This behaviour is captured in MC dropout as
well. Even though the models in ﬁgures 2 have an incorrect
predictive mean, the increased standard deviation expresses
the models’ uncertainty about the point.
Note that the uncertainty is increasing far from the data
for the ReLU model, whereas for the TanH model it stays
bounded.
Figure 3. Predictive mean and uncertainties on the Mauna Loa
CO2 concentrations dataset for the MC dropout model with ReLU
non-linearities, approximated with 10 samples.

--- Page 6 ---
Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
(a) Softmax input scatter
(b) Softmax output scatter
Figure 4. A scatter of 100 forward passes of the softmax input and output for dropout LeNet. On the X axis is a rotated image of
the digit 1. The input is classiﬁed as digit 5 for images 6-7, even though model uncertainty is extremly large (best viewed in colour).
This is not surprising, as dropout’s uncertainty draws its
properties from the GP in which different covariance func-
tions correspond to different uncertainty estimates. ReLU
and TanH approximate different GP covariance functions
(section 3.1 in the appendix) and TanH saturates whereas
ReLU does not. For the TanH model we assessed the uncer-
tainty using both dropout probability 0.1 and dropout prob-
ability 0.2. Models initialised with dropout probability 0.1
initially exhibit smaller uncertainty than the ones initialised
with dropout probability 0.2, but towards the end of the op-
timisation when the model has converged the uncertainty is
almost indistinguishable. It seems that the moments of the
dropout models converge to the moments of the approxi-
mated GP model – its mean and uncertainty. It is worth
mentioning that we attempted to ﬁt the data with models
with a smaller number of layers unsuccessfully.
The number of forward iterations used to estimate the un-
certainty (T) was 1000 for drawing purposes.
A much
smaller numbers can be used to get a reasonable estima-
tion to the predictive mean and uncertainty (see ﬁg. 3 for
example with T = 10).
5.2. Model Uncertainty in Classiﬁcation Tasks
To assess model classiﬁcation conﬁdence in a realistic ex-
ample we test a convolutional neural network trained on
the full MNIST dataset (LeCun & Cortes, 1998).
We
trained the LeNet convolutional neural network model (Le-
Cun et al., 1998) with dropout applied before the last fully
connected inner-product layer (the usual way dropout is
used in convnets). We used dropout probability of 0.5. We
trained the model for 106 iterations with the same learning
rate policy as before with γ = 0.0001 and p = 0.75. We
used Caffe (Jia et al., 2014) reference implementation for
this experiment.
We evaluated the trained model on a continuously rotated
image of the digit 1 (shown on the X axis of ﬁg. 4). We
scatter 100 stochastic forward passes of the softmax input
(the output from the last fully connected layer, ﬁg. 4a), as
well as of the softmax output for each of the top classes
(ﬁg. 4b). For the 12 images, the model predicts classes [1
1 1 1 1 5 5 7 7 7 7 7].
The plots show the softmax input value and softmax output
value for the 3 digits with the largest values for each corre-
sponding input. When the softmax input for a class is larger
than that of all other classes (class 1 for the ﬁrst 5 images,
class 5 for the next 2 images, and class 7 for the rest in
ﬁg 4a), the model predicts the corresponding class. Look-
ing at the softmax input values, if the uncertainty envelope
of a class is far from that of other classes’ (for example
the left most image) then the input is classiﬁed with high
conﬁdence. On the other hand, if the uncertainty envelope
intersects that of other classes (such as in the case of the
middle input image), then even though the softmax output
can be arbitrarily high (as far as 1 if the mean is far from
the means of the other classes), the softmax output uncer-
tainty can be as large as the entire space. This signiﬁes the
model’s uncertainty in its softmax output value – i.e. in the
prediction. In this scenario it would not be reasonable to
use probit to return class 5 for the middle image when its
uncertainty is so high. One would expect the model to ask
an external annotator for a label for this input. Model un-
certainty in such cases can be quantiﬁed by looking at the
entropy or variation ratios of the model prediction.
5.3. Predictive Performance
Predictive log-likelihood captures how well a model ﬁts the
data, with larger values indicating better model ﬁt. Un-
certainty quality can be determined from this quantity as
well (see section 4.4 in the appendix). We replicate the
experiment set-up in Hern´andez-Lobato & Adams (2015)
and compare the RMSE and predictive log-likelihood of
dropout (referred to as “Dropout” in the experiments)
to that of Probabilistic Back-propagation (referred to as

--- Page 7 ---
Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
Avg. Test RMSE and Std. Errors
Avg. Test LL and Std. Errors
Dataset
N
Q
VI
PBP
Dropout
VI
PBP
Dropout
Boston Housing
506
13
4.32 ±0.29 3.01 ±0.18
2.97 ±0.19
-2.90 ±0.07 -2.57 ±0.09
-2.46 ±0.06
Concrete Strength
1,030
8
7.19 ±0.12 5.67 ±0.09
5.23 ±0.12
-3.39 ±0.02 -3.16 ±0.02
-3.04 ±0.02
Energy Efﬁciency
768
8
2.65 ±0.08 1.80 ±0.05
1.66 ±0.04
-2.39 ±0.03 -2.04 ±0.02
-1.99 ±0.02
Kin8nm
8,192
8
0.10 ±0.00 0.10 ±0.00
0.10 ±0.00
0.90 ±0.01
0.90 ±0.01
0.95 ±0.01
Naval Propulsion
11,934
16
0.01 ±0.00 0.01 ±0.00
0.01 ±0.00
3.73 ±0.12
3.73 ±0.01
3.80 ±0.01
Power Plant
9,568
4
4.33 ±0.04 4.12 ±0.03
4.02 ±0.04
-2.89 ±0.01 -2.84 ±0.01
-2.80 ±0.01
Protein Structure
45,730
9
4.84 ±0.03 4.73 ±0.01
4.36 ±0.01
-2.99 ±0.01 -2.97 ±0.00
-2.89 ±0.00
Wine Quality Red
1,599
11
0.65 ±0.01 0.64 ±0.01
0.62 ±0.01
-0.98 ±0.01 -0.97 ±0.01
-0.93 ±0.01
Yacht Hydrodynamics
308
6
6.89 ±0.67 1.02 ±0.05
1.11 ±0.09
-3.43 ±0.16 -1.63 ±0.02
-1.55 ±0.03
Year Prediction MSD
515,345
90
9.034 ±NA 8.879 ±NA 8.849 ±NA
-3.622 ±NA -3.603 ±NA -3.588 ±NA
Table 1. Average test performance in RMSE and predictive log likelihood for a popular variational inference method (VI, Graves
(2011)), Probabilistic back-propagation (PBP, Hern´andez-Lobato & Adams (2015)), and dropout uncertainty (Dropout). Dataset size
(N) and input dimensionality (Q) are also given.
“PBP”, (Hern´andez-Lobato & Adams, 2015)) and to a pop-
ular variational inference technique in Bayesian NNs (re-
ferred to as “VI”, (Graves, 2011)). The aim of this exper-
iment is to compare the uncertainty quality obtained from
a naive application of dropout in NNs to that of specialised
methods developed to capture uncertainty.
Following our Bayesian interpretation of dropout (eq. (4))
we need to deﬁne a prior length-scale, and ﬁnd an opti-
mal model precision parameter τ which will allow us to
evaluate the predictive log-likelihood (eq. (8)). Similarly
to (Hern´andez-Lobato & Adams, 2015) we use Bayesian
optimisation (BO, (Snoek et al., 2012; Snoek & authors,
2015)) over validation log-likelihood to ﬁnd optimal τ, and
set the prior length-scale to 10−2 for most datasets based on
the range of the data. Note that this is a standard dropout
NN, where the prior length-scale l and model precision τ
are simply used to deﬁne the model’s weight decay through
eq. (7). We used dropout with probabilities 0.05 and 0.005
since the network size is very small (with 50 units follow-
ing (Hern´andez-Lobato & Adams, 2015)) and the datasets
are fairly small as well. The BO runs used 40 iterations
following the original setup, but after ﬁnding the optimal
parameter values we used 10x more iterations, as dropout
takes longer to converge. Even though the model doesn’t
converge within 40 iterations, it gives BO a good indication
of whether a parameter is good or not. Finally, we used
mini-batches of size 32 and the Adam optimiser (Kingma
& Ba, 2014). Further details about the various datasets are
given in (Hern´andez-Lobato & Adams, 2015).
The results are shown in table3 1. Dropout signiﬁcantly
outperforms all other models both in terms of RMSE as
well as test log-likelihood on all datasets apart from Yacht,
for which PBP obtains better RMSE. All experiments were
averaged on 20 random splits of the data (apart from Pro-
3Update [October 2016]: Note that in an earlier version of this
paper our reported dropout standard error was erroneously scaled-
up by a factor of 4.5 (i.e. for Boston RMSE we reported standard
error 0.85 instead of 0.19 for example).
tein for which only 5 splits were used and Year for which
one split was used). The median for most datasets gives
much better performance than the mean.
For example,
on the Boston Housing dataset dropout achieves median
RMSE of 2.68 with an IQR interval of [2.45, 3.35] and
predictive log-likelihood median of -2.34 with IQR [-2.54,
-2.29]. In the Concrete Strength dataset dropout achieves
median RMSE of 5.15.
To implement the model we used Keras (Chollet, 2015),
an open source deep learning package based on Theano
(Bergstra et al., 2010). In (Hern´andez-Lobato & Adams,
2015) BO for VI seems to require a considerable amount
of additional time compared to PBP. However our model’s
running time (including BO) is comparable to PBP’s
Theano implementation4. On Naval Propulsion for exam-
ple our model takes 276 seconds on average per split (start-
to-ﬁnish, divided by the number of splits). With the opti-
mal parameters BO found, model training took 95 seconds.
This is in comparison to PBP’s 220 seconds. For Kin8nm
our model requires 188 seconds on average including BO,
65 seconds without, compared to PBP’s 156 seconds.
Dropout’s RMSE in table 1 is given by averaging stochas-
tic forward passes through the network following eq. (6)
(MC dropout). We observed an improvement using this es-
timate compared to the standard dropout weight averaging,
and also compared to much smaller dropout probabilities
(near zero). For the Boston Housing dataset for example,
repeating the same experiment with dropout probability 0
results in RMSE of 3.07 and predictive log-likelihood of
4 Update [October 2016]: In the results above we attempted
to match PBP’s run time (hence used only 10x more epochs
compared to PBP’s 40 epochs). Experimenting with 100x more
epochs compared to PBP (10x more epochs compared to the re-
sults in table 1) gives a considerable improvement both in terms of
test RMSE as well as test log-likelihood over the results in table
1. We further assessed a model with two hidden layers instead of
one (using the same number of units for the second layer). Both
experiments are shown in table 2 at the end of this document.

--- Page 8 ---
Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
-2.59. This demonstrates that dropout signiﬁcantly affects
the predictive log-likelihood and RMSE, even though the
dropout probability is fairly small.
We used dropout following the same way the method would
be used in current research – without adapting model struc-
ture.
This is to demonstrate the results that could be
obtained from existing models when evaluated with MC
dropout. Experimenting with different network architec-
tures we expect the method to give even better uncertainty
estimates.
5.4. Model Uncertainty in Reinforcement Learning
In reinforcement learning an agent receives various rewards
from different states, and its aim is to maximise its expected
reward over time. The agent tries to learn to avoid transi-
tioning into states with low rewards, and to pick actions that
lead to better states instead. Uncertainty is of great impor-
tance in this task – with uncertainty information an agent
can decide when to exploit rewards it knows of, and when
to explore its environment.
Recent advances in RL have made use of NNs to estimate
agents’ Q-value functions (referred to as Q-networks), a
function that estimates the quality of different actions an
agent can take at different states. This has led to impres-
sive results on Atari game simulations, where agents su-
perseded human performance on a variety of games (Mnih
et al., 2015). Epsilon greedy search was used in this set-
ting, where the agent selects the best action following its
current Q-function estimation with some probability, and
explores otherwise. With our uncertainty estimates given
by a dropout Q-network we can use techniques such as
Thompson sampling (Thompson, 1933) to converge faster
than epsilon greedy while avoiding over-ﬁtting.
We use code by (Karpathy & authors, 2014–2015) that
replicated the results by (Mnih et al., 2015) with a sim-
pler 2D setting. We simulate an agent in a 2D world with
9 eyes pointing in different angles ahead (depicted in ﬁg.
5). Each eye can sense a single pixel intensity of 3 colours.
The agent navigates by using one of 5 actions controlling
two motors at its base. An action turns the motors at dif-
ferent angles and different speeds. The environment con-
sists of red circles which give the agent a positive reward
for reaching, and green circles which result in a negative
reward. The agent is further rewarded for not looking at
(white) walls, and for walking in a straight line.
We trained the original model, and an additional model
with dropout with probability 0.1 applied before the every
weight layer. Note that both agents use the same network
structure in this experiment for comparison purposes. In
a real world scenario using dropout we would use a larger
model (as the original model was intentially selected to be
small to avoid over-ﬁtting). To make use of the dropout Q-
network’s uncertainty estimates, we use Thompson sam-
pling instead of epsilon greedy. In effect this means that
we perform a single stochastic forward pass through the
network every time we need to take an action. In replay,
we perform a single stochastic forward pass and then back-
propagate with the sampled Bernoulli random variables.
Exact experiment set-up is given in section E.2 in the ap-
pendix.
In ﬁg. 6 we show a log plot of the average reward obtained
by both the original implementation (in green) and our ap-
proach (in blue), as a function of the number of batches.
Not plotted is the burn-in intervals of 25 batches (random
Figure 5. Depiction of the reinforcement learning problem used in
the experiments. The agent is in the lower left part of the maze,
facing north-west.
Figure 6. Log plot of average reward obtained by both epsilon
greedy (in green) and our approach (in blue), as a function of the
number of batches.

--- Page 9 ---
Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
moves).
Thompson sampling gets reward larger than 1
within 25 batches from burn-in. Epsilon greedy takes 175
batches to achieve the same performance. It is interesting
to note that our approach seems to stop improving after
1K batches. This is because we are still sampling random
moves, whereas epsilon greedy only exploits at this stage.
6. Conclusions and Future Research
We have built a probabilistic interpretation of dropout
which allowed us to obtain model uncertainty out of exist-
ing deep learning models. We have studied the properties
of this uncertainty in detail, and demonstrated possible ap-
plications, interleaving Bayesian models and deep learning
models together. This extends on initial research studying
dropout from the Bayesian perspective (Wang & Manning,
2013; Maeda, 2014).
Bernoulli dropout is only one example of a regularisation
technique corresponding to an approximate variational dis-
tribution which results in uncertainty estimates. Other vari-
ants of dropout follow our interpretation as well and cor-
respond to alternative approximating distributions. These
would result in different uncertainty estimates, trading-off
uncertainty quality with computational complexity. We ex-
plore these in follow-up work.
Furthermore, each GP covariance function has a one-to-
one correspondence with the combination of both NN non-
linearities and weight regularisation. This suggests tech-
niques to select appropriate NN structure and regularisa-
tion based on our a priori assumptions about the data. For
example, if one expects the function to be smooth and
the uncertainty to increase far from the data, cosine non-
linearities and L2 regularisation might be appropriate. The
study of non-linearity–regularisation combinations and the
corresponding predictive mean and variance are subject of
current research.
ACKNOWLEDGEMENTS
The authors would like to thank Dr Yutian Chen, Mr
Christof Angermueller, Mr Roger Frigola, Mr Rowan
McAllister, Dr Gabriel Synnaeve, Mr Mark van der Wilk,
Mr Yan Wu, and many other reviewers for their helpful
comments. Yarin Gal is supported by the Google European
Fellowship in Machine Learning.
References
Anjos, O, Iglesias, C, Peres, F, Mart´ınez, J, Garc´ıa, ´A,
and Taboada, J. Neural networks applied to discrimi-
nate botanical origin of honeys. Food chemistry, 175:
128–136, 2015.
Baldi, P, Sadowski, P, and Whiteson, D. Searching for ex-
otic particles in high-energy physics with deep learning.
Nature communications, 5, 2014.
Barber, D and Bishop, C M. Ensemble learning in Bayesian
neural networks.
NATO ASI SERIES F COMPUTER
AND SYSTEMS SCIENCES, 168:215–238, 1998.
Bergmann, S, Stelzer, S, and Strassburger, S. On the use of
artiﬁcial neural networks in simulation-based manufac-
turing control. Journal of Simulation, 8(1):76–90, 2014.
Bergstra, James, Breuleux, Olivier, Bastien, Fr´ed´eric,
Lamblin, Pascal, Pascanu, Razvan, Desjardins, Guil-
laume, Turian, Joseph, Warde-Farley, David, and Ben-
gio, Yoshua. Theano: a CPU and GPU math expression
compiler.
In Proceedings of the Python for Scientiﬁc
Computing Conference (SciPy), June 2010. Oral Pre-
sentation.
Blei, D M, Jordan, M I, and Paisley, J W.
Variational
Bayesian inference with stochastic search.
In ICML,
2012.
Blundell, C, Cornebise, J, Kavukcuoglu, K, and Wierstra,
D. Weight uncertainty in neural networks. ICML, 2015.
Chen, W, Wilson, J T, Tyree, S, Weinberger, K Q, and
Chen, Y. Compressing neural networks with the hash-
ing trick. In ICML-15, 2015.
Chollet, Franc¸ois.
Keras.
https://github.com/
fchollet/keras, 2015.
Damianou, A and Lawrence, N. Deep Gaussian processes.
In AISTATS, 2013.
Ghahramani, Z. Probabilistic machine learning and artiﬁ-
cial intelligence. Nature, 521(7553), 2015.
Graves, A. Practical variational inference for neural net-
works. In NIPS, 2011.
Hern´andez-Lobato, J M and Adams, R P.
Probabilistic
backpropagation for scalable learning of bayesian neu-
ral networks. In ICML-15, 2015.
Herzog, S and Ostwald, D. Experimental biology: Some-
times Bayesian statistics are better. Nature, 494, 2013.
Hinton, G E and Van Camp, D. Keeping the neural net-
works simple by minimizing the description length of the
weights. In Proceedings of the sixth annual conference
on Computational learning theory, 1993.
Hoffman, M D, Blei, D M, Wang, C, and Paisley, J.
Stochastic variational inference. The Journal of Machine
Learning Research, 14(1):1303–1347, 2013.

--- Page 10 ---
Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
Jia, Y, Shelhamer, E, Donahue, J, Karayev, S, Long, J, Gir-
shick, R, Guadarrama, S, and Darrell, T. Caffe: Convo-
lutional architecture for fast feature embedding. arXiv
preprint arXiv:1408.5093, 2014.
Karpathy, A and authors.
A Javascript implementa-
tion of neural networks.
https://github.com/
karpathy/convnetjs, 2014–2015.
Keeling, C D, Whorf, T P, and the Carbon Dioxide Re-
search Group. Atmospheric CO2 concentrations (ppmv)
derived from in situ air samples collected at Mauna Loa
Observatory, Hawaii, 2004.
Kingma, D P and Welling, M. Auto-encoding variational
Bayes. arXiv preprint arXiv:1312.6114, 2013.
Kingma,
Diederik
and
Ba,
Jimmy.
Adam:
A
method for stochastic optimization.
arXiv preprint
arXiv:1412.6980, 2014.
Krzywinski, M and Altman, N.
Points of signiﬁcance:
Importance of being uncertain. Nature methods, 10(9),
2013.
Lean, J. Solar irradiance reconstruction. NOAA/NGDC
Paleoclimatology Program, USA, 2004.
LeCun, Y and Cortes, C. The mnist database of handwrit-
ten digits, 1998.
LeCun, Y, Bottou, L, Bengio, Y, and Haffner, P. Gradient-
based learning applied to document recognition. Pro-
ceedings of the IEEE, 86(11):2278–2324, 1998.
Linda, O, Vollmer, T, and Manic, M. Neural network based
intrusion detection system for critical infrastructures. In
Neural Networks, 2009. IJCNN 2009. International Joint
Conference on. IEEE, 2009.
MacKay, D J C. A practical Bayesian framework for back-
propagation networks. Neural computation, 4(3), 1992.
Maeda, S. A Bayesian encourages dropout. arXiv preprint
arXiv:1412.7003, 2014.
Mnih, V, Kavukcuoglu, K, Silver, D, Rusu, A A, Veness, J,
et al. Human-level control through deep reinforcement
learning. Nature, 518(7540):529–533, 2015.
Neal, R M. Bayesian learning for neural networks. PhD
thesis, University of Toronto, 1995.
Nuzzo, Regina. Statistical errors. Nature, 506(13):150–
152, 2014.
Rasmussen, C E and Williams, C K I. Gaussian Processes
for Machine Learning (Adaptive Computation and Ma-
chine Learning). The MIT Press, 2006.
Rezende, D J, Mohamed, S, and Wierstra, D. Stochastic
backpropagation and approximate inference in deep gen-
erative models. In ICML, 2014.
Snoek, Jasper and authors.
Spearmint.
https://
github.com/JasperSnoek/spearmint, 2015.
Snoek, Jasper, Larochelle, Hugo, and Adams, Ryan P.
Practical Bayesian optimization of machine learning al-
gorithms. In Advances in neural information processing
systems, pp. 2951–2959, 2012.
Srivastava, N, Hinton, G, Krizhevsky, A, Sutskever, I, and
Salakhutdinov, R. Dropout: A simple way to prevent
neural networks from overﬁtting. The Journal of Ma-
chine Learning Research, 15(1), 2014.
Szepesv´ari, C. Algorithms for reinforcement learning. Syn-
thesis Lectures on Artiﬁcial Intelligence and Machine
Learning, 4(1), 2010.
Thompson, W R. On the likelihood that one unknown prob-
ability exceeds another in view of the evidence of two
samples. Biometrika, 1933.
Titsias, M and L´azaro-Gredilla, M. Doubly stochastic vari-
ational Bayes for non-conjugate inference.
In ICML,
2014.
Traﬁmow, D and Marks, M. Editorial. Basic and Applied
Social Psychology, 37(1), 2015.
Wan, L, Zeiler, M, Zhang, S, LeCun, Y, and Fergus, R.
Regularization of neural networks using dropconnect. In
ICML-13, 2013.
Wang, S and Manning, C. Fast dropout training. ICML,
2013.
Williams, C K I. Computing with inﬁnite networks. NIPS,
1997.

--- Page 11 ---
Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
A. Appendix
The appendix for the paper is given at http://arxiv.
org/abs/1506.02157.

--- Page 12 ---
Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
Avg. Test RMSE and Std. Errors
Avg. Test LL and Std. Errors
Dataset
Dropout
10x Epochs
2 Layers
Dropout
10x Epochs
2 Layers
Boston Housing
2.97 ±0.19
2.80 ±0.19
2.80 ±0.13
-2.46 ±0.06 -2.39 ±0.05
-2.34 ±0.02
Concrete Strength
5.23 ±0.12
4.81 ±0.14
4.50 ±0.18
-3.04 ±0.02 -2.94 ±0.02
-2.82 ±0.02
Energy Efﬁciency
1.66 ±0.04
1.09 ±0.05
0.47 ±0.01
-1.99 ±0.02 -1.72 ±0.02
-1.48 ±0.00
Kin8nm
0.10 ±0.00
0.09 ±0.00
0.08 ±0.00
0.95 ±0.01
0.97 ±0.01
1.10 ±0.00
Naval Propulsion
0.01 ±0.00
0.00 ±0.00
0.00 ±0.00
3.80 ±0.01
3.92 ±0.01
4.32 ±0.00
Power Plant
4.02 ±0.04
4.00 ±0.04
3.63 ±0.04
-2.80 ±0.01 -2.79 ±0.01
-2.67 ±0.01
Protein Structure
4.36 ±0.01
4.27 ±0.01
3.62 ±0.01
-2.89 ±0.00 -2.87 ±0.00
-2.70 ±0.00
Wine Quality Red
0.62 ±0.01
0.61 ±0.01
0.60 ±0.01
-0.93 ±0.01 -0.92 ±0.01
-0.90 ±0.01
Yacht Hydrodynamics
1.11 ±0.09
0.72 ±0.06
0.66 ±0.06
-1.55 ±0.03 -1.38 ±0.01
-1.37 ±0.02
Table 2. Average test performance in RMSE and predictive log likelihood for dropout uncertainty as above (Dropout), the same
model optimised with 10 times the number of epochs and identical model precision (10x epochs), and the same model again with 2
layers instead of 1 (2 Layers).
