# A Survey of Deep Meta-Learning

**Authors**: Huisman, van Rijn, Plaat
**Year**: 2021
**arXiv**: 2010.03522
**Topic**: fewshot
**Relevance**: Meta-learning survey for few-shot defect classification

---


--- Page 1 ---
A Survey of Deep Meta-Learning
Mike Huisman · Jan N. van Rijn · Aske Plaat
Abstract Deep neural networks can achieve great successes when presented
with large data sets and suﬃcient computational resources. However, their
ability to learn new concepts quickly is limited. Meta-learning is one approach
to address this issue, by enabling the network to learn how to learn. The
ﬁeld of Deep Meta-Learning advances at great speed, but lacks a uniﬁed, in-
depth overview of current techniques. With this work, we aim to bridge this
gap. After providing the reader with a theoretical foundation, we investigate
and summarize key methods, which are categorized into i) metric-, ii) model-,
and iii) optimization-based techniques. In addition, we identify the main open
challenges, such as performance evaluations on heterogeneous benchmarks,
and reduction of the computational costs of meta-learning.
Keywords Meta-learning · Learning to learn · Few-shot learning · Transfer
learning · Deep learning
1 Introduction
In recent years, deep learning techniques have achieved remarkable successes
on various tasks, including game-playing (Mnih et al., 2013; Silver et al., 2016),
image recognition (Krizhevsky et al., 2012; He et al., 2015), machine trans-
lation (Wu et al., 2016), and automatic classiﬁcation in biomedical domains
(Goceri, 2019a; Goceri and Karakas, 2020; Iqbal et al., 2020, 2019b,a). Despite
these advances and recent solutions (Goceri, 2019b, 2020), ample challenges
remain to be solved, such as the large amounts of data and training that are
needed to achieve good performance. These requirements severely constrain
the ability of deep neural networks to learn new concepts quickly, one of the
M. Huisman · J.N. van Rijn · A. Plaat
Leiden Institute of Advanced Computer Science
Niels Bohrweg 1, 2333CA Leiden, The Netherlands
E-mail: m.huisman@liacs.leidenuniv.nl
arXiv:2010.03522v2  [cs.LG]  21 Apr 2021

--- Page 2 ---
2
Mike Huisman et al.
32-32-32-32
64-64-64-64
64-96-128-256
96-192-384-512
64-64-64-64-64
ResNet-12
WRN-28-10
45
50
55
60
65
Accuracy (%)
Reptile
MAML
LSTM
meta-learner
Matching nets
LLAMA
Meta-SGD
iMAML
Prototypical nets
GNN
Relation nets
LR-D2
R2-D2
Meta nets
BMAML
SNAIL
MetaOptNet
LEO
Metric-based
Model-based
Optimization-based
Fig. 1 The accuracy scores of the covered techniques on 1-shot miniImageNet classiﬁcation.
The used feature extraction backbone is displayed on the x-axis. As one can see, there is a
strong relationship between the network complexity and the classiﬁcation performance.
deﬁning aspects of human intelligence (Jankowski et al., 2011; Lake et al.,
2017).
Meta-learning has been suggested as one strategy to overcome this chal-
lenge (Naik and Mammone, 1992; Schmidhuber, 1987; Thrun, 1998). The key
idea is that meta-learning agents improve their learning ability over time, or
equivalently, learn to learn. The learning process is primarily concerned with
tasks (set of observations) and takes place at two diﬀerent levels: an inner- and
an outer-level. At the inner-level, a new task is presented, and the agent tries
to quickly learn the associated concepts from the training observations. This
quick adaptation is facilitated by knowledge that it has accumulated across
earlier tasks at the outer-level. Thus, whereas the inner-level concerns a single
task, the outer-level concerns a multitude of tasks.
Historically, the term meta-learning has been used with various scopes. In
its broadest sense, it encapsulates all systems that leverage prior learning ex-
perience in order to learn new tasks more quickly (Vanschoren, 2018). This
broad notion includes more traditional algorithm selection and hyperparame-
ter optimization techniques for Machine Learning (Brazdil et al., 2008). In this
work, however, we focus on a subset of the meta-learning ﬁeld which devel-
ops meta-learning procedures to learn a good inductive bias for (deep) neural

--- Page 3 ---
A Survey of Deep Meta-Learning
3
networks.1 Henceforth, we use the term Deep Meta-Learning to refer to this
subﬁeld of meta-learning.
The ﬁeld of Deep Meta-Learning is advancing at a quick pace, while it
lacks a coherent, unifying overview, providing detailed insights into the key
techniques. Vanschoren (2018) has surveyed meta-learning techniques, where
meta-learning was used in the broad sense, limiting its account of Deep Meta-
Learning techniques. Also, many exciting developments in deep meta-learning
have happened after the survey was published. A more recent survey by
Hospedales et al. (2020) adopts the same notion of deep meta-learning as
we do, but aims to give a broad overview, omitting technical details of the
various techniques.
We attempt to ﬁll this gap by providing detailed explications of contempo-
rary Deep Meta-Learning techniques, using a uniﬁed notation. More speciﬁ-
cally, we cover modern techniques in the ﬁeld for supervised and reinforcement
learning, that have achieved state-of-the-art performance, obtained popular-
ity in the ﬁeld, and presented novel ideas. Extra attention is paid to MAML
(Finn et al., 2017), and related techniques, because of their impact on the ﬁeld.
We show how the techniques relate to each other, detail their strengths and
weaknesses, identify current challenges, and provide an overview of promis-
ing future research directions. One of the observations that we make is that
the network complexity is highly related to the few-shot classiﬁcation perfor-
mance (see Figure 1). One might expect that in a few-shot setting, where only
a few examples are available to learn from, the number of network parameters
should be kept small to prevent overﬁtting. Clearly, the ﬁgure shows that this
does not hold, as techniques that use larger backbones tend to achieve better
performance. One important factor might be that due to the large number
of tasks that have been seen by the network, we are in a setting where simi-
larly large amounts of observations have been evaluated. This result suggests
that the size of the network should be taken into account when comparing
algorithms.
This work can serve as an educational introduction to the ﬁeld of Deep
Meta-Learning, and as reference material for experienced researchers in the
ﬁeld. Throughout, we will adopt the taxonomy used by Vinyals (2017), which
identiﬁes three categories of Deep Meta-Learning approaches: i) metric-based,
ii) model-based, and iii) optimization-based meta-learning techniques.
The remainder of this work is structured as follows. Section 2 builds a com-
mon foundation on which we will base our overview of Deep Meta-Learning
techniques. Sections 3, 4, and 5 cover the main metric-, model-, and optimization-
based meta-learning techniques, respectively. Section 6 provides a helicopter
view of the ﬁeld and summarizes the key challenges and open questions. Ta-
ble 1 gives an overview of notation that we will use throughout this paper.
1 Here, inductive bias refers to the assumptions of a model which guide predictions on
unseen data (Mitchell, 1980).

--- Page 4 ---
4
Mike Huisman et al.
Expression
Meaning
Meta-learning
Learning to learn
Tj = (Dtr
Tj , Dtest
Tj )
A task consisting of a labeled support and query set
Support set
The train set Dtr
Tj associated with a task Tj
Query set
The test set Dtest
Tj
associated with a task Tj
xi
Example input vector i in the support set
yi
(One-hot encoded) label of example input xi from the support set
k
Number of examples per class in the support set
N
Number of classes in the support and query sets of a task
x
Input in the query set
y
A (one-hot encoded) label for input x
(f/g/h)◦
Neural network function with parameters ◦
Inner-level
At the level of a single task
Outer-level
At the meta-level: across tasks
Fast weights
A term used in the literature to denote task-speciﬁc parameters
Base-learner
Learner that works at the inner-level
Meta-learner
Learner that operates at the outer-level
θ
The parameters of the base-learner network
LD
Loss function with respect to task/dataset D
Input embedding
Penultimate layer representation of the input
Task embedding
An internal representation of a task in a network/system
SL
Supervised Learning
RL
Reinforcement Learning
Table 1 Some notation and meaning, which we use throughout this paper.
2 Foundation
In this section, we build the necessary foundation for investigating Deep Meta-
Learning techniques in a consistent manner. To begin with, we contrast regular
learning and meta-learning. Afterwards, we brieﬂy discuss how Deep Meta-
Learning relates to diﬀerent ﬁelds, what the usual training and evaluation
procedure looks like, and which benchmarks are often used for this purpose.
We ﬁnish this section by describing the context and some applications of the
meta-learning ﬁeld.
2.1 The Meta Abstraction
In this subsection, we contrast base-level (regular) learning and meta-learning
for two diﬀerent paradigms, i.e., supervised and reinforcement learning.
2.1.1 Regular Supervised Learning
In supervised learning, we wish to learn a function fθ : X →Y that learns to
map inputs xi ∈X to their corresponding outputs yi ∈Y . Here, θ are model
parameters (e.g. weights in a neural network) that determine the function’s
behavior. To learn these parameters, we are given a data set of m observations:
D = {(xi, yi)}m
i=1. Thus, given a data set D, learning boils down to ﬁnding

--- Page 5 ---
A Survey of Deep Meta-Learning
5
the correct setting for θ that minimizes an empirical loss function LD, which
must capture how the model is performing, such that appropriate adjustments
to its parameters can be made. In short, we wish to ﬁnd
θSL := arg min
θ
LD(θ),
(1)
where SL stands for “supervised learning". Note that this objective is speciﬁc
to data set D, meaning that our model fθ may not generalize to examples
outside of D. To measure generalization, one could evaluate the performance
on a separate test data set, which contains unseen examples. A popular way to
do this is through cross-validation, where one repeatedly creates train and test
splits Dtr, Dtest ⊂D and uses these to train and evaluate a model respectively
(Hastie et al., 2009).
Finding globally optimal parameters θSL is often computationally infeasi-
ble. We can, however, approximate them, guided by pre-deﬁned meta-knowledge
ω (Hospedales et al., 2020), which includes, e.g., the initial model parameters
θ, choice of optimizer, and learning rate schedule. As such, we approximate
θSL ≈gω(D, LD),
(2)
where gω is an optimization procedure that uses pre-deﬁned meta-knowledge
ω, data set D, and loss function LD, to produce updated weights gω(D, LD)
that (presumably) perform well on D.
2.1.2 Supervised Meta-Learning
In contrast, supervised meta-learning does not assume that any meta-knowledge
ω is given, or pre-deﬁned. Instead, the goal of meta-learning is to ﬁnd the best
ω, such that our (regular) base-learner can learn new tasks (data sets) as
quickly as possible. Thus, whereas supervised regular learning involves one
data set, supervised meta-learning involves a group of data sets. The goal is
to learn meta-knowledge ω such that our model can learn many diﬀerent tasks
well. Thus, our model is learning to learn.
More formally, we have a probability distribution of tasks p(T ) and wish
to ﬁnd optimal meta-knowledge
ω∗:= arg min
ω
ETj∽p(T )
|
{z
}
Outer-level
[LTj(gω(Tj, LTj))
|
{z
}
Inner-level
].
(3)
Here, the inner-level concerns task-speciﬁc learning, while the outer-level con-
cerns multiple tasks. One can now easily see why this is meta-learning: we
learn ω, which allows for quick learning of tasks Tj at the inner-level. Hence,
we are learning to learn.

--- Page 6 ---
6
Mike Huisman et al.
2.1.3 Regular Reinforcement Learning
In reinforcement learning, we have an agent that learns from experience. That
is, it interacts with an environment, modeled by a Markov Decision Process
(MDP) M = (S, A, P, r, p0, γ, T). Here, S is the set of states, A the set of
actions, P the transition probability distribution deﬁning P(st+1|st, at), r :
S × A →R the reward function, p0 the probability distribution over initial
states, γ ∈[0, 1] the discount factor, and T the time horizon (maximum number
of time steps) (Sutton and Barto, 2018; Duan et al., 2016).
At every time step t, the agent ﬁnds itself in state st, in which the agent per-
forms an action at, computed by a policy function πθ (i.e., at = πθ(st)), which
is parameterized by weights θ. In turn, it receives a reward rt = r(st, πθ(st)) ∈
R and a new state st+1. This process of interactions continues until a termi-
nation criterion is met (e.g. ﬁxed time horizon T reached). The goal of the
agent is to learn how to act in order to maximize its expected reward. The
reinforcement learning (RL) goal is to ﬁnd
θRL := arg min
θ
Etraj
T
X
t=0
γtr(st, πθ(st)),
(4)
where we take the expectation over the possible trajectories traj = (s0, πθ(s0),
. . . sT , πθ(sT )) due to the random nature of MDPs (Duan et al., 2016). Note
that γ is a hyperparameter that can prioritize short- or long-term rewards by
decreasing or increasing it, respectively.
Also in the case of reinforcement learning it is often infeasible to ﬁnd the
global optimum θRL, and thus we settle for approximations. In short, given a
learning method ω, we approximate
θRL ≈gω(Tj, LTj),
(5)
where again Tj is the given MDP, and gω is the optimization algorithm, guided
by pre-deﬁned meta-knowledge ω.
Note that in a Markov Decision Process (MDP), the agent knows the state
at any given time step t. When this is not the case, it becomes a Partially
Observable Markov Decision Process (POMDP), where the agent receives only
observations O, and uses these to update its belief with regard to the state it
is in (Sutton and Barto, 2018).
2.1.4 Meta Reinforcement Learning
The meta abstraction has as its object a group of tasks, or Markov Decision
Processes (MDPs) in the case of reinforcement learning. Thus, instead of max-
imizing the expected reward on a single MDP, the meta reinforcement learning
objective is to maximize the expected reward over various MDPs, by learning

--- Page 7 ---
A Survey of Deep Meta-Learning
7
Fig. 2 The diﬀerence between multi-task learning and meta-learning2.
meta-knowledge ω. Here, the MDPs are sampled from some distribution p(T ).
So, we wish to ﬁnd a set of parameters
ω∗:= arg min
ω
ETj∽p(T )
|
{z
}
Outer-level


Etraj
T
X
t=0
γtr(st, πgω(Tj,LTj )(st))
|
{z
}
Inner-level


.
(6)
2.1.5 Contrast with other Fields
Now that we have provided a formal basis for our discussion for both supervised
and reinforcement meta-learning, it is time to contrast meta-learning brieﬂy
with two related areas of machine learning that also have the goal to improve
the speed of learning. We will start with transfer learning.
Transfer Learning In Transfer Learning, one tries to transfer knowledge
of previous tasks to new, unseen tasks (Pan and Yang, 2009; Taylor and Stone,
2009), which can be challenging when the new task comes from a diﬀerent
distribution than the one used for training Iqbal et al. (2018). The distinction
between Transfer Learning and Meta-Learning has become more opaque over
time. A key property of meta-learning techniques, however, is their meta-
objective, which explicitly aims to optimize performance across a distribution
over tasks (as seen in previous sections by taking the expected loss over a
distribution of tasks). This objective need not always be present in Transfer
Learning techniques, e.g., when one pre-trains a model on a large data set,
and ﬁne-tunes the learned weights on a smaller data set.
Multi-task learning Another, closely related ﬁeld, is that of multi-task
learning. In multi-task learning, a model is jointly trained to perform well on
multiple ﬁxed tasks (Hospedales et al., 2020). Meta-learning, in contrast, aims
to ﬁnd a model that can learn new (previously unseen) tasks quickly. This
diﬀerence is illustrated in Figure 2.
2 Adapted from https://meta-world.github.io/

--- Page 8 ---
8
Mike Huisman et al.
Fig. 3 Illustration of N-way, k-shot classiﬁcation, where N = 5, and k = 1. Meta-validation
tasks are not displayed. Adapted from Ravi and Larochelle (2017).
2.2 The Meta-Setup
In the previous section, we have described the learning objectives for (meta)
supervised and reinforcement learning. We will now describe the general set-
ting that can be used to achieve these objectives. In general, one optimizes a
meta-objective by using various tasks, which are data sets in the context of
supervised learning, and (Partially Observable) Markov Decision Processes in
the case of reinforcement learning. This is done in three stages: the i) meta-
train stage, ii) meta-validation stage, and iii) meta-test stage, each of which
is associated with a set of tasks.
First, in the meta-train stage, the meta-learning algorithm is applied to
the meta-train tasks. Second, the meta-validation tasks can then be used to
evaluate the performance on unseen tasks, which were not used for training. Ef-
fectively, this measures the meta-generalization ability of the trained network,
which serves as feedback to tune, e.g., hyper-parameters of the meta-learning
algorithm. Third, the meta-test tasks are used to give a ﬁnal performance
estimate of the meta-learning technique.
2.2.1 N-way, k-shot Learning
A frequently used instantiation of this general meta-setup is called N-way,
k-shot classiﬁcation (see Figure 3). This setup is also divided into the three
stages—meta-train, meta-validation, and meta-test—which are used for meta-
learning, meta-learner hyperparameter optimization, and evaluation, respec-
tively. Each stage has a corresponding set of disjoint labels, i.e., Ltr, Lval, Ltest ⊂
Y , such that Ltr ∩Lval = ∅, Ltr ∩Ltest = ∅, and Lval ∩Ltest = ∅. In a given
stage s, tasks/episodes Tj = (Dtr
Tj, Dtest
Tj ) are obtained by sampling examples
(xi, yi) from the full data set D, such that every yi ∈Ls. Note that this re-

--- Page 9 ---
A Survey of Deep Meta-Learning
9
quires access to a data set D. The sampling process is guided by the N-way,
k-shot principle, which states that every training data set Dtr
Tj should contain
exactly N classes and k examples per class, implying that |Dtr
Tj| = N · k. Fur-
thermore, the true labels of examples in the test set Dtest
Tj
must be present
in the train set Dtr
Tj of a given task Tj. Dtr
T j acts as a support set, literally
supporting classiﬁcation decisions on the query set Dtest
Tj . Importantly, note
that with this terminology, the query set (or test set) of a task is actually used
during the meta-training phase. Furthermore, the fact that the labels across
stages are disjoint ensures that we test the ability of a model to learn new
concepts.
The meta-learning objective in the training phase is to minimize the loss
function of the model predictions on the query sets, conditioned on the support
sets. As such, for a given task Tj, the model ‘sees’ the support set, and extracts
information from the support set to guide its predictions on the query set. By
applying this procedure to diﬀerent episodes/tasks Tj, the model will slowly
accumulate meta-knowledge ω, which can ultimately speed up learning on new
tasks.
The easiest way to achieve this is by doing this with regular neural net-
works, but as was pointed out by various authors (see, e.g., Finn et al. (2017))
more sophisticated architectures will vastly outperform such networks. In the
remainder of this work, we will review such architectures.
At the meta-validation and meta-test stages, or evaluation phases, the
learned meta-information in ω is ﬁxed. The model is, however, still allowed
to make task-speciﬁc updates to its parameters θ (which implies that it is
learning). After task-speciﬁc updates, we can evaluate the performance on the
test sets. In this way, we test how well a technique performs at meta-learning.
N-way, k-shot classiﬁcation is often performed for small values of k (since
we want our models to learn new concepts quickly, i.e., from few examples).
In that case, one can refer to it as few-shot learning.
2.2.2 Common Benchmarks
Here, we brieﬂy describe some benchmarks that can be used to evaluate meta-
learning algorithms.
– Omniglot (Lake et al., 2011): This data set presents an image recogni-
tion task. Each image corresponds to one out of 1 623 characters from 50
diﬀerent alphabets. Every character was drawn by 20 people. Note that in
this case, the characters are the classes/labels.
– ImageNet (Deng et al., 2009): This is the largest image classiﬁcation
data set, containing more than 20K classes and over 14 million colored
images. miniImageNet is a mini variant of the large ImageNet data set
(Deng et al., 2009) for image classiﬁcation, proposed by Vinyals et al.
(2016) to reduce the engineering eﬀorts to run experiments. The mini data
set contains 60 000 colored images of size 84 × 84. There are a total of 100
classes present, each accorded by 600 examples. tieredImageNet (Ren et al.,

--- Page 10 ---
10
Mike Huisman et al.
2018) is another variation of the large ImageNet data set. It is similar to
miniImageNet, but contains a hierarchical structure. That is, there are 34
classes, each with its own sub-classes.
– CIFAR-10 and CIFAR-100 (Krizhevsky, 2009): Two other image
recognition data sets. Each one contains 60K RGB images of size 32 × 32.
CIFAR-10 and CIFAR-100 contain 10 and 100 classes respectively, with a
uniform number of examples per class (6 000 and 600 respectively). Every
class in CIFAR-100 also has a super-class, of which there are 20 in the full
data set. Many variants of the CIFAR data sets can be sampled, giving rise
to e.g. CIFAR-FS (Bertinetto et al., 2019) and FC-100 (Oreshkin et al.,
2018).
– CUB-200-2011 (Wah et al., 2011): The CUB-200-2011 data set con-
tains roughly 12K RGB images of birds from 200 species. Every image has
some labeled attributes (e.g. crown color, tail shape).
– MNIST (LeCun et al., 2010): MNIST presents a hand-written digit
recognition task, containing ten classes (for digits 0 through 9). In total,
the data set is split into a 60K train and 10K test gray scale images of
hand-written digits.
– Meta-Dataset (Triantaﬁllou et al., 2020): This data set comprises
several other data sets such as Omniglot (Lake et al., 2011), CUB-200
(Wah et al., 2011), ImageNet (Deng et al., 2009), and more (Triantaﬁllou
et al., 2020). An episode is then constructed by sampling a data set (e.g.
Omniglot) and selecting a subset of labels to create train and test splits as
before. In this way, broader generalization is enforced since the tasks are
more distant from each other.
– Meta-world (Yu et al., 2019): A meta reinforcement learning data set,
containing 50 robotic manipulation tasks (control a robot arm to achieve
some pre-deﬁned goal, e.g. unlocking a door, or playing soccer). It was
speciﬁcally designed to cover a broad range of tasks, such that meaningful
generalization can be measured (Yu et al., 2019).
2.2.3 Some Applications of Meta-Learning
Deep neural networks have achieved remarkable results on various tasks in-
cluding image recognition, text processing, game playing, and robotics (Silver
et al., 2016; Mnih et al., 2013; Wu et al., 2016), but their success depends on
the amount of available data (Sun et al., 2017) and computing resources. Deep
meta-learning reduces this dependency by allowing deep neural networks to
learn new concepts quickly. As a result, meta-learning widens the applicability
of deep learning techniques to many application domains. Such areas include
few-shot image classiﬁcation (Finn et al., 2017; Snell et al., 2017; Ravi and
Larochelle, 2017), robotic control policy learning (Gupta et al., 2018; Naga-
bandi et al., 2019) (see Figure 4), hyperparameter optimization (Antoniou
et al., 2019; Schmidhuber et al., 1997), meta-learning learning rules (Bengio
et al., 1991, 1997; Miconi et al., 2018, 2019), abstract reasoning (Barrett et al.,

--- Page 11 ---
A Survey of Deep Meta-Learning
11
Fig. 4 Learning continuous robotic control tasks is an important application of Deep Meta-
Learning techniques. Image taken from (Yu et al., 2019).
2018), and many more. For a larger overview of applications, we refer inter-
ested readers to Hospedales et al. (2020).
2.3 The Meta-Learning Field
As mentioned in the introduction, meta-learning is a broad area of research,
as it encapsulates all techniques that leverage prior learning experience to
learn new tasks more quickly (Vanschoren, 2018). We can classify two distinct
communities in the ﬁeld with a diﬀerent focus: i) algorithm selection and
hyperparameter optimization for machine learning techniques, and ii) search
for inductive bias in deep neural networks. We will refer to these communities
as group i) and group ii) respectively. Now, we will give a brief description of
the ﬁrst ﬁeld, and a historical overview of the second.
Group i) uses a more traditional approach, to select a suitable machine
learning algorithm and hyperparameters for a new data set D (Peng et al.,
2002). This selection can for example be made by leveraging prior model eval-
uations on various data sets D′, and by using the model which achieved the
best performance on the most similar data set (Vanschoren, 2018). Such tra-
ditional approaches require (large) databases of prior model evaluations, for
many diﬀerent algorithms. This has led to initiatives such as OpenML (Van-
schoren et al., 2014), where researchers can share such information. The usage
of these systems would limit the freedom in picking the neural network ar-
chitecture as they would be constrained to using architectures that have been
evaluated beforehand.
In contrast, group ii) adopts the view of a self-improving (neural) agent,
which improves its learning ability over time by ﬁnding a good inductive bias

--- Page 12 ---
12
Mike Huisman et al.
Metric
Model
Optimization
Key idea
Input similarity
Internal task
representation
Optimize for fast
adaptation
Strength
Simple and eﬀective
Flexible
More robust general-
izability
pθ(Y |x, Dtr
Tj )
P
(xi,yi)∈Dtr
Tj
kθ(x, xi)yi
fθ(x, Dtr
Tj )
fgϕ(θ,Dtr
Tj
,LDtr
Tj
)(x)
Table 2 High-level overview of the three Deep Meta-Learning categories, i.e., i) metric-,
ii) model-, and iii) optimization-based techniques, and their main strengths and weaknesses.
Recall that Tj is a task, Dtr
Tj the corresponding support set, kθ(x, xi) a kernel function
returning the similarity between the two inputs x and xi, yi are true labels for known
inputs xi, θ are base-learner parameters, and gϕ is a (learned) optimizer with parameters
ϕ.
(a set of assumptions that guide predictions). We now present a brief histor-
ical overview of developments in this ﬁeld of Deep Meta-Learning, based on
Hospedales et al. (2020).
Pioneering work was done by Schmidhuber (1987) and Hinton and Plaut
(1987). Schmidhuber developed a theory of self-referential learning, where the
weights of a neural network can serve as input to the model itself, which then
predicts updates (Schmidhuber, 1987, 1993). In that same year, Hinton and
Plaut (1987) proposed to use two weights per neural network connection, i.e.,
slow and fast weights, which serve as long- and short-term memory respec-
tively. Later came the idea of meta-learning learning rules (Bengio et al., 1991,
1997). Meta-learning techniques that use gradient-descent and backpropaga-
tion were proposed by Hochreiter et al. (2001) and Younger et al. (2001). These
two works have been pivotal to the current ﬁeld of Deep Meta-Learning, as the
majority of techniques rely on backpropagation, as we will see on our journey
of contemporary Deep Meta-Learning techniques.
2.4 Overview of the rest of this Work
In the remainder of this work, we will look in more detail at individual meta-
learning methods. As indicated before, the techniques can be grouped into
three main categories (Vinyals, 2017), namely i) metric-, ii) model-, and iii) opti-
mization-based methods. We will discuss them in that order.
To help give an overview of the methods, we draw your attention to the fol-
lowing tables. Table 2 summarizes the three categories and provides key ideas,
and strengths of the approaches. The terms and technical details are explained
more fully in the remainder of this paper. Table 3 contains an overview of all
techniques that are discussed further on.

--- Page 13 ---
A Survey of Deep Meta-Learning
13
Name
RL
Key idea
Bench.
Metric-based
Input similarity
-
Siamese networks

Two-input, shared-weight, class identity
network
1, 8
Matching networks

Learn
input
embeddings
for
cosine-
similarity weighted predictions
1, 2
Prototypical networks

Input embeddings for class prototype
clustering
1, 2, 7
Relation networks

Learn input embeddings and similarity
metric
1, 2, 7
ARC

LSTM-based input fusion through inter-
leaved glimpses
1, 2
GNN

Propagate label information to unlabeled
inputs in a graph
1, 2
Model-based
Internal
and
stateful
latent
task
representations
-
Reccurrent ml.
✓
Deploy Recurrent networks on RL prob-
lems
-
MANNs

External short-term memory module for
fast learning
1
Meta networks
✓
Fast reparameterization of base-learner
by distinct meta-learner
1, 2
SNAIL
✓
Attention mechanism coupled with tem-
poral convolutions
1, 2
CNP

Condition predictive model on embedded
contextual task data
1, 8
Neural stat.

Similarity between latent task embed-
dings
1, 8
Opt.-based
Optimize
for
fast
task-speciﬁc
adaptation
-
LSTM optimizer

RNN proposing weight updates for base-
leaner
6, 8
LSTM ml.
✓
Embed base-learner parameters in cell
state of LSTM
2
RL optimizer

View optimization as RL problem
4, 6
MAML
✓
Learn initialization weights θ for fast
adaptation
1, 2
iMAML
✓
Approx. higher-order gradients, indepen-
dent of optimization path
1, 2
Meta-SGD
✓
Learn both the initialization and updates
1, 2
Reptile
✓
Move initialization towards task-speciﬁc
updated weights
1, 2
LEO

Optimize in lower-dimensional latent pa-
rameter space
2, 3
Online MAML

Accumulate task data for MAML-like
training
4, 8
LLAMA

Maintain probability distribution over
post-update parameters θ′
j
2
PLATIPUS

Learn a probability distribution over
weight initializations θ
-
BMAML
✓
Learn multiple initializations Θ, jointly
optimized by SVGD
2
Diﬀ. solvers

Learn input embeddings for simple base-
learners
1, 2, 3, 4, 5
Table 3 Overview of the discussed Deep Meta-Learning techniques. The table is partitioned
into three sections, i.e., metric-, model-, and optimization-based techniques. All methods
in one section adhere to the key idea of its corresponding category, which is mentioned
in bold font. The columns RL and Bench show whether the techniques are applicable to
reinforcement learning settings and the used benchmarks for testing the performance of
the techniques. Note that all techniques are applicable to supervised learning, with the
exception of RMLs. The benchmark column displays which benchmarks from Section 2.2.2
were used in the paper proposing the technique. The used coding scheme for this column is
the following. 1: Omniglot, 2: miniImageNet, 3: tieredImageNet, 4: CIFAR-100, 5: CIFAR-
FS, 6: CIFAR-10, 7: CUB, 8: MNIST, “-": used other evaluation method that are non-
standard in Deep Meta-Learning and thus not covered in Section 2.2.2. Used abbreviations:
“opt.": optimization, “diﬀ.": diﬀerentiable, “bench.": benchmarks.

--- Page 14 ---
14
Mike Huisman et al.
3 Metric-based Meta-Learning
At a high level, the goal of metric-based techniques is to acquire—among
others—meta-knowledge ω in the form of a good feature space that can be
used for various new tasks. In the context of neural networks, this feature
space coincides with the weights θ of the networks. Then, new tasks can be
learned by comparing new inputs to example inputs (of which we know the
labels) in the meta-learned feature space. The higher the similarity between a
new input and an example, the more likely it is that the new input will have
the same label as the example input.
Metric-based techniques are a form of meta-learning as they leverage their
prior learning experience (meta-learned feature space) to ‘learn’ new tasks
more quickly. Here, ‘learn’ is used in a non-standard way since metric-based
techniques do not make any network changes when presented with new tasks,
as they rely solely on input comparisons in the already meta-learned feature
space. These input comparisons are a form of non-parametric learning, i.e.,
new task information is not absorbed into the network parameters.
More formally, metric-based learning techniques aim to learn a similarity
kernel, or equivalently, attention mechanism kθ (parameterized by θ), that
takes two inputs x1 and x2, and outputs their similarity score. Larger scores
indicate larger similarity. Class predictions for new inputs x can then be made
by comparing x to example inputs xi, of which we know the true labels yi.
The underlying idea being that the larger the similarity between x and xi, the
more likely it becomes that x also has label yi.
Given a task Tj = (Dtr
Tj, Dtest
Tj ) and an unseen input vector x ∈Dtest
Tj , a
probability distribution over classes Y is computed/predicted as a weighted
combination of labels from the support set Dtr
Tj, using similarity kernel kθ, i.e.,
pθ(Y |x, Dtr
Tj) =
X
(xi,yi)∈Dtr
Tj
kθ(x, xi)yi.
(7)
Importantly, the labels yi are assumed to be one-hot encoded, meaning that
they are represented by zero vectors with a ‘1’ on the position of the true
class. For example, suppose there are ﬁve classes in total, and our example
x1 has true class 4. Then, the one-hot encoded label is y1 = [0, 0, 0, 1, 0].
Note that the probability distribution pθ(Y |x, Dtr
Tj) over classes is a vec-
tor of size |Y |, in which the i-th entry corresponds to the probability that
input x has class Yi (given the support set). The predicted class is thus
ˆy = arg maxi=1,2,...,|Y | pθ(Y |x, S)i, where pθ(Y |x, S)i is the computed proba-
bility that input x has class Yi.
3.1 Example
Suppose that we are given a task Tj = (Dtr
Tj, Dtest
Tj ). Furthermore, suppose that
Dtr
Tj = {([0, −4], 1), ([−2, −4], 2), ([−2, 4], 3), ([6, 0], 4)}, where a tuple denotes

--- Page 15 ---
A Survey of Deep Meta-Learning
15
Fig. 5 Illustration of our metric-based example. The blue vector represents the new input
from the query set, whereas the red vectors are inputs from the support set which can be
used to guide our prediction for the new input.
a pair (xi, yi). For simplicity, the example will not use an embedding function,
which maps example inputs onto an (more informative) embedding space. Our
query set only contains one example Dtest
Tj
= {([4, 0.5], y)}. Then, the goal is
to predict the correct label for new input [4, 0.5] using only examples in Dtr
Tj.
The problem is visualized in Figure 5, where red vectors correspond to example
inputs from our support set. The blue vector is the new input that needs to be
classiﬁed. Intuitively, this new input is most similar to the vector [6, 0], which
means that we expect the label for the new input to be the same as that for
[6, 0], i.e., 4.
Suppose we use a ﬁxed similarity kernel, namely the cosine similarity,
i.e., k(x, xi) =
x·xT
i
||x||·||xi||, where ||v|| denotes the length of vector v, i.e.,
||v|| =
p
(P
n v2n). Here, vn denotes the n-th element of placeholder vector
v (substitute v by x or xi). We can now compute the cosine similarity be-
tween the new input [4, 0.5] and every example input xi, as done in Table 4,
where we used the facts that ||x|| = || [4, 0.5] || =
√
42 + 0.52 ≈4.03, and
x
||x|| ≈[4,0.5]
4.03 = [0.99, 0.12].
From this table and Equation 7, it follows that the predicted probability
distribution pθ(Y |x, Dtr
Tj) = −0.12y1−0.58y2−0.37y3+0.99y4 = −0.12[1, 0, 0, 0]−
0.58[0, 1, 0, 0]−0.37[0, 0, 1, 0]+0.99[0, 0, 0, 1] = [−0.12, −0.58, −0.37, 0.99]. Note
that this is not really a probability distribution. That would require normal-
ization such that every element is at least 0 and the sum of all elements is 1.
For the sake of this example, we do not perform this normalization, as it is
clear that class 4 (the class of the most similar example input [6, 0]) will be
predicted.

--- Page 16 ---
16
Mike Huisman et al.
xi
yi
||xi||
xi
||xi||
xi
||xi|| ·
x
||x||
[0, −4]
[1, 0, 0, 0]
4
[0, −1]
−0.12
[−2, −4]
[0, 1, 0, 0]
4.47
[−0.48, −0.89]
−0.58
[−2, 4]
[0, 0, 1, 0]
4.47
[−0.48, 0.89]
−0.37
[6, 0]
[0, 0, 0, 1]
6
[1, 0]
0.99
Table 4 Example showing pair-wise input comparisons. Numbers were rounded to two
decimals.
One may wonder why such techniques are meta-learners, for we could take
any single data set D and use pair-wise comparisons to compute predictions.
At the outer-level, metric-based meta-learners are trained on a distribution of
diﬀerent tasks, in order to learn (among others) a good input embedding func-
tion. This embedding function facilitates inner-level learning, which is achieved
through pair-wise comparisons. As such, one learns an embedding function
across tasks to facilitate task-speciﬁc learning, which is equivalent to “learning
to learn", or meta-learning.
After this introduction to metric-based methods, we will now cover some
key metric-based techniques.
3.2 Siamese Neural Networks
A Siamese neural network (Koch et al., 2015) consists of two neural networks
fθ that share the same weights θ. Siamese neural networks take two inputs
x1, x2, and compute two hidden states fθ(x1), fθ(x2), corresponding to the
activation patterns in the ﬁnal hidden layers. These hidden states are fed into
a distance layer, which computes a distance vector d = |fθ(x1) −fθ(x2)|,
where di is the absolute distance between the i-th elements of fθ(x1) and
fθ(x2). From this distance vector, the similarity between x1, x2 is computed
as σ(αT d), where σ is the sigmoid function (with output range [0,1]), and α
is a vector of free weighting parameters, determining the importance of each
di. This network structure can be seen in Figure 6.
Koch et al. (2015) applied this technique to few-shot image recognition
in two stages. In the ﬁrst stage, they train the twin network on an image
veriﬁcation task, where the goal is to output whether two input images x1 and
x2 have the same class. The network is thus stimulated to learn discriminative
features. In the second stage, where the model is confronted with a new task,
the network leverages its prior learning experience. That is, given a task Tj =
(Dtr
Tj, Dtest
Tj ), and previously unseen input x ∈Dtest
Tj , the predicted class ˆy is
equal to the label yi of the example (xi, yi) ∈Dtr
Tj which yields the highest
similarity score to x. In contrast to other techniques mentioned further in this
section, Siamese neural networks do not directly optimize for good performance
across tasks (consisting of support and query sets). However, they do leverage
learned knowledge from the veriﬁcation task to learn new tasks quickly.

--- Page 17 ---
A Survey of Deep Meta-Learning
17
Fig. 6 Example of a Siamese neural network. Source: Koch et al. (2015).
In summary, Siamese neural networks are a simple and elegant approach
to perform few-shot learning. However, they are not readily applicable outside
the supervised learning setting.
3.3 Matching Networks
Matching networks (Vinyals et al., 2016) build upon the idea that underlies
Siamese neural networks (Koch et al., 2015). That is, they leverage pair-wise
comparisons between the given support set Dtr
Tj = {(xi, yi)}m
i=1 (for a task
Tj), and new inputs x ∈Dtest
Tj
from the query set which we want to classify.
However, instead of assigning the class yi of the most similar example input
xi, matching networks use a weighted combination of all example labels yi in
the support set, based on the similarity of inputs xi to new input x. More
speciﬁcally, predictions are computed as follows: ˆy = Pm
i=1 a(x, xi)yi, where a
is a non-parametric (non-trainable) attention mechanism, or similarity kernel.
This classiﬁcation process is shown in Figure 7. In this ﬁgure, the input to fθ
has to be classiﬁed, using the support set Dtr
Tj (input to gθ).
The attention that is used consists of a softmax over the cosine similarity
c between the input representations, i.e.,
a(x, xi) =
ec(fφ(x),gϕ(xi))
Pm
j=1 ec(fφ(x),gϕ(xj)) ,
(8)
where fφ and gϕ are neural networks, parameterized by φ and ϕ, that map raw
inputs to a (lower-dimensional) latent vector, which corresponds to the output
of the ﬁnal hidden layer of a neural network. As such, the neural networks act as
embedding functions. The larger the cosine similarity between the embeddings

--- Page 18 ---
18
Mike Huisman et al.
Fig. 7 Architecture of matching networks. Source: Vinyals et al. (2016).
of x and xi, the larger a(x, xi), and thus the inﬂuence of label yi on the
predicted label ˆy for input x.
Vinyals et al. (2016) propose two main choices for the embedding functions.
The ﬁrst is to use a single neural network, granting us θ = φ = ϕ and thus
fφ = gϕ. This setup is the default form of matching networks, as shown in
Figure 7. The second choice is to make fφ and gϕ dependent on the support
set Dtr
Tj using Long Short-Term Memory networks (LSTMs). In that case, fφ is
represented by an attention LSTM, and gϕ by a bidirectional one. This choice
for embedding functions is called Full Context Embeddings (FCE), and yielded
an accuracy improvement of roughly 2% on miniImageNet compared to the
regular matching networks, indicating that task-speciﬁc embeddings can aid
the classiﬁcation of new data points from the same distribution.
Matching networks learn a good feature space across tasks for making
pair-wise comparisons between inputs. In contrast to Siamese neural networks
(Koch et al., 2015), this feature space (given by weights θ) is learned across
tasks, instead of on a distinct veriﬁcation task.
In summary, matching networks are an elegant and simple approach to
metric-based meta-learning. However, these networks are not readily applica-
ble outside of supervised learning settings and suﬀer from performance degra-
dation when label distributions are biased (Vinyals et al., 2016).
3.4 Prototypical Networks
Just like matching networks (Vinyals et al., 2016), prototypical networks (Snell
et al., 2017) base their class predictions on the entire support set Dtr
Tj. How-
ever, instead of computing the similarity between new inputs and examples
in the support set, prototypical networks only compare new inputs to class
prototypes (centroids), which are single vector representations of classes in

--- Page 19 ---
A Survey of Deep Meta-Learning
19
some embedding space. Since there are fewer (or equal) class prototypes than
the number of examples in the support set, the amount of required pair-wise
comparisons decreases, saving computational costs.
Fig. 8 Prototypical networks for the case of few-shot learning. The ck are class prototypes
for class k which are computed by averaging the representations of inputs (colored circles) in
the support set. Note that the representation space is partitioned into three disjoint areas,
where each area corresponds to one class. The class with the closest prototype to the new
input x in the query set is then given as prediction. Source: Snell et al. (2017).
The underlying idea of class prototypes is that for a task Tj, there ex-
ists an embedding function that maps the support set onto a space where
class instances cluster nicely around the corresponding class prototypes (Snell
et al., 2017). Then, for a new input x, the class of the prototype nearest to
that input will be predicted. As such, prototypical networks perform nearest
centroid/prototype classiﬁcation in a meta-learned embedding space. This is
visualized in Figure 8.
More formally, given a distance function d : X × X →[0, +∞) (e.g. Eu-
clidean distance) and embedding function fθ, parameterized by θ, prototypical
networks compute class probabilities pθ(Y |x, Dtr
Tj) as follows
pθ(y = k|x, Dtr
Tj) =
exp[−d(fθ(x), ck)]
P
yi exp[−d(fθ(x), cyi)],
(9)
where ck is the prototype/centroid for class k and yi are the classes in the
support set Dtr
Tj. Here, a class prototype for class k is deﬁned as the average of
all vectors xi in the support set such that yi = k. Thus, classes with prototypes
that are nearer to the new input x obtain larger probability scores.
Snell et al. (2017) found that the squared Euclidean distance function as
d gave rise to the best performance. With that distance function, prototypical
networks can be seen as linear models. To see this, note that −d(fθ(x), ck) =
−||fθ(x) −ck||2 = −fθ(x)T fθ(x) + 2cT
k fθ(x) −cT
k ck. The ﬁrst term does not
depend on the class k, and does thus not aﬀect the classiﬁcation decision. The
remainder can be written as wT
k fθ(x)+bk, where wk = 2ck and bk = −cT
k ck.

--- Page 20 ---
20
Mike Huisman et al.
Fig. 9 Relation network architecture. First, the embedding network fϕ embeds all inputs
from the support set Dtr
Tj (the ﬁve example inputs on the left), and the query input (be-
low the fϕ block). All support set embeddings fϕ(xi) are then concatenated to the query
embedding fϕ(x). These concatenated embeddings are passed into a relation network gφ,
which computes a relation score for every pair (xi, x). The class of the input xi that yields
the largest relation score gφ([fϕ(x), fϕ(xi)]) is then predicted. Source: Sung et al. (2018).
Note that this is linear in the output of network fθ, not linear in the input
of the network x. Also, Snell et al. (2017) show that prototypical networks
(coupled with Euclidean distance) are equivalent to matching networks in one-
shot learning settings, as every example in the support set will be its prototype.
In short, prototypical networks save computational costs by reducing the
required number of pair-wise comparisons between new inputs and the sup-
port set, by adopting the concept of class prototypes. Additionally, prototyp-
ical networks were found to outperform matching networks (Vinyals et al.,
2016) in 5-way, k-shot learning for k = 1, 5 on Omniglot (Lake et al., 2011)
and miniImageNet (Vinyals et al., 2016), even though they do not use com-
plex task-speciﬁc embedding functions. Despite these advantages, prototypical
networks are not readily applicable outside of supervised learning settings.
3.5 Relation Networks
In contrast to previously discussed metric-based techniques, Relation networks
(Sung et al., 2018) employ a trainable similarity metric, instead of a pre-
deﬁned one (e.g. cosine similarity as used in matching networks (Vinyals et al.,
2016)). More speciﬁcally, matching networks consist of two chained, neural
network modules: the embedding network/module fϕ which is responsible for
embedding inputs, and the relation network gφ which computes similarity
scores between new inputs x and example inputs xi of which we know the

--- Page 21 ---
A Survey of Deep Meta-Learning
21
labels. A classiﬁcation decision is then made by picking the class of the example
input which yields the largest relation score (or similarity). Note that Relation
networks thus do not use the idea of class prototypes, and simply compare new
inputs x to all example inputs xi in the support set, as done by, e.g., matching
networks (Vinyals et al., 2016).
More formally, we are given a support set Dtr
Tj with some examples (xi, yi),
and a new (previously unseen) input x. Then, for every combination (x, xi),
the Relation network produces a concatenated embedding [fϕ(x), fϕ(xi)],
which is vector obtained by concatenating the respective embeddings of x
and xi. This concatenated embedding is then fed into the relation module gφ.
Finally, gφ computes the relation score between x and xi as
ri = gφ([fϕ(x), fϕ(xi)]).
(10)
The predicted class is then ˆy = yarg maxi ri. This entire process is shown in
Figure 9. Remarkably enough, Relation networks use the Mean-Squared Error
(MSE) of the relation scores, rather than the more standard cross-entropy
loss. The MSE is then propagated backwards through the entire architecture
(Figure 9).
The key advantage of Relation networks is their expressive power, induced
by the usage of a trainable similarity function. This expressivity makes this
technique very powerful. As a result, it yields better performance than previ-
ously discussed techniques that use a ﬁxed similarity metric.
3.6 Graph Neural Networks
Graph neural networks (Garcia and Bruna, 2017) use a more general and
ﬂexible approach than previously discussed techniques for N-way, k-shot clas-
siﬁcation. As such, graph neural networks subsume Siamese (Koch et al., 2015)
and prototypical networks (Snell et al., 2017). The graph neural network ap-
proach represents each task Tj as a fully-connected graph G = (V, E), where
V is a set of nodes/vertices and E a set of edges connecting nodes. In this
graph, nodes vi correspond to input embeddings fθ(xi), concatenated with
their one-hot encoded labels yi, i.e., vi = [fθ(xi), yi]. For inputs x from the
query set (for which we do not have the labels), a uniform prior over all N
possible labels is used: y = [ 1
N , . . . , 1
N ]. Thus, each node contains an input and
label section. Edges are weighted links that connect these nodes.
The graph neural network then propagates information in the graph using
a number of local operators. The underlying idea is that label information can
be transmitted from nodes of which we do have the labels, to nodes for which
we have to predict labels. Which local operators are used, is out of scope for
this paper, and the reader is referred to Garcia and Bruna (2017) for details.
By exposing the graph neural network to various tasks Tj, the propagation
mechanism can be altered to improve the ﬂow of label information in such a
way that predictions become more accurate. As such, in addition to learning

--- Page 22 ---
22
Mike Huisman et al.
Fig. 10 Processing in an attentive recurrent comparator. At every time step, the model
takes a glimpse of a part of an image and incorporates this information into the hidden state
ht. The ﬁnal hidden state after taking various glimpses of a pair of images is then used to
compute a class similarity score. Source: Shyam et al. (2017).
a good input representation function fθ, graph neural networks also learn to
propagate label information from labeled examples to unlabeled inputs.
Graph neural networks achieve good performance in few-shot settings (Gar-
cia and Bruna, 2017) and are also applicable in semi-supervised and active
learning settings.
3.7 Attentive Recurrent Comparators (ARCs)
Attentive recurrent comparators (ARCs) (Shyam et al., 2017) diﬀer from pre-
viously discussed techniques as they do not compare inputs as a whole, but
by parts. This approach is inspired by how humans would make a decision
concerning the similarity of objects. That is, we shift our attention from one
object to the other, and move back and forth to take glimpses of diﬀerent
parts of both objects. In this way, information of two objects is fused from the
beginning, whereas other techniques (e.g., matching networks (Vinyals et al.,
2016) and graph neural networks (Garcia and Bruna, 2017)) only combine
information at the end (after embedding both images) (Shyam et al., 2017).
Given two inputs xi and x, we feed them in interleaved fashion repeatedly
into a recurrent neural network (controller): xi, x, . . . , xi, x. Thus, the image
at time step t is given by It = xi if t is even else x. Then, at each time
step t, the attention mechanism focuses on a square region of the current

--- Page 23 ---
A Survey of Deep Meta-Learning
23
image: Gt = attend(It, Ωt), where Ωt = Wght−1 are attention parameters,
which are computed from the previous hidden state ht−1. The next hidden
state ht+1 = RNN(Gt, ht−1) is given by the glimpse at time t, i.e., Gt, and
the previous hidden state ht−1. The entire sequence consists of g glimpses per
image. After this sequence is fed into the recurrent neural network (indicated
by RNN(◦)), the ﬁnal hidden state h2g is used as combined representation of xi
relative to x. This process is summarized in Figure 10. Classiﬁcation decisions
can then be made by feeding the combined representations into a classiﬁer.
Optionally, the combined representations can be processed by bi-directional
LSTMs before passing them to the classiﬁer.
The attention approach is biologically inspired, and biologically plausible.
A downside of attentive recurrent comparators is the higher computational
cost, while the performance is often not better than less biologically plausible
techniques, such as graph neural networks (Garcia and Bruna, 2017).
3.8 Metric-based Techniques, in conclusion
In this section, we have seen various metric-based techniques. The metric-
based techniques meta-learn an informative feature space that can be used to
compute class predictions based on input similarity scores. Figure 11 shows
the relationships between the various metric-based techniques that we have
covered.
As we can see, Siamese networks (Koch et al., 2015) mark the beginning
of metric-based, deep meta-learning techniques in few-shot learning settings.
They are the ﬁrst to use the idea of predicting classes by comparing inputs
from the support and query sets. This idea was generalized in graph neu-
ral networks (GNNs) (Hamilton et al., 2017; Garcia and Bruna, 2017) where
the information ﬂow between support and query inputs is parametric and thus
more ﬂexible. Matching networks (Vinyals et al., 2016) are directly inspired by
Siamese networks as they use the same core idea (comparing inputs for making
predictions), but directly train in the few-shot setting and use cosine similarity
as a similarity function. Thus, the auxiliary, binary classiﬁcation task used by
Siamese networks is left out, and matching networks directly train on tasks.
Prototypical networks (Snell et al., 2017) increase the robustness of input com-
parisons by comparing every query set input with a class prototype instead of
individual support set examples. This reduces the number of required input
comparisons for a single query input to N instead of k · N. Relation networks
(Sung et al., 2018) replace the ﬁxed, pre-deﬁned similarity metrics used in
matching and prototypical networks by a neural network, which allows for
learning a domain-speciﬁc similarity function. Lastly, attentive recurrent com-
parators (Shyam et al., 2017) take a more biologically plausible approach by
not comparing entire inputs but by taking multiple interleaved glimpses at
various parts of the inputs that are being compared.
Key advantages of these metric-based techniques are that i) the underlying
idea of similarity-based predictions is conceptually simple, and ii) they can

--- Page 24 ---
24
Mike Huisman et al.
Siamese nets
(Koch et al. 2015)
Matching nets
(Vinyals et al.
2016)
No auxiliary
task. Train
under same
conditions as
those at test time
GNN
(Garcia et al.
2017)
Use graph
neural
networks
Prototypical nets
(Snell et al. 2017)
Compare
query inputs
with class
prototypes
Relation nets
(Sung et al. 2018)
Use a neural network
as trainable similarity
function instead of
cosine similarity
ARC
(Shyam et al.
2017)
Compare inputs
by taking
biologically
inspired glimpses
Fig. 11 The relationships between the covered metric-based meta-learning techniques.
be fast at test-time when tasks are small, as the networks do not need to
make task-speciﬁc adjustments. However, when tasks at meta-test time become
more distant from the tasks that were used at meta-train time, metric-learning
techniques are unable to absorb new task information into the network weights.
Consequently, performance may degrade.
Furthermore, when tasks become larger, pair-wise comparisons may be-
come prohibitively expensive. Lastly, most metric-based techniques rely on
the presence of labeled examples, which make them inapplicable outside of
supervised learning settings.
4 Model-based Meta-Learning
A diﬀerent approach to Deep Meta-Learning is the model-based approach. On
a high level, model-based techniques rely upon an adaptive, internal state, in
contrast to metric-based techniques, which generally use a ﬁxed neural network
at test-time.
More speciﬁcally, model-based techniques maintain a stateful, internal rep-
resentation of a task. When presented with a task, a model-based neural net-
work processes the support set in a sequential fashion. At every time step,
an input enters and alters the internal state of the model. Thus, the inter-
nal state can capture relevant task-speciﬁc information, which can be used to
make predictions for new inputs.
Because the predictions are based on internal dynamics that are hidden
from the outside, model-based techniques are also called black-boxes. Infor-
mation from previous inputs must be remembered, which is why model-based
techniques have a memory component, either in- or externally.
Recall that the mechanics of metric-based techniques were limited to pair-
wise input comparisons. This is not the case for model-based techniques, where
the human designer has the freedom to choose the internal dynamics of the

--- Page 25 ---
A Survey of Deep Meta-Learning
25
algorithm. As a result, model-based techniques are not restricted to meta-
learning good feature spaces, as they can also learn internal dynamics, used
to process and predict input data of tasks.
More formally, given a support set Dtr
Tj corresponding to task Tj, model-
based techniques compute a class probability distribution for a new input x
as
pθ(Y |x, Dtr
Tj) = fθ(x, Dtr
Tj),
(11)
where f represents the black-box neural network model, and θ its parameters.
4.1 Example
Using the same example as in Section 3, suppose we are given a task support
set Dtr
Tj = {([0, −4], 1), ([−2, −4], 2), ([−2, 4], 3), ([6, 0], 4)}, where a tuple de-
notes a pair (xi, yi). Furthermore, suppose our query set only contains one
example Dtest
Tj
= {([4, 0.5], 4)}. This problem has been visualized in Figure 5
(in Section 3). For the sake of the example, we do not use an input embed-
ding function: our model will operate on the raw inputs of Dtr
Tj and Dtest
Tj . As
an internal state, our model uses an external memory matrix M ∈R4×(2+1),
with four rows (one for each example in our support set), and three columns
(the dimensionality of input vectors, plus one dimension for the correct label).
Our model proceeds to process the support set sequentially, reading the ex-
amples from Dtr
Tj one by one, and by storing the i-th example in the i-th row
of the memory module. After processing the support set, the memory matrix
contains all examples, and as such, serves as internal task representation.
Given the new input [4, 0.5], our model could use many diﬀerent techniques
to make a prediction based on this representation. For simplicity, assume that
it computes the dot product between x, and every memory M(i) (the 2-D
vector in the i-th row of M, ignoring the correct label), and predicts the
class of the input which yields the largest dot product. This would produce
scores −2, −10, −6, and 24 for the examples in Dtr
Tj respectively. Since the last
example [6, 0] yields the largest dot product, we predict that class, i.e., 4.
Note that this example could be seen as a metric-based technique where
the dot product is used as a similarity function. However, the reason that
this technique is model-based is that it stores the entire task inside a memory
module. This example was deliberately easy for illustrative purposes. More
advanced and successful techniques have been proposed, which we will now
cover.
4.2 Recurrent Meta-Learners
Recurrent meta-learners (Duan et al., 2016; Wang et al., 2016) are, as the name
suggests, meta-learners based on recurrent neural networks. The recurrent

--- Page 26 ---
26
Mike Huisman et al.
Fig. 12 Workﬂow of recurrent meta-learners in reinforcement learning contexts. As men-
tioned in Section 2.1.3, st, rt, and dt denote the state, reward, and termination ﬂag at time
step t. ht refers to the hidden state at time t. Source: Duan et al. (2016).
network serves as dynamic task embedding storage. These recurrent meta-
learners were speciﬁcally proposed for reinforcement learning problems, hence
we will explain them in that setting.
The recurrence is implemented by e.g. an LSTM (Wang et al., 2016) or
a GRU (Duan et al., 2016). The internal dynamics of the chosen Recurrent
Neural Network (RNN) allows for fast adaptation to new tasks, while the
algorithm used to train the recurrent net gradually accumulates knowledge
about the task structure, where each task is modelled as an episode (or set of
episodes).
The idea of recurrent meta-learners is quite simple. That is, given a task Tj,
we simply feed the (potentially processed) environment variables [st+1, at, rt, dt]
(see Section 2.1.3) into an RNN at every time step t. Recall that s, a, r, d de-
note the state, action, reward, and termination ﬂag respectively. At every time
step t, the RNN outputs an action and a hidden state. Conditioned on its hid-
den state ht, the network outputs an action at. The goal is to maximize the
expected reward in each trial. See Figure 12 for a visual depiction. From this
ﬁgure, it also becomes clear why these techniques are model-based. That is,
they embed information from previously seen inputs in the hidden state.
Recurrent meta-learners have shown to perform almost as well as asymp-
totically optimal algorithms on simple reinforcement learning tasks (Wang
et al., 2016; Duan et al., 2016). However, their performance degrades in more
complex settings, where temporal dependencies can span a longer horizon.
Making recurrent meta-learners better at such complex tasks is a direction for
future research.
4.3 Memory-Augmented Neural Networks (MANNs)
The key idea of memory-augmented neural networks (MANNs) (Santoro et al.,
2016) is to enable neural networks to learn quickly with the help of an external
memory. The main controller (the recurrent neural network interacting with
the memory) then gradually accumulates knowledge across tasks, while the

--- Page 27 ---
A Survey of Deep Meta-Learning
27
external memory allows for quick task-speciﬁc adaptation. For this, Santoro
et al. (2016) used Neural Turing Machines (Graves et al., 2014). Here, the
controller is parameterized by θ and acts as the long-term memory of the
memory-augmented neural network, while the external memory module is the
short-term memory.
The workﬂow of memory-augmented neural networks is displayed in Fig-
ure 13. Note that the data from a task is processed as a sequence, i.e., data
are fed into the network one by one. The support set is fed into the memory-
augmented neural network ﬁrst. Afterwards, the query set is processed. During
the meta-train phase, training tasks can be fed into the network in arbitrary
order. At time step t, the model receives input xt with the label of the pre-
vious input, i.e., yt−1. This was done to prevent the network from mapping
class labels directly to the output (Santoro et al., 2016).
-
Fig. 13 Workﬂow of memory-augmented neural networks. Here, an episode corresponds
to a given task Tj. After every episode, the order of labels, classes, and samples should be
shuﬄed to minimize dependence on arbitrarily assigned orders. Source: Santoro et al. (2016).
The interaction between the controller and memory is visualized in Fig-
ure 14. The idea is that the external memory module, containing representa-
tions of previously seen inputs, can be used to make predictions for new inputs.
In short, previously obtained knowledge is leveraged to aid the classiﬁcation
of new inputs. Note that neural networks also attempt to do this, however,
their prior knowledge is slowly accumulated into the network weights, while
an external memory module can directly store such information.
Given an input xt at time t, the controller generates a key kt, which can
be stored in memory matrix M and can be used to retrieve previous repre-
sentations from memory matrix M. When reading from memory, the aim is
to produce a linear combination of stored keys in memory matrix M, giving
greater weight to those which have a larger cosine similarity with the current
key kt. More speciﬁcally, a read vector wr
t is created, in which each entry i

--- Page 28 ---
28
Mike Huisman et al.
Fig. 14 Controller-memory interaction in memory-augmented neural networks. Source:
Santoro et al. (2016).
denotes the cosine similarity between key kt and the memory (from a previous
input) stored in row i, i.e., Mt(i). Then, the representation rt = P
i wr
t (i)M(i)
is retrieved, which is simply a linear combination of all keys (i.e., rows) in
memory matrix M.
Predictions are made as follows. Given an input xt, memory-augmented
neural networks use the external memory to compute the corresponding rep-
resentation rt, which could be fed into a softmax layer, resulting in class
probabilities. Across tasks, memory-augmented neural networks learn a good
input embedding function fθ and classiﬁer weights, which can be exploited
when presented with new tasks.
To write input representations to memory, Santoro et al. (2016) propose
a new mechanism called Least Recently Used Access (LRUA). LRUA either
writes to the least, or most recently used memory location. In the former case,
it preserves recent memories, and in the latter it updates recently obtained
information. The writing mechanism works by keeping track of how often
every memory location is accessed in a usage vector wu
t , which is updated
at every time step according to the following update rule: wu
t := γwu
t−1 +
wr
t +ww
t , where superscripts u, w and r refer to usage, write and read vectors,
respectively. In words, the previous usage vector is decayed (using parameter
γ), while current reads (wr
t) and writes (ww
t ) are added to the usage. Let n
be the total number of reads to memory, and ℓu(n) (ℓu for ‘least used’) be the
n-th smallest value in the usage vector wu
t . Then, the least-used weights are

--- Page 29 ---
A Survey of Deep Meta-Learning
29
Fig. 15 Architecture of a Meta Network. Source: Munkhdalai and Yu (2017).
deﬁned as follows:
wℓu
t (i) =
(
0
if wu
t (i) > ℓu(n)
1
else
.
Then, the write vector ww
t is computed as ww
t = σ(α)wr
t−1 + (1 −σ(α))wℓu
t−1,
where α is a parameter that interpolates between the two weight vectors. As
such, if σ(α) = 1, we write to the most recently used memory, whereas when
σ(α) = 0, we write to the least recently used memory locations. Finally, writing
is performed as follows: Mt(i) := Mt−1(i) + ww
t (i)kt, for all i.
In summary, memory-augmented neural networks (Santoro et al., 2016)
combine external memory and a neural network to achieve meta-learning. The
interaction between a controller, with long-term memory parameters θ, and
memory M, may also be interesting for studying human meta-learning (San-
toro et al., 2016). In contrast to many metric-based techniques, this model-
based technique is applicable to both classiﬁcation and regression problems.
A downside of this approach is the architectural complexity.
4.4 Meta Networks
Meta networks are divided into two distinct subsystems (consisting of neural
networks), i.e., the base- and meta-learner (whereas in memory-augmented
neural networks the base- and meta-components are intertwined). The base-
learner is responsible for performing tasks, and for providing the meta-learner
with meta-information, such as loss gradients. The meta-learner can then com-
pute fast task-speciﬁc weights for itself and the base-learner, such that it can

--- Page 30 ---
30
Mike Huisman et al.
perform better on the given task Tj = (Dtr
Tj, Dtest
Tj ). This workﬂow is depicted
in Figure 15.
The meta-learner consists of neural networks uφ, mϕ, and dψ. Network uφ
is used as input representation function. Networks dψ and mϕ are used to
compute task-speciﬁc weights φ∗and example-level fast weights θ∗. Lastly, bθ
is the base-learner which performs input predictions. Note that we used the
term fast-weights throughout, which refers to task- or input-speciﬁc versions
of slow (initial) weights.
In similar fashion to memory-augmented neural networks (Santoro et al.,
2016), meta networks (Munkhdalai and Yu, 2017) also leverage the idea of
an external memory module. However, meta networks use the memory for a
diﬀerent purpose. The memory stores for each observation xi in the support
set two components, i.e., its representation ri and the fast weights θ∗
i . These
are then used to compute a attention-based representation and fast weights
for new inputs, respectively.
Algorithm 1 Meta networks, by Munkhdalai and Yu (2017)
1: Sample S = {(xi, yi) ∽Dtr
Tj }T
i=1 from the support set
2: for (xi, yi) ∈S do
3:
Li = error(uφ(xi), yi)
4: end for
5: φ∗= dψ({∇φLi}T
i=1)
6: for (xi, yi) ∈Dtr
Tj do
7:
Li = error(bθ(xi), yi)
8:
θ∗
i = mϕ(∇θLi)
9:
Store θ∗
i in i-th position of example-level weight memory M
10:
ri = uφ,φ∗(xi)
11:
Store ri in i-th position of representation memory R
12: end for
13: Ltask = 0
14: for (x, y) ∈Dtest
Tj
do
15:
r = uφ,φ∗(x)
16:
a = attention(R, r)
▷ak is the cosine similarity between r and R(k)
17:
θ∗= softmax(a)T M
18:
Ltask = Ltask + error(bθ,θ∗(x), y)
19: end for
20: Update Θ = {θ, φ, ψ, ϕ} using ∇ΘLtask
The pseudocode for meta networks is displayed in Algorithm 1. First, a
sample of the support set is created (line 1), which is used to compute task-
speciﬁc weights φ∗for the representation network uφ (lines 2-5). Note that
uφ has two tasks, i) it should compute a representation for inputs (xi (line
10 and 15), and ii) it needs to make predictions for inputs (xi, in order to
compute a loss (line 3). To achieve both goals, a conventional neural network
can be used that makes class predictions. The states of the ﬁnal hidden layer
are then used as representations. Typically, the cross entropy is calculated
over the predictions of representation network uφ. When there are multiple

--- Page 31 ---
A Survey of Deep Meta-Learning
31
Fig. 16 Layer augmentation setup used to combine slow and fast weights. Source:
Munkhdalai and Yu (2017).
examples per class in the support set, an alternative is to use a contrastive
loss function (Munkhdalai and Yu, 2017).
Then, meta networks iterate over every example (xi, yi) in the support
set Dtr
Tj. The base-learner bθ attempts to make class predictions for these
examples, resulting in loss values Li (line 7-8). The gradients of these losses are
used to compute fast weights θ∗for example i (line 8), which are then stored in
the i-th row of memory matrix M (line 9). Additionally, input representations
ri are computed and stored in memory matrix R (lines 10-11).
Now, meta networks are ready to address the query set Dtest
Tj . They iterate
over every example (x, y), and compute a representation r of it (line 15). This
representation is matched against the representations of the support set, which
are stored in memory matrix R. This matching gives us a similarity vector a,
where every entry k denotes the similarity between input representation r and
the k-th row in memory matrix R, i.e., R(k) (line 16). A softmax over this
similarity vector is performed to normalize the entries. The resulting vector
is used to compute a linear combination of weights that were generated for
inputs in the support set (line 17). These weights θ∗are speciﬁc for input x
in the query set and can be used by the base-learner b to make predictions for
that input (line 18). The observed error is added to the task loss. After the
entire query set is processed, all involved parameters can be updated using
backpropagation (line 20).
Note that some neural networks use both slow- and fast-weights at the
same time. Munkhdalai and Yu (2017) use a so-called augmentation setup for
this, as depicted in Figure 16.

--- Page 32 ---
32
Mike Huisman et al.
Fig. 17 Architecture and workﬂow of SNAIL for supervised and reinforcement learning
settings. The input layer is red. Temporal Convolution blocks are orange; attention blocks
are green. Source: Mishra et al. (2018).
In short, meta networks rely on a reparameterization of the meta- and
base-learner for every task. Despite the ﬂexibility and applicability to both
supervised and reinforcement learning settings, the approach is quite complex.
It consists of many components, each with its own set of parameters, which can
be a burden on memory usage and computation time. Additionally, ﬁnding the
correct architecture for all the involved components can be time-consuming.
4.5 Simple Neural Attentive Meta-Learner (SNAIL)
Instead of an external memory matrix, SNAIL (Mishra et al., 2018) relies on
a special model architecture to serve as memory. Mishra et al. (2018) argue
that it is not possible to use Recurrent Neural Networks for this, as they
have limited memory capacity, and cannot pinpoint speciﬁc prior experiences
(Mishra et al., 2018). Hence, SNAIL uses a diﬀerent architecture, consisting of
1D temporal convolutions (Oord et al., 2016) and a soft attention mechanism
(Vaswani et al., 2017). The temporal convolutions allow for ‘high bandwidth’
memory access, and the attention mechanism allows one to pinpoint speciﬁc
experiences. Figure 17 visualizes the architecture and workﬂow of SNAIL for
supervised learning problems. From this ﬁgure, it becomes clear why this tech-
nique is model-based. That is, model outputs are based upon the internal state,
computed from earlier inputs.

--- Page 33 ---
A Survey of Deep Meta-Learning
33
Fig. 18 Schematic view of how conditional neural processes work. Here, h denotes a net-
work outputting a representation for a observation, a denotes an aggregation function for
these representations, and g denotes a neural network that makes predictions for unlabelled
observations, based on the aggregated representation. Source: Garnelo et al. (2018).
SNAIL consists of three building blocks. The ﬁrst is the DenseBlock, which
applies a single 1D convolution to the input, and concatenates (in the fea-
ture/horizontal direction) the result. The second is a TCBlock, which is sim-
ply a series of DenseBlocks with exponentially increasing dilation rate of the
temporal convolutions (Mishra et al., 2018). Note that the dilation is nothing
but the temporal distance between two nodes in a network. For example, if we
use a dilation of 2, a node at position p in layer L will receive the activation
from node p−2 from layer L−1. The third block is the AttentionBlock, which
learns to focus on the important parts of prior experience.
In similar fashion to memory-augmented neural networks (Santoro et al.,
2016) (Section 4.3), SNAIL also processes task data in sequence, as shown in
Figure 17. However, the input at time t is accompanied by the label at time
t, instead of t −1 (as was the case for memory-augmented neural networks).
SNAIL learns internal dynamics from seeing various tasks so that it can make
good predictions on the query set, conditioned upon the support set.
A key advantage of SNAIL is that it can be applied to both supervised
and reinforcement learning tasks. In addition, it achieves good performance
compared to previously discussed techniques. A downside of SNAIL is that
ﬁnding the correct architecture of TCBlocks and DenseBlocks can be time-
consuming.
4.6 Conditional Neural Processes (CNPs)
In contrast to previous techniques, a conditional neural process (CNP) (Gar-
nelo et al., 2018) does not rely on an external memory module. Instead, it
aggregates the support set into a single aggregated latent representation. The
general architecture is shown in Figure 18. As we can see, the conditional neu-
ral process operates in three phases on task Tj. First, it observes the support
set Dtr
Tj, including the ground-truth outputs yi. Examples (xi, yi) ∈Dtr
Tj are
embedded using a neural network hθ into representations ri. Second, these

--- Page 34 ---
34
Mike Huisman et al.
Fig. 19 Neural statistician architecture. Edges are neural networks. All incoming inputs to
a node are concatenated.
representations are aggregated using operator a to produce a single represen-
tation r of Dtr
Tj (hence it is model-based). Third, a neural network gφ processes
this single representation r, new inputs x, and produces predictions ˆy.
Let the entire conditional neural process model be denoted by QΘ, where
Θ is a set of all involved parameters {θ, φ}. The training process is diﬀerent
compared to other techniques. Let xTj and yTj denote all inputs and corre-
sponding outputs in Dtr
Tj. Then, the ﬁrst ℓ∽U(0, . . . , k · N −1) examples in
Dtr
Tj are used as a conditioning set Dc
Tj (eﬀectively splitting the support set
in a true training set and a validation set). Given a value of ℓ, the goal is to
maximize the log likelihood (or minimize the negative log likelihood) of the
labels yTj in the entire support set Dtr
Tj
L(Θ) = −ETj∽p(T )
h
Eℓ∽U(0,...,k·N−1)

QΘ(yTj|Dc
Tj, xTj)
i
.
(12)
Conditional neural processes are trained by repeatedly sampling various tasks
and values of ℓ, and propagating the observed loss backwards.
In summary, conditional neural processes use compact representations of
previously seen inputs to aid the classiﬁcation of new observations. Despite its
simplicity and elegance, a disadvantage of this technique is that it is often out-
performed in few-shot settings by other techniques such as matching networks
(Vinyals et al., 2016) (see Section 3.3).
4.7 Neural Statistician
A neural statistician (Edwards and Storkey, 2017) diﬀers from earlier ap-
proaches as it learns to compute summary statistics, or meta-features, of data
sets in an unsupervised manner. These latent embeddings (making the ap-
proach model-based) can then later be used for making predictions. Despite
the broad applicability of the model, we discuss it in the context of Deep
Meta-Learning.

--- Page 35 ---
A Survey of Deep Meta-Learning
35
A neural statistician performs both learning and inference. In the learning
phase, the model attempts to produce generative models ˆPi for every data set
Di. The key assumption that is made by Edwards and Storkey (2017) is that
there exists a generative process Pi, which conditioned on a latent context
vector ci, can produce data set Di. At inference time, the goal is to infer a
(posterior) probability distribution over the context q(c|D).
The model uses a variational autoencoder, which consists of an encoder
and decoder. The encoder is responsible for producing a distribution over
latent vectors z: q(z|x; φ), where x is an input vector, and φ are the encoder
parameters. The encoded input z, which is often of lower dimensionality than
the original input x, can then be decoded by the decoder p(x|z; θ). Here, θ
are the parameters of the decoder. To capture more complex patterns in data
sets, the model uses multiple latent layers z1, . . . , zL, as shown in Figure 19.
Given this architecture, the posterior over c and z1, .., zL (shorthand z1:L) is
given by
q(c, z1:L|D; φ) = q(c|D; φ)
Y
x∈D
q(zL|x, c; φ)
L−1
Y
i=1
q(zi|zi+1, x, c; φ).
(13)
The neural statistician is trained to minimize a three-component loss function,
consisting of the reconstruction loss (how well it models the data), context loss
(how well the inferred context q(c|D; φ) corresponds to the prior P(c), and
latent loss (how well the inferred latent variables zi are modelled).
This model can be applied to N-way, few-shot learning as follows. Con-
struct N data sets for every of the N classes, such that one data set contains
only examples of the same class. Then, the neural statistician is provided with
a new input x, and has to predict its class. It computes a context posterior
Nx = q(c|x; φ) depending on new input x. In similar fashion, context poste-
riors are computed for all of the data sets Ni = q(c|Di; φ). Lastly, it assigns
the label i such that the diﬀerence between Ni and Nx is minimal.
In summary, the neural statistician (Edwards and Storkey, 2017) allows
for quick learning on new tasks through data set modeling. Additionally, it is
applicable to both supervised and unsupervised settings. A downside is that
the approach requires many data sets to achieve good performance (Edwards
and Storkey, 2017).
4.8 Model-based Techniques, in conclusion
In this section, we have discussed various model-based techniques. Despite ap-
parent diﬀerences, they all build on the notion of task internalization. That
is, tasks are processed and represented in the state of the model-based sys-
tem. This state can then be used to make predictions. Figure 20 displays the
relationships between the covered model-based techniques.
Memory-augmented neural networks (MANNs) (Santoro et al., 2016) mark
the beginning of the deep model-based meta-learning techniques. They use

--- Page 36 ---
36
Mike Huisman et al.
MANNs
(Santoro et al.
2016)
Meta Nets
(Munkhdalai et al.
2017)
Store and compute
task-specific
weights in
memory
RMLs
(Duan et al. 2016);
(Wang et al. 2016)
Adapt model-
based meta-
learning to
reinforcement
learning
setting
SNAIL
(Mishra et al.
2018)
CNP
(Garnelo et al. 2018)
Use attention and
special temporal
layers
Neural stat.
(Edwards et al.
2016)
Task-
dependent
classifiers
Fig. 20 The relationships between the covered model-based meta-learning techniques. The
neural statistician and conditional neural process (CNP) form an island in the model-based
approaches.
the idea of feeding the entire support set in sequential fashion into the model
and then making predictions for the query set inputs using the internal state
of the model. Such a model-based approach, where inputs sequentially enter
the model was also taken by recurrent meta-learners (Duan et al., 2016; Wang
et al., 2016) in the reinforcement learning setting. Meta networks (Munkhdalai
and Yu, 2017) also use a large black-box solution but generate task-speciﬁc
weights for every task that is encountered. SNAIL (Mishra et al., 2018) tries
to improve the memory capacity and ability to pinpoint memories, which is
limited in recurrent neural networks, by using attention mechanisms coupled
with special temporal layers. Lastly, the neural statistician and conditional
neural process (CPN) are two techniques that try to learn the meta-features
of data sets in an end-to-end fashion. The neural statistician uses the distance
between meta-features to make class predictions, while the conditional neural
process conditions classiﬁers on these features.
Advantages of model-based approaches include the ﬂexibility of the in-
ternal dynamics of the systems, and their broader applicability compared to
most metric-based techniques. However, model-based techniques are often out-
performed by metric-based techniques in supervised settings (e.g. graph neu-
ral networks (Garcia and Bruna, 2017); Section 3.6), may not perform well
when presented with larger data sets (Hospedales et al., 2020), and generalize
less well to more distant tasks than optimization-based techniques (Finn and
Levine, 2018). We discuss this optimization-based approach next.
5 Optimization-based Meta-Learning
Optimization-based techniques adopt a diﬀerent perspective on meta-learning
than the previous two approaches. They explicitly optimize for fast learning.
Most optimization-based techniques do so by approaching meta-learning as a
bi-level optimization problem. At the inner-level, a base-learner makes task-
speciﬁc updates using some optimization strategy (such as gradient descent).
At the outer-level, the performance across tasks is optimized.

--- Page 37 ---
A Survey of Deep Meta-Learning
37
Fig. 21 Example of an optimization-based technique, inspired by Finn et al. (2017).
More formally, given a task Tj = (Dtr
Tj, Dtest
Tj ) with new input x ∈Dtest
Tj
and base-learner parameters θ, optimization-based meta-learners return
p(Y |x, Dtr
Tj) = fgϕ(θ,Dtr
Tj
,LTj )(x),
(14)
where f is the base-learner, gϕ is a (learned) optimizer that makes task-speciﬁc
updates to the base-learner parameters θ using the support data Dtr
Ti, and loss
function LTj.
5.1 Example
Suppose we are faced with a linear regression problem, where every task is
associated with a diﬀerent function f(x). For this example, suppose our model
only has two parameters: a and b, which together form the function ˆf(x) =
ax + b. Suppose further that our meta-training set consists of four diﬀerent
tasks, i.e., A, B, C, and D. Then, according to the optimization-based view, we
wish to ﬁnd a single set of parameters {a, b} from which we can quickly learn
the optimal parameters for each of the four tasks, as displayed in Figure 21.
In fact, this is the intuition behind the popular optimization-based technique
MAML (Finn et al., 2017). By exposing our model to various meta-training
tasks, we can update the parameters a and b to facilitate quick adaptation.
We will now discuss the core optimization-based techniques in more detail.
5.2 LSTM Optimizer
Standard gradient update rules have the form
θt+1 := θt −α∇θtLTj(θt),
(15)

--- Page 38 ---
38
Mike Huisman et al.
Fig. 22 Workﬂow of the LSTM optimizer. Gradients can only propagate backwards through
solid edges. ft denotes the observed loss at time step t. Source: Andrychowicz et al. (2016).
where α is the learning rate, and LTj(θt) is the loss function with respect to
task Tj and network parameters at time t, i.e., θt. The key idea underlying
LSTM optimizers (Andrychowicz et al., 2016) is to replace the update term
(−α∇LTj(θt)) by an update proposed by an LSTM g with parameters ϕ.
Then, the new update becomes
θt+1 := θt + gϕ(∇θtLTj(θt)).
(16)
This new update allows the optimization strategy to be tailored to a speciﬁc
family of tasks. Note that this is meta-learning, i.e., the LSTM learns to learn.
As such, this technique basically learns an update policy.
The loss function used to train an LSTM optimizer is:
L(ϕ) = ELTj
" T
X
t=1
wtLTj(θt)
#
,
(17)
where T is the number of parameter updates that are made, and wt are weights
indicating the importance of performance after t steps. Note that generally, we
are only interested in the ﬁnal performance after T steps. However, the authors
found that the optimization procedure was better guided by equally weighting
the performance after each gradient descent step. As is often done, second-
order derivatives (arising from the dependency between the updated weights
and the LSTM optimizer) were ignored due to the computational expenses as-
sociated with the computation thereof. This loss function is fully diﬀerentiable
and thus allows for training an LSTM optimizer (see Figure 22). To prevent a
parameter explosion, the same network is used for every coordinate/weight in
the base-learner’s network, causing the update rule to be the same for every
parameter. Of course, the updates depend on their prior values and gradients.
The key advantage of LSTM optimizers is that they can enable faster
learning compared to hand-crafted optimizers, also on diﬀerent data sets than
those used to train the optimizer. However, Andrychowicz et al. (2016) did
not apply this technique to few-shot learning. In fact, they did not apply it

--- Page 39 ---
A Survey of Deep Meta-Learning
39
Fig. 23 LSTM meta-learner computation graph. Gradients can only propagate backwards
through solid edges. The base-learner is denoted as M. (Xt, Yt) are training sets, whereas
(X, Y ) is the test set. Source: Ravi and Larochelle (2017).
across tasks at all. Thus, it is unclear whether this technique can perform
well in few-shot settings, where few data per class are available for training.
Furthermore, the question remains whether it can scale to larger base-learner
architectures.
5.3 LSTM Meta-Learner
Instead of having an LSTM predict gradient updates, Ravi and Larochelle
(2017) embed the weights of the base-learner parameters into the cell state
(long-term memory component) of the LSTM, giving rise to LSTM meta-
learners. As such, the base-learner parameters θ are literally inside the LSTM
memory component (cell state). In this way, cell state updates correspond to
base-learner parameter updates. This idea was inspired by the resemblance be-
tween the gradient and cell state update rules. Gradient updates often have the
form as shown in Equation 15. The LSTM cell state update rule, in contrast,
looks as follows
ct := ft ⊙ct−1 + αt ⊙¯ct,
(18)
where ft is the forget gate (which determines which information should be
forgotten) at time t, ⊙represents the element-wise product, ct is the cell state
at time t, and ¯ct the candidate cell state for time step t, and αt the learning rate
at time step t. Note that if ft = 1 (vector of ones), αt = α, ct−1 = θt−1, and
¯ct = −∇θt−1LTt(θt−1), this update is equivalent to the one used by gradient-
descent. This similarity inspired Ravi and Larochelle (2017) to use an LSTM
as meta-learner that learns to make updates for a base-learner, as shown in
Figure 23.
More speciﬁcally, the cell state of the LSTM is initialized with c0 = θ0,
which will be adjusted by the LSTM to a good common initialization point
across diﬀerent tasks. Then, to update the weights of the base-learner for the

--- Page 40 ---
40
Mike Huisman et al.
next time step t + 1, the LSTM computes ct+1 and sets the weights of the
base-learner equal to that. There is thus a one-to-one correspondence between
ct and θt. The meta-learner’s learning rate αt (see Equation 18), is set equal
to σ(wα · [∇θt−1LTt(θt−1), LTt(θt), θt−1, αt−1] + bα), where σ is the sigmoid
function. Note that the output is a vector, with values between 0 and 1, which
denote the the learning rates for the corresponding parameters. Furthermore,
wα and bα are trainable parameters that part of the LSTM meta-learner. In
words, the learning rate at any time depends on the loss gradients, the loss
value, the previous parameters, and the previous learning rate. The forget
gate, ft, determines what part of the cell state should be forgotten, and is
computed in a similar fashion, but with diﬀerent weights.
To prevent an explosion of meta-learner parameters, weight-sharing is used,
in similar fashion to LSTM optimizers proposed by Andrychowicz et al. (2016)
(Section 5.2). This implies that the same update rule is applied to every weight
at a given time step. The exact update, however, depends on the history of that
speciﬁc parameter in terms of the previous learning rate, loss, etc. For simplic-
ity, second-order derivatives were ignored, by assuming the base-learner’s loss
does not depend on the cell state of the LSTM optimizer. Batch normalization
was applied to stabilize and speed up the learning process.
In short, LSTM optimizers can learn to optimize a base-learner by main-
taining a one-to-one correspondence over time between the base-learner’s weights
and the LSTM cell state. This allows the LSTM to exploit commonalities in
the tasks, allowing for quicker optimization. However, there are simpler ap-
proaches (e.g. MAML (Finn et al., 2017)) that outperform this technique.
5.4 Reinforcement Learning Optimizer
Li and Malik (2018) proposed a framework that casts optimization as a rein-
forcement learning problem. Optimization can then be performed by existing
reinforcement learning techniques. At a high-level, an optimization algorithm
g takes as input an initial set of weights θ0 and a task Tj with corresponding
loss function LTj, and produces a sequence of new weights θ1, . . . , θT , where
θT is the ﬁnal solution found. On this sequence of proposed new weights,
we can deﬁne a loss function L that captures unwanted properties (e.g. slow
convergence, oscillations, etc.). The goal of learning an optimizer can then be
formulated more precisely as follows. We wish to learn an optimal optimizer
g∗= argming ETj∽p(T ),θ0∽p(θ0)[L(g(LTj, θ0))]
(19)
The key insight is that the optimization can be formulated as a Partially
Observable Markov Decision Process (POMDP). Then, the state corresponds
to the current set of weights θt, the action to the proposed update at time step
t, i.e., ∆θt, and the policy to the function that computes the update. With
this formulation, the optimizer g can be learned by existing reinforcement
learning techniques. In their paper, they used a recurrent neural network as an
optimizer. At each time step, they feed it observation features, which depend

--- Page 41 ---
A Survey of Deep Meta-Learning
41
on the previous set of weights, loss gradients, and objective functions, and use
guided policy search to train it.
In summary, Li and Malik (2018) made the ﬁrst step towards general
optimization through reinforcement learning optimizers, which were shown
able to generalize across network architectures and data sets. However, the
base-learner architecture that was used was quite small. The question remains
whether this approach can scale to larger architectures.
5.5 MAML
Fig. 24 MAML learns an initialization point from which it can perform well on various
tasks. Source: Finn et al. (2017).
Model-agnostic meta-learning (MAML) (Finn et al., 2017) uses a simple
gradient-based inner optimization procedure (e.g. stochastic gradient descent),
instead of more complex LSTM procedures or procedures based on reinforce-
ment learning. The key idea of MAML is to explicitly optimize for fast adap-
tation to new tasks by learning a good set of initialization parameters θ. This
is shown in Figure 24: from the learned initialization θ, we can quickly move
to the best set of parameters for task Tj, i.e., θ∗
j for j = 1, 2, 3. The learned
initialization can be seen as the inductive bias of the model, or simply the
set of assumptions (encapsulated in θ) that the model makes concerning the
overall task structure.
More formally, let θ denote the initial model parameters of a model. The
goal is to quickly learn new concepts, which is equivalent to achieving a mini-
mal loss in few gradient update steps. The amount of gradient steps s has to be
speciﬁed upfront, such that MAML can explicitly optimize for achieving good
performance within that number of steps. Suppose we pick only one gradient
update step, i.e., s = 1. Then, given a task Tj = (Dtr
Tj, Dtest
Tj ), gradient descent
would produce updated parameters (fast weights)
θ′
j = θ −α∇θLDtr
Tj (θ),
(20)

--- Page 42 ---
42
Mike Huisman et al.
speciﬁc to task j. The meta-loss of quick adaptation (using s = 1 gradient
steps) across tasks can then be formulated as
ML :=
X
Tj∽p(T )
LDtest
Tj (θ′
j) =
X
Tj∽p(T )
LDtest
Tj (θ −α∇θLDtr
Tj (θ)),
(21)
where p(T ) is a probability distribution over tasks. This expression contains
an inner gradient (∇θLTj(θj)). As such, by optimizing this meta-loss using
gradient-based techniques, we have to compute second-order gradients. One
can easily see this in the computation below
∇θML = ∇θ
X
Tj∽p(T )
LDtest
Tj (θ′
j)
=
X
Tj∽p(T )
∇θLDtest
Tj (θ′
j)
=
X
Tj∽p(T )
L′
Dtest
Tj (θ′
j)∇θ(θ′
j)
=
X
Tj∽p(T )
L′
Dtest
Tj (θ′
j)∇θ(θ −α∇θLDtr
Tj (θ))
=
X
Tj∽p(T )
L′
Dtest
Tj (θ′
j)
|
{z
}
FOMAML
(∇θθ −α∇2
θLDtr
Tj (θ)),
(22)
where we used L′
Dtest
Tj (θ′
j) to denote the derivative of the loss function with re-
spect to the query set, evaluated at the post-update parameters θ′
j. The term
α∇2
θLDtr
Tj (θ) contains the second-order gradients. The computation thereof is
expensive in terms of time and memory costs, especially when the optimiza-
tion trajectory is large (when using a larger number of gradient updates s per
task). Finn et al. (2017) experimented with leaving out second-order gradients,
by assuming ∇θθ′
j = I, giving us First Order MAML (FOMAML, see Equa-
tion 22). They found that FOMAML performed reasonably similar to MAML.
This means that updating the initialization using only ﬁrst order gradients
P
Tj∽p(T ) L′
Dtest
Tj (θ′
j) is roughly equal to using the full gradient expression of
the meta-loss in Equation 22. One can extend the meta-loss to incorporate
multiple gradient steps by substituting θ′
j by a multi-step variant.
MAML is trained as follows. The initialization weights θ are updated by
continuously sampling a batch of m tasks B = {Tj ∽p(T )}m
i=1. Then, for
every task Tj ∈B, an inner update is performed to obtain θ′
j, in turn granting
an observed loss LDtest
Tj (θ′
j). These losses across a batch of tasks are used in
the outer update
θ := θ −β∇θ
X
Tj∈B
LDtest
Tj (θ′
j).
(23)

--- Page 43 ---
A Survey of Deep Meta-Learning
43
The complete training procedure of MAML is displayed in Algorithm 2.
At test-time, when presented with a new task Tj, the model is initialized with
θ, and performs a number of gradient updates on the task data. Note that
the algorithm for FOMAML is equivalent to Algorithm 2, except for the fact
that the update on line 8 is done diﬀerently. That is, FOMAML updates the
initialization with the rule θ = θ −β P
Tj∽p(T ) L′
Dtest
Tj (θ′
j).
Algorithm 2 One-step MAML for supervised learning, by Finn et al. (2017)
1: Randomly initialize θ
2: while not done do
3:
Sample batch of J tasks B = T1, . . . , TJ ∽p(T )
4:
for Tj = (Dtr
Tj , Dtest
Tj ) ∈B do
5:
Compute ∇θLDtr
Tj
(θ)
6:
Compute θ′
j = θ −α∇θLDtr
Tj
(θ)
7:
end for
8:
Update θ = θ −β∇θ
P
Tj∈B LDtest
Tj
(θ′
j)
9: end while
Antoniou et al. (2019), in response to MAML, proposed many technical
improvements that can improve training stability, performance, and gener-
alization ability. Improvements include i) updating the initialization θ after
every inner update step (instead of after all steps are done) to increase gra-
dient propagation, ii) using second-order gradients only after 50 epochs to
increase the training speed, iii) learning layer-wise learning rates to improve
ﬂexibility, iv) annealing the meta-learning rate β over time, and v) some Batch
Normalization tweaks (keep running statistics instead of batch-speciﬁc ones,
and using per-step biases).
MAML has obtained great attention within the ﬁeld of Deep Meta-Learning,
perhaps due to its i) simplicity (only requires two hyperparameters), ii) general
applicability, and iii) strong performance. A downside of MAML, as mentioned
above, is that it can be quite expensive in terms of running time and memory
to optimize a base-learner for every task and compute higher-order derivatives
from the optimization trajectories.
5.6 iMAML
Instead of ignoring higher-order derivatives (as done by FOMAML), which
potentially decreases the performance compared to regular MAML, iMAML
(Rajeswaran et al., 2019) approximates these derivatives in a way that is less
memory-consuming.
Let A denote an inner optimization algorithm (e.g., stochastic gradient
descent), which takes a support set Dtr
Tj corresponding to task Tj and initial
model weights θ, and produces new weights θ′
j = A(θ, Dtr
Tj). MAML has to
compute the derivative

--- Page 44 ---
44
Mike Huisman et al.
∇θLDtest
Tj (θ′
j) = L′
Dtest
Tj (θ′
j)∇θ(θ′
j),
(24)
where Dtest
Tj
is the query set corresponding to task Tj. This equation is a simple
result of applying the chain rule. Importantly, note that ∇θ(θ′
j) diﬀerentiates
through A(θ, Dtr
Tj), while L′
Dtest
Tj (θ′
j) does not, as it represents the gradient of
the loss function evaluated at θ′
j. Rajeswaran et al. (2019) make use of the
following lemma.
If (I+ 1
λ∇2
θLDtr
Tj (θ′
j)) is invertible (i.e., (I+ 1
λ∇2
θLDtr
Tj (θ′
j))−1 exists), then
∇θ(θ′
j) =

I + 1
λ∇2
θLDtr
Tj (θ′
j)
−1
.
(25)
Here, λ is a regularization parameter. The reason for this is discussed below.
Combining Equation 24 and Equation 25, we have that
∇θLDtest
Tj (θ′
j) = L′
Dtest
Tj (θ′
j)

I + 1
λ∇2
θLDtr
Tj (θ′
j)
−1
.
(26)
The idea is to obtain an approximate gradient vector gj that is close to
this expression, i.e., we want the diﬀerence to be small
gj −L′
Dtest
Tj (θ′
j)

I + 1
λ∇2
θLDtr
Tj (θ′
j)
−1
= ϵ,
(27)
for some small tolerance vector ϵ. If we multiply both sides by the inverse of
the inverse factor, i.e.,

I + 1
λ∇2
θLDtr
Tj (θ′
j)

, we get
gT
j

I + 1
λ∇2
θLDtr
Tj (θ′
j)

gj −gT
j L′
Dtest
Tj (θ′
j) = ϵ′,
(28)
where ϵ′ absorbed the multiplication factor. We wish to minimize this expres-
sion for gj, and that can be performed using optimization techniques such as
the conjugate gradient algorithm (Rajeswaran et al., 2019). This algorithm
does not need to store Hessian matrices, which decreases the memory cost
signiﬁcantly. In turn, this allows iMAML to work with more inner gradient
update steps. Note, however, that one needs to perform explicit regularization
in that case to avoid overﬁtting. The conventional MAML did not require this,
as it uses only a few number of gradient steps (equivalent to an early stopping
mechanism).
At each inner loop step, iMAML computes the meta-gradient gj. After
processing a batch of tasks, these gradients are averaged and used to update
the initialization θ. Since it does not diﬀerentiate through the optimization
process, we are free to use any other (non-diﬀerentiable) inner-optimizer.

--- Page 45 ---
A Survey of Deep Meta-Learning
45
Fig. 25 Meta-SGD learning process. Source: Li et al. (2017).
In summary, iMAML reduces memory costs signiﬁcantly as it need not dif-
ferentiate through the optimization trajectory, also allowing for greater ﬂexi-
bility in the choice of inner optimizer. Additionally, it can account for larger
optimization paths. The computational costs stay roughly the same compared
to MAML (Finn et al., 2017). Future work could investigate more inner opti-
mization procedures (Rajeswaran et al., 2019).
5.7 Meta-SGD
Meta-SGD (Li et al., 2017), or meta-stochastic gradient descent, is similar
to MAML (Finn et al., 2017) (Section 5.5). However, on top of learning an
initialization, Meta-SGD also learns learning rates for every model parameter
in θ, building on the insight that the optimizer can be seen as a trainable
entity.
The standard SGD update rule is given in Equation 15. The meta-SGD
optimizer uses a more general update, namely
θ′
j ←θ −α ⊙∇θLDtr
Tj (θ),
(29)
where ⊙is the element-wise product. Note that this means that alpha (learning
rate) is now a vector—hence the bold font— instead of scalar, which allows for
greater ﬂexibility in the sense that each parameter has its own learning rate.
The goal is to learn the initialization θ, and learning rate vector α, such that
the generalization ability is as large as possible. More mathematically precise,
the learning objective is
minα,θETj∽p(T )[LDtest
Tj (θ′
j)] = ETj∽p(T )[LDtest
Tj (θ −α ⊙∇θLDtr
Tj (θ))],
(30)
where we used a simple substitution for θ′
j. LDtr
Tj and LDtest
Tj
are the losses
computed on the support and query set respectively. Note that this formula-
tion stimulates generalization ability (as it includes the query set loss LDtest
Tj ,
which can be observed during the meta-training phase). The learning process
is visualized in Figure 25. Note that the meta-SGD optimizer is trained to

--- Page 46 ---
46
Mike Huisman et al.
maximize generalization ability after only one update step. Since this learning
objective has a fully diﬀerentiable loss function, the meta-SGD optimizer itself
can be trained using standard SGD.
In summary, Meta-SGD is more expressive than MAML as it does not only
learn an initialization but also learning rates per parameter. This, however,
does come at the cost of an increased number of hyperparameters.
5.8 Reptile
Reptile (Nichol et al., 2018) is another optimization-based technique that, like
MAML (Finn et al., 2017), solely attempts to ﬁnd a good set of initialization
parameters θ. The way in which Reptile attempts to ﬁnd this initialization is
quite diﬀerent from MAML. It repeatedly samples a task, trains on the task,
and moves the model weights towards the trained weights (Nichol et al., 2018).
Algorithm 3 displays the pseudocode describing this simple process.
Algorithm 3 Reptile, by Nichol et al. (2018)
1: Initialize θ
2: for i = 1, 2, . . . do
3:
Sample task Tj = (Dtr
Tj , Dtest
Tj ) and corresponding loss function LTj
4:
θ′
j = SGD(LDtr
Tj
, θ, k)
▷Perform k gradient update steps to get θ′
j
5:
θ := θ + ϵ(θ′
j −θ)
▷Move initialization point θ towards θ′
j
6: end for
Nichol et al. (2018) note that it is possible to treat (θ −θ′
j)/α as gradients,
where α is the learning rate of the inner stochastic gradient descent optimizer
(line 4 in the pseudocode), and to feed that into a meta-optimizer (e.g. Adam).
Moreover, instead of sampling one task at a time, one could sample a batch
of n tasks, and move the initialization θ towards the average update direction
¯θ = 1
n
Pn
j=1(θ′
j −θ), granting the update rule θ := θ + ϵ¯θ.
The intuition behind Reptile is that updating the initialization weights
towards updated parameters will grant a good inductive bias for tasks from
the same family. By performing Taylor expansions of the gradients of Reptile
and MAML (both ﬁrst-order and second-order), Nichol et al. (2018) show that
the expected gradients diﬀer in their direction. They argue, however, that in
practice, the gradients of Reptile will also bring the model towards a point
minimizing the expected loss over tasks.
A mathematical argument as to why Reptile works goes as follows. Let θ
denote the initial parameters, and θ∗
j the optimal set of weights for task Tj.
Lastly, let d be the Euclidean distance function. Then, the goal is to minimize
the distance between the initialization point θ and the optimal point θ∗
j, i.e.,
minθ ETj∽p(T )[1
2d(θ, θ∗
j)2].
(31)

--- Page 47 ---
A Survey of Deep Meta-Learning
47
The gradient of this expected distance with respect to the initialization θ
is given by
∇θETj∽p(T )[1
2d(θ, θ∗
j)2] = ETj∽p(T )[1
2∇θd(θ, θ∗
j)2]
= ETj∽p(T )[θ −θ∗
j],
(32)
where we used the fact that the gradient of the squared Euclidean distance
between two points x1 and x2 is the vector 2(x1 −x2). Nichol et al. (2018) go
on to argue that performing gradient descent on this objective would result in
the following update rule
θ = θ −ϵ∇θ
1
2d(θ, θ∗
j)2
= θ −ϵ(θ∗
j −θ).
(33)
Since we do not know θ∗
Tj, one can approximate this by term by k steps of gra-
dient descent SGD(LTj, θ, k). In short, Reptile can be seen as gradient descent
on the distance minimization objective given in Equation 31. A visualization is
shown in Figure 26. The initialization θ is moving towards the optimal weights
for tasks 1 and 2 in interleaved fashion (hence the oscillations).
Fig. 26 Schematic visualization of Reptile’s learning trajectory. Here, θ∗
1 and θ∗
2 are the
optimal weights for tasks T1 and T2 respectively. The initialization parameters θ oscillate
between these. Adapted from Nichol et al. (2018).
In conclusion, Reptile is an extremely simple meta-learning technique,
which does not need to diﬀerentiate through the optimization trajectory like,
e.g., MAML (Finn et al., 2017), saving time and memory costs. However, the
theoretical foundation is a bit weaker due to the fact that it does not directly
optimize for fast learning as done by MAML, and performance may be a bit
worse than that of MAML in some settings.
5.9 Latent embedding optimization (LEO)
Latent Embedding Optimization, or LEO, was proposed by Rusu et al. (2018)
to combat an issue of gradient-based meta-learners, such as MAML (see Sec-
tion 5.5), in few-shot settings (N-way, k-shot). These techniques operate in a

--- Page 48 ---
48
Mike Huisman et al.
high-dimensional parameter space using gradient information from only a few
examples, which could lead to poor generalization.
Fig. 27 Workﬂow of LEO. adapted from Rusu et al. (2018).
LEO alleviates this issue by learning a lower-dimensional latent embed-
ding space, which indirectly allows us to learn a good set of initial parameters
θ. Additionally, the embedding space is conditioned upon tasks, allowing for
more expressivity. In theory, LEO could ﬁnd initial parameters for the en-
tire base-learner network, but the authors only experimented with setting the
parameters for the ﬁnal layers.
The complete workﬂow of LEO is shown in Figure 27. As we can see, given
a task Tj, the corresponding support set Dtr
Tj is fed into an encoder, which
produces hidden codes for each example in that set. These hidden codes are
paired and concatenated in every possible manner, granting us (Nk)2 pairs,
where N is the number of classes in the training set, and k the number of
examples per class. These paired codes are then fed into a relation net (Sung
et al., 2018) (see Section 3.5). The resulting embeddings are grouped by class,
and parameterize a probability distribution over latent codes zn (for class n)
in a low dimensional space Z. More formally, let xℓ
n denote the ℓ-th example of
class n in Dtr
Tj. Then, the mean µe
n and variance σe
n of a Gaussian distribution
over latent codes for class n are computed as
µe
n, σe
n =
1
Nk2
k
X
ℓp=1
N
X
m=1
k
X
ℓq=1
gφr
 gφe(xℓp
n ), gφe(xℓq
m)

,
(34)
where φr, φe are parameters for the relation net and encoder respectively.
Intuitively, the three summations ensure that every example with class n in
Dtr
Tj is paired with every example from all classes n. Given µe
n, and σe
n, one
can sample a latent code zn ∽N(µe
n, diag(σe2
n )) for class n, which serves as
latent embedding of the task training data.
The decoder can then generate a task-speciﬁc initialization θn for class n as
follows. First, one computes a mean and variance for a Gaussian distribution

--- Page 49 ---
A Survey of Deep Meta-Learning
49
using the latent code
µd
n, σd
n = gφd(zn).
(35)
These are then used to sample initialization weights θn ∽N(µd
n, diag(σd2
n )).
The loss from the generated weights can then be propagated backwards to
adjust the embedding space. In practice, generating such a high-dimensional
set of parameters from a low-dimensional embedding can be quite problematic.
Therefore, LEO uses pre-trained models, and only generates weights for the
ﬁnal layer, which limits the expressivity of the model.
A key advantage of LEO is that it optimizes in a lower-dimensional la-
tent embedding space, which aids generalization performance. However, the
approach is more complex than e.g. MAML (Finn et al., 2017), and its appli-
cability is limited to few-shot learning settings.
5.10 Online MAML (FTML)
Online MAML (Finn et al., 2019) is an extension of MAML (Finn et al., 2017)
to make it applicable to online learning settings (Anderson, 2008). In the online
setting, we are presented with a sequence of tasks Tt with corresponding loss
functions {LTt}T
t=1, for some potentially inﬁnite time horizon T. The goal is to
pick a sequence of parameters {θt}T
t=1 that performs well on the presented loss
functions. This objective is captured by the RegretT over the entire sequence,
which is deﬁned by Finn et al. (2019) as follows
RegretT =
T
X
t=1
LTt(θ′
t) −minθ
T
X
t=1
LTt(θ′
t),
(36)
where θ are the initial model parameters (just as MAML), and θ′
t are param-
eters resulting from a one-step gradient update (starting from θ) on task t.
Here, the left term reﬂects the updated parameters chosen by the agent (θt),
whereas the right term presents the minimum obtainable loss (in hindsight)
from a single ﬁxed set of parameters θ. Note that this setup assumes that the
agent can make updates to its chosen parameters (transform its initial choice
at time t from θt to θ′
t).
Finn et al. (2019) propose FTML (Follow The Meta Leader), inspired by
FTL (Follow The Leader) (Hannan, 1957; Kalai and Vempala, 2005), to mini-
mize the regret. The basic idea is to set the parameters for the next time step
(t + 1) equal to the best parameters in hindsight, i.e.,
θt+1 := argminθ
t
X
k=1
LTk(θ′
k).
(37)
The gradient to perform meta-updates is then given by

--- Page 50 ---
50
Mike Huisman et al.
gt(θ) := ∇θETk∽pt(T )LTk(θ′
k),
(38)
where pt(T ) is a uniform distribution over tasks 1, . . . , t (at time t).
Algorithm 4 contains the full pseudocode for FTML. In this algorithm,
MetaUpdate performs a few (Nmeta) meta-steps. In each meta-step, a task
is sampled from B, together with train and test mini-batches to compute
the gradient gt in Equation 37. The initialization θ is then updated (θ :=
θ −βgt(θ)), where β is the meta-learning rate. Note that the memory usage
keeps increasing over time, as at every time step t, we append tasks to the
buﬀer B, and keep task data sets in memory.
Algorithm 4 FTML by Finn et al. (2019)
Require: Performance threshold γ
1: Initialize empty task buﬀer B
2: for t = 1, . . . do
3:
Initialize data set Dt = ∅
4:
Append Tt to B
5:
while |Dt| < N do
6:
Append batch of data {(xi, yi)}n
i=1 to Dt
7:
θt = MetaUpdate(θt, B, t)
8:
Compute θ′
t
9:
if LDtest
Tt
(θ′
t) < γ then
10:
Save |Dt| as the eﬃciency for task Tt
11:
end if
12:
end while
13:
Save ﬁnal performance LDtest
Tt
(θ′
t)
14:
θt+1 = θt
15: end for
In summary, Online MAML is a robust technique for online-learning (Finn
et al., 2019). A downside of this approach is the computational costs that keep
growing over time, as all encountered data are stored. Reducing these costs
is a direction for future work. Also, one could experiment with how well the
approach works when more than one inner gradient update steps per task are
used, as mentioned by Finn et al. (2019).
5.11 LLAMA
Grant et al. (2018) mold MAML into a probabilistic framework, such that a
probability distribution over task-speciﬁc parameters θ′
j is learned, instead of a
single one. In this way, multiple potential solutions can be obtained for a task.
The resulting technique is called LLAMA (Laplace Approximation for Meta-
Adaptation). Importantly, LLAMA is only developed for supervised learning
settings.

--- Page 51 ---
A Survey of Deep Meta-Learning
51
A key observation is that a neural network fθ′
j, parameterized by updated
parameters θ′
j (obtained from few gradient updates using Dtr
Tj), outputs class
probabilities p(yi|xi, θ′
j). To minimize the error on the query set Dtest
Tj , the
model must output large probability scores for the true classes. This objective
is captured in the maximum log-likelihood loss function
LDtest
Tj (θ′
j) = −
X
xi,yi∈Dtest
Tj
log p(yi|xi, θ′
j).
(39)
Simply put, if we see a task j as a probability distribution over examples
pTj, we wish to maximize the probability that the model predicts the correct
class yi, given an input xi. This can be done by plain gradient descent, as
shown in Algorithm 5, where β is the meta-learning rate. Line 4 refers to
ML-LAPLACE, which is a subroutine that computes task-speciﬁc updated
parameters θ′
j, and estimates the negative log likelihood (loss function) which
is used to update the initialization θ, as shown in Algorithm 6. Grant et al.
(2018) approximated the quadratic curvature matrix ˆH using K-FAC (Martens
and Grosse, 2015).
The trick is that the initialization θ deﬁnes a distribution p(θ′
j|θ) over task-
speciﬁc parameters θ′
j. This distribution was taken to be a diagonal Gaussian
(Grant et al., 2018). Then, to sample solutions for a new task Tj, one can
simply generate possible solutions θ′
j from the learned Gaussian distribution.
Algorithm 5 LLAMA by Grant et al. (2018)
1: Initialize θ randomly
2: while not converged do
3:
Sample a batch of J tasks: B = T1, . . . , TJ ∽p(T )
4:
Estimate E(xi,yi)∽pTj [−log p(yi|xi, θ)] ∀Tj ∈B using ML-LAPLACE
5:
θ = θ −β∇θ
P
j E(xi,yi)∽pTj [−log p(yi|xi, θ)
6: end while
Algorithm 6 ML-LAPLACE (Grant et al., 2018)
1: θ′
j = θ
2: for k = 1, . . . , K do
3:
θ′
j = θ′
j + α∇θ′
j log p(yi ∈Dtr
Tj |θ′
j, xi ∈Dtr
Tj )
4: end for
5: Compute
curvature
matrix
ˆH
=
∇2
θ′
j [−log p(yi
∈
Dtest
Tj |θ′
j, xi
∈
Dtest
Tj )] +
∇2
θ′
j [−log p(θ′
j|θ)]
6: return −log p(yi ∈Dtest
Tj |θ′
j, xi ∈Dtest
Tj ) + η log[det( ˆH)]
In short, LLAMA extends MAML in a probabilistic fashion, such that one
can obtain multiple solutions for a single task, instead of one. This does, how-

--- Page 52 ---
52
Mike Huisman et al.
ever, increase the computational costs. On top of that, the used Laplace ap-
proximation (in ML-LAPLACE) can be quite inaccurate (Grant et al., 2018).
5.12 PLATIPUS
PLATIPUS (Finn et al., 2018) builds upon the probabilistic interpretation of
LLAMA (Grant et al., 2018), but learns a probability distribution over ini-
tializations θ, instead of task-speciﬁc parameters θ′
j. Thus, PLATIPUS allows
one to sample an initialization θ ∽p(θ), which can be updated with gradient
descent to obtain task-speciﬁc weights (fast weights) θ′
j.
Algorithm 7 PLATIPUS training algorithm by Finn et al. (2018)
1: Initialize Θ = {µθ, σ2
θ, vq, γp, γq}
2: while Not done do
3:
Sample batch of tasks B = {Tj ∽p(T )}m
i=1
4:
for Tj ∈B do
5:
Dtr
Tj , Dtest
Tj
= Tj
6:
Compute ∇µθ LDtest
Tj
(µθ)
7:
Sample θ ∽q = N(µθ −γq∇µθ LDtest
Tj
(µθ), vq)
8:
Compute ∇θLDtr
Tj
(θ)
9:
Compute fast weights θ′
i = θ −α∇θLDtr
Tj
(θ)
10:
end for
11:
p(θ|Dtr
Tj ) = N(µθ −γp∇µθ LDtr
Tj
(µθ), σ2
θ)
12:
Compute ∇Θ
P
Tj LDtest
Tj
(φi) + DKL(q(θ|Dtest
Tj ), p(θ|Dtr
Tj ))

13:
Update Θ using the Adam optimizer
14: end while
The approach is best explained by its pseudocode, as shown in Algorithm 7.
In contrast to the original MAML, PLATIPUS introduces ﬁve more parameter
vectors (line 1). All of these parameters are used to facilitate the creation of
Gaussian distributions over prior initializations (or simply priors) θ. That is,
µθ represents the vector mean of the distributions. σ2
q, and vq represent the
covariances of train and test distributions respectively. γx for x = q, p are
learning rate vectors for performing gradient steps on distributions q (line 6
and 7) and P (line 11).
The key diﬀerence with the regular MAML is that instead of having a single
initialization point θ, we now learn distributions over priors: q and P, which
are based on query and support data sets of task Tj respectively. Since these
data sets come from the same task, we want the distributions q(θ|Dtest
Tj ), and
p(θ|Dtr
Tj) to be close to each other. This is enforced by the Kullback–Leibler
divergence (DKL) loss term on line 12, which measures the distance between
the two distributions. Importantly, note that q (line 7) and P (line 11) use
vector means which are computed with one gradient update steps using the

--- Page 53 ---
A Survey of Deep Meta-Learning
53
query and support data sets respectively. The idea is that the mean of the
Gaussian distributions should be close to the updated mean µθ because we
want to enable fast learning. As one can see, the training process is very
similar to that of MAML (Finn et al., 2017) (Section 5.5), with some small
adjustments to allow us to work with the probability distributions over θ.
At test-time, one can simply sample a new initialization θ from the prior
distribution p(θ|Dtr
Tj) (note that q cannot be used at test-time as we do not
have access to Dtest
Tj ), and apply a gradient update on the provided support
set Dtr
Tj. Note that this allows us to sample multiple potential initializations θ
for the given task.
The key advantage of PLATIPUS is that it is aware of its uncertainty, which
greatly increases the applicability of Deep Meta-Learning in critical domains
such as medical diagnosis (Finn et al., 2018). Based on this uncertainty, it can
ask for labels of some inputs it is unsure about (active learning). A downside
to this approach, however, is the increased computational costs, and the fact
that it is not applicable to reinforcement learning.
5.13 Bayesian MAML (BMAML)
Bayesian MAML (Yoon et al., 2018) is another probabilistic variant of MAML
that can generate multiple solutions. However, instead of learning a distri-
bution over potential solutions, BMAML simply keeps M possible solutions,
and optimizes them in joint fashion. Recall that probabilistic MAMLs (e.g.,
PLATIPUS) attempt to maximize the data likelihood of task Tj, i.e., p(ytest
j
|θ′
j),
where θ′
j are task-speciﬁc fast weights obtained by one or more gradient up-
dates. Yoon et al. (2018) model this likelihood using Stein Variational Gradient
Descent (SVGD) (Liu and Wang, 2016).
To obtain M solutions, or equivalently, parameter settings θm, SVGD keeps
a set of M particles Θ = {θm}M
i=1. At iteration t, every θt ∈Θ is updated as
follows
θt+1 = θt + ϵ(φ(θt))
(40)
where φ(θt) = 1
M
M
X
m=1

k(θm
t , θt)∇θm
t log p(θm
t ) + ∇θm
t k(θm
t , θt)

.
(41)
Here, k(x, x′) is a similarity kernel between x and x′. The authors used a
radial basis function (RBF) kernel, but in theory, any other kernel could be
used. Note that the update of one particle depends on the other gradients
of particles. The ﬁrst term in the summation (k(θm
t , θt)∇θm
t log p(θm
t )) moves
the particle in the direction of the gradients of other particles, based on parti-
cle similarity. The second term (∇θm
t k(θm
t , θt)) ensures that particles do not
collapse (repulsive force) (Yoon et al., 2018).
These particles can then be used to approximate the probability distribu-
tion of the test labels

--- Page 54 ---
54
Mike Huisman et al.
p(ytest
j
|θ′
j) ≈1
M
M
X
m=1
p(ytest
j
|θm
Tj),
(42)
where θm
Tj is the m-th particle obtained by training on the support set Dtr
Tj of
task Tj.
Yoon et al. (2018) proposed a new meta-loss to train BMAML, called the
Chaser Loss. This loss relies on the insight that we want the approximated
parameter distribution (obtained from the support set pn
Tj(θTj|Dtr, Θ0)) and
true distribution p∞
Tj(θTj|Dtr ∪Dtest) to be close to each other (since the task
is the same). Here, n denotes the number of SVGD steps, and Θ0 is the set of
initial particles, in similar fashion to the initial parameters θ seen by MAML.
Since the true distribution is unknown, Yoon et al. (2018) approximate it by
running SVGD for s additional steps, granting us the leader Θn+s
Tj , where the
s additional steps are performed on the combined support and query set. The
intuition is that as the number of updates increases, the obtained distributions
become more like the true ones. Θn
Tj in this context is called the chaser as it
wants to get closer to the leader. The proposed meta-loss is then given by
LBMAML(Θ0) =
X
Tj∈B
M
X
m=1
||θn,m
Tj
−θn+s,m
Tj
||2
2.
(43)
The full pseudocode of BMAML is shown in Algorithm 8. Here, Θn
Tj(Θ0)
denotes the set of particles after n updates on task Tj, and SG means “stop
gradients" (we do not want the leader to depend on the initialization, as the
leader must lead).
Algorithm 8 BMAML by Yoon et al. (2018)
1: Initialize Θ0
2: for t = 1, . . . until convergence do
3:
Sample a batch of tasks B from p(T )
4:
for task Tj ∈B do
5:
Compute chaser Θn
Tj (Θ0) = SV GDn(Θ0; Dtr
Tj , α)
6:
Compute leader Θn+s
Tj
(Θ0) = SV GDs(Θn
Tj (Θ0); Dtr
Tj ∪Dtest
Tj , α)
7:
end for
8:
Θ0 = Θ0 −β∇Θ0
P
Tj∈B d(Θn
Tj (Θ0), SG(Θn+s
Tj
(Θ0)))
9: end for
In summary, BMAML is a robust optimization-based meta-learning tech-
nique that can propose M potential solutions to a task. Additionally, it is
applicable to reinforcement learning by using Stein Variational Policy Gradi-
ent instead of SVGD. A downside of this approach is that one has to keep M
parameter sets in memory, which does not scale well. Reducing the memory

--- Page 55 ---
A Survey of Deep Meta-Learning
55
costs is a direction for future work (Yoon et al., 2018). Furthermore, SVGD
is sensitive to the selected kernel function, which was pre-deﬁned in BMAML.
However, Yoon et al. (2018) point out that it may be beneﬁcial to learn the
kernel function instead. This is another possibility for future research.
5.14 Simple Diﬀerentiable Solvers
Bertinetto et al. (2019) take a quite diﬀerent approach. That is, they pick
simple base-learners that have an analytical closed-form solution. The intu-
ition is that the existence of a closed-form solution allows for good learning
eﬃciency. They propose two techniques using this principle, namely R2-D2
(Ridge Regression Diﬀerentiable Discriminator), and LR-D2 (Logistic Regres-
sion Diﬀerentiable Discriminator). We cover both in turn.
Let gφ : X →Re be a pre-trained input embedding model (e.g. a CNN),
which outputs embeddings with a dimensionality of e. Furthermore, assume
that we use a linear predictor function f(gφ(xi)) = gφ(xi)W, where W is a
e × o weight matrix, and o is the output dimensionality (of the label). When
using (regularized) Ridge Regression (done by R2-D2), one uses the optimal
W, i.e.,
W ∗= arg min
W
||XW −Y ||2
2 + γ||W||2
= (XT X + γI)−1XT Y,
(44)
where X ∈Rn×e is the input matrix, containing n rows (one for each em-
bedded input gφ(xi)), Y ∈Rn×o is the output matrix with correct outputs
corresponding to the inputs, and γ is a regularization term to prevent overﬁt-
ting. Note that the analytical solution contains the term (XT X) ∈Re×e, which
is quadratic in the size of the embeddings. Since e can become quite large when
using deep neural networks, Bertinetto et al. (2019) use Woodburry’s identity
W ∗= XT (XXT + γI)−1Y,
(45)
where XXT ∈Rn×n is linear in the embedding size, and quadratic in the
number of examples, which is more manageable in few-shot settings, where n
is very small. To make predictions with this Ridge Regression based model,
one can compute
ˆY = αXtestW ∗+ β,
(46)
where α and β are hyperparameters of the base-learner that can be learned
by the meta-learner, and Xtest ∈Rm×e corresponds to the m test inputs of a
given task. Thus, the meta-learner needs to learn α, β, γ, and φ (embedding
weights of the CNN).

--- Page 56 ---
56
Mike Huisman et al.
The technique can also be applied to iterative solvers when the optimiza-
tion steps are diﬀerentiable (Bertinetto et al., 2019). LR-D2 uses the Logistic
Regression objective and Newton’s method as solver. Outputs y ∈{−1, +1}n
are now binary. Let w denote a parameter row of our linear model (param-
eterized by W). Then, the i-th iteration of Newton’s method updates wi as
follows
wi = (XT diag(si)X + γI)−1XT diag(si)zi,
(47)
where µi = σ(wT
i−1X), si = µi(1 −µi), zi = wT
i−1X + (y −µi)/si, and σ is
the sigmoid function. Since the term XT diag(si)X is a matrix of size e × e,
and thus again quadratic in the embedding size, Woodburry’s identity is also
applied here to obtain
wi = XT (XXT + λdiag(si)−1)−1zi,
(48)
making it quadratic in the input size, which is not a big problem since n is
small in the few-shot setting. The main diﬀerence compared to R2-D2 is that
the base-solver has to be run for multiple iterations to obtain W.
In the few-shot setting, the base-level optimizers compute the weight ma-
trix W for a given task Ti. The obtained loss on the query set of a task LDtest
is then used to update the parameters φ of the input embedding function (e.g.
CNN) and the hyperparameters of the base-learner.
Lee et al. (2019) have done similar work to Bertinetto et al. (2019), but
with linear Support Vector Machines (SVMs) as base-learner. Their approach
is dubbed MetaOptNet and achieved state-of-the-art performance on few-
shot image classiﬁcation.
In short, simple diﬀerentiable solvers are simple, reasonably fast in terms
of computation time, but limited to few-shot learning settings. Investigating
the use of other simple base-learners is a direction for future work.
5.15 Optimization-based Techniques, in conclusion
Optimization-based aim to learn new tasks quickly through (learned) opti-
mization procedures. Note that this closely resembles base-level learning, which
also occurs through optimization (e.g., gradient descent). However, in contrast
to base-level techniques, optimization-based meta-learners can learn the opti-
mizer and/or are exposed to multiple tasks, which allows them to learn how to
learn new tasks quickly. Figure 28 shows the relationships between the covered
optimization-based techniques.
As we can see, the LSTM optimizer (Andrychowicz et al., 2016), which
replaces hand-crafted optimization procedures such as gradient descent by a
trainable LSTM, can be seen as the starting point for these optimization-
based meta-learning techniques. Li and Malik (2018) also aim to learn the
optimization procedure with reinforcement learning instead of gradient-based

--- Page 57 ---
A Survey of Deep Meta-Learning
57
LSTM optimizer
(Andrychowicz et
al. 2016)
Adapt for few-
shot learning
LSTM meta-
learner
(Ravi et al. 2017)
MAML
(Finn et al. 2017)
Replace trainable
optimizer by gradient
descent
iMAML
(Rajeswaran et al.
2019)
Meta-SGD
(Li et al. 2017)
Reptile
(Nichol et al.
2018)
LEO
(Rusu et al. 2018)
LLAMA
(Grant et al.
2018)
PLATIPUS
(Finn et al. 2018)
BMAML
(Yoon et al. 2018)
R2-D2/LR-D2
(Bertinetto et al.
2019)
MetaOptNet
(Lee et al. 2019)
Online MAML
(Finn et al. 2019)
Trainable
learning rates
First-order
simplification
Bayesian
interpretation
Learn a prior over initialization
parameters instead of a posterior
Learn
multiple
initializations
Optimize in lower-
dimensional space
More flexible inner-
loop optimization
Adapt for
the online
setting
RL optimizer
(Li et al. 2018)
Use reinforcement
learning for optimization
Use simple ridge
regression and
logistic regression
as classifier
Use SVM as
classifier
Fig. 28 The relationships between the covered optimization-based meta-learning tech-
niques. As one can see, MAML has a central position in this graph of techniques as it
has inspired many other works.
methods. The LSTM meta-learner (Ravi and Larochelle, 2017) extends the
LSTM optimizer to the few-shot setting by not only learning the optimization
procedure but also a good set of initial weights. This way, it can be used
across tasks. MAML (Finn et al., 2017) is a simpliﬁcation of the LSTM meta-
learner as it replaces the trainable LSTM optimizer by hand-crafted gradient
descent. MAML has received considerable attention within the ﬁeld of deep
meta-learning, and has, as one can see, inspired many other works.
Meta-SGD is an enhancement of MAML that not only learns the initial
parameters, but also the learning rates (Li et al., 2017). LLAMA (Grant et al.,
2018), PLATIPUS (Finn et al., 2018), and online MAML (Finn et al., 2019)
extend MAML to the active and online learning settings. LLAMA and PLATI-
PUS are probabilistic interpretations of MAML, which allow them to sample
multiple solutions for a given task and quantify their uncertainty. BMAML
(Yoon et al., 2018) takes a more discrete approach as it jointly optimizes
a discrete set of M initializations. iMAML (Rajeswaran et al., 2019) aims
to overcome the computational expenses associated with the computation of

--- Page 58 ---
58
Mike Huisman et al.
second-order derivatives, which is needed by MAML. Through implicit diﬀer-
entiation, they also allow for the use of non-diﬀerentiable inner loop optimiza-
tion procedures. Reptile (Nichol et al., 2018) is an elegant ﬁrst-order meta-
learning algorithm for ﬁnding a set of initial parameters and removes the need
for computing higher-order derivatives. LEO (Rusu et al., 2018) tries to im-
prove the robustness of MAML by optimizing in lower-dimensional parameter
space through the use of an encoder-decoder architecture. Lastly, R2-D2, LR-
D2 (Bertinetto et al., 2019), and Lee et al. (2019) use simple classical machine
learning methods (ridge regression, logistic regression, SVM, respectively) as
a classiﬁer on top of a learned feature extractor.
A key advantage of optimization-based approaches is that they can achieve
better performance on wider task distributions than, e.g., model-based ap-
proaches (Finn and Levine, 2018). However, optimization-based techniques op-
timize a base-learner for every task that they are presented with and/or learn
the optimization procedure, which is computationally expensive (Hospedales
et al., 2020).
Optimization-based meta-learning is a very active area of research. We
expect future work to be done in order to reduce the computational demands of
these methods and improve the solution quality and level of generalization. We
think that benchmarking and reproducibility research will play an important
role in these improvements.
6 Concluding Remarks
In this section, we give a helicopter view of all that we discussed, and the ﬁeld
of Deep Meta-Learning in general. We will also discuss challenges and future
research.
6.1 Overview
In recent years, there has been a shift in focus in the broad meta-learning
community. Traditional algorithm selection and hyperparameter optimization
for classical machine learning techniques (e.g. Support Vector Machines, Lo-
gistic Regression, Random Forests, etc.) have been augmented by Deep Meta-
Learning, or equivalently, the pursuit of self-improving neural networks that
can leverage prior learning experience to learn new tasks more quickly. Instead
of training a new model from scratch for diﬀerent tasks, we can use the same
(meta-learning) model across tasks. As such, meta-learning can widen the ap-
plicability of powerful deep learning techniques to domains where fewer data
are available and computational resources are limited.
Deep Meta-Learning techniques are characterized by their meta-objective,
which allows them to maximize performance across various tasks, instead of a
single one, as is the case in base-level learning objectives. This meta-objective
is reﬂected in the training procedure of meta-learning methods, as they learn

--- Page 59 ---
A Survey of Deep Meta-Learning
59
on a set of diﬀerent meta-training tasks. The few-shot setting lends itself nicely
towards this end, as tasks consist of few data points. This makes it computa-
tionally feasible to train on many diﬀerent tasks, and it allows us to evaluate
whether a neural network can learn new concepts from few examples. Task
construction for training and evaluation does require some special attention.
That is, it has been shown beneﬁcial to match training and test conditions
(Vinyals et al., 2016), and perhaps train in a more diﬃcult setting than the
one that will be used for evaluation (Snell et al., 2017).
On a high level, there are three categories of Deep Meta-Learning tech-
niques, namely i) metric-, ii) model-, and iii) optimization-based ones, which
rely on i) computing input similarity, ii) task embeddings with states, and
iii) task-speciﬁc updates, respectively. Each approach has strengths and weak-
nesses. Metric-learning techniques are simple and eﬀective (Garcia and Bruna,
2017) but are not readily applicable outside of the supervised learning setting
(Hospedales et al., 2020). Model-based techniques, on the other hand, can
have very ﬂexible internal dynamics, but lack generalization ability to more
distant tasks than the ones used at meta-train time (Finn and Levine, 2018).
Optimization-based approaches have shown greater generalizability, but are in
general computationally expensive, as they optimize a base-learner for every
task (Finn and Levine, 2018; Hospedales et al., 2020).
Table 2 provides a concise, tabular overview of these approaches. Many
techniques have been proposed for each one of the categories, and the under-
lying ideas may vary greatly, even within the same category. Table 3, therefore,
provides an overview of all methods and key ideas that we have discussed in
this work, together with their applicability to supervised learning (SL) and
reinforcement learning (RL) settings, key ideas, and benchmarks that were
used for testing them. Table 5 displays an overview of the 1- and 5-shot clas-
siﬁcation performances (reported by the original authors) of the techniques
on the frequently used miniImageNet benchmark. Moreover, it displays the
used backbone (feature extraction module) as well as the ﬁnal classiﬁcation
mechanism. From this table, it becomes clear that the 5-shot performance is
typically better than the 1-shot performance, indicating that data scarcity is
a large bottleneck for achieving good performance. Moreover, there is a strong
relationship between the expressivity of the backbone and the performance.
That is, deeper backbones tend to give rise to better classiﬁcation performance.
The best performance is achieved by MetaOptNet, yielding a 1-shot accuracy
of 64.09% and a 5-shot accuracy of 80.00%. Note however that MetaOptNet
used a deeper backbone than most of the other techniques.
6.2 Open Challenges and Future Work
Despite the great potential of Deep Meta-Learning techniques, there are still
open challenges, which we discuss here.
Figure 1 in Section 1 displays the accuracy scores of the covered meta-
learning techniques on 1-shot miniImageNet classiﬁcation. Techniques that

--- Page 60 ---
60
Mike Huisman et al.
Name
Backbone
Classiﬁer
1-shot
5-shot
Metric-based
Siamese nets
-
-
-
Matching nets
64-64-64-64
Cosine sim.
43.56 ± 0.84
55.31 ± 0.73
Prototypical nets
64-64-64-64
Euclidean dist.
49.42 ± 0.78
68.20 ± 0.66
Relation nets
64-96-128-256
Sim. network
50.44 ± 0.82
65.32 ± 0.70
ARC
-
64-1 dense
49.14 ± −
-
GNN
64-96-128-256
Softmax
50.33 ± 0.36
66.41 ± 0.63
Model-based
RMLs
-
-
-
MANNs
-
-
-
Meta nets
64-64-64-64-64
64-Softmax
49.21 ± 0.96
-
SNAIL
Adj. ResNet-12
Softmax
55.71 ± 0.99
68.88 ± 0.92
CNP
-
-
-
Neural stat.
-
-
-
Opt.-based
LSTM optimizer
-
-
-
LSTM ml.
32-32-32-32
Softmax
43.44 ± 0.77
60.60 ± 0.71
RL optimizer
-
-
-
MAML
32-32-32-32
Softmax
48.70 ± 1.84
63.11 ± 0.92
iMAML
64-64-64-64
Softmax
49.30 ± 1.88
-
Meta-SGD
64-64-64-64
Softmax
50.47 ± 1.87
64.03 ± 0.94
Reptile
32-32-32-32
Softmax
48.21 ± 0.69
66.00 ± 0.62
LEO
WRN-28-10
Softmax
61.76 ± 0.08
77.59 ± 0.12
Online MAML
-
-
-
LLAMA
64-64-64-64
Softmax
49.40 ± 1.83
-
PLATIPUS
-
-
-
BMAML
64-64-64-64-64
Softmax
53.80 ± 1.46
-
Diﬀ. solvers
R2-D2
96-192-384-512
Ridge regr.
51.8 ± 0.2
68.4 ± 0.2
LR-D2
96-192-384-512
Log. regr.
51.90 ± 0.20
68.70 ± 0.20
MetaOptNet
ResNet-12
SVM
64.09 ± 0.62
80.00 ± 0.45
Table 5 Comparison of the accuracy scores of the covered meta-learning techniques on 1-
and 5-shot miniImageNet classiﬁcation. Scores are taken from the original papers. The ±
indicates the 95% conﬁdence interval. The backbone is the used feature extraction module.
The classiﬁer column shows the ﬁnal layer(s) that were used to transform the features
into class predictions. Used abbreviations: “sim.": similarity, “Adj.": adjusted, and “dist.":
distance, “log.": logistic, “regr.": regression, “ml.": meta-learner, “opt.": optimization.
were not tested in this setting by the original authors are omitted. As we
can see, the performance of the techniques is related to the expressivity of
the used backbone (ordered in increasing order on the x-axis). For example,
the best-performing techniques, LEO and MetaOptNet, use the largest net-
work architectures. Moreover, the fact that diﬀerent techniques use diﬀerent
backbones poses a problem as it is diﬃcult to fairly compare their classiﬁca-
tion performance. An obvious question arises to which degree the diﬀerence in
performance is due to methodological improvements, or due to the fact that
a better backbone architecture was chosen. For this reason, we think that it
would be useful to perform a large-scale benchmark test where techniques are
compared when they use the same backbones. This would also allow us to

--- Page 61 ---
A Survey of Deep Meta-Learning
61
get a more clear idea of how the expressivity of the feature extraction module
aﬀects the performance.
Another challenge of Deep Meta-Learning techniques is that they can be
susceptible to the memorization problem (meta-overﬁtting), where the neural
network has memorized tasks seen at meta-training time and fails to generalize
to new tasks. More research is required to better understand this problem.
Clever task design and meta-regularization may prove useful to avoid such
problems (Yin et al., 2020).
Another problem is that most of the meta-learning techniques discussed
in this work are evaluated on narrow benchmark sets. This means that the
data that the meta-learner used for training are not too distant from the data
used for evaluating its performance. As such, one may wonder how well these
techniques are able to adapt to more distant tasks. Chen et al. (2019) showed
that the ability to adapt to new tasks decreases as they become more distant
from the tasks seen at training time. Moreover, a simple non-meta-learning
baseline (based on pre-training and ﬁne-tuning) can outperform state-of-the-
art meta-learning techniques when meta-test tasks come from a diﬀerent data
set than the one used for meta-training.
In reaction to these ﬁndings, Triantaﬁllou et al. (2020) have recently pro-
posed the Meta-Dataset benchmark, which consists of various previously used
meta-learning benchmarks such as Omniglot (Lake et al., 2011) and ImageNet
(Deng et al., 2009). This way, meta-learning techniques can be evaluated in
more challenging settings where tasks are diverse. Following Hospedales et al.
(2020), we think that this new benchmark can prove to be a good means to-
wards the investigation and development of meta-learning algorithms for such
challenging scenarios.
As mentioned earlier in this section, Deep Meta-Learning has the appeal-
ing prospect of widening the applicability of deep learning techniques to more
real-world domains. For this, increasing the generalization ability of these tech-
niques is very important. Additionally, the computational costs associated with
the deployment of meta-learning techniques should be small. While these tech-
niques can learn new tasks quickly, meta-training can be quite computationally
expensive. Thus, decreasing the required computation time and memory costs
of Deep Meta-Learning techniques remains an open challenge.
Some real-world problems demand systems that can perform well in online,
or active learning settings. The investigation of Deep Meta-Learning in these
settings (Finn et al., 2018; Yoon et al., 2018; Finn et al., 2019; Munkhdalai
and Yu, 2017; Vuorio et al., 2018) remains an important direction for future
work.
Yet another direction for future research is the creation of compositional
Deep Meta-Learning systems, which instead of learning ﬂat and associative
functions x →y, organize knowledge in a compositional manner. This would
allow them to decompose an input x into several (already learned) components
c1(x), . . . , cn(x), which in turn could help the performance in low-data regimes
(Tokmakov et al., 2019).

--- Page 62 ---
62
Mike Huisman et al.
The question has been raised whether contemporary Deep Meta-Learning
techniques actually learn how to perform rapid learning, or simply learn a
set of robust high-level features, which can be (re)used for many (new) tasks.
Raghu et al. (2020) investigated this question for the most popular Deep Meta-
Learning technique MAML and found that it largely relies on feature reuse. It
would be interesting to see whether we can develop techniques that rely more
upon fast learning, and what the eﬀect would be on performance.
Lastly, it may be useful to add more meta-abstraction levels, giving rise
to, e.g., meta-meta-learning, meta-meta-...-learning (Hospedales et al., 2020;
Schmidhuber, 1987).
Acknowledgements Thanks to Herke van Hoof for an insightful discussion on LLAMA.
Thanks to Pavel Brazdil for his encouragement and feedback on a preliminary version of
this work.
References
Anderson T (2008) The Theory and Practice of Online Learning. AU Press,
Athabasca University
Andrychowicz M, Denil M, Colmenarejo SG, Hoﬀman MW, Pfau D, Schaul T,
Shillingford B, de Freitas N (2016) Learning to learn by gradient descent by
gradient descent. In: Advances in Neural Information Processing Systems
29, Curran Associates Inc., NIPS’16, pp 3988–3996
Antoniou A, Edwards H, Storkey A (2019) How to train your MAML. In:
International Conference on Learning Representations, ICLR’19
Barrett DG, Hill F, Santoro A, Morcos AS, Lillicrap T (2018) Measuring ab-
stract reasoning in neural networks. In: Proceedings of the 35th International
Conference on Machine Learning, JLMR.org, ICML’18, pp 4477–4486
Bengio S, Bengio Y, Cloutier J, Gecsei J (1997) On the optimization of a
synaptic learning rule. In: Optimality in Artiﬁcial and Biological Neural
Networks, Lawrance Erlbaum Associates, Inc.
Bengio Y, Bengio S, Cloutier J (1991) Learning a synaptic learning rule. In:
International Joint Conference on Neural Networks, IEEE, IJCNN’91, vol 2
Bertinetto L, Henriques JF, Torr PHS, Vedaldi A (2019) Meta-learning with
diﬀerentiable closed-form solvers. In: International Conference on Learning
Representations, ICLR’19
Brazdil P, Carrier CG, Soares C, Vilalta R (2008) Metalearning: Applications
to Data Mining. Springer-Verlag Berlin Heidelberg
Chen WY, Liu YC, Kira Z, Wang YC, Huang JB (2019) A Closer Look at
Few-shot Classiﬁcation. In: International Conference on Learning Represen-
tations, ICLR’19
Deng J, Dong W, Socher R, Li LJ, Li K, Fei-Fei L (2009) ImageNet: A Large-
Scale Hierarchical Image Database. In: Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, IEEE, pp 248–255

--- Page 63 ---
A Survey of Deep Meta-Learning
63
Duan Y, Schulman J, Chen X, Bartlett PL, Sutskever I, Abbeel P (2016)
RL2: Fast Reinforcement Learning via Slow Reinforcement Learning. arXiv
preprint arXiv:161102779
Edwards H, Storkey A (2017) Towards a Neural Statistician. In: International
Conference on Learning Representations, ICLR’17
Finn C, Levine S (2018) Meta-Learning and Universality: Deep Representa-
tions and Gradient Descent can Approximate any Learning Algorithm. In:
International Conference on Learning Representations, ICLR’18
Finn C, Abbeel P, Levine S (2017) Model-agnostic Meta-learning for Fast
Adaptation of Deep Networks. In: Proceedings of the 34th International
Conference on Machine Learning, JMLR.org, ICML’17, pp 1126–1135
Finn C, Xu K, Levine S (2018) Probabilistic Model-Agnostic Meta-Learning.
In: Advances in Neural Information Processing Systems 31, Curran Asso-
ciates Inc., NIPS’18, pp 9516–9527
Finn C, Rajeswaran A, Kakade S, Levine S (2019) Online Meta-Learning. In:
Chaudhuri K, Salakhutdinov R (eds) Proceedings of the 36th International
Conference on Machine Learning, JLMR.org, ICML’19, pp 1920–1930
Garcia V, Bruna J (2017) Few-Shot Learning with Graph Neural Networks.
In: International Conference on Learning Representations, ICLR’17
Garnelo M, Rosenbaum D, Maddison C, Ramalho T, Saxton D, Shanahan M,
Teh YW, Rezende D, Eslami SMA (2018) Conditional neural processes. In:
Dy J, Krause A (eds) Proceedings of the 35th International Conference on
Machine Learning, JMLR.org, ICML’18, vol 80, pp 1704–1713
Goceri E (2019a) Capsnet topology to classify tumours from brain images and
comparative evaluation. IET Image Processing 14(5):882–889
Goceri E (2019b) Challenges and recent solutions for image segmentation in
the era of deep learning. In: 2019 ninth international conference on image
processing theory, tools and applications (IPTA), IEEE, pp 1–6
Goceri E (2020) Convolutional neural network based desktop applications to
classify dermatological diseases. In: 2020 IEEE 4th International Conference
on Image Processing, Applications and Systems (IPAS), IEEE, pp 138–143
Goceri E, Karakas AA (2020) Comparative evaluations of cnn based net-
works for skin lesion classiﬁcation. In: 14th International Conference on
Computer Graphics, Visualization, Computer Vision and Image Processing
(CGVCVIP), Zagreb, Croatia, pp 1–6
Grant E, Finn C, Levine S, Darrell T, Griﬃths T (2018) Recasting Gradient-
Based Meta-Learning as Hierarchical Bayes. In: International Conference on
Learning Representations, ICLR’18
Graves A, Wayne G, Danihelka I (2014) Neural Turing Machines. arXiv
preprint arXiv:14105401
Gupta A, Mendonca R, Liu Y, Abbeel P, Levine S (2018) Meta-Reinforcement
Learning of Structured Exploration Strategies. In: Advances in Neural Infor-
mation Processing Systems 31, Curran Associates Inc., NIPS’18, pp 5302–
5311
Hamilton WL, Ying R, Leskovec J (2017) Inductive representation learning
on large graphs. In: Advances in Neural Information Processing Systems,

--- Page 64 ---
64
Mike Huisman et al.
Curran Associates Inc., NIPS’17, vol 30, p 1025–1035
Hannan J (1957) Approximation to bayes risk in repeated play. Contributions
to the Theory of Games 3:97–139
Hastie T, Tibshirani R, Friedman J (2009) The Elements of Statistical Learn-
ing: Data Mining, Inference, and Prediction, 2nd edn. Springer, New York,
NY
He K, Zhang X, Ren S, Sun J (2015) Delving Deep into Rectiﬁers: Surpassing
Human-Level Performance on ImageNet Classiﬁcation. In: Proceedings of
the IEEE International Conference on Computer Vision, pp 1026–1034
Hinton GE, Plaut DC (1987) Using Fast Weights to Deblur Old Memories. In:
Proceedings of the 9th Annual Conference of the Cognitive Science Society,
pp 177–186
Hochreiter S, Younger AS, Conwell PR (2001) Learning to Learn Using Gra-
dient Descent. In: International Conference on Artiﬁcial Neural Networks,
Springer, pp 87–94
Hospedales T, Antoniou A, Micaelli P, Storkey A (2020) Meta-Learning in
Neural Networks: A Survey. arXiv preprint arXiv:200405439
Iqbal MS, Luo B, Khan T, Mehmood R, Sadiq M (2018) Heterogeneous trans-
fer learning techniques for machine learning. Iran Journal of Computer Sci-
ence 1(1):31–46
Iqbal MS, El-Ashram S, Hussain S, Khan T, Huang S, Mehmood R, Luo B
(2019a) Eﬃcient cell classiﬁcation of mitochondrial images by using deep
learning. Journal of Optics 48(1):113–122
Iqbal MS, Luo B, Mehmood R, Alrige MA, Alharbey R (2019b) Mitochondrial
organelle movement classiﬁcation (ﬁssion and fusion) via convolutional neu-
ral network approach. IEEE Access 7:86570–86577
Iqbal MS, Ahmad I, Bin L, Khan S, Rodrigues JJ (2020) Deep learning recog-
nition of diseased and normal cell representation. Transactions on Emerging
Telecommunications Technologies p e4017
Jankowski N, Duch W, Grąbczewski K (2011) Meta-Learning in Computa-
tional Intelligence, vol 358. Springer-Verlag Berlin Heidelberg
Kalai A, Vempala S (2005) Eﬃcient algorithms for online decision problems.
Journal of Computer and System Sciences 71(3):291–307
Koch G, Zemel R, Salakhutdinov R (2015) Siamese Neural Networks for One-
shot Image Recognition. In: Proceedings of the 32nd International Confer-
ence on Machine Learning, JMLR.org, ICML’15, vol 37
Krizhevsky A (2009) Learning Multiple Layers of Features from Tiny Images.
Tech. rep., University of Toronto
Krizhevsky A, Sutskever I, Hinton GE (2012) ImageNet Classiﬁcation with
Deep Convolutional Neural Networks. In: Advances in Neural Information
Processing Systems, pp 1097–1105
Lake B, Salakhutdinov R, Gross J, Tenenbaum J (2011) One shot learning of
simple visual concepts. In: Proceedings of the annual meeting of the cogni-
tive science society, vol 33, pp 2568–2573
Lake BM, Ullman TD, Tenenbaum JB, Gershman SJ (2017) Building machines
that learn and think like people. Behavioral and brain sciences 40

--- Page 65 ---
A Survey of Deep Meta-Learning
65
LeCun Y, Cortes C, Burges C (2010) MNIST handwritten digit database.
http://yann.lecun.com/exdb/mnist, accessed: 7-10-2020
Lee K, Maji S, Ravichandran A, Soatto S (2019) Meta-Learning with Diﬀer-
entiable Convex Optimization. In: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, IEEE, pp 10657–10665
Li K, Malik J (2018) Learning to Optimize Neural Nets. arXiv preprint
arXiv:170300441
Li Z, Zhou F, Chen F, Li H (2017) Meta-SGD: Learning to Learn Quickly for
Few-Shot Learning. arXiv preprint arXiv:170709835
Liu Q, Wang D (2016) Stein Variational Gradient Descent: A General Purpose
Bayesian Inference Algorithm. In: Advances in neural information processing
systems 29, Curran Associates Inc., NIPS’16, pp 2378–2386
Martens J, Grosse R (2015) Optimizing Neural Networks with Kronecker-
factored Approximate Curvature. In: Proceedings of the 32th International
Conference on Machine Learning, JMLR.org, ICML’15, pp 2408–2417
Miconi T, Stanley K, Clune J (2018) Diﬀerentiable plasticity: training plastic
neural networks with backpropagation. In: Dy J, Krause A (eds) Proceed-
ings of the 35th International Conference on Machine Learning, JLMR.org,
ICML’18, pp 3559–3568
Miconi T, Rawal A, Clune J, Stanley KO (2019) Backpropamine: training self-
modifying neural networks with diﬀerentiable neuromodulated plasticity. In:
International Conference on Learning Representations, ICLR’19
Mishra N, Rohaninejad M, Chen X, Abbeel P (2018) A Simple Neural At-
tentive Meta-Learner. In: International Conference on Learning Represen-
tations, ICLR’18
Mitchell TM (1980) The need for biases in learning generalizations. Tech. Rep.
CBM-TR-117, Rutgers University
Mnih V, Kavukcuoglu K, Silver D, Graves A, Antonoglou I, Wierstra D, Ried-
miller M (2013) Playing Atari with Deep Reinforcement Learning. arXiv
preprint arXiv:13125602
Munkhdalai T, Yu H (2017) Meta networks. In: Proceedings of the 34th Inter-
national Conference on Machine Learning, JLMR.org, ICML’17, pp 2554–
2563
Nagabandi A, Clavera I, Liu S, Fearing RS, Abbeel P, Levine S, Finn C (2019)
Learning to Adapt in Dynamic, Real-World Environments Through Meta-
Reinforcement Learning. In: International Conference on Learning Repre-
sentations, ICLR’19
Naik DK, Mammone RJ (1992) Meta-neural networks that learn by learning.
In: International Joint Conference on Neural Networks, IEEE, IJCNN’92,
vol 1, pp 437–442
Nichol A, Achiam J, Schulman J (2018) On First-Order Meta-Learning Algo-
rithms. arXiv preprint arXiv:180302999
Oord Avd, Dieleman S, Zen H, Simonyan K, Vinyals O, Graves A, Kalchbren-
ner N, Senior A, Kavukcuoglu K (2016) WaveNet: A Generative Model for
Raw Audio. arXiv preprint arXiv:160903499

--- Page 66 ---
66
Mike Huisman et al.
Oreshkin B, López PR, Lacoste A (2018) Tadam: Task dependent adaptive
metric for improved few-shot learning. In: Advances in Neural Information
Processing Systems 31, Curran Associates Inc., NIPS’18, pp 721–731
Pan SJ, Yang Q (2009) A Survey on Transfer Learning. IEEE Transactions on
knowledge and data engineering 22(10):1345–1359
Peng Y, Flach PA, Soares C, Brazdil P (2002) Improved Dataset Characteri-
sation for Meta-learning. In: International Conference on Discovery Science,
Springer, Lecture Notes in Computer Science, vol 2534, pp 141–152
Raghu A, Raghu M, Bengio S, Vinyals O (2020) Rapid Learning or Feature
Reuse? Towards Understanding the Eﬀectiveness of MAML. In: Interna-
tional Conference on Learning Representations, ICLR’20
Rajeswaran A, Finn C, Kakade SM, Levine S (2019) Meta-Learning with Im-
plicit Gradients. In: Advances in Neural Information Processing Systems 32,
Curran Associates Inc., NIPS’19, pp 113–124
Ravi S, Larochelle H (2017) Optimization as a Model for Few-Shot Learning.
In: International Conference on Learning Representations, ICLR’17
Ren M, Triantaﬁllou E, Ravi S, Snell J, Swersky K, Tenenbaum JB, Larochelle
H, Zemel RS (2018) Meta-Learning for Semi-Supervised Few-Shot Classiﬁ-
cation. In: International Conference on Learning Representations, ICLR’18
Rusu AA, Rao D, Sygnowski J, Vinyals O, Pascanu R, Osindero S, Hadsell
R (2018) Meta-Learning with Latent Embedding Optimization. In: Interna-
tional Conference on Learning Representations, ICLR’18
Santoro A, Bartunov S, Botvinick M, Wierstra D, Lillicrap T (2016) Meta-
learning with Memory-augmented Neural Networks. In: Proceedings of the
33rd International Conference on International Conference on Machine
Learning, JMLR.org, ICML’16, pp 1842–1850
Schmidhuber J (1987) Evolutionary principles in self-referential learning.
Diploma Thesis, Technische Universität München
Schmidhuber J (1993) A neural network that embeds its own meta-levels. In:
IEEE International Conference on Neural Networks, IEEE, pp 407–412
Schmidhuber J, Zhao J, Wiering M (1997) Shifting Inductive Bias with
Success-Story Algorithm, Adaptive Levin Search, and Incremental Self-
Improvement. Machine Learning 28(1):105–130
Shyam P, Gupta S, Dukkipati A (2017) Attentive Recurrent Comparators.
In: Proceedings of the 34th International Conference on Machine Learning,
JLMR.org, ICML’17, pp 3173–3181
Silver D, Huang A, Maddison CJ, Guez A, Sifre L, van den Driessche G,
Schrittwieser J, Antonoglou I, Panneershelvam V, Lanctot M, Dieleman
S, Grewe D, Nham J, Kalchbrenner N, Sutskever I, Lillicrap T, Leach M,
Kavukcuoglu K, Graepel T, Hassabis D (2016) Mastering the game of Go
with deep neural networks and tree search. Nature 529(7587):484
Snell J, Swersky K, Zemel R (2017) Prototypical Networks for Few-shot Learn-
ing. In: Advances in Neural Information Processing Systems 30, Curran As-
sociates Inc., NIPS’17, pp 4077–4087
Sun C, Shrivastava A, Singh S, Gupta A (2017) Revisiting Unreasonable Ef-
fectiveness of Data in Deep Learning Era. In: Proceedings of the IEEE In-

--- Page 67 ---
A Survey of Deep Meta-Learning
67
ternational Conference on Computer Vision, pp 843–852
Sung F, Yang Y, Zhang L, Xiang T, Torr PH, Hospedales TM (2018) Learning
to Compare: Relation Network for Few-Shot Learning. In: Proceedings of
the IEEE Conference on Computer Vision and Pattern Recognition, IEEE,
pp 1199–1208
Sutton RS, Barto AG (2018) Reinforcement Learning: An Introduction, 2nd
edn. MIT press
Taylor ME, Stone P (2009) Transfer Learning for Reinforcement Learning
Domains: A Survey. Journal of Machine Learning Research 10(7)
Thrun S (1998) Lifelong Learning Algorithms. In: Learning to learn, Springer,
pp 181–209
Tokmakov P, Wang YX, Hebert M (2019) Learning Compositional Represen-
tations for Few-Shot Recognition. In: Proceedings of the IEEE International
Conference on Computer Vision, pp 6372–6381
Triantaﬁllou E, Zhu T, Dumoulin V, Lamblin P, Evci U, Xu K, Goroshin R,
Gelada C, Swersky K, Manzagol PA, Larochelle H (2020) Meta-Dataset: A
Dataset of Datasets for Learning to Learn from Few Examples. In: Interna-
tional Conference on Learning Representations, ICLR’20
Vanschoren
J
(2018)
Meta-Learning:
A
Survey.
arXiv
preprint
arXiv:181003548
Vanschoren J, van Rijn JN, Bischl B, Torgo L (2014) OpenML: Networked
Science in Machine Learning. SIGKDD Explorations 15(2):49–60
Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser
Ł, Polosukhin I (2017) Attention Is All You Need. In: Advances in Neural
Information Processing Systems 30, Curran Associates Inc., NIPS’17, pp
5998–6008
Vinyals O (2017) Talk: Model vs optimization meta learning. http:
//metalearning-symposium.ml/files/vinyals.pdf, neural Information
Processing Systems (NIPS’17); accessed 06-06-2020
Vinyals O, Blundell C, Lillicrap T, Kavukcuoglu K, Wierstra D (2016) Match-
ing Networks for One Shot Learning. In: Advances in Neural Information
Processing Systems 29, Curran Associates Inc., NIPS’16, pp 3637–3645
Vuorio R, Cho DY, Kim D, Kim J (2018) Meta Continual Learning. arXiv
preprint arXiv:180606928
Wah C, Branson S, Welinder P, Perona P, Belongie S (2011) The Caltech-
UCSD Birds-200-2011 Dataset. Tech. Rep. CNS-TR-2011-001, California
Institute of Technology
Wang JX, Kurth-Nelson Z, Tirumala D, Soyer H, Leibo JZ, Munos R, Blundell
C, Kumaran D, Botvinick M (2016) Learning to reinforcement learn. arXiv
preprint arXiv:161105763
Wu Y, Schuster M, Chen Z, Le QV, Norouzi M, Macherey W, Krikun M, Cao
Y, Gao Q, Macherey K, Klingner J, Shah A, Johnson M, Liu X, Łukasz
Kaiser, Gouws S, Kato Y, Kudo T, Kazawa H, Stevens K, Kurian G, Patil
N, Wang W, Young C, Smith J, Riesa J, Rudnick A, Vinyals O, Corrado
G, Hughes M, Dean J (2016) Google’s Neural Machine Translation System:
Bridging the Gap between Human and Machine Translation. arXiv preprint

--- Page 68 ---
68
Mike Huisman et al.
arXiv:160908144
Yin M, Tucker G, Zhou M, Levine S, Finn C (2020) Meta-Learning without
Memorization. In: International Conference on Learning Representations,
ICLR’20
Yoon J, Kim T, Dia O, Kim S, Bengio Y, Ahn S (2018) Bayesian Model-
Agnostic Meta-Learning. In: Advances in Neural Information Processing
Systems 31, Curran Associates Inc., NIPS’18, pp 7332–7342
Younger AS, Hochreiter S, Conwell PR (2001) Meta-learning with backprop-
agation. In: International Joint Conference on Neural Networks, IEEE,
IJCNN’01, vol 3
Yu T, Quillen D, He Z, Julian R, Hausman K, Finn C, Levine S (2019) Meta-
World: A Benchmark and Evaluation for Multi-Task and Meta Reinforce-
ment Learning. arXiv preprint arXiv:191010897
