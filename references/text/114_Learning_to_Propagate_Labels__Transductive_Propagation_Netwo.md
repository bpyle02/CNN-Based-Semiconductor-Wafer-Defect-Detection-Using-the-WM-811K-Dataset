# Learning to Propagate Labels: Transductive Propagation Network

**Authors**: Liu, Lee, Park, Kim, Yang, Hwang
**Year**: 2019
**arXiv**: 1805.10002
**Topic**: fewshot
**Relevance**: Label propagation for semi-supervised wafer map classification

---


--- Page 1 ---
Published as a conference paper at ICLR 2019
LEARNING TO PROPAGATE LABELS: TRANSDUCTIVE
PROPAGATION NETWORK FOR FEW-SHOT LEARNING
Yanbin Liu1∗, Juho Lee2,3, Minseop Park3, Saehoon Kim3, Eunho Yang3,4,
Sung Ju Hwang3,4 & Yi Yang1,5†
1CAI, University of Technology Sydney, 2University of Oxford
3AITRICS, 4KAIST, 5Baidu Research
csyanbin@gmail.com, juho.lee@stats.ox.ac.uk,
{mike_seop, shkim}@aitrics.com, {eunhoy, sjhwang82}@kaist.ac.kr,
Yi.Yang@uts.edu.au
ABSTRACT
The goal of few-shot learning is to learn a classiﬁer that generalizes well even
when trained with a limited number of training instances per class. The recently
introduced meta-learning approaches tackle this problem by learning a generic
classiﬁer across a large number of multiclass classiﬁcation tasks and generalizing
the model to a new task. Yet, even with such meta-learning, the low-data problem
in the novel classiﬁcation task still remains. In this paper, we propose Transductive
Propagation Network (TPN), a novel meta-learning framework for transductive
inference that classiﬁes the entire test set at once to alleviate the low-data problem.
Speciﬁcally, we propose to learn to propagate labels from labeled instances to
unlabeled test instances, by learning a graph construction module that exploits the
manifold structure in the data. TPN jointly learns both the parameters of feature
embedding and the graph construction in an end-to-end manner. We validate TPN
on multiple benchmark datasets, on which it largely outperforms existing few-shot
learning approaches and achieves the state-of-the-art results.
1
INTRODUCTION
Recent breakthroughs in deep learning (Krizhevsky et al., 2012; Simonyan and Zisserman, 2015; He
et al., 2016) highly rely on the availability of large amounts of labeled data. However, this reliance
on large data increases the burden of data collection, which hinders its potential applications to the
low-data regime where the labeled data is rare and difﬁcult to gather. On the contrary, humans have
the ability to recognize new objects after observing only one or few instances (Lake et al., 2011).
For example, children can generalize the concept of “apple” after given a single instance of it. This
signiﬁcant gap between human and deep learning has reawakened the research interest on few-shot
learning (Vinyals et al., 2016; Snell et al., 2017; Finn et al., 2017; Ravi and Larochelle, 2017; Lee
and Choi, 2018; Xu et al., 2017; Wang et al., 2018).
Few-shot learning aims to learn a classiﬁer that generalizes well with a few examples of each of
these classes. Traditional techniques such as ﬁne-tuning (Jia et al., 2014) that work well with deep
learning models would severely overﬁt on this task (Vinyals et al., 2016; Finn et al., 2017), since a
single or only a few labeled instances would not accurately represent the true data distribution and
will result in learning classiﬁers with high variance, which will not generalize well to new data.
In order to solve this overﬁtting problem, Vinyals et al. (2016) proposed a meta-learning strat-
egy which learns over diverse classiﬁcation tasks over large number of episodes rather than only
on the target classiﬁcation task. In each episode, the algorithm learns the embedding of the few
labeled examples (the support set), which can be used to predict classes for the unlabeled points
(the query set) by distance in the embedding space. The purpose of episodic training is to mimic
∗This work was done when Yanbin Liu was an intern at AITRICS.
†Part of this work was done when Yi Yang was visiting Baidu Research during his Professional Experience
Program.
1
arXiv:1805.10002v5  [cs.LG]  8 Feb 2019

--- Page 2 ---
Published as a conference paper at ICLR 2019
...
Transductive Propagation Network
Task 1
...
Task 2
Test Task
unlabeled
labeled
Meta-train
Meta-test
!
!
Figure 1: A conceptual illustration of our transductive meta-learning framework, where lines between nodes
represent graph connections and their colors represent the potential direction of label propagation. The neigh-
borhood graph is episodic-wisely trained for transductive inference.
the real test environment containing few-shot support set and unlabeled query set. The consistency
between training and test environment alleviates the distribution gap and improves generalization.
This episodic meta-learning strategy, due to its generalization performance, has been adapted by
many follow-up work on few-shot learning. Finn et al. (2017) learned a good initialization that can
adapt quickly to the target tasks. Snell et al. (2017) used episodes to train a good representation and
predict classes by computing Euclidean distance with respect to class prototypes.
Although episodic strategy is an effective approach for few-shot learning as it aims at generalizing
to unseen classiﬁcation tasks, the fundamental difﬁculty with learning with scarce data remains for
a novel classiﬁcation task. One way to achieve larger improvements with limited amount of training
data is to consider relationships between instances in the test set and thus predicting them as a whole,
which is referred to as transduction, or transductive inference. In previous work (Joachims, 1999;
Zhou et al., 2004; Vapnik, 1999), transductive inference has shown to outperform inductive methods
which predict test examples one by one, especially in small training sets. One popular approach for
transduction is to construct a network on both the labeled and unlabeled data, and propagate labels
between them for joint prediction. However, the main challenge with such label propagation (and
transduction) is that the label propagation network is often obtained without consideration of the
main task, since it is not possible to learn them at the test time.
Yet, with the meta-learning by episodic training, we can learn the label propagation network as the
query examples sampled from the training set can be used to simulate the real test set for transductive
inference. Motivated by this ﬁnding, we propose Transductive Propagation Network (TPN) to deal
with the low-data problem. Instead of applying the inductive inference, we utilize the entire query
set for transductive inference (see Figure 1). Speciﬁcally, we ﬁrst map the input to an embedding
space using a deep neural network. Then a graph construction module is proposed to exploit the
manifold structure of the novel class space using the union of support set and query set. According
to the graph structure, iterative label propagation is applied to propagate labels from the support
set to the query set and ﬁnally leads to a closed-form solution. With the propagated scores and
ground truth labels of the query set, we compute the cross-entropy loss with respect to the feature
embedding and graph construction parameters. Finally, all parameters can be updated end-to-end
using backpropagation.
The main contribution of this work is threefold.
• To the best of our knowledge, we are the ﬁrst to model transductive inference explicitly
in few-shot learning. Although Nichol et al. (2018) experimented with a transductive set-
ting, they only share information between test examples by batch normalization rather than
directly proposing a transductive model.
• In transductive inference, we propose to learn to propagate labels between data instances
for unseen classes via episodic meta-learning. This learned label propagation graph is
2

--- Page 3 ---
Published as a conference paper at ICLR 2019
shown to signiﬁcantly outperform naive heuristic-based label propagation methods (Zhou
et al., 2004).
• We evaluate our approach on two benchmark datasets for few-shot learning, namely
miniImageNet and tieredImageNet. The experimental results show that our Transductive
Propagation Network outperforms the state-of-the-art methods on both datasets. Also, with
semi-supervised learning, our algorithm achieves even higher performance, outperforming
all semi-supervised few-shot learning baselines.
2
RELATED WORK
Meta-learning
In
recent
works,
few-shot
learning
often
follows
the
idea
of
meta-
learning (Schmidhuber, 1987; Thrun and Pratt, 2012). Meta-learning tries to optimize over batches
of tasks rather than batches of data points. Each task corresponds to a learning problem, obtaining
good performance on these tasks helps to learn quickly and generalize well to the target few-shot
problem without suffering from overﬁtting. The well-known MAML approach (Finn et al., 2017)
aims to ﬁnd more transferable representations with sensitive parameters. A ﬁrst-order meta-learning
approach named Reptile is proposed by Nichol et al. (2018). It is closely related to ﬁrst-order
MAML but does not need a training-test split for each task. Compared with the above methods,
our algorithm has a closed-form solution for label propagation on the query points, thus avoiding
gradient computation in the inner updateand usually performs more efﬁciently.
Embedding and metric learning approaches
Another category of few-shot learning approach
aims to optimize the transferable embedding using metric learning approaches.
Matching net-
works (Vinyals et al., 2016) produce a weighted nearest neighbor classiﬁer given the support set
and adjust feature embedding according to the performance on the query set. Prototypical net-
works (Snell et al., 2017) ﬁrst compute a class’s prototype to be the mean of its support set in the
embedding space. Then the transferability of feature embedding is evaluated by ﬁnding the near-
est class prototype for embedded query points. An extension of prototypical networks is proposed
in Ren et al. (2018) to deal with semi-supervised few-shot learning. Relation Network (Sung et al.,
2018) learns to learn a deep distance metric to compare a small number of images within episodes.
Our proposed method is similar to these approaches in the sense that we all focus on learning deep
embeddings with good generalization ability. However, our algorithm assumes a transductive set-
ting, in which we utilize the union of support set and query set to exploit the manifold structure of
novel class space by using episodic-wise parameters.
Transduction
The setting of transductive inference was ﬁrst introduced by Vapnik (Vapnik, 1999).
Transductive Support Vector Machines (TSVMs) (Joachims, 1999) is a margin-based classiﬁcation
method that minimizes errors of a particular test set. It shows substantial improvements over induc-
tive methods, especially for small training sets. Another category of transduction methods involves
graph-based methods (Zhou et al., 2004; Wang and Zhang, 2006; Rohrbach et al., 2013; Fu et al.,
2015). Label propagation is used in Zhou et al. (2004) to transfer labels from labeled to unlabeled
data instances guided by the weighted graph. Label propagation is sensitive to variance parameter
σ, so Linear Neighborhood Propagation (LNP) (Wang and Zhang, 2006) constructs approximated
Laplacian matrix to avoid this issue. In Zhu and Ghahramani (2002), minimum spanning tree heuris-
tic and entropy minimization are used to learn the parameter σ. In all these prior work, the graph
construction is done on a pre-deﬁned feature space using manually selected hyperparamters since
it is not possible to learn them at test time. Our approach, on the other hand, is able to learn the
graph construction network since it is a meta-learning framework with episodic training, where at
each episode we simulate the test set with a subset of the training set.
In few-shot learning, Nichol et al. (2018) experiments with a transductive setting and shows im-
provements. However, they only share information between test examples via batch normaliza-
tion (Ioffe and Szegedy, 2015) rather than explicitly model the transductive setting as in our algo-
rithm.
3

--- Page 4 ---
Published as a conference paper at ICLR 2019
CNN 
CNN 
Support
Query
f'
f'(X)
σ
y
gφ
Wij = exp
!
−1
2d(f'(xi)
σi
, f'(xj)
σj
)
"
Query
Label
LOSS
!
!
!
Graph Construction
Feature Embedding
Label Propagation
Loss
!
Figure 2: The overall framework of our algorithm in which the manifold structure of the entire query set helps to
learn better decision boundary. The proposed algorithm is composed of four components: feature embedding,
graph construction, label propagation, and loss generation.
3
MAIN APPROACH
In this section, we introduce the proposed algorithm that utilizes the manifold structure of the given
few-shot classiﬁcation task to improve the performance.
3.1
PROBLEM DEFINITION
We follow the episodic paradigm (Vinyals et al., 2016) that effectively trains a meta-learner for few-
shot classiﬁcation tasks, which is commonly employed in various literature (Snell et al., 2017; Finn
et al., 2017; Nichol et al., 2018; Sung et al., 2018; Mishra et al., 2018). Given a relatively large
labeled dataset with a set of classes Ctrain, the objective of this setting is to train classiﬁers for an
unseen set of novel classes Ctest, for which only a few labeled examples are available.
Speciﬁcally, in each episode, a small subset of N classes are sampled from Ctrain to construct a
support set and a query set. The support set contains K examples from each of the N classes (i.e.,
N-way K-shot setting) denoted as S = {(x1, y1), (x2, y2), . . . , (xN×K, yN×K)}, while the query
set Q = {(x∗
1, y∗
1), (x∗
2, y∗
2), . . . , (x∗
T , y∗
T )} includes different examples from the same N classes.
Here, the support set S in each episode serves as the labeled training set on which the model is
trained to minimize the loss of its predictions for the query set Q. This procedure mimics training
classiﬁers for Ctest and goes episode by episode until convergence.
Meta-learning implemented by the episodic training reasonably performs well to few-shot classi-
ﬁcation tasks. Yet, due to the lack of labeled instances (K is usually very small) in the support
set, we observe that a reliable classiﬁer is still difﬁcult to be obtained. This motivates us to con-
sider a transductive setting that utilizes the whole query set for the prediction rather than predicting
each example independently. Taking the entire query set into account, we can alleviate the low-data
problem and provide more reliable generalization property.
3.2
TRANSDUCTIVE PROPAGATION NETWORK (TPN)
We introduce Transductive Propagation Network (TPN) illustrated in Figure 2, which consists of
four components: feature embedding with a convolutional neural network; graph construction that
produces example-wise parameters to exploit the manifold structure; label propagation that spreads
labels from the support set S to the query set Q; a loss generation step that computes a cross-
entropy loss between propagated labels and the ground-truths on Q to jointly train all parameters in
the framework.
3.2.1
FEATURE EMBEDDING
We employ a convolutional neural network fϕ to extract features of an input xi, where fϕ(xi; ϕ)
refers to the feature map and ϕ indicates a parameter of the network. Despite the generality, we adopt
the same architecture used in several recent works (Snell et al., 2017; Sung et al., 2018; Vinyals et
al., 2016). By doing so, we can provide more fair comparisons in the experiments, highlighting
the effects of transductive approach. The network is made up of four convolutional blocks where
each block begins with a 2D convolutional layer with a 3 × 3 kernel and ﬁlter size of 64. Each
4

--- Page 5 ---
Published as a conference paper at ICLR 2019
convolutional layer is followed by a batch-normalization layer (Ioffe and Szegedy, 2015), a ReLU
nonlinearity and a 2 × 2 max-pooling layer. We use the same embedding function fϕ for both the
support set S and the query set Q.
3.2.2
GRAPH CONSTRUCTION
Manifold learning (Chung and Graham, 1997; Zhou et al., 2004; Yang et al., 2016) discovers the
embedded low-dimensional subspace in the data, where it is critical to choose an appropriate neigh-
borhood graph. A common choice is Gaussian similarity function:
Wij = exp

−d(xi, xj)
2σ2

,
(1)
where d(·, ·) is a distance measure (e.g., Euclidean distance) and σ is the length scale parameter.
The neighborhood structure behaves differently with respect to various σ, which means that it needs
to carefully select the optimal σ for the best performance of label propagation (Wang and Zhang,
2006; Zhu and Ghahramani, 2002). In addition, we observe that there is no principled way to tune the
scale parameter in meta-learning framework, though there exist some heuristics for dimensionalty
reduction methods (Zelnik-Manor and Perona, 2004; Sugiyama, 2007).
Example-wise length-scale parameter
To obtain a proper neighborhood graph in meta-learning,
we propose a graph construction module built on the union set of support set and query set: S ∪Q.
This module is composed of a convolutional neural network gφ which takes the feature map fϕ(xi)
for xi ∈S ∪Q to produce an example-wise length-scale parameter σi = gφ(fϕ(xi)). Note that the
scale parameter is determined example-wisely and learned in an episodic training procedure, which
adapts well to different tasks and makes it suitable for few-shot learning. With the example-wise σi,
our similarity function is then deﬁned as follows:
Wij = exp

−1
2d
fϕ(xi)
σi
, fϕ(xj)
σj

(2)
where W ∈R(N×K+T )×(N×K+T ) for all instances in S ∪Q. We only keep the k-max values
in each row of W to construct a k-nearest neighbour graph. Then we apply the normalized graph
Laplacians (Chung and Graham, 1997) on W, that is, S = D−1/2WD−1/2, where D is a diagonal
matrix with its (i, i)-value to be the sum of the i-th row of W.
f'(xi)
f'(xj)
σi
σj
Wij = exp
✓
−1
2d(f'(xi)
σi
, f'(xj)
σj
)
◆
3 ⇥3 conv
BatchNorm
ReLU
2 ⇥2 max-pool
3 ⇥3 conv
BatchNorm
ReLU
2 ⇥2 max-pool
gφ
FC layer 1
FC layer 2
Figure 3: Detailed architecture of the graph construction module, in which the length-scale parameter is
example-wisely determined.
Graph construction structure
The structure of the proposed graph construction module is shown
in Figure 3. It is composed of two convolutional blocks and two fully-connected layers, where
each block contains a 3-by-3 convolution, batch normalization, ReLU activation, followed by 2-
by-2 max pooling. The number of ﬁlters in each convolutional block is 64 and 1, respectively. To
provide an example-wise scaling parameter, the activation map from the second convolutional block
is transformed into a scalar by two fully-connected layers in which the number of neurons is 8 and
1, respectively.
Graph construction in each episode
We follow the episodic paradigm for few-shot meta-learner
training. This means that the graph is individually constructed for each task in each episode, as
shown in Figure 1. Typically, in 5-way 5-shot training, N = 5, K = 5, T = 75, the dimension of
W is only 100 × 100, which is quite efﬁcient.
5

--- Page 6 ---
Published as a conference paper at ICLR 2019
3.2.3
LABEL PROPAGATION
We now describe how to get predictions for the query set Q using label propagation, before the last
cross-entropy loss step. Let F denote the set of (N × K + T) × N matrix with nonnegative entries.
We deﬁne a label matrix Y ∈F with Yij = 1 if xi is from the support set and labeled as yi = j,
otherwise Yij = 0. Starting from Y , label propagation iteratively determines the unknown labels of
instances in the union set S ∪Q according to the graph structure using the following formulation:
Ft+1 = αSFt + (1 −α)Y ,
(3)
where Ft ∈F denotes the predicted labels at the timestamp t, S denotes the normalized weight, and
α ∈(0, 1) controls the amount of propagated information. It is well known that the sequence {Ft}
has a closed-form solution as follows:
F ∗= (I −αS)−1Y ,
(4)
where I is the identity matrix (Zhou et al., 2004). We directly utilize this result for the label propa-
gation, making a whole episodic meta-learning procedure more efﬁcient in practice.
Time complexity
Matrix inversion originally takes O(n3) time complexity, which is inefﬁcient
for large n. However, in our setting, n = N × K + T (80 for 1-shot and 100 for 5-shot) is very
small. Moreover, there is plenty of prior work on the scalability and efﬁciency of label propagation,
such as Liang and Li (2018); Fujiwara and Irie (2014), which can extend our work to large-scale
data. More discussions are presented in A.4
3.2.4
CLASSIFICATION LOSS GENERATION
The objective of this step is to compute the classiﬁcation loss between the predictions of the union
of support and query set via label propagation and the ground-truths. We compute the cross-entropy
loss between predicted scores F ∗and ground-truth labels from S ∪Q to learn all parameters in an
end-to-end fashion, where F ∗is converted to probabilistic score using softmax:
P( ˜yi = j|xi) =
exp(F ∗
ij)
PN
j=1 exp(F ∗
ij)
.
(5)
Here, ˜yi denotes the ﬁnal predicted label for ith instance in the union of support and query set and
F ∗
ij denotes the jth component of predicted label from label propagation. Then the loss function is
computed as:
J(ϕ, φ) =
N×K+T
X
i=1
N
X
j=1
−I(yi == j) log(P( ˜yi = j|xi)) ,
(6)
where yi means the ground-truth label of xi and I(b) is an indicator function, I(b) = 1 if b is true
and 0 otherwise.
Note that in Equation (6), the loss is dependent on two set of parameters ϕ, φ (even though the
dependency is implicit through F ∗
ij). All these parameters are jointly updated by the episodic training
in an end-to-end manner.
4
EXPERIMENTS
We evaluate and compare our TPN with state-of-the-art approaches on two datasets, i.e.,
miniImageNet (Ravi and Larochelle, 2017) and tieredImageNet (Ren et al., 2018). The former
is the most popular few-shot learning benchmark and the latter is a much larger dataset released
recently for few-shot learning.
4.1
DATASETS
miniImageNet. The miniImageNet dataset is a collection of Imagenet (Krizhevsky et al., 2012) for
few-shot image recognition. It is composed of 100 classes randomly selected from Imagenet with
each class containing 600 examples. In order to directly compare with state-of-the-art algorithms for
6

--- Page 7 ---
Published as a conference paper at ICLR 2019
few-shot learning, we rely on the class splits used by Ravi and Larochelle (2017), which includes
64 classes for training, 16 for validation, and 20 for test. All images are resized to 84 × 84 pixels.
tieredImageNet. Similar to miniImageNet , tieredImageNet (Ren et al., 2018) is also a subset of
Imagenet (Krizhevsky et al., 2012), but it has a larger number of classes from ILSVRC-12 (608
classes rather than 100 for miniImageNet). Different from miniImageNet, it has a hierarchical struc-
ture of broader categories corresponding to high-level nodes in Imagenet. The top hierarchy has
34 categories, which are divided into 20 training (351 classes), 6 validation (97 classes) and 8 test
(160 classes) categories. The average number of examples in each class is 1281. This high-level
split strategy ensures that the training classes are distinct from the test classes semantically. This is
a more challenging and realistic few-shot setting since there is no assumption that training classes
should be similar to test classes. Similarly, all images are resized to 84 × 84 pixels.
4.2
EXPERIMENTAL SETUP
For fair comparison with other methods, we adopt a widely-used CNN (Finn et al., 2017; Snell et
al., 2017) as the feature embedding function fϕ (Section 3.2.1). The hyper-parameter k of k-nearest
neighbour graph (Section 3.2.2) is set to 20 and α of label propagation is set to 0.99, as suggested in
Zhou et al. (2004).
Following Snell et al. (2017), we adopt the episodic training procedure, i.e, we sample a set of
N-way K-shot training tasks to mimic the N-way K-shot test problems. Moreover, Snell et al.
(2017) proposed a “Higher Way ” training strategy which used more training classes in each episode
than test case. However, we ﬁnd that it is beneﬁcial to train with more examples than test phase
(Appendix A.1). This is denoted as “Higher Shot” in our experiments. For 1-shot and 5-shot test
problem, we adopt 5-shot and 10-shot training respectively. In all settings, the query number is set
to 15 and the performance are averaged over 600 randomly generated episodes from the test set.
All our models were trained with Adam (Kingma and Ba, 2015) and an initial learning rate
of 10−3.
For miniImageNet, we cut the learning rate in half every 10, 000 episodes and for
tieredImageNet, we cut the learning rate every 25, 000 episodes. The reason for larger decay step is
that tieredImageNet has more classes and more examples in each class which needs larger training
iterations. We ran the training process until the validation loss reached a plateau.
4.3
FEW-SHOT LEARNING RESULTS
We compare our method with several state-of-the-art approaches in various settings. Even though
the transductive method has never been used explicitly, batch normalization layer was used transduc-
tively to share information between test examples. For example, in Finn et al. (2017); Nichol et al.
(2018), they use the query batch statistics rather than global BN parameters for the prediction, which
leads to performance gain in the query set. Besides, we propose two simple transductive methods
as baselines that explicitly utilize the query set. First, we propose the MAML+Transduction with
slight modiﬁcation of loss function to: J (θ) = PT
i=1 yi log P(byi|xi) + PN×K+T
i,j=1
Wij∥byi −byj∥2
2
for transductive inference. The additional term serves as transductive regularization. Second, the
naive heuristic-based label propagation methods (Zhou et al., 2004) is proposed to explicitly model
the transductive inference.
Experimental results are shown in Table 1 and Table2. Transductive batch normalization methods
tend to perform better than pure inductive methods except for the “Higher Way” PROTO NET. Label
propagation without learning to propagate outperforms other baseline methods in most cases, which
veriﬁes the necessity of transduction. The proposed TPN achieves the state-of-the-art results and
surpasses all the others with a large margin even when the model is trained with regular shots. When
“Higher Shot” is applied, the performance of TPN continues to improve especially for 1-shot case.
This conﬁrms that our model effectively ﬁnds the episodic-wise manifold structure of test examples
through learning to construct the graph for label propagation.
Another observation is that the advantages of 5-shot classiﬁcation is less signiﬁcant than that of 1-
shot case. For example, in 5-way miniImageNet , the absolute improvement of TPN over published
state-of-the-art is 4.13% for 1-shot and 1.66% for 5-shot. To further investigate this, we experi-
mented 5-way k-shot (k = 1, 2, · · · , 10) experiments. The results are shown in Figure 4. Our TPN
performs consistently better than other methods with varying shots. Moreover, it can be seen that
7

--- Page 8 ---
Published as a conference paper at ICLR 2019
Table 1: Few-shot classiﬁcation accuracies on miniImageNet. All results are averaged over 600 test episodes.
Top results are highlighted.
5-way Acc
10-way Acc
Model
Transduction
1-shot
5-shot
1-shot
5-shot
MAML (Finn et al., 2017)
BN
48.70
63.11
31.27
46.92
MAML+Transduction
Yes
50.83
66.19
31.83
48.23
Reptile (Nichol et al., 2018)
No
47.07
62.74
31.10
44.66
Reptile + BN (Nichol et al., 2018)
BN
49.97
65.99
32.00
47.60
PROTO NET (Snell et al., 2017)
No
46.14
65.77
32.88
49.29
PROTO NET (Higher Way) (Snell et al., 2017)
No
49.42
68.20
34.61
50.09
RELATION NET (Sung et al., 2018)
BN
51.38
67.07
34.86
47.94
Label Propagation
Yes
52.31
68.18
35.23
51.24
TPN
Yes
53.75
69.43
36.62
52.32
TPN (Higher Shot)
Yes
55.51
69.86
38.44
52.77
* “Higher Way” means using more classes in training episodes. “Higher Shot” means using more shots
in training episodes. “BN” means information is shared among test examples using batch normalization.
† Due to space limitation, we report the accuracy with 95% conﬁdence intervals in Appendix.
Table 2: Few-shot classiﬁcation accuracies on tieredImageNet.
All results are averaged over 600 test
episodes. Top results are highlighted.
5-way Acc
10-way Acc
Model
Transduction
1-shot
5-shot
1-shot
5-shot
MAML (Finn et al., 2017)
BN
51.67
70.30
34.44
53.32
MAML + Transduction
Yes
53.23
70.83
34.78
54.67
Reptile (Nichol et al., 2018)
No
48.97
66.47
33.67
48.04
Reptile + BN (Nichol et al., 2018)
BN
52.36
71.03
35.32
51.98
PROTO NET (Snell et al., 2017)
No
48.58
69.57
37.35
57.83
PROTO NET (Higher Way) (Snell et al., 2017)
No
53.31
72.69
38.62
58.32
RELATION NET (Sung et al., 2018)
BN
54.48
71.31
36.32
58.05
Label Propagation
Yes
55.23
70.43
39.39
57.89
TPN
Yes
57.53
72.85
40.93
59.17
TPN (Higher Shot)
Yes
59.91
73.30
44.80
59.44
* “Higher Way” means using more classes in training episodes. “Higher Shot” means using more shots
in training episodes. “BN” means information is shared among test examples using batch normalization.
† Due to space limitation, we report the accuracy with 95% conﬁdence intervals in Appendix.
TPN outperforms other methods with a large margin in lower shots. With the shot increase, the
advantage of transduction narrows since more labelled data are used. This ﬁnding agrees with the
results in TSVM (Joachims, 1999): when more training data are available, the bonus of transductive
inference will be decreased.
4.4
COMPARISON WITH SEMI-SUPERVISED FEW-SHOT LEARNING
Table 3: Semi-supervised comparison on miniImageNet.
Model
1-shot
5-shot
1-shot w/D
5-shot w/D
Soft k-Means (Ren et al., 2018)
50.09
64.59
48.70
63.55
Soft k-Means+Cluster (Ren et al., 2018)
49.03
63.08
48.86
61.27
Masked Soft k-Means (Ren et al., 2018)
50.41
64.39
49.04
62.96
TPN-semi
52.78
66.42
50.43
64.95
* “w/D” means with distraction. In this setting, many of the unlabelled data are from the
so-called distraction classes , which is different from the classes of labelled data.
† Due to space limitation, we report the accuracy with 95% conﬁdence intervals in
Appendix.
The main difference of traditional semi-supervised learning and transduction is the source of un-
labeled data. Transductive methods directly use test set as unlabeled data while semi-supervised
learning usually has an extra unlabeled set. In order to compare with semi-supervised methods,
8

--- Page 9 ---
Published as a conference paper at ICLR 2019
Figure 4: 5-way performance with various training/test shots.
Table 4: Semi-supervised comparison on tieredImageNet.
Model
1-shot
5-shot
1-shot w/D
5-shot w/D
Soft k-Means (Ren et al., 2018)
51.52
70.25
49.88
68.32
Soft k-Means+Cluster (Ren et al., 2018)
51.85
69.42
51.36
67.56
Masked Soft k-Means (Ren et al., 2018)
52.39
69.88
51.38
69.08
TPN-semi
55.74
71.01
53.45
69.93
* “w/D” means with distraction. In this setting, many of the unlabelled data are from the
so-called distraction classes , which is different from the classes of labelled data.
† Due to space limitation, we report the accuracy with 95% conﬁdence intervals in
Appendix.
we propose a semi-supervised version of TPN, named TPN-semi, which classiﬁes one test example
each time by propagating labels from the labeled set and extra unlabeled set.
We use miniImageNet and tieredImageNet with the labeled/unlabeled data split proposed by Ren
et al. (2018). Speciﬁcally, they split the images of each class into disjoint labeled and unlabeled
sets. For miniImageNet, the ratio of labeled/unlabeled data is 40% and 60% in each class. Likewise,
the ratio is 10% and 90% for tieredImageNet. All semi-supervised methods (including TPN-semi)
sample support/query data from the labeled set (e.g, 40% from miniImageNet) and sample unlabeled
data from the unlabeled sets (e.g, 60% from miniImageNet). In addition, there is a more challenging
situation where many unlabelled examples from other distractor classes (different from labelled
classes).
Following Ren et al. (2018), we report the average accuracy over 10 random labeled/unlabeled splits
and the uncertainty computed in standard error. Results are shown in Table 3 and Table 4. It can
be seen that TPN-semi outperforms all other algorithms with a large margin, especially for 1-shot
case. Although TPN is originally designed to perform transductive inference, we show that it can
be successfully adapted to semi-supervised learning tasks with little modiﬁcation. In certain cases
where we can not get all test data, the TPN-semi can be used as an effective alternative algorithm.
5
CONCLUSION
In this work, we proposed the transductive setting for few-shot learning. Our proposed approach,
namely Transductive Propagation Network (TPN), utilizes the entire test set for transductive infer-
ence. Speciﬁcally, our approach is composed of four steps: feature embedding, graph construction,
label propagation, and loss computation. Graph construction is a key step that produces example-
wise parameters to exploit the manifold structure in each episode. In our method, all parameters
are learned end-to-end using cross-entropy loss with respect to the ground truth labels and the
prediction scores in the query set. We obtained the state-of-the-art results on miniImageNet and
tieredImageNet. Also, the semi-supervised adaptation of our algorithm achieved higher results than
other semi-supervised methods. In future work, we are going to explore the episodic-wise distance
metric rather than only using example-wise parameters for the Euclidean distance.
9

--- Page 10 ---
Published as a conference paper at ICLR 2019
ACKNOWLEDGMENTS
Saehoon Kim, Minseop Park, and Eunho Yang were supported by Samsung Research Funding &
Incubation Center of Samsung Electronics under Project Number SRFC-IT1702-15. Yanbin Liu and
Yi Yang are in part supported by AWS Cloud Credits for Research.
REFERENCES
Fan RK Chung and Fan Chung Graham. Spectral graph theory. American Mathematical Soc., 1997.
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep
networks. In International Conference on Machine Learning, pages 1126–1135, 2017.
Yanwei Fu, Timothy M Hospedales, Tao Xiang, and Shaogang Gong. Transductive multi-view zero-shot learn-
ing. IEEE transactions on pattern analysis and machine intelligence, 37(11):2332–2345, 2015.
Yasuhiro Fujiwara and Go Irie. Efﬁcient label propagation. In International Conference on Machine Learning,
pages 784–792, 2014.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In
Computer Vision and Pattern Recognition, pages 770–778, 2016.
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing
internal covariate shift. In International Conference on Machine Learning, pages 448–456, 2015.
Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadar-
rama, and Trevor Darrell. Caffe: Convolutional architecture for fast feature embedding. In ACM Interna-
tional Conference on Multimedia, pages 675–678. ACM, 2014.
Thorsten Joachims. Transductive inference for text classiﬁcation using support vector machines. In Interna-
tional Conference on Machine Learning, volume 99, pages 200–209, 1999.
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference
on Learning Representations (ICLR), volume 5, 2015.
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classiﬁcation with deep convolutional neural
networks. In Advances in Neural Information Processing Systems, pages 1097–1105, 2012.
Brenden Lake, Ruslan Salakhutdinov, Jason Gross, and Joshua Tenenbaum. One shot learning of simple visual
concepts. In Conference of the Cognitive Science Society, volume 33, 2011.
Yoonho Lee and Seungjin Choi. Gradient-based meta-learning with learned layerwise metric and subspace. In
International Conference on Machine Learning, pages 2933–2942, 2018.
De-Ming Liang and Yu-Feng Li. Lightweight label propagation for large-scale network data. In IJCAI, pages
3421–3427, 2018.
Bauerïij ˇN Matthias, Rojas-Carulla Mateo, Jakub Bartłomiej ´Swi ˛atkowski, Bernhard Schölkopf, and Richard E
Turner. Discriminative k-shot learning using probabilistic models. arXiv preprint arXiv:1706.00326, 2017.
Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, and Pieter Abbeel. A simple neural attentive meta-learner. In
International Conference on Learning Representations, 2018.
Tsendsuren Munkhdalai, Xingdi Yuan, Soroush Mehri, and Adam Trischler. Rapid adaptation with condition-
ally shifted neurons. In International Conference on Machine Learning, pages 3661–3670, 2018.
Alex Nichol, Joshua Achiam, and John Schulman. On ﬁrst-order meta-learning algorithms. arXiv preprint
arXiv:1803.02999, 2018.
Boris N Oreshkin, Alexandre Lacoste, and Pau Rodriguez. Tadam: Task dependent adaptive metric for im-
proved few-shot learning. In Advances in Neural Information Processing Systems, 2018.
Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. International Conference
on Learning Representations, 2017.
Mengye Ren, Eleni Triantaﬁllou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B Tenenbaum, Hugo
Larochelle, and Richard S Zemel. Meta-learning for semi-supervised few-shot classiﬁcation. International
Conference on Learning Representations, 2018.
10

--- Page 11 ---
Published as a conference paper at ICLR 2019
Marcus Rohrbach, Sandra Ebert, and Bernt Schiele. Transfer learning in a transductive setting. In Advances in
Neural Information Processing Systems, pages 46–54, 2013.
Jürgen Schmidhuber. Evolutionary principles in self-referential learning, or on learning how to learn: the
meta-meta-... hook. PhD thesis, Technische Universität München, 1987.
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition.
International Conference on Learning Representations, 2015.
Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In Advances in
Neural Information Processing Systems, pages 4080–4090, 2017.
Masashi Sugiyama. Dimensionality reduction of multimodal labeled data by local ﬁsher discriminant analysis.
Journal of Machine Learning Research, 8:1027–1061, 2007.
Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M Hospedales. Learning to
compare: Relation network for few-shot learning. In Computer Vision and Pattern Recognition, 2018.
Sebastian Thrun and Lorien Pratt. Learning to learn. Springer Science & Business Media, 2012.
Vladimir Naumovich Vapnik. An overview of statistical learning theory. IEEE transactions on neural networks,
10(5):988–999, 1999.
Oriol Vinyals, Charles Blundell, Tim Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning.
In Advances in Neural Information Processing Systems, pages 3630–3638, 2016.
Fei Wang and Changshui Zhang. Label propagation through linear neighborhoods. In International Conference
on Machine Learning, pages 985–992. ACM, 2006.
Yu-Xiong Wang, Ross Girshick, Martial Hebert, and Bharath Hariharan. Low-shot learning from imaginary
data. In Computer Vision and Pattern Recognition, 2018.
Zhongwen Xu, Linchao Zhu, and Yi Yang. Few-shot object recognition from machine-labeled web images. In
Computer Vision and Pattern Recognition, 2017.
Zhilin Yang, William Cohen, and Ruslan Salakhudinov. Revisiting semi-supervised learning with graph em-
beddings. In International Conference on Machine Learning, pages 40–48, 2016.
Lihi Zelnik-Manor and Pietro Perona. Self-tuning spectral clustering. In Advances in Neural Information
Processing Systems, 2004.
Denny Zhou, Olivier Bousquet, Thomas N Lal, Jason Weston, and Bernhard Schölkopf. Learning with local
and global consistency. In Advances in Neural Information Processing Systems, pages 321–328, 2004.
Xiaojin Zhu and Zoubin Ghahramani. Learning from labeled and unlabeled data with label propagation. Tech.
Rep., Technical Report CMU-CALD-02–107, Carnegie Mellon University, 2002.
11

--- Page 12 ---
Published as a conference paper at ICLR 2019
A
ABLATION STUDY
In this section, we performed several ablation studies with respect to training shots and query number.
A.1
TRAINING SHOTS
Figure 5: Model performance with different training shots. The x-axis indicates the number of shots in training,
and the y-axis indicates 5-way test accuracy for 1-shot and 5-shot. Error bars indicate 95% conﬁdence intervals
as computed over 600 test episodes.
A.2
QUERY NUMBER
Table 5: Accuracy with various query numbers
miniImageNet 1-shot
5
10
15
20
25
30
Train=15
52.29
52.95
53.75
53.92
54.57
54.47
Test=15
53.53
53.72
53.75
52.79
52.84
52.47
Train=Test
51.94
53.47
53.75
54.00
53.59
53.32
miniImageNet 5-shot
5
10
15
20
25
30
Train=15
66.97
69.30
69.43
69.92
70.54
70.36
Test=15
68.50
68.85
69.43
69.26
69.12
68.89
Train=Test
67.55
69.22
69.43
69.85
70.11
69.94
At ﬁrst, we designed three experiments to study the inﬂuence of the query number in both training and test
phase: (1) ﬁx training query to 15; (2) ﬁx test query to 15; (3) training query equals test query. The results
are shown in Table 5. Some conclusions can be drawn from this experiment: (1) When training query is ﬁxed,
increasing the test query will lead to the performance gain. Moreover, even a small test query (e.g., 5) can
yield good performance; (2) When test query is ﬁxed, the performance is relatively stable with various training
query numbers; (3) If the query number of training matches test, the performance can also be improved with
increasing number.
A.3
RESULTS ON RESNET
In this paper, we use a 4-layer neural network structure as described in Section 3.2.1 to make a fair comparison.
Currently, there are two common network architectures in few-shot learning: 4-layer ConvNets (e.g., Finn et
al. (2017); Snell et al. (2017); Sung et al. (2018)) and 12-layer ResNet (e.g., Mishra et al. (2018); Munkhdalai
et al. (2018); Matthias et al. (2017); Oreshkin et al. (2018)). Our method belongs to the ﬁrst one, which
contains much fewer layers than the ResNet setting. Thus, it is more reasonable to compare algorithms such as
TADAM (Oreshkin et al., 2018) with ResNet version of our method. To make this comparison, we implemented
our algorithm with ResNet architecture on miniImagenet dataset and show the results in Table 6.
It can be seen that we beat TADAM for 1-shot setting. For 5-shot, we outperform all other recent high-
performance methods except for TADAM.
12

--- Page 13 ---
Published as a conference paper at ICLR 2019
Table 6: ResNet results on miniImageNet
Method
1-shot
5-shot
SNAIL (Mishra et al., 2018)
55.71
68.88
adaResNet (Munkhdalai et al., 2018)
56.88
71.94
Discriminative k-shot (Matthias et al., 2017)
56.30
73.90
TADAM (Oreshkin et al., 2018)
58.50
76.70
TPN
59.46
75.65
A.4
CLOSED-FORM SOLUTION VS ITERATIVE UPDATES
There is a potential concern that the closed-form solution of label propagation can not scale to large-scale
matrix. We relieve this concern from two aspects. On one hand, the few-shot learning problem assumes that
training examples in each class is quite small (only 1 or 5). In this situation, Eq 3 and the closed-form version
can be efﬁciently solved, since the dimension of S is only 80 × 80 (5-way, 1-shot, 15-query) or 100 × 100
(5-way, 5-shot, 15-query). On the other hand, there are plenty of prior work on the scalability and efﬁciency
of label propagation, such as Liang and Li (2018); Fujiwara and Irie (2014), which can extend our work to
large-scale data.
Furthermore, on miniImagenet, we performed iterative optimization and got 53.05/68.75 for 1-shot/5-shot ex-
periments with only 10 steps. This is slightly worse than closed-form version (53.75/69.43). We attribute this
slightly worse accuracy to the inaccurate computation and unstable gradients caused by multiple step iterations.
A.5
ACCURACY WITH 95% CONFIDENCE INTERVALS
Table 7: Few-shot classiﬁcation accuracies on miniImageNet. All results are averaged over 600 test episodes
and are reported with 95% conﬁdence intervals. Top results are highlighted.
5-way Acc
10-way Acc
Model
Transduction
1-shot
5-shot
1-shot
5-shot
MAML
BN
48.70±1.84
63.11±0.92
31.27±1.15
46.92±1.25
MAML+Transduction
Yes
50.83±1.85
66.19±1.85
31.83±0.45
48.23±1.28
Reptile
No
47.07±0.26
62.74±0.37
31.10±0.28
44.66±0.30
Reptile + BN
BN
49.97±0.32
65.99±0.58
32.00±0.27
47.60±0.32
PROTO NET
No
46.14±0.77
65.77±0.70
32.88±0.47
49.29±0.42
PROTO NET (Higher Way)
No
49.42±0.78
68.20±0.66
34.61±0.46
50.09±0.44
RELATION NET
BN
51.38±0.82
67.07±0.69
34.86±0.48
47.94±0.42
Label Propagation
Yes
52.31±0.85
68.18±0.67
35.23±0.51
51.24±0.43
TPN
Yes
53.75±0.86
69.43±0.67
36.62±0.50
52.32±0.44
TPN (Higher Shot)
Yes
55.51±0.86
69.86±0.65
38.44±0.49
52.77±0.45
* “Higher Way” means using more classes in training episodes. “Higher Shot” means using more shots in
training episodes. “BN” means information is shared among test examples using batch normalization.
13

--- Page 14 ---
Published as a conference paper at ICLR 2019
Table 8: Few-shot classiﬁcation accuracies on tieredImageNet. All results are averaged over 600 test episodes
and are reported with 95% conﬁdence intervals. Top results are highlighted.
5-way Acc
10-way Acc
Model
Transduction
1-shot
5-shot
1-shot
5-shot
MAML
BN
51.67±1.81
70.30±1.75
34.44±1.19
53.32±1.33
MAML + Transduction
Yes
53.23±1.85
70.83±1.78
34.78±1.18
54.67±1.26
Reptile
No
48.97±0.21
66.47±0.21
33.67±0.28
48.04±0.30
Reptile + BN
BN
52.36±0.23
71.03±0.22
35.32±0.28
51.98±0.32
PROTO NET
No
48.58±0.87
69.57±0.75
37.35±0.56
57.83±0.55
PROTO NET (Higher Way)
No
53.31±0.89
72.69±0.74
38.62±0.57
58.32±0.55
RELATION NET
BN
54.48±0.93
71.31±0.78
36.32±0.62
58.05±0.59
Label Propagation
Yes
55.23±0.96
70.43±0.76
39.39±0.60
57.89±0.55
TPN
Yes
57.53±0.96
72.85±0.74
40.93±0.61
59.17±0.52
TPN (Higher Shot)
Yes
59.91±0.94
73.30±0.75
44.80±0.62
59.44±0.51
* “Higher Way” means using more classes in training episodes. “Higher Shot” means using more shots in
training episodes. “BN” means information is shared among test examples using batch normalization.
Table 9: Semi-supervised comparison on miniImageNet.
Model
1-shot
5-shot
1-shot w/D
5-shot w/D
Soft k-Means
50.09±0.45
64.59±0.28
48.70±0.32
63.55±0.28
Soft k-Means+Cluster
49.03±0.24
63.08±0.18
48.86±0.32
61.27±0.24
Masked Soft k-Means
50.41±0.31
64.39±0.24
49.04±0.31
62.96±0.14
TPN-semi
52.78±0.27
66.42±0.21
50.43±0.84
64.95±0.73
* “w/D” means with distraction. In this setting, many of the unlabelled data are from
the so-called distraction classes , which is different from the classes of labelled data.
Table 10: Semi-supervised comparison on tieredImageNet.
Model
1-shot
5-shot
1-shot w/D
5-shot w/D
Soft k-Means
51.52±0.36
70.25±0.31
49.88±0.52
68.32±0.22
Soft k-Means+Cluster
51.85±0.25
69.42±0.17
51.36±0.31
67.56±0.10
Masked Soft k-Means
52.39±0.44
69.88±0.20
51.38±0.38
69.08±0.25
TPN-semi
55.74±0.29
71.01±0.23
53.45±0.93
69.93±0.80
* “w/D” means with distraction. In this setting, many of the unlabelled data are from
the so-called distraction classes , which is different from the classes of labelled data.
14
