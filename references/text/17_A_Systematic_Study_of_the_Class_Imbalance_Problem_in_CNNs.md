# A Systematic Study of the Class Imbalance Problem in CNNs

**Authors**: Buda, Maki, Mazurowski
**Year**: 2018
**arXiv**: 1710.05381
**Topic**: imbalance
**Relevance**: Imbalance mitigation strategies

---


--- Page 1 ---
This manuscript is published in Neural Networks. Please cite it as:
Mateusz Buda, Atsuto Maki, and Maciej A Mazurowski. A systematic study of the class
imbalance problem in convolutional neural networks. Neural Networks, 106:249–259, 2018.
A systematic study of the class imbalance problem
in convolutional neural networks∗
Mateusz Buda1, 2
Atsuto Maki2
Maciej A. Mazurowski1, 3
1Department of Radiology, Duke University School of Medicine, Durham, NC, USA
2Royal Institute of Technology (KTH), Stockholm, Sweden
3Department of Electrical and Computer Engineering, Duke University, Durham, NC, USA
{buda, atsuto}@kth.se
maciej.mazurowski@duke.edu
Abstract
In this study, we systematically investigate the impact of class imbalance on classiﬁcation
performance of convolutional neural networks (CNNs) and compare frequently used methods
to address the issue. Class imbalance is a common problem that has been comprehensively
studied in classical machine learning, yet very limited systematic research is available in
the context of deep learning. In our study, we use three benchmark datasets of increasing
complexity, MNIST, CIFAR-10 and ImageNet, to investigate the eﬀects of imbalance on
classiﬁcation and perform an extensive comparison of several methods to address the issue:
oversampling, undersampling, two-phase training, and thresholding that compensates for
prior class probabilities. Our main evaluation metric is area under the receiver operating
characteristic curve (ROC AUC) adjusted to multi-class tasks since overall accuracy metric
is associated with notable diﬃculties in the context of imbalanced data. Based on results
from our experiments we conclude that (i) the eﬀect of class imbalance on classiﬁcation
performance is detrimental; (ii) the method of addressing class imbalance that emerged as
dominant in almost all analyzed scenarios was oversampling; (iii) oversampling should be
applied to the level that completely eliminates the imbalance, whereas the optimal under-
sampling ratio depends on the extent of imbalance; (iv) as opposed to some classical machine
learning models, oversampling does not cause overﬁtting of CNNs; (v) thresholding should be
applied to compensate for prior class probabilities when overall number of properly classiﬁed
cases is of interest.
Keywords: Class Imbalance, Convolutional Neural Networks, Deep Learning, Image Clas-
siﬁcation
1
Introduction
Convolutional neural networks (CNNs) are gaining signiﬁcance in a number of machine learning
application domains and are currently contributing to the state of the art in the ﬁeld of computer
vision, which includes tasks such as object detection, image classiﬁcation, and segmentation.
They are also widely used in natural language processing or speech recognition where they are
replacing or improving classical machine learning models [1]. CNNs integrate automatic feature
extraction and discriminative classiﬁer in one model, which is the main diﬀerence between them
and traditional machine learning techniques. This property allows CNNs to learn hierarchical
∗R
⃝2018. This manuscript version is made available under the CC-BY-NC-ND 4.0 license.
1
arXiv:1710.05381v2  [cs.CV]  13 Oct 2018

--- Page 2 ---
representations [2]. The standard CNN is built with fully connected layers and a number of
blocks consisting of convolutions, activation function layer and max pooling [3, 4, 5].
The
complex nature of CNNs requires a signiﬁcant computational power for training and evaluation
of the networks, which is addressed with the help of modern graphical processing units (GPUs).
A common problem in real life applications of deep learning based classiﬁers is that some
classes have a signiﬁcantly higher number of examples in the training set than other classes. This
diﬀerence is referred to as class imbalance. There are plenty of examples in domains like computer
vision [6, 7, 8, 9, 10], medical diagnosis [11, 12], fraud detection [13] and others [14, 15, 16] where
this issue is highly signiﬁcant and the frequency of one class (e.g., cancer) can be 1000 times
less than another class (e.g., healthy patient). It has been established that class imbalance can
have signiﬁcant detrimental eﬀect on training traditional classiﬁers [17] including multi-layer
perceptrons [18]. It aﬀects both convergence during the training phase and generalization of a
model on the test set. While the issue very likely also aﬀects deep learning, no systematic study
on the topic is available.
Methods of dealing with imbalance are well studied for classical machine learning models [19,
17, 20, 18]. The most straightforward and common approach is the use of sampling methods.
Those methods operate on the data itself (rather than the model) to increase its balance. Widely
used and proven to be robust is oversampling [21]. Another option is undersampling. Na¨ıve
version, called random majority undersampling, simply removes a random portion of examples
from majority classes [17].
The issue of class imbalance can be also tackled on the level of
the classiﬁer. In such case, the learning algorithms are modiﬁed, e.g. by introducing diﬀerent
weights to misclassiﬁcation of examples from diﬀerent classes [22] or explicitly adjusting prior
class probabilities [23].
Some previous studies showed results on cost sensitive learning of deep neural networks [24,
25, 26]. New kinds of loss function for neural networks training were also developed [27]. Re-
cently, a new method for CNNs was introduced that trains the network in two-phases in which
the network is trained on the balanced data ﬁrst and then the output layers are ﬁne-tuned [28].
While little systematic analysis of imbalance and methods to deal with it is available for deep
learning, researchers employ some methods that might be addressing the problem likely based on
intuition, some internal tests, and systematic results available for traditional machine learning.
Based on our review of the literature, the method most commonly applied in deep learning is
oversampling.
The reminder of this paper is organized as follows. Section 2 gives an overview of methods to
address the problem of imbalance. In Section 3 we describe the experimental setup. It provides
details about compared methods, datasets and models used for evaluation. Then, in Section 4
we present the results from our experiments and compare methods. Finally, Section 5 concludes
the paper.
2
Methods for addressing imbalance
Methods for addressing class imbalance can be divided into two main categories [29]. The ﬁrst
category is data level methods that operate on training set and change its class distribution.
They aim to alter dataset in order to make standard training algorithms work.
The other
category covers classiﬁer (algorithmic) level methods. These methods keep the training dataset
unchanged and adjust training or inference algorithms. Moreover, methods that combine the
two categories are available. In this section we give an overview of commonly used approaches
in both classical machine learning models and deep neural networks.
2

--- Page 3 ---
2.1
Data level methods
Oversampling.
One of the most commonly used method in deep learning [16, 30, 31, 32]. The
basic version of it is called random minority oversampling, which simply replicates randomly
selected samples from minority classes. It has been shown that oversampling is eﬀective, yet it
can lead to overﬁtting [33, 34]. A more advanced sampling method that aims to overcome this
issue is SMOTE [33]. It augments artiﬁcial examples created by interpolating neighboring data
points. Some extensions of this technique were proposed, for example focusing only on examples
near the boundary between classes [35].
Another type of oversampling approach uses data
preprocessing to perform more informed oversampling. Cluster-based oversampling ﬁrst clusters
the dataset and then oversamples each cluster separately [36]. This way it reduces both between-
class and within-class imbalance. DataBoost-IM, on the other hand, identiﬁes diﬃcult examples
with boosting preprocessing and uses them to generate synthetic data [37]. An oversampling
approach speciﬁc to neural networks optimized with stochastic gradient descent is class-aware
sampling [38]. The main idea is to ensure uniform class distribution of each mini-batch and
control the selection of examples from each class.
Undersampling.
Another popular method [16] that results in having the same number of
examples in each class. However, as opposed to oversampling, examples are removed randomly
from majority classes until all classes have the same number of examples. While it might not
appear intuitive, there is some evidence that in some situations undersampling can be preferable
to oversampling [39]. A signiﬁcant disadvantage of this method is that it discards a portion of
available data. To overcome this shortcoming, some modiﬁcations were introduced that more
carefully select examples to be removed. E.g. one-sided selection identiﬁes redundant examples
close to the boundary between classes [40]. A more general approach than undersampling is
data decontamination that can involve relabeling of some examples [41, 42].
2.2
Classiﬁer level methods
Thresholding.
Also known as threshold moving or post scaling, adjusts the decision threshold
of a classiﬁer. It is applied in the test phase and involves changing the output class probabilities.
There are many ways in which the network outputs can be adjusted. In general, the threshold
can be set to minimize arbitrary criterion using an optimization algorithm [23]. However, the
most basic version simply compensates for prior class probabilities [43]. These are estimated
for each class by its frequency in the imbalanced dataset before sampling is applied. It was
shown that neural networks estimate Bayesian a posteriori probabilities [43]. That is, for a
given datapoint x, their output for class i implicitly corresponds to
yi(x) = p(i|x) = p(i) · p(x|i)
p(x)
.
Therefore, correct class probabilities can be obtained by dividing the network output for each
class by its estimated prior probability p(i) =
|i|
P
k |k|, where |i| denotes the number of unique
examples in class i.
Cost sensitive learning.
This method assigns diﬀerent cost to misclassiﬁcation of examples
from diﬀerent classes [44]. With respect to neural networks it can be implemented in various
ways. One approach is threshold moving [22] or post scaling [23] that is applied in the inference
phase after the classiﬁer is already trained. Similar strategy is to adapt the output of the network
3

--- Page 4 ---
and also use it in the backward pass of backpropagation algorithm [45]. Another adaptation of
neural network to be cost sensitive is to modify the learning rate such that higher cost examples
contribute more to the update of weights. And ﬁnally we can train the network by minimizing
the misclassiﬁcation cost instead of standard loss function [45]. The results of this approach
are equivalent to oversampling [22, 26] described above and therefore this method will not be
implemented in our study.
One-class classiﬁcation.
In the context of neural networks it is usually called novelty de-
tection.
This is a concept learning technique that recognizes positive instances rather than
discriminating between two classes. Autoencoders used for this purpose are trained to perform
autoassociative mapping, i.e. identity function. Then, the classiﬁcation of a new example is
made based on a reconstruction error between the input and output patterns, e.g. absolute
error, squared sum of errors, Euclidean or Mahalanobis distance [46, 47, 48]. This method has
proved to work well for extremely high imbalance when classiﬁcation problem turns into anomaly
detection [49].
Hybrid of methods.
This is an approach that combines multiple techniques from one or
both abovementioned categories.
Widely used example is ensembling.
It can be viewed as
a wrapper to other methods.
EasyEnsemble and BalanceCascade are methods that train a
committee of classiﬁers on undersampled subsets [50]. SMOTEBoost, on the other hand, is a
combination of boosting and SMOTE oversampling [51]. Recently introduced and successfully
applied to CNN training for brain tumor segmentation, is two-phase training [28]. Even though
the task was image segmentation, it was approached as a pixel level classiﬁcation. The method
involves network pre-training on balanced dataset and then ﬁne-tuning the last output layer
before softmax on the original, imbalanced data.
3
Experiments
3.1
Forms of imbalance
Class imbalance can take many forms particularly in the context of multiclass classiﬁcation,
which is typical in CNNs.
In some problems only one class might be underrepresented or
overrepresented and in other every class will have a diﬀerent number of examples. In this study
we deﬁne and investigate two types of imbalance that we believe are representative of most of
the real-world cases.
The ﬁrst type is step imbalance. In step imbalance, the number of examples is equal within
minority classes and equal within majority classes but diﬀers between the majority and minority
classes.
This type of imbalance is characterized by two parameters.
One is the fraction of
minority classes deﬁned by
µ = |{i ∈{1, . . . , N} : Ci is minority}|
N
,
(1)
where Ci is a set of examples in class i and N is the total number of classes. The other parameter
is a ratio between the number of examples in majority classes and the number of examples in
minority classes deﬁned as follows.
ρ = maxi{|Ci|}
mini{|Ci|}
(2)
4

--- Page 5 ---
An example of this type of imbalance is the situation when among the total of 10 classes, 5 of
them have 500 training examples and another 5 have 5 000. In this case ρ = 10 and µ = 0.5,
as shown in Figure 1a. A dataset with the same number of examples in total that has smaller
imbalance ratio, corresponding to parameter ρ = 2, but more classes being minority, µ = 0.9, is
presented in Figure 1b.
(a) ρ = 10, µ = 0.5
(b) ρ = 2, µ = 0.9
(c) ρ = 10
Figure 1: Example distributions of imbalanced set together with corresponding values of parameters ρ
and µ for step imbalance (a - b) and ρ for linear imbalance (c).
The second type of imbalance we call linear imbalance. We deﬁne it with one parameter
that is a ratio between the maximum and minimum number of examples among all classes,
as in Equation 2 for imbalance ratio in step imbalance. However, the number of examples in
the remaining classes is interpolated linearly such that the diﬀerence between consecutive pairs
of classes is constant. An example of linear imbalance distribution with ρ = 10 is shown in
Figure 1c.
3.2
Methods of addressing imbalance compared in this study
In total, we examine seven methods to handle CNN training on a dataset with class imbalance
which cover most of the commonly used approaches in the context of deep learning:
1. Random minority oversampling
2. Random majority undersampling
3. Two-phase training with pre-training on randomly oversampled dataset
4. Two-phase training with pre-training on randomly undersampled dataset
5. Thresholding with prior class probabilities
6. Oversampling with thresholding
7. Undersampling with thresholding
We examine two variants of two-phase training method. One on oversampled and the other
on undersampled dataset. For the second phase, we keep the same hyperparameters and learn-
ing rate decay policy as in the ﬁrst phase. Only the base learning rate from the ﬁrst phase is
multiplied by the factor of 10−1. Regarding thresholding, this method originally uses the im-
balanced training set to train a neural network. We, in addition, combine it with oversampling
and undersampling.
Selected methods are representative of the available approaches. Sampling can be used to
explicitly incorporate cost of the examples by their appearance. It makes them one of many
5

--- Page 6 ---
implementations of cost-sensitive learning [22]. Thresholding is another way of applying cost-
sensitiveness by moving the output threshold such that higher cost examples are harder to
misclassify. Ensemble methods require training of multiple classiﬁers. Because of considerable
time needed to train deep models, it is often not practical and may be even infeasible to train
multiple deep neural networks. One-class methods have a very limited application to datasets
with extremely high imbalance. Moreover, they are applied to anomaly detection problem that
is beyond the scope of our study.
Importantly, we focused on methods that are widely used and relatively straightforward to
implement as our aim is to draw conclusions that will be practical and serve as a guidance to a
large number of deep learning researchers and engineers.
3.3
Datasets and models
In our study, we used three benchmark datasets: MNIST [52], CIFAR-10 [53] and ImageNet
Large Scale Visual Recognition Challenge (ILSVRC) 2012 [54]. All of them are provided with a
split on training and test set that are both labeled. For each dataset we choose diﬀerent model
with a set of hyperparameters used for its training that is known to perform well based on
the literature. Datasets together with their corresponding models are of increasing complexity.
This allows us to draw some conclusions on simple task and then verify how they scale to more
complex ones.
All networks for the same dataset were trained with equal number of iterations. It means
that the number of epochs diﬀers between the imbalanced versions of dataset. This way we
keep the number of weights’ updates constant. Also, all networks were trained from a random
initialization of weights and no pretraining was applied.
An overview of some information
about the datasets and their corresponding models is given in Table 1. All experiments were
implemented in the deep learning framework Caﬀe [55].
Image dimensions
Images per class
Dataset
Width
Height
Depth
No. classes
Training
Test
CNN model
MNIST
28
28
1
10
5 000
1 000
LeNet-5
CIFAR-10
32
32
3
10
5 000
1 000
All-CNN
ILSVRC-2012
≥256
≥256
3
1 000
1 000
50
ResNet-10
Table 1: Summary of the used datasets. The number of images per class refers to the perfectly
balanced subsets used for experiments. Provided image dimensions for ImageNet are given after
rescaling.
3.3.1
MNIST
MNIST is considered simple and solved problem that involves digits’ images classiﬁcation. The
dataset consists of grayscale images of size 28×28. There are ten classes corresponding to digits
from 0 to 9. The number of examples per class in the original training dataset ranges from
5421 in class 5 to 6742 in class 1. In artiﬁcially imbalanced versions we uniformly at random
subsample each class to contain no more than 5 000 examples.
The CNN model that we use for MNIST is the modern version of LeNet-5 [52]. The network
architecture is presented in Table 2.
All networks for this dataset were trained for 10 000
iterations. Optimization algorithm is stochastic gradient descent (SGD) with momentum value of
6

--- Page 7 ---
µ = 0.9 [56]. The learning rate decay policy is deﬁned as ηt = η0 · (1 + γ · t)−α, where η0 = 0.01
is a base learning rate, γ = 0.0001 and α = 0.75 are decay parameters and t is the current
iteration. Furthermore, we used a batch size of 64 and a weight decay value of λ = 0.0005.
Network weights were initialized randomly with uniform distribution and Xavier variance [57]
whereas the biases were initialized with zero. No data augmentation was used. Test error of the
model trained as described above on the original MNIST dataset was below 1%.
Data dimensions
Layer
Width
Height
Depth
Kernel size
Stride
Input
28
28
1
-
-
Convolution
24
24
20
5
1
Max Pooling
12
12
20
2
2
Convolution
8
8
50
5
1
Max Pooling
4
4
50
2
2
Fully Connected
1
1
500
-
-
ReLU
1
1
500
-
-
Fully Connected
1
1
10
-
-
Softmax
1
1
10
-
-
Table 2: Architecture of LeNet-5 CNN used in MNIST experiments.
Experiments on MNIST dataset are performed on the following imbalance parameters space.
For linear imbalance we test values of ρ ∈{10, 25, 50, 100, 250, 500, 1 000, 2 500, 5 000}. For step
imbalance the set of ρ values is the same and for each we use all possible number of minority
classes from 1 to 9, which corresponds to µ ∈{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}. The experi-
ment for each combination of parameters is repeated 50 times. Every time the subset of minority
classes is randomized. This way, we have created 4 050 artiﬁcially imbalanced training sets for
step imbalance and 450 for linear imbalance. As we have evaluated four methods that require
training a model, the total number of trained networks, including baseline, is 22 500.
3.3.2
CIFAR-10
CIFAR-10 is a signiﬁcantly more complex image classiﬁcation problem than MNIST. It contains
32 × 32 color images with ten classes of natural objects. It does not have any natural imbalance
at all. There are exactly 5 000 training and 1 000 test examples in each class. We do not use any
data augmentation but follow standard preprocessing comprising global contrast normalization
and ZCA whitening [58].
For CIFAR-10 experiments we use one of the best performing type of CNN model on this
dataset, i.e. All-CNN [59]. The network architecture is presented in Table 3. The networks were
trained for 70 000 iterations using SGD with momentum µ = 0.9. The base learning rate was
multiplied by a ﬁxed multiplier of 0.1 after 40 000, 50 000 and 60 000 iterations. The number of
examples in a batch was 256 and a weight decay value was λ = 0.001. Network weights were
initialized with Xavier procedure and the biases set to zero. Test error of the model trained as
described above on the original CIFAR-10 dataset was 9.75%.
We have found the network training to be quite sensitive to initialization and the choice of
base learning rate. Sometimes the network gets stuck in a very poor local minimum. Also, for
more imbalanced datasets the training required lower base learning rate to train at all. Therefore,
7

--- Page 8 ---
for each case we were searching for the best one from the ﬁxed set η0 ∈{0.05, 0.005, 0.0005, 0.00005}.
Similar procedure was used by the authors of the model architecture [59]. Moreover, each train-
ing was repeated twice on the same dataset. For a particular method and imbalanced dataset,
we pick the model with the best score on the test set over all eight runs.
Data dimensions
Layer
Width
Height
Depth
Kernel size
Stride
Padding
Input
32
32
3
-
-
-
Dropout (0.2)
32
32
3
-
-
-
2×(Convolution + ReLU)
32
32
96
3
1
1
Convolution + ReLU
16
16
96
3
2
1
Dropout (0.5)
16
16
96
-
-
-
2×(Convolution + ReLU)
16
16
192
3
1
1
Convolution + ReLU
8
8
192
3
2
1
Dropout (0.5)
8
8
192
-
-
-
Convolution + ReLU
6
6
192
3
1
0
Convolution + ReLU
6
6
192
1
1
0
Convolution + ReLU
6
6
10
1
1
0
Average Pooling
1
1
10
6
-
-
Softmax
1
1
10
-
-
-
Table 3: Architecture of All-CNN used in CIFAR-10 experiments.
The network architecture does not have any fully connected layers. Therefore, during the
ﬁne-tuning in two-phase training method we update the weights of two last convolutional layers
with kernels of size 1.
The imbalance parameters space used in CIFAR-10 experiments is considerably sparser than
the one used for MNIST due to the signiﬁcantly longer time required to train one network. The
set of tested values was narrowed to make the experiment run in a reasonable time. For linear
and step imbalance, we test values of ρ ∈{2, 10, 20, 50}. In step imbalance, for each value of ρ,
the set of values of parameter µ was µ ∈{0.2, 0.5, 0.8}, which corresponds to having two, ﬁve
and eight minority classes, respectively. And for all the cases, the classes chosen to be minority
were the ones with the lowest label value. It means that for a ﬁxed number of minority classes
the same classes were always picked as minority. Also, all of them were included in a larger set
of minority classes. In total we trained 640 networks on this dataset.
3.3.3
ImageNet
For evaluation we use a ILSVRC-2012 competition subset of ImageNet, widely used as a bench-
mark to compare classiﬁers’ performance.
The number of examples in majority classes was
reduced from 1 200 to 1 000. Classes with less than 1 000 cases were always chosen as a minor-
ity ones for imbalanced subsets. The only data preprocessing applied is resizing such that the
smaller dimension is 256 pixels long and the aspect ratio is preserved. During training, as input
we use a randomly cropped 224 × 224 pixel square patch and a single centered crop in a test
phase. Moreover, during training we randomly mirror images, but there is no color, scale or
aspect ratio augmentation.
A model architecture employed for this dataset is ResNet-10 [60], i.e. a residual network [61]
8

--- Page 9 ---
with batch normalization layers that are known to accelerate deep networks training [62]. It
consists of four residual blocks that give us nine convolutional layers and one fully connected.
The ﬁrst residual block outputs data tensor of depth 64 and then each one increases it by a
factor of two. Fully connected layer outputs 1 000 values to softmax that transforms them to
class probabilities. The architecture of one residual block is presented in Figure 2.
Figure 2: Architecture of a single residual block in ResNet used in ILSVRC-2012 experiments.
The networks were trained for 320 000 iterations using SGD with momentum µ = 0.9. The
base learning rate is set to η0 = 0.1 and decays linearly to 0 in the last iteration. The number of
examples in a batch was 256 and a weight decay λ = 0.0001. Network weights were initialized
with Kaiming (also known as MSRA) initialization procedure [63].
Top-1 test error of the
model trained as described above on the original ILSVRC-2012 dataset was 62.56% and 99.50
multi-class ROC AUC for a single centered crop.
We have chosen relatively small ResNet for the sake of faster training 1. We test only one
case of small and two cases of large step imbalance and run it on the baseline, undersampling
and oversampling methods.
Speciﬁcally, all three step imbalanced subsets are deﬁned with
µ = 0.1, ρ = 10, µ = 0.8, ρ = 50 and µ = 0.9, ρ = 100. They correspond to 100 minority classes
with imbalance ratio of 10, 800 minority classes with imbalance of 50, and 900 minority classes
with imbalance ratio of 100, respectively. Moreover, for the highest imbalance, we train three
networks for each method with randomized selection of minority classes and subsampled set of
examples in each class. This is done in order to estimate variability in performance of methods.
In total, this gives us 15 ResNet-10 networks trained on ﬁve artiﬁcially imbalanced subsets of
ILSVRC-2012.
3.4
Evaluation metrics and testing
The metric that is most widely used to evaluate a classiﬁer performance in the context of multi-
class classiﬁcation with CNNs is overall accuracy which is the proportion of test examples that
were correctly classiﬁed. However, it has some signiﬁcant and long acknowledged limitations,
particularly in the context of imbalanced datasets [19]. Speciﬁcally, when the test set is im-
balanced, accuracy will favor classes that are overrepresented in some cases leading to highly
misleading assessment. An example of this is a situation when the majority class represents 99%
of all cases and the classiﬁer assigns the label of the majority class to all test cases. A misleading
accuracy of 99% will be assigned to a classiﬁer that has a very limited use. Another issue might
arise when the test set is balanced and a training set is imbalanced. This might result in a
situation when a decision threshold is moved to reﬂect the estimated class prior probabilities
and cause a low accuracy measure in the test set while the true discriminative power of the
classiﬁer does not change.
A measure that addresses these issues is area under the receiver operating characteristic
curve (ROC AUC) [64] which is a plot of the false positive rate to the true positive rate for all
1It takes ﬁve days to train one ResNet-10 network on Nvidia GTX 1070 GPU.
9

--- Page 10 ---
possible prediction thresholds. We used a speciﬁc implementation of the ROC AUC available
in scikit-learn python package [65]. It calculates sensitivities and speciﬁcities at all thresholds
deﬁned by the responses of the classiﬁer in the test set followed by the AUC calculation using
the trapezoid rule. ROC AUC is a well-studied and sound measure of discrimination [66] and
has been widely used as an evaluation metric for classiﬁers. ROC has also been used to compare
performance of classiﬁers trained on imbalanced datasets [18, 20]. Since the basic version of
ROC is only suitable for binary classiﬁcation, we use a multi-class modiﬁcation of it [67]. The
multi-class ROC is calculated by taking the average of AUCs obtained independently for each
class for the binary classiﬁcation task of distinguishing a given class from all the other classes.
Test set of all used datasets has equal number of examples in each class.
Usually, it is
assumed that the class distribution of a test set follows the one of a training set. We do not
change a test set to match artiﬁcially imbalanced training set. The reason is that the score
achieved by each classiﬁer on the same test set is more comparable and the largest number of
cases in each of the classes provides the most accurate performance estimation.
(a) 2 minority classes
(b) 5 minority classes
(c) 8 minority classes
(d) 2 minority classes
(e) 5 minority classes
(f) 8 minority classes
Figure 3: Comparison of methods with respect to multi-class ROC AUC on MNIST (a - c) and CIFAR-10
(d - f) for step imbalance with ﬁxed number of minority classes.
4
Results
4.1
Eﬀects of class imbalance on classiﬁcation performance and comparison
of methods to address imbalance
The results showing the impact of class imbalance on classiﬁcation performance and comparison
of methods for addressing imbalance are shown in Figures 3 and 4. Figure 3 shows the results
10

--- Page 11 ---
with respect to multi-class ROC AUC for a ﬁxed number of minority classes on MNIST and
CIFAR-10. Figure 4 presents the result from the perspective of ﬁxed ratio of imbalance, i.e.
parameter ρ, for the same two datasets.
Regarding the eﬀect of class imbalance on classiﬁcation performance, we observed the follow-
ing. First, the deterioration of performance due to class imbalance is substantial. As expected,
the increasing ratio of examples between majority and minority classes as well as the number of
minority classes had a negative eﬀect on performance of the resulting classiﬁers. Furthermore,
by comparing the results from MNIST and CIFAR-10 we observed that the eﬀect of imbalance
is signiﬁcantly stronger for the task with higher complexity. A similar drop in performance for
MNIST and CIFAR-10 corresponded to approximately 100 times stronger level of imbalance in
the MNIST dataset.
(a) Imbalance ratio of 100
(b) Imbalance ratio of 500
(c) Imbalance ratio of 1 000
(d) Imbalance ratio of 10
(e) Imbalance ratio of 20
(f) Imbalance ratio of 50
Figure 4: Comparison of methods with respect to multi-class ROC AUC on MNIST (a - c) and CIFAR-10
(d - f) for step imbalance with ﬁxed imbalance ratio.
Regarding performance of diﬀerent methods for addressing imbalance, in almost all of the
situations oversampling emerged as the best method. It also showed notable improvement of
performance over the baseline (i.e. do-nothing strategy) in majority of the situations and never
showed a considerable decrease in performance for the two datasets analyzed in this section
making it a clear recommendation for tasks similar to MNIST and CIFAR-10.
Undersampling showed a generally poor performance. In a large number of analyzed scenar-
ios, undersampling showed decrease in performance as compared to the baseline. In scenarios
with a large proportion of minority classes undersampling showed some improvement over the
baseline but never a notable advantage over oversampling (Figure 3).
For a ﬁxed imbalance ratio undersampling is always trained on the subset of equal size.
As a result, its performance does not change with the number of minority classes. For both
11

--- Page 12 ---
datasets and each case of imbalance ratio, the gap between undersampling and oversampling is
the biggest for smaller number of minority classes and decreases with the number of minority
classes, as shown in Figure 4. This is expected since with all classes being minority these two
methods become equivalent.
Two-phase training methods with both undersampling and oversampling tend to perform
between the baseline and their corresponding method (undersampling or oversampling).
If
the baseline is better than one of these methods, ﬁne-tuning improves the original method.
Otherwise, performance deteriorates. However, if the baseline is better, there is still no gain from
using two-phase training method. As oversampling is almost always better than the baseline,
ﬁne-tuning always gives lower score.
The variability of patters, visual structures, and objects in CIFAR-10 is considerably higher
then in MNIST. For this reason, we run the step imbalance experiment three times on a reshuﬄed
stratiﬁed training and test split to validate our results.
Additional results are available in
Appendix A.
(a) MNIST
(b) CIFAR-10
Figure 5: Comparison of methods with respect to multi-class ROC AUC for linear imbalance.
In Figure 5 we show the results for linear imbalance on MNIST and CIFAR-10 datasets. The
highest possible linear imbalance ratio for MNIST dataset is 5 000, which means only one example
in the most underrepresented class. However, even in this case the decrease in performance
according to multi-class ROC AUC score for the baseline model is not signiﬁcant, as shown in
Figure 5a. Nevertheless, oversampling improves the score on both datasets and for all tested
values of ρ, whereas the score for undersampling decreases approximately linearly with imbalance
ratio.
4.2
Results on ImageNet dataset
The results from experiments performed on ImageNet (ILSVRC-2012) dataset conﬁrm the im-
pact of imbalance on classiﬁer’s performance. Table 4 compares methods with respect to multi-
class ROC AUC. The drop in performance for the largest tested imbalance was from 99 to 90, in
terms of multi-class ROC AUC. The results conﬁrm that the oversampling approach performs
consistently better than undersampling approach across all scenarios. A small decrease in per-
formance as compared to baseline was observed for oversampling for extreme imbalances. Please
note, however, that these results should be treated with caution and not as strong evidence that
oversampling is inferior for highly complex tasks with extreme imbalance. The absolute diﬀer-
ence in performance between three runs with respect to multi-class ROC AUC was even higher
than 4 (for undersampling). Therefore, diﬀerences of 1 - 2 might be due to variability of results
12

--- Page 13 ---
between diﬀerent runs of neural networks. Moreover, the highest tested imbalanced training set
was only about 10% of the original ILSVRC-2012 introducing confounding issues such as the
optimal training hyperparameters for this signiﬁcantly changed dataset. Therefore, while these
results indicate that caution should be taken when any sampling technique is applied to highly
complex tasks with extreme imbalances, it needs a more extensive study devoted to this speciﬁc
issue.
Method
µ = 0.1, ρ = 10
µ = 0.8, ρ = 50
µ = 0.9, ρ = 100
Baseline
99.41
96.31
90.74
90.46
90.05
Oversampling
99.35
95.06
88.38
88.39
88.17
Undersampling
96.85
94.98
88.35
84.08
83.74
Table 4: Comparison of results on ImageNet with respect to multi-class ROC AUC.
4.3
Separation of eﬀects from reduced number of examples and class imbal-
ance
An important question that needs to be considered in the context of our study is whether the
decrease in performance for imbalanced datasets is merely caused by the fact that our imbalanced
datasets simply had fewer training examples or is it truly caused by the fact that the datasets
are imbalanced.
First, we notice that oversampling method uses the same amount of data as the baseline.
It only eliminates the imbalance which is enough to improve the performance in almost all the
cases. Still, it does not reach the performance of a classiﬁer trained on the original dataset. This
is an indication that the eﬀect of imbalance is not trivial.
Second, for some cases undersampling, which reduces the total number of cases performs bet-
ter than the baseline (see Figures 3c and 3f). Moreover, there are even cases when undersampling
can perform on a par with oversampling. It means that, between two sampling methods that
eliminate imbalance, even using fewer data can be comparable.
In addition, for the same value of parameter ρ we have equal number of examples in the
training set for linear imbalance and step imbalance with µ = 0.5, which corresponds to half
of the classes being minority.
The drop in performance is much higher for step imbalance.
This additionally demonstrates that not only the total number of examples matters but also its
distribution between classes.
4.4
Improving accuracy score with multi-class thresholding
While our focus is on ROC AUC, we also provide the evaluation of the methods based on overall
accuracy measure with results on step imbalance shown in Figure 6. As explained in Section 3.4,
accuracy has some known limitations and in some scenarios does not reﬂect the discriminative
power of a classiﬁer but rather the prevalence of classes in the training or test set. Nevertheless,
it is still commonly used evaluation score [16] and therefore we provide some results according
to this metric.
Our results show that thresholding is an appropriate approach to take to oﬀset the prior
probabilities of diﬀerent classes learned by a network based on imbalanced datasets and provided
an improvement in overall accuracy. In general, thresholding worked particularly well when
applied jointly with oversampling.
13

--- Page 14 ---
(a) 2 minority classes
(b) 5 minority classes
(c) 8 minority classes
(d) 2 minority classes
(e) 5 minority classes
(f) 8 minority classes
Figure 6: Comparison of methods with respect to accuracy on MNIST (a - c) and CIFAR-10 (d - f) for
step imbalance with ﬁxed number of minority classes.
Please note that thresholding does not have an actual eﬀect on the ability of the classiﬁer
to discriminate between a given class from another but rather helps to ﬁnd a threshold on the
network output that guarantees a large number of correctly classiﬁed cases. In terms of ROC,
multiplying a decision variable by any positive number does not change the area under the ROC
curve. However, ﬁnding an optimal operating point on the ROC curve is important when the
overall number of correctly classiﬁed cases is of interest.
4.5
Undersampling and oversampling to smaller imbalance ratio
The default version of oversampling is to increase the number of cases in the minority classes
so that the number matches the majority classes. Similarly, the default of undersampling is to
decrease the number of cases in the majority classes to match the minority classes. However,
a more moderate version of these algorithms could be applied. For the case of MNIST with
imbalance ratio of 1 000 we have tried to gradually decrease the imbalance with oversampling
and undersampling. The results are shown in Figure 7.
The results show that the default version of oversampling was always the best. Any reduction
of imbalance improves the score regardless of the number of minority classes, as shown in Fig-
ure 7a. For undersampling, in some cases of moderate number of minority classes, intermediate
levels of undersampling performed better than both full undersampling and the baseline.
Moreover, comparing undersampling and oversampling to reduced level of imbalance, we can
notice that for each case of oversampling there is a level to which we can apply undersampling and
achieve equivalent performance. However, that level is not known a priori rendering oversampling
still the method of choice.
14

--- Page 15 ---
(a) Oversampling
(b) Undersampling
Figure 7: Comparison of oversampling and undersampling to reduced imbalance ratios on MNIST with
original imbalance of 1 000.
4.6
Generalization of sampling methods
In some cases undersampling and oversampling perform similarly. In those cases, one would
probably prefer the model that generalizes better.
For classical machine learning models it
was shown that oversampling can cause overﬁtting, especially for minority classes [33]. As we
repeat small number of examples multiple times, the trained model ﬁts them too well. Thus,
according to this prior knowledge undersampling would be a better choice. The results from our
experiments do not conﬁrm this conclusion for convolutional neural networks.
In Figure 8 we compare the convergence of baseline and sampling methods for CIFAR-10
experiments with respect to accuracy. Both oversampling and undersampling methods helped
to train a better classiﬁer in terms of performance and generalization. They also made training
more stable. As opposed to traditional machine learning methods, in this case oversampling did
not lead to overﬁtting. The gap between accuracy on the training and test set does not increase
with iterations for oversampling, Figure 8b.
Furthermore, we validated this phenomenon in
multiple additional scenarios for all analyzed datasets and have not observed overﬁtting in any
of these scenarios. This observation also holds for MNIST and ImageNet datasets and other
cases of imbalance. The additional plots are included in Appendix A.
(a) Baseline
(b) Oversampling
(c) Undersampling
Figure 8: Comparison of networks convergence between baseline and sampling methods. Training on
CIFAR-10 step imbalanced with 5 minority classes and imbalance ratio of 50.
15

--- Page 16 ---
5
Conclusions
In this study, we examined the impact of class imbalance on classiﬁcation performance of convo-
lutional neural networks and investigated the eﬀectiveness of diﬀerent methods of addressing the
issue. We deﬁned and parametrized two representative types of imbalance, i.e. step and linear.
Then we subsampled MNIST, CIFAR-10 and ImageNet (ILSVRC-2012) datasets to make them
artiﬁcially imbalanced. We have compared common sampling methods, basic thresholding, and
two-phase training.
The conclusions from our experiments related to the class imbalance are as follows.
• The eﬀect of class imbalance on classiﬁcation performance is detrimental.
• The inﬂuence of imbalance on classiﬁcation performance increases with the scale of a task.
• The impact of imbalance cannot be explained simply by the lower total number of training
cases and depends on the distribution of examples among classes.
Regarding the choice of a method to handle CNN training on imbalanced dataset we conclude
the following.
• The method that in most of the cases outperforms all others with respect to multi-class
ROC AUC was oversampling.
• For extreme ratio of imbalance and large portion of classes being minority, undersampling
performs on a par with oversampling. If training time is an issue, undersampling is a
better choice in such a scenario since it dramatically reduces the size of the training set.
• To achieve the best accuracy, one should apply thresholding to compensate for prior class
probabilities. A combination of thresholding with baseline and oversampling is the most
preferable, whereas it should not be combined with undersampling.
• Oversampling should be applied to the level that completely eliminates the imbalance,
whereas the optimal undersampling ratio depends on the extent of imbalance. The higher
a fraction of minority classes in the imbalanced training set, the more imbalance ratio
should be reduced.
• Oversampling does not cause overﬁtting of convolutional neural networks, as opposed to
some classical machine learning models.
Appendix A. Supplementary data
Supplementary material can be found online at https://doi.org/10.1016/j.neunet.2018.07.011.
16

--- Page 17 ---
References
[1] Jiuxiang Gu, Zhenhua Wang, Jason Kuen, Lianyang Ma, Amir Shahroudy, Bing Shuai, Ting
Liu, Xingxing Wang, and Gang Wang. Recent advances in convolutional neural networks.
arXiv preprint arXiv:1512.07108, 2015.
[2] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks.
In European conference on computer vision, pages 818–833. Springer, 2014.
[3] Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard,
Wayne Hubbard, and Lawrence D Jackel.
Backpropagation applied to handwritten zip
code recognition. Neural computation, 1(4):541–551, 1989.
[4] Alex Krizhevsky, Ilya Sutskever, and Geoﬀrey E Hinton. Imagenet classiﬁcation with deep
convolutional neural networks. In Advances in neural information processing systems, pages
1097–1105, 2012.
[5] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale
image recognition. arXiv preprint arXiv:1409.1556, 2014.
[6] Grant Van Horn, Oisin Mac Aodha, Yang Song, Alex Shepard, Hartwig Adam, Pietro
Perona, and Serge Belongie.
The inaturalist challenge 2017 dataset.
arXiv preprint
arXiv:1707.06642, 2017.
[7] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun
database: Large-scale scene recognition from abbey to zoo. In Computer vision and pattern
recognition (CVPR), 2010 IEEE conference on, pages 3485–3492. IEEE, 2010.
[8] Brian Alan Johnson, Ryutaro Tateishi, and Nguyen Thanh Hoan. A hybrid pansharpening
approach and multiscale object-based image analysis for mapping diseased pine and oak
trees. International journal of remote sensing, 34(20):6969–6982, 2013.
[9] Miroslav Kubat, Robert C Holte, and Stan Matwin. Machine learning for the detection of
oil spills in satellite radar images. Machine learning, 30(2-3):195–215, 1998.
[10] Oscar Beijbom, Peter J Edmunds, David I Kline, B Greg Mitchell, and David Kriegman.
Automated annotation of coral reef survey images. In Computer Vision and Pattern Recog-
nition (CVPR), 2012 IEEE Conference on, pages 1170–1177. IEEE, 2012.
[11] Jerzy W Grzymala-Busse, Linda K Goodwin, Witold J Grzymala-Busse, and Xinqun Zheng.
An approach to imbalanced data sets based on changing rule strength. In Rough-Neural
Computing, pages 543–553. Springer, 2004.
[12] Brian Mac Namee, Padraig Cunningham, Stephen Byrne, and Owen I Corrigan.
The
problem of bias in training data in regression problems in medical decision support. Artiﬁcial
intelligence in medicine, 24(1):51–70, 2002.
[13] Philip K Chan and Salvatore J Stolfo. Toward scalable learning with non-uniform class
and cost distributions: A case study in credit card fraud detection. In KDD, volume 1998,
pages 164–168, 1998.
17

--- Page 18 ---
[14] Predrag Radivojac, Nitesh V Chawla, A Keith Dunker, and Zoran Obradovic. Classiﬁcation
and knowledge discovery in protein databases. Journal of Biomedical Informatics, 37(4):
224–239, 2004.
[15] Claire Cardie and Nicholas Howe. Improving minority class prediction using case-speciﬁc
feature weights. In ICML, pages 57–65, 1997.
[16] Guo Haixiang, Li Yijing, Jennifer Shang, Gu Mingyun, Huang Yuanyue, and Gong Bing.
Learning from class-imbalanced data: Review of methods and applications. Expert Systems
with Applications, 2016.
[17] Nathalie Japkowicz and Shaju Stephen. The class imbalance problem: A systematic study.
Intelligent data analysis, 6(5):429–449, 2002.
[18] Maciej A Mazurowski, Piotr A Habas, Jacek M Zurada, Joseph Y Lo, Jay A Baker, and
Georgia D Tourassi. Training neural network classiﬁers for medical decision making: The
eﬀects of imbalanced datasets on classiﬁcation performance. Neural networks, 21(2):427–
436, 2008.
[19] Nitesh V Chawla. Data mining for imbalanced datasets: An overview. In Data mining and
knowledge discovery handbook, pages 853–867. Springer, 2005.
[20] Marcus A Maloof. Learning when data sets are imbalanced and when costs are unequal
and unknown. In ICML-2003 workshop on learning from imbalanced data sets II, 2003.
[21] Charles X Ling and Chenghui Li. Data mining for direct marketing: Problems and solutions.
In KDD, volume 98, pages 73–79, 1998.
[22] Zhi-Hua Zhou and Xu-Ying Liu.
Training cost-sensitive neural networks with methods
addressing the class imbalance problem. IEEE Transactions on Knowledge and Data En-
gineering, 18(1):63–77, 2006.
[23] Steve Lawrence, Ian Burns, Andrew Back, Ah Chung Tsoi, and C Lee Giles. Neural network
classiﬁcation and prior class probabilities. In Neural networks: tricks of the trade, pages
299–313. Springer, 1998.
[24] Salman H Khan, Mohammed Bennamoun, Ferdous Sohel, and Roberto Togneri.
Cost
sensitive learning of deep feature representations from imbalanced data. arXiv preprint
arXiv:1508.03422, 2015.
[25] Vidwath Raj, Sven Magg, and Stefan Wermter. Towards eﬀective classiﬁcation of imbal-
anced data with convolutional neural networks. In IAPR Workshop on Artiﬁcial Neural
Networks in Pattern Recognition, pages 150–162. Springer, 2016.
[26] Yu-An Chung, Hsuan-Tien Lin, and Shao-Wen Yang. Cost-aware pre-training for multiclass
cost-sensitive deep learning. arXiv preprint arXiv:1511.09337, 2015.
[27] Shoujin Wang, Wei Liu, Jia Wu, Longbing Cao, Qinxue Meng, and Paul J Kennedy. Train-
ing deep neural networks on imbalanced data sets. In Neural Networks (IJCNN), 2016
International Joint Conference on, pages 4368–4374. IEEE, 2016.
[28] Mohammad Havaei, Axel Davy, David Warde-Farley, Antoine Biard, Aaron Courville,
Yoshua Bengio, Chris Pal, Pierre-Marc Jodoin, and Hugo Larochelle. Brain tumor seg-
mentation with deep neural networks. Medical image analysis, 35:18–31, 2017.
18

--- Page 19 ---
[29] Haibo He and Edwardo A Garcia. Learning from imbalanced data. IEEE Transactions on
knowledge and data engineering, 21(9):1263–1284, 2009.
[30] Gil Levi and Tal Hassner. Age and gender classiﬁcation using convolutional neural networks.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Work-
shops, pages 34–42, 2015.
[31] Andrew Janowczyk and Anant Madabhushi.
Deep learning for digital pathology image
analysis: A comprehensive tutorial with selected use cases. Journal of pathology informatics,
7, 2016.
[32] Nicolas Jaccard, Thomas W Rogers, Edward J Morton, and Lewis D Griﬃn. Detection
of concealed cars in complex cargo x-ray imagery using deep learning. Journal of X-Ray
Science and Technology, pages 1–17, 2016.
[33] Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. Smote:
synthetic minority over-sampling technique. Journal of artiﬁcial intelligence research, 16:
321–357, 2002.
[34] Kung-Jeng Wang, Bunjira Makond, Kun-Huang Chen, and Kung-Min Wang. A hybrid
classiﬁer combining smote with pso to estimate 5-year survivability of breast cancer patients.
Applied Soft Computing, 20:15–24, 2014.
[35] Hui Han, Wen-Yuan Wang, and Bing-Huan Mao. Borderline-smote: a new over-sampling
method in imbalanced data sets learning. Advances in intelligent computing, pages 878–887,
2005.
[36] Taeho Jo and Nathalie Japkowicz. Class imbalances versus small disjuncts. ACM Sigkdd
Explorations Newsletter, 6(1):40–49, 2004.
[37] Hongyu Guo and Herna L Viktor. Learning from imbalanced data sets with boosting and
data generation: the databoost-im approach. ACM Sigkdd Explorations Newsletter, 6(1):
30–39, 2004.
[38] Li Shen, Zhouchen Lin, and Qingming Huang. Relay backpropagation for eﬀective learning
of deep convolutional neural networks. In European Conference on Computer Vision, pages
467–482. Springer, 2016.
[39] Chris Drummond, Robert C Holte, et al. C4.5, class imbalance, and cost sensitivity: why
under-sampling beats over-sampling. In Workshop on learning from imbalanced datasets II,
volume 11, pages 1–8, 2003.
[40] Miroslav Kubat, Stan Matwin, et al. Addressing the curse of imbalanced training sets:
one-sided selection. In ICML, volume 97, pages 179–186. Nashville, USA, 1997.
[41] Jack Koplowitz and Thomas A Brown. On the relation of performance to editing in nearest
neighbor rules. Pattern Recognition, 13(3):251–255, 1981.
[42] Ricardo Barandela, E Rangel, Jos´e Salvador S´anchez, and Francesc J Ferri. Restricted
decontamination for the imbalanced training sample problem. In Iberoamerican Congress
on Pattern Recognition, pages 424–431. Springer, 2003.
19

--- Page 20 ---
[43] Michael D Richard and Richard P Lippmann. Neural network classiﬁers estimate bayesian
a posteriori probabilities. Neural computation, 3(4):461–483, 1991.
[44] Charles Elkan. The foundations of cost-sensitive learning. In International joint conference
on artiﬁcial intelligence, volume 17, pages 973–978. Lawrence Erlbaum Associates Ltd,
2001.
[45] Matjaz Kukar, Igor Kononenko, et al. Cost-sensitive learning with neural networks. In
ECAI, pages 445–449, 1998.
[46] Nathalie Japkowicz, Catherine Myers, Mark Gluck, et al. A novelty detection approach to
classiﬁcation. In IJCAI, volume 1, pages 518–523, 1995.
[47] Nathalie Japkowicz, Stephen Jose Hanson, and Mark A Gluck. Nonlinear autoassociation
is not equivalent to pca. Neural computation, 12(3):531–545, 2000.
[48] Hoon Sohn, Keith Worden, and Charles R Farrar. Novelty detection using auto-associative
neural network. In Symposium on Identiﬁcation of Mechanical Systems: international me-
chanical engineering congress and exposition, New York, NY, pages 573–580, 2001.
[49] Hyoung-joo Lee and Sungzoon Cho. The novelty detection approach for diﬀerent degrees
of class imbalance. In Neural Information Processing, pages 21–30. Springer, 2006.
[50] Xu-Ying Liu, Jianxin Wu, and Zhi-Hua Zhou.
Exploratory undersampling for class-
imbalance learning. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cy-
bernetics), 39(2):539–550, 2009.
[51] Nitesh V Chawla, Aleksandar Lazarevic, Lawrence O Hall, and Kevin W Bowyer. Smote-
boost: Improving prediction of the minority class in boosting. In European Conference on
Principles of Data Mining and Knowledge Discovery, pages 107–119. Springer, 2003.
[52] Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Haﬀner. Gradient-based learning
applied to document recognition. Proceedings of the IEEE, 1998.
[53] Alex Krizhevsky and Geoﬀrey Hinton. Learning multiple layers of features from tiny images.
2009.
[54] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhi-
heng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large
scale visual recognition challenge. International Journal of Computer Vision, 115(3):211–
252, 2015.
[55] Yangqing Jia, Evan Shelhamer, JeﬀDonahue, Sergey Karayev, Jonathan Long, Ross Gir-
shick, Sergio Guadarrama, and Trevor Darrell. Caﬀe: Convolutional architecture for fast
feature embedding. In Proceedings of the 22nd ACM international conference on Multime-
dia, pages 675–678. ACM, 2014.
[56] Ning Qian. On the momentum term in gradient descent learning algorithms. Neural net-
works, 12(1):145–151, 1999.
[57] Xavier Glorot and Yoshua Bengio. Understanding the diﬃculty of training deep feedforward
neural networks. In Aistats, volume 9, pages 249–256, 2010.
20

--- Page 21 ---
[58] Ian J Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron C Courville, and Yoshua Ben-
gio. Maxout networks. ICML (3), 28:1319–1327, 2013.
[59] Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. Striv-
ing for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806, 2014.
[60] Marcel Simon, Erik Rodner, and Joachim Denzler. Imagenet pre-trained models with batch
normalization. arXiv preprint arXiv:1612.01452, 2016.
[61] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for
image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 770–778, 2016.
[62] Sergey Ioﬀe and Christian Szegedy. Batch normalization: Accelerating deep network train-
ing by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.
[63] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectiﬁers:
Surpassing human-level performance on imagenet classiﬁcation. In Proceedings of the IEEE
international conference on computer vision, pages 1026–1034, 2015.
[64] Andrew P Bradley. The use of the area under the roc curve in the evaluation of machine
learning algorithms. Pattern recognition, 30(7):1145–1159, 1997.
[65] Fabian Pedregosa, Ga¨el Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand
Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg,
et al. Scikit-learn: Machine learning in python. Journal of machine learning research, 12
(Oct):2825–2830, 2011.
[66] Charles X Ling, Jin Huang, and Harry Zhang. Auc: a statistically consistent and more
discriminating measure than accuracy. In IJCAI, volume 3, pages 519–524, 2003.
[67] Foster Provost and Pedro Domingos. Tree induction for probability-based ranking. Machine
learning, 52(3):199–215, 2003.
Mateusz Buda, Atsuto Maki, and Maciej A Mazurowski. A systematic study of the class imbal-
ance problem in convolutional neural networks. (Master’s thesis) Royal Institute of Technology
(KTH), 2017. Retrieved from http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-219872
21
