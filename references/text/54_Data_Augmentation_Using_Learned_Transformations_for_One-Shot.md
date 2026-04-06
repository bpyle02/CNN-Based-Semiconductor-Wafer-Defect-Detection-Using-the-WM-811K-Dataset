# Data Augmentation Using Learned Transformations for One-Shot Medical Image Segmentation

**Authors**: Zhao, Data, Greenspan, St-Onge, Bhatt, Murphy, Grady
**Year**: 2019
**arXiv**: 1902.09383
**Topic**: wafer
**Relevance**: Learned augmentation for small datasets applicable to rare defects

---


--- Page 1 ---
Data augmentation using learned transformations
for one-shot medical image segmentation
Amy Zhao
MIT
xamyzhao@mit.edu
Guha Balakrishnan
MIT
balakg@mit.edu
Fr´edo Durand
MIT
fredo@mit.edu
John V. Guttag
MIT
guttag@mit.edu
Adrian V. Dalca
MIT, MGH
adalca@mit.edu
Abstract
Image segmentation is an important task in many med-
ical applications.
Methods based on convolutional neu-
ral networks attain state-of-the-art accuracy; however, they
typically rely on supervised training with large labeled
datasets. Labeling medical images requires signiﬁcant ex-
pertise and time, and typical hand-tuned approaches for
data augmentation fail to capture the complex variations
in such images.
We present an automated data augmentation method for
synthesizing labeled medical images. We demonstrate our
method on the task of segmenting magnetic resonance imag-
ing (MRI) brain scans. Our method requires only a sin-
gle segmented scan, and leverages other unlabeled scans
in a semi-supervised approach. We learn a model of trans-
formations from the images, and use the model along with
the labeled example to synthesize additional labeled exam-
ples. Each transformation is comprised of a spatial defor-
mation ﬁeld and an intensity change, enabling the synthesis
of complex effects such as variations in anatomy and im-
age acquisition procedures. We show that training a super-
vised segmenter with these new examples provides signif-
icant improvements over state-of-the-art methods for one-
shot biomedical image segmentation.
1. Introduction
Semantic image segmentation is crucial to many
biomedical imaging applications, such as performing pop-
ulation analyses, diagnosing disease, and planning treat-
ments. When enough labeled data is available, supervised
deep learning-based segmentation methods produce state-
of-the-art results. However, obtaining manual segmentation
labels for medical images requires considerable expertise
and time. In most clinical image datasets, there are very
Ground truth 
Ours
Random aug
Single-atlas 
segmentation
Figure 1: Biomedical images often vary widely in anatomy,
contrast and texture (top row). Our method enables more
accurate segmentation of anatomical structures compared to
other one-shot segmentation methods (bottom row).
few manually labeled images. The problem of limited la-
beled data is exacerbated by differences in image acquisi-
tion procedures across machines and institutions, which can
produce wide variations in resolution, image noise, and tis-
sue appearance [45].
To overcome these challenges, many supervised biomed-
ical segmentation methods focus on hand-engineered pre-
processing steps and architectures [53, 57]. It is also com-
mon to use hand-tuned data augmentation to increase the
number of training examples [2, 55, 57, 63, 65]. Data aug-
mentation functions such as random image rotations or ran-
dom nonlinear deformations are easy to implement, and are
effective at improving segmentation accuracy in some set-
tings [55, 57, 63, 65]. However, these functions have lim-
ited ability to emulate real variations [26], and can be highly
sensitive to the choice of parameters [25].
We address the challenges of limited labeled data by
learning to synthesize diverse and realistic labeled exam-
ples. Our novel, automated approach to data augmentation
1
arXiv:1902.09383v2  [cs.CV]  6 Apr 2019

--- Page 2 ---
leverages unlabeled images. Using learning-based registra-
tion methods, we model the set of spatial and appearance
transformations between images in the dataset. These mod-
els capture the anatomical and imaging diversity in the un-
labeled images. We synthesize new examples by sampling
transformations and applying them to a single labeled ex-
ample.
We demonstrate the utility of our method on the task of
one-shot segmentation of brain magnetic resonance imag-
ing (MRI) scans. We use our method to synthesize new
labeled training examples, enabling the training of a su-
pervised segmentation network.
This strategy outper-
forms state-of-the art one-shot biomedical segmentation ap-
proaches, including single-atlas segmentation and super-
vised segmentation with hand-tuned data augmentation.
2. Related work
2.1. Medical image segmentation
We focus on the segmentation of brain MR images,
which is challenging for several reasons. Firstly, human
brains exhibit substantial anatomical variations [28, 59, 76].
Secondly, MR image intensity can vary as a result of
subject-speciﬁc noise, scanner protocol and quality, and
other imaging parameters [45]. This means that a tissue
class can appear with different intensities across images –
even images of the same MRI modality.
Many existing segmentation methods rely on scan pre-
processing to mitigate these intensity-related challenges.
Pre-processing methods can be costly to run, and devel-
oping techniques for realistic datasets is an active area of
research [14, 73]. Our augmentation method tackles these
intensity-related challenges from another angle: rather than
removing intensity variations, it enables a segmentation
method to be robust to the natural variations in MRI scans.
A large body of classical segmentation methods use
atlas-based or atlas-guided segmentation, in which a la-
beled reference volume, or atlas, is aligned to a target vol-
ume using a deformation model, and the labels are prop-
agated using the same deformation [6, 13, 22, 32]. When
multiple atlases are available, they are each aligned to
a target volume, and the warped atlas labels are fused
[36, 41, 68, 78].
In atlas-based approaches, anatomical
variations between subjects are captured by a deformation
model, and the challenges of intensity variations are miti-
gated using pre-processed scans, or intensity-robust metrics
such as normalized cross-correlation. However, ambiguities
in tissue appearances (e.g., indistinct tissue boundaries, im-
age noise) can still lead to inaccurate registration and seg-
mentations. We address this limitation by training a seg-
mentation model on diverse realistic examples, making the
segmenter more robust to such ambiguities. We focus on
having a single atlas, and demonstrate that our strategy out-
performs atlas-based segmentation. If more than one seg-
mented example is available, our method can leverage them.
Supervised learning approaches to biomedical segmenta-
tion have gained popularity in recent years. To mitigate the
need for large labeled training datasets, these methods of-
ten use data augmentation along with hand-engineered pre-
processing steps and architectures [2, 40, 53, 57, 63, 65, 82].
Semi-supervised and unsupervised approaches have also
been proposed to combat the challenges of small training
datasets. These methods do not require paired image and
segmentation data. Rather, they leverage collections of seg-
mentations to build anatomical priors [21], to train an adver-
sarial network [39], or to train a novel semantic constraint
[29]. In practice, collections of images are more readily
available than segmentations. Rather than rely on segmen-
tations, our method leverages a set of unlabeled images.
2.2. Spatial and appearance transform models
Models of shape and appearance have been used in a
variety of image analyses.
Parametric spatial transform
models have been used to align and classify handwrit-
ten digits [31, 44, 50].
In medical image registration,
a spatial deformation model is used to establish seman-
tic correspondences between images.
This mature ﬁeld
spans optimization-based methods [4, 7, 67, 70], and re-
cent learning-based methods [8, 9, 20, 42, 62, 72, 80]. We
leverage VoxelMorph [8, 9], a recent unsupervised learning-
based method, to learn spatial transformations.
Many medical image registration methods focus on
intensity-normalized images or intensity-independent ob-
jective functions, and do not explicitly account for varia-
tions in image intensity. For unnormalized images, models
of intensity transforms have used to remove bias ﬁeld ef-
fects from MRI [44, 79]. Spatial and appearance transform
models have been used together to register objects that dif-
fer in shape as well as texture. Many works build upon the
framework of Morphable Models [38] or Active Appear-
ance Models (AAMs) [15, 16], in which statistical models
of shape and texture are constructed. AAMs have been used
to localize anatomical landmarks [17, 58] and perform seg-
mentation [52, 56, 77]. We build upon these concepts by us-
ing convolutional neural networks to learn models of uncon-
strained spatial and intensity transformations. Rather than
learning transform models for the end goal of registration or
segmentation, we sample from these models to synthesize
new training examples. As we show in our experiments,
augmenting a segmenter’s training set in this way can pro-
duce more robust segmentations than performing segmen-
tation using the transform models directly.
2.3. Few-shot segmentation of natural images
Few-shot segmentation is a challenging task in seman-
tic segmentation and video object segmentation.
Exist-

--- Page 3 ---
4) Synthesize training image and segmentation maps
3) Sample transformations
1) Learn spatial transform model
Spatial 
transformation 𝝉𝒔
CNN
…
Unlabeled subject 𝒚
Warp
Train to align
2) Learn appearance transform model
CNN
…
+
Train to match
Inverse-warped
unlabeled subject ෥𝒚
Appearance 
transformation 𝝉𝒂
Sampled appearance 
target 𝒚(𝒋)
Learned 
spatial transform 
model
Sampled spatial  
target 𝒚(𝒊)
Sampled spatial 
transformation 𝝉𝒔
(𝒊)
Sampled appearance 
transformation 𝝉𝒂
(𝒋)
Atlas 𝒙
Learned 
appearance 
transform model
Atlas labels 𝒍𝒙
Atlas 𝒙
+
Sampled appearance 
transformation 𝝉𝒂
(𝒋)
Synthesized image 
ෝ𝒚(𝒊,𝒋)
Synthesized labels 
෡𝒍𝒚
(𝒊,𝒋)
Warp
Sampled spatial 
transformation 𝝉𝒔
(𝒊)
Warp
Atlas 𝒙
Atlas 𝒙
Figure 2: An overview of the proposed method. We learn independent spatial and appearance transform models to capture
the variations in our image dataset. We then use these models to synthesize a dataset of labeled examples. This synthesized
dataset is used to train a supervised segmentation network.
ing approaches focus mainly on natural images. Methods
for few-shot semantic segmentation incorporate informa-
tion from prototypical examples of the classes to be seg-
mented [24, 69]. Few-shot video segmentation is frequently
implemented by aligning objects in each frame to a labeled
reference frame [37, 75]. Other approaches leverage large
labeled datasets of supplementary information such as ob-
ject appearances [11], or incorporate additional information
such as human input [60]. Medical images present different
challenges from natural images; for instance, the visual dif-
ferences between tissue classes are very subtle compared to
the differences between objects in natural images.
2.4. Data augmentation
In image-based supervised learning tasks, data augmen-
tation is commonly performed using simple parameterized
transformations such as rotation and scaling. For medical
images, random smooth ﬂow ﬁelds have been used to simu-
late anatomical variations [51, 63, 64]. These parameterized
transformations can reduce overﬁtting and improve test per-
formance [34, 43, 51, 63, 64]. However, the performance
gains imparted by these transforms vary with the selection
of transformation functions and parameter settings [25].
Recent works have proposed learning data augmentation
transformations from data. Hauberg et al. [31] focus on
data augmentation for classifying MNIST digits. They learn
digit-speciﬁc spatial transformations, and sample training
images and transformations to create new examples aimed
at improving classiﬁcation performance. We learn an ap-
pearance model in addition to a spatial model, and we fo-
cus on the problem of MRI segmentation.
Other recent
works focus on learning combinations of simple transfor-
mation functions (e.g., rotation and contrast enhancement)
to perform data augmentation for natural images [18, 61].
Cubuk et al. [18] use a search algorithm to ﬁnd augmenta-
tion policies that maximize classiﬁcation accuracy. Ratner
et al. [61] learn to create combinations of transformations
by training a generative adversarial network on user input.
These simple transformations are insufﬁcient for capturing
many of the subtle variations in MRI data.
3. Method
We propose to improve one-shot biomedical image seg-
mentation by synthesizing realistic training examples in a
semi-supervised learning framework.
Let {y(i)} be a set of biomedical image volumes, and let
the pair (x, lx) represent a labeled reference volume, or at-

--- Page 4 ---
U
3×3×3 conv
and upsample
Image similarity loss 
𝓛𝒔𝒊𝒎= 𝑪𝑪or 𝐌𝐒𝐄
𝑥𝑦
Input volume
H×W×D×(1×2)
C
C
3×3×3 conv
and max pool
C
3×3×3 conv
C
C
U
U
U
C
concatenate
C
𝑢or 𝜓
ො𝑦
Output 
volume 
H×W×D×1
Smoothness loss 
𝓛𝒔𝒎𝒐𝒐𝒕𝒉
C
Apply
Transformation 
volume
H×W×D×3
or 
H×W×D×1
Figure 3: We use a convolutional neural network based on the U-Net architecture [63] to learn each transform model. The
application of the transformation is a spatial warp for the spatial model, and a voxel-wise addition for the appearance model.
Each convolution uses 3 × 3 × 3 kernels, and is followed by a LeakyReLU activation layer. The encoder uses max pooling
layers to reduce spatial resolution, while the decoder uses upsampling layers.
las, and its corresponding segmentation map. In brain MRI
segmentation, each x and y is a grayscale 3D volume. We
focus on the challenging case where only one labeled atlas
is available, since it is often difﬁcult in practice to obtain
many segmented volumes. Our method can be easily ex-
tended to leverage additional segmented volumes.
To perform data augmentation, we apply transformations
τ (k) to the labeled atlas x. We ﬁrst learn separate spatial and
appearance transform models to capture the distribution of
anatomical and appearance differences between the labeled
atlas and each unlabeled volume. Using the two learned
models, we synthesize labeled volumes {(ˆy(k), ˆl(k)
y )} by ap-
plying a spatial transformation and an appearance transfor-
mation to the atlas volume, and by warping the atlas label
maps using the spatial transformation. Compared to single-
atlas segmentation, which suffers from uncertainty or er-
rors in the spatial transform model, we use the same spa-
tial transformation to synthesize the volume and label map,
ensuring that the newly synthesized volume is correctly la-
beled. These synthetic examples form a labeled dataset that
characterizes the anatomical and appearance variations in
the unlabeled dataset. Along with the atlas, this new train-
ing set enables us to train a supervised segmentation net-
work. This process is outlined in Fig. 2.
3.1. Spatial and appearance transform models
We describe the differences between scans using a com-
bination of spatial and intensity transformations. Speciﬁ-
cally, we deﬁne a transformation τ(·) from one volume to
another as a composition of a spatial transformation τs(·)
and an intensity or appearance transformation τa(·), i.e.,
τ(·) = τs(τa(·)).
We assume a spatial transformation takes the form of
a smooth voxel-wise displacement ﬁeld u. Following the
medical registration literature, we deﬁne the deformation
function φ = id + u, where id is the identity function.
We use x ◦φ to denote the application of the deforma-
tion φ to x. To model the distribution of spatial transfor-
mations in our dataset, we compute the deformation that
warps atlas x to each volume y(i) using φ(i) = gθs(x, y(i)),
where gθs(·, ·) is a parametric function that we describe
later. We write approximate inverse deformation of y(i) to
x as φ−1(i) = gθs(y(i), x).
We model the appearance transformation τa(·) as
per-voxel addition in the spatial frame of the atlas.
We compute this per-voxel volume using the function
ψ(i) = hθa(x, y(i) ◦φ−1(i)), where y(i) ◦φ−1(i) is a vol-
ume that has been registered to the atlas space using our
learned spatial model. In summary, our spatial and appear-
ance transformations are:
τ (i)
s (x) = x ◦φ(i),
φ = gθs(x, y(i))
(1)
τ (i)
a (x) = x + ψ(i),
ψ(i) = hθa(x, y(i) ◦φ−1(i)).
(2)
3.2. Learning
We aim to capture the distributions of the transforma-
tions τs and τa between the atlas and the unlabeled vol-
umes. We estimate the functions gθs(·, ·) and hθa(·, ·) in
Eqs. (1) and (2) using separate convolutional neural net-
works, with each network using the general architecture
outlined in Fig. 3. Drawing on insights from Morphable
Models [38] and Active Appearance Models [16, 17], we
optimize the spatial and appearance models independently.
For our spatial model, we leverage VoxelMorph [8, 9,
20], a recent unsupervised learning-based approach with an
open-source implementation. VoxelMorph learns to output

--- Page 5 ---
a smooth displacement vector ﬁeld that registers one image
to another by jointly optimizing an image similarity loss and
a displacement ﬁeld smoothness term. We use a variant of
VoxelMorph with normalized cross-correlation as the im-
age similarity loss, enabling the estimation of gθs(·, ·) with
unnormalized input volumes.
We use a similar approach to learn the appearance model.
Naively, one might deﬁne hθa(·, ·) from Eq. (2) as a sim-
ple per-voxel subtraction of the volumes in the atlas space.
While this transformation would perfectly reconstruct the
target image, it would include extraneous details when the
registration function φ−1 is imperfect, resulting in image
details in x + ψ that do not match the anatomical labels.
We instead design hθa(·, ·) as a neural network that pro-
duces a per-voxel intensity change in an anatomically con-
sistent manner. Speciﬁcally, we use an image similarity loss
as well as a semantically-aware smoothness regularization.
Given the network output ψ(i) = hθa(x, y(i) ◦φ−1), we de-
ﬁne a smoothness regularization function based on the atlas
segmentation map:
Lsmooth(cx, ψ) = (1 −cx)∇ψ,
(3)
where cx is a binary image of anatomical boundaries com-
puted from the atlas segmentation labels lx, and ∇is the
spatial gradient operator. Intuitively, this term discourages
dramatic intensity changes within the same anatomical re-
gion.
In the total appearance transform model loss La, we
use mean squared error for the image similarity loss
Lsim(ˆy, y) = ||ˆy −y||2. In our experiments, we found that
computing the image similarity loss in the spatial frame of
the subject was helpful. We balance the similarity loss with
the regularization term Lsmooth:
La(x, y(i), φ(i), φ−1(i), ψ(i), cx)
= Lsim
 (x + ψ(i)) ◦φ(i), y(i)
+ λaLsmooth(cx, ψ(i)),
where λa is a hyperparameter.
3.3. Synthesizing new examples
The models described in Eqs.
(1) and (2) enable us
to sample spatial and appearance transformations τ (i)
s , τ (j)
a
by sampling target volumes y(i), y(j) from an unlabeled
dataset. Since the spatial and appearance targets can be dif-
ferent subjects, our method can combine the spatial varia-
tions of one subject with the intensities of another into a
single synthetic volume ˆy. We create a labeled synthetic ex-
ample by applying the transformations computed from the
target volumes to the labeled atlas:
ˆy(i,j) = τ (i)
s (τ (j)
a (x)),
ˆl(i,j)
y
= τ (i)
s (lx).
This process is visualized in steps 3 and 4 in Fig. 2. These
new labeled training examples are then included in the la-
beled training set for a supervised segmentation network.
3.4. Segmentation network
The newly synthesized examples are useful for improv-
ing the performance of a supervised segmentation network.
We demonstrate this using a network based on the state-of-
the-art architecture described in [66]. To account for GPU
memory constraints, the network is designed to segment one
slice at a time. We train the network on random slices from
the augmented training set. We select the number of train-
ing epochs using early stopping on a validation set. We
emphasize that the exact segmentation network architecture
is not the focus of this work, since our method can be used
in conjunction with any supervised segmentation network.
3.5. Implementation
We implemented all models using Keras [12] and Ten-
sorﬂow [1]. The application of a spatial transformation to
an image is implemented using a differentiable 3D spa-
tial transformer layer [8]; a similar layer that uses near-
est neighbor interpolation is used to transform segmenta-
tion maps. For simplicity, we capture the forward and in-
verse spatial transformations described in Section 3.1 using
two identical neural networks. For the appearance trans-
form model, we use the hyperparameter setting λa = 0.02.
We train our transform models with a single pair of vol-
umes in each batch, and train the segmentation model
with a batch size of 16 slices.
All models are trained
with a learning rate of 5e−4.
Our code is available at
https://github.com/xamyzhao/brainstorm.
4. Experiments
We
demonstrate
that
our
automatic
augmentation
method can be used to improve brain MRI segmentation.
We focus on one-shot segmentation of unnormalized scans
– a challenging but practical scenario. Intensity normal-
ization methods such as bias ﬁeld correction [27, 71, 74]
can work poorly in realistic situations (e.g., clinical-quality
scans, or scans with stroke [73] or traumatic brain injury).
4.1. Data
We use the publicly available dataset of T1-weighted
MRI brain scans described in [8].
The scans are com-
piled from eight databases:
ADNI [54], OASIS [46],
ABIDE [48], ADHD200 [49], MCIC [30], PPMI [47],
HABS [19], and Harvard GSP [33]; the segmentation labels
are computed using FreeSurfer [27]. As in [8], we resample
the brains to 256 × 256 × 256 with 1mm isotropic voxels,
and afﬁnely align and crop the images to 160 × 192 × 224.
We do not apply any intensity corrections, and we perform

--- Page 6 ---
skull-stripping by zeroing out voxels with no anatomical la-
bel. For evaluation, we use segmentation maps of the 30
anatomical labels described in [8].
We focus on the task of segmentation using a single la-
beled example. We randomly select 101 brain scans to be
available at training time. In practice, the atlas is usually
selected to be close to the anatomical average of the popu-
lation. We select the most similar training example to the
anatomical average computed in [8]. This atlas is the single
labeled example that is used to train our transform models;
the segmentation labels of the other 100 training brains are
not used. We use an additional 50 scans as a validation set,
and an additional 100 scans as a held-out test set.
4.2. Segmentation baselines
Single-atlas segmentation (SAS): We use the same state-
of-the-art registration model [8] that we trained for our
method’s spatial transform model in a single-atlas segmen-
tation framework. We register the atlas to each test vol-
ume, and warp the atlas labels using the computed defor-
mation ﬁeld [6, 13, 22, 32, 41]. That is, for each test im-
age y(i), we compute φ(i) = gθs(x, y(i)) and predict labels
ˆl(i)
y
= lx ◦φ(i).
Data
augmentation
using
single-atlas
segmentation
(SAS-aug): We use SAS results as labels for the unanno-
tated training brains, which we then include as training ex-
amples for supervised segmentation.
This adds 100 new
training examples to the segmenter training set.
Hand-tuned random data augmentation (rand-aug):
Similarly to [51, 63, 64], we create random smooth defor-
mation ﬁelds by sampling random vectors on a sparse grid,
and then applying bilinear interpolation and spatial blurring.
We evaluated several settings for the amplitude and smooth-
ness of the deformation ﬁeld, including the ones described
in [63], and selected the settings that resulted in the best
segmentation performance on a validation set. We synthe-
size variations in imaging intensity using a global inten-
sity multiplicative factor sampled uniformly from the range
[0.5, 1.5], similarly to [35, 40]. We selected the range to
match the intensity variations in the dataset; this is repre-
sentative of how augmentation parameters are tuned in prac-
tice. This augmentation method synthesizes a new randomly
transformed brain in each training iteration.
Supervised:
We train a fully-supervised segmentation
network that uses ground truth labels for all 101 examples
in our training dataset. Apart from the atlas labels, these
labels are not available for any of the other methods. This
method serves as an upper bound.
Table 1:
Segmentation performance in terms of Dice
score [23], evaluated on a held-out test set of 100 scans.
We report the mean Dice score (and standard deviation in
parentheses) across all 30 anatomical labels and 100 test
subjects. We also report the mean pairwise improvement of
each method over the SAS baseline.
Method
Dice score
Pairwise Dice
improvement
SAS
0.759 (0.137)
-
SAS-aug
0.775 (0.147)
0.016 (0.041)
Rand-aug
0.765 (0.143)
0.006 (0.088)
Ours-coupled
0.795 (0.133)
0.036 (0.036)
Ours-indep
0.804 (0.130)
0.045 (0.038)
Ours-indep + rand-aug
0.815 (0.123)
0.056 (0.044)
Supervised (upper bound)
0.849 (0.092)
0.089 (0.072)
4.3. Variants of our method
Independent sampling (ours-indep): As described in Sec-
tion 3.3, we sample spatial and appearance target images
independently to compute τ (i)
s , τ (j)
a .
With 100 unlabeled
targets, we obtain 100 spatial and 100 appearance transfor-
mations, enabling the synthesis of 10, 000 different labeled
examples. Due to memory constraints, we synthesize a ran-
dom labeled example in each training iteration, rather than
adding all 10, 000 new examples to the training set.
Coupled sampling (ours-coupled): To highlight the ef-
ﬁcacy of our independent transform models, we compare
ours-indep to a variant of our method where we sample each
of the spatial and appearance transformations from the same
target image. This results in 100 possible synthetic exam-
ples. As in ours-indep, we synthesize a random example in
each training iteration.
Ours-indep + rand-aug: When training the segmenter, we
alternate between examples synthesized using ours-indep,
and examples synthesized using rand-aug.
The addition
of hand-tuned augmentation to our synthetic augmentation
could introduce additional variance that is unseen even in the
unlabeled set, improving the robustness of the segmenter.
4.4. Evaluation metrics
We evaluate the accuracy of each segmentation method
in terms of Dice score [23], which quantiﬁes the overlap be-
tween two anatomical regions. A Dice score of 1 indicates
perfectly overlapping regions, while 0 indicates no overlap.
The predicted segmentation labels are evaluated relative to
anatomical labels generated using FreeSurfer [27].
4.5. Results
4.5.1
Segmentation performance
Table 1 shows the segmentation accuracy attained by each
method.
Our methods outperform all baselines in mean

--- Page 7 ---
Figure 4: Pairwise improvement in mean Dice score (with
the mean computed across all 30 anatomical labels) com-
pared to the SAS baseline, shown across all test subjects.
Figure 5: Pairwise improvement in mean Dice score (with
the mean computed across all 30 anatomical labels) com-
pared to the SAS baseline, shown for each test subject.
Subjects are sorted by the Dice improvement of ours-
indep+rand-aug over SAS.
Dice score across all 30 evaluation labels, showing signif-
icant improvements over the next best baselines rand-aug
(p < 1e-15 using a paired t-test) and SAS-aug (p < 1e-20).
In Figs. 4 and 5, we compare each method to the single-
atlas segmentation baseline. Fig. 4 shows that our methods
attain the most improvement on average, and are more con-
sistent than hand-tuned random augmentation. Fig. 5 shows
that ours-indep + rand-aug is consistently better than each
baseline on every test subject. Ours-indep alone is always
better than SAS-aug and SAS, and is better than rand-aug on
95 of the 100 test scans.
Fig. 6 shows that rand-aug improves Dice over SAS on
large anatomical structures, but is detrimental for smaller
ones. In contrast, our methods produce consistent improve-
ments over SAS and SAS-aug across all structures. We show
several examples of segmented hippocampi in Fig. 7.
4.5.2
Synthesized images
Our independent spatial and appearance models enable the
synthesis of a wide variety of brain appearances. Fig. 8
shows some examples where combining transformations
produces realistic results with accurate labels.
5. Discussion
Why do we outperform single-atlas segmentation?
Our
methods rely on the same spatial registration model that is
used for SAS and SAS-aug. Both ours-coupled and SAS-aug
augment the segmenter training set with 100 new images.
To understand why our method produces better segmen-
tations, we examine the augmented images. Our method
warps the image in the same way as the labels, ensuring
that the warped labels match the transformed image. On the
other hand, SAS-aug applies the warped labels to the origi-
nal image, so any errors or noise in the registration results
in a mis-labeled new training example for the segmenter.
Fig. 9 highlights examples where our method synthesizes
image texture within the hippocampus label that is more
consistent with the texture of the ground truth hippocam-
pus, resulting in a more useful synthetic training example.
Extensions
Our framework lends itself to several plausi-
ble future extensions. In Section 3.1, we discussed the use
of an approximate inverse deformation function for learning
the appearance transformation in the reference frame of the
atlas. Rather than learning a separate inverse spatial trans-
form model, in the future we will leverage existing work in
diffeomorphic registration [3, 5, 10, 20, 81].
We sample transformations from a discrete set of spatial
and appearance transformations. This could be extended to
span the space of transformations more richly, e.g., through
interpolation between transformations, or using composi-
tions of transformations.
We demonstrated our approach on brain MRIs. Since
the method uses no brain- or MRI-speciﬁc information, it is
feasible to extend it to other anatomy or imaging modalities,
such as CT.
6. Conclusion
We presented a learning-based method for data augmen-
tation, and demonstrated it on one-shot medical image seg-
mentation.
We start with one labeled image and a set of unlabeled
examples. Using learning-based registration methods, we
model the set of spatial and appearance transformations be-
tween the labeled and unlabeled examples. These transfor-
mations capture effects such as non-linear deformations and
variations in imaging intensity. We synthesize new labeled
examples by sampling transformations and applying them

--- Page 8 ---
Figure 6: Segmentation accuracy of each method across various brain structures. Labels are sorted by the volume occupied
by each structure in the atlas (shown in parentheses), and labels consisting of left and right structures (e.g., Hippocampus)
are combined. We abbreviate the labels: white matter (WM), cortex (CX), ventricle (vent), and cerebrospinal ﬂuid (CSF).
Figure 7: Hippocampus segmentation predictions for two
test subjects (rows). Our method (column 2) produces more
accurate segmentations than the baselines (columns 3 and
4).
to the labeled example, producing a wide variety of realistic
new images.
We use these synthesized examples to train a supervised
segmentation model. The segmenter out-performs existing
one-shot segmentation methods on every example in our
test set, approaching the performance of a fully supervised
model. This framework enables segmentation in many ap-
plications, such as clinical settings where time constraints
permit the manual annotation of only a few scans.
In summary, this work shows that:
• learning independent models of spatial and appear-
ance transformations from unlabeled images enables
the synthesis of diverse and realistic labeled examples,
and
• these synthesized examples can be used to train a seg-
mentation model that out-performs existing methods in
a one-shot scenario.
References
[1] M. Abadi et al.
Tensorﬂow: Large-scale machine learn-
ing on heterogeneous distributed systems.
arXiv preprint
arXiv:1603.04467, 2016. 5
[2] Z. Akkus, A. Galimzianova, A. Hoogi, D. L. Rubin, and B. J.
Erickson. Deep learning for brain mri segmentation: state
of the art and future directions. Journal of digital imaging,
30(4):449–459, 2017. 1, 2
[3] J. Ashburner. A fast diffeomorphic image registration algo-
rithm. Neuroimage, 38(1):95–113, 2007. 7
[4] J. Ashburner and K. Friston. Voxel-based morphometry-the
methods. Neuroimage, 11:805–821, 2000. 2
[5] B. B. Avants, C. L. Epstein, M. Grossman, and J. C. Gee.
Symmetric diffeomorphic image registration with cross-
correlation: evaluating automated labeling of elderly and
neurodegenerative brain. Medical image analysis, 12(1):26–
41, 2008. 7
[6] C. Baillard, P. Hellier, and C. Barillot. Segmentation of brain
3d mr images using level sets and dense registration. Medical
image analysis, 5(3):185–194, 2001. 2, 6

--- Page 9 ---
+
+
Atlas and labels 
𝑥, 𝑙𝑥
Sampled appearance and spatial 
transform targets
𝑦(𝑗), 𝑦(𝑖)
Synthesized image and labels
ො𝑦(𝑖,𝑗), መ𝑙(𝑖,𝑗)
Figure 8: Since we model spatial and appearance transformations independently, we are able to synthesize a variety of com-
bined effects. We show some examples synthesized using transformations learned from the training set; these transformations
form the bases of our augmentation model. The top row shows a synthetic image where the appearance transformation pro-
duced a darkening effect, and the spatial transformation shrunk the ventricles and widened the whole brain. In the second
row, the atlas is brightened and the ventricles are enlarged.
Figure 9: Synthetic training examples produced by SAS-aug
(column 2) and ours-coupled (column 3). When the spatial
model (used by both methods) produces imperfect warped
labels, SAS-aug pairs the warped label with incorrect image
textures. Our method still produces a useful training exam-
ple by matching the synthesized image texture to the label.
[7] R. Bajcsy and S. Kovacic. Multiresolution elastic matching.
Computer Vision, Graphics, and Image Processing, 46:1–21,
1989. 2
[8] G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, and
A. V. Dalca. An unsupervised learning model for deformable
medical image registration. In Proceedings of the IEEE Con-
ference on Computer Vision and Pattern Recognition, pages
9252–9260, 2018. 2, 4, 5, 6
[9] G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, and
A. V. Dalca.
Voxelmorph: a learning framework for de-
formable medical image registration. IEEE transactions on
medical imaging, 2019. 2, 4
[10] M. F. Beg, M. I. Miller, A. Trouv´e, and L. Younes. Comput-
ing large deformation metric mappings via geodesic ﬂows of
diffeomorphisms. International journal of computer vision,
61(2):139–157, 2005. 7
[11] S. Caelles, K.-K. Maninis, J. Pont-Tuset, L. Leal-Taix´e,
D. Cremers, and L. Van Gool. One-shot video object seg-
mentation. In CVPR 2017. IEEE, 2017. 3
[12] F. Chollet et al.
Keras.
https://github.com/
fchollet/keras, 2015. 5
[13] C. Ciofolo and C. Barillot. Atlas-based segmentation of 3d
cerebral structures with competitive level sets and fuzzy con-
trol. Medical image analysis, 13(3):456–470, 2009. 2, 6
[14] D. Coelho de Castro and B. Glocker. Nonparametric den-
sity ﬂows for mri intensity normalisation. In International
Conference on Medical Image Computing and Computer As-
sisted Intervention, pages 206–214, 09 2018. 2
[15] T. F. Cootes, C. Beeston, G. J. Edwards, and C. J. Taylor.
A uniﬁed framework for atlas matching using active appear-
ance models. In Biennial International Conference on In-
formation Processing in Medical Imaging, pages 322–333.
Springer, 1999. 2

--- Page 10 ---
[16] T. F. Cootes, G. J. Edwards, and C. J. Taylor. Active ap-
pearance models. IEEE Transactions on Pattern Analysis &
Machine Intelligence, (6):681–685, 2001. 2, 4
[17] T. F. Cootes and C. J. Taylor. Statistical models of appear-
ance for medical image analysis and computer vision.
In
Medical Imaging 2001: Image Processing, volume 4322,
pages 236–249. International Society for Optics and Photon-
ics, 2001. 2, 4
[18] E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le.
Autoaugment: Learning augmentation policies from data.
arXiv preprint arXiv:1805.09501, 2018. 3
[19] A. Dagley, M. LaPoint, W. Huijbers, T. Hedden, D. G.
McLaren, J. P. Chatwal, K. V. Papp, R. E. Amariglio,
D. Blacker, D. M. Rentz, et al. Harvard aging brain study:
dataset and accessibility. NeuroImage, 144:255–258, 2017.
5
[20] A. V. Dalca, G. Balakrishnan, J. Guttag, and M. R. Sabuncu.
Unsupervised learning for fast probabilistic diffeomorphic
registration. In International Conference on Medical Image
Computing and Computer-Assisted Intervention, pages 729–
738. Springer, 2018. 2, 4, 7
[21] A. V. Dalca, J. Guttag, and M. R. Sabuncu.
Anatomical
priors in convolutional networks for unsupervised biomed-
ical segmentation. In Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, pages 9290–
9299, 2018. 2
[22] B. M. Dawant, S. L. Hartmann, J.-P. Thirion, F. Maes,
D. Vandermeulen, and P. Demaerel. Automatic 3-d segmen-
tation of internal structures of the head in mr images using
a combination of similarity and free-form transformations. i.
methodology and validation on normal subjects. IEEE trans-
actions on medical imaging, 18(10):909–916, 1999. 2, 6
[23] L. R. Dice. Measures of the amount of ecologic association
between species. Ecology, 26(3):297–302, 1945. 6
[24] N. Dong and E. P. Xing. Few-shot semantic segmentation
with prototype learning. In BMVC, volume 3, page 4, 2018.
3
[25] A. Dosovitskiy, P. Fischer, J. T. Springenberg, M. Ried-
miller, and T. Brox.
Discriminative unsupervised feature
learning with exemplar convolutional neural networks. IEEE
transactions on pattern analysis and machine intelligence,
38(9):1734–1747, 2016. 1, 3
[26] Z. Eaton-Rosen, F. Bragman, S. Ourselin, and M. J. Cardoso.
Improving data augmentation for medical image segmenta-
tion. In International Conference on Medical Imaging with
Deep Learning, 2018. 1
[27] B. Fischl. Freesurfer. Neuroimage, 62(2):774–781, 2012. 5,
6
[28] M. A. Frost and R. Goebel. Measuring structural–functional
correspondence:
spatial variability of specialised brain
regions after macro-anatomical alignment.
Neuroimage,
59(2):1369–1381, 2012. 2
[29] P.-A. Ganaye, M. Sdika, and H. Benoit-Cattin.
Semi-
supervised learning for segmentation under semantic con-
straint. In International Conference on Medical Image Com-
puting and Computer-Assisted Intervention, pages 595–602.
Springer, 2018. 2
[30] R. L. Gollub et al.
The mcic collection: a shared repos-
itory of multi-modal, multi-site brain image data from a
clinical investigation of schizophrenia.
Neuroinformatics,
11(3):367–388, 2013. 5
[31] S. Hauberg, O. Freifeld, A. B. L. Larsen, J. Fisher, and
L. Hansen. Dreaming more data: Class-dependent distribu-
tions over diffeomorphisms for learned data augmentation.
In Artiﬁcial Intelligence and Statistics, pages 342–350, 2016.
2, 3
[32] P. Hellier and C. Barillot. A hierarchical parametric algo-
rithm for deformable multimodal image registration. Com-
puter Methods and Programs in Biomedicine, 75(2):107–
115, 2004. 2, 6
[33] A. J. Holmes et al. Brain genomics superstruct project ini-
tial data release with structural, functional, and behavioral
measures. Scientiﬁc data, 2, 2015. 5
[34] G. Huang, Z. Liu, K. Q. Weinberger, and L. van der Maaten.
Densely connected convolutional networks. arXiv preprint
arXiv:1608.06993, 2016. 3
[35] Z. Hussain, F. Gimenez, D. Yi, and D. Rubin. Differential
data augmentation techniques for medical imaging classiﬁ-
cation tasks. In AMIA Annual Symposium Proceedings, vol-
ume 2017, page 979. American Medical Informatics Associ-
ation, 2017. 6
[36] J. E. Iglesias and M. R. Sabuncu. Multi-atlas segmentation
of biomedical images: a survey. Medical image analysis,
24(1):205–219, 2015. 2
[37] S. D. Jain, B. Xiong, and K. Grauman. Fusionseg: Learn-
ing to combine motion and appearance for fully automatic
segmention of generic objects in videos.
In Proc. CVPR,
volume 1, 2017. 3
[38] M. J. Jones and T. Poggio.
Multidimensional morphable
models: A framework for representing and matching ob-
ject classes.
International Journal of Computer Vision,
29(2):107–131, 1998. 2, 4
[39] T. Joyce, A. Chartsias, and S. A. Tsaftaris. Deep multi-class
segmentation without ground-truth labels. 2018. 2
[40] K. Kamnitsas, C. Ledig, V. F. Newcombe, J. P. Simpson,
A. D. Kane, D. K. Menon, D. Rueckert, and B. Glocker. Efﬁ-
cient multi-scale 3d cnn with fully connected crf for accurate
brain lesion segmentation. Medical image analysis, 36:61–
78, 2017. 2, 6
[41] A. Klein and J. Hirsch. Mindboggle: a scatterbrained ap-
proach to automate brain labeling. NeuroImage, 24(2):261–
280, 2005. 2, 6
[42] J. Krebs et al. Robust non-rigid registration through agent-
based action learning. In International Conference on Med-
ical Image Computing and Computer-Assisted Intervention
(MICCAI), pages 344–352. Springer, 2017. 2
[43] A. Krizhevsky, I. Sutskever, and G. E. Hinton.
Imagenet
classiﬁcation with deep convolutional neural networks. In
Advances in neural information processing systems, pages
1097–1105, 2012. 3
[44] E. G. Learned-Miller.
Data driven image models through
continuous joint alignment. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 28(2):236–250, 2006. 2

--- Page 11 ---
[45] K. K. Leung, M. J. Clarkson, J. W. Bartlett, S. Clegg, C. R.
Jack Jr, M. W. Weiner, N. C. Fox, S. Ourselin, A. D. N. Ini-
tiative, et al. Robust atrophy rate measurement in alzheimer’s
disease using multi-site serial mri:
tissue-speciﬁc inten-
sity normalization and parameter selection.
Neuroimage,
50(2):516–523, 2010. 1, 2
[46] D. S. Marcus et al. Open access series of imaging studies
(oasis): cross-sectional mri data in young, middle aged, non-
demented, and demented older adults. Journal of cognitive
neuroscience, 19(9):1498–1507, 2007. 5
[47] K. Marek et al. The parkinson progression marker initiative.
Progress in neurobiology, 95(4):629–635, 2011. 5
[48] A. D. Martino et al. The autism brain imaging data exchange:
towards a large-scale evaluation of the intrinsic brain ar-
chitecture in autism. Molecular psychiatry, 19(6):659–667,
2014. 5
[49] M. P. Milham et al. The ADHD-200 consortium: a model to
advance the translational potential of neuroimaging in clin-
ical neuroscience. Frontiers in systems neuroscience, 6:62,
2012. 5
[50] E. G. Miller, N. E. Matsakis, and P. A. Viola.
Learning
from one example through shared densities on transforms.
In Proceedings IEEE Conference on Computer Vision and
Pattern Recognition. CVPR 2000 (Cat. No. PR00662), vol-
ume 1, pages 464–471. IEEE, 2000. 2
[51] F. Milletari, N. Navab, and S.-A. Ahmadi.
V-net: Fully
convolutional neural networks for volumetric medical im-
age segmentation. In 3D Vision (3DV), 2016 Fourth Inter-
national Conference on, pages 565–571. IEEE, 2016. 3, 6
[52] S. C. Mitchell, J. G. Bosch, B. P. Lelieveldt, R. J. Van der
Geest, J. H. Reiber, and M. Sonka. 3-d active appearance
models: segmentation of cardiac mr and ultrasound images.
IEEE transactions on medical imaging, 21(9):1167–1178,
2002. 2
[53] P. Moeskops, M. A. Viergever, A. M. Mendrik, L. S.
de Vries, M. J. Benders, and I. Iˇsgum. Automatic segmen-
tation of mr brain images with a convolutional neural net-
work. IEEE transactions on medical imaging, 35(5):1252–
1261, 2016. 1, 2
[54] S. G. Mueller et al.
Ways toward an early diagnosis in
alzheimer’s disease: the alzheimer’s disease neuroimaging
initiative (adni). Alzheimer’s & Dementia, 1(1):55–66, 2005.
5
[55] A. Oliveira, S. Pereira, and C. A. Silva. Augmenting data
when training a cnn for retinal vessel segmentation: How to
warp? In Bioengineering (ENBENG), 2017 IEEE 5th Por-
tuguese Meeting on, pages 1–4. IEEE, 2017. 1
[56] B. Patenaude, S. M. Smith, D. N. Kennedy, and M. Jenkin-
son. A bayesian model of shape and appearance for subcor-
tical brain segmentation. Neuroimage, 56(3):907–922, 2011.
2
[57] S. Pereira, A. Pinto, V. Alves, and C. A. Silva. Brain tumor
segmentation using convolutional neural networks in mri im-
ages. IEEE transactions on medical imaging, 35(5):1240–
1251, 2016. 1, 2
[58] V. Potesil, T. Kadir, G. Platsch, and M. Brady. Personal-
ized graphical models for anatomical landmark localization
in whole-body medical images.
International Journal of
Computer Vision, 111(1):29–49, 2015. 2
[59] J. Rademacher, U. B¨urgel, S. Geyer, T. Schormann, A. Schle-
icher, H.-J. Freund, and K. Zilles. Variability and asymme-
try in the human precentral motor system: a cytoarchitec-
tonic and myeloarchitectonic brain mapping study. Brain,
124(11):2232–2258, 2001. 2
[60] K. Rakelly, E. Shelhamer, T. Darrell, A. A. Efros, and
S. Levine. Few-shot segmentation propagation with guided
networks. arXiv preprint arXiv:1806.07373, 2018. 3
[61] A. J. Ratner, H. R. Ehrenberg, Z. Hussain, J. Dunnmon, and
C. R´e. Learning to compose domain-speciﬁc transformations
for data augmentation.
arXiv preprint arXiv:1709.01643,
2017. 3
[62] M.-M. Roh´e et al. Svf-net: Learning deformable image reg-
istration using shape matching. In International Conference
on Medical Image Computing and Computer-Assisted Inter-
vention (MICCAI), pages 266–274. Springer, 2017. 2
[63] O. Ronneberger, P. Fischer, and T. Brox.
U-net: Convo-
lutional networks for biomedical image segmentation.
In
International Conference on Medical image computing and
computer-assisted intervention, pages 234–241. Springer,
2015. 1, 2, 3, 4, 6
[64] H. R. Roth, C. T. Lee, H.-C. Shin, A. Seff, L. Kim, J. Yao,
L. Lu, and R. M. Summers.
Anatomy-speciﬁc classiﬁca-
tion of medical images using deep convolutional nets. arXiv
preprint arXiv:1504.04003, 2015. 3, 6
[65] H. R. Roth, L. Lu, A. Farag, H.-C. Shin, J. Liu, E. B. Turk-
bey, and R. M. Summers.
Deeporgan: Multi-level deep
convolutional networks for automated pancreas segmenta-
tion.
In International conference on medical image com-
puting and computer-assisted intervention, pages 556–564.
Springer, 2015. 1, 2
[66] A. G. Roy, S. Conjeti, D. Sheet, A. Katouzian, N. Navab, and
C. Wachinger. Error corrective boosting for learning fully
convolutional networks with limited data. In International
Conference on Medical Image Computing and Computer-
Assisted Intervention, pages 231–239. Springer, 2017. 5
[67] D. Rueckert et al. Nonrigid registration using free-form de-
formation: Application to breast mr images. IEEE Transac-
tions on Medical Imaging, 18(8):712–721, 1999. 2
[68] M. R. Sabuncu, B. T. Yeo, K. Van Leemput, B. Fischl, and
P. Golland.
A generative model for image segmentation
based on label fusion. IEEE transactions on medical imag-
ing, 29(10):1714–1729, 2010. 2
[69] A. Shaban, S. Bansal, Z. Liu, I. Essa, and B. Boots. One-
shot learning for semantic segmentation.
arXiv preprint
arXiv:1709.03410, 2017. 3
[70] D. Shen and C. Davatzikos. Hammer: Hierarchical attribute
matching mechanism for elastic registration. IEEE Transac-
tions on Medical Imaging, 21(11):1421–1439, 2002. 2
[71] J. G. Sled, A. P. Zijdenbos, and A. C. Evans. A nonpara-
metric method for automatic correction of intensity nonuni-
formity in mri data. IEEE transactions on medical imaging,
17(1):87–97, 1998. 5
[72] H. Sokooti et al. Nonrigid image registration using multi-
scale 3d convolutional neural networks.
In International

--- Page 12 ---
Conference on Medical Image Computing and Computer-
Assisted Intervention (MICCAI), pages 232–239. Springer,
2017. 2
[73] R. Sridharan, A. V. Dalca, K. M. Fitzpatrick, L. Cloonan,
A. Kanakis, O. Wu, K. L. Furie, J. Rosand, N. S. Rost, and
P. Golland. Quantiﬁcation and analysis of large multimodal
clinical image studies: Application to stroke.
In Interna-
tional Workshop on Multimodal Brain Image Analysis, pages
18–30. Springer, 2013. 2, 5
[74] M. Styner, C. Brechbuhler, G. Szckely, and G. Gerig. Para-
metric estimate of intensity inhomogeneities applied to mri.
IEEE Trans. Med. Imaging, 19(3):153–165, 2000. 5
[75] Y.-H. Tsai, M.-H. Yang, and M. J. Black. Video segmenta-
tion via object ﬂow. In Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, pages 3899–
3908, 2016. 3
[76] D. C. Van Essen and D. L. Dierker.
Surface-based and
probabilistic atlases of primate cerebral cortex.
Neuron,
56(2):209–225, 2007. 2
[77] G. Vincent, G. Guillard, and M. Bowes. Fully automatic seg-
mentation of the prostate using active appearance models.
MICCAI Grand Challenge: Prostate MR Image Segmenta-
tion, 2012, 2012. 2
[78] H. Wang, J. W. Suh, S. R. Das, J. B. Pluta, C. Craige, and
P. A. Yushkevich. Multi-atlas segmentation with joint label
fusion. IEEE transactions on pattern analysis and machine
intelligence, 35(3):611–623, 2013. 2
[79] W. M. Wells, W. E. L. Grimson, R. Kikinis, and F. A. Jolesz.
Adaptive segmentation of mri data. IEEE transactions on
medical imaging, 15(4):429–442, 1996. 2
[80] X. Yang et al.
Quicksilver:
Fast predictive im-
age registration–a deep learning approach.
NeuroImage,
158:378–396, 2017. 2
[81] M. Zhang, R. Liao, A. V. Dalca, E. A. Turk, J. Luo, P. E.
Grant, and P. Golland. Frequency diffeomorphisms for efﬁ-
cient image registration. In International conference on in-
formation processing in medical imaging, pages 559–570.
Springer, 2017. 7
[82] W. Zhang, R. Li, H. Deng, L. Wang, W. Lin, S. Ji, and
D. Shen.
Deep convolutional neural networks for multi-
modality isointense infant brain image segmentation. Neu-
roImage, 108:214–224, 2015. 2
[83] A. Zlateski, R. Jaroensri, P. Sharma, and F. Durand.
On
the importance of label quality for semantic segmentation.
In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pages 1479–1487, 2018.
