# Panoptic Feature Pyramid Networks

**Authors**: Kirillov, Girshick, He, Dollar
**Year**: 2019
**arXiv**: 1901.02446
**Topic**: multiscale
**Relevance**: Unified multi-scale architecture for segmentation and classification

---


--- Page 1 ---
Panoptic Feature Pyramid Networks
Alexander Kirillov
Ross Girshick
Kaiming He
Piotr Doll´ar
Facebook AI Research (FAIR)
Abstract
The recently introduced panoptic segmentation task has
renewed our community’s interest in unifying the tasks of
instance segmentation (for thing classes) and semantic seg-
mentation (for stuff classes).
However, current state-of-
the-art methods for this joint task use separate and dis-
similar networks for instance and semantic segmentation,
without performing any shared computation. In this work,
we aim to unify these methods at the architectural level,
designing a single network for both tasks. Our approach
is to endow Mask R-CNN, a popular instance segmenta-
tion method, with a semantic segmentation branch using
a shared Feature Pyramid Network (FPN) backbone. Sur-
prisingly, this simple baseline not only remains effective for
instance segmentation, but also yields a lightweight, top-
performing method for semantic segmentation. In this work,
we perform a detailed study of this minimally extended ver-
sion of Mask R-CNN with FPN, which we refer to as Panop-
tic FPN, and show it is a robust and accurate baseline for
both tasks. Given its effectiveness and conceptual simplic-
ity, we hope our method can serve as a strong baseline and
aid future research in panoptic segmentation.
1. Introduction
Our community has witnessed rapid progress in seman-
tic segmentation, where the task is to assign each pixel a
class label (e.g. for stuff classes), and more recently in in-
stance segmentation, where the task is to detect and segment
each object instance (e.g. for thing classes). These advances
have been aided by simple yet powerful baseline methods,
including Fully Convolutional Networks (FCN) [41] and
Mask R-CNN [24] for semantic and instance segmentation,
respectively. These methods are conceptually simple, fast,
and ﬂexible, serving as a foundation for much of the sub-
sequent progress in these areas. In this work our goal is
to propose a similarly simple, single-network baseline for
the joint task of panoptic segmentation [30], a task which
encompasses both semantic and instance segmentation.
While conceptually straightforward, designing a sin-
gle network that achieves high accuracy for both tasks is
(a) Feature Pyramid Network
(b) Instance Segmentation Branch
(c) Semantic Segmentation Branch
Figure 1: Panoptic FPN: (a) We start with an FPN back-
bone [36], widely used in object detection, for extracting
rich multi-scale features.
(b) As in Mask R-CNN [24],
we use a region-based branch on top of FPN for instance
segmentation. (c) In parallel, we add a lightweight dense-
prediction branch on top of the same FPN features for se-
mantic segmentation. This simple extension of Mask R-
CNN with FPN is a fast and accurate baseline for both tasks.
challenging as top-performing methods for the two tasks
have many differences. For semantic segmentation, FCNs
with specialized backbones enhanced by dilated convolu-
tions [57, 10] dominate popular leaderboards [18, 14]. For
instance segmentation, the region-based Mask R-CNN [24]
with a Feature Pyramid Network (FPN) [36] backbone
has been used as a foundation for all top entries in re-
cent recognition challenges [37, 60, 43]. While there have
been attempts to unify semantic and instance segmentation
[46, 1, 9], the specialization currently necessary to achieve
top performance in each was perhaps inevitable given their
parallel development and separate benchmarks.
Given the architectural differences in these top methods,
one might expect compromising accuracy on either instance
or semantic segmentation is necessary when designing a
single network for both tasks. Instead, we show a simple,
ﬂexible, and effective architecture that can match accuracy
for both tasks using a single network that simultaneously
generates region-based outputs (for instance segmentation)
and dense-pixel outputs (for semantic segmentation).
1
arXiv:1901.02446v2  [cs.CV]  10 Apr 2019

--- Page 2 ---
Figure 2: Panoptic FPN results on COCO (top) and Cityscapes (bottom) using a single ResNet-101-FPN network.
Our approach starts with the FPN [36] backbone pop-
ular for instance-level recognition [24] and adds a branch
for performing semantic segmentation in parallel with the
existing region-based branch for instance segmentation, see
Figure 1. We make no changes to the FPN backbone when
adding the dense-prediction branch, making it compatible
with existing instance segmentation methods. Our method,
which we call Panoptic FPN for its ability to generate both
instance and semantic segmentations via FPN, is easy to im-
plement given the Mask R-CNN framework [23].
While Panoptic FPN is an intuitive extension of Mask R-
CNN with FPN, properly training the two branches for si-
multaneous region-based and dense-pixel prediction is im-
portant for good results.
We perform careful studies in
the joint setting for how to balance the losses for the two
branches, construct minibatches effectively, adjust learning
rate schedules, and perform data augmentation. We also ex-
plore various designs for the semantic segmentation branch
(all other network components follow Mask R-CNN). Over-
all, while our approach is robust to exact design choices,
properly addressing these issues is key for good results.
When trained for each task independently, our method
achieves excellent results for both instance and semantic
segmentation on both COCO [37] and Cityscapes [14]. For
instance segmentation, this is expected as our method in this
case is equivalent to Mask R-CNN. For semantic segmen-
tation, our simple dense-prediction branch attached to FPN
yields accuracy on par with the latest dilation-based meth-
ods, such as the recent DeepLabV3+ [12].
For panoptic segmentation [30], we demonstrate that
with proper training, using a single FPN for solving both
tasks simultaneously yields accuracy equivalent to training
two separate FPNs, with roughly half the compute. With
the same compute, a joint network for the two tasks out-
performs two independent networks by a healthy margin.
Example panoptic segmentation results are shown in Fig. 2.
Panoptic FPN is memory and computationally efﬁcient,
incurring only a slight overhead over Mask R-CNN. By
avoiding the use of dilation, which has high overhead, our
method can use any standard top-performing backbone (e.g.
a large ResNeXt [55]). We believe this ﬂexibility, together
with the fast training and inference speeds of our method,
will beneﬁt future research on panoptic segmentation.
We used a preliminary version of our model (semantic
segmentation branch only) as the foundation of the ﬁrst-
place winning entry in the COCO Stuff Segmentation [6]
track in 2017. This single-branch model has since been
adopted and generalized by several entries in the 2018
COCO and Mapillary Challenges1, showing its ﬂexibility
and effectiveness. We hope our proposed joint panoptic seg-
mentation baseline is similarly impactful.
2. Related Work
Panoptic segmentation: The joint task of thing and stuff
segmentation has a rich history, including early work on
scene parsing [51], image parsing [52], and holistic scene
understanding [56]. With the recent introduction of the joint
panoptic segmentation task [30], which includes a simple
task speciﬁcation and carefully designed task metrics, there
has been a renewed interest in the joint task.
This year’s COCO and Mapillary Recognition Chal-
lenge [37, 43] featured panoptic segmentation tracks that
proved popular. However, every competitive entry in the
panoptic challenges used separate networks for instance
and semantic segmentation, with no shared computation.1
Our goal is to design a single network effective for both
tasks that can serve as a baseline for future work.
1For details of not yet published winning entries in the 2018 COCO and
Mapillary Recognition Challenge please see: http://cocodataset.
org/workshop/coco-mapillary-eccv-2018.html. TRI-ML
used separate networks for the challenge but a joint network in their recent
updated tech report [33] (which cites a preliminary version of our work).
2

--- Page 3 ---
Instance segmentation: Region-based approaches to ob-
ject detection, including the Slow/Fast/Faster/Mask R-CNN
family [22, 21, 48, 24], which apply deep networks on
candidate object regions, have proven highly successful.
All recent winners of the COCO detection challenges have
built on Mask R-CNN [24] with FPN [36], including in
2017 [39, 45] and 2018.1 Recent innovations include Cas-
cade R-CNN [7], deformable convolution [15], and sync
batch norm [45]. In this work, the original Mask R-CNN
with FPN serves as the starting point for our baseline, giving
us excellent instance segmentation performance, and mak-
ing our method fully compatible with these recent advances.
An alternative to region-based instance segmentation is
to start with a pixel-wise semantic segmentation and then
perform grouping to extract instances [31, 38, 1]. This di-
rection is innovative and promising. However, these meth-
ods tend to use separate networks to predict the instance-
level information (e.g., [31, 1, 38] use a separate network to
predict instance edges, bounding boxes, and object break-
points, respectively). Our goal is to design a single network
for the joint task. Another interesting direction is to use
position-sensitive pixel labeling [35] to encode instance in-
formation fully convolutionally; [46, 9] build on this.
Nevertheless, region-based approaches remain dominant
on detection leaderboards [37, 60, 43]. While this moti-
vates us to start with a region-based approach to instance
segmentation, our approach would be fully compatible with
a dense-prediction branch for instance segmentation.
Semantic segmentation: FCNs [41] serve as the founda-
tion of modern semantic segmentation methods. To increase
feature resolution, which is necessary for generating high-
quality results, recent top methods [12, 58, 5, 59] rely heav-
ily on the use of dilated convolution [57] (also known as
atrous convolution [10]). While effective, such an approach
can substantially increase compute and memory, limiting
the type of backbone network that can be used. To keep this
ﬂexibility, and more importantly to maintain compatibility
with Mask R-CNN, we opt for a different approach.
As an alternative to dilation, an encoder-decoder [2] or
‘U-Net’ [49] architecture can be used to increase feature
resolution [26, 44, 20, 47]. Encoder-decoders progressively
upsample and combine high-level features from a feedfor-
ward network with features from lower-levels, ultimately
generating semantically meaningful, high-resolution fea-
tures (see Figure 5). While dilated networks are currently
more popular and dominate leaderboards, encoder-decoders
have also been used for semantic segmentation [49, 2, 20].
In our work we adopt an encoder-decoder framework,
namely FPN [36]. In contrast to ‘symmetric’ decoders [49],
FPN uses a lightweight decoder (see Fig. 5). FPN was de-
signed for instance segmentation, and it serves as the default
backbone for Mask R-CNN. We show that without changes,
FPN can also be highly effective for semantic segmentation.
Multi-task learning: Our approach is related to multi-task
learning. In general, using a single network to solve mul-
tiple diverse tasks degrades performance [32], but various
strategies can mitigate this [29, 42]. For related tasks, there
can be gains from multi-task learning, e.g. the box branch in
Mask R-CNN beneﬁts from the mask branch [24], and joint
detection and semantic segmentation of thing classes also
shows gains [3, 8, 17, 46]. Our work studies the beneﬁts of
multi-task training for stuff and thing segmentation.
3. Panoptic Feature Pyramid Network
Our approach, Panoptic FPN, is a simple, single-network
baseline whose goal is to achieve top performance on both
instance and semantic segmentation, and their joint task:
panoptic segmentation [30]. Our design principle is to start
from Mask R-CNN with FPN, a strong instance segmenta-
tion baseline, and make minimal changes to also generate a
semantic segmentation dense-pixel output (see Figure 1).
3.1. Model Architecture
Feature Pyramid Network: We begin by brieﬂy review-
ing FPN [36]. FPN takes a standard network with features
at multiple spatial resolutions (e.g., ResNet [25]), and adds
a light top-down pathway with lateral connections, see Fig-
ure 1a. The top-down pathway starts from the deepest layer
of the network and progressively upsamples it while adding
in transformed versions of higher-resolution features from
the bottom-up pathway. FPN generates a pyramid, typically
with scales from 1/32 to 1/4 resolution, where each pyramid
level has the same channel dimension (256 by default).
Instance segmentation branch: The design of FPN, and
in particular the use of the same channel dimension for all
pyramid levels, makes it easy to attach a region-based object
detector like Faster R-CNN [48]. Faster R-CNN performs
region of interest (RoI) pooling on different pyramid levels
and applies a shared network branch to predict a reﬁned box
and class label for each region. To output instance segmen-
tations, we use Mask R-CNN [24], which extends Faster
R-CNN by adding an FCN branch to predict a binary seg-
mentation mask for each candidate region, see Figure 1b.
Panoptic FPN: As discussed, our approach is to modify
Mask R-CNN with FPN to enable pixel-wise semantic seg-
mentation prediction. However, to achieve accurate predic-
tions, the features used for this task should: (1) be of suit-
ably high resolution to capture ﬁne structures, (2) encode
sufﬁciently rich semantics to accurately predict class labels,
and (3) capture multi-scale information to predict stuff re-
gions at multiple resolutions. Although FPN was designed
for object detection, these requirements – high-resolution,
rich, multi-scale features – identify exactly the characteris-
tics of FPN. We thus propose to attach to FPN a simple and
fast semantic segmentation branch, described next.
3

--- Page 4 ---
conv→2×→conv→2×→conv→2×
conv→4×
256 × 1/16
256 × 1/8
256 × 1/4
C × 1
128 × 1/4
128 × 1/4
128 × 1/4
1/32
conv→2×→conv→2×
conv→2×
conv
128 × 1/4
Figure 3: Semantic segmentation branch. Each FPN level
(left) is upsampled by convolutions and bilinear upsampling
until it reaches 1/4 scale (right), theses outputs are then
summed and ﬁnally transformed into a pixel-wise output.
Semantic segmentation branch: To generate the seman-
tic segmentation output from the FPN features, we propose
a simple design to merge the information from all levels of
the FPN pyramid into a single output. It is illustrated in
detail in Figure 3. Starting from the deepest FPN level (at
1/32 scale), we perform three upsampling stages to yield
a feature map at 1/4 scale, where each upsampling stage
consists of 3×3 convolution, group norm [54], ReLU, and
2× bilinear upsampling. This strategy is repeated for FPN
scales 1/16, 1/8, and 1/4 (with progressively fewer upsam-
pling stages). The result is a set of feature maps at the same
1/4 scale, which are then element-wise summed. A ﬁnal
1×1 convolution, 4× bilinear upsampling, and softmax are
used to generate the per-pixel class labels at the original im-
age resolution. In addition to stuff classes, this branch also
outputs a special ‘other’ class for all pixels belonging to ob-
jects (to avoid predicting stuff classes for such pixels).
Implementation details: We use a standard FPN conﬁgu-
ration with 256 output channels per scale, and our seman-
tic segmentation branch reduces this to 128 channels. For
the (pre-FPN) backbone, we use ResNet/ResNeXt [25, 55]
models pre-trained on ImageNet [50] using batch norm
(BN) [28]. When used in ﬁne-tuning, we replace BN with a
ﬁxed channel-wise afﬁne transformation, as is typical [25].
3.2. Inference and Training
Panoptic inference: The panoptic output format [30] re-
quires each output pixel to be assigned a single class label
(or void) and instance id (the instance id is ignored for stuff
classes). As the instance and semantic segmentation outputs
from Panoptic FPN may overlap; we apply the simple post-
processing proposed in [30] to resolve all overlaps. This
post-processing is similar in spirit to non-maximum sup-
pression and operates by: (1) resolving overlaps between
different instances based on their conﬁdence scores, (2) re-
solving overlaps between instance and semantic segmenta-
tion outputs in favor of instances, and (3) removing any stuff
regions labeled ‘other’ or under a given area threshold.
1/32
1/4
0.3
1.5
3.0
4.5
6.0
Mult-adds ×1012
FPN
Dilation 
Symmetric decoder
1/16 
1/8 
Output scale
1/32
1/4
0.5
2.0
4.0
6.0
Activations ×109
1/16 
1/8 
Output scale
Figure 4: Backbone architecture efﬁciency.
We com-
pare methods for increasing feature resolution for seman-
tic segmentation, including dilated networks, symmetric de-
coders, and FPN, see Figure 5. We count multiply-adds and
memory used when applying ResNet-101 to a 2 megapixel
image.
FPN at output scale 1/4 is similar computation-
ally to dilation-16 (1/16 resolution output), but produces a
4× higher resolution output. Increasing resolution to 1/8
via dilation uses a further ∼3× more compute and memory.
Joint training: During training the instance segmentation
branch has three losses [24]: Lc (classiﬁcation loss), Lb
(bounding-box loss), and Lm (mask loss). The total instance
segmentation loss is the sum of these losses, where Lc and
Lb are normalized by the number of sampled RoIs and Lm is
normalized by the number of foreground RoIs. The seman-
tic segmentation loss, Ls, is computed as a per-pixel cross
entropy loss between the predicted and the ground-truth la-
bels, normalized by the number of labeled image pixels.
We have observed that the losses from these two
branches have different scales and normalization policies.
Simply adding them degrades the ﬁnal performance for
one of the tasks. This can be corrected by a simple loss
re-weighting between the total instance segmentation loss
and the semantic segmentation loss. Our ﬁnal loss is thus:
L = λi (Lc + Lb + Lm) + λsLs. By tuning λi and λs it
is possible to train a single model that is comparable to two
separate task-speciﬁc models, but at about half the compute.
3.3. Analysis
Our motivation for predicting semantic segmentation us-
ing FPN is to create a simple, single-network baseline
that can perform both instance and semantic segmentation.
However, it is also interesting to consider the memory and
computational footprint of our approach relative to model
architectures popular for semantic segmentation. The most
common designs that produce high-resolution outputs are
dilated convolution (Figure 5b) and symmetric encoder-
decoder models that have a mirror image decoder with lat-
eral connections (Figure 5c). While our primary motiva-
tion is compatibility with Mask R-CNN, we note that FPN
is much lighter than a typically used dilation-8 network,
∼2× more efﬁcient than the symmetric encoder-decoder,
and roughly equivalent to a dilation-16 network (while pro-
ducing a 4× higher resolution output). See Figure 4.
4

--- Page 5 ---
b5×1024×1/32
b4×512×1/16
b3×256×1/8
(a) Original
(b) +Dilation
(c) +Symmetric Decoder
(d) +Assymetric Decoder (FPN)
b5×1024×1/8
b4×512×1/8
b3×256×1/8
b5×1024×1/32
b4×512×1/16
b3×256×1/8
b5×1024×1/32
b4×512×1/16
b3×256×1/8
b5×1024×1/32
b4×512×1/16
b3×256×1/8
1×256×1/32
1×256×1/16
1×256×1/8
Figure 5: Backbone architectures for increasing feature resolution. (a) A standard convolutional network (dimensions are
denoted as #blocks×#channels×resolution). (b) A common approach is to reduce the stride of select convolutions and use
dilated convolutions after to compensate. (c) A U-Net [49] style network uses a symmetric decoder that mirrors the bottom-up
pathway, but in reverse. (d) FPN can be seen as an asymmetric, lightweight decoder whose top-down pathway has only one
block per stage and uses a shared channel dimension. For a comparison of the efﬁciency of these models, please see Figure 4.
4. Experiments
Our goal is to demonstrate that our approach, Panoptic
FPN, can serve as a simple and effective single-network
baseline for instance segmentation, semantic segmentation,
and their joint task of panoptic segmentation [30]. For in-
stance segmentation, this is expected, since our approach
extends Mask R-CNN with FPN. For semantic segmenta-
tion, as we simply attach a lightweight dense-pixel predic-
tion branch (Figure 3) to FPN, we need to demonstrate it
can be competitive with recent methods. Finally, we must
show that Panoptic FPN can be trained in a multi-task set-
ting without loss in accuracy on the individual tasks.
We therefore begin our analysis by testing our approach
for semantic segmentation (we refer to this single-task vari-
ant as Semantic FPN). Surprisingly, this simple model
achieves competitive semantic segmentation results on the
COCO [37] and Cityscapes [14] datasets. Next, we ana-
lyze the integration of the semantic segmentation branch
with Mask R-CNN, and the effects of joint training. Lastly,
we show results for panoptic segmentation, again on COCO
and Cityscapes. Qualitative results are shown in Figures 2
and 6. We describe the experimental setup next.
4.1. Experimental Setup
COCO: The COCO dataset [37] was developed with a fo-
cus on instance segmentation, but more recently stuff an-
notations were added [6]. For instance segmentation, we
use the 2017 data splits with 118k/5k/20k train/val/test im-
ages and 80 thing classes. For semantic segmentation, we
use the 2017 stuff data with 40k/5k/5k splits and 92 stuff
classes. Finally, panoptic segmentation [30] uses all 2017
COCO images with 80 thing and 53 stuff classes annotated.
Cityscapes: Cityscapes [14] is an ego-centric street-scene
dataset. It has 5k high-resolution images (1024×2048 pix-
els) with ﬁne pixel-accurate annotations: 2975 train, 500
val, and 1525 test. An additional 20k images with coarse
annotations are available, we do not use these in our exper-
iments. There are 19 classes, 8 with instance-level masks.
Single-task metrics: We report standard semantic and in-
stance segmentation metrics for the individual tasks using
evaluation code provided by each dataset. For semantic seg-
mentation, the mIoU (mean Intersection-over-Union) [18]
is the primary metric on both COCO and Cityscapes. We
also report fIoU (frequency weighted IoU) on COCO [6]
and iIoU (instance-level IoU) on Cityscapes [14]. For in-
stance segmentation, AP (average precision averaged over
categories and IoU thresholds) [37] is the primary metric
and AP50 and AP75 are selected supplementary metrics.
Panoptic segmentation metrics: We use PQ (panoptic
quality) as the default metric to measure Panoptic FPN per-
formance, for details see [30]. PQ captures both recognition
and segmentation quality, and treats both stuff and thing cat-
egories in a uniﬁed manner. This single, uniﬁed metric al-
lows us to directly compare methods. Additionally, we use
PQSt and PQTh to report stuff and thing performance sepa-
rately. Note that PQ is used to evaluate Panoptic FPN pre-
dictions after the post-processing merging procedure is ap-
plied to the outputs of the semantic and instance branches.
COCO training: We use the default Mask R-CNN 1×
training setting [23] with scale jitter (shorter image side in
[640, 800]). For semantic segmentation, we predict 53 stuff
classes plus a single ‘other’ class for all 80 thing classes.
Cityscapes training: We construct each minibatch from
32 random 512×1024 image crops (4 crops per GPU) af-
ter randomly scaling each image by 0.5 to 2.0×. We train
for 65k iterations starting with a learning rate of 0.01 and
dropping it by a factor of 10 at 40k and 55k iterations. This
differs from the original Mask R-CNN setup [24] but is ef-
fective for both instance and semantic segmentation. For the
largest backbones for semantic segmentation, we perform
color augmentation [40] and crop bootstrapping [5]. For
semantic segmentation, predicting all thing classes, rather
than a single ‘other’ label, performs better (for panoptic in-
ference we discard these predictions). Due to the high vari-
ance of the mIoU (up to 0.4), we report the median perfor-
mance of 5 trials of each experiment on Cityscapes.
5

--- Page 6 ---
backbone
mIoU
FLOPs
memory
DeeplabV3 [11]
ResNet-101-D8
77.8
1.9
1.9
PSANet101 [59]
ResNet-101-D8
77.9
2.0
2.0
Mapillary [5]
WideResNet-38-D8
79.4
4.3
1.7
DeeplabV3+ [12]
X-71-D16
79.6
0.5
1.9
Semantic FPN
ResNet-101-FPN
77.7
0.5
0.8
Semantic FPN
ResNeXt-101-FPN
79.1
0.8
1.4
(a) Cityscapes Semantic FPN. Performance is reported on the val set and
all methods use only ﬁne Cityscapes annotations for training. The back-
bone notation includes the dilated resolution ‘D’ (note that [12] uses both
dilation and an encoder-decoder backbone). All top-performing methods
other than ours use dilation. FLOPs (multiply-adds ×1012) and memory
(# activations ×109) are approximate but informative. For these larger
FPN models we train with color and crop augmentation. Our baseline is
comparable to state-of-the-art methods in accuracy and efﬁciency.
backbone
mIoU
fIoU
Vllab [13]
Stacked Hourglass
12.4
38.8
DeepLab VGG16 [10]
VGG-16
20.2
47.5
Oxford [4]
ResNeXt-101
24.1
50.6
G-RMI [19]
Inception ResNet v2
26.6
51.9
Semantic FPN
ResNeXt-152-FPN
28.8
55.7
(b) COCO-Stuff 2017 Challenge results. We submitted an early version
of Semantic FPN to the 2017 COCO Stuff Segmentation Challenge held
at ECCV (http://cocodataset.org/#stuff-2017). Our en-
try won ﬁrst place without ensembling, and we outperformed competing
methods by at least a 2 point margin on all reported metrics.
Width
Cityscapes
COCO
64
74.1
39.6
128
74.5
40.2
256
74.6
40.1
(c) Ablation (mIoU): Channel width
of 128 for the features in the seman-
tic branch strikes a good balance be-
tween accuracy and efﬁciency.
Aggr.
Cityscapes COCO
Sum
74.5
40.2
Concat
74.4
39.9
(d) Ablation (mIoU): Sum aggre-
gation of the feature maps in the
semantic branch is marginally bet-
ter and is more efﬁcient.
Table 1: Semantic Segmentation using FPN.
4.2. FPN for Semantic Segmentation
Cityscapes: We start by comparing our baseline Seman-
tic FPN to existing methods on the Cityscapes val split
in Table 1a. We compare to recent top-performing meth-
ods, but not to competition entires which typically use en-
sembling, COCO pre-training, test-time augmentation, etc.
Our approach, which is a minimal extension to FPN, is
able to achieve strong results compared to systems like
DeepLabV3+ [12], which have undergone many design iter-
ations. In terms of compute and memory, Semantic FPN is
lighter than typical dilation models, while yielding higher
resolution features (see Fig. 4). We note that adding dila-
tion into FPN could potentially yield further improvement
but is outside the scope of this work. Moreover, in our base-
line we deliberately avoid orthogonal architecture improve-
ments like Non-local [53] or SE [27], which would likely
yield further gains. Overall, these results demonstrate that
our approach is a strong baseline for semantic segmentation.
COCO: An earlier version of our approach won the 2017
COCO-Stuff challenge. Results are reported in Table 1b.
As this was an early design, the the semantic branch dif-
fered slightly (each upsampling module had two 3×3 conv
layers and ReLU before bilinear upscaling to the ﬁnal reso-
lution, and features were concatenated instead of summed,
please compare with Figure 3). As we will show in the ab-
lations shortly, results are fairly robust to the exact branch
design. Our competition entry was trained with color aug-
mentation [40] and at test time balanced the class distribu-
tion and used multi-scale inference. Finally, we note that
at the time we used a training schedule speciﬁc to semantic
segmentation similar to our Cityscapes schedule (but with
double learning rate and halved batch size).
Ablations: We perform a few ablations to analyze our pro-
posed semantic segmentation branch (shown in Figure 3).
For consistency with further experiments in our paper, we
use stuff annotations from the COCO Panoptic dataset
(which as discussed differ from those used for the COCO
Stuff competition).
Table 1c shows ResNet-50 Seman-
tic FPN with varying number of channels in the semantic
branch. We found that 128 strikes a good balance between
accuracy and efﬁciency. In Table 1d we compare element-
wise sum and concatenation for aggregating feature maps
from different FPN levels. While accuracy for both is com-
parable, summation is more efﬁcient. Overall we observe
that the simple architecture of the new dense-pixel labelling
branch is robust to exact design choices.
4.3. Multi-Task Training
Single-task performance of our approach is quite effec-
tive; for semantic segmentation the results in the previous
section demonstrate this, for instance segmentation this is
known as we start from Mask R-CNN. However, can we
jointly train for both tasks in a multi-task setting?
To combine our semantic segmentation branch with the
instance segmentation branch in Mask R-CNN, we need
to determine how to train a single, uniﬁed network. Pre-
vious work demonstrates that multi-task training is often
challenging and can lead to degraded results [32, 29]. We
likewise observe that for semantic or instance segmentation,
adding the secondary task can degrade the accuracy in com-
parison with the single-task baseline.
In Table 2 we show that with ResNet-50-FPN, using a
simple loss scaling weight on the semantic segmentation
loss, λs, or instance segmentation loss, λi, we can obtain
a re-weighting that improves results over single-task base-
lines. Speciﬁcally, adding a semantic segmentation branch
with the proper λs improves instance segmentation, and
vice-versa. This can be exploited to improve single-task
results. However, our main goal is to solve both tasks si-
multaneously, which we explore in the next section.
6

--- Page 7 ---
λs
mIoU
AP
AP50
AP75
PQTh
0.0
-
33.9
55.6
35.9
46.6
0.1
37.2
34.0
55.6
36.0
46.8
0.25
39.6
33.7
55.3
35.5
46.1
0.5
41.0
33.3
54.9
35.2
45.9
0.75
41.1
32.6
53.9
34.6
45.0
1.0
41.5
32.1
53.2
33.6
44.6
+0.1
+0.0
+0.1
+0.2
(a) Panoptic FPN on COCO for instance segmentation (λi = 1).
λs
mIoU
AP
AP50
PQTh
0.0
-
32.2
58.7
51.3
0.1
68.3
32.5
59.2
52.9
0.25
71.8
32.8
59.6
52.7
0.5
72.0
32.7
59.5
52.9
0.75
73.4
32.8
58.8
52.3
1.0
74.2
33.2
59.7
52.4
+1.0
+1.0
+1.1
(b) Panoptic FPN on Cityscapes for instance segmentation (λi = 1).
λi
AP
mIoU
fIoU
PQSt
0.0
-
40.2
67.2
27.9
0.1
20.1
40.6
67.5
28.4
0.25
25.5
41.0
67.8
28.6
0.5
29.2
41.3
68.0
28.9
0.75
30.8
41.1
68.2
28.9
1.0
32.1
41.5
68.2
29.0
+1.2
+1.0
+1.1
(c) Panoptic FPN on COCO for semantic segmentation (λs = 1).
λi
AP
mIoU
iIoU
PQSt
0.0
-
74.5
55.8
62.4
0.1
27.4
75.3
57.6
62.5
0.25
30.5
75.5
58.3
62.5
0.5
32.0
75.0
58.2
62.2
0.75
32.6
74.3
58.2
61.7
1.0
33.2
74.2
57.4
61.4
+1.0
+2.5
+0.1
(d) Panoptic FPN on Cityscapes for semantic segmentation (λs = 1).
Table 2: Multi-Task Training: (a,b) Adding a semantic segmentation branch can slightly improve instance segmentation
results over a single-task baseline with properly tuned λs (results bolded). Note that λs indicates the weight assigned to the
semantic segmentation loss and λs = 0.0 serves as the single-task baseline. (c,d) Adding an instance segmentation branch
can provide even stronger beneﬁts for semantic segmentation over a single-task baseline with properly tuned λi (results
bolded). As before, λi indicates the weight assigned to the instance segmentation loss and λi = 0.0 serves as the single-task
baseline. While promising, we are more interested in the joint task, for which results are shown in Table 3.
backbone
AP
PQTh
mIoU
PQSt
PQ
COCO
R50-FPN×2
33.9
46.6
40.2
27.9
39.2
R50-FPN
33.3
45.9
41.0
28.7
39.0
-0.6
-0.7
+0.8
+0.8
-0.2
Cityscapes
R50-FPN×2
32.2
51.3
74.5
62.4
57.7
R50-FPN
32.0
51.6
75.0
62.2
57.7
-0.2
+0.3
+0.5
-0.2
+0.0
(a) Panoptic Segmentation: Panoptic R50-FPN vs. R50-FPN×2. Us-
ing a single FPN network for solving both tasks simultaneously yields
comparable accuracy to two independent FPN networks for instance
and semantic segmentation, but with roughly half the compute.
backbone
AP
PQTh
mIoU
PQSt
PQ
COCO
R50-FPN×2
33.9
46.6
40.2
27.9
39.2
R101-FPN
35.2
47.5
42.1
29.5
40.3
+1.3
+0.9
+1.9
+1.6
+1.1
Cityscapes
R50-FPN×2
32.2
51.3
74.5
62.4
57.7
R101-FPN
33.0
52.0
75.7
62.5
58.1
+0.8
+0.7
+1.3
+0.1
+0.4
(b) Panoptic Segmentation: Panoptic R101-FPN vs. R50-FPN×2.
Given a roughly equal computational budget, a single FPN network for
the panoptic task outperforms two independent FPN networks for in-
stance and semantic segmentation by a healthy margin.
loss
AP
PQTh
mIoU
PQSt
PQ
COCO
alternate
31.7
43.9
40.2
28.0
37.5
combine
33.3
45.9
41.0
28.7
39.0
+1.6
+2.0
+0.8
+0.7
+1.5
Cityscapes
alternate
32.0
51.4
74.3
61.3
57.4
combine
32.0
51.6
75.0
62.2
57.7
+0.0
+0.2
+0.7
+0.9
+0.3
(c) Training Panoptic FPN. During training, for each minibatch we can
either combine the semantic and instances loss or we can alternate which
loss we compute (in the latter case we train for twice as long). We ﬁnd
that combining the losses in each minibatch performs much better.
FPN
AP
PQTh
mIoU
PQSt
PQ
COCO
original
33.3
45.9
41.0
28.7
39.0
grouped
33.1
45.7
41.2
28.4
38.8
-0.2
-0.2
+0.2
-0.3
-0.2
Cityscapes
original
32.0
51.6
75.0
62.2
57.7
grouped
32.0
51.8
75.3
61.7
57.5
+0.0
+0.2
+0.3
-0.5
-0.2
(d) Grouped FPN. We test a variant of Panoptic FPN where we group
the 256 FPN channels into two sets and apply the instance and semantic
branch to its own dedicated group of 128. While this gives mixed gains,
we expect better multi-task strategies can improve results.
Table 3: Panoptic FPN Results.
7

--- Page 8 ---
Figure 6: More Panoptic FPN results on COCO (top) and Cityscapes (bottom) using a single ResNet-101-FPN network.
PQ
PQTh
PQSt
Artemis
16.9
16.8
17.0
LeChen
26.2
31.0
18.9
MPS-TU Eindhoven [16]
27.2
29.6
23.4
MMAP-seg
32.1
38.9
22.0
Panoptic FPN
40.9
48.3
29.7
(a) Panoptic Segmentation on COCO test-dev. We submit Panoptic FPN
to the COCO test-dev leaderboard (for details on competing entries, please
see
http://cocodataset.org/#panoptic-leaderboard).
We only compare to entires that use a single network for the joint task.
We do not compare to competition-level entires that utilize ensembling
(including methods that ensemble separate networks for semantic and
instance segmentation). For methods that use one network for panoptic
segmentation, our approach improves PQ by an ∼9 point margin.
coarse
PQ
PQTh
PQSt
mIoU
AP
DIN [1, 34]
✓
53.8
42.5
62.1
80.1
28.6
Panoptic FPN
58.1
52.0
62.5
75.7
33.0
(b) Panoptic Segmentation on Cityscapes. For Cityscapes, there is no
public leaderboard for panoptic segmentation at this time. Instead, we com-
pare on val to the recent work of Arnab and Torr [1, 34] who develop a novel
approach for panoptic segmentation, named DIN. DIN is representative of
alternatives to region-based instance segmentation that start with a pixel-
wise semantic segmentation and then perform grouping to extract instances
(see the related work). Panoptic FPN, without extra coarse training data or
any bells and whistles, outperforms DIN by a 4.3 point PQ margin.
Table 4: Comparisons of ResNet-101 Panoptic FPN to the state of the art.
4.4. Panoptic FPN
We now turn to our main result: testing Panoptic FPN
for the joint task of panoptic segmentation [30], where the
network must jointly and accurately output stuff and thing
segmentations. For the following experiments, for each set-
ting we select the optimal λs and λi from {0.5, 0.75, 1.0},
ensuring that results are not skewed by ﬁxed choice of λ’s.
Main results: In Table 3a we compare two networks
trained separately to Panoptic FPN with a single backbone.
Panoptic FPN yields comparable accuracy but with roughly
half the compute (the backbone dominates compute, so the
reduction is almost 50%). We also balance computational
budgets by comparing two separate networks with ResNet-
50 backbones each and Panoptic FPN with ResNet-101,
see Table 3b. Using roughly equal computational budget,
Panoptic FPN signiﬁcantly outperforms two separate net-
works. Taken together, these results demonstrate that the
joint approach is strictly beneﬁcial, and that our Panoptic
FPN can serve as a solid baseline for the joint task.
Ablations: We perform additional ablations on Panoptic
FPN with ResNet-50. First, by default, we combine the
instance and semantic losses together during each gradi-
ent update. A different strategy is to alternate the losses on
each iteration (this may be useful as different augmentation
strategies can be used for the two tasks). We compare these
two options in Table 3c; the combined loss demonstrates
better performance. Next, in Table 3d we compare with an
architecture where FPN channels are grouped into two sets,
and each task uses one of the two features sets as its input.
While the results are mixed, we expect more sophisticated
multi-task approaches could give stronger gains.
Comparisons: We conclude by comparing Panoptic FPN
with existing methods.
For these experiments, we use
Panoptic FPN with a ResNet-101 backbone and without
bells-and-whistles. In Table 4a we show that Panoptic FPN
substantially outperforms all single-model entries in the re-
cent COCO Panoptic Segmentation Challenge. This estab-
lishes a new baseline for the panoptic segmentation task. On
Cityscapes, we compare Panoptic FPN with an approach for
panoptic segmentation recently proposed in [1] in Table 4b.
Panoptic FPN outperforms [1] by a 4.3 point PQ margin.
5. Conclusion
We introduce a conceptually simple yet effective base-
line for panoptic segmentation.
The method starts with
Mask R-CNN with FPN and adds to it a lightweight se-
mantic segmentation branch for dense-pixel prediction. We
hope it can serve as a strong foundation for future research.
8

--- Page 9 ---
References
[1] A. Arnab and P. H. Torr. Pixelwise instance segmentation
with a dynamically instantiated network. In CVPR, 2017. 1,
3, 8
[2] V. Badrinarayanan, A. Kendall, and R. Cipolla. Segnet: A
deep convolutional encoder-decoder architecture for image
segmentation. PAMI, 2017. 3
[3] S. Bell, C. Lawrence Zitnick, K. Bala, and R. Girshick.
Inside-outside net: Detecting objects in context with skip
pooling and recurrent neural networks. In CVPR, 2016. 3
[4] P. Bilinski and V. Prisacariu. COCO-Stuff 2017 Challenge:
Oxford Active Vision Lab team. 2017. 6
[5] S. R. Bul`o, L. Porzi, and P. Kontschieder.
In-place acti-
vated batchnorm for memory-optimized training of DNNs.
In CVPR, 2018. 3, 5, 6
[6] H. Caesar, J. Uijlings, and V. Ferrari. COCO-Stuff: Thing
and stuff classes in context. In CVPR, 2018. 2, 5
[7] Z. Cai and N. Vasconcelos. Cascade R-CNN: Delving into
high quality object detection. In CVPR, 2018. 3
[8] J. Cao, Y. Pang, and X. Li. Triply supervised decoder net-
works for joint detection and segmentation. arXiv preprint
arXiv:1809.09299, 2018. 3
[9] L.-C. Chen, A. Hermans, G. Papandreou, F. Schroff,
P. Wang, and H. Adam. MaskLab: Instance segmentation
by reﬁning object detection with semantic and direction fea-
tures. In CVPR, 2018. 1, 3
[10] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and
A. L. Yuille. DeepLab: Semantic image segmentation with
deep convolutional nets, atrous convolution, and fully con-
nected CRFs. PAMI, 2018. 1, 3, 6
[11] L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam. Re-
thinking atrous convolution for semantic image segmenta-
tion. arXiv:1706.05587, 2017. 6
[12] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam.
Encoder-decoder with atrous separable convolution for se-
mantic image segmentation. In ECCV, 2018. 2, 3, 6
[13] J.-T. Chien and H.-T. Chen. COCO-Stuff 2017 Challenge:
Vllab team. 2017. 6
[14] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler,
R. Benenson, U. Franke, S. Roth, and B. Schiele.
The
cityscapes dataset for semantic urban scene understanding.
In CVPR, 2016. 1, 2, 5
[15] J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, and Y. Wei.
Deformable convolutional networks. In ICCV, 2017. 3
[16] D. de Geus, P. Meletis, and G. Dubbelman. Panoptic seg-
mentation with a joint semantic and instance segmentation
network. arXiv:1809.02110, 2018. 8
[17] N. Dvornik, K. Shmelkov, J. Mairal, and C. Schmid.
BlitzNet: A real-time deep network for scene understanding.
In ICCV, 2017. 3
[18] M. Everingham, S. A. Eslami, L. Van Gool, C. K. Williams,
J. Winn, and A. Zisserman.
The PASCAL visual object
classes challenge: A retrospective. IJCV, 2015. 1, 5
[19] A. Fathi and K. Murphy. COCO-Stuff 2017 Challenge: G-
RMI team. 2017. 6
[20] G. Ghiasi and C. C. Fowlkes. Laplacian pyramid reconstruc-
tion and reﬁnement for semantic segmentation. In ECCV,
2016. 3
[21] R. Girshick. Fast R-CNN. In ICCV, 2015. 3
[22] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich fea-
ture hierarchies for accurate object detection and semantic
segmentation. In CVPR, 2014. 3
[23] R. Girshick,
I. Radosavovic,
G. Gkioxari,
P. Doll´ar,
and K. He.
Detectron.
https://github.com/
facebookresearch/detectron, 2018. 2, 5
[24] K. He, G. Gkioxari, P. Doll´ar, and R. Girshick. Mask R-
CNN. In ICCV, 2017. 1, 2, 3, 4, 5
[25] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning
for image recognition. In CVPR, 2016. 3, 4
[26] S. Honari, J. Yosinski, P. Vincent, and C. Pal. Recombinator
networks: Learning coarse-to-ﬁne feature aggregation. In
CVPR, 2016. 3
[27] J. Hu, L. Shen, and G. Sun.
Squeeze-and-excitation net-
works. In CVPR, 2018. 6
[28] S. Ioffe and C. Szegedy. Batch normalization: Accelerating
deep network training by reducing internal covariate shift. In
ICML, 2015. 4
[29] A. Kendall, Y. Gal, and R. Cipolla. Multi-task learning using
uncertainty to weigh losses for scene geometry and seman-
tics. In CVPR, 2018. 3, 6
[30] A. Kirillov, K. He, R. Girshick, C. Rother, and P. Doll´ar.
Panoptic segmentation. In CVPR, 2019. 1, 2, 3, 4, 5, 8
[31] A. Kirillov, E. Levinkov, B. Andres, B. Savchynskyy, and
C. Rother. InstanceCut: from edges to instances with multi-
cut. In CVPR, 2017. 3
[32] I. Kokkinos. UberNet: Training a universal convolutional
neural network for low-, mid-, and high-level vision using
diverse datasets and limited memory. In CVPR, 2017. 3, 6
[33] J. Li, A. Raventos, A. Bhargava, T. Tagawa, and A. Gaidon.
Learning to fuse things and stuff. arXiv:1812.01192, 2018.
2
[34] Q. Li, A. Arnab, and P. H. Torr. Weakly-and semi-supervised
panoptic segmentation. In ECCV, 2018. 8
[35] Y. Li, H. Qi, J. Dai, X. Ji, and Y. Wei. Fully convolutional
instance-aware semantic segmentation. In CVPR, 2017. 3
[36] T.-Y. Lin, P. Doll´ar, R. Girshick, K. He, B. Hariharan, and
S. Belongie. Feature pyramid networks for object detection.
In CVPR, 2017. 1, 2, 3
[37] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ra-
manan, P. Doll´ar, and C. L. Zitnick. Microsoft COCO: Com-
mon objects in context. In ECCV, 2014. 1, 2, 3, 5
[38] S. Liu, J. Jia, S. Fidler, and R. Urtasun.
SGN: Sequen-
tial grouping networks for instance segmentation. In CVPR,
2017. 3
[39] S. Liu, L. Qi, H. Qin, J. Shi, and J. Jia. Path aggregation
network for instance segmentation. In CVPR, 2018. 3
[40] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y.
Fu, and A. C. Berg. SSD: Single shot multibox detector. In
ECCV, 2016. 5, 6
[41] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional
networks for semantic segmentation. In CVPR, 2015. 1, 3
9

--- Page 10 ---
[42] I. Misra, A. Shrivastava, A. Gupta, and M. Hebert. Cross-
stitch networks for multi-task learning. In CVPR, 2016. 3
[43] G. Neuhold, T. Ollmann, S. Rota Bul`o, and P. Kontschieder.
The mapillary vistas dataset for semantic understanding of
street scenes. In CVPR, 2017. 1, 2, 3
[44] A. Newell, K. Yang, and J. Deng. Stacked hourglass net-
works for human pose estimation. In ECCV, 2016. 3
[45] C. Peng, T. Xiao, Z. Li, Y. Jiang, X. Zhang, K. Jia, G. Yu,
and J. Sun. Megdet: A large mini-batch object detector. In
CVPR, 2018. 3
[46] V.-Q. Pham, S. Ito, and T. Kozakaya. BiSeg: Simultaneous
instance segmentation and semantic segmentation with fully
convolutional networks. In BMVC, 2017. 1, 3
[47] P. O. Pinheiro, T.-Y. Lin, R. Collobert, and P. Doll´ar. Learn-
ing to reﬁne object segments. In ECCV, 2016. 3
[48] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: To-
wards real-time object detection with region proposal net-
works. In NIPS, 2015. 3
[49] O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolu-
tional networks for biomedical image segmentation. In MIC-
CAI, 2015. 3, 5
[50] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh,
S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein,
A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual
Recognition Challenge. IJCV, 2015. 4
[51] J. Tighe, M. Niethammer, and S. Lazebnik.
Scene pars-
ing with object instances and occlusion ordering. In CVPR,
2014. 2
[52] Z. Tu, X. Chen, A. L. Yuille, and S.-C. Zhu. Image parsing:
Unifying segmentation, detection, and recognition.
IJCV,
2005. 2
[53] X. Wang, R. Girshick, A. Gupta, and K. He. Non-local neural
networks. In CVPR, 2018. 6
[54] Y. Wu and K. He. Group normalization. In ECCV, 2018. 4
[55] S. Xie, R. Girshick, P. Doll´ar, Z. Tu, and K. He. Aggregated
residual transformations for deep neural networks. In CVPR,
2017. 2, 4
[56] J. Yao, S. Fidler, and R. Urtasun. Describing the scene as
a whole: Joint object detection, scene classiﬁcation and se-
mantic segmentation. In CVPR, 2012. 2
[57] F. Yu and V. Koltun. Multi-scale context aggregation by di-
lated convolutions. In ICLR, 2016. 1, 3
[58] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia. Pyramid scene
parsing network. In CVPR, 2017. 3
[59] H. Zhao, Y. Zhang, S. Liu, J. Shi, C. C. Loy, D. Lin, and
J. Jia.
PSANet: Point-wise spatial attention network for
scene parsing. In ECCV, 2018. 3, 6
[60] B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Tor-
ralba. Scene parsing through ADE20K dataset. In CVPR,
2017. 1, 3
10
