# Class-Conditional Diffusion for Imbalanced Data Augmentation

**Authors**: Trabucco, Doherty, Gurinas, Salakhutdinov
**Year**: 2023
**arXiv**: 2211.10959
**Topic**: augmentation
**Relevance**: Diffusion-based rare class generation

---


--- Page 1 ---
Proton quark distributions from a light-front Faddeev-Bethe-Salpeter approach
E. Ydreforsa, T. Fredericob
aInstitute of Modern Physics, Chinese Academy of Sciences, Lanzhou 730000, China
bInstituto Tecnol´ogico de Aeron´autica, DCTA, 12228-900 S˜ao Jos´e dos Campos, Brazil
Abstract
The projection onto the Light-Front of a Minkowski space Faddeev-Bethe-Salpeter equation model truncated at the
valence level is applied to study the proton structure with constituent quarks. The dynamics of the model has built-in:
(i) a bound diquark brought by a contact interaction, and (ii) the separation by ∼ΛQCD of the infrared and ultraviolet
interaction regions. The model parameters are ﬁne tuned to reproduce the proton Dirac electromagnetic form factor
and mass. From that, the non-polarized longitudinal and transverse momentum distributions were computed. The
results for the evolved non-polarized valence parton distributions suggest that: (i) the explicit consideration of the
spin degree of freedom of both quark and diquark seems not relevant to it, and (ii) the comparison with the global ﬁt
from the NNPDF4.0 calls for higher Fock components of the wave function beyond the valence one.
Keywords: Faddeev-Bethe-Salpeter equation, Light-Front, proton structure, valence state, momentum distributions
Introduction. The distinctive characteristic of Quan-
tum Chromodynamics (QCD) is its ability to dynami-
cally enhance the interaction in the infrared (IR) region
or at long distances. Phenomena like the mass gener-
ation of the light hadrons is outside the known Higgs
mechanism, but it is born in the enhancement of the IR
interaction among quarks and gluons, leading to chiral
symmetry breaking and the quark dressing [1, 2, 3, 4].
At the same time gluons are also dressed and in ex-
treme IR region they behave as massive particles of
about 600 MeV [5] (for a recent discussion of the gluon
propagator in the Landau gauge see [6]). Any success-
ful representation of the QCD dynamics by means of
models has to incorporate the strengthening of the ef-
fective interaction in IR region [7] and dressed degrees
of freedom. The interaction among these dressed quarks
is based on the exchange of dressed gluons, which has
a range of about ∼0.3 fm inside the hadron, where the
interaction at larger distances becomes strong, or below
momentum of 600 MeV∼2ΛQCD. Therefore, to some
extend the constituent quarks have a short-range inter-
action at low momentum, which could be parameterized
eﬀectively as a contact one, as happens in the success-
ful phenomenology provided by Nambu-Jona-Lasinio
quark models (see e.g. [8]).
Email addresses: ydrefors@kth.se (E. Ydrefors),
tobias@ita.br (T. Frederico)
The implementation of eﬀective interactions among
the constituent quarks on the hypersurface (x+ = t + z =
0) within light-cone quantization [9, 10, 11, 12], al-
lows to ﬁnd the hadron state as an eigenstate of the
light-front (LF) Hamiltonian. On the other hand such
description leads to the partonic picture of the hadron
in the ultraviolet (UV) momentum region. The hadron
state expanded in the LF Fock basis deﬁned with con-
stituent degrees of freedom, ultimately allows to build
the hadron image through the diﬀerent probability den-
sities (see e.g. [13]). The associated parton distributions
to be studied in future facilities as the Electron Ion Col-
lider (EIC) [14] and the one in China (EicC) [15], will
provide precise information on the QCD nonperturba-
tive IR physics.
Among the Fock-components of the hadron wave
function, the dominant valence one can also be ob-
tained from the projection onto the LF of the Minkowski
space Bethe-Salpeter amplitude (see e.g. [16, 17]). Fur-
thermore, the valence wave function is an eigenstate
of the eﬀective LF mass squared operator reduced to
the valence sector, which can be derived from the
BS equation using the quasi-potential approach applied
to two bosons [18], two fermions [19] and to three-
particles [20, 21]. The eﬀective interaction contains the
inﬁnite sum over intermediate states in the Fock-space.
Alternatively, the eigenvalue equation for the eﬀective
LF mass squared operator can be derived using the ”it-
Preprint submitted to Physics Letters B
November 22, 2022
arXiv:2211.10959v1  [hep-ph]  20 Nov 2022

--- Page 2 ---
erated resolvent method” [9].
In this work, we further study the proton within
the framework of the LF projected Minkowski space
Faddeev-Bethe-Salpeter (LF-FBS) equation for three
particles with a contact pairwise interaction [22, 23]
(see [24] for the detailed derivation within the quasi-
potential approach). This valence model has been used
to investigate the proton structure with the totally sym-
metric momentum component of the valence wave func-
tion [25] and applied recently [24] to study the pro-
ton image on the null-plane.
The essential dynami-
cal ingredient is a diquark [26], which is introduced
in the model through the quark-quark amplitude, anal-
ogous to formulations of the nucleon with Euclidean
continuum methods [27]. Other recent approaches to
the nucleon emphasize diquark degrees of freedom (see
e.g. [28, 29]).
The kernel of the LF-FBS equation applied for the
proton is improved to take into account the separation
between the IR and the UV interaction regions in the
three-quark dynamics. This is achieved by introducing
a soft cutoﬀof the free three-quark LF propagation in
states of high virtuality. The deep unphysical ground
state found previously [30] is now naturally removed. In
the present formulation of the LF-FBS model the quark
spin degree of freedom is not considered, as it is our
goal to study the spatial non-polarized distributions of
the quarks in the proton valence state, namely, the par-
ton distribution function (PDF) and transverse momen-
tum distribution (TMD).
LF three-quark model. We consider only the totally
symmetric momentum part of the the colorless three-
quark wave function corresponding to the valence nu-
cleon state, as we are interested for the time being on
the investigation of the properties associated with the
momentum distributions and the image of the nucleon
onto the null-plane. In this case, the valence LF wave
function is written as [24]:
Ψ3({x,⃗k⊥}) =
3
X
i=1
Γ(xi, ki⊥)
√x1x2x3

M2
N −M2
0({x,⃗k⊥})
,
(1)
{x,⃗k⊥} ≡{x1,⃗k1⊥, x2,⃗k2⊥, x3,⃗k3⊥} with Γ(xi, ki⊥), where
ki⊥= |⃗ki⊥|, being the Faddeev component of the ver-
tex function for the bound state, x1 + x2 + x3 = 1,
⃗k1⊥+ ⃗k2⊥+ ⃗k3⊥= ⃗0⊥and M2
0({x,⃗k⊥}) = P3
i=1
⃗k2
i⊥+m2
xi
, is
the free three-body squared mass for on-mass-shell con-
stituents. The factorized form of the valence wave func-
tion, namely with a vertex function depending solely on
the bachelor quark LF momenta, is a consequence of
the eﬀective contact interaction between the constituent
quarks, which is an idealized model resembling the suc-
cessful Nambu-Jona-Lasinio model applied to model
QCD [31]. It should be understood as an eﬀective low-
energy model which is meant to have signiﬁcance in
the IR region where constituent quarks are massive and
bound forming the nucleon.
The bound state homogeneous Faddeev equation for
the vertex component of the nucleon valence LF wave
function, with four-point local interaction between the
constituents was described in Refs. [22, 23] and applied
to study the proton in [25, 24]. One physical key in-
gredient was missing so far in these previous studies,
namely the kernel of the dynamical integral equation
has to take into account the IR enhancement of the QCD
interaction between the quarks [7] and the weakening at
large momentum scale. However, our model four-point
local interaction has its action in the UV region, while
in IR quarks and gluons interact strongly according to
QCD. In order to represent the physics of QCD, which
undoubtedly distinguish the IR and UV dynamics, we
model the kernel in a way that it is stronger at low mo-
menta and weaker for large ones. For this aim we in-
troduced a smooth cutoﬀin the integral equation for the
Faddeev component of the vertex such that:
Γ(x, k⊥) = F(M2
12)
(2π)3
Z 1−x
0
dx′
x′(1 −x −x′)
×
Z ∞
0
d2k′
⊥
Λ( b
M2
0)
b
M2
0 −M2
N
Γ(x′, k′
⊥) ,
(2)
where F(M2
12) is the quark-quark amplitude, b
M2
0 =
M2
0(x,⃗k⊥, x′,⃗k′
⊥, 1 −x −x′, −(⃗k⊥+ ⃗k′
⊥)) . Eq. (2) for
Λ( b
M2
0) = 1 was derived in detail in [24] resorting to the
quasi-potential technique to perform the projection onto
the LF of the three-boson BS equation in Minkowski
space. Such LF equation corresponds to the truncation
of the LF Fock-space only at the valence level. The form
factor is introduced to cut the three-quark resolvent at
large virtuality, such that
1
b
M2
0 −M2
N
−
1
b
M2
0 + µ2 =
Λ( b
M2
0)
b
M2
0 −M2
N
,
(3)
where Λ( b
M2
0) = (M2
N + µ2)/( b
M2
0 + µ2) , which damp-
ens the kernel when b
M2
0 >> µ2, without changing the
low momentum region. In the IR region F( b
M2
0) ∼1,
as the minimum value of the three-quark free-mass is
3m, which is about the nucleon mass. The IR scale of
the model is chosen µ ∼ΛQCD. Noteworthy that the IR
enhancement of the kernel also should simulate the rel-
evance of the coupling of the valence component to the
2

--- Page 3 ---
Model
m [MeV]
a.m
µ/m
Mdq [MeV]
(a)
366
2.70
1
644
(b)
362
3.60
∞
682
(c)
317
-1.84
∞
-
Table 1: Model parameters: constituent quark mass (2nd column),
scattering length in units of m−1 (3rd column), cutoﬀmass in units of
m (4th column) and diquark mass (5th column). The nucleon mass is
940 MeV.
higher Fock-components at large distances. In addition,
the choice of the form factor eliminates the unphysical
solution with M2
N < 0 appearing for the bound diquark
case when Λ( b
M2
0) = 1 [24], as we are going to show.
We observe that the quark-exchange kernel of the LF-
FBS equation (2) regularized at the mass scale µ can
be recognized by the appearance of the regulated free
resolvent given by Eq. (3). The quark-exchange kernel
is also present in the Euclidean four-dimensional three-
quark BSE [4] when diquarks dominate the quark-quark
interaction.
The model takes into account the quark-quark ampli-
tude, F(M2
12), which weights the kernel of the LF-FBS
equation for the vertex function, and has the following
expression in the limit of an eﬀective contact interaction
between the constituent quarks:
F(M2
12) =
Θ(−M2
12)
1
16π2y log 1+y
1−y −
1
16πma
+ Θ(M2
12) Θ(4m2 −M2
12)
1
8π2y′ arctan y′ −
1
16πma
,
(4)
where the Θ(x) denotes the Heaviside theta function.
and its argument is the eﬀective oﬀ-shell mass of the
two-quark subsystem squared, given by
M2
12 = (1 −x)M2
N −k2
⊥+ (1 −x)m2
x
,
y′ =
M12
q
4m2 −M2
12
, y =
q
−M2
12
q
4m2 −M2
12
.
(5)
The scalar diquark is a pole of the quark-quark am-
plitude, Eq. (4), for scattering lengths π/(2m) > a > 0
associated with a diquark mass Mdq, which is a model
parameter. The value of Mdq is suggested by the recent
literature [26] to be around 600 MeV. In the case a is
negative no physical two-body bound-state exists and
the nucleon is a Borromean state. In both situations,
when π/(2m) > a > 0 and a < 0, the quark-quark am-
plitude has a pole. In the former case the pole appears
in the physical complex-energy sheet, while in the latter
in the 2nd one, meaning the virtual state. The strong
diquark correlation is manifested either for bound or
 0
 0.5
 1
 1.5
k⊥/m
 0
 0.2
 0.4
 0.6
 0.8
 1
x
 0
 0.2
 0.4
 0.6
 0.8
 1
 1.2
−70
−60
−50
−40
−30
−20
−10
 0
 10
 1
 10
 100
 1000
M2
3 [m2]
µ [m]
a = −1.84/m, ground state
a = 3.60/m, ground state
a = 3.60/m, 1st excited state
Figure 1: Upper panel: Vertex function, Γ(x, k⊥), in arbitrary units
for model (a). Lower panel: Computed values of the squared three-
body bound state mass, M2
3, versus the cutoﬀmass, µ, for a/m = 3.60
ground (solid line) and ﬁrst excited state (dot-dashed line), and a/m =
−1.84 (dashed line). The asymptotic limits of µ →∞correspond to
model (b) (dot-dashed line) and to model (c) (dashed line).
virtual states and it is a consequence of the enhance-
ment of interaction between the constituent quarks in
the IR region [26]. Some choices of parameters of the
model are given in Table 1, with (a) being the new one
for µ = m, (b) and (c) for µ = ∞, which were already
studied in [24]. The choice (a) of model parameters will
become clear later on.
In the upper panel of Fig. 1, the Faddeev compo-
nent of the vertex, Γ(x, k⊥), is shown. The quark mass
in model (a) is chosen close to 350 MeV which is the
IR value obtained in a recent LQCD calculation in the
Landau gauge [32]. We note that the diquark mass of
644 MeV presents a diﬀerence of 278 MeV with respect
to the quark mass, comparable to the gauge invariant re-
sult from the LQCD calculation [33] of 319(1) MeV at
the physical pion mass. For the model (a) with param-
eters inspired by LQCD results, the three-quark system
has only one bound state, which is identiﬁed with the
nucleon. The vertex function is peaked at x ∼1/3 and
it spreads out up to k⊥∼ΛQCD, which is about the con-
stituent quark mass and turns even more reasonable our
3

--- Page 4 ---
 0
 0.1
 0.2
 0.3
 0.4
 0.5
 0.6
 0.7
 0.8
 0
 2
 4
 6
 8
 10
Q2 F1(Q2)
Q2 [GeV2]
Fit exp. data, Z. Ye et al
Model (a)
Model (b)
Model (c)
Figure 2: F1(Q2) for model (a) (solid line), (b) (dot-dashed line) and
(c) (dotted line). The empirical ﬁt (dashed line) obtained in Ref. [34].
choice of µ. In particular, model (a) provides a ﬁt to the
Dirac form factor of the proton, as it will be shown.
To be complete, we present in the lower panel of
Fig. 1, the computed values of the squared three-body
mass, M2
3, as a function of the cutoﬀmass µ for the
ground and ﬁrst excited states for a = 3.6/m, and
ground state in the case of a = −1.84/m. By decreasing
the value of µ the unphysical ground state of the model
with M2
3 < 0 for a = 3.6/m disappears and turns to a
physical one. A similar eﬀect is found for a = 2.7/m
which for µ = m deﬁnes the model parametrization (a)
given in Table 1.
Dirac form factor. The present model accounts only
for the valence state of the nucleon wave function, and
it allows to obtain the Dirac form factor as discussed
in detail in [24]. Resorting to the Drell-Yan condition
where the plus component of the momentum transfer
vanishes (q+ = 0), the form factor computed with the
valence component of the wave function is given by:
F1(Q2) =
Z
{dx d2k⊥} Ψ†
3({x,⃗kf
⊥})Ψ3({x,⃗ki
⊥}),
(6)
where the phase-space integral is
Z
{dx d2k⊥} =
2
Y
i=1
Z d2ki⊥
(2π)2
Z 1
0
dxi Θ 1 −x1 −x2
 , (7)
with P3
i=1 ⃗ki⊥= 0, P3
i=1 xi = 1 and Q2 = ⃗q⊥· ⃗q⊥. Fur-
thermore, choosing the Breit frame the momenta of the
quarks in Eq. (6) are:
⃗kf(i)
i⊥= ⃗ki⊥± ⃗q⊥
2 xi (i = 1, 2)
and
⃗kf(i)
3⊥= ±⃗q⊥
2 (x3 −1) −⃗k1⊥−⃗k2⊥,
(8)
with -(+) for f(i).
 0
 0.5
 1
 1.5
 2
 2.5
 3
 3.5
 0
 0.2
 0.4
 0.6
 0.8
 1
Contribution
x1
I11
I22 + I33
I12 + I13
I23
Total (a)
Total (b)
Total (c)
Figure 3: Valence PDF computed with model (a): I11 (solid thin line),
I22 +I33 (dashed line), I12 +I13 (dot-dashed line), I23 (dotted line) and
the total (solid thick line) normalized to 1. Results for the total PDF
for model (b) (full squares) and (c) (full circles) are connected by the
dashed lines.
In Fig. 2, we present the results for the proton Dirac
form factor, F1(Q2), with model (a), which was ﬁne
tuned to be consistent with the global ﬁt to experimen-
tal data by Ye et al [34] up to 10 GeV2. In addition, we
present the previous results [24] obtained with model
(b) having a bound diquark, and with model (c) present-
ing a virtual diquark, both calculations have µ = ∞.
The IR dynamics is not privileged by models (b) and
(c) and the calculated Dirac form factor is not able to
reproduce the experimental ﬁt, which is now possible.
The enhancement of the interaction kernel of Eq. (2) in
the IR with respect to the UV region, was achieved with
the introduction of the regularization scale µ ∼ΛQCD in
Eq. (3), which ﬁnally lead the model to the experimental
Dirac form factor.
Valence quark momentum distribution. The valence
PDF should be given by the sum over all Fock compo-
nents of the proton light-front wave function, namely:
f1(x) =
∞
X
n=3
(
n
Y
i
Z d2ki⊥
(2π)2
Z 1
0
dxi
)
× δ (x −x1) δ
1 −
n
X
i=1
xi
δ

n
X
i=1
⃗ki⊥

×
Ψn(x1,⃗k1⊥, x2,⃗k2⊥, ...)
2 ,
(9)
where n indicates the number of partons in each Fock
contribution to the probability amplitude. However, in
the present model, we consider only the valence compo-
nent of the proton LF wave function, namely the trun-
cation takes into account only n = 3, which presumably
is the dominant one. It has been veriﬁed for the pion
state [35] where the valence contribution accounts for
70% of the LF wave function described in terms of con-
stituent quark degrees of freedom.
4

--- Page 5 ---
The results for the valence parton distribution (PDF)
at the hadron scale are shown in Fig. 3 for model (a),
(b) and (c). The PDF is obtained from the integrand of
Eq. (6) for the Dirac form factor at Q2 = 0 or equiva-
lently from Eq. (9) truncated at the valence state:
f1(x) =
X
3≥j≥i≥1
(2 −δij) Iij(x) ,
(10)
where
Iij(x) =
Z
{dx d2k⊥}δ(x −x1)
x1x2x3
Γ(xi,⃗ki⊥)Γ(x j,⃗kj⊥)
 M2
N −M2
0({x,⃗k⊥})2 ,
(11)
with P3
i=1 ⃗ki⊥= 0 and P3
i=1 xi = 1. The contributions
to the PDF indicated in the ﬁgure are identiﬁed by Ii j
deﬁned in Eq. (11). Due to the symmetry of the nucleon
wave function under the exchange of quarks 2 and 3,
it follows that I22 = I33 and I12 = I13, these relations
are taken into account in Eq. (10). Noteworthy to ﬁnd
that all the contributions have about the same size, and
are peaked around 1/3, despite our choice of quark 1 to
obtain the PDF. This property can be traced back to the
denominator appearing in the valence wave function in
Eq. (1), which is the three-quark resolvent and has its
maximum value at the smallest virtuality of the three-
quark system. The contribution from I11 corresponds
to the situation where the quark 1 is picked up while the
pair of quarks 2 and 3 are in a diquark correlation, which
is a small fraction of the total PDF, showing that the
symmetrization of the valence wave function is relevant
for building the proton PDF.
All contributions to the PDF are indeed similar in
magnitude, and no one is dominant, as we have already
shown previously in the study of models (b) and (c) in
Ref. [24]. Interesting to observe that models (a) and (b),
which both present a bound diquark and provide results
close to the experimental proton Dirac form factor, as
seen in Fig. 2, also have very similar PDF’s. This sug-
gests that for these observables, form factor and PDF,
the formation of the diquark is a dominant feature, quite
independent on the cutoﬀmass. However, model (b)
presents an unphysical ground state, which is eliminated
by the introduction of the cutoﬀ, as we have discussed.
Evolved valence PDFs. We present results for model
(a) at Q = 3.097 GeV for the u and d valence quarks
in Fig. 4.
We compare with the recent results ob-
tained with the Dyson-Schwinger Equation (DSE) ap-
proach [27]. In this work we adopted the same method
as the one used in [37] for the evolution of the pion
PDF. Namely, the DGLAP equation with lowest-order
splitting function was used together with the eﬀective
charge from [38].
 0
 0.2
 0.4
 0.6
 0.8
 1
 1.2
 1.4
 0.01
 0.1
 1
xu(x), xd(x)
x
xu(x), this work
xd(x), this work
xu(x), DSE
xd(x), DSE
xu(x), NNPDF4
xd(x), NNPDF4
Figure 4:
Valence u and d quark PDFs evolved to Q = 3.097 GeV.
Comparison with DSE [27]. The shaded areas correspond to model (a)
for Q0 = 0.33 ± 0.03 GeV. NNPDF 4.0 global ﬁt [36] (dotted lines).
We also took into account an uncertainty with respect
to the initial proton scale: Q0 = 0.33 ± 0.03 GeV [38],
and compared our calculations with the results from the
NNPDF 4.0 global ﬁt1 [36]. The model (a) produces
results quite consistent with the DSE approach, which
privileges the strong diquark correlation, only at large x
the discrepancy becomes visible with a softer behavior
from the DSE approach. The present model has a strong
UV damping due to the cutoﬀ, being not asymptoti-
cally free at large momentum, which is associated with
a harder behavior at the end-point. Furthermore, it has
only the contribution for the PDF from the valence light-
front wave function, which is strongly peaked around
1/3. It is expected that the higher Fock-components con-
tribution to the PDF moves the peak towards smaller x,
as the proton longitudinal momentum is shared between
more than three quarks. Therefore, the value of ⟨xq⟩
should be somewhat decreased, as we observe from the
comparison of model (a) with NNPDF4.0 in Table 2 for
the ﬁrst Mellin moments.
In the present LF-FBS model the proton has 100%
probability to be in the valence state, as we have con-
sidered only n = 3 in the Fock space decomposition of
Eq. (9). We veriﬁed that the sum rule ⟨xu⟩+ ⟨xd⟩= 1
is saturated at the proton initial scale by adopting the
standard normalization:
Z 1
0
dx fu(x) = 2
and
Z 1
0
dx fd(x) = 1 ,
(12)
where fq(x) = (2 −δq,d) f1(x) with f1(x) normalized
to 1 and obtained from Eq. (10). The assumption of
1Shown in the Fig. 4 are the central values of the NNPDF 4.0
global ﬁt available in the PDF database at https://lhapdf.
hepforge.org/pdfsets.html and the data was extracted by using
the LHAPDF software [39].
5

--- Page 6 ---
q
u
d
Model (a)
⟨xq⟩
0.296 ± 0.025
0.148 ± 0.012
⟨x2
q⟩
0.071 ± 0.009
0.036 ± 0.005
⟨x3
q⟩
0.021 ± 0.004
0.011 ± 0.002
⟨x4
q⟩
0.007 ± 0.002
0.004 ± 0.001
DSE [27]
⟨xq⟩
0.303
0.137
⟨x2
q⟩
0.077
0.032
⟨x3
q⟩
0.032
0.009
⟨x4
q⟩
0.010
0.003
NNPDF4.0
⟨xq⟩
0.261
0.101
⟨x2
q⟩
0.072
0.023
⟨x3
q⟩
0.027
0.007
⟨x4
q⟩
0.012
0.003
Table 2: Mellin moments, ⟨xn
q⟩, of the valence quark PDF (q = u, d)
at Q = 3.097 GeV for model (a) compared with the DSE [27] and the
NNPDF 4.0 global ﬁt [36], obtained by integrating the PDFs shown
in Fig. 4.
only valence state in the proton at the initial scale is
not fully sustained by the comparison with NNPDF4.0
global ﬁt as shown in Table 2 at Q = 3.097 GeV, as the
model Mellin moments ⟨xu⟩and ⟨xd⟩are clearly over
estimated. On the other hand the comparison with the
continuum DSE results from Ref. [27], which takes into
account the detailed spin structure of the quarks and
diquarks, suggests that the spin contribution is aver-
aged out in the non-polarized PDF, as shown in both
Fig. 4 and in Table 2 for the ﬁrst few moments at
Q = 3.097 GeV.
Noteworthy, the present model represents the trunca-
tion of the FBS equation at the valence level, and its full
representation on the LF has to consider the contribu-
tion of an induced three-body interaction from the cou-
pling of the valence with higher Fock-states [40, 30].
In principle, this four-dimensional FB equation model
only takes into account the quark sea, and despite of that
a quite relevant eﬀect in the binding energy was found
in [30]. Here, with the introduction of the soft cutoﬀ,
this eﬀect would be somewhat reduced. However, glu-
ons are not taken into account in this model which pre-
sumably are the main degrees of freedom to share the
quark longitudinal momentum.
Transverse momentum distribution at the proton
scale. The single quark transverse momentum distri-
bution in the forward limit [41] is associated with the
probability density to ﬁnd a quark with momentum k⊥
and x, when truncated to the valence component is:
˜f1(k⊥, x) =
Z 1
0
dx1δ(x −x1)
Z
dk1⊥
(2π)2 δ(k⊥−k1⊥)
×
Z 2π
0
dθ1
Z d2k2⊥
(2π)2
Z 1−x
0
dx2 |Ψ3({x,⃗k⊥})|2 ,
(13)
where only the dependence on k⊥= |⃗k⊥| remains due
to the symmetry of the wave function under rotations in
the transverse plane.
The PDF is the integrated TMD on the transverse mo-
mentum:
f1(x) =
Z
dk⊥k⊥˜f1(k⊥, x)
(14)
and the integrated TMD in the longitudinal momentum
is
L1(k⊥) = k⊥
Z 1
0
dx ˜f(k⊥, x) ,
(15)
which represents the probability density of a single
quark with transverse momentum k⊥.
In the upper panel of Fig. 5 the valence TMD given
by Eq. (13) is presented for model (a), and in the lower
panel the result for the single quark transverse momen-
tum density from Eq. (15), for models (a), (b) and (c)
are shown. It is interesting to notice that the momentum
scale that dominates the TMD and the integrated one is
about 0.15 GeV, deep in the IR region, which is associ-
ated with the proton size in the transverse direction of
∼1.3 fm. As seen in the lower panel of the ﬁgure, the
models (a) and (b) with a bound diquark present quite
similar result, and model (c) with the virtual diquark
peaks at considerably lower transverse momentum, re-
ﬂecting the lower binding energy (see Table 1). The
TMD and the integrated one reﬂect and are a source of
information on the IR dynamics of QCD, which in part
is reﬂected in the binding energy of the present model.
Summary. We further develop the LF eﬀective three-
quark model for the proton, based on the notion of
the dynamics dominated by the formation of diquarks.
We have disregarded the momentum structure of quark-
diquark vertex function, where the quark-quark ampli-
tude is obtained from a contact interaction. The im-
proved version of the proton model includes a soft cut-
oﬀin the kernel of the LF Faddeev equation for the
vertex function [24]. This soft cutoﬀallows to sepa-
rate the IR and UV interaction regions, with the physi-
cal eﬀect of damping the contributions from three-quark
conﬁgurations at large virtualities. This new develop-
ment eliminated the unphysical ground state for bound
diquarks achieved in previous calculations [30]. The
present model was tuned to reproduce the Dirac pro-
ton form factor with a reasonable set of parameters: a
6

--- Page 7 ---
 0
 0.05  0.1  0.15  0.2  0.25  0.3
k⊥ [GeV]
 0
 0.2
 0.4
 0.6
 0.8
 1
x
 0
 50
 100
 150
 200
 250
 0
 0.5
 1
 1.5
 2
 2.5
 3
 3.5
 4
 0
 0.2
 0.4
 0.6
 0.8
 1
L1(k⊥) [GeV−1]
k⊥ [GeV]
Model (a)
Model (b)
Model (c)
Figure 5: Transverse momentum distribution at the proton scale. Up-
per panel: transverse momentum distribution, ˜f1(k⊥, x) in GeV−2 for
model (a). Lower panel: integrated transverse momentum density vs
k⊥for the (a), (b) and (c) models.
quark mass of 366 MeV, a diquark mass of 644 MeV
and a cutoﬀof 366 MeV (∼ΛQCD), which was enough
to give the proton mass. From this parameter set we ex-
plored the proton non-polarized quark longitudinal and
transverse momentum distributions obtained from the
valence wave function. We found that the explicit con-
sideration of the spin degree of freedom of both quark
and diquark is not relevant for the evolved non-polarized
valence PDF. However, the comparison with the global
ﬁt from the NNPDF 4.0 at Q = 3.097 GeV suggests that
the higher Fock-components, missed in the model wave
function, could be relevant to improve the valence PDF.
Future challenges for the advance of the present nu-
cleon eﬀective model: the treatment Bethe-Salpeter am-
plitude in the four-dimensional Minkowski space [42,
43], the spin degree of freedom for polarized PDFs, and
quark dressing, which will lead to further insights into
the nucleon structure.
This work is a part of the project INCT-FNA
#464898/2014-5. This study was ﬁnanced in part by
Conselho Nacional de Desenvolvimento Cient´ıﬁco e
Tecnol´ogico (CNPq) under the grant 308486/2015-3
(TF). E.Y. thanks for the ﬁnancial support of the grants
#2016/25143-7 and #2018/21758-2 from FAPESP. We
thank the FAPESP Thematic grants #2017/05660-0 and
#2019/07767-1.
References
[1] A. Bashir, L. Chang, I. C. Clo¨et, B. El-Bennich, Y.-X. Liu,
C. D. Roberts, P. C. Tandy, Collective perspective on advances
in Dyson-Schwinger Equation QCD, Commun. Theor. Phys.
58 (2012) 79–134.
arXiv:1201.3366,
doi:10.1088/
0253-6102/58/1/16.
[2] I. C. Clo¨et, C. D. Roberts, Explanation and Prediction of Ob-
servables using Continuum Strong QCD, Prog. Part. Nucl. Phys.
77 (2014) 1–69. arXiv:1310.2651, doi:10.1016/j.ppnp.
2014.02.001.
[3] T. Horn, C. D. Roberts, The pion: an enigma within the Standard
Model, J. Phys. G: Nucl. Part. Phys. 43 (7) (2016) 073001.
[4] G. Eichmann, H. Sanchis-Alepuz, R. Williams, R. Alkofer,
C. S. Fischer, Baryons as relativistic three-quark bound states,
Prog. Part. Nucl. Phys. 91 (2016) 1–100. arXiv:1606.09602,
doi:10.1016/j.ppnp.2016.07.001.
[5] O. Oliveira, P. Bicudo, Running Gluon Mass from Landau
Gauge Lattice QCD Propagator, J. Phys. G: Nucl. Part. Phys.
38 (2011) 045003.
arXiv:1002.4151,
doi:10.1088/
0954-3899/38/4/045003.
[6] S. W. Li, P. Lowdon, O. Oliveira, P. J. Silva, The generalised
infrared structure of the gluon propagator, Phys. Lett. B 803
(2020) 135329.
[7] O. Oliveira, T. Frederico, W. de Paula, The soft-gluon limit
and the infrared enhancement of the quark-gluon vertex,
Eur. Phys. J. C 80 (5) (2020) 484. arXiv:2006.04982, doi:
10.1140/epjc/s10052-020-8037-0.
[8] T. Hatsuda, T. Kunihiro, QCD phenomenology based on a chiral
eﬀective Lagrangian, Phys. Rept. 247 (1994) 221–367. arXiv:
hep-ph/9401310, doi:10.1016/0370-1573(94)90022-1.
[9] S. J. Brodsky, H.-C. Pauli, S. S. Pinsky, Quantum chromody-
namics and other ﬁeld theories on the light cone, Phys. Rep. 301
(1998) 299–486.
arXiv:hep-ph/9705477, doi:10.1016/
S0370-1573(97)00089-6.
[10] J.
Vary,
H.
Honkanen,
J.
Li,
P.
Maris,
S.
Brodsky,
A. Harindranath, G. de Teramond, P. Sternberg, E. Ng, C. Yang,
Hamiltonian light-front ﬁeld theory in a basis function approach,
Phys. Rev. C 81 (2010) 035205.
arXiv:0905.1411, doi:
10.1103/PhysRevC.81.035205.
[11] B. Bakker, A. Bassetto, S. Brodsky, W. Broniowski, S. Dal-
ley, T. Frederico, S. Głazek, J. Hiller, C.-R. Ji, V. Karmanov,
et al., Light-Front Quantum Chromodynamics: A framework
for the analysis of hadron physics, Nucl. Phys. B Proc. Suppl.
251 (2014) 165–174.
[12] J. P. Vary, L. Adhikari, G. Chen, Y. Li, P. Maris, X. Zhao,
Basis Light-Front Quantization: Recent Progress and Future
Prospects, Few Body Syst. 57 (8) (2016) 695–702. doi:10.
1007/s00601-016-1117-x.
[13] J. Arrington, C. A. Gayoso, P. C. Barry, V. Berdnikov, D. Bi-
nosi, L. Chang, M. Diefenthaler, M. Ding, R. Ent, T. Frederico,
et al., Revealing the structure of light pseudoscalar mesons at the
electron–ion collider, J. Phys. G: Nucl. Part. Phys. 48 (7) (2021)
075106. doi:10.1088/1361-6471/abf5c3.
[14] R. Abdul Khalek, et al., Science Requirements and Detector
Concepts for the Electron-Ion Collider: EIC Yellow Report,
Nucl. Phys. A 1026 (2022) 122447.
arXiv:2103.05419,
doi:10.1016/j.nuclphysa.2022.122447.
[15] D. P. Anderle, et al., Electron-ion collider in China, Front. Phys.
7

--- Page 8 ---
(Beijing) 16 (6) (2021) 64701. arXiv:2102.09222, doi:10.
1007/s11467-021-1062-0.
[16] T. Frederico, G. Salm`e, Projecting the Bethe-Salpeter Equation
onto the Light-Front and back: A Short Review, Few Body
Syst. 49 (2011) 163–175. arXiv:1011.1850, doi:10.1007/
s00601-010-0163-z.
[17] C. Mezrag, H. Moutarde, J. Rodriguez-Quintero, From Bethe–
Salpeter Wave functions to Generalised Parton Distributions,
Few Body Syst. 57 (9) (2016) 729–772. arXiv:1602.07722,
doi:10.1007/s00601-016-1119-8.
[18] J.
H.
O.
Sales,
T.
Frederico,
B.
V.
Carlson,
P.
U.
Sauer, Light front Bethe-Salpeter equation, Phys. Rev. C 61
(2000) 044003.
arXiv:nucl-th/9909029, doi:10.1103/
PhysRevC.61.044003.
[19] J. H. O. Sales, T. Frederico, B. V. Carlson, P. U. Sauer, Renor-
malization of the ladder light front Bethe-Salpeter equation in
the Yukawa model, Phys. Rev. C 63 (2001) 064003.
doi:
10.1103/PhysRevC.63.064003.
[20] J. A. O. Marinho, T. Frederico, Next-to-leading order light-
front three-body dynamics, PoS LC2008 (2008) 036.
doi:
10.22323/1.061.0036.
[21] K. S. F. F. Guimar˜aes, O. Lourenc¸o, W. de Paula, T. Frederico,
A. C. dos Reis, Final state interaction in D+ →K−π+π+ with
Kπ I = 1/2 and 3/2 channels, JHEP 08 (2014) 135. arXiv:
1404.3797, doi:10.1007/JHEP08(2014)135.
[22] T. Frederico, Null-plane model of three bosons with zero-range
interaction, Phys. Lett. B 282 (3) (1992) 409–414. doi:https:
//doi.org/10.1016/0370-2693(92)90661-M.
[23] J. Carbonell, V. A. Karmanov, Three-boson relativistic bound
states with zero-range two-body interaction, Phys. Rev. C 67 (3)
(2003) 037001. doi:10.1103/physrevc.67.037001.
[24] E. Ydrefors, T. Frederico, Proton image and momentum distri-
butions from light-front dynamics, Phys. Rev. D 104 (11) (2021)
114012.
arXiv:2108.02146, doi:10.1103/PhysRevD.
104.114012.
[25] W. R. B. de Ara´ujo, J. P. B. C. de Melo, T. Frederico, Faddeev
null-plane model of the nucleon, Phys. Rev. C 52 (1995) 2733–
2737. doi:10.1103/PhysRevC.52.2733.
[26] M. Y. Barabanov, M. A. Bedolla, W. K. Brooks, G. D.
Cates, C. Chen, Y. Chen, E. Cisbani, M. Ding, G. Eich-
mann, R. Ent, J. Ferretti, R. W. Gothe, T. Horn, S. Liuti,
C. Mezrag, A. Pilloni, A. J. R. Puckett, C. D. Roberts, P. Rossi,
G. Salm´e, E. Santopinto, J. Segovia, S. N. Syritsyn, M. Tak-
izawa, E. Tomasi-Gustafsson, P. Wein, B. B. Wojtsekhowski,
Diquark correlations in hadron physics: Origin, impact and ev-
idence, Prog. Part. Nucl. Phys. 116 (2021) 103835.
arXiv:
2008.07630, doi:10.1016/j.ppnp.2020.103835.
[27] Y. Lu, L. Chang, K. Raya, C. D. Roberts, J. Rodr´ıguez-Quintero,
Proton and pion distribution functions in counterpoint, Phys.
Lett. B 830 (2022) 137130.
arXiv:2203.00753, doi:10.
1016/j.physletb.2022.137130.
[28] T. J. Hobbs, M. Alberg, G. A. Miller, Euclidean bridge to the
relativistic constituent quark model, Phys. Rev. C 95 (3) (2017)
035205. arXiv:1608.07319, doi:10.1103/PhysRevC.95.
035205.
[29] J. H. Alvarenga Nogueira, D. Colasante, V. Gherardi, T. Fred-
erico, E. Pace, G. Salm`e, Solving the Bethe-Salpeter Equation
in Minkowski Space for a Fermion-Scalar system, Phys. Rev. D
100 (1) (2019) 016021. arXiv:1907.03079, doi:10.1103/
PhysRevD.100.016021.
[30] E. Ydrefors, J. H. Alvarenga Nogueira, V. Gigante, T. Frederico,
V. A. Karmanov, Three-body bound states with zero-range inter-
action in the Bethe–Salpeter approach, Phys. Lett. B 770 (2017)
131–137. arXiv:1703.07981, doi:10.1016/j.physletb.
2017.04.035.
[31] S. P. Klevansky, The Nambu-Jona-Lasinio model of quantum
chromodynamics, Rev. Mod. Phys. 64 (1992) 649–708. doi:
10.1103/RevModPhys.64.649.
[32] O. Oliveira, P. J. Silva, J.-I. Skullerud, A. Sternbeck, Quark
propagator with two ﬂavors of O(a)-improved Wilson fermions,
Phys. Rev. D 99 (9) (2019) 094506.
arXiv:1809.02541,
doi:10.1103/PhysRevD.99.094506.
[33] A. Francis, P. de Forcrand, R. Lewis, K. Maltman, Diquark
properties from full QCD lattice simulations (6 2021). arXiv:
2106.09080.
[34] Z. Ye, J. Arrington, R. J. Hill, G. Lee, Proton and Neutron
Electromagnetic Form Factors and Uncertainties, Phys. Lett. B
777 (2018) 8–15.
arXiv:1707.09063, doi:10.1016/j.
physletb.2017.11.023.
[35] W. de Paula, E. Ydrefors, J. H. Alvarenga Nogueira, T. Fred-
erico, G. Salm`e, Observing the Minkowskian dynamics of
the pion on the null-plane, Phys. Rev. D 103 (1) (2021)
014002.
arXiv:2012.04973, doi:10.1103/PhysRevD.
103.014002.
[36] R. D. Ball, et al., The path to proton structure at 1% accuracy,
Eur. Phys. J. C 82 (5) (2022) 428. arXiv:2109.02653, doi:
10.1140/epjc/s10052-022-10328-7.
[37] W. de Paula, Y. E., J. H. Alvarenga Nogueira, T. Fred-
erico,
G. Salm`e,
Parton distribution function in a pion
with Minkowskian dynamics,
Phys. Rev. D 105 (2022)
L071505.
arXiv:2203.07106, doi:10.1103/PhysRevD.
105.L071505.
[38] Z. F. Cui, M. Ding, J. M. Morgado, K. Raya, D. Binosi,
J. Rodr´ıguez.Quintero, S. M. Schmidt, Eur. Phys. J. A 58 (2022)
10.
[39] A. Buckley, J. Ferrando, S. Lloyd, K. Nordstrom, B. Page,
M. Ruefenacht, M. Schoenherr, G. Watt, Lhapdf6:
par-
ton density access in the lhc precision era, Eur. Phys. J
C
75
132.
arXiv:1412.7420,
doi:10.1140/epjc/
s10052-015-3318-8.
[40] V. A. Karmanov, P. Maris, Manifestation of three-body forces in
three-body Bethe-Salpeter and light-front equations, Few Body
Syst. 46 (2009) 95–113. arXiv:0811.1100, doi:10.1007/
s00601-009-0054-3.
[41] C. Lorc´e, B. Pasquini, M. Vanderhaeghen, Uniﬁed framework
for generalized and transverse-momentum dependent parton dis-
tributions within a 3Q light-cone picture of the nucleon, JHEP
05 (5) (2011) 041. doi:{10.1007/jhep05(2011)041}.
[42] E. Ydrefors, J. H. Alvarenga Nogueira, V. A. Karmanov,
T. Frederico, Solving the three-body bound-state Bethe-Salpeter
equation in Minkowski space, Phys. Lett. B 791 (2019)
276–280. arXiv:1903.01741, doi:10.1016/j.physletb.
2019.02.046.
[43] E. Ydrefors, J. H. Alvarenga Nogueira, V. A. Karmanov,
T. Frederico, Three-boson bound states in Minkowski space
with contact interactions,
Phys. Rev. D 101 (9) (2020)
096018.
arXiv:2005.07943, doi:10.1103/PhysRevD.
101.096018.
8
