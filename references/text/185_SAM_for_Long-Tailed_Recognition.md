# SAM for Long-Tailed Recognition

**Authors**: Zhou et al.
**Year**: 2023
**arXiv**: 2304.06827
**Topic**: optimization
**Relevance**: SAM optimizer tuned for long-tail

---


--- Page 1 ---
Reachability Analysis Using Hybrid Zonotopes
and Functional Decomposition
Jacob A. Siefert, Trevor J. Bird, Andrew F. Thompson, Jonah J. Glunt,
Justin P. Koeln, Neera Jain, and Herschel C. Pangborn
Abstract— This paper proposes methods for reachability
analysis of nonlinear systems in both open loop and closed
loop with advanced controllers. The methods combine
hybrid zonotopes, a construct called a state-update set,
functional decomposition, and special ordered set approx-
imations to enable linear growth in reachable set memory
complexity with time and linear scaling in computational
complexity with the system dimension. Facilitating this
combination are new identities for constructing nonconvex
sets that contain nonlinear functions and for efficiently
converting a collection of polytopes from vertex repre-
sentation to hybrid zonotope representation. Benchmark
numerical examples from the literature demonstrate the
proposed methods and provide comparison to state-of-the-
art techniques.
I. INTRODUCTION
Reachability analysis—the process of calculating reachable
sets—is used to evaluate system performance and ensure
constraint satisfaction in safety-critical applications. However,
the scalability of existing approaches for nonlinear systems
is limited by their nonconvexity and the computational com-
plexity that this induces. Several methods have been pro-
posed to calculate reachable sets for continuous-time nonlinear
systems, including Hamilton-Jacobi reachability [1]–[5] and
monotonicity-based techniques [6]–[8]. This paper addresses
set propagation techniques, which calculate reachable sets by
recursively propagating successor sets [9].
1) Gaps
in
literature:
Set
propagation
techniques
for
continuous-time systems often resemble their discrete-time
counterparts, as the former often still propagate reachable
sets over discrete time intervals [10]. For this reason, there
are many shared challenges in calculating reachable sets for
continuous-time and discrete-time systems. Abstraction of the
state space helps to mitigate the computational challenges of
generating continuous-time and discrete-time reachable sets by
approximating nonlinear functions with affine functions over
Jacob A. Siefert, Andrew F. Thompson, Jonah J. Glunt, and Herschel
C. Pangborn are with the Department of Mechanical Engineering, The
Pennsylvania State University, University Park, PA 16802 USA (e-mails:
jas7031@psu.edu; thompson@psu.edu; jglunt@psu.edu;
hcpangborn@psu.edu).
Trevor J. Bird is with P. C. Krause and Associates, West Lafayette, IN
47906 USA (e-mail: tbird@pcka).
Justin P. Koeln is with the Mechanical Engineering Department,
University of Texas at Dallas, Richardson, TX 75080-3021 USA (e-mail:
justin.koeln@utdallas.edu).
Neera Jain is with the School of Mechanical Engineering, Purdue Uni-
versity, West Lafayette, IN 47907 USA (e-mail: neerajain@purdue.edu).
partitioned regions of the domain. In some cases, higher-order
polynomial abstractions compatible with specific set represen-
tations are possible [11], [12]. A collection of polyhedral over-
approximations of the nonlinear dynamics can be generated
by combining bounds for the error associated with higher-
order terms with affine abstractions. State-space abstractions
can either be generated in a time-invariant [13], [14] or
time-varying manner [15]. The primary challenge with time-
invariant abstractions is that they exhibit hybrid behaviour,
which can cause exponential growth in the number of convex
reachable sets as a result of taking intersections across guards
associated with the domain of each abstraction [9], [16], [17].
Recent work by the authors has proposed scalable ap-
proaches for reachability analysis of discrete-time hybrid
systems using a new set representation called the hybrid
zonotope, a construct called the state-update set, and efficient
identities for using the state-update set to calculate successor
and precursor sets [18]–[22]. These methods were extended for
over-approximated reachability analysis of nonlinear systems
by representing special ordered sets as hybrid zonotopes [23],
where open-loop state-update sets were combined with a set
representation of the controller called the state-input map to
construct closed-loop state-update sets. Figure 1 depicts this
process of combining an open-loop state-update set Ψ with a
state-input map Θ to obtain a closed-loop state-update set Φ,
which can then be applied iteratively to calculate reachable
sets. While construction of an open-loop state-update set
and state-input map were detailed for a particular numerical
example in [23], a generalized method was not provided.
2) Contribution: This paper provides efficient identities to
construct sets that contain general nonlinear functions. The
identities are compatible with hybrid zonotopes and can be
used to generate both open-loop state-update sets and state-
input maps, which can then be used within successor set
identities to calculate over-approximations of reachable sets.
The method decomposes functions with an arbitrary number
of arguments into functions with only one or two scalar
arguments, thus avoiding the so-called curse of dimensionality
(i.e., exponential growth in memory complexity with respect
to the state dimension). Complementary results for hybrid
zonotopes are provided, including efficient conversion from
collections of vertex representation polytopes and efficient
over-approximation of common binary functions. Theoretical
results are applied to several numerical examples. The first
performs reachability of an inverted pendulum with a neural
network controller. This aligns with a benchmark problem
arXiv:2304.06827v2  [eess.SY]  22 Feb 2024

--- Page 2 ---
Fig. 1: The closed-loop successor set identity uses a set-based representation of the closed-loop dynamics, called the closed-loop
state-update set Φ, to generate the one-step forward reachable set Rk+1 from Rk. The closed-loop state-update set is created
by combining sets representing the open-loop dynamics and a state-feedback controller, called the open-loop state-update set
Ψ and the state-input map Θ, respectively.
from [24], [25], facilitating comparison to other state-of-the-
art methods. While hybrid zonotopes have previously been
used for reachability analysis of neural networks [26], [27],
the methods in this paper achieve improved scalability and
go beyond the scope of previous work by analyzing the
coupling of a neural network controller to a nonlinear plant.
The second numerical example performs reachability of a
high-dimensional logical system from [28] to demonstrate
applications to logical systems and scalability with the number
of states. The final numerical example performs reachability of
the Vertical Collision Avoidance System (VCAS) from [24],
[25] to demonstrate how the proposed techniques can address
a complex combination of neural networks and logical opera-
tions, while other state-of-the-art methods require simplifying
assumptions.
3) Outline: The remainder of this paper is organized as
follows. Section II provides notation and basic definitions.
Section III defines open-loop and closed-loop state-update sets
and presents an identity to construct a closed-loop state-update
set from an open-loop state-update set and a state-input map.
This section also provides a generalized method for efficiently
constructing open-loop state-update sets and state-input maps
by leveraging functional decomposition. Section IV shows
how to generate over-approximated sets of nonlinear functions
by efficiently converting a collection of vertex-representation
polytopes into a single hybrid zonotope. Section V applies the
theoretical results to several numerical examples.
II. PRELIMINARIES AND PREVIOUS WORK
A. Notation
Matrices are denoted by uppercase letters, e.g., G ∈Rn×ng,
and sets by uppercase calligraphic letters, e.g., Z ⊂Rn.
Vectors and scalars are denoted by lowercase letters. The
ith column of a matrix G is denoted by G(·,i). Commas in
subscripts are used to distinguish between properties that are
defined for multiple sets, e.g., ng,z describes the complexity of
the representation of Z while ng,w describes the complexity
of the representation of W. The topological boundary of a set
Z is denoted by ∂Z and its interior by Z◦. The n-dimensional
unit hypercube is denoted by Bn
∞= {x ∈Rn | ∥x∥∞≤1}.
The set of all n-dimensional binary vectors is denoted by
{−1, 1}n and the interval set between a lower bound bl and an
upper bound bu is denoted by [bl, bu]. Matrices of all 0 and 1
elements are denoted by 0 and 1, respectively, of appropriate
dimension and I denotes the identity matrix. The ith row of
the identity matrix is denoted by ei. The concatenation of
two column vectors to a single column vector is denoted by
(g1, g2) = [gT
1 gT
2 ]T and the rounding of a scalar s to the next
largest integer is denoted by ⌈s⌉.
Given the sets Z, W
⊂
Rn, Y
⊂
Rm, and matrix
R ∈Rm×n, the linear transformation of Z by R is RZ =
{Rz | z ∈Z}, the Minkowski sum of Z and W is Z ⊕W =
{z + w | z ∈Z, w ∈W}, the generalized intersection of Z
and Y under R is Z ∩R Y = {z ∈Z | Rz ∈Y}, the Cartesian
product of Z and Y is Z × Y = {(z, y)| z ∈Z, y ∈Y}.
B. Set representations
Definition 1: A set P ∈Rn is a convex polytope if it is
bounded and ∃V ∈Rn×nv with nv < ∞such that
P =
n
V λ | λj ≥0 ∀j ∈{1, ..., nv}, 1T λ = 1
o
.
A convex polytope is the convex hull of a finite set of
vertices given by the columns of V . A convex polytope
defined using a matrix of vertices is said to be given in
vertex-representation (V-rep). The memory complexity and
computational complexity of many key set operations scales
poorly when V-rep is used [9, Table 1], often limiting analysis
to systems with few states. The methods proposed in this paper
use functional decomposition to represent high-dimensional
systems using unary or binary functions, thus we only use
vertex representation in one- or two-dimensional space. The
resulting V-rep polytopes are then converted to hybrid zono-
topes, which are combined in a higher-dimensional space.
Definition 2:
[19, Def. 3] The set Zh ⊂Rn is a hybrid
zonotope if there exist Gc ∈Rn×ng, Gb ∈Rn×nb, c ∈Rn,

--- Page 3 ---
Ac ∈Rnc×ng, Ab ∈Rnc×nb, and b ∈Rnc such that
Zh =



h
Gc Gbi h ξc
ξb
i
+ c

h ξc
ξb
i
∈Bng
∞× {−1, 1}nb,
h
Ac Abi h ξc
ξb
i
= b


.
(1)
A hybrid zonotope is the union of 2nb constrained zono-
topes corresponding to all combinations of binary factors.
The hybrid zonotope is given in hybrid constrained gen-
erator representation and the shorthand notation of Zh =
⟨Gc, Gb, c, Ac, Ab, b⟩⊂Rn is used to denote the set given by
(1). Continuous and binary generators refer to the columns of
Gc and Gb, respectively. A hybrid zonotope with no binary
factors is a constrained zonotope, Zc = ⟨G, c, A, b⟩⊂Rn, and
a hybrid zonotope with no binary factors and no constraints is a
zonotope, Z = ⟨G, c⟩⊂Rn. Identities and time complexities
of linear transformations, Minkowski sums, generalized inter-
sections, and generalized half-space intersections are reported
in [19, Section 3.2]. An identity and time complexity for
Cartesian products is given in [21]. Methods for removing
redundant generators and constraints of a hybrid zonotope
were explored in [19] and developed further in [21].
C. Graphs of functions
A set-valued mapping ϕ : DΦ →Q assigns each element
p ∈DΦ to a subset, potentially a single point, of Q. We refer
to the set Φ = {(p, q) | p ∈DΦ, q ∈ϕ(p) ⊆Q} as a graph
of the function ϕ. The graph of a function consists of ordered
pairs of a function over a given domain. The set DΦ is referred
to as the domain set of Φ and can be chosen by a user as the
set of inputs of interest. Graphs of functions, and the related
concept of relations, have been used for reachability analysis
of nonlinear and hybrid systems, e.g., [29]–[31].
The authors have provided identities that leverage graphs
of functions for reachability of hybrid system [18], set-
valued state estimation [32], and reachability of nonlinear
system [23]. Some of the latter are included in Section III
(Theorem 1 and Theorem 3) for completeness and to motivate
the proposed identities for efficient construction of graphs of
functions for nonlinear systems. The proposed methods for
constructing graphs of nonlinear functions leverage hybrid
zonotopes, special ordered set approximations and functional
decomposition.
D. Successor sets
Consider a class of discrete-time nonlinear dynamics given
by f : Rn × Rnu →Rn
xk+1 = f(xk, uk) ,
(2)
with state and input constraint sets given by X ⊂Rn and
U ⊂Rnu, respectively. The ith row of f(xk, uk) is a scalar-
valued function and denoted by fi(xk, uk). Disturbances are
omitted for simplicity of exposition, although the results in
this paper extend to systems with disturbances. Because hybrid
zonotopes are the set representation of interest for this paper
and are inherently bounded, Assumption 1 is made.
Assumption 1: For all (x, u) ∈X × U, ||f(x, u)|| < ∞.
The successor set is defined as follows.
Definition 3: The successor set from Rk ⊆X with inputs
bounded by Uk ⊆U is given by
Suc(Rk, Uk) ≡
f(x, u) | x ∈Rk, u ∈Uk
	
.
(3)
Forward reachable sets from an initial set can be found by
recursion of successor sets (3), i.e., Rk+1 = Suc(Rk, Uk).
E. Special ordered sets
Special Ordered Set (SOS) approximations, a type of
piecewise-affine (PWA) approximations, were originally de-
veloped to approximate solutions of nonlinear optimization
programs [33]. We define an SOS approximation equivalently
to [34, Section 1.2]. An incidence matrix is introduced to
mirror the structure given to collections of V-rep polytopes
in Theorem 5.
Definition 4 (SOS Approximation): An SOS approximation
S of a scalar-valued function g(x) : Rn →R is the union of
N polytopes, i.e., S = ∪N
i=1Pi. The collection of polytopes
is defined by a vertex matrix V = [v1 · · · vnv] ∈R(n+1)×nv,
where vi = (xi, g(xi)) , ∀i, and a corresponding incidence
matrix M ∈Rnv×N with entries M(j,i) ∈{0, 1} , ∀i, j, such
that
Pi =





V λ

λj ∈
(h
0 ,
1
i
,
if j ∈{k | M(k,i) = 1}
{0},
if j ∈{k | M(k,i) = 0}
,
1Tnvλ = 1





,
(4)
([In 0]Pi)◦∩[In 0]Pj = ∅, ∀i ̸= j , and
(5)
1T M(·,i) ≤n + 1 , ∀i ∈{1, ..., N} .
(6)
The set of points xi, i ∈{1, ..., nv} are referred to as
the sampling of the first n dimensions. Each polytope Pi
given by (4) is the convex hull of the vertices given by V(·,i)
corresponding to the index of all entries of M(·,i) that equal
1. The constraint (5) enforces that none of the simplices’
interiors “overlap” while allowing for sharing of topological
boundaries. The constraint (6) enforces that Pi will be at most
an n-dimensional simplex (1T M(·,i) = n + 1) and a lower
dimensional simplex otherwise (1T M(·,i) < n + 1).
Example 1: Consider y = sin(x) for x ∈[−4, 4]. An SOS
approximation with 21 evenly spaced breakpoints is given by
vertex matrix V and incidence matrix M as
V =

−4
−3.6
−3.2
. . .
4
sin(−4)
sin(−3.6)
sin(−3.2)
. . .
sin(4)

, and
M =
 I20
01×20

+

01×20
I20

.
The first column of the incidence matrix
M(1,:) =

1
1
0
· · ·
0
T
corresponds to one section of the SOS approximation for the
domain x ∈[−4, −3.6], which is a 1-dimensional simplex.
F. Functional decomposition
Functions can be decomposed into unary and binary func-
tions with one or two scalar arguments, respectively. Con-
structing SOS approximations of the decomposed functions

--- Page 4 ---
TABLE I: Functional decomposition of T2 (9) with K = 5
compositions.
wj(wj1, ...)
hℓ
Dℓ
w1 = x1,k
[−4, 4]
w2 = x2,k
[−8, 8]
w3 = uk
[−20, 20]
w4(w1)
sin(w1)
[−1, 1]
w5(w1)
cos(w1)
[−1, 1]
w6(w5, w2)
w5w2
[−8, 8]
w7(w1, w2, w3, w4)
w1 + w2
10 + w3
200 + w4
20
[−4.95, 4.95]
w8(w2, w4, w3, w6)
w2 + w4 + w3
10 + w6
20
[−11.4, 11.4]
avoids exponential growth with respect to the argument di-
mension [34]. A function h(x) : Rn →Rm is decomposed by
introducing intermediate variables
wj =
(
xj ,
if j = 1, ..., n ,
hj(wj1{, wj2}) ,
if j = n + 1, ..., n + K ,
(7)
where j1, j2 < j, giving
h(x) =


wn+K−m+1
...
wn+K

.
The first n assignments directly correspond to the n elements
of the argument vector x, assignments n + 1, ..., n + K are
defined by the unary function or binary function hj, and the
final m assignments are associated with h(x). In the case that
hj is unary, the second argument is omitted.
Remark 1 (Affine Decompositions): Because hybrid zono-
topes are closed under linear transformation, functional com-
positions are also allowed to admit functions hj(·) that have
more than two arguments, provided that hj(·) is an affine
function of lower-indexed variables.
Example 2: Consider the inverted pendulum dynamics
given by
 ˙x1
˙x2

=

x2
g
l sin(x1) + u
I

,
(8)
with gravity g = 10, length l = 1, mass m = 1, and moment of
inertia I = ml2 = 1. The continuous-time nonlinear dynamics
are discretized with time step h = 0.1 using a 2nd-order Taylor
polynomial T2(xk) given by
T2(xk) =
("
x1,k +
x2,k
10 +
sin(x1,k)
20
+ uk
200
x2,k + sin(x1,k) +
x2,k cos(x1,k)
20
+ uk
10
#)
.
(9)
A functional decomposition of T2(xk) is shown in Table I.
For a chosen domain (x1,k, x2,k.uk) ∈DH = D1 × D2 × D3,
bounds on the intermediate and output variables can be found
by domain propagation via interval arithmetic.
Remark 2 (Existence of a Functional Decomposition):
The Kolmogorov Superposition Theorem [35] proves that
a continuous function f(·) defined on the n-dimensional
hypercube can be represented as the sum and superposition
of continuous functions of only one variable. This result
allows us to analyze the scalability of the proposed approach
in Section IV-C, although functional decompositions are
system-specific, not unique, and not always obvious to
find. Fortunately, decompositions of the form in (7) are
readily available for large classes of functions, such as
those containing basic operators (e.g., addition, subtraction,
multiplication, division), polynomials, trigonometric functions,
and boolean functions (e.g., AND). Additionally, they can
often be obtained by analyzing the order of operations within
an expression, although a decomposition strictly based on the
order of operations may not be most concise or useful.
III. REACHABILITY VIA STATE-UPDATE SETS
This section first introduces the open-loop state-update set
(Definition 5), which encodes all possible state transitions
of (2) over a user-specified domain of states and inputs,
and is used to calculate successor sets over discrete time
steps (Theorem 1). Then, after defining a state-input map
as all possible inputs of a given control law over a user-
specified domain of states (Definition 6), the set of possible
state transitions of the closed-loop system is constructed by
combining the state-input map and the open-loop state-update
set (Theorem 2). It is shown how this closed-loop state-update
set can be used to calculate successor sets of the closed-loop
system (Theorem 3). Then, a general method to construct
complex nonlinear sets using functional decomposition, later
used to construct open-loop state update sets and state-input
maps, is provided (Theorem 4). Finally, the effect of over-
approximations on the theoretical results of this section is
addressed (Corollary 3).
Definition 5: The open-loop state-update set Ψ ⊆R2n+nu
is defined as
Ψ ≡





xk
u
xk+1



xk+1 ∈Suc({xk}, {u}),
(xk, u) ∈DΨ


.
(10)
We refer to DΨ ⊂Rn+nu as the domain set of Ψ, typically
chosen as the region of interest for analysis.
Theorem 1: [23, Theorem 1] Given sets of states Rk ⊆Rn
and inputs Uk ⊆Rnu, and an open-loop state-update set Ψ, if
Rk × Uk ⊆DΨ, then the open-loop successor set is given by
Suc(Rk, Uk) =
0
In
  Ψ ∩[In+nu 0] (Rk × Uk)

.
(11)
The containment condition in Theorem 1, Rk × Uk ⊆DΨ,
is not restrictive as modeled dynamics are often only valid
over some region of states and inputs, which the user may
specify as DΨ = X × U.
Consider a set-valued function C(xk) corresponding to a
state-feedback controller, such that C(xk) is the set of all
possible inputs that the controller may provide given the
current state, xk. For example, for a linear feedback control
law given by u(xk) = Kxk with no actuator uncertainty,
C(xk) = {Kxk} would be a single vector. In the case of
a linear feedback control law with actuator uncertainty given
by u = Kxk + δu where δu ∈∆u, we would have C(xk) =
{Kxk + δu | δu ∈∆u}. The state-input map encodes the
feedback control law given by C(xk) as a set over a domain
of states.
Definition 6: The state-input map is defined as Θ
≡
{(xk, u) | u ∈C(xk), xk ∈DΘ}, where DΘ is the domain
set of Θ.

--- Page 5 ---
Next, the closed-loop state-update set under a controller
given by C(xk) is defined. Then it will be shown how to
construct a closed-loop state-update set given an open-loop
state-update set and a state-input map.
Definition 7: The closed-loop state-update set Φ ⊆R2n for
a controller given by C(xk) is defined as
Φ ≡
 xk
xk+1
 
xk+1 ∈Suc ({xk}, C(xk)) ,
xk ∈DΦ

,
(12)
where DΦ ⊂Rn is the domain set of Φ.
Theorem 2:
[23, Theorem 2] Given an open-loop state-
update set Ψ and state-input map Θ, the closed-loop state-
update set Φ with DΦ =
In 0
(DΨ ∩Θ) is given by
Φ =
In
0
0
0
0
In
 
Ψ ∩h
In+nu 0
i Θ

.
(13)
Theorem 3 provides an identity for the successor set of a
closed-loop system with the feedback control law described
by the set-valued function C(xk). For closed-loop successor
sets, the input set argument Uk is omitted and the successor
set is instead denoted by Suc(Rk, C).
Theorem 3:
[23, Theorem 3] Given a set of states Rk ⊆
Rn and closed-loop state-update set Φ, if Rk ⊆DΦ then the
closed-loop successor set is given by
Suc(Rk, C) =
0
In
  Φ ∩[In 0] Rk

.
(14)
The identities in (11), (13), and (14) utilize the open-loop
state-update set and state-input map. In general, these sets can
be complex and difficult to construct in a high-dimensional
space using methods that sample or partition the state space.
To more efficiently construct these sets, Theorem 4 utilizes
functional decomposition as given in Section II-F, thus only
considering unary and binary functions. While Theorem 4
addresses nonlinear functions with a single vector argument x,
it is easily applied to functions with multiple arguments, such
as (2), by concatenating the arguments into a single vector.
Theorem 4 can be used to construct open-loop state-update
sets and state-input maps, as is demonstrated in Section V.
Theorem 4: Consider a general nonlinear function h(x) :
Rn →Rm and its decomposition (7). Define the set H as
H ≡







w1
...
wn+K



(w1, w2, . . . , wn) ∈DH,
wj = hj(wj1{, wj2})
∀j ∈{n + 1, ..., n + K}





.
(15)
Given Dj ⊇[ej]H ∀j = n + 1, ..., n + K,1 and
Hℓ=





wℓ1
{wℓ2}
wℓ



(wℓ1{, wℓ2}) ∈Dℓ1{×Dℓ2},
wℓ= hℓ(wℓ1{, wℓ2})


,
(16)
for ℓ∈{n + 1, ..., n + K}, then H is given by initializing
H1:n = DH and iterating K times through
H1:ℓ= (H1:ℓ−1 × Dℓ) ∩

eℓ1
{eℓ2}
eℓ


Hℓ.
(17)
This yields H1:n+K = H.
1Sufficiently large Dj can be found using interval arithmetic.
Proof: From H1:n = DH and (17),
H1:n+K =







w1
...
wn+K



(w1, w2, ..., wn) ∈DH,
wj ∈Dj ∀j ∈{n + 1, ..., n + K},
wj = hj(wj1{, wj2}) ∀j ∈{n + 1, ..., n + K}





.
The constraint wn+1
∈Dn+1 is redundant given that
(w1, w2, ..., wn) ∈DH and wj = hj(wj1{, wj2}) for j =
n + 1, and therefore can be removed. The same is then true
∀j ∈{n + 2, ..., n + K}, yielding H as defined by (15).
Theorem 4 addresses compositions of unary and binary
functions, however the identity (17) is easily extended to en-
force relations across many input and output variables, similar
to how (13) couples an arbitrary number of signals between
an open-loop plant and a closed-loop controller. Corollary 1
addresses the special case when the decomposition includes
affine functions.
Corollary 1: Consider a general nonlinear function h(x) :
Rn →Rm, its decomposition (7), the set H defined by
(15), and the recursion (17). For all ℓwhere hℓ(·) is affine,
i.e., hℓ(wℓ1{, wℓ2}) = mℓ1wℓ1{+mℓ2 wℓ2} + bℓwhere mℓ1,
{mℓ2}, and bℓare scalars, the set H is equivalently given when
(17) is replaced by
H1:ℓ=

Il−1
mℓ1eℓ1{+mℓ2eℓ2}

H1:ℓ−1 +

0
1

bℓ.
(18)
Proof:
The
proof
only
requires
recognizing
that
mℓ1wℓ1{+mℓ2 wℓ2} + bℓ= hℓ(wℓ1{, wℓ2}) when hℓ(·) is
affine to arrive at (18) and then follows the same procedure
as the proof of Theorem 4.
Although Corollary 1 is written for unary or binary affine
functions, it is easily extended to functions with an arbitrary
number of arguments, i.e., hℓ(w1, ...) = bℓ+ P
i mℓiwℓi.
Applying Corollary 1 reduces the memory complexity of
H (15) when implemented with hybrid zonotopes, as the
affine transformation in (18) does not increase the memory
complexity of the hybrid zonotope.
Corollary 2 provides an identity to remove intermediate
variables that arise from the functional decomposition, and
only retains dimensions corresponding to the n arguments and
m outputs of h(x).
Corollary 2: Consider a general nonlinear function h(x) :
Rn →Rm and its decomposition (7). Given the set H as
defined by (15),
(
x
y
 
x ∈DH
y = h(x)
)
=

In
0
0
0
0
Im

H .
(19)
Proof:
By definition of the linear transformation, the
right side of (19) yields





















w1
...
wn
wn+K−m+1
...
wn+K



(w1, w2, . . . , wn) ∈DH,
wj = hj(wj1{, wj2})
∀j ∈{n + 1, ..., n + K}



















.

--- Page 6 ---
Substituting hj(·) ∀j ∈{n + 1, ..., n + K} yields the desired
result.
A fundamental challenge of reachability analysis is that a
given set representation and its operations can only achieve
efficient computation of exact successor sets for a limited
class of systems [36]. To obtain formal guarantees for broader
system classes, over-approximations of successor sets are often
computed instead [37]. To this end, Corollary 3 extends
the previous results by first considering the effect of using
over-approximations of sets on the right side of the identities
in (11), (13), (14), and (17). Then Corollary 4 provides
an identity to produce sets that over-approximate nonlinear
functions from set-based approximations of those functions
and bounds on their approximation error.
Corollary 3: For the identities provided by Theorems 1-4
and Corollary 2, if any set on the right side is replaced by
an over-approximation, then the identity will instead yield an
over-approximation of the left side.
Proof: Set containment is preserved under linear trans-
formation and generalized intersection.
For example, if an over-approximation of the open-loop
state-update set, given by ¯Ψ, is used in place of Ψ in (11), then
the right side of the equation will yield an over-approximation
of the reachable set, Suc(Rk, Uk).
Corollary 4: Given
an
approximation
of
a
function
ˆhℓ(wℓ1{, wℓ2}) ≈hℓ(wℓ1{, wℓ2}), its corresponding set ˆHℓ≈
Hℓfrom (16) defined over the same domain D ˆ
Hℓ= DHℓ,
and an error bound given by the interval [a, b] such that
hℓ(wℓ1{, wℓ2}) ∈[ˆhℓ(wℓ1{, wℓ2}) + a, ˆhℓ(wℓ1{, wℓ2}) + b]
for all (wℓ1, {wℓ2}) ∈DHℓ, an over-approximation of Hℓis
given by
Hℓ⊆ˆHℓ⊕


0
{0}
1

a, b
.
(20)
Proof: Defining the right side of (20) as ¯Hℓ,
¯
Hℓ=





ωℓ1
{ωℓ2}
ωℓ





(ωℓ1, {ωℓ2}) ∈DHℓ,
ωℓ∈
[ˆhℓ(wℓ1, {wℓ2}) + a, ˆhℓ(wℓ1, {wℓ2}) + b]




.
Corollary 4 assumes that a bound on the approximation
error is known. Generally, for a function f(x) approximated
over a domain by an affine approximation ¯f(x), error bounds
can be posed as the nonlinear programs
a = min
x∈X
  ¯f(x) −f(x)

,
(21)
b = max
x∈X
  ¯f(x) −f(x)

,
(22)
where X is the domain of the approximation. In general
this problem is challenging, though by decomposing func-
tions into unary or binary functions, (21) only needs to be
solved for each unary and binary nonlinear function of the
decomposition, for which x ∈R1 or R2. Additional properties,
such as whether f(x) is continuous, differentiable, convex, or
concave, can significantly reduce the complexity of solving
(21). Error bounds for affine approximations of some nonlinear
functions can be found in [38, Chapter 3], e.g., for x2 a closed-
form solution to (21) is found by leveraging its convexity
and differentiability. In the case of PWA approximations, the
process can be repeated for each partition of the domain and
the worst-case error bounds should be used.
Although Theorem 4 is primarily beneficial for functions
with many arguments, Example 3 uses a system with one input
to facilitate exposition and visualization. More complex exam-
ples of constructing state-update sets of nonlinear systems are
considered in Section V.
Example 3: A
functional
decomposition
of
xk+1
=
cos ( π sin (xk)) over a domain DH = [−π, π]
and its visual representation are given by Table II and
Figure 2, respectively. Over-approximations
¯H2 and
¯H3
are constructed using SOS approximations and closed-form
solution error bounds from [38], which are combined using
Corollary 4. The effect of over-approximation per Corollary
3 is demonstrated in Figure 2(e)-(h). In Figure 2(d) and 2(h),
Corollary 2 is used as the last step in construction of the
state-update set Φ and its over-approximation ¯Φ, which are
shown in magenta.
TABLE II: Functional decomposition and domain propagation
of xk+1 = cos(π sin(xk)) over a domain DH = [−π, π].
wj
Hj
Dj
w1 = xk
[−π π]
w2
π sin(w1)
[−π π]
w3 = xk+1
cos(w2)
[−1 1]
IV. REACHABILITY OF NONLINEAR SYSTEMS USING
HYBRID ZONOTOPES
The remainder of this paper assumes that reachable sets
and state-update sets are represented as hybrid zonotopes.
Hybrid zonotopes are closed under linear transformations,
generalized intersections [19], and Cartesian products [21].
The time complexity of the open-loop and closed-loop suc-
cessor set identities in (11) and (14), respectively, is O(n),
as the linear transformations [I 0] under the generalized
intersections amount to matrix concatenations. The resulting
memory complexity is given by
Open
ng,Suc = ng,r + ng,u + ng,ψ,
nb,Suc = nb,r + nb,u + nb,ψ,
nc,Suc = nc,r + nc,u + nc,ψ + n,
Closed
ng,Suc = ng,r + ng,ϕ,
nb,Suc = nb,r + nb,ϕ,
nc,Suc = nc,r + nc,ϕ + n .
Reachable sets found by recursion of these identities exhibit
linear memory complexity growth with respect to time.
Because hybrid zonotopes cannot exactly represent general
nonlinear functions, a trade-off between over-approximation
error and complexity arises. The scalability of Theorem 4 for
constructing hybrid zonotopes containing nonlinear functions
(for state-update sets, state-input maps, etc.) depends both
on the functional decomposition (7) and the complexity of
over-approximating unary and binary functions with hybrid
zonotopes. Theorem 5 addresses the latter and additional
results for efficient representation of common binary functions
are given in Section IV-B. These are followed in Section IV-
C with a discussion of the scalability when a decomposition
based on the Kolmogorov Superposition Theorem is used.

--- Page 7 ---
(a)
(b)
(c)
(d)
(e)
(f)
(g)
(h)
Fig. 2: Visual depiction of functional decomposition and Theorem 4 applied to xk+1 = cos(π sin(xk)) with the decomposition
shown in Table II. Here, H1:ℓdenotes a stage in the recursion of Theorem 4. H1:1 and H1:1 × D2 are a line and a rectangle,
respectively, and are not shown. (a) The first of two recursions of (17) yields H1:2 (blue). (b) The first step in the second
recursion of (17) gives H1:2 ×D3 (also shown in blue). (c) The second step of the second recursion of (17) takes a generalized
intersection with H3 (red) and yields H1:3 (black). (d) Corollary 2 eliminates the intermediate variable w2 using a projection
and is the last step in producing the state-update set Φ (magenta). (e-h) Per Corollary 3, over-approximation of the functions
used for decomposition results in an over-approximation of the state-update set ¯Φ ⊃Φ.
A. Converting vertex representation to hybrid zonotope
An identity to convert SOS approximations to hybrid zono-
topes was provided in [23]. Theorem 5 provides a more
general result for converting a collection of V-rep polytopes
into a single hybrid zonotope. Beyond SOS approximations,
the result can be applied to represent complex initial or unsafe
sets efficiently as a hybrid zonotope.
Theorem 5: A set S consisting of the union of N V-rep
polytopes, S = ∪N
i=1Pi, with a total of nv vertices can be
represented as a hybrid zonotope with memory complexity
ng = 2nv, nb = N, nc = nv + 2 .
(23)
Proof:
Define the vertex matrix V = [v1, . . . , vnv] ∈
Rn×nv and construct a corresponding incidence matrix M ∈
Rnv×N with entries M(j,i) ∈{0, 1} ∀i, j, such that
Pi =





V λ
 λj ∈
(
[0 1],
if j ∈{k | M(k,i) = 1}
{0},
if j ∈{k | M(k,i) = 0}
1T
nvλ = 1





.
Define the hybrid zonotope
Q = 1
2
Inv
0

,
 0
IN

,
1nv
1N

,

1T
nv
0

,
 0
1T
N

,
2 −nv
2 −N

,
and the polyhedron H = {h ∈Rnv | h ≤0}, and let
D = Q ∩[Inv −M] H .
(24)
Then the set S is equivalently given by the hybrid zonotope
ZS =
V
0
D .
(25)
By direct application of set operation identities provided in
[19, Section 3.2], it can be shown that ZS has the complexity
given by (23). The remainder of the proof shows equivalency
of ZS and S. For any (λ, δ) ∈D there exists some (ξc, ξb) ∈
Bnv
∞× {−1, 1}N such that 1T
nvξc = 2 −nv, 1T
Nξb = 2 −N,
λ = 0.5ξc+0.51nv, δ = 0.5ξb+0.51N, and λ−Mδ ∈H =⇒
λ ≤Mδ. Thus λ ∈[0, 1]nv, δ ∈{0, 1}N, Pnv
i=1 λi = 1, and
PN
i=1 δi = 1 results in δi = 1 =⇒δj̸=i = 0. Let δi = 1,
then λ ≤Mδ enforces λj ∈[0, 1] ∀j ∈{k | M(k,i) = 1} and
λj = 0 ∀j ∈{k | M(k,i) = 0}. Therefore given any z ∈ZS
corresponding to δi = 1,
z =
X
λjvj ∀j ∈{k | M(k,i) = 1} ,
(26)
thus z ∈Pi ⊆S and ZS ⊆S.
Conversely, given any x ∈S, ∃i such that x ∈Pi =
P λjvj ∀j ∈{k | M(k,i) = 1}. This is equivalent to (26),
therefore x ∈ZS, S ⊆ZS, and ZS = S.
The
computational
complexity
of
Theorem
5
is
O(n(nv + N)2). The pre-existing approach for converting a
collection of V-rep polytopes to a single hybrid zonotope
would be to convert each polytope from V-rep to H-rep to
HCG-rep and then take iterative unions using the method given
in [22]. However, this would involve greater computational
complexity, as conversion between V-rep and H-rep alone has
worst-case exponential computational complexity [39], and
would produce a set with memory complexity that scales
quadratically with the number of polytopes. Furthermore The-
orem 5 allows for the intuitive representation of sets as hybrid
zonotopes through the use of the vertex matrix and incidence
matrix, as demonstrated in Example 4.
Example 4: Consider a vertex matrix V
= [v1, v2, v3]
consisting of the vertices of a triangle. The set of vertices, the
set of points along the edges of the triangle, and the convex

--- Page 8 ---
TABLE III: Functional decompositions of xy,
x
y , and xy
using 2, 3, and 4 unary nonlinear functions, respectively,
with additional affine functions. Equivalency of each binary
function with the highest indexed variable can be shown using
substitution.
xy
x
y
xy
w1
x
x
x
w2
y
y
y
w3
w1 + w2
1
w2
ln w1
w4
w1 −w2
w1 + w3
w1 + w3
w5
w2
3
w1 −w3
w1 −w3
w6
w2
4
w2
4
w2
4
w7
1
4 (w5 −w6)
w2
5
w2
5
w8
-
1
4 (w6 −w7)
1
4 (w6 −w7)
w9
-
-
ew8
hull of the vertices can be found as a hybrid zonotope using
Theorem 5 and the respective incidence matrices
Mvertices = I3 , Medges =


1
0
1
1
1
0
0
1
1

, M∆=


1
1
1

.
Example 5 demonstrates the use of Theorem 5 to represent
an SOS approximation of sin(x) as a hybrid zonotope.
Example 5: Figure 3 shows y = sin(x) (green) for x ∈
[−4, 4] and an SOS approximation (red) with the vertex and
incidence matrices given in Example 1. The SOS approxima-
tion ZS is represented as a hybrid zonotope using Theorem
5. An envelope ¯Zsin(x) ⊃{(x, sin(x)) | x ∈[−4, 4]}, shown
in blue, is calculated using Corollary 4 and rigorous error
bounds for SOS approximations given by [38, Chapter 3].
B. Common binary functions
This section demonstrates how the memory complexity of
hybrid zonotopes approximating graphs of several common
binary functions can be reduced as compared to sampling
in both arguments of the functions. This is achieved using
functional decompositions composed exclusively of unary
nonlinear functions and affine functions, leveraging separa-
ble functions [40, Chapter 7.3] and extensions presented in
[41, Section 4.2]. When nonlinear functions within functional
decompositions are strictly unary, SOS approximations can be
constructed by sampling one-dimensional spaces. For example,
in the case of xy, x
y , and xy, where x, y ∈R, this reduces
the task from sampling a two-dimensional space to sampling
a one-dimensional space 2, 3, and 4 times respectively. The
decompositions for each of these expressions are shown in
Table III and graphs of the decompositions can be constructed
using methods presented in Section III.
Example 6: To demonstrate the scalability advantage of
constructing graphs of binary functions using functional de-
compositions with only affine functions and unary nonlinear
functions, consider three methods for building a hybrid zono-
tope approximation of the bilinear function xy. These methods
are: (M1) uniformly sampling the two-dimensional input space
and generating a hybrid zonotope using Theorem 5, (M2)
TABLE IV: Complexity comparison of three methods for ap-
proximating the bilinear function f(x, y) = xy. The leftmost
column denotes (nx for M1, n3 for M2, nx for M3) and these
values are selected for parity in their approximation accuracy.
Case
M1
nx = ny
M2
n3 = n4
M3
nx (ny = 2)
(3,9,18)
ng = 19
nb = 5
nc = 11
ng = 24
nb = 10
nc = 14
ng = 41
nb = 10
nc = 22
(6,17,34)
ng = 73
nb = 26
nc = 38
ng = 40
nb = 18
nc = 22
ng = 73
nb = 18
nc = 38
(9,26,52)
ng = 163
nb = 65
nc = 83
ng = 56
nb = 26
nc = 30
ng = 105
nb = 26
nc = 54
(12,34,68)
ng = 289
nb = 122
nc = 146
ng = 72
nb = 34
nc = 38
ng = 137
nb = 34
nc = 70
decomposing the function according to Table III, sampling
the w5(w3) and w6(w4) unary functions, generating hybrid
zonotope approximations of the quadratic functions using
Theorem 5, and constructing a graph of the bilinear function
using methods from Section III, and (M3) uniformly sampling
in one dimension and generating a hybrid zonotope using
Theorem 5. The intuition of M3 is that for a fixed value
of x, xy is linear with respect to the y dimension. Thus it is
possible to to obtain tighter over-approximations of xy by only
increasing sampling of the x dimension (or the y dimension).
The domain of interest is specified as (x, y) ∈[−1, 1]2 →
(w3, w4) ∈[−2, 2]2. For M1 and M2, nx,M1 = ny,M1 and
nw3,M2 = nw4,M2. The number of samples in the w3 and w4
spaces, nw3,M2, is chosen as nw3,M2 = ⌈
√
2nx,M1⌉to ensure
that the spacing of the approximation for M2 partitions the
domain into squares with a side length at least as small as
those using M1. To compare the scalability of M3 with M2,
the sampling for M3 is chosen to result in the same number of
binary factors as M2 by setting nx,M3 = 2n5,M2 and ny,M3 =
2.
Table IV compares the complexity of the three methods,
demonstrating how decomposing binary functions into affine
functions and unary nonlinear functions enables more scalable
over-approximations of the bilinear function as the sampling
is increased and over-approximations become more accurate.
Figure 4 plots over-approximations of the bilinear function
using all three methods, corresponding to Case (6, 17, 34)
in Table IV. The partitioning in Figure 4(a) and Figure
4(c) corresponds to sampling in the (x, y) space while the
partitioning in Figure 4(b) corresponds to sampling in the
(w3, w4) space.
C. Avoiding the curse of dimensionality
This section provides a theoretical bound on the memory
complexity of the open-loop state-update set Ψ and state-input
map Θ (together these bound the complexity of the closed-loop
state-update set Φ via Theorem 2) to quantify the scalability of
the proposed methods with respect to the state dimension and
number of vertices used to approximate nonlinear functions.
Consider a general nonlinear function h(x) : Rn →Rm.
We assume that the functional decomposition consists of n

--- Page 9 ---
Fig. 3: A sinusoid (green) is approximated using an SOS approximation for x ∈[−4, 4] and represented as a hybrid zonotope
(red). Using formal bounds for SOS approximation error, the SOS approximation is bloated in the output dimension to create
an enclosure of a sine wave for x ∈[−4, 4], which is also represented as a hybrid zonotope (blue).
(a) M1: nx = 6
(b) M2: n3 = 17
(c) M3: nx = 34, ny = 2
Fig. 4: Comparison of three methods for over-approximating {
x
y
xyT ) | (x, y) ∈[−1 1]2}.
intermediate variables corresponding to the dimensions of
x, Kaff intermediate variables corresponding to multivariate
affine functions, and KNL intermediate variables correspond-
ing to nonlinear functions that are unary or binary. We refer to
the set of indices such that hℓ(·) is nonlinear as N. Assuming
DH and Dℓ(17) are intervals, the memory complexity of H
as constructed by Theorem 4 and Corollary 1 in terms of the
complexities of Hℓ, ∀ℓsuch that hℓ(·) is nonlinear, is given
by
ng,H = n + KNL +
X
ℓ∈N
ng,Hℓ,
(27)
nb,H =
X
ℓ∈N
nb,Hℓ,
(28)
nc,H =
X
ℓ∈N
 nc,Hℓ+ nℓ

,
nℓ=
(
2
if hℓ(·) is unary ,
3
if hℓ(·) is binary . (29)
The computational complexity of constructing H using Theo-
rem 4 and Corollary 1 is dominated by the KNL generalized
intersection recursions of (17). Each generalized intersection
has computational complexity O(ℓ(ng + nb)), where ng and
nb grow with the complexity of Hℓ.
For a given system, the decomposition plays a critical role in
the memory complexity of the resulting set H defined in (15).
Thus, quantifying the scalability of the proposed approaches
for constructing graphs of functions with respect to the state
dimension is challenging, as it is expected that KNL will
grow with n. To provide a theoretical bound for the proposed
method, let us assume that a decomposition based on the Kol-
mogorov Superposition Theorem [35] is performed for each
hi(x), ∀i ∈{1, ..., m}. Then the number of unary nonlinear
decomposition functions is given by KNL = 2mn2 + mn
and no binary nonlinear decomposition functions are required.
Assuming that each unary nonlinear decomposition function is
approximated using an SOS approximation with nv vertices,
the resulting complexity of H is given by
ng,H = n + (2mn2 + mn)(2nv + 1) ,
nb,H = (2mn2 + mn)(nv −1) ,
(30)
nc,H = 8mn2 + 4mn .
It is clear from (30) that the memory complexity of H scales
as a polynomial with n, avoiding the exponential growth as-
sociated with the curse of dimensionality. In comparison, any
method that sampled the n-dimensional space to generate an
approximation would scale as nn
v, not including any additional
complexity incurred to generate H from those samples.
V. NUMERICAL EXAMPLES
Results in this section were generated with MATLAB on a
desktop computer with a 3.0 GHz Intel i7 processor and 16
GB of RAM. Reachable sets were plotted using techniques
from [19] and [21].
A. Single pendulum controlled by a neural network
This numerical example demonstrates reachability analysis
of a nonlinear inverted pendulum in closed loop with a
neural network controller trained to mimic Nonlinear Model
Predictive Control (NMPC). This is accomplished by con-
structing an over-approximation of the open-loop state-update
set Ψ (V-A.1) and a state-input map for the neural network
controller Θ (V-A.2), which are then combined to form an
over-approximation of the closed-loop state-update set Φ (V-
A.3) and used for reachability analysis (V-A.4).
Consider the dynamics of an inverted pendulum given by (8)
with gravity g = 10, length l = 1, mass m = 1, and moment of
inertia I = ml2 = 1. The continuous-time nonlinear dynamics

--- Page 10 ---
are discretized with time step h = 0.1 using a 2nd-order Taylor
polynomial T2(xk) (9), and over-approximated as
xk+1 ∈T2(xk) ⊕L ,
(31)
where L is constructed using the Taylor inequality to bound the
error due to truncating higher-order terms. The input torque is
controlled in discrete time with a zero-order hold and bounded
as uk ∈[−20, 20] ∀k.
1) Construction of open-loop state-update set: A functional
decomposition of T2(xk) is performed and, for a chosen
D(H) = D1 × D2 × D3, domain propagation is performed
to determine Dj ∀j ∈{4, 5, 6}. This is shown in Table I. This
choice of the domain also allows us to construct L in (31) as
L =
−0.02, 0.02
×
−0.26, 0.26
. The sets ¯Hℓ∀ℓ∈{4, 5, 6}
are constructed using Theorem 5, Corollary 4 and error
bounds given for SOS approximations in [34]. The set ¯H using
Theorem 4 with a Minkowski summation to account for the
Taylor remainder L yields an over-approximation of the open
loop state-update set,
Ψ ⊂¯Ψ = ¯H ⊕


0
0
0
0
0
0
1
0
0
1


L .
(32)
A projection of ¯Ψ is shown in Figure 5(a). The thickness of
the set in the x2,k dimension is primarily due to the open-
loop state-update set capturing variability in the input uk ∈
[−20, 20], though some of this is also a result of the Taylor
remainder L and over-approximation of nonlinear functions,
with ¯Hℓ⊃Hℓ∀ℓ∈{4, 5, 6}.
Computing ¯Hℓ∀ℓ∈{4, 5, 6} required 0.02 seconds, and ap-
plication of exact complexity reduction techniques from [19],
[21] required an additional 2 seconds. From this, ¯Ψ with
memory complexity given in Table V, was computed in 8
milliseconds using Theorem 4 and Corollary 1.
TABLE V: Memory complexity of the open-loop state-update
set, state-input map, and closed-loop state-update set.
Set
ng
nb
nc
¯Ψ
113
45
63
Θ
76
20
56
¯Φ
184
63
117
2) Construction of state-input map: To train the neural net-
work controller, an NLMPC is formulated as
min
u(·)
10
X
k=1
xT
k
100
0
0
1

xk + uT
k−1uk−1
(33)
s.t.
trapezoidal discretization of (8) holds ,
uk ∈[−20, 20] ∀k ∈{0, ..., 9} .
The solution of (33) is found using the MATLAB Model
Predictive Control Toolbox [42] for 400 uniformly sampled
initial conditions. The sampled initial conditions and the first
optimal input of the solution trajectory u∗
0 are then used as
input-output pairs to train a neural network with 2 hidden
(a) Projection of ¯Ψ
(b) Θ
(c) Projection of ¯Φ
Fig. 5: (a) Projection of over-approximated open-loop state-
update set ¯Ψ bounding dynamics of a pendulum at discrete
time steps. (b) State-input map Θ of a neural network trained
to mimic NMPC. (c) Projection of over-approximated closed-
loop state-update set found using Theorem 2 by coupling ¯Ψ
and Θ.
layers, each with 10 nodes and Rectified Linear Unit (ReLU)
activation functions, using the MATLAB Deep Learning Tool-
box. A functional decomposition of the neural network with
saturated output to obey torque constraints is performed and
Theorem 4 is used to generate the state-input map Θ in a
similar fashion as the over-approximation of the open-loop
state-update set Ψ. This decomposition is omitted for brevity,
however we note that the state-input map can be represented
exactly in this case, as the only nonlinear functions involved
are the ReLU activation function and saturation, which can
both be represented exactly using hybrid zonotopes. The state-
input map is shown in Figure 5(b). Using Theorem 4 and
Corollary 1, this took 4 seconds to compute and an additional
45 seconds to apply the exact complexity reduction techniques
from [19], [21].
3) Construction of closed-loop state-update set: Given an
over-approximation of the open-loop state-update set ¯Ψ and
the exact state-input map Θ, an over-approximation of the
closed-loop state-update set ¯Φ was constructed using the
identity in Theorem 2 in less than 1 millisecond and exact re-
duction methods were completed in an additional 35 seconds.
A projection of ¯Φ is shown in Figure 5(c). This projection is a
subset of the projection of ¯Ψ in Figure 5(a), as variability in the
input is eliminated when creating the closed-loop state-update

--- Page 11 ---
set using the state-input map Θ. Although difficult to perceive
in the figure, some thickness in the xk,2 dimension remains
as a result of the Taylor remainder L and over-approximating
nonlinear functions of the plant model. Table V reports the
memory complexity of ¯Ψ, Θ, and ¯Φ for this example.
4) Forward reachability: Using Corollary 3 and iteration
over the identity in (14), over-approximations of forward
reachable sets Ri, i ∈{1, ..., 15} are calculated from an initial
set given by
R0 =
−π,
π
×
−0.1,
0.1
.
(34)
The over-approximated reachable sets up to R3 are plotted in
Figure 6(a), overlaid by exact closed-loop trajectories found
by randomly sampling points in X0 and propagating using
(8). Examination of the exact trajectories exemplifies suc-
cessful nonconvex over-approximation of the reachable sets.
Computation time to execute the successor set identity was 5
milliseconds per time step on average.
To handle growth in set memory complexity over time steps,
set propagation methods often utilize over-approximations
to reduce complexity. Using techniques from [21], over-
approximations of the reachable set are taken periodically
every three time steps beginning at k = 3, resulting in
4 total approximations that took an average of 49 seconds
each to compute, in a manner similar to [43]. At the time
steps corresponding to over-approximations, the set is first
saved and analyzed before being over-approximated. The over-
approximation is used to calculate the reachable set of the
subsequent time step. Figure 6(b) plots the over-approximated
reachable sets and Figure 6(c) plots the corresponding memory
complexity. The periodic memory complexity reduction is
apparent in Figure 6(c). It is clear that the containment
condition of Theorem 3, Ri ⊆D(Φ), is met each time step.
While plotting the reachable sets can be used to visually
confirm performance and the containment condition of The-
orem 3, this is a computationally expensive process, taking
7378 seconds to produce Figure 6(b). Much of the same
information can be obtained with lower computational burden
by sampling the support function in the axis-aligned directions,
which took a total of 34 seconds for all 15 steps, less than
0.5% of the time to plot.
B. ARCH AINNCS benchmark: Single pendulum
This section compares the proposed approach to the state-of-
the-art using a modified version of the single pendulum bench-
mark in the Artificial Intelligence and Neural Network Con-
trol Systems (AINNCS) category of the Applied veRification
for Continuous and Hybrid Systems (ARCH) workshop [24,
Section 3.5] , [25, Section 3.5]. This benchmark problem is
similar to that presented in Section V-A, with different values
for g, l, m and a neural network with two hidden layers, each
with 25 ReLU activation nodes. While the stated objective
of the benchmark is to falsify or verify a safety condition, to
demonstrate the proposed approach, this is modified to instead
focus exclusively on calculating reachable sets over the time
period t ∈

0,
1

seconds with a time step of 0.05 seconds.
(a)
(b)
(c)
Fig. 6: (a) Over-approximation of reachable sets R0 →R3 of
the inverted pendulum in closed-loop with a saturated neural
network controller, overlaid by samples of exact trajectories in
green. (b) Over-approximated reachable sets R0 →R15 with
over-approximations taken every three time steps. (c) Memory
complexity of the over-approximated reachable sets.
We consider a small initial set matching that in [25, Section
3.5] and a large initial set. These sets are given by
Small initial set: R0 =
1,
1.2
×
0,
0.2
, and
Large initial set: R0 =

0,
1

×

−0.1,
0.1

.
The proposed method follows the same procedure to con-
struct the open-loop state-update set, state-input map, closed-
loop state-update set, and reachable sets as done in the
previous example. The complexity of the closed-loop state-
update set is ng,ϕ = 321, nb,ϕ = 75, nc,ϕ = 240. As done for

--- Page 12 ---
TABLE VI: Comparison of computation times in seconds of
state-of-the-art tools for reachability analysis of an inverted
pendulum with a neural network controller. Diverging approx-
imation error is noted when experienced.
Tool
Small Initial Set
Large Initial Set
Proposed Method
312
2377
CORA
0.5
0.6
Diverged
JuliaReach
0.5
0.7
Diverged
NNV
2086
>7200
Diverged
POLAR
0.2
0.15
Terminated after
7 steps
the previous example, reachable sets are over-approximated
every 3 time steps to handle growth in memory complexity.
The proposed method is compared to all four state-of-the-
art tools that participated in the AINNCS category in 2022
and 2023, namely CORA [44], JuliaReach [45], NNV [46],
and POLAR [47]. Computation times and an indication when
reachable sets diverge with untenable over-approximation error
for the small initial set and large initial set are given in
Table VI. For the small initial set, all methods successfully
generate reachable sets without diverging approximation error
and the proposed method is substantially slower than the other
tools. For the large initial set, the proposed method was the
only method to compute the reachable set without diverging
approximation error, though it required 40 minutes to do so.
CORA, JuliaReach, NNV, and POLAR each have tuning
parameters that allow the user to adjust a trade-off between
computation time and error. The results reported for the large
initial set were generated with the same tuning parameters
as used for the small initial set. The authors were unable
to find alternative tuning parameters that avoided diverging
over-approximating error for the large initial set. It is possible
for CORA, JuliaReach, NNV, and POLAR to accommodate
the large initial set via partitioning. For example, when the
initial set is partitioned into 160 subsets, NNV can compute
the reachable sets without diverging approximation error. In
general, this may require a significant number of partitions,
especially for systems with many states, that must be tuned to
the problem at hand.
More than 96% of the total computation time of the
proposed method is spent calculating periodic convex over-
approximations. The time to reduce the order of the closed-
loop state-update set is greater than 3% of the total compu-
tation time. The time spent constructing the open-loop state-
update set, state-input map, closed-loop state-update set, and
successor sets for all time steps is only 8 seconds. Periodic
over-approximations are taken to limit the complexity of the
resulting sets, which also limits the complexity of analyzing
them. This motivates work to efficiently generate hybrid zono-
tope over-approximations with acceptable approximation error,
though approximations of hybrid zonotopes are challenging
due to their implicit nature [21].
C. High-dimensional Boolean function
We adopt the following example from [28]. Consider the
Boolean function with xi, ui ∈{0, 1}20, i ∈{1, 2, 3}
x1,k+1 = u1,k ∨(x2,k ⊙x1,k) ,
x2,k+1 = x2,k ⊙(x1,k ∧u2,k) ,
(35)
x3,k+1 = x3,k∼∧(u2,k ⊙u3,k) ,
where ∨, ⊙, ∧, and ∼∧denote the standard Boolean functions
OR, XNOR, AND, and NAND, respectively. Boolean func-
tions can easily be represented as hybrid zonotopes, e.g., the
set representing OR





s1
s2
s1 ∨s2


 (s1, s2) ∈{0, 1}2


,
(36)
is equivalently given as the set of four points





0
0
0

,


1
0
1

,


0
1
1

,


1
1
1




,
(37)
which can be converted to a hybrid zonotope by concatenating
the points (37) into a matrix V , using the incidence matrix
M = I4, and applying the identity given in Theorem 5. Note
that the resulting set maps two inputs consisting of the four
combinations of Boolean values 0 and 1 to a single output
of a Boolean value. Using a functional decomposition and
Theorem 4, an exact open loop state-update set Ψ for (35)
can be generated. Generating Φ in HCG-rep, including order
reduction, took 0.82 seconds.
An initial set for (x1,k, x2,k, x3,k) consists of 8 possible
values. Input sets for (u1,k, u2,k,
u3,k) also consist of 8
possible values. Reachable sets are calculated for 30 time
steps. Computation times are plotted in Figure 7 comparing
the methods presented here using HCG-rep to those developed
for polynomial logical zonotopes. Beyond 5 steps, the compu-
tation time for polynomial zonotopes is on the order of hours
due to exponential growth in the set complexity resulting from
iterative AND (and NAND) operations, consistent with the
results given in [28, Table 3]. Computation time using hybrid
zonotopes and the proposed methods scales polynomially with
time.
D. ARCH AINNCS benchmark: Vertical collision
avoidance system
As a second comparison to state-of-the-art reachability
tools, we study the Vertical Collision Avoid System (VCAS)
benchmark from the ARCH AINNCS category [24], [25]. The
plant is a linear discrete-time model with 3 states, given by
hk+1 = hk −˙h0,k −1
2
¨hk ,
˙h0,k+1 = ˙h0,k + ¨hk ,
τk+1 = τk −1 ,
where hk is the relative height of the ownship from an intruder
flying at a constant altitude, ˙h0,k is the derivative of the height
of the ownship, and τk ∈{25, 24, ..., 15} is the time until the

--- Page 13 ---
Fig. 7: Computation time of reachable sets of a logical
function. Two lines are plotted for HCG-rep, one including
the time to generate the state-update set (total) and one that
only includes the computation time associated with generation
of reachable sets using (14). The latter better demonstrates
that computation time scales with N. Computation times for
polynomial logical zonotopes are also shown.
ownship and intruder are no longer horizontally separated. A
controller state advk ∈{1, 2, ..., 9}, denotes the flight advisory,
for which there are corresponding choices of ¨h to be selected
by the pilot. At each time step, one of 9 neural networks, each
with 5 fully connected hidden layers and 20 ReLU nodes per
layer, is selected based on the previous time step advisory.
Each neural network has 3 inputs, corresponding to hk, ˙h0,k,
and τk, and has 9 outputs, i.e., fNN,i(·) : R3 →R9 ∀i ∈
{1, ..., 9}. The index of the largest output of the neural network
corresponding to the previous time step advisory determines
the advisory for the current time step. If the previous advisory
and current advisory coincide, and ˙h0,k complies with the
advisory, then ¨hk = 0. Otherwise, ¨hk is selected according
to the current advisory. The advisories, their associated ranges
of compliant ˙h0,k, and corresponding choices of ¨hk are listed
in Table VII.
In [24], [25], CORA, JuliaReach, and POLAR do not
support reachability analysis of the closed-loop VCAS dynam-
ics. Both CORA and JuliaReach provide custom simulation
algorithms to falsify the VCAS benchmark, but do not employ
reachability algorithms. NNV is the only tool to employ
reachability algorithms for this benchmark problem and is able
to verify/falsify various initial conditions by partitioning the
set and omitting the multiple choices for ¨h given an advisory
by assuming a “middle” (middle ¨h is chosen) or “worst-case”
(¨h that results in driving h closest to 0) strategy.
The proposed method is able to perform reachability anal-
ysis of this benchmark problem without this modification. We
first generate a hybrid zonotope graph of the neural network
associated with each of the 9 advisories. To reduce the com-
plexity of the problem, it is first shown that only 4 advisories,
advk ∈{1, 5, 7, 9}, are achievable for a large region of the
state space. Specifically, starting from an initial set where
the previous advisory is COC, advk−1 = 1, only these
advisories will occur for trajectories that remain within the
TABLE VII: VCAS Advisories
adv
Advisory
Compliant ˙h0,k
Choice of ¨hk
1
COC
∅
{−g
8 , 0,
g
8 }
2
DNC
˙h0,k ≤0
{−g
3 , −7g
24 , −g
4 }
3
DND
0 ≤˙h0,k
{ g
4 ,
7g
24 ,
g
3 }
4
DES1500
˙h0,k ≤−1500
{−g
3 , −7g
24 , −g
4 }
5
CL1500
1500 ≤˙h0,k
{ g
4 ,
7g
24 ,
g
3 }
6
SDES1500
˙h0,k ≤−1500
{−g
3 }
7
SCL1500
1500 ≤˙h0,k
{
g
3 }
8
SDES2500
˙h0,k ≤−2500
{−g
3 }
9
SCL2500
2500 ≤˙h0,k
{
g
3 }
constraints hk ∈
−400
−100
and ˙h0,k ∈
−100
100
.
Compliant regions of advisories 4−9 are never achieved within
this domain. It will be shown that under these conditions,
advisories 2 and 3 do not occur and can be neglected.
The functional decomposition in Table VIII relates hk, ˙h0,k,
τk, and advk to the next advisory, under the assumption
that advisories 2 and 3 do not occur. To compactly write
the decomposition, we use −→
w i,j:k to denote the vector of
variables
wi,j
wi,j+1
· · ·
wi,k
T . Using the proposed
methods and the functional decomposition, the graph of the
function advk = f(hk, ˙h0,k, τk, advk−1) can be generated.
Using the identity (14), the set of advisories that can be active,
given the set of states and previous advisory, is calculated. This
is done iteratively in Table IX. Iteration 4 results in the same
potential advisories {1, 5, 7, 9} in the output set as the input
set. The assumption that advisories 2 and 3 do not occur is
confirmed, and the only advisories ever achieved starting from
the input set in iteration 4, and remaining within the bounds for
(hk, ˙hk, τk), is {1, 5, 7, 9}. This knowledge allows the analysis
that follows to neglect 5 of the neural networks from the
closed-loop dynamics, significantly reducing the complexity
of the problem over a large domain.
Now consider the domain (hk, ˙h0,k, τk, advk) ∈D1 ×
D2 × D3 × D4
=
−400
−100
×
−100
100
×
{25, 24, ..., 15} × {1, 5, 7, 9}. Using this domain and the
functional decomposition given by Table X, a closed-loop
state-update set is generated in 2.8 seconds with complex-
ity (ng,ϕ, nb,ϕ, nc,ϕ) = (2339, 842, 2049), which encodes
the transition from (w1, w2, w3, w4) →(w14, w15, w16, w12),
i.e., from (hk, ˙h0,k, τk, advk) to (hk+1, ˙h0,k+1, τk+1, advk+1).
Reachable sets are generated by recursion of (14) and checked
for falsification each step, taking a total of 0.8 seconds to
falsify the VCAS benchmark. Figure 8 plots the resulting
reachable sets.
VI. CONCLUSION
This paper studies the reachability analysis of nonlinear
systems, with focus on methods for efficiently constructing
sets containing nonlinear functions. By leveraging the hy-
brid zonotope set representation and state-update sets, the
proposed methods provide a unified framework for scal-
able calculation of reachable sets and their approximations
spanning broad classes of systems including hybrid, logical,

--- Page 14 ---
TABLE VIII: Functional decomposition of VCAS: States
→Advisory
w1
=
hk
w2
=
˙h0,k
w3
=
τk
w4
=
advk−1
−
→
w i+4,1:9
=
(
fNN,i(w1, w2, w3),
if advk−1 = i
0,
otherwise
,
∀i ∈{1, 2, 3, 4, 5, 6, 7, 8, 9}
−
→
w 14,1:9
=
P13
i=5 −
→
w i
−
→
w 15,1:7
=


max(w14,1, w14,2)
max(w15,1, w14,3)
max(w15,2, w14,4)
max(w15,3, w14,5)
max(w15,4, w14,6)
max(w15,5, w14,7)
max(w15,6, w14,8)


−
→
w 16,1:8
=


w16,1 =
(
0,
if w14,1 ≥w14,2
1,
if w14,1 ≤w14,2
w16,2 =
(
0,
if w15,1 ≥w14,3
1,
if w15,1 ≤w14,3
...
w16,8 =
(
0,
if w15,7 ≥w14,9
1,
if w15,7 ≤w14,9


w17
=































1,
if (w16,1:8) = 0
2,
if (w16,2:8) = 0 ∧w16,1 = 1
3,
if (w16,3:8) = 0 ∧w16,2 = 1
4,
if (w16,4:8) = 0 ∧w16,3 = 1
5,
if (w16,5:8) = 0 ∧w16,4 = 1
6,
if (w16,6:8) = 0 ∧w16,5 = 1
7,
if (w16,7:8) = 0 ∧w16,6 = 1
8,
if (w16,8:8) = 0 ∧w16,7 = 1
9,
if
w16,8 = 1
Fig. 8: Reachable sets for the VCAS benchmark, calculated
and falsified in 0.8 seconds, and plotted in 4.3 seconds.
and nonlinear dynamics. The resulting reachable set memory
complexity grows linearly with time and scales linearly in
computational complexity with the state dimension. Numer-
ical results demonstrate efficient computation and tight over-
approximation of discrete-time reachable sets for a continuous-
time nonlinear system in closed-loop with a neural network
controller, scalability for a high-dimensional logical system,
and exact and efficient reachability of a complicated vertical
TABLE IX: Iterative domain propagation of advisories for an
assumed domain of interest.
Iteration
Input Set
(hk, ˙hk, τk, advk−1)
Reachable Advisories
advk
1
[−400, −100]...
×[−100, 100]...
×{25, 24, ...15}...
×{1}
{1, 5}
2
[−400, −100]...
×[−100, 100]...
×{25, 24, ...15}...
×{1, 5}
{1, 5, 7}
3
[−400, −100]...
×[−100, 100]...
×{25, 24, ...15}...
×{1, 5, 7}
{1, 5, 7, 9}
4
[−400, −100]...
×[−100, 100]...
×{25, 24, ...15}...
×{1, 5, 7, 9}
{1, 5, 7, 9}
TABLE X:
Functional
decomposition
of
VCAS:
States
→Updated States
w1
=
hk
w2
=
˙h0,k
w3
=
τk
w4
=
advk−1
−→
w i+4,1:4
=





[eT
1 , eT
5 , eT
7 , eT
9 ]T fNN,i(w1, w2, w3) ,
if advk−1 = i
0 ,
otherwise
∀i ∈{1, 5, 7, 9}
−→
w 9,1:4
=
P8
i=5 −→
w i
−→
w 10,1:2
=
 max(w9,1, w9,2)
max(w10,1, w9,3)

−→
w 11,1:3
=


w11,1 =
(
0,
if w9,1 ≥w9,2
1,
if w9,1 ≤w9,2
w11,2 =
(
0,
if w10,1 ≥w9,3
1,
if w10,1 ≤w9,3
w11,3 =
(
0,
if w10,2 ≥w9,4
1,
if w10,2 ≤w9,4


w12
=









1,
if (w11,1:3) = 0
5,
if (w11,2:3) = 0 ∧w11,1 = 1
7,
if (w11,3:3) = 0 ∧w11,2 = 1
9,
if
w11,3 = 1
w13
∈









{−g
8, 0, g
8},
if w12 = 1
{ g
4, 7g
24, g
3},
if w12 = 5
{ g
3},
if w12 = 7
{ g
3},
if w12 = 9
w14
=
w1 −w2 −w13
w15
=
w2 + w13
w16
=
w3 −1

--- Page 15 ---
collision avoidance system that combines 9 neural networks
with logic-based rules.
REFERENCES
[1] S. Bansal, M. Chen, S. L. Herbert, and C. J. Tomlin, “Hamilton-
Jacobi reachability: A brief overview and recent advances,” IEEE 56th
Conference on Decision and Control, pp. 2242–2253, 2017.
[2] M. Bui, M. Lu, R. Hojabr, M. Chen, and A. Shriraman, “Real-time
Hamilton-Jacobi reachability analysis of autonomous system with an
fpga,” 2021 International Conference on Intelligent Robots and Systems,
pp. 1666–1673, 2021.
[3] M. Chen, S. L. Herbert, M. S. Vashishtha, S. Bansal, and C. J. Tomlin,
“Decomposition of reachable sets and tubes for a class of nonlinear
systems,” IEEE Transactions on Automatic Control, pp. 3675–3688,
2018.
[4] M. Chen, S. Herbert, and C. J. Tomlin, “Fast reachable set approxi-
mations via state decoupling disturbances,” IEEE 55th Conference on
Decision and Control, pp. 191–196, 2016.
[5] I. M. Mitchell, A. M. Bayen, and C. J. Tomlin, “A time-dependent
hamilton-jacobi formulation of reachable sets for continuous dynamic
games,” IEEE Transactions on automatic control, pp. 947–957, 2005.
[6] D. Angeli and E. D. Sontag, “Monotone control systems,” pp. 1684–
1698, 2003.
[7] N. Ramdani, N. Meslem, and Y. Candau, “Reachability of uncertain non-
linear systems using a nonlinear hybridization,” International Workshop
on Hybrid Systems: Computation and Control, pp. 415–428, 2008.
[8] S. Coogan, “Mixed monotonicity for reachability and safety in dynami-
cal systems,” 59th IEEE Conference on Decision and Control, pp. 5074–
5085, 2020.
[9] M. Althoff, G. Frehse, and A. Girard, “Set propagation techniques
for reachability analysis,” Annual Review of Control, Robotics, and
Autonomous Systems, pp. 369–395, 2021.
[10] M. Althoff, “Reachability analysis and its application to the safety
assessment of autonomous cars,” Institute of Automatic Control Engi-
neering, Technische Universit¨at M¨unchen, Munich, Germany, 2010.
[11] N. Kochdumper and M. Althoff, “Constrained polynomial zonotopes,”
Acta Informatica, vol. 60, pp. 279–316, 2023.
[12] ——, “Sparse polynomial zonotopes: A novel set representation for
reachability analysis,” IEEE Transactions on Automatic Control, pp.
4043–4058, 2021.
[13] E. Asarin, T. Dang, and A. Girard, “Hybridization methods for the
analysis of nonlinear systems,” Acta Informatica, pp. 451–476, 2007.
[14] ——, “Reachability analysis of nonlinear systems using conservative ap-
proximation,” International Workshop on Hybrid Systems: Computation
and Control, pp. 20–35, 2003.
[15] M. Althoff, O. Stursberg, and M. Buss, “Reachability analysis of nonlin-
ear systems with uncertain parameters using conservative linearization,”
IEEE Conference on Decision and Control, pp. 4042–4048, 2008.
[16] R. A. Rajeev, C. Courcoubetis, N. Halbwachs, T. A. Henzinger, P.-H.
Ho, X. Nicollin, A. Olivero, J. Sifakis, and S. Yovine, “The algorithmic
analysis of hybrid systems,” Theoretical Computer Science, pp. 3–34,
1995.
[17] A. Bemporad, “Modeling, control, and reachability analysis of discrete-
time hybrid systems,” University of Sienna, 2003.
[18] J. A. Siefert, T. J. Bird, J. P. Koeln, N. Jain, and H. C. Pangborn, “Robust
successor and precursor sets of hybrid systems using hybrid zonotopes,”
IEEE Control Systems Letters, pp. 355–360, 2023.
[19] T. J. Bird, H. C. Pangborn, N. Jain, and J. P. Koeln, “Hybrid zonotopes:
a new set representation for reachability analysis of mixed logical
dynamical systems,” Automatica, vol. 154, 2023.
[20] T. Bird, N. Jain, H. Pangborn, and J. Koeln, “Set-based reachability and
the explicit solution of linear MPC using hybrid zonotopes,” American
Control Conference 2022, pp. 158–165.
[21] T. Bird, “Hybrid zonotopes: A mixed-integer set representation for the
analysis of hybrid systems,” Purdue University Graduate School, 2022.
[22] T. J. Bird and N. Jain, “Unions and complements of hybrid zonotopes,”
IEEE Control Systems Letters, pp. 1778–1783, 2022.
[23] J. A. Siefert, T. J. Bird, J. P. Koeln, N. Jain, and H. C. Pangborn, “Suc-
cessor sets of discrete-time nonlinear systems using hybrid zonotopes,”
American Control Conference, pp. 1383–1389, 2023.
[24] D. M. Lopez, M. Althoff, L. Benet, X. Chen, J. Fan, M. Forets,
C. Huang, T. T. Johnson, T. Ladner, W. Li, C. Schilling, and Q. Zhu,
“ARCH-COMP22 category report: Artificial intelligence and neural
network control systems (AINNCS) for continuous and hybrid systems
plants,” 9th International Workshop on Applied Verification of Contin-
uous and Hybrid Systems, pp. 142–184, 2022.
[25] D. M. Lopez, M. Althoff, M. Forets, T. T. Johnson, T. Ladner, and
C. Schilling, “ARCH-COMP23 category report: Artificial intelligence
and neural network control systems (AINNCS) for continuous and
hybrid systems plants,” 10th International Workshop on Applied Ver-
ification of Continuous and Hybrid Systems, pp. 89–125, 2023.
[26] Y. Zhang and X. Xu, “Reachability analysis and safety verification
of neural feedback systems via hybrid zonotopes,” American Control
Conference, pp. 1915–1921, 2023.
[27] J. Ortiz, A. Vellucci, J. Koeln, and J. Ruths, “Hybrid zonotopes exactly
represent relu neural networks,” Proceedings of Machine Learning
Research, 2022.
[28] A. Alanwar, F. J. Jiang, and K. H. Johansson, “Polynomial logical
zonotopes: A set representation for reachability analysis of logical
systems,” arXiv: 2306.12508 [v1 of 1], 2023.
[29] S. V. Rakovic, M. Baric, and M. Morari, “Mixed monotonicity for
reachability and safety in dynamical systems,” 47th IEEE Conference
on Decision and Control, pp. 333–338, 2008.
[30] S. Sankaranarayanan and A. Tiwari, “Relational abstractions for contin-
uous and hybrid systems,” Computer Aided Verification, Lecture Notes
in Computer Science, pp. 686–702, 2011.
[31] A. Zutshi, S. Sankaranarayanan, and A. Tiwari, “Relational abstractions
for continuous and hybrid systems,” Computer Aided Verification,
Lecture Notes in Computer Science, pp. 343–361, 2012.
[32] J. A. Siefert, A. F. Thompson, J. J. Glunt, , and H. C. Pangborn,
“Set-valued State Estimation for Nonlinear Systems Using Hybrid
Zonotopes,” IEEE 62nd Conference on Decision and Control, 2023.
[33] E. M. L. Beale and J. A. Tomlin, “Special facilities in a general math-
ematical programming system for non-convex problems using ordered
sets of variables,” Operational Research, pp. 447–454, 1970.
[34] S. Leyffer, A. Sartenaer, and E. Wanufelle, “Branch-and-refine for
mixed-integer nonconvex global optimization,” Preprint ANL/MCS-
P1547-0908, Mathematics and Computer Science Division, Argonne
National Laboratory, pp. 40–78, 2008.
[35] V. K˚urkov´a, “Kolmogorov’s theorem is relevant,” Neural Computation,
vol. 3, no. 4, pp. 617–622, 1991.
[36] T. Gan, M. Chen, Y. Li, B. Xia, and N. Zhan, “Reachability analysis for
solvable dynamical systems,” IEEE Transactions on Automatic Control,
pp. 2003–2018, 2018.
[37] M. Wetzlinger, A. Kulmburg, A. Le Penven, and M. Althoff, “Adaptive
reachability algorithms for nonlinear systems using abstraction error
analysis,” Nonlinear Analysis: Hybrid Systems, 2022.
[38] E. Wanufelle, “A global optimization method for mixed integer nonlin-
ear nonconvex problems related to power systems analysis,” Facult´es
Universitaires Notre-Dame de la Paix, Namur, Belgium, 2007.
[39] J. K. Scott, D. M. Raimondo, G. R. Marseglia, and R. D. Braatz,
“Constrained zonotopes: A new tool for set-based estimation and fault
detection,” Automatica, pp. 126–136, 2016.
[40] H. P. Williams, Model building in mathematical programming.
John
Wiley & Sons, 2013.
[41] A. Sz˝ucs, M. Kvasnica, and M. Fikar, “Optimal piecewise affine
approximations of nonlinear functions obtained from measurements,”
IFAC Proceedings Volumes, vol. 45, no. 9, pp. 160–165, 2012.
[42] MATLAB, “version 9.10.0 (r2021a),” The MathWorks Inc., Natick,
Massachusetts, 2022.
[43] J. A. Siefert, D. D. Leister, J. P. Koeln, and H. C. Pangborn, “Discrete
Reachability Analysis with Bounded Error Sets,” IEEE Control Systems
Letters, 2021.
[44] M. Althoff, “An introduction to cora 2015,” in Applied Verification for
Continuous and Hybrid Systems, vol. 34, 2015, pp. 120–151.
[45] S. Bogomolov, M. Forets, G. Frehse, K. Potomkin, and C. Schilling,
“Juliareach: A toolbox for set-based reachability,” in ACM Conference
on Hybrid Systems: Computation and Control, 2019, p. 39–44.
[46] H.-D. Tran, X. Yang, D. Manzanas Lopez, P. Musau, L. V. Nguyen,
W. Xiang, S. Bak, and T. T. Johnson, “NNV: The neural network
verification tool for deep neural networks and learning-enabled cyber-
physical systems,” in Computer Aided Verification, 2020, pp. 3–17.
[47] C. Huang, J. Fan, X. Chen, W. Li, and Q. Zhu, “Polar: A polynomial
arithmetic framework for verifying neural-network controlled systems,”
in Symposium on Automated Technology for Verification and Analysis,
2022, pp. 414–430.
ACKNOWLEDGEMENTS
The authors thank Matthias Althoff, Tobias Ladner, Taylor
Johnson, Diego Lopez, Christian Schilling, and Xin Chen for
their insight regarding the implementation of CORA, NNV,

--- Page 16 ---
JuliaReach, and POLAR for the ARCH inverted pendulum
benchmark. This work was supported by the Department of
Defense through the National Defense Science & Engineering
Graduate Fellowship Program.
Jacob Siefert is a Ph.D. student and Research
Assistant at The Pennsylvania State University,
where he studies verification of advanced con-
trollers using set-theoretic methods. He received
his B.S. degree in Mechanical Engineering from
the University of Maryland in 2016, and his M.S.
degree in Mechanical Engineering from the Uni-
versity of Minnesota in 2021. His research in-
terests include optimal control, co-design, hybrid
systems, and reachability-based verification.
Trevor J. Bird is a Senior Lead Engineer at P.
C. Krause and Associates, where he develops
tools for the analysis, design, and optimization
of transient systems. He received his B.S. de-
gree in Mechanical Engineering from Utah State
University in 2017, and went on to complete his
M.S. and Ph.D. degrees in Mechanical Engineer-
ing from Purdue University in 2020 and 2022,
respectively. His research interests include op-
timization, discrete mathematics, and set-based
methods.
Andrew F. Thompson is a Ph.D. student and
Research Assistant at The Pennsylvania State
University. He received B.S. degrees in Mechan-
ical Engineering and Computer Science from
The University of Delaware in 2021, and an M.S.
degree in Mechanical Engineering from The
Pennsylvania State University in 2023. His re-
search interests include trajectory optimization,
reachability analysis, and applications to thermal
systems, hybrid electric aircraft, and high-speed
vehicles.
Jonah G. Glunt is a Ph.D. student and Re-
search Assistant at The Pennsylvania State Uni-
versity, where he also received his B.S. degree
in Mechanical Engineering in 2022. He is a
recipient of the National Defense Science & En-
gineering Graduate Fellowship. His research in-
terests include optimization, autonomy, and set-
theoretic methods for verified control and path
planning.
Justin P. Koeln received his B.S. degree in
2011 from Utah State University in Mechani-
cal and Aerospace Engineering. He received
M.S. and Ph.D. degrees in 2013 and 2016,
respectively, from the University of Illinois at
Urbana–Champaign in Mechanical Science and
Engineering. He is an Assistant Professor at the
University of Texas at Dallas in the Mechanical
Engineering Department. He was a NSF Gradu-
ate Research Fellow and a Summer Faculty Fel-
low with the Air Force Research Laboratory. He
was a recipient of the 2022 Office of Naval Research Young Investigator
Award. His research interests include dynamic modeling and control
of thermal management systems, model predictive control, set-based
methods, and hierarchical and distributed control for electro-thermal
systems.
Neera Jain received the S.B. degree in Me-
chanical Engineering from the Massachusetts
Institute of Technology in 2006. She received
M.S. and Ph.D. degrees in 2009 and 2013,
respectively, from the University of Illinois at
Urbana–Champaign in Mechanical Science and
Engineering. She is an Associate Professor of
Mechanical Engineering at Purdue University.
She is a recipient of the National Science Foun-
dation CAREER Award (2022) and served as
a National Research Council Senior Research
Associate at the Air Force Research Laboratory (2022-2023). Her re-
search interests include dynamic modeling, optimal control, and control
co-design for complex energy systems and human-machine teaming.
Herschel C. Pangborn received the B.S. de-
gree in Mechanical Engineering from The Penn-
sylvania
State
University
in
2013
and
the
M.S. and Ph.D. degrees in Mechanical Engi-
neering from the University of Illinois at Ur-
bana–Champaign in 2015 and 2019, respec-
tively. He was an NSF Graduate Research Fel-
low and a Postdoctoral Research Associate at
the University of Illinois, and a Summer Fac-
ulty Fellow with the Air Force Research Lab-
oratory. He is currently an Assistant Professor
with the Department of Mechanical Engineering and the Department
of Aerospace Engineering (by courtesy) at The Pennsylvania State
University. His research interests include model predictive control, op-
timization, and set-based verification of dynamic systems, including
electro-thermal systems in vehicles and buildings.
