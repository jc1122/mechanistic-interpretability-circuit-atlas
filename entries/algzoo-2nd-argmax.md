# Second-argmax in ARC's AlgZoo

In the list `1, 3, 2`, the second-largest value is at position 2 (positions start at 0); a network reading one number at a time must remember enough to make that choice later. Short-list networks often compare inputs, while some longer-list explanations use a recent window of inputs, the largest value so far or stored cutoff values, sometimes reproducing the network's mistakes.

Interpretation is partly solved and the formula status is [partial](../GLOSSARY.md#verifying-a-circuit): the tests cover sampled inputs at fixed list lengths, not exact algorithms for every input or longer lists.

## The circuit at a glance

Figure S1 shows how parallel signed memories of the first value meet the next input at the final comparison neurons; this sketch covers only two-input networks.

<img src="../figures/circuits/algzoo-2nd-argmax.svg" width="640" alt="The two-input recurrent family stores positive and negative parts of the first scalar in parallel. For input 1 their coefficients are 1 and 0, not measured neuron amplitudes. Final comparison neurons approximately compare that memory with the separate next input 3 and choose position 0, the smaller value 1. The comparison is dashed: sampled family evidence is not all-input identity.">

*Figure S1 shows the two-sided binary family storing the first value and approximately comparing the next with final comparison neurons (35/40 two-input recurrent networks read this way, matching the hand-built comparator role); status is Partial: after the 2026-11-14 specificity audit these are READ readings, and only the two width-2 networks among them keep a proven count, as whole-network exceptions with nothing outside the circuit to resample; there is no all-input equivalence or longer-list generalisation, and the values are a task example, not a measurement.*

## At a glance

| Fact | Value |
|---|---|
| Task | Return the zero-based position of the second largest independent Gaussian input |
| Input → output example | Values 1, 3, 2 → position 2 |
| Model | Bias-free, one-layer ReLU recurrent network, widths 2, 3, 4, 5, 6, 8, 16, 32 |
| Rungs and seeds | External TRAIN: five original seeds in each of 56 width/length cells; 280 final nets; nine HAND references; no PART or KD |
| Closed form status | Partial: 16 proven / 257 read / 7 unexplained among 280 networks after the 2026-11-14 specificity audit; the 16 are 4 width-2 whole-network exceptions, 2 resample proofs and 10 flagged optimal-only tables (which fail resampling), so 6 count without the optimal-only exception; sampled domains apply |
| Status | Partly solved (interpretation count, not a task-accuracy seed count); analysis iteration 61 (2026-11-14) with the 2026-11-14 specificity audit applied; later analysis excluded |
| Weights | [Hugging Face: algzoo-2nd-argmax](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/algzoo-2nd-argmax), hand-built archive revision `1a04d29`; ARC nets linked, not re-uploaded |
| Reproduce | Not available yet: analysis code for this task is not yet in `code/`; see [archive](#25-archive) |

This study examined 280 already-trained recurrent networks, which update a small memory after each number, from the Alignment Research Center's (ARC) AlgZoo; no new networks were trained.

By analysis iteration 61 (2026-11-14), 252 networks had passed registered tests that replaced a computation with a formula and changed the activity outside the proposed circuit (15 of them only with chosen constants), and 28 had not. A specificity audit on 2026-11-14 then required each claimed circuit to beat the strongest of 64 random circuits with matched output effects on every one of ten fresh row sets ([what “proven” means](#what-proven-means)). Only 16 of the 252 keep a proven count; the other 236 are now read, giving 16 proven, 257 read and 7 unexplained. The audit did not retest the 28 networks that were already read or unexplained.

HAND denotes formula-set weights and TRAIN denotes training from scratch ([rungs](../GLOSSARY.md#the-three-rungs)). A ReLU keeps only positive input ([glossary](../GLOSSARY.md#standard-machine-learning-terms)). A closed form is an equation-based replacement; resampling the complement replaces everything outside the claimed circuit using other inputs ([verification](../GLOSSARY.md#verifying-a-circuit)). Here “read” means a named interpretation without a passing certificate, including every form the audit demoted; “proven” is the registered-test category defined in [Section 6](#what-proven-means), not an all-input theorem.

## Limitations (read first)

- Most named forms in this entry are READ. The 2026-11-14 specificity audit demoted 236 of the 252 earlier proofs: 108 fail the strongest magnitude-matched random control on at least one of ten row sets, 76 have a final readout-kernel complement that is algebraically null (resampling it cannot change the decision, so it gives no specificity evidence), 51 have too few matched random circuits to build the control, and one table passes the new gate but fails its original .01 margin on 2 of 10 row sets. Of the 16 that remain, 10 are flagged optimal-only tables, 4 are width-2 whole-network exceptions with nothing outside the circuit, and only 2 are resample proofs that pass every gate; 6 count without the optimal-only exception.
- The 15 optimal-ablation-only proofs fail resampling; 10 of them keep the proven count, flagged, and 5 are read. Optimal ablation replaces a discarded component with a chosen constant; it is weaker than keeping accuracy when that component receives independent donor values. Two width-16 proofs sit exactly at the size bar. The T8 width-32 seed-3 table alone agrees only .891/.896 and needs constants.
- Eight proofs use activation-level complements, replacing hidden activity rather than establishing an identity for the recurrence. Three have a codimension-one control, which keeps all but one linear direction, instead of a size-matched control. Thin passes and oversized tables remain read.
- A formula may explain a weak network, including its wrong answers, rather than solve second-argmax. Exact final-kernel complements can preserve the native output algebraically; the joint test then repeats standalone substitution and is not independent mechanism evidence. Checked auxiliary boxes prove native contrast/ideal factoring algebra, not approximate classifier equivalence. The binary ridge additionally has complete native regions and a rational sign lemma.
- Failed tests remain failed: five empty-unit failures, the width-2 T5 seed-0 specificity failure, old floating-projector and quotient failures, the auxiliary expression-size failure, the separate width-4 seed-4 omission, and both shared-veto attempts. Later accepted successors have separately registered tests and fresh rows; they do not repair those failures retrospectively.
- The literal h3 T3 seed-0 final-cut circuit is only a rank2/rank1 reduction. The width-3 T4/T5/T6 seed-4 envelopes explain weak nets with activation-residual complements and rank-2 rather than rank-3 random controls. Modest recurrent-scalar savings are not full low-dimensional algorithms.
- No fixed-length position readout defines longer-length generalisation. This is a dated count: of the 28 networks never proven, 21 are read and 7 unexplained, and none is shown to be impossible to explain. Fresh replication is finite-sample evidence, not ARC's sampling-efficiency or surprise-accounting standard. [Section 6](#6-controls-limitations-and-open-questions) retains the deciding margins and failed-test details.

## 1. Task and data

Given T independent standard-Gaussian scalars (single numbers drawn from a normal distribution with mean 0 and variance 1), the label is the zero-based position of the second largest; T∈{2,3,4,5,6,8,10}. Values 1, 3, 2 have label 2. T denotes input length and h denotes recurrent width; a label such as h3 T3 seed 1 specifies all three. There are 56 width/length cells, each with five original seeds, for 280 networks. Task accuracy is the fraction of correct labels; formula agreement is the fraction matching the native network, the original frozen ARC model, which may itself be wrong.

The interpretation count is frozen at analysis iteration 61 with the 2026-11-14 specificity audit applied; later research is excluded. Table 1 groups the networks by mechanism, and Appendix B retains the earlier iteration-12 accounting, registered additions and failed tests. A later passing test never changes the recorded outcome of an earlier failed one, and a failed audit gate overrides an earlier pass.

### How to read the mechanisms

For a list of two inputs, second-largest means the smaller value. For three, it means the middle value. Short-list networks often reduce this to a comparison or an angle between input differences. Longer-list forms may keep only a recent window, a running maximum or a table of thresholds; a form can reproduce a weak network's wrong answers as well as its correct ones. The mechanism groups are in Section 4; the complete registered-addition and failed-test record is in [Appendix B](#appendix-b-reading-the-detailed-evidence).
## 2. Model

### 2.1 Architecture

The network has no attention, embedding or biases. With zero initial state h_(−1)=0, its recurrence is h_t=relu(I x_t+H h_(t−1)) and its logits (output scores) are O h_(T−1) ([logits](../GLOSSARY.md#standard-machine-learning-terms)). The matrices I, H and O are the input, recurrent and output weights. Width h∈{2,3,4,5,6,8,16,32}; parameter count h²+h+Th, under 1,500.

### 2.2 Frozen and trained parts

All tensors were trained by ARC and frozen for this study. There was no partly trained (PART) or knowledge-distillation (KD) arm.

### 2.3 Training and analysis settings

Credit: [ARC AlgZoo code/models, MIT-0](https://github.com/alignment-research-center/alg-zoo), [ARC's blog](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters). Synthetic iid N(0,1) rows; CPU native fp32, formulas fp64. Every deciding form, row seed, complement, control and bar was registered before its test. Standard bar: substitution and complement each retain ≥.90 of native accuracy; answer agreement ≥.90. Later tests require ≥.01 margins on two row sets. New interval test: 32,768 held-out rows each at seeds 51129/51229, three donor permutations; unit-length rows decide, raw rows reported separately. Publishing replication uses seed 60129; older binary-subspace replays use 61129/61229 plus sign strata. New exact-kernel T3 replications use 68329/68629,71329/71629 and72329/72629/72929; the binary ridge uses raw seed69329. Replication coefficients are frozen; the newer coefficients are weight-derived or frozen from declared diagnostics, while the older behavioural tables are fitted. Tests from iteration 13 onward require all six margins ≥.01 on two 32,768-row main sets and on a separately registered fresh replication, each with three donors. Training precision and optimiser-step grok dates are unavailable. A grok date would locate delayed generalisation during training ([glossary](../GLOSSARY.md#counting-results)); these frozen checkpoints do not supply it.

### 2.4 Reproduction

Reproduce: not available yet: analysis code for this task is not yet in `code/`; the hand-built networks and recorded checks are on [Hugging Face](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/algzoo-2nd-argmax).

### 2.5 Archive

The archive contains our nine hand-built nets at revision `1a04d29`. ARC's 280 networks are listed there by public URL and hash, not re-uploaded. A hash fingerprints the unchanged contents ([glossary](../GLOSSARY.md#standard-machine-learning-terms)).

## 3. Hand-built solution
HAND-a: positive-input delay and leave-one-out maxima, L_i=(x_i)⁺+2 max(0,x_j:j≠i), exact on positive separated inputs. HAND-b: signed binary comparator and two signed T3 projections. HAND-c: T3 medoid L_i=−Σ_j|x_i−x_j|. None is assumed to be the learned mechanism merely because it computes the task.

Table F1 records the fixed reference alternatives alongside the worked circuit above. Scalar inputs take separate reference routes: HAND-a uses delay and leave-one-out maxima, HAND-b uses signed projections, and HAND-c uses absolute distances. Each produces position scores on its stated domain. No part was trained in a PART cell; HAND-a remains restricted to positive separated inputs.

<details>
<summary>Technical reference alternatives and their domains (Table F1)</summary>

*Table F1. Three fixed routes produce position scores (logits) on their stated domains. The rows are alternatives, not sequential stages. There was no PART cell. HAND-a's exactness is restricted to positive separated inputs; a matching task output does not assign a learned mechanism.*

| Fixed reference | Input, stored information and decision |
|---|---|
| HAND-a | Numbers in a list. Delay each input and compare with maxima of the other inputs. Produces position scores on positive separated inputs only. |
| HAND-b | Binary or T3 inputs. Compare signed values using projections to produce position scores on the stated domain. |
| HAND-c | T3, a list of three numbers. Sum absolute distances; the smallest sum picks the position. Produces position scores on the stated domain. |

</details>

## 4. What training finds

The networks are grouped below by the mechanism read at analysis iteration 61; the groups are disjoint and cover all 280. The first eleven rows hold the 252 networks proven by iteration 61, each now carrying its audit outcome; the last row holds the 28 never proven, which the audit did not retest. “Read” is not certified. Interpretation coverage is not task accuracy, and all 40 binary models are accounted for, including the weak and constant predictors. Here h is width, T input length and s the original seed.

*Table 1. A disjoint mechanism grouping of all 280 final networks at analysis iteration 61 with the 2026-11-14 specificity audit applied. Proven includes the ten separately flagged optimal-only tables.*

| group | nets | proven | read | unexplained | mechanism |
|---|---|---|---|---|---|
| Binary comparisons (T2, all widths) | 40 | 2 | 38 | 0 | 35 two-sided comparators, 3 one-sided near-7/8 solutions, 2 constants; h6 s3 one-ridge sign form |
| Angle maps (T3, h≥4) | 25 | 0 | 25 | 0 | one-angle median maps; five also have per-neuron two-entry threshold tables |
| Width-2 arc and register forms (T3) | 4 | 1 | 3 | 0 | arc tables and register forms |
| Clamps, veto, fans and carrier (T3, h3–h5) | 10 | 0 | 10 | 0 | interval-veto, saturated fall/rise, balanced and double clamps, fall/memory, falling fan, compensated carrier |
| Width-2 longer lists (T≥4) | 16 | 1 | 15 | 0 | last-two arc tables, maximum registers, three-input detector/delay forms |
| Width-3 windows and envelopes (T≥4) | 7 | 0 | 7 | 0 | 4 recent-input windows, 3 running-maximum/inhibitory envelopes |
| Width-3 nonfinal ramps (T≥5) | 8 | 0 | 8 | 0 | 3 compensated carriers, 3 helper floors, 2 late compensated-ramp sections |
| Wide threshold tables (h16, h32) | 23 | 12 | 11 | 0 | top-position threshold tables: 15 optimal-only, 5 activation-level, 3 other resample-tested |
| Wide one-scalar fans (h≥4) | 65 | 0 | 65 | 0 | own scalar fan/memory composition with one balanced coupling transferred; literal final kernel |
| Wide nonfinal sections (h≥4) | 51 | 0 | 51 | 0 | 38 compensated-ramp sections at one step, 6 minimum-norm sections, 3 pair-gap, 1 three-threshold gap, 3 two-site sections |
| Wide discrete identity quotients (h5, h6) | 3 | 0 | 3 | 0 | falling, weighted falling and rising orbit forms |
| Never proven by iteration 61 | 28 | 0 | 21 | 7 | 10 width-2 (rounded-recurrence readings and the detector/memory band), 10 width-3, 8 wider; none shown impossible |
| Total | 280 | 16 | 257 | 7 | |

*Table 1a. Audit outcome of the 252 networks proven by iteration 61 (classes defined in [METHODS §9.1](../METHODS.md#91-specificity-audit-of-the-algzoo-entries-2026-11-14)). Counts are networks.*

| group | still proven | strongest-control failure | null readout kernel | insufficient pool |
|---|---|---|---|---|
| Binary comparisons | 2 (width-2 exceptions) | 34 | 4 | 0 |
| Angle maps | 0 | 25 | 0 | 0 |
| Width-2 arc and register forms | 1 (width-2 exception) | 3 | 0 | 0 |
| Clamps, veto, fans and carrier | 0 | 1 | 7 | 2 |
| Width-2 longer lists | 1 (width-2 exception) | 15 | 0 | 0 |
| Width-3 windows and envelopes | 0 | 7 | 0 | 0 |
| Width-3 nonfinal ramps | 0 | 4 | 0 | 4 |
| Wide threshold tables | 12 (2 resample, 10 optimal-only) | 11 (one of them an original-gate failure) | 0 | 0 |
| Wide one-scalar fans | 0 | 0 | 65 | 0 |
| Wide nonfinal sections | 0 | 6 | 0 | 45 |
| Wide discrete identity quotients | 0 | 3 | 0 | 0 |
| Total | 16 | 109 | 76 | 51 |

The strongest-control column includes the width-32 T4 seed-3 table, which passes the new gate on 10/10 row sets but its original .01 margin on only 8/10 (minimum margin .00673828125). Section 5 describes the groups; [Later readings](#later-readings-iterations-1361) covers the families added after iteration 12.

*Table 2. Best logged validation accuracy (%) among five seeds per width/length cell. This is task accuracy, not the certificate count or formula agreement.*
| h | T2 | T3 | T4 | T5 | T6 | T8 | T10 |
|---|---|---|---|---|---|---|---|
| 2 | 99.9919 | 41.0300 | 32.7963 | 30.5511 | 26.9049 | 21.3631 | 20.2094 |
| 3 | 99.9975 | 90.8432 | 43.3364 | 37.5299 | 34.0642 | 28.5172 | 23.8887 |
| 4 | 99.9963 | 98.4534 | 60.1992 | 42.8359 | 39.4123 | 34.5943 | 31.7937 |
| 5 | 99.9978 | 99.9134 | 74.5662 | 54.2300 | 45.0205 | 38.3694 | 34.6842 |
| 6 | 99.9918 | 99.9223 | 86.4660 | 70.3007 | 48.4254 | 40.4798 | 37.5098 |
| 8 | 99.9979 | 99.9454 | 98.7488 | 88.1508 | 78.8639 | 46.5771 | 42.6648 |
| 16 | 99.9974 | 99.9511 | 99.8319 | 99.2499 | 98.4539 | 97.6339 | 94.8445 |
| 32 | 99.9979 | 99.9533 | 99.8795 | 99.7301 | 99.3825 | 99.3058 | 99.4402 |
## 5. Mechanism and verification

The population status is partial. After the specificity audit only 16 networks keep a proven count (2 resample proofs, 10 flagged optimal-only tables and 4 width-2 whole-network exceptions, listed in [Section 6](#what-proven-means)). Every other form below is a READ reading: a named description that reproduced network decisions on registered sampled domains and passed its original tests, but did not pass the audit or was never proven. Algebraically invariant final kernels and auxiliary box checks do not independently prove an exact recurrent algorithm.

Table F2 records the learned families and their limits. Short lists use comparison, angle, clamp or fan descriptions; longer lists use window, envelope, threshold-table, scalar-fan, nonfinal-section or orbit descriptions. Each family describes externally trained networks and ends at their native scores or decisions. The population status remains partial: most descriptions are READ, the 16 proven include optimal-only and whole-network exceptions, and sampled agreement is not a global recurrent identity.

<details>
<summary>Technical learned-family key and failed-test limits (Table F2)</summary>

*Table F2. Learned rules can reproduce the networks' errors. These are alternative families identified in externally ARC-trained networks, not newly trained here. Status: partial; after the 2026-11-14 audit 16 networks are proven (10 flagged optimal-only, 4 width-2 whole-network exceptions, 2 resample) and every other description is READ; sampled agreement and exact final-kernel algebra are not global recurrent identities.*

| Input family | Learned description and output |
|---|---|
| Short Gaussian lists | T2 compares two inputs. T3 uses angle, clamp or fan forms, which are geometric decision rules. Each describes the native position scores or decision. |
| Longer Gaussian lists | Window forms keep recent inputs; envelope forms keep bounds; tables use thresholds; scalar fans and nonfinal sections keep compensated ramps at chosen steps; orbit forms forget which of two neurons holds a value. Each describes the native position scores or decision. |
| Limits of these descriptions | Sampled tests only; most failed the audit's strongest matched control, had an algebraically null complement or lacked a matched pool. Some tests use fixed constants or replace hidden activity. Optimal-only and activation-level exceptions remain flagged; no all-input recurrent identity is established. |

</details>

Table F3 gives the dated coverage census without suggesting that a certificate is a perfect task algorithm.

*Table F3. Longer lists leave more networks unexplained or only read. Analysis iteration 61 with the 2026-11-14 specificity audit applied: 280 fixed-length ARC networks. Counts are networks, not task accuracy. Status: partial; the optimal-only tables fail resampling and the width-2 exceptions have nothing outside the circuit to resample. Binary models include weak and constant predictors. Counts describe sampled, domain-qualified evidence, not exact task algorithms or longer-length generalisation. The superseded iteration-12 census is Table B1.*

| Input-length and width group | Evidence counts |
|---|---|
| Two inputs; all widths (n = 40) | 2 proven (width-2 whole-network exceptions); 38 read; 0 unexplained |
| Three inputs; all widths (n = 40) | 1 proven (width-2 whole-network exception); 39 read; 0 unexplained |
| Longer lists; width 2 (n = 25) | 1 proven (width-2 whole-network exception); 24 read; 0 unexplained |
| Longer lists; width 3 (n = 25) | 0 proven; 21 read; 4 unexplained |
| Longer lists; width ≥ 4 (n = 150) | 12 proven (2 resample, 10 optimal-only); 135 read; 3 unexplained |
| Recorded totals | 16 proven; 257 read; 7 unexplained |

### Binary decisions (T2)

With two inputs the second-largest value is the smaller one, so a correct model chooses position 1 when the second value is below the first. The accounted-for population nevertheless includes two constant predictors and three one-sided weak solutions, alongside the 35 two-sided comparators. An arc table assigns an answer to each angular sector in the two-input plane; it describes what a model predicts, not necessarily the correct comparison. The notation v⁺ means max(v,0), applied to each entry for a vector or matrix, and atan2(y,x) is the angle of the point (x,y).

The exact first state is I⁺x0⁺+(−I)⁺(−x0)⁺, so each final neuron is one ramp in x1. For h6 seed3, all contrast gains are positive: answer1 iff −x1−tau(−x0)⁺>0, tau=439424844068038381/439424838562480128; ties0. The decision is an arc lookup of atan2(x1,x0). All 40 tables agree with their nets on all 65,536 held-out rows. Two-sided rays are within .038° of the comparison boundary; one-sided nets forget positive x0 and are right on approximately 7/8 of the plane.
### Angle maps (T3, 25 accurate nets)

For three inputs, only their differences matter to the correct middle-value choice: shifting every number by the same amount leaves its position unchanged. The following projection removes that shared shift; each model's angular lookup table supplies its own decisions.

Project onto e_a=(1,−1,0)/√2 and e_b=(1,1,−2)/√6; answer by each net's own arc table of atan2(x·e_b,x·e_a). Agreement .9928–.9998. Five with an empty unit complement are instead proven by a per-neuron two-entry threshold table and a nonempty function-space complement, replicated on fresh rows.
### Interval and veto (T3, width 3, seed 1)

A veto suppresses the answer within a silent interval. Here p(u)=u⁺=max(u,0), and dist measures distance outside an interval.

 For (u,v,w), a=.9392046332, c=6.1445968844, k=33.0218429626, let b=|u|, L=a u⁺−c(−u)⁺, U=c u⁺−a(−u)⁺, and dist(z,[l,h])=(l−z)⁺+(z−h)⁺. Set R=dist(v,[L,U]), P=(v−L)⁺, Q=(U−v)⁺, L2=aP−cQ+kb, U2=cP−aQ−kb. Then S=(U2−L2)⁺+dist(w,[min(L2,U2),max(L2,U2)]); substitute (S/2,R,S/2) with the native readout. The second interval is inverted on about .929 of sampled rows: the veto creates a silent band. The pair's summed activity, not its sign, carries the decision. Three scalars replace nine recurrent entries; none is fitted.
### Saturated fall (T3, width 3, seed 0)

A saturated fall is a difference of decreasing ramps that stops changing beyond its thresholds.

 For (u,v,w), P=(−p u⁺+a(−u)⁺−v)⁺, Q=(−q u⁺+c(−u)⁺−v)⁺, R=(v−r u⁺+d(−u)⁺)⁺, K=Q−P; substitute ((aQ+gK−pR−w)⁺,0,(cQ+kK−qR−w)⁺). Eight exact weight-derived scalars: (p,q,r,a,c,d,g,k)≈(.506833,.234368,.804196,.096235,−.233314,3.514283,3.621294,2.646011). Final helper is the native complement; no retained-pair residual is hidden in it.
### Additional clamps (T3, width 3, seeds 3/4)

A clamp limits a value between thresholds. A gauge rescales hidden coordinates without changing the output; M denotes recurrent weights in the stated gauge.

 All scalars below are weight-derived in positive gauge M: input neurons have unit-magnitude input, memory neurons unit principal drive. Seed3: g=−M00, a=−M00−M02, l=(M20+M22)/2, ell=1+M12, j=−M01, s=−M11, k=−M21, epsilon=I1. Put p=u⁺, n=(−u)⁺, A=(−v−an−j epsilon p)⁺, C=(−v+2ln−k epsilon p)⁺, E=(epsilon v+ell n−s epsilon p)⁺, K=C−A; F=((−w−aC+gK−jE)⁺,(epsilon w+ell C−K−sE)⁺,(−w+2lC−lK−kE)⁺). Seed4: a=−M00−M01,b=−M02,c=M10+M11,d=M12,e=M20+M21,f=M22,alpha=−(M00+M11)/2,t=M21; lo=−cp−dn, hi=ap+bn, A=(v−hi)⁺, K=min((v−lo)⁺,hi−lo), C=(ep+fn−v)⁺; F=((w−aA+(alpha−a)K−bC)⁺,(w+cA−alpha K+dC)⁺,(−w+eA+tK+fC)⁺). Keep the native readout. Their exact final decision/kernel tests pass, not the earlier floating-projector tests; neither is an all-input approximate/native equality theorem.
### Wide short networks

A signed capped ramp is a difference of ReLU ramps; unlike an unsigned clamp, it can be negative. A population fan combines a common rising response, a common falling response and bounded differences between individual neurons. The full equations retain those differences.

 A final neuron compares the last input with a threshold indexed by the earlier top positions. With m1≥m2≥m3, p1,p2 their positions, the top-two table is `relu(s(x_last−m3)+g[p1](m1−m2)+g[{p1,p2}](m2−m3)+a)`. Four-to-six inputs need 6/10/15 table numbers per neuron; optimal-only proofs add one fitted constant. Other forms that passed their original tests sort a zero with the prefix (15 numbers at T4), watch one position in three cases (7), or use an additive pair table (14 at T8). These are fitted behavioural tables, not recovered recurrent identities.

Separately, width4 T4 seeds1/2/4 have a weight-derived two-stage clamp: order neurons as rising A/B and falling C/D; a_i=M_iA+M_iB,b_i=M_iB except b_A=0,c_i=M_iC+M_iD,d_i=M_iC. For (u,v,w,z), prefix ramps have thresholds ta=−a_Au⁺−c_A(−u)⁺,tb=−a_Bu⁺−c_B(−u)⁺,tc=a_Cu⁺+c_C(−u)⁺,td=a_Du⁺+c_D(−u)⁺. Put A1=(v−ta)⁺,D1=(td−v)⁺,Kr1=(v−tb)⁺−A1,Kf1=(tc−v)⁺−D1,r_i=a_iA1+b_iKr1+c_iD1+d_iKf1. Then A2=(w+r_A)⁺,D2=(r_D−w)⁺,Kr2=(w+r_B)⁺−A2,Kf2=(r_C−w)⁺−D2,F_i=(s_i z+a_iA2+b_iKr2+c_iD2+d_iKf2)⁺,s=(1,1,−1,−1),with the native readout. Both differences are signed capped ramps. This preserves the first prefix and omits only one later rising-contrast coupling;15vs16 scalars,not a full low-dimensional algorithm.

Width4 T4 seed0 uses the same paired form in native order with b2=0 instead of bA=0, removing M21=.0102879 into M20.

Seed3 uses S_i=Σ_(j∈{0,2,3})M_ij,R_i=M_i1,C0_i=M_i0 except C0_2=0,C3_i=M_i3; at each cut the common fall is neuron2, the rise is1 and the signed falling differences are0−2 and3−2. The only transfer is M20=.268892 into M22, preserving both first-input rays; its full two-stage input formula and 15vs16 scalar reduction pass independently.

Width5 T4 seeds1/2/4 use rising/falling populations with reference pairs(3,2),(0,3),(1,3),respectively: population sums a_i,c_i drive two common amplitudes,while three signed bounded differences carry every other prefix ramp. At each of two cuts the drive is a_i A+c_i D+Σ_j d_ij K_j;final F_i=(s_i z+a_i A2+c_i D2+Σ_j d_ij K2_j)⁺. Only the registered transfers M14->M13,M31->M33,M14->M11,respectively,are omitted from the differential paths;24vs25 scalars. The references fix thresholds from u⁺,(−u)⁺,then the first-cut drives fix the second-cut thresholds;both initial rays are preserved exactly. All three full input forms pass their own tests,not an extrapolation to seeds0/3.
### Maximum and envelope networks

An envelope bounds the alternating partial sums defined below.

 M_t=max(0,x_0,…,x_t), A_t=M_t−M_(t−1), f_t=−x_t+alpha A_(t−1)+beta M_(t−2). The third coordinate is the alternating min/max envelope of partial sums Σ_(j=0..k)(−rho)^j f_(t−j), with an outer maximum with zero. Use (A_t,M_(t−1),E_t) and the native readout. Three weight-derived constants, no fitted recurrence; agreement on native errors ≥.9935. These explain weak nets, not the task.
### Narrow windows and a read-only band

 Earlier proofs use last-two angles or named detector/delay recurrences on at most half the inputs, flagged as truncated rounded recurrences. Width2 T3 seed2 has an explicit detector/memory band: P=(v−a u⁺−b epsilon(−u)⁺)⁺,E=(u⁺−epsilon v−d epsilon(−u)⁺)⁺,L=aP+bE,F=P−dE,U=F/epsilon,mix(t)=(F+tL)/(epsilon+t),T2=min(U,mix(tau_high)),T1=max(L,mix(tau_low)); answer2 belowT2,1 aboveT1,else0. Exact weight-derived values a≈.999364,b≈1.000699,d≈4.013977,epsilon≈.000024396,tau_low≈.093438,tau_high≈1327.289. Two fresh sampled sets agree1, but its complement is empty and near-chance task controls fail necessity/specificity: READ, not proven. The band box verifier is unknown; exact native-region coverage does not certify band equality. Whole-input rounded recurrences and unpassed approximations remain readings.
### Later readings (iterations 13–61)

These families were added between iterations 13 and 61. Each passed its own registered test on two 32,768-row main sets and an independently registered fresh replication, all six margins ≥.01, before the audit; all are READ after it (Table 1a). Every gain is read from the network's own weights; nothing is transferred between seeds, and none of these is a complete recurrent or task algorithm.

The shared building block is an explicit scalar fan. In a positive gauge, let P and N be the pure rising and falling neuron populations, the helpers be the remaining explicit neurons m with input sign s_m, and rp, rn the network's own rising and falling reference neurons. With a_i=Σ_(j∈P)H_ij, c_i=Σ_(j∈N)H_ij and b_im=H_im, the first state is f_i=a_i x0⁺+c_i(−x0)⁺+Σ_m b_im(s_m x0)⁺. Each later input q applies G(q;t)_i=a_i(q+t_rp)⁺+c_i(t_rn−q)⁺+Σ_(j∈J)H_ij K_j(q;t)+Σ_m b_im(s_m q+t_m)⁺, where t holds the thresholds set by the previous step, K_j is neuron j's signed capped ramp against its same-sign reference and J is P∪N without the references. The final neurons are F_i=(s_i x_last+drive_i)⁺ with the native readout.

Wide one-scalar fans (65 networks). Each composes its own fan, fixes its own references and transfers one balanced coupling H_ij into H_ik (or one declared pair sharing), preserving both first-input rays; every other cap, memory, tiny input and feedback remains. This removes one free scalar, not the full state. The complement is the literal final kernel of rank T−1, which the audit found algebraically null for the decision, so these tests carry no specificity evidence.

Wide nonfinal sections (51 networks). At its own step c an explicit fan builds R_i=(s_i x_c+drive_i)⁺. The form keeps h−1 compensated ramps Z_j=R_j−d_j R_p for a pivot neuron p with d_p=1. With y_p=0, y_j=Z_j, L=max over d_i>0 of (−y_i/d_i) and U=min over d_i<0 of (−y_i/d_i), 37 boundary forms use the state C=y+L·d and minimum-norm forms (h8 T6 s1, h8 T10 s3, h5 T8 s2, h6 T8 s3, h6 T8 s4) use C=y+clip(−y·d/(d·d),[L,U])·d; h5 T10 s0 instead minimises the norm of the next drive. The next drive Σ_(j≠p)H_ij Z_j+(Hd)_i λ keeps every gain of Hd, where λ is the scalar multiplying d in C, so only the actual signed rank-1 amplitude along d is resampled. Pair-gap forms (h6 T4 s1, h6 T6 s0, h5 T6 s4) keep every other next-step bias and each pair's weighted gap and omit only the pair's common centre; the three-threshold gap form (h8 T8 s2) keeps two input-normalised group gaps and omits one common centre; the two-site forms (h4 T8 s0, h4 T10 s1, h5 T6 s0) omit one amplitude at each of two steps; h6 T8 s1, the 38th compensated-ramp section, keeps one signed falling cap instead of a boundary state. Some directions d are frozen from fit-only suffix-Jacobian diagnostics, not weight-only null vectors. In the audit 45 of these networks had no matched random pool and 6 failed the strongest control.

Wide discrete identity quotients (3 networks). These forget only which of two neurons holds a value. For h5 T5 s1 at step 2, the form keeps the three outside amplitudes, R1+R4 and |R1−R4|, replaces the pair by its minimum on neuron 1 and maximum on neuron 4, and composes the network's own fan suffix; one identity bit is omitted and resampled, and the continuous dimension stays 5. For h5 T6 s2 at steps 3 and 4, with q_j=Σ_i|H_ij| fixed from the weights, it keeps S=q1R1+q4R4 and A=|q1R1−q4R4| and sets C1=(S−A)/(2q1), C4=(S+A)/(2q4), omitting two identity bits. h6 T6 s2 is the rising-orbit analogue for neurons 1 and 2 at steps 3 and 4. All three fail the strongest matched identity control.

Width-3 nonfinal ramps (8 networks). h3 T6 s0–s2 keep growing and deficit ramps with an exactly compensated carrier; h3 T5 s0–s2 add a minimal nonnegative helper floor; h3 T10 s0 and s1 keep two growing/falling compensated ramps after nine inputs (h3 T10 s1 keeps the falling-2-to-falling-1 feedback 14.174440). They describe weak classifiers' behaviour, including errors, not second-argmax. Four failed the strongest control and four had no matched pool.

Never proven (28 networks). Width 2: h2 T3 s2 (the detector/memory band below), h2 T4 s1/s3/s4, T5 s0/s2, T6 s2, T8 s1 and T10 s2/s3 are read. Width 3: h3 T4 s0–s3, T6 s3 and T10 s2 are read; h3 T5 s3 and T8 s0/s1/s4 are unexplained. Wider: h4 T5 s1/s2/s4, h4 T6 s4 and h5 T6 s1 are read; h4 T5 s0, h4 T6 s3 and h5 T10 s2 are unexplained. Their recorded gaps apply only to the forms tried. For the ten width-2 networks the singleton swap control cannot pass by construction: a complement retention of at least .90B+.01 forces a specificity drop of at most .10B−.01, below the .05 bar for base accuracy B ≤ .5. h4 T6 s4's two-site form fails its fresh replication (temporal agreement .907409668, margin .007409668<.01). None is shown impossible to explain.

### Relation to published work
[ARC's repository](https://github.com/alignment-research-center/alg-zoo) (code/models, MIT-0) supplies our frozen weights. [ARC's blog](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters) (primary blog) already gives the binary reading, approximate four-neuron T3 symmetries and five of sixteen neurons of its separate T10 example; our binary/max roles overlap, not a new discovery. [Leeuwenhoek](https://github.com/Xabilimon1/leeuwenhoek) (code, November 2026) derives rise/fall comparison and memory pairs, and exact paths for argmedian h4 T3: task-equivalent to second argmax at T3, not a reading of all our seeds. [mogar](https://github.com/mogar/algzoo_interp) (code) reads h4 T3 and h32 T10 event/value/clock/position codes; its position-only surrogate reproduces correct answers but none of the net's errors, the same limitation our failed rank chart exposes.
[DraggonZ](https://github.com/DraggonZ/algzoo-interp-public) (code, October 2026) builds hand-written copies of ARC's separate h16 T10 example, with partial job-by-job fidelity; [mrsirrisrm](https://gist.github.com/mrsirrisrm/d6850ff8647d1ed2f67cc92d5bce3ed0) (code gist) uses canonical activation/matrix-product lookups for two top impulses, not arbitrary Gaussian lists. [Snider](https://www.lesswrong.com/posts/sAhFp6gYoyw3xeCaF) (forum, 2026-10-26) fits population-level accuracy estimators, not a deductive closed form of these nets. None supplies our 280-net certificates. [Dutta et al. (2026)](#references), an arXiv preprint, constructs shallow ReLU rank-value approximators; relevant at T3, not evidence for a learned recurrent index circuit.
## 6. Controls, limitations and open questions

### What “proven” means

A network counts as proven only if its form passed the registered tests of its iteration and then the 2026-11-14 specificity gate. The registered tests are substitution, complement resampling and their joint intervention, each keeping ≥.90 of native accuracy with answer agreement ≥.90, with necessity and same-size or same-rank random controls; from iteration 13 all six margins must reach ≥.01 on two 32,768-row main sets and on an independent fresh replication. The specificity gate ([METHODS §9.1](../METHODS.md#91-specificity-audit-of-the-algzoo-entries-2026-11-14)) draws 64 random circuits matched to the claim by output effect, uses ten independent 4,096-row streams and three donors, and requires on every stream that the claim keep at least .01 more than the strongest control, with every original gate still passing on the same rows. Claim and controls are resampled plainly in this gate, also for optimal-only tables. The failure classes are a strongest-control failure, an algebraically null readout-kernel complement and an insufficient matched pool (Table 1a). The networks already read or unexplained were not retested. Proven means finite-sample evidence, not exhaustive over inputs and not unique or minimal.

*Table 1b. The 16 networks that remain proven, and why.*

| Networks | Kind | Why they count |
|---|---|---|
| h2 T2 s1, s2; h2 T3 s1; h2 T4 s2 | Width-2 whole-network exception, flagged | The claimed circuit is the whole network, so nothing is resampled and the control does not apply; inherited status |
| h32 T4 s0 (message-level), s1 (activation-level) | Resample proof | Every original gate and the new gate pass on 10/10 streams; advantage over the strongest control +.01401793 to +.02646795 and +.03250244 to +.04110595 |
| h16 T6 s2; h32 T5 s0–s4; h32 T6 s0, s2–s4 | Optimal-only threshold table, flagged | Original fixed-constant gates and the new gate pass on 10/10 streams; advantage +.01728516 to +.11124704 over 100 streams; fixed-constant accuracy shares .9168094–.9673860, while plain complement resampling keeps only .614–.723 and fails .90 |

Without the optimal-only exception 6 networks count, the 4 width-2 exceptions and the 2 resample proofs; only those 2 are new measured proofs. h32 T4 s3 passes the new gate on 10/10 streams (advantage +.05560984 to +.06749817) but its original .01 margin on only 8/10 (minimum .00673828125), so it is read.

### Earlier controls and failed tests

All items in the [near-top Limitations](#limitations-read-first) apply. Distributed-variable interventions are motivated by [Geiger et al. (2023)](#references); replacing an ablated component by an optimised constant follows [Li and Janson (2024)](#references), which is weaker evidence than resampling in this study. Region-by-region recurrent analysis is contextualised by [Durstewitz (2017)](#references), and explicit affine-region constraints by [Chu et al. (2018)](#references). These are method sources, not proofs of the circuits or their numerical acceptance bars.

For the exact-kernel tests, the kernel is the discarded hidden-state direction that the final readout maps to zero. “FC=native” means the complement intervention algebraically leaves the native output unchanged; “joint=standalone” means adding that intervention to substitution adds no independent evidence.

The new interval form passes substitution, native complement resampling and their joint intervention on both row sets; endpoints reconstruct every native/donor label, same-rank random controls and necessity drop >.05, and formula/native common-bundle swaps agree ≥.9960. Its complement contains .484 of hidden energy, so this is not an empty-complement exception. No all-input exact learned identity or independently causal scalar variables are claimed.
The 15 optimal-only proofs fail resampling and are flagged; two width-16 proofs use exactly half the width in numbers per neuron. The T8 width-32 seed-3 table alone agrees only .891/.896 and passes with constants. Eight activation-level proofs retain that flag; three also have a codimension-one, not size-matched, control. Thin passes and oversized tables stay read. No fixed-T position readout defines longer-length generalisation. Fresh-row table and later-form replications all pass, including the weakest optimal-only table (margin +.01138); unit replications are complete, with one additional specificity demotion; all three proper binary-subspace replications pass. Wide late-step top-two charts fail despite near-full native interchange fidelity; no new long-wide proof from those charts. The saturated-fall proof replicates on a third fresh row set (agreement .9932, joint .9684). Two earlier quotient tests failed: the rise form agreed .9987–.9989 alone but .8420–.8446 under joint resampling; the signed-fall quotient also failed. Those failed tests remain failed. New, separately registered exact-kernel tests pass for both nets, retaining the same rise form but a lower-row-only tied fall form with the helper input preserved. At iteration 61, 21 never-proven networks were read and 7 unexplained; with the audit applied the census is 16 proven, 257 read and 7 unexplained.
### Detailed contrast forms and failed alternatives

 Let R(v;a,b)=(v−a)⁺−(v−b)⁺ and D(v;a,b)=(a−v)⁺−(b−v)⁺, each a signed capped ramp. Width4 seed2 has common rise A, common fall B and contrasts Kr,Kf; F_i=(s_i w+S_i A+U_i Kr+N_i B+V_i Kf)⁺, with S=M[:,0]+M[:,2],U=M[:,0],N=M[:,1]+M[:,3],V=M[:,3] except V1=0. Width4 seed3 has common fall A, its contrast K, rise B and helper E: F_i=(s_i w+S_i A+C_i K+M_i1 B+M_i3 E)⁺, S=M[:,0]+M[:,2],C=M[:,2] except C0=0. Width5 seed0 has common fall A, rise B and K0,K2,K3: F_i=(s_i w+S_i A+M_i1 B+M_i0 K0+M_i2 K2+C_i K3)⁺, S=Σ_(j∈{0,2,3,4})M[:,j],C=M[:,3] except C1=0. Prefix thresholds are direct weight contractions of u⁺,(−u)⁺; no fitted values or recurrent rollout. These modest 15/16,14/15,24/25 scalar reductions pass both main sets and fresh replications; full bounded contrasts and tiny helper inputs survive elsewhere. Width3 seed2/width4 seed1 retain their earlier balanced clamps. One separate width4 seed4 omission fails and stays failed. A shared-veto form fails standalone/joint agreement(.885–.889); a separately registered prefix-preserving compensation still fails(.8884–.8915), despite passing accuracy-share and every other gate. Both native final veto gains must remain distinct. The T4 clamp/fan proofs use exact final rank3/rank1 kernels at width4 and rank3/rank2 at width5, with genuinely same-rank controls; all current proofs have fresh replications. Exact-kernel limits: FC=native and joint=standalone algebraically, not independent mechanism evidence. Nonempty kernels, exact endpoints, native precision, necessity and truly same-rank controls pass. Checked boxes certify native contrast algebra/ideal factoring, NOT approximate classifier equivalence; the binary additionally has complete native regions and a rational sign lemma.
## 7. Prediction record

Written before opening the zoo weights, not before ARC's training: universal binary comparison, symmetric projections at T3, maximum/leave-one-out hybrids at longer T. Comparison is right for 35/40 binary nets, not universal; good T3 nets depend nearly only on input differences; individual maximum registers recur at longer T but do not explain the whole population. No retrospective prediction edits.

## Appendix A. Weights excerpt and reading

*Table 3. Weight-derived examples. These slices support the stated reading but do not replace the substitution and complement tests.*

| Net | Weight-derived structure | Interpretation |
|---|---|---|
| h2 T2 seed 1 | input (4.7229,−6.2045); effective threshold .9995; rays −134.978°,44.986° | opposite ramps balance at x1=x0; neither neuron alone has that kink |
| h3 T3 seed 1 | pair self −.938/−.940, cross 6.144/6.145, memory inhibition −33.029/−33.015; memory self −5.207 | near-mirror pair, distance memory and common veto; the near-equal readout columns make the pair contrast dispensable |

## Appendix B. Reading the detailed evidence

Tables 1 and 2 distinguish mechanism-certificate counts from task accuracy. In the detailed tests, substitution is formula-only evaluation, a joint intervention applies substitution and complement resampling together, and specificity asks whether a size- or rank-matched random circuit also works. Rank is the number of independent linear directions ([analysis measures](../GLOSSARY.md#analysis-measures)); final rank2/rank1 and rank3/rank2 labels describe kept/discarded directions, not a minimal recurrent state. The accounting below is the superseded iteration-12 record (2026-11-13), kept for its margins and failed tests; Table 1 replaces its counts, and its proofs are now read unless listed in Table 1b.

### Iteration-12 accounting (superseded)

*Table B1. Superseded iteration-12 census. Longer lists leave more networks unexplained. Iteration-12 snapshot, 2026-11-13: 280 fixed-length ARC networks. Counts are networks, not task accuracy. Status: partial; the 15 optimal-only certificates fail resampling, and resample-tested forms retain their activation-level and control flags. Binary models include weak and constant predictors. Counts describe sampled, domain-qualified evidence, not exact task algorithms or longer-length generalisation.*

| Input-length and width group | Evidence counts |
|---|---|
| Two inputs; all widths (n = 40) | 40 resample-tested; 0 optimal-only; 0 reading only; 0 unexplained |
| Three inputs; all widths (n = 40) | 38 resample-tested; 0 optimal-only; 2 reading only; 0 unexplained |
| Longer lists; width 2 (n = 25) | 16 resample-tested; 0 optimal-only; 9 reading only; 0 unexplained |
| Longer lists; width 3 (n = 25) | 7 resample-tested; 0 optimal-only; 10 reading only; 8 unexplained |
| Longer lists; width ≥ 4 (n = 150) | 16 resample-tested; 15 optimal-only; 12 reading only; 107 unexplained |
| Recorded totals | 117 resample-tested; 15 optimal-only; 33 reading only; 115 unexplained |

At iteration 12 the wide proofs were 16 resample proofs (5 activation-level, 8 modest two-stage clamps and fans) and 15 optimal-only threshold tables; the width-3 full-state register form stayed read, and width-2 T5 seed 0 had failed fresh specificity.

The recorded partition is 132 proven / 33 read / 115 unexplained. Proven = 117 resample proofs and 15 optimal-ablation-only proofs. Resample-only accounting: 117 proven / 43 read / 120 unexplained. The publishing audit demoted h3 T10 seed2 and h6 T2 seed3 for empty old complements; the latter is now restored only by its new one-ridge/native-kernel proof. Fresh-row replication passes all 28 frozen tables and every form then counted. The unit batch's serialisation-only replay finishes 80/86: five empty-unit failures and one new specificity failure. Width 2, T5, seed 0 leaves proven for read (random-control accuracy drop .0344<.05). All three proper older binary-subspace certificates also replicate on two fresh row sets and sign strata; failed unit tests remain failed.

### Registered additions and retained failures

Width5 T4 seeds1/2/4 pass two-stage signed population fans, agreement .9259–.9802 across both main sets and independent fresh replications; twenty-four recurrent scalars replace twenty-five, with one fixed balanced differential coupling removed. The remaining width4 T4 seeds0/3 pass distinct two-stage paired-clamp/falling-fan forms, agreement .943–.998 on main sets and fresh replications; all five width4 T4 seeds are now proven under their own tests, not one identical mechanism. Each new form removes one balanced differential coupling, fifteen versus sixteen scalars.

Earlier width4 T4 seeds1/2/4 pass a two-stage signed-clamp form, agreement .9462–.9671 on both main sets and fresh replications; fifteen recurrent scalars replace sixteen. Their original tests failed the auxiliary verifier expression-size gate and remain failed. A new registered complete test removes only redundant zero-state syntax, preserves every cap/form/control and proves the required auxiliary algebra.

Width4 T3 seeds2/3 and width5 T3 seed0 pass explicit double-clamp, fall/memory and falling-fan forms, with main-set agreement .9870–.9946 and independently registered fresh replications. Width3 T3 seed2 and width4 T3 seed1 previously passed balanced rising/falling clamps, with main-set agreement .9991–.9996 and fresh replications. Exact-prefix-preserving transfers remove one final differential coupling each: modest seven-versus-eight and fourteen-versus-fifteen scalar reductions, not full recurrent algorithms.

Earlier width3, T3, seeds3/4 pass saturated fall/rise forms in exact native decision coordinates (agreement .9973–.9990), with independent fresh replications. Their final rank2/rank1 kernel complement is algebraically invariant, not independent algorithm evidence. Width6, T2, seed3 passes a one-threshold one-sided ridge, with checked region coverage, rational all-real sign proof and fresh controls. All old floating-projector failures remain failed.

Earlier width3, T3, seed0 uses a saturated fall contrast and two falling thresholds. Agreement .9922/.9923, joint resampling .9665/.9668; checked auxiliary pair-extraction algebra on the unit box. Flag: modest eight-versus-nine recurrent-scalar reduction, literal final-cut rank2/rank1 circuit only; no all-input classifier proof.

Width 3, T3, seed 1 is a two-interval distance comparison with an inhibitory veto, using three weight-derived recurrent scalars. Standalone agreement .9961–.9962, wrong-row agreement .9798–.9828; joint substitution/resampling agreement .9204–.9214 on two fresh row sets. The discarded pair contrast is a nonempty rank-1 complement; truly same-rank-2 random controls drop to .367–.441 accuracy. The earlier failed formula with a missing feedback term stays failed; this is a separately tested successor.

Also collected: width 16, T5, seed 2 passes its earlier registered one-position-table optimal-ablation test (agreement .9153/.9195), exactly at the size bar; resampling fails (.634 exact), and the table alone agrees .894 on one set. Width 16, T5, seed 1 becomes read, not proven: its additive table exceeds the size bar.

Earlier proofs: width-3 T4/T5/T6 seed 4, running maximum plus inhibitory envelope, agreement .9946–.9983. Flags: weak nets, activation-residual complement and rank-2 rather than rank-3 random control. Five earlier wide T4 proofs also use activation-level complements. The 15 optimal-only proofs remain separate; none is called a resample proof.

## What to read next

Compare [AlgZoo median](algzoo-median.md), where the output is a value rather than a position and formulas can also explain inaccurate networks. For the difference between explaining a model and solving its task, read [the verification guide](../GLOSSARY.md#verifying-a-circuit).

## References

The earlier analysis read the public pages and READMEs below and checked the scholarly DOI metadata with scite. This is source
credit, not a claim that those sources certify these networks or that our tests establish ARC's sampling-efficiency or
surprise-accounting standard.

- Alignment Research Center (n.d.). AlgZoo: Uninterpreted models with fewer than 1,500 parameters. *ARC blog*.
  https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters (published binary and partial T3/T10 readings)
- Alignment Research Center (n.d.). AlgZoo. *GitHub software and model repository; MIT-0*.
  https://github.com/alignment-research-center/alg-zoo (the frozen networks analysed here)
- Chu, L., Hu, X., Hu, J., Wang, L., & Pei, J. (2018). Exact and consistent interpretation for piecewise linear neural networks.
  *ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (conference paper).
  https://doi.org/10.1145/3219819.3220063 (region-wise interpretation, not these recurrent-network certificates)
- DraggonZ (2026, October). algzoo-interp-public. *GitHub software repository*.
  https://github.com/DraggonZ/algzoo-interp-public (hand-written copies of a separate ARC example, with partial fidelity)
- Durstewitz, D. (2017). A state space approach for piecewise-linear recurrent neural networks for identifying computational
  dynamics from neural measurements. *PLOS Computational Biology* (journal article).
  https://doi.org/10.1371/journal.pcbi.1005542 (piecewise-linear recurrent dynamics)
- Dutta, A., Safran, I., & Valiant, P. (2026). The median is easier than it looks: Approximation with a constant-depth, linear-width
  ReLU network. *arXiv preprint*. https://doi.org/10.48550/arxiv.2602.07219 (rank-value approximation, not a learned recurrent index circuit)
- Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. D. (2023). Finding alignments between interpretable causal variables and
  distributed neural representations. *arXiv preprint*. https://doi.org/10.48550/arxiv.2303.02536 (intervention-based causal descriptions)
- Li, M., & Janson, L. (2024). Optimal ablation for interpretability. *arXiv preprint*.
  https://doi.org/10.48550/arxiv.2409.09951 (input-independent ablation constants, distinct from resampling)
- mogar (n.d.). algzoo_interp. *GitHub software repository*. https://github.com/mogar/algzoo_interp
  (event/value/clock/position readings and a task-only surrogate for a separate example)
- mrsirrisrm (n.d.). AlgZoo interpretation code. *GitHub code gist*.
  https://gist.github.com/mrsirrisrm/d6850ff8647d1ed2f67cc92d5bce3ed0 (activation and matrix-product lookups for two top impulses)
- Snider (2026, October 26). [Forum post on AlgZoo population accuracy]. *LessWrong forum post*.
  https://www.lesswrong.com/posts/sAhFp6gYoyw3xeCaF (population accuracy estimators, not deductive forms of these networks)
- Xabilimon1 (2026, November). Leeuwenhoek. *GitHub software repository*. https://github.com/Xabilimon1/leeuwenhoek
  (rise/fall and memory-pair readings; separate h4 T3 argmedian paths)
