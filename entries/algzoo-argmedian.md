# Argmedian in ARC's AlgZoo recurrent networks

In the list `0.3, −1.2, 2.0` the median is 0.3, and it sits at position 0 (positions start at 0). A network that reads one number at a time must keep enough about the earlier numbers to name that position at the end. The best three-input networks keep the first two numbers as an ordered pair and place the third against them; longer lists are handled by large banks of comparisons and fading memories, many of them weak heuristics rather than median algorithms.

Interpretation is partly solved and the formula status is [partial](../GLOSSARY.md#verifying-a-circuit): after the 2026-11-14 specificity audit 22 of 100 networks keep a proven count, 20 of them only as whole-network exceptions, and no all-input equivalence is proved.

## The circuit at a glance

Figure S1 shows the ordered first pair meeting the next input at the final neurons, which cut the next input's place relative to the pair near 0 and 1; this sketch covers only wide three-input networks.

<img src="../figures/circuits/algzoo-argmedian.svg" width="640" alt="The wide three-input recurrent networks keep the first two values as an ordered pair: which one came first, and the lower bound, upper bound and gap. For 0.3 then −1.2 the lower bound −1.2 came second, the upper 0.3 came first, and the gap is 1.5. Final neurons place the next input 2.0 relative to the pair, (2.0 minus −1.2) divided by 1.5, about 2.13, and cut that ratio near 0 and 1: above 1 the upper value is the median, so the output is position 0. Learned roles are dashed: two networks are proven after the specificity audit, the others are read.">

*Figure S1 shows the length-3 prefix table: the first pair's order, its lower and upper values and their gap, and final neurons that cut τ=(next−lower)/gap near 0 and 1 (in all 15 wide three-input networks the recovered table is the median table on both pair orders, worst cut .035 from 0 or 1); status is Partial: after the 2026-11-14 specificity audit only width 16 seeds 1 and 4 are proven, the other wide three-input forms are READ, lower/upper/gap are variables of the formula rather than decoded neurons, and the values are a task example, not a measurement.*

## At a glance

| Fact | Value |
|---|---|
| Task | Return the position of the median of an odd-length iid Gaussian scalar list; ties excluded |
| Input → output example | Values 0.3, −1.2, 2.0 → position 0 (the median 0.3) |
| Model | Bias-free one-layer ReLU recurrent network (DistRNN); widths 2, 4, 8, 16, 32; lengths 3, 5, 7, 11 |
| Rungs and seeds | External TRAIN: 20 width/length cells × five original seeds = 100 terminal nets; HAND references A, A+ and B; no PART or KD |
| Closed form status | Partial: 22 proven / 78 read / 0 unexplained after the 2026-11-14 specificity audit; 20 are width-2 whole-network exceptions with an empty complement, 2 are resample proofs (width 16, length 3, seeds 1 and 4). Before the audit 99 / 1 / 0. No global native equivalence. |
| Status | Partly solved (22/100 networks proven, 78 read, 0 unexplained); analysis iteration 24 (2026-11-14) with the 2026-11-14 specificity audit applied |
| Weights | [Hugging Face: algzoo-argmedian](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/algzoo-argmedian), hand-built weights/evidence archive revision `22977ef`; ARC nets linked by URL and hashes, not redistributed |
| Reproduce | Not available yet: analysis code for this task is not yet in `code/`; see [archive](#25-archive) |

This study examined 100 recurrent networks already trained by the Alignment Research Center (ARC), which update a small memory after each number; no new networks were trained.

By analysis iteration 24 (2026-11-14), 99 networks had passed registered tests that replaced a computation with a formula and resampled the activity outside the proposed circuit, and one had not. A specificity audit on 2026-11-14 then required each claimed circuit to beat the strongest of 64 random circuits with matched output effects on every one of ten fresh row sets ([what “proven” means](#what-proven-means)). Two of the 79 testable proofs passed; the 20 width-2 forms keep their status because nothing lies outside them. The count is therefore 22 proven, 78 read and 0 unexplained, and all 40 paired-register charts are READ readings with their named variable.

Here HAND means weights set by formula and TRAIN means weights trained from scratch ([rungs](../GLOSSARY.md#the-three-rungs)). A ReLU keeps the positive part of a number ([glossary](../GLOSSARY.md#standard-machine-learning-terms)). A closed form replaces a network computation with equations; a complement test resamples everything outside the claimed circuit ([verification](../GLOSSARY.md#verifying-a-circuit)). “Read” means a named reading without a passing certificate, including every form the audit demoted; “proven” is the registered sampled-certificate category defined in [Section 6](#what-proven-means), not an all-input theorem.

## Limitations (read first)

- Most named forms in this entry are READ. 77 of the 79 earlier proofs did not beat the strongest matched control by .01 on any of the ten streams, and for 60 of them all 64 matched controls coincided with the claim, so those claims could not be told apart from size-matched controls at all.
- The 20 width-2 proofs keep their count only as whole-network exceptions: the claimed circuit is the whole network, so the complement test is vacuous and says nothing about which parts could be discarded. Their forms explain weak networks, not the median task.
- Only two networks, both width 16 and length 3, are measured proofs. Width 16 seed 4's worst individual donor agreement is .897217; the original gate uses the three-donor mean.
- Most explanations retain nearly the whole network state: the paired-register charts keep 3 of 4, 7 of 8, 15 of 16 or 31 of 32 coordinates, and the 19 full-state banks keep every register. Many of these are weak heuristics, not compact median algorithms or dimension reductions.
- Tests use finite fresh Gaussian rows; they are not exhaustive over inputs, unique or minimal. Fixed-length position readouts cannot emit new indices, so length tests are stress tests, not generalisation.
- Width 4, length 11, seed 3 was never proven: every registered form fails its necessity or specificity drop. Every failed test listed in [Appendix B](#appendix-b-detailed-evidence) stays failed.

## 1. Task and data

Inputs are iid standard-Gaussian sequences of length T=3,5,7,11; the label is the zero-based position of the median; ties are excluded. All preservation tests use synthetic independent inputs, not ARC's validation set. ARC saved terminal nets after 2^30 sequences and 1,100 earlier snapshots; only the terminal nets were interpreted. Finite tests are sampled, not exhaustive over Gaussian support. Labels such as h16 T3 s1 give width h, length T and original seed s.

## 2. Model

### 2.1 Architecture

Bias-free one-layer ReLU DistRNN: h=2,4,8,16,32; T=3,5,7,11; h₀=0; h′=relu(ux+Wh), logits=Vh_T. No attention, embeddings, vocabulary, frozen HAND blocks or extra computational steps.

### 2.2 Frozen and trained parts

All ARC tensors were frozen for this analysis. There was no partly trained (PART) arm or knowledge-distillation (KD) arm.

### 2.3 Training and analysis settings

[Shared methods](https://github.com/jc1122/mechanistic-interpretability-circuit-atlas/blob/main/METHODS.md). Credit: ARC's [AlgZoo code/networks, MIT-0](https://github.com/alignment-research-center/alg-zoo) and [introducing blog](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters). Five original seeds in 20 cells; terminal 2^30 sequences. Original training arithmetic is unspecified; zoo best-snapshot accuracies and grok steps are unavailable. Interventions run on CPU with frozen weights.

### 2.4 Reproduction

Reproduce: not available yet: analysis code for this task is not yet in `code/`. The external networks come from [ARC's code/model repository](https://github.com/alignment-research-center/alg-zoo).

### 2.5 Archive

Six hand-built nets, `config.json`, the native loader and a manifest with sha256 hashes passed a local lossless round-trip on 2026-11-13 (106/106 reloads byte-identical with identical accuracy on 8,192 fresh rows); the 100 ARC nets are referenced by public URL and hash, not redistributed. They are on [Hugging Face](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/algzoo-argmedian) (revision `22977ef`). An independent anonymous download re-checked every file hash and the round-trip record and reloaded all six nets; every reproduced accuracy lies inside the net's fp32 near-tie band and every fp64 accuracy equals the manifest. Raw outputs were not bit-identical to the recording host, so the check is on decisions, not output bits.

Near ties. On 8,192 Gaussian rows the positive-domain A+ net (n=11, width 32) never ranks a wrong position strictly above the median position. On 4,160 rows (51%) the median position ties exactly with other positions, because the circuit ignores negative inputs, so the tie-break decides the answer, not the circuit. 48.9% of rows are decided strictly correctly. With exact (fp64) arithmetic and a first-index tie-break the accuracy is 56.47%; fp32 runs give 56.46–56.63% depending on the CPU's summation order. The manifest records the fp64 accuracy, the tie-break and the band [48.9%, 100%] that a rigorous fp32 rounding bound allows (the bound holds for any summation order). The other five nets have fp64 accuracy 100%. The rounding bound certifies the n=3, width-16 signed net on any host. The n=7 signed net has 20 rows inside the bound, all correct in every order tried. For the three B nets the bound is too loose to certify (recurrent gain about 32), and no decision changed across three summation orders. Exact verification has to model floating-point details ([Jia & Rinard, 2021](https://doi.org/10.1007/978-3-030-88806-0_9), peer-reviewed conference paper).

## 3. Hand-built solution

A streams signed delay taps, comparison hinges and accumulators: L_i=−Σ_j|x_i−x_j|−Σ_jx_j, selecting the L1 medoid; A+ loses negative taps. B uses z₁=x₀−(λ+1)x₁+λx₂, z₂=−x₀−λx₁+(λ+1)x₂, λ=31.5, with logits (|z₁|+|z₂|,−λ|z₁|+(λ+1)|z₂|,(λ+1)|z₁|−λ|z₂|).

Independent harness: 40,960,720 bounded/gapped rows, zero errors; B and A+ are not exact on unrestricted Gaussian support. There was no PART cell, so no hand-built part is marked as trained.

## 4. What training finds

Zero heads throughout. Neurons are axis-used coordinates across time. Terminal accuracy rises with width and falls with length (Table 2 in Appendix B): width-8 length-3 nets reach .999258–.999454, while width-2 length-11 nets reach .104–.106. Thirty-eight recurrent reductions and twenty-five finite comparison/filter forms were found in earlier rounds (2026-11-13), and by analysis iteration 24 all but one net had a registered form. Higher variable decoding does not establish causal use.

*Table 1. A disjoint grouping of all 100 terminal networks at analysis iteration 24 with the 2026-11-14 specificity audit applied. Counts are networks, not task accuracy. “0/10” means the claim beat the strongest matched control by .01 on none of the ten streams.*

| Group | Cells (h, T, seeds) | Nets | Proven | Read | Unexplained | Audit outcome |
|---|---|---|---|---|---|---|
| Width-2 whole-network forms | h2, every T, s0–4 | 20 | 20 | 0 | 0 | whole-network exception; nothing to resample |
| Paired-register charts (q omitted) | h4 T3 s0, s2–4, T5 s0, s2, T7 s0, s1, s4, T11 s0–2, s4; h8 T3 s0, T11 s0, s3; h16 T5 s0–4, T7 s0–2, T11 s3; h32 T5, T7, T11 s0–4 | 40 | 0 | 40 | 0 | 0/10 for all 40; 33 pools all claim-identical |
| Full-state comparison/filter banks | h8 T5, T7 s0–4, T11 s1, s2, s4; h16 T7 s3, s4, T11 s0–2, s4 | 19 | 0 | 19 | 0 | 0/10; all 64 controls identical to the claim |
| Length-3 prefix tables | h8 T3 s1–4; h16 T3 s0, s1, s4 | 7 | 2 | 5 | 0 | h16 s1, s4 pass 10/10; the other five 0/10 |
| Length-3 final-unit complements | h16 T3 s2, s3; h32 T3 s0–4 | 7 | 0 | 7 | 0 | 0/10; 2 pools all claim-identical |
| Finite width-4 function-residual forms | h4 T3 s1, T5 s1, s3, s4, T7 s2, s3 | 6 | 0 | 6 | 0 | 0/10; all 64 controls identical to the claim |
| Four-register reading, never proven | h4 T11 s3 | 1 | 0 | 1 | 0 | not retested (read before the audit) |
| Total | | 100 | 22 | 78 | 0 | |

The 60 claim-identical pools are counted in the [specificity audit](#what-proven-means); identical controls were run and retained, not discarded. Width 16 T3 seed 0 beats its strongest control on every stream, but only by .0051–.0095, under the .01 bar.

## 5. Mechanism and verification

The sections below describe the earlier registered tests (2026-11-13, iterations up to 24). There, “proof”, “pass” and “close” refer to those tests; after the audit every form below is a READ reading except the 20 width-2 forms and the two h16 T3 tables. Before the audit: 99 proven / 1 read / 0 unexplained; the resample column had 97, including 20 existing two-neuron whole-net exceptions with empty complements; 77 had nonempty resampled complements; two further proofs, h32 T3 seeds 0 and 4, were flagged optimal-ablation-only (fresh optimal joint shares .926–.928/.923–.924; resample .902/.895–.896). All 99 counted forms passed fresh publication replications.

### 5.1 Length-3 prefix tables (two proven)

At length 3 the final preactivations are s_i x₂+a_i x₀+b_i x₁+e_i|x₀−x₁|. Population tents, ramps and flags yield a table over input order and τ=(x₂−lo)/|x₁−x₀|, where lo and hi are the smaller and larger of the first two inputs. The OS2 form writes each final preactivation as gap·(s_i τ+κ_i[sheet]), a table over the pair's order (the sheet) and τ; CF3c is the same decision table with every preactivation made shift invariant. All 15 wide tables are the median table on both sheets, with the worst cut .035 from 0 or 1: below 0 the lower value is the median, between 0 and 1 the new input, above 1 the upper value.

Seven earlier wide length-3 proofs passed the full residual complement, five more a spare-neuron complement; h32 seeds 0 and 4 passed only the flagged constant complement. Of these, only the h16 seeds 1 and 4 tables survive the audit: h16 T3 s1 is the OS2 median-of-three table on the two middle prefix order statistics, and h16 T3 s4 the CF3c decision table over input order and τ (Table 1b).

### 5.2 Width-2 whole-network forms (20, proven as exceptions)

Two-neuron forms: nine signed dead-zone traces, four two-tap excursions, two accumulator thresholds, three one-comparison heuristics and two running-max/veto solutions. Whole-net recurrence plus ratio readout agrees ≥.99899 on 200,000 rows; the nine one-scalar dead-zone reductions agree exactly on 70,000 fresh rows. These explain weak nets, not the median task.

A separate readout theorem: h2 T3 seeds 0, 3 and 4 never emit position 2, whose readout is convexly dominated on h≥0; their complete recurrence certificates, not that readout theorem alone, supply their counted forms.

### 5.3 Paired-register charts (40, read)

Every one of the 40 paired-register charts stays READ with its named variable z_j=h_p/g_p−h_n/g_n, the signed difference of a gain-normalised neuron pair, omitting q_j=min(h_p/g_p,h_n/g_n). In positive gauge n=h/g, M=diag(g)⁻¹Wdiag(g), the chart retains z_k=n_p−n_n and unpaired r, and discards q_k=min(n_p,n_n). Let A_ik=(M_ip−M_in)/2, B_ik=(M_ip+M_in)/2. Set a=sx+Az+B|z|+M_rest r; z′_k=[a_p]₊−[a_n]₊, r′=[a_rest]₊; reconstruct n_p=[z]₊, n_n=[−z]₊ and use Vdiag(g). The complement consists of q at every step, not a missing final label or a nearly complete edge mask; small q energy was not treated as algebraic nullity. Balanced pairs use g_pair=g0_pair v, with v the positive small right-singular vector of the original normalized recurrent/readout pair columns; no gauge is fitted to deciding rows.

Width 4. Existing N0 h4 T3 s0 uses pairs (0,3),(1,2); N2 s2 uses (0,1),(2,3). New h4 pairs: T7 s0:(1,0), s1:(0,2), s4:(1,2); T11 s0:(1,2), s1:(0,2), s2:(3,1), s4:(0,2); T3 s3:(1,2), s4:(0,2); T5 s0:(0,3), s2:(2,3). The other two coordinates retain all feedback, so the h4 quotients retain three state coordinates. Nine are weak heuristics (accuracy .19–.40); all twelve have a nonlinear-chart/random-control geometry caveat.

The six magnitude-delay-line forms are a signed live trace plus two leaky magnitude registers, not `argmin |x|` or exact finite windows. Their substitution agreement is .980–.987, joint agreement .962–.974; complement accuracy share .989–.995. T7 s4: .958 substitution, .932 joint, .971 complement share. T3 s3 and s4: .955–.965 substitution, .952–.960 joint, .931–.948 complement share.

Width 8. h8 T3 s0 passes pair (1,0), combining ramp/carrier neurons while retaining six other comparator coordinates (seven in all): substitution agreement .966, complement share .921, joint agreement .950–.952; its discarded common mode is active on .375–.377 of final rows. Two seven-register recurrent filter banks closed h8 T11 seeds 0 and 3.

Widths 16 and 32. Seven fifteen-register comparison/filter banks closed h16 T5 seeds 0, 1, 3, 4; h16 T7 seeds 0, 1; h16 T11 seed 3: high-retention recurrent reductions from 16 to 15 registers with every retained drive/readout specified, not pure medoid or compact task algorithms. Two fully specified 31-register h32 T7 banks, seeds 3 and 4, retain 31 coordinates: seed 3 pair (17,30), a central/compressed-history comparison plus 13 reflected filters and 17 refractory comparisons; seed 4 pair (31,2), an approximate lag-4 comparison plus 17 reflected filters and 13 refractory comparisons. All sixty retained drive/readout rows are explicit; complete recurrent q is the complement. Joint agreement .999913–.999980/.997607–.997853, minimum margins .097640/.092795. Six further 31-register banks closed h32 T7 seeds 1, 2 and h32 T5 seeds 0, 1, 2, 4; four 31-register fading-history banks closed h32 T11 seeds 0, 2, 3, 4 (all four weak task heuristics); five further banks closed h16 T5 seed 2, h16 T7 seed 2, h32 T5 seed 3, h32 T7 seed 0, h32 T11 seed 1 (two use downstream-balanced positive gauges; all retain fifteen or thirty-one state coordinates).

### 5.4 Full-state comparison/filter banks (19, read)

In the full banks, write d_i=s_i x+A_i z+B_i|z|+Σ_{k≠i}M_ik r_k. Positive-self rows have explicit reflected suffix solution r_i(t)=max(0,max_{k<t}Σ_{l=k}^{t−1}M_ii^(t−1−l)d_i(l)); negative-self rows retain [d_i+M_ii r_i]₊. All feedback uses previous registers. In the eighteen finite banks, all registers remain: d_i=s_i x+Σ_{k≠i}M_ik r_k, positive-self rows use the reflected suffix formula, negative-self rows refractory comparisons. The selected warm-up carrier instead uses m_t=a^(t−K)m_K+Σ_{l=K}^{t−1}a^(t−1−l)d_l after K reflected updates; a selected reset comparison uses [d_i]₊. The full accumulated prefix-message error, including downstream feedback, is the complement; C=J algebraically, not independent evidence.

Eighteen all-register finite filter banks closed every remaining h8 net, h16 T7 seed 3 and h16 T11 seeds 0, 1, 2, 4: four reset one refractory comparison; fourteen retain reflected warm-ups then use signed convolution tails. All 184 register roles, complete feedback and positional scores were frozen before testing; the complete accumulated message error was resampled. All eighteen are weak heuristics, not compact median algorithms or dimension reductions. A sixteen-register filter bank closed h16 T7 seed 4: it retains one reflected warm-up in its low-history register 5, then a signed leaky tail, all other comparisons/feedback/readout, and resamples the entire accumulated message error.

### 5.5 Finite width-4 function-residual forms (6, read)

Four finite comparison/filter forms, h4 T3 seed 1, h4 T5 seed 1, h4 T7 seeds 2 and 3, are not recurrent quotients: each passed four independent Gaussian streams. For T3 s1, set α=M[s]₊, β=M[−s]₊, b_i=[s_i x₁+α_i[x₀]₊+β_i[−x₀]₊]₊ for i=0,1,2; replace b₃ by k=α₃[x₀]₊+β₃[−x₀]₊ and read Vg relu(sx₂+Mb). Its full complement is M[:,3]·(b₃_native−k), not a partial residual; substitution agreement .966, joint .944–.945, complement share .972–.975.

For h4 T5 s1, v=−d+.856110e+.432401p and m_t=max(0,max_{k<t}Σ_{l=k}^{t−1}1.481344^(t−1−l)v_l); d/e are asymmetric low-excursion gates, p a high-excursion gate, with all cross-feedback retained. Removing only the carrier's .001373 direct input and resampling the COMPLETE accumulated message error preserves joint agreement .99865–.99876 across four streams.

The h4 T7 filters retain K reflected carrier updates, then m_t=a^(t−K)m_K+Σ_{l=K}^{t−1}a^(t−1−l)v_l with all excursion/delay feedback retained: seed 2 uses K2, a=.729356, v=x+9.398256n0−.411201n1+.321702n3; seed 3 uses K1/2/3, a=.702177, v=−x+.627384n0−.719079n2+.011452n3. They remove at least three late carrier gates; joint agreement is .90179–.90399 for seed 2 and .90743–.95349 across seed 3's three separately registered forms. The h4 T7 seed 2 proof was weak and fragile: specificity cleared its fixed bar by only .000474. Two further finite forms closed h4 T5 seeds 3 and 4.

### 5.6 Width 4, length 11, seed 3 (read, never proven)

The remaining h4 T11 s3 has an exact FOUR-state reading: two reflected suffix-history filters m, d, a common low-excursion q, and signed branch contrast z. Set e=q+[z]₊, f=q+[−z]₊; with c2=M2·(m,d,e,f), c3=M3·(m,d,e,f), δ=c2−c3, update q'=[−x+min(c2,c3)]₊ and z'=sign(δ)min(|δ|,[−x+max(c2,c3)]₊). Keep both current-input leaks, every feedback term, and all positional scores. It is a clipped delayed-balance heuristic with an EMPTY complement, NOT an additional empty-complement exception, and stays READ. Its failed tests are in Appendix B.

### 5.7 Relation to published work

Leeuwenhoek already reads the same argmedian h4 T3 best-seed net as rise/fall flags plus memory, enumerates its paths, and tabulates exact T3 accuracy for h2/4/8; our N0 contrasts overlap that reading, adding empirical complement tests, not priority for the mechanism. ARC's blog fully treats 2nd-argmax M(2,2), approximately M(4,3), and 5/16 neurons of M(16,10); T3 is the same task but not necessarily the same weights.

Mogar additionally treats 2nd-argmax h4 T3 exactly and h32 T10 with a top-two tracker surrogate; DraggonZ builds h16 T10 copies; mrsirrisrm's gist predicts from top-two impulses; Snider fits accuracy estimators. Their T10 results do not explain our argmedian nets. [Dutta et al. (2026)](#references) construct parallel median-value approximators, not this tied streaming position circuit. Standalone ideal taps/costs and pure-medoid identities remain readings, not what those tests substitute.

## 6. Limitations and open questions

### What “proven” means

A network counts as proven only if its form passed the registered tests of its iteration and then the 2026-11-14 specificity gate. In the registered quotient tests, substitution (S), complement resampling (C) and their joint intervention (J) each retain ≥.90 of base accuracy, S and J agree ≥.90 with the net, and necessity and specificity drops exceed .05, on two independent 50,000-row Gaussian streams with three non-self donors at every step and a fresh replication.

The specificity gate ([METHODS §9.1](../METHODS.md#91-specificity-audit-of-the-algzoo-entries-2026-11-14)) draws 64 random circuits matched to the claim by output effect, uses ten independent 4,096-row streams and three donors, and requires on every stream that the claim keep at least .01 more than the strongest control, with every original gate still passing on the same rows. Of the 79 tested argmedian proofs, 77 fail on all ten streams (claim-minus-strongest margins −.0142646335 to +.0095284632; claim shares .8922069082–1.0128369704); 60 of the 79 had matched pools whose 64 controls all coincided with the claim mask, and 49 of the 77 failures were hits of the earlier audit. A failure of the new gate suffices for READ; passing it alone never proves. The one network already read was not retested. Proven means finite-sample evidence, not exhaustive over inputs and not unique or minimal.

*Table 1b. The 22 networks that remain proven, and why.*

| Networks | Kind | Why they count |
|---|---|---|
| h2 T3, T5, T7, T11, seeds 0–4 (20) | Width-2 whole-network exception, flagged | The claimed circuit is the whole network, so nothing is resampled and the control does not apply; inherited status |
| h16 T3 s1 | OS2 resample proof | The new gate and every original gate pass on 10/10 streams; advantage over the strongest control +.0398958 to +.0481545; claim shares .9075883406–.9165648170 |
| h16 T3 s4 | CF3c resample proof | The new gate and every original gate pass on 10/10 streams; advantage +.0621179 to +.0720215; claim shares .9044943820–.9118437118; worst individual donor agreement .897217, while the original gate uses the three-donor mean |

### Limits of the audit and earlier tests

The audit result is the verdict: 77 earlier proofs fail the strongest matched control on every stream. The 20 width-2 forms have empty complements, so their proven status says nothing about which parts could be discarded. The 60 coincident matched pools mean those claims could not be distinguished from size-matched controls at all. These are finite fresh-row specificity tests, not exhaustive population proofs or a search for replacement mechanisms.

Earlier quotient tests used thirty random subspaces, full/empty guards, zero/mean, necessity and specificity, in addition to the gates above. All sixty-three proofs added in the last rounds passed fresh replication; three older T3 proofs cleared a gate by .0066–.0091. h16 T5 seed 3 and h16 T7 seed 1 are thin: minimum margins .005647/.008680. The weak h4 T7 seed 2 K2 form is especially thin: joint agreement .90179–.90399 and specificity drop .050474 at its weakest stream; no bar was relaxed. The carrier form uses thirty matched wrong-prefix rotations/reflections, not random state subspaces; C=J algebraically. Native fp32/fp64 differences are separate.

Discarded q is genuinely nonzero: last-step occupancy .080–.096 for the delay lines, .110–.111 for T7 s4, .015–.017 for T3 survivors. Random controls match dimension, not nonlinear chart geometry. Aggregate empirical certificates do not prove exact full-support equality, unique semantics, minimality, or every-stratum preservation.

Fixed-readout length tests use 8,192 Gaussian rows at T+2/T+4; new indices cannot be emitted, so these are stress tests, not architecture-compatible generalisation. Structural-edge/path claims, balanced separating semantics and public reproduction remain incomplete. All failed tests listed in Appendix B stay failed.

Dependency-aware partial circuits were motivated by causal abstraction ([Geiger et al., 2023](#references)) and piecewise-linear recurrent dynamics ([Durstewitz, 2017](#references)); the width-4 length-11 transfer bounds use bounded-sum concentration ([Hoeffding, 1963](#references)). For comparison, read [AlgZoo median](algzoo-median.md) (scalar median value) and [AlgZoo second-argmax](algzoo-2nd-argmax.md) (another position task under the same audit).

## 7. Prediction record

Prediction preceded our probes, not ARC's training: expect a four-unit comparator at T3, possible medoid at h32 T5, no supplied full-signed T11 match. Narrow comparator and high-retention comparison/filter-bank replacements passed the earlier tests; wider positive-medoid compatibility does not establish a Gaussian mechanism. After the audit only two wide T3 tables and the width-2 forms remain proven.

## Appendix A. Weights excerpt and reading

For h4 T11 s0, pair (1,2), let m=n0, e=n3. The memory equations are m′=[−.001056x+.003247z+.996753|z|−.169618m−.010939e]₊ and e′=[.077260x+.007040z−.992960|z|+24.688733m−.078763e]₊ (rounded excerpts only). Signed trace branches use a_p=x−.073468z+.126282|z|+.098935m−.006936e and a_n=−x+.074494z+.130433|z|+.096987m−.006743e. Coefficients are weight algebra, not fitted predictors. N0 gains=(3.965263,4.699488,4.716320,3.965202), N2=(7.440723,3.774178,7.961251,3.756567). This chart is READ after the audit.

## Appendix B. Detailed evidence

*Table 2. Descriptive diagnostics and terminal accuracy per length; each cell lists widths h2/h4/h8/h16/h32 over five seeds. HAND R² is best held-out over seeds/times, often a trivial early state: A=(taps,hinges,accumulators,ghost,cost,margin), B=(z₁,z₂,|z₁|,|z₂|,pair-order,margin). Participation rank r_eff is (Σσ²)²/Σσ⁴, not circuit size; residual is a fit-only affine HAND remainder. Attention JS is NA throughout (no heads); zoo best accuracies are NR (not recorded) and terminal values are external, not our diagnostic accuracies.*

|T|Neurons h2/h4/h8/h16/h32|Best HAND R²|r_eff native; residual A/A+/B|Terminal accuracy|Fixed-readout +2 / +4 accuracy|
|--|--|--|--|--|--|
|3|0–2/4/6–8/8–10/10–13|h2 NA; h4 B:1,1,1,1,.763,.420; h8 A+:1,1,1,NA,1,.765, B:1,1,1,1,.763,.698; h16 A:1,1,1,1,1,.778, A+:1,1,1,NA,1,.872, B:1,1,1,1,.797,.855; h32 A:1,1,1,1,1,.861, A+:1,1,1,NA,1,.828, B:1,1,1,1,.811,.931|h2 1.085–1.970; NA. h4 1.114–1.858; NA/NA/1.023–3.056. h8 2.044–3.019; NA/2.740–3.334/1.658–1.834. h16 2.818–3.187; 2.975–5.207/2.443–4.116/1.944–2.638. h32 2.969–3.466; 3.018–5.579/2.587–4.711/2.114–3.089|.321–.410/.759–.985/.999258–.999454/.999387–.999511/.999382–.999533|.198–.220 / .138–.141; .165–.197 / .134–.156; .179–.213 / .128–.156; .156–.197 / .126–.141; .168–.193 / .132–.154|
|5|0–1/3–4/8/16/25–30|h2–h8 NA; h16 A+:1,1,1,NA,1,.697; h32 A:1,1,1,1,1,.733, A+:1,1,1,NA,1,.821|h2 1.626–1.819; NA. h4 1.305–2.398; NA. h8 1.388–2.999; NA. h16 5.548–6.477; NA/3.785–5.405/NA. h32 7.730–8.477; 5.590–9.442/7.771–9.676/NA|.228–.264/.342–.394/.665–.699/.953–.983/.993–.996|.139–.150 / .107–.116; .113–.149 / .100–.114; .088–.145 / .082–.121; .102–.119 / .098–.126; .099–.130 / .104–.125|
|7|0/0–4/7–8/15–16/29–32|h2–h16 NA; h32 A:1,1,1,1,1,.677, A+:1,1,1,NA,1,.552|h2 1.801–1.968; NA. h4 2.328–3.408; NA. h8 2.125–3.040; NA. h16 3.079–5.393; NA. h32 8.848–10.278; 2.390–6.115/5.624–8.591/NA|.159–.164/.224–.275/.462–.493/.690–.714/.937–.978|.094–.111 / .090–.094; .093–.102 / .088–.097; .084–.099 / .080–.092; .064–.084 / .066–.092; .080–.105 / .058–.085|
|11|0/0–2/4–8/12–14/30–32|h2, h4, h16 NA; h8 A+: unavailable at this width; h32 A+:1,1,1,NA,1,.325|h2 1.764–1.993; NA. h4 2.305–3.300; NA. h8 2.381–4.269; NA. h16 4.732–8.624; NA. h32 8.939–12.934; NA/4.424–9.783/NA|.104–.106/.136–.191/.321–.336/.430–.460/.673–.748|.071–.076 / .064–.068; .062–.075 / .067–.073; .062–.068 / .057–.066; .042–.072 / .047–.078; .044–.063 / .055–.070|

### Failed and limited tests (all retained)

Pair and register alternatives. Twenty-nine opposed-pair and two downstream-balanced alternatives fail; the latter retain only .473–.530 substitution agreement and .451–.574 complement share despite static cancellation. Three pooled-trace reductions fail (.449–.542 agreement, cosines .939–.990). In h4 T5 s0, retaining sign succeeds where magnitude pooling fails. Planted identities, endpoints and fresh N0/N2 controls pass. All five dependent-register tests and ninety wider reconstruction variants fail: drive relations do not survive separate gates.

Width-4 carriers. The checked h4 T3 s1 audit has ten prefix regions; two further checked certificates prove its carrier identity/bound only, not positional equality. Original carrier runs remain implementation-void for seed reuse; only corrected tests and their prospective replication count. The original opposite-drive audit has 27/90 descriptive reference passes, none counted directly. Twelve leaky-carrier audits give two descriptive passes. The h4 T5 s1 reflected excursion-balance amplifier passed deciding AND prospectively fixed replication streams (.9991–.9994 substitution agreement, .99865–.99876 joint). The h4 T7 s3 signed leaky filter fails initial joint agreement .89835 and replication .89942–.89948, not rescued by averaging. Twelve further checked complete prefix polyforms and six local carrier identities prove only bounded hidden-state facts; both serialization-limited prefix-3 audits are recovered, while four local bounds remain unknown. All ninety wide local-map/kernel cohorts pass numerical guards but give lookup kernels, not geometric/Prony/Jordan compression; map counts are not Gaussian coverage. Seed 2 K1 fails joint agreement and K3 fails replication specificity (.04832 drop), so neither alternative counts. Coupled-amplifier/reset-comparator forms for h4 T5 seeds 3, 4 passed all four initial/replication streams (joint agreement .909–.911/.969–.972); its metadata-key defect affects no native test.

Wide banks. h8 T11 seed 0 is thin (joint agreement .90917–.91208, complement share .96884–.97718); seed 3 has near-silent but active q (joint .99752–.99783, final q occupancy .00164–.00212). h16 bank joint agreement ranges .92609–.99958 at T5, .90868–.99041 at T7, .96750–.97042 at T11, minimum margins .005647/.008680 for the thin pair. Discarded q energy is only 2.4–3.3e−10 for h32 T7 seed 3 and 6.4–7.1e−7 for seed 4. Six h32 banks (T7 seeds 1, 2; T5 seeds 0, 1, 2, 4) with 180 retained rows frozen beforehand: T7 joint agreement 1/.999607–.999727; T5 .998300–.998533/.998740–.999067/.950407–.951753/.997307–.998100; the T7 seed 1 q energy is 2.94–5.13e−13. Four h32 T11 banks with all 120 retained rows frozen: joint agreement .990807–.991467/.996633–.997047/.988360–.988913/.999333–.999553; seed 4 q energy 7.65e−11–1.12e−10. A broader finite screen examines 2,152 frozen pair/gauge forms; only 9/48 independent descriptive audits pass reference bars, none counted directly. Five first-ranked banks with all 118 retained rows fixed: joint agreement .99777–.99829/.95618–.95787/.999933–.999993/.999073–.999293/.99280–.99328; minimum margins .09585/.05618/.09955/.09313/.06702; balanced h16 uses highly unequal gain factors .003721/.999993; q energy as low as 6.3e−10–1.1e−9. All 39 failed selected audits remain failures.

Full-state filter screen. A further 374-form finite filter screen gives 18/19 descriptive selected passes; the h16 T7 seed 4 warm-up has an empty complete residual and is rejected. The eighteen passing forms: joint agreement .959147–1, complement accuracy share .993166–1.001398, minimum original margin .030945; residual occupancy .00016–.97414 and energy 5.14e−9–.00049387. The h16 T7 seed 4 bank (five reflected filters/eleven refractory comparisons; carrier 5 a=.51027226448): joint agreement .99070–.991527, complement accuracy share .998685–.999372, minimum original margin +.070528; residual occupancy .05330–.05666, negative-tail occupancy .04722–.05030. Collection-only defects (two missing-input, one phase-suffix filename, one output-directory, one missing-log, one metadata-key) were corrected without changing native tests.

Other forms. Five pre-registered exact-slope group tests fail (.28–.71 substitution agreement). All ten general gap-graph variants fail; h32 T7 s3 reaches .930 substitution agreement but only .845–.847 group ties. All ten h32 T5 central-distance-blend variants fail (substitution .731–.900, joint .720–.857). Bounded/positive HAND-only replacements remain readings; error bounds or high R² alone never count.

### Width 4, length 11, seed 3 (open before and after the audit)

All ten fixed input/balanced quotients fail: best substitution agreement .915649 falls to .833862 jointly, and all necessity/specificity drops miss .05. Exhaustive fourteen proper coordinate masks across three time modes also miss that drop (maximum .046468 on 8192 rows). Four fixed singular-mode quotients and eight orthogonal/weighted geometry cohorts also fail. The exact FOUR-register reading agrees on 100000 fresh Gaussian rows (accuracy .1360–.1365, max state/logit errors 3.6e−15/4.9e−14). Checked two-input prefix geometry gives 10 paths/5 maps and 34 rational vertices, attaining a normalized hidden-contrast bound .0120083374062.

A four-state form removing only the common history's .000150371 contribution has .99992–1 substitution agreement, .999927–.999973 joint agreement and nonempty complete-message error (occupancy .7522–.7573, energy≈1.65e−9) but FAILS necessity/specificity on every stream (drops .0271–.0301, required>.05); all 182 exact clipped-coordinate subset/time maps also miss that bar (max .042440).

Exact final-gate envelopes on 1024 frozen donor prefixes enclose the uniform-mixture control accuracy at .107269495308–.107269495667; native fresh accuracy .135850 (100000 rows) has an ideal-iid-qualified upper .144367235; all 1024 certificates pass independent checking, including 1,008,579 rational winner inequalities; nine native last-winning intervals are noncentral, five below chance. With 8192 new donor prefixes and the checked conditional bound |K−1/11|≤.044518846, under explicit ideal-iid and exact-real/floating-transfer assumptions the Gaussian-donor accuracy lies in .103292–.111163, and native-upper minus donor-lower is ≤.042773 on both cohorts.

A threshold-drive chart (m,d,b,delta) fails all eight deletions; fifty-six signed partial maps reach at most .048340. All 5,376 recipient-specific partial controls over 128 prefixes have integral widths <7e−10, largest paired mean drop .0391781. Eight cone-preserving amplitude/fraction quotients fail (best substitution .917603, joint .877889); all 112 partial maps give maximum drops .04537/.04594; an exact final positive-scale quotient fails needed/specific drops .044759/.034828; all 2688 nonlinear controls over 32 prefixes give maximum paired drop [.048703793295,.048703793382]. Two score-block grammars give 0/1408 all-gate passes on 16384 fresh rows (random-control maxima .039439/.044671). A joint-history shared-reflection removal keeps .99718–.99788 substitution and .99640–.99716 joint agreement but fails needed/specific drops .04424–.04585/.03695–.04070 (complement active .02654–.02709). The final shared-floor law (iteration 24) fails needed/specific drops on all four 50,000-row streams. These are finite conditional results, not Gaussian-population bounds or global impossibility.

## What to read next

Compare [AlgZoo median](algzoo-median.md) for the median value rather than its position, and [AlgZoo second-argmax](algzoo-2nd-argmax.md), whose length-3 networks solve the same task. Read [the verification guide](../GLOSSARY.md#verifying-a-circuit) for why an empty complement and a resample proof have different evidential roles.

## References

DOI metadata and abstracts were verified with scite; these sources motivate methods, not exact identity or certified sampler claims. No argmedian proof is attributed to the T10 work.

- ARC. (n.d.). AlgZoo. *Code repository, MIT-0*. https://github.com/alignment-research-center/alg-zoo (Original model source.)
- ARC. (n.d.). AlgZoo: Uninterpreted models with fewer than 1,500 parameters. *Blog post*. https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters (Project context; detailed interpretations concern 2nd argmax.)
- DraggonZ. (2026). AlgZoo interpretation code. *Code repository*. https://github.com/DraggonZ/algzoo-interp-public (h16,T10 2nd-argmax copies, October 2026.)
- Durstewitz, D. (2017). A state space approach for piecewise-linear recurrent neural networks for identifying computational dynamics from neural measurements. *PLOS Computational Biology* (journal article). https://doi.org/10.1371/journal.pcbi.1005542 (Piecewise-linear dynamics.)
- Dutta, A., Safran, I., & Valiant, P. (2026). The Median is Easier than it Looks: Approximation with a Constant-Depth, Linear-Width ReLU Network. *arXiv preprint; abstract read*. https://doi.org/10.48550/arxiv.2602.07219 (Constant-depth linear-width median approximation.)
- Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. D. (2023). Finding alignments between interpretable causal variables and distributed neural representations. *arXiv preprint*. https://doi.org/10.48550/arxiv.2303.02536 (Causal abstraction motivation.)
- Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* (journal article; metadata only verified, no abstract or full paper read). https://doi.org/10.1080/01621459.1963.10500830 (Bounded-independent-sum concentration.)
- Leeuwenhoek. (2026). Leeuwenhoek. *Code repository*. https://github.com/Xabilimon1/leeuwenhoek (Argmedian h4/T3 comparison/memory reading, mechanism twins and exact path geometry, plus h2/4/8 T3 exact-accuracy tables, November 2026; direct overlap with N0.)
- Mogar. (n.d.). AlgZoo interpretation code. *Code repository*. https://github.com/mogar/algzoo_interp (2nd-argmax h4/T3 and h32/T10.)
- mrsirrisrm. (n.d.). Top-two impulse/lookup predictor. *Code gist*. https://gist.github.com/mrsirrisrm/d6850ff8647d1ed2f67cc92d5bce3ed0
- Snider. (2026). AlgZoo population accuracy estimators. *Forum post, 2026-10-26*. https://www.lesswrong.com/posts/sAhFp6gYoyw3xeCaF (Fitted accuracy estimators.)
