# Longest cycle in ARC's AlgZoo transformer networks

A function on the nodes 0, 1, 2 can be drawn as a graph with one arrow out of each node. For `[1,0,2]`, node 0 points to 1, node 1 points back to 0 and node 2 points to itself, so the longest cycle has two nodes and the answer, its length minus one, is 1. Two attention layers must follow these pointers far enough to see which cycles close; how the trained networks do that is still unknown.

Interpretation is unsolved and the formula status is [partial](../GLOSSARY.md#verifying-a-circuit): **0 proven / 94 read / 6 unexplained**. Five historical low-rank substitutions have current supporting evidence, but they are numerical readings, not identified graph algorithms.

## The circuit at a glance

No circuit figure is shown, because the sources describe no graph mechanism exactly. The most complete reading, the rational path-code chart of width 2, n 3, seed 3 (Section 5.2), reproduces all 27 answers exactly, but its strengths are observed encoding hypotheses, not derived cycle counts; drawing it as pointer-following would claim more than the evidence supports.

## At a glance

| Fact | Value |
|---|---|
| Task | Longest cycle length minus one in a finite functional graph |
| Input → output example | [1,0,2] → 1: a two-cycle and a self-loop |
| Model | Two residual attention-only layers, one head each, width 2–8 |
| Rungs and seeds | Two bounded-domain HANDs; 100 external trained finals; five seeds per cell |
| Closed form status | Partial numerical readings; no native graph algorithm proven |
| Status | Unsolved interpretation, 0/100; last analysis 2026-11-14 (second attack); specificity audit 2026-11-14 not applicable (no proven net) |
| Weights | [Hugging Face: algzoo-longest-cycle](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/algzoo-longest-cycle), hand-built weights/evidence archive revision `22977ef`; ARC nets linked by URL and hashes, not redistributed |
| Reproduce | Public reproduction package not available yet |

We study 100 external final transformers, five seeds in each of 20 cells, with no training by us. Exact finite certificates and passing interventions below do not waive the distinction between a numerical reading and a graph algorithm. A second attack on 2026-11-14 (twelve preregistered analysis iterations on 21 fixed nets and a 60-source literature scout) added no proof. The 2026-11-14 specificity audit of the AlgZoo entries ([METHODS §9.1](../METHODS.md#91-specificity-audit-of-the-algzoo-entries-2026-11-14)) changes nothing here, because no network was proven before it.

Here HAND means weights set by formula ([rungs](../GLOSSARY.md#the-three-rungs)). A closed form replaces a network computation with equations; a complement test resamples everything outside the claimed circuit ([verification](../GLOSSARY.md#verifying-a-circuit)). “Read” means a named reading without a passing certificate; “proven” would require an identified graph operation that passes the tests in [Section 6](#what-proven-means). Labels [h,n,seed] give width, graph size and original seed.

## Limitations (read first)

- No network is proven. The 94 readings are numerical substitutions, exact finite charts or output screens; none supplies an explicit received-state graph operation that survives complete outside resampling with matched specificity controls.
- Only five historical low-rank readings currently replicate on fresh rows; [4,3,4] lost its current count when its complement failed a fresh split, and the other 82 historical reductions were not re-replicated.
- Exact rational arithmetic covers declared finite domains (27 inputs at n=3, stated treatment sets); it is not exact native hidden-state equality, an arbitrary continuous-box theorem or algorithm uniqueness.
- The second attack's failures are finite failures, not impossibility; the six unexplained nets remain open.
- Longer inputs require new vocabulary, position and readout conventions, so length generalisation is undefined.

## 1. Task and data

Input f:{0,…,n−1}→{0,…,n−1}, n=3–6; uniform functions, output longest cycle length minus one. Example: [1,0,2] gives 1. CPU census sizes are 27/256/3,125/46,656. All 702 distinct ordered recipient/donor pairs at n=3 are finite treatments, not independent samples; paired-write domains contain 18,252 treatments each. Historical fit/test splits were disjoint functions; census-guided hypothesis selection is not unseen-input validation.

## 2. Model

### 2.1 Architecture

Two unmasked residual attention layers, one head per layer, d_head=d_model=h∈{2,3,4,6,8}; no bias, MLP, normalization or dropout. Vocabulary, positions and classes each n; final-position readout. Parameter count 8h²+3nh, 50–656.

### 2.2 Frozen parts and trained part

ARC trained E[n,h], P[n,h], Q/K/V/O[h,h] per layer and U[n,h]. Our interventions freeze weights and replace declared computations only. Received-state replacements preserve separate query/source/key/root-query/direct roles; no original-input closure supplies mixed-state semantics.

### 2.3 Training

External terminal budget 2^30 sequences, seeds 0–4, twelve saved snapshots per seed. ARC reports A100 training; no local retraining, new optimizer, or GPU analysis. Source-designated best seeds differ from terminal checkpoint selection; best-checkpoint comparisons, grok steps and cross-length generalization are not established.

### 2.4 Reproduction

Public runnable reproduction package not available yet. No reproduction command is claimed; internal checked scripts and immutable reports are not a public release.

### 2.5 Archive

Six hand-built nets, `config.json`, the native loader and a manifest with sha256 hashes are on [Hugging Face](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/algzoo-longest-cycle) (revision `22977ef`; the folder is unchanged since `9764b25`). An independent anonymous download re-checked every file hash and the local round-trip record, and reloaded two nets (n=3 and n=4), reproducing their recorded accuracy on 8,192 fresh rows and their outputs bit-exactly. The 100 ARC nets are referenced by public URL and hash, not redistributed. ARC supplies original weights and checkpoints under MIT-0; a safetensors conversion is not a ready-to-load Transformers model or resumable training state.

## 3. Hand-built solution

Circular address u(j)=(cos(2πj/n),sin(2πj/n)); the first head matches f(i) to position j and reads f²(i). The second reads original values for f³ (HAND-a) or composed values for f⁴ (HAND-b); earliest root return selects length.

Disjoint blocks, lookup strength 128 and nonreturn penalties ≥8 make both exact only when the root lies on a longest cycle of length ≤4. Neither solves arbitrary graphs; all 50 aligned bounded-domain learned-net substitutions fail.

## 4. What training finds

*Table 1. Exhaustive census accuracy ranges over five seeds per cell.*

|n|Census accuracy ranges, h=2 / 3 / 4 / 6 / 8|
|---|---|
|3|.889–1 / .963–1 / .593–1 / .963–1 / 1–1|
|4|.660–.723 / .668–.902 / .828–1 / .891–1 / .941–1|
|5|.564–.631 / .598–.651 / .623–.810 / .821–.947 / .961–.997|
|6|.505–.523 / .521–.631 / .593–.722 / .671–.841 / .830–.854|

There are 25/100 zoo-exact finals. All 100 need both heads under resampling; each keeps its own provisional mechanism identifier. Angular f² probe R² reaches .818, but decodability is not causal use. Separate native f³/f⁴ identification, matched attention JS and effective used ranks remain unavailable; shared width or accuracy does not establish a shared algorithm.

*Table 2. A disjoint grouping of all 100 final networks by their strongest current evidence (2026-11-14). Counts are networks. The historical reductions are numerical substitutions counted in an earlier round, not graph algorithms.*

| Group | Networks | Proven | Read | Unexplained | Current evidence |
|---|---|---|---|---|---|
| Historical low-rank reductions, replicated on fresh rows | [2,3,4], [3,3,0], [3,3,2], [3,3,3], [3,4,3] | 0 | 5 | 0 | fresh-row replication passes; three decision-exact on all 27 inputs |
| Historical low-rank reduction, failed fresh replication | [4,3,4] | 0 | 1 | 0 | fresh complement/joint 37/42 below .90 |
| Other historical numerical reductions | 82 nets | 0 | 82 | 0 | earlier certificates, not re-replicated; no graph identification |
| Read without a historical certificate | [4,3,0] (weak), [3,4,4] (thin), [3,6,3] (thin), [2,5,3], [6,5,3], [2,6,2] | 0 | 6 | 0 | readings only |
| Unexplained | [4,4,2], [3,5,1], [3,5,2], [3,5,3], [3,6,4], [4,6,3] | 0 | 0 | 6 | no form passes; none shown impossible |
| Total | 100 | 0 | 94 | 6 | |

## 5. Mechanism and verification

### 5.1 Historical low-rank readings

Five current historical low-rank readings: [3,3,0], [3,3,2], [3,3,3] are decision-exact on all 27 inputs; [2,3,4] has one disagreement and [3,4,3] six. Fresh-row replication supports these numerical forms. [4,3,4] fails fresh complement/joint 37/42 and whole-domain complement 72/81<.90, with seven counterexample graphs, mostly two-cycles. Earlier 88 numerical reductions are not graph-algorithm identifications.

### 5.2 Rational path codes in [2,3,3]

In [2,3,3], rational local path codes replace both first-layer score computations. Reader 0 mixes sources 0, 1 for query 1, else 0, 2; reader 1 mixes 0, 1; root copies itself. For reader 0 queries 0/1/2, latter-source masses are (31/32,63/64,1), (0,3/4,1/6), (15/16,2/3,1), indexed by that source's target. Reader 1 queries 0/1 give masses 0/1; query 2 gives (1,1/24,1/3), indexed by source 0's target. These strengths are observed encoding hypotheses, not derived cycle counts.

Write w_i=Σ_(u,v)mu_i[u,v]·(P_u+E_v)B0ᵀ, received s_i=P_i+E[a_i]+w_i. Root-query y<tau gives c=1/64, else 1/96; select the first minimum of s_jx+c*s_jy over keys 0, 1 and emit (s_RD+s_selected B1ᵀ)Uᵀ. All 27 answers, native/joint complement 678/702, 684/702, 10/17 paths retained; each inside/random control passes, minimum slack 77/1170. Supported singleton/pair all-and-changed correspondence exceeds .91. Independent exact-rational replay checks 69,525 continuations, zero ties/unknowns, class margin >0.4003792348. Natural root-following equivalence is not a transported graph-operation proof.

The rational [2,3,3] chart decodes 111 states on 729 delivered-role/current-pointer cases. Local returns cap node-1 WRITE correspondence at 562/702; authoritative row overwriting retains 22/27 natural answers, joint 554/702. Independent replay confirms coloured-return collisions and an injective eight-junction extension on this chart; high ceilings prove no program. Adding historical/current/root edges without overwriting retains 26/27 natural answers, joint 658/702, but fails EVERY supported single/pair correspondence gate. The checked endpoint law preserves all 69,525 rational continuations and separates selector errors from value masking; source/destination weights still lack a graph-operation interpretation. Generalized equal-marginal/different-cycle fixtures are outside the declared rational chart, not native impossibility.

### 5.3 Other width-2, n=3 confidence readings

Other h2/n3 confidence readings retain all 27 native answers, with nonempty tested complements: [2,3,0] has 12,690 checked natural/complement/control/single-write decisions; [2,3,2] has a checked natural path-sign/extremum mechanism but node-1 support 20 remains low; [2,3,4] retains two native errors and one supported continuation mismatch. Native confidence or frozen linear encodings remain unexplained graph operations, so all stay read.

### 5.4 Output symmetry screens at n=4

Root-fixed S3 output projections pass a standard-component screen in [3,4,2] (244/256 native agreement) and an invariant screen in [3,4,4] (233/256). The latter gets 248/256 task answers right but agrees on only 4/25 native errors. Replacing the first score with 51 weight-derived root-pinned equality-pattern coefficients fails: 155/256 correct in [3,4,4], 121/256 in [3,4,2]; their joint complements fail too. Output symmetry does not establish that internal operation.

### 5.5 Second attack (2026-11-14)

An independent analysis line ran twelve preregistered iterations on 21 fixed nets (the six unexplained and fifteen read nets) while a literature scout read 60 sources and ranked hypotheses; neither submitted a proof, so the count stays 0 / 94 / 6.

- Raw-weight audits confirm 0/588 whole clean passes across 12 nets; an output-only group screen fails five of the six unexplained nets.
- Restricted affine/bilinear motif and dictionary searches find 0/108 native-block memberships.
- Received-root-conditional vote supports reach only 17/480 initial and 24/480 replication rows on the unexplained nets, below the frozen usefulness gates.
- First-write boundary substitutions preserve complete complement/joint accuracy on 0/840 donor streams.
- A paired reciprocal-current law, which omits only the shared reciprocal attention mass, passed 16 clean and 14 complete two-phase diagnostics, including unexplained [3,6,4] and [4,6,3], but new permutation-matched specificity controls rejected all 21 qualifications: only 72/280 positive-net streams clear the strict >.06 bar, against 280/280 for Haar-random controls, and [4,6,3] has matched drops .010345–.018476 against Haar .240605–.332762.
- A final traffic-mediation screen found one diagnostic without a measured failure ([4,6,3] payload arm: 420/420 supported role-stream passes, 46091/46656 native agreement), but it uses an annotated package that is not decodable from the received state, and no nonempty complement, needed or specificity test or fresh replication exists, so it cannot count.
- The scout's tested hypotheses (truncation 0/10, common-linear-primitive eligibility 0/10, homogeneous-cone eligibility 0/80) are exclusions, not mechanisms.

Nothing passed because no candidate supplied an explicit received-state operation that survives complete outside resampling with matched specificity controls; these are finite failures, not impossibility, and the six unexplained nets remain [4,4,2], [3,5,1], [3,5,2], [3,5,3], [3,6,4] and [4,6,3].

### 5.6 Relation to published work

[ARC's blog](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters) (blog) reads only 2nd argmax: M(2,2) fully, M(4,3) approximately, and 5/16 neurons of its separate M(16,10) example. [ARC's repository](https://github.com/alignment-research-center/alg-zoo) (code/weights, MIT-0) supplies these longest-cycle nets and task; it identifies no public longest-cycle circuit there.

Related readings do not overlap these graph nets: [leeuwenhoek](https://github.com/Xabilimon1/leeuwenhoek) (code repository, Nov 2026) reads argmedian h4 T3 rise/fall flags and activation paths; [mogar](https://github.com/mogar/algzoo_interp) (code repository) reads a 2nd-argmax h32 T10 online maximum/second-maximum tracker surrogate. [DraggonZ](https://github.com/DraggonZ/algzoo-interp-public) (code repository, Oct 2026) builds partial h16 T10 2nd-argmax copies; [mrsirrisrm](https://gist.github.com/mrsirrisrm/d6850ff8647d1ed2f67cc92d5bce3ed0) (code gist) supplies a top-two lookup predictor for that example, discarding other inputs.

[Snider](https://www.lesswrong.com/posts/sAhFp6gYoyw3xeCaF) (forum, 2026-10-26) fits RNN accuracy estimators, not graph mechanisms. [Dutta, Safran & Valiant](https://doi.org/10.48550/arxiv.2602.07219) (arXiv, 2026) construct ReLU median approximators, not these transformers. All eight listed pages were read; source-scope comparisons are not transferred evidence.

## 6. Limitations and open questions

### What “proven” means

A network would count as proven only with an explicit received-state graph operation that passes substitution, a nonempty complement resampled from donors, the joint intervention, needed and specificity controls and a fresh replication; numerical substitutions and empty complements remain readings. No longest-cycle network met this standard before or after the 2026-11-14 specificity audit, so the audit's matched-control gate ([METHODS §9.1](../METHODS.md#91-specificity-audit-of-the-algzoo-entries-2026-11-14)) had nothing to test here; the second attack applied permutation-matched specificity controls of the same kind to its own candidates (Section 5.5).

### Scope of the readings

Approximate forms and empty complements remain readings. Exact rational candidate arithmetic is not exact native hidden-state equality, an arbitrary continuous-box theorem, or conventional algorithm uniqueness. Native natural/complement/singleton real-arithmetic certificates keep their declared scope; other paired native targets/support remain checked fp64 operational evidence. Role-conflict/register exclusions cover only their declared treatment domains and register classes, never all graph programs. The first-reader replay lacks a complete registered end-to-end role-mutation control suite; both candidates fail regardless. Longer inputs require new vocabulary/position/readout conventions. Intended use is finite model-organism research, not a safety or generalization guarantee.

### Method sources

Typed, intervention-consistent variables follow causal abstraction ([Geiger et al., 2023a](#references)) and distributed variable alignment ([Geiger et al., 2023b](#references)); exact finite replays were motivated by proof-producing verification ([Isac et al., 2022](#references)) and compact proofs ([Gross et al., 2024](#references)). Substructure-count notions ([Chen et al., 2020](#references)) and equality-pattern coordinates ([Maron et al., 2018](#references)) motivated the graph screens. None is evidence for these nets.

## 7. Prediction record

Before inspecting ARC weights, after external training, we predicted no complete rooted-HAND match; zero HAND certificates is consistent. Wider-net pointer composition and serial-versus-doubling preference remain unconfirmed. No new pretraining prediction is claimed retrospectively.

## Appendix A. Weights excerpt and reading

For R[j,b]=P[j]+E[b], A_l=Q_lᵀK_l/√h and B_l=O_lV_l. In [2,3,3], E[0]=(9637423/8388608,5797343/1048576), exactly from archived fp32 weights; tau=13720734544499850543691/73786976294838206464 comes from exact raw-weight root atoms. The two scalar directions and native OV/U implement the numerical continuation above; their graph meaning is not assumed from natural correctness.

## Appendix B. Detailed evidence

Finite input-register ceilings exclude .90-agreement root-orbit-only readings in 60/100 nets, named indegrees plus root pointer in 100/100, and named one/two-step indegrees plus root pointer in 90/100. High ceilings prove nothing. The five numerical readings retain full-domain exceptions and fresh seed 202611132 replication; donor evidence is not every possible intervention. Independent return/overwrite replays preserve failed candidates. Public detailed artifacts and mechanism-tracking across checkpoints remain pending.

## What to read next

Compare [AlgZoo second-argmax](algzoo-2nd-argmax.md) and [AlgZoo argmedian](algzoo-argmedian.md), recurrent networks from the same zoo where the specificity audit did find proofs, and [grid BFS](grid-bfs.md) for a trained graph task. Read [the verification guide](../GLOSSARY.md#verifying-a-circuit) for why an exact numerical chart is not yet an identified algorithm.

## References

Overlap with the zoo sources is delimited in Section 5.6; none identifies these graph algorithms. All cited DOIs were scite-verified metadata/abstracts, not independent reproductions.

- ARC. (n.d.). AlgZoo. *Code repository, MIT-0*. https://github.com/alignment-research-center/alg-zoo (code/weights; the networks analysed here.)
- ARC. (n.d.). AlgZoo: Uninterpreted models with fewer than 1,500 parameters. *Blog post*. https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters (reads 2nd argmax only.)
- Chen, Z., Chen, L., Villar, S., & Bruna, J. (2020). Can graph neural networks count substructures? *arXiv preprint*. https://doi.org/10.48550/arxiv.2002.04025 (distinguishes substructure-count notions; its GNN expressivity bounds are not transferred to these transformers or used to declare the open nets impossible.)
- DraggonZ. (2026). AlgZoo interpretation code. *Code repository*. https://github.com/DraggonZ/algzoo-interp-public (partial h16 T10 2nd-argmax copies, October 2026.)
- Dutta, A., Safran, I., & Valiant, P. (2026). The Median is Easier than it Looks: Approximation with a Constant-Depth, Linear-Width ReLU Network. *arXiv preprint*. https://doi.org/10.48550/arxiv.2602.07219 (ReLU median approximators, not these transformers.)
- Geiger, A., Ibeling, D., Zur, A., et al. (2023a). Causal abstraction: A theoretical foundation for mechanistic interpretability. *arXiv preprint*. https://doi.org/10.48550/arxiv.2301.04709 (intervention-consistent typed/multi-source variables.)
- Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. D. (2023b). Finding alignments between interpretable causal variables and distributed neural representations. *arXiv preprint*. https://doi.org/10.48550/arxiv.2303.02536 (distributed variable alignment, not evidence for these nets.)
- Gross, J. N., Agrawal, R., Kwa, T., et al. (2024). Compact proofs of model performance via mechanistic interpretability. *arXiv preprint*. https://doi.org/10.48550/arxiv.2406.11779 (compact-proof motivation; our exact claims are finite rational/Taylor replays, not borrowed theorems about softmax transformers.)
- Isac, O., Barrett, C., Zhang, M., & Katz, G. (2022). Neural network verification with proof production. *arXiv preprint*. https://doi.org/10.48550/arxiv.2206.00512 (proof-producing verification motivation.)
- Leeuwenhoek. (2026). Leeuwenhoek. *Code repository*. https://github.com/Xabilimon1/leeuwenhoek (argmedian h4 T3 rise/fall flags and activation paths, November 2026; no longest-cycle circuit.)
- Maron, H., Ben-Hamu, H., Shamir, N., & Lipman, Y. (2018). Invariant and equivariant graph networks. *arXiv preprint; abstract read*. https://doi.org/10.48550/arxiv.1812.09902 (motivates symmetry/equality-pattern coordinates; our output screens and rejected first-reader replacements are not its theorem identifying a trained net.)
- Mogar. (n.d.). AlgZoo interpretation code. *Code repository*. https://github.com/mogar/algzoo_interp (2nd-argmax h32 T10 tracker surrogate.)
- mrsirrisrm. (n.d.). Top-two impulse/lookup predictor. *Code gist*. https://gist.github.com/mrsirrisrm/d6850ff8647d1ed2f67cc92d5bce3ed0 (top-two lookup predictor for a 2nd-argmax example.)
- Snider. (2026). AlgZoo population accuracy estimators. *Forum post, 2026-10-26*. https://www.lesswrong.com/posts/sAhFp6gYoyw3xeCaF (RNN accuracy estimators, not graph mechanisms.)
