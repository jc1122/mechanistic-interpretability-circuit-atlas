# GF(2) matrix product

> - **Task.** Multiply two 3×3 bit matrices over GF(2) (AND for products, XOR for sums), writing either only the 9 answer bits or a trace of running parities. Example: row (1, 1, 0) times column (1, 0, 1) gives running parities 1, 1, 1, so the entry is 1.
> - **Hand-built net.** Each trace token is a 3-bit XOR-AND lookup on (previous running parity, A_ij, B_jk), after two attention heads fetch A_ij and B_jk.
> - **Trained nets.** With frozen hand-built routing, one retrained MLP relearns the lookup in 10/10 seeds within 500 steps. Trained from scratch, the trace solves 10/10 (grokking at 3–7.5k steps) and computes the same variables, but fetches A_ij and B_jk with 2–7 heads one layer later, in another code; answer only also solves 10/10. Nothing generalises to m = 4, 5 (0).
> - **Verdict.** Solved (10/10 in every cell). Closed form: partial (frozen-part stage only: substituted 1.0 = native in 20/20, zeroed 0, shuffled keys ≤ .002; none for the from-scratch nets).
> - **Links.** [Weights](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/gf2-matrix-product) · Reproduce: not available yet · [Terms](../GLOSSARY.md)

## Task and status
**Task.** C = A B over GF(2), m = 3. Prompt `B column-major ; A row-major =` (bits, `,` between groups). T: per column k a segment of running
parities s_ij = s_i,j−1 ⊕ A_ij B_jk (s_i2 = C_ik; `,` between rows, `;` between columns); A: the 9 bits of C column-major (answer only).
Per seed 100k train / 2k test disjoint random (A, B) pairs of 2^18; exact match of all output tokens; extra evals m = 2, 4, 5 (500 random).

**Formats.**
- **T**: trace of running parities s_ij, per column k
- **A**: answer only, the 9 bits of C column-major

**Status.** 2026-11-05. HAND T; PART T, TRAIN T, TRAIN A, 10 seeds each: 10/10 solved in every cell (no retry needed); best = final in every
cell. Controls 2026-11-08: the PART closed form passes the full bar (best and final snapshots, matched zero and shuffle controls); the TRAIN
candidates of the earlier audit still lack these controls.
## Summary
The hand-built net with a trace (HAND T) computes each trace token as a lookup on (s_i,j−1, A_ij, B_jk) after two 3-D attribute-match heads fetch A_ij and B_jk. PART (layer-3 MLP
re-learnt on frozen HAND) solves 10/10 in 500 steps and reads only HAND's two fetch heads
(diag: used = L2H0, L2H1, MLP3 in 10/10, attention divergence JS 0 to HAND). TRAIN T (trained from scratch, trace format) groks (Power et al., 2022) at 3–7.5k and computes
HAND's variables (and, s flip 1.0 under interchange at L4) but fetches A_ij/B_jk with 2–7 L3 heads, one layer later than HAND, in another code.
TRAIN A solves 10/10 with no compact HAND (13 heads + 4 MLPs on s0). Nothing generalises to m = 4, 5 (0 everywhere).
## Setup
[METHODS.md](../METHODS.md). AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000, cosine (PART 10k, TRAIN 30k), clip 1, batch 256; eval every
500, best-test snapshot; early stop at test 1.0 or 5k-plateau after 15k. PART fp32, TRAIN TF32. Consumer GPUs, all seeds of a cell in one process.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/gf2-matrix-product](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/gf2-matrix-product)
(15: 1 HAND, 10 PART, 4 TRAIN; train/test split per seed in `split.safetensors`).
**Reproduce:** training code for this task is not yet in `code/`; the weights, held-out splits and recorded accuracies are on Hugging Face (link above).
## Architecture as used
GPT-NeoX 4 layers × 12 heads (d_head 32), d_model 384, MLP 256 ReLU, no LayerNorm, NoPE; vocab 6 (BOS `0 1 , ; =`), 256 positions (3.16M
params). HAND T: 1295 non-zero weights; 7 heads, neurons L0 256 / L1 72 / L2 4 / L3 7. PART trains the L3 MLP only (dense_h_to_4h [256, 384]
+ bias, dense_4h_to_h [384, 256] + bias). TRAIN: all weights from the default initialisation. Archive `gf2-matrix-product/`: hand_T, part_T_s0–s9 (stage
tensors), train_T_s2/s6 and train_A_s2/s5 (earliest/latest best step), split.safetensors, config.json, manifest.json (sha256, roundtrip_all_ok true).
## Prediction before training
Recorded in the task specification before any PART/TRAIN run: PART 10/10 at m = 3 and < .1 at m = 2, 4, 5; TRAIN T ≥ 8/10, grok 5–15k; TRAIN A
≥ 6/10; nothing > .2 at m = 4, 5. Held except PART at m = 2 (up to .40, predicted < .1); TRAIN T grokked earlier (best 4.5–7.5k).
## Hand-built reference
L0: position R from a uniform head + 256 reciprocal neurons. L1: 4 first/last heads (first separator → row length, first `;` → segment length,
last terminator, last separator); MLP: clean steps give segment SEG, group G, offset J (and the next-token versions SEGN, GN, JN). L2: H0 matches
(SEG 1, G = GN, J = JN) → MA = A_ij; H1 matches (SEG 0, G = SEGN − 2, J = JN) → MB = B_jk (score 2·20·[match] − 20 per attribute); MLP: separator
flags TS, TSEMI, TB = 1 − TS. L3 MLP: 4 neurons relu(±tok1 ± MA ± MB + 100 TB − (#ones − 1) − 100), one per cell with tok1 ⊕ MA·MB = 1, writing
Y1 − Y0; + Y0 bias neuron and `,` / `;` neurons. Exact: all 2^8 pairs at m = 2, every m = 3 pair, 2000 random at m = 4, 5.
## What training finds
- PART T: 10/10 test 1.0 at step 500; m = 2 .00–.40, m = 4, 5 0. TRAIN T: 10/10, t(.5) 2.5–5.5k, t(.9) 3–7k, best 4.5–7.5k; m = 2, 4, 5 0.
  TRAIN A: 10/10, t(.5) 2.5–8k, t(.9) 2.5–8.5k, train .99 at 3.5–9k, best 5–11k; m = 2 .07–.27, m = 4, 5 0.
- Circuit types: PART one type (HAND's L2 fetch heads + new L3 lookup) 10/10. TRAIN T: frame in L0–L2, fetch in L3, XOR-AND in MLP2/MLP3
  (9/9 diag seeds use 2–7 L3 heads; a, b linear only at L4, R² ≤ .39 at L3 in 8/9).
- Diagnostics ([METHODS.md](../METHODS.md), 250 test pairs, best snapshots; probes = best-layer R²/acc; interchange flips as in Geiger et al. (2023); JS of used heads to the closest HAND head):

| Cell (diag seeds) | used heads + MLPs | best probe R² (layer); interchange flip | r99 L4 | JS used heads | best / final |
|---|---|---|---|---|---|
| PART T (10) | 2 + 1: L2H0, L2H1, MLP3 (10/10) | a, b, prev 1 (L3); and .87–.92, s .92–.96 (L4); flip and 1 (L3), s .81–.92 | 7 (HAND frame) | 0 | 1.0 / 1.0 |
| TRAIN T (9; s4 pending) | 2–9 + 2–4 | a .85–.99, b .93–.99, and .89–.99, s .93–.99 (L4); flip and, s 1 (L4) | 1–14 | .00–.60 | 1.0 / 1.0 |
| TRAIN A (2: s0, s1) | 13–14 + 4 | out .95–.96, cnt .96–.98 (L4); flip out 1, cnt .79 (L4) | 24–28 | — (no HAND) | 1.0 / 1.0 |

- Failures: none (all cells 10/10). Length: PART keeps HAND's frame, yet m = 4 is 0 because the re-learnt L3 also keys on m = 3 features (TB/TS
  scale); TRAIN learns absolute separator positions (col/grp probes).
## Closed form
PART: stage → its mean output per (tok1, MA, MB, TS, TSEMI) key: exact 1.0 and agreement 1.0 with the net on 2000 test pairs, 10/10 seeds (24 keys).
Controls (2,048 fit / 2,000 held-out pairs, best and final snapshot of all 10 seeds; the two snapshots are identical files): substituted 1.0 =
native in 20/20; stage output zeroed 0, closed form fed shuffled keys ≤ .002, per-query mean ≤ .016, previous bit negated 0; another
seed's table also gives 1.0 (one shared truth table). Accepted. TRAIN: no shared basis; and, s linear at L4 (flip 1.0); align
(s0, s1) fit_r2 < 0, QK/OV mean_ov ≤ .05, edge Jaccard .09 vs HAND: HAND's variables, not HAND's wiring.
## Weights excerpt and reading
HAND L3 cell neuron 0: relu(−tok1 + MA + MB + 100 TB − 101) fires only at (s_i,j−1, A_ij, B_jk) = (0, 1, 1) and writes Y1 − Y0.
PART: only 5.2–6.1% of the stage's W_in norm lies on HAND's six L3 inputs. Strongest Y1 − Y0 neuron (W_in / row max): s0 n243
(+.60 tok1, −1.00 MA, −.61 MB, +.26 TB; out +.26), s7 n46 (−.54, +.93, +.96; out +.24), s3 n209 (−.45, −.76, −1.00; out −.27).
Reading: in 10/10 seeds the top neuron weighs tok1, MA, MB at ≥ .45 of its row max with the sign pattern of one truth-table cell and votes
that cell's answer: (1,0,0)→1 in 5 seeds, (0,1,1)→1 in 3, (0,0,0)→0 and (0,0,1)→0 in one each. Same cell-detector form as HAND, different cells.
## Comparison
Same algorithm as HAND in PART (the same truth table as a matrix-vector trace, with the B fetch in place of the vector bit). TRAIN T computes HAND's variables but moves
the fetch to L3 and spreads the AND/XOR over MLP2–MLP3; TRAIN A needs ≥ 13 heads, consistent with a query-dependent partner per output bit. Charton (2021)
trains transformers on real-valued matrix products; here the field is GF(2) and the trace makes each step a 3-bit lookup.
## Controls and limits
m = 3 only; 38% of all pairs seen in training. Linear probes; 250-pair diagnostics; JS ties on uniform heads. Diagnostics on best snapshots
(best = final for every seed); TRAIN T s4 and TRAIN A s2–s9 not diagnosed.
## Circuit found in trained nets
Circuit probes (2026-11-06). **Verified** = causal intervention plus accuracy-preserving internal substitution; **supported** = causal
only. Exact = teacher-forced complete-sequence accuracy. Layers zero-based. The deep probes were run on a matrix-vector variant of this
task (trace s_i,j = s_i,j−1 ⊕ (M_ij ∧ v_j), matrix size 4, same format and network family); the matrix-product cells of this entry were
covered by the seed census only.

**Matrix product (this entry's cells), seed census.** T PART: early two-fetch + XOR-AND/separator macro, verified in 10/10 seeds
(consistent with the Closed form's 24-key table). T TRAIN: a Boolean oracle table suffices in 6, a query-dependent one in 3, richer
context in 1; no neural decoder was verified, so the TRAIN macro is unverified. A TRAIN: count-only oracle 2, frame/weight-dependent 2,
richer context 6; internal codes are hypotheses.

**Matrix-vector trace, best TRAIN (seed 0, replicate seed 8).**
| Component | Role | Evidence (verified) |
|---|---|---|
| L3H6 / L3H2 | matrix / vector bit transport | donor bit patches flip 64/64 each; hard positional routes separately and jointly 1.0, drop 0 |
| L3H4 | previous-parity message, conditioned on the query | previous-token donor flips 64/64, M/v donors 0/64; zeroed .012; μ[query, prev] substitution 1.0 |
| MLP3 | s = prev ⊕ (M ∧ v), plus separator | μ[M̂, v̂, prev, sep] substitution 1.0, drop 0; no single neuron necessary |
| PART L2H0/H1 → MLP2 | same macro one layer earlier, on frozen fetches | operand donor flips 64/64 in all ten seeds; 8- and 16-cell substitutions 1.0 (seed 0) |
| A PART MLP3 | parity of the count c = round(1/X − 1) | MLP3 zeroed 0; μ[c] substitution 1.0 in 10/10 seeds (seed 3 .9955 → 1 is a repair) |

- μ = training conditional-mean contribution at the replaced boundary; hats = live neural decoders. All TRAIN forms together (hard M/v
  routes + μ[query, prev] head + MLP3 table) give 100% in seeds 0 and 8, also on 2000 fresh instances: a verified dense decision macro, not
  exact amplitudes or a closed form of the early layers. The eight-cell (AND, prev, sep) macro holds in 1/10 TRAIN seeds, the
  separate-operand macro in 8/10.
- Supported: L0–L2 form a position/current-token frame (dense whole scrub 100 / 99.85–99.90%); the same story fails on the pruned circuits
  (69.2–71.0% seed 0, 88.7–89.8% seed 8), so it is not a complete minimal mechanism.
- Size (joint mean-ablation local minima at ≥ 99% of native): seed 0 35 heads + 284 neurons, seed 8 33 + 272, vs HAND 6 + 364; on fresh
  instances 98.70 / 98.45%, below 99%. Joint substitution on the pruned circuits: seed 0 54.5%, seed 8 100%.
- Size transfer (matrix-vector): count-parity and separator-relative tables restore A PART m = 6, 8 and T PART m = 2, 3, 5, 6, 8 from 0 to 1 (repairs):
  the learned stage keys on absolute separator positions, as the Length note in What training finds describes for this entry.
- Unlike HAND: trained nets fetch operands and update in L3, not L2, and add a previous-parity path; PART keeps HAND's fetches but uses
  redundant decision neurons. Open: the early frame code; answer-only TRAIN decoders (only 2 A TRAIN best snapshots diagnosed, no finals);
  the matrix-product TRAIN neural macro.

## References
- Charton, F. (2021). Linear algebra with transformers. https://doi.org/10.48550/arxiv.2112.01898
- Power, A., Burda, Y., Edwards, H., et al. (2022). Grokking: generalization beyond overfitting on small algorithmic datasets. https://doi.org/10.48550/arxiv.2201.02177
- Geiger, A., Wu, Z., Potts, C., et al. (2023). Finding alignments between interpretable causal variables and distributed neural representations. https://doi.org/10.48550/arxiv.2303.02536
