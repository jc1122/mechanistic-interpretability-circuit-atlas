# Addition in four output formats (router, inductive scratchpad, answer only): one shared column step; length needs a stage that cannot read position
**Task.** a + b, operands n digits, most significant digit first, base 2 (N = 16 bits) and base 10 (N = 6 digits); training draws fresh
pairs every batch with n ~ U[1, N]. Four output formats. AM / AL (answer only): `BOS a + b =` then the sum most- / least-significant digit
first. I (inductive scratchpad): `... = IDX_0`, then per column the line `s_k c_(k+1) IDX_(k+1)`. R (router): every operand digit is one
token (digit, significance id), coupled across the two operands; lines `ADD s_k CARRY c_(k+1) IDX_(k+1)`, so each written op token
names the sub-step that follows. I and R ids = column + a random offset (ids up to 2N are trained). Test: 1000 fixed pairs at n = N (in
range), N + 1 and 2N; exact = every scored token.
**Status.** 2026-11-06. HAND exact for all 8 (base, format) pairs. PART 10 seeds per cell (+ an arm with the stage's position inputs masked),
TRAIN 10 seeds per cell; failed cells retried once (3 seeds) where the retry rules applied. Audit 2026-11-07: the PART R stage rule and the masked
decimal PART AL spline pass the full bar (best and final snapshots, zero, mean and shuffle controls); the other forms below were run on best snapshots only.

## Summary
The hand-built step is one output-MLP circuit shared by every column (sum, carry and next-id neuron groups never overlap). Re-learning that
stage on the frozen hand-built net (PART) length-generalises exactly when the stage cannot read position: R without masking (2N ≥ .93 in
8/10 binary and ≥ .99 in 10/10 decimal seeds) and I, AM, AL once the stage's position inputs are masked (10/10 at 2N); unmasked I and A are 0 at 2N.
From scratch (TRAIN) I, R and AL solve in range (binary AL 8/10), most-significant-first answers do not (binary 0/10, decimal 2/10), and none reaches 2N (R included): the
router token alone does not buy length generalisation. Trained nets deliver the two operand digits by input-dependent attention mixtures
and compute the sum in distributed MLPs; no compact trained arithmetic formula was verified.

## Setup
[METHODS.md](../METHODS.md) for the shared conventions. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000, cosine, clip 1, batch 256 (fresh pairs
every batch: examples = 256 × steps); PART fp32 10k steps (masked I/A arms up to 20k), TRAIN TF32 30k; eval every 500 steps, best in-range
snapshot; stop at in-range 1.0, at the budget, or on a 5k plateau after min_steps. All seeds of a cell packed in one process, consumer GPUs.
"Masked" arm: the stage's input weights on HAND's position dimensions held at 0.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/router-addition](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/router-addition)
(105: 8 HAND, 80 PART, 17 TRAIN). Every file reloads bit-identically; fp32 re-scoring at n = N on 1000 fresh pairs matches the recorded
in-range accuracy within sampling error (3 binomial standard errors).
**Reproduce:** the training and analysis code for this task is not yet in `code/`; the weights and recorded accuracies are in the folder above.

## Architecture as used
GPT-NeoX 4 layers × 4 heads (d_head 64), d_model 256, MLP 512 ReLU, no LayerNorm, no positional encoding, untied readout; vocabulary 12
specials + B digits + K ids (K = 40 binary / 20 decimal) + B·K operand tokens (R), 512 positions. PART: HAND frozen, the stage MLP
re-initialised (normal .02, bias 0) and trained: I, R layer 2, AM, AL layer 3 (dense_h_to_4h [512, 256] + bias, dense_4h_to_h [256, 512]
+ bias). TRAIN: all weights from the default initialisation. Archive `router-addition/`: `hand_b<B>_<fmt>` (8 state dicts); PART stage
tensors for all seeds of the solved arms (R; masked I; masked AM/AL decimal; AM/AL binary); TRAIN full nets (earliest and latest best step
per cell, plus the circuit-probe seeds); `config.json`, `manifest.json` (base, format, seed, rung, masked, best step, recorded and re-scored
in-range accuracy, sha256).

## Prediction before training
Recorded in the task specification before the runs. (1) HAND R needs no absolute position, I and A need derived positions: held (R uses
recency and id matching; I/A index heads; all exact). (2) PART fits all formats in range quickly; R generalises in length, I and A only
with masked positions: length part held (R 2N final ≥ .93 in 8/10 binary, .99–1.0 decimal; masked I/AM/AL 10/10 at 2N); the in-range part
did not: unmasked I 0/10 (binary 0 at the 5k plateau stop, decimal .57–.79 at the 10k budget), unmasked decimal A 0/10 at 10k.
(3) TRAIN in range: binary AM fails, AL ≥ 7/10, I and R ≥ 8/10, R fastest: binary AM 0/10, AL 8/10 binary / 10/10 decimal, I and R 10/10
(held); R is not the fastest (decimal best step R 11.5–16.5k, I 8.5–16k); decimal AM 2/10 (not predicted; budget-limited). (4) TRAIN
length: only R generalises: refuted, R is 0 at 2N in both bases. (5) Routing visible in trained R nets (ADD and CARRY neuron sets
disjoint): not tested on trained nets; it holds by construction in HAND (below).

## Hand-built reference
R: layer 1 reads the id one-hot J of the last IDX, the value LD of the last digit written and the last op LOP (ADD / CARRY); layer-2 match
heads fetch XA / XB = the a and b digits whose id equals J (no absolute position). Stage MLP2: at ADD s = (XA + XB + LD) mod B (LD = the
carry just written); at CARRY c = [XA + XB > LD] (LD = the sum digit just written, so the carry is read off the written digit); after a
digit with LOP = CARRY the successor IDX_(J+1); the op cycle IDX → ADD → CARRY. I: the same stage; index heads locate a_k, b_k from the
id offset and slot flags from the distance to the last IDX. AL: index heads on derived position, MLP3 s = (a_k + b_k + CY) mod B; AM adds
a lookahead carry head (exact for n ≤ 30, so binary AM 2N = 32 is outside its range). Size (ablation-used heads; stage neurons per slot):
R and I 7 heads, sum 9 / 57 neurons (binary / decimal), carry 3, next-id one per id; AL 6 heads, AM 5. The stage's active neuron sets
overlap .71–.97 (Jaccard) across columns and 0 between slots. Exact (logit margin 30) on all or 500 random pairs per n = 1..N+1 and 2N.

## What training finds
| cell | solved (in range 1.0) | best step | n = N / N+1 / 2N (final, range over seeds) |
|---|---|---|---|
| PART R binary / decimal | 10/10 / 10/10 | 1.5–3k / 3.5–5k | 1 / 1 / .001–1 (8/10 ≥ .93); 1 / 1 / .99–1 |
| PART I binary / decimal (unmasked) | 0/10 / 0/10 | — | 0 / 0 / 0; .57–.78 / .14–.34 / 0 |
| PART I masked binary / decimal | 10/10 / 10/10 | 3.5–4.5k / 8.5–10k | 1 / 1 / 1 |
| PART AM / AL decimal (unmasked; 20k retry, 3 seeds) | 0/10 / 0/10; 2/3 (3/3 ≥ .99) / 0/3 | 16–19k (retry AM) | .99–1 / .24–.26 / 0; .59–.71 / .02–.04 / 0 (retry) |
| PART AM / AL decimal masked | 10/10 / 10/10 | 5.5–7k / 7.5–9.5k | 1 / 1 / 1 |
| PART AM / AL binary (unmasked) | 10/10 / 8/10 | 7.5–9k / 8–9.5k | 1 / .26–1 / 0; .39–1 / .16–1 / 0 |
| TRAIN I binary / decimal | 10/10 / 10/10 | 20–25k / 8.5–16k | 1 / .975–1 / 0; 1 / .27–.94 / 0 |
| TRAIN R binary / decimal | 10/10 / 10/10 | 7–22.5k / 11.5–16.5k | 1 / .995–1 / 0; 1 / .38–.92 / 0 |
| TRAIN AL binary / decimal | 8/10 / 10/10 | 23–29.5k / 13.5–16k | .99–1 / .61–1 / 0; 1 / .14–.91 / 0 |
| TRAIN AM binary / decimal | 0/10 / 2/10 | — / 25.5k, 27k | .001–.99 / 0–.94 / 0; .99–1 / .02–.47 / 0 |

- PART R: all diagnosed snapshots recover HAND's wiring (edge attribution: Spearman .87–1 vs exact patching, Jaccard 1 with HAND's edges,
  alignment to HAND's head/MLP weights R² .83–1, CKA .84–1). Binary seed 6 has that wiring and is exact in range, yet 2N .001: the shared
  in-range mechanism does not decide length. Unmasked PART I fails at the sum slot (carry and next-id 1, sum .74–.91; MLP2 patching drop 1):
  the stage latches onto position.
- TRAIN (I, AL, AM): 3–9 of 16 heads used under mean ablation (10–16 under zero ablation), 1–4 MLP blocks; no orthogonal map to HAND's residual
  (alignment fit R² < 0 in every diagnosed net), CKA ≤ .12 (in range), edge overlap with HAND's wiring Jaccard ≤ .23: HAND's step, not
  HAND's wiring. At 2N TRAIN I / AL are 0 and the patching drop moves to MLP1/MLP3; failed binary AM and the residual binary AL errors
  are not diagnosed, and trained R nets have no diagnostics yet.
- Circuit probes (best snapshots; layers zero-based; "verified" = causal intervention + accuracy-preserving substitution on the stated
  finite test, best snapshot, no shuffle control; "supported" = causal only):

| component | role | evidence |
|---|---|---|
| TRAIN I decimal s4: L3H0–H2 (sum), H1/H2 (carry) | input-dependent delivery of the two current digits | donor flips .31 / .25 (sum, H1), .49 / .43 (carry, H2); joint two-source form exact and agreement 1 on 512 pairs (verified; also seed 8) |
| TRAIN I decimal s4: L0H1/H3, MLP0 (56 neurons) | token-frequency aggregation; ReLU feature bank | solo substitution .9995 on 2048 pairs, native predictions kept (verified) |
| TRAIN AL decimal s1: L3H3/H2; L3H0 | separate a / b delivery; previous-column carry context | operand patches 128/128; joint lookup 256/256 at n = 6; carry patches 118/128, substitution exact 1 (verified) |
| TRAIN R decimal s3: L1H3 | reads the written carry | carry patches 128/128; latest-carry lookup exact 1, drop 0 (verified) |
| PART AL decimal masked s2: frozen retrieval + MLP2; MLP3 | operands and carry; reads their raw sum | MLP3 zeroed .99 → 0; spline substitution 1 / 1 / 1 on 2048 pairs per length (verified) |
| TRAIN I decimal s4: L2H1; MLP3 | carry information; mediates sum and carry | carry patches 128/128 but unit-gain replacement .043; deleting 98 carry-preferring MLP3 units: sum .995, carry .361 (supported) |

## Closed form
PART R stage: s = (XA + XB + LD) mod B at ADD, c = [XA + XB > LD] at CARRY, then the successor id and op cycle. Substituting it for MLP2
gives 1 / 1 / 1 (n = 1 / 6 / 12, decimal, 512 pairs each) vs the native net's 1 / 1 / .246; across 95 PART best snapshots the same rule
gives native .984–1 → 1 on 64 pairs (no fitted reals). Controls (20 seeds, binary and decimal, best and final, 512 held-out pairs at n = N):
substituted 1 vs native .996–1; MLP2 output zeroed or set to its mean 0; the rule fed row-shuffled inputs .002: accepted in range. At 2N it is a
repair, not the net's behaviour. PART AL masked (decimal s2): u = XA + XB + CY, Y_d = Σ_{j=0..19} C[j, d]·max(0, 1 − |u − j|)/30 (a 200-number
fitted spline of the raw sum): 1 / 1 / 1 vs native 1 / 1 / .9995, an accuracy-level abstraction. Controls (seeds 0–9, best and final, 512 held-out pairs at n = 6): spline 1 vs native 1 (HAND's MLP3 also 1);
zero, mean, sign-flipped C 0; shuffled inputs .002; C taken from another seed 1 (the fit is not seed-specific): accepted in range. TRAIN: operand delivery
α = sigmoid((Q·K_a − Q·K_b)/8), z = α V_a + (1 − α) V_b with Q, K, V recomputed in the net; all 11 component forms of TRAIN I s4 together
give 8188/8192 exact vs native 8187/8192 (one native error repaired; agreement 8191/8192). The arithmetic stays in dense fits
(2,979,840 coefficients per head form): no compact trained formula is verified. Failed unmasked PART AL decimal s0 is repaired by
MLP3 = (XA + XB + CY) mod 10 (1 / .82 / 0 → 1 / 1 / 1), which does not explain its errors.

## Weights excerpt and reading
HAND R stage (layer-2 MLP): 56 (binary) / 84 (decimal) neurons with non-zero input weights, reading 8 residual groups: the id one-hot J, the last digit LD,
the last op LOP, the operand digits XA / XB, DIG, ISI, SPC. PART R binary (all 10 seeds): 63–78% of the squared norm of the trained
dense_h_to_4h lies on those 8 groups; the id one-hot J alone takes 48–58% (seed 4: J .58, SPC .05, LOP .035, XA .03, XB .03), the
operand digits 2–3% each. Reading: most input weight serves the successor map IDX_J → IDX_(J+1), which in HAND needs one neuron per id;
the sum and carry tables need only the few XA, XB, LD, LOP dimensions. Values = squared column norms of `gpt_neox.layers.2.mlp.dense_h_to_4h.weight`
of `b2_part_IR_R_s<seed>` over the HAND input dimensions (non-zero columns of the same matrix in `hand_b2_R`), grouped by the HAND
residual variable, divided by the matrix's total squared norm.

## Comparison
R and I share HAND's stage and differ upstream (id matching vs index heads). Coupled significance ids (Cho et al. 2024; McLeish et al. 2024)
give the stage nothing positional, which is what masking gives I and A; trained from scratch the upstream attention still fails at 2N.
Length generalisation of this kind is expected only when long-sequence internals are reproduced at trained lengths (Izzo et al. 2025).
Small transformers learn addition with sharp transitions in steps and data (Lee et al. 2023), as in the late rise of the unmasked PART I
and A stages. The inductive format follows Abbe et al. (2024). Same per-column lookup as the binary-addition and decimal-hex-addition
entries.

## Controls and limits
Masked vs unmasked PART arms control for the position shortcut. Binary AM HAND is exact only to n = 30. Decimal AM TRAIN (2/10) is
budget-limited; its 60k retry has not run past a smoke test. Unmasked PART A and I were budget-limited at 10k (retries from scratch at
20k: decimal AM 2/3, AL 0/3). Diagnostics on 2–10 seeds per cell; trained R nets, failed binary AM and failed PART I/A cells are not
diagnosed; routing in trained R nets is untested. Teacher-forced scratchpads; one size (d_model 256).
Open: final-snapshot and shuffle controls for the TRAIN operand-delivery form and the PART AL s0 repair; five PART best seeds without an interpreted stage;
binary TRAIN R per-seed length scores; dense PART stage tables (2.6–25M fitted entries) exceed the stage's 262,912 parameters, so they are
not compact forms (rejected). Follow-up, binary subtraction (a ≥ b, borrow), PART I / R, 3 seeds: native 512/512 at n = 16 on best and
final snapshots; at 2N I is 0 in every seed, R best 1 / .06 / 1 and final 1 / 1 / 1; a graph of 7 heads plus 2–3 whole MLPs keeps 512/512
and its complement gives 0/512 (no graph shuffle control yet). Decimal subtraction R (3 seeds) reached in-range 1 at its 20k budget.

## References
- Abbe, E., et al. (2024). How far can transformers reason? The globality barrier and inductive scratchpad. arXiv preprint. https://doi.org/10.48550/arxiv.2406.06467
- Cho, H., et al. (2024). Position coupling: improving length generalization of arithmetic transformers using task structure. arXiv preprint. https://doi.org/10.48550/arxiv.2405.20671
- McLeish, S., et al. (2024). Transformers can do arithmetic with the right embeddings. arXiv preprint. https://doi.org/10.48550/arxiv.2405.17399
- Lee, N., et al. (2023). Teaching arithmetic to small transformers. arXiv preprint. https://doi.org/10.48550/arxiv.2307.03381
- Izzo, Z., Nichani, E., & Lee, J. D. (2025). Quantitative bounds for length generalization in transformers. arXiv preprint. https://doi.org/10.48550/arxiv.2510.27015
