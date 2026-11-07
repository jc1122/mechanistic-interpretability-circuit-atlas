# Polynomial integration with reduced fractions: the scratchpad speeds up PART and transfers to unseen fractions; trained nets verify only local readers, not a gcd algorithm

**Task.** Given integer coefficients c₀…c_{m−1} of a polynomial (ascending powers), emit the integral's coefficients as reduced
fractions: per term j = 1…m the tokens `J:j N:n D:d` with n = c_{j−1}/g, d = j/g, g = gcd(|c_{j−1}|, j) (constant implicitly 0, zero
= 0/1). Prompt `BOS M:m C:c₀ … C:c_{m−1} =`; integers are typed tokens, not digit strings. Format **A** writes only `J N D` per term;
format **T** first writes every subtractive-Euclid register row `A:a B:b`, updating (a, b) → (|a − b|, min(a, b)) until b = 0, then
`G:g`, then `N D`. Training draws 2–4 terms with coefficients −9…9 excluding ±8, and excludes the reduced fractions ±2/3 and ±3/4;
a whole-polynomial hash gives a ~80/20 train/test split (69,649 legal mixed-length inputs before splitting, so draws repeat).
256 unique held-out 4-term polynomials per seed; exact = every output token incl. EOS, teacher-forced.
**Status.** 2026-11-07. HAND A and T exact. Three-seed screening runs (seeds 0–2) for PART and TRAIN in both formats, all solved.
Final cell T TRAIN, ten seeds: 9/10 solved (seed 4 .875 at the 30k budget). Pending: dynamics of six T TRAIN seeds and two diagnostics, public code.

## Summary
All four screening cells reach whole-output exactness in 3/3 seeds; the ten-seed T TRAIN final solves 9/10, and the unsolved seed
fails only on the signed numerator. The scratchpad cuts the PART stop step from 26,000 to 5,500 but raises
TRAIN's from 14,500 to 27,000. Only T PART generalises beyond the training distribution: it scores 1.0 on the withheld fractions
(A PART 0) and .59–.88 at 5 terms; every TRAIN net scores 0 on 5 or more terms, held-out ±8 and withheld fractions. In trained T nets,
layer 0 is exactly lexical prefix pooling (substituted formula keeps every prediction, 6/6 snapshots), and in seed 0 a layer-2 head is a
categorical coefficient reader (substitution 512/512). The same reader fails in seeds 1–2, so no complete gcd/division closed form is
accepted.

## Setup
[METHODS.md](../METHODS.md) for shared conventions. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1,000, cosine over 30k, clip 1,
batch 16 fresh polynomials per step, eval every 500 steps on the 256 held-out polynomials (best = highest exact, then lowest CE),
checkpoints every 2,500, stop when all seeds are exact (5k plateau after 22.5k otherwise). PART fp32, TRAIN TF32; all seeds of a cell
in one process on consumer GPUs; re-scores fp32 on CPU. Out-of-distribution sets (64 polynomials per seed): 1, 2, 3, 5, 6, 8 terms,
held-out ±8, withheld fractions, wide and outer coefficients.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-integration](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-integration)
(38 nets + 2 HAND: best and final snapshots of seeds 0–2 in every cell and of T TRAIN final seeds 3–9; held-out and OOD rows in
`test_integ` (seeds 0–2) and `test_integ_2c76ff` (seeds 3–9)). Reloaded in fp32, all 38 reproduce their recorded fp32 held-out exact
(1.0; seed 4 .875) and both HANDs score 1.0.
**Reproduce:** the generator, HAND and trainer for this task are not yet in `code/`; weights, test rows and accuracies are in the folder above.

## Architecture as used
GPT-NeoX, 4 layers, d_model 64, 4 heads of 16, ReLU MLP 384 (A) / 448 (T), no LayerNorm, no positional encoding (causal mask only;
a BOS-derived positional frame is built by HAND), untied read-out, vocabulary 130, context up to 218 tokens. TRAIN learns every tensor
(A 16 heads + 1,536 neurons; T 16 + 1,792). PART: HAND frozen, layer-3 MLP (`dense_h_to_4h` [384 or 448, 64] + bias,
`dense_4h_to_h` [64, 384 or 448] + bias) re-initialised (normal .02) and trained, input columns masked to the HAND interface (zero-based):
ONE 0, FLAG 2–9, NUM 10–17, TERM 21, COEF/AC/NEG 36–38, OLD 39–40, ABS/MIN/ZERO/FIRST/LAST 41–45, EQ 63. Files: `hand_<fmt>`,
`part_<fmt>_<best|final>_s<seed>` (4 trained tensors; insert into the HAND), `train_*` (full state dicts), `test_integ`, `config.json`,
`manifest.json` (seed, best/stop step, recorded and re-scored exact, sha256).

## Prediction before training
Recorded before the runs: PART T ≥ 2/3 solved by 10k steps (held: 5.5k) and PART A ≤ 1/3 by 30k (refuted: 3/3 by 26k); TRAIN T ≥ 2/3
by 25k (held: screen 2/3 at 24k, the third seed needed 27k; final 7/10 first exact by 24.5k); TRAIN A ≥ 1/3 by 30k (held, 3/3 by 14.5k). PART length transfer ≥ .90 was predicted
and refuted (.59–.88 at 5 terms); a T-over-A advantage on withheld fractions held for PART only.

## Hand-built reference
A compiles the finite reduction table into triangular ReLU hinges on the joint (coefficient, exponent) code. T shares coefficient and
register fetches, runs the subtractive Euclid update until b = 0, then applies bounded signed quotient staircases for N and D. A
BOS-derived positional frame and clock addresses do the routing. Formula-active size A = 4 heads + 428 neurons, T = 6 + 769 (not
learned minima). Exact on 2,804 polynomials per format (exhaustive for 1–2 terms, stratified above), minimum target margin 10.

## What training finds
| Cell | Solved | First ≥ 50% / ≥ 90% test (s0, s1, s2) | Stop | Draws / seed | 5 terms | held ±8 | withheld fractions |
|---|---|---|---|---|---|---|---|
| A PART | 3/3 | 16.0k, 15.5k, 13.0k / 19.5k, 22.0k, 20.0k | 26,000 | 416k | .05–.12 | .16–.31 | 0 |
| T PART | 3/3 | 2.0k, 1.5k, 2.5k / 2.0k, 2.0k, 2.5k | 5,500 | 88k | .59–.88 | .53–.77 | 1.0 (3/3) |
| A TRAIN | 3/3 | 7.0k, 7.0k, 9.5k / 8.0k, 7.5k, 11.5k | 14,500 | 232k | 0 | 0 | 0–.19 |
| T TRAIN | 3/3 | 14.5k, 15.0k, 17.0k / 16.0k, 16.0k, 22.0k | 27,000 | 432k | 0 | 0 | 0 |
| T TRAIN final s3–9 | 6/7 (cell 9/10) | — / 14.5k, —, 18.0k, 22.0k, 16.5k, 20.5k, 17.5k | 30,000 | 480k | 0 | 0 | 0 |

OOD columns: best snapshot, 64 polynomials per seed, from the training logs. All cells score 1.0 at 2–3 terms; 8 terms and outer coefficients
are 0 everywhere, wide coefficients ≤ .09; T PART scores 1.0 and A PART .5–.88 on single terms. New term counts use unseen M/J token ids, so the
term-count columns are not pure length tests.
Unsolved seed 4 (best = final = step 30k): exact .875, numerator token .972, every other role ≥ .999; numerator interchange .933
(nf 75), DAS .920, gcd DAS .984: the state is present, signed-numerator selection at G→N fails (budget-limited; not rerun).
Diagnostics (available snapshots): A PART and T PART each form one frozen-interface family, 3/3; removing the trained MLP destroys
exactness; residual rank at layer 4 (r90) 1 = HAND's; A PART coefficient probe R² ≈ .9998 at layer 3, T PART gcd R² .42–.61.
T TRAIN: one distributed, MLP-heavy family, 10/10; any MLP zero/mean lesion gives exact 0; 12–14 mean-sensitive heads; residual
r90 [13, 9, 12] vs HAND 1; numerator R² [.74, .30, .27] and gcd R² [.96, .94, .87] at layer 4, yet numerator interchange flips 1.0 with
answer-fixing cases (used, coded non-linearly); layer-3 attention JS to HAND .46–.60. All 448 last-MLP single-neuron lesions per T TRAIN
snapshot cost ≤ 5 points (redundancy, not a joint minimum). A TRAIN: one distributed four-MLP family, 3/3 (sign bottleneck in layer 3
in s0/s2, head L2H3 in s1); probe R² numerator .49–.88, denominator .93–.95; interchange flips 1/1. LMC barriers between TRAIN seeds
3.5–9.4 nats (separate basins). Dynamics (interchange flip ≥ .5): T TRAIN denominator, gcd and minimum by ≤ 2.5k steps, the signed
numerator last (5k–12.5k; seed 4 latest); no memorisation or stall alert in any of 11 trajectories.

## Closed form
Two substitutions on T TRAIN, each on a fresh 512-polynomial hash-test batch per seed; bar: substituted exact ≥ .99 × native.
1. Layer 0, all heads: `P_h(t) = Σ_u N_u exp(S_h[t,u]) v_h[u] / Σ_u N_u exp(S_h[t,u])`, S = Q(E_t)·K(E_u)/4, v = OV(E_u), N_u = prefix
   count of token u. Keeps every prediction in all 6 best/final snapshots of seeds 0–2 (max head-vector error 1.8e-7): lexical prefix pooling.
2. Coefficient reader at G queries: layer-2 head h output replaced by `b_h + u_h[c] + v_h[j]` (fit on 1,024 train polynomials), or its
   attention by `1[k = j + 1]` (hard fetch).

| T TRAIN seed | Native | H0 categorical | H1 categorical | Joint | Hard-fetch H0 |
|---|---|---|---|---|---|
| 0 | 512 | 512 PASS | 512 PASS | 512 PASS | 512 PASS |
| 1 | 512 | 504 FAIL | 512 PASS | 468 FAIL | 7 FAIL |
| 2 | 511 | 412 FAIL | 502 FAIL | 408 FAIL | 37 FAIL |
Best and final snapshots give identical counts. No closed form for the Euclid/division computation is accepted; PART and A TRAIN have
no substitution yet.

## Weights excerpt and reading
HAND T layer-2 register update (formula weights, zero bias): three ReLUs with input → output `(a − b) → (ABS +1, MIN −1)`,
`(b − a) → ABS +1`, `a → MIN +1`; hence ABS = |a − b|, MIN = a − ReLU(a − b) = min(a, b).
T TRAIN seed 0 best, layer 0 head 0, raw float32 weight slices (columns 0–2) and bias:

| Row | w[0] | w[1] | w[2] | bias |
|---|---|---|---|---|
| Q₀ | −0.0403105 | −0.1027140 | −0.0268110 | 0.0550319 |
| K₀ | −0.0468027 | −0.0292281 | −0.2562236 | 0.0423894 |
| V₀ | 0.0778282 | 0.0074196 | −0.0264612 | −0.0090742 |
Contracted score from the embeddings and QK weights, `S₀[G:1, C:{−3, 0, 3}] = [−0.2216, 0.1847, −0.3445]`: repeated tokens contribute
by prefix count and lexical score, not through a scalar arithmetic coordinate. Fitted reader coefficients (activation fits, not raw
weights; coordinate 0): seed 0 H0 b = −0.1019, u[c=0] = −0.0537, v[j=1] = 0.0421 (final −0.1030, −0.0551, 0.0389); seed 1 H1
(accepted) −0.6105, −0.3814, 0.9723; seed 2 H0 (rejected) 0.1060, 0.1241, −0.1414.

## Comparison
HAND runs a formula-defined subtractive Euclid; the trained pieces that are verified are prefix pooling and a local coefficient
reader, not the same complete algorithm. The scratchpad helps PART (faster, unseen-fraction transfer, partial term-count transfer)
but slows TRAIN and gives it no transfer. This finite typed-register task is far narrower than symbolic integration [Lample & Charton];
the intermediate-state comparison follows [Nye et al.].

## Controls and limits
Seed 0 G-query zero-ablation of H0: 216/512 (best), 215/512 (final); wrong coefficient address 217/512 vs correct 512/512. Seed 2's
one fresh IID error is a numerator sign (coefficients [−6, 6, −6, 1], term 2: N:−3 instead of N:3). Teacher-forced checks only; no
free-rollout or OOD substitution guarantees. Not established: learned-neuron minima, a complete arithmetic abstraction, a closed form for A TRAIN or the PART stages, substitutions
on seeds 3–9. Pending: dynamics of six T TRAIN seeds and two diagnostics, public code.

## References
[Lample & Charton, Deep Learning for Symbolic Mathematics](https://arxiv.org/abs/1912.01412) — research preprint; task context.
[Nye et al., Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/abs/2112.00114) — research preprint; scratchpad comparison.
[Power et al., Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) — research preprint; terminology context only.
