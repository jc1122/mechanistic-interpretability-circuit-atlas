# Linear systems over F₅/F₇: a microstep scratchpad keeps the hand-built arithmetic-to-router split; a fused update stage bypasses it

**Task.** Solve an invertible system Ax = b over F₅ or F₇, dimension m = 2 or 3, by Gauss-Jordan elimination with the first unused
nonzero pivot; one residue per token. Format **A** emits only the solution; **T** emits every updated augmented matrix; **S** also
writes the microstates of each update (pivot inverse, normalised operand, product). Output lengths A/T/S: 2/25/61 tokens at m = 2,
3/61/169 at m = 3. Fresh systems every batch; a seed-specific matrix hash (mod 5 = 0) selects held-out matrices, all of whose
right-hand sides are excluded from training; 2,000 held-out systems per seed. Exact = whole teacher-forced continuation.
**Status.** 2026-11-07. HAND T and S exact. Three-seed screens (30k steps; T TRAIN retries 60k or double width); ten-seed final cell
F₅m2 S TRAIN 10/10. Lost (not relaunched): the initial F₅m2 A/T TRAIN nets and the unfinished F₇m3 S PART run. Pending: diagnostics
of final seeds 3–9 and of the F₇m2 T width retry, public code.

## Summary
With the fused T format, training only the update MLP on frozen hand-built routing (PART) learns a direct readout that bypasses the
frozen subtractor: one bypass family in all 12 diagnosed seed-runs, solved 3/3 only at F₅m3. A categorical decision surrogate replaces
that MLP with 1024/1024 exact in all 24 best/final snapshots. With the S microstep format, PART keeps the hand-built decomposition
(multiply/subtract candidates, then a typed router): a three-map form keeps 1024/1024 in all 8 snapshots of the 4 solved seeds, but
lacks a substitute-shuffle control, so it is not accepted.
From scratch, S solves every screen (3/3 in all four cells) and the F₅m2 ten-seed final (10/10); T only after a budget retry at m = 2
(F₅ 3/3, F₇ 2/3; a further width retry at F₇m2 0/3), A never; no TRAIN arithmetic closed form passes.

## Setup
[METHODS.md](../METHODS.md) for shared conventions. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1,000, cosine, clip 1, batch 64
(F₅m3 S TRAIN 16 after a memory failure); eval every 500 steps on the held-out systems, best = highest held-out exact; checkpoints every
2,500; stop when solved or on a 5k plateau after 75% of the steps. T budget retries 60k steps; m = 3 width retries double d_head and MLP.
PART fp32, TRAIN TF32; consumer GPUs; re-scores fp32 on CPU.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/linear-systems-finite-fields](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/linear-systems-finite-fields)
(158 nets: best and final snapshot of every available pack incl. the F₅m2 S final seeds 3–9 and the F₇m2 T width retry, full state
dicts; each run's held-out systems in `test_<run>_p<p>_m<m>`). Reloaded as GPT-NeoX in fp32, all 158 reproduce their recorded held-out
exact (TF32 nets within 3/2000).
**Reproduce:** the generator, HAND and trainers for this task are not yet in `code/`; weights, test systems and accuracies are in the folder above.

## Architecture as used
GPT-NeoX, 5 layers, ReLU, no LayerNorm, no positional encoding, untied read-out. T/A: d_model 96, 6 heads of 16, MLP 128, vocabulary
16, context 128 (width retry: d_model 192, heads of 32, MLP 256). S: d_model 176, 11 heads of 16, MLP 223, vocabulary 37, context 256.
TRAIN learns every tensor; A has no HAND. PART: HAND frozen; layer-3 MLP re-initialised and trained (T: W_in [128, 96], W_out [96, 128]
+ biases, 24,800 parameters; S: W_in [223, 176], W_out [176, 223], inputs masked to typed-token blocks 16–22/23–29/30–36 and PB/K/OLD
106–112/113–119/120–126, outputs NORMAL/PROD/DIFF 148–154/155–161/162–168; zero-based). Files `<rung>_F<p>m<m>_<fmt>_<run>_<best|final>_s<seed>`
hold full state dicts (PART includes the frozen HAND, so the HAND weights are every PART file minus its layer-3 MLP; `trained_keys` in `manifest.json`), with per-net config, seed, steps, recorded
and re-scored exact, sha256.

## Prediction before training
Recorded before the runs: T PART ≥ 9/10 solved by 10k; T TRAIN ≥ 7/10 at m = 2 and ≥ 5/10 at m = 3, pivots the bottleneck; S PART
3/3 by 5k with cross-dimension transfer. Not confirmed: T PART solves 3/12 seed-runs, updates (not pivot choice) dominate the failures,
S PART solves 4/6 with categorical mapping holes, and T PART m2 → m3 transfer is 0/3 in each field.

## Hand-built reference
Layer 0 builds a position R = q/64 from uniform-BOS attention and ReLU hinges; layer 1 derives block/row/column and grammar gates;
layer 2 fetches operands with scores −24‖key − query‖²; layer 3 picks the pivot by coordinate match plus 120[nz] − 120[used] − 20r.
T arithmetic: a scalar discrete-log lookup g^(log k + log pb + log d⁻¹) (zero handled separately) onto (cos 2πt/p, −sin 2πt/p);
layer 4 subtracts by paired ReLUs on |⟨(e(old) + e(−t))/2, e(c/2)⟩| ("pizza" subtraction). S instead writes categorical candidates
ReLU(e_a + e_b − 1) → e_(ab mod p) or e_(a−b mod p), and a frozen typed router emits the right candidate or pivot-row copy.
Formula-nonzero size T 14 heads + 195/211 neurons, S 15 + 400/482 (F₅/F₇). Exact on all 12,000/98,784 m = 2 systems, exhaustive local
arithmetic, stratified m = 3 fixtures and greedy rollouts (m = 3 not exhausted).

## What training finds
Seeds with best held-out exact ≥ .99 (screens; initial → retry):

| Cell | PART T | PART S | TRAIN T | TRAIN S | TRAIN A |
|---|---|---|---|---|---|
| F₅m2 | 0/3 | 2/3 | 0/3 → 3/3 (60k) | 3/3; ten-seed final 10/10 | 0/3 |
| F₅m3 | 3/3 | not run | 0/3 → 0/3 | 3/3 | 0/3 → 0/3 |
| F₇m2 | 0/3 | 2/3 | 0/3 → 2/3 (60k) → 0/3 (width 2) | 3/3 | 0/3 → 0/3 |
| F₇m3 | 0/3 | run lost | 0/3 → 0/3 | 3/3 (stop 22.5k) | 0/3 → 0/3 |
Curves: F₅m2 S TRAIN first reaches all-seed test ≥ .5/.9/.99 at 7.5k/9.5k/10.5k (train ≥ .99 also at 10.5k: no separate memorisation
phase visible); its final seeds 3–9 first-any/all ≥ .99 at 7.5k/13.5k (stop 14.5k); F₇m3 S TRAIN 10.5k/18.5k (stop 22.5k); F₅m3 T PART
at 9.5k/18.5k/21.5k; F₅m2 T retry all-seed ≥ .99 at 46.5k. F₇m2 T width retry plateaus at .47–.50 (stop 51.5k); the cell is closed.
Circuit census: all 12 diagnosed T PART seed-runs: MLP3 patch recovery/drop 1/1 and MLP4 drop 0, one bypass family incl. failures;
4 solved S PART seed-runs keep the router, 2 failed seeds keep the wiring but mis-map categories (failed F₇m2 S PART emits 0/136 correct
NORMAL4 tokens despite near-perfect decoding). Solved TRAIN snapshots need all five MLPs, not HAND's sparse staging.

| Setup (best/final) | Heads / MLPs needed | Product R² | Residual r90 vs HAND | MLP3 in/out r90; attention JS | Exact B → F |
|---|---|---|---|---|---|
| F₅m3 T PART | 9 / 2 | .75–.78 | 4 vs 5 | 16–21 / 6–20; 0 | 1 → 1 |
| F₇m2 S PART | 10 / 3 | .98–1.0 | 19–20 vs 22 | 28–30 / 15–17; 0 | .56–1 → .56–1 |
| F₅m2 T TRAIN retry | 16–21 / 5 | .50–.52 | 16–22 vs 9 | 20–24 / 23–29; 0–.52 | 1 → 1 |
| F₅m2 S TRAIN | 14–19 / 5 | .88–.91 | 16–21 vs 17 | 28–33 / 38–42; 0–.58 | 1 → 1 |
Needed = individual mean ablation costs > .05 exact (256 T/A, 128 S systems). A 128-system zero-ablation audit of solved F₅m3 T PART
needs 14 heads + 151–175 neurons, no MLP4 neuron (not a joint minimum).

## Closed form
T PART: with n = pb·d⁻¹, t = k·n, f = n on the pivot row and old − t elsewhere (mod p), replace the whole MLP3 by m̂₃(x) = ι_R θ[g(x)],
keeping MLP4: g = (equal, f) at value slots, otherwise a prefix-derived grammar category; θ = mean native response in that category
(fit on 1,024 fresh train-partition systems), R = the 16 token-score coordinates + OC, OS, PC, PS, TB, EQ (22). On 1,024 fresh
held-matrix systems per snapshot all 24 best/final snapshots of the four T PART cells score 1024/1024 (bar ≥ .99 × native), with
484/506/616/638 scalars (F₅m2/F₅m3/F₇m2/F₇m3) vs 24,800. Solved F₅m3 native scores 1022–1024, so the surrogate is a decision
surrogate, and on failed cells a repair, not an explanation of their errors. Controls: zero MLP3 and shifted arithmetic 0/1024; zero
MLP4 keeps native accuracy.
S PART (candidate, not accepted): MLP3 → {I_a → NORMAL_(a·pb), N_a → PROD_(a·k), M_b → DIFF_(old − b)}, MLP4 kept: 1024/1024,
token agreement 1, in all 8 snapshots of the 4 solved seeds; zero MLP3/MLP4 0; the substitute-shuffle control is missing. TRAIN: a phase-conditioned additive HAND-variable MLP4 formula is rejected (2/18 numerical passes,
both on failed snapshots; 0 on solved ones).

## Weights excerpt and reading
F₅m3 T PART seed 1 best, MLP3, raw float32 weight slices (Y₀/Y₁ = residue 0/1 score rows):

| Neuron | Input weights (bias) | Output weights |
|---|---|---|
| 61 | TS .17479, TSEMI −.21531, TB −.24831 (−.02143) | Y_comma .14114 |
| 97 | KZ .32052, PZ .30478, OZ −.22918 (−.01176) | Y₀ −.30414, Y₁ .24864 |
θ[g = (nonpivot, 0), Y₀] (fitted category means, seeds 0/1/2): F₅m3 .0641, .0609, .0422; F₇m2 .0351, .0409, .0396.
Neuron 61 favours comma over semicolon/value contexts; neuron 97 mixes zero-operand flags with old-state information and writes residue
scores directly. Neither is an isolated arithmetic gate: where HAND writes product-circle coordinates for MLP4's paired subtraction, the
trained stage writes answer scores itself, and deleting MLP4 costs nothing (the causal bypass evidence).

## Comparison
S PART keeps HAND's factorisation; T PART bypasses it; TRAIN uses distributed typed-workspace transport, not a recovered HAND program.
S beats answer-only in the completed screens, but its larger width, longer output and one smaller batch prevent a format-only causal
conclusion. HAND's subtraction is the "pizza" construction [Zhong et al.]; written microstates follow scratchpad computation [Nye et al.].

## Controls and limits
Teacher-forced only. The T surrogate is a reachable-input decision surrogate, not continuous-response equality. Not established:
a TRAIN arithmetic formula, an S PART formula, basin identity, coefficient-holdout transfer, length generalisation (T PART m2 → m3 0/3
per field). Lost: initial F₅m2 A/T TRAIN nets, F₇m3 S PART. Pending: diagnostics of the final seeds 3–9 and the width retry, public code.

## References
[Zhong et al., The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks](https://arxiv.org/abs/2306.17844) — research preprint; modular-arithmetic circuit comparison.
[Nye et al., Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/abs/2112.00114) — research preprint; scratchpad computation.
