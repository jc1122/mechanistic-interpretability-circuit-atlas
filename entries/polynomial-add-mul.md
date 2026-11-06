# Integer polynomial addition and multiplication: addition is learned in every seed; multiplication fails unless the hand-built product gates stay frozen

**Task.** Prompt `BOS a_0..a_3 (+|*) b_0..b_3 =`, one token per signed integer (vocabulary 3156: integers −1575…1575 and `+ * = ;`),
coefficients in ascending powers, degree 3, values −9…9 without ±8 (±8 held out by value). Addition: c_k = a_k + b_k (4 answer
tokens). Multiplication (convolution): c_k = Σ_j a_{k−j} b_j (7 answer tokens). Format **A** (answer only) writes c. Format **T**
(scratchpad): addition writes rows `a_k b_k c_k ;`; multiplication writes the partial products P_j = P_{j−1} + b_j x^j·a, one row per j,
the last row being the answer (31 tokens). Split: input hash mod 97 (one residue class = test); fresh training pairs every batch;
1000 held-out pairs per seed. Exact = whole teacher-forced output.
**Status.** 2026-11-06. HAND exact (add/mul, A/T). Final cells (10 seeds): PART add A/T, TRAIN add A/T. Three-seed cells: PART mul
A/T (whole stage, 2x width, split-sum, split-prod), TRAIN mul A/T (30k, 2x width, running-sum auxiliary loss, T at 60k). A third
product-line scratchpad format (S) is unfinished and not reported here.

## Summary
Addition is easy at every rung: training only the last arithmetic MLP on frozen hand-built parts (PART) solves A and T in 10/10 seeds
within 1–2k steps, and training from scratch (TRAIN) solves both in 10/10 seeds (best snapshot at 12.5k–26.5k steps). Multiplication is
not learned by gradient descent in these nets: PART on the whole stage MLP fails 0/3 in A and T (also at 2x width), TRAIN A fails 0/3
(≤ .003), TRAIN T fails 0/3 at 30k and reaches 2/3 seeds ≥ .99 (none exact) only at 60k. When the hand-built signed-bit product gates
are kept frozen and only the read-out is trained (split-sum), A reaches .999 in 3/3 seeds, and a 123-number read-out formula replaces
the trained layer with no loss in all 12 tested snapshots. No trained addition net transfers to other degrees.

## Setup
[METHODS.md](../METHODS.md) for the shared conventions. AdamW lr 1e-3, wd .1, warmup 1000, cosine over the run (30k steps; 60k for
TRAIN mul T), clip 1, batch 128 fresh pairs per step, eval every 500 steps on the 1000 held-out pairs; best snapshot = highest held-out
exact; stop when all seeds are exact or on a 5k plateau after 75% of the steps. PART fp32, TRAIN TF32 updates; all re-scores fp32 on CPU.
All seeds of a cell in one process on consumer GPUs. Extension sets (256 rows): degrees 0–2 and 4–6, the held-out value ±8, wider
coefficient ranges.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-add-mul](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-add-mul)
(55 nets + 4 HAND: PART add A/T final s0–9, PART mul A/T split-sum s0–2, TRAIN add A/T final s0–9, TRAIN mul A/T 30k s0–2,
TRAIN mul T 60k s0–2; best snapshots; each run's held-out pairs in `test_<run>`). Reloaded in fp32, all 55 reproduce their recorded
held-out exact (TF32 nets within 3/1000; HAND 1.0).
**Reproduce:** the generator, HAND and trainer for this task are not yet in `code/`; weights, test pairs and accuracies are in the folder above.

## Architecture as used
GPT-NeoX, ReLU, no LayerNorm, no positional encoding (causal mask only), untied read-out, vocabulary 3156. d_model 96 with 4 heads of
24, MLP 128 in every layer, except mul A: d_model 112 with 14 heads of 8 (one head pair per product
channel). **A: 3 layers; T: 4 layers.** TRAIN learns every tensor of its cell's shape. PART: HAND frozen; the stage MLP (A layer 2,
T layer 3: `dense_h_to_4h` [128, d_model] + bias, `dense_4h_to_h` [d_model, 128] + bias) re-initialised and trained; **split-sum**
trains only `dense_4h_to_h` + bias and keeps the hand-built product gates. Files: `hand_{add,mul}_{A,T}`, `part_*` (only the trained
tensors, listed in the manifest's `trained_keys`; insert into the matching HAND), `train_*` (full state dicts), `config.json` (token map,
HF configs, optimiser) and `manifest.json` (seed, best step, recorded and re-scored held-out exact, final exact, sha256).

## Prediction before training
Recorded before any run. PART add A 10/10 within 5k: **held** (1–1.5k). PART add T ≥ 9/10: **held** (10/10). PART mul A/T ≥ 8/10
within 20k: **broken** (0/3 each). TRAIN add A/T ≥ 8/10: **held** (10/10 each), timing optimistic. TRAIN mul A ≤ 3/10: held (0/3).
TRAIN mul T ≥ 6/10 within 10–25k: **broken** (0/3 at 30k). TRAIN exact < .5 at degree ≥ 4: held; lower degrees fail too (not
predicted). The split-sum arm was designed after the failures to isolate the product step, with its own recorded prediction.

## Hand-built reference
A reciprocal-position frame and quadratic index heads fetch the coefficients each output needs (A: a_k and b_k, or all pairs a_{k−j},
b_j; T: the shifted a, b_j and the previous partial coefficient). Addition: signed ReLU pairs. Multiplication: b_j is written as 5-bit
two's complement with weights (1, 2, 4, 8, −16), and each product is a·bit = ReLU(a + 32·bit − 32) − ReLU(−a + 32·bit − 32), summed with
those weights; A adds the parallel products, T adds one product row to the previous partial polynomial. The numeric read-out logit is
2vS − v²·ONE (S = value scalar, ONE = constant channel), maximal at v = S. Exact on degrees 0–6 and coefficients ±15: exhaustive
primitives plus 17,676 stratified A/T sequences, minimum logit margin .9985. Formula sizes (nonzero heads + neurons, frame included):
add A 4 + 48, add T 4 + 111, mul A 16 + 170, mul T 5 + 307; allocations, not measured minima.

## What training finds
| Cell | Solved / seeds | Best / final exact | Best step (k) | Diagnostics (seeds 0–2) |
|---|---|---|---|---|
| PART add A final | 10/10 | 1.0 / 1.0 | 1–1.5 | 2 heads + stage MLP needed; a, b, c R² 1.0 (L3); r99 4 vs HAND 3 |
| PART add T final | 10/10 | 1.0 / 1.0 | 1.5–2 | 2 heads; a, b 1.0 (L3), c .97–.98 (L4); r99 5–6 vs 6 |
| PART mul A whole stage | 0/3 | 0 / 0 | – | products R² .19–.26, c .93–.96 (L3); stage MLP patch drop 1.0 |
| PART mul T whole stage | 0/3 | ≤ .004 / ≤ .003 | – | product .76–.80, c .91–.92 (L4); previous partial, bits ≥ .99 |
| PART mul A split-sum | 3/3 (≥ .99) | .999 / .998–.999 | 28.5–29.5 | 8 heads; c R² 1.0 (L3); r99 9 vs 6 |
| PART mul T split-sum | 0/3 | .63–.69 / .59–.66 | – | product .9999, c 1.0 (L4); fails on unit precision of c |
| TRAIN add A final | 10/10 | 1.0 / 1.0 | 16.5–26.5 | 5–12 heads; a .53–1.0, b .57–1.0 (L2/3), c .87–1.0 (L3); JS 0–.55 |
| TRAIN add T final | 10/10 | 1.0 / 1.0 | 12.5–23 | 3–7 heads; c .99 (L4), row phase .999 (L3); JS .04–.58 |
| TRAIN mul A 30k | 0/3 | .002 / ≤ .001 | – | a, b, products not decodable (R² ≤ .03); r99 32–52 vs 6 |
| TRAIN mul T 30k | 0/3 | .05–.17 / same | – | first product row exact; later partials (U R² .24–.61) fail |
| TRAIN mul T 60k | 2/3 (≥ .99) | .993, .322, .996 / .991, .318, .995 | 57–59.5 | 4–7 heads; product .27–.59, c .46–.74 (L4) |
Retries that did not change the outcome (3 seeds each): PART mul 2x width (A 0, T .05–.09), PART mul split-prod (gates trained,
read-out frozen: 0/3 both), TRAIN mul A 2x width (≤ .002), TRAIN mul with a running-sum auxiliary probe loss (A ≤ .002, T .02–.37).
- **Learning curve.** PART addition is exact at 1–2k steps. TRAIN add A seeds 0/1 first reach .99 at 16k/19k and
  exact at 17.5k/23k, with no memorisation-first phase (training data are fresh every step).
- **Circuit types.** PART addition reads the frozen operand fetches in 10/10 A and 10/10 T seeds (token-ablation census). TRAIN add A
  clusters into a sparse fetch (2/10) and a distributed fetch (8/10); TRAIN add T reads its own written trace rows in 10/10 seeds rather
  than the original operands, unlike HAND. These are ablation families, not separate basins.
- **Why multiplication fails.** With intact frozen fetches and bits (R² ≥ .99), the trained PART stage builds only approximate products
  (T R² ≤ .80) and an uncalibrated output; TRAIN T computes the first product row exactly but loses the accumulation over later rows.
  Freezing the product gates (split-sum) removes the A failure entirely; what remains in T is unit precision of the accumulated
  coefficient, not a missing variable. The step gradient descent does not find is the signed product of two scalars at unit resolution.

## Closed form
- **PART mul split-sum read-out (A and T, best and final, seeds 0–2).** From the frozen fetched a, bits β and (T) previous partial U, form
  g± = ReLU(±a + 32β − 32); A uses 40 signed bit gates + 1, T uses 10 gates, ReLU(±U) and 1. Replace the whole stage by
  Δ(S, ONE, SEP) = φ·C (other writes zero), fitted by least squares on 512 sequences: **123 (A) / 39 (T) numbers** vs 14,448 / 12,384
  trained read-out parameters. On 4096 fresh held-out sequences the substituted net equals the original in **12/12** snapshots
  (A .992–.998, T .852–.874; minimum ratio 1.000), confirmed by an independent re-fit on fresh seeds. Faithful to the trained read-out;
  T remains unsolved.
- **TRAIN add A, last MLP (best and final, seeds 0–2), lossy.** Decode z = (a/9, b/9, current token/18) and replace the final MLP by
  Δx = W_out ReLU(Θ Φ(z)) + b_out, Φ = 15 terms (1, z_i, z_i z_j, |z_i|, |z_1 + z_2|, |z_1 − z_2|); 8,456 fitted numbers. Keeps ≥ 99% of
  the original exact in 6/6 snapshots on 8192 rows (original 99.88–100%, substituted 99.24–100%); a first 1024-row audit failed once.
  A response abstraction, not an exact decompilation; the same fit fails for TRAIN add T and the auxiliary-loss mul nets (0/6 each).

## Weights excerpt and reading
PART add A, seed 1 (best), stage neuron 0 (weight slices of `part_add_A_final10_s1`, layer 2): W_in[a, b] = (−.142369, −.158027) on the
fetched-operand channels, bias .133309, W_out[S, ONE] = (−.241497, +.028603); its largest input weights (.213, .153) sit on the current
residual's bit channels. Equal-sign operand weights make a ramp on −(a + b); the positive ONE write rescales, so the read-out's argmax gives
round(S/ONE), not raw S. HAND (formula weights): addition uses signed ReLU pairs on a + b; a product neuron is ReLU(±a + 32·bit − 32), written into S with the
bit weight (1, 2, 4, 8, −16); split-sum keeps these gates and learns only the sum over them (formula above).

## Comparison
Same algorithm in the easy direction: PART addition reuses HAND's fetches exactly, and TRAIN addition keeps a, b and c decodable but
fetches them differently (T reads its own scratchpad). Multiplication separates the rungs: frozen hand-built product gates plus a trained
sum learn (A), trained gates on a frozen sum do not (split-prod 0/3), and from-scratch T needs 2x the budget for 2/3 non-exact seeds.
Structured arithmetic units ([Trask et al.](https://arxiv.org/abs/1808.00508), [Madsen & Johansen](https://arxiv.org/abs/2001.05016))
build multiplication into the architecture, which is what split-sum's frozen gates do here; [Bai et al.](https://arxiv.org/abs/2510.00184)
report that standard training of multiplication misses long-range partial-product dependencies, and the running-sum auxiliary loss
they motivate did not rescue it here. Scratchpad vs answer only: both solve addition;
for multiplication T trains further than A (2/3 at 60k vs 0/3) but is one layer deeper, so it is not a size-matched comparison.

## Controls and limits
Length: PART addition best snapshots at degree 6 score A .26–.93, T .62–.89; TRAIN addition (screening seeds) scores ≤ .12 at degrees
0–2 and 4–6, and TRAIN add A best seeds 0/1 score 0% on outer values 10–15. Sizes are mean-ablation head counts; no individual-neuron
ablations were run, so used-neuron counts are not given. Probe R², interchange with no flipped answers and ties on uniform heads were
not taken as use or absence. Teacher-forced exact; no greedy or exhaustive check of trained nets. Pending: diagnostics of the final
cells' seeds 3–9, the product-line scratchpad format (S), public code.

## References
- Trask et al. (2018). Neural Arithmetic Logic Units (preprint). https://arxiv.org/abs/1808.00508
- Madsen & Johansen (2020). Neural Arithmetic Units (preprint). https://arxiv.org/abs/2001.05016
- Bai et al. (2025). Why Can't Transformers Learn Multiplication? Reverse-Engineering Reveals Long-Range Dependency Pitfalls (preprint). https://arxiv.org/abs/2510.00184
