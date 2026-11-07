# Polynomial differentiation: a frozen fetch plus one trained MLP learns the low-bit multiplier on every seed; from scratch the bottleneck is fetching a_k by position, not arithmetic

**Task.** `BOS a_0 … a_d =` → the derivative of Σ a_k x^k, ascending k, zero terms kept; one token per integer. Answer-only A emits
`(k·a_k, k−1)` per term; scratchpad T emits `(k, a_k, k·a_k, k−1)`. Training: degree 3, coefficients −9…9, the input pairs
(k, a) = (2, ±3), (3, ±2) withheld (their product labels still occur elsewhere); complete-input hash split, 103,251 train / 1,078 test
polynomials; fresh draws every batch. Exact = whole teacher-forced output correct.
**Status.** 2026-11-07. HAND A/T; PART 3 seeds per format (screening) and a ten-seed final per format (10/10 each); TRAIN screening
3 seeds per cell (width 1 30k, width 2 30k, width 2 60k); selected ten-seed TRAIN final T width 2, 60k: 6/10. 3 layers × 1 head.
Open: no ten-seed TRAIN A final (only the T format got a final cell); no closed form for TRAIN (PART accepted); no
training-dynamics run on the final cells; public reproduction code pending.

## Summary
With addressing frozen from the hand-built net, one masked MLP learns the local multiply k·a_k in 2–3k steps in every seed (PART 26/26
seed-runs), but never the multiplier bit it never saw (degree 4: .05). From scratch the same 3-layer NoPE shape needs 2× width and 60k
steps and still solves only 6/10 final seeds: every failed seed is exact on every token except the coefficient copy a_1 (s9 also a_2),
stuck near 1/3 — the missing variable is the position-indexed fetch, not the product. Solved TRAIN nets do not extrapolate (degree
1/2/4–6, withheld pairs, wide coefficients: 0–.016). A three-constant replacement of the PART MLP, ΔS = a·(b_0 + 2b_1) = k·a,
is exact on all 40 final snapshots and passes its zero/shuffle controls (accepted).

## Setup
[METHODS.md](../METHODS.md) for shared conventions. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1,000, cosine, clip 1, batch 256
fresh draws per step; eval on 1,024 held-out polynomials per seed every 500 steps; checkpoints every 2,500. PART: fp32, 30k budget,
stop when all seeds exact. TRAIN: TF32 updates, fp32 evals; final cell `--width 2 --steps 60000 --min_steps 45000 --plateau 5000`,
seeds 3–12, plateau stop at 57.5k (14.72 M draws per seed). Solved = best held-out exact ≥ .99. GPU accelerators for training,
CPU fp32 for re-scoring. Public reproduction code for this entry is pending; the archived networks re-score to the recorded accuracies.

## Architecture as used
GPT-NeoX, 3 layers, 1 head per layer, d_model = d_head 16, MLP 48 ReLU; TRAIN width 2: d_model 32, MLP 96. Vocabulary 183 (token =
integer + 91), context 64, untied embeddings, no LayerNorm, no positional encoding. PART freezes everything except layer-2 MLP:
W_in [48, 16], b_in [48], W_out [16, 48], b_out [16], masked to input columns AA = 12 (fetched coefficient) and bits 9–11 (multiplier
bits of k) and to output row S = 13 (answer value); 289 free scalars. TRAIN learns all tensors.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-differentiation](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-differentiation):
`hand_{A,T}`, `part_{A,T}_{best,final}_s0…9` (the 4 MLP tensors; insert into the matching HAND), `train_T_{best,final}_s3…12` (full
state dicts, width 2), `test_*.safetensors` (held-out rows per seed), `config.json`, `manifest.json` (seed, steps, recorded and fp32
re-scored exact, sha256). 60 nets; every one re-scores to its recorded exact within one row.

## Prediction before training
Recorded before the runs: PART 3/3 per format within 5k, old terms kept under extension, failure on the unseen multiplier bit —
**confirmed** (2–3k steps; degree 4 .05). TRAIN ≥ 2/3 per format by 30k — **refuted** (width 1: A 0/3, T 1/3); it needed width 2 and
60k (A 2/3, T 2/3 screening; T final 6/10).

## Hand-built reference
Layer 0: uniform BOS attention writes P = 1/(r+1); reciprocal ReLU hinges recover r/64 and r²/4096. Layer 1: attention anchors `=`;
paired-ReLU steps in relative output position write the term index, phase, the multiplier bits b_j of k and the exponent label.
Layer 2: fetch score `24[2(k+1)t − t²]` peaks at the coefficient token k+1 and writes AA = a_k; the MLP adds Σ_j 2^j g(AA, b_j) with
g(a, b) = ReLU(a + 32b − 32) − ReLU(−a + 32b − 32) = a·b for binary b, |a| ≤ 15, so S = k·a_k. Readout: integer logits 30(2vS − v²).
Size 3 heads + 48 neurons (A) / 82 (T). Exact on 34,320 polynomials per format (exhaustive degree 1–2, stratified 3–6) plus 217 primitive
products; minimum margin 30.0 (A) / 29.9998 (T). Degree 6 not enumerated exhaustively.

## What training finds
| Cell (seeds) | Solved | Best → final held-out exact | ≥ .99 step (k) |
|---|---|---|---|
| PART A / T screening (0–2) | 3/3 each | 1 → 1 | 2–3 |
| PART A / T final (0–9) | 10/10 each | 1 → 1 | 2–3 |
| TRAIN width 1, 30k, A / T (0–2) | 0/3 / 1/3 | A ≤ .09; T .007 / .998 / .073 | T s1 26 |
| TRAIN width 2, 30k, A / T (0–2) | 1/3 / 0/3 | A .29 / .89 / 1; T .19 / .83 / .22 | A s2 14.5 |
| TRAIN width 2, 60k, A / T (0–2) | 2/3 / 2/3 | A .56 / 1 / 1; T .34 / 1 / 1 | A 19–22; T 21–27 |
| **TRAIN width 2, 60k, T final (3–12)** | **6/10** | solved 1 → 1; s3 / s4 / s9 / s11 .34 / .34 / .12 / .36 → .33 / .33 / .11 / .35 | 16.5–49 |

Fresh data every batch: train = test, no memorisation step. Scratchpad vs answer-only: no uniform win (width 1: T 1/3 vs A 0/3;
width 2 30k: 0/3 vs 1/3; width 2 60k: 2/3 each); A has no ten-seed final, so the final cell does not compare formats.
Circuit types: PART — one type in all seeds: frozen fetch + learned local low-bit writer. TRAIN — solved nets carry k and the exponent
linearly (probe R² > .9 at layer 1) but a_k and the product weakly (< .5): a categorical, higher-rank code (hand-built CKA ≤ .06), not
the hand-built scalar one; no common local formula across seeds. Failed TRAIN seeds (final s3, s4, s11, screening width-2 60k T s0) are
wrong only at token 1, the copy of a_1 (token accuracy .32–.35, all other 11 tokens 1.0); s9 also at token 5 (copy of a_2, .45).

**Diagnostics** (R² = best linear-probe R² over layers; rank = layer-3 residual 99 %-variance rank; JS = layer-2 attention vs the
hand-built fetch head; used = mean-ablation drop > .05; hand-built and PART frames survive mean ablation, so only head 2 + MLP2 show).
| Cell | Used heads + MLPs | R² k / a / product / multiplier | Rank | JS | Best → final |
|---|---|---|---|---|---|
| HAND A / T | 1 + 1 | 1 / 1 / 1 / 1 (A); 1 / 1 / .87 / .999 (T) | 3 | 0 | 1 → 1 |
| PART A, T (13 seeds each) | 1 + 1 | = HAND to 3 decimals | 3 | 0 | 1 → 1 |
| TRAIN width 1 T (0–2) | 3 + 3 | .27–.90 / .10–.25 / .07–.17 / .31–.37 | 3–4 | .45–.66 | table above |
| TRAIN width 2 60k A (0–2) | 3 + 3 | .94–.98 / .13–.23 / .14–.20 / .85–.93 | 2–11 | .29–.47 | .56 / 1 / 1 → same |
| TRAIN width 2 60k T (0–2) | 3 + 3 | .94–.96 / .24–.37 / .18–.29 / .46–.58 | 6–9 | .52–.60 | .34 / 1 / 1 → .32 / 1 / 1 |
| TRAIN final T solved (5, 6, 7, 8, 10, 12) | 3 + 3 | .91–.97 / .23–.49 / .16–.43 / .41–.59 | 2–12 | .53–.62 | 1 → 1 |
| TRAIN final T failed (3, 4, 9, 11) | 3 + 3 | .89–.95 / .20–.32 / .17–.25 / .51–.62 | 5–13 | .54–.59 | table above |
Failures explained: answer interchange at layer 3 stays .81–.89 (DAS ≈ interchange) and activation patching needs all six components,
so the product/exponent path works in failed seeds; the missing variable is the a_k fetch at the copy slot. Edge attribution
(EAP-IG) check Spearman .49–.99; the .49 seed (s8) is not trusted.

## Closed form
**PART: accepted.** Replace the whole MLP2 write by ΔS = c + u·g(AA, b_0) + v·g(AA, b_1) (the frozen frame kept), with (c, u, v)
fitted to the MLP's outputs (512 training-hash draws), never to labels; fit R² .998–.9996. Controls, on the same 1,024 disjoint held-out
draws for every A/T × best/final × seed 0–9 snapshot (exact; agreement = fraction of answer tokens equal to the network's):
| Variant (40/40 snapshots, min = max) | Exact | Agreement |
|---|---:|---:|
| network | 1.000 | 1 |
| fitted closed form | 1.000 | 1 |
| zero write / mean write | 0 / 0 | .529 / .529 |
| closed form, (AA, b_0, b_1) jointly permuted across rows | .002 | .532 |
| closed form, u and v sign-flipped | 0 | .529 |
| (c, u, v) from another seed / hand-built (0, 1, 2) | 1.000 / 1.000 | 1 / 1 |
Half the answer tokens are written by the frozen frame, so agreement on the MLP-dependent tokens alone is ≈ .06 (zero, flip: correct
only where a = 0). The bar (substituted ≥ .99 × network) holds 40/40 while zero and shuffle collapse; constants are interchangeable
across seeds and equal to the hand-built ones, so the learned MLP computes ΔS = a·(b_0 + 2b_1) = k·a for k ≤ 3 (degree-3 inputs only;
the unseen bit b_2 is absent). TRAIN: a categorical codebook writer fed with the
true hand-built variables replaces MLP2 exactly in two screening T seeds (s1, s2) but fails all other solved snapshots (.73–.98); it
uses oracle variables and is not a closed form of the network. The selected final TRAIN nets have no substitution test.

## Weights excerpt and reading
Hand-built gate rows (W_AA, W_bit, bias) = (±1, 32, −32), outgoing S weights ±2^j. Trained PART A best s0, layer-2 MLP
(`gpt_neox.layers.2.mlp.*`, from `part_A_best_s0.safetensors`; incoming columns [AA, b_0, b_1, b_2] = [12, 9, 10, 11], outgoing row 13,
rounded to 4 decimals; output bias −.1837; all non-zero weights lie in these columns / this row):
| Neuron | W_in [AA, b_0, b_1, b_2] | b_in | W_out[13] |
|---:|---|---:|---:|
| 0 | −.0638, .1919, 1.1357, .0050 | −.5618 | −1.2408 |
| 1 | .0525, .0686, .9442, .0030 | −.5404 | 1.1179 |
| 37 | .1427, .4706, 1.1447, −.0043 | .1380 | .8728 |
Reading: rows are overlapping signed hinges, not single recovered gates. The b_1 column is large (max |W| 1.23), the b_2 column tiny
(max .056): b_2 is set only for k ≥ 4, never seen in training, so the net has no 4a term and degree-4 terms fail. Fitted PART constants
(c, u, v) for the final snapshots, A: s0 (−.0071, 1.0019, 2.0057) … s9 (.0096, 1.0037, 2.0116); T: s0 (−.0144, .9991, 2.0105) … s9
(.0001, 1.0047, 2.0035); all 20 final fits lie within .045 of (0, 1, 2), i.e. the trained MLP computes a·b_0 + 2a·b_1 = k·a for k ≤ 3
(least-squares fit of the MLP output on g(AA, b_0), g(AA, b_1), recomputable from the published weights and the hand-built frame).

## Comparison
PART reproduces the hand-built function in the hand-built basis (JS 0, rank 3, probes equal to HAND) and fails only the unseen bit.
TRAIN finds a different, higher-rank categorical code, and its hard part is addressing: NoPE must align the output index k with the
coefficient position, the bottleneck that position coupling targets in arithmetic transformers (Cho et al., 2024; not adopted here).

## Controls and limits
Length/OOD on the final cells (256 rows each): degree 1/2/4/5/6, withheld pairs and wide coefficients give 0–.016 exact on all ten TRAIN
seeds, solved ones included; PART withheld pairs 1.0 in 9/10 seeds per format, degree 4 .05, old terms exact. Scores are
teacher-forced; greedy rollout checked for PART and the solved screening T seeds only. No training-dynamics run on the final cells
(formation steps unknown). Repeated seed labels across cells are not independent replications.

## References
- Cho, H. et al. (2024). Position Coupling: Improving Length Generalization of
  Arithmetic Transformers Using Task Structure. arXiv preprint. https://doi.org/10.48550/arXiv.2405.20671 (research preprint)
