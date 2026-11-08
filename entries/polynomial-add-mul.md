# Integer polynomial addition and multiplication (convolution)

> - **Task.** Add or multiply two degree-3 integer polynomials, one token per signed coefficient, answer only or with a trace of partial results. Example (in words): adding the coefficient lists (1, 2, 0, 0) and (3, 0, 1, 0), a_0 first, gives (4, 2, 1, 0).
> - **Hand-built net.** Attention fetches a_k and b_k by position; addition is two ReLU ramps, S = relu(a + b) − relu(−a − b), and a read-out picks the integer closest to S/ONE. Multiplication forms each product from signed bit gates on b (two's complement).
> - **Trained nets.** Addition is learnt in all 40/40 final nets (frozen-part and from scratch, answer only and trace); the frozen-part stage is HAND's sum with a free logit scale. Multiplication is solved in no setup: the signed product is never built to unit resolution; even with the hand-built product gates frozen, 3/3 seeds reach only .998-.999, not exact.
> - **Verdict.** Partly solved (addition 40/40; multiplication 0/3 in the screenings, no final run). Closed form: partial (frozen-part addition: exact 1.0 in 40/40 nets; from scratch, trace rejected, answer-only decoder lossy).
> - **Links.** [Weights](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-add-mul) · Reproduce: not available yet · [Terms](../GLOSSARY.md)

## Task and status
**Task.** `BOS a_0..a_3 (+|*) b_0..b_3 =` then the answer; one token per signed integer (vocab 3156, |v| <= 1575). A: answer only (add c_k =
a_k + b_k; mul c_k = sum_j a_{k-j} b_j). T: trace (add rows `a_k b_k c_k ;`; mul partial polynomials P_j = P_{j-1} + b_j x^j A). Coefficients -9..9
without ±8 (held out); fresh pairs every batch, train/test split by input hash mod 97 (1000 test pairs per seed).

**Formats.**
- **A**: answer only
- **T**: trace (add: rows `a_k b_k c_k ;`; mul: partial polynomials P_j)

**Status.** 2026-11-08. Final run:
addition PART and TRAIN, A and T, 10 seeds each, 40/40 solved; multiplication 3-seed screenings and arms only (none solved, no final run).
## Summary
Addition is learnt in every seed. The PART stage (one MLP trained on frozen hand-built parts) is HAND's sum up to one logit scale per row type, (S, ONE) = κ·HAND: exact 1.0 in 40/40 PART nets.
TRAIN A: sparse-fetch 2/10, distributed-fetch 8/10; TRAIN T reads its own written trace (10/10; no accepted closed form). Multiplication fails in
every setup: the signed product is never built to unit resolution.
## Setup
[METHODS.md](../METHODS.md). AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000, cosine over 30k steps (60k arm), clip 1, batch 128 fresh pairs; eval
every 500 steps; best = highest held-out exact; early stop when all seeds are exact (TRAIN min 22.5k) or on a 5k plateau after 75%. PART fp32, TRAIN
TF32, re-scores fp32 CPU; 10 seeds per process on one consumer GPU. Extension sets of 256: degrees 0-2/4-6, ±8. Exact = whole teacher-forced answer.
**Reproduce:** training code for this task is not yet in `code/`; the weights, held-out splits and recorded accuracies are on Hugging Face (link above). Networks: huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-add-mul
## Architecture as used
GPT-NeoX, NoPE, no LayerNorm, ReLU, untied lm_head; d_model 96 (4 x 24; mul A 112, 14 x 8); layers A 3, T 4; MLP 128 (2x arms 256). PART: frozen
HAND except the stage MLP (A layer 2, T layer 3; W_in [128, d], b_in, W_out [d, 128], b_out; split-sum trains W_out, b_out only). Blocks: ONE 8,
flags 6-39, AA 41-47 (a), BB 48-82 (b or its bits), S 83. TRAIN: all from random init. [Nets](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-add-mul):
4 HAND + 55 nets (best snapshots; PART files hold the trained tensors), config JSON, manifest (seed, rung, accuracy, sha256), test pairs `test_<run>`.
## Prediction before training
Committed before any run (PREDICTION.md). PART add A 10/10 <= 5k, add T >= 9/10: held. TRAIN add A/T >= 8/10 <= 15k: count held, speed missed (best
12.5-26.5k). PART mul A/T >= 8/10: refuted (0/3). TRAIN mul A <= 3/10: held (0/3). TRAIN mul T >= 6/10 in 10-25k: refuted (0/3; 2/3 >= .99 only at
60k). TRAIN degree >= 4 < .50: held, degrees 0-2 fail too (not predicted). PART > TRAIN out of range: held.
## Hand-built reference
Reciprocal-position frame; quadratic index heads (score -β(r_key - target)²) fetch a_k, b_k (A) or shifted a, b_j and the previous coefficient (T).
Add: S = relu(a + b) - relu(-a - b) (T: ±a, ±b ramps per row). Mul: b_j in 5-bit two's complement, a·bit = relu(a + 32 bit - 32) - relu(-a + 32 bit
- 32), weights (1, 2, 4, 8, -16). Read-out logit(v) = 2vS - v²ONE, answer v* = S/ONE. Heads+neurons: add A 4+48, T 4+111; mul A 16+170, T 5+307.
Exact on 17,676 stratified sequences (degree 0-6, |coefficient| <= 15), min margin .9985.
## What training finds
Final run, 10 seeds per cell. H+N = heads and neurons whose mean ablation drops exact > .05; R² = best linear probe (A L3, T L4); rank = residual
r99 (net/HAND); JS = attention divergence to the matching HAND head; acc = exact best/final.
| cell | solved | best step | H+N | R² a / b / coefficient | rank | JS | acc |
|---|---|---|---|---|---|---|---|
| PART add A | 10/10 | 1-1.5k | 2 + 35-84 | 1.0 / 1.0 / 1.0 | 4/3 | 0 | 1.0/1.0 |
| PART add T | 10/10 | 1.5-2k | 2 + 86-102 | 1.0 / 1.0 / .97-.98 | 5-6/6 | 0 | 1.0/1.0 |
| TRAIN add A | 10/10 | 16.5-26.5k | 5-12 + 23-66 | .53-1.0 / .52-1.0 / .87-1.0 | 3-5/3 | 0-.69 | 1.0/1.0 |
| TRAIN add T | 10/10 | 12.5-23k | 2-14 + 26-72 | .75-.93 / .48-.60 / .97-1.0 | 7-10/6 | 0-.69 | 1.0/1.0 |
Types: PART A fixed-fetch sum 10/10; PART T input-dominant 10/10 (sum rows follow the operands: patch a .996-1.0, b .75-.95, trace 0-.06); TRAIN A
sparse-fetch 2/10 (5 needed heads) vs distributed 8/10 (8-12); TRAIN T trace-dominant 10/10 (written a_k, b_k .79-.87, operands 0). Screening (3 seeds,
best/final): PART mul A 0/0, T .004/.003 (product R² .76-.80); TRAIN mul A .002/.001, mul T .05-.17 (60k arm .993/.322/.996); width, running-sum and
split-gates arms 0/3; PART mul A split-sum (gates frozen) 3/3 at .998-.999. Length (best, seed range): PART A degree 4 .98-1.0, 5 .72-1.0, 6 .26-.93;
PART T 4 .98-1.0, 5 .90-1.0, 6 .62-.89; TRAIN A/T <= .13 at every degree but 3 (absolute routing); ±8: PART A 1.0, TRAIN <= .03.
## Closed form
PART add: (S, ONE) = κ_g·(S, ONE)_HAND, S_HAND = a + b (T: a, b or a + b by row), one κ per row group (A 2.2-4.9; T four groups 1.9-7.1), other read
dims a group constant. Substituted for the stage MLP (1000 held-out pairs): exact 1.0 in 40/40 nets (seeds 0-9, best and final); the bare HAND stage
also 40/40. The nets' |S/ONE - HAND S/ONE| is .03-.07 (A), .26-.32 (T), under .5. A per-dim linear fit in (a, b, 1) fails (.001-.057, 0/40): it misses
the ratio. TRAIN A: fitted MLP2 decoder .993-1.0 vs net .999-1.0 (seeds 0-2, lossy). TRAIN T: rejected (.05-.59 vs 1.0, seeds 0-2).
## Weights excerpt and reading
Weight slice, stage MLP: [a, b, ONE] | bias -> W_out[S], W_out[ONE]. HAND A: [1, 1, 0] | 0 -> 1 and [-1, -1, 0] | 0 -> -1. PART A best: 80-100% of
the top-20 S writers read a and b with equal weight, s0 n84 [.49, .50, .59] | .59 -> .321, .026; n83 [-.50, -.48, .61] | .59 -> -.289, .024: HAND's
±(a + b) ramps duplicated with offsets, each also writing ONE (this sets κ). PART T: 0% equal weight, a and b of opposite sign (s0 n111 [-.29, .37,
-.33] | -.33 -> -1.02, -.22), phase-gated rows. Mul split-sum s0 W_out[S] on HAND gates 0/1/8/9: .0765, -.0765, -1.2187, 1.2186 (ratio 15.93 ≈ 1 : -16).
## Comparison
PART = HAND's sum, free logit scale; TRAIN: degree-specific routing, T via its trace; mul fails as published (Madsen 2020; Trask 2018; Bai 2025).
## Controls and limits
κ fit on 512 disjoint pairs; off range the κ form scores 1.0 where nets fail (not a model of errors); no complement resample; mul: no final run.
## References
- [Madsen & Johansen 2020, Neural Arithmetic Units](https://doi.org/10.48550/arxiv.2001.05016) — preprint; exact learned multiplication needs structure.
- [Trask et al. 2018, Neural Arithmetic Logic Units](https://doi.org/10.48550/arxiv.1808.00508) — preprint; MLPs fit products only approximately.
- [Bai et al. 2025](https://doi.org/10.48550/arxiv.2510.00184) — preprint; multiplication needs long-range partial-product dependencies.
