# Polynomial differentiation by the power rule (deriv)

> - **Task.** Differentiate an integer polynomial of degree 3 term by term, one token per integer, answer only or with a trace. Example (answer only): `5 4 1 1 =` (coefficients a_0 to a_3) → `4 0 2 1 3 2`, the pairs (k·a_k, k−1).
> - **Hand-built net.** Uniform attention builds the position; quadratic-score attention fetches a_k; signed gates on the bits of k sum to k·a_k and an integer read-out picks the closest integer.
> - **Trained nets.** With addressing frozen, one masked MLP learns the low-bit multiplier on all 10 seeds in 2–3k steps, but never the unseen bit. From scratch the net needs width 2 and 60k steps and solves 6/10 (scratchpad format); failed seeds miss only the position-indexed copy of a_k, stuck near 1/3.
> - **Verdict.** Partly solved (frozen-part 10/10 per format; from scratch 6/10 in the selected final cell). Closed form: partial (frozen-part nets: MLP2 = k·a on the two observed bits, 1.000 on 40/40 snapshots; from scratch only a numerical compression, 4 of 6 solved seeds).
> - **Links.** [Weights](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-differentiation) · Reproduce: not available yet · [Terms](../GLOSSARY.md)

## Task and status
**Task.** Input `BOS a0 … a_d =`; output the derivative of Σ a_k x^k, ascending k, no zero trimming. Answer-only A emits `(k·a_k, k−1)` per term; scratchpad T emits `(k, a_k, k·a_k, k−1)`. One token per integer. HAND: degree 1–6, coefficients −15..15. Training: degree 3, −9..9; pairs (k,a)=(2,±3),(3,±2) withheld as inputs (their product labels still occur elsewhere).

**Formats.**
- **A**: answer only, (k·a_k, k−1) per term
- **T**: scratchpad, (k, a_k, k·a_k, k−1) per term

**Status 2026-11-08.** All cells finished and diagnosed (108/108 trained best/final snapshots + 2 HAND controls). PART A/T 10/10; selected TRAIN T (width 2) 6/10. PART closed form accepted (MLP2 = k·a on the two observed bits); TRAIN only a numerical substitute (4 of 6 solved final seeds).
## Summary
With addressing frozen from HAND (PART), one masked MLP learns the local low-bit multiplier in 2–3k steps on every seed. From scratch (TRAIN), the same 3-layer NoPE shape learns the multiply but stalls on **fetching a_k by position**: every failed TRAIN seed is exact on all tokens except the coefficient-copy slot(s), stuck near 1/3. Solved TRAIN nets split into several copy-routing types and do not extrapolate (degree 1/2/4–6, unseen pairs, wide coefficients: ≤ .06).
## Setup
Fresh hash-disjoint draws with replacement (104329 primary polynomials -> 103251 train / 1078 test). AdamW lr 1e-3, wd .1, warmup 1000, cosine, clip 1, bs 256; eval 1024 draws / 500 steps; ckpt every 2500. PART fp32, 30k budget; TRAIN TF32 updates, fp32 evals.
Selected final: `train.py --fmts T --kinds train --width 2 --steps 60000 --min_steps 45000 --plateau 5000 --tf32 1 --seeds 3 … 12` (launch_final_train.sh; 14.72M draws/seed, plateau stop 57.5k). PART final: launch_final_part.sh, seeds 0–9. Networks: huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-differentiation.
**Reproduce:** training code for this task is not yet in `code/`; the weights, held-out splits and recorded accuracies are on Hugging Face (link above).
## Architecture as used
GPT-NeoX shape, 3 layers × 1 head, d_model = d_head 16, ReLU MLP 48 (TRAIN width 2: 32/96), vocab 183, ctx 64, untied, no LayerNorm, no
positional encoding. PART freezes all but masked MLP2 (W_in 48×16 on input cols AA=12 / bits 9–11, output col S=13; 289 free scalars).
## Prediction before training (PREDICTION.md, commit 61df19ce)
PART 3/3 within 5k, keeps old terms under extension, fails the unseen multiplier bit: **confirmed**. TRAIN >= 2/3 per format by 30k:
**refuted** (A 0/3, T 1/3); it needed width 2 and 60k (screening A 2/3, T 2/3; T final 6/10).
## Hand-built reference
Uniform BOS attention gives P=1/(r+1); reciprocal hinges recover r/64, r²/4096; '=' anchor + relative-step thresholds write term/phase, multiplier bits and index; quadratic-score attention fetches a_k; six signed bit gates + integer readout give k·a_k. Size 3H+48N A / 3H+82N T. Exact on 34320 polynomials/format (exhaustive degree 1/2, stratified 3–6) + 217 primitive products; min margin 30.0 / 29.9998.
## What training finds
| Cell (seeds) | Solved | Best -> final IID exact | grok (>=.99), k steps |
|---|---|---|---|
| PART A / T final (0–9) | 10/10 each | 1 -> 1 | 2–3 |
| TRAIN w1 A / T (0–2), 30k | 0/3 / 1/3 | A <= .09; T .007/.998/.073 | T s1 26 |
| TRAIN w2 A / T (0–2), 30k | 1/3 / 0/3 | A .29/.89/1; T .17/.83/.22 | A s2 14.5 |
| TRAIN w2 A / T (0–2), 60k | 2/3 / 2/3 | A .56/1/1; T .34/1/1 | A 19–22; T 21–27 |
| **TRAIN w2 T final (3–12), 60k** | **6/10** | solved 1; s3/4/9/11 final .33/.33/.11/.35 | 16.5–49 |
Scratchpad vs answer-only: no uniform win (w1 T 1/3 vs A 0/3; w2-30k 0/3 vs 1/3; w2-60k 2/3 each). Fresh data: train = test, no memorisation.
Circuit types (donor-flip query patches on the copied a_k; products read the trace, k=2,3 trace flip .56–1 vs input ≤ .14). PART: one type, masked bit-gate multiplier, 10/10 per format. TRAIN T final (4 of 6 solved seeds analysed; best = final type): split copies (a_1 by H1, a_2/a_3 by H2) 1/4; first copy H2, later H1 1/4; H1 copies with H2 mixed or idle 2/4. Screening adds A early/late fetch and T late-copy types: no single TRAIN circuit.
## Diagnostics
R² = best linear-probe R² over layers; rank = residual L3 r99 (HAND 3); JS = L2H0 attention vs same-index HAND head. Used = mean-ablation drop > .05 (neurons not counted); HAND/PART frames survive mean ablation, so only H2+MLP2 show. TRAIN inter on `answer` at L3: flip .81–1.0, DAS ≈ inter.
| Cell | Used H+MLP | R² k / a / product / multiplier | rank | JS | best -> final |
|---|---|---|---|---|---|
| HAND A / T (formula 3H) | 1+1 | 1/1/1/1 (A); 1/1/.87/.999 (T) | 3 | 0 | 1 -> 1 |
| PART A, T (13 seeds each) | 1+1 | = HAND to 3 decimals | 3 | 0 | 1 -> 1 |
| TRAIN w1 T (0–2) | 3+3 | .27–.90 / .10–.25 / .07–.17 / .31–.37 | 3–4 | .45–.66 | table above |
| TRAIN w2-60k A (0–2) | 3+3 | .94–.98 / .13–.23 / .14–.20 / .85–.93 | 2–11 | .29–.47 | .56/1/1 -> .56/1/1 |
| TRAIN w2-60k T (0–2) | 3+3 | .94–.96 / .24–.37 / .18–.29 / .46–.58 | 6–9 | .52–.60 | .34/1/1 -> .32/1/1 |
| TRAIN w2 T final solved (5,6,7,8,10,12) | 3+3 | .91–.97 (L1) / .23–.49 / .16–.43 / .41–.59 | 2–12 | .53–.62 | 1 -> 1 all |
| TRAIN w2 T final failed (3,4,9,11) | 3+3 | .89–.95 (L1) / .20–.32 / .17–.25 / .51–.62 | 5–13 | .54–.59 | table above |
Failure location (final10 err/tok_acc): s3/s4/s11 wrong only at token 1 (copy of a_1, acc .32–.35), all 11 other tokens 1.0; s9 also at
token 5 (copy of a_2, .45). Same as w2-60k T s0. Answer interchange stays .81–.89 and patch needs all six components, so the product/exponent
path works; the missing variable is the a_k fetch at the copy slot (NoPE index-operand alignment, cf. Cho et al. 2024). Solved TRAIN nets carry
k/exponent linearly (R² > .9 at L1) but a and product weakly (< .5): a categorical, higher-rank code, not HAND's scalar one (hand_cka <= .06).
EAP-IG edge check spearman .49–.99 (s8 .49: edges not trusted).
## Closed form
**PART accepted:** MLP2's whole write -> ΔS = c + u·g(a,b0) + v·g(a,b1), g = ReLU(a+32b−32) − ReLU(−a+32b−32) = a·b, fit (c,u,v) ≈ (0,1,2). Same
1024 held-out draws, 40/40 snapshots: substituted 1.000 = net; zero/mean 0, (a,b0,b1) row-shuffle .002, u/v flip 0; seed-swapped or HAND (0,1,2) constants 1.000,
so ΔS = a·(b0+2b1) = k·a. Frame + 4 gates, 3 constants (3H+46N A / 3H+80N T); IID degree 3 only (bit b2 absent). **TRAIN (numerical only):**
every head A = causal_softmax((zQ_L)(zQ_R)ᵀ), ΔH = (A zO_L)O_R + b and every MLP ΔM = ReLU(zI_L I_R)O_L O_R + b (z = [x;1], low rank, fit on
internal activations): all six replaced together keep s7 .997/.998, s8 .997/1, s10 .9985/1, s12 1/1 (best/final, native 1; 2048 draws), with
8956–15908 of 31488 scalars; screening 12/12 ≥ .99 × native. A compression, not recovered power-rule algebra.
## Weights excerpt and reading (HAND)
Gate rows `(W_AA, W_bit, bias) = (±1, 32, −32)`, outgoing S weights `±2^j`; fetch score `24[2(k+1)t − t²]`; integer logits `30(2vS − v²)`.
Gate j passes ±a only when bit j of k is set; S sums a·2^j = k·a; the readout picks the integer closest to S. PART final (MLP2 rows with the 4 largest |W_out[S]|, all 20 nets): W_AA ±.06–.11, W_b0/W_b1 .0–1.16, W_b2 ≈ 0 (|.04| max), bias −.46…−.96, out S ±1.10–1.48, sign(W_AA) = sign(out S): the same ±a·bit gates spread over many soft rows (small a-slope, bias cancels the bit weight), no row reads the unseen bit b2.
## Comparison
PART reproduces HAND's function in HAND's basis (JS 0, rank 3, probes = HAND) and fails only the multiplier bit it never saw (degree 4 .05, 5/6 0; old terms exact). TRAIN finds a different, higher-rank code with seed-dependent copy routing; its hard part is addressing, not arithmetic.
## Controls and limits
Length/OOD (final10 TRAIN, 256 rows): degree 1/2/4/5/6, unseen pairs, wide = 0–.016 exact at final (best ≤ .062) on all ten seeds, solved ones included; PART: degree 1/2 1.0, unseen pairs 1.0 in 9/10, wide 0. T scores are teacher-forced; greedy rollout checked for PART and solved screening T only. No checkpoint dynamics on final10.
TRAIN circuit types and the substitute cover 4 of 6 solved final seeds (two jobs never started: compute queue). Repeated seed labels across cells are not independent replications.
## References
Cho et al. 2024, Position coupling, https://doi.org/10.48550/arXiv.2405.20671 (preprint): index-operand alignment as the arithmetic bottleneck.
