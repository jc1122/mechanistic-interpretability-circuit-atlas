# Explicit Euler for y' = −ky in fixed point (euler-decay-ode) — unsolved (negative result)
**Task.** `BOS C<c> N<T> Y<y₀>` → scratchpad **T** `Y<y₁> … Y<y_T> R<y_T> EOS` or answer-only **A** `R<y_T> EOS`; y_{t+1} = ⌊(8 − c)y_t/8⌋, c 0…7,
T 1…6, y₀ 0…63 (3,072; r2 TRAIN 0…319), one token per value; (y₀, c) pair-hash split, all T together: 2,592 / 480 (r2 12,390 / 2,970).
**Status.** HAND T/A exact. r1 (3 seeds × {A, T} × {PART, TRAIN}, TRAIN wd .1/.3/1), r2 retry (A PART, A/T TRAIN): 0/3 everywhere; final T PART 0/10.
## Summary
Unsolved. Best: scratchpad PART keeps HAND's 5-edge route in 10/10 seeds but misses 1-5 held-out first steps, mostly at c = 5 (final
.963-.988). From scratch both formats memorise per (y, c) state: T decodes y but does not use it (held-out c = 0 exact 0); A never composes T steps.
## Setup
[METHODS.md](../METHODS.md). Teacher-forced exact of every target token. AdamW 1e-3, betas (.9, .98), warmup 1000, cosine, clip 1, batch 128 fresh,
eval 500. PART 20k fp32 wd .1 (r2 A: wd .03/.1, MLP 896); TRAIN 40k (min 30k, plateau 5k) TF32 wd .1/.3/1; evals fp32; one GPU per pack. Reproduce: task trainer `--kinds part --steps 20000 --wd 0.1 --tf32 0 --mask_pos 1`, `--kinds train --steps 40000 --wd 0.1
0.3 1.0` (r2 `--ymax 319`; `--part_width 896`); keys `best_test`, `final_test.exact`. Networks: huggingface.co/jc1122/circuit-atlas-nets/tree/main/euler-decay-ode.
## Architecture as used
GPT-NeoX, 7 layers, d_model 64, 2 heads × 32, ReLU MLP 448, NoPE, no LayerNorm, untied; vocab 146 (r2 TRAIN 658). PART: HAND frozen, layer-1
MLP re-initialised (W_in [448, 64] masked to YT, CV_c, ON₀; b_in [448]; W_out [64, 448]; b_out [64]). TRAIN: all weights. Archive `hand_{T,A}`,
`part_{T,A}_r1_{best,final}_s0-2`, `part_T_final_{best,final}_s0-9` (4 tensors), `train_{T,A}_r1_*`, `train_T_r2_*` (s0-2) + config/manifest (sha256).
## Prediction before training
Before r1: T PART unsolved, .80-.95 (unsolved held; range missed: .963-.988 in r1 and final); A PART train 1.0 (.37-.43); T TRAIN unsolved unless z = (8 − c)y/8 forms (held,
no z); A TRAIN memorises (held). r2: T TRAIN .60-.90, c = 0 as a rule (wrong: .41-.57, c = 0 exact 0); A PART up with width (wrong).
## Hand-built reference
L0H0 fetches c (one-hot CV_c) and T; L0H1 counts Y tokens; L0 MLP: ON_l = [T − lines ≥ l]. MLP l+1 = Euler step: YT −= ⌈c·YT/8⌉ =
Σ_{τ∈THR_c}[YT ≥ τ], 224 pairs relu(2x − 2τ + 1.5) − relu(2x − 2τ + .5) gated by CV_c, ON_l (G 1000). Readout Y<v> = 2v·Y1 − v²·ONE,
R<v> = 2v·YT + (1 − v²)·ONE. Exact on all 3,072 inputs, both formats (A = T minus the Y rows).
## What training finds
| cell | solved | best test | final test | train | stop | T = 1 / T = 6 (final) | failure |
|---|---|---|---|---|---|---|---|
| T PART final (r1) | 0/10 (0/3) | 1.0 in 9/10 (test-selected) | .963-.988 (.963-.988) | 1.0 (s5 .994, s7 .996) | 20k budget | .96-.99, flat in T | step-1 thresholds, c = 5 |
| A PART r1 / r2 | 0/3, 0/6 | .44-.50 / .32-.37 | .41-.50 / .11-.35 | .37-.43 / .12-.36 | budget / plateau | .16-.20 / .54-.71 | can't fit: YT inexact after step 1 |
| T TRAIN r1 / r2 | 0/9, 0/9 | .53-.74 / .41-.57 | .51-.72 / .41-.56 | 1.0 (wd 1: .79-.98) | plateau 35-38k | flat in T | memorises states; c = 0 exact <= .09 |
| A TRAIN r1 / r2 | 0/9, 0/9 | .74-.83 / .64-.77 | .73-.82 / .64-.77 | 1.0 (wd 1: .78-.99) | plateau/budget | .60-.70 / .81-.89 | no T-fold composition |
No grok. One type per cell: T PART = HAND-T route 10/10 (discover = HAND-T's 5 edges, faith 1.0 in 9/10, s9 .98; L0H0 + MLP1); T TRAIN = dense
MLP chain (55-147 edges MLPk→MLPk+1, HAND-T jaccard ≤ .04); A TRAIN = HAND-A recall .47-.80, jaccard ≤ .32; A PART = 4-13 edges around L0H0.
## Diagnostics (wd .1, best snapshot; final within .03; R² = best layer; n90 = neurons for 90% of MLP mass)
| cell | used blocks; n90 | R² hand_YT / hand_Y1 / y_new / c | flip (nf > 0) | CKA vs HAND (L7) | attn JS | diag exact best / final |
|---|---|---|---|---|---|---|
| T PART final (10) | 2 of 21; 130-160 (HAND 448) | .87-.88 / .94 / .91-.96 / .98-.99 (L1) | Y1 .99-1.0 (L1) | .41-.65 | .28 (frozen) | .99-1.0 / .96-.99 |
| A PART r1 | 1-7; 56-63 | .91-.95 / .94-.96 / — / .96-.98 | YT .36-.51 (L1) | .14-.19 | .20 (frozen) | .57-.72 / .54-.70 |
| T TRAIN r1 | 16; 1418-1507 | .85-.89 / .91-.95 / .98-.99 / .95-.98 | y_new .69-.85 (L6-7) | .05-.07 | .33-.35 | .61-.70 / .55-.69 |
| T TRAIN r2 | 14-16; 1348-1450 | .32-.73 / .55-.78 / .96-.99 / .97 | y_new ≤ .07 | .05-.10 | .31-.32 | .49-.57 / .48-.56 |
| A TRAIN r1 | 17-20; 1044-1159 | .75-.84 / .82-.90 / — / .99-1.0 | YT .31-.50 (L1-2) | .23-.32 | .28-.30 | .87-.90 / .82-.91 |
r90 (L7) 1-2 (r2 T TRAIN 8-13); c/y_prev/dec untested (nf 0). T TRAIN holds y_new linearly (dict corr .93) but swapping it does not move the
output (flip ≤ .38, DAS .15): a state lookup. A TRAIN loses YT after L3 (R² ~.1). T PART: hand_Y1 swappable on wrong rows too (flip 1.0).
## Closed form
none (unsolved): no solved cell, so no substitution or complement check.
## Weights excerpt and reading (PART layer-1 MLP on x = embedding row Y<y> + e_CV_c + e_ON₀, 512 (y, c); excerpts.py `ode`; HAND: 448 neurons, W_in {2, 1000}, readout 512/512, YT error 0)
| net | active neurons | line-1 readout ok (of 512) | YT off ≥ ¼ | largest W_out rows (L1 norm) |
|---|---|---|---|---|
| T PART final best / final (s0-9) | 353-386 / 354-384 | 506-512 / 509-511 (misses at c 1, 3, 5; final: c = 5 in 10/10) | 511-512 | YT 215-242, ONE 133-154, Y1 127-152 |
| T PART r1 best / final (s0-2) | 360-383 / 365-386 | 507-510 / 509-510 | 512 | YT 216-230, ONE 127-139, Y1 125-133 |
| A PART r1 final (s0-2) | 321 / 336 / 335 | — | 506 / 509 / 507 | YT 71-94, ONE 58-64 |
Reading: the step writes ONE about as strongly as Y1; readout 2v·Y1 − v²·ONE peaks at v = Y1/ONE, so T PART encodes line 1 as a ratio and
YT is off everywhere. A PART does the same, but the frozen steps threshold absolute YT (tolerance ¼): the can't-fit failure.
## Comparison
HAND: 224 gated threshold pairs per step; T PART: same route, 353-386 active neurons, ratio code, held-out thresholds underdetermined (Madras
et al. 2019). Other cells match no HAND: unseen (y, c) states, and one opaque token per value gives nothing to interpolate (Kim et al. 2021; Kreitner et al. 2025).
## Controls and limits
Diag: 128 held-out T = 6 rows; hand_* probes y₀ ≤ 63; no T > 6 test (no N<T> token); "best" = picked on test (not a solve); edges spearman_exact
.88-1.0 (T PART final 6/10), .90-.94 (TRAIN); complement needed but not specific (random_same_size drops alike).
## References
Kim et al. 2021 https://doi.org/10.18653/v1/2021.emnlp-main.563 (EMNLP conference paper); Kreitner et al. 2025 https://doi.org/10.48550/arxiv.2510.06824 (arXiv); Ootani 2026
https://doi.org/10.48550/arxiv.2607.05104 (arXiv); Fu et al. 2024 https://doi.org/10.48550/arxiv.2401.10463 (arXiv); Madras et al. 2019 https://doi.org/10.48550/arxiv.1910.09573 (arXiv).
