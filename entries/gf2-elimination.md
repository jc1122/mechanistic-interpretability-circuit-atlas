# GF(2) Gauss-Jordan elimination: the hand-built row update is relearned at once; trained nets solve only with enough distinct matrices
**Task.** Solve A x = b over GF(2), A invertible m × m. Input `f r_0 , .. , f r_{m-1} ;`, row r_i = a_i0 .. a_i,m-1 b_i, flag f `.` unused / `*`
pivot used. T (scratchpad): per step k a block `p_k f r_0 , .. ;` (pivot p_k = first unused row with a 1 in column k−1; new_r = old_r ⊕
(old_r[k−1] ∧ [r ≠ p_k]) old_{p_k}), then `= x_0 .. x_{m-1}`; A (answer only): input, `= x`. Split 70/30 *by matrix* (test = unseen matrices).
m = 3: all 168 matrices (944 / 400 systems); m = 4: 6000 of 20160, 67.2k train, 2000 test. Exact match; size axis 500 systems at m = 2–5.
**Status.** 2026-11-08. HAND T exact. m = 3: PART, PARTp, TRAIN T, TRAIN A, 10 seeds each; m = 4 retry (3 seeds per format); final cell A TRAIN
m = 4, 10 seeds: 10/10 test 1.0. GPT-NeoX 4 × 12 heads, d_model 384; PART 10k, TRAIN 30k steps.
## Summary
HAND fetches three bits of the previous block with 3-D match heads and writes bit = OLD ⊕ (K ∧ PB ∧ ¬EQ) in its last MLP; PART relearns it in
10/10 seeds (1000 steps; mean-table substitution exact in 20/20 snapshots, controls pass). From scratch m = 3 gives 5/10 per format (118 matrices
memorised); with 6000 matrices at m = 4 answer-only solves 10/10 in three types (6/10 read out with L3 heads, 2/10 diffusely in L3, 2/10 finish by L2). No size transfer.
## Setup
[METHODS.md](../METHODS.md). AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000, cosine, clip 1, batch 256 (m = 4: 6 micro-batches); eval every
500, best snapshot; stop at test 1.0 or 5k plateau after min_steps (m = 3 15k, m = 4 22.5k). PART fp32, TRAIN TF32, consumer GPUs. Networks:
huggingface.co/jc1122/circuit-atlas-nets/tree/main/gf2-elimination. Reproduce: code not yet in `code/`; weights, splits, accuracies are there.
## Architecture as used
4 layers × 12 heads (d_head 32), d_model 384, MLP 256 ReLU, no LayerNorm, no positional encoding, untied readout; vocab 13 (BOS, `0 1 , ; = . *`,
`a`..`e`), 256 positions. PART re-initialises HAND's L3 MLP (dense_h_to_4h [256, 384], dense_4h_to_h [384, 256], biases); PARTp its L3 attention
(query_key_value [1152, 384], dense [384, 384], biases); TRAIN all weights. Archive `gf2-elimination/`: hand_T, part_T_m3_s0–9, partp_T_m3_s0–9,
train_T_m3_s6/s0, train_A_m3_s4/s3, train_A_m4_s0–9, split_m3/m4, config.json, manifest.json (seed, rung, step, test, sha256; bit-identical reload).
## Prediction before training
Recorded before the runs. PART 10/10: held. PARTp ≥ 8/10, 0 at m = 4, 5: held. TRAIN T m = 3 ≥ 5/10, pivot labels last to .99: 5/10 held, but
labels/answers are 1.0 and the misses are row-update bits (refuted). TRAIN A m = 3 ≤ 1/10: refuted (5/10).
## Hand-built reference
L0–L1: frame BLK, ROW, J and next-token versions (uniform head + 256 position neurons, five anchor heads, relu(4z+3) − relu(4z+2)). L2: five 3-D
match heads fetch OLD, OLDF, K, PB, USED, PX; MLP: slot flags and EQ. L3: H0 picks the pivot (−B·dist² + G·bit − G·USED − A·ROW), H1 the answer
bit; MLP: one neuron per 1-cell of (OLD, K, PB, EQ), relu(Σ ±x + 100 TB − (#ones − 1) − 100) → Y1 − Y0, plus TB → Y0. 13 heads, neurons
256/80/14/32, 1630 non-zero weights. Exact on all systems at m = 2, 3, 4 and 200 random at m = 5.
## What training finds
- T PART 10/10 at 1000 steps (m = 2 1.0 in 6/10, m = 4, 5 0); PARTp 10/10, best 1.5–4k. TRAIN m = 3: T 5/10, A 5/10; train ≥ .99 by 3.5–5k,
  failed seeds memorise the 118 matrices. m = 4 retry: A 3/3, T 3/3 (nets lost). Final A m = 4: 10/10 best and final 1.0, best step 12–21.5k.
- Circuit types: PART one (HAND fetch + cell-detector MLP3) 10/10; PARTp one, its attention also carries the update in 7/20 snapshots; TRAIN m = 3
  distributed, HAND inputs present in another basis (align R² < 0, edge Jaccard .07–.10); A final: L3-head readout 6/10 (4–10 L3 heads, answer
  R² at L3 ≤ .26), L3 diffuse 2/10 (no single L3 head used, whole L3 needed, R² at L3 .54) and L2-complete 2/10 (L3 idle, R² at L3 .89).
- Diagnostics (250 test systems; used = mean-ablation drop > .05, neurons not ablated; r99 = residual rank at L4; JS = used heads vs HAND):

| Cell (diag seeds) | used heads + MLPs | best probe R² per HAND variable (layer) | r99 (HAND 11) | JS | best / final |
|---|---|---|---|---|---|
| T PART (7) | 8 + 2 (frozen fetches, MLP3) | OLD, K, PB, EQ 1 (L3); interchange K 1, EQ, OLD → DAS 1 | 11 | 0 | 1 / 1 |
| T PARTp (4) | 10–13 + 2 | OLD, K, PB, EQ 1 (L3); pivot .92–.96 (L4) | 6–12 | .11 | 1 / 1 |
| T TRAIN m3 (6) | 11–15 + 4 | failed seeds OLD, K, PB, EQ .12–.70 (L3); interchange OLD, K ≥ .97 | 27–36 | .47 | .73–1 / .70–1 |
| A TRAIN m4 final (10 B + 10 F) | 6–19 + 4 (final) | answer .80–.94 (L4); flip .996–1 (L4) | 20–51 | — (no HAND) | 1 / 1 |
## Closed form
PART: μ[OLD, K·PB·¬EQ, query] on HAND's live registers replaces MLP3's row-bit write: 1.0 = native in 20/20 snapshots (944 fit / 400 test);
zeroed 0, shuffled ≤ .01, negated 0, other seed's table 1.0: accepted. PARTp: 1.0 in 20/20, zero/shuffle keep ≥ .2 in 7/20: not accepted. TRAIN
m = 3: Boolean tables .05–.06. A final: oracle readout μ_c + (2x_c − 1)g_c 4/10 best, 10/10 final, zero-MLP3 control fails: not accepted. L3
attention → per-slot mean (2000 test): L2-complete 1.0 = native in 4/4 snapshots (accepted); L3 diffuse .38–.48, L3-head .05–.18: no closed form.
## Weights excerpt and reading
HAND L3 cell neuron: relu(Σ ±x + 100·TB − (#ones − 1) − 100), + on the cell's 1-inputs, writing Y1 − Y0. PART top |Y1 − Y0| neuron (layer-3
dense_h_to_4h row at OLD, K, PB, EQ, TB / row max; out = Y1 − Y0 difference of dense_4h_to_h): s0 n55 (−1.00, −.71, −1.00, +.47, +.33; −.53) =
cell (0,0,0,1) → 0; s3 n213 (+.89, +.84, +.71, −1.00, −.16; −.69) = (1,1,1,0) → 0; s9 n84 (+1.00, −1.00, −.47, −.44, −.11; +.65) = (1,0,0,0) → 1.
Reading: in 9/10 seeds the top neuron's signs are one truth-table cell and it votes that cell's bit (HAND's form, other cells); 8.6–10.1% of W_in.
## Comparison
PART = HAND's update on HAND's bits, as in [gf2-matrix-product](gf2-matrix-product.md); from scratch 6–19 heads and all MLPs. Data decided TRAIN
(118 matrices memorise, 6000 generalise): a critical training fraction (Liu et al. 2022); diverse matrices generalise (Charton 2021).
## Controls and limits
Unseen-matrix tests. m = 4 T: 3 seeds, nets lost. Diagnosed: PART 7/10, PARTp 4/10, T TRAIN m3 6/10, A m3 none, A final 20/20. Linear probes, no
neuron ablation, no training dynamics, one width. Size: exact only at the training m (A final m = 2 .09–.45, m = 3 .10–.27, m = 5 .03–.10).
## References
- Liu, Z., et al. (2022). Towards understanding grokking. arXiv. https://doi.org/10.48550/arxiv.2205.10343 · Charton, F. (2021). Linear algebra with transformers. arXiv. https://doi.org/10.48550/arxiv.2112.01898
