# Group automata (parity, Z3, S3): a scratchpad state table is learned at once; answer-only nets learn count → residue only in range
**Task.** Words s_1..s_n over a group: parity (Z2, 2 symbols), a Z3 counter (3 symbols), S3 (the 6 permutations of 3 objects).
q_0 = identity, q_t = q_{t-1} · s_t. A (answer only): `BOS s_1 .. s_n = Q(q_n)`. B (scratchpad): `BOS s_1 Q(q_1) .. s_n Q(q_n)` (teacher
forced). n uniform in 1..16; per seed 20k unique training words and 2k disjoint test words; exact match of every answer token. Length
axis: 1000 random words at n = 32, 64, 128.
**Status.** 2026-11-05. HAND exact for all 6 cells (3 groups × 2 formats). PART and TRAIN 10 seeds per cell; every failed cell retried
(fresh words per batch, then tail-balanced symbol counts); all runs kept. Pending: S3 answer-only TRAIN at d_model 512 (size-ladder step 1).

## Summary
With a scratchpad (B), PART and TRAIN learn the transition table δ(q_{t-1}, s_t) within 2500 steps in all three groups (10/10 each). PART
keeps the hand-built recency head and stays exact to n = 128; TRAIN fetches q_{t-1} with its own heads and breaks at about twice the
training length. Answer-only (A): S3 PART reproduces the hand-built composition tables (10/10, exact to n = 31, the hand-built depth
bound). The abelian PART stage computes the symbol sum S exactly but learns S mod G only for S ≤ 11–18, even when large S are made
frequent. A TRAIN with fresh words: parity and Z3 10/10 (count, then a residue table; bound to the training length), S3 3/10.

## Setup
[METHODS.md](../METHODS.md). AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000 + cosine, clip 1, batch 256, fp32, eval every 500 steps,
best-test snapshot. r1: static 20k words; PART 10k steps (Z3 A 30k), TRAIN 40k, plateau stop 5k after 15k. r2 (data rule): fresh random
words every batch (test words masked), PART 30k, TRAIN 40k. r3: PART with tail-balanced symbol counts (per-word symbol probabilities
~ Dirichlet(1)). "mask": the PART stage's input weights on the position dimensions held at 0. Consumer GPUs (10–24 GB).
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/dfa-group-automata](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/dfa-group-automata)
(219: 6 HAND, 200 PART, 13 TRAIN; held-out test words per group and seed in `test_<group>.safetensors`).
**Reproduce:** the training and analysis code for this task is not yet in `code/`; the weights, test words and accuracies are in the folder above.

## Architecture as used
GPT-NeoX 6 layers × 4 heads (d_head 64), d_model 256, MLP 512 ReLU, no LayerNorm, no positional encoding (causal mask only), untied
readout; vocabulary 7 / 9 / 15 (BOS, PAD, G symbols, `=`, G states). PART re-initialises and trains one MLP on the frozen HAND (B: layer 1;
A parity and Z3: layer 1; A S3: layer 5; dense_h_to_4h [512, 256] + bias, dense_4h_to_h [256, 512] + bias). TRAIN: all weights from the
default initialisation. Archive: `hand_<group>_<F>` (state dicts), `part_<run>_<F>_s<seed>` (stage tensors, all seeds),
`train_<run>_<F>_s<seed>` (full nets, best and worst seed per run), `config.json`, `manifest.json` (seed, rung, run arguments, fp32 test
accuracy, recorded accuracy, sha256); every file reloaded bit-identically and its test accuracy reproduced.

## Prediction before training
Recorded in the task specification before any PART/TRAIN run. (1) B PART exact, masked exact at 8× length: held, unmasked too.
(2) B TRAIN ≥ 7/10, hand-built circuit, half the seeds ≥ .9 at 2× length: 10/10, hand-built single fetch head 1/10 (parity), ≥ .9 at
2× length only 1–2/10. (3) A abelian PART exact: failed (large S). (4) A abelian TRAIN ≤ 5/10 with a length-bound count shortcut: r1 parity
0/10, Z3 9/10; r2 10/10 and length-bound (shortcut held, count broken). (5) A S3 PART = HAND including length: held. (6) A S3 TRAIN ≤ 3/10,
depth-bound: r2 3/10 with a length frontier, held. (7) Grokking order B < A abelian < A S3: B first held; at PART the order is inverted.

## Hand-built reference
Layer 0: BOS-sink position stage (P = 1/(j+1); hinges give R = j/256). B: a layer-1 recency head (score 100·[state token] + 20 j) copies
q_{t-1} into QP; the layer-1 MLP has G² neurons relu(QP_a + TOK_b − 1) → OUT[a·b]: 1 head + 4 / 9 / 36 neurons at any n.
A abelian: a BOS-sink head averages the symbols, giving S = sum of symbol values; (G−1)·n hinge neurons relu(S − k) are read out as
[S mod G = r] (1 head + 16 / 32 neurons at n = 16; grows with n). A S3: prefix doubling; in layer k an index head fetches the block 2^(k−1)
back and a 36-neuron table composes the two blocks: 5 heads + 180 neurons, exact for n ≤ 31. Exact on 2000 words, n = 1..16, every cell.

## What training finds
- B PART (all groups, masked or not): 10/10 exact by step 500; the recency head is kept (previous-state attention mass 1.0); the stage is a
  (q_prev, symbol) table; exact at n = 128.
- B TRAIN: groks at steps 500–2500 (10/10 per group); the late MLPs are the δ table (substitution 1.0: parity MLP5, Z3 MLP3–5, 10/10). The
  previous-state fetch differs from HAND: parity spreads it over sharp heads in layers 2–5 (single HAND-like head 1/10); Z3 has one sharp
  head (mass .85–.95) in 7/10. Length: n = 32 .05–.99, n = 64 ≤ .14; token accuracy breaks at t = 24–48.
- A S3 PART: 10/10 by step 1000; the block states ST1..ST4 are decodable at accuracy 1.0 in layers 2..5 (the HAND layers); n = 32 .18 = HAND's .18.
- A abelian PART: S exact (probe R² 1.00), stage = f(S), but whole large S values are wrong. r2 fresh words: parity S = 13 wrong in 10/10,
  Z3 every S ≥ 17–19 wrong. r3 tail-balanced S (large S now ~12% of words): parity masked 7/10 exact, unmasked 1/10, Z3 0/10. Not data.
- A TRAIN r1 (20k static words): parity counts S but has no S mod 2 (test .50–.52, train also chance); Z3 9/10 (q decodable 1.00 at L4);
  S3 memorises and learns only the sign of the permutation (right sign in 8/10 seeds).
- A TRAIN r2 (fresh words): parity and Z3 10/10 (grok at 13–21k / 3.5–18k steps): S at L1–L2, q 1.00 from L2–L3, MLPs = f(S): the HAND
  type, but the residue staircase covers only the trained S (parity n = 32 chance; Z3 n = 32 ≥ .95 in 3/10). S3: sign 1.0 (10/10), exact up to
  n ≈ 9–13, best .70–1.0 (3/10 ≥ .99) and still rising at the budget; ST2..ST4 at chance: no doubling.
- Diagnostics (median over seeds; probe = best layer, one-hot accuracy, R² for S and position t; r90 = principal components for 90% of the
  last residual's variance; attention divergence to HAND ≈ 0 throughout because HAND's unused heads are uniform):

| cell | used heads + MLPs | probes | r90 | best / final |
|---|---|---|---|---|
| B PART (3 groups) | 1 + 1 | q_prev, q 1.00 (L2) | 2–10 | 1 / 1 |
| A S3 PART | 5 + 5 | ST1..ST4 1.00 (L2..L5), q .94 | 35 | 1 / 1 |
| B TRAIN parity / Z3 | 1–6 + 4–6 / 0–3 + 2–6 | q_prev, q 1.00 (L5–6 / L3), t R² .97 / .98 | 1.5 / 2 | 1 / 1 |
| A parity PART r1 / r2 / r3 (+mask) | 1 + 1 | S 1.00 (L2), q .63 / .97 / .96–.99 | 1 | .96–.99 / .986–.999 / .988–1.0 |
| A Z3 PART r1 / r2 / r3 | 1 + 1 | S 1.00 (L2), q .80 / .76 / .79 | 1 | .91–.98 / .92–.97 / .88–.97 |
| A parity TRAIN r1 / r2 | 0 + 0 / 3–10 + 4–6 | S 1.00 (L1), q .52 / 1.00 (L3) | 2 / 1 | .50–.52 / 1 |
| A Z3 TRAIN r1 / r2 | 2–6 + 5–6 / 3–6 + 4–6 | S R² .95 / .98 (L1 / L2), q 1.00 (L4 / L2) | 2 | .998–1 / 1 |
| A S3 TRAIN r1 / r2 | 0–4 + 0–5 / 4–20 + 4–6 | ST1 .37 / .55, ST2..4 .18–.23, q .34 / .64 | 2.5 / 4.5 | .21–.37 / .70–1.0 |

## Closed form
B: q_t = δ(q_{t-1}, s_t), a G × G table applied to the fetched previous state; substituting the (q_prev, symbol) mean table for the stage
keeps accuracy 1.0 (PART 30/30; TRAIN parity and Z3 10/10). A S3 PART: MLP5 = f(q) and MLP4 = f(ST4) tables, 10/10. A abelian: each stage is
a table f(S) (substitution = the net's own accuracy, PART 40/40, TRAIN r2 parity 10/10, Z3 8–9/10); replacing it by the exact table
[S mod G] gives 1.0, so the closed form exists and the trained stage misses only its large-S teeth.

## Weights excerpt and reading
B PART (masked) layer-1 stage: 65–85% of the input-weight energy (squared norm of dense_h_to_4h columns) sits on the QP and
symbol dimensions. The neurons are group characters, not HAND's AND gates: parity seed 0 neuron 116 reads QP (−.77, .22) and the symbol
(.73, −.26), i.e. a ± pattern that fires on (q, s) of unequal parity = XOR; S3 seed 0 neuron 106 reads QP and the symbols with the
permutation-sign pattern (+ − − + + −). Values are dense_h_to_4h rows of `part_<group>_mask_B_s0`, restricted to the QP / symbol blocks.

## Comparison
Scratchpad: one G² table at any length (HAND uses 4–36 neurons), learned at once; the length limit comes from the previous-state fetch
(PART n = 128, TRAIN ≈ 2× training length). Answer-only: HAND needs (G−1)·n neurons (abelian) or log2 n layers (S3); PART reproduces the S3
depth bound; the abelian residue staircase has only ~16–18 correct teeth, a resolution limit of a one-layer ReLU MLP learning a period-G
sawtooth on a scalar (consistent with the spectral bias of ReLU networks, Hong et al. 2022), not a data limit. Transformers are known to
learn logarithmic-depth "shortcut" solutions to automata that are brittle out of distribution (Liu et al. 2022); here S3 PART is exactly such
a log-depth solution, and TRAIN answer-only nets are length-bound. Fresh samples per step rescue the parity/Z3 count → residue cells
(r1 0/10 → r2 10/10), in line with the advantage of fresh-sample SGD on parities (Abbe & Sandon 2023). Scratchpads as an aid: Nye et al. 2021.

## Controls and limits
Linear probes only; teacher-forced scratchpad; one size (d 256; the d 512 S3 run is pending); 10 seeds per cell; length evaluated on random
words. The masked/unmasked PART pairs control for the position shortcut. The abelian PART r3 retry rules out coverage; a structured
(hinge) initialisation of the stage was not tried.

## References
- Liu, B., Ash, J. T., Goel, S., et al. (2022). Transformers learn shortcuts to automata. https://doi.org/10.48550/arxiv.2210.10749
- Hong, Q., et al. (2022). On the activation function dependence of the spectral bias of neural networks. https://doi.org/10.48550/arxiv.2208.04924
- Abbe, E., & Sandon, C. (2023). Polynomial-time universality and limitations of deep learning. https://doi.org/10.1002/cpa.22121
- Nye, M., et al. (2021). Show your work: scratchpads for intermediate computation with language models. https://doi.org/10.48550/arxiv.2112.00114
