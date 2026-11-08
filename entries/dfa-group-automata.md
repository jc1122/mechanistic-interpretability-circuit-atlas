# Group automata (parity, Z3, S3)

> - **Task.** Give the final state of a parity, Z3 or S3 automaton after a word of 1..16 symbols. Example: parity word with three symbols of value 1 → state 1.
> - **Hand-built net.** A 6-layer net, every weight set by formula, exact in all 6 cells: with a scratchpad one recency head copies the previous state and a G² neuron table applies the transition; answer-only uses a symbol sum (parity, Z3) or prefix doubling (S3, exact for n ≤ 31).
> - **Trained nets.** With a scratchpad, PART (last MLP trained on frozen hand-built parts) and TRAIN (from scratch) learn the transition table within 2500 steps in 10/10 seeds per group; PART is exact to n = 128, TRAIN breaks at about twice the training length. Answer only: S3 PART 10/10; abelian PART learns S mod G only for S ≤ 11–18; TRAIN with fresh words parity and Z3 10/10, S3 2/10 at ≥ .99.
> - **Verdict.** Partly solved (scratchpad 10/10 per group; answer-only S3 TRAIN 2/10 at ≥ .99). Closed form: partial (δ table keeps accuracy 1.0 on the stated words but is rejected as universal; audit 2026-11-07: no closed form passes the full bar).
> - **Links.** [Weights](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/dfa-group-automata) · Reproduce: not available yet · [Terms](../GLOSSARY.md)

## Task and status
**Task.** Words s_1..s_n over a group: parity (Z2, 2 symbols), a Z3 counter (3 symbols), S3 (the 6 permutations of 3 objects).
q_0 = identity, q_t = q_{t-1} · s_t. A (answer only): `BOS s_1 .. s_n = Q(q_n)`. B (scratchpad): `BOS s_1 Q(q_1) .. s_n Q(q_n)` (teacher
forced). n uniform in 1..16; per seed 20k unique training words and 2k disjoint test words; exact match of every answer token. Length
axis: 1000 random words at n = 32, 64, 128.
**Formats.**
- **A**: answer only, `BOS s_1 .. s_n = Q(q_n)`.
- **B**: scratchpad, `BOS s_1 Q(q_1) .. s_n Q(q_n)` (teacher forced).

**Status.** 2026-11-05. HAND exact for all 6 cells (3 groups × 2 formats). PART and TRAIN 10 seeds per cell; every failed cell retried
(fresh words per batch, then tail-balanced symbol counts; S3 answer-only TRAIN then one size step, d_model 512); all saved nets kept.
Audit 2026-11-07 (unrounded curves): S3 answer-only TRAIN r2 has 2/10 seeds ≥ .99 (max .9995, none exact), not 3/10; no closed form
below passes the full bar (every substitution is on best snapshots; none on final snapshots).

## Summary
With a scratchpad (format B), PART (the last MLP trained on frozen hand-built parts) and TRAIN (all weights trained from scratch) learn the transition table δ(q_{t-1}, s_t) within 2500 steps in all three groups (10/10 each). PART
keeps the hand-built recency head and stays exact to n = 128; TRAIN fetches q_{t-1} with its own heads and breaks at about twice the
training length. Answer-only (A): S3 PART reproduces the hand-built composition tables (10/10, exact to n = 31, the hand-built depth
bound). The abelian PART stage computes the symbol sum S exactly but learns S mod G only for S ≤ 11–18, even when large S are made
frequent. A TRAIN with fresh words: parity and Z3 10/10 (count, then a residue table; bound to the training length), S3 2/10 at ≥ .99.

## Setup
[METHODS.md](../METHODS.md). AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000 + cosine, clip 1, batch 256, fp32, eval every 500 steps,
best-test snapshot. r1: static 20k words; PART 10k steps (Z3 A 30k), TRAIN 40k, plateau stop 5k after 15k. r2 (data rule): fresh random
words every batch (test words masked), PART 30k, TRAIN 40k. r3: PART with tail-balanced symbol counts (per-word symbol probabilities
~ Dirichlet(1)). "mask": the PART stage's input weights on the position dimensions held at 0. Consumer GPUs (10–24 GB).
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/dfa-group-automata](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/dfa-group-automata)
(219: 6 HAND, 200 PART, 13 TRAIN; held-out test words per group and seed in `test_<group>.safetensors`).
**Reproduce:** training code for this task is not yet in `code/`; the weights, held-out splits and recorded accuracies are on Hugging Face (link above).

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
depth-bound: r2 2/10 (≥ .99) with a length frontier, held. (7) Grokking order B < A abelian < A S3: B first held; at PART the order is inverted.

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
  Z3 every S ≥ 17–19 wrong. r3 tail-balanced S (large S now ~12% of words): parity masked 4/10 exact on the original test split (7/10 on fresh batches), unmasked 1/10, Z3 0/10. Not data.
- A TRAIN r1 (20k static words): parity counts S but has no S mod 2 (test .50–.52, train also chance); Z3 9/10 (q decodable 1.00 at L4);
  S3 memorises and learns only the sign of the permutation (right sign in 8/10 seeds).
- A TRAIN r2 (fresh words): parity and Z3 10/10 (grok at 13–21k / 3.5–18k steps): S at L1–L2, q 1.00 from L2–L3, MLPs = f(S): HAND's
  variables (interchange below) in a different code (no orthogonal map to HAND's residual, CKA .11–.23), but the residue staircase covers only the trained S (parity n = 32 chance; Z3 n = 32 ≥ .95 in 3/10). S3: sign 1.0 (10/10), exact up to
  n ≈ 9–13, best .70–.9995 (2/10 ≥ .99) and still rising at the budget; ST2..ST4 at chance: no doubling.
- Diagnostics (median over seeds; probe = best layer, one-hot accuracy, R² for S and position t; r90 = principal components for 90% of the
  last residual's variance; attention divergence to HAND ≈ 0 throughout because HAND's unused heads are uniform; patch = activation patching
  on a shuffled batch, component with the largest accuracy drop; uses = interchange flip → DAS flip of the HAND variable at its best layer,
  ranges over 167 seeds; "untested" = no answer-fixing pairs, not unused):

| cell | used heads + MLPs | probes | r90 | patch (drop) | uses (inter → DAS) | best / final |
|---|---|---|---|---|---|---|
| B PART (3 groups) | 1 + 1 | q_prev, q 1.00 (L2) | 2–10 | MLP1 1.0, fetch head L1H0 .94 | sym 1, q .93–1 → 1; q_prev untested | 1 / 1 |
| A S3 PART | 5 + 5 | ST1..ST4 1.00 (L2..L5), q 1.00 | 35 | index heads L1–L3 ≈ 1 | q .26–.54 → .78–1; ST1–4 untested | 1 / 1 |
| B TRAIN parity / S3 | 1–6 + 4–6 / 0–3 + 4–6 | q_prev, q 1.00 (L3–6), t R² .88–.99 | 1.5 / 2 | MLP0 (6/10, 7/9), late MLPs .84–.98 | sym 1, q .99–1 → 1 | 1 / 1 |
| A parity PART r1 / r2 / r3 (+mask) | 1 + 1 | S 1.00 (L2), q .84–.97 / .93–1 / .93–1 | 1 | L1H0 and MLP1 1.0 (55/55) | S .92–1; q .48–.79 → .85–1 | .96–.99 / .986–.999 / .988–1.0 |
| A Z3 PART r1 / r2 / r3 | 1 + 1 | S 1.00 (L2), q .80 / .76 / .79 | 1 | L1H0 and MLP1 1.0 (11/11) | S .82–.90 (2 seeds); q .64–.86 → .70–.88 | .91–.98 / .92–.97 / .88–.97 |
| A parity TRAIN r1 / r2 | 0 + 0 / 3–10 + 4–6 | S 1.00 (L1), q .52 / 1.00 (L3) | 2 / 1 | — / layer-0 heads + MLP0 1.0 | S, q .48–.53 → .48–.53 / S 1.00, q .85–1 → .97–1 | .50–.52 / 1 |
| A Z3 TRAIN r1 / r2 | 2–6 + 5–6 / 3–6 + 4–6 | S R² .95 / .98 (L1 / L2), q 1.00 (L4 / L2) | 2 | MLP5 or layer-0 heads ≈ 1 | q .94–.96 → 1.00 (2 seeds) | .998–1 / 1 |
| A S3 TRAIN r1 / r2 | 0–4 + 0–5 / 4–20 + 4–6 | ST1 .37 / .55, ST2..4 .18–.23, q .34 / .64 | 2.5 / 4.5 | layer-0 heads, MLP5 ≈ 1 | untested / q .62–.99 → .57–.99 (5/7); ST1–4 untested | .21–.37 / .70–1.0 |
| A S3 TRAIN r3 / r3b (d 512) | not diagnosed (no best snapshot) | — | — | — | — | .49–.99 at 40k / .36–.51 at 27.5k (logged test) |
- A S3 TRAIN size step (d_model 512, fresh words): r3 (10 seeds, 40k) reached test .487–.988, still rising, but the nets were lost (GPU
  out of memory in the final length evaluation, before the save); r3b (3 seeds, 80k from scratch, 2× budget) stopped at step 27.5k when the
  rented host's disk filled, test .36–.51. No best snapshot exists, so neither run is
  diagnosed or archived. This was the one size step of the retry order; the answer-only S3 TRAIN result stands at 2/10 ≥ .99 (r2).
- Causal verdicts. Failed abelian PART stages use S (interchange .92–1) and carry q, recoverable by DAS (.85–1) though only half linearly
  readable (.48–.79): the miss is the large-S readout, not the count. Deleting the S/q subspace drops every parity A PART and parity TRAIN r2 net to
  chance (.46–.52), keeping only it leaves accuracy unchanged. Parity TRAIN r1 has S but no answer-controlling variable (all flips at chance).
  Solved TRAIN nets use HAND's variables in their own basis (residual alignment fit R² < 0); a high CKA does not mark success (failed parity
  r1 .99, solved r2 ≤ .23). S3 TRAIN: no linear trace of HAND's block states; whether some other code carries them is not tested.

## Closed form
B: q_t = δ(q_{t-1}, s_t), a G × G table applied to the fetched previous state; substituting the (q_prev, symbol) mean table for the stage
keeps accuracy 1.0 (PART 30/30; TRAIN parity and Z3 10/10) on the stated words; as a universal reduction it is rejected: on a re-check
parity seed 9 keeps 126/128 (< .99 of the net). A S3 PART: MLP5 = f(q) and MLP4 = f(ST4) tables, 10/10. A abelian: each stage is
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
Probes, ablation, activation patching, interchange and DAS on the best snapshots (final snapshots unchecked; neuron
counts not measured); teacher-forced scratchpad; size: d 256 throughout, the one d 512 step (S3 answer-only TRAIN) ended without saved nets
(out of memory at 40k, disk full at 27.5k of 80k), so size vs solved is untested; 10 seeds per cell; length evaluated on random
words. The masked/unmasked PART pairs control for the position shortcut. The abelian PART r3 retry rules out coverage; a structured
(hinge) initialisation of the stage was not tried.

## Circuit found in trained nets
Circuit probes on selected best snapshots (2026-11-06). **Verified** = causal intervention plus an accuracy-preserving substitution on the
stated finite domain; **supported** = causal only. Layers zero-based; H = heads, N = neurons; B is teacher-forced.

| Component | Role | Evidence |
|---|---|---|
| S3 B TRAIN, all L0 heads | weighted prefix histogram | substitution 512/512 in both tested seeds (verified) |
| S3 B TRAIN, joint forms (histogram heads, ideal L2 routing, MLP2..5 tables) | answer-preserving reduction | seed 4 full net 100%; both pruned circuits 100% with refitted coefficients, 512 words (verified answer reduction only) |
| S3 B PART, L1 fetch + MLP1 | displayed-state transport, local product | fetch zeroed: exact 0; changed-state patch 100%; 36-number substitution 256/256 (verified) |
| S3 A PART, MLP5 | composes the last frozen blocks | 36-number substitution 256/256, drop 0 (verified) |
| S3 A TRAIN (run 2), MLP5 | sign + inverse-image-of-zero readout | rank-4 projection 250/256, agreement 100%; sign-only 88/256, point-only 171/256 (verified readout only) |
| Z3 A TRAIN (run 2), MLP5 | in-range periodic readout | seven-number cosine form 256/256, drop 0; MLP5 deletion 45.3% (verified at n = 16) |

- Closed forms. Histogram head: `H = Σ_t C_t exp(q·k_t/8) OV_t / Σ_t C_t exp(q·k_t/8)` (C_t = prefix token counts). S3 PART: state-logit
  write `T[MUL[q_a, q_b]]` with 36 fitted numbers (q_a, q_b = fetched/frozen-block states). S3 A TRAIN: `δ_e = c(x) + u(x) sign(e) + v_{e⁻¹(0)}(x)`,
  99.96% of the state-write variance, coefficients still neural (no input algorithm). Z3 A TRAIN: `d = max(0, 1 − (S − 6)/29.5)`,
  `δ_{6+e} = d a_e cos(2π(S − e)/3)`, other channels `d b_{S mod 3}`, a = (24.439, 27.467, 24.795), b = (−62.595, 2.495, −38.438); write R² .764;
  n = 32 228/256, n = 64 65/256: not faithful out of range.
- Supported: S3 B TRAIN previous-state transport (whole-L2 changed-state patch 100%; ideal routing 512/512 in both seeds, but seed 0
  improves 115/128 → 128/128 on random-display words: a repair). S3 A TRAIN sign/point factors (DAS transfers 256/256 and 181/187); the two
  late heads alone transfer the point in only 61/351 donor patches, so the upstream point-product algorithm is unresolved.
- Critical limit: scrubbing the whole S3 B TRAIN hypothesis at once gives 71.42% (seed 4) and 1.69% (seed 0) on the sufficient circuits
  (mean of 3 draws) although each is 100% unscrubbed and under joint substitution: the replacements do not prove independent causal modules.
- Size (sufficient upper bounds, not minima): S3 B TRAIN seed 4 7H + 2202N, seed 0 8H + 1333N (of 24H + 3072N), 512/512 each; HAND B
  2H + 292N (256 position hinges + 36 pair gates), not re-minimised. PART learned stage: 34/512 (B) and 22/512 (A) neurons suffice, 256/256.
- Seed census (150 best snapshots in 17 cells; mechanism classes, not basins): B PART parity/Z3/S3 20/20/20 frozen fetch + product, compact
  substitution 100%. B TRAIN (10 each) display-local / history-sensitive: parity 3/7, Z3 7/3, S3 9/1; the strongest-attention layer is an
  exactly sufficient fetch in 0/10, 2/10, 6/10. A S3 PART (20): 14 faithful compact products, 6 one-error repairs, all on frozen doubling.
  A TRAIN parity run 2: 10 late-count codes; Z3 run 1: 9 count + context; Z3 run 2: 1 faithful late count, 9 count + context. A S3 TRAIN
  run 2: seed 4 has the faithful sign + point readout; seed 0 (also ≥ .99) and seed 3 (.99 only after rounding) need extra directions. A parity PART run 3: 8 scalar-hinge codes. Z3 A PART: 0 solved.
- Unlike HAND: S3 B TRAIN fetches later and redundantly with distributed pair/position thresholds; PART keeps HAND's interfaces but not
  its single-neuron AND gates (pair dependence R² .9993 vs product-only .148 for the best-fitting neuron).
- Note: the Weights excerpt reads B PART neurons as group characters; the probes verify pair dependence but not a unique group-character
  role. The Closed form's f(S) tables for Z3 A TRAIN hold on random test words; on a balanced histogram grid MLP2..5 = f(S) gives only
  90.41% agreement, so a global f(S) reduction is refuted. The parity A PART run-3 nets exact on test words still miss 2 of 17 counts on that grid.
- Open: final-snapshot substitutions for every form above (0 final-weight diagnostics); the d 512 S3 answer-only size final (not
  completed); six solved seeds (6/9) of a context-reset S TRAIN follow-up, mechanism unclassified; the S3 answer-only point product; what the failed whole-hypothesis scrub omits; smallest sufficient circuits under length extension.

## References
- Liu, B., Ash, J. T., Goel, S., et al. (2022). Transformers learn shortcuts to automata. https://doi.org/10.48550/arxiv.2210.10749 (arXiv preprint)
- Hong, Q., et al. (2022). On the activation function dependence of the spectral bias of neural networks. https://doi.org/10.48550/arxiv.2208.04924 (arXiv preprint)
- Abbe, E., & Sandon, C. (2023). Polynomial-time universality and limitations of deep learning. https://doi.org/10.1002/cpa.22121 (journal article, Communications on Pure and Applied Mathematics)
- Nye, M., et al. (2021). Show your work: scratchpads for intermediate computation with language models. https://doi.org/10.48550/arxiv.2112.00114 (arXiv preprint)
