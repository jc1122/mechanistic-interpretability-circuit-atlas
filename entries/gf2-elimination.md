# GF(2) Gauss-Jordan elimination: the hand-built row update is relearned at once; trained nets solve only with enough distinct matrices
**Task.** Solve A x = b over GF(2), A an invertible m × m bit matrix. Input `f r_0 , f r_1 , .. , f r_{m-1} ;`, row r_i = a_i0 .. a_i,m-1 b_i,
row flag f = `.` (not yet a pivot) or `*` (used as a pivot). T (scratchpad): for k = 1..m a block `p_k f r_0 , .. ;` = pivot label p_k
(`a`..`e`: the first unused row with a 1 in column k−1) and the rows after eliminating column k−1:
new_r = old_r ⊕ (old_r[k−1] ∧ [r ≠ p_k]) old_{p_k}, flag_r |= [r = p_k]; then `= x_0 .. x_{m-1}` with x_c = b of row p_{c+1} in the last
block. A (answer only): input, then `= x_0 .. x_{m-1}`. The invertible matrices are split 70/30 *by matrix* (every b of a matrix on one
side), so test systems use matrices never seen in training. Exact match of every output token. m = 3: all 168 invertible matrices,
118 train / 50 test (944 / 400 systems). m = 4: 6000 random invertible matrices of 20160, 67.2k train systems, 2000 test systems drawn
from the unseen-matrix systems. Size axis: 500 random systems at the other m (2, 3, 4, 5).
**Status.** 2026-11-06. HAND T exact (m = 2–5). m = 3: PART (two stages) and TRAIN T / A, 10 seeds each. Failed TRAIN cells retried with
the data rule at m = 4 (3 seeds per format); final cell: A TRAIN at m = 4, 10 seeds, 10/10 at test 1.0. Audit 2026-11-07: no learned closed form passes the full bar (best and final snapshots, matched zero and shuffle controls); across the two GF(2) entries 9 candidate
groups were rejected on re-check, 12 need reruns.

## Summary
The hand-built net computes each scratchpad token by fetching three bits of the previous block with 3-D attribute-match heads and
applying bit = OLD ⊕ (K ∧ PB ∧ ¬EQ) in a last-layer MLP. Re-learning that MLP (PART) or the pivot/answer attention (PARTp) on the frozen
hand-built net solves 10/10 within 1000–4000 steps, and the mean table on HAND's registers substitutes exactly in all 20 nets. From
scratch at m = 3 only 5/10 seeds per format generalise to unseen matrices: the 118 training matrices are memorised. With 6000 distinct
matrices at m = 4 the answer-only net solves 10/10 (best 12–21.5k steps) with a distributed circuit (11–17 heads + 4 MLPs). No cell
generalises to a larger m.

## Setup
[METHODS.md](../METHODS.md) for the shared conventions. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000, cosine (PART 10k, TRAIN 30k),
clip 1, batch 256 (m = 4: 6 micro-batches), eval every 500 steps, best-test snapshot; stop at test 1.0 or no gain > .005 over 5k steps after
min_steps (m = 3: 15k; m = 4 final: 22.5k). PART fp32, TRAIN TF32. Per-seed snapshots every 2500 steps at m = 4. All seeds of a cell in one
process on consumer GPUs.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/gf2-elimination](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/gf2-elimination)
(35: 1 HAND, 20 PART, 14 TRAIN; matrices and per-seed train masks in `split_m3`, `split_m4`).
**Reproduce:** the training and analysis code for this task is not yet in `code/`; the weights, splits and accuracies are in the folder above.

## Architecture as used
GPT-NeoX 4 layers × 12 heads (d_head 32), d_model 384, MLP 256 ReLU, no LayerNorm, no positional encoding (causal mask only), untied
readout; vocabulary 13 (BOS, `0 1 , ; = . *`, labels `a`..`e`), 256 positions. PART re-initialises one stage of the frozen HAND T
(normal .02, bias 0): PART = layer-3 MLP (dense_h_to_4h [256, 384] + bias, dense_4h_to_h [384, 256] + bias); PARTp = layer-3 attention
(query_key_value [1152, 384] + bias, dense [384, 384] + bias). TRAIN: all weights from the default initialisation. Archive
`gf2-elimination/`: `hand_T`; `part_T_m3_s0–9`, `partp_T_m3_s0–9` (stage tensors); `train_T_m3_s6`/`s0` (best; solved / failed),
`train_A_m3_s4`/`s3` (final; solved / failed); `train_A_m4_s0–9` (final cell, best); `config.json`, `manifest.json` (seed, rung,
snapshot, best step, recorded and re-scored test accuracy, sha256). Every file reloads bit-identically; fp32 re-scoring reproduces the
recorded test accuracy.

## Prediction before training
Recorded in the design file before any PART/TRAIN run. (1) PART row update 10/10: held (1000 steps). (2) PARTp ≥ 8/10 at m = 3; it keeps
m = 4, 5 only if it learns the first-row tie-break as a ramp: 10/10 in range, 0 at m = 4, 5. (3) TRAIN T at m = 3 ≥ 5/10 on unseen
matrices, pivot labels the last tokens to reach .99, m = 4 < .2: 5/10 (held at the edge), but labels and answers are 1.0 in every seed and
the misses are row-update bits (refuted); m = 4 0 (held). (4) TRAIN A at m = 3 ≤ 1/10 (memorises): refuted, 5/10, which also meets the
stated refutation criterion "TRAIN A ≥ TRAIN T".

## Hand-built reference
Every position carries frame attributes (block BLK, row ROW, offset J from the last separator or header) and their next-token versions,
computed in layers 0–1 (uniform head + 256 position neurons; five first/last/anchor heads; integer clean-up steps
relu(4z + 3) − relu(4z + 2)). Layer 2: five 3-D match heads fetch OLD (same slot one block back), OLDF (its flag), K (the old row's bit in
the eliminated column), PB (the pivot row's bit in this column), USED (own row flag) and PX (answer label); the MLP sets slot-type flags
and EQ = [this row is the pivot]. Layer 3: H0 picks the pivot (score −B·dist² + G·bit − G·USED − A·ROW: the first unused row with a 1),
H1 fetches the answer bit; the MLP writes bit = OLD ⊕ (K ∧ PB ∧ ¬EQ), flag = OLDF ∨ EQ, separators, label one-hot and answer. Size:
13 heads; neurons L0 256 / L1 80 / L2 14 / L3 32; 1630 non-zero of 3,164,416 parameters. Exact on every system at m = 2, 3 and 4 (all 20160 × 16) and on 200 random systems at
m = 5. No compact hand-built A net (x = A⁻¹ b).

## What training finds
- T PART (layer-3 MLP): 10/10 test 1.0 at step 1000; m = 2 1.0 in 6/10 seeds (0 in the rest), m = 4, 5 0.
- T PARTp (layer-3 attention): 10/10 test 1.0, best at 1.5–4k; m = 2 0–.37, m = 4, 5 0.
- T TRAIN m = 3: 5/10 best, 3/10 final (best .73–1.0, final .70–1.0, plateau stop 16.5k); label and answer tokens 1.0 in every seed, row-update tokens
  .992–1.0; m = 2, 4, 5 0. A TRAIN m = 3: 5/10 (best .845–1.0, final .805–1.0, plateau stop 20.5k); m = 2 .27–.64, m = 4 .04–.16, m = 5 .02–.06.
- Failure (m = 3, both formats): train ≥ .99 by 3.5–5k steps in every seed, then test oscillates with a train–test gap up to .40 in the
  failed seeds: they memorise matrix-specific updates / answers for the 118 training matrices (m = 3 has no more data).
- Retry (data rule, m = 4, 6000 matrices, 67.2k train systems, 3 seeds): A 3/3 test 1.0 (best 10–18k), train ≈ test throughout;
  T 3/3 test 1.0 at 8.5k, but the best/final nets were lost when the rented host's disk filled (step-7500 snapshot test .996–1.0);
  a replacement T run logs 3/3 best and final 1.0 by 6k, its nets not yet retrieved.
- Final A TRAIN m = 4 (10 seeds): 10/10 best and final test 1.0, best step 12–21.5k; test at 2.5k .22–.52, 5k .51–.96, 7.5k .74–1.0;
  m = 2 .09–.45, m = 3 .10–.27, m = 5 .03–.10 (final).
- Diagnostics (best snapshots, 250 test systems; used = ablation-used heads + MLP blocks, neurons not ablated; probe = best layer;
  inter → DAS = interchange flip, then distributed alignment search; "untested" = no answer-fixing pairs, not unused):

| cell (seeds diagnosed) | used heads + MLPs | probes (layer) | inter → DAS | r99 last layer | JS to HAND | best / final |
|---|---|---|---|---|---|---|
| T PART (7) | 8 + 2 (frozen fetches + MLP3) | OLD, K, PB, EQ R² 1 (L3) | OLD .42–.70 → 1; K 1; EQ .78–.91 → 1; PB untested | 11 | 0 | 1 / 1 |
| T PARTp (4) | 10–13 + 2 | OLD, K, PB, EQ 1 (L3); pivot .92–.96 (L4) | OLD .95–1 → 1; pivot .39–.54 → .26–.77 | 6–12 | .11 | 1 / 1 |
| T TRAIN m3 (6; 3 failed) | 11–15 + 4 | failed: OLD, K, PB, EQ R² .12–.70 (L3) | OLD .97–.98, K 1, EQ .60–.74 → ≥ .998 | 27–36 | .47 | .73–1 / .70–1 |
| A TRAIN m4 final (3) | 14–16 + 4 (final 11–17 + 4) | answer R² .87–.93 (L4) | answer ≈ 1 → 1 | 30–46 | — (no HAND) | 1 / 1 |

- Failed T TRAIN seeds (s0, s7, s8) miss 56 / 16 / 5 of 250 sequences, only row-update tokens. They carry the update inputs causally
  (interchange on OLD and K ≥ .97) though they are linearly weak at L3 (R² ≤ .70 vs 1 in PARTp); projecting onto the probe subspace
  does not repair them: the defect is the integration of the inputs, not a missing variable. The six diagnosed m = 3 TRAIN T nets (solved and failed) carry HAND's variables in
  another basis (residual alignment fit R² < 0, CKA .19–.44, edge overlap with HAND's wiring Jaccard .07–.10). In the A final cell the
  answer becomes linearly exact only at L4 (logit lens L3 .06–.84 → L4 ≈ 1); one of three diagnosed seeds uses no single L3 head above the
  ablation threshold. A TRAIN m = 3 and seven final-cell seeds are not diagnosed; no training-dynamics measurements exist for this task.

## Closed form
PART and PARTp: the row-update bit is bit = OLD ⊕ (K ∧ PB ∧ ¬EQ). Replacing the trained stage's row-bit output by the mean table
μ[OLD, K·PB·¬EQ, query slot] computed from HAND's live registers keeps accuracy 1.0 in all 10 PART and all 10 PARTp nets (coverage 1,
drop 0); the other slots stay native, and PARTp needs the registers before its own attention. TRAIN: no closed form accepted. Substituting
fitted Boolean tables for a solved T TRAIN net (seed 6) gives .0475–.0625; repairing failed seed 0 with an oracle row update lifts it
.735 → 1 but reproduces 0 of its 144 native errors (a repair, not an explanation). Oracle control on the A TRAIN m = 4 final cell
(terminal MLP replaced by a fitted readout of the externally supplied solution bits, 1,024 fresh systems, bar ≥ .99 × net): passes on
4/10 best and 10/10 final snapshots (an earlier 10-seed run: 5/10 / 9/10). The answer is readable at the end; this does not identify an
internal elimination.

## Weights excerpt and reading
HAND layer-3 row-update neurons (one per cell of (OLD, K, PB, EQ) whose output bit is 1, 8 cells):
relu(Σ ±x + 100·TB − (#ones − 1) − 100) with + for the cell's 1-inputs and − for its 0-inputs, writing Y["1"] − Y["0"]; one more neuron
TB → Y["0"] sets the default. The cell neuron fires (value 1) only on its own cell and only on bit slots (TB = 1).
PART (trained layer-3 MLP, per seed): only 8.6–10.1% of the squared norm of dense_h_to_4h lies on the four HAND inputs. The neuron with the
largest |Y["1"] − Y["0"]| output weight (input weights / row max | output):

| seed | neuron | OLD | K | PB | EQ | TB | out (Y1 − Y0) | cell (signs) → bit |
|---|---|---|---|---|---|---|---|---|
| 0 | 55 | −1.00 | −.71 | −1.00 | +.47 | +.33 | −.53 | (0,0,0,1) → 0 |
| 3 | 213 | +.89 | +.84 | +.71 | −1.00 | −.16 | −.69 | (1,1,1,0) → 0 |
| 8 | 120 | +.94 | −.95 | −1.00 | −.51 | +.09 | +.51 | (1,0,0,0) → 1 |
| 9 | 84 | +1.00 | −1.00 | −.47 | −.44 | −.11 | +.65 | (1,0,0,0) → 1 |
| 2 | 45 | −1.00 | +.67 | +.59 | +.59 | −.09 | +.53 | (0,1,1,1) → 0 (vote 1: exception) |

Reading: in 9/10 seeds the top neuron's sign pattern on (OLD, K, PB, EQ) is one cell of the truth table and its output votes that cell's
bit (cells (0,0,0,1) in 4 seeds, (1,0,0,0) in 2); the same cell-detector form as HAND, mostly on different cells; seed 2's top neuron does
not match a single cell. Values are slices of `part_T_m3_s<seed>` (rows of `gpt_neox.layers.3.mlp.dense_h_to_4h.weight` at the OLD, K,
PB, EQ, TB residual dimensions given in `config.json`'s generator, divided by the row's largest |weight|; out = difference of the
`dense_4h_to_h` rows of the Y["1"] and Y["0"] dimensions).

## Comparison
PART relearns HAND's update on HAND's fetched bits, like the matrix-product entry ([gf2-matrix-product](gf2-matrix-product.md)), whose
trace step is the same kind of XOR-AND lookup. From scratch the step is distributed over 11–17 heads and all four MLPs, with HAND's inputs
present but not HAND's wiring. Data, not format, decided TRAIN: at m = 3 the by-matrix split leaves 118 training matrices and half of the
seeds memorise; with 6000 matrices every seed generalises. This matches the picture of grokking as needing a training fraction above a
critical value, with a memorisation phase below it (Liu et al. 2022), and the finding that transformers trained on linear algebra
generalise better from more diverse matrix sets (Charton 2021). Nothing transfers to another m: the frame and the trained stages key on
the trained block layout.

## Controls and limits
Unseen-matrix test sets (by-matrix split). m = 3 TRAIN cells are the 10-seed screening; the m = 4 T retry lost its best/final nets
(disk full), so T at m = 4 is a 3-seed result from logs and a step-7500 snapshot. Diagnostics cover 7/10 PART, 4/10 PARTp, 6/10 T TRAIN
m = 3 and the A TRAIN m = 4 final cell on 10 best + 10 final snapshots (2026-11-07; its earlier run 10 best + 6 final); A TRAIN m = 3
has none. Open: el4 intermediate variables (old/k/pb/eq/pivot are not probed in the answer-only nets), training dynamics (checkpoints
lost), final-snapshot substitutions. Linear probes; interchange/DAS only for answer-fixing variables; no neuron-level
ablation; no training-dynamics measurements. The archived A m = 3 nets are final snapshots (their best-snapshot file was truncated).
One size (d_model 384).

## References
- Liu, Z., et al. (2022). Towards understanding grokking: an effective theory of
  representation learning. arXiv preprint. https://doi.org/10.48550/arxiv.2205.10343
- Charton, F. (2021). Linear algebra with transformers. arXiv preprint. https://doi.org/10.48550/arxiv.2112.01898
