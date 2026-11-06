# KRK mate (king + rook vs king): an exact hand-built rule router; a scratchpad net reaches .99, but its circuit is not the hand-built one
**Task.** King + rook vs king, white to move. Domain: all 175,168 won white-to-move positions; a fixed permutation holds out 10%
(17,516 positions; 157,652 train). Prompt: 73 tokens (BOS, side, castling, en passant, 64 squares, `=`). The label is the move of a
deterministic 8-rule fence policy (FLEE, SHRINK, CHECK, WAIT, REPOS, APPROACH, DIVIDE, FAR, in priority order; integer keys break ties,
one move per position), not distance-to-mate (DTM) optimal play. Formats: **A** (answer only): from, to. **G** (scratchpad): the fence
box fl fh rl rh, area, rook-safety flag, active rule (phase), then from, to. **C**: G plus the active rule's candidate destinations in
key order (padded to 14) before from, to. Sequence lengths A/G/C 75/82/96. Metric: exact match of the whole output, teacher-forced.
**Status.** 2026-11-06. HAND exact on all positions (sparse feature-level check, below). PART: 10 seeds per format and setting.
TRAIN: 30k steps (10 seeds per format), 60k (3), 120k (3 per format A/G/C). The 10-seed final G cell at 120k is incomplete (below).

## Summary
The rule policy mates from every position (worst-case defender: max 48, mean 24.1 moves; DTM-optimal mean 9.86) and compiles into a
hand-built circuit of 4 heads + 6,666 ReLU neurons. Given the frozen hand-built rule features, a trained 3 × 512 router reaches
.993–.999 held-out exact in 60/60 seed-runs but none is exact; a closed form for its phase output (eight per-rule templates gated by the
hand-built rule priority, 4,096 fitted numbers) substitutes without loss in 20/20 snapshots. From scratch, the scratchpad G grows with
budget (30k .72–.86, 60k .96–.98, 120k **.993–.998, 3/3 seeds ≥ .99**); answer-only A stays below (.970–.977, 0/3) and wins only
.53–.60 of games against a tablebase defender (G .85–1.0, C 1.0). The trained nets read the kings as coordinates, not squares; their
last-layer MLP is not an additive function of the hand-built variables (substitution .995 → .34). The destination tie-break is the
last token learned everywhere.

## Setup
[METHODS.md](../METHODS.md) for the shared conventions. TRAIN: AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000, cosine, clip 1,
batch 64 of fresh positions from the train split each step, TF32, eval every 250 steps on a fixed 2,048-position held-out batch,
best-test snapshot; stop on a 5k plateau after 75% of the budget. PART: fp32, batch 256 from all train positions, same schedule, eval on
all 17,516 held-out positions; OHEM variant keeps the 256 highest-loss rows of 1,024 per step ([Köksal et al.](https://doi.org/10.1109/ICIP46576.2022.9897806)),
run from scratch and as a continuation of each baseline seed (dependent runs). Games: best snapshots, greedy, 512 held-out starts,
60-move cap, DTM-maximising tablebase defender and a random defender. Legality via [python-chess](https://python-chess.readthedocs.io/).
Consumer GPUs. All seeds of a cell in one process.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-krk](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-krk)
(35: TRAIN G and C best + final seeds 0–2, A best seeds 0–2; PART G OHEM-continuation best + final seeds 0–9; held-out states and the
2,048-row test batch per format in `heldout.safetensors`).
**Reproduce:** the training, hand-built feature and analysis code for this task is not yet in `code/`; the weights, test batches and
recorded accuracies are in the folder above. Re-scored in fp32 on the archived batch, all 15 TRAIN nets reproduce their recorded
test accuracy (G, C exactly; A within .001 of its 3-digit training log); the PART routers reload bit-identically (their accuracy needs the HAND features).

## Architecture as used
TRAIN (A, G, C): GPT-NeoX 6 layers × 12 heads (d_head 32), d_model 384, MLP 1536 ReLU, no LayerNorm, no positional encoding (causal
mask only), untied readout, vocabulary 293; 72 heads + 9,216 neurons allocated. HAND: sparse ReLU levels on the three piece one-hots
(white king, rook, black king) plus output-position flags; its token front end is one uniform counting head and three gather heads
(one per piece). PART: frozen HAND levels below the rule-priority level give 3,122 features per position (uint8, computed once); trained
router h1 = ReLU(x W1 + pos[p] + b1), h2 = ReLU(h1 W2 + b2), h3 = ReLU(h2 W3 + b3), logits = h3 W4 + b4; W1 [3122, 512], W2, W3 [512, 512],
W4 [512, 293], pos [9, 512] (one row per output position p). Archive `chess-krk/`: `train_{G,C}_s{0-2}_{best,final}`,
`train_A_s{0-2}_best`, `part_G_ohemc_s{0-9}_{best,final}`, `part_features.json`, `heldout.safetensors`, `config.json`, `manifest.json`
(seed, rung, snapshot, step, recorded and re-scored exact match, sha256). The HAND net is defined by formula; no HAND weight file is published.

## Prediction before training
Recorded before any rule-label training step. PART ≥ .99 in ≥ 8/10 seeds: held (10/10 per format and setting), exact never predicted.
G box/area/flag tokens ≥ .99: held (1.0 at 120k). G move exact .65–.95 with 2–6/10 seeds ≥ .99 at 30k: broken (.72–.86, 0/10); it
needed 4x the budget. A move exact .5–.8: held (.77–.88). Fence variables linearly decodable (R² ≥ .9): broken (≤ .76; ub/uk ≤ .35).
Interchanging the phase changes the *move* in ≥ 80%: not tested as stated (the .81–.86 interchange flip measured the phase/flag tokens,
not from/to). Win ≥ .8 whenever move exact ≥ .8: broken for A (win 0 at 30k, .53–.60 at 120k with move exact ≥ .97).

## Hand-built reference
The policy confines the black king in a box fenced by the rook and shrinks it. From the three piece coordinates the net forms the fence
frame (box corners, area, rook safety); each rule p has a candidate mask m_pt over the 64 destinations t (safe, exposed, box after the
move, Chebyshev distances) and integer keys. Router: E_p = 1[Σ_t m_pt > 0], priority g_p = ReLU(E_p − Σ_{r<p} E_r) (first eligible
rule), destination = argmax_t of ReLU(10000 g_p + 10000 m_pt + 10000 P_to − 64 key_pt − 20000 − t) (key, then square index breaks
ties). Bricks: one-square indicators, steps [z ≥ c], AND / AND-NOT, |z|, max. Check: evaluating the sparse levels on the piece one-hots
gives 0 wrong tokens on all 175,168 positions (G 9 outputs, A 2); the dense transformer form with the attention front end was not
checked exhaustively. The policy itself mates 175,168/175,168 against every defender reply (worst 48 moves, mean 24.11).

## What training finds
- **PART** (frozen HAND features, trained router; best exact on all held-out positions): baseline A .99343–.99458, G .99646–.99755
  (10/10 ≥ .99 each, ≥ .99 reached at 16.75–19.75k); OHEM from scratch A .9942–.9951, G .9970–.9981 (no gain); OHEM continuation A
  .99657–.99732, G .99829–.99892. Errors sit at the destination; no single hidden unit costs > .5 points. Circuit type (50/50 best
  snapshots, best and final of the 30 selected): one coarse dependency cluster: mean-ablating the frozen rook-line features (rl) or the
  king-distance features (q) each costs ≥ 1 point, removing the raw piece one-hots jointly < 1 point. The router reads the hand-built
  rook rays and king distances, not the board.
- **TRAIN, budget ladder** (best / final, 2,048 held-out): G 30k .722–.859 (10 seeds, 0 ≥ .99), 60k .962–.979 (3, 0), 120k
  .993–.998 / .990–.996 (3, **3/3**); A 30k .765–.876, 60k .929–.950, 120k .970–.977 / .968–.976 (0/3); C 120k .987–.991 / .985–.991
  (1/3). Fresh positions every step: no cell memorises; every shortfall was budget.
- **Scratchpad vs answer only** (120k, equal shape, 3 seeds): first test ≥ .9 at G 63.5–75.8k, C 76.0–80.8k, A 86.3–89.5k; win vs
  tablebase G .85–1.0, C .996–1.0, A .53–.60 (vs random G .97–1.0). Per position (G final): box/area/flag 1.0, phase .996–.999,
  from .9995–1.0, to .996–.998. Writing the candidates (C) moves the remaining errors from 'to' into the candidate list.
- **Diagnostics, G 120k** (best + final, seeds 0–2, 512 rows): exact .998–1.0 / .992–.998. Mean-ablation screen: 23–34 of 72 heads +
  6 MLP blocks used (zero-ablation: 39–49 heads); activation patching: MLP5 drop .98–.99, MLP0 .88–.90, MLP4 .38–.58; logit lens decodes
  the answer only at layer 6. Linear probes: king file/rank R² .85–.99 (layers 2–5), box corners .44–.76, inner fence coordinates ub/uk
  ≤ .35, phase .12–.18; 64-way square probes for the kings ≤ .64 and editing the black-king square direction moves nothing (.001 vs
  full patch .999): **kings are coordinates, not squares**. Interchange at layer 6 flips box/area .91–1.0, phase .81–.86 (DAS .89–.95),
  but these tokens are written in the scratchpad, so the flips partly read the written token. EAP-IG edge scores agree with exact
  patching at spearman .80–.86. No HAND attention-divergence or rank control exists for TRAIN (no dense HAND for these labels).
- **Size:** HAND 4 heads + 6,666 neurons (positional thermometer excluded). Cumulative-ablation sufficient graphs (upper bounds, not
  minima): PART G 0 heads + 4,484 neurons (frozen ancestors included), exact .989; TRAIN G best s1/s2 56 heads + 4,669 neurons and
  63 heads + 4,069 neurons, full held-out exact .987/.986 vs native .996/.995. Measured on the selected seeds only.
- Not finished: final G cell seeds 3–9 (6/7 completed at best .996/.995, final .994/.994, nets not yet harvested; 3–5 and 8–9 rerunning);
  A/C diagnostics; diagnostics of the PART routers with the standard tools (they are not GPT-NeoX nets).

## Closed form
- **PART, phase output** (output position 6). Replace all 512 h3 activations at that position by Σ_p g_p A_p, where g_p is the
  HAND priority gate computed from the frozen features and A_p = mean of h3 over training positions with g_p = 1 (8 × 512 = 4,096
  fitted numbers per snapshot, 1.6% of W3 + b3); W4, b4 and the other positions stay native. On 8,192 fresh held-out draws (6,583
  distinct positions, disjoint from the fit) exact goes from .99780–.99939 (native) to .99805–.99951 (substituted): **20/20 snapshots
  pass** (best and final, seeds 0–9; bar ≥ .99 × native), phase accuracy 1.0; an independent rerun with a different sampling seed
  passed 20/20. This is decoder-level sufficiency: activation R² is only .15–.18, an eligibility-only affine replacement fails
  (.969–.971), and the destination computation is not covered.
- **TRAIN G 120k, last MLP.** Patching names MLP5. An additive model of its whole output in the HAND variables (fence frame, area,
  phase, piece coordinates; 525k fitted numbers, ridge) **fails**: seed 1 best/final .995/.994 → .340/.344 (n = 1,024); A 60k
  .940 → .417. A per-token template for the seven trace tokens only (17,280 numbers) passes (.996/.997) with from/to left native.
  Joint scrubbing of the full proposed explanation gives .034/.082: rejected. No closed form for the destination is verified.

## Weights excerpt and reading
HAND (formula): priority neuron g_p has weight +1 on E_p, −1 on every earlier E_r, bias 0; destination neuron (p, t):
ReLU(10000 g_p + 10000 m_pt + 10000 P_to − 64 key_pt − 20000 − t), so only the active rule's candidates survive and the largest value
is the smallest key, then the lowest square. PART: the source of the move (king or rook) is read off one direction of h3:
king iff h3 · (W4[:, WK] − W4[:, WR]) + b4[WK] − b4[WR] > 0; restricting the readout to that difference keeps exact .99732/.99892
(A/G continuation, all held-out, 100% agreement). A single-unit Boolean form ReLU(9.28 I_APPROACH + .019) for a learned phase unit
lowers exact .998915 → .998858: the router is distributed, not a set of Boolean rows. These numbers are fits and readout slices of the
archived PART weights.

## Comparison
The hand-built net and the trained routers share the inputs that matter (rook rays and king distances, by ablation) and the phase
decoder (template substitution), but not a verified priority or destination computation. Trained-from-scratch nets reach the same
accuracy with a different code: king coordinates linear, fence frame non-linear, a last-layer MLP that is not additive in the HAND
variables. Scratchpad formats learn earlier and play much better at similar token accuracy. With fresh positions every step there is
no memorisation phase: the curves rise smoothly with budget, unlike the delayed generalisation of [Power et al.](https://arxiv.org/abs/2201.02177).
Verifying a learned board model by its behaviour rather than by probes is the approach of [Balogh & Jelasity](https://arxiv.org/abs/2602.05903).

## Controls and limits
Best and final differ by ≤ .006. "Grok" here is the first test ≥ .9, an operational threshold. TRAIN accuracies use a 2,048-row
held-out batch (PART: all held-out positions). The PART "frozen features" bypass the token front end. High probe R² was not taken
as use: the phase is decodable from PART h3 (R² .89–.91) yet its rank-8
interchange flips ≤ .012. No net is exact; length/board-size generalisation does not apply (fixed 8 × 8 board). DTM-label cells (a
different target) memorise or stall (.20–.40) and are not reported as replications. Pending: ten-seed final G cell, A/C 120k
diagnostics, public code, raw substitution logs.

## References
- Köksal et al. (2022). Improved Hard Example Mining Approach for Single Shot Object Detectors. ICIP (conference paper; OHEM). https://doi.org/10.1109/ICIP46576.2022.9897806
- Power et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets (preprint). https://arxiv.org/abs/2201.02177
- Balogh & Jelasity (2026). Verification of the Implicit World Model in a Generative Model via Adversarial Sequences (preprint). https://arxiv.org/abs/2602.05903
- python-chess (software and documentation; legality checks). https://python-chess.readthedocs.io/
