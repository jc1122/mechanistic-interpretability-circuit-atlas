# KRK mate (king + rook vs king): an exact hand-built rule router; a scratchpad net reaches .99, but its circuit is not the hand-built one
**Task.** King + rook vs king, white to move; all 175,168 won white-to-move positions, 10% held out by a fixed permutation (17,516; 157,652
train). Prompt 73 tokens (BOS, side, castling, en passant, 64 squares, `=`). Label: the move of a deterministic 8-rule fence policy (FLEE,
SHRINK, CHECK, WAIT, REPOS, APPROACH, DIVIDE, FAR in priority order; integer keys break ties), not DTM-optimal play. Formats: **A** from, to;
**G** scratchpad fl fh rl rh (box), area, rook-safety flag, phase (active rule), from, to; **C** G plus the active rule's candidates in key
order (padded to 14). Metric: exact match of the whole output, teacher-forced.
**Status.** 2026-11-08. HAND exact. PART 10 seeds per setting. TRAIN 30k (10 seeds), 60k (3), 120k (3 per A/G/C); final G 120k: 7 of 10 seeds.
## Summary
The policy mates from every position (worst case 48, mean 24.1 moves; DTM 9.86) and compiles to 4 heads + 6,666 ReLU neurons. A 3 x 512
router over frozen HAND features reaches .993-.999 (60/60, none exact); the HAND priority replaces its phase output without loss (20/20). From
scratch G grows with budget: 30k .72-.86, 60k .96-.98, 120k **.990-.997 in 7/7 final seeds (none exact)**, win vs tablebase .85-1.0; A .968-.976
(win .53-.60). G nets read kings as coordinates, pick the source from the written phase (closed form, 7/7) and decode 'to' in a non-additive last MLP.
## Setup
[METHODS.md](../METHODS.md). TRAIN: AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000, cosine, clip 1, batch 64 fresh train positions per step,
TF32, eval every 250 steps on a fixed 2,048-row held-out batch, best snapshot, 5k plateau stop after 75%. PART: fp32, batch 256, eval on all
held-out; OHEM keeps the 256 highest-loss of 1,024 rows ([Köksal et al.](https://doi.org/10.1109/ICIP46576.2022.9897806)). Games: greedy, 512
held-out starts, 60-move cap, tablebase and random defenders, legality by [python-chess](https://python-chess.readthedocs.io/). Consumer GPUs.
Nets: [jc1122/circuit-atlas-nets/chess-krk](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-krk) (manifest `recorded_test`, `rescored_test`); training and analysis code not yet in `code/`.
## Architecture as used
TRAIN: GPT-NeoX 6 layers x 12 heads (d_head 32), d_model 384, MLP 1536 ReLU, no LayerNorm, no positional encoding, untied readout, vocab 293,
length A/G/C 75/82/96. HAND: sparse ReLU levels on the piece one-hots (1 counting + 3 gather heads; 2592/662/647/430/1282/137/2/7/907). PART: 3,122 frozen HAND features below the priority level; h1 = ReLU(x W1 + pos[p] + b1), two ReLU layers, logits = h3 W4 + b4;
W1 [3122, 512], W2, W3 [512, 512], W4 [512, 293], pos [9, 512]. Archive `chess-krk/`: `train_{G,C}_s{0-2}_{best,final}`, `train_A_s{0-2}_best`,
`train_G_s{6,7}_final`, `train_G_s{8,9}_{best,final}`, `part_G_ohemc_s{0-9}_{best,final}`, `heldout.safetensors`, `config.json`, `manifest.json`.
## Prediction before training
Recorded before any rule-label step. PART >= .99 in >= 8/10: held (10/10). G box/area/flag >= .99: held (1.0). G move exact .65-.95, 2-6/10
>= .99 at 30k: broken (.72-.86, 0/10; 4x budget gives 7/7). A .5-.8: held (.77-.88). Fence variables linear (R^2 >= .9): broken (<= .76; ub/uk
<= .35). Phase interchange flips the move >= 80%: tested on the phase token only (.81-.86). Win >= .8 at move exact >= .8: broken for A.
## Hand-built reference
Fence frame from the piece coordinates; per rule p a candidate mask m_pt over destinations t and integer keys. E_p = 1[sum_t m_pt > 0],
g_p = ReLU(E_p - sum_{r<p} E_r), to = argmax_t ReLU(10000 g_p + 10000 m_pt + 10000 P_to - 64 key_pt - 20000 - t). 0 wrong tokens (all states).
## What training finds
- PART: G .9965-.9976, OHEM continuation .99829-.99892; A .9934-.9946 / .99657-.99732; >= .99 by 16.8-19.8k. **Ray+distance router** 50/50
  seed-runs: zeroing rook rays or king distances costs >= 1 pt each, raw piece one-hots < 1 pt. Errors at 'to'.
- TRAIN G 120k final: best .993-.998, final .990-.997 (7/7 >= .99, 0 exact); first test >= .9 at 63.5-75.8k; per position box/area/flag 1.0,
  phase .996-.999, from .9995-1.0, to .996-.998. C .985-.991 (errors move into the candidate list); A .968-.976.
- Types (diagnosed G s0-2, C s0-1, 6 A nets): **G coordinate decoder** 3/3 (all 12 G nets 30k-120k): MLP0 kings as file/rank, L5 heads, MLP5
  decodes (patch .98-.99); **phase-conditioned source copy** 7/7 final seeds; **C candidate pointer** 2/3; **A MLP0 + one L0 histogram head** 6/6.

| cell | used | best probe R^2 (layer) | eff. rank | attn JS | best / final |
|---|---|---|---|---|---|
| PART G | 1,334-1,340 of 1,536 units | phase from h3 .89-.91 | W4 phase block 7 of 8 | n/a (no heads) | .9989 / .9987 max |
| TRAIN G 120k | 23-34 of 72 heads + 6 MLPs | kings .85-.99 (L2-5), fl .66-.76 (L4-6), area .56-.62, ub/uk <= .35, phase .12-.18 | MLP ranks 41-239 | U | .993-.998 / .990-.997 |
| TRAIN C 120k | 27-39 heads + 6 MLPs | kings .81-.97, box .33-.73, phase .40-.49 | 75-239 | U | .987-.991 / .985-.991 |
| TRAIN A 120k | 22-31 heads + 6 MLPs | kings .78-1.0, rl <= .37, rest < .3 | U | U | .970-.977 / .968-.976 |
## Closed form
- PART phase: phase = first p with E_p (0 numbers): exact .99829-.99897 vs native .99772-.99892, 20/20; linear in E fails (.968-.970).
- TRAIN G source: from = square(WK) if written phase = APPROACH else square(WR) (0 numbers): 7/7 final seeds keep (.99035-.99623 vs .99018-.99623).
- TRAIN C: z_h(to) = W_V x[candidate 0] at L5H4: 1,024/1,024 kept. G's 'to': an additive HAND-variable model of MLP5 fails (.995 -> .34).
## Weights excerpt and reading
HAND: g_p has +1 on E_p, -1 on each earlier E_r; the -64 key_pt term makes the smallest key win. PART W4 phase block (centred svd, `excerpts.py
krk`): rank 7, s_7/s_1 .47-.53. TRAIN G's source rule needs no weights: it reads the written phase and the piece squares.
## Comparison
Trained nets share HAND's inputs (rays, distances) and phase decoder, not its priority or argmin; kings are coordinates, not squares. Scratchpads
learn earlier and play better; fresh positions per step give no memorisation phase, unlike [Power et al.](https://arxiv.org/abs/2201.02177).
## Controls and limits
Best and final differ by <= .007; "grok" = first test >= .9; length generalisation does not apply (fixed board). Source controls: always the rook .869-.875, shuffled phase .780-.786.
U = unavailable: no dense same-architecture HAND for rule labels (no HAND JS or rank). Final seeds 6-9 are undiagnosed; 3-5 never finished.
## References
Köksal et al. 2022, https://doi.org/10.1109/ICIP46576.2022.9897806 (conference). Power et al. 2022, https://arxiv.org/abs/2201.02177 (preprint).
Jin et al. 2026, https://doi.org/10.48550/arxiv.2604.10158 (preprint; transcoder reading of chess MLPs, the proposed next test).
