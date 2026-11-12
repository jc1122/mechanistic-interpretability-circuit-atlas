# KBBK mate (king + two bishops vs king): unsolved; trained nets learn a distributed board-to-move lookup with no distance-to-mate code
**Task.** White king + opposite-colour bishops vs black king, white to move; all 2,503,608 won positions (bishops identified by square colour; 520
drawn excluded), fixed split 2,253,248 train / 250,360 held out. Prompt 73 tokens (board). Label: the move of a verified hybrid policy: geometric
cage rules (MATE, RESCUE, BARRIER, APPROACH, REGROUP, WAIT) where their closed attractor holds (0.37% of states), else a DTM-optimal move with a
geometric tie-break key (**99.63%**). Formats: **A** from, to; **T** scratchpad phase, cage area, 8 ordered candidate to-squares, from, to.
Metric: whole-output exact, teacher-forced. **Status.** 2026-11-11. TRAIN only, 3 seeds per cell, A r1, T r1, A r2 (width step); **0 of 9 seeds solved.**
## Summary
No cell is solved: A r1 best .58-.63, T r1 .39-.41, A r2 (width 768, cut at 18% of budget) .18-.20; train = test throughout (underfit, no
memorisation, no grokking). No hand-built network exists: the labels are a DTM table on 99.6% of states. Every net computes the move as a
distributed lookup (embedding -> MLP0/MLP3 -> MLP5 readout); no discovered sub-circuit is faithful (faith .05-.69) and the DTM ordering is
not linearly decodable at any layer (R² <= .03). T copies its own written first candidate into the move (causal, 3/3), so its errors are in candidate generation.
## Setup
[METHODS.md](../METHODS.md). AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000, cosine, clip 1, batch 64 fresh train positions per step, TF32,
eval every 500 steps on a fixed 2,048-row held-out screen, best snapshot, 5k plateau stop after 75%. Budgets: A 132k, T 150k steps; A r2 was
wall-capped at 23.5k of 132k. wd .3 and 1 not run. Labels: exhaustive retrograde (0 Bellman discrepancies on 4,194,304 slots) checked against
[python-chess](https://python-chess.readthedocs.io/) (0 move-set mismatches on 3,000 states per side; 150 games per policy mate at the certified ply).
The hybrid mates every won state against every defence (max 22, mean 14.37 moves; DTM max 19). Consumer GPUs. Nets: [jc1122/circuit-atlas-nets/chess-kbbk-mate](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-kbbk-mate); training code not in `code/`.
## Architecture as used
GPT-NeoX 6 layers x 12 heads, no LayerNorm, no positional encoding, ReLU MLP 4 x width, untied readout, vocab 280, length A 75 / T 85. r1: d_model 384
(d_head 32, MLP 1536; 72 heads + 9,216 neurons); A r2: d_model 768 (d_head 64, MLP 3072; 18,432 neurons); all weights trained. Archive `chess-kbbk-mate/`: `train_{A,T}_r1_{best,final}_s{0-2}`, `train_A_r2_final_s{0-2}` (step 22,500), `screen_{A,T}`, `config.json`, `manifest.json`.
## Prediction before training
Recorded before TRAIN: A .3-.7 (held: .58-.63), T .4-.8 (broken: .39-.41), T phase near 1 (held: .98-1.0), T candidate0 ~ move with no lift
over A (held), budget-limited and still rising at the end (broken for A r1: plateau; not recorded for T, A r2), wall-cap truncation recorded as such (held for A r2).
## Hand-built reference
none (unsolved). The geometric rule list certifies only 9,257 of 2,503,608 won states within 50 moves; a formula-weight network for the
remaining labels would be a 2.5M-state DTM table, so no HAND, PART or closed-form comparison exists.
## What training finds
| Cell | Seeds solved | Best exact (s0/s1/s2) | Final exact | Train (final) | Stop | Failure class |
|---|---|---|---|---|---|---|
| TRAIN A r1, d 384 | 0/3 | .627/.590/.582 | .620/.586/.575 | .63/.61/.62 | plateau 130.5k | can't fit |
| TRAIN T r1, d 384 | 0/3 | .410/.394/.407 | .402/.391/.405 | ≈ test (gap <= .03) | ran to 146k | can't fit |
| TRAIN A r2, d 768 | 0/3 | .190/.195/.178 | .171/.175/.149 (22.5k) | .19/.18/.15 | wall cap 23.5k | can't fit |
Width check at matched step 23.5k: d 384 .183-.217, d 768 .149-.184 (no gain at a fixed lr). Circuit types: one in 9/9 nets, "distributed
lookup"; T adds a causal copy of candidate0 into from/to (3/3). Diagnostics (128 held-out rows per snapshot; variables = symbolic policy quantities):
| Snapshot | exact | used heads + MLPs | best probe R² (layer) | eff. rank L6 r90 (vs HAND) | attn JS | discover faith / complement |
|---|---|---|---|---|---|---|
| A r1 best s1/s2 | .55/.50 | 15+6 / 12+6 | dtm_plies, eligible, optimal_count, king_distance <= .03; cage_u_hi .30-.40 (L2-3) | 49/46 (no HAND) | N/A | - |
| A r1 final s0/s1/s2 | .55/.55/.53 | 22/16/16 + 6 | same; policy_to <= .08; cage_v_hi .40 (L6, s1) | 59/49/46 | N/A | .27/.69/.41 / pass |
| T r1 best s0/s1/s2 | .41/.42/.49 | 14/10/31 + 6 | policy_to .79/.71/.40 (L6); dtm_plies .16-.45 (L6); candidate_to < 0 | 52/44/62 | N/A | - |
| T r1 final s0/s1/s2 | .43 (s1) | 11+6 (s1) | policy_to .70 (L6, s1) | 44 (s1) | N/A | .05/.08/.10 / pass |
| A r2 final s0/s2 (s1) | .16/.24 (.16) | 2H+2M / 2H+4M | every var <= .30; DTM vars <= .021 all layers | 46/42 | N/A | .44/.33 (s0/s1); complement .75/.48 |
Used = single-component mean-ablation drop > .05; neurons not counted. Logit lens: exact <= .06 through L5, .16-.55 only after MLP5. Interchange:
every A variable untested (nf 0); T policy_to at L6 flips 1.0 (nf 63, 3/3), also on wrong rows (bucket): the break is upstream, in candidate0.
## Closed form
none (unsolved): no faithful circuit to substitute; an affine geometry substitution (44 features, one MLP) kept >= 99% of baseline in 0/12 snapshots.
## Weights excerpt and reading
Per final net, MLP5 -> destination logits U[SQ] W_out: 90%-mass rank A r1 21-22/64, T r1 42/64, A r2 17/64; neurons for 50/90% of readout mass A r1 185-211/399-457 of 1,536, T r1 537-551/1130-1164 of 1,536, A r2 180-210/1119-1247 of 3,072. Reading: no sparse readout; T spreads it over ~3x A's neurons.
## Comparison
No hand-built algorithm to compare. Unlike KRK (a compact 8-rule policy, solved .99 with a scratchpad), DTM-optimal KBBK labels admit no compact
intermediate; the scratchpad hurts whole-output exact (T < A). Consistent with tablebase-trained engines that still misplay endgames.
Answer-only A 0/6 (72 heads + 9,216 / 18,432 neurons) vs scratchpad T 0/3 (72 + 9,216); no grokking in either; length N/A (fixed board).
## Controls and limits
3 seeds per cell; wd .1 only (sweep not run); T log lost after 67k (nets kept); r2 cut at 18% of budget, so the width step is untested at full
schedule and a width-scaled lr was never tried. Missing diagnostics: A r2 s1 ablation set, s2 discover/dict, checkpoint dynamics. Exact is
teacher-forced on 2,048 held-out rows; free play unmeasured. Length generalisation N/A (fixed 8x8 board).
## References
- Ruoss et al. 2024, Amortized Planning with Large-Scale Transformers: A Case Study on Chess (preprint): https://doi.org/10.48550/arxiv.2402.04494; Haworth 2019, Chess endgame news: an endgame challenge for neural nets (journal article): https://doi.org/10.3233/icg-190109
- de Brito Neto et al. 2023, Evaluating Deep Learning-based Chess-Engine Endgame Strategies (conference paper): https://doi.org/10.5753/sbgames_estendido.2023.233983
- Haque et al. 2022, On the Road to Perfection? Evaluating Leela Chess Zero Against Endgame Tablebases (conference paper): https://doi.org/10.1007/978-3-031-11488-5_13; Kaplan et al. 2020, Scaling Laws for Neural Language Models (preprint): https://doi.org/10.48550/arxiv.2001.08361
- Kim et al. 2026, Let's Scale Step by Step: Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts (preprint; lr shifts with width without parameterised transfer): https://doi.org/10.48550/arxiv.2608.20061
