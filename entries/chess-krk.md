# KRK mate (king + rook vs king)

A king and rook can trap a lone king by shrinking the space it can use. A six-layer transformer that wrote the box around the defender and the active rule before its move reached .990–.997 whole-output accuracy in all 7 completed final runs on 2,048 held-out positions, with correct earlier output tokens supplied; none was exact and 3 launched runs never finished.

The written rule suffices to choose whether the king or rook moves, but the destination remains unexplained.

The picture carries the supplied strategy step, called its phase, along a shared line of running internal values ([residual stream](../GLOSSARY.md#standard-machine-learning-terms)).

## The circuit at a glance

Figure S1 shows the phase-to-source copy bypassing the destination decoder, whose coordinate-based rule remains unknown.

<img src="../figures/circuits/chess-krk.svg" width="640" alt="One residual stream retains board and supplied strategy-phase information. The source copy reads that phase directly: APPROACH selects the white king's square, otherwise the white rook's. Independently, early neurons encode king file/rank, late layer-5 heads read coordinates, and layer-5 neurons decode an unresolved destination. The source path bypasses destination decoding; both contribute to the move. Learned roles are dashed: source substitution passes but complement is untested and the additive destination fit fails. The phase example asserts no board move.">

*Figure S1 shows the written phase selecting the source independently of the unresolved coordinate-based destination, differing from the hand-built priority/key selector (strategy-scratchpad networks trained from scratch, seven completed final seeds); status is Partial because source substitution passes but the complement is untested and the additive destination fit fails, with APPROACH illustrating a supplied phase rather than a legal board move.*

| Fact | Value |
|---|---|
| Task | In a king + rook vs king position with white to move, write the move chosen by a fixed 8-rule fence policy (not the tablebase-optimal move) |
| Input → output example | In words: the 64-square board with the three pieces → the policy's move as its from square and to square (answer only) |
| Model | From scratch: 6-layer transformer, 12 heads per layer, width 384, feed-forward width 1536; on frozen features: a 3 × 512 feed-forward router reading 3,122 inputs |
| Rungs and seeds | Hand-built network: 4 heads + 6,666 neurons, exact on all states; frozen-feature router: 10 seeds per setting; from scratch: 30k (10 seeds), 60k (3), 120k (3 per output format); final strategy scratchpad: 7 of 10 launched seeds completed |
| Closed form status | Partial (phase rule keeps accuracy in 20/20 router snapshots; source-square rule in 7/7 final scratchpad seeds on all 17,516 held-out positions; complement test missing in 7/7); destination model fails (.995 → .34) |
| Status | Unsolved (final strategy scratchpad, 0/7 exact on 2,048 held-out positions; 7/7 ≥ .99, 3 unfinished); last run 2026-11-08; diagnostic update 2026-11-09 |
| Weights | [Hugging Face: chess-krk](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-krk) |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

These networks learned to imitate a fixed 8-rule version of that strategy, rather than the fastest move from an exhaustive endgame table.

The task is unsolved at the final audit's exact criterion despite 7/7 passing the separate ≥ .99 milestone: the output formulas are [partial](../GLOSSARY.md#verifying-a-circuit), supported by replacement, while failed diagnostic jobs left no test showing that activity outside the proposed circuit can be changed without harming accuracy.

## 1. Task and data

The data are all 175,168 won king + rook vs king positions with white to move. A fixed permutation holds out 10% (17,516 positions) and
leaves 157,652 for training. The prompt has 73 tokens: BOS, side to move, castling, en passant, the 64 squares and `=`.

The label is the move of a deterministic fence policy with 8 rules in priority order: FLEE, SHRINK, CHECK, WAIT, REPOS, APPROACH, DIVIDE
and FAR. Integer keys break ties between candidate moves. The policy is not optimal play by depth to mate (DTM): it mates from every
position in at most 48 moves (mean 24.1) against a mean DTM of 9.86.

An output counts as correct only when every token matches the label. Scoring is
[teacher-forced](../GLOSSARY.md#task-formats): each output token is given the correct earlier tokens, rather than the network's own
previous outputs. The ≥ .99 whole-output milestone and the final audit's stricter solved-seed criterion of 1.0 are reported separately.
No seed met the latter; the earlier partly-solved headline described the ≥ .99 milestone. Length generalisation
does not apply, because the board is fixed. Games were played greedily from 512 held-out starts with a 60-move cap, against a tablebase defender (one that plays optimally from an exhaustive endgame table)
and random defenders, with legality checked by python-chess ([Fiekas, n.d.](#references)).

*Table 1. Output formats.*

| Format | Output |
|---|---|
| `A` (answer only) | from, to |
| `G` (strategy scratchpad, intermediate results before the move) | the fence box fl fh rl rh, area, rook-safety flag, phase (the active rule), then from, to |
| `C` (scratchpad with candidates) | G plus the active rule's candidate moves in key order (padded to 14) |

A [scratchpad](../GLOSSARY.md#task-formats) is the network's written intermediate computation. Here the box describes the fence
around the defending king; the phase names the policy rule that applies. The labels fl, fh, rl and rh are the box-boundary variables
of the hand-built policy: fl/fh are the low/high file bounds and rl/rh the low/high rank bounds. Files are columns and ranks are rows
on the board. In the fence's own coordinates, v measures distance across the rook's line and u position along it; ub and uk are the
black and white kings' u coordinates. These labels match the archived diagnostics. A move's source is the square it starts
from; its destination is the square it ends on. The fact-table example describes this format in words rather than inventing a move
without a board position.

## 2. Model

The shared conventions are described in [METHODS.md](../METHODS.md). The three [rungs](../GLOSSARY.md#the-three-rungs) are HAND (every
weight set by formula), PART (frozen hand-built parts with one trained stage) and TRAIN (all weights trained from scratch).

### 2.1 Architecture

TRAIN networks follow the [GPT-NeoX decoder layout](../GLOSSARY.md#standard-machine-learning-terms), with 6 layers × 12 attention
heads (readers of earlier positions). Each head has width d_head 32; the running residual vector has width d_model 384. Each
feed-forward block (MLP) has 1536 ReLU units, which set negative inputs to zero. The networks have no LayerNorm, no positional
encoding and an untied readout, with a vocabulary of 293 tokens; sequence length is 75, 82 and 96 for A, G and C. HAND is a stack of
sparse ReLU levels on the piece one-hots with 1 counting head and 3 gather heads; the levels have 2592/662/647/430/1282/137/2/7/907
neurons. The total sparse HAND circuit has 4 heads and 6,666 ReLU neurons.

### 2.2 Frozen parts and trained part

PART reads the 3,122 frozen HAND features below the priority level and replaces the priority and argmax levels by a trained router:
h1 = ReLU(x W1 + pos[p] + b1), two further ReLU layers, and logits = h3 W4 + b4, with W1 [3122, 512], W2 and W3 [512, 512], W4 [512, 293]
and pos [9, 512], where p is the output position.

### 2.3 Training

TRAIN used AdamW with learning rate 1e-3, weight decay .1, betas (.9, .98), warmup 1000 steps, a cosine schedule and gradient clip 1.
Each step drew a batch of 64 fresh training positions, in TF32. Test accuracy was evaluated every 250 steps on a fixed 2,048-row
held-out batch; the best snapshot was kept, with a plateau stop after 5k steps without improvement once 75% of the budget had passed.
PART ran in [fp32](../GLOSSARY.md#hand-built-machinery) with batch 256 and was evaluated on all held-out positions. Its continuation
used online hard example mining (OHEM), which keeps the 256 highest-loss rows of 1,024 ([Koksal et al., 2022](#references)). Training
ran on consumer GPUs.

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-krk](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-krk)
(manifest keys `recorded_test` and `rescored_test`).

### 2.5 Archive

The Hugging Face folder `chess-krk/` holds `train_{G,C}_s{0-2}_{best,final}`, `train_A_s{0-2}_best`, `train_G_s{6,7}_final`,
`train_G_s{8,9}_{best,final}`, `part_G_ohemc_s{0-9}_{best,final}`, `heldout.safetensors`, `config.json` and `manifest.json`.

## 3. Hand-built solution

The hand-built network computes the fence frame from the piece coordinates, then, for each rule p, a candidate mask m_pt over
destination squares t together with integer keys key_pt (Table F1). A rule applies when it has a candidate, E_p = 1[Σ_t m_pt > 0]. The
priority level keeps the first applicable rule, g_p = ReLU(E_p − Σ_{r<p} E_r). The destination is

to = argmax_t ReLU(10000 g_p + 10000 m_pt + 10000 P_to − 64 key_pt − 20000 − t),

so the smallest key among the active rule's candidates wins. P_to flags the output position at which the destination is written. The network writes 0 wrong tokens on all states.

Table F1 records the fixed-feature policy and router alternative. Piece indicators produce the fence frame, each rule's candidate mask and keys, and flags saying which rules apply. The first applicable rule and its smallest key select the move. PART instead trains a router on those frozen features to write phase, source and destination; it does not train the feature construction.

<details>
<summary>Technical fixed policy and trained-router alternative (Table F1)</summary>

*Table F1. The hand-built network and PART alternative. The frozen levels compute the fence frame, each rule's candidates and keys, and whether each rule applies; the priority and argmax levels pick the move. The router is the trained part of PART; it reads the frozen features below the priority level in place of the priority and argmax levels.*

| Component or alternative | Input and operation |
|---|---|
| Fixed coordinate gathering | Piece indicators (one-hots) on 64 squares feed 1 counting head and 3 gather heads. Piece coordinates define the fence frame. |
| Fixed per-rule features | For rule p, the frame gives candidate mask m_pt and keys key_pt. E_p = 1 if rule p has a candidate. |
| Fixed priority and destination selection | Compute g_p = ReLU(E_p − sum of earlier E_r). Read g_p together with the candidate masks and keys; to = argmax over t, with the smallest key winning. |
| PART router, trained | A 3 × 512 ReLU router reads 3,122 frozen features, including candidates, keys and E_p. It replaces priority and selection, writing phase, from and to. |

</details>

## 4. What training finds

### 4.1 PART

A router is the trained feed-forward network that selects the active rule and move from the supplied frozen features. The router on frozen features reached G .9965–.9976 and, with the OHEM continuation, .99829–.99892; A reached .9934–.9946 and, with the OHEM
continuation, .99657–.99732. Test accuracy reached ≥ .99 by 16.8–19.8k steps; in total .993–.999 in 60/60 seed-runs, none exact. The
router is a ray and distance router in the 50/50 diagnosed seed-runs: zeroing the rook rays (squares along the rook's lines) or the
king distances costs ≥ 1 percentage point each, while zeroing the raw piece one-hots costs < 1 point. These 50 diagnosed seed-runs
are a subset of the 60/60 outcome runs, not 60 independent seeds. The remaining errors are at the `to` token.

### 4.2 TRAIN

From scratch, the scratchpad format G improves with budget: 30k steps give .72–.86, 60k .96–.98, and 120k .990–.997 in 7/7 final
seeds (none exact), with a win rate against the tablebase defender of .85–1.0. Answer only (A) reached .968–.976 (win .53–.60).
Scratchpads learn earlier and play better. Fresh positions at every step give no memorisation phase, unlike the grokking runs of
[Power et al. (2022)](#references).

In the 120k G final (seeds 0–2 and 6–9; seeds 3–5 never finished), best test is .993–.998 and final test .990–.997 (7/7 ≥ .99, 0 exact). Test accuracy first passed .9 at
63.5–75.8k steps. Per position, box, area and flag are 1.0, phase .996–.999, from .9995–1.0 and to .996–.998. Format C reached
.985–.991, with its errors moved into the candidate list; A reached .968–.976.

*Table 2. Results per cell.*

| Cell | Seeds | Best / final test | ≥ .99 |
|---|---|---|---|
| PART G | 10 per setting | .9965–.9976; OHEM continuation .99829–.99892 | by 16.8–19.8k |
| PART A | 10 per setting | .9934–.9946; OHEM continuation .99657–.99732 | by 16.8–19.8k |
| TRAIN G 30k | 10 | .72–.86 | 0/10 |
| TRAIN G 60k | 3 | .96–.98 | |
| TRAIN G 120k final | 7 of 10 | .993–.998 / .990–.997 | 7/7, 0 exact |
| TRAIN C 120k | 3 | .987–.991 / .985–.991 | |
| TRAIN A 120k | 3 | .970–.977 / .968–.976 | |

### 4.3 Circuit types

The original diagnostic set covered G seeds 0–2, C seeds 0–1 and 6 A networks. A later diagnostic pass added G seeds 6–9, extending
the coordinate-decoder evidence to all 7/7 completed final G seeds. Four circuit types were found (Table 3; diagnostics and failed
verification jobs in Appendix B). The trained networks share HAND's inputs (rays, distances) and its phase decoder, but not its priority or argmin, and they code the
kings as coordinates rather than squares.

*Table 3. Circuit types.*

| Type | Seeds | Description |
|---|---|---|
| G coordinate decoder | Original 3/3 diagnosed final seeds (same type across all 12 earlier G nets, 30k–120k); later 4/4 added, now 7/7 final seeds | MLP0 codes the kings as file and rank, then layer-5 heads, then MLP5 decodes (patch .98–.99) |
| Phase-conditioned source copy | 7/7 final seeds | the source square is copied according to the written phase |
| C candidate pointer | 2/3 | a layer-5 head points at the first candidate |
| A histogram | 6/6 | MLP0 + one layer-0 histogram head |

## 5. Mechanism and verification

### 5.1 Closed forms

Table 4 lists the substitutions. The PART phase rule and the TRAIN G source rule have 0 fitted numbers.

*Table 4. Substitutions: whole-output accuracy with a formula against the unmodified (native) network. TRAIN G uses all 17,516
held-out positions, unlike the 2,048-position training evaluation. The source-square formula changes one output while the other 8
remain native. PART counts snapshots; TRAIN G counts final seeds.*

| Component | Closed form | Substituted vs native | Kept |
|---|---|---|---|
| PART phase | phase = first p with E_p | .99829–.99897 vs .99772–.99892 | 20/20 |
| PART phase, alternative | linear in E | .968–.970 | fails |
| TRAIN G source | from = square(WK) if the written phase is APPROACH, else square(WR) | .99035–.99623 vs .99018–.99623 | 7/7 final seeds |
| TRAIN C pointer | z_h(to) = W_V x[candidate 0] at L5H4 | 1,024/1,024 kept | |
| TRAIN G `to` | additive model of MLP5 on HAND variables | .995 → .34 | fails |

Key: WK, WR = white king, white rook; L5H4 = layer 5, head 4; z_h = the head's output.

The PART phase rule and the TRAIN G source rule are supported by substitution; no
[resample ablation of the complement](../GLOSSARY.md#verifying-a-circuit) is available, so the closed-form status is partial. This
would replace everything outside the claimed circuit with its values on other inputs; failed discovery jobs produced no such test
in any of the 7 final seeds. Two controls
test the source rule: always copying the rook gives .869–.875, and a shuffled phase gives .780–.786. How G decodes the destination
square stays open, because the additive model fails (Table F2).

Table F2 records the separate source and destination routes shown in the worked circuit above. Trained king-coordinate features feed late heads and the destination decoder. A separate source-copy description uses the written phase to choose the white king or rook square. Status remains partial: source substitution passes with complement untested, while an additive destination formula fails. The source rule is not a complete destination mechanism.

<details>
<summary>Technical source rule and unresolved destination (Table F2)</summary>

*Table F2. The partial mechanism in TRAIN G (120k budget). The components were trained; the source rule describes one output. The source rule keeps whole-output accuracy in 7/7 final seeds under substitution on 17,516 held-out positions; the complement was not tested. MLP5 has no accepted destination closed form: an additive model fails (from .995 to .34).*

| Separate route | Recorded computation and limit |
|---|---|
| Destination | Board tokens feed MLP0's king file/rank features, then layer-5 heads and MLP5. MLP5 decodes to, but is not additive in HAND variables and has no accepted destination formula. |
| Source | Board tokens and the written phase supply the rule: from = square of WK if phase is APPROACH, else square of WR. WK is the white king; WR is the white rook. This source-copy rule is not a complete destination mechanism. |

</details>

### 5.2 Relation to the hand-built network and to published work

The trained networks share HAND's inputs and phase decoder; they do not reproduce HAND's priority and key argmin, and they code the
kings as coordinates. A transcoder, a sparse approximation to a feed-forward block that makes its features easier to inspect, as in [Lin et al. (2026)](#references), is the proposed next test for
G's destination decoder.

## 6. Limitations and open questions

Best and final differ by ≤ .007. "Grok" here means the first evaluation with test ≥ .9
([glossary](../GLOSSARY.md#counting-results)). Some diagnostics are unavailable (U in Appendix B), because there is no dense
hand-built network of the same architecture for the rule labels, and so no HAND JS or rank. Later diagnostics cover all 7 completed final seeds, superseding the earlier absence for seeds 6–9, but automatic
discovery and the complement, dictionary and bucket tests remain missing in 7/7 after stalls, timeouts and memory failures. Seeds 3–5
never finished. No seed is exact, and the destination algorithm remains open. Probes and interchange at late scratchpad positions
partly read already-written box and phase tokens; their causal use does not by itself explain how the hidden computation produced
them.

## 7. Prediction record

The predictions were recorded before any rule-label step (Table 5).

*Table 5. Predictions and outcomes.*

| Prediction | Outcome |
|---|---|
| PART ≥ .99 in ≥ 8/10 | held (10/10) |
| G box/area/flag ≥ .99 | held (1.0) |
| G move exact .65–.95, 2–6/10 ≥ .99 at 30k | broken (.72–.86, 0/10; 4x budget gives 7/7) |
| A .5–.8 | held (.77–.88) |
| Fence variables linear (R² ≥ .9) | broken (≤ .76; ub/uk ≤ .35) |
| Phase interchange flips the move ≥ 80% | tested on the phase token only (.81–.86) |
| Win ≥ .8 at move exact ≥ .8 | broken for A |

## Appendix A. Weights excerpt and reading

In HAND, g_p has +1 on E_p and −1 on each earlier E_r, and the −64 key_pt term makes the smallest key win. The PART W4 phase block,
read by a centred singular value decomposition (SVD, which separates a matrix into independent weighted directions;
`excerpts.py krk`), has rank 7 (7 of 8 phases), with s_7/s_1 .47–.53. Here s_7/s_1 is the seventh singular value divided by the largest. TRAIN G's source rule needs no weights: it
reads the written phase and the piece squares.

## Appendix B. Detailed evidence

*Table B1. Diagnostics per cell.*

| cell | used | best probe R² (layer) | eff. rank | attn JS | best / final |
|---|---|---|---|---|---|
| PART G | 1,334–1,340 of 1,536 units | phase from h3 .89–.91 | W4 phase block 7 of 8 | n/a (no heads) | .9989 / .9987 max |
| TRAIN G 120k | 23–34 of 72 heads + 6 MLPs | kings .85–.99 (L2–5), fl .66–.76 (L4–6), area .56–.62, ub/uk ≤ .35, phase .12–.18 | MLP ranks 41–239 | U | .993–.998 / .990–.997 |
| TRAIN C 120k | 27–39 heads + 6 MLPs | kings .81–.97, box .33–.73, phase .40–.49 | 75–239 | U | .987–.991 / .985–.991 |
| TRAIN A 120k | 22–31 heads + 6 MLPs | kings .78–1.0, rl ≤ .37, rest < .3 | U | U | .970–.977 / .968–.976 |

Key: used = components whose ablation drops accuracy; R² = best linear [probe](../GLOSSARY.md#analysis-measures) (layer); fl, rl and
ub/uk are fence-frame variables of HAND; eff. rank = effective rank; attn JS = attention divergence to HAND; U = unavailable (no dense
HAND of the same architecture). The compact TRAIN G row describes the original diagnosed seeds 0–2, not the later full final cohort.

*Table B2. Later final-seed diagnostics on 512 positions, copied from the method-v2 results. MLP5, MLP0 and MLP4 identify feed-forward
blocks by layer. Patch drop is the fraction of the clean-to-corrupted effect caused by corrupting that component alone
([glossary](../GLOSSARY.md#analysis-measures)); it is not an accuracy. A used component changes accuracy under ablation.*

| Seed | Whole-output accuracy | Used heads and MLPs | MLP5 / MLP0 / MLP4 patch drop | Edge check, jobs b / f |
|---|---|---|---|---|
| 6 | .9941 | 38/72 heads + 6 MLPs | .983 / .901 / .417 | .728 / .761 (fail) |
| 7 | .9961 | 36 heads + 6 MLPs | .989 / .931 / .428 | .886 / .841 |
| 8 | .9961 | 27 heads + 5 MLPs (no MLP1) | .980 / .873 / .465 | .846 / .844 |
| 9 | .9961 | 28 heads + 6 MLPs | .981 / .928 / .506 | missing / .838 |

The edge check compares attribution scores with exact single-edge interventions; values below .8 fail the check
([glossary](../GLOSSARY.md#analysis-measures)). They must not be read as circuit certificates. For the four added seeds, the best
linear readout of king coordinates has R² .74–.99, while replacing all final-layer heads by their means gives exact match 0 in 4/4.
Those interventions support the coordinate-decoder type, without recovering its destination algorithm.

Automatic circuit discovery, complement resampling, sparse-dictionary analysis and error-bucket diagnostics are missing in 7/7 final
seeds. The jobs reached edge attribution, then stalled, timed out or ran out of memory while discovering circuits in a 2764-edge
graph. Thus there is no discovered circuit size, measured held-out faithfulness, complement result or error-bucket diagnosis.
The available dynamics checkpoints were all before the 63.5–75.8k improvement window, so they do not explain that improvement.

## What to read next

Compare [king and two bishops versus king](chess-kbbk-mate.md) for a different label policy and a harder policy-imitation result. [Mate-in-one selection](chess-mate-in-one.md) separates a local move-selection computation from endgame play.

## References

- Fiekas, N. (n.d.). python-chess: a chess library for Python (software). *GitHub*. https://github.com/niklasf/python-chess;
  https://python-chess.readthedocs.io/ (legality checks in games)
- Koksal, A., Tuzcuoglu, O., Ince, K. G., Ataseven, Y., & Alatan, A. A. (2022). Improved hard example mining approach for single shot
  object detectors. *2022 IEEE International Conference on Image Processing (ICIP)*. https://doi.org/10.1109/ICIP46576.2022.9897806
  (online hard example mining)
- Lin, R., Jin, Z., Zhou, G., et al. (2026). Tracing the thought of a grandmaster-level chess-playing transformer. *arXiv preprint*.
  https://doi.org/10.48550/arxiv.2604.10158 (transcoder reading of chess MLPs, the proposed next test)
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization beyond overfitting on small
  algorithmic datasets. *arXiv preprint*. https://arxiv.org/abs/2201.02177 (grokking with a memorisation phase)
