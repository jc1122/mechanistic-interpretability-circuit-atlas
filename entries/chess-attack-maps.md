# Chess attack maps

A rook threatens squares along a straight line until another piece blocks it; an attack map collects such threats from every piece of a chosen colour, including friendly-occupied squares. Four-layer transformers learned this task more often when they first listed attacks piece by piece as written working, a [scratchpad](../GLOSSARY.md#task-formats), before drawing the final map. On the recorded draw of 2048 held-out full boards per run, 7/10 networks trained from scratch reached whole-output accuracy 1.0, reported to three decimals, and 10/10 reached ≥ .998, with correct earlier output tokens supplied.

The task is partly solved: on sparse boards containing sliding pieces only 1/10 reached recorded 1.0, and no tested answer-only setting reached 1.0. The proposed formula was [rejected](../GLOSSARY.md#verifying-a-circuit) both when it replaced the learned computation and when activity outside its proposed circuit was changed; the shipped archive also uses a different test-board draw from the headline results.

## At a glance

| Fact | Value |
|---|---|
| Task | Mark the 64 squares attacked by the requested colour, including friendly-occupied squares. Exact means every scored output token. |
| Input → output example | A white knight alone on a1 attacks b3 and c2; only those squares are marked for White. |
| Model | Fully trained four-layer decoder; full boards use width 256, sparse slider boards width 512. |
| Rungs and seeds | HAND fixes weights; PART retrains attention and MLP stages; TRAIN learns all weights ([rungs](../GLOSSARY.md#the-three-rungs)). Screening 3–10 seeds per cell; final TRAIN 10 seeds each for full boards and sparse slider boards. |
| Closed form status | Rejected (uniform-head replacement leaves exact match 0 in all 13 tested snapshots; resampling outside the proposed circuit leaves 0 on three tested seeds). |
| Status | Partly solved (recorded-test full-board hint TRAIN 7/10 at 1.0, 10/10 ≥ .998; slider hint TRAIN 1/10 at 1.0, 3/10 ≥ .99; 2048 boards/seed, scores to three decimals); last run analysis 2026-11-09; archive check 2026-11-10. |
| Weights | [Hugging Face: chess-attack-maps](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-attack-maps), published revision `3919dc14f8ab`; recorded and archived test draws differ. |
| Reproduce | Not available yet: training code for this task is not yet in `code/`. |

## The circuit at a glance

Figure S1 separates the late piece-token and board-square reads from the neuron write; these roles occur at different token queries, and the dashed routes do not establish a square-by-square algorithm.

<img src="../figures/circuits/chess-attack-maps.svg" width="640" alt="The board has White knight a1 and king h1, and Black king h8; White is requested. Independently supplied correct past scratchpad tokens include square b3. Early weighted counts and necessary middle layers supply context. Two layer-3 read roles are separated: at a direction marker heads read the piece token, and at a piece/square token heads read the named board square. In this task illustration the piece is a knight and b3 is empty. Neurons, not a next-square head, write steps and map bits; the targets include listing b3 and marking its bit 1. These are roles at different token queries, not two recovered exact heads or a measured b3 output. All learned routes are dashed because uniform/dominant-key substitution and complement tests fail.">

*Figure S1 shows marker-to-piece and square-token-to-board reads feeding neuron writes, rather than the hand-built next-square head (type T1: ten full-board and four slider hint-scratchpad networks trained from scratch with recorded teacher-forced exact match ≥ .93; a marker denotes a direction, and the two read roles occur at different token queries); status is Rejected because uniform/dominant-key substitutions and complement tests fail, so this is not a recovered formula, correct past tokens are supplied, and the White-knight example’s empty b3 and attacked bit 1 are task targets, not measured outputs.*

## 1. Task and data

The label is a 64-bit attack map for colour `c`, checked against [Fiekas (n.d.)](#references), the python-chess software documentation.
A bit marks whether that square is attacked, including friendly-occupied squares. Scoring is teacher-forced: each next output token is
predicted after the correct preceding tokens ([glossary](../GLOSSARY.md#task-formats)); it is not a free-running ray walk. For example, a white knight alone on a1 attacks b3 and c2; those are the only marked squares when White is requested.

*Table 1. Board distributions and output formats. Exact match means every scored output token is correct.*

| Label | Meaning |
|---|---|
| C1 | One piece; all 768 (piece, square) cases, split 384/384 per seed. |
| C2 | 1–9 sliding pieces of random colour; fresh training boards every batch. |
| C3 | Both kings and random armies; fresh training boards every batch. |
| A | Map only. |
| S | Map, then a square list. |
| T | Square list only. |
| H | Hint scratchpad, then the map: for each colour-c piece, its square, then a marker for each direction and its attacked squares nearest-first. |

C2/C3 tests contain 2048 fixed fresh boards per seed. The archived test files and the recorded training tests are different draws from this distribution (Section 6). Longer-scratchpad and other board-distribution tests are separate from the in-range result.

## 2. Model

### 2.1 Architecture

All decoders use [ReLU](../GLOSSARY.md#standard-machine-learning-terms), which clips negative values to zero, no positional embeddings (NoPE), and no LayerNorm. The TRAIN rung trains every weight from scratch; HAND fixes weights by formula; PART retrains a stage in the fixed network ([rungs](../GLOSSARY.md#the-three-rungs)).

*Table 2. Architectures as used. Width is the residual-stream dimension; head width is each head's query/key/value dimension.*

| Network | Layers | Width | Heads × head width | MLP width | Vocabulary / positions |
|---|---|---|---|---|---|
| TRAIN C3 H | 4 | 256 | 8 × 32 | 1024 | H: 184 / 323 |
| TRAIN C2 H | 4 | 512 | 8 × 64 | ReLU MLP | H: 184 / 323 |
| HAND A/S/T | 2–3 | 792 | 12 × 66 | 512 | 84 |
| HAND H | 3 | 1122 | 17 × 66 | 768 | H vocabulary |

### 2.2 Frozen and trained parts

A/S/T PART re-initialises L1 attention and its MLP, with QKV shape [2376, 792] and MLP matrices [512, 792] and [792, 512]. The frame, embedding and readout stay frozen. H PART trains L1–L2. H TRAIN trains the complete network. Layer labels start at L0; probe labels L1–L4 refer to the residual after each layer.

### 2.3 Training

AdamW used learning rate 1e-3, weight decay .1, betas (.9, .98), warmup 1000, cosine decay over 60k steps, clipping at 1, batch 64 (microbatch 16), evaluation every 250 steps and a plateau stop of 5000 after 45k. PART used fp32; TRAIN used TF32 ([precision](../GLOSSARY.md#hand-built-machinery)). One GPU handled each pack of 2–5 seeds. The legacy final logs did not record machine or device identity; final weight decay .1 was not swept.

Screening A/S/T budgets were PART 8k and TRAIN ≤ 25k. H TRAIN first ran for 20k, then 60k. Final seeds 10–19 used C3 H width 256 and C2 H width 512. Diagnostics used 32 boards; substitution used 128.

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded accuracies are on Hugging Face at [chess-attack-maps](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-attack-maps). The recorded training invocation is an archival setting, not a runnable command in this repository:

```text
python train.py --rung C3 --fmt H --kinds train --seeds 10 11 12 13 14 --steps 60000 --min_steps 45000 --plateau 5000 --ckpt_every 2500 --lr 1e-3 --wd 0.1 --warmup 1000 --bs 64 --micro 16
```

Here `--rung C3` names the board distribution, not the atlas's HAND/PART/TRAIN rung. Headline output keys are `logs.train.best_fp32` and `logs.train.log[-1].test`.

### 2.5 Archive

The published archive at revision `3919dc14f8ab` contains HAND-H C2/C3, TRAIN best and final snapshots of C3 seeds 10, 11, 13, 14 and C2 seeds 10, 13, 14, test sets, config and a manifest with seed, rung, accuracy and sha256. Weights are safetensors. This is a subset of the analysed final seeds. The original inventory described upload with publication. The archive check kept the three failed accuracy flags; Section 6 explains the differences without changing those flags.

## 3. Hand-built solution

For map-only output, L0 constructs square and colour coordinates. Each L1 ray head finds the nearest occupied square in its direction;
leaper heads for knights and kings, and pawn heads, read fixed offsets. A colour-gated OR then marks the attacked squares. With ray relation `Rel`, occupancy indicator `occupied`, ray-scale constants `G` and `A`, and square coordinates `q,k`, the attention score is:

```text
Rel(q,k) + G*occupied(k)
Rel = G - A*distance
BOS score = 1.5G
step(z) = ReLU(z) - ReLU(z-1)
y = step(16(AW + M*CW - M)) + step(16(AB + M*CB - M))
```

`AW/AB` are White/Black attack signals; `CW/CB` select the requested colour; `M` is the gate scale. A ray is a straight line of board squares; its blocker is the first occupied square. The colour gate selects pieces of the requested colour before the walk begins. In H, `cur` is the current square, `g` the direction, `occ_t` target occupancy, `nx` the next square and `y` the map bit. L1 reads piece, direction and target occupancy. L2 one-step heads move along the ray (`nx = cur + g` unless `cur` is occupied). The map head checks whether square `s` appears in the scratchpad.

Table F1 records the fixed ray-step route alongside the worked circuit above. The board and requested colour enter the coordinate stage. Piece, direction and occupancy reads then drive a ray step; the scratchpad lists attacked squares, and the map head checks whether each square appears. Only the middle read and step stages are trained in hint-format PART; the rest stays frozen.

<details>
<summary>Technical ray-step route and trained stages (Table F1)</summary>

*Table F1. The hand-built H network reads and advances one ray step before forming the map. L1–L2 are the trained stages in H PART; the other stages stay fixed.*

| Component | Input, operation and output |
|---|---|
| Fixed layer 0 (L0) | Read the board and requested colour; construct square and colour coordinates. |
| Layer 1 (L1), trained in H PART | Use those coordinates to read piece, direction and target occupancy. |
| Layer 2 (L2), trained in H PART | One-step ray heads use current square cur and direction g to find next square nx. The scratchpad lists attacked squares. |
| Fixed map head | Check whether square s appears in the scratchpad; write the 64-bit attack map. |

</details>

Exactness checks covered C1's 768 cases; A C2 5000 random and 5000 out-of-distribution boards; A C3 5000 random and 4096 playout boards; H C2/C3 3000 each. A C3 uses 14 heads and 264 neurons. H uses 24 heads and 1041 neurons, of which the task part is 22 heads and 397 neurons. Only one HAND design was built; no alternative HAND was tested.

## 4. What training finds

### 4.1 Results and failures

*Table 3. Recorded screening and final results. Solved means exact match 1.0 on the cell's held-out test; ≥ .99 is a separate, weaker criterion. A/S/T cells all had zero exact solves.*

| Cell (seeds) | Solved 1.0 / ≥ .99 | Best; final exact | First ≥ .99 | Reading |
|---|---|---|---|---|
| A PART C1/C2/C3 (10 each) | 0 / 0 | C1 .07–.19; C2 .107 (all-0 map); C3 0 | none | Ray attention never forms. |
| A/S/T TRAIN C1/C2/C3; width retry w2 | 0 / 0 | C1 .04–.39; C2 .09–.32; C3 0–.08 | none | Blocker R² ≤ .38; C1 memorises. |
| A TRAIN C3, 6 layers, width 512 (5) | 0 / 1 | .63–.997 | ~54.5k | Depth and budget help, without an exact solve. |
| H PART C1/C2/C3 (10 each) | 0 / 0 | C1 .01–.02; C2 .10; C3 0 | none | Map read missing. |
| H TRAIN C2 final, width 512 (10) | 1 / 3 | seed 14 1.0/1.0; seed 13 .992/.990; seed 17 .999/.997; seed 18 .938/.937; six failures summarised as .11–.12 | 47.5–56.5k | First-piece colour choice at token 0. |
| H TRAIN C3 final, width 256 (10) | 7 / 10 | .999–1.0; .998–1.0 | 47.75–52.5k | T1 in 10/10. |

The C2 .11–.12 shorthand is the earlier compact account, not an assertion that every final lies in that interval: the final source table gives .104–.111 for the failed group, and seed 18 is .937. No A/S/T cell solved. The six-layer A seed at .997 is a near-solve, not evidence that answer-only prediction is impossible at any depth.

With fresh data, exact match stayed at 0 for C3 or .11 for C2 for 16–45k steps, then rose to ≥ .99 as the learning rate decayed. Best and final differed by at most .003. Training accuracy was not recorded in the legacy final logs, so this is a delayed test improvement, not a measured train-before-test grokking gap ([definition](../GLOSSARY.md#counting-results)).

A cells lacked occupancy and nearest-blocker distance at every layer (R² ≤ .38); their maps were per-square priors. H PART retained the scratchpad but never formed the map read (`y` R² ≤ .24). Six failed C2 final seeds missed colour `c` at the first scratch query: token 0 accuracy .47–.51 corresponds to choosing the first occupied square while ignoring colour. Colour interchange was untested (`nf = 0`), so this diagnosis rests on probes and the error location, not a causal colour swap.

### 4.2 Circuit type and generalisation

Every final net reaching ≥ .93 had type T1, a count block followed by late contextual pointers and MLP ray steps: C3 10/10 and C2 4/4 (seeds 13, 14, 17, 18). L0's eight near-uniform heads write weighted token counts. L1–L2 heads remain necessary. L3 uses 3–6 contextual pointers, heads that attend to a piece token at a marker or to the board square named by a piece/square token. The ray step and map bit are computed in MLP0–3, feed-forward blocks ([glossary](../GLOSSARY.md#standard-machine-learning-terms)), without a next-square head. Path drops are .74 for MLP3 to output and .66 for embedding to MLP0 ([path drop](../GLOSSARY.md#analysis-measures)). One-hot pointer keys fit 4/14; the remaining 10/14 use mixed keys. This is the fourteen-network classification in the source. The numeric pointer-substitution table below covers ten C3 and two high-performing C2 seeds, not all fourteen; it must not be read as fourteen intervention certificates.

HAND-free discovery retained every head and MLP in all 14 nets reaching ≥ .93: 188–317 of 591 edges, held-out faithfulness .375–.92 ([faithfulness](../GLOSSARY.md#analysis-measures)). This is neither a faithful sparse circuit nor a separately identified algorithm.

Longer or shifted inputs remained difficult. C3 playout-board exact match was .65–.82; scratchpads longer than the trained 192 tokens scored 0/64. C2 out-of-distribution boards scored .51–.63 in the near-solved group. These tests do not inherit the in-range solve count.

## 5. Mechanism and verification

The proposed closed form replaced L0 attention by a uniform causal mean and L3 pointer attention by a one-hot choice of its dominant key, leaving the rest trained. The acceptance rule requires both substitution and complement resampling to keep at least .90 times native exact match ([verification](../GLOSSARY.md#verifying-a-circuit)). It failed.

*Table 4. Pointer substitutions on 128 boards. Values are exact match after replacing the L3 pointers; L0 uniform replacement separately gave 0/13 nets.*

| C3 seed | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|
| Exact | .30 | .97 | .41 | 1.0 | .68 | 0 | .88 | .34 | .07 | 1.0 |

C2 seed 13 gave .93 and seed 14 .16. Thus pointer substitutions covered 12 high-performing seeds, while the uniform-head test's 13
also includes failed C2 seed 15 (native exact .125); neither denominator is the 14-seed circuit classification. Complement resampling
kept L0, L3 and the MLPs but resampled L1–L2 heads; C3 seeds 12 and 15 and C2 seed 14 all fell to exact 0 (zero and mean controls also 0). The discovered circuits' complements gave .55–.89 across 14 nets, all below .90, with circuit-resampled and random-same-size controls 0. These failures leave the proposed form rejected, even in seeds whose pointer substitution alone passed.

At the four sites with an exact or near-exact sparse-dictionary splice, replacing the residual by reconstructed sparse features, no feature tracked a HAND variable above the relevant basis baseline: colour correlation ≤ .22 and map-bit correlation ≤ .35. The compact account calls these four splices exact; the source includes a .97 splice. No feature-level closed form was available to substitute. This is the distinction between reconstruction and feature recovery discussed by [Korznikov et al. (2026)](#references).

Table F2 records the observed component roles and rejected replacement. Weighted token counts feed the needed middle contextual heads, then late piece and board-square pointers. The output MLP writes ray steps and map bits. All shown heads and MLPs were trained. The proposed form remains rejected: uniform-head and dominant-key substitutions or complement tests fail, so this family sketch is not a faithful recovered formula.

<details>
<summary>Technical observed roles and rejected replacements (Table F2)</summary>

*Table F2. T1 in the fully trained H networks. All listed heads and MLPs are trained. Closed-form status: rejected; near-uniform and dominant-key descriptions fail substitution or complement resampling and do not identify a faithful circuit.*

| Trained components | Input and observed role |
|---|---|
| Layer 0 (L0) and MLP0 | Read board tokens; form weighted token counts. |
| Layers 1–2 (L1–L2) and MLP1–2 | Needed contextual heads and middle computation, following the count block. |
| Layer 3 (L3) pointers | Later contextual reads of pieces and board squares. |
| MLP3 and output | Use the late context to write ray steps and map bits, producing the scratchpad and map. This role inventory is not a faithful recovered formula. |

</details>

Table F3 compares the two board generators at their recorded thresholds.

*Table F3. Perfect maps depend on the board generator. Final hint-scratchpad TRAIN outcomes on 2048 boards per seed, reported to three decimals. The whole network is trained. These are recorded-test results, not scores on the different shipped draw. Closed-form status is rejected.*

| Board generator | Seeds at whole-output 1.0 | Weaker threshold |
|---|---|---|
| Full boards with generated armies | 7/10 | 10/10 at ≥ .998 |
| Sparse random-colour sliders | 1/10 | 3/10 at ≥ .99 |

The trained layout differs from HAND-H: the latter steps in L2 and uses one map head; TRAIN adds the L0 count block, requires middle
heads, points at L3 and steps in MLPs one to two layers later. In the final C3 diagnostics, current square `cur` has linear-probe
R² .93–.94, direction `g` .80–.94 and map bit `y` .92–.97 (Appendix B). Only `y` has a well-powered causal interchange test; the
small-n tests for `cur` and `g` do not establish their causal use. In C3 at four layers, hint TRAIN reached ≥ .99 in 10/10 versus 0/10 for answer only; exact hint solves were 7/10.

[Wies et al. (2022)](#references) motivate local sub-task supervision. [Toshniwal et al. (2022)](#references) provide the comparison with chess language models that track board state more readily than complete legal-move sets. [Trockman and Kolter (2023)](#references) motivated the unsuccessful same-token-prior initialisation retry. Sparse-pattern repetition in [Zucchet et al. (2025)](#references) and coverage sensitivity in [Ootani (2026)](#references) suggest a possible explanation for C2's 3/10 ≥ .99, not a controlled proof of that cause. [Bali et al. (2026)](#references), previously misattributed as Stanley et al., contextualise the variable middle-layer head sets.

## 6. Limitations and open questions

The final run used weight decay .1 only, without a sweep, and did not log machine or device identity. Diagnostics used 32 boards. Interchange for `cur` and `g` had fewer than 30 flipping queries and gives no verdict; colour had `nf = 0`, meaning untested, not unused ([interchange](../GLOSSARY.md#analysis-measures)). Neuron counts and HAND effective rank were not measured. HAND-free discovery was read on all C3 10/10 and C2 10/10 seeds. Only one HAND was built. The accepted reference tests do not prove all-real exactness for the trained networks, and no trained closed form passed the two-part rule.

The manifest's `acc_test_fp32` is fp32 accuracy on the shipped C2/C3 test files, a different 2048-board draw from `recorded_test`. The training tests were drawn on the training GPU and were not shown reproducible on another device from the seed alone. Device-dependent
random-number generation was suggested as the cause of the draw difference, not proven. Weights were bit-identical; the recorded check
found agreement between three GPU forward paths apart from a one-board TF32 difference for C2 final seed 10, so the archive gap was
attributed to the board draw rather than precision. This does not make the two tests interchangeable: the three `archive_roundtrip_ok = false` accuracy flags remain false.

*Table 5. Exceptions beyond archive tolerance. Tolerance was one board for best and three for final; the other 11 TRAIN nets were within two boards. Each score is on 2048 boards; recorded precision is preserved.*

| Snapshot | Shipped-test fp32 | Recorded training test |
|---|---|---|
| C3 best seed 14 | .999 | 1.0 |
| C2 best seed 13 | .9951 | .9917 |
| C2 final seed 13 | .9932 | .9902 |

Two valid draws moved a network by up to 14 of 2048 boards. The archive's accuracy-check pass count is therefore not the training solve count. The unidentified middle-layer computation and colour gate remain open; neither failed dictionary features nor weak linear probes exclude a nonlinear representation.

## 7. Prediction record

Predictions were written before each rung, rather than inferred from the final mechanism.

*Table 6. Recorded predictions and outcomes. A and H are the formats in Table 1; PART and TRAIN are the atlas rungs.*

| Prediction | Outcome |
|---|---|
| A PART C2 10/10, C3 ≥ 8/10 | Refuted: 0 solved. |
| A TRAIN C2 5–10/10 | Refuted: 0 solved. |
| Width step stays below .2 | Held. |
| H PART C2 ≥ 8/10, C3 ≥ 6/10 | Refuted: 0, map read missing. |
| H TRAIN C2 4–8/10, C3 2–6/10 ≥ .99 within budget | Refuted at 20k: 0/10, rising. At 60k, C3 10/10 was above range and C2 3/10 below. |
| Errors at ray ends and the next-piece step | Wrong for C2: first error was token 0, the colour gate. |
| Used size at least HAND task part, 22 heads | Held for heads in 9/10 C3; neurons were not measured. |

## Appendix A. Weights excerpt and reading

The archived excerpt reports slices, not certified TRAIN coefficients. HAND's ray score is `G - A*distance + G*occupied`, its BOS score `1.5G`, OR gain 16 and offsets `M*(CW - 1)`. Its H one-step head uses `Rel(next(cur), k)`.

For trained networks the excerpt computes, from fused QKV slices, each L0 head's singular-value ratio `sv2/sv1` of `W_Q^T W_K` and each L3 head's absolute `W_O` block norm. Across 17 best/final snapshot records from the final C3 run, 83/136 L0 QK maps were near rank-1 (`sv2/sv1 ≤ .05`). The 17 records are the excerpt's snapshot denominator, not 17 independent final seeds. Near-rank-one QK is consistent with weighted token counting; exact uniform substitution still fails. Pointer content arrives at L3. The source gives no small numeric TRAIN matrix to reproduce here.

## Appendix B. Detailed evidence

*Table B1. Recorded diagnostics, 32 boards. Used means mean-ablation drop > .05; neurons NA means whole MLPs were ablated. R² is best-layer linear-probe fit; effective rank measures residual dimension; JS compares with HAND attention and can match idle uniform heads ([measures](../GLOSSARY.md#analysis-measures)).*

| Cell / snapshots | Used heads + neurons [MLPs] | Best probe R² @ layer | Effective rank (HAND NA) | JS | y flip (nf) | Best / final exact |
|---|---|---|---|---|---|---|
| C3 H final (10 seeds) | 19–30 + NA [4] | sq 1.0@L2; cur .93–.94@L3; g .80–.94@L4; y .92–.97@L4; occ_t .59–.82@L4; nx .70–.74@L4; c .25–.75 | 17–30 | .004–.036 | 1.0 (511) | 1.0 / 1.0 |
| C2 H final solved/near (13, 14, 17, 18) | 22–32 + NA [4] | sq 1.0@L2; cur .89–.93@L4; g .74–.92@L4; y .85–.93@L4; nx .70–.85@L4; c .05–.56@L3–4 | NA | NA | 1.0 (476) | 1.0 / .97–1.0 |
| C2 H final failed (6) | 16–25 + NA [4] | As above, but c < 0 at every layer. | NA | NA | 1.0 (476) | .13–.25 / .09–.19 |
| A PART C2/C3 | 0–2 + NA [0] | sq, c 1@L1; occ ≤ .06; blocker ≤ .38; y ≤ .35 | — | .63–.66 | — | .107 / 0 |
| A TRAIN C3 width retry w2 (3) | 7–21 + NA [4] | sq, c ~1; occ ≤ .31; blocker ≤ .38; y ≤ .65 | — | ≥ .30 | .88–.93 | .04–.08 |

`sq` is square identity, `occ` occupancy and `blocker` nearest-blocker distance; other variables are defined in Section 3. The compact y-flip row does not override the small-n or zero-n cautions for other variables.

## What to read next

Read [mate-in-one selection](chess-mate-in-one.md) for how attacked squares and escape counts enter a local mating decision. [KRK mate](chess-krk.md) instead studies imitation of a complete endgame policy.

## References

- Bali, K., Stanley, J., Suresh, P., & Bzdok, D. (2026). Quantifying LLM attention-head stability: Implications for circuit universality. *arXiv preprint*. https://doi.org/10.48550/arxiv.2602.16740 (middle-layer head variability).
- Fiekas, N. (n.d.). python-chess: A chess library for Python. https://python-chess.readthedocs.io/en/latest/core.html (software documentation; attack-map labels).
- Korznikov, A., Galichin, A., Dontsov, A., Rogov, O., Oseledets, I., & Tutubalina, E. (2026). Sanity checks for sparse autoencoders: Do SAEs beat random baselines? *arXiv preprint*. https://doi.org/10.48550/arxiv.2602.14111 (reconstruction versus feature recovery).
- Ootani, Y. (2026). Grokking is conditional and fragile: A fully-tractable, multi-seed study at 12K parameters. *arXiv preprint*. https://doi.org/10.48550/arxiv.2607.05104 (coverage-sensitive grokking).
- Toshniwal, S., Wiseman, S., Livescu, K., & Gimpel, K. (2022). Chess as a testbed for language model state tracking. *Proceedings of the AAAI Conference on Artificial Intelligence, 36*(10), 11385–11393. https://doi.org/10.1609/aaai.v36i10.21390 (conference paper; board-state tracking and move prediction).
- Trockman, A., & Kolter, J. Z. (2023). Mimetic initialization of self-attention layers. *arXiv preprint*. https://doi.org/10.48550/arxiv.2305.09828 (mimetic initialisation).
- Wies, N., Levine, Y., & Shashua, A. (2022). Sub-task decomposition enables learning in sequence to sequence tasks. *arXiv preprint*. https://doi.org/10.48550/arxiv.2204.02892 (local sub-task supervision).
- Zucchet, N., d’Angelo, F., Lampinen, A. K., & Chan, S. C. Y. (2025). The emergence of sparse attention: Impact of data distribution and benefits of repetition. *arXiv preprint*. https://doi.org/10.48550/arxiv.2505.17863 (emergence of sparse attention).
