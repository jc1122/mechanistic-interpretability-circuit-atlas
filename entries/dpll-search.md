# DPLL search on small 3-SAT: a learned local update inside a hand-built DPLL transformer

**Task.** Ordered 3-SAT, n = 4-6 variables, m = 10-25 clauses. A (answer-only) outputs SAT/UNSAT; T (scratchpad) writes, after every unit/decide/backtrack step, the full DPLL state (assignment, decision levels and tried flags, clock, the clause used, conflict), then SAT/UNSAT. Formulas are split train/monitor/test 8:1:1 by a canonical clause-multiset hash; 192 test formulas per seed.
**Status.** Screening (r1, seeds 0-2): A PART, T PART, A TRAIN, T TRAIN. Final run (2026-11-08): T PART, 10 seeds, 30k-step budget, all exact (seeds 0-2 stop at 11.5k, 3-9 at 15.5k).

## Summary
With every hand-built (HAND) weight frozen except the last MLP, training re-learns HAND's local DPLL update on 10/10 seeds. The discovered circuit is the HAND circuit (36 edges, Jaccard 1.0). A parameter-free closed form of the learned MLP (conditional copy/clear/force/decide/backtrack of state and level) replaces it with 192/192 exact traces per seed, and resample ablation of everything outside the circuit keeps exact 1.0. On longer formulas (n = 7) the trained MLP is only .67-.83 exact, the closed form 1.0. From scratch, neither format learns the task (A .74-.86, T ~0).

## Setup
Shared architecture and instruments: [METHODS.md](https://github.com/jc1122/mechanistic-interpretability-circuit-atlas/blob/main/METHODS.md). Fresh, outcome-balanced formulas every step (8 per step); UNSAT proposals are core-enriched, not uniform. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000, cosine, clip 1, eval every 500, checkpoints every 2500; stop when all seeds are test-exact, or a 5k plateau after 22.5k, or the 30k budget. PART fp32, TRAIN TF32. Best = monitor exact, then CE. Exact = every teacher-forced output token right (T: the whole trace). Hardware: one consumer or rented GPU per pack; diagnostics on CPU.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/dpll-search](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/dpll-search) (safetensors, config.json, manifest with seed, rung, accuracy, sha256). Training and analysis code is not yet in the atlas `code/` folder.

## Architecture as used
GPT-NeoX, ReLU, no LayerNorm, no positional embedding (explicit clock tokens), untied readout, vocabulary 1,104, 16,384 positions; indices zero-based.
T: 6 layers, d_model 160, 5 heads/layer (d_head 32), MLP 96. PART trains only L5 MLP: W_in [96, 160], b_in [96], W_out [160, 96], b_out [160]; input masked to old state/level (OLD, OL, OL2), MLP4's equality/keep flags (EQF, EQD, EQB, KEEP) and the controller block (CTRL); output masked to new state/level (Y, YL, YL2). All other weights frozen HAND.
A: 2 layers, d_model 272, 1 head/layer (d_head 272), MLP 128; PART trains only L1 MLP (reads witness-count channel 258, writes answer channel 259). TRAIN: all weights from a seeded init.
Files: `hand_T`, `hand_A` (full HAND), `part_T_final_<best|final>_s<0-9>` (the 4 MLP5 tensors; insert into hand_T), r1 `part_*`/`train_*` nets.

## Prediction before training
Written before training: T PART >= 2/3 screening seeds at .99 by 5k and 8-10/10 in the final run; PART follows HAND variables in a shared basis; closed form only after substitution; n = 7 token accuracy >= .99 with whole-trace possibly lower; T TRAIN 1-2/3; A PART >= 2/3, A TRAIN <= 1/3. Held: T PART screening 2/3 exact by 5k (but unstable, with regressions), T PART 10/10 final, HAND variables shared, A PART 3/3, A TRAIN 0/3. Partly: n = 7 token >= .99 on 7/10 seeds. Failed: T TRAIN 0/3.

## Hand-built reference
T: L0 clock/header retrieval; L1 previous state and clause-operand retrieval (L1H0 = old state/level of the updated variable); L2 clause copy and unknown/deepest-decision selection; L3 first unit clause, conflict and action; L4 controller copy (L4H0) and MLP4 equality flags; MLP5 = conditional copy/clear/flip of state and level. Clock equality uses a signed binary ID code, so a stale conflict never outranks a current clause. Weights are formulas (gates +-128, unit steps): T 14 heads + 71 neurons [3,12,8,5,11,32], A 1 head + 130 neurons. Exact on 264 stratified traces (172,395 tokens) through n = 7, all 216 local clause cells exhaustively, and an independent SAT oracle; formula space is sampled, not exhausted.

## What training finds
| Cell | Exact seeds | Best / final test | Grok (first test 1.0) | Stop |
|---|---|---|---|---|
| T PART final | 10/10 | 1.0 / 1.0 every seed | median 4.0k (2.5k-7k); later dips to .67-.92 | 11.5k (s0-2), 15.5k (s3-9) |
| T PART r1 | 3/3 | 1 / 1 | 4.5k, 3k, 5.5k | 5.5k |
| A PART r1 | 3/3 | 1 / 1 | .5k-1k | 1k |
| A TRAIN r1 | 0/3 | .72, .83, .86 / .74, .83, .86 | never | plateau 28k |
| T TRAIN r1 | 0/3 | .010, .005, .005 / .016, 0, .005 | never | budget 30k |
Circuit types: T PART final, one type on 10/10 seeds, "HAND path + learned seven-field update": the discovered circuit equals the HAND circuit; MLP5 reads exactly L1H0, L4H0 and MLP4 (resampling one of these paths drops exact to .03-.25, all three to .00-.06). Seeds reach different MLP5 weights (linear-mode barrier .88 and .26 between seed pairs): one mechanism, several weight solutions. A PART r1 (3/3) sits on HAND's 4-edge pooling -> count -> threshold path; A/T TRAIN have no verified algorithm.
| Cell (final snapshot) | Used heads + MLPs | Best probe R² per HAND variable (layer) | Residual r99 / HAND | Attention JS vs HAND | Best / final |
|---|---|---|---|---|---|
| T PART final | 11-12 of 14 HAND heads + MLP1-5; MLP5 n90 24-29 of 96 | clock .94 (L1), variable .76 (L0), action .75 (L6), unit_sign .55 (L0); categorical codes low by design (HAND CKA 1.0 every layer) | 2 / 2 | 0 on all 14 heads | 1.0 / 1.0 |
| A PART r1 | 1 head | witness_count 1.0 (L2) | 6-8 / - | 0 | 1 / 1 |
| A TRAIN r1 | 1 head | sat .16-.31, witness_count <= .19 | 2-3 / - | .28-.44 | .72-.86 / .74-.86 |
| T TRAIN r1 | 13-17 heads, 3-55 neurons | step .95-.98 (L3-4), variable .99 (L0), new_state .37-.48 | 19-25 / 2 | .19-.66 | .005-.010 / 0-.016 |

## Closed form
MLP5 = F, with old = one_hot(argmax OLD), L = clamp(round(OL), 0, 7), flags f, d, b, s = CTRL[0..3] > .5 (force, decide, backtrack, sign) and ef, ed, eb, k = EQF, EQD, EQB, KEEP > .5 (this variable is the target; keep):
copy = f(1-ef) + d(1-ed) + b(1-eb)k; Y = copy·old + [b(1-eb)(1-k), d·ed, b·eb, f·ef(1-s), f·ef·s] over states (U, D0, D1, F0, F1); YL = copy·L + b·eb·BL + f·ef·DL + d·ed·(DL+1); YL2 likewise with squares (BL, DL = backtrack and decision level from CTRL).
No fitted numbers. Substituted for MLP5: 192/192 = native on every seed (agreement 1.0); with every frozen part also replaced by its formula: 192/192. Resample ablation outside the 36-edge circuit: exact 1.0 (base 1.0) on 10/10, circuit resampled 0, random same-size 0. Accepted, 10/10 seeds (bar .90 x net exact).

## Weights excerpt and reading
Weight slices of L5 MLP (final snapshot; recompute from the published files): HAND MLP5 has 32 live neurons, W_in values {-128, -1, 1, 2, 128}, |W_in|_F 652.7, W_out 90%-energy rank 7; its input energy sits on the gates (EQ/KEEP .54, CTRL .46). Trained MLP5, seeds 0-9: all 96 neurons live, |W_in|_F 7.2-7.5, |W_out|_F 3.3-3.7, W_out rank 4; input energy OLD .39-.44, OL .13-.19, EQ/KEEP .20-.27, CTRL .17-.20; output energy 1.00 on Y/YL/YL2. Reading: HAND gates a copy with huge constants; training learns the same truth table with small, spread weights that read the old state directly, which holds in range but breaks on longer formulas (decision level 7 never occurs at n <= 6; this cause is not tested).

## Comparison: scratchpad vs answer-only
Both PART cells solve; neither TRAIN cell does. The scratchpad does not rescue from-scratch learning here (T TRAIN ~0), and HAND sizes differ (T 14 + 71, A 1 + 130), so this is not a matched size comparison. Length, n = 7, 24 traces/seed: T PART final native .67-.83 exact (token .979-.996), closed form 1.0 on every seed; A TRAIN r1 .63-.79; T TRAIN 0. Related work: Pan et al. build a transformer that runs DPLL with a chain of thought; here a trained stage inside such a construction is checked against its hand-written rule.

## Controls and limits
HAND: independent SAT oracle, prefix causality, right-padding, clock shifts and clause reversal pass. PART masks keep HAND intact; substitutes read live activations, never labels. Teacher-forced traces only: free rollout is checked on one formula per seed. The closed form discretises (threshold flags, argmax state, round level), so it matches decisions, not raw activations. Edge attribution (EAP-IG) ranks are untrusted (Spearman < .8); interchange and DAS flips are low, so localisation rests on HAND-component evidence and resample ablation. Diagnostics use 32 traces per seed. A PART's threshold form is not substituted.

## References
[Pan et al. 2024, Can Transformers Reason Logically? A Study in SAT Solving](https://arxiv.org/abs/2410.07432) (arXiv preprint). [Abbe et al. 2024, How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad](https://arxiv.org/abs/2406.06467) (arXiv preprint).
