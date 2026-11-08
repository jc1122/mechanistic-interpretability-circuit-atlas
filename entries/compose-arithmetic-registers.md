# Composing arithmetic through scratchpad registers (compose)
**Task.** Base 2. STEP runs ADD, SHL or REV on registers named by tags in the prompt header (operands up to 15 bits). SUM k (k operands), MUL n (n multiplier bits) and POW e compose STEP lines. Format O writes register lines; format S also writes gathered bits and carry. Fresh samples every step; eval sets of 128 per cell; score = every teacher-forced trace token exact. **Status.** Final cell O c TRAIN, 10 seeds, 30k steps: **unsolved (0/10)**. Solved: S a PART (3/3, screen). 2026-11-08.
## Summary
Trained from scratch on STEP plus SUM2-5/MUL1-4/POW2-3 (O c), the net solves every trained composition in 10/10 seeds. It fails the bare STEP primitive in 10/10 (final .54-.61), so the cell is unsolved (final .871-.897). The cause is a missing variable. The net never builds HAND's register-address code PA/PS (R² .29-.55 at L2, ~0 at L4; HAND .99). The carry is present and causal (interchange flip .96-1.00). The net fetches operands by their position in the trace, which works only while the compositions keep a fixed register convention. With HAND's tag router frozen and only the column-stage MLP trained (S a PART), the stage learns HAND's rule exactly in 3/3 seeds: d = t mod 2, c' = ⌊t/2⌋. It passes substitution (1.0) and resample ablation of the complement (1.0).
## Setup
Shared architecture: [METHODS.md](../METHODS.md). AdamW lr 1e-3, wd 0.1, betas (.9,.98), warmup 1000, cosine, clip 1; TF32 TRAIN, fp32 PART and all evals. Each step trains one cell (STEP .4, compositions .6), batch ≈ 8000 tokens. Eval every 1000 steps. Arms: a = STEP only; b adds SUM2-3/MUL1-2/POW2; c adds SUM2-5/MUL1-4/POW2-3. Depth+1 cells (SUM6, MUL5, POW4) are never trained. Seeds: O c final 0-9; screens 3 (S) or 10 (O). Single consumer or datacenter GPU per run. Reproduce: none (unsolved task; the training code is not yet ported to code/). Networks: https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/compose-arithmetic-registers.
## Architecture as used
O (TRAIN): 4 layers, d_model 256, 16 heads × 16, MLP 1024, ReLU, NoPE, no LayerNorm, untied readout, vocabulary 296; all weights trained. S (PART): 4 layers, d_model 208, 13 heads × 16, MLP 134, vocabulary 424. The S hand-built net is frozen; only layer-3 MLP is trained (W_in [134, 208], W_out [208, 134] masked to HAND's stage ports, 8,728 parameters). Archived: hand_O, hand_S, part_S_a_{best,final}_s0-2 (4 stage tensors), train_O_c_final_{best,final}_s{0,1,8}; config.json and a manifest (seed, rung, accuracy, sha256).
## Prediction before training
Before r1 (O c): depth+1 full ≥ .9 in ≥ 8/10 seeds, with ADD-stage neurons shared across tasks as in HAND. S: PART ≥ .99 in 3/3 by 10k steps; c TRAIN depth+1 ≥ .9 in ≥ 1/3. All falsified, except S a PART (3/3, solved by 16k). The O c final run had no pre-registered prediction and is reported without one.
## Hand-built reference
Each header names a program counter and register tags; every digit carries a significance tag. L0H0 is uniform (position code). Eleven L1 recency heads copy the latest P/A/S/GA/ST fields (PA, PS, GA_RD, PC). L2 match heads fetch operands by tag equality: L2H0 → XA1, L2H1 → XS0, L2H2 → XR, scored 30(2v·id − v²). The L3 MLP is the stage: t = GA_RD + NUM + CI with CI = ReLU(PC + NF − 1); d = t mod 2, c' = ⌊t/2⌋; SHL and REV copy GA_RD. Exact (margin 30) on every cell in bases 2/8/10/16, with 200 or 50 examples per cell. Used: 21/64 heads. The ADD-stage neuron set is shared by STEP, SUM and MUL (Jaccard 1.00).
## What training finds
| Rung / fmt / arm | Seeds | best → final (mean trained cells) | Outcome |
|---|---|---|---|
| **TRAIN O c (final)** | **10** | **.877-.899 → .871-.897** | **0/10; compositions 1.00, STEP ADD/SHL/REV ≤ .59/.61/.54, depth+1 0/10** |
| TRAIN O a / O b (r1) | 10 each | .51-.68 / .81-.83 | O a run lost; O b: SUM4/MUL3/POW3 never ≥ .9 |
| **PART S a (r1)** | **3** | **1.00 → 1.00** | **solved 3/3 (STEP primitives)** |
| PART S b / S c (r1, r2) | 3 each | .968-.970 / .80-.83 | b: untrained depth+1 cells; c: budget-limited, still rising |
| TRAIN S a/b/c (r1, r2) | 3 each | .37-.48 | underfit (train .42-.56); 2× budget gave nothing |
Grok step (O c, first full ≥ .9): SUM5 17-18k, MUL4 14k; STEP never. Depth+1 final, max over seeds (full/exec): SUM6 .24/.81, MUL5 0/.17, POW4 .07/.52.
Circuit types: S a PART has one type, **frozen tag router + HAND column stage** (3 of 3). O c TRAIN has one failure type, **positional-convention fetch + MLP3 adder** (10 of 10): compositions are solved, STEP has no tag fetch. No discovery pass or alternative HAND was run.
| Cell (seeds) | exact | used heads / MLPs | R² CY / PA / PS / XA1 | inter flip (nf>0) | r90 L4 vs HAND 3 | JS vs HAND | patch |
|---|---|---|---|---|---|---|---|
| O c STEP_ADD final (s0-9) | .41-.535 (s8 .785) | 0-5 / 2-4 | .85-.89@4 / .29-.55@2 / .29-.55@2 / .65-.70 | CY .96-1.00@4 | 9-11 | .11-.13 | MLP0 / MLP3 |
| O c SUM5 final (s0-9) | .99-.9975 | 11-21 / 4 | .92@4 / .40@2 / .60@2 / .87@4 | XA1 DAS .92-1.0 | 26-29 | .19-.21 | MLP3 |
| S a PART STEP_ADD final (s0-2) | 1.0 | 7 (mean) / MLP2, MLP3 | carry .40-.44; operands .98-.99 | carry .52-.91 | 2 | 0 (frozen) | MLP3 |
Best vs final: O c best .877-.899, final .871-.897; S a PART 1.00 at both snapshots.
## Closed form
S a PART, trained MLP3, per column: t = GA_RD + NUM + CI; d = t mod 2; c' = ⌊t/2⌋ (ADD); SHL and REV copy GA_RD with c' = 0 (0 fitted numbers). Substituting HAND's stage for the trained MLP3 gives 1.0 in 18/18 tests (best and final, s0-2, ADD/SHL/REV, 256 fresh traces each; native .992-1.0). Zeroing MLP3 gives 0, and HAND's MLP3 with shuffled hidden-to-output wiring gives 0. Resample ablation of everything outside the claimed circuit (L0H0, MLP0, L1H0/1/2/5/7/8/9/11/12, MLP1, L2H0/1/2, MLP2, MLP3) gives 1.0 = base (ADD s0-2, SHL s0, REV s0). Resampling the circuit itself gives .15-.42, and a random same-size set gives .19-.54. O c TRAIN: none (unsolved).
## Weights excerpt and reading
HAND stage: tag-match scores are 30(2v·id − v²); the stage bias offsets are integers. Per-seed trained stage neurons (top 3 by |W_out| × |W_in on ports|, final) are printed by the archive's excerpt script `compose`: HAND's stage has 134 live neurons with W_in values in {-1000, 1, 10, 1000, 10000}; each PART seed keeps 134 live neurons, and its top neurons read mostly GA_RD and CI (|w| up to .85) and write the digit outputs YD[0]/YD[1] (|w| .5-.76) with real-valued, seed-specific weights. Reading: the substitution and complement results show the trained stage computes the same function as HAND's stage on its ports, but no neuron-level identity is claimed.
## Comparison
PART: same algorithm as HAND (stage function equal under substitution). TRAIN: a different code. L4 CKA vs HAND is .01-.05, and 55-64 of 64 heads best-match HAND's uniform head. Operands come from the trace convention, not from HAND's tag lookup. This matches order-based variable binding (Cui et al. 2026) and positional shortcuts under low layout diversity (Nevermann & Gros 2026).
## Controls and limits
Scoring is teacher-forced, not rollout. Depth and length are confounded at depth+1. Lengths beyond 16 digits lie outside the frozen router's id range, so S a PART has no longer-input test; its position shift (pad 256) is exact. The complement test used final snapshots only, with SHL/REV on one seed. Method v2 discovery (discover, dict, bucket) was not run. The O c final has no pre-registered prediction. Lost runs (O a r1, S b TRAIN r2, S c PART r2 final) are not counted.
## References
- Lee et al., Teaching Arithmetic to Small Transformers, https://arxiv.org/abs/2307.03381 (preprint).
- Cho et al., Position Coupling, https://arxiv.org/abs/2405.20671 (preprint).
- Nevermann & Gros, https://arxiv.org/abs/2609.11913 (preprint): low layout diversity, positional shortcuts.
- Shintani, https://arxiv.org/abs/2603.26828 (preprint): mixed-layout exposure.
- Cui et al., https://doi.org/10.48550/arxiv.2603.22278 (preprint): ordering identifiers for variable binding.
- Wu, Geiger & Millière, https://doi.org/10.48550/arxiv.2505.20896 (preprint): dereferencing in symbolic programs.
