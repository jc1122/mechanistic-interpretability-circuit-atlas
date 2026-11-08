# Router scratchpad: step tokens as a router for long addition (and a subtraction final run)
**Task.** a + b (final run a - b), n digits MSB first, base 2 (N 16) / 10 (N 6), n ~ U[1, N], fresh pairs per batch. Formats: AM / AL (answer only,
MSB / LSB first), I (scratchpad `s_k c_(k+1) IDX_(k+1)`), R (router: operand tokens carry a coupled id; `ADD s_k CARRY c_(k+1) IDX_(k+1)`); ids = column
+ random offset. Test: 1000 pairs at n = N, N+1, 2N; exact = every token. **Status.** HAND exact in all 8 cells; PART/TRAIN 10 seeds; final sub TRAIN AL 10.
## Summary
The per-column step is one shared stage: frozen-upstream PART stages solve 10/10 in every router or masked cell; HAND's stage (0 fitted reals) substitutes
at 1.0 on 60 snapshots and length-generalises when it cannot read position (R; masked I/A 2N = 1). From scratch no format reaches 2N (the router token
alone does not buy length); TRAIN nets use HAND's causal variables via seed-specific code; subtraction's MLP3 is a table of (a_k, b_k, borrow).
## Setup
Shared architecture: [METHODS.md](../METHODS.md). AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1k, cosine, clip 1, bs 256; PART fp32 10k-20k steps,
TRAIN TF32 30k; eval every 500; stop at in-range 1.0 or a 5k plateau after 75% of steps; seeds 0-9 packed per GPU process. Controls: zero /
mean / shuffle substitutions, position-masked PART arms. Headline keys `best`, `best_step`, `log`. Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/router-addition](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/router-addition).
    python train.py --op add|sub --base 2|10 --kind part|train --fmts AM AL I R --seeds 0 1 2 3 4 5 6 7 8 9 --steps 30000
## Architecture as used
GPT-NeoX 4L, d_model 256, 4 heads x 64, MLP 512 ReLU, no LayerNorm, no position embedding; vocab 12 specials + B digits + K ids (K 40 / 20) + B*K
operand tokens (134 / 242). PART: HAND frozen, stage MLP re-initialised and trained (I, R: layer 2; AM, AL: layer 3; W_in [512, 256], W_out
[256, 512] + biases); masked arm holds the stage inputs on HAND position dims at 0. TRAIN: every tensor. Archive: HAND per (base, format), all
PART seeds of the solved arms, 2-4 TRAIN seeds per cell, final-run seeds (safetensors + config.json + manifest: seed, rung, accuracy, sha256);
networks on Hugging Face (circuit-atlas-nets, entry router-addition).
## Prediction before training
Recorded in advance: PART R length-generalises, I/A only when masked - HELD (masked 2N = 1; R .80 / .999). TRAIN in range: binary A-MSB fails
(HELD, 0/10); A-LSB, I, R solve (HELD: I 10/10, AL b10 10/10 and b2 8/10, R 10/10). TRAIN length: only R generalises - REFUTED (R 2N = 0).
Routing visible as disjoint ADD/CARRY neuron sets - NOT TESTED (no neuron-level ablation).
## Hand-built reference
R: L0 heads by recency, L1 heads match operand ids to the line's IDX (no absolute position), L2 MLP stage gated by the op token: sum s = (a_k +
b_k + c) mod B (9 / 57 neurons), carry c' = [a_k + b_k > s_k] read from the digit just written (3), successor IDX_j -> IDX_(j+1) (one per id).
I: same stage, slot flags from r - r(last IDX). A: index heads on derived position; AM adds a lookahead thermometer (exact n <= 30). Exact
(margin 30) on all / 500 pairs per n <= N+1 and 2N. Subtraction: s = a - b - c, borrow [b - a + s >= 1].
## What training finds
- First in-range exact (k steps, b2 / b10): PART R 1.5-3 / 3.5-5, masked I 3.5-4.5 / 8.5-10, masked AM/AL b10 5.5-9.5; TRAIN I 20-25 / 8.5-16,
  AL 23-29.5 / 13.5-16, R b10 11.5-16.5. Final cell: 50% at 7-14k, 90% at 8.5-15k, exact at 14-19.5k (9 seeds; seed 4 stuck at .11, all errors in answer column 1).
- Circuit types (diagnosed seeds): R-frozen-stage 18 of 18 (one basin up to permutation); I-masked-stage 5 of 5; A-frozen-readout 18 (out =
  T[a_k + b_k + c]); TRAIN sum/carry/successor 20 (I; lmc barrier >= 2.63: different solutions); TRAIN answer-only sum/carry 19 (AL 10, AM 9);
  final cell: sub-digit-table 9 of 9 solved (MLP3 a table of a_k, b_k, borrow, not a - b - c alone). Unmasked PART I fails at the sum slot
  (.74-.91). Parts read: R L1H0 (drop .955-.991), A L2H0/L2H1 (.89-.95), TRAIN MLP3 (.995-1).

| Cell | Used heads / neurons | Best probe R^2 (layer) a, b, s, cout, k | Eff. rank L4 net / HAND | Attn JS | Best / final N |
|---|---|---|---|---|---|
| PART R b10 / b2 | 4 / - | 1, 1, .79, .93, 1 (4,4,4,4,2) / - | 4.22 / 2.65 | 0 | 1 / 1 |
| PART I masked b2 / b10; AM / AL b2 | 5; 3 / 4 / - | - | - | 0 | 1 / 1; 1 / .89 |
| PART AL masked b10 | 4 / - | 1.0, 1.0, .70, .88, 1.0 (3,3,4,4,1) | 1.03 / 1.02 | 0 | 1 / 1 |
| TRAIN I b10 / b2 | 4-7 / 4-9 / - | .54, .46, .95, .97, 1.0 (4,4,4,4,3) / - | 9.52 / 1.00 | 0-.51 | 1 / 1 |
| TRAIN AL / AM b10; R b10 / b2 | 3-4 / 4-8 / -; - | - | - | 0-.50 | 1 / 1; .998 / .997; 1 / 1 |
| Final: sub TRAIN AL b10 | 5 (L3H0-3 + one L2) / MLP3 n90 206-229 | s4 (failed): .72, .68, .72, .70, 1.0 (4,4,4,4,1) | - | s4 L3 .38-.43 | 9/10 at 1; s4 .114 / .099 |
"-" = not measured. Length (final, N+1 / 2N): TRAIN I .99 / 0 (b2), .55 / 0 (b10); AL .45 / 0; AM .25 / 0; R .65 / 0; final cell .21 / 0.
## Closed form
PART R stage, masked PART AL readout: HAND stage (s = (a + b + c) mod B, c' = [a + b > s], successor; 0 fitted reals), AL spline out = sum_j
C[j, d] max(0, 1 - |u - j|), u = a_k + b_k + c (200 reals): 1 vs original .996-1 on 60 snapshots, zero / mean / shuffle <= .002. Final cell:
MLP3 out := T[a_k, b_k, c_k] (200 x 256 means) .944-1 on 18/18 solved snapshots, T[a - b - c] alone 8/18, controls <= .002; resample of all but
MLP3>out among direct writes .902 / .950 (s0, s3 best; accepted) but .127 on s1 final, whose L3 heads also write the answer to the logits.
Discovered: MLP0 -> MLP1 -> MLP2 -> L3 heads -> MLP3 (+ L3 heads -> out), 22-32 edges, faith .95-.98 on all 3.
## Weights excerpt and reading
Excerpt `router`: HAND R stage neurons/inputs, per PART R seed the W_in energy share on them; final cell: neurons with 90% of ||W_U W_out(MLP3)||^2.
## Comparison
R and I share HAND's stage (upstream: id match vs index heads); coupled ids act like masking; from scratch retrieval loses operands at 2N (b_k R^2 <= .06).
## Controls and limits
Binary AM HAND exact only n <= 30. Teacher-forced exact. No TRAIN addition closed form; final-cell table keyed on true variables (bypasses upstream).
## References
Abbe et al. 2024 https://doi.org/10.52202/079017-0874; Cho et al. 2024 https://doi.org/10.48550/arxiv.2405.20671; McLeish et al. 2024 https://doi.org/10.48550/arxiv.2405.17399;
Lee et al. 2023 https://doi.org/10.48550/arxiv.2307.03381; Izzo et al. 2025 https://doi.org/10.48550/arxiv.2510.27015 (preprints / proceedings, scite-verified).
