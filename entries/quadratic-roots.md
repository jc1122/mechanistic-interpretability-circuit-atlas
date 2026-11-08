# Quadratic roots

> - **Task.** Given integers b and c, compute for x² + bx + c the discriminant D = b² − 4c, its integer square root s, the two floored roots and a flag (negative / integer / other real). Example: b = −3, c = 2 → D = 1, s = 1, roots 1 and 2, flag integer.
> - **Hand-built net.** Four variants (answer only, sub-result, full trace, register executor), all exact at n 3–5. The executor runs the whole computation as a fixed program of Boolean NAND updates, each fetching two addressed registers by binary address match and computing the new bit with one AND neuron whose readout emits its complement (NAND).
> - **Trained nets.** Only the scaffolded NAND executor is learned. With frozen hand-built routing a trained MLP learns it in 500 steps on 10/10 seeds, and the whole trained MLP can be replaced by one neuron, ReLU(A + B − 1). Trained from scratch (2× width) it gets 0/10 and never learns to fetch the addressed registers. The other completed cells memorise or underfit (0/10).
> - **Verdict.** Partly solved (executor with frozen routing 10/10, from scratch 0/10; answer only at n 6 pending). Closed form: partial (accepted for the executor stage only, 1,792/1,792 packets kept per seed; none for the parent cells).
> - **Links.** [Weights](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/quadratic-roots) · Reproduce: not available yet · [Terms](../GLOSSARY.md)

## Task and status
**Task.** x² + bx + c with integers |b|, |c| ≤ R in binary sign/magnitude (n = 5 bits: R 15, 961 pairs; R 31, 3,969; n = 6: R 63,
16,129). Outputs are specified integers, not exact irrational roots: D = b² − 4c, s = ⌊√max(D, 0)⌋, r∓ = ⌊(−b ∓ s)/2⌋ and a flag
(negative / integer / other real). Formats: A answer only (29 tokens at n 5); S writes D and s first (49); T writes every arithmetic step
(170); **S_exec** (scratchpad): the whole computation as an input-independent program of Boolean NAND updates on a register file, one
update per step — each packet supplies the shuffled register bank and the gate's two source addresses, and the net predicts the new bit.
Exact = every output token correct, teacher forced.

**Formats.**
- **A**: answer only
- **S**: sub-result (writes D and s first)
- **T**: full trace (every arithmetic step)
- **S_exec**: executor scratchpad (one Boolean NAND update per step)

**Status.** 2026-11-07. HAND A/S/T/S_exec exact at n 3–5 (T not exact at n 6, closed); ten-seed finals: S_exec PART 10/10, S_exec TRAIN
(2× width) 0/10. Open: **answer-only A at n 6 / R 63** (its 40k run's host was lost at 23.5k with test .91–.93; the cosine tail is being
rerun as a decay from the 22.5k snapshot, pending); most parent-format nets (A/S TRAIN, T TRAIN, S/T PART r2/r3) were lost with rented
hosts, so for those cells the run-time diagnostics below are the evidence and the networks are not archived; S_exec checks are sampled
(rollouts, not all 16,129 traces); public reproduction code pending.

## Summary
The hand-built trace (format T) shrinks the arithmetic (compact T: 62 neurons for every n vs answer-only A's ~6R), but every trained parent stage fails: one
frozen-parts MLP must place unit-resolution thresholds on scalars up to ~2¹¹ and memorises or underfits (0/10 in every cell). The
scaffolded executor reduces the learned part to one Boolean rule: PART (one MLP trained on frozen hand-built parts) learns it in 500 steps on 10/10 seeds, keeps it exact at widths 7
and 8 and in its own predicted-state rollouts, and its whole trained MLP is replaced by the hand-built neuron AND = ReLU(A + B − 1) with
1,792/1,792 packets kept per seed. From scratch (TRAIN, 2× width) the executor never learns to fetch the addressed registers and outputs
the majority bit. This is a learned addressed-NAND stage inside supplied routing and microcode, not a from-scratch quadratic solver.

## Setup
[METHODS.md](../METHODS.md) for shared conventions. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1,000, cosine, clip 1; finals 10 seeds
(seed = init + split), 40k budget, plateau 5k / minimum 30k, checkpoints every 2,500. Early parent retries r1–r3 used constant lr and wd 1.
PART fp32, TRAIN TF32 updates with fp32 evals. Parent grids train on 480/961 (R 15) and 1,984/3,969 (R 31) pairs. S_exec: 12,903/16,129
R 63 coefficient pairs for training (held-out pairs per seed in `split_S_exec`), fresh reachable register states every batch (128), mixed
widths 3–6, shuffled memory, renamed addresses; program, source/destination addresses and state copying are supplied. Tests: local =
next-bit exact on 1,024 held-out packets; trace = all updates of a whole program for held-out pairs (8 per cell at n 5/R 15, n 5/R 31,
n 6/R 63); length n 7/8; rollout = the net's own predictions fed back through a whole program, answer checked by a verifier.
**Reproduce:** training code for this task is not yet in `code/`; the weights, held-out splits and recorded accuracies are on Hugging Face (link above).

## Architecture as used
GPT-NeoX, ReLU, no LayerNorm, no positional encoding, untied embeddings. Parent A/S/T: 4 layers, d_model 768, 12 heads (d_head 64), MLP
512, vocabulary 10, context 512 (n 6 A: d_model 1,024, 16 heads, MLP 1,024). PART trains one stage MLP on frozen HAND (A: layer 2; S/T:
layer 1). S_exec: 2 layers, d_model 128, 2 heads, MLP 128, vocabulary 1,029 (register/address tokens), context 512; PART trains the
layer-1 MLP (W_in [128, 128] masked to the A/B residual columns, W_out [128, 128], both biases); TRAIN width 2 = d_model 256, 4 heads,
MLP 256, all weights. Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/quadratic-roots](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/quadratic-roots):
`hand_{A,S,T}`, `hand_T_compact`, `hand_S_exec`, `part_S_exec_final_s0…9` (the 4 MLP tensors; insert into `hand_S_exec`; best = final
byte-identical), `train_S_exec_final_s0…2` (full state dicts; best = step-0 init, not kept), `part_T_R15_{r1,r2}_s0…9` (early parent
PART), `split_S_exec`, `config.json`, `manifest.json` (recorded and fp32 re-scored accuracy, sha256).

## Prediction before training
Recorded before the runs: PART A 10/10 — **refuted** (best .05–.31; r2 .58–.62); PART S ≥ 7/10 — **refuted** (r2 .03–.06); PART T 10/10
— **refuted** (r2 .61, r3 .83, memorised); TRAIN A ≤ 2/10 and T ≤ 3/10 — **confirmed** (0/10); HAND T ~62 neurons vs A ~6R —
**confirmed**. S_exec PART 3/3 within 2.5k — **confirmed** (500); S_exec TRAIN ≥ 2/3 at local .999 — **refuted** (0/3, then 0/10).

## Hand-built reference
Parent: a layer-0 frame head reads 1/(1 + position) from BOS and an MLP turns it into a position one-hot; table heads read a written
number as a weighted token sum; bit k of V is [V − W ≥ 2ᵏ] via ReLU(2u + 1.5) − ReLU(2u + .5), gated by −G per failed condition. A: layer 1
b², layer 2 isqrt [D ≥ m²], layer 3 roots; S: layer 1 b² + isqrt of the written |D|; T: all rules in layer 1 on token values (compact T
moves bit thresholds into query-position weights). S_exec: address codes Cⱼ(a) = 2·bitⱼ(a) − 1; layer-0 heads find the A/B pointer
tokens, layer-1 heads fetch registers a, b by binary address match (a wrong code bit costs 40 logits); one neuron AND = ReLU(a + b − 1)
writes Y; the readout (other −10, bit0 10·Y, bit1 5) emits the complement, NAND. Exact at n 3–8, margin 5.
| n (R) | A designed / used | S | T | T compact | S_exec |
|---|---|---|---|---|---|
| 5 (31) | 6h 217 / 7h + 194 | 9h 321 / 10h + 260 | 9h 322 / 10h + 187 | 9h 62 / 10h + 44 | 4h + 1 |
| 6 (63) | 6h 385 (exact) | 9h 507 | not exact | untested | 4h + 1 (1,806 → 2,296 steps) |
Used = ablation minima (heads + neurons, frame head included, frame neurons excluded) on 200–300 rows.

## What training finds
| Cell | Format | Best / final test (10 seeds) | Train | Stop | Verdict |
|---|---|---|---|---|---|
| S_exec PART final | scratchpad | trace 1 / 1 (10/10); n 7/8 and rollouts R 15/31/63 1 | 1 | solved 500 | solved |
| S_exec TRAIN width 2 final | scratchpad | trace 0 (best = step 0); local .58–.62 | .58–.62 | plateau 30k | constant output |
| A n 6 R 63 TRAIN | answer only | .91–.93 at 22.5k; **final pending** (decay) | .97–1 | host lost 23.5k | memorising slowly |
| A PART R 15 r1 / r2; R 31 r2 | answer only | .34 → .17 / .59 → .58; .63 → .62 (medians) | | | 0/10 each |
| A TRAIN r2 R 15 / R 31 | answer only | .73 → .70 / .88 → .87 | 1 | plateau | memorised, 0/10 |
| S R 31 PART / TRAIN r2 | sub-result | .06 / .86 best | .09–.17 / 1 | 60k / 40k | underfit / memorised |
| T R 15 PART r2 / r3; T R 31 PART r2 | trace | .63 → .61 / .85 → .83; .65 → .62 | 1; 1; .71 | budget | memorised; underfit |
| T R 31 TRAIN r2 | trace | .80 | 1 | budget | memorised |
Scratchpad vs answer-only: the executor's single learned rule solves at 500 steps and transfers to n 7/8; no answer-only or parent-trace
cell solves at any rung. No solved delayed-grokking transition exists in the parent cells (A TRAIN r2 crosses 50% test at median 1,750
steps and 99% train at 6,125 / 10,375). Circuit types: S_exec PART — one functional family (addressed AND-hinge bank, NAND readout),
10/10 seeds; S_exec TRAIN — constant majority output, 10/10.

**Diagnostics** (used = mean-ablation drop; R² = best linear probe (layer); rank = effective rank (99% energy); JS = attention
divergence to the HAND head; S_exec on 128 held-out packets per seed):
| Cell | Used | Best probe R² (layer) | Rank (HAND) | JS | Best / final |
|---|---|---|---|---|---|
| S_exec PART (10) | 4 heads + 28–37 of 128 hinges (HAND 4 + 1) | a, b, sum 1; NAND .994–.999 (2) | stage input 2 (1); residual 17–18 (17–18) | 0 | 1 / 1 |
| S_exec TRAIN width 2 (10) | none (every ablation within ±.01) | a, b ≤ .011 | residual 2–4 (17–18) | .65 (uniform = .69) | 0 / local .58–.62 |
| S R 31 PART r2 (s0–2) | stage only (patch drop 1) | D .96, s .97, b² .88 (2) | stage W_in 41 | 0 (frozen) | .06 / .05 |
| A n 6 R 63 at 22.5k (s0–2) | 10–12 heads + MLPs | s .94–.95, b² .93–.94, D .92–.94 | W_in 362–634 | n/a | .945–.955 (200 rows) |
| A R 31 TRAIN r2 (3) | 14 heads + 4 MLPs | b² .96, D .95, s .94 (3) | W_in 300 | .44–.49 | .88 / .87 |
| T R 15 PART r2 (s0–2) | 9 heads + 1 MLP block | D ≈ .94 (2) | stage input 80 (51) | 0 | .63 / .61 |
S_exec PART causal checks: every head's query-mean ablation lowers accuracy to .63–.88; MLP1 patch drop = recovery = 1; interchange on
the sum / NAND variables flips .93–1 / .57–.84 (answer-fixing n 25–38); A/B have no answer-fixing pairs (untested, not unused). Failures
explained: S_exec TRAIN predicts the majority bit (P(NAND = 1) ≈ .61) everywhere — its heads never attend to the addressed registers
(JS .65), so a and b are never read. S R 31 PART decodes D (R² .96) but bits 2⁵…2⁰ fail (.47–.77, first errors there): unit thresholds
on |D| ≤ 1,085. T PART: inexact high-value / borrow / late-sqrt thresholds, not a missing D. A TRAIN: D decodable late and non-linearly
(DAS rescues D in 3/6 audited seeds), no early D → isqrt path.

## Closed form
**S_exec PART (accepted):** the native bank is hⱼ = ReLU(cⱼ + aⱼA + bⱼB), O = 1 + β_O + Σuⱼhⱼ, Y = β_Y + Σvⱼhⱼ, logits (−10·O, 10·Y,
5·O) for other / bit0 / bit1 (direct tensor slices, not a fit). Overwriting all trained MLP tensors with h = ReLU(A + B − 1) writing only
Y (frozen readout unchanged) keeps 1,792/1,792 → 1,792/1,792 fresh held-out packets per seed, agreement 1, on 10/10 seeds (17,920 unique
packets; bar ≥ .99 × native); packets balance the four Boolean cases at register counts 3–256 plus equal-address controls. In
predicted-state execution native and substituted nets agree at every update (507,680 unique updates; 2–8 held-out pairs per width/range
at n 5–6, 4 boundary/sign/zero pairs at n 7/8). Best = final weights, so this is one set of 10 nets, not 20. It certifies the decision
on the tested distribution, not logit equality or one basin. **Parent cells:** no complete form accepted; earlier contextual-bank
substitutions on seven parent finals only preserved their errors, and those originals are no longer available for fresh checks.

## Weights excerpt and reading
HAND stage row (A, B, bias) = (1, 1, −1), writing Y with weight 1; all other stage entries are zero. Trained PART, per seed the neuron
with the largest bit1 − bit0 margin contribution wΔ = 5u − 10v (`part_S_exec_final_s<seed>`; W_in columns A = 37, B = 38):
| Seed / neuron | bias c | a | b | wΔ | mean-pruned hinges |
|---|---:|---:|---:|---:|---:|
| 0 / 25 | −.0044 | .1240 | .0926 | −1.346 | 33 |
| 1 / 56 | −.0128 | .0842 | .0894 | −1.509 | 35 |
| 2 / 71 | −.0099 | .1286 | .0833 | −1.283 | 37 |
| 3 / 42 | .0034 | .1217 | .1214 | −1.160 | 31 |
| 6 / 110 | −.0061 | .1198 | .1646 | −1.050 | 31 |
| 9 / 45 | −.0086 | .1146 | .1118 | −1.365 | 29 |
Reading: seed 0's neuron responds (00, 01, 10, 11) = (0, .088, .120, .212) and writes ONE / Y with −.056 / .107: positive operand weights
raise suppression of the bit1 margin, strongest at 11 — a rescaled AND hinge whose readout gives NAND. About 30 such hinges share what
HAND does with one; seeds 3/4 have small positive biases, so not every hinge is off at 00. The decision is a bank-level sum.

## Comparison
Answer-only and parent-trace stages need unit-resolution thresholds on large scalars; the executor scratchpad turns each step into one bounded-bit rule, which a frozen-parts MLP learns at once and a
from-scratch net of 2× width does not (it never learns the addressing). Scratchpads as supervised intermediate computation follow Nye et
al. (2021); training curves alone are not circuit evidence (Nanda et al., 2023). Supplied program, routing and state copying rule out a
size-crossover or from-scratch quadratic claim.

## Controls and limits
Mean pruning is ordered and calibrated on 256 balanced packets, validated on 1,792 per seed: not a global minimum. Head ablations are
query-mean, not token-global minima. Executor checks are sampled traces and rollouts, not an exhaustive sweep of the coefficient grid.
Parent TRAIN curves used TF32 without full-test fp32 confirmation. Missing: A n 6 decay finals; archived parent TRAIN and r2/r3 nets
(lost); public reproduction code.

## References
- Nye, M. et al. (2021). Show Your Work: Scratchpads for Intermediate Computation with Language Models. arXiv preprint.
  https://doi.org/10.48550/arXiv.2112.00114 (research preprint)
- Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic
  interpretability. arXiv preprint. https://doi.org/10.48550/arXiv.2301.05217 (research preprint)
