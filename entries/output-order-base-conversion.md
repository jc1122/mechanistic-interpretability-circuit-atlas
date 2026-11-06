# Output order in 16-bit base conversion: most-significant-first answers are threshold ladders and are learned; least-significant-first answer-only cells need a residue readout and mostly fail

**Task.** Convert integers x = 0…65535 between 5 decimal digits and 16 bits, both directions (d2b: decimal to binary, b2d: binary to
decimal). The answer is written MSB-first or LSB-first. Format **A** writes only the answer; format **B** first writes 16 scratchpad
lines (d2b: halving lines `r_{k−1} q_k`, q_k = x >> k; b2d: doubling lines v_k = x >> (16−k)), then copies the answer from them.
Prompt `BOS input =`, vocabulary 18. Split per seed: a random permutation of all 65536 integers, 90% train (58982), 10% test (6554).
Exact = every teacher-forced output token of the row.
**Status.** 2026-11-06. HAND exact in all 8 conversion cells (65536/65536 each). Published here: A cells b2d MSB (PART, TRAIN 10 seeds),
d2b MSB (PART, TRAIN 10), d2b LSB (PART, TRAIN 10), b2d LSB (PART 10, failed), B cells d2b MSB, b2d MSB, b2d LSB (10 seeds trained,
seeds 0–2 archived). Pending: d2b B LSB (training interrupted), a companion integer-Newton square-root trace task, a controlled-routing
scratchpad format, ten-seed substitution checks and most best/final diagnostics.

## Summary
Answer order changes difficulty in the opposite direction from the recorded prediction: MSB-first answer-only conversion needs one
threshold ladder on a magnitude F = x − (value already written) and is learned (PART ≥ .99 in all seeds, TRAIN d2b ≥ .996 in 10/10),
while LSB-first answer-only conversion needs a periodic readout (u mod base) of a summed residue and mostly fails (PART .46–.59 d2b,
.18–.38 b2d; TRAIN b2d ≤ .045, also at 2x width). Scratchpads remove the difference: b2d B is solved by PART in 10/10 seeds in both
orders. For PART (trained last MLP on frozen hand-built routing) two weight-derived closed forms — a 366-threshold thermometer for b2d
MSB and a 1054-coefficient spline for the failed d2b LSB parity — replace the trained MLP with identical predictions on every token in
12/12 tested snapshots. A scalar template for the TRAIN nets is rejected (0/12).

## Setup
[METHODS.md](../METHODS.md) for the shared conventions. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000, cosine over the run, clip 1,
batch 128 from the train split; eval every 250 steps on the first 2048 test integers (best snapshot = highest there), full test at the
end; stop when all seeds are exact or on a 5k plateau after the minimum steps. Budgets: A cells PART 20k / TRAIN 80k (d2b MSB PART 10k),
B cells PART 10k / TRAIN 40k. PART fp32, TRAIN TF32; all seeds of a cell in one process on consumer GPUs; re-scores fp32 on CPU.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/output-order-base-conversion](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/output-order-base-conversion)
(88 nets + 7 HAND; best snapshots; each run's held-out integers in `test_<run>`). Reloaded in fp32, every net reproduces its recorded
eval-subset exact (TF32 nets within 3/2048) and every HAND scores 1.0.
**Reproduce:** the generator, HAND and trainer for this task are not yet in `code/`; weights, test integers and accuracies are in the folder above.

## Architecture as used
GPT-NeoX, 3 layers, d_model 512, 4 heads of 128, MLP 256, ReLU, no LayerNorm, no positional encoding (causal mask only), untied
read-out, vocabulary 18, positions up to 256; sequence lengths A 23, d2b B 120, b2d B 104. TRAIN learns every tensor of this shape
(the b2d A LSB width retry, d_model 1024, is not archived). PART: HAND frozen; the layer-2 MLP (`dense_h_to_4h` [256, 512] + bias,
`dense_4h_to_h` [512, 256] + bias) re-initialised (normal .02, bias 0) and trained. PART input blocks (inclusive residual indices):
TOK 0–17, frame scalar 18, raw positions 19…L+17, cleaned positions L+18…2L+16, task scalars from 2L+17 (A: F = 63; b2d MSB unit
T = 64, thermometer Z = 65–74; d2b LSB Y = 64, ONE = 65). Files: `hand_<cell>`, `part_<cell>_s<seed>` (the 4 trained tensors; insert
into the HAND), `train_<cell>_s<seed>` (full state dicts), `test_<run>` (test integers per seed), `config.json` (HF configs, split,
optimiser), `manifest.json` (seed, best step, recorded and re-scored exact, full-test best/final, sha256).

## Prediction before training
Recorded before the runs: PART answer-only MSB fails, LSB succeeds, B succeeds 10/10; TRAIN LSB at least as good as MSB, B ≥ 8/10.
The answer-only order result is the reverse (MSB learned, LSB fails); the B prediction held for b2d in both orders.

## Hand-built reference
L0 builds a positional frame (BOS head, position one-hot), the L1 MLP cleans positions to an exact one-hot, L2 heads fetch exact
weighted mixtures of input digits (BOS sink), and the L2 MLP is the arithmetic stage. **MSB:** heads form F = x − W (W = value of the
written higher digits) and a per-row unit T (2^i or 10^k); digit a wins when F − aT ≥ 0, a comparator ReLU(s+1) − ReLU(s), s = F − aT;
the read-out telescopes the thermometer to a digit. **LSB:** F = (Σ_j digit_j·(base_in^j mod base_out^(q+1)) − W)/base_out^q is an
integer and the digit is F mod base_out (binary parity: alternating signed hinges; decimal: thermometer then mod-10 read-out).
**B:** local halving/doubling rules per scratch line, then the answer copies the last scratch digits (reversed table for B MSB).
Exhaustive check: 65536/65536 integers in every cell. Nonzero heads + neurons incl. the frame: d2b A MSB 3+67, A LSB 3+99; b2d A MSB
3+83, A LSB 3+215; d2b B 3+382, b2d B 2+346.

## What training finds
Full-test exact (6554 integers per seed), best snapshot unless marked; 10 seeds per cell (B TRAIN 10 trained, 3 archived).
| Cell | PART | TRAIN | Reading |
|---|---:|---:|---|
| d2b A MSB | .999–1 | .996–1 | threshold on F |
| b2d A MSB | .993–.997 | .77–.9997 (3 seeds ≥ .998) | thermometer on F (PART); TRAIN not identified |
| d2b A LSB | .46–.59 | .21–1 (3/10 ≥ .9995) | PART keeps F but misses parity precision |
| b2d A LSB | .18–.38 | .03–.045 (also d_model 1024) | decimal residue never read out |
| d2b B MSB | .89–1 (7/10 = 1) | .999–1 | scratchpad |
| b2d B MSB / LSB | 1 in 10/10 (5.3k / 5.5k steps) | .9986–1 / .9985–1 | scratchpad |
b2d B TRAIN first test crossings 50%/90%/99%: MSB 9.3–12.8k/13.5–19.5k/24.8–30k, LSB 9–11k/14.3–17.3k/23.3–28k; no separate
memorisation phase. Diagnostics (1000 rows, seeds 0–2): PART b2d MSB / d2b LSB decode F with probe R² = 1 at L3, 2 heads matter, patching
the L2 MLP drops/recovers everything, attention divergence to HAND 0, residual rank99 2 / 15 = HAND. TRAIN b2d MSB: F R² .978–.985 at L3,
12 heads matter, rank99 49–97 vs HAND 2, attention JS .18–.21; TRAIN d2b: F R² .75–.92 at L2, 9–12 heads, rank99 20–38 vs 15, JS .32–.37.
The TRAIN nets carry F but spread it over many heads; their sufficient neuron counts are not measured.

## Closed form
**PART b2d MSB magnitude (3/3 measured seeds):** M′ = e_Z0 + Σ_{a=1..9} e_Za·[F ≥ θ(q, t, a)], q = query position, t = preceding symbol;
366 thresholds per snapshot found by bisection on the component's own response (no label fitting).
**PART d2b LSB imperfect parity (3/3):** f = clip(F, 0, 33), k = min(⌊f⌋, 32), r = f − k, m = (1−r)c(q,t,k) + r·c(q,t,k+1),
M′ = e_ONE + ((10+m)/20)e_Y; 1054 weight-derived grid coefficients per snapshot. It reproduces a failing read-out, not exact parity.
Substitution (whole L2 MLP output replaced, calibration 512 integers, evaluation 1024 disjoint integers): native = substituted correct
rows, best/final, b2d seeds 0: 1024/1024, 1: 1020/1019, 2: 1018/1018; d2b 0: 585/562, 1: 563/566, 2: 493/470. Every token prediction
agrees in 12/12 snapshots (9362/12288 rows correct both ways); zeroing the MLP scores 0.
**TRAIN (rejected):** decode a scalar û_q = h_q·w_q + b_q and replace the L2 MLP by query × digit mean logit profiles: 0/12 snapshots
pass (b2d .002–.009 vs native .93–1; d2b .009–.18 vs .25–.90). This rejects this template, not every scalar algorithm.

## Weights excerpt and reading
HAND MSB hinge pair: input coefficients (F, T) = (1, −a), biases (1, 0), output (+1, −1), read-out scale 10. Binary parity: input F = 1,
biases −a, output 1 at a = 0 and 2(−1)^a after. Trained PART, final snapshots (fitted response parameters, not raw entries):
| Seed | b2d θ(q=0, t='=', a=1) | d2b c(q=2, t='0', k=14…17) |
|---|---:|---|
| 0 | 9999.44 | (−1.305, −.310, .017, .325) |
| 1 | 10001.96 | (.488, −.131, −.750, −.641) |
| 2 | 9999.44 | (−1.323, −1.275, −1.235, −1.246) |

Raw weights, seed 0 final (W_in columns are slices of `dense_h_to_4h.weight`; β = bias + remaining inputs at q = 0, t = '=', F = 0;
last column = (U_1 − U_0)·W_out[:, j], the actual output column projected on the digit-1 minus digit-0 unembedding):
| Neuron | W_in[j, F] | W_in[j, T] | β | (U₁−U₀)W_out[:, j] |
|---|---:|---:|---:|---:|
| b2d 87 | −.22669 | .21495 | 2149.97 | −20.42 |
| b2d 22 | −.20950 | .19968 | 1997.30 | −17.36 |
| d2b 252 | −.43030 | — | .654 | 103.86 |
| d2b 164 | −.21479 | — | .641 | −102.91 |
Reading: b2d neurons are decreasing hinges in F with a T-scaled offset, i.e. a ladder of cuts at multiples of the place value (θ ≈ 10^4
for the leading digit); their negative writes release evidence for the larger digit as F rises. The d2b pair writes opposite signs and
makes oscillations in F, but their sum has wrong bands (c changes sign irregularly across k): the learned parity is approximate. Only
the whole-block substitution is verified, not a detector identity for single neurons.

## Comparison
Same variable as HAND in PART (F decodable at R² = 1, same heads, same residual rank) and the same threshold algorithm for MSB; for LSB
PART keeps F but fails the periodic read-out. TRAIN nets carry F with many more heads and a higher-rank code. The output-order advantage
reported for reversed addition [Lee et al.] does not transfer to base conversion: there, LSB-first needs a mod on a sum and is harder.
Scratchpads remove the order effect, in line with scratchpad and inductive-scratchpad results [Abbe et al.].

## Controls and limits
HAND controls exact; substitution on calibration/evaluation sets disjoint from the earlier fit sets; zeroed-stage control 0. Six
cell/seed units (best and final), not twelve independent seeds; seeds 3–9 untested for substitution. Displayed 1.000 values from logs are
rounded; per-seed full-test values are in the manifest. Length transfer is not tested (fixed width). Pending: d2b B LSB, the Newton
square-root trace task (PART fails, TRAIN .85–.95 in 5 seeds), the controlled-routing scratchpad format, remaining best/final
diagnostics, checkpoint dynamics, public code.

## References
[Lee et al., Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381) — research preprint; output-order context.
[Abbe et al., How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad](https://arxiv.org/abs/2406.06467) — research preprint; scratchpad motivation.
[Zhou et al., Transformers Can Achieve Length Generalization But Not Robustly](https://arxiv.org/abs/2402.09371) — research preprint; length/padding controls.
