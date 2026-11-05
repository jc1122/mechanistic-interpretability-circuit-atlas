# Subtraction (base 10 and 2): a per-column lookup on the signed difference a − b − borrow, not the adder's sum
**Task.** `a-b=` → a − b with a ≥ b; base 10, 3 digits (per seed 20k train / 2k test random pairs); base 2, 6 bits (train fraction .6 of
the 2080 pairs: 1248 / 832). A: n answer digits, MSB first; B: d_0 k_1 d_1 .. k_n (LSB first, k = borrow token); C: signed column
differences e_i = a_i − b_i (n tokens), `=`, d_{n−1}..d_0. Exact match of all output tokens.
**Status.** 2026-11-05. HAND per base × format (6 cells); PART and TRAIN 10 seeds per cell; failed cells retried once (all retries 10/10; both runs kept).
## Summary
Every trained stage that works computes its digit as a lookup keyed on diff = a_i − b_i − borrow_in (closed form f(diff); the prediction held):
PART 10/10 in base 10 A/B and base 2 A/B/C, 9/10 in base 10 C; TRAIN's last MLP in 21/30 (base 10, run 1), 29/30 (base 2, run 1) and 34/40 (retries).
The adder's key a + b + c never substitutes (≤ .01 in base 10, ≤ .88 in base 2, except binary A where the two keys coincide). PART A extrapolates
to 4 and 6 digits; B and C do not, because their trained stage reads absolute position; holding those weights at 0 restores length (B .87–.98 at n + 1).
TRAIN never length-generalises.
## Setup
[METHODS.md](../METHODS.md). AdamW lr 1e-3, wd 1, betas (.9, .98), batch 256, TF32, eval every 500; PART 10k, TRAIN 20k, stop on 1.0 or a 2k
plateau (base 2 20k); best-test snapshot. Retries: warmup 1000 + cosine + clip 1, 30k (base 2 wd 1, base 10 wd .1). Control: PART with the
stage's position inputs held at 0. Length: 4 and 6 digits (base 10), 7 and 12 bits (base 2). Consumer GPUs.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/subtraction](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/subtraction) (112: 6 HAND, 60 PART, 30 PART with position mask, 16 TRAIN).
**Reproduce:** the training and analysis code for this task is not yet in `code/` (`python run.py` has no subtraction task), so the learned cells cannot be rerun from this repository; the weights, test splits and the table of accuracies are in the folder above.
## Architecture as used
GPT-NeoX 4 layers × 4 heads, d_model 256, d_head 64, MLP 256 ReLU, no LayerNorm, NoPE; vocabulary BOS, unk, the 2b − 1 signed values
−(b−1)..b−1 and `- = . c` (25 tokens in base 10, 9 in base 2). PART trains the stage MLP only (A layer 3, B/C layer 2: dense_h_to_4h [256, 256] + bias,
dense_4h_to_h [256, 256] + bias), reading TOK, X1/X2/X3 (fetched column digits), CY (borrow), PH/EQS (C phase) and the position blocks
P/R/R2/T/RP/VT; TRAIN: all 1.6M weights from init.
Archive (Hugging Face folder above): `hand_b<base>_<F>` (HAND state dicts), `part_b<base>_<F>_s<seed>` (all PART seeds; the four stage tensors),
`part_b10_<F>_maskpos_s<seed>` (position control), `train_b<base>_<F>[_r2]_s<seed>` (full nets, best and worst seed per cell and run; `_r2` = retry),
`config.json`, `manifest.json` (sha256, fp32 re-score); each file reloaded bit-identically.
## Prediction before training
Recorded in the hand-built design notes before the runs: "per column (a − b − c) mod b", binary digit = a XOR b XOR borrow; open question
"does TRAIN reuse the adder shape (f(a + b + c))?". Held: f(diff) in PART and in most TRAIN last MLPs; the adder shape is not reused. Nothing was
recorded about grokking time or length.
## Hand-built reference
The [decimal-hex-addition](decimal-hex-addition.md) bricks with borrow for carry. A: layer-2 index heads copy a_i, b_i, and a_j at b_j's position;
the L2 MLP on D = b_j − a_j writes EQ = [D ≠ 0] and GEN = [D ≥ 1]; the L3H0 chain head attends to the first lower column with EQ and copies GEN
(= borrow_in); the L3 MLP is the one-hot of (a_i − b_i − c) mod b (2b hinge triangles). B: the L2 MLP on z = a_i − b_i − [own token `c`] + b gives
digit z mod b and borrow 1 − [z ≥ b]; L3H0 copies the previous borrow. C: phase 1 writes e_i = a_i − b_i; the chain head finds the first e_j ≠ 0
(borrow = [e_j < 0]); phase 2 writes the one-hot of (e_i − c) mod b. Exact for n = 1..12 in both bases. Size (heads; neurons L0/L1/L2/L3):
base 10 A 6; 64/114/4/22, B 5; 64/64/3/14, C 5; 64/119/43/0 (base 2: L3 6/6/0, L2 4/3/11).
## What training finds
- PART base 10: test 1.0 / .995–1 / .97–1 (A/B/C), t(.9) ≤ 1k. Length n + 1 / 2n: A .80–.98 / .80–.96, B ≤ .20 / ≤ .04, C ≤ .02 / 0; with the stage's
  position inputs at 0: A .99–1 / .98–1, B .87–.98 / .39–.60, C .95–.99 / .52–.59 (the 2n errors fall on new joint (value, role) inputs).
  PART base 2: A, B 1.0, C .93–.95 (fp32 re-score 1.0); length n + 1: A .78, B 0–.82, C .13–.42; 2n ≤ .67.
- TRAIN base 10 best (final): A .72–.99 (.01–.90), B .94–1 (.01–1), C .48–1 (0–.98); seeds ≥ .9: 7, 10, 9; t(.9) 3–12k. Base 2 run 1: A .40–.95 (1/10
  ≥ .9), B .79–1 (9/10), C .55–.95 (4/10). Retries (base 10 A, C; base 2 A, C): 10/10 = 1.0 each (final = best), t(.9) base 10 A 2.5–5.5k, C 2–4.5k,
  base 2 A 8.5–14.5k, C 6–11.5k. Length n + 1 ≤ .06 (base 2 ≤ .35), 2n ≤ .001.
- Circuit types (last-MLP substitution ≥ net − .01): PART f(diff) base 10 A 10, B 10, C 9 + f(a, b, c) 1; base 2 A/B/C 10/10. TRAIN f(diff) base 10 A 7,
  B 8, C 6 (+ f(a, b, c) pair lookup 4: the compact 3–4-head nets); base 2 run 1 A 9, B 10, C 10; retries base 10 A 10 (substitution ≥ .99), C 8 + f(a, b, c) 2;
  base 2 A 6 + f(a, b, c) 3 + none 1, C 10. Earlier TRAIN MLPs have no closed form.
- Frozen parts: PART reads all the HAND heads it needs (attention substitution 1.0). TRAIN uses 3–16 heads + 300–980 neurons (HAND 5–6 + 75–230);
  HAND's fetch roles (L2 index heads) are re-found in the retries (base 10 A 9/10, C 9/10, base 2 C 9/10, base 2 A 2/10; 3–16 heads + 226–762 neurons).
- Diagnostics (used heads + MLPs; best-layer linear-probe R² of a, b (X1/X2 or a_i/b_i), borrow c (and diff); residual participation ratio PR at L4;
  attention JS of used heads to the matching HAND head; test best (final)):

| cell | PART | TRAIN best (final) |
|---|---|---|
| b10 A | 4+2; a, b, c 1; PR 3.4–3.6; JS 0; 1 | 4–16+4; a .74–.90, b .70–.89, c .52–.62; PR 2.1–3.2; JS .09–.35; .72–.98 (.01–.90). Retry 4–8+4; a .75–.88, b .79–.92, c .59–.74; JS .16–.39; 1 (1) |
| b10 B | 2–3+2; a, b .83–.86, c .64–.67, diff .85–.91; JS 0–.16; .91–.99 | 10–16+4; a .58–.84, c .65–.86, diff .85–.98; JS .08–.19; .94–1 (.01–1) |
| b10 C | 3+1; a, b, c, phase 1; JS 0; .97–1 | 3–16+4; a .61–.83, b .55–.73, c .65–.89; PR 2.5–4.7; JS .08–.39; .48–1 (0–.97). Retry 3–15+3–4; a .70–.81, c .80–.96; JS .10–.34; 1 (1) |
| b2 A | 4+2; a, b, c 1; PR 5.4; JS 0; 1 | 11–16+4; a .39–.50, b .58–.70, c .28–.38; JS .04–.14; .39–.95. Retry 6–16+4; a .41–.68, b .61–.75, c .32–.48; JS .08–.31; 1 (1) |
| b2 B | 3+2; a, b .79–.84, c .72–.83, diff .89–1; JS .15–.17; 1 | 5–13+4; a .47–.72, c .58–.85, diff .84–.94; JS .05–.19; .79–1 |
| b2 C | 3+1; a, b, c, phase 1; JS 0; 1 | run 1 6–16+4; c .50–.64; .54–.95. Retry 3–9+3–4; a .56–.75, c .51–.89; JS .18–.38; 1 (1) |

- Failures explained: failed TRAIN nets keep the stage key (diff R² .84–.98) but carry a, b and the borrow only partly linearly (R² .3–.9);
  the base 10 A/C failures oscillate under wd 1 with a constant learning rate (final ≪ best); the retries (warmup + cosine + clip) solved every
  seed (base 2 C at wd 1, base 10 at wd .1). PART B/C length failure: the stage reads position (clamping those inputs to their mean drops n = 3 to ~.01).
## Closed form
Last MLP replaced by its mean output given diff = a_i − b_i − c (C phase 1: given a_i − b_i): accuracy = net accuracy (PART .97–1; TRAIN f(diff)
nets ± .01; base 2 C retry 1.00). digit = (a_i − b_i − c) mod b, borrow_out = [a_i − b_i − c < 0].
## Weights excerpt and reading
Numbers computed from the archived PART base 10 stage tensors (W_in rows on X2/X1/CY; B: X1/X2/token `c`; neurons weighted by ||W_out column|| × ||W_in row||).
HAND A L3 hinges read (a, b, c) = (1, −1, −1) with bias −v (n18/n19: −7, −8). Trained PART: the output-norm-weighted |cos| of each neuron's
(w_a, w_b, w_c) with (1, −1, −1) is A .88–.93, B .72–.78; C phase 2 (e, c) .63–.68. Examples: A s0 n16 (.04, −.39, −.39), A s2 n7 (−.05, .05, .06),
B s2 n142 (−.20, −.21, −.19) bias .02.
Reading: the A/B neurons are hinges on the line a − b − c (b and c with equal weight, as in HAND), so the stage is a function of diff; B mixes in
sum-like neurons (cos .72) that the borrow token disambiguates.
## Comparison
PART = HAND (diff lookup, wider). TRAIN finds diff lookups without HAND's chain head. [decimal-hex-addition](decimal-hex-addition.md) and
[binary-addition](binary-addition.md) give f(sum): the column key follows the operation. The position shortcut of B/C and its mask control
match binary addition.
## Controls and limits
One training length per base; the position-masked PART is the length control. Probes are linear. Nets trained in TF32; fp32 re-score: base 10 B
PART .91–.99, base 2 C PART 1.0. Binary A cannot separate diff from a + b + c (the keys coincide mod 2).

## References
None.
