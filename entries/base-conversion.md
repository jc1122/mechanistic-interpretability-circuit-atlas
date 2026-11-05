# Base conversion: hexadecimal to binary, binary to decimal, decimal to binary

**Task.** x = 0..255 (all 256 inputs, one sequence each). Hex to binary `^h1h0=` then 8 bits; binary to decimal `^b7..b0=` then 3 digits;
decimal to binary `^d2d1d0=` then 8 bits (`^` = start token). Format A: answer only. Format B: scratchpad (hex to binary: each hex digit
echoed, then its 4 bits; binary to decimal: Horner doubling v_k = 2 v_(k-1) + b_k written as 3 digits per step; decimal to binary:
halving, remainder bit then quotient, then `=` and the answer). Metric: exact match on all output tokens, teacher forced. Split over
inputs: train fraction .5 for every cell (.8 for TRAIN as a data control).
**Status.** 2026-11-05. HAND exact in 6/6 cells. PART and TRAIN: 10 seeds per cell (TRAIN also at .8, 10 seeds), 2-layer GPT-NeoX.

## Summary
Only the most local conversion is learned from scratch: hex to binary groks in TRAIN (A: 1/10 seeds at fraction .5 and 6/10 at .8; B: 2/10 and
7/10), and 38 of the 40 trained nets carry HAND's variable, the one-hot of the source hex digit (substituting it keeps exact match at
1.00 in 36 of 40), spread over 2-4 heads and 310-490 neurons where HAND uses 1 head and 4-7 neurons. With frozen hand-built attention,
decimal to binary (answer only) learns HAND's rule in 6/10 seeds: every answer bit is one threshold on a remainder scalar, with
theta in [.94, 1.06] instead of 1. The carry-type cells (binary to decimal, scratchpad decimal to binary) fail in PART and in TRAIN (0 of 80 TRAIN nets, and 0 of 30 PART nets, reach test .99).

## Setup
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/base-conversion](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/base-conversion) (6 HAND, 77 PART/TRAIN).
Architecture, optimiser and instrumentation: [METHODS.md](../METHODS.md), section 7. Task-specific: vocabulary of 18 tokens, sequence
lengths 12 / 14 (hex to binary A / B), 13 / 34 (binary to decimal), 13 / 46 (decimal to binary); minibatch 64 of the training
inputs; evaluation every 500 steps; PART at most 10,000 steps, TRAIN at most 50,000; seeds 0-9. Controls: TRAIN at fraction .8, and
for decimal to binary A one TRAIN at lr 3e-4, weight decay .1 (fraction .8, up to 100,000 steps). The published runs packed the 10
seeds in one GPU process (one consumer GPU per cell, 5-25 minutes); `code/` trains one network at a time on CPU.

**Reproduce** (from `code/`, `pip install -r requirements.txt`):
```
python run.py convert check                                                   # the 6 hand-built networks, exact on all 256 inputs (min margin >= 5)
python run.py convert --variant hex2bin --fmt A --rung part  --seeds 0 1 2 3 4 5 6 7 8 9      # PART: exact within ~500 steps
python run.py convert --variant hex2bin --fmt A --rung train --steps 50000 --frac 0.8 --seeds 0 1 2 3 4 5 6 7 8 9
python run.py convert --variant dec2bin --fmt A --rung part  --seeds 0 1 2 3 4 5 6 7 8 9      # thresholds on the remainder
```
(`--variant hex2bin|bin2dec|dec2bin`, `--fmt A|B`, `--frac` default .5.) Smoke run: `--steps 400 --every 100 --seeds 0 1` (seconds for
the answer-only cells; about 30 s per network for the long scratchpads). Output keys per network: `best_test`, `final_test`, `grok_step`
(first evaluation with test >= .99), `analysis.best.exact_substituted` (substitution) and `analysis.best.thresholds` (decimal to binary A).
A full 50,000-step TRAIN network takes on the order of an hour on 4 CPU threads at length 14; reruns do not reproduce the published
seeds bit for bit (different initialiser order, batching and hardware) but the cell-level outcomes.

## Architecture as used
- **All nets (HAND, PART, TRAIN):** GPT-NeoX, 2 layers, d_model 256, 4 heads x 64, MLP 256, ReLU, no LayerNorm, no positional embedding or
  rotary, sequential residual, vocabulary 18 (start, 0-9, a-f, `=`), untied readout. Parameters: embed_in [18, 256], per layer
  qkv [768, 256] + [768], dense [256, 256] + [256], h_to_4h [256, 256] + [256], 4h_to_h [256, 256] + [256], lm_head [18, 256]; 798,720 in total.
  Non-zero HAND weights: 225 / 318 (hex to binary A / B), 271 / 904 (binary to decimal), 241 / 1,796 (decimal to binary).
- **Position frame (all HAND nets, frozen in PART).** Layer-0 head 0 attends the start token with score ln 64 and writes P = 64/(64 + r); the
  layer-0 MLP has one hinge neuron per position (L - 2 of them) that interpolates the position one-hot POS[m] on the P grid (coefficients from one linear
  solve). Layer-1 "table" heads use key = POS one-hot and query = sum over designed (r, j, w) of POS_r times a row, score 30 + ln w.
- **PART:** everything except the layer-1 MLP is HAND. Trained: `gpt_neox.layers.1.mlp.dense_h_to_4h` W_in [256, 256], b_in [256] and
  `dense_4h_to_h` W_out [256, 256], b_out [256] (131,584 parameters), initialised N(0, .02) / 0.
- **TRAIN:** all 798,720 weights trained from N(0, .02) / 0.
- **Input blocks of the trained stage in PART** (residual dims at the layer-1 MLP input): TOK one-hot 0-17, P 18, POS one-hot 19..(18 + L - 1), then
  HAND's layer-1 variable: X (16-wide nibble one-hot, hex to binary), F (remainder scalar, binary to decimal and decimal to binary A) or C, X (decimal to binary B).
- **Archive.** One file per network (PART: the four stage tensors; TRAIN: the full state dict; both with `test_idx`), the six HAND files,
  `config.json` and `manifest.json` (seed, rung, fraction, test accuracy, circuit type, sha256): 77 trained nets (hex to binary: all 20 PART and 40 TRAIN nets; decimal to binary A: all 10 PART; one best PART and one best TRAIN-at-.8 net
  for each other cell) and 6 HAND nets, each reloaded and checked to reproduce its recorded test accuracy to one input. Public hosting
  of the weights is pending; `code/convert.py` rebuilds the six HAND nets bit for bit.

## Prediction before training
Recorded in the design notes before any PART or TRAIN run (committed 2026-11-04 23:24; the first training run finished after 23:44): look in the
trained nets for weighted-mixture attention heads (place-value weights) versus a per-digit lookup, and for thresholds on a scalar remainder
(answer-only formats) versus doubling / halving tables (scratchpad formats). Held: hex to binary is a per-digit lookup on the read nibble
(PART and TRAIN), and decimal to binary A in PART is a remainder threshold. Not testable: no carry-type cell was learned from scratch, so the mixture-head prediction was never confronted by a trained net. No
prediction about grokking, net size or which cells would fail was recorded; the failure of the carry-type cells and the 50-100x size gap are post-hoc observations.

## Hand-built reference
All six nets share the position frame above. The per-cell stage (layer-1 heads, then MLP):

| Cell | Layer-1 heads | MLP stage | Readout | Used heads + neurons |
|---|---|---|---|---|
| hex to binary A | head 0 reads the source hex digit (h1 for answer bits b7..b4, h0 for b3..b0): X = its one-hot | 4 neurons relu(sum over v with bit j set of X_v + gate_j - 1) | output one-hot, '1' by 2 x neuron, '0' by bias | 1 + 4 |
| hex to binary B | as A; also copies the echoed digit | 8 neurons (bit-1 and bit-0 per j) | same | 1 + 7 |
| binary to decimal A | head 0: exact weighted mixture of the 8 bits, F = N/s (s = 100, 10, 1); head 1 subtracts the written digits | 18 step neurons [F >= c], c = 1..9 | logit_d = Z_d - Z_(d+1) (thermometer) | 3 + 18 |
| binary to decimal B | head 0: F = 2 x previous value digit + new bit | 38 steps at 1..19 | floor mod 10 | 1 + 37 |
| decimal to binary A | head 0: F = N/2^i (mixture of d2, d1, d0); head 1 subtracts the written higher bits | 2 neurons: [F >= 1] | '1' vs bias '0' | 2 + 2 |
| decimal to binary B | heads read the current digit C and X (higher digit, or the written bit) | 26: parity of X (hinges) and floor(C/2) | quadratic logit in the digit | 2 + 26 |

Mechanism in words. A table head scores 30 + ln w on a designed (query position, key position, weight) triple, so softmax over those keys
(plus the start token as a value-0 sink) returns exactly the weighted mixture of the key tokens' values. Hex to binary needs one weight 1 (copy a digit);
decimal to binary A needs the weights 100/(111 2^i), 10/(111 2^i), 1/(111 2^i) and a sink weight 1 - 2^-i, which with value 111 x digit yield F = N/2^i;
the written bits are then subtracted with weights 2^(j-i)/254, so F is the remainder x/2^i minus the higher bits, in [0, 2), and the bit is [F >= 1].
Exactness: all six nets are exact on 256/256 inputs (token accuracy 1.0000), minimum logit margin 10.0 / 10.0 (hex to binary), 9.0 / 9.9 (binary to decimal), 8.8 / 5.0 (decimal to binary), checked with `python run.py convert check`.
Answer-only is smaller than the scratchpad here because with a fixed length one mixture head reads both the input and the already written answer digits.

## What training finds
Test = exact match on held-out inputs; *n* = nets whose best test reached .99; grok = first evaluation with test >= .99. Best / final: best
evaluation over the run / last evaluation (final weights can be far lower).

| Cell | PART best / final, n, grok | TRAIN .5 best / final, n, grok | TRAIN .8 best / final, n, grok |
|---|---|---|---|
| hex to binary A | 1.00 / 1.00, 10/10, 500 | .70-.99 / .52-.86, 1/10, 35.5k | .94-1.00 / .08-.98, 6/10, 8.5k-23.5k |
| hex to binary B | 1.00 / 1.00, 10/10, 500 | .81-.99 / .75-.99, 2/10, 39k and 40.5k | .96-1.00 / .02-1.00, 7/10, 3k-29k |
| binary to decimal A | .56-.76 / .50-.71, 0 | .02-.05 / .00-.03, 0 | .06-.19 / .00-.10, 0 |
| binary to decimal B | .40-.48 / .27-.48, 0 | .23-.47 / .03-.44, 0 | .31-.73 / .04-.63, 0 |
| decimal to binary A | .98-1.00 / .89-1.00, 6/10, 500-8k | .04-.16 / .02-.09, 0 | .12-.58 / .04-.52, 0; lr 3e-4, wd .1: .10-.25, 0 |
| decimal to binary B | .33-.51 / .15-.42, 0 | .11-.28 / .00-.18, 0 | .29-.62 / .13-.35, 0 |

- **Hex to binary, PART (20/20).** Exact by step 500. The only head read is the frozen nibble head; the trained MLP is a lookup on the nibble one-hot
  (substituting HAND's X fit: 1.00); the nibble as a linear *value* does not suffice for format B (.02-.04). 7-38 (A) and 51-72 (B) neurons vs HAND 4 / 7.
- **Hex to binary, TRAIN.** Memorisation by 2.5-13k steps, grokking 3k-40.5k steps (16/40 nets); instability afterwards (final test down to .02 while the best snapshot is exact).
  38/40 nets are the *nibble one-hot* type (one hex-digit one-hot fit of the residual after layer 1 suffices); the read is spread over 2-4 layer-1
  heads with fixed attention patterns (2-7 keys, cosine to HAND's pattern .5-.97) after 2-4 layer-0 heads that average all positions;
  4-8 heads + 310-490 neurons vs HAND 1 + 4 / 7.
- **Decimal to binary A, PART (10/10 read the remainder).** The nets read HAND's remainder F (substitution .95-1.00 = full); 6/10 reach test >= .99
  at their best; each answer bit is a single threshold [F >= theta], agreement with the net .984-1.000 per position.
- **Binary to decimal A, PART (unsolved).** Reads F (substituting F *raises* exact from .72-.81 to .82-.94 in 8/10), errors are off-by-one digits
  in tens and units: a blurred threshold ladder plus memorising features; about 200 neurons vs 18.
- **Binary to decimal B, PART (unsolved).** F alone gives .02; the read tokens (or the current token) are needed (.54-.64).
- **All carry-type TRAIN nets use all 8 heads and both MLPs (420-510 neurons) and have no closed form.**
- Diagnostics (medians over solved nets, exact ≥ .9; * = cell without a solved net, all 10 nets): used heads+MLPs; patch = component with the
  largest patching drop (nets); var inter/DAS = linear interchange / DAS flip, best layer; edges = EAP-IG jaccard/hand_recall vs HAND, trusted nets (spearman_exact ≥ .8); diag exact best / final:

| cell | HAND | PART | TRAIN frac .5 | TRAIN frac .8 |
|---|---|---|---|---|
| hex2bin A | 1+1; L1H0; X 1/1; edges 1/1 | 1+1; L1H0 6, MLP1 4; X 1/1; edges 1/1 (10); 1 / 1 | 7-8+2; MLP1 9/9; X untested (nf < 30); edges .50/.67 (3); .85-1 / .54-.93 | 7-8+2; MLP1 10/10; X untested; edges .20/.67 (7); .98-1 / .09-1 |
| hex2bin B | 1+1; L1H0; X 1/1; edges 1/1 | 1+1; L1H0 10/10; X .94/1; edges 1/1 (9); 1 / 1 | 5-8+2; MLP1 9, MLP0 1; X 1/1; edges .33/.50 (7); .91-1 / .87-1 | 4-8+2; MLP1 10/10; X 1/1; edges .33/.50 (9); .96-1 / .01-1 |
| dec2bin A | 2+1; MLP1; F 1/1; edges untrusted | 2+1; MLP1 10/10; F 1/1; edges 1/1 (4 of 9); final .93-.99 | * 5-8+2; MLP1 9, MLP0 1; F .87/.87; edges .25/.50 (4); final .11-.55 | * 8+2; MLP1 7, MLP0 3; F .89/.95 (lr 3e-4: .82/.94); edges .25/.40 (8); final .24-.86 |
| dec2bin B | 2+1; MLP1; C .24/.49, X .42/.42; edges 1/1 | * 2+1; MLP1 10/10; C .47/.98, X .63/.98; edges 1/1 (5); final .21-.52 | * 0-8+0-2; MLP0 7, MLP1 2; C .18/.91, X .32/.86; edges .25/.40 (7); final .00-.52 | * 8+2; MLP0 4, L1H2 2; C .15/.93, X .29/.44; edges .25/.40 (4); final .24-.66 |
| bin2dec A | 3+1; MLP1; F 0/.32; edges untrusted | * 2+1; MLP1 10/10; F .11/.42; edges 1/1 (4); final .71-.81 | * 7-8+2; MLP0 6, MLP1 4; F .07/.30; edges .11/.40 (8); final .43-.51 | * 8+2; MLP0 6, MLP1 4; F .05/.30; edges .11/.40 (10); final .73-.80 |
| bin2dec B | 1+1; L1H0; F .09/–; no edges | * 1+1; MLP1 10/10; F .17/.48; edges 1/1 (8); final .43-.64 | * 8+2; MLP0 10/10; F .10/.31; edges .20/.33 (6); final .08-.71 | * 0-8+0-2; MLP0 7, MLP1 2; F .12/.31; edges .20/.33 (7); final .03-.88 |
- Causal: PART patch top = the frozen read head L1H0 (hex2bin, 16/20) or the trained L1 MLP; PART edges = HAND wiring (jaccard 1.0, hand_recall 1.0; 43/48 solved nets trusted,
  also in the failed PART cells). TRAIN wiring differs (jaccard .11-.50; heads not aligned: a different-wiring flag, not a match score); TRAIN patch top MLP1 (63/64 solved). Present but not linear: dec2bin B C, X (PART inter .47/.63, DAS .98). bin2dec F is neither (inter ≤ .17, DAS ≤ .48 in every rung; HAND 0 / .32).

## Closed form
- Hex to binary: bit_j = bit_j(X) of the read nibble X. Substituting the per-position least-squares fit of the nibble one-hot for the residual after layer 1
  gives exact 1.00 on all 256 inputs in 36 of 40 TRAIN nets (A .5: 9/10, A .8: 10/10, B .5: 8/10, B .8: 9/10; the others .94), above the nets' own
  accuracy (.85-.99): the trained nets carry HAND's variable and their errors come from extra residual components.
- Decimal to binary A (PART): bit_q = [F_q >= theta_q], theta_q in [.94, 1.06]; the net agrees with its own best-threshold rule on .984-1.000 of inputs
  per answer position; with theta = 1 (HAND's rule) the rule is HAND (exact). All errors of the nets lie between theta and 1, at the high bits where F's grid is finest (1/128).

## Weights excerpt and reading
**HAND hex to binary A** (non-zero pattern). Layer-1 head 0 has 8 / 11 / 16 non-zero entries in q / k / v. The four MLP neurons j = 0..3 have bias -1, weight 1 on
X_v for the 8 nibble values with bit j set (j = 0: v = 1, 3, 5, ..., 15; j = 1: 2, 3, 6, 7, ...; j = 2: 4-7, 12-15; j = 3: 8-15) and weight 1 on the position one-hot
entries of the answer slots that carry bit j (j = 0: slots 6 and 10; j = 1: 5, 9; j = 2: 4, 8; j = 3: 3, 7).
**HAND decimal to binary A.** The two step neurons have weight 128 on F and bias -127 and -128 with output weights +2 / -2 on the '1' logit, i.e. [F >= 1] on the 1/128 grid;
the value rows are digit value x 111 (head 0) and x -254 (head 1).

**PART hex to binary A, seed 0** (test 1.00). Mean over inputs of logit('1') - logit('0') at the answer bits from the first hex digit (b7..b4), by its value v:

| v | b7 | b6 | b5 | b4 | bits of v |
|---|---|---|---|---|---|
| 0 | -11.9 | -11.2 | -10.8 | -11.1 | 0000 |
| 1 | -9.2 | -8.5 | -7.9 | **7.4** | 0001 |
| 2 | -9.2 | -7.5 | **7.4** | -9.2 | 0010 |
| 5 | -8.6 | **7.9** | -8.0 | **8.8** | 0101 |
| 8 | **7.9** | -9.0 | -8.0 | -8.5 | 1000 |
| a | **8.3** | -7.8 | **7.9** | -8.2 | 1010 |
| d | **10.1** | **7.5** | -7.5 | **9.8** | 1101 |
| f | **12.3** | **11.0** | **10.5** | **11.4** | 1111 |

(bold = positive; the sign pattern equals the bits for all 16 values; the h0-driven bits b3..b0 give the same table to within 1.) Computed from the
archived net: forward pass on all inputs, group by the source digit.

**PART decimal to binary A, per seed**: best threshold theta (agreement with the net) per answer bit, `convert.thresholds`; bits b3..b0 are 1.000 (1.00) in every seed.

| seed (test) | b7 | b6 | b5 | b4 |
|---|---|---|---|---|
| 0 (.953) | 1.008 | 1.000 | 1.031 | 1.000 |
| 1 (.945) | 0.961 | 1.000 | 0.969 | 1.000 |
| 2 (.922) | 1.055 | 0.984 | 0.938 (.98) | 1.000 |
| 3 (1.000) | 1.000 | 1.016 | 1.000 | 1.000 |
| 4 (.891) | 1.016 | 1.000 | 1.063 | 1.063 |
| 9 (.977) | 1.000 | 1.000 | 1.000 | 1.000 |

(agreement 1.00 unless shown.) **PART hex to binary B / TRAIN hex to binary**: substitution rows in "Closed form"; the A TRAIN net at fraction .8, seed 0 (test 1.00) has the
same sign pattern as the PART table but with logit differences of 6-42 instead of 7-12: the extra residual components inflate the margins.

Reading. In hex to binary the frozen head delivers the nibble one-hot, so the stage is a 16-entry table per answer bit; a ReLU layer realises it as a bit-pattern
lookup: the sign table above is HAND's bit pattern, with the same +-8 magnitude for all values (HAND's own margin is 10). In decimal to binary the frozen heads deliver F exactly
(a multiple of 1/2^i), so a learned single step must sit between the largest F below 1 and 1; 128 training inputs
pin theta only to the nearest observed F values, so theta drifts by up to .06 at the high bits, where the F grid is finest and unseen values fall between theta and 1.

## Comparison
Same algorithm where it is learned: hex to binary is a per-digit lookup in HAND, PART and TRAIN; decimal to binary A in PART is HAND's remainder threshold. Trained nets are 50-100x larger than HAND
(4-8 heads and 310-490 neurons vs 1 head and 4-7 neurons), and TRAIN spreads the digit read over several heads. HAND's answer-only mixture heads (place-value weights, written digits subtracted) were
not found by training: no trained net solved a carry-type cell from scratch. No published circuit for these conversions was checked for this entry, so no comparison to the literature is made.

## Controls and limits
- Length generalisation is not testable: every cell has a fixed input width and sequence length, and HAND and PART read by absolute position (position one-hot from the start-token head).
- The substitution fit uses all 256 inputs (test included). Mean-ablation sizes do not count the input-independent position frame.
- TRAIN outcome depends on the data size: 128 (fraction .5) or 205 (.8) training inputs; the carry-type cells may need more data, more steps or a different optimiser; lr 3e-4 / wd .1 did not help decimal to binary A.
- Final weights are unstable under weight decay 1 (a net can fall from exact to .02 after grokking), so "best" numbers are best snapshots on the test split, an optimistic selection.
- 10 seeds per cell; "n = 0" means no seed reached .99, not that none can.

## References
- Lee, N., et al. (2023). Teaching arithmetic to small transformers. https://doi.org/10.48550/arxiv.2307.03381 (output order, scratchpads)
- Nye, M., et al. (2021). Show your work: scratchpads for intermediate computation with language models. https://doi.org/10.48550/arxiv.2112.00114
