# Binary long division and long-hand square root: a scratchpad step is one learned compare; answer-only nets need 16–90x more data
**Task.** Division `n..n0 / d3..d0 =` → A (answer only): quotient bits, remainder bits; B (scratchpad): per step [q_i, r3..r0]. Square root
`n..n0 =` → A: root bits, remainder bits; B: per step [b_i, remainder bits]. 8-bit range: dividend N 8 bits / divisor D 4 bits (3840 pairs),
square root N 8 bits (256 numbers), train fraction .5. Wide range: division N 12 bits (61440 pairs, train .5), square root N 14 bits
(16384 numbers, train .7). Exact match of the whole answer on the held-out split.
**Status.** 2026-11-05. HAND exact in all 8 cells (2 operations × 2 formats × 2 ranges). PART and TRAIN 10 seeds per cell; every failed cell
retried (all runs kept). Pending: 2x-width division B seeds 0–4; square root B TRAIN 2x-width rerun; continued runs (+30k from the final nets at
lr 3e-4) of the cells still rising at their budget: division B and square root B TRAIN (wide), square root A PART (wide, annealed), square root B PART (8-bit).
## Summary
With a scratchpad (B) the trained stage reproduces the hand-built per-step compare-and-subtract as one circuit shared by all steps: division B
PART 10/10 exact at both ranges (90% test by step 250–500), square root B PART .985–.992 at the wide range (errors only at the last step).
Answer-only (A) PART stages fail at every range and schedule (≤ .29): they get every input exactly but cannot place unit-resolution thresholds
on 2^8–2^14-scale values. From scratch nothing generalises at 8 bits (train 1.0, test ≤ .81), but at the wide range (16–90x the data) it does,
even answer-only: division A .995–1.0 (10/10 ≥ .99), square root A .981–.993 (2/10; 6/10 at 2x width), division B .88–.999 (7/10), square
root B .21–.95 at the 30k budget and still rising. No trained net is exact on all inputs.
## Setup
[METHODS.md](../METHODS.md). AdamW betas (.9, .98), batch 256, eval every 250 steps, best and final snapshot. PART: fp32, frozen HAND, lr 1e-3,
wd 1, constant for 10k steps; retry warmup 1000 + cosine + clip 1, 30k. TRAIN: TF32, all weights from the default initialisation, lr 1e-3,
wd .1, warmup 1000, cosine, clip 1, 30–60k steps, stop on a 5k plateau after 15k (8-bit also constant lr with wd 1, and lr 3e-4). Size ladder
step 1 = hidden 512. Consumer GPUs (8–24 GB).
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/long-division-square-root](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/long-division-square-root) (202: 8 HAND, 160 PART, 34 TRAIN; continued runs added when finished).
**Reproduce:** the training and analysis code for this task is not yet in `code/` (`python run.py` has no division or square-root task), so the learned cells cannot be rerun from this repository; the weights, test splits (`test_idx` in each file) and the table of accuracies are in the folder above.
## Architecture as used
GPT-NeoX 3 layers × 4 heads (d_head 64), d_model 256 (ladder: 512, d_head 128), MLP 1024 ReLU, no LayerNorm, NoPE; vocabulary 5 (BOS, `0 1 / =`).
PART re-initialises and trains the compare stage on frozen HAND: 8-bit the layer-1 MLP, wide the layer-2 MLP (dense_h_to_4h [1024, 256] + bias,
dense_4h_to_h [256, 1024] + bias); 8-bit second shots also train layers 1 and 2. TRAIN: all weights.
Archive (Hugging Face folder above): `hand_<op>_<F>[_wide]` (HAND state dicts), `part_<op>_<F><run>_s<seed>` (all PART seeds; the stage tensors),
`train_<op>_<F><run>_s<seed>` (full nets, best and worst seed per run), `config.json`, `manifest.json` (seed, rung, shape, fp32 test accuracy,
recorded accuracy, sha256); each file reloaded bit-identically.
## Prediction before training
8-bit (recorded before the runs): division A PART < .5 held; division A/B TRAIN < .3 broken (.64–.81 / .53–.69); division B errors grow with the
step held. Wide range (recorded before the runs): PART division B 10/10 held; square root B ≥ 8/10 broken (1/10 ≥ .99); square root A 10/10 and
division A ≥ 5/10 broken (≤ .29); TRAIN division B ≥ 5/10 held (7/10); square root B ≥ 3/10 broken at 30k (0/10 ≥ .99, 3/10 ≥ .9; continuation
pending); answer-only TRAIN < .5 broken (division A .995+, square root A .98+).
## Hand-built reference
B: layer-1 heads write X = 2·rem + next bit (square root: 4·rem + 2 bits), TT = D (square root: 4p + 1, p = root so far), B = own bit, W = remainder
bits already written; one MLP serves all steps: bit = [X ≥ TT], remainder bits = steps of X − B·TT − W (8-bit division 4 heads + 14 neurons,
square root 4 + 20; wide 7 + 18 and 7 + 34). A 8-bit: the layer-1 MLP holds all thresholds, division 255 compares [N − mD ≥ 0] → Q (5 heads + 186
neurons), square root 15 [N ≥ m²] (2 + 46). A wide (sequential): each output reads the written prefix; division q bit = [N − d(Wq + 2^j) ≥ 0] gated
by one-hot D (4 + 525); square root one step [N − SQ − T ≥ 0] for all outputs (4 + 3). Exact on all inputs at both ranges (fp32, margin 10).
A scratchpad shrinks the circuit: division 14 vs 186 neurons, square root 20 vs 46 (8-bit).
## What training finds
- PART B: one per-step circuit shared by all steps (8-bit division step-Jaccard .985, 7.9 of 8 steps per neuron; wide division .98, 11.9/12;
  square root .87, 6.5/7), reading the hand-built heads, 600–700 neurons vs HAND 14–34. Square root B: 8-bit .91–1.0 annealed (2/10 = 1.0), wide
  .985–.992; all errors at the last step (largest x, T).
- PART A: the stage inputs are exact (N, x, T R² 1.0), but unit-resolution thresholds on 2^8–2^14-scale values are not placed: 8-bit ≤ .16
  (Q exact 4 %, r0 = n0 ⊕ b0 at chance); wide constant lr ≤ .006, annealed square root .12–.29 (rises only as lr → 0), division .04–.06 (q0 .59, r0 .60).
- TRAIN 8-bit: train 1.0, test division A ≤ .81, division B ≤ .69, square root ≤ .28 (128 training numbers): lookup; x − T is never linear.
- TRAIN wide: division A errors only at q1, q0, r3, no linear Q (quotient R² ≤ .86); division B errors at the late steps with x − T R² ≤ .67 (a
  nonlinear route; at 2x width, seeds 5–9: .969–.998, 3/5 ≥ .99, same probes); square root A errors at b0 and the top remainder bits, 11 heads +
  700–1100 neurons vs HAND 4 + 3; square root B at 30k errors at steps 6–7 only (T R² 1.0, x − T ≤ .73), still rising: under-trained, not memorising.
- Diagnostics (best + worst seed, final / best snapshot; probes = best-layer ridge R²; PR = residual participation ratio at layer 3 (HAND 1.0–1.9);
  JS = attention divergence of HAND's heads):

| cell | used components | probes | PR | JS | best / final | patch (nets) | inter/DAS | edges (trusted) |
|---|---|---|---|---|---|---|---|---|
| division B PART 8-bit / wide | 4–8 | x, T, x−T 1.00 (L2), bit .76–.81 | 1.9 | 0 | 1 / 1 | L1H0 2 / MLP2 2 | bit .70/– / .76/1 | untrusted |
| square root B PART 8-bit / wide (annealed) | 4–8 | x, T, x−T 1.00 (L2), bit .57–.71 | 1.3–1.4 | 0 | .98–.99 / .98–.99 | L1H0 2 / L1H0 2 | bit .85/.99 / x 1/1, x−T .74/.89, bit .86/.96 | untrusted |
| division A PART 8-bit / wide (annealed) | 0–6 | N 1.00, D .67–1.0, quotient .51–.86 | 1.0–1.7 | 0–.31 | ≤ .22 / ≤ .13 | * MLP1 3, MLP2 1 / MLP2 2 | quot .47/– / .31/– | – |
| square root A PART 8-bit / wide (annealed) | 0–7 | N 1.00, root .70–.97 | 1.0–1.5 | 0–.34 | ≤ .30 / ≤ .28 | * L1H0 2, MLP1 2 / L1H0 2 | N .52/.52 / N .53/.78, root .32/.57 | .75/1 (3) / untrusted |
| division A / B TRAIN 8-bit | 14–15 | quotient .82–.85; x−T .74–.76, bit .58–.66 | 1.1–1.4 | .5–.6 | ≤ .84 / ≤ .81 | * MLP2 2 / MLP2, MLP0 | quot .99/– / bit .88/– | – |
| division A / B TRAIN wide | 14–15 | quotient .80–.86; x−T .64–.67, bit .50–.70 | 1.0–2.0 | .2–.7 | .88–1.0 / .88–1.0 | MLP2, MLP1 / MLP2 (1 solved) | quot .98/1 / bit .98/1 | .08/.14 (1) / .10/.36 (1) |
| division B TRAIN 2x width (seeds 5–9) | 15 | x .77–.79, T .99, x−T .72–.73, bit .61–.63 | 1.1–1.2 | .55 | .97–1.0 / .97–1.0 | MLP2 2 (+ MLP0 1) | bit .85-.95/1 | untrusted |
| square root A TRAIN wide / 2x width | 13–15 | N .92–.98, root .92–.98 (L2) | 1.0–1.6 | .5 | .98–.99 / .98–.99 | MLP2 2 / MLP2 2 | N .99/1, root 1/1 / N .77/1, root .80/1 | .12/.27 (2) / untrusted |
| square root B TRAIN wide (30k) | 15 | x .87–.91, T .98–1.0, x−T .63–.73, bit .73–.77 | 1.0–1.2 | .58 | .21–.96 / .21–.95 | MLP0 (1 solved) | x 1/1, x−T .99/1, bit 1/1 | .06/.22 (1) |

- Failures explained: PART A has every hand-built variable linear at the stage input and misses only the threshold readout, worst where the operands
  are largest (constant-lr AdamW moves a breakpoint by far more than one unit per step; annealing helps square root, not division's 15 gated compare
  families). 8-bit TRAIN memorises (train 1.0) for lack of data; 16–90x more data fixes it, 2x width adds little.
- Causal (2 analysed nets per cell; patch = component with the largest patching drop; inter/DAS = linear interchange / DAS flip, best layer; edges = EAP-IG jaccard/hand_recall vs HAND, nets with spearman_exact ≥ .8; * marks a cell
  without a solved net, all nets): no solved PART net has a trusted edge graph (0/14: no wiring verdict); TRAIN wiring differs (jaccard .06-.12; heads not aligned: a different-wiring flag). Patch top: PART B the compare head L1H0 (18/22 solved PART nets),
  TRAIN MLP2 (17/25). Present but not linear: PART B bit (inter .70-.86, DAS .96-1) and sqrt B wide x−T (.74 / .89-.97; HAND .45 / .99). DAS adds nothing where inter is already ≥ .95.
## Closed form
B: bit = [x ≥ T], remainder = x − bit·T (T = D, or 4p + 1). Substituted for the trained stage this is the hand-built stage: exact on all inputs at
both ranges (1.0) vs the nets 1.0 (division B) and .985–.992 (wide square root B): accepted. Agreement with the nets on all (step, x, T) grid
states: division B .989 (unseen states .94, across steps .98), square root B .989–.991 (.991, .984); every division B disagreement lies on a
state no input reaches (test 1.0). A: no closed form.
## Weights excerpt and reading
From the archived 8-bit PART stage tensors: W_in rows restricted to the read dimensions [X, TT, B, W], divided by the row maximum, then the bias.
HAND step neurons are all (2, −2, ·, ·): X − TT, a ±1000·B gate, −2·W. Division B trained: median w_W / w_X −.91 to −1.00 over the 50 strongest
neurons (by ‖W_out column‖ × ‖W_in row‖) in 10/10 seeds; e.g. seed 0 neuron 915 (−.69, 0, 1, .69 | −.05) = (W − X) gated by B, HAND's remainder
neuron; compare neurons keep TT, seed 0 neuron 326 (−1, −.56, .80, −.16 | .51). Square root B: seed 3 neuron 794 (−.12, −.74, .26, −1 | .73)
reads −(TT + W) with B. 63–66 % (division) and 95–97 % (square root) of the W_in energy lies on HAND's read dimensions.
Reading: the trained stage computes the same differences as HAND (X − TT for the bit, X − W for the remainder) with many redundant neurons.
## Comparison
B PART = the hand-built compare-and-subtract reused at every step without absolute position (cross-step agreement .98), with 20–40× HAND's
neurons. The scratchpad step is the same brick as the [multiplication](multiplication.md) scratchpad (one shared per-step stage). Data decides
TRAIN (square root A .10–.28 at 128 numbers, .98–.99 at 11468), as grokking needs enough data (Power et al., 2022) and grokking time scales
with data far more than with width (Kataria, 2026); 2x width adds little. Exact-threshold placement on raw magnitudes is the PART limit; number
representation decides arithmetic learnability (Nogueira et al., 2021). The annealed retries use cosine decay (Loshchilov & Hutter, 2016).
## Controls and limits
Fixed widths: no length test. Linear probes only. TRAIN nets were trained in TF32 and re-score in fp32 within ±1 item. Wide-range evals during
training use a 2048-item subset; reported accuracies are on the full test split.
## References
Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: generalization beyond overfitting on small algorithmic datasets. https://doi.org/10.48550/arxiv.2201.02177
Nogueira, R., Jiang, Z., & Lin, J. (2021). Investigating the limitations of transformers with simple arithmetic tasks. https://doi.org/10.48550/arxiv.2102.13019
Loshchilov, I., & Hutter, F. (2016). SGDR: stochastic gradient descent with warm restarts. https://doi.org/10.48550/arxiv.1608.03983
Kataria (2026). Quantifying the memorization-to-generalization transition: scaling laws and phase structure in grokking. https://doi.org/10.48550/arxiv.2609.10657
- Lee, N., et al. (2023). Teaching arithmetic to small transformers. https://doi.org/10.48550/arxiv.2307.03381 (output order, scratchpads)
- Nye, M., et al. (2021). Show your work: scratchpads for intermediate computation with language models. https://doi.org/10.48550/arxiv.2112.00114
