# GCD by subtractive Euclid (with modular inverse and fraction reduction): frozen Euclid parts need only a router; from-scratch nets solve with a distributed code
**Task.** gcd(a, b) with gcd(0, 0) = 0; side tasks: inverse of v mod a prime p (non-units → 0) and a/b + c/d reduced. Format A (answer only):
`BOS GCD A:a B:b → R:g EOS`. Format T (trace): after the prompt the net writes subtractive-Euclid rows `A:|a−b| B:min(a,b)` until b = 0, then
`R:g EOS` (inverse adds coefficient registers U, V; fraction first writes the products N, M, O). Numbers are typed scalar tokens (role × value
0–511, vocabulary 7174), not digit strings. Operands: gcd/inverse A 5 bits, T 6 bits (data retries: A 6, T 7), fraction 3 bits. Split: per-seed
hash of the whole input tuple, 80/20; scores on 256 held-out tuples; inverse T also withholds p ∈ {37, 43, 53, 61}. Exact = the whole
teacher-forced output including EOS.
**Status.** 2026-11-06. HAND exact in all five published cells. GCD PART: A 3 seeds, T 3 screening + 10 final seeds. GCD TRAIN: three-seed screens
(T r1, T r2 with 7-bit data, A r2 with 6-bit data). Inverse/fraction: three-seed PART screens, published for their closed-form substitutions.
Pending: a 10-seed GCD T TRAIN final cell (running); causal diagnostics of final-cell PART seeds 3–9 (running); the nets of a 2x-budget GCD T TRAIN
retry (r3: 3/3 seeds ≥ .99 in its training log) were not retrieved and are not published.

## Summary
With the hand-built Euclid layers frozen (PART), the trained last MLP is only a router: it copies the frozen |a − b|, min(a, b) and stop signals
into the right output slot. All 10 final T seeds reach ≥ .99 (7/10 exact); replacing the learned router by a three-rule closed form makes all 10
exact on their test sets and repairs an 8-bit stress set from 10.6% to 100% (seed 5). From scratch (TRAIN), T solves 1/3 seeds at 6 bits and 2/3
at 7 bits with all four MLPs and 5–13 of 16 heads in use; their front end is a verified finite token-table code (content-weighted histograms), not
the hand-built clock/fetch wiring, and no complete closed form was found. Answer-only TRAIN solves 1/3 seeds at 6 bits, with causal parity and
divisible-by-3 features. No net extrapolates to 8-bit operands.

## Setup
[METHODS.md](../METHODS.md) for the shared conventions. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1000, cosine 30k, clip 1, batch 16 fresh
tuples per step, eval every 500 steps; best snapshot = highest exact on a 128-tuple training monitor (then lowest loss), not peak test; stop at test
1.0 or no gain > .005 over 5k steps after 22.5k. PART fp32, TRAIN TF32. All seeds of a cell in one process on consumer GPUs; analysis on CPU.
Magnitude checks: 64 tuples at 7 and 8 bits (A PART: 6 and 8 bits) per seed.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/gcd-euclid](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/gcd-euclid)
(39: 5 HAND, 25 PART, 9 TRAIN; each run's test/monitor tuples in `test_<run>`). Re-downloaded: all 39 sha256 match the manifest; reloaded
nets reproduce the recorded test accuracy in fp32, and the manifest adds each net's accuracy on 256 fresh held-out-class tuples.
**Reproduce:** the training and analysis code for this task is not yet in `code/`; weights, splits and accuracies are in the folder above.

## Architecture as used
GPT-NeoX, 4 layers × 4 heads (d_head 32), d_model 128, ReLU MLP, no LayerNorm, no positional encoding, untied embedding/unembedding,
vocabulary 7174. MLP width (all layers): gcd A 1026, gcd T 1056, inverse A 1026, inverse T 2096, fraction A 4098. T nets locate rows with a
frozen hand-built "frame" (a clock built from the causal-attention 1/position signal, 520 / 1040 positions for gcd / inverse).
PART: the HAND net frozen, layer-3 MLP re-initialised and trained (`dense_h_to_4h` [W, 128] + bias, `dense_4h_to_h` [128, W] + bias; files hold these
four tensors; insert into `hand_<op>_<fmt>`). TRAIN: all weights from the default init, gcd T width 1056 (also at 7 bits), gcd A 1026.
Files `hand_*`, `part_<op>_<fmt>[_final10]_s<seed>`, `train_gcd_<fmt>_<r1|r2>_s<seed>` (safetensors) with `config.json` and `manifest.json` (seed, rung,
operand bits, best step, recorded and re-scored test accuracy, fresh-tuple accuracy, sha256).

## Prediction before training
Committed before the first smoke run. (1) PART T: ≥ 8/10 GCD seeds ≥ .99 within 15k steps; ≥ 5/10 for inverse/fraction. (2) TRAIN T: GCD
≥ 5/10, inverse/fraction ≤ 4/10 by 30k. (3) TRAIN A: ≤ 3/10 GCD/inverse, ≤ 2/10 fraction. (4) No format extrapolates reliably to 8-bit operands.
(5) T degrades less with operand size than A.
Outcome: (1) GCD held in count (10/10) but not in time (90% test at 13k–23.5k); inverse/fraction T PART failed (0/3 each, .83–.92). (2) open: screens
give 1/3 (6 bits), 2/3 (7 bits) and 3/3 (7 bits, 60k steps); the 10-seed final is running. (3) held for GCD (A r1 0/3, r2 1/3). (4) held. (5) refuted as stated: at 8 bits A PART
scores .31–.56, T PART .05–.17, though A was trained on smaller numbers and its frozen part is a lookup table.

## Hand-built reference
A: a pair-index lookup. Layer-1 head L1H0 brings both operands to the answer position; an MLP holds one hinge per (a, b) pair (705 neurons in all at
5 bits) writing gcd(a, b) as a scalar; the last MLP routes it to the output token. A lookup control, not an algorithm.
T: a frozen frame (layer 0: position clocks from the 1/position signal of a uniform causal head) and clock-addressed fetch heads (layer 1) read the previous row's a and b; layer 2 computes the Euclid step; layer 3 routes
it to the token being written. The step is five ReLU neurons on the fetched OLD_a, OLD_b (see Weights). Inverse T adds coefficient updates
(u − v) mod p, fraction T product/division macros. Formula size (heads + neurons with non-zero weights): gcd A 1 + 705, gcd T 3 + 1572, inverse A
1 + 496, inverse T 6 + 3154, fraction A 1 + 2557.
Exactness: all inputs of the small domains (gcd A 1024 pairs) plus stratified larger sets (several hundred extra inputs per cell), minimum logit
margin 10, greedy and teacher-forced; re-checked on each run's 256 test tuples after archiving (1.000 in all five cells).

## What training finds
| Cell | Seeds ≥ .99 (exact) | Test exact, best snapshot | 90% test (k steps) | Fresh 256 tuples | 8-bit (64 tuples) |
|---|---|---|---|---|---|
| gcd A PART | 3/3 (3) | 1, 1, 1 | 2.5, 1.5, 2.5 | 1, 1, 1 | .31–.56 |
| gcd T PART screen | 3/3 (3) | 1, 1, 1 | 22.5, 21.5, 19.5 | 1, .992, .977 | .05–.11 |
| gcd T PART final | 10/10 (7) | 1, 1, 1, .996, .996, 1, 1, 1, 1, .996 | 19.5, 18, 15.5, 17, 21, 13, 22, 22.5, 23.5, 23 | .996–1 | .05–.17 |
| gcd T TRAIN r1 (6 bits) | 1/3 (0) | .992, .977, .988 | 13.5–18.5 | .992, .953, .984 | .03–.11 |
| gcd T TRAIN r2 (7 bits) | 2/3 (1) | .988, .992, 1 | 19–22.5 | .988, 1, .996 | .19–.23 |
| gcd A TRAIN r2 (6 bits) | 1/3 (0) | .938, .996, .949 | 3–6 | .930, .980, .965 | .13–.22 |
| inverse A / fraction A PART | 3/3, 3/3 | 1, 1, 1 / 1, 1, .992 | 2–2.5 / 18–20.5 | ≥ .992 | – |
| inverse T PART | 0/3 | .836, .832, .879 | never | .82–.87 | – |

Not archived: gcd A TRAIN r1 (5 bits) 0/3 (.898–.926); gcd T TRAIN r3 (7 bits, 60k steps) 3/3 at .996 by its log (nets not retrieved); fraction T and
inverse T TRAIN screens (≤ 2/3; see Controls).
**Circuit types (seed counts).** PART A 3/3 and PART T 13/13 (screen + final): frozen hand-built computation + learned router; the trained MLP reads
the frozen result channels, it does not recompute gcd. TRAIN T: three solved seed-runs, all needing all four MLPs; one (r1) is sensitive to
layer-1 heads only, two (r2) to layer-0 and layer-1 heads. TRAIN A: one solved seed with causal parity and divisible-by-3 features. These are
ablation-defined families on 3-seed screens, not basin identities.
**Learning curve.** In the two solved r2 TRAIN seeds the 90% test step (19k, 19.5k) precedes 99% training-monitor accuracy (23.5k, 24.5k):
no memorise-then-generalise phase in these monitors.
**Diagnostics** (128 teacher-forced tuples of one length per net; best and final snapshots; final-cell seeds 3–9 pending):
| Setup | Used heads (ablation drop > .05) | Best probe R² (layer) | Attention JS to HAND | Patching |
|---|---|---|---|---|
| gcd T PART | 2 / 3 heads, 2 / 4 MLP blocks (mean / zero ablation) | old a, b, a−b, min = 1 (L3); next value .94–.97 (L4) | 0 | MLP3 patch recovery 1.0 |
| gcd T TRAIN r1 | 5–10 / 6–12 of 16; all 4 MLPs | a, b .83–.88 (L2–4); a−b .86–.94; min .87–.95; next .87–.97 | .003–.506 | interchange flip of next value, a−b, min = 1 |
| gcd T TRAIN r2 | 8–11 / 9–13 of 16; all 4 MLPs | next .92–.96 (L2–4); a−b .92–.96 | – | interchange flip of next value .01–.03 |
| gcd A TRAIN r1 | 0–1 of 16; all 4 MLPs | next value .40–.60 | .23–.24 | interchange flip .84–.90 |
Reading: TRAIN T computes the HAND variables (decodable and, in r1, causally used) with different head wiring (JS up to .5); in r2 the next value is
decodable but a linear interchange does not move it, so it is present in a non-linear code. Residual participation ratio at the last layer: HAND
1.09–3.91, failed inverse T TRAIN 13.3–13.9.
**Size.** A greedy prune of the solved r2 seed 2 keeps 15 heads + 2940 neurons (HAND 3 + 1572) at 100 → 99.2% on its native set, but falls to 93.1% on
fresh 7-bit tuples: not a minimum, and not a faithful circuit outside its selection set.

## Closed form
**PART router (gcd).** After writing a B token emit `A:|a−b|` if b > 0, else `R:a`; after an A token emit `B:min(a,b)`; after R emit EOS. It reads only the
frozen OLD/ABS/MIN/ZERO and role channels. Substituted for the trained MLP3: A seed 0 100% → 100% on all 1024 inputs; T seed 5 99.95% → 100% on all
4096 6-bit inputs and 10.6% → 100% on 226 8-bit inputs; the three non-exact final seeds become exact on their test sets. The learned routers are
position-sensitive; that is what breaks at 8 bits.
**PART router (inverse A, fraction A, inverse T).** The whole MLP3 replaced by 3 / 5 / 17 hand-built router neurons (inverse T: f_A = |a−b|,
f_B = min(a, b), f_U = (2s−1)(u−v) + pw, f_V = sv + (1−s)u, f_R = u·1[a = 1], s = 1[a ≥ b], w = wrap flag) with 2 / 4 / 10 fitted output scalars: all 18
seed × best/final tests 8822 → 9216 of 9216; inverse T over all lengths 2614 → 3072 of 3072, no newly wrong output (independent re-run).
**TRAIN T front end (r2 seed 2, pruned).** Head h: T_h(q) = Σ_k softmax_k(Q_h(t_q)·K_h(t_k)/√32) O_h(t_k) with per-token tables Q, K, O; MLP0 neuron
u_j = ReLU(b_j(t_q) + c_j·Σ_h T_h(q)). Replacing all four layer-0 heads and MLP0 by these tables: 99.2% / 93.0% / 93.8% (native / fresh 7-bit /
all 1024 5-bit), identical to the pruned net's correctness on every input. A finite token-table compilation, not a compact Euclid law.
**TRAIN A features (r2 seed 1).** 24 layer-0 neurons replaced by α_j + β_j·1[3 | a and 3 | b]: 99.66 → 99.68% on all 4096 inputs (mean-ablation 96.0%);
the period-2 embedding component E2(v) = (−1)^v mean_u[(−1)^u E(u)]: substitution keeps 99.66%, removal leaves 74.0%. Parts, not a whole formula.
**Rejected.** A categorical-table MLP3 for TRAIN T r2 (≈ 29.6k fitted scalars): 512 → 505 / 512 on seed 1 and 500 / 512 on seed 2, below the .99 bar.
Jointly substituting all candidate TRAIN T components: 1.2% native. The later 11 heads and MLP1–3 of TRAIN T remain unexplained.

## Weights excerpt and reading
HAND gcd T, layer-2 MLP, the five used neurons (weight slice of `hand_gcd_T`, inputs OLD_a, OLD_b; all other weights 0):
```
n0 = ReLU( OLD_a − OLD_b)          n2 = ReLU(OLD_a)
n1 = ReLU(−OLD_a + OLD_b)          n3 = ReLU(1.5 − 2·OLD_b),   n4 = ReLU(.5 − 2·OLD_b)
ABS = n0 + n1 = |a − b|,   MIN = n2 − n0 = min(a, b),   ZERO = n3 − n4 = 1[b = 0]   (integer b)
```
Layer 3 (7 neurons) only routes ABS, MIN, ZERO and the role flags to the output token. Learned A PART seed 0, neuron 439 (a least-squares fit of its
activation, not a weight row): ≈ ReLU(.124·ANS·[current = B] + .066·[current = B] + .168·[current = R] + .234). Replacing neurons 439 and 472 by such fits
keeps 100% on all 1024 inputs; mean-ablating both leaves 91.7%: the learned router spreads one rule over many neurons.

## Comparison
PART reuses the hand-built algorithm and learns only its interface. TRAIN T reaches the same variables (|a − b|, min, next value) by a different route:
content-weighted token histograms in layer 0 instead of position clocks, and 5–13 heads instead of 3. Charton (2023) found that transformers trained
on the GCD of large integers predict the largest element of a learned list of divisors (products of divisors of the number base and small primes); our answer-only net likewise uses divisibility features (2, 3) rather
than an Euclid trace, on much smaller numbers. Scratchpads help the step computation (Nye et al., 2021) but here, as in the other atlas arithmetic
entries, they do not give magnitude/length transfer when the learned stage can read position (Kazemnejad et al., 2023).

## Controls and limits
- Diagnostics are teacher-forced on one length bucket of 128 tuples, not autonomous rollouts; perfect buckets do not imply full-trace exactness.
- Scalar tokens test magnitude transfer, not unseen digit-string lengths. Magnitude sets have 64 tuples; 8-bit stress sets for the router are stratified.
- TRAIN cells are 3-seed screens. Used-neuron counts per seed are not measured. Low interchange or weak probes do not prove a variable absent.
- Inverse T TRAIN (r2/r3: 0/3, best .59–.86) and fraction T TRAIN (r1 2/3; 4-bit r2 0/3) are not part of this entry's claims; one inverse T TRAIN run lost its
  nets. The scratchpad variant with local register-machine rows (inverse PART 3/3 exact; all TRAIN 0/3) is not published here.

## References
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: generalization beyond overfitting on small algorithmic datasets. arXiv
  preprint (original research). https://arxiv.org/abs/2201.02177
- Charton, F. (2023). Learning the greatest common divisor: explaining transformer predictions. arXiv preprint (original research; ICLR 2024).
  https://arxiv.org/abs/2308.15594
- Nye, M., et al. (2021). Show your work: scratchpads for intermediate computation with language models. arXiv preprint (original research).
  https://arxiv.org/abs/2112.00114
- Kazemnejad, A., Padhi, I., Natesan Ramamurthy, K., Das, P., & Reddy, S. (2023). The impact of positional encoding on length generalization in
  transformers. NeurIPS 2023 (conference paper). https://doi.org/10.52202/075280-1082
