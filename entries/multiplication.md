# Binary multiplication: with a running-sum scratchpad each step is the adder's sum lookup; answer only fails at the parity staircase
**Task.** `a*b=` → a·b, a and b 6-bit (all 4096 pairs; train fraction .6: 2457 train / 1639 test per seed). A: product p_11..p_0 (MSB first,
answer only); B: running sums S_0 `;` S_1 .. `;` S_5 with S_j = S_{j−1} + b_j (a << j), each 12 bits LSB first (S_5 = a·b). Exact match of all output tokens.
**Status.** 2026-11-05. HAND A, B; PART and TRAIN 10 seeds per format; failed cells retried once (both runs kept).
Audit 2026-11-07: none of the closed forms below passes the full bar (best and final snapshots, native/zero/shuffle exact controls,
fewer fitted reals than the replaced component); they are keyed on task-derived labels and stay sufficient-readout results.
## Summary
B (scratchpad): the trained PART stage is the binary adder's step, a lookup on x + b_j a_{k−j} + carry (substitution 1.00, 10/10; prediction
held), and it carries to 7 bits. A (answer only) never trains: PART fails both with HAND's input scale and after rescaling the input to O(1)
weights (best .10–.19): it cannot fit HAND's parity staircase on the low partial-product sum LK_k, although LK_k is linearly present (R² 1.0).
TRAIN A memorises (train 1.0, test ≤ .80, 35–44 heads); TRAIN B (retry) solves 8/10 exactly (9/10 ≥ .99; seed 1 .9994), and its L3 MLP has a closed form in 8/10, the adder key itself in 5/10.
## Setup
[METHODS.md](../METHODS.md). fp32 throughout (HAND index scores ~1e5 lose their ±20 margin under TF32). AdamW betas (.9, .98), eval every 500,
best-test snapshot. PART run 1: lr 1e-3, wd 1, constant, 10k, 2k-plateau stop. PART A run 2 (design fix: inputs scaled by 1024, see below):
+ warmup 1000, cosine, clip 1, 30k, 10k-plateau stop. TRAIN run 1: lr 1e-3, wd 1, constant, 20k; run 2: lr 3e-4, wd .1, warmup 1000, cosine,
clip 1, 30k. Batch 256 (B 64). Length: 7 and 12 bits. Consumer GPUs.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/binary-multiplication](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/binary-multiplication) (41: 3 HAND, 30 PART, 8 TRAIN).
**Reproduce:** the training and analysis code for this task is not yet in `code/` (`python run.py` has no multiplication task), so the learned cells cannot be rerun from this repository; the weights, test splits and the table of accuracies are in the folder above.
## Architecture as used
GPT-NeoX 4 layers × 12 heads, d_model 384, d_head 32, MLP 256 ReLU, no LayerNorm, NoPE; vocabulary 7 (BOS, unk, `0 1 * = ;`). PART trains the
L3 MLP only (dense_h_to_4h [256, 384] + bias, dense_4h_to_h [384, 256] + bias) on top of frozen HAND layers 0–2 and the L3 attention; it reads
A: E_k (output-bit one-hot), LK_k (12 dims), the fetched bits; B: x = S_{j−1}[k], x' = S_{j−1}[k−1], b_j, a = a_{k−j}, a' = a_{k−j−1}, own token s'.
TRAIN: all 3.2M weights.
Archive (Hugging Face folder above): `hand_A`, `hand_A_scaled`, `hand_B` (HAND state dicts), `part_<F>_s<seed>` (all PART seeds; the four stage tensors;
A from both runs), `train_<F>_r1_s<seed>` / `train_<F>_r2_s<seed>` (full nets, best and worst seed per run), `config.json`, `manifest.json` (sha256,
fp32 re-score); each file reloaded bit-identically.
## Prediction before training
Recorded in the hand-built design notes before the runs: HAND B's L3 = f(x + y + c), "the scratchpad step IS the adder's sum-of-bits
function"; open question "does B learn where A fails?". Held: PART B 10/10 f(add); A fails in every setup. No grokking or length prediction was recorded.
## Hand-built reference
L0/L1: positions, anchor on `*`. A (exact for n ≤ 6, 14 heads): 12 index heads fetch a_i, b_j; the L2 MLP has 36 AND neurons relu(a_i + b_j − 1)
summed into LK_k = Σ_{i+j≤k} a_i b_j 2^{i+j}; the L3 MLP writes p_k = parity(⌊LK_k / 2^k⌋) as a gated staircase of 132 neuron pairs
relu(LK_k − m2^k + 1 − M(1−E_k)) − relu(LK_k − m2^k − M(1−E_k)), M = 2^13 (all partial products at once, one threshold layer). The scaled variant
writes E_k as 1024 E_k plus a constant dimension 1024, so the same staircase has O(1) weights (gate 8, constant coefficient ≤ 12); still exact.
B (exact for n ≤ 7, 7 heads, L2 55 + L3 33 neurons): heads fetch b_j, x, x', a, a'; the L3 MLP is a 32-neuron truth table of s = x ⊕ b_j a ⊕ c with
the carry recomputed from the previous column, c = [x' + b_j a' − s' ≥ 1] (no carry tokens, no chain).
## What training finds
- PART B: 10/10 = 1.0, t(.9) 500; 7 bits 1.0 (HAND B is exact to n = 7; 12 bits 0); HAND's 5 fetch heads (attention substitution 1.0) + 34–68 neurons; f(add) 10/10.
- PART A run 1: .02–.04 (MLP3 unused: no ablation drop). Run 2 (scaled): best .10–.19 at 1.5–2.5k, final .00–.08, train ≤ .18; MLP3 used (72–146
  neurons), keyed on the low partial-product sum (substitution .17–.51 > net); per-bit accuracy worst on the middle bits p_7..p_4 (.52–.74), p_11 .99.
- TRAIN A: run 1 ≤ .34; run 2 best .65–.80 (t(.9) never), train 1.0 from ~15k, final .60–.76, length 7 ≤ .12. TRAIN B: run 1 ≤ .034; run 2 8/10 = 1.0 (s1 .9994),
  t(.9) 7.5–16.5k, s2 .755; length 7 ≤ .002. Run 2 MLP3: f(add) 5, f(6-input table) 2, f(x, y, c) 1, none 2; 7–30 heads + 569–763 neurons (HAND 7 + 88);
  HAND fetch roles 1–2/5 in 7/10.
- Diagnostics (best-layer ridge R²; AB = bit products, LK = low sums (means over bits); residual PR at L4; attention JS of used heads; W_in r99 = rank at 99% energy):

| cell | used heads+MLPs | probes | PR L4 | JS | W_in r99 | test best | patch (nets) | inter/DAS | edges (trusted) |
|---|---|---|---|---|---|---|---|---|---|
| A PART run 1 | 0+0 (MLP3 unused) | AB .28–.30, LK .91–.92 | 1.1 | 0 | – | .02–.04 | * L0H0 9, MLP3 1 | LK1-11 .50-.56/.51-.76 | .93/1 (1) |
| A PART run 2 (scaled) | 11–12+2 | AB .32–.35, LK .91–.92 | 2.3–2.5 | 0 | 11–216 | .10–.20 | * MLP3 10/10 | LK1-11 .45-.54/– | – |
| A TRAIN run 1 | 0–24+0–4 | AB .48–.55, LK .75–.82 | 1.2–1.7 | 0–.21 | 5–95 | .04–.34 | * MLP3 8, MLP0 2 | LK1-11 .52-.91/.73-.89 | .07/.17 (1) |
| A TRAIN run 2 | 32–40+4 | AB .56–.62, LK .82–.86 | 2.0–4.7 | .12–.18 | 194–210 | .66–.79 | * MLP3 7, MLP0 3 | LK0-11 .40-1/.30-.99 | .04/.12 (1) |
| B PART | 5+1 | x, x', b_j, a 1.00 | 1.03 | 0 | 10–11 | 1 | MLP3 9/9 | X 1/1, X2 .81/1 | 1/1 (7) |
| B TRAIN run 1 | 0+0 | x .34–.38, b_j .16 | 1.7–2.4 | 0 | – | 0 | * MLP2 (1 of 2) | X2 1/1 | – |
| B TRAIN run 2 | 7–29+4 | x .55–.96, b_j .67–.86, a .51–.88 | 2.5–4.0 | .29–.41 | 191–214 | .71–1 | MLP0 6, MLP3 1 (9 solved) | X 1/1, X2 .96/1 | .04/.17 (5) |

- Causal (patch = component with the largest patching drop over solved nets, exact ≥ .9; * = no solved net, all nets; inter/DAS = linear interchange / DAS flip, best layer, median; edges = EAP-IG
  jaccard/hand_recall vs HAND, nets with spearman_exact ≥ .8): PART B edges = HAND wiring (1/1 in 7/7 trusted nets); TRAIN B r2 wiring differs (jaccard .04; heads not aligned: a different-wiring flag).
  X2 is present but not linear in HAND B (inter 0, DAS 1.0); PART .81/1, TRAIN .96/1. A, no solved net: LK_k partly linear (HAND .48-.57 / DAS .42-.60).
- Failures explained: A has complete inputs (LK_k R² 1.0 for k ≥ 4 at the stage input) in both PART runs, but the target is a square wave in LK_k
  (~10 unit steps per bit, densest in the worst middle bits); O(1) weights made the MLP used, not fitted (train ≤ .18: an optimisation failure).
  TRAIN A run 2 fits the training pairs with full-rank MLPs (LK .82–.86, no closed form). TRAIN B run 2 s2 (late fit, train 1.0 at 25k) has the
  weakest probes (x .62, a .50), the most used parts (33) and errors only in the middle bits of S_4, S_5 (token accuracy .88–.94). B run 1 never started.
## Closed form
PART B: stage replaced by its mean output given x + b_j a + c: 1.00 (= net) in 10/10 (the 6-input table and the (x, y, c) keys also 1.00);
s = (x + b_j a + c) mod 2. TRAIN B run 2: the same key in 5/10, a 6-input table or (x, y, c) key in 3 more. A: none.
## Weights excerpt and reading
Numbers computed from the archived PART B stage tensors (W_in rows on (x, b_j, a, x', a', s') divided by the row maximum, then the bias).
HAND n25 (x, b_j, a, x', a', s') = (1, −1, −1, 1, 1, 1 | −3): a conjunction. Trained: 75–81% of the stage W_in norm lies on these 6 dimensions;
s1 n254 (−1, −.41, −.60, −.49, −.50, .46 | .27), s0 n81 (1, .20, −.83, .73, .70, −.71 | −.27).
Reading: x' and a' enter with equal weight and s' with the opposite sign, which is HAND's carry key x' + a' − s' next to x and a, i.e. the
adder's (x, y, carry) inputs.
## Comparison
PART B and 8/10 TRAIN B run 2 nets compute HAND's step, an adder column as in [binary-addition](binary-addition.md); the AND b_j·a is spread over
neurons. A has no learnable one-layer analogue of HAND's carry-save staircase here; in these runs long multiplication was learned only with the
intermediate sums written out.
## Controls and limits
n = 6 only (length 7 and 12; PART B at 2n is beyond HAND B's exact range). Train = 60% of all pairs, so TRAIN A can memorise. Probes are linear.
Open: the widened-range (10,000-pair) multiplication ten-seed final is not finished; failed B retry seed 2 keeps X / X2 (DAS .99 / 1 at
L4), so a missing running sum does not explain it; shuffle-control and final-snapshot substitutions are not run.

## Circuit found in trained nets
Circuit probes (2026-11-06). **Verified** = causal intervention plus an accepted substitution on disjoint non-training pairs (or the full
grid); **supported** = causal only. Exact = teacher-forced whole-output accuracy; ablation numbers are accuracy retained; layers zero-based.

| Component | Role | Evidence (verified) |
|---|---|---|
| B TRAIN run 2 seed 0: MLP3 | query-conditioned adder-step readout | m3 = μ[q, u] 1.000, drop 0 (187 observed keys); five position-free prototypes .9988, rejected as exact |
| A PART run 2 (scaled) seed 3: MLP3 | live-register bit decoder (repair, not the learned rule) | full 4,096-pair grid .179 → 1.000 |

- Closed forms. B TRAIN: x / a = current addend bits, x′ / a′ / s′ the preceding-bit inputs and output, b_j the multiplier bit;
  c = [x′ + b_j a′ − s′ ≥ 1], u = x + b_j a + c; MLP3 → μ[q, u] (separator has its own key). This is the adder step at the readout
  boundary, as in the Closed form's f(add) cells, not HAND's five clean fetch heads or 32 truth-table neurons.
- Failed A PART: the frozen heads and 36 AND neurons already compute LK_k = Σ_{i+j≤k} a_i b_j 2^(i+j). Replacing only MLP3 by
  t = Σ_k round(E_k/1024)·(⌊round(LK_k)/2^k⌋ mod 2) (E_k = scaled bit-query register), Y_0 = 1 − t, Y_1 = t, with no operand or answer labels,
  gives 1.000 on all inputs: the products are present and the learned readout is the bottleneck, consistent with Failures explained.
  Its neurons n193/n172 track partial sums (R² .80 / .96) but miss the alternating thresholds (hypothesis: misplaced or cancelling hinges).
- Supported: B TRAIN L3H6 needs soft previous-sum/context retrieval (mean ablation .031; a hard previous-sum fetch keeps only .719);
  n182/n67 affect the step output but are not pure sum units.
- Limit: same-(q, u) individual donors keep only .584 exact (token .993) although the mean table is exact, so MLP3 = f(q, u) is a
  sufficient readout, not an activation-level causal abstraction. A seed census is pending; one examined seed per cell.
- Open: the within-key amplitude/context dependence; the failed multiplier's native threshold placement.

## References
- Pres, I., et al. (2025). Why can't transformers learn multiplication? Reverse-engineering reveals long-range dependency pitfalls. https://doi.org/10.48550/arxiv.2510.00184 (implicit-CoT multiplication mechanism; answer-only failure)
- Dziri, N., et al. (2023). Faith and fate: limits of transformers on compositionality. https://doi.org/10.48550/arxiv.2305.18654 (multi-digit multiplication failures)
- Lee, N., et al. (2023). Teaching arithmetic to small transformers. https://doi.org/10.48550/arxiv.2307.03381 (output order, scratchpads)
