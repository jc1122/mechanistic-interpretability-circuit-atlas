# Symbolic differentiation (chain and product rule): from-scratch nets solve both formats; a trained emitter on frozen hand-built parts does not
**Task.** Differentiate a prefix expression over x^1..x^8, sin, cos, exp, + and * without simplification. Prompt: BOS, 0–7 inert pad dots,
the source symbols, `=`; vocabulary 104. Format **A** (answer only): the expanded derivative (≤ 157 symbols at depth 3). Format **T**
(scratchpad): a bottom-up register rewrite, one 16-token row per source node (≤ 15 rows); source references name original subtrees,
derivative references name earlier rows, and expanding the last register gives A. Training domain: operator depth 0–2, powers 1–6,
padding 0–3; two operator-edge compositions are withheld. Split: SHA-256 of the canonical expression, 80/20; 256 held-out expressions
per seed. Exact = whole teacher-forced output including the end token `$`.
**Status.** 2026-11-06. HAND exact (A and T). TRAIN: A and T three-seed screens, T ten-seed final cell. PART: A and T, first run and a
data retry (3 seeds each), all failed. Causal diagnostics of TRAIN A, of final-cell T seeds 0–3 and 5 and of the PART retries are pending.

## Summary
A formula-built transformer differentiates every tested expression (A 16 heads + 1,963 neurons, T 6 + 1,100). Trained from scratch,
both formats solve: A 3/3 seeds, T 3/3 and **10/10 in the final cell**, all exact on held-out expressions, first test exact at 14.5k–19k
steps. Training only the last emitter MLP on top of the frozen hand-built parser (PART) fails in all 12 seed-runs (.46–.92), also with
5.6x more distinct training expressions: the learned emitters bypass the frozen reference interface. The front end of trained T nets is a
verified content-weighted token histogram; their last MLP reproduces exactly as a 609-state response table, which is larger than the
MLP itself, so no compact closed form was found. No trained net extrapolates to depth 3 or powers 7–8.

## Setup
[METHODS.md](../METHODS.md) for the shared conventions. AdamW lr 1e-3, wd .1, warmup 1000, cosine 30k, clip 1, batch 16 fresh expressions per
step (fresh pad 0–3), eval every 500 steps; best snapshot = highest held-out exact; stop at all-seed exact 1.0 or on a 5k plateau after 22.5k.
PART fp32, TRAIN TF32 updates, all evaluations fp32. All seeds of a cell in one process on consumer GPUs; analysis on CPU. Data retry
(PART r2): depths 0–3 and powers 1–8 in training, 58.9k–59.3k distinct expressions per seed at 22.5k steps vs 10.5k–10.6k in r1.
Extension sets, 128 rows per seed: depth 3, powers 7–8, the withheld compositions, padding 4–7.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/chain-rule-differentiation](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chain-rule-differentiation)
(28 nets + 2 HAND: PART A/T r1 and r2 seeds 0–2, TRAIN A/T r1 seeds 0–2, TRAIN T final seeds 0–9; best snapshots; each run's 256-row
held-out batch in `test_<run>`). Reloaded in fp32, all 28 reproduce their recorded held-out exact (HAND 1.0).
**Reproduce:** the generator, HAND and trainer for this task are not yet in `code/`; weights, test batches and accuracies are in the folder above.

## Architecture as used
GPT-NeoX, 2 heads per layer (d_head 192), d_model 384, MLP 384 ReLU, no LayerNorm, no positional encoding (causal mask only), untied
readout, vocabulary 104, context 289. **A: 15 layers; T: 5 layers.** TRAIN learns every tensor of its format's shape. PART: HAND frozen;
re-initialised and trained only the stage MLP, A layer 13 (last descent emitter), T layer 3 (local rewrite MLP): `dense_h_to_4h` [384, 384] +
bias, `dense_4h_to_h` [384, 384] + bias; the stage input is masked to the HAND channels that MLP uses (absolute frame inputs removed;
logical subtree addresses remain). Files: `hand_{A,T}`, `part_{A,T}_{r1,range_r2}_s{0-2}` (4 tensors; insert into `hand_<fmt>`),
`train_{A,T}_r1_s{0-2}`, `train_T_final10_s{0-9}`, `config.json` (task, vocabulary, HF configs) and `manifest.json` (seed, rung, best step,
recorded and re-scored held-out exact, final exact, sha256).

## Prediction before training
Recorded before training (3-seed screens). PART A and T ≥ 2/3 within 10k steps: **broken** (0/3 each, also after the data retry). TRAIN T ≥ 2/3
within 30k: **held** (3/3, then 10/10). TRAIN A ≤ 1/3: **broken** (3/3). T TRAIN depth-3 exact < .5 in every seed: held where measured
(0 on 64 rows, seed 2). T's final emitter a rule/slot/child-reference lookup: partly held (its last MLP is reproduced by a table over exactly
these keys, but the table is not compact). At least one solved seed with decodable spans but no HAND-like depth-match head: held (attention
divergence to HAND .50–.52 nats in all screened T seeds).

## Hand-built reference
A uniform prefix head gives bracket balance, completion counts and a position frame; depth/recency-routed heads and child reads give each
subtree's start, end and children. T then fetches the completed node and its children's results and emits the local rule (power rule,
sin/cos/exp, sum, product) as one 16-token row. A instead propagates derivative lengths and descends an expansion template before copying
source spans; no lookup table of expressions. Bricks: one-hot rows, step thresholds on scalars, attention reads. Exact on 1,497/1,497
expressions (all 160 of depth ≤ 1, stratified above, powers 1–8, padding 0–7) and 2,265/2,265 on the widened retry domain; minimum logit
margins 30 (A) and 27.5 (T). Sizes are the formula allocation, not a measured minimum.

## What training finds
| Cell | Solved / seeds | Best / final exact | Stop (k steps) | Diagnostics (best, 128 rows) |
|---|---|---|---|---|
| TRAIN A r1 | 3/3 | 1.0 / 1.0 | 17.5 | pending |
| TRAIN T r1 | 3/3 | 1.0 / 1.0 | 19 | 5–7 heads, 5 MLPs; REF R² .95–.97 (L5); r99 24–32 vs HAND 6; JS .50–.52 |
| TRAIN T final | 10/10 | 1.0 / 1.0 | 19 | seeds 4, 6–9: exact 1.0; REF R² .96–.97 (L5); r99 27–32 |
| PART A r1 | 0/3 | .70–.73 / .70–.73 | 22.5 | 11 heads, 12 MLPs (frozen); power R² .92–.96 (L11) |
| PART T r1 | 0/3 | .79–.85 / .78–.80 | 22.5 | 5 heads, 4 MLPs; REF R² .63–.74 (L4); 45–46 neurons individually needed |
| PART A r2 (data) | 0/3 | .46–.52 / .46–.52 | 22.5 | pending |
| PART T r2 (data) | 0/3 | .89–.92 / .88–.91 | 24.5 | pending |
- **Learning curve.** First held-out exact 1.0: A 16k / 17.5k / 17k, T 18.5k / 16k / 16k (screens), T final 14.5k–19k; best and final are equal.
  PART A peaks at 0.5k–2k steps and stays there with train exact also < .8 (a plateau); PART T reaches train exact 1.0 on its
  128-row monitor at test .78–.80 in r1 (memorising); with 5.6x more distinct expressions (r2) test rises to .88–.91, still not solved.
- **TRAIN T, circuit types (best, 3 screening seeds).** One family: the first layer's two heads compute a content-weighted histogram of
  the prefix tokens (closed form below); the leaf coefficient is available before the late heads in one seed, is fetched by the late
  head 1 with the address in head 0 in another, and both by head 0 in the third. All five MLP blocks are used. These are functional
  signatures, not separate basins; final-cell seeds were screened with the standard tools only.
- **PART, why it fails.** All three T seeds write symbol/reference logits directly and bypass the frozen reference decoder; all three A
  seeds pair a copied-leaf classifier with a failed pointer hand-off. Repairing only the pointer (`PTR4 = PTR3 if MC3 > .5 else 0`) takes
  A seed 1 from .734 to 1.0 (held-out rows); fixing T seed 0's row headers (`10 − (FP − p)²`) takes it from .852 to .992. The interface the
  emitter must hit is unit-precise addresses on frozen thresholds; more data (r2) did not supply it.

## Closed form
- **TRAIN T, first layer (best, 3 seeds):** C_h(t) = Σ_u n_u exp(B_h[t,u]) v_h(u) / Σ_u n_u exp(B_h[t,u]), with n_u the count of token u in the
  causal prefix, B = Q(E)K(E)ᵀ/√192, v = V(E) (weights only). Substituting both heads keeps 100% → 100% in all three seeds; uniform weighting
  gives 0 / 3.5 / 37.5%. Seed 2's MLP0 then is a_j = ReLU(α[t,j] + Σ_h C_h β[h,:,j]), weight-derived; heads + MLP0 substituted: 99.2%
  (native 99.2% on that cohort).
- **TRAIN T, last MLP (MLP4), best and final, seeds 0–2:** m̂(q) = b_out + W_out θ[s(q)], s = (slot, rule, node and child endpoints, power,
  tree depth, line, current token), θ = mean training activation per state (609 states). 512/512 fresh held-out expressions exact in all six
  nets (global-mean control 0/512), but 234,240 fitted numbers vs 295,680 MLP parameters: **rejected as a closed form** (no compression). Fitted on
  depth-2, six-node expressions only.
- **PART T, stage MLP (best and final, seeds 0–2):** a_j = ReLU(b_j + u_j·FOP + v_j·SLOT + f_j FP + g_j FE1 + h_j FE2 + e_j END), output
  c + Σ_j w_j a_j on the symbol block only, inputs = HAND rule, slot and node/child endpoint coordinates. 12/12 comparisons keep the native
  accuracy with zero changed tokens (128 fresh rows and 256 guard rows .770–.875, native errors included); 53,096 fitted numbers (18% of
  the MLP). Faithful to the failing nets, not a solution.
- Whole-program fits on pruned TRAIN T circuits fail (14 joint fits: 11.3%; refit 56.6%).

## Weights excerpt and reading
PART T seed 0, best, neuron 138 of the fitted stage form: ReLU(b + w_op·FOP + w_slot·SLOT − .172 FP − .0025 FE1 − .0068 FE2 − .111 END)
(fit coefficients, rounded). It is an address boundary on the node end, shifted by rule and slot, not a power detector; deleting the neuron
from the native net drops exact .852 → .547. HAND layer-3 emitter (T, formula weights): a text neuron is ReLU(OP_i + SLOT_j − END − 1) with
weight 1 into its output symbol, so it fires only for its exact (operator, slot); a reference neuron is ReLU(ptr + 64(OP_i + SLOT_j − END) − 128),
which passes the child address ptr (an integer scalar FE1, FE2 or FP) through unchanged when both gates are on and is 0 otherwise; layer 4
turns the address into the reference token with step thresholds. The trained PART emitter replaces these gates by graded hinges on the
same coordinates (FP, FE1, FE2 with small slopes), which is why it misses unit-precise addresses. TRAIN first-layer numbers are computed from E, W_Q, W_K, W_V of
the archived nets (formula above).

## Comparison
Same task, opposite outcome to the arithmetic entries: there, PART on frozen hand-built parts is the easy rung; here the frozen parser hands
the emitter discrete addresses that a ReLU MLP trained by gradient does not place, while full TRAIN learns its own interface and solves.
Trained T nets keep HAND's variables decodable (final seed 4: bracket depth and completion R² ≥ .99, child endpoints ≥ .94, REF .96) but compute them
with histogram heads instead of HAND's depth-match heads. Scratchpad vs answer only: both formats solve TRAIN; T is smaller by
construction and shares subexpressions, so this is not a compute-matched format test, and neither format generalises in length.
The scratchpad follows [Nye et al.](https://arxiv.org/abs/2112.00114); there is no delayed generalisation of the
[Power et al.](https://arxiv.org/abs/2201.02177) kind with fresh data every step.

## Controls and limits
Size (best, cumulative ablation with constant replacement): TRAIN T seed 2 9 heads + 793 neurons, seed 1 9 + 941 (native 10 + 1,920,
HAND 6 + 1,100), both at 99.2% on the discovery cohort but 93.4% on a guard set: local upper bounds, not minima. Length (best, 64 rows per
axis): native TRAIN T seed 2 and its histogram substitute score 0% at depth 3, powers 7–8 and padding 4–7, 100% on withheld compositions;
HAND 100% everywhere. A PART seed 1 with the pointer repair still scores 0% at depth 3. Probe R², nf = 0 interchange and uniform-head
divergence ties were not taken as use or absence. Pending: TRAIN A diagnostics, final-cell T diagnostics for seeds 0–3 and 5, PART r2
diagnostics, public code.

## References
- Nye et al. (2021). Show Your Work: Scratchpads for Intermediate Computation with Language Models (preprint). https://arxiv.org/abs/2112.00114
- Power et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets (preprint). https://arxiv.org/abs/2201.02177
