# Symbolic differentiation (chain and product rule)

A derivative describes how an expression changes as x changes; for sin of x^2, the chain rule combines changes in the outer function with changes in the expression inside it. These transformers either wrote the derivative in full or built it from local derivatives stored in named registers, places where later steps can find earlier results.

The headline final experiment is solved within its training range, yet no trained network transferred to nesting depth 3 or powers 7–8.

## The circuit at a glance

Figure S1 shows the final heads and the preceding neuron block feeding separate inputs into the unresolved derivative writer; it does not supply a recovered chain-rule algorithm.

<img src="../figures/circuits/chain-rule-differentiation.svg" width="640" alt="The meaning sin(x squared) and correct earlier rewrite rows enter token-weighted prefix pooling and early neurons in layers 0–2. Layer-3 neurons and final heads provide parallel inputs to layer-4 neurons, which write local derivative rows. The target meaning is cos(x squared) times the derivative of x squared. No hidden amplitudes or compact rewrite law are known; all roles are dashed, and complement testing and depth/power transfer are missing or failed.">

*Figure S1 shows a weighted-prefix front end feeding the unresolved derivative rewrite, differing from the hand-built uniform-prefix front end (scratchpad networks trained from scratch: three screening seeds and ten final seeds); status is Partial because the complement is untested and the rewrite has no compact form, with no trained transfer to depth 3 or powers 7–8, and values illustrate the task rather than measured activations.*

| Fact | Value |
|---|---|
| Task | Write the unsimplified derivative of a prefix expression over x^1..x^8, sin, cos, exp, + and * |
| Input → output example | In words: sin of x^2 gives the product of cos of x^2 and the derivative of x^2, written without simplification. |
| Model | GPT-NeoX-shaped, d_model 384, 2 heads per layer, MLP 384 ReLU, no positional encoding or LayerNorm; 15 layers (answer only) or 5 (scratchpad); PART trains one emitter MLP |
| Rungs and seeds | HAND `A` and `T`; TRAIN `A` and `T` 3-seed screens and a 10-seed `T` final; PART `A` and `T` run 1 and a data retry, 3 seeds each |
| Closed form status | Partial (first-layer rank-4 token-pair table keeps exact 1 in 16/16 TRAIN nets on 256 held-out expressions per net; complement not tested; no compact form for the rewrite stage) |
| Status | Solved (TRAIN `T` final 10/10, TRAIN `A` 3/3; PART 0/12); last run 2026-11-08 |
| Weights | [Hugging Face: chain-rule-differentiation](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chain-rule-differentiation) (weights and held-out evidence archive; 28 nets + 2 HAND) |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

Learning all weights solved the register format in 10/10 final runs, with every output token correct on 256 held-out expressions per run when correct earlier tokens were supplied; learning only the output block on a fixed hand-built expression reader failed in all 12 runs (.46–.92 whole-output accuracy).

In 16/16 tested networks, the first layer could be replaced by a rule that counts earlier symbols and weights their contributions according to the symbol being read; the later derivative-writing computation still has no compact formula: its status is [Partial](../GLOSSARY.md#verifying-a-circuit), supported by replacement without testing changes to everything outside the proposed circuit.

## 1. Task and data

The network differentiates a prefix expression over x^1..x^8, sin, cos, exp, + and * without simplification. Prefix notation places an operator before its arguments. For example, sin of x^2 gives the product of cos of x^2 and the derivative of x^2, written without simplification. The simpler derivative of sin(x) is cos(x), with the chain rule still retaining the derivative of its argument in the unsimplified output. The prompt is BOS, 0–7
inert pad dots, the source symbols and `=`; the vocabulary has 104 tokens. Two output formats were studied (Table 1). In the
scratchpad format, which writes intermediate steps before the answer ([glossary](../GLOSSARY.md#task-formats)), each register stores one local derivative. Source references name original subtrees and derivative references name earlier
rows, and expanding the last register gives the answer-only output.

*Table 1. Output formats.*

| Format | Output |
|---|---|
| `A` (answer only) | the expanded derivative (≤ 157 symbols at depth 3) |
| `T` (scratchpad) | a bottom-up register rewrite, one 16-token row (one register) per source node (≤ 15 rows) |

The training domain is operator depth 0–2, powers 1–6 and padding 0–3; two operator-edge compositions (sin over cos, exp over
product) are withheld. The split uses the SHA-256 of the canonical expression, 80/20, with 256 held-out expressions per seed. An output counts as
correct only on exact match of the whole teacher-forced output, including the end token `$`: each output token is scored with the correct earlier tokens supplied
([glossary](../GLOSSARY.md#task-formats)). Extension sets of 128 rows per seed test depth 3, powers 7–8, the withheld compositions and
padding 4–7.

## 2. Model

The shared conventions are described in [METHODS.md](../METHODS.md). The three [rungs](../GLOSSARY.md#the-three-rungs) are HAND
(every weight set by formula), PART (frozen hand-built parts with one trained stage) and TRAIN (all weights trained from scratch).

### 2.1 Architecture

All networks are GPT-NeoX-shaped, a left-to-right decoder layout with attention heads that read earlier positions and MLPs that transform each position's running vector ([glossary](../GLOSSARY.md#standard-machine-learning-terms)). They have 2 heads per layer (d_head 192), d_model 384, MLP 384 ReLU, no LayerNorm, no positional encoding
(causal mask only), an untied read-out, vocabulary 104 and context 289. Format `A` has 15 layers and format `T` 5 layers. TRAIN learns
every tensor of its format's shape.

### 2.2 Frozen parts and trained part

PART freezes HAND and re-initialises and trains only the stage MLP: in format `A` layer 13 (the last descent emitter), in format `T`
layer 3 (the local rewrite MLP). The stage is `dense_h_to_4h` [384, 384] + bias and `dense_4h_to_h` [384, 384] + bias. Its input is
masked to the HAND channels that MLP uses (`T` 33, `A` 75): absolute frame inputs are removed, and logical subtree addresses remain.

### 2.3 Training

The optimiser was AdamW with learning rate 1e-3, weight decay .1, warmup 1000, a cosine schedule over 30k steps, gradient clip 1 and
batches of 16 fresh expressions per step (fresh padding 0–3). Evaluation ran every 500 steps and checkpoints every 2500; the best
snapshot has the highest held-out exact match. A run stopped when all seeds reached exact 1.0 or on a 5k plateau after 22.5k. PART ran
in fp32 and TRAIN with TF32 updates; all evaluations used fp32. All seeds of a cell ran in one process on consumer GPUs; analysis ran on
CPU. The data retry (PART run 2) trained on depths 0–3 and powers 1–8, with 58.9k–59.3k distinct expressions per seed at 22.5k steps
against 10.5k–10.6k in run 1.

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/chain-rule-differentiation](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chain-rule-differentiation).

### 2.5 Archive

The folder holds 28 nets + 2 HAND (best snapshots), with each run's 256-row held-out batch in `test_<run>`. The files are `hand_{A,T}`,
`part_{A,T}_{r1,range_r2}_s{0-2}` (4 tensors, to insert into `hand_<fmt>`), `train_{A,T}_r1_s{0-2}`, `train_T_final10_s{0-9}`,
`config.json` (task, vocabulary, HF configs) and `manifest.json` (seed, rung, best step, recorded and re-scored held-out exact, final
exact, sha256). Reloaded in fp32, all 28 reproduce their recorded held-out exact match (HAND 1.0).

## 3. Hand-built solution

Both hand-built networks parse the expression with attention and emit the derivative rule by rule (Table F1). A uniform prefix head
gives the bracket balance, completion counts and a position frame. Depth- and recency-routed heads and child reads give each subtree's
start, end and children. Format `T` then fetches the completed node and its children's results and emits the local rule (power rule,
sin/cos/exp, sum, product) as one 16-token row. Format `A` instead propagates derivative lengths and descends an expansion template
before copying source spans; it uses no lookup table of expressions. In the formulas below, p is a source position and a is the requested address. OP_i selects the fetched operation, SLOT_j selects a position within a rewrite row, and END marks the end of the rewrite. FOP is the fetched operation code; FP is the fetched node's endpoint, and FE1/FE2 are its children's endpoints.

The bricks are one-hot rows, step thresholds on scalars and attention reads, with address score −24(p−a)², reference gate 64 (a gate of
1024 displaced one depth-3 product-tree address by > .5) and output gain 30. In the layer-3 emitter of `T`, a text neuron is
ReLU(OP_i + SLOT_j − END − 1) and a reference neuron is ReLU(ptr + 64(OP_i + SLOT_j − END) − 128); layer 4 turns an address into the
reference token with step thresholds (Appendix A). `A` uses 16 heads + 1,963 neurons, `T` 6 + 1,100; these are the formula
allocation and not a measured minimum. The networks are exact on 1,497/1,497 expressions (all 160 of depth ≤ 1, stratified above,
powers 1–8, padding 0–7) and 2,265/2,265 on the widened retry domain, with minimum logit margins 30 (`A`) and 27.5 (`T`).

Table F1 records the fixed-parser routes alongside the worked circuit above. The shared parser supplies subtree spans and children. Scratchpad T fetches a node and its children's results, emits a local rewrite, and converts addresses into reference tokens. Answer-only A descends the expansion template and copies source spans. PART trains only the rewrite emitter in T or descent emitter in A; parsing remains frozen.

<details>
<summary>Technical shared parser and alternative derivative writers (Table F1)</summary>

*Table F1. Parse first, then write a derivative. A shared parsing front end gives each subtree's span and children; the rows are alternative output routes. Format `T` emits one rewrite row per node and format `A` expands a template and copies spans. Only the emitter MLPs (layer 3 in `T`, layer 13 in `A`) are re-initialised and trained in PART.*

| Route or component | Fixed computation and trained interface |
|---|---|
| Shared frozen parser | Read pad dots, prefix source symbols and =. A uniform prefix head counts bracket balance and completions and supplies a position frame. Depth- and recency-routed heads read subtree start, end and children. |
| T: register rewrites, 5 layers | Fetch the completed node and its children's results. The layer-3 rewrite MLP, trained in PART, uses text and reference neurons on OP, SLOT and END (operation, slot and end flag). Fixed layer 4 thresholds the address into a reference token. |
| A: expanded derivative, 15 layers | Propagate derivative lengths and descend the expansion template. The layer-13 descent emitter MLP is trained in PART; source spans are copied into the expanded output. |

</details>

## 4. What training finds

The 2026-11-08 analysis found that the formula-built HAND differentiates every tested expression (`A` 16 heads + 1,963 neurons, `T` 6 + 1,100). Table 2 gives
every trained cell. Trained from scratch, both formats solve, all exact on held-out expressions: `A` 3/3 seeds, `T` 3/3 in the screen
and 10/10 in the final cell, the 16 TRAIN networks of Section 5.1. PART solves 0/12 seed-runs (.46–.92), also with 5.6x more distinct training expressions.

*Table 2. Results per cell (ranges over seeds; run 2 = the data retry).*

| Cell | Solved / seeds | Best / final exact | Stop (k steps) | First held-out exact 1.0 (k steps) |
|---|---|---|---|---|
| TRAIN A r1 | 3/3 | 1.0 / 1.0 | 17.5 | 16, 17.5, 17 |
| TRAIN T r1 | 3/3 | 1.0 / 1.0 | 19 | 18.5, 16, 16 |
| TRAIN T final (wd .1) | 10/10 | 1.0 / 1.0 | 19 | 14.5–19 |
| PART A r1 / r2 (data) | 0/3 / 0/3 | .70–.73 / .46–.52 (final .70–.73 / .46–.52) | 22.5 / 22.5 | — |
| PART T r1 / r2 (data) | 0/3 / 0/3 | .79–.85 / .89–.92 (final .78–.80 / .88–.91) | 22.5 / 24.5 | — |

Key: r1 = run 1, r2 = run 2 (data retry); a dash = never reached.

### 4.1 Learning curve

Held-out exact match reached 50% and 90% at 3.5–4.5k and 10–11k steps in format `T`, and at 1.5k and 5–6k in format `A`. The first
held-out 1.0 lagged the first train 1.0 by 0–5k steps. PART `A` peaked at 0.5k–2k steps with train exact also < .8, a plateau. PART
`T` reached train exact 1.0 on its 128-row monitor at test .78–.80 in run 1, which is memorisation; with 5.6x more distinct
expressions (run 2) test rose to .88–.91, still unsolved.

### 4.2 Circuit types

Circuit types were clustered by ablation map, patching, attention divergence, edges and alignment; each TRAIN cell has one type.
TRAIN `T` has one type in 10/10 final seeds and 3/3 screens, a weighted-prefix count with a late-MLP rewrite read-out: all five MLPs
are needed, last-layer heads are used in 19–20 of 20 snapshots, MLP4 is the top patching recovery in 20/20, and the strongest edges are
MLP4 to output and MLP3 to MLP4 (Table F2). Layer-3 and layer-4 head signatures differ by seed and switch between best and final in 3 seeds; a
seed-0/seed-1 interpolation barrier of 1.40 nats shows different basins of the same type. TRAIN `A` has one type in 3/3, a late MLP14
read-out: 1–4 heads and 3–10 MLPs are used (HAND 11 + 12), the MLP14 patch drop is .81–.95, and the pointer, start and node-end
interchange flips are ≈ 1 only in layers 12–15. HAND as a causal model of the network scores .10–.18 (HAND on itself .84).

### 4.3 Why PART fails

All six `T` runs write symbol and reference logits directly and bypass the frozen reference decoder: ablating it leaves exact 1.0 on
the diagnostic rows. All six `A` runs route through a different frozen head and pair a copied-leaf classifier with a failed pointer
hand-off. Repairing only the pointer (`PTR4 = PTR3 if MC3 > .5 else 0`) takes `A` seed 1 from .734 to 1.0 (held-out rows); fixing
`T` seed 0's row headers (`10 − (FP − p)²`) takes it from .852 to .992. PTR3/PTR4 are the source-copy addresses before and after the final hand-off, MC3 flags the source-copy phase, and p in the header repair is the candidate address. The emitter must hit unit-precise addresses on frozen
thresholds, and more data (run 2) did not supply them. The final snapshot of `T` run 2 seed 0 scores 63/64 at depth 3; its one error is
a root-product source reference.

PART networks share HAND's frozen state (identical upstream probes), so their failures lie in the interface of the trained stage and
not in missing upstream variables. Diagnostics per cell are in Appendix B.

## 5. Mechanism and verification

### 5.1 First layer of TRAIN: a content-weighted count

In all 16 TRAIN nets (best snapshots), the first-layer attention was replaced by a_ij = softmax_{j≤i} S_r[t_i, t_j], with
S = (E W_Q + b_Q)(E W_K + b_K)ᵀ/√192 and S_r its rank-r singular-value decomposition (SVD) truncation, retaining only r independent directions: a 104 × 104 token-pair table per head, computed from the
weights only, with the values and the rest of the network unchanged. E contains token embeddings; W_Q/W_K are the query/key matrices and b_Q/b_K their biases. Equivalently, each head returns
Σ_u n_u exp(S[t,u]) v(u) / Σ_u n_u exp(S[t,u]), with n_u the count of token u in the causal prefix, which is a content-weighted
histogram; v(u) is the head's value vector for token u. On 256 held-out rows per net, whole-output exact match goes from native 1 to 1 at rank 4, 8, 16 and with the full table in
16/16 nets (2·4·104 = 832 numbers per head); rank 2 keeps it in 12/16 and rank 1 in 7/16 (`A` 3/3, `T` 4/13). HAND's uniform average
(S = 0) gives `T` 0–.38 and `A` .81 / 1 / .93. The front end of `T` is therefore a content-weighted count of rank ≤ 4 and not HAND's
average, while that of `A` is near-uniform. This is supported by substitution; no
[resample ablation of the complement](../GLOSSARY.md#verifying-a-circuit) was reported, so the status is partial.

For seed 2 (`T` run 1), with MLP0 also substituted as a_j = ReLU(α[t,j] + Σ_h C_h β[h,:,j]), the network keeps 99.2% (native 99.2% on
that cohort). A rank-8 front plus all MLP0 hinges uses 205,440 numbers against 887,040 native and keeps exact 1 (native 1) on fixed-shape batches in
all three run-1 seeds.

### 5.2 Rewrite stage of TRAIN T: no compact form

For the last MLP (MLP4) of `T` run 1, seeds 0–2, best and final, a table m̂(q) = b_out + W_out θ[s(q)] was fitted, with state
s = (slot, rule, node and child endpoints, power, tree depth, line, current token) and θ the mean training activation per state (609
states). It scores 512/512 on fresh held-out expressions in all six nets (global-mean control 0/512), but it uses 234,240 fitted numbers
against 295,680 MLP parameters and scores 5/256 on mixed-shape rows, so it was rejected: it does not compress and holds only for a fixed
shape. The write alone is compact: m̂ = μ + a[y]·B over the 53 emitted tokens y (rank 24, 10,872 numbers) is exact on two mixed-shape
cohorts in 6/6 nets, but only when y is supplied by the hand-built parse. A linear read-out of y from the MLP4 input gives .87–.98, with
derivative addresses off by ±1, so the token decision has no compact form. Whole-program fits on pruned TRAIN `T` circuits fail (14
joint fits: 11.3%; refit 56.6%).

Table F2 records the supported front end and unresolved rewrite stage. Trained content-weighted prefix heads feed the MLP stack, while final-layer heads also enter MLP4's rewrite readout. All displayed computational stages were trained. The partial status covers first-layer substitution with complement untested; the later rewrite readout still has no compact closed form.

<details>
<summary>Technical learned front end and unresolved rewrite (Table F2)</summary>

*Table F2. A prefix summary feeds the rewrite in format `T` trained from scratch (10/10 final seeds, 3/3 screens); all five MLPs are needed, and MLP3 to MLP4 is the strongest MLP edge. MLP4 also receives final-head inputs; the heads are not claimed to follow MLP3. The first layer is a content-weighted prefix count. Its status is Partial, supported by substitution with the complement not tested; the rewrite read-out in MLP4 has no compact closed form. All listed computational components are trained in TRAIN.*

| Trained component | Input and recorded role |
|---|---|
| Layer-0 heads | Weight earlier prompt and row tokens by content, forming a prefix count with a rank-4 token-pair table. |
| MLP0 to MLP2, then MLP3 | Process the prefix summary in the needed MLP stack. This is not a compact formula for the later rewrite. |
| Final-layer heads and MLP4 | Heads supply extra reads to MLP4 alongside MLP3's write. MLP4 writes the output rewrite token and has the top patching recovery. Only first-layer substitution is supported; the complement is untested and the rewrite has no compact form. |

</details>

### 5.3 PART T stage: a faithful fit of a failing network

For the PART `T` stage (best and final, seeds 0–2), the form a_j = ReLU(b_j + u_j·FOP + v_j·SLOT + f_j FP + g_j FE1 + h_j FE2 + e_j END),
with output c + Σ_j w_j a_j on the symbol block only and inputs the HAND rule, slot and node and child endpoint coordinates, keeps the
native accuracy with zero changed tokens in 12/12 comparisons (128 fresh rows and 256 guard rows, .770–.875, native errors included).
It uses 53,096 fitted numbers (18% of the MLP). It is faithful to the failing networks and is not a solution.

### 5.4 Relation to the hand-built network and to published work

This task has the opposite outcome to the arithmetic entries. There, PART on frozen hand-built parts is the easy rung; here the frozen
parser hands the emitter discrete addresses that a ReLU MLP trained by gradient does not place, while full TRAIN learns its own
interface and solves. Trained `T` networks keep HAND's variables decodable (final seed 4: bracket depth and completion R² ≥ .99, child
endpoints ≥ .94, REF .96) but compute them with weighted-count heads instead of HAND's depth-match heads. TRAIN networks carry the
reference address and use the rule more swappably than HAND (rule flip `A` .39–.85, `T` .49–.60 against HAND .35 / .20), although its
probe R² is lower in `T`: a different rule code, causally used. TRAIN `A` uses far fewer components (1–4 heads) than HAND `A` (11).

Both formats solve TRAIN at similar steps (`A` 16–17.5k, `T` 14.5–19k). Format `T` is smaller by construction and shares
subexpressions, so this is not a compute-matched comparison of formats. The scratchpad follows [Nye et al. (2021)](#references); with
fresh data every step there is no delayed generalisation of the kind reported by [Power et al. (2022)](#references).

## 6. Limitations and open questions

Length was tested on the trainer's extension sets (fp32, 128 rows per seed; best / final). Every TRAIN net scores 0 at depth 3 and at
powers 7–8, whose outputs need tokens never emitted in training. On the withheld compositions `A` scores 1, `T` run 1 1 / .48 / 1 and
the `T` final .47–1 (6/10 best and 5/10 final at 1); on padding 4–7 `A` scores .79–1 and `T` 0–.09. PART on the same sets: `A` depth 3
0 (run 1), `T` depth 3 .29–.39 (run 1). HAND scores 100% everywhere.

Size was measured by cumulative ablation with constant replacement (best snapshot): TRAIN `T` seed 2 keeps 9 heads + 793 neurons and
seed 1 9 + 941 (native 10 + 1,920, HAND 6 + 1,100), both at 99.2% on the discovery cohort but 93.4% on a guard set. These are local
upper bounds and not minima.

The diagnostic cohort is one fixed shape. Edges, alignment and interpolation were run on part of the networks (interpolation on one
pair). Probe R² measures linear decodability, not causal use; interchange with nf = 0 has no eligible answer-fixing queries, so it is untested ([glossary](../GLOSSARY.md#analysis-measures)). Neither that result nor ties with a uniform head in attention divergence establish use or absence. Neuron-level ablation and checkpoint dynamics for most seeds were not run. The residual failure of the widened-range PART scratchpad lies outside both diagnostic cohorts and remains undiagnosed. Ten-seed finals of TRAIN `A` and the PART cells were not run; public code is unavailable.

## 7. Prediction record

The predictions were recorded before training (3-seed screens). PART `A` and `T` ≥ 2/3 within 10k steps was broken (0/3 each, also
after the data retry). TRAIN `T` ≥ 2/3 within 30k held (3/3, then 10/10). TRAIN `A` ≤ 1/3 was broken (3/3). Poor depth transfer of
TRAIN `T` held (0 at depth 3 in every seed). That the final emitter of `T` is a rule/slot/child-reference lookup partly held: its last
MLP is reproduced by a table over exactly these keys, but the table is not compact. At least one solved seed with decodable spans but
no HAND-like depth-match head held: the first layer is a weighted count, not a depth match.

## Appendix A. Weights excerpt and reading

PART `T` seed 0, best, neuron 138 of the fitted stage form is ReLU(b + w_op·FOP + w_slot·SLOT − .172 FP − .0025 FE1 − .0068 FE2 −
.111 END) (fit coefficients, rounded in the fit output). It is an address boundary on the node end, shifted by rule and slot, and not a
power detector; deleting the neuron from the native network drops exact match from .852 to .547.

In the HAND layer-3 emitter of `T` (formula weights), a text neuron is ReLU(OP_i + SLOT_j − END − 1) with weight 1 into its output
symbol, so it fires only for its exact (operator, slot). A reference neuron is ReLU(ptr + 64(OP_i + SLOT_j − END) − 128), which passes
the child address ptr (an integer scalar FE1, FE2 or FP) through unchanged when both gates are on and is 0 otherwise; layer 4 turns
the address into the reference token with step thresholds. The trained PART emitter replaces these gates by graded hinges on the same
coordinates, which is why it misses unit-precise addresses.

In TRAIN `T` MLP4 (run 1 seed 0, best; vote = W_U·W_out column, a weight slice), syntax and coefficient tokens have dedicated detector
neurons (`:` n381 vote +1.88, runner-up +.33), while address neurons are broad bumps that also vote for the neighbouring address (`d7`
n18 +1.22, `d8` +1.14): an address is the overlap of several bumps. The TRAIN first-layer tables are computed from E, W_Q, W_K and W_V
of the archived networks (formula in Section 5.1).

## Appendix B. Detailed evidence

Diagnostics use 128 in-distribution depth-2 rows, on which the failed PART networks score 1.0; their errors are at other shapes.

*Table B1. Diagnostics per cell (ranges over seeds, best and final snapshots).*

| Cell | Diag exact | Used H / M | Rule R² | Address R² (A power, T REF) | Flip rule / address; DAS | r99 | JS |
|---|---|---|---|---|---|---|---|
| HAND A | 1 | 11 / 12 | .928 (9) | .961 (11) | .35 / 1; 1 | 9 | 0 |
| HAND T | 1 | 5 / 5 | .865 (5) | 1.0 (4) | .20 / .15; .86 | 6 | 0 |
| PART A r1 / r2 | 1 | 11 / 12 | .85–.93 (9) | .92–.96 (11) | .32–.43 / 1; .89–1 / .33–1 | 9 | .001 |
| PART T r1 / r2 | 1 | 5 / 4 | .95–.96 (4) | .63–.74 / .68–.73 (4) | .10–.29 / .07–.11; .85–.98 / .24–.99 | 7–8 | 0 |
| TRAIN A r1 | .95–1 | 1–4 / 3–10 | .93–.95 (9–14) | .97–.999 (12–14) | .39–.85 / .50–.61; .72–1 | 20–28 | .12–.19 |
| TRAIN T r1 | 1 | 5–7 / 5 | .66–.72 (5) | .95–.97 (5) | .50–.60 / .27–.34; .86–1 | 24–32 | .13–.28 |
| TRAIN T final, 10 seeds | .97–1 | 5–10 / 5 | .65–.77 (3–5) | .94–.98 (5) | .49–.57 / .29–.39; .93–1 | 21–32 | .13–.27 |

Key: Used = mean-ablation exact drop > .05, H heads / M whole MLPs (neurons not ablated). R² = best linear probe (layer in brackets).
Flip = maximum-over-layers interchange flip toward the source answer; DAS = distributed-alignment flip of the rule
([glossary](../GLOSSARY.md#analysis-measures)). r99 = final-residual rank at 99% variance. JS = mean over the network's heads of the
attention divergence to the best-matched HAND head (the per-head maximum reaches .44–.60 in TRAIN `T`). Best and final held-out exact
are in Table 2. Low flips on HAND, whose codes are scalar, make the flip a weak test of absence here.

## What to read next

Read [polynomial differentiation](polynomial-differentiation.md) for a related symbolic task without the same nested-expression rewrite. [Arithmetic registers](compose-arithmetic-registers.md) provide a comparison for intermediate results stored in named slots.

## References

- Nye, M., Andreassen, A. J., Gur-Ari, G., et al. (2021). Show your work: Scratchpads for intermediate computation with language
  models. *arXiv preprint*. https://arxiv.org/abs/2112.00114 (scratchpad format)
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization beyond overfitting on small
  algorithmic datasets. *arXiv preprint*. https://arxiv.org/abs/2201.02177 (delayed generalisation)
