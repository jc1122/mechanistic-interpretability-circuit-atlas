# Subtraction (base 10 and 2)

In 742 minus 358, the units column must borrow before subtracting 8. Each of four retried direct-answer/column-difference settings got whole outputs right in 10/10 runs on 2k decimal or 832 binary test pairs per run.

A table reproduced most local responses when supplied the signed difference after borrowing. That label came from outside the network, so the table did not explain how the network computed it. The stronger whole-mechanism test failed. Scores supplied correct earlier tokens.

## The circuit at a glance

Figure 3 shows the observed direct digit write that bypasses the hand-built store ([mechanism and verification](#5-mechanism-and-verification)).

<img src="../figures/circuits/subtraction.svg" width="640" alt="The trained block writes digit scores, bypassing the hand-built store. One upward residual stream has separate read/write branches for Read first bit (layer 2, head 0); Read other bit (layer 2, head 1); Write digit (layer 2 block). Local input: 1 − 0; given incoming borrow 1; Digit target: 0. Task values are illustrative; dashed boxes and branches are not verified.">

*Figure 3. Binary borrow-token partly trained seed 0 differs from the hand-built design through its layer-2 feed-forward block writing digit scores around the supplied digit store at digit queries, a sufficient-readout result rather than endogenous difference/borrow proof; separate from-scratch whole-role controls failed.*

| Fact | Value |
|---|---|
| Task | Subtract b from a (a ≥ b), 3-digit decimal or 6-bit binary, written as `a-b=` followed by the difference |
| Input → output example | Base 10, answer only: 742 − 358 → `384` |
| Model | Four-layer transformer, 4 heads per layer, width 256; PART trains one feed-forward block, MLP3 in answer only or MLP2 in the scratchpads |
| Rungs and seeds | HAND: formula weights in 6 base/format cells; PART: frozen routing plus one learned block; TRAIN: all weights learned; 10 seeds per learned cell, with retries retained |
| Closed form status | Sufficient readout only (external-label tables; whole-role scrubs .938/.933 in two tested TRAIN seeds; the 2026-11-07 audit's full bar was not passed) |
| Status | Partly solved (four TRAIN retry cells, each 10/10 at test exact match 1.0; borrow-token cells use a different criterion); last run 2026-11-05; audit 2026-11-07 |
| Weights | [Hugging Face: subtraction](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/subtraction) (112 networks) |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

Four-layer networks had to learn both the answer digit and how borrowing affects the next column. Whole-network training did not transfer to longer numbers.

The task is partly solved. Its formulas are [sufficient readouts only](../GLOSSARY.md#verifying-a-circuit): swapping internal states between examples with matching claimed arithmetic labels kept only .938 and .933 whole-output accuracy in the two tested binary borrow-token runs, below the .99 control bar.

Read next: [Results](#4-what-training-finds) and [verification](#5-mechanism-and-verification), then [router addition](router-addition.md) for shared column labels, or [binary multiplication](multiplication.md) for a different sufficient-readout example.

## 1. Task and data

The prompt is `a-b=` and the target is a − b with a ≥ b. For example, 742 − 358 = 384: the units column needs to borrow before subtracting 8. Write a_i and b_i for operand digits, c for the incoming borrow (0 or 1), d_i for an answer digit and β for the number base. Older formulas below use b for the base, not for an operand digit. In base 10 the operands have 3 digits and each seed draws 20k random
training pairs and 2k test pairs. In base 2 the operands have 6 bits; of the 2080 pairs with a ≥ b, a train fraction of .6 gives 1248
training and 832 test pairs. An output counts as correct only on exact match of all output tokens. Length generalisation, accuracy on
operands longer than any seen in training ([glossary](../GLOSSARY.md#task-formats)), was tested at n + 1 and 2n, that is 4 and 6 digits in base 10 and 7
and 12 bits in base 2. Three output formats were studied (Table 1); B and C are [scratchpad](../GLOSSARY.md#task-formats) formats, in
which the network writes intermediate results before or between the answer digits.

*Table 1. Output formats (n = number of digits, k = borrow token).*

| Format | Output |
|---|---|
| `A` (answer only) | n answer digits, most significant first (MSB first) |
| `B` (borrow token) | d_0 k_1 d_1 .. k_n, least significant first (LSB first) |
| `C` (signed differences) | the signed column differences e_i = a_i − b_i (n tokens), `=`, then d_{n−1}..d_0 |

## 2. Model

The shared architecture and instrumentation are described in [METHODS.md](../METHODS.md). The three
[rungs](../GLOSSARY.md#the-three-rungs) are HAND (every weight set by formula), PART (frozen hand-built parts with one trained stage)
and TRAIN (all weights trained from scratch).

### 2.1 Architecture

All networks are GPT-NeoX-shaped decoders, transformers that use earlier tokens to predict the next one, with 4 layers × 4 attention heads, d_model 256 and d_head 64. Here d_model is the running-state width and d_head each head's vector width; a head reads information from earlier positions. Each MLP, a feed-forward block at one position, has width 256 and uses ReLU, which keeps positive inputs and sets negative inputs to zero. There is no LayerNorm, which would centre and rescale the running state, and no positional encoding (NoPE), meaning no added position vectors. These terms are defined in the [glossary](../GLOSSARY.md#standard-machine-learning-terms). The vocabulary holds BOS, unk, the 2b − 1 signed values −(b−1)..b−1 for base b, and `- = . c`: 25 tokens in base 10
and 9 in base 2. Layers are numbered from zero. TRAIN trains all 1.6M weights from initialisation.

### 2.2 Frozen hand-built network and trained part

PART trains the stage MLP only, in layer 3 for format A and in layer 2 for formats B and C (dense_h_to_4h [256, 256] + bias,
dense_4h_to_h [256, 256] + bias), on top of the frozen hand-built layers. The stage reads the token embedding TOK, the fetched column
digits X1, X2 and X3, the borrow CY, the phase flags PH and EQS of format C, and the position blocks P, R, R2, T, RP and VT. TOK represents the current token by a one-hot vector, with one coordinate per token. The position blocks encode absolute positions, their powers or cumulative thresholds; their exact scale matters to the mask control, not to the borrowing rule ([number codes](../GLOSSARY.md#standard-machine-learning-terms)).

### 2.3 Training

The optimiser was AdamW with learning rate 1e-3, weight decay 1, betas (.9, .98) and batch 256, in TF32, with test accuracy evaluated
every 500 steps and the best-test snapshot kept. This reuses the test set for selection; it is not an untouched final test. TF32 is reduced-precision matrix multiplication; fp32 re-scoring uses full precision ([glossary](../GLOSSARY.md#hand-built-machinery)). PART and TRAIN used 10 seeds per cell. The last recorded run was 2026-11-05. Runs were on consumer GPUs. Table 2 gives the per-run settings. A control repeated
PART with the stage's position inputs held at 0.

*Table 2. Training settings per run.*

| Run | Steps | Schedule | Weight decay | Stop |
|---|---|---|---|---|
| PART | 10k | constant | 1 | on 1.0 or a 2k plateau (base 2: 20k) |
| TRAIN run 1 | 20k | constant | 1 | on 1.0 or a 2k plateau (base 2: 20k) |
| TRAIN run 2 (retry) | 30k | warmup 1000, cosine, clip 1 | base 2: 1; base 10: .1 | |

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/subtraction](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/subtraction)
(112: 6 HAND, 60 PART, 30 PART with position mask, 16 TRAIN).

### 2.5 Archive

The Hugging Face folder holds `hand_b<base>_<F>` (HAND state dicts); `part_b<base>_<F>_s<seed>` (all PART seeds; the four stage
tensors); `part_b10_<F>_maskpos_s<seed>` (the position control); `train_b<base>_<F>[_r2]_s<seed>` (full networks, best and worst seed
per cell and run; `_r2` marks the retry); `config.json`; and `manifest.json` (sha256, fp32 re-score). Each file was reloaded
bit-identically.

## 3. Hand-built solution

The hand-built networks reuse the bricks of [decimal-hex-addition](decimal-hex-addition.md), with a borrow in place of the carry.
Each answer digit is (a_i − b_i − borrow) mod b, and the formats differ in where the borrow comes from (Table F1).

In format A, layer-2 index heads, which attend to one operand position and copy its digit, copy a_i and b_i, and a_j at b_j's
position. The L2 MLP reads D = b_j − a_j and writes EQ = [D ≠ 0] and GEN = [D ≥ 1]. The chain head L3H0 attends to the first lower
column with EQ set and copies its GEN, which is the incoming borrow borrow_in. The L3 MLP writes the one-hot of (a_i − b_i − c) mod b,
where c is the borrow, with 2b hinge triangles. In format B the L2 MLP reads z = a_i − b_i − [own token `c`] + b and gives the digit
z mod b and the outgoing borrow 1 − [z ≥ b]; L3H0 copies the previous borrow token. In format C, phase 1 writes e_i = a_i − b_i, the
chain head finds the first e_j ≠ 0 (borrow = [e_j < 0]), and phase 2 writes the one-hot of (e_i − c) mod b. All six networks are exact
for n = 1..12 in both bases. Table 3 gives their sizes.

*Table 3. Size of the base 10 hand-built networks (heads; neurons in L0/L1/L2/L3).*

| Format | Base 10 |
|---|---|
| A | 6; 64/114/4/22 |
| B | 5; 64/64/3/14 |
| C | 5; 64/119/43/0 |

For base 2, the compact note “L3 6/6/0, L2 4/3/11” lists formats A/B/C in that order. The source design notes give A 64/114/4/6, B 64/64/3/6 and C 64/119/11/0 neurons in L0/L1/L2/L3, with 6, 5 and 5 heads respectively. These are hand-built counts, not measured TRAIN minima.

Table F1 records the three alternative borrow sources. The shared readers do not make the answer-only, borrow-token and difference-trace formats successive stages.

<details>
<summary>Technical alternative borrow routes and trained stages (Table F1)</summary>

*Table F1. The hand-built networks for the three formats. Each alternative reads the common input; the formats are not successive stages. Index heads fetch the column digits; the borrow comes from a chain head that finds the first lower column where the digits differ (A, C) or from the previous borrow token (B). The MLPs (layer 3 in A, layer 2 in B and C, which in C writes both phases) are the parts that PART replaces with a trained stage. The learned tables are sufficient readouts only: external-label substitutions do not establish how TRAIN computes its difference or borrow, and the stronger audit controls failed.*

| Part or alternative format | Input, computation and destination |
|---|---|
| Common input and readers; frozen | Prompt `a-b=` and output so far feed layer-2 index heads copying a_i, b_i and a_j at b_j's position. |
| A: comparison and borrow chain; frozen | Layer-2 feed-forward block (MLP) computes D = b_j − a_j, EQ = [D ≠ 0], GEN = [D ≥ 1]. Layer 3, head 0 (L3H0) finds the first lower column with EQ and copies GEN as borrow_in. |
| A: digit stage; trained in PART | Layer-3 MLP writes the one-hot category of (a_i − b_i − c) mod b, where c is the borrow and b the base. |
| B: local digit/borrow stage; trained in PART | Layer-2 MLP writes digit z mod b and borrow 1 − [z ≥ b], with z = a_i − b_i − [own token c] + b. Frozen L3H0 copies the previous borrow token for the output route. |
| C: signed-difference stage; trained in PART | Phase 1 writes e_i = a_i − b_i. A frozen chain head finds the first e_j ≠ 0; borrow is 1 if e_j is negative. Phase 2 writes the one-hot of (e_i − c) mod b. Both phases use the layer-2 trained stage. |
| Audit boundary | Later task-label readout tables do not recover endogenous TRAIN difference or borrow. Their stronger whole-mechanism controls failed. |

</details>

## 4. What training finds

Table 4 summarises the cells. Exact match is on held-out pairs; "steps to .9" is the first evaluation at which test exact match
reached .9.

*Table 4. Results per cell (ranges over seeds; "position masked" = the stage's position inputs held at 0; run 2 is the retry).*

| Cell | Test exact match | Steps to .9 | Length n + 1 | Length 2n |
|---|---|---|---|---|
| PART base 10 A / B / C | 1.0 / .995–1 / .97–1 | ≤ 1k | A .80–.98, B ≤ .20, C ≤ .02 | A .80–.96, B ≤ .04, C 0 |
| PART base 10, position masked, A / B / C | | | A .99–1, B .87–.98, C .95–.99 | A .98–1, B .39–.60, C .52–.59 |
| PART base 2 A / B / C | A, B 1.0, C .93–.95 (fp32 re-score 1.0) | | A .78, B 0–.82, C .13–.42 | ≤ .67 |
| TRAIN base 10 run 1, best (final) | A .72–.99 (.01–.90), B .94–1 (.01–1), C .48–1 (0–.98); seeds ≥ .9: 7, 10, 9 | 3–12k | ≤ .06 | ≤ .001 |
| TRAIN base 2 run 1 | A .40–.95 (1/10 ≥ .9), B .79–1 (9/10), C .55–.95 (4/10) | | ≤ .35 | ≤ .001 |
| TRAIN run 2: base 10 A, C; base 2 A, C | 10/10 = 1.0 each (final = best) | base 10 A 2.5–5.5k, C 2–4.5k; base 2 A 8.5–14.5k, C 6–11.5k | base 10 ≤ .06, base 2 ≤ .35 | ≤ .001 |

### 4.1 Partly trained networks

Most PART stages admit f(diff), a readout table indexed by diff = a_i − b_i − borrow_in; Table 5 retains the exceptions. The original family counts are 10/10 in base 10 A/B and base 2 A/C, 9/10 in base 10 C and 8/10 in base 2 B. These are sufficient-readout classifications, not verified mechanisms. The adder's key a + b + c gives ≤ .01 in base 10 and ≤ .88 in base 2, except binary A: both keys have the same parity for the answer bit, so that test cannot distinguish addition from subtraction. Borrowing must be checked separately.

PART retains the needed HAND routes, with attention substitution 1.0. A transfers partially to longer operands; B/C depend on absolute position. Holding the stage's position weights at 0 improves B to .87–.98 at n + 1 but only .39–.60 at 2n; C reaches .52–.59 at 2n. This is an incomplete repair. Remaining errors occur on new combinations of value and token role even when each coordinate is within its training range. Clamping those inputs to their mean drops accuracy at n = 3 to ~.01, which locates
the length failure of PART B and C in the stage's position read.

### 4.2 Networks trained from scratch

In run 1 the base 10 cells reached .9 in 7, 10 and 9 seeds (A, B, C), and the failures of A and C oscillate under weight decay 1 with
a constant learning rate, so the final accuracy is far below the best. The retries with warmup, cosine decay and clipping solved every
seed (base 2 C at weight decay 1, base 10 at weight decay .1). Failed TRAIN networks keep the stage key (diff R² .84–.98) but carry a,
b and the borrow only partly linearly (R² .3–.9). No TRAIN network generalises to longer inputs.

TRAIN uses 3–16 heads + 300–980 neurons, against HAND's 5–6 + 75–230. HAND's fetch roles (the L2 index heads) are re-found in the
retries in base 10 A 9/10, C 9/10, base 2 C 9/10 and base 2 A 2/10 seeds, with 3–16 heads + 226–762 neurons.

### 4.3 Circuit types

Seeds were grouped into circuit types by the key whose substitution for the last MLP keeps accuracy within .01 of the network
(substitution ≥ net − .01; Table 5). Over all cells, the last MLP of TRAIN uses f(diff) in 21/30 networks in base 10 run 1, 29/30 in
base 2 run 1 and 34/40 in the retries. Earlier TRAIN MLPs have no closed form.

*Table 5. Circuit types per cell (seed counts out of 10). f(a, b, c) = a lookup on the pair of digits and the borrow.*

| Cell | Base 10 | Base 2 |
|---|---|---|
| PART A | f(diff) 10 | f(diff) 10 |
| PART B | f(diff) 10 | f(diff) 8 + f(a, b, c) 2 (its frozen MLP3 takes f(diff) in all 10) |
| PART C | f(diff) 9 + f(a, b, c) 1 | f(diff) 10 |
| TRAIN run 1 A | f(diff) 7 | f(diff) 9 |
| TRAIN run 1 B | f(diff) 8 | f(diff) 10 |
| TRAIN run 1 C | f(diff) 6 + f(a, b, c) pair lookup 4 (the compact 3–4-head networks) | f(diff) 10 |
| TRAIN retry A | f(diff) 10 (substitution ≥ .99) | f(diff) 6 + f(a, b, c) 2 + unclassified 2 |
| TRAIN retry C | f(diff) 8 + f(a, b, c) 2 | f(diff) 10 |

The primary result summary instead classifies binary A retry as 6 difference + 3 pair + 1 unclassified. Both records are retained. The raw separate analysis for seed 0 gives native 1.0 and pair substitution `0.9893333315849304`, below the native-minus-.01 bar despite a rounded .99 display. No replacement census or merged total is inferred from rounding. The original 34/40 difference-readout count is unchanged by this pair/unclassified disagreement.

Per-cell diagnostics (probes, participation ratio, attention divergence, patching, interchange and edges) are in Appendix B.

## 5. Mechanism and verification

### 5.1 Closed forms

The last MLP was replaced by its mean output given diff = a_i − b_i − c (in phase 1 of format C, given a_i − b_i). The original overview reports PART .97–1, TRAIN f(diff) networks within ± .01 and base 2 C retry 1.00. That overview is not a universal accuracy range: decimal B PART separately scored .91–.99 in fp32 versus .995–1.0 in GPU training. Failed and unclassified substitutions remain outside this family claim. The digit is (a_i − b_i − c) mod b and the
outgoing borrow is [a_i − b_i − c < 0].

These are [sufficient readouts](../GLOSSARY.md#verifying-a-circuit), not verified mechanisms. The audit of 2026-11-07 found that none
of the closed forms passes the full bar (best and final snapshots, native/zero/shuffle exact controls, fewer fitted reals than the replaced
component); they
are keyed on task-derived labels and stay sufficient-readout results.

The 2026-11-07 check used a tent-spline table F_q(z), a piecewise-linear table on the signed difference z = a_i − b_i − c. Decimal fits used 4,000 pairs and evaluation used 2,000 new pairs, with a bar of ≥ .99 × the network's accuracy (Table 6). Binary A instead fitted the 1248 training pairs and evaluated the 832 finite-domain nontraining pairs. Only their ordering was fresh; this was not a never-evaluated binary domain. The replacement operated at scored queries only, leaving other positions and components native. The TRAIN f(diff) claim is not universal: TRAIN seed 8 fails it, its
f(a, b, c) table fails too (.014) and would need 204,800 fitted reals, more than the MLP's 131,584. A decimal B rank-six readout card
also fails exact.

*Table 6. Tent-spline substitutions at scored queries. Decimal evaluation has 2,000 fresh pairs per seed; binary A reuses its 832-pair nontraining domain. Accuracy cells show native, then substituted. These are sufficient readouts only.*

| Stage | Passing seeds | Accuracy |
|---|---|---|
| binary A PART MLP3 | 10/10 | 1 → 1 |
| decimal B PART MLP2 | 10/10 | .92–.99 → .99–1 (a repair) |
| decimal B final TRAIN MLP3 | 9/10 | |
| decimal B TRAIN seed 8 (final) | counterexample | 23/2,000 correct → 22/2,000 |
| decimal B rank-six readout card | fails exact | .9987 → .9827 |

### 5.2 Wiring and causal tests

The component with the largest patching drop in PART is the stage MLP in 60/60 networks. Edge attribution (EAP-IG) recovers HAND's
wiring in PART (hand_recall 1.0; Jaccard overlap 1.0 in A and C, .60-.80 in B; 51/60 networks trusted). TRAIN is wired differently
(Jaccard 0-.20; heads not aligned, which raises a different-wiring flag). The borrow-chain variable X2 of format C is present but not
linear: it flips .00-.06 by linear interchange in every rung, against a DAS flip of 1.0 (base 2), .97 (base 10 TRAIN retry) and .36
(base 10 PART). In base 10 PART B the digit flips .32 by interchange and .96 by DAS. The [glossary](../GLOSSARY.md#analysis-measures)
defines patching, interchange, DAS and edges.

### 5.3 Circuit probes

Circuit probes were run on 2026-11-06 on seed-0 best snapshots unless noted (Table 7). The table records causal interventions plus component substitutions fitted and tested on disjoint non-training pairs. It does not verify a complete mechanism under the atlas acceptance rule. Exact is teacher-forced whole-output accuracy: correct earlier output tokens are supplied, rather than generated by the network ([glossary](../GLOSSARY.md#task-formats)). Ablation numbers are accuracy retained, and layers are zero-based. In the table d = a_i − b_i −
c_i, q is the index of the output query, and μ[key] is the mean MLP-output vector over the fit set for each value of the key.

*Table 7. Components probed (seed 0 unless noted).*

| Component | Role | Component evidence, not whole-mechanism verification |
|---|---|---|
| Base 10 A PART: L2H0/H1/H2, L3H0 | HAND operand, comparison-column and borrow routes | exact HAND route substitutions 1.000 each; mean ablation 0 / 0 / .602 / .313 |
| Base 2 B PART: L2H0/H1, L3H0 | operand fetches, borrow relay | hard routes 1.000 each; mean ablation .016 / .031 / .391 |
| Base 2 B PART: trained MLP2 → answer logits | digit path bypasses HAND's DIG register | deleting its digit write at digit queries: 0; deleting DIG: 1.000; m2 = μ[q, d] 1.000 |
| Base 10 A PART / TRAIN retry: MLP3 | final digit readout | m3 = μ[q, d] 1.000 / 1.000 on 2,000 fresh pairs; μ[d mod 10] (ten vectors) 1.000 / 1.000 |
| Base 2 B TRAIN: MLP3 | four-template digit/borrow decoder | m3 = μ[p, v] 1.000, drop 0, in 2/2 solved seeds tested |

In base 2 B PART the trained MLP2 writes the digit straight to the answer logits and bypasses HAND's digit register DIG (Table F2).
At digit queries m2 = μ[q, d], and the final m3 = μ[q, [d < 0]] (1.000 each). Keeping only the HAND-like registers DIG = d mod 2 and
MAJ = [d < 0] at digit queries gives .423, so the learned borrow calibration matters. In base 2 B TRAIN, MLP3 is four vectors indexed
by the phase p = 0 (digit) / 1 (borrow) and v = (1 − p)(d mod 2) + p[d < 0], with no column dependence. These keys use external
arithmetic labels. They are sufficient readouts only: a correct label-conditioned decoder does not explain how earlier layers compute d.

Supported, causal only: TRAIN operand receivers are context-sensitive and do not act as copies. In the base 10 A TRAIN retry, hard
a/b routing keeps .602 / .273 (mean ablation .117 / .031); in base 2 B TRAIN, hard routing of L3H0/H2 keeps .977 / .984 (mean
ablation .281 / .016). Needed populations are 95 pair-selective neurons in base 2 B PART (ablated .051) and 164 difference-selective
neurons in base 2 B TRAIN (.012).

Size was measured by finite-set 1-minimal keep-sets at ≥ .99, which are order-dependent and not global minima, with the pruning set
reused for selection. Base 2 B TRAIN seed 0 keeps 14 heads + 516 neurons and seed 3 keeps 12 + 550 (of 16 + 1,024), with full-grid
exact .967 / .977, so neither is 99%-faithful on the whole domain. HAND under the same search keeps 3 + 5 (1.000; unablated nonzero
wiring 5 + 137).

The whole hypothesis is rejected: resampling every retained head, neuron population and residual coordinate within its claimed role
class gives .938 / .933 (seeds 0 / 3), and same-key individual donors keep only .957 for the base 10 TRAIN readout although its mean
table is exact. A complete endogenous TRAIN algorithm is not verified. A seed census is pending; the phenotypes above come from one or
two examined seeds per cell and are not frequencies.

The whole-role test drew donors from 416 non-training fit pairs and used 416 disjoint evaluation pairs, which had also been used
adaptively to select the keep-sets. Across three donor draws, the recorded means were .93830 (range .93510–.93990) for seed 0 and
.93349 (range .92548–.94231) for seed 3. The .938 / .933 figures above are the earlier display precision, not independent tests.
These are teacher-forced whole-output scores, not evidence from an untouched validation set.

Section 4.2 reports HAND's L2 fetch roles re-found in 9/10 base 10 A retry seeds by attention mass. In the probed seed the receivers
sit in L3 and forcing HAND's hard b route keeps only .273, so attention-mass labels overstate copy behaviour. The base 2 B PART bypass
refines the stage-level f(diff) closed form and does not contradict it.

Table F2 records the direct digit write and the separate frozen borrow relay. Bypassing the digit register at digit queries does not establish that the entire register is globally unused.

<details>
<summary>Technical binary digit bypass and task-labelled readout (Table F2)</summary>

*Table F2. The digit path that training finds in base 2 format B (PART, seed 0). The trained MLP2 writes the digit straight to the answer scores (logits) and bypasses the hand-built DIG register. The mean tables are a local readout on task-derived labels; the stage-level closed form remains a sufficient readout only, not a verified mechanism (Section 5.1).*

| Part or intervention | Route, evidence and boundary |
|---|---|
| Layer 2, heads 0 and 1 (L2H0/L2H1); frozen | Fetch operands for the trained layer-2 feed-forward block MLP2. |
| MLP2; trained | Local replacement m2 = μ[q, d] uses the mean output for query q and task-derived signed difference d. It writes digit scores directly; deleting that write at digit queries gives 0. |
| Hand-built DIG register | Stores d mod 2. Deleting DIG keeps 1.000 in the tested digit path, supporting a bypass at those queries. |
| Frozen layer 3 | L3H0 relays borrowing; MLP3 admits m3 = μ[q, [d < 0]], a mean table on query and the negative-difference flag. This route also contributes to output. |
| Meaning of the test keys | d = a_i − b_i − c_i comes from external arithmetic labels, not an accepted native decoder. The local tables do not verify the endogenous difference/borrow computation. |
| Separate whole-role controls | The failed from-scratch controls reported above remain failures; this one-seed PART bypass does not certify a whole TRAIN mechanism. |

</details>

### 5.4 Relation to the hand-built network and to published work

Most PART responses admit HAND-like difference tables with a wider stage, but binary B can bypass HAND's digit register and requires its learned borrow calibration. Many TRAIN last-MLP responses admit the same key without a recovered HAND chain head. These are sufficient readouts only; the endogenous computation of that key is unresolved. In
[decimal-hex-addition](decimal-hex-addition.md) and [binary-addition](binary-addition.md) the trained stages compute f(sum), so the
column key follows the operation; the position shortcut of formats B and C and its mask control match binary addition. A shared
skeleton for addition and subtraction with a borrow was described by [Quirke et al. (2024)](#references), and causal mediation analysis
of arithmetic in language models by [Stolfo et al. (2023)](#references).

## 6. Limitations and open questions

One training length was used per base; the position-masked PART is the length control. Probes are linear. Networks were trained in
TF32; the fp32 re-score gives base 10 B PART .91–.99 and base 2 C PART 1.0. Binary A cannot separate diff from a + b + c, because the
keys coincide mod 2.

Open questions:

- missing or failed matched shuffle, zero and final-snapshot controls for proposed forms; the existing failures are not treated as unrun tests;
- complete best and final packs per cell (only the best and worst seeds are archived for TRAIN);
- neuron-level necessity and training dynamics;
- the endogenous difference and borrow computation, including the borrow compensation in B PART;
- the causes of the donor failures, and seed frequencies of the probed phenotypes.

## 7. Prediction record

The prediction was recorded in the hand-built design notes before the runs: "per column (a − b − c) mod b", with the binary digit
a XOR b XOR borrow, and the open question "does TRAIN reuse the adder shape (f(a + b + c))?". Difference-readout substitutions supported the prediction in most stages, subject to the 2026-11-07 sufficient-readout audit. Decimal substitutions distinguish the operation-specific keys; binary answer-only parity cannot do so. Nothing was recorded about grokking time or length.

## Appendix A. Weights excerpt and reading

The numbers are computed from the archived PART base 10 stage tensors: the W_in rows on X2, X1 and CY (format B: X1, X2 and the token
`c`), with neurons weighted by ||W_out column|| × ||W_in row||. HAND A's L3 hinges read (a, b, c) = (1, −1, −1) with bias −v (n18/n19:
−7, −8). In the trained PART stages, the output-norm-weighted |cos| of each neuron's (w_a, w_b, w_c) with (1, −1, −1) is .88–.93 in A
and .72–.78 in B; in phase 2 of C, (e, c) gives .63–.68. Examples are A s0 n16 (.04, −.39, −.39), A s2 n7 (−.05, .05, .06) and B s2
n142 (−.20, −.21, −.19) with bias .02.

The alignment statistics support a difference-directed population, not exact equality for every neuron: the displayed cosine ranges are below 1. A hinge is a ReLU response that is zero below a threshold and linear above it ([glossary](../GLOSSARY.md#hand-built-machinery)). B also mixes in sum-like neurons, with cos .72, whose interpretation depends on the borrow token. These weight observations do not lift the sufficient-readout status.

## Appendix B. Detailed evidence

*Table B1. Diagnostics per cell. Each cell lists used heads + MLPs; the best-layer linear-probe R² of a, b (X1/X2 or a_i/b_i), the
borrow c and diff; the residual participation ratio PR at L4; the attention JS of used heads to the matching HAND head; test best
(final); then, after ‖, the causal results.*

| cell | PART | TRAIN best (final) |
|---|---|---|
| b10 A | 4+2; a, b, c 1; PR 3.4–3.6; JS 0; 1 ‖ MLP3 10/10; edges 1/1 (2; mask 3) | 4–16+4; a .74–.90, b .70–.89, c .52–.62; PR 2.1–3.2; JS .09–.35; .72–.98 (.01–.90). Retry 4–8+4; a .75–.88, b .79–.92, c .59–.74; JS .16–.39; 1 (1) ‖ r1 7 solved: MLP3 4, MLP0 3; edges .11/.20 (7). Retry MLP3 8, L3H0 1; edges .11/.20 (8) |
| b10 B | 2–3+2; a, b .83–.86, c .64–.67, diff .85–.91; JS 0–.16; .91–.99 ‖ MLP2 10/10; diff .99/.94, digit .32/.96, borrow_out .80/.99; edges .60/1 (9) | 10–16+4; a .58–.84, c .65–.86, diff .85–.98; JS .08–.19; .94–1 (.01–1) ‖ MLP3 8, MLP0 2; diff 1/1, digit .98/1, borrow_out .92/1; edges .14/.33 (10) |
| b10 C | 3+1; a, b, c, phase 1; JS 0; .97–1 ‖ MLP2 10/10; X2 .03/.36; edges 1/1 (10) | 3–16+4; a .61–.83, b .55–.73, c .65–.89; PR 2.5–4.7; JS .08–.39; .48–1 (0–.97). Retry 3–15+3–4; a .70–.81, c .80–.96; JS .10–.34; 1 (1) ‖ r1 8 solved: MLP3 7, MLP0 1; X2 .04/.83; edges 0/0 (8). Retry MLP3 10/10; X2 .06/.97; edges 0/0 (10) |
| b2 A | 4+2; a, b, c 1; PR 5.4; JS 0; 1 ‖ MLP3 10/10; edges 1/1 (10) | 11–16+4; a .39–.50, b .58–.70, c .28–.38; JS .04–.14; .39–.95. Retry 6–16+4; a .41–.68, b .61–.75, c .32–.48; JS .08–.31; 1 (1) ‖ r1 1 solved: MLP0; edges .20/.50 (1). Retry MLP0 9, MLP3 1; edges .20/.33 (10) |
| b2 B | 3+2; a, b .79–.84, c .72–.83, diff .89–1; JS .15–.17; 1 ‖ MLP2 10/10; diff .93/.95, digit 1/1, borrow_out .86/1; edges .80/1 (10) | 5–13+4; a .47–.72, c .58–.85, diff .84–.94; JS .05–.19; .79–1 ‖ 9 solved: MLP3 7, MLP1 1; diff .99/1, digit .75/1, borrow_out 1/1; edges .20/.44 (8) |
| b2 C | 3+1; a, b, c, phase 1; JS 0; 1 ‖ MLP2 10/10; X2 0/1; edges 1/1 (10) | run 1 6–16+4; c .50–.64; .54–.95. Retry 3–9+3–4; a .56–.75, c .51–.89; JS .18–.38; 1 (1) ‖ r1 4 solved: MLP3 4; edges 0/0 (4). Retry MLP3 9, MLP1 1; X2 0/1; edges 0/0 (8) |

Key: b10 / b2 = base 10 / base 2; r1 = run 1. After ‖, medians over solved networks (exact ≥ .9): the component with the largest
patching drop (number of networks); var inter/DAS = linear interchange / DAS flip at the best layer; edges = EAP-IG Jaccard/hand_recall
against HAND over trusted networks (spearman_exact ≥ .8), with the count of
trusted networks in parentheses. HAND reference values: b10 B diff 1/.37, digit 1/.17, borrow_out 1/1; b10 C X2 0/.01; b2 B diff
.31/.31; b2 C X2 0/1; format A has no answer-fixing variable. Some test accuracies in this table (b10 C TRAIN final 0–.97, b2 C TRAIN
run 1 .54–.95, b2 A TRAIN .39–.95, b10 B PART .91–.99) differ from the values in Table 4 (0–.98, .55–.95, .40–.95, .995–1); the
records do not say why, and both are kept as recorded. The b10 B PART value equals the fp32 re-score given in Section 6.

## References

- Quirke, P., Neo, C., & Barez, F. (2024). Understanding addition and subtraction in transformers. *arXiv preprint*.
  https://doi.org/10.48550/arxiv.2402.02619 (shared addition/subtraction skeleton, borrow)
- Stolfo, A., Belinkov, Y., & Sachan, M. (2023). A mechanistic interpretation of arithmetic reasoning in language models using causal
  mediation analysis. *arXiv preprint*. https://doi.org/10.48550/arxiv.2305.15054 (causal mediation analysis of arithmetic)
