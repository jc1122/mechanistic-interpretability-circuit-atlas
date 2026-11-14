# Quadratic roots

The quadratic x²−3x+2 reaches zero at its roots, 1 and 2. This task requested integer versions of the formula's results and a label for the kind of roots. Every end-to-end answer or written-arithmetic experiment failed, including the completed ten-run answer-only follow-up.

A different approach supplied the program, storage addresses and routing. The network only learned a [NAND](../GLOSSARY.md#standard-machine-learning-terms) update, which returns 0 only when both input bits are 1. With that help, 10/10 runs learned the local rule after 500 steps. The task is partly solved.

## The circuit at a glance

Figure 3 shows the supplied-program executor, not a learned quadratic solver ([mechanism and verification](#5-mechanism-and-verification)).

<img src="../figures/circuits/quadratic-roots.svg" width="640" alt="Supplied addresses feed a learned local NAND vote. One upward residual stream has separate read/write branches for Read first bit (pointer, then value); Read other bit (pointer, then value); Vote together (trained neurons). Input: supplied program; two addressed bits are 1; New register bit: 0. Task values are illustrative; dashed boxes and branches are not verified.">

*Figure 3. The ten final partly trained executor seeds match the hand-built local NAND decision through a collective neuron group, verified by substitution and complement resampling on sampled checks, with program, addressing and state copying supplied rather than a learned quadratic solver.*

| Fact | Value |
|---|---|
| Task | For x² + bx + c, output D = b² − 4c, s = ⌊√max(D, 0)⌋, task-defined integer root expressions and a flag (negative / integer / other real). |
| Input → output example | b = −3, c = 2 → D = 1, s = 1, roots 1 and 2, flag integer |
| Model | GPT-NeoX-shaped, ReLU, no LayerNorm, no positions; parent formats 4 layers, d_model 768; executor 2 layers, d_model 128. PART trains one MLP stage (layer 1 in the executor); TRAIN learns all weights. |
| Rungs and seeds | HAND A/S/T/S_exec exact at n 3–5; HAND T fails at n = 6. Executor PART/TRAIN and answer-only n = 6 finals: 10 seeds each; earlier parent runs 1–4: 3–10 seeds per cell. |
| Closed form status | Verified (executor PART only: 1,792/1,792 substituted packets per seed; complement 1.0 versus base 1.0 on 512 rows with 3 draws, 10/10 seeds). None for trained parent cells. |
| Status | Partly solved (scaffolded executor PART 10/10; executor TRAIN and answer-only n = 6 TRAIN 0/10; earlier parent cells 0/n); final analysis 2026-11-12. |
| Weights | [Hugging Face: quadratic-roots](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/quadratic-roots) (weights, splits and recorded test archive; published revision `e417b9b92a75`). |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

Training the whole network at twice the width succeeded in 0/10 runs: it never learned to fetch the addressed bits.

The local [formula is verified](../GLOSSARY.md#verifying-a-circuit): a one-neuron replacement kept 1,792/1,792 held-out packets right per run, and swapping the other parts' signals between examples also passed in every run. This verifies a local rule within supplied computation, not a learned quadratic solver.

Read next: [Results](#4-what-training-finds) and [verification](#5-mechanism-and-verification), then [composed arithmetic registers](compose-arithmetic-registers.md) for another supplied program, or [long division and square root](long-division-square-root.md) for arithmetic substeps.

## 1. Task and data

The coefficients are integers with |b|, |c| ≤ R, written in binary sign/magnitude: n = 5 bits gives R 15 (961 pairs) or R 31 (3,969),
and n = 6 gives R 63 (16,129). The prompt is `BOS sb |b| sc |c| =`, with coefficient-sign tokens sb/sc. The answer is `f sr- |r-| sr+ |r+|`, with flag f and root-sign tokens sr−/sr+.

For example, b = −3 and c = 2 give D = 1, s = 1, roots 1 and 2 and flag integer. Outputs are specified integers: D = b² − 4c, s = ⌊√max(D, 0)⌋, the root expressions r∓ = ⌊(−b ∓ s)/2⌋, and a flag (negative / integer / other real). D is the discriminant, whose sign and square status distinguish the root types. The floor sign means round down; flag N/I/R means negative discriminant, integer roots or other real roots. Because the task first takes an integer square root, its root expressions are defined outputs, not exact irrational roots or a claim to floor the true roots in every case.

An output is exact when every output token is correct, teacher
forced, that is, each token is scored given the correct previous tokens ([glossary](../GLOSSARY.md#task-formats)). Four formats were
studied (Table 1); formats `A`, `S` and `T` are called the parent formats, and `S_exec` the executor.

*Table 1. Output formats (output length in tokens at n 5).*

| Format | Output |
|---|---|
| `A` (answer only) | the answer (29 tokens) |
| `S` (sub-result) | D and s first, then the answer (49) |
| `T` (full trace) | every arithmetic step (170) |
| `S_exec` (executor scratchpad: intermediate working before the answer; [glossary](../GLOSSARY.md#task-formats)) | the whole computation as an input-independent program of Boolean NAND updates on a register file, one update per step |

In S_exec each packet supplies the shuffled register bank and the gate's two source addresses, and the network predicts the new bit.
Training used 12,903/16,129 R 63 coefficient pairs, an 80/20 pair split (held-out pairs per seed in `split_S_exec`), fresh reachable register states every
batch (128), mixed widths 3–6, shuffled memory and renamed addresses; the program, the source and destination addresses and the state
copying are supplied. The parent grids train on 480/961 (R 15) and 1,984/3,969 (R 31) pairs. Publication metadata records held-out counts of 481 (R 15), 1,985 (R 31) and 8,065 (R 63), with 8,064 training pairs at R 63.

S_exec was tested four ways. Local is next-bit exact on 1,024 held-out packets. Trace is all updates of a whole program for held-out
pairs (8 per cell at n 5/R 15, n 5/R 31 and n 6/R 63). Length is n 7/8. Rollout feeds the network's own predictions back through a
whole program and checks the answer with a verifier.

## 2. Model

The shared conventions are described in [METHODS.md](../METHODS.md). The three [rungs](../GLOSSARY.md#the-three-rungs) are HAND
(every weight set by formula), PART (frozen hand-built parts with one trained stage) and TRAIN (all weights trained from scratch).

### 2.1 Architecture

All networks are GPT-NeoX-shaped decoders, with attention heads reading earlier positions and feed-forward MLP blocks ([glossary](../GLOSSARY.md#standard-machine-learning-terms)). They use ReLU, no LayerNorm, no positional encoding and untied embeddings. The parent A/S/T networks have
4 layers, d_model 768, 12 heads (d_head 64), MLP 512, vocabulary 10 and context 512 (n 6 A: d_model 1,024, 16 heads, MLP 1,024). The
S_exec networks have 2 layers, d_model 128, 2 heads × 64, MLP 128, vocabulary 1,029 (register and address tokens) and context 512. TRAIN
width 2 is d_model 256, 4 heads and MLP 256.

### 2.2 Frozen hand-built network and trained part

In the parent formats PART trains one stage MLP on the frozen HAND (A: layer 2; S/T: layer 1). In S_exec PART trains the layer-1 MLP
(W_in [128, 128] masked to the A/B residual columns, W_out [128, 128], both biases). TRAIN trains all weights.

### 2.3 Training

The loss was cross-entropy on output tokens. AdamW used learning rate 1e-3, weight decay .1, betas (.9, .98), warmup 1,000, cosine decay and clip 1. Finals used
10 seeds (seed = initialisation + split), a 40k budget, a 5k plateau stop with a minimum of 30k and checkpoints every 2,500 steps. The later source entry gives a shared evaluation interval of 500; the 2026-11-06 publication metadata separately records 250 for parents and 500 for the executor. These setup records are retained at their own dates; no per-run reconciliation is reported here. The
original parent run 1 used a constant learning rate and weight decay 1. Tuned retries used warmup and cosine decay: T R15 PART run 2 kept weight decay 1, while run 3 used .1. PART ran in fp32; TRAIN used TF32 updates with
fp32 evaluations.

The completed answer-only n = 6/R 63 result is a legacy recovery, not an independent from-scratch run. After the original run was lost at 23.5k, recovery used the 22.5k snapshot on the same splits, fresh optimiser moments, learning rate 1e-4, warmup 1000 and a cosine tail over the lost 17.5k steps. It plateaued after 13.5k at test .929–.944, 0/10 exact. This historical continuation is not an adopted retry recipe.

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/quadratic-roots](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/quadratic-roots).
The recorded result keys are `test` for whole-output exact match and `local` for next-bit exactness. The source names `launch_final.sh`, parent and executor `test_hand.py` harnesses and `subst_check.py`; these are recorded artifact names, not runnable commands supplied in this repository. Public reproduction code is pending.

### 2.5 Archive

Published revision `e417b9b92a75` of the Hugging Face folder holds `hand_{A,S,T}`, `hand_T_compact` and `hand_S_exec`; `part_S_exec_final_s0…9` (the 4 MLP tensors, to be
inserted into `hand_S_exec`; best and final are byte-identical); `train_S_exec_final_s0…2` (full state dicts; the best is the step-0
initialisation, which was not kept); `part_T_R15_{r1,r2}_s0…9` (early parent PART); `split_S_exec`; `config.json`; and `manifest.json`
(recorded and fp32 re-scored accuracy, sha256). Most parent-format networks (A/S TRAIN, T TRAIN, S/T PART run 2 and run 3) were lost
with rented hosts, so for those cells the run-time diagnostics in Section 4 are the evidence and the networks are not archived. The later public-archive verification checked 42 files with no missing file or hash mismatch; that file count is not a census of independent networks.

## 3. Hand-built solution

In the parent networks, a layer-0 frame head reads 1/(1 + position) from BOS and an MLP turns it into a position one-hot; table heads
read a written number as a weighted token sum. Bit k of V is [V − W ≥ 2ᵏ], computed as ReLU(2u + 1.5) − ReLU(2u + .5) and gated by −G
per failed condition. V is the current arithmetic value, W is the value of higher bits already written, u = V − W − 2ᵏ and G is a fixed gating penalty. Format A computes b² in layer 1, isqrt [D ≥ m²] in layer 2 and the roots in layer 3; S computes b² and the isqrt
of the written |D| in layer 1; T puts all rules in layer 1 on token values. Compact T moves the bit thresholds into query-position
weights, which shrinks the arithmetic to 62 neurons for every n, against roughly 6R for answer-only A.

The executor (Table F1) uses address codes Cⱼ(a) = 2·bitⱼ(a) − 1. Layer-0 heads find the A and B pointer tokens, and layer-1 heads
fetch registers a and b by binary address match, where a wrong code bit costs 40 logits. One neuron, AND = ReLU(a + b − 1), writes Y,
and the readout (other −10, bit0 10·Y, bit1 5) emits the complement, NAND. The executor is exact at n 3–8 with margin 5. Table 2 gives
the sizes.

*Table 2. Size of the hand-built networks: designed heads (h) and neurons / used heads + neurons.*

| n (R) | A designed / used | S | T | T compact | S_exec |
|---|---|---|---|---|---|
| 5 (31) | 6h 217 / 7h + 194 | 9h 321 / 10h + 260 | 9h 322 / 10h + 187 | 9h 62 / 10h + 44 | 4h + 1 |
| 6 (63) | 6h 385 (exact) | 9h 507 | not exact | untested | 4h + 1 (1,806 → 2,296 steps) |

Used = ablation minima (heads + neurons, frame head included, frame neurons excluded) on 200–300 rows. HAND A, S, T and S_exec are
exact at n 3–5; the source also reports A/S exact on all pairs through n 6. T is not exact at n 6, and that cell is closed. This HAND design failed in both fp32 and fp64 and was not repaired; it is not a trained-network failure or a precision-only effect.

Table F1 records the local executor inside the supplied program. Addressing and register retrieval are frozen; the learned part is a bit operation, not an end-to-end quadratic solver.

<details>
<summary>Technical supplied addresses and local NAND executor (Table F1)</summary>

*Table F1. The hand-built register executor (format S_exec). Frozen heads locate and fetch the two addressed registers; one AND neuron and a fixed readout write their NAND. The layer-1 feed-forward block (MLP) is the part that PART trains.*

| Part and training status | Input, rule and destination |
|---|---|
| Supplied packet | Contains a shuffled register bank and source addresses A and B from the supplied program. |
| Layer-0 heads; frozen | Find the A and B pointer tokens. |
| Layer-1 heads; frozen | Fetch register bits a and b by binary address match. |
| Layer-1 MLP; trained in PART | The reference uses one AND neuron, Y = ReLU(a + b − 1), active only when both input bits are 1. |
| Fixed readout; frozen | Scores other tokens −10, bit 0 as 10Y and bit 1 as 5; this writes NAND(a, b), which is 0 only for two 1 inputs. |

</details>

## 4. What training finds

The final interpretation was refreshed on 2026-11-12. Table 3 summarises the cells; earlier parent runs 1–4 used 3–10 seeds per cell. Arrows give the best and then the final test accuracy.

*Table 3. Results per cell. The three final cells use 10 seeds; earlier parent screens/retries use 3–10, with medians where marked. Parent full-test sets contain 481, 1,985 and 8,065 pairs at R 15, 31 and 63 respectively; these are separate from the 200-row diagnostics. Exact match means every output token correct under teacher forcing.*

| Cell | Format | Best / final test | Train | Stop | Verdict |
|---|---|---|---|---|---|
| S_exec PART final | scratchpad | trace 1 / 1 (10/10); n 7/8 and rollouts R 15/31/63 1 | 1 | solved 500 | solved |
| S_exec TRAIN width 2 final | scratchpad | trace 0 (best = step 0); local .58–.62 | .58–.62 | plateau 30k | constant output |
| A n 6 R 63 TRAIN, interrupted original | answer only | .91–.93 at 22.5k | .97–1 | lost at 23.5k | Historical partial run. |
| A n 6 R 63 TRAIN, completed decay | answer only | .929–.944 / same (from 22.5k snapshot) | .97–1 at recovery snapshot | plateau 13.5k | 0/10, memorised late lookup. |
| A PART R 15 r1 / r2; R 31 r2 | answer only | .34 → .17 / .59 → .58; .63 → .62 (medians) | | | 0/10 each |
| A TRAIN r2 R 15 / R 31 | answer only | .73 → .70 / .88 → .87 | 1 | plateau | memorised, 0/10 |
| S R 31 PART / TRAIN r2 | sub-result | .06 / .86 best | .09–.17 / 1 | 60k / 40k | underfit / memorised |
| T R 15 PART r2 / r3; T R 31 PART r2 | trace | .63 → .61 / .85 → .83; .65 → .62 | 1; 1; .71 | budget | memorised; underfit |
| T R 31 TRAIN r2 | trace | .80 | 1 | budget | memorised |

Key: r1, r2, r3 = run 1 to run 3 of a cell. The original 40k A n = 6 run was interrupted at 23.5k with test .91–.93. The later cosine-decay completion from the 22.5k snapshot plateaued after 13.5k and closed at .929–.944, 0/10. This is a labelled continuation of the legacy interrupted run, not an independent from-scratch replication.

### 4.1 Executor

PART learns the executor's single rule in 500 steps on 10/10 seeds, and the rule transfers to n 7/8. PART keeps it exact at widths 7 and 8 and in its own
predicted-state rollouts. The S_exec PART networks form one functional family, an addressed AND-hinge bank with a NAND readout, in
10/10 seeds. Its native-bank partition is 7/10 OR-pool-sufficient and 3/10 mixed OR/affine/complement banks, a causal sub-split rather than distinct algorithms. Aligned interpolation loss barriers .001–.007 (naive .48–.70) support one basin up to neuron permutation. S_exec TRAIN gives a constant majority output in 10/10: it predicts the majority bit (P(NAND = 1) ≈ .61) everywhere,
because its heads never attend to the addressed registers (JS .65), so a and b are never read.

### 4.2 Parent formats

No trained answer-only or parent-trace cell solves (0/n in earlier 3–10-seed cells, and 0/10 in the completed answer-only final); every trained parent cell memorises or underfits. HAND's exact constructed solutions are separate and must not be counted as failed training cells. A
PART stage here is one MLP on frozen parts that must place unit-resolution thresholds on scalars up to ~2¹¹. There is no solved delayed-grokking transition in the
parent cells: A TRAIN run 2 crosses 50% test at a median of 1,750 steps and 99% train at 6,125 / 10,375
([grokking](../GLOSSARY.md#counting-results) is test accuracy rising long after training accuracy is perfect).

The failures are explained from diagnostics (Table B1). S R 31 PART decodes D (R² .96), but bits 2⁵…2⁰ fail (.47–.77, where the first
errors fall): these are unit thresholds on |D| ≤ 1,085. T PART fails on inexact high-value, borrow and late-sqrt thresholds and does
not miss D. In A TRAIN, D is decodable late and non-linearly (DAS rescues D in 3/6 audited seeds), and there is no early D-to-isqrt
path.

### 4.3 Executor causal checks

In S_exec PART, the query-mean ablation of every head lowers accuracy to .63–.88; the MLP1 patching drop and recovery are both 1; and
interchange on the sum / NAND variables flips .93–1 / .57–.84 (answer-fixing n 25–38, the number of answer-fixing pairs tested). A and B have no answer-fixing pairs, which
means untested, not unused. Interchange and patching are defined in the [glossary](../GLOSSARY.md#analysis-measures).

## 5. Mechanism and verification

### 5.1 Executor closed form

The native bank of S_exec PART is hⱼ = ReLU(cⱼ + aⱼA + bⱼB), O = 1 + β_O + Σuⱼhⱼ and Y = β_Y + Σvⱼhⱼ, with logits (−10·O, 10·Y, 5·O)
for other / bit0 / bit1; these are direct tensor slices, not a fit. Overwriting all trained MLP tensors with h = ReLU(A + B − 1)
writing only Y (frozen readout unchanged) takes the fresh held-out packets per seed from 1,792/1,792 (native) to 1,792/1,792, with
agreement 1, on 10/10 seeds (17,920 unique packets; bar ≥ .99 × native). The packets balance the four Boolean cases at register counts 3–256 and
add equal-address controls. In predicted-state execution the native and substituted networks agree at every update (507,680 unique
updates; 2–8 held-out pairs per width/range at n 5–6, 4 boundary/sign/zero pairs at n 7/8). Best and final weights are identical, so
this is one set of 10 networks, not 20.

The substitution certifies the decision on the tested distribution, not logit equality. A later complement test used 512 rows and 3 draws per seed: resampling everything outside HAND's 9 edges (of 26) keeps exact 1.0 = base 1.0 in 10/10 seeds, above the .90 × base acceptance share. Resampling those 9 edges instead gives .49–.54, and random same-size sets .52–.55; full_eq_base is true and empty_eq_source is 1.0. HAND-free discovery keeps the same 9 edges in 10/10, with held-out faith 1.0. The formula is therefore Verified by substitution and complement resampling ([glossary](../GLOSSARY.md#verifying-a-circuit)). This places the trained stage inside frozen routing and supplied microcode; it does not show that training found the routing.

The later 512-packet substitution plus n 6/7/8 rollout check also has native_min 1.0 and substituted_min 1.0 in 20/20 best/final snapshots, which are byte-identical pairs of 10 networks. The native bank's collective decision is NAND-equivalent; an individual native hinge need not compute AND.

Table F2 records the collective native decision and the smaller replacement that preserves it on the measured checks. An individual trained neuron need not be an AND gate.

<details>
<summary>Technical collective neuron bank and sampled local verification (Table F2)</summary>

*Table F2. The executor stage is Verified on the measured checks in 10/10 seeds: one-neuron substitution preserves 1,792/1,792 packets per seed and the 9-edge complement keeps 1.0. Only the bank is trained. The native decision is collective; the individual hinges can respond to either single operand, unlike HAND's replacement AND neuron. This verifies a local rule with the program and routing supplied, not an end-to-end quadratic solver.*

| Part or check | Response and scope |
|---|---|
| Layer-1 addressed readers; frozen | Supply fetched bits a and b to the learned bank. |
| Learned bank; trained | Uses 28–37 of 128 hinges, threshold-response neurons with additive operand contributions. Individual hinges can respond to a single operand. |
| Bank-level sum | Combines responses into Y and O channels; the NAND decision belongs to this collective, not one native AND neuron. |
| Readout; frozen | Scores other tokens −10O, bit 0 as 10Y and bit 1 as 5O, producing NAND(a, b). |
| Sampled local checks | One-neuron replacement preserves 1,792/1,792 packets per seed; the 9-edge complement keeps 1.0 in 10/10 seeds. This does not verify learned program generation, addressing or a complete quadratic solver. |

</details>

### 5.2 Parent cells

No complete form was accepted for the parent cells. Earlier contextual-bank substitutions on seven parent finals only preserved their
errors, and those originals are no longer available for fresh checks.

### 5.3 Relation to published work

Answer-only and parent-trace stages need unit-resolution thresholds on large scalars. The executor scratchpad turns each step into one
bounded-bit rule, which a frozen-parts MLP learns at once (500 steps, 10/10) and a from-scratch network of 2× width does not, because it never learns the
addressing. Scratchpads as supervised intermediate computation follow [Nye et al. (2021)](#references); training curves alone are not
circuit evidence ([Nanda et al., 2023](#references)). Because the program, routing and state copying are supplied, the results support
neither a size-crossover claim nor a claim of a from-scratch quadratic solver. Earlier numerical-root and fuzzy-quadratic work studied different tasks: [Buckley and Eslami (1997)](#references), [Buckley et al. (2002)](#references), [Huang and Chi (2001a, 2001b)](#references), [Huang et al. (2003a, 2003b, 2004, 2005)](#references), [Huang (2004)](#references), [Epitropakis and Vrahatis (2005)](#references), [Mourrain et al. (2006)](#references), [Dilanian et al. (2006)](#references), [Zhang et al. (2008)](#references), [Zhu et al. (2008)](#references), [Zhang and Zeng (2009)](#references) and [Das and Seal (2012)](#references).

The retained related-task citations also cover sequence-model polynomial roots ([Saxton et al., 2019](#references)), approximate roots ([Freitas et al., 2021](#references); [Bukhsh et al., 2022](#references); [Wang et al., 2026](#references)), worked mathematical solutions ([Hendrycks et al., 2021](#references)), program synthesis ([Palermo et al., 2021](#references)), and root-type classification rather than extraction ([Thomas and Bani-Yaghoub, 2026](#references)). [Qiu et al. (2026)](#references) concern integer arithmetic and [Nunley (2026)](#references) token operators; none supplies evidence for this quadratic executor's learned mechanism.

## 6. Limitations and open questions

Mean pruning is ordered and calibrated on 256 balanced packets and validated on 1,792 per seed, so it is not a global minimum. Head
ablations are query-mean, not token-global minima. Executor checks are sampled traces and rollouts (not all 16,129 traces), not an
exhaustive sweep of the coefficient grid. Earlier parent PART range-generalisation tests solved 0/10, unlike the executor's sampled width transfer. The full-trace HAND design at n = 6 failed in both fp32 and fp64 and was not repaired; this construction failure is separate from the trained-parent failures. Parent TRAIN curves used TF32 without full-test fp32 confirmation.

Missing:

- most archived parent TRAIN and run 2 / run 3 networks (lost);
- public reproduction code.

## 7. Prediction record

Predictions were recorded before the runs (Table 4).

*Table 4. Prediction record.*

| Prediction | Outcome |
|---|---|
| PART A 10/10 | refuted (best .05–.31; r2 .58–.62) |
| PART S ≥ 7/10 | refuted (r2 .03–.06) |
| PART T 10/10 | refuted (r2 .61, r3 .83, memorised) |
| TRAIN A ≤ 2/10 and T ≤ 3/10 | confirmed (0/10) |
| HAND T ~62 neurons vs A ~6R | confirmed for constructed sizes, not a trained crossover |
| S_exec PART 3/3 within 2.5k | confirmed (500) |
| S_exec TRAIN ≥ 2/3 at local .999 | refuted (0/3, then 0/10) |

## Appendix A. Weights excerpt and reading

The parent HAND constants are BIG 40 (table-attention score offset), U 10 (readout scale) and C_POS 128 (position-signal scale); the bit-threshold pair is (2, 1.5) / (2, .5). The executor HAND stage row (A, B, bias) is (1, 1, −1), writing Y with weight 1; all other stage entries are zero. For the trained PART, Table A1
gives per seed the neuron with the largest contribution wΔ = 5u − 10v to the bit1 − bit0 margin (u and v are the neuron's output weights into O and Y, Section 5.1) (`part_S_exec_final_s<seed>`; W_in
columns A = 37, B = 38).

*Table A1. Strongest margin neuron per seed and the number of hinges kept by mean pruning.*

| Seed / neuron | bias c | a | b | wΔ | mean-pruned hinges |
|---|---:|---:|---:|---:|---:|
| 0 / 25 | −.0044 | .1240 | .0926 | −1.346 | 33 |
| 1 / 56 | −.0128 | .0842 | .0894 | −1.509 | 35 |
| 2 / 71 | −.0099 | .1286 | .0833 | −1.283 | 37 |
| 3 / 42 | .0034 | .1217 | .1214 | −1.160 | 31 |
| 6 / 110 | −.0061 | .1198 | .1646 | −1.050 | 31 |
| 9 / 45 | −.0086 | .1146 | .1118 | −1.365 | 29 |

Reading: seed 0's neuron responds (00, 01, 10, 11) = (0, .088, .120, .212) and writes ONE / Y with −.056 / .107. Positive operand
weights suppress the bit1 margin most strongly at 11, but the nonzero 01 and 10 responses mean this individual neuron is not an AND gate. About 30 hinges collectively implement the NAND decision that HAND implements with one AND neuron and a complementing readout. Seeds 3/4 have small positive biases, so not every hinge is off at 00. The decision
is a bank-level sum. The earlier rounded seed-0 excerpt was A .124, B .093, bias −.004, ONE −.056, Y .107 and margin −1.35; it also reported top-hinge operand weights .04–.16 and margins −1.05..−1.60. These are native additive responses and outgoing margin contributions, not individual Boolean-gate identifications.

## Appendix B. Diagnostics per cell

*Table B1. Earlier diagnostics, recorded on 2026-11-07. S_exec on 128 held-out packets per seed.*

| Cell | Used | Best probe R² (layer) | Rank (HAND) | JS | Best / final |
|---|---|---|---|---|---|
| S_exec PART (10) | 4 heads + 28–37 of 128 hinges (HAND 4 + 1) | a, b, sum 1; NAND .994–.999 (2) | stage input 2 (1); residual 17–18 (17–18) | 0 | 1 / 1 |
| S_exec TRAIN width 2 (10) | none (every ablation within ±.01) | a, b ≤ .011 | residual 2–4 (17–18) | .65 (uniform = .69) | 0 / local .58–.62 |
| S R 31 PART r2 (s0–2) | stage only (patch drop 1) | D .96, s .97, b² .88 (2) | stage W_in 41 | 0 (frozen) | .06 / .05 |
| A n 6 R 63 at 22.5k (s0–2) | 10–12 heads + MLPs | s .94–.95, b² .93–.94, D .92–.94 | W_in 362–634 | n/a | .945–.955 (200 rows) |
| A R 31 TRAIN r2 (3) | 14 heads + 4 MLPs | b² .96, D .95, s .94 (3) | W_in 300 | .44–.49 | .88 / .87 |
| T R 15 PART r2 (s0–2) | 9 heads + 1 MLP block | D ≈ .94 (2) | stage input 80 (51) | 0 | .63 / .61 |

Key: used = components with a mean-ablation drop; R² = best linear probe (layer); rank = effective rank at 99% energy; JS = attention
divergence to the HAND head; s0–2 = seeds 0 to 2.

*Table B2. Refreshed final-source diagnostic summary. This retains the final source's reporting alongside Table B1's earlier diagnostic summary; differing probe/rank values are not silently overwritten.*

| Cell | Used heads and neurons | Best probe R² | Rank and attention comparison | Best / final diagnostic exact |
|---|---|---|---|---|
| S_exec PART, 10 seeds | 4 heads + 28–37 of 128 hinges | a, b, sum 1; NAND .998 at L2 | MLP1 2, L2 17; HAND 1 / 18; JS 0 | 1 / 1 |
| S_exec TRAIN width 2, 10 seeds | None at the individual-ablation screen (±.01). | a, b, sum, NAND ≤ 0 | L1–L2 2–3; JS .65 (ln 2 .69) | 0 / 0 |
| A n6 R63 completed decay, 10 seeds | 5–9 heads + 4 MLPs | s .95–.97, D .92–.96, b² .93–.96 at L3 | MLP1 W_in 582, seed 0; HAND overlap approximately 0; JS .38 at L0 | .945–.975 / .93–.98 (200 rows) |
| S R31 PART run 2, 3 seeds | Stage only, patch drop 1.0 | D .97, s .97, b² .88 at L2 | W_in 41; HAND overlap .03; JS 0 | .06 / .025–.035 |

The earlier parent-result summary gives S R31 PART train .13, compared with the wider .09–.17 band in Table 3. The executor's original HAND margin was reported as 5.0. The source's approximate design crossover n* ≈ 3 concerns hand-built sizes; a trained size-crossover claim is not established by the supplied-program experiment. The parent half-grid split is .5 and the final A R63 uses .5 of the coefficient grid. The source also reported scalar-threshold range |D| ≤ 1085 and a HAND-free failure graph of 38–55% of 1935 edges; neither is a new success criterion.

The completed A n = 6 decay diagnostics use 200 rows: best exact .945–.975 and final .93–.98, 5–9 heads plus all 4 MLPs, probes at L3 s .95–.97, D .92–.96 and b² .93–.96. HAND-free discovery retains 736–1073 of 1935 edges (38–55%), with edge overlap .29–.42 against HAND's discovered graph. These describe a failed dense late lookup, not a solved circuit type. The updated S R31 PART source reports .06 best and .025–.035 final; Table B1 retains the earlier diagnostic cohort's .06 / .05 rather than silently choosing one.

## References

Titles and venues of the inherited root-finding references were checked against public bibliographic records. Where older registry records omit dates or do not separate author names, the source attribution and year are retained. These works concern related tasks, not verification of this executor; checking their metadata is not a full-text review.

- Buckley, J. J., & Eslami, E. (1997). Neural net solutions to fuzzy problems: The quadratic equation. *Fuzzy Sets and Systems*. https://doi.org/10.1016/S0165-0114(95)00412-2 (Journal article; fuzzy quadratic equations.)
- Buckley, J. J., Feuring, T., & Hayashi, Y. (2002). Solving fuzzy equations using evolutionary algorithms and neural nets. *Soft Computing*. https://doi.org/10.1007/s005000100147 (Journal article; fuzzy-equation methods.)
- Bukhsh, M., Ali, M. S., Ashraf, M. U., Alsubhi, K., & Chen, W. (2022). An Interpretation of Long Short-Term Memory Recurrent Neural Network for Approximating Roots of Polynomials. *IEEE Access*. https://doi.org/10.1109/ACCESS.2022.3157306 (Journal article; approximate polynomial roots.)
- Das, M., & Seal, P. (2012). Polynomial real roots finding using Feed forward Neural Network: A simple approach. *2012 NATIONAL CONFERENCE ON COMPUTING AND COMMUNICATION SYSTEMS*. https://doi.org/10.1109/NCCCS.2012.6413004 (Conference paper; numerical root finding.)
- Dilanian, R. A., Nikulin, A. Y., Darahanau, A. V., Hester, J., & Zaumseil, P. (2006). Application of phase-retrieval x-ray diffractometry to carbon doped SiGe(C)∕Si(C) superlattice structures. II. High resolution reconstruction using neural network root finder technique. *Journal of Applied Physics*. https://doi.org/10.1063/1.2201447 (Journal article; numerical root finding.)
- Epitropakis, M. G., & Vrahatis, M. N. (2005). Root finding and approximation approaches through neural networks. *ACM SIGSAM Bulletin*. https://doi.org/10.1145/1140378.1140382 (Journal article; numerical approximation.)
- Freitas, D., Guerreiro Lopes, L., & Morgado-Dias, F. (2021). A Neural Network-Based Approach for Approximating Arbitrary Roots of Polynomials. *Mathematics*. https://doi.org/10.3390/math9040317 (Journal article; approximate roots.)
- Hendrycks, D., Burns, C., Kadavath, S., et al. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. *arXiv preprint*. https://doi.org/10.48550/arXiv.2103.03874 (Worked mathematics problems.)
- Huang, D.-S., & Chi, Z. (2001a). Finding complex roots of polynomials by feedforward neural networks. *IJCNN'01. International Joint Conference on Neural Networks. Proceedings (Cat. No.01CH37222)*. https://doi.org/10.1109/IJCNN.2001.1016716 (Conference paper; complex-root methods.)
- Huang, D.-S., & Chi, Z. (2001b). Neural networks with problem decomposition for finding real roots of polynomials. *IJCNN'01. International Joint Conference on Neural Networks. Proceedings (Cat. No.01CH37222)*. https://doi.org/10.1109/IJCNN.2001.1016718 (Conference paper; real-root decomposition.)
- Huang, D.-S., Ip, H. H. S., Chi, Z., & Wong, H. S. (2003a). Dilation method for finding close roots of polynomials based on constrained learning neural networks. *Physics Letters A*. https://doi.org/10.1016/S0375-9601(03)00216-0 (Journal article; numerical root separation.)
- Huang et al. (2003b). Finding the ordered roots of arbitrary polynomials using constrained partitioning neural networks. *Proceedings of the International Joint Conference on Neural Networks, 2003.* https://doi.org/10.1109/IJCNN.2003.1223844 (Conference paper; numerical root finding; source author attribution retained.)
- Huang, D.-S. (2004). A Constructive Approach for Finding Arbitrary Roots of Polynomials by Neural Networks. *IEEE Transactions on Neural Networks*. https://doi.org/10.1109/TNN.2004.824424 (Journal article; constructive numerical methods.)
- Huang, D.-S., Ip, H. H. S., & Chi, Z. (2004). A Neural Root Finder of Polynomials Based on Root Moments. *Neural Computation*. https://doi.org/10.1162/089976604774201668 (Journal article; numerical root finding.)
- Huang, D.-S., Chi, Z., & Siu, W.-C. (2005). A case study for constrained learning neural root finders. *Applied Mathematics and Computation*. https://doi.org/10.1016/j.amc.2004.04.070 (Journal article; numerical root finding.)
- Mourrain, B., Pavlidis, N. G., Tasoulis, D. K., & Vrahatis, M. N. (2006). Determining the number of real roots of polynomials through neural networks. *Computers & Mathematics with Applications*. https://doi.org/10.1016/j.camwa.2005.07.012 (Journal article; root counting.)
- Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *International Conference on Learning Representations (ICLR 2023)*; arXiv preprint. https://doi.org/10.48550/arXiv.2301.05217 ; https://arxiv.org/abs/2301.05217 (Training curves are not circuit evidence.)
- Nunley, J. (2026). TANGO: Treating Tokens as Operators. *arXiv preprint*. https://doi.org/10.48550/arXiv.2608.22117 (Operator-based token computation.)
- Nye, M., Andreassen, A. J., Gur-Ari, G., et al. (2021). Show your work: Scratchpads for intermediate computation with language models. *arXiv preprint*. https://doi.org/10.48550/arXiv.2112.00114 ; https://arxiv.org/abs/2112.00114 (Scratchpads as supervised intermediate computation.)
- Palermo, J., Ye, J., & Singh, A. (2021). A Reinforcement Learning Environment for Mathematical Reasoning via Program Synthesis. *arXiv preprint*. https://doi.org/10.48550/arXiv.2107.07373 (Solving programs rather than NAND traces.)
- Qiu, L., Li, J., Kim, H., et al. (2026). Explaining and Tuning Transformer-based LLMs in Arithmetic Tasks with Human Strategies. *arXiv preprint*. https://doi.org/10.48550/arXiv.2607.17166 (Integer arithmetic, not quadratic evidence.)
- Saxton, D., Grefenstette, E., Hill, F., & Kohli, P. (2019). Analysing Mathematical Reasoning Abilities of Neural Models. *arXiv preprint*. https://doi.org/10.48550/arXiv.1904.01557 (Sequence-model polynomial roots.)
- Thomas, R., & Bani-Yaghoub, M. (2026). On the Limits of Interpretable Machine Learning in Quintic Root Classification. *arXiv preprint*. https://doi.org/10.48550/arXiv.2602.23467 (Root-type classification, not extraction.)
- Wang, Y., Khan, A. G., Ali, N. Z., et al. (2026). A Function-Conditioned Neural Framework for High-Order Caputo Fractional King–Neta Root Estimator with Computational Analysis. *Fractal and Fractional*. https://doi.org/10.3390/fractalfract10100669 (Journal article; approximate numerical roots.)
- Zhang et al. (2008). Finding multiple real roots by neural networks based on complete discrimination system of polynomial. *2008 IEEE Conference on Cybernetics and Intelligent Systems*. https://doi.org/10.1109/ICCIS.2008.4670918 (Conference paper; numerical root finding; source author attribution retained.)
- Zhang, Y., & Zeng, Z.-Z. (2009). Neural Network Algorithm for the Simultaneous Extraction of All Roots of Algebraic Polynomial. *2009 International Conference on Computational Intelligence and Security*. https://doi.org/10.1109/CIS.2009.78 (Conference paper; numerical root finding.)
- Zhu et al. (2008). A fast neural-network algorithm for simultaneous extraction of all roots of algebraic polynomial. *2008 9th International Conference on Signal Processing*. https://doi.org/10.1109/ICOSP.2008.4697750 (Conference paper; numerical root finding; source author attribution retained.)
