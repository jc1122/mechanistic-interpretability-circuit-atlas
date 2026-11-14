# Binary Gaussian elimination

If x xor y = 1 and y = 1, then x = 0: here adding two 1 bits gives 0, and xor means addition with that wraparound rule. Four-layer transformers learned either to give the solution directly or to write each row update of Gauss–Jordan elimination, a method that removes variables from the other equations one at a time.

The headline experiment is solved, yet its networks used three different internal routes to the same accuracy. The available formulas are [Partial](../GLOSSARY.md#verifying-a-circuit), supported by local replacements without testing changes to everything outside the proposed circuit, and no setting established reliable exact transfer across all tested matrix sizes.

## The circuit at a glance

<img src="../figures/circuits/gf2-elimination.svg" width="640" alt="Earlier trained layers leave an unknown answer code on one residual stream. Mutually exclusive seed families use individual layer-3 heads, a layer-3 head group, or an attention bypass before the final bit writer. Identity matrix with right side 1010 targets solution 1010, not a new model test.">

*Figure S1. Final answer-only networks trained from scratch (10 seeds: 6 head-readout, 2 diffuse and 2 that finish before the last attention) have three last-layer routes and no hand-built answer-only counterpart; only the two early-finish seeds have a partial constant-attention substitute, with complement untested, no complete algorithm and failed size transfer.*

Figure S1 shows mutually exclusive seed families: individual-head readout, a head group or a bypass when the answer is already available.

| Fact | Value |
|---|---|
| Task | Solve A x = b with binary entries and arithmetic modulo two. |
| Input → output example | Equations x xor y = 1 and y = 1 give x = 0, y = 1. |
| Model | 4-layer transformer, width 384, 12 heads per layer, ReLU MLP width 256; PART trains only the final row-update MLP. |
| Rungs and seeds | HAND: formula-set solver; m = 3 PART, attention-only PARTp, and both whole-network TRAIN formats: 10 seeds each; m = 4 retry: 3 seeds per format; final answer-only TRAIN: 10 seeds. |
| Closed form status | Partial (PART row-write substitution: 20/20 snapshots, 400 test systems; all 1344 invertible 3×3 systems in the 10 archived best snapshots; complement not tested). A local attention-removal formula covers only 2/10 final TRAIN seeds. |
| Status | Solved (answer-only TRAIN m = 4, 10/10 final seeds at exact 1.0 on 2000 unseen-matrix tests); last run 2026-11-08. No reliable size transfer. |
| Weights | [Hugging Face: gf2-elimination](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/gf2-elimination) (weights and split archive). |
| Reproduce | Not available yet: training code for this task is not yet in `code/`. |

Learning only an update inside a fixed hand-built solver succeeded in 10/10 runs, while whole-network training at the larger data setting reached whole-output accuracy 1.0 on 2000 unseen-matrix systems in all 10/10 final runs, with correct earlier output tokens supplied.

## 1. Task and data

GF(2) is the field with values 0 and 1: addition is exclusive-or (xor), so 1 + 1 = 0, and multiplication is AND. For example, x xor y = 1 together with y = 1 has solution x = 0, y = 1. Each input is an invertible m × m matrix A and a right-hand-side vector b. A pivot row provides the next variable to eliminate; adding that row flips the target row's bits wherever the pivot row has a 1.

The token input is `f r_0 , .. , f r_{m-1} ;`, with row r_i = a_i0 .. a_i,m-1 b_i. Flag f is `.` for unused and `*` for pivot used.

*Table 1. Output formats. A matrix row includes its right-hand-side bit.*

| Format | Output |
|---|---|
| T, scratchpad | At step k, `p_k f r_0 , .. ;`, with p_k the first unused row having a 1 in column k−1; then `= x_0 .. x_{m-1}`. |
| A, answer only | `= x`. |

A scratchpad writes intermediate steps before the answer ([glossary](../GLOSSARY.md#task-formats)). The split was 70/30 by matrix, so test matrices were unseen. At m = 3, all 168 invertible matrices supplied 944 / 400 train/test systems. At m = 4, 6000 of 20160 matrices supplied 67.2k training systems and 2000 test systems. Exact match scores the whole output with the correct preceding tokens supplied (teacher forcing; [glossary](../GLOSSARY.md#task-formats)). Size tests used 500 systems at m = 2–5.

The source's historical diagnosis prose says 940 training systems, whereas its entry and substitution cohort say 944. Both are retained here as a source discrepancy; the reported substitution uses 944 fit / 400 test, and no replacement count is inferred.

## 2. Model

### 2.1 Architecture

The GPT-NeoX-shaped decoder has 4 layers × 12 heads, d_head 32, d_model 384, an MLP of width 256 with ReLU activation, no LayerNorm, no positional encoding and an untied readout. An MLP is a feed-forward block applied at each position; an attention head reads information from earlier positions ([glossary](../GLOSSARY.md#standard-machine-learning-terms)). The vocabulary has 13 tokens: BOS, `0 1 , ; = . *` and `a`..`e`; the context has 256 positions.

### 2.2 Frozen parts and trained part

HAND sets all weights by formula; PART freezes that network except one trained stage; TRAIN learns all weights from scratch ([glossary](../GLOSSARY.md#the-three-rungs)). PART reinitialises HAND's L3 MLP, with input weights [256, 384], output weights [384, 256] and biases. PARTp instead trains L3 attention: query/key/value weights [1152, 384], output weights [384, 384] and biases. These are alternative cells, not jointly trained stages.

### 2.3 Training

AdamW used learning rate 1e-3, weight decay .1, betas (.9, .98), warmup 1000, cosine schedule, clipping at 1 and batch 256. The m = 4 cell used 6 micro-batches. Evaluation was every 500 steps; the best snapshot was retained. Training stopped at test 1.0 or a 5k plateau after the minimum steps, 15k for m = 3 and 22.5k for m = 4. PART used fp32 and TRAIN TF32 on consumer GPUs. PART's budget was 10k and TRAIN's 30k steps. Shared conventions are in [METHODS.md](../METHODS.md).

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded accuracies are on Hugging Face at [gf2-elimination](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/gf2-elimination).

### 2.5 Archive

The archive contains `hand_T`, `part_T_m3_s0–9`, `partp_T_m3_s0–9`, selected `train_T_m3_s6/s0` and `train_A_m3_s4/s3`, `train_A_m4_s0–9`, `split_m3/m4`, config.json and manifest.json. The manifest records seed, rung, step, test score and sha256. Reload was bit-identical. The m = 4 scratchpad retry's final/best weights were lost; those results are log evidence, not downloadable final networks.

## 3. Hand-built solution

The row update is new_r = old_r xor ((old_r[k−1] AND [r ≠ p_k]) old_p_k). For one output bit this becomes OLD xor (K AND PB AND NOT EQ): OLD is the target row's previous bit, K its pivot-column bit, PB the corresponding pivot-row bit and EQ whether this is the pivot row itself. The pivot row is copied; another row is flipped only when its pivot-column bit is 1.

L0–L1 compute the block, row and column frames (BLK, ROW, J) and their next-token versions. A uniform head and 256 position neurons, five anchor heads and paired ReLU steps `relu(4z+3) − relu(4z+2)` provide the frame. L2 uses five 3-D match heads, which fetch values by address, to retrieve OLD, OLDF, K, PB, USED and PX; its MLP computes slot flags and EQ ([glossary](../GLOSSARY.md#hand-built-machinery)). OLDF is the previous row's pivot-used flag, USED marks an already chosen pivot row, and PX is the row address for reading the solution. L3 H0 chooses the pivot with score −B·dist² + G·bit − G·USED − A·ROW; H1 reads the answer bit. Here dist is address distance, ROW is the candidate row index, and B, G and A are fixed score gains. The last MLP has one neuron per 1-cell of the row-update truth table, which lists the output for every Boolean input combination.

Table F1 records the parallel row-update and pivot/readout interfaces. A frozen row and column frame supplies fetched operand bits and the equality flag. Pivot and answer reads run alongside MLP3's XOR-AND row update before the matrix and solution are written. Only MLP3 is trained in PART; the separate attention-only PARTp cell trains different components.

<details>
<summary>Technical row-update, pivot and readout interfaces (Table F1)</summary>

*Table F1. The hand-built row-update circuit. Pivot/answer reads and MLP3's row-bit write have parallel output routes. MLP3 is the only trained part in PART. The separate PARTp cell trains L3 attention instead; it has different trainable components.*

| Component | Input, operation and output |
|---|---|
| Fixed layers 0–1 (L0–L1) | Construct the block, row and column frame. |
| Fixed layer 2 (L2) | Fetch OLD, K and PB and compute equality flag EQ. These are the row-update inputs defined above. |
| Fixed layer 3, heads 0–1 (L3 H0–H1) | Read pivot and answer information, alongside the row-update MLP rather than after it. |
| MLP3, trained in PART | Write OLD xor (K and PB and not EQ). Together with the parallel pivot/answer route, this produces the updated matrix block, then solution bits. |

</details>

HAND uses 13 heads, neurons 256/80/14/32 and 1630 non-zero weights. It was exact on all systems at m = 2, 3, 4 and on 200 random systems at m = 5. These allocations are not measured minimum sizes.

## 4. What training finds

The final run was analysed on 2026-11-08.

*Table 2. Training outcomes. A solved seed reached whole-output exact 1.0. m = 3 tests use unseen matrices; the final m = 4 test contains 2000 unseen-matrix systems. Best and final values refer to saved snapshots ([glossary](../GLOSSARY.md#counting-results)).*

| Cell | Solved / seeds | Best / final test | Timing |
|---|---|---|---|
| T PART, m = 3 | 10/10 | 1.0 / 1.0 | 1000 steps. |
| T PARTp, m = 3 | 10/10 | 1.0 / 1.0 | Best at 1.5–4k. |
| T TRAIN, m = 3 | 5/10 | .73–1 / .70–1 | Training ≥ .99 by 3.5–5k. |
| A TRAIN, m = 3 | 5/10 | See diagnosis below. | Training ≥ .99 by 3.5–5k. |
| A TRAIN, m = 4 retry | 3/3 | 1.0 / 1.0 | Data retry. |
| T TRAIN, m = 4 retry | 3/3 | 1.0 reported; final/best weights lost. | Log-only result. |
| A TRAIN, m = 4 final | 10/10 | 1.0 / 1.0 | Best step 12–21.5k. |

Failed m = 3 seeds memorised the 118 seen matrices. The data retry increased matrix diversity to 6000 at m = 4, and the final answer-only cell solved. This is consistent with training-fraction effects discussed by [Liu et al. (2022)](#references) and diverse-matrix generalisation studied by [Charton (2021)](#references), without isolating data diversity from the change of size.

PART has one type in 10/10 seeds: frozen HAND fetches plus a learned cell-detector MLP3. PARTp has one type, but its attention also carries the update in 7/20 best/final snapshots. Whole-network m = 3 circuits are distributed: interchange supports use of HAND's inputs (OLD and K flips ≥ .97). The attempted alignment has R² < 0, so that mapping failed; it does not identify the network's algorithm or a shared basis. Edge Jaccard .07–.10 measures the low overlap of the compared graphs ([glossary](../GLOSSARY.md#analysis-measures)).

The final answer-only networks form three types: L3-head readout in 6/10 seeds (4–10 L3 heads, answer R² at L3 ≤ .26), diffuse L3 readout in 2/10 (no individual L3 head crosses the ablation threshold, but the whole layer is needed; R² at L3 .54), and L2-complete in 2/10 (L3 attention idle; R² at L3 .89).

Table F2 records the alternative last-layer roles shown above. Earlier trained layers feed alternative last-layer roles: head-based readout, diffuse group readout, or an already-complete representation with replaceable attention. The final MLP and output readout write solution bits. All computational nodes were trained. Partial status covers only constant-attention substitution in the already-complete family, with complement untested; this is a family-role inventory, not a whole-network formula or minimum circuit.

<details>
<summary>Technical last-attention roles and limited substitution scope (Table F2)</summary>

*Table F2. Roles of the last-layer attention in the three final TRAIN families; every computational component is trained. Status: Partial for the two L2-complete seeds' constant-attention substitution, with complement untested and no whole-network formula. This is a family-role inventory, not a minimum circuit or an edge-discovery claim.*

| Alternative last-layer role | Recorded scope and common output |
|---|---|
| Head readout | 6 of the ten final seeds use layer-3 (L3) attention to read the answer from earlier trained layers. |
| Diffuse group readout | 2 of the ten final seeds use L3 attention as a group. |
| Layer-2-complete (L2-complete) | 2 of the ten final seeds already have the needed representation before L3 attention, which is replaceable in the scoped constant-attention test. |
| Common readout | Each alternative feeds trained MLP3 and the output readout to write solution bits. No whole-network formula or minimum-circuit claim follows. |

</details>

Table F3 counts the three last-layer roles rather than drawing them a second time.

*Table F3. Correct answers do not imply one circuit. The dimension-4 final answer-only run has 10 seeds, all at exact 1.0 on 2000 unseen-matrix systems. The whole network is trained. The layer-2-complete family's constant-attention substitution is partial evidence only: no complement test is reported, and this is not a whole-network formula or minimum circuit.*

| Last-layer role | Seeds | What the role means |
|---|---|---|
| Layer 3 head readout | 6/10 | Late heads read the answer. |
| Layer 3 diffuse readout | 2/10 | No single head passes the drop test. |
| Answer complete by layer 2 (L2) | 2/10 | Layer 3 attention can be replaced by a per-slot constant. |

## 5. Mechanism and verification

PART replaces MLP3's row-bit write by μ[OLD, K·PB·¬EQ, query], a table of mean writes indexed by live HAND registers and output query. On 944 fit / 400 test systems it keeps native exact 1.0 in 20/20 best/final snapshots. Zeroed, shuffled and negated controls give 0, ≤ .01 and 0; another seed's table gives 1.0. This supports the local update, as in [binary matrix multiplication](gf2-matrix-product.md). Its atlas status is Partial: substitution is supported, but resampling everything outside the claimed circuit (the complement test) was not reported ([glossary](../GLOSSARY.md#verifying-a-circuit)).

A later full-domain check covers the ten archived best PART snapshots. The frozen tables, fitted on 944 rows, make exactly the same
decisions as the native networks and are task-exact on all 1344 invertible 3×3 systems, covering every matrix and right-hand-side
vector. There are no missing keys or counterexamples. Zero, mean and update-flip controls each score 0/1344. This upgrades
substitution coverage only; a matching exhaustive resample-complement certificate is not established. It makes no claim for
unarchived final tensors or the rejected PARTp forms.

The corresponding PARTp substitution gives 1.0 in 20/20 snapshots, but zero/shuffle controls keep ≥ .2 in 7/20. It is rejected as a separable account of the row update. For TRAIN m = 3, fitted Boolean tables score .05–.06, so no closed form is accepted.

In the final answer-only cell, an oracle readout μ_c + (2x_c − 1)g_c passes in 4/10 best and 10/10 final snapshots, but its zero-MLP3 control fails. Here x_c is the supplied correct answer bit; this is not an accepted decision mechanism. Replacing L3 attention by a per-slot mean on 2000 test systems preserves native 1.0 in 4/4 snapshots of the two L2-complete seeds. It yields .38–.48 in the diffuse family and .05–.18 in the head-readout family. The constant-attention account is therefore Partial, local to those two seeds, with complement untested; the other final circuits have no formula.

## 6. Limitations and open questions

No cell established reliable exact transfer across all the tested sizes. T PART gives m = 2 exact 1.0 in 6/10 seeds and 0 at m = 4, 5. Final answer-only TRAIN gives m = 2 .09–.45, m = 3 .10–.27 and m = 5 .03–.10 on 500-system size tests. No size-generalisation claim follows from the fixed-size success.

Diagnostics covered PART 7/10 seeds, PARTp 4/10, T TRAIN m = 3 6/10, no A TRAIN m = 3 seeds, and all 20/20 best/final snapshots of the final answer-only cell. The m = 4 T retry had 3 seeds but lost final weights. Probes are linear; there was no neuron-level ablation or training-dynamics analysis, and only one width was used. Individual-head ablation missed a layer-level need in the diffuse family, so no single-head absence claim is made.

## 7. Prediction record

Before the runs, PART 10/10 was predicted and held. PARTp ≥ 8/10 and 0 accuracy at m = 4, 5 held. TRAIN T m = 3 ≥ 5/10 held at 5/10, but the prediction that pivot labels would be last to reach .99 was refuted: labels and answers reached 1.0; the remaining errors were row-update bits. TRAIN A m = 3 ≤ 1/10 was refuted by 5/10.

## Appendix A. Weights excerpt and reading

HAND's L3 cell neuron is `relu(Σ ±x + 100·TB − (#ones − 1) − 100)`, with a positive sign on the cell's 1-inputs. TB marks a row-bit output position. The neuron writes Y1 − Y0, with a baseline TB write into Y0, implementing bit = OLD xor (K AND PB AND NOT EQ).

*Table 3. PART's largest absolute Y1 − Y0 vote neuron in selected seeds. Inputs are a row of the layer-3 input matrix at OLD, K, PB, EQ and TB, divided by that row's maximum absolute value. The output is the Y1 − Y0 difference of its output-matrix column.*

| Seed / neuron | OLD, K, PB, EQ, TB; output | Truth-table cell and bit |
|---|---|---|
| s0 n55 | −1.00, −.71, −1.00, +.47, +.33; −.53 | (0,0,0,1) → 0 |
| s3 n213 | +.89, +.84, +.71, −1.00, −.16; −.69 | (1,1,1,0) → 0 |
| s9 n84 | +1.00, −1.00, −.47, −.44, −.11; +.65 | (1,0,0,0) → 1 |

In 9/10 seeds, the top neuron's signs select a truth-table cell and its output votes for that cell's bit. This matches HAND's cell-detector form with different cells. The source reports a share of 8.6–10.1% of W_in for these input columns; these rows do not describe the whole trained response.

## Appendix B. Detailed evidence

*Table 4. Diagnostics on 250 test systems. Used means mean-ablation exact-match drop > .05; neurons were not ablated. r99 is residual rank at L4 explaining 99% of variance; JS is the divergence of used heads from HAND. R² is linear decodability, while interchange/DAS test causal swaps ([glossary](../GLOSSARY.md#analysis-measures)).*

| Cell (diag seeds) | Used heads + MLPs | Best probe R² per HAND variable (layer) | r99 (HAND 11) | JS | Best / final |
|---|---|---|---|---|---|
| T PART (7) | 8 + 2 (frozen fetches, MLP3) | OLD, K, PB, EQ 1 (L3); interchange K 1, EQ, OLD / DAS 1 | 11 | 0 | 1 / 1 |
| T PARTp (4) | 10–13 + 2 | OLD, K, PB, EQ 1 (L3); pivot .92–.96 (L4) | 6–12 | .11 | 1 / 1 |
| T TRAIN m3 (6) | 11–15 + 4 | Failed seeds OLD, K, PB, EQ .12–.70 (L3); interchange OLD, K ≥ .97 | 27–36 | .47 | .73–1 / .70–1 |
| A TRAIN m4 final (10 B + 10 F) | 6–19 + 4 (final) | Answer .80–.94 (L4); flip .996–1 (L4) | 20–51 | — (no HAND) | 1 / 1 |

B and F mean best and final snapshots. From scratch the circuit uses 6–19 heads and all MLPs; the local PART circuit reads HAND's bits. This does not establish a minimal circuit or a common weight basis across whole-network seeds.

## What to read next

Read [binary matrix product](gf2-matrix-product.md) for another local XOR-and-AND lookup. [Linear systems over finite fields](linear-systems-finite-fields.md) broaden the comparison beyond binary coefficients.

## References

- Charton, F. (2021). Linear algebra with transformers. *arXiv preprint*. https://arxiv.org/abs/2112.01898 ; https://doi.org/10.48550/arxiv.2112.01898 (Diverse matrix datasets and linear-algebra generalisation.)
- Liu, Z., Kitouni, O., Nolte, N., Michaud, E. J., Tegmark, M., & Williams, M. (2022). Towards understanding grokking: An effective theory of representation learning. *arXiv preprint*. https://arxiv.org/abs/2205.10343 ; https://doi.org/10.48550/arxiv.2205.10343 (Training-fraction effects and memorisation.)
