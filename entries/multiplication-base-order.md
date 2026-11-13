# Multiplication across bases and digit orders

Writing a product's units digit first lets a model build the answer in the same direction as the carries in long multiplication. These small transformers multiplied numbers below 256: on 2000 held-out pairs per seed, decimal units-first output reached best whole-output exact match ≥ .99 in 20/20 seeds, while largest-place-first output met that bar in only 1/60 seeds across the three tested bases at the original width. Writing running sums before the final product succeeded in all 18/18 screened seeds at that width. The task is partly solved, with no reliable transfer to longer numbers and no hand-built network for comparison. The formula is Partial and covers only outgoing scratchpad digit votes, with decimal passing in the same 6/6 seeds at both best and final snapshots and hexadecimal in 3/6 at best versus 4/6 at final snapshots of the same seed cohort; complement resampling was not tested.

| Fact | Value |
|---|---|
| Task | Multiply x and y in [0, 256), written in base 2, 10 or 16, as `x*y=` followed by the product |
| Input → output example | Base 10, answer only, most significant digit first: 7 × 3 → 21, written as the 5 zero-padded digits 00021 |
| Model | GPT-NeoX-shaped, 4 layers, d_model 384 (12 heads × 32), MLP 256 ReLU; all weights trained from scratch |
| Rungs and seeds | TRAIN only (no HAND or PART, an explicit exception); answer only 6 cells × 4 mixes × 5 seeds (120); data retry 25, auxiliary loss 12, scratchpad 18, final cell 10. Width-retry inventory reports 45 but documents seven five-seed cells; unresolved in Section 2.3. |
| Closed form status | Partial (outgoing scratchpad digit-vote map only; ≥ .99 × native on 2048 fresh pairs: decimal 6/6 best and final, hex 3/6 best / 4/6 final; complement not tested). No answer-only formula. |
| Status | Partly solved (original-width largest-place-first answer only 1/60; scratchpad 18/18 at best exact ≥ .99 on 2000 pairs per seed; binary units-first final 10/10); last run 2026-11-08 |
| Weights | [Hugging Face: multiplication-base-order](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/multiplication-base-order) (weights and held-out evidence archive). |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

Read next: [Results](#4-what-training-finds) and [verification](#5-mechanism-and-verification), then [binary multiplication](multiplication.md) for the running-sum step, or [output-order base conversion](output-order-base-conversion.md) for the same ordering question in conversion.

## 1. Task and data

The prompt is `x*y=` and the target is the product x·y, with x, y in [0, 256). In decimal, 7 × 3 = 21; the five-digit largest-place-first output is 00021, while units-first starts with the units digit. Least-significant-first (LSB) means units first; most-significant-first (MSB) means the largest place first ([glossary](../GLOSSARY.md#task-formats)). Numbers are zero-padded: in base 2 the operands have
8+8 digits and the product 16, in base 10 3+3 and 5, in base 16 2+2 and 4. The test set is 2000 held-out pairs; training draws fresh
pairs from the other 63,536, so train and test accuracy stay close. An output is exact when every output token is right; a seed is
solved when its best test exact match is ≥ .99. Scoring is teacher-forced, with the correct earlier tokens supplied ([glossary](../GLOSSARY.md#task-formats)). Table 1 lists the output formats and the retry variants.

*Table 1. Output formats and retry variants.*

| Label | Meaning |
|---|---|
| AO, MSB | answer only, most significant digit first (MSB first) |
| AO, LSB | answer only, least significant digit first (LSB first) |
| SP | A scratchpad, intermediate working before the answer ([glossary](../GLOSSARY.md#task-formats)): the running sums S_j = x (y mod B^(j+1)), each LSB first and followed by `;`, then the product MSB first |
| R1 | retry with 2x width |
| D3 | retry with 3x data |
| aux | retry with a running-sum probe loss |

Each cell also sets a curriculum mix f: a training sample is a sub-problem with probability f (digit × digit, or y with one or two
nonzero digits), and otherwise a uniform pair. The mixes are f = 0, .3, .6 and anneal (f from .6 to 0 by step 15k).

## 2. Model

The shared architecture and instrumentation are described in [METHODS.md](../METHODS.md). This entry has only the TRAIN
[rung](../GLOSSARY.md#the-three-rungs), in which all weights are trained from scratch.

### 2.1 Architecture

The networks are GPT-NeoX-shaped decoders, reading left to right with attention heads and feed-forward MLP blocks ([glossary](../GLOSSARY.md#standard-machine-learning-terms)). They have 4 layers, d_model 384 (12 heads × 32), MLP 256 ReLU (W_in [256, 384], reading the whole
residual stream), no LayerNorm, no positional encoding and untied embeddings. The vocabulary has 19 tokens (BOS, `*`, `=`, digits
0-15), and 20 in SP, which adds `;`. The R1 retry uses d 768 (12 × 64); D3 (3x data) uses batch 768, three times the base batch of 256; aux adds a running-sum probe loss at
resid_3.

### 2.2 Frozen parts and trained part

Nothing is frozen; every weight is trained.

### 2.3 Training

The optimiser was AdamW with learning rate 1e-3, weight decay .1, betas (.9, .98), warmup 1k, cosine decay to 30k steps, gradient
clip 1 and batch 256, in TF32, with test accuracy evaluated every 500 steps. All cells, in base 2, 10 and 16, used 4 layers, d 384 and 30k steps. The answer-only screen ran 6 cells × 4 mixes × 5 seeds
(120 networks); the retries were R1 2x width (45), D3 3x data (25) and aux (12); SP ran 18 networks. The final cell (binary LSB, anneal
mix) reran the same recipe with 10 seeds and a plateau stop. The reported width-retry inventory of 45 seed-runs is not reconciled with the seven five-seed R1 cells documented by the results table. It is retained as an unresolved inventory claim, not used to infer a new denominator.

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/multiplication-base-order](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/multiplication-base-order).

### 2.5 Archive

Files are `<cell>_s<seed>[_final].safetensors`, `config.json` and `manifest.json` (sha256, fp32 re-score). The round trip passes for
all 43 networks, including SP and the final cell; the Hugging Face copy was re-downloaded and sha256-checked, and 2 networks were
re-scored.

## 3. Hand-built solution

There is no hand-built network for this entry (an explicit exception; no HAND or PART rung). The trained networks were probed instead
for the variables of long multiplication in base B (Figure 1): the product digit p_k, the column sum col_k = Σ_{i+j=k} x_i y_j, the
carry carry_k into column k, and low_k / high_k, the product mod / div B^k. For SP lines the same variables were probed for
S_j = S_{j−1} + x y_j B^j.

```mermaid
flowchart TD
    accTitle: Long-multiplication variables used for probes
    accDescr: A trained network is probed for column sums, carries, product digits and low and high product parts. No hand-built network or PART cell exists for this study; no node is shaded as a PART stage.
    subgraph VARIABLES["Reference arithmetic variables"]
        direction TB
        X["operand digits x_i and y_j"] --> C["column sum col_k =<br/>sum of x_i y_j<br/>over i + j = k"]
        C --> K["carry carry_k"]
        C --> P["product digit p_k"]
        K --> P
        C --> LH["low_k and high_k:<br/>product mod and div B^k"]
        %% Invisible link stacks outputs; it is not a computational path.
        P ~~~ LH
    end
    subgraph PROBE["Separate measurement"]
        N["TRAIN network: 4 layers,<br/>all weights trained;<br/>probed for col_k and carry_k"]
    end
    %% Invisible link stacks panels; it is not a computational path.
    VARIABLES ~~~ PROBE
```

*Figure 1. The reference variables of long multiplication that the trained networks were probed for. No hand-built network or PART cell exists for
this entry, so no node is shaded. The network is trained from scratch in full.*

## 4. What training finds

The 2026-11-08 analysis found different answer-only outcomes by digit order in this tested range (Table 2). Least significant digit first solves base 10 in 20/20 seeds (5/5 in
every mix) and base 2 in 17/20 (final cell 10/10); most significant digit first solves 1/60. Width rescues decimal and hex MSB (7/10,
6/10), never binary MSB (0/26 over width, data and aux). The aux retries solved base 2 MSB in 0/6 (.90-.98) and base 10 MSB in 0/6
(.87-.98). A scratchpad solves every MSB cell at 1x (18/18).

*Table 2. Solved seeds per cell (best test exact match ≥ .99). f = curriculum mix, a = anneal; values in parentheses are best test
accuracies of unsolved seeds; the SP column gives the first step at which test exact match reached .99.*

| base | order | f 0 | .3 | .6 | anneal | R1 2x width | D3 3x data | SP 1x f0 / anneal (first .99) |
|---|---|---|---|---|---|---|---|---|
| 2 | LSB | 3/5 | 4/5 | 5/5 | 5/5; final 10/10 | - | - | - |
| 10 | LSB | 5/5 | 5/5 | 5/5 | 5/5 | not reported | not reported | not reported |
| 16 | LSB | 1/5 | 1/5 | 2/5 | 0/5 | anneal 5/5 | - | - |
| 2 | MSB | 0/5 | 0/5 | 0/5 | 0/5 | f0 0/5, a 0/5 (.91-.96) | f0 0/5 (.90-.93) | 3/3 / 3/3 (8.5-10.5k) |
| 10 | MSB | 0/5 | 0/5 | 0/5 | 1/5 | f0 3/5, f.3 4/5 | f0 1/5, a 2/5 | 3/3 / 3/3 (11.5-16k) |
| 16 | MSB | 0/5 | 0/5 | 0/5 | 0/5 | f0 5/5, a 1/5 | f0 0/5, a 0/5 (.97-.99) | 3/3 / 3/3 (17.5-21.5k) |

Base 10 LSB solved 5/5 in every mix, as shown in the decimal units-first row.

In the final cell, test accuracy reached .5 at 6-10k steps and .99 at 16-23.5k (median 18.75k); training stopped at 28.5k, with best =
final = 1.0 (10/10). Train accuracy was close to test accuracy because pairs are fresh. The answer-only MSB retries grokked at 25-28k
of 30k steps, so they were limited by the step budget ([grokking](../GLOSSARY.md#counting-results): test accuracy rising long after
training accuracy). Binary MSB under-fits (train .95-.999), and decimal and hex MSB at 1x memorise (train 1.0). In SP,
best = final = 1.0.

### 4.1 Circuit types

Four circuit types were found, one per base for answer only plus one for SP, each reading column variables at MLP3 (Table 3). Types
were assigned from the per-seed interchange flip at L4 and the DAS flip and do not overlap; an interchange flip counts how often the
answer changes as predicted when one variable is swapped between two inputs, and DAS does the same through a learned rotation
([glossary](../GLOSSARY.md#analysis-measures)). Unsolved answer-only networks carry their base's type.

*Table 3. Circuit types (seed counts are of diagnosed networks).*

| Type | Cells | Evidence |
|---|---|---|
| A, linear column/carry (binary) | b2 LSB 17/17; final cell 10/10, all diagnosed | col_k flip .99-1.0, carry_k .87-1.0 |
| B, non-linear column (decimal) | b10 LSB 20/20, R1 1/1 | col_k flip .16-.26, DAS .99-1.0 |
| C, per-digit table (hex) | b16 LSB 4/4, R1 5/5 | col_k flip .03-.15, DAS .21-.28, carry_k flip .99-1.0 |
| D, SP scratch-line column | b16 SP 6/6, b10 SP 4/4 diagnosed; b2 SP undiagnosed | col_k flip 1.0 at the scratch queries |

The variables col_k and carry_k fix the answer: excluding them at L4 drops exact match to about 0 (the restrict column of Table B1).
The failures lie in computing them. MSB networks use more components, carry is decoded with lower R², and errors fall at the longest
carry chains (b2 bits 6-9, b10 digit 2, b16 digits 1-2). Per-cell diagnostics are in Appendix B.

## 5. Mechanism and verification

### 5.1 Closed form of the scratchpad readout

In SP, the digit map of the final MLP, E = W_out^T U_digits^T (+ bias), where U_digits is the unembedding of the digit tokens, was
replaced by a few Fourier modes, sine and cosine patterns over the digit d ([glossary](../GLOSSARY.md#analysis-measures)). This replaces the outgoing vote map only, not how the network computes the column variables or activates the neurons:

Ehat[j,d] = c[j,0] + Σ_{k ∈ K} (c[j,k] cos 2πkd/B + s[j,k] sin 2πkd/B),

with coefficients taken as exact DFT (discrete Fourier transform) projections of the weights and the mode sets K_dec = {1,2,3,5} and K_hex = {1,2,4,8}. The bar was
exact ≥ .99 × native accuracy on 2048 fresh pairs. Decimal passed in 6/6 best and 6/6 final seeds (lossless in 2/6 and 4/6), hex in
3/6 and 4/6 (lossless in 1/6). The full DFT keeps all accuracy and the mean-only map keeps none (0). This is supported by substitution;
the complement was not tested, so the closed form status is partial.

```mermaid
flowchart TD
    accTitle: Scratchpad outgoing digit-vote substitute
    accDescr: The trained network keeps its native final-MLP activations and other writes while selected Fourier modes replace outgoing digit votes. This is partial, scratchpad-only substitution support; complement resampling is untested.
    X["prompt and scratch lines so far"] --> R["TRAIN network:<br/>native final-MLP activations,<br/>unchanged"]
    R --> E["MLP3 outgoing<br/>digit-vote map E"]
    F["Weight-derived Fourier modes<br/>K_dec or K_hex"] -.->|"SP substitution only"| E
    E --> D["digit logits, with other<br/>native writes retained"]
    %% Invisible link stacks inputs; it is not a computational path.
    R ~~~ F
    classDef trained fill:gold,stroke:darkgoldenrod,color:black
    class R,E trained
```

*Figure 2. Substitution of the outgoing MLP3 digit-vote map in SP, leaving native neuron activations and other writes unchanged. The column-variable interventions were at L4, after the fourth layer; they do not show that MLP3 receives either variable as a single scalar. The Fourier closed form is Partial (substitution only, SP only); answer-only types A-C have no closed form. The shaded network and digit-vote map are trained; decimal passes in 6/6 best and final seeds, hexadecimal in 3/6 best and 4/6 final, and complement resampling is untested.*

![Multiplication at the original width: units-first successes are 17/20 binary, 20/20 decimal and 4/20 hexadecimal, versus 0/20, 1/20 and 0/20 largest-place-first.](../figures/multiplication-base-order-success.svg)

*Figure 3. Original-width answer-only screens, 20 seeds per base and order across four data mixes. Success means best whole-output exact match ≥ .99 on 2000 held-out pairs. Scratchpad 18/18 covers a separate screened comparison; neither result establishes length generalisation or a compute-matched format effect.*

[Open Figure 3 at full size](../figures/multiplication-base-order-success.svg).

### 5.2 Answer only

Types A-C have no accepted closed form. Substituting a scalar col_k / carry_k (decimal, probe R² .998/.959) drops exact match from
2041 to 3 of 2048 pairs. This rejects that scalar replacement; it does not rule out every possible scalar representation.

### 5.3 Relation to published work

There is no hand-built network to compare with. The digit-order effect agrees with [Lee et al. (2023)](#references), who found that
reversed output helps small transformers learn arithmetic, and with [Nogueira et al. (2021)](#references), who found that the number
representation decides learnability. The auxiliary running-sum loss used by [Bai et al. (2025)](#references) did not rescue binary
MSB here; a scratchpad did. The non-linear code for decimal col_k fits the periodic number features described by
[Fu et al. (2026)](#references). The entry's own new
finding is that the base effect depends on digit order.

## 6. Limitations and open questions

Screening cells used 5 seeds (aux and SP 3 per mix, final cell 10); base 2 diagnostics used 1000 pairs. Length generalisation was
tested with SP padding or input ranges from [0, 512) to [0, 10000): 0/256 (one network 2/256). The final cell scored 1.0 natively (20/20
networks: 10 best and 10 final), .019–.29 with 9-bit padding and .002–.10 on [0, 512), where the middle bits fail. The earlier entry reported these ranges at lower precision as .02–.29 and .00–.10; these are legacy reporting, not separate measurements. Binary MSB answer only with 2x the step
budget is untested (the run was killed).

The formats differ in output length, data mix and sometimes width or batch size. These comparisons do not isolate a format-only causal effect. Missing diagnostics:

- SP base 2 (out of memory);
- SP base 10, 2 of 6 networks;
- 11 R1/D3 decimal networks.

## 7. Prediction record

Predictions were recorded in the task specification before training (Table 4).

*Table 4. Prediction record.*

| Source | Prediction | Outcome |
|---|---|---|
| SPEC.md | b16 and b10 solve, b2 MSB fails | half right (hex fails at 1x) |
| SPEC.md | LSB helps every base | right |
| SPEC.md | curriculum speeds base 2/10, rescues nothing | right (anneal > .3, not ~) |
| SPEC.md | width does not rescue b2 MSB | right |
| MUL_STUDY (SP) | b10/b16 ≥ 2/3 per mix, faster than 2x | right |
| MUL_STUDY (SP) | b2 ≥ 2/6 | right |
| MUL_STUDY (SP) | b2 slower than b10/16 | wrong |
| Final cell | ≥ 9/10 | right (10/10) |

## Appendix A. Weights excerpt and reading

The excerpt is taken from the f0 mix, seed 0, best snapshot, final MLP. For neuron j, W_in[j, 0:2] is mapped to E[j, digits 0-3],
computed through the full W_out and the unembedding; rows are chosen by digit-contrast norm (Table A1).

*Table A1. Final-MLP neurons: W_in[j, 0:2] and the digit votes E[j, digits 0-3].*

| Base | Neuron | W_in[j, 0:2] | E[j, digits 0-3] |
|---|---|---|---|
| decimal | N123 | (.0731, .1140) | (1.431, -1.567, 1.452, -1.673) |
| decimal | N8 | (.1037, .0168) | (-.967, 1.478, -1.314, 1.279) |
| hex | N215 | (.0499, -.0325) | (-2.169, -1.655, -.871, .071) |

Reading: N123 votes for even digits and N8 for odd ones (parity mode c5 = +1.518 / -1.307); hex N215 has a broad preference
(c1 = -1.829, s1 = .462). These are outgoing votes and do not show how the inputs activate the neurons.

## Appendix B. Diagnostics per cell

*Table B1. Diagnostics on best snapshots (seed ranges).*

| cell | used H+M of 52 | patch top (drop) | p_k | col_k | carry_k | low_k | high_k | R^2 col/carry | PR | restrict | best / final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b2 LSB 1x | 22-42 | MLP3 (MLP2) .94-1.0 | .87-.92/1.0 | 1.0/1.0 | 1.0/1.0 | .83-.88/1.0 | .41-.63/1.0 | .88-.95/.93-.95 (L4) | 1.5-3.1 | .01 | .63-1.0 |
| final cell (10 best, 3 final) | 22-32 | MLP3 (MLP0) .98-1.0 | .91/1.0 | 1.0/1.0 | 1.0/1.0 | .85/1.0 | .58/1.0 | .94/.94 (L4) | 2.4 / 2.9 | .01 | 1.0 / 1.0 |
| b2 MSB 1x / 2x / aux | 40-52 | MLP3 .95-.99 | .97-.99/.99 | 1.0/1.0 | .99-1.0/.99 | .97-.99/.99 | .91-.97/.99 | .83-.96/.83-.92 (L3-4) | 1.2-1.7 | .01 | .74-.98 |
| b10 LSB 1x | 23-37 | MLP3 .96-1.0 | .99/.99 | .22-.24/1.0 | u | .10-.12/.36-.44 | .15-.16/.36-.45 | .92-.93/.93-.96 (L3) | 3.6-4.5 | .00 | 1.0 |
| b10 MSB 1x / aux | 33-47 | MLP3 (MLP0) .97-1.0 | .96-.98/.95-.98 | .17-.24/.99 | u | .09-.11/.35-.45 | .15-.18/.35-.44 | .88-1.0/.89-.96 (L3) | 6.0-7.0 | .00 | .87-.99 |
| b16 LSB 1x | 25-34 | MLP3 (MLP0) .97-1.0 | .97-.98/.96-.97 | .10-.14/.27-.28 | 1.0/.98-.99 | .07-.08/.22-.25 | .08/.23-.25 | .92-.93/.96-.97 (L3) | 3.9-6.3 | .00-.01 | .84-1.0 |
| b16 MSB 1x | 33-46 | MLP3 (MLP0) .99-1.0 | .91-.97/.91-.96 | .10-.13/.26-.31 | .98-.99/.95-.97 | .06-.08/.22-.24 | .06-.08/.22-.24 | .90-.92/.89-.93 (L3) | 7.0-7.5 | .00 | .72-.98 |
| b16 2x (LSB+MSB) | 27-31 | MLP3 .98-1.0 | .99-1.0/.98-.99 | .09-.11/.21-.24 | 1.0/.97-.98 | .03-.06/.19-.23 | .05-.09/.18-.23 | .92-.95/.92-.96 (L3) | 4.9-8.5 | .00-.01 | .99-1.0 |
| SP b16 (6+1 final); b10 (4+1) | 23-30; 13-19 | MLP3 .99-1.0; .98-1.0 | 1.0/.99; .99-1.0/1.0 | 1.0/.99-1.0; 1.0/1.0 | u | .07-.10/.26-.29; .11-.16/.33-.43 | .08-.09/.26-.29; .13-.19/.33-.43 | .81-.83/.62-.66 (L4/L3); .88-.89/.67-.79 (L4) | 4.4-6.7; 4.9-7.7 | .00 | 1.0 / 1.0 |

Key: used H+M = used heads and MLPs, of 52 components; patch top (drop) = the component with the largest
patching drop and that drop, with a second component in parentheses as recorded; variable columns = interchange / DAS flip at L4, u = untested; R^2 = best
ridge-probe R² of col / carry (layer); PR = effective rank at L4 (no HAND to compare); restrict = exact match at L4 with the variables
excluded; JS is not applicable (no HAND); final ≈ best. Terms are defined in the [glossary](../GLOSSARY.md#analysis-measures).

## References

- Bai, X., Pres, I., Deng, Y., et al. (2025). Why can't transformers learn multiplication? Reverse-engineering reveals long-range
  dependency pitfalls. *arXiv preprint*. https://doi.org/10.48550/arxiv.2510.00184 ; https://arxiv.org/abs/2510.00184 (auxiliary running-sum loss for multiplication)
- Fu, D., Zhou, T., Belkin, M., Sharan, V., & Jia, R. (2026). Convergent evolution: How different language models learn similar number representations. *Conference on Language Modeling (COLM 2026); arXiv preprint*.
  https://doi.org/10.48550/arxiv.2604.20817 ; https://arxiv.org/abs/2604.20817 (periodic number features)
- Lee, N., Sreenivasan, K., Lee, J. D., Lee, K., & Papailiopoulos, D. (2023). Teaching arithmetic to small transformers.
  *arXiv preprint*. https://doi.org/10.48550/arxiv.2307.03381 ; https://arxiv.org/abs/2307.03381 (reversed output order)
- Nogueira, R., Jiang, Z., & Lin, J. (2021). Investigating the limitations of transformers with simple arithmetic tasks.
  *arXiv preprint*. https://doi.org/10.48550/arxiv.2102.13019 ; https://arxiv.org/abs/2102.13019 (representation decides learnability)
