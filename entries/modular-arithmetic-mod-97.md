# Modular arithmetic mod 97 in six number formats

The number 96 can be one symbol or two decimal digits. These networks learned differently from that choice, although both formats ask the same arithmetic: 96+2 leaves remainder 1 after division by 97. Whole-number multiplication produced circular codes ordered by powers of 5. An exponent counts the multiplications by 5; this code, called a [discrete logarithm](../GLOSSARY.md#analysis-measures), turns multiplication into addition.

Every whole-number multiplication run developed circles, but only 6/10 reached held-out accuracy ≥ .99 on 941 inputs per run. The tested digit formats had no circular code; digit multiplication remained limited by memorisation. The task is partly solved.

Its output-only formula is [partial](../GLOSSARY.md#verifying-a-circuit): replacement worked on all nonzero pairs in eight runs with held-out accuracy ≥ .98, but included training inputs and used externally supplied true exponents. It does not explain how the network calculates those exponents. No test swapping the other parts' signals between examples established that internal mechanism.

| Fact | Value |
|---|---|
| Task | Compute a + b, a·b (a, b in 0..96) or x² mod 97, with the numbers written in one of six formats |
| Input → output example | a = 96, b = 2, addition: (96 + 2) mod 97 = 1 |
| Model | 4-layer transformer, 4 heads per layer, width 256, feed-forward width 512 with ReLU; no LayerNorm or positional encoding; all weights trained |
| Rungs and seeds | Hand-built: 17 networks, exact on all 9409 inputs; from scratch: 10 seeds in each of 6 final cells, after 3-seed rounds |
| Closed form status | Partial (output readout only: substitution accuracy 1.0 on all 9216 nonzero pairs, including training pairs, for 8 seeds with native test ≥ .98; true discrete logarithms supplied externally; internal circuit and complement unverified) |
| Status | Partly solved (token multiplication 6/10 ≥ .99 at fraction .9, 0/10 at .3; token addition 10/10 ≥ .99; digit multiplication memorises); last run 2026-11-08 |
| Weights | [Hugging Face: modular-arithmetic-formats](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/modular-arithmetic-formats) (earlier round: [modular-arithmetic-mod-97](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/modular-arithmetic-mod-97)) |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

Read next: [Results](#4-what-training-finds) and [verification](#5-mechanism-and-verification), then [mod-59 addition](modular-addition-mlp.md) for a simpler Fourier clock, or [multiplication across bases](multiplication-base-order.md) for digit-order comparisons.

## The circuit at a glance

Figure S1 compares scalar exponents in the hand-built lane with circular codes in the trained lane; the dashed trained arithmetic is unrecovered.

<a href="../figures/circuits/modular-arithmetic-mod-97.svg"><img src="../figures/circuits/modular-arithmetic-mod-97.svg" width="640" alt="Alternative hand-built and trained-from-scratch lanes have separate residual streams. Hand-built embeddings turn tokens 5 and 25 into scalar exponents 1 and 2, add to 3 and look up 28. Trained embeddings use circular exponent codes; their internal arithmetic is dashed, not verified, and 28 is only a task target."></a>

*Figure S1 shows circular exponent codes in all ten final token multipliers at training fraction .9, differing from the hand-built scalar code; the separate Partial output-only surrogate needs true external logarithms rather than recovered activations, leaving the internal arithmetic and complement unverified ([evidence](#5-mechanism-and-verification)).*

## 1. Task and data

Three operations were studied: a + b and a·b with a, b in 0..96, and x² with x in 0..9408, each mod 97. Every operation has 9409 inputs,
all of which were used. The answer is r = S mod 97, where S is the exact sum, product or square, and an output is correct only on exact
match. For example, with a = 96 and b = 2, addition gives (96 + 2) mod 97 = 1. The residue is the remainder; a discrete logarithm of
a nonzero residue is its exponent as a power of a chosen generator, here 5
([glossary](../GLOSSARY.md#analysis-measures)). A clock is a circular number code. These terms describe representations, not success
criteria: Table 2 separately counts exact outputs and those with held-out accuracy ≥ .99. Table 1 lists the six number formats.

*Table 1. Number formats.*

| Format | Numbers written as |
|---|---|
| `tok` | one token per residue |
| `b2l` / `b2m` | binary, least / most significant bit first |
| `b10` | decimal digits |
| `b16` | hexadecimal digits |
| `s10` | decimal [scratchpad](../GLOSSARY.md#task-formats), writing intermediate results: S `;` S div 97 `;` r |

The split was drawn per seed at train fraction .3 (2823 training inputs) or .9 (8468 training, 941 test inputs). In the cell names,
x3 means three times the data, that is, fraction .9 instead of .3; "mul", "add" and "sq" are multiplication, addition and squaring.
Length generalisation is not defined, because the domain is fixed.

## 2. Model

The shared architecture and instrumentation are described in [METHODS.md](../METHODS.md). Two [rungs](../GLOSSARY.md#the-three-rungs)
were run: HAND (every weight set by formula) and TRAIN (all weights trained from scratch).

### 2.1 Architecture

The trained networks follow the [GPT-NeoX decoder layout](../GLOSSARY.md#standard-machine-learning-terms), with 4 layers x 4
attention heads (readers of earlier positions), each of width d_head 64. The residual stream, the running vector each layer updates,
has width d_model 256. Each feed-forward block (MLP) has 512 ReLU units, which set negative inputs to zero. There is no LayerNorm, no positional encoding (NoPE), and
the readout weights are separate from the input embedding. The vocabulary has 103 tokens: BOS, 97 residues or digits, and `+ * ^ = ;`. All parameters are
trained. The hand-built networks are 8 layers x 8 heads with d_model 512 and the same vocabulary, with every weight set by formula.
In probe locations, L0 is the embedding.

### 2.2 Frozen parts and trained part

No part is frozen: TRAIN trains every weight, and HAND trains none.

### 2.3 Training

All runs used AdamW with learning rate 1e-3, weight decay 1.0 (.1 in the `x3wd` cells), betas (.9, .98), a warmup of 1000 steps,
a cosine schedule over 40k steps, gradient clipping at 1 and batch 512, in TF32. A run stopped early at test accuracy 1.0 or after a
5k-step plateau. The best-training and final snapshots were kept, with checkpoints every 2500 steps. Training ran on consumer and rented
GPUs; diagnostics and substitutions ran on CPU in fp32. The final run trained 10 seeds in each of six cells: mul tok x3 (the headline
cell), add tok x3, add b16, add s10, mul b2l and sq b16.

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/modular-arithmetic-formats](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/modular-arithmetic-formats)
(earlier round: [model archive](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/modular-arithmetic-mod-97)).

### 2.5 Archive

The folder `modular-arithmetic-formats/` holds `hand_{mul,add}_tok` (HAND state dicts),
`train_{mul,add}_tok_f90_{best,final}_s<seed>` (multiplication seeds 0, 1, 2, 4, 7, 9; addition seeds 0–2), `split_{mul,add}_tok`
(permutation, number of training inputs, seeds), `config.json` and `manifest.json` (seed, rung, circuit type, fp32 accuracy, sha256).

## 3. Hand-built solution

The 17 hand-built networks use one algorithm per operation and format (Table F1). All 17 are exact on all 9409 inputs, with a logit
margin ≥ 23.6.

Token addition is a pizza, the variant of the Fourier clock described by [Zhong et al. (2023)](#references) in which the answer is read
from the sum of the two angle vectors, at frequencies 17, 25, 32 and 43. Token multiplication is a discrete-log adder. Every nonzero
residue mod 97 is a power of 5, v = 5^l, so multiplying residues is adding their discrete logs l = log_5 v mod 96. The embedding writes
log_5 v and [v = 0], one averaging head adds the two logs, and an MLP looks up 5^(la+lb mod 96), where la and lb are the logs of a and
b.

In the digit formats, index heads fetch the digits and form place-value sums. Addition is a lookup on A+B, the sum of the two
operands. Multiplication builds thermometer partial products and then compares and subtracts, [P >= 97j], to give the quotient q and
the remainder r. Squaring compares and subtracts on x and then looks up the square.

Table F1 records alternative reference algorithms for each number format. The token multiplier adds exponents, while the token adder reads summed circle vectors; neither is a claim about the trained network's internal arithmetic.

<details>
<summary>Technical hand-built exponent, circle and digit routes (Table F1)</summary>

*Table F1. The hand-built networks. In token multiplication the embedding writes the discrete log, one head adds the logs and the feed-forward block (MLP) looks up the power of 5; token addition uses the “pizza” circle-vector readout. Digit formats fetch digits, sum place values and compare and subtract. No part is trained in these networks; the empirical networks are trained from scratch in full.*

| Alternative reference route | Inputs, computation and output |
|---|---|
| Token multiplication: embedding | Writes log_5 v, the exponent of 5 giving nonzero value v, and the v = 0 flag. |
| Token multiplication: head and MLP | An averaging head adds la + lb. The MLP looks up 5^(la + lb mod 96), giving r = a·b mod 97; zero is handled separately. |
| Token addition | Embeds angles at frequencies 17, 25, 32 and 43. The “pizza” readout obtains remainder r from the sum of the angle vectors. |
| Digit formats: shared readers | Index heads fetch digits and form place-value sums for operands A and B. |
| Digit addition | Looks up A+B. This is an alternative to the product route, not a later multiplication stage. |
| Digit multiplication | Thermometer partial products form P; comparisons P ≥ 97j and subtraction give quotient q and remainder r. |
| Squaring | Compares and subtracts on x, then looks up the square. |

</details>

## 4. What training finds

Table 2 summarises the final cells. Exact match is on held-out inputs; "≥ .99" counts seeds whose test exact match reached .99.

*Table 2. Results of the final run (10 seeds per cell; seeds named s0..s9).*

| cell (final, 10 seeds) | Seeds at exact 1.0 | Seeds at ≥ .99 | test range | circuit types (seeds) |
|---|---|---|---|---|
| mul tok x3 | 4/10 | 6/10 | .425-1 | K-high log clock 8 (s0,2-8), K-low ramp-clock 2 (s1,s9) |
| add tok x3 | 6/10 | 10/10 | .998-1 | Fourier clock on Z_97 10 |
| add b16 | 0/10 | 9/10 | .990-.999 | r read at L4 (DAS .47-.72): r-only 9, S+r 1 |
| add s10 | 7/10 | 9/10 | .988-1 | q-transfer route (3/3 measured) |
| mul b2l, sq b16 (each) | 0/10 | 0/10 | .013-.077 | memorised (failed); 3-seed rounds: mul b10/b16 memorise at every data/wd |

The rounded .990 lower bound for hexadecimal addition and .988 for the decimal scratchpad hide threshold failures: the source
records .9898 for seed 1 and .9875 for seed 7, respectively. Both are below .99, which is why each row has 9/10 at that threshold.
These more precise extrema and the legacy rounded ranges are retained separately.

Key: both success columns count seeds out of the same ten, at exact match 1.0 and at exact match ≥ .99 respectively (the first row has 4 exact seeds and 6 seeds at ≥ .99, as in the 6/10 above). The number-format labels and x3 data label are defined in Section 1. K-high and K-low = the two clock types as labelled in the run records, a multi-frequency clock and a ramp-like clock; DAS = [distributed alignment
search](../GLOSSARY.md#analysis-measures), testing variable swaps in learned rotated directions; r-only, S+r = which of the HAND variables the network reads at layer 4; q-transfer = a route
through the quotient q; data/wd = training fraction and weight decay.

Token formats are the only ones in which a network carries a circular code of a or r. Token multiplication at fraction .9 embeds
the residues as circles in discrete-log order in 10/10 seeds: 8 seeds form a multi-frequency clock (K-high) and 2 a ramp-like clock
with few frequencies (K-low). Six of them reach test ≥ .99. In 6 seeds the step at which test exact match reached .9 (recorded as
te90; reading of the label) was 21k–30.5k. No HAND variable reaches probe R² .9 at any checkpoint. Token addition is a [Fourier
clock](../GLOSSARY.md#analysis-measures) on Z_97 in 10/10 seeds.

In the digit formats no network forms circles. The solved addition networks read r out as a categorical code at layer 4: in add b16 9
seeds read r only and 1 reads S and r; in add s10 the measured networks follow a q-transfer route (3/3 measured). Binary
multiplication and hexadecimal squaring memorise (test .013–.077), and in the 3-seed rounds decimal and hexadecimal multiplication
memorised at every training fraction and weight decay tried.

Per-cell diagnostics (used heads, probes, effective rank, attention divergence) are in Appendix B.

## 5. Mechanism and verification

For token multiplication, the partial output-only closed form fits cosine and sine waves to differences of externally supplied true
discrete logs. It is a sufficient output readout, not a recovered internal circuit or a complement-verified mechanism:

logit(c) = Σ_{k ∈ K} α_k cos(2πk d/96) + β_k sin(2πk d/96),  d = (log5 a + log5 b − log5 c) mod 96, c ≠ 0,

with answer 0 iff a or b = 0. K is the set of the top-m frequencies of the network's mean logit against d, and α, β are fitted on the
training rows. Substitution gives task accuracy 1.0 on all 9216 nonzero pairs, including training pairs, for every seed whose native
held-out score is ≥ .98 (8 seeds). The smallest m per final seed is s4 1, s5 1, s2 2, s8 2, s9 3, and s0/s1/s3 6. Its agreement with
the network equals the network's
accuracy (≥ .99 on the six seeds that reach ≥ .99). Seed 6 (.838) agrees on .837 despite formula task accuracy 1.0 at m = 4: it idealises the task rather than explaining that seed's
errors. Seed 7 has no passing fit (m = 6 gives task accuracy .508 and agreement .31). For token addition the same form on
Z_97 with d = (a + b − c) mod 97 is proposed; its substitution was not yet measured.

The closed form is output-level only: it uses the true discrete log, not a variable read from the network. Internally, replacing MLP3
by a lookup T[ls] on the summed log ls keeps accuracy in seeds 0, 2, 4, 5, 6 and 9. No [resample ablation of the
complement](../GLOSSARY.md#verifying-a-circuit) was reported, so the status is partial. A complement test would replace the circuitry
outside a proposed mechanism with its values on other inputs. The externally supplied logarithms and the trained network's internal
path must remain distinct (Table F2).

Table F2 separates the native network from the external-log surrogate. The surrogate receives externally supplied logarithms of the same inputs; its scores are fitted and compared with the native output scores. These are separate evaluations, not successive network stages.

<details>
<summary>Technical native network versus externally supplied logarithms (Table F2)</summary>

*Table F2. The trained network and its separate output surrogate. The embeddings, transformer and native readout are trained; the true-log step is supplied externally. Status: partial, output readout only, with no recovered internal circuit or complement test. Substitution accuracy 1.0 on all 9216 nonzero pairs in 8 eligible seeds does not establish faithfulness to the failed seeds' errors.*

| Separate route or test | Information source, computation and limit |
|---|---|
| Native token input | Tokens a and b enter learned embeddings with circles in discrete-log order. |
| Native transformer and readout; trained | Produce native output scores. No complete internal closed form is recovered. |
| External surrogate keys | True discrete logarithms of a, b and candidate c are supplied externally, not recovered from activations. |
| External surrogate difference | d = (log5 a + log5 b − log5 c) mod 96 feeds the fitted cosine-and-sine output readout. |
| External surrogate answer | Candidate c is scored by that fitted readout; the zero-input rule is handled separately. |
| Scope of substitution | Accuracy 1.0 covers all 9216 nonzero pairs, including training pairs, in 8 eligible seeds. It does not reproduce all failed seeds' errors or verify native arithmetic or the complement. |

</details>

The hand-built token multiplier and the trained networks share the
isomorphism Z_97* ≅ Z_96, meaning that nonzero residues under multiplication correspond to exponents added modulo 96, but not the code: the hand-built network uses a ramp and a lookup, the trained networks use circles. The
trained code is the clock of [Nanda et al. (2023)](#references) and [Zhong et al. (2023)](#references) moved by the discrete log; the
clock solution of modular arithmetic was also derived analytically by [Gromov (2023)](#references). Digit formats never form circles;
the solved digit networks read r out as a categorical code at layer 4.

## 6. Limitations and open questions

The Fourier label was checked on all 17 hand-built networks (token addition F, the other 16 L, as recorded). The closed form is
output-level with the oracle log, meaning that true logarithms are supplied rather than read from the network. It is not an
accepted full internal mechanism; there is no complement certificate. The seeds differ in their weights (loss barrier, LMC, 12–19): linear mode connectivity (LMC) measures the rise in loss when
interpolating between two seeds' weights ([glossary](../GLOSSARY.md#analysis-measures)). The run-1 networks for add b16 and add
s10 were lost (only their logged numbers remain); the final-run networks exist. Neuron counts and an alignment of TRAIN with HAND
(`align`) were not measured, and the substitution of the addition closed form is open.

## 7. Prediction record

The prediction was recorded before training: token addition would be Fourier, token multiplication Fourier in log coordinates, the
digit formats would carry no p-periodic code, and s10 would beat b10. It held, except that fraction .3 was below the grokking data
threshold for token addition (3/10) and token multiplication (0/10). Three times the data (fraction .9) improved both: addition
reached held-out accuracy ≥ .99 in 10/10 seeds, while multiplication reached it in 6/10, not every seed. That representational
priors control the delay of grokking was studied by [Howe (2026)](#references).

## Appendix A. Weights excerpt and reading

Table A1 gives the clock frequencies K and the amplitudes √(α² + β²) per seed, from the final snapshots (`clock_fit`). K is the frequency set of the smallest m per seed given in Section 5; the K-high and K-low labels are the run records' circuit types (Table 2 key), so a K-high seed can have one frequency here.

*Table A1. Clock frequencies and amplitudes per seed, token multiplication.*

| seed | frequencies k | amplitudes |
|---|---|---|
| s4 | 43 | 23.5 |
| s5 | 7 | 39.2 |
| s2 | 38, 29 | 16.5, 8.3 |
| s8 | 17, 16 | 9.5, 6.4 |
| s9 | 6, 2, 9 | 28.8, 9.9, 9.3 |
| s1 | 6, 4, 8, 2, 10, 1 | 52.4 ... 7.4 |

These parameters belong to a partial output readout using true logarithms, not a verified internal mechanism. The top log-order
frequencies of the embeddings are computed by `excerpts.py modp` (DFT over the residues sorted by log_5, power
share). A discrete Fourier transform (DFT) measures how much of a vector follows each sine and cosine frequency
([glossary](../GLOSSARY.md#analysis-measures)). The fitted output for each seed rotates log a and log b on a few circles and reads cos(k(la + lb − lc)).

## Appendix B. Detailed evidence

*Table B1. Diagnostics per cell (final snapshot, ranges over seeds).*

| cell | used heads + MLPs | best probe R² per HAND var @L | eff. rank (PR) vs HAND | attn JS | best / final test |
|---|---|---|---|---|---|
| mul tok x3 | 11-16 h + 4 M | A/B/S/q/r <= .12; la .62, lb .40, ls .78 (K-low) @3-4 | U vs 1.01 | .17-.37 | .425-1 / .425-1 |
| add tok x3 | 13-16 h + 4 M | A, B, S, q <= .04; r <= .31 @4 | 4.1-8.7 vs 4.41 | .26-.28 | .997-1 / .998-1 |
| add b16 | 9-16 h + 4 M | S .99, q .93, r .93 @4; A .88, B .80 | 3.1-6.4 vs 4.02 | .15-.26 | .993-.999 / .990-.999 |
| add s10 | 7-8 h + 4 M | S .86, q .89, r .57 @4 | 5.0-7.1 vs 7.04 | .18-.24 | .997-1 / .988-1 |
| mul b16 (r1/x3) | 0-4 h | A .80, B .72, S .80 @4; r .49 @0 (chance) | 37-43 vs 1 | .19-.26 | .02-.25 |

Key: L0 = embedding; h = heads, M = MLPs (neuron counts unmeasured, whole MLPs only); probe R² per HAND variable at the best layer
(@L), with A, B the operands, S the exact result, q the quotient, r the remainder, and la, lb, ls the discrete logs of a, b and their
sum; PR = [participation ratio](../GLOSSARY.md#analysis-measures) of the residual stream against HAND's (U as recorded); attn JS =
attention divergence from the closest HAND head; r1/x3 = run 1 and the three-times-data run.

## References

- Gromov, A. (2023). Grokking modular arithmetic. *arXiv preprint*. https://doi.org/10.48550/arxiv.2301.02679 (analytic clock
  solution)
- Howe, G. L. (2026). Structure-specific representational priors causally control the grokking delay. *arXiv preprint*.
  https://doi.org/10.48550/arxiv.2607.04333 (grokking delay)
- Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic
  interpretability. *ICLR 2023 (arXiv preprint)*. https://doi.org/10.48550/arxiv.2301.05217 (the modular-addition clock)
- Zhong, Z., Liu, Z., Tegmark, M., & Andreas, J. (2023). The clock and the pizza: Two stories in mechanistic explanation of neural
  networks. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)* (conference paper). https://doi.org/10.52202/075280-1186, arXiv
  https://doi.org/10.48550/arxiv.2306.17844 (clock and pizza algorithms)
