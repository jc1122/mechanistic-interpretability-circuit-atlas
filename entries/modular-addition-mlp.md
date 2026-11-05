# Modular addition, (x + y) mod 59: MLP on a frozen menu

**Task.** Prompt `SUM: x y\nANSWER: `, x, y in 0..58, answer (x + y) mod 59 read at the answer position. Train fraction .4 of all 59² pairs.
**Status.** 2026-11-04. 10 seeds (+ 6 control runs). Trained part: one MLP layer, 256 ReLU neurons. AdamW lr 1e-3, weight decay 1.0,
betas (.9, .98), full batch, 300k steps.

## Summary
All 10 networks converge to the same circuit: the Fourier "clock" solution of a wide MLP (Gromov, 2023). Each neuron uses one
frequency and the output weights obey a phase rule. The per-neuron closed form, substituted for the trained layer, scores
.996–1.000, higher than the networks themselves (.94–.98), whose residual errors are memorization not yet removed by weight decay.

## Setup
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/modular-addition-mlp](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/modular-addition-mlp) (hand-built base in `_shared/m2`).
Full architecture, optimiser and instrumentation: [METHODS.md](../METHODS.md). Task-specific conditions:
A frozen, hand-built transformer (6 layers, every weight set by formula) reads the two numbers and places a "menu" of encodings
at the answer position. Each menu block is centered and scaled to rms 1:
- scalar x/100, y/100;
- one-hot x, y (59 + 59 dims);
- thermometer [x ≥ k], [y ≥ k] (58 + 58 dims);
- distractors: a copy pointer, a position value, a counter and a bracket balance (the last two are constant on this prompt).

Only the MLP on top is trained. The question is which encodings it reads and which algorithm it builds from them.

- Menu width 310; hidden width 256 ReLU, 59 outputs; AdamW lr 1e-3, weight decay 1.0, betas (.9, .98), full batch, 300,000 steps, no early
  stop, evaluation every 500 steps. Train fraction .4 of the 3481 pairs (1392 training pairs); seed = initialisation and split.
- Controls: random frozen base and scalar-only menu, 3 seeds each. Runs were on CPU, 14-29 minutes per network.
- Grok steps are the first evaluation (every 500 steps) with test accuracy >= 50% / 90%.

**Reproduce** (from `code/`; one network per call, add seeds as needed):
```
python run.py modadd --seeds 0 1 2 3 4 5 6 7 8 9                 # menu, 10 seeds (headline: neuron_formula_acc .996-1.000)
python run.py modadd --kind random --seeds 0 1 2                  # controls
python run.py modadd --kind scalar --seeds 0 1 2
```
Each run prints the learning-curve steps, the circuit type, dd, the phase error and the substitution accuracies, and writes `out/*.json`.

## Architecture as used
- **Frozen base, hand-built m2** (no training): GPT-NeoX-shaped decoder, 6 layers, d_model 528, 4 heads (d_head 132), MLP width 512, exact
  GELU, fixed-gain LayerNorm (eps 1e-5), no positional embedding and no rotary, sequential residual, character vocabulary of 70 ids.
  10,041,408 parameters, of which 13,443 are non-zero; read at the last prompt position, residual before the final LayerNorm.
  `code/m2.py` rebuilds it bit for bit (safetensors sha256 `3dbb5226…cd9a6a`).
- **Frozen menu** (formula, no parameters learned): m2's number readers give NA = x/100, NB = y/100 (exact to 2e-4). Blocks, index
  ranges of the 310-wide input: sx 0, sy 1 (scalars), ox 2–60, oy 61–119 (one-hot, `step(100 NA; k − ½) − step(100 NA; k + ½)`), tx
  120–177, ty 178–235 (thermometer `step(100 NA; k − ½)`, k = 1..58), count 236 and brackets 237–238 (constant here, so 0), copy
  pointer 239–308, position 309. Each block centred and divided by its rms row norm.
- **Trained stage:** `logits = W2 ReLU(W1 x + b1) + b2`, W1 [256, 310], b1 [256], W2 [59, 256], b2 [59] (94,779 parameters); only this
  stage is trained, from the default PyTorch initialisation.
- **Archive.** Per network one safetensors file (`W1`, `b1`, `W2`, `b2`, `test_idx`), plus `config.json`, `inputs.safetensors` (the frozen
  menu rows X, the integers and the labels) and `manifest.json` (seed, rung, train fraction, accuracy, circuit type, sha256). Every file
  was re-loaded and its forward pass reproduces the recorded test accuracy exactly. Public hosting of the weights is pending. 10 networks (`s0_f40` … `s9_f40`, train fraction .4).

## Prediction before training
No prediction was recorded before these runs (the entry predates this section). Retrospective, labelled as such: the published
analytic solution of a wide MLP (Gromov, 2023) predicts one frequency per neuron and the phase rule θ = φ + ψ; this held in 10/10
networks. Not predicted: which encodings training reads (one-hot plus thermometer, scalar unused) and the spread over 22–27 frequencies.

## What training finds
- **Learning curve (grokking).** Train accuracy 99% at step 500. Test accuracy stays at chance (.02) to step 20k, reaches 50% at
  49–69k and 90% at 105–295k, and ends at .91–.97. Generalization is delayed but gradual, not a sharp jump. The weight norm rises from
  10 to 101 during memorization, then decays to 85.
- **Circuit type.** 10/10 clock (no pizza, no lookup table): the correct logit does not depend on x − y.
- **Encodings read (test accuracy after mean-ablation; full network .94):**

  | Block | Ablated | Kept alone (pair) |
  |---|---|---|
  | one-hot x / y | .08 / .02 | .65–.79 |
  | thermometer x / y | .89 / .87 | .08–.11 |
  | scalar x / y | .93 / .94 | .02–.03 |
  | copy / position distractors | .86 / .89 | |

  One-hot is necessary and nearly sufficient; the thermometer supplies the low frequencies k = 1–2 (73–77% of its contribution at k = 1);
  the scalar is unused.
- **Frequencies.** Neurons spread over 22–27 of the 29 frequencies in every network (no sparse key set). Frequencies 1 and 2 are in the
  logit top-5 of all 10 networks; the rest are seed-specific (mean Jaccard overlap of top-5 sets .37).

## Closed form
For neuron n with frequency k_n (ω_n = 2πk_n/59):

  h_n(x, y) = ReLU(c_n + a_n cos(ω_n x + φ_n) + b_n cos(ω_n y + ψ_n))
  logit_c = Σ_n A_n cos(ω_n c + φ_n + ψ_n) · h_n(x, y)

The phase rule θ_n = φ_n + ψ_n holds with median error .04–.05 rad. The convex ReLU turns the sum of cosines into a cross term
cos(ω_n(x + y − c)), which peaks at c = x + y mod 59. Fitting k, φ, ψ from the DFTs of the input weights and substituting the formula
for the trained layer gives .996–1.000 accuracy on all pairs (agreement with the network .94–.98).

## Weights excerpt and reading
Seed 0, the 8 neurons with the largest output-weight norm. E_x(x) = pre(x, 0) and E_y(y) = pre(0, y) (centred); k = the dominant
DFT frequency of E_x; (a, φ), (b, ψ), (A, θ) = amplitude and phase at k of E_x, E_y and of the neuron's output column W2[:, n] over c.

| neuron | k | a, φ | b, ψ | A, θ | θ − φ − ψ |
|---|---|---|---|---|---|
| 178 | 29 | .60, −1.51 | .61, +.64 | .74, −.94 | −.07 |
| 149 | 22 | .59, +2.13 | .60, +2.60 | .73, −1.56 | −.01 |
| 55 | 25 | .59, −.58 | .59, −1.85 | .73, −2.37 | +.06 |
| 42 | 5 | .74, +.42 | .71, +.10 | .72, +.50 | −.02 |
| 76 | 12 | .59, −.33 | .59, −2.51 | .73, −2.85 | −.02 |
| 217 | 28 | .59, +3.06 | .59, +.71 | .71, −2.53 | −.02 |
| 9 | 13 | .61, −2.60 | .61, −1.05 | .71, +2.59 | −.04 |
| 231 | 29 | .58, +1.85 | .57, −2.76 | .71, −.88 | +.03 |

Median |θ − φ − ψ| over the live neurons: .047 rad. One input curve against its fit (neuron 42, x = 0..11):
```
E_x(x)                 .60  .60  .22 -.18 -.43 -.71 -.65 -.43  .15  .63  .60  .60
.74 cos(2π·5x/59+.42)  .67  .43  .06 -.32 -.61 -.74 -.66 -.39 -.02  .35  .63  .74
```
Reading. Every neuron is tuned to one frequency with nearly equal input amplitudes for x and y (a ≈ b ≈ .6), so its pre-activation is
c_n + .6 cos(ωx + φ) + .6 cos(ωy + ψ). The ReLU's quadratic part contains the product of the two cosines, i.e. a term cos(ω(x + y) + φ + ψ).
The output column is A cos(ωc + θ) with θ = φ + ψ, so the product term contributes ∝ cos(ω(x + y − c)) to logit c, maximal at
c = x + y mod 59; summing over neurons with different k cancels every other c. The measured E_x is a cosine flattened at its crests
(x = 0, 1, 10, 11), which the single-cosine closed form ignores at no cost in accuracy.

## Comparison
Matches the analytic MLP solution of Gromov (2023) and the clock algorithm of Zhong et al. (2023). Unlike the sparse key-frequency
transformer circuit, a wide MLP uses many frequencies. New here: given a menu of encodings, training consistently selects one-hot
plus a low-frequency thermometer helper and ignores the scalar.

## Controls and limits
- Random frozen base (same shape, random weights): memorizes, test stays flat at .25, no phase rule.
- Scalar-only menu: cannot fit the training set (train .47–.50).
- One modulus, one width, one hyperparameter set; 300k steps did not reach 99% test accuracy (longer training not run).

## References
- Gromov, A. (2023). Grokking modular arithmetic. https://doi.org/10.48550/arxiv.2301.02679
- Zhong, Z., Liu, Z., Tegmark, M., & Andreas, J. (2023). The clock and the pizza. https://doi.org/10.52202/075280-1186
- Chughtai, B., Chan, L., & Nanda, N. (2023). A toy model of universality. https://doi.org/10.48550/arxiv.2302.03025
- Nanda, N., et al. (2023). Progress measures for grokking via mechanistic interpretability. https://doi.org/10.48550/arxiv.2301.05217 (original modular-addition Fourier circuit)
