# Open-bracket count: MLP on a frozen menu

**Task.** Prompt `VALID_BRACKETS: ()\nANSWER: YES\nVALID_BRACKETS: <line>\nANSWER: `, answer the number of `(` = (L + b)/2 (46 classes),
where b is the final depth (#`(` − #`)`) and L the length, L = 1..45, b = −L..L in steps of 2 (1080 cells, one shuffled line each).
**Status.** 2026-11-04. 30 networks (10 seeds × train fractions .4/.6/.8) + 18 control runs; setup as in [count mod 5](count-mod-5.md),
with the frozen network supplying the depth ratio b/(L+1) and L.

## Setup
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/open-bracket-count](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/open-bracket-count) (hand-built base in `_shared/m2`).
Frozen m2 and menu as in [count mod 5](count-mod-5.md) but on the bracket prompt: u = b (depth, from the bracket head's b/(L+1)), v = L; menu
width 416 (the b thermometer is a second step stage gated by the one-hot of L). Hidden width 256, 46 outputs, 200,000 steps, evaluation
every 500 steps, 10 seeds at train fractions .4 / .6 / .8, controls 3 seeds each; see [METHODS.md](../METHODS.md).

**Reproduce** (from `code/`; the clock sum appears at fraction .8):
```
python run.py atlas --task open_count --frac 0.8 --seeds 0 1 2 3 4 5 6 7 8 9    # also --frac 0.4 and 0.6
python run.py atlas --task open_count --kind random --seeds 0 1 2 --probe
python run.py atlas --task open_count --kind scalar --seeds 0 1 2
python run.py table out/open_count_*.json
```
Headline per network: `final`, `t50`, `form_clock` / `form_clock_agree`, `clock_k_top`.

## Summary
The only task in this set that groks, and only with 80% of cells in training: 10/10 networks build a low-frequency trigonometric sum
(frequencies k = 1–3 of 92), a Gromov-style clock on a non-modular domain, plus a
small parity channel at the Nyquist frequency. With 40% or 60% training data all networks memorize.

## Architecture as used
- **Frozen base, hand-built m2** (no training): GPT-NeoX-shaped decoder, 6 layers, d_model 528, 4 heads (d_head 132), MLP width 512, exact
  GELU, fixed-gain LayerNorm (eps 1e-5), no positional embedding and no rotary, sequential residual, character vocabulary of 70 ids.
  10,041,408 parameters, of which 13,443 are non-zero; read at the last prompt position, residual before the final LayerNorm.
  `code/m2.py` rebuilds it bit for bit (safetensors sha256 `3dbb5226…cd9a6a`).
- **Frozen menu** (formula) on m2's depth ratio BAL = b/(L+1) (layer 5 head 0: uniform attention over the bracket tokens and the
  newline of the current line, value +1 for `(` and −1 for `)`) and R (gives L). Blocks, index ranges of the 416-wide input: su 0 (BAL),
  sv 1 (R), ou 2–92 (one-hot b = −45..45), ov 93–138 (one-hot L = 0..45), tu 139–228 ([b ≥ k], k = −44..45), tv 229–273 ([L ≥ k]),
  distractors count 274 and neg 275 (constant here, so 0), copy pointer 276–345, last letter 346–415. The b thermometer is gated by the
  one-hot of L: [b ≥ k] = Σ_L step(BAL + 100(onehot_L − 1); (k − ½)/(L+1), 40(L+1)).
- **Trained stage:** `logits = W2 ReLU(W1 x + b1) + b2`, W1 [256, 416], b1 [256], W2 [46, 256], b2 [46] (118,574 parameters).
- **Archive.** Per network one safetensors file (`W1`, `b1`, `W2`, `b2`, `test_idx`), plus `config.json`, `inputs.safetensors` (the frozen
  menu rows X, the integers and the labels) and `manifest.json` (seed, rung, train fraction, accuracy, circuit type, sha256). Every file
  was re-loaded and its forward pass reproduces the recorded test accuracy exactly. Public hosting of the weights is pending. 30 networks (`s<seed>_f<40|60|80>`; the clock sum is in the `_f80` files).

## Prediction before training
No prediction was recorded before these runs (the entry predates this section). Retrospective, labelled as such: the nearest
hand-built design adds the two integers linearly and reads the sum with step staircases; it predicts a linear sum, which did not occur
(a straight-line form scores .09–.11). The parity channel (below) was not predicted.

## What training finds
- **Learning curve.** Train 100% at step 500 everywhere. Fraction .4: final test .006–.019 (below chance; wrong answers are neighbouring
  cells' answers). Fraction .6: .04–.11. Fraction .8: test .02–.05 at 10k steps, 50% at 20.5k–28.5k, then flat at .78–.88 from ~150k while
  the weight norm falls 80 → 73. None reaches 90%.
- **Circuit types.** Low-frequency clock sum: 10/10 at fraction .8. Memorization: 20/20 at .4 and .6.
- **Encodings read (fraction .8):** all three encodings are used together, the one-hot as the main input; distractors have no effect.

## Closed form
pre_n = c_n + a_n cos(ω_n b/2 + φ_n) + a′_n cos(ω_n L/2 + ψ_n), ω_n = 2πk_n/92, k_n ∈ 1..45 chosen per neuron (most often 1–3), with the trained output weights:
accuracy .91–.93 (networks .96–.98), agreement .88–.92. The pre-activation separates into a b part plus an L part (substitution
.95–.98) but is not linear (a straight-line form scores .09–.11). The low-k wavelengths (31–92) exceed half the domain, so those cosines do not alias.

## Weights excerpt and reading
Seed 0, fraction .8. Each neuron's pre-activation is split into c_n + E_b(b) + E_L(L) and fitted by one frequency k ∈ 1..45 per neuron
(ω = 2πk/92, E_b ≈ a cos(ωb/2 + φ), E_L ≈ a′ cos(ωL/2 + ψ)). Neurons per best k: k = 1: 58, k = 2: 43, k = 3: 14, k = 4: 7; the other 134
spread over k = 6..45 (k = 45: 10, k = 23: 11). The strongest low-k neurons (W2 column norm):

| neuron | k | a, φ | a′, ψ | fit R² | W2 column, classes 0, 9, 18, 27, 36, 45 |
|---|---|---|---|---|---|
| 216 | 1 | .49, −1.75 | .05, −2.52 | .30 | .17, −.68, .21, .56, −.41, .00 |
| 231 | 1 | .97, −1.81 | .86, −2.36 | .47 | .45, .55, .56, −.60, .29, −.01 |
| 47 | 1 | .27, +1.70 | .21, +.78 | .22 | −.17, −.60, −.51, .39, −.36, .00 |
| 171 | 2 | .28, +.67 | .37, +3.12 | .30 | −.42, .26, .11, .62, −.37, −.01 |

A **parity channel**, found in the archived weights (10/10 networks at fraction .8): 7–10 neurons whose output column alternates with the
class parity (seed 0, neuron 58, classes 0..9: −.67, .75, −.72, .70, −.73, .74, −.74, .76, −.75, .73). Their pre-activations are fitted by
cos/sin(πb/2), cos/sin(πL/2) and their products with median R² .96–.99. Removing them lowers the all-cell accuracy by .01–.05 (seed 0:
.956 → .934); the networks get the parity of the answer right on .98–.99 of cells.

Reading. The low-k neurons are smooth, broad bumps over b and L (single-cosine R² only .15–.47, so "one cosine per neuron" is a coarse
description); with output columns that vary slowly over the 46 classes, their sum locates the answer (L + b)/2 (without the parity neurons
the networks still score .92–.96 on all cells). The parity neurons resolve the remaining neighbour ambiguity: (−1)^((L+b)/2) = cos(π(L+b)/2) = cos(πL/2)cos(πb/2) − sin(πL/2)sin(πb/2),
a product of Nyquist terms that the ReLU forms, and their alternating output column adds ± the parity to every class.

## Comparison
No hand-built open-count brick was given. The nearest hand-built design forms the sum linearly in the residual and reads it with step
staircases; the trained stage never forms the sum linearly and builds a trigonometric sum instead. Modular addition (x + y mod 59) used
22–27 frequencies; this non-modular task uses only k = 1–3.

## Controls and limits
- Random frozen base: .57–.60 / .72–.77 / .81–.86 (fractions .4 / .6 / .8), better than the menu at .4 and .6. Its residual holds b and L
  linearly (ridge R² 1.000).
- Scalar-only menu: .35–.56.
- Open: is a 90% clock reachable with more data or lower weight decay (more steps do not help: accuracy and norm are flat after ~150k)?

## References
- Gromov, A. (2023). Grokking modular arithmetic. https://doi.org/10.48550/arxiv.2301.02679
- Yao, S., et al. (2021). Self-attention networks can process bounded hierarchical languages. https://doi.org/10.48550/arxiv.2105.11115 (bounded Dyck)
- Sharma et al. (2026). Dissociating decodability and causal use in bracket-sequence transformers. https://doi.org/10.48550/arxiv.2604.22128 (Dyck depth / top of stack)
- Liu, B., et al. (2022). Transformers learn shortcuts to automata. https://doi.org/10.48550/arxiv.2210.10749
