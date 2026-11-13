# Open-bracket count

Count the opening brackets in a line, even though its prompt asks about bracket validity. A frozen, hand-built transformer supplies
the length and final depth (opening minus closing brackets); only the small feed-forward network combining those features was
trained. With 80% of the 1080 depth/length cells used for training, all 10 seeds showed delayed improvement, but final held-out
accuracy was .78–.88 and none reached .90. Their responses suggest a clock-like sum of broad trigonometric features, helped by a
smaller channel that distinguishes odd from even answers. The task remains unsolved at the .90 held-out criterion; the fitted
cosine rule is a partial output approximation evaluated on all cells, including training cells, with no resample-complement test.

| Fact | Value |
|---|---|
| Task | Count the `(` in a line of up to 45 brackets (46 possible answers) |
| Input → output example | `(()` → `2` |
| Model | Frozen hand-built 6-layer transformer and 416-wide feature menu; trained: one feed-forward block (MLP), 256 hidden units, 46 outputs |
| Rungs and seeds | Hand-built base supplies b/(L+1) and L, not the answer; trained MLP, 10 seeds × training fractions .4/.6/.8 (30 networks), 200,000 steps each; 18 control runs; no from-scratch cell |
| Closed form status | Partial (10/10 at fraction .8: cosine-rule task accuracy .91–.93 and agreement .88–.92 on all 1080 cells, including training; native networks .96–.98; complement not tested) |
| Status | Unsolved (fraction .8, 0/10 with held-out accuracy ≥ .90; fractions .4 and .6, memorisation in 20/20); last run 2026-11-04 |
| Weights | [Hugging Face: open-bracket-count](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/open-bracket-count) (hand-built base in `_shared/m2`) |
| Reproduce | [Commands](#24-reproduction), run from `code/` |

Read next: [Results](#4-what-training-finds) and [verification](#5-mechanism-and-verification), then [compare counts](compare-counts.md) for a related counting task, or [mod-59 addition](modular-addition-mlp.md) for another periodic readout.

## 1. Task and data

Each prompt holds a fixed one-shot example followed by the query line:
`VALID_BRACKETS: ()\nANSWER: YES\nVALID_BRACKETS: <line>\nANSWER: `. The answer is the number of `(`, which is (L + b)/2, where b is the
final depth (#`(` − #`)`) and L the length; there are 46 classes. The data cover every cell with L = 1..45 and b = −L..L in steps of
2, which gives 1080 cells with one shuffled line each. The split into training and test data is over cells, with train fractions .4,
.6 and .8. Accuracy is the share of held-out cells whose single count answer is correct. For example, the line `(()` has count `2`; the
`VALID_BRACKETS` wording is inherited from the frozen base and does not change the counting label. No seed reached the .90 held-out
criterion. Formula substitutions in Section 5 instead score all 1080 cells, including those used for training.

## 2. Model

The shared architecture, optimiser and instrumentation are described in [METHODS.md](../METHODS.md); this section gives only what is
specific to the task. The [rungs](../GLOSSARY.md#the-three-rungs) distinguish a network set entirely by formula (HAND), a trained stage
on frozen hand-built features (PART), and training from scratch (TRAIN). This experiment used HAND and PART only. An MLP is a
feed-forward block, and ReLU is the activation that sets negative values to zero
([glossary](../GLOSSARY.md#standard-machine-learning-terms)).

### 2.1 Frozen hand-built network

The base is [m2](../GLOSSARY.md#hand-built-machinery), a GPT-NeoX-shaped decoder with 6 layers, d_model 528 and 4 heads (d_head 132),
MLP width 512, exact GELU, fixed-gain LayerNorm (eps 1e-5), no positional embedding and no rotary embedding, a sequential residual
stream and a character vocabulary of 70 ids. It has 10,041,408 parameters, of which 13,443 are non-zero, and nothing in it was
trained. It is read at the last prompt position, from the residual stream before the final LayerNorm. `code/m2.py` rebuilds it bit for
bit (safetensors sha256 `3dbb5226…cd9a6a`). On the bracket prompt it supplies the depth ratio BAL = b/(L+1) and the position scalar R,
which gives L (Section 3).

### 2.2 Frozen menu and trained part

The [menu](../GLOSSARY.md#hand-built-machinery) is a frozen layer, defined by formula, that places three encodings of each integer at
the answer position: a scalar (one number, here BAL for b and R for L), a one-hot code (one indicator per possible value), and a
thermometer code (indicators of the thresholds crossed, with k-th unit [n ≥ k]), together with distractor features. Here the two integers are u = b, the depth from the bracket head's b/(L+1), and v = L. Each block is centred and
divided by its root-mean-square (rms) row norm, a scale measure
([glossary](../GLOSSARY.md#standard-machine-learning-terms)). Table 1 lists the blocks of the 416-wide input.

*Table 1. Input blocks of the frozen menu (index ranges of the 416-wide input).*

| Block | Indices | Content |
|---|---|---|
| su | 0 | BAL |
| sv | 1 | R |
| ou | 2–92 | one-hot b = −45..45 |
| ov | 93–138 | one-hot L = 0..45 |
| tu | 139–228 | [b ≥ k], k = −44..45 |
| tv | 229–273 | [L ≥ k] |
| count (distractor) | 274 | constant here, so 0 |
| neg (distractor) | 275 | live prefix-negativity feature ANY; not constant |
| copy pointer (distractor) | 276–345 | |
| last letter (distractor) | 346–415 | |

The trained stage is one MLP, `logits = W2 ReLU(W1 x + b1) + b2`, with W1 [256, 416], b1 [256], W2 [46, 256] and b2 [46], which is
118,574 parameters. The logits are output scores; the largest score selects the count answer. ANY is the base's signal that a
bracket prefix has negative depth, a different property from the final count.
Its effect is small rather than identically zero: for seed 0 at fraction .8, held-out accuracy is 0.7824074029922485 and becomes
0.7546296119689941 when neg is ablated. These are the unrounded recorded values, not the ten-seed ranges below.

### 2.3 Training

Training followed [count mod 5](count-mod-5.md): hidden width 256, 46 outputs, 200,000 steps, an evaluation every 500 steps, and 10
seeds at each train fraction .4 / .6 / .8. Two controls, a random frozen base of the same shape and a menu holding only the scalars,
were run with 3 seeds each at each fraction, which gives 18 control runs.

### 2.4 Reproduction

The commands below run from `code/`; the clock sum appears at fraction .8:

```
python run.py atlas --task open_count --frac 0.8 --seeds 0 1 2 3 4 5 6 7 8 9    # also --frac 0.4 and 0.6
python run.py atlas --task open_count --kind random --seeds 0 1 2 --probe
python run.py atlas --task open_count --kind scalar --seeds 0 1 2
python run.py table out/open_count_*.json
```

Each command writes one JSON file per network. Its top-level `final.test_acc` is held-out accuracy; the nested `row` object
holds `row.final`, `row.t50` (the first evaluation with test accuracy ≥ 50%), `row.form_clock` / `row.form_clock_agree`
(substitution task accuracy and agreement) and `row.clock_k_top` (the dominant frequencies). The table command reads `row`
objects but does not print the clock fields; those remain in the individual JSON files. Both control commands default to training
fraction .4; repeat each with `--frac 0.6` and `--frac 0.8` for the other fractions.

### 2.5 Archive

Each network is stored as one safetensors file (`W1`, `b1`, `W2`, `b2`, `test_idx`), with `config.json`, `inputs.safetensors` (the
frozen menu rows X, the integers and the labels) and `manifest.json` (seed, rung, train fraction, accuracy, circuit type, sha256). Every
file was re-loaded, and its forward pass reproduces the recorded test accuracy exactly. The 30 networks (`s<seed>_f<40|60|80>`; the
clock sum is in the `_f80` files) are published on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/open-bracket-count](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/open-bracket-count).

## 3. Hand-built solution

The hand-built network supplies the inputs of the count but not the count itself (Figure 1).

The depth ratio is computed by layer 5, head 0 of m2. It attends uniformly over the bracket tokens and the newline of the current
line, with value +1 for `(` and −1 for `)`, so it writes BAL = b/(L+1). The length L comes from the position scalar R.

The menu decodes BAL into a thermometer code of b with a second step stage gated by the one-hot of L:
[b ≥ k] = Σ_L step(BAL + 100(onehot_L − 1); (k − ½)/(L+1), 40(L+1)), where `step` is the GELU step pair of
[count mod 5](count-mod-5.md#3-hand-built-solution). The one-hot of b is the difference of neighbouring thermometer units.

No hand-built open-count brick was given. The nearest hand-built design adds the two integers linearly in the residual stream and reads
the sum (L + b)/2 with step staircases.

```mermaid
flowchart TD
    accTitle: Frozen bracket-depth and length features
    accDescr: The hand-built base reads opening-minus-closing depth and line length, and the frozen menu re-encodes them. Only the final count MLP is trained; no hand-built count reader is supplied.
    P["Prompt:<br/>VALID_BRACKETS line"] --> READ["Frozen m2 readers:<br/>layer 5 head 0,<br/>uniform attention, value<br/>+1 for open and -1 for close<br/>BAL = b/(L+1)<br/>Position scalar R gives L"]
    READ --> MENU["Frozen menu inputs:<br/>step stage gated by one-hot L<br/>gives one-hot and thermometer<br/>of b (ou, tu)<br/>Direct scalar BAL (su)<br/>Scalar, one-hot and<br/>thermometer of L (sv, ov, tv)<br/>Distractors: count, neg,<br/>copy pointer, last letter"]
    MENU --> M["TRAINED: MLP,<br/>256 ReLU units"]
    M --> Y["46 logits: number of<br/>open brackets (L + b)/2"]
    classDef trained fill:gold,stroke:darkgoldenrod,color:black
    class M trained
```

*Figure 1. The hand-built network and the trained stage. Frozen parts, set by formula, compute the depth ratio b/(L+1) through layer
5, head 0 of m2 and re-encode b and the line length L in the 416-wide menu; there is no hand-built count reader. The shaded MLP is the
only trained part.*

## 4. What training finds

Within the three frozen-menu tasks compared together (count modulo 5, compare counts, and opening-bracket count), this was the only
case of [grokking](../GLOSSARY.md#counting-results), delayed test improvement after training accuracy becomes perfect. It occurred
only with 80% of the cells in training. Train accuracy was 100% at step 500 in every run (Table 2); Figure 3 shows one recorded
learning curve.

*Table 2. Held-out accuracy per training fraction (ranges over 10 seeds; 0/10 reach .90 at each fraction). The legacy rounded ranges
are retained; the source gives .037–.109 for fraction .6 and .778–.880 for fraction .8.*

| Fraction | Train 100% step | Test 50% step | Final test | Circuit type |
|---|---|---|---|---|
| .4 | 500 | | .006–.019 (below chance) | memorisation |
| .6 | 500 | | .04–.11 | memorisation |
| .8 | 500 | 20.5k–28.5k | .78–.88, flat from ~150k | low-frequency clock sum, 10/10 |

At fraction .4 the wrong answers are neighbouring cells' answers. At fraction .8, test accuracy was .02–.05 at 10k steps, reached 50%
at 20.5k–28.5k and then stayed flat at .78–.88 from about 150k steps, while the weight norm fell from 80 to 73. None reaches 90%.

The circuit types are a low-frequency clock sum in 10/10 networks at fraction .8 and memorisation in 20/20 at .4 and .6. At fraction .8
all three encodings are used together, with the one-hot as the main input. Distractor ablations have small effects; the live neg
feature must not be described as constant or causally inert.

The controls are summarised in Table 3. The random base is better than the menu at fractions .4 and .6; its residual stream holds b and
L linearly (ridge R² 1.000). A menu holding only the scalars reached .35–.56.

*Table 3. Random frozen base control (test accuracy, ranges over 3 seeds).*

| Control | Fraction .4 | Fraction .6 | Fraction .8 |
|---|---|---|---|
| Random frozen base | .57–.60 | .72–.77 | .81–.86 |

## 5. Mechanism and verification

At fraction .8, each hidden neuron's pre-activation, the value before ReLU clips negative inputs, was fitted with the closed form

pre_n = c_n + a_n cos(ω_n b/2 + φ_n) + a′_n cos(ω_n L/2 + ψ_n),  ω_n = 2πk_n/92,

where c_n is an offset, a_n and a′_n are amplitudes, φ_n and ψ_n are phase offsets, and k_n ∈ 1..45 chooses a frequency for each neuron
(most often 1–3). The trained output weights were kept. For all 10 seeds, substitution gives task accuracy .91–.93 against native
.96–.98, and agreement with the native answers is .88–.92. These are all-cell scores, not the held-out .78–.88 in Table 2. Table 4
retains the source's more precise all-cell ranges alongside the old rounded values. The pre-activation separates into a b part plus
an L part (substitution .95–.98) but is not linear: a straight-line form scores .09–.11. The low-k wavelengths (31–92)
exceed half the domain, so those cosines do not alias. The closed form is supported by substitution; no [resample ablation of the
complement](../GLOSSARY.md#verifying-a-circuit) was reported, so the status is partial. Such a test would replace the circuitry
outside the proposed mechanism with its values on other inputs.

*Table 4. Substitutions on all 1080 cells, including training cells, for 10 seeds at fraction .8. Task accuracy is correctness against
the count labels; agreement is equality with the native network's answer. The precise and legacy rounded ranges are copied separately.*

| Readout | Task accuracy, source | Legacy rounded range | Native agreement |
|---|---|---|---|
| Native network | .956–.976 | .96–.98 | |
| Separate b and L contributions | .954–.975 | .95–.98 | |
| One cosine per neuron, trained output weights | .906–.932 | .91–.93 | .88–.92 |
| Straight-line replacement | .09–.11 | .09–.11 | |

The networks combine two channels (Figure 2; numbers in Appendix A). The low-k (low-frequency) neurons are smooth, broad bumps over b and L (their
single-cosine R² is only .15–.47, so "one cosine per neuron" is a coarse description); with output columns that vary slowly over the 46
classes, their sum locates the answer (L + b)/2. Without the parity neurons the networks still score .92–.96 on all cells. A parity
(odd/even) channel, found in the archived weights of 10/10 networks at fraction .8, resolves the remaining ambiguity between neighbouring answers:
7–10 neurons whose output column alternates with the class parity. They use the Nyquist frequency, the highest frequency the grid can
carry (one sign change per step): (−1)^((L+b)/2) = cos(π(L+b)/2) = cos(πL/2)cos(πb/2) − sin(πL/2)sin(πb/2) is a product of Nyquist
terms that the ReLU forms, and their alternating output column adds ± the parity to every class. Removing them lowers the all-cell
accuracy by .01–.05; the networks get the parity of the answer right on .98–.99 of cells.

```mermaid
flowchart TD
    accTitle: Approximate clock and parity count channels
    accDescr: The trained hidden neurons combine broad low-frequency responses with a smaller odd-even channel before producing count scores. The cosine description is partial and tested on training-inclusive all-cell data, with no complement test.
    IN["menu codes of b and L,<br/>one-hot main input"] --> LK["Mostly low-frequency neurons:<br/>broad bumps over b and L"]
    IN --> PN["7 to 10 parity neurons:<br/>Nyquist terms cos and sin<br/>of pi b/2 and pi L/2"]
    LK --> SUM["slowly varying output columns:<br/>sum locates (L + b)/2"]
    PN --> PAR["alternating output column:<br/>adds the parity of the answer"]
    SUM --> Y["46 logits"]
    PAR --> Y
    %% Invisible link stacks independent channels; it is not a computational path.
    SUM ~~~ PN
    classDef trained fill:gold,stroke:darkgoldenrod,color:black
    class LK,PN trained
```

*Figure 2. The circuit training finds at fraction .8. Low-frequency neurons form a trigonometric sum that locates the answer, and a
small set of parity neurons adds the parity of the answer. Shading marks the trained hidden neurons. The cosine-sum description is
partial: a coarse substitution on all cells, with the complement not tested.*

The nearest hand-built design forms the sum linearly and reads it with step
staircases. The failed straight-line substitution does not support that linear readout for the trained stage; its fitted
responses are trigonometric. This is a Gromov-style clock
([Gromov, 2023](#references)) on a non-modular domain. Modular addition (x + y mod 59, [modular-addition-mlp](modular-addition-mlp.md))
used 22–27 frequencies; here k = 1–3 were the three most common frequencies in every seed, not the only frequencies present. Bracket and automaton tasks in transformers were studied as bounded
Dyck languages by [Yao et al. (2021)](#references), as shortcuts to automata by [Liu et al. (2022)](#references), and for the
difference between decodable and causally used depth by [Sharma et al. (2026)](#references).

## 6. Limitations and open questions

It is open whether a 90% clock is reachable with more data or a lower weight decay. The recorded accuracy and weight norm were flat
after about 150k steps, so extending the same training was not supported as a remedy by these curves. None of the 10 seeds at
fraction .8 reached .90 held-out accuracy. The higher native and formula scores in Section 5 include training cells and do not
remove that failure. One cosine per neuron remains a coarse fit (R² .15–.47), and no complement test establishes the fitted rule as
the learned mechanism. The frozen transformer supplies no running maximum depth, so that alternative bracket task was not tested.

## 7. Prediction record

No prediction was recorded before these runs; the entry predates the prediction record. A retrospective prediction, labelled as such,
follows from the nearest hand-built design, which adds the two integers linearly and reads the sum with step staircases: it predicts a
linear sum, which did not occur (a straight-line form scores .09–.11). The parity channel was not predicted.

## Appendix A. Weights excerpt and reading

For seed 0 at fraction .8, each neuron's pre-activation was split into c_n + E_b(b) + E_L(L) and fitted by one frequency k ∈ 1..45 per
neuron (ω = 2πk/92, E_b ≈ a cos(ωb/2 + φ), E_L ≈ a′ cos(ωL/2 + ψ)). Neurons per best k: k = 1: 58, k = 2: 43, k = 3: 14, k = 4: 7; the
other 134 spread over k = 6..45 (k = 45: 10, k = 23: 11). Table A1 lists the strongest low-k neurons by W2 column norm.

*Table A1. Strongest low-k neurons, seed 0, fraction .8.*

| neuron | k | a, φ | a′, ψ | fit R² | W2 column, classes 0, 9, 18, 27, 36, 45 |
|---|---|---|---|---|---|
| 216 | 1 | .49, −1.75 | .05, −2.52 | .30 | .17, −.68, .21, .56, −.41, .00 |
| 231 | 1 | .97, −1.81 | .86, −2.36 | .47 | .45, .55, .56, −.60, .29, −.01 |
| 47 | 1 | .27, +1.70 | .21, +.78 | .22 | −.17, −.60, −.51, .39, −.36, .00 |
| 171 | 2 | .28, +.67 | .37, +3.12 | .30 | −.42, .26, .11, .62, −.37, −.01 |

The parity neurons of seed 0 include neuron 58, whose output column over classes 0..9 is −.67, .75, −.72, .70, −.73, .74, −.74, .76,
−.75, .73. The pre-activations of the parity neurons are fitted by cos/sin(πb/2), cos/sin(πL/2) and their products with median R²
.96–.99. Removing them lowers the all-cell accuracy of seed 0 from .956 to .934.

The low-k neurons carry the coarse location of the answer through output columns that vary slowly over the classes; the
parity neurons, whose output column alternates in sign from class to class (neuron 58), separate neighbouring answers.

## Appendix B. Detailed evidence

![Training accuracy rises before held-out accuracy in seed 0 at training fraction .8](../figures/qa1-open-count-learning.svg)

*Figure 3. Recorded training and held-out accuracy for seed 0 at fraction .8, joined without smoothing or averaging. Training is
perfect at step 500, while held-out performance improves later and remains below .90 through 200,000 steps. This is one seed, not
the ten-seed range. Formula scores on all 1080 cells are a separate, training-inclusive test; the partial cosine description has no
resample-complement certificate.*

[Open Figure 3 at full size](../figures/qa1-open-count-learning.svg).

## References

- Gromov, A. (2023). Grokking modular arithmetic. *arXiv preprint*. https://doi.org/10.48550/arxiv.2301.02679 (the clock solution of
  a wide MLP)
- Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A., & Zhang, C. (2022). Transformers learn shortcuts to automata. *arXiv preprint*.
  https://doi.org/10.48550/arxiv.2210.10749 (automaton tasks)
- Sharma, A., Dawes, C., & Raval, S. (2026). Dissociating decodability and causal use in bracket-sequence transformers. *arXiv
  preprint*. https://doi.org/10.48550/arxiv.2604.22128 (Dyck depth, top of stack)
- Yao, S., Peng, B., Papadimitriou, C., & Narasimhan, K. (2021). Self-attention networks can process bounded hierarchical languages.
  *arXiv preprint*. https://doi.org/10.48550/arxiv.2105.11115 (bounded Dyck)
