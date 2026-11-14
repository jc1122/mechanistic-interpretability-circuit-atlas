# Compare two counts

Picking the more common letter in a shuffled line of a's and b's sounds like a counting problem; recognising a tie turned out to be the stumbling block.

Every network got every held-out tie wrong, and both a random fixed base network and inputs containing only the numerical values beat the richer hand-built representations. The task is partly solved with equality unsolved: the fitted comparison formula is [partial](../GLOSSARY.md#verifying-a-circuit), never returns a tie, and was not tested while changing activity outside its proposed circuit.

## The circuit at a glance

Figure S1 shows two score paths inside one learned block: letter comparison and a separately memorised tie score, whose held-out failures limit the rule.

<img src="../figures/circuits/compare-counts.svg" width="640" alt="A frozen feature menu supplies count 12 and length 20 for a line containing 12 a's and 8 b's. Inside ONE trained neuron block, a tilted letter-comparison score and a separately memorised tie score read these features in parallel, then meet at the answer. The task example answers a; every held-out tie fails. Dashed scores are approximate, fitted on all count/length cells, with the complement untested.">

*Figure S1 shows supplied counts feeding an approximate letter comparator and a separately memorised tie score (all 30 menu-trained networks, differing from the hand-built equality-by-default rule); status is Partial, the formula fit includes every count/length cell, complement resampling was not tested, and values are a task example, not measured answer scores.*

| Fact | Value |
|---|---|
| Task | Say whether a's or b's are more frequent in a line of up to 45 letters, or `=` if they tie |
| Input → output example | A line of 20 letters with 12 a's and 8 b's → `a` |
| Model | Frozen hand-built 6-layer transformer and a frozen menu of encodings; trained: one MLP layer, 256 ReLU units, 3 outputs |
| Rungs and seeds | Hand-built network, exact on all cells; MLP trained on frozen hand-built parts, 10 seeds × train fractions .4/.6/.8 (30 networks), 200,000 steps each; 18 control runs |
| Closed form status | Partial (approximate two-way comparator: substituted accuracy .928–.958, agreement .916–.935 over all 1080 cells, not an independent formula-fit split; never outputs a tie; complement not tested) |
| Status | Partly solved (greater/less learned; at fraction .8, test .917–.963; `=` 0 correct test cells in 30/30); last run 2026-11-04 |
| Weights | [Hugging Face: compare-counts](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/compare-counts) (hand-built base in `_shared/m2`) |
| Reproduce | [Commands](#24-reproduction), run from `code/` |

A fixed hand-built transformer supplied the count and line length in several representations, leaving one small output-processing block to learn the comparison.

All 30 networks, ten runs at each of three training fractions, learned an approximate dividing line between “more a's” and “more b's”; at training fraction .8, held-out accuracy was .917–.963.

## 1. Task and data

The prompt has a `COUNT a:` header and a shuffled list of a's and b's, preceded by one solved example, as in
[count mod 5](count-mod-5.md). Let c be the number of a's and L the total number of letters, so the b count is L − c. For example,
a line of 20 letters with 12 a's and 8 b's should answer `a`.
The answer is `a` if c > L − c, `b` if c < L − c and `=` if 2c = L, so there are 3 classes; as labels, class 0 if 2c > L, 1 if
2c < L and 2 (`=`) if 2c = L. The data cover every cell (c, L) with L = 1..45, which gives 1080 cells with one prompt each, of which 22
are `=`. The trained stage must therefore compare c with L/2. The split into training and test data is over cells, with train fractions
.4, .6 and .8. Accuracy is the share of held-out cells answered correctly; the `=` cells are also scored on their own.

## 2. Model

The shared architecture, optimiser and instrumentation are described in [METHODS.md](../METHODS.md); this section gives only what is
specific to the task.

### 2.1 Frozen hand-built network

The base is [m2](../GLOSSARY.md#hand-built-machinery), a GPT-NeoX-shaped decoder with 6 layers, d_model 528 and 4 heads (d_head 132),
MLP width 512, exact GELU, fixed-gain LayerNorm (eps 1e-5), no positional embedding and no rotary embedding, a sequential residual
stream and a character vocabulary of 70 ids. It has 10,041,408 parameters, of which 13,443 are non-zero, and nothing in it was
trained. It is read at the last prompt position, from the residual stream before the final LayerNorm. `code/m2.py` rebuilds it bit for
bit (safetensors sha256 `3dbb5226…cd9a6a`). The base computes the count c through a segment-match attention head, the line length L
through a position scalar, and, in its layer 5 MLP, a hand-built comparator (Section 3).

### 2.2 Frozen menu and trained part

The [menu](../GLOSSARY.md#hand-built-machinery) is a frozen layer, defined by formula with no learned parameters, that places three
encodings of each integer at the answer position: the raw scalar, a one-hot code and a thermometer code (a vector whose k-th unit is
[n ≥ k]), together with distractor features. It is computed from m2's count X = 1/(c+1) and position scalar R, which gives L. A one-hot code has one unit per possible value; a thermometer code has one unit per threshold
([glossary](../GLOSSARY.md#standard-machine-learning-terms)). Each block is centred and divided by its rms row norm, the root mean
square of the row's entries. Table 1 lists the blocks of the 326-wide input.

*Table 1. Input blocks of the frozen menu (index ranges of the 326-wide input).*

| Block | Indices | Content |
|---|---|---|
| su | 0 | X |
| sv | 1 | R |
| ou | 2–47 | one-hot c = 0..45 |
| ov | 48–93 | one-hot L = 0..45 |
| tu | 94–138 | [c ≥ k], k = 1..45 |
| tv | 139–183 | [L ≥ k] |
| brackets (distractor) | 184–185 | constant, so 0 |
| copy pointer (distractor) | 186–255 | |
| last letter (distractor) | 256–325 | |

The trained stage is one MLP, `logits = W2 ReLU(W1 x + b1) + b2`, with W1 [256, 326], b1 [256], W2 [3, 256] and b2 [3], which is
84,483 parameters; the three outputs are the classes a, b and `=`.

### 2.3 Training

Training followed [count mod 5](count-mod-5.md): hidden width 256, 200,000 steps on the full batch, an evaluation every 500 steps, and
10 seeds at each train fraction .4 / .6 / .8. Two controls, a random frozen base of the same shape and a menu holding only the scalars,
were run with 3 seeds each at each fraction, which gives 18 control runs.

### 2.4 Reproduction

The commands below run from `code/`:

```
python run.py atlas --task compare_counts --frac 0.4 --seeds 0 1 2 3 4 5 6 7 8 9    # repeat with --frac 0.6 and 0.8
python run.py atlas --task compare_counts --kind random --seeds 0 1 2 --probe
python run.py atlas --task compare_counts --kind scalar --seeds 0 1 2
python run.py table out/compare_counts_*.json
```

The headline numbers of each network are in the output keys `final` (test accuracy), `eq_test_acc` (test accuracy on the `=` cells),
`cmp_ratio` and `cmp_r2` (the linear comparator), and `form_line` / `form_line_agree` (substitution accuracy and agreement).

### 2.5 Archive

Each network is stored as one safetensors file (`W1`, `b1`, `W2`, `b2`, `test_idx`), with `config.json`, `inputs.safetensors` (the
frozen menu rows X, the integers and the labels) and `manifest.json` (seed, rung, train fraction, accuracy, circuit type, sha256). Every
file was re-loaded, and its forward pass reproduces the recorded test accuracy exactly. The 30 networks (`s<seed>_f<40|60|80>`) are
published on Hugging Face at [huggingface.co/jc1122/circuit-atlas-nets/tree/main/compare-counts](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/compare-counts).

## 3. Hand-built solution

The hand-built network counts with a single attention head, decodes the count with step functions and compares with three step pairs, two of which act on
the difference of the counts (Table F1).

The count is computed by layer 4, head 1 of m2. Its query is the question letter, supplied by an earlier has-letter head. Every key
token equal to that letter, and the newline that ends the counted line, scores 300, plus a recency term of 3 per position of the
segment anchor, which selects the current `:` segment; the scores are centred so that tied keys score exactly 0. Attention is
therefore uniform over the c matches and the 1 newline. The value reads only the newline indicator, so the head writes X = 1/(c+1).
The head has 95 non-zero entries in the query weights, 52 in the key weights and 2 in the value and output weights.

The menu decodes X with pairs of GELU steps, `step(z; c, s) = (gelu(s(z − c) + 8) − gelu(s(z − c) − 8))/16`. The thermometer unit is
[c ≥ k] = step(−X; −m_k, 32/g_k), where m_k = (1/k + 1/(k+1))/2 is the midpoint and g_k = 1/k − 1/(k+1) the gap between neighbouring
values of X. The one-hot unit is the difference of neighbouring thermometer units. The decoder is exact on all cells.

The hand-built comparator, `compare_answer`, sits in m2's layer 5 MLP as three step pairs with slope 40, gated by
g = 200·[task flag] + 200·[space token]. The `=` unit fires at g ≥ 300 and writes +`=`; the `>` unit fires at
100(NA − NB) + g ≥ 400.5, where NA and NB are the two letter counts, and writes +`>` and −`=`; the `<` unit is symmetric. Equality is
therefore the default output, what remains when neither difference unit fires, and it is exact on unseen cells. Weights are in
Appendix A.

Table F1 records the supplied count/length features and the fixed comparator alternative. The frozen count head supplies X and the position scalar supplies length L. Count decoders form the one-hot and thermometer codes; these, scalar inputs, length codes and distractors enter the trained menu-reading MLP. A separate frozen comparator supplies the hand-built reference answer. Only the menu-reading MLP is trained.

<details>
<summary>Technical supplied-feature menu and fixed comparator (Table F1)</summary>

*Table F1. The hand-built network and the trained stage. Frozen parts, set by formula, compute count c through layer 4, head 1 of m2 and re-encode it, together with line length L, in the 326-wide menu; m2's layer 5 MLP holds the hand-built comparator. The menu-reading MLP is the only trained part.*

| Component or feature route | Input, operation and output |
|---|---|
| Fixed count head | The prompt is a shuffled line of a's and b's. m2 layer 4, head 1 attends uniformly over the c matches and the newline, giving X = 1/(c+1). |
| Fixed length route | m2 position scalar R gives line length L. |
| Fixed count decoder | A GELU step ladder at midpoints m_k reads X to form one-hot count indicators (ou) and thermometer count flags (tu). The scalar X route is su. |
| Fixed length codes and distractors | Scalar, one-hot and thermometer codes of L are sv, ov and tv. Brackets, copy pointer and last letter are distractors. All enter the menu alongside count codes. |
| Trained menu-reading MLP | 256 ReLU units read the supplied menu and write 3 scores (logits): a, b and =. |
| Separate fixed hand-built comparator | m2 layer 5 MLP reads the prompt's counts. It answers = by default, with greater/less step units on NA − NB choosing a or b. |

</details>

## 4. What training finds

Test accuracy reached its plateau at the first evaluation (step 500) and never improved. The held-out ranges were .910–.941 at
fraction .4, .921–.947 at fraction .6 and .917–.963 at fraction .8 (Table 2). There was no [grokking](../GLOSSARY.md#counting-results), a delayed rise of test accuracy after training
accuracy: accuracy was already at its plateau, and the weight norm stayed flat.

*Table 2. Results per cell. Test accuracy is the source-reported range over seeds.*

| Cell | Seeds | Plateau step | Final test | Correct `=` test cells |
|---|---|---|---|---|
| MLP on the frozen hand-built menu, fraction .4 | 10 | 500 | .910–.941 | 0 in every network |
| MLP on the frozen hand-built menu, fraction .6 | 10 | 500 | .921–.947 | 0 in every network |
| MLP on the frozen hand-built menu, fraction .8 | 10 | 500 | .917–.963 | 0 in every network |
| Control: random frozen base | 3 per fraction | | .99–1.000 | |
| Control: scalar-only menu | 3 per fraction | | .97–.99 | .17–.44 |

All 30 networks fall into one [circuit type](../GLOSSARY.md#counting-results), a grouping by mechanism (30/30): an approximate linear
comparator with a tilted boundary and `=` memorised. This is a classification count, not 30 solved tasks. The a − b logit difference is 83–87% linear in (c, L), with an L/c slope ratio (β/α in Section 5) of −.56 to −.64 where the exact rule is
−.50; the errors sit at |2c − L| ≤ 2. The `=` class never generalises: 0 correct `=` test cells in 30/30.

At fraction .4 the thermometer code of c is the main input: ablating it lowers test accuracy to .59, while the one-hot and the scalar
contribute little.

## 5. Mechanism and verification

Each network was fitted with the closed form

answer = a if αc + βL + γ > 0 else b (never `=`),

with α, β and γ fitted per network. Substituted for the network, it reaches accuracy .928–.958 (the corresponding networks score
.946–.963) and agrees with the network on .916–.935 of cells. The fit and substitution screen covered all 1080 count/length cells, not an independently
held-out formula-fit split. A form built on the exact variable 2c − L scores lower (.90–.94), so that variable alone does not
reproduce the measured decisions as well; this does not prove it is absent internally. The closed form is supported by substitution; no [resample ablation of the
complement](../GLOSSARY.md#verifying-a-circuit) was reported, so the status is partial.

The weights show the tilt (Table F2; numbers in Appendix A). The a − b difference is a tilted line, β/α ≈ −.6 instead of the exact
−.5, so the boundary drifts away from 2c = L as L grows. Of the 22 equality cells in the full domain, the `=` logit is large only on
those assigned to the training split and very negative on the held-out ones (margin over the best other class +13.7 on training cells and −15.5 on test cells for seed 0): it
is a per-cell lookup on the one-hot codes, not a function of 2c − L.

Table F2 records the separate letter and tie scores shown in the worked circuit above. Thermometer count and length features feed the trained letter-score difference, approximated by a tilted linear rule. One-hot codes feed a separate trained tie score that is memorised on training cells and fails on unseen ties. Status remains partial: substitution supports the approximation, with complement untested; it is not an exact equality-aware comparator.

<details>
<summary>Technical letter comparison and memorised tie score (Table F2)</summary>

*Table F2. The circuit type found in 30/30 networks. The trained MLP's a − b difference is approximated by a tilted linear function of c and L, while `=` has a separate score memorised over training cells. Both score computations are trained. The linear rule is partial: supported by substitution, with the complement not tested, and never an equality-aware exact comparator.*

| Separate trained score | Supplied inputs and recorded behaviour |
|---|---|
| Letter-score difference | Thermometer count flags tu are the main input, alongside length blocks ov, tv and sv. The a − b score is approximated by alpha c + beta L + gamma, with beta/alpha −.56 to −.64. The a/b boundary is tilted away from 2c = L. |
| Tie score | One-hot codes of c and L (ou, ov) feed a per-cell lookup. The = score is large on training cells only; it gives 0 correct test tie cells. |

</details>

The hand-built comparator makes `=` the default output and subtracts two steps on the difference (greater, less), so equality is
"neither" and is exact on unseen cells. The trained stage instead gives `=` its own logit. On one-hot or thermometer codes, equality is a
non-linear (XOR-like) feature, and these 30 runs memorised it. This is observed training behaviour, not a representational
impossibility: the hand-built comparator itself implements exact equality by making it the default.

Both controls did better than the hand-built menu: a random frozen base scored .99–1.0 against .91–.96 for the menu
networks. A random frozen base reached .99–1.000; its residual stream encodes c,
L and 2c − L linearly (ridge R² 1.000), so a few ReLUs give an exact comparator, `=` included. A menu with only the scalars reached
.97–.99 and got .17–.44 of the `=` test cells right across all three training fractions, also better than the full menu. The older
source summary gave .25–.44; the full control record includes fraction .4 with a separately recorded .167 result. The broader
.17–.44 display includes all three fractions; these summaries are not silently interchanged. Counting and histogram tasks in small
transformers were studied by [Behrens et al. (2024)](#references), and the length generalisation of formal-language tasks such as
counting by [Delétang et al. (2022)](#references).

## 6. Limitations and open questions

One hypothesis, untested, is that exact one-hot codes make `=` and the boundary cells memorisable at a lower weight norm than learning
the line 2c − L. No test on lines longer than 45 letters was reported. The comparator is an approximate fit over the full cell domain,
not a separately validated formula for the network; no complement resample was reported.

## 7. Prediction record

No prediction was recorded before these runs; the entry predates the prediction record. A retrospective prediction, labelled as such,
is that the hand-built comparator's default "neither greater nor less" output (`compare_answer`) might favour equality transfer.
This is not a claim that it is the only learnable design. The trained
stage gives `=` its own logit instead and never generalises it (0 correct `=` test cells in 30/30), so the hand-built design did not
predict the trained circuit.

## Appendix A. Weights excerpt and reading

The hand-built comparator in m2's layer 5 MLP has three step pairs with slope 40 and gate g = 200·[task flag] + 200·[space token]. The
`=` unit fires at g ≥ 300 (writes +`=`); the `>` unit at 100(NA − NB) + g ≥ 400.5 (writes +`>`, −`=`); the `<` unit is symmetric.
Equality is what remains when neither difference unit fires.

Table A1 gives the least-squares fit of the a − b logit difference of each trained network to αc + βL + γ over all cells, at
fraction .4.

*Table A1. Linear comparator per seed, fraction .4.*

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| α | 4.43 | 4.47 | 4.27 | 4.40 | 3.84 | 4.22 | 4.19 | 4.17 | 4.12 | 4.31 |
| β | −2.55 | −2.51 | −2.74 | −2.72 | −2.31 | −2.65 | −2.52 | −2.40 | −2.62 | −2.60 |
| β/α | −.58 | −.56 | −.64 | −.62 | −.60 | −.63 | −.60 | −.58 | −.64 | −.60 |

Table A2 gives the logits of seed 0 at L = 20 on test cells.

*Table A2. Logits (a, b, `=`) of seed 0 at L = 20, all test cells.*

| c | a | b | `=` |
|---|---|---|---|
| 8 | −1.0 | 6.8 | −10.1 |
| 9 | −6.6 | 10.9 | −8.3 |
| 10 (`=`) | 6.4 | −0.4 | −9.9 |
| 11 | 1.6 | 0.7 | −4.7 |
| 12 | 2.8 | −1.7 | −2.9 |

The margin of the `=` logit over the best other class on `=` cells was +13.7 on training cells and −15.5 on test cells. The W1 column
norm per input block for seed 0 was tu 7.3, ov 7.0, ou 6.4, tv 4.1, sv 2.5, su 2.4.

Reading. β/α ≈ −.6 instead of −.5 tilts the boundary, so the errors sit at |2c − L| ≤ 2. At c = 10, L = 20 (a held-out `=` cell) the
`=` logit is the lowest of the three.

## Appendix B. Detailed evidence

*Table B1. Source-precision final accuracies, ten seeds per training fraction. Each score is a share of held-out count/length cells.*

| Train fraction | Final test | Test a/b cells only | Correct held-out ties |
|---|---|---|---|
| .4 | .910–.941 | .926–.957 | 0 in every seed |
| .6 | .921–.947 | .941–.967 | 0 in every seed |
| .8 | .917–.963 | .943–.977 | 0 in every seed |

Key: these are the source's per-fraction ranges, also used in Table 2. The earlier lede and Status both displayed .91–.96; the
earlier Table 2 and Section 4 used .91–.94 at fraction .4 and .92–.96 at fraction .8. These historical rounded displays are retained
here rather than silently selected over the source precision.

The substituted two-way rule scores .928–.958 and agrees with the network on .916–.935 of all 1080 cells; the corresponding network
scores .946–.963. The earlier Section 5 and fact table displayed .93–.96 substitution accuracy and .92–.94 agreement; the earlier
Section 5 network display was .95–.96. These are retained reporting precisions, not additional experiments. None of these scores
counts a held-out equality success.

## What to read next

Compare [count mod 5](count-mod-5.md), which uses the same frozen count encodings for a different readout. Read [the verification guide](../GLOSSARY.md#verifying-a-circuit) for the distinction between formula task accuracy and agreement with the trained network.

## References

- Behrens, F., Biggio, L., & Zdeborová, L. (2024). Counting in small transformers: The delicate interplay between attention and
  feed-forward layers. *ICML 2025 (arXiv preprint)*. https://doi.org/10.48550/arxiv.2407.11542 (counting and histogram tasks)
- Delétang, G., Ruoss, A., Grau-Moya, J., et al. (2022). Neural networks and the Chomsky hierarchy. *arXiv preprint*.
  https://doi.org/10.48550/arxiv.2207.02098 (length generalisation on formal-language tasks)
