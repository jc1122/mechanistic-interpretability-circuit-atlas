# Count mod 5

Seven a's in a line leave a remainder of 2 after division by 5, however many b's surround them. A fixed hand-built transformer supplied the count in several representations, and only a small output-processing network learned to turn it into a remainder.

A formula using two cosine waves describes that lookup, but here the waves simply group counts with the same remainder; the fit does not show that the network internally uses trigonometry. The task is solved at that criterion and the formula is [partial](../GLOSSARY.md#verifying-a-circuit): replacement works on all 1080 count/length combinations, including training inputs, without a test changing activity outside the proposed circuit.

## The circuit at a glance

<img src="../figures/circuits/count-mod-5.svg" width="640" alt="The frozen transformer head counts seven a tokens using the reciprocal signal one eighth. Its residual stream ends before two external frozen count indicators: count 2 is off and count 7 is on. Both feed nearly the same learned remainder-2 pattern in the feed-forward block. The task answer is 2; this is a partial lookup account, not a claim of identical individual neurons.">

*Figure S1. Frozen counting matches the hand-built reference, while all 30 trained feed-forward blocks (MLPs; 10 seeds at each fraction .4/.6/.8) group counts such as 2 and 7 into nearly the same remainder pattern (remainder 2 after division by 5); this partial fit uses all 1080 cells, including training cells, with complement resampling and transfer to unseen counts untested.*

Figure S1 shows count indicators 2 and 7 converging on the learned remainder-2 group: seven a’s turn on the count-7 indicator, while the count-2 indicator stays off; the transformer and feature menu stay frozen.

| Fact | Value |
|---|---|
| Task | Count how often a letter occurs in a line of a's and b's (up to 45 letters) and answer the count mod 5 |
| Input → output example | After a fixed solved example, `COUNT a: a a b a a a b a a` → `2` (seven a's) |
| Model | Frozen hand-built 6-layer transformer and feature menu; trained: one feed-forward block (MLP), 256 hidden units |
| Rungs and seeds | Hand-built base exact on all cells; MLP trained on frozen features, 10 seeds × training fractions .4/.6/.8 (30 networks), 200,000 steps each; 18 control runs; no from-scratch cell |
| Closed form status | Partial (two-cosine output fit: task accuracy 1.000 on all 1080 cells in 30/30 networks; not a prediction-agreement score; complement not tested) |
| Status | Solved (frozen-feature MLP, 30/30 with held-out accuracy ≥ .99); last run 2026-11-04 |
| Weights | [Hugging Face: count-mod-5](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/count-mod-5) (hand-built base in `_shared/m2`) |
| Reproduce | [Commands](#24-reproduction), run from `code/` |

All 30 networks, 10 independent runs at each of three training fractions, reached held-out accuracy ≥ .99 by looking up the count's remainder and largely ignoring line length.

## 1. Task and data

Each prompt holds a fixed one-shot example followed by the query line:
`COUNT b: a b\nANSWER: 1\nCOUNT a: <c a's and L−c b's, shuffled>\nANSWER: `. The answer is c mod 5, so there are 5 classes. The data
cover every cell (c, L) with line length L = 1..45 and count c = 0..L, which gives 1080 cells with one prompt each. The split into
training and test data is over cells, rather than a split deliberately withholding whole count values; train fractions of .4, .6 and .8 were used. Accuracy
is the share of held-out cells whose single answer is correct. For example, `COUNT a: a a b a a a b a a` has seven a's, so the answer
is `2` after the fixed example prefix. The success criterion is held-out accuracy ≥ .99; the formula tests in Section 5 instead use
all 1080 cells, including the training cells.

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
bit (safetensors sha256 `3dbb5226…cd9a6a`). The base computes the count c through a segment-match attention head, which gives
1/(c+1), and an MLP step ladder that decodes it; it also computes the line length L (Section 3).

### 2.2 Frozen menu

The [menu](../GLOSSARY.md#hand-built-machinery) is a frozen layer, defined by formula with no learned parameters, that places three
encodings of each integer at the answer position: a scalar (one number), a one-hot code (one indicator per possible value), and a
thermometer code (indicators of the thresholds crossed, with k-th unit [n ≥ k]), together with distractor features. It is computed from m2's count X = 1/(c+1) and position scalar R, which gives L. Each
block is centred and divided by its root-mean-square (rms) row norm, a scale measure, so every block has rms 1. Table 1 lists the blocks of the 326-wide input.

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

### 2.3 Trained part and training

The trained stage is one MLP, `logits = W2 ReLU(W1 x + b1) + b2`, where logits are the output scores used to choose an answer. It has
W1 [256, 326], b1 [256], W2 [5, 256] and b2 [5], which is
84,997 parameters: hidden width 256 ReLU units and 5 outputs. It was trained with AdamW, learning rate 1e-3 and weight decay 1.0, on
the full batch for 200,000 steps, with an evaluation every 500 steps and no early stop. Ten seeds were run at each train
fraction .4 / .6 / .8. Two controls, a random frozen base of the same shape and a menu holding only the scalars, were run with 3 seeds
at each fraction, giving the 18 control runs in the fact table.

### 2.4 Reproduction

The commands below run from `code/`:

```
python run.py atlas --task count_mod5 --frac 0.4 --seeds 0 1 2 3 4 5 6 7 8 9     # repeat with --frac 0.6 and 0.8
python run.py atlas --task count_mod5 --kind random --seeds 0 1 2 --probe        # random-base control with ridge probes
python run.py atlas --task count_mod5 --kind scalar --seeds 0 1 2
python run.py table out/count_mod5_*.json                                         # min-max over seeds, as quoted here
```

Each command writes one JSON file per network. Its top-level `final.test_acc` is held-out accuracy; the nested `row` object
holds the flat analysis summary: `row.final`, `row.form_res5`, `row.form_u_only`, `row.form_fourier` and `row.fourier_r2`.
The `form_` fields report task accuracy under substitution, not agreement with the network. The table command reads these `row`
objects. Both control commands default to training fraction .4; repeat each with `--frac 0.6` and `--frac 0.8` for the other fractions.

### 2.5 Archive

Each network is stored as one safetensors file (`W1`, `b1`, `W2`, `b2`, `test_idx`), with `config.json`, `inputs.safetensors` (the
frozen menu rows X, the integers and the labels) and `manifest.json` (seed, rung, train fraction, accuracy, circuit type, sha256).
Every file was re-loaded, and its forward pass reproduces the recorded test accuracy exactly. The 30 networks (`s<seed>_f<40|60|80>`)
are published on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/count-mod-5](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/count-mod-5).

## 3. Hand-built solution

The hand-built network counts with a single attention head and decodes the count with step functions (Table F1).

The count is computed by layer 4, head 1 of m2. Its query is the question letter, supplied by an earlier has-letter head. Every key
token equal to that letter, and the newline that ends the counted line, scores 300, plus a recency term of 3 per position of the
segment anchor, which selects the current `:` segment; the scores are centred so that tied keys score exactly 0. Attention is
therefore uniform over the c matching letters and the 1 newline. The value reads only the newline indicator, so the head writes
X = 1/(c+1). The head has 95 non-zero entries in the query weights, 52 in the key weights and 2 in the value and output weights.

The menu decodes X with pairs of GELU steps, `step(z; c, s) = (gelu(s(z − c) + 8) − gelu(s(z − c) − 8))/16`. The thermometer unit is
[c ≥ k] = step(−X; −m_k, 32/g_k), where m_k = (1/k + 1/(k+1))/2 is the midpoint and g_k = 1/k − 1/(k+1) the gap between neighbouring
values of X. The one-hot unit is the difference of neighbouring thermometer units. The decoder is exact on all cells.

Table F1 records the fixed feature menu alongside the worked circuit above. The frozen matching head writes reciprocal count X, and a position scalar supplies length L. Step decoders form count indicators; these, the scalar count, length codes and distractors enter the sole trained MLP to score remainders. The table separates the frozen encodings from the learned readout.

<details>
<summary>Technical supplied count/length menu and trained readout (Table F1)</summary>

*Table F1. The hand-built network and the trained stage. Frozen parts, set by formula, compute count c through layer 4, head 1 of m2 and re-encode it, together with line length L, in the 326-wide menu; the menu-reading MLP is the only trained part.*

| Component or feature route | Input, operation and output |
|---|---|
| Fixed count head | Read COUNT a: and the line of a's and b's. m2 layer 4, head 1 attends uniformly over the c matches and the newline, giving X = 1/(c+1). |
| Fixed length route | m2 position scalar R gives line length L. |
| Fixed count decoder | A GELU step ladder at midpoints m_k reads X to form one-hot count indicators (ou) and thermometer count flags (tu). The scalar X route is su. |
| Fixed length codes and distractors | Scalar, one-hot and thermometer codes of L are sv, ov and tv. Brackets, copy pointer and last letter are distractors. All enter the menu alongside count codes. |
| Trained menu-reading MLP | 256 ReLU units read the supplied menu and write 5 scores (logits) for c mod 5. |

</details>

## 4. What training finds

Training was fast and showed no delay between fitting and generalising. For every train fraction, train accuracy and test accuracy
both reached 99% at step 500, the first evaluation, and the final test accuracy was .992–1.000 (Table 2). This is not
[grokking](../GLOSSARY.md#counting-results).

*Table 2. Results per cell. Final test accuracy is the range over seeds.*

| Cell | Seeds | Solved | Train 99% step | Test 99% step | Final test |
|---|---|---|---|---|---|
| MLP on the frozen hand-built menu, fraction .4 | 10 | 10/10 ≥ .99 | 500 | 500 | .992–1.000 |
| MLP on the frozen hand-built menu, fraction .6 | 10 | 10/10 ≥ .99 | 500 | 500 | .995–1.000 |
| MLP on the frozen hand-built menu, fraction .8 | 10 | 10/10 ≥ .99 | 500 | 500 | .995–1.000 |
| Control: random frozen base | 3 per fraction | | | | .24–.40 |
| Control: scalar-only menu | 3 per fraction | | | | .29–.34 (chance .2) |

All 30 networks fall into one circuit type, a residue lookup on c, meaning that counts with the same remainder share a readout
(30/30). The lookup largely ignores L. Setting the L input to its mean kept
.995–1.000 test accuracy, and replacing each c embedding by the mean of its residue class kept 1.000.

The networks read mainly the one-hot code of c (Table 3). With all c features ablated, test accuracy at fraction .4 fell to .16; with
all L features ablated it stayed at .997. Kept alone, the one-hot pair gave .997, the thermometer .86 and the scalars .21. The one-hot
is the primary input, and the information is redundant across the one-hot and thermometer codes.

*Table 3. Test accuracy after ablating or isolating input encodings (fraction .4).*

| Inputs | Test accuracy |
|---|---|
| all c features ablated | .16 |
| all L features ablated | .997 |
| one-hot pair kept alone | .997 |
| thermometer kept alone | .86 |
| scalars kept alone | .21 |

## 5. Mechanism and verification

Each network was fitted with the closed form

logit_r(c) = A cos(2π(c − r)/5) + B cos(4π(c − r)/5),

with fit R² .989–.999 and task accuracy 1.000 on all 1080 cells for each of the 30 networks when the formula replaces the trained
readout. Here a logit is an output score and R² measures the fit to those scores
([glossary](../GLOSSARY.md#analysis-measures)). The 1.000 is accuracy against the task labels, not agreement with every native
prediction: native held-out accuracy was .992–1.000. The closed form is partial, supported by
[substitution](../GLOSSARY.md#verifying-a-circuit); no resample ablation replaced the circuitry outside the claimed mechanism with
values from other inputs. The two amplitudes are nearly equal
(A ≈ B, for example 6.77 and 6.68). With A = B the profile is a delta: 2 at c ≡ r and −1/2 elsewhere, in units of A. This fit describes a flat lookup and does not by itself establish an internal
[Fourier mechanism](../GLOSSARY.md#analysis-measures), in which numbers are represented by angles.

The weights show how the lookup is wired (Table F2; numbers in Appendix A). The largest first-layer weights sit on the one-hot of c
(block ou), so each count value is mapped by its own column to the hidden pattern of its residue class, and the logits take nearly the
same value for c and c + 5 (14.25 and 13.40 for c = 0 and 5 in Table A1). The L blocks receive small weights whose effect the network
averages out, since setting L to its mean keeps .995–1.000.

Table F2 records remainder pooling and distinguishes the ideal profile from native scores. Count indicators feed the trained input weights and a hidden remainder pattern before the output scores. Length inputs have small effects that can be replaced by their mean. The equal-amplitude cosine profile describes an ideal lookup, not identical native scores or a trigonometric mechanism. Status remains partial, with complement untested.

<details>
<summary>Technical learned pooling and ideal-versus-native scores (Table F2)</summary>

*Table F2. The trained readout pools counts by remainder. The equal-height delta is the ideal A = B output profile; learned logits are nearly flat rather than identical. The MLP's input weights, hidden pattern and output weights are trained. Status: partial, supported by substitution, with the complement not tested. The equal-amplitude cosine description does not prove trigonometric computation.*

| Trained readout part | Supplied input and recorded behaviour |
|---|---|
| Input weights W1 | One-hot count indicators ou have the largest W1 column norm; thermometer flags tu give a redundant copy. Columns for c and c + 5 map to nearly the same hidden pattern. |
| Hidden response | Represents residue class r = c mod 5. Length blocks sv, ov and tv have small weights; replacing their effects by the mean keeps accuracy. |
| Ideal output profile | Scores are 2A at r = c mod 5 and −A/2 elsewhere in the ideal A = B profile. Native scores are nearly flat, not identical. |

</details>

The two controls change the frozen encodings. A random frozen base of the same shape reached .24–.40 test accuracy; its
residual stream holds c linearly (ridge R² 1.000) but carries no residue feature. A menu with only the scalars reached .29–.34 against
chance .2. Thus the scalar-only control did not learn a residue partition of 1/(c+1) with 256 ReLU units at this weight decay.

Relation to the hand-built design: the hand-built count decoder is a step ladder on 1/(c+1), the same construction as the menu's
one-hot and thermometer codes. The trained stage reads the one-hot output and pools it by residue, which is one more linear map. No
hand-built mod-5 component was given. Counting and histogram tasks in small transformers, and the division of labour between
attention and feed-forward layers in them, were studied by [Behrens et al. (2024)](#references). The entry reports no
[resample ablation of the complement](../GLOSSARY.md#verifying-a-circuit); the evidence rests on substitution of the closed form
and on the input ablations above, so the closed form is supported by substitution with the complement not tested.

## 6. Limitations and open questions

Holding out values of c, rather than cells, to test whether the period extrapolates was not run. Extrapolation to unseen count
values was expected to fail, but that expectation was not tested. It is also open why the one-hot code is preferred over the
thermometer code when both are exact, and whether its norm cost under weight decay explains the preference. Substitution and
input-family ablations do not replace the missing resample-complement test.

## 7. Prediction record

No prediction was recorded before these runs; the entry predates the prediction record. A retrospective prediction, labelled as such,
is that because the hand-built decoder already turns X into exact one-hot and thermometer codes, the cheapest solution is a linear
pooling of the one-hot code by residue, with L ignored. This held in 30/30 networks. It did not anticipate that the thermometer alone
also carries most of the accuracy (.86).

## Appendix A. Weights excerpt and reading

Table A1 gives the logits of seed 0 at fraction .4, centred over the 5 classes and averaged over L, for c = 0..9; c = 10..45 repeat
the pattern.

*Table A1. Centred logits, seed 0, fraction .4. Bold marks the class r = c mod 5.*

| c | r = 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | **14.25** | −2.59 | −3.42 | −3.93 | −4.30 |
| 1 | −2.95 | **13.56** | −3.30 | −3.42 | −3.88 |
| 2 | −3.35 | −3.10 | **13.35** | −3.12 | −3.78 |
| 3 | −3.57 | −3.33 | −3.31 | **13.83** | −3.63 |
| 4 | −3.50 | −3.40 | −3.28 | −3.22 | **13.40** |
| 5 | **13.40** | −3.29 | −3.39 | −3.34 | −3.38 |
| 9 | −3.57 | −3.63 | −3.42 | −3.34 | **13.97** |

The two-cosine fit (A, B) of the logits over seeds 0–9 at fraction .4 gave A = 6.73–6.77 and B = 6.61–6.68 (seed 0: 6.77, 6.68). The
W1 column norm per input block for seed 0 was ou 14.6, tu 6.5, su 4.2; ov 1.9, tv 4.0, sv 3.1; copy 3.3, last 2.9.

With A = B the closed form gives 2A ≈ 13.5 at r = c mod 5 and A(cos 72° + cos 144°) = −A/2 ≈ −3.4 elsewhere, which matches Table A1:
a flat delta of nearly equal height on the residue, nearly identical for c and c + 5 (14.25 against 13.40 at r = 0).

## Appendix B. Detailed evidence

*Table B1. Native training and test accuracy at the first evaluation, step 500, for seed 0 at fraction .4. Both cross .99, but they
are not equal. Values are copied from the recorded output log without rounding.*

| Evaluation cells | Accuracy |
|---|---|
| Training cells | 1.0 |
| Held-out cells | 0.9984567761421204 |

*Table B2. Random frozen-base control, final held-out accuracy over 3 seeds per fraction. The pooled range .24–.40 in Table 2 keeps
the earlier summary; these source ranges separate the training fractions.*

| Training fraction | Final test accuracy |
|---|---|
| .4 | .24–.26 |
| .6 | .31–.34 |
| .8 | .32–.40 |

## What to read next

Compare [two-count comparison](compare-counts.md): the same frozen encodings support a different readout, with held-out ties remaining a failure. Read [the verification guide](../GLOSSARY.md#verifying-a-circuit) for what substitution alone establishes.

## References

- Behrens, F., Biggio, L., & Zdeborová, L. (2024). Counting in small transformers: The delicate interplay between attention and
  feed-forward layers. *ICML 2025 (arXiv preprint)*. https://doi.org/10.48550/arxiv.2407.11542 (counting and histogram tasks)
