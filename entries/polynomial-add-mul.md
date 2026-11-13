# Integer polynomial addition and multiplication

Adding (1+2x) and (3+x²) gives 4+2x+x². This combines coefficients, the numbers in front of matching powers of x. Multiplication also needs products across different powers. Small networks learned addition on degree-3 inputs in 40/40 final runs across direct answers, written working, hand-built retrieval and training from scratch. Each got whole outputs right on 1000 held-out pairs, with correct preceding tokens supplied. No learned multiplication experiment was exact.

Networks trained from scratch with working read coefficients back from that working to produce the sum. Supplying hand-built products and training only their sum helped multiplication, but still stopped short of exact outputs. Learning to combine products differed from learning to construct them.

The task is partly solved. The [addition formula is partial](../GLOSSARY.md#verifying-a-circuit): replacement preserved in-range answers only for partly trained networks. It repaired rather than reproduced their out-of-range errors, and swapping the other parts' signals between examples was not tested.

| Fact | Value |
|---|---|
| Task | Add or multiply two degree-3 integer polynomials, one token per signed coefficient |
| Input → output example | Addition, answer only: `BOS 1 2 0 0 + 3 0 1 0 =` → `4 2 1 0` (coefficients a_0 first: (1, 2, 0, 0) + (3, 0, 1, 0) = (4, 2, 1, 0)) |
| Model | GPT-NeoX-shaped, NoPE, no LayerNorm, d_model 96, 3 layers (answer only) or 4 (trace), MLP 128 ReLU; the partly trained networks train only one stage MLP |
| Rungs and seeds | [HAND / PART / TRAIN](../GLOSSARY.md#the-three-rungs): formula-set weights / one trained stage / all weights trained; HAND for each operation and format (4 networks); addition PART and TRAIN, answer only and trace, 10 seeds each (final run); multiplication 3-seed screenings and arms only |
| Closed form status | Partial (addition PART scale-ratio form: exact 1.0 on substitution in 40/40 best/final snapshots, seeds 0–9 per format; complement not tested; TRAIN trace rejected, TRAIN answer-only decoder lossy) |
| Status | Partly solved (addition 40/40 seeds at held-out whole-output exact match 1.0; multiplication split-sum 3/3 at .998–.999, none at 1.0; no multiplication final run); last run 2026-11-08 |
| Weights | [Hugging Face: polynomial-add-mul](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-add-mul) (4 HAND + 55 nets) |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

Read next: [Results](#4-what-training-finds) and [verification](#5-mechanism-and-verification), then [polynomial differentiation](polynomial-differentiation.md) for coefficient arithmetic, or [symbol-level algebra](symbolic-algebra-atoms.md) for collecting and expanding terms.

## The circuit at a glance

Figure 4 shows the partly trained answer-only addition cell ([mechanism and verification](#5-mechanism-and-verification)).

<img src="../figures/circuits/polynomial-add-mul.svg" width="640" alt="Addition writes a value and scale; the fixed output reads their ratio. One upward residual stream has separate read/write branches for Read first list (coefficient fetch); Read other list (coefficient fetch); Add this pair (write value + scale). Add lists: (1, 2, 0, 0) + (3, 0, 1, 0); Constant-term target: 4. Task values are illustrative; dashed boxes and branches are not verified.">

*Figure 4. Final partly trained answer-only addition seeds 0–9 match the hand-built in-range sum through a value/scale ratio, a partial substitution form with no complement test that repairs rather than describes out-of-range errors; multiplication is not covered.*

## 1. Task and data

The prompt is `BOS a_0..a_3 (+|*) b_0..b_3 =` followed by the answer, with one token per signed integer (vocabulary 3156, |v| ≤ 1575).
Addition writes c_k = a_k + b_k; multiplication writes c_k = Σ_j a_{k−j} b_j, the convolution: sum all products whose powers add to k.
For example, `(1, 2, 0, 0) + (3, 0, 1, 0)` gives `(4, 2, 1, 0)`, with the constant coefficient first. Training coefficients range over
−9..9 without ±8, which are held out. Fresh pairs were drawn for every batch, and the train/test split is by
input hash mod 97, with 1000 test pairs per seed. An output counts as correct only on exact match of the whole teacher-forced answer
([glossary](../GLOSSARY.md#task-formats)): each token is scored given the correct previous tokens. Extension sets of 256 sequences
test degrees 0–2 and 4–6 and the held-out coefficients ±8.

Two output formats were studied (Table 1). The trace is a [scratchpad](../GLOSSARY.md#task-formats), in which the network writes
intermediate results before the answer.

*Table 1. Output formats.*

| Format | Output |
|---|---|
| `A` (answer only) | the coefficients c_k |
| `T` (trace) | addition: rows `a_k b_k c_k ;`; multiplication: partial polynomials P_j = P_{j−1} + b_j x^j A |

## 2. Model

The shared architecture and instrumentation are described in [METHODS.md](../METHODS.md). The three
[rungs](../GLOSSARY.md#the-three-rungs) are HAND (every weight set by formula), PART (frozen hand-built parts with one trained stage)
and TRAIN (all weights trained from scratch).

### 2.1 Architecture

All networks are GPT-NeoX-shaped with no positional encoding (NoPE), no LayerNorm, ReLU MLPs and an untied lm_head. d_model is 96
(4 heads × 24; multiplication `A` uses 112, 14 × 8). Format `A` has 3 layers and format `T` 4. The MLP width is 128 (2x arms 256).

### 2.2 Frozen parts and trained part

PART freezes the whole HAND network except the stage MLP (layer 2 in format `A`, layer 3 in format `T`), which is re-initialised and
trained: W_in [128, d], b_in, W_out [d, 128] and b_out. The split-sum arm trains W_out and b_out only, so the hand-built product
gates stay frozen. The stage reads blocks of the residual stream, the running vector at each position
([glossary](../GLOSSARY.md#standard-machine-learning-terms)), listed in Table 2. S is the score numerator and ONE its scale; the output
represents the answer as the ratio S/ONE. TRAIN trains all weights from random initialisation.

*Table 2. Residual blocks read by the stage MLP (index ranges, as recorded).*

| Block | Indices | Content |
|---|---|---|
| ONE | 8 | ONE, the read-out scale |
| flags | 6–39 | flags |
| AA | 41–47 | a |
| BB | 48–82 | b, or its bits |
| S | 83 | S, read by the read-out |

### 2.3 Training

The optimiser was AdamW with learning rate 1e-3, weight decay .1, betas (.9, .98), warmup 1000 and a cosine schedule over 30k steps
(60k in one arm), gradient clip 1 and batches of 128 fresh pairs. Test exact match was evaluated every 500 steps, and the best snapshot
is the one with the highest held-out exact match. A run stopped early when all seeds were exact (TRAIN after a minimum of 22.5k steps)
or on a 5k-step plateau after 75% of the budget. PART ran in fp32 and TRAIN in TF32, with re-scores in fp32 on CPU. 10 seeds were
packed per process on one consumer GPU.

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-add-mul](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-add-mul).

### 2.5 Archive

The Hugging Face folder holds 4 HAND networks and 55 trained networks (best snapshots; PART files hold the trained tensors), a config
JSON, a manifest (seed, rung, accuracy, sha256) and the test pairs of each run (`test_<run>`).

## 3. Hand-built solution

The hand-built network fetches the operand coefficients by position and combines them with ReLU ramps or bit gates (Table F1).

A reciprocal-position frame gives every token a position scalar. Quadratic index heads, with score −β(r_key − target)², fetch a_k and
b_k in format `A`, or the shifted a, b_j and the previous coefficient in format `T`. Addition writes S = relu(a + b) − relu(−a − b),
which is a + b (in format `T`, ±a and ±b ramps per row). Multiplication writes b_j in 5-bit two's complement and forms each product
with signed bit gates a·bit = relu(a + 32 bit − 32) − relu(−a + 32 bit − 32), summed with weights (1, 2, 4, 8, −16). The read-out has
logit(v) = 2vS − v²ONE, whose maximum is at the integer closest to v* = S/ONE.

The sizes in heads and neurons are 4+48 for addition `A`, 4+111 for addition `T`, 16+170 for multiplication `A` and 5+307 for
multiplication `T`. The networks are exact on 17,676 stratified sequences (degree 0–6, |coefficient| ≤ 15), with minimum margin .9985. This is 4,419
full-polynomial checks per cell, alongside exhaustive scalar primitives (31^2 signed factor pairs), not exhaustive enumeration of
the 19^8 degree-3 input combinations.

Table F1 records the two alternative coefficient operations and their common readout. The separate split-sum arm in Table F3 trains only output weights after supplied product gates; it is not this full-stage PART comparison.

<details>
<summary>Technical coefficient readers, sum/product gates and readout (Table F1)</summary>

*Table F1. The hand-built networks for addition and multiplication. Each operation uses the common input and readout; addition and multiplication are alternatives, not successive stages. Index heads on a reciprocal-position frame fetch the operands; the stage feed-forward block (MLP) adds them with two ReLU ramps or multiplies them with signed bit gates on b_j; the readout picks the integer closest to S/ONE. Only the stage MLP is trained in PART (layer 2 in format A, layer 3 in format T).*

| Part or alternative operation | Input, formula and destination |
|---|---|
| Common input and frame; frozen | Prompt `a_0..a_3, + or *, b_0..b_3, =` supplies coefficients to a reciprocal-position frame. |
| Quadratic index heads; frozen | Fetch a_k and b_k, or shifted a, b_j and the previous coefficient, for the current operation. |
| Addition stage; trained in PART | S = ReLU(a + b) − ReLU(−a − b), a signed sum from two threshold ramps. |
| Multiplication stage; alternative, trained in PART | Signed bit gates ReLU(a + 32·bit − 32) − ReLU(−a + 32·bit − 32), with weights 1, 2, 4, 8, −16, combine products selected by bits of b_j. |
| Common integer readout; frozen | Score candidate v by logit(v) = 2vS − v²·ONE; choose the integer nearest S/ONE, with ONE the scale channel. |
| Separate split-sum arm | Frozen product gates supply the terms; only output weights are trained. Its screening results do not establish a verified multiplication mechanism. |

</details>

## 4. What training finds

Table 3 gives the final run of addition: PART and TRAIN, formats `A` and `T`, 10 seeds each. Every addition cell solved 10/10 seeds (40/40 in all); the best snapshots of the networks trained on frozen parts came at 1–1.5k (`A`) and
1.5–2k (`T`) steps, those of the networks trained from scratch at 16.5–26.5k and 12.5–23k.

*Table 3. Addition, final run, 10 seeds per cell (seeds 0–9).*

| cell | solved | best step | H+N | R² a / b / coefficient | rank | JS | acc |
|---|---|---|---|---|---|---|---|
| PART add A | 10/10 | 1–1.5k | 2 + 35–84 | 1.0 / 1.0 / 1.0 | 4/3 | 0 | 1.0/1.0 |
| PART add T | 10/10 | 1.5–2k | 2 + 86–102 | 1.0 / 1.0 / .97–.98 | 5–6/6 | 0 | 1.0/1.0 |
| TRAIN add A | 10/10 | 16.5–26.5k | 5–12 + 23–66 | .53–1.0 / .52–1.0 / .87–1.0 | 3–5/3 | 0–.69 | 1.0/1.0 |
| TRAIN add T | 10/10 | 12.5–23k | 2–14 + 26–72 | .75–.93 / .48–.60 / .97–1.0 | 7–10/6 | 0–.69 | 1.0/1.0 |

Key: H+N = heads and neurons whose mean [ablation](../GLOSSARY.md#verifying-a-circuit) drops exact match by > .05; R² = best linear
probe ([glossary](../GLOSSARY.md#analysis-measures); `A` at L3, `T` at L4); rank = residual r99, the rank at 99% of the variance (net/HAND);
JS = attention divergence to the matching HAND head; acc = exact match best/final.

### 4.1 Addition: circuit types

The networks fall into one [circuit type](../GLOSSARY.md#counting-results) per PART cell and two in TRAIN `A`. PART `A` is a
fixed-fetch sum (10/10). PART `T` is input-dominant (10/10): its sum rows follow the operands, with patching scores of .996–1.0 for a,
.75–.95 for b and 0–.06 for the trace. TRAIN `A` splits into a sparse-fetch type with 5 needed heads (2/10) and a distributed type with 8–12
(8/10). TRAIN `T` is trace-dominant (10/10): it reads its own written a_k and b_k (patching .79–.87) and not the operands in the prompt
(0).

### 4.2 Multiplication

No learned multiplication cell reached whole-output exact match 1.0 (Table 4). PART reached .004/.003 in format `T` (best/final)
although the product was partly decodable (product R² .76–.80). TRAIN `T` reached .05–.17, and its 60k arm .993/.322/.996 in its
three seeds: 2/3 at ≥ .99, still none at 1.0. Arms with double width, a running sum, or trained split gates solved 0/3. When the
hand-built product gates stayed frozen and only W_out, b_out were trained (split-sum, PART `A`), 3/3 seeds scored .998–.999. This
separates the difficulty of building signed products from learning their output sum.

*Table 4. Multiplication screenings, 3 seeds per cell (best/final exact match).*

| Cell | Exact match |
|---|---|
| PART mul A | 0/0 |
| PART mul T | .004/.003 (product R² .76–.80) |
| TRAIN mul A | .002/.001 |
| TRAIN mul T | .05–.17 (60k arm .993/.322/.996) |
| width, running-sum and split-gates arms | 0/3 |
| PART mul A split-sum (gates frozen) | 3/3 at .998–.999 |

### 4.3 Length and range

Length was tested on the extension sets (best snapshot, seed range). PART `A` scored .98–1.0 at degree 4, .72–1.0 at
degree 5 and .26–.93 at degree 6; PART `T` scored .98–1.0, .90–1.0 and .62–.89. TRAIN `A` and `T` scored ≤ .13 at every degree but 3,
which the analysis reads as absolute routing by position. On the held-out coefficients ±8, PART `A` scored 1.0 and TRAIN ≤ .03.

## 5. Mechanism and verification

### 5.1 Closed form

For PART addition, the stage output is (S, ONE) = κ_g·(S, ONE)_HAND, with S_HAND = a + b (in format `T`, a, b or a + b by row) and one
scale κ per row group (`A` 2.2–4.9; `T` four groups 1.9–7.1); the other dimensions the stage writes are constant within a group.
Substituted for the stage MLP on 1000 held-out pairs, the closed form keeps exact 1.0 on 40/40 best/final snapshots (seeds 0–9 in
each of the two PART formats, 20 snapshots per format, not 40 independent seeds); the bare HAND stage also scores 40/40. This is a
partial form, supported by substitution without a complement-resampling test. The networks' |S/ONE − HAND S/ONE| is .03–.07 (`A`) and .26–.32 (`T`), under .5, so the
read-out picks the same integer. A per-dimension linear fit in (a, b, 1) fails (.001–.057, 0/40), because it misses the ratio S/ONE that the read-out uses.
The closed form is supported by substitution; no [resample ablation of the complement](../GLOSSARY.md#verifying-a-circuit) was run,
so its status is partial.

For TRAIN, a fitted MLP2 decoder in format `A` scores .993–1.0 against the network's .999–1.0 (seeds 0–2), so it is lossy. In format
`T` the closed form was rejected (.05–.59 against 1.0, seeds 0–2).

Table F2 distinguishes the supplied earlier trace from the original prompt. Reading the correct trace under teacher forcing does not establish unaided generation or an accepted TRAIN arithmetic formula.

<details>
<summary>Technical trace reuse and rejected from-scratch sum formula (Table F2)</summary>

*Table F2. The circuit type training from scratch finds in format T (10/10 seeds): the sum row reads the network's own written a_k and b_k (patching .79–.87) and not the prompt operands (0). The retrieval/readout is trained. This TRAIN closed form was rejected (.05–.59 against 1.0); the entry's partial status applies to the separate PART sum, whose complement was not tested.*

| Source, component or test | Evidence and boundary |
|---|---|
| Original prompt operands a_k and b_k | Patching contribution 0 to the tested sum row. This is the unused route in that test, not a claim that prompts are never used elsewhere. |
| Written trace operands a_k and b_k | Patching .79–.87. Scoring supplies the correct earlier tokens, even though those slots represent the network's preceding writes. |
| Attention and feed-forward blocks (MLPs); trained | Retrieve from the trace and write coefficient c_k on the sum row. |
| Proposed TRAIN formula; rejected | Substitution .05–.59 against native 1.0 does not support a recovered from-scratch arithmetic law. |
| Separate PART sum | Partial substitution support only, with complement untested; it must not be promoted by this trace-reuse result. |

</details>

### 5.2 Relation to the hand-built network and to published work

PART addition admits HAND's sum with a free logit scale as a partial, substitution-supported form; its complement was not tested.
TRAIN uses degree-specific routing, and in format `T` it routes through its own trace (Table F2). The separate split-sum arm (Table F3)
shows that accumulation can be learned once the products are supplied. [Madsen and Johansen (2020)](#references) studied structured
learned arithmetic units, [Trask et al. (2018)](#references) studied approximate product fitting and extrapolation, and
[Bai et al. (2025)](#references) analysed long-range partial-product dependencies in transformers. These comparisons do not establish
that the failed cells here are impossible to learn.

*Table F3. Supplied products make the output sum easier to learn, but leave errors. This is the separate answer-only split-sum multiplication arm, not the main ten-seed result. Its 3/3 screening seeds reached .998–.999 whole-output exact match on 1000 held-out pairs per seed, none 1.0. No verified learned multiplication mechanism is claimed; the entry's partial closed form covers addition only, without a complement test.*

| Part of the computation | Fixed or trained | Role |
|---|---|---|
| Operand-coefficient readers | Frozen | Read the input numbers for each power. |
| Signed-product gates | Frozen | Supply the products to be added. |
| Output weights W_out and bias b_out | Trained | Combine those products into coefficient scores. |

## 6. Limitations and open questions

κ was fitted on 512 disjoint pairs. Outside the training range the κ form scores 1.0 where the networks fail, so it is not a model of
their errors. No resample ablation of the complement was run. Multiplication had no final run, only 3-seed screenings and arms.
HAND full-polynomial checks were stratified rather than exhaustive at degree 3. The two extension evaluations in Section 4.3 and
Appendix B have different recorded ranges and are kept separately.

## 7. Prediction record

The predictions were committed before any run (in the task's prediction file). PART addition `A` 10/10 within 5k steps and addition
`T` ≥ 9/10 held. TRAIN addition `A` and `T` ≥ 8/10 within 15k held in count but missed in speed (best 12.5–26.5k). PART multiplication
`A` and `T` ≥ 8/10 was refuted (0/3). TRAIN multiplication `A` ≤ 3/10 held (0/3). TRAIN multiplication `T` ≥ 6/10 in 10–25k was refuted
(0/3; 2/3 ≥ .99 only at 60k). TRAIN at degree ≥ 4 below .50 held, but degrees 0–2 failed too, which was not predicted. PART above TRAIN
out of range held.

## Appendix A. Weights excerpt and reading

Table A1 gives weight slices of the stage MLP: for each neuron, its input weights on (a, b, ONE), its bias, and its output weights
into S and ONE.

*Table A1. Stage MLP rows. HAND rows are formula weights; PART rows are from the archived best snapshots.*

| Network | Neuron | W_in [a, b, ONE] | bias | W_out[S], W_out[ONE] |
|---|---|---|---|---|
| HAND `A` | ramp + | [1, 1, 0] | 0 | 1 |
| HAND `A` | ramp − | [−1, −1, 0] | 0 | −1 |
| PART `A` seed 0 | n84 | [.49, .50, .59] | .59 | .321, .026 |
| PART `A` seed 0 | n83 | [−.50, −.48, .61] | .59 | −.289, .024 |
| PART `T` seed 0 | n111 | [−.29, .37, −.33] | −.33 | −1.02, −.22 |

In the best PART `A` networks, 80–100% of the top-20 S writers read a and b with equal weight. Rows such as n84 and n83 are HAND's
±(a + b) ramps duplicated with offsets, each also writing ONE, which sets κ. PART `T` has 0% equal-weight rows: a and b enter with
opposite signs, in phase-gated rows. In the multiplication split-sum arm, seed 0's W_out[S] on HAND gates 0/1/8/9 is .0765, −.0765,
−1.2187, 1.2186 (ratio 15.93 ≈ 1 : −16), the two's-complement weight of the sign bit.

## Appendix B. Detailed evidence

The addition substitution evaluation used 20 best/final snapshots per format, 512 disjoint fit pairs and 1000 held-out test pairs.
Its native extension scores differ from the best-snapshot extension sets in Section 4.3; neither range replaces the other.

*Table B1. Native addition PART extension scores on the substitution evaluation batch, not the Section 4.3 extension batch.*

| Format | Held-out coefficients ±8 | Degree 6 |
|---|---|---|
| `A` | .996–1.0 | .27–.92 |
| `T` | .25–.51 | .55–.88 |

Key: ranges span the best/final snapshots of seeds 0–9 per format. Both the bare HAND stage and the scale-ratio replacement scored
1.0 on these extension tests, repairing native errors rather than reproducing them. The fitted TRAIN decoder and rejected trace
formula covered seeds 0–2 only.

## References

- Bai, X., Pres, I., Deng, Y., et al. (2025). Why can't transformers learn multiplication? Reverse-engineering reveals long-range
  dependency pitfalls. *arXiv preprint*. https://doi.org/10.48550/arxiv.2510.00184 (multiplication needs long-range partial-product
  dependencies)
- Madsen, A., & Johansen, A. R. (2020). Neural arithmetic units. *arXiv preprint*. https://doi.org/10.48550/arxiv.2001.05016 (exact
  learned multiplication needs structure)
- Trask, A., Hill, F., Reed, S., Rae, J., Dyer, C., & Blunsom, P. (2018). Neural arithmetic logic units. *arXiv preprint*.
  https://doi.org/10.48550/arxiv.1808.00508 (MLPs fit products only approximately)
