# Methods

This document describes the methods shared by all entries of the atlas: the three experimental rungs, the frozen hand-built model and
menu used by the counting and modular-addition entries, the small whole-network models, the instrumentation, the diagnostics applied
to every trained network, and the rule by which a closed form is accepted. Each entry gives its task-specific values in its Section 2
(Model), including the training settings and, where available, the command that reproduces it. The code in [code/](code/README.md)
(MIT licence) reruns the five entries listed in [code/README.md](code/README.md); the other entries publish their networks, test sets
and recorded accuracies on Hugging Face (links in each entry's fact table). The numbers below are those of the code. Plain-language definitions of the terms are in
[GLOSSARY.md](GLOSSARY.md), and works are cited by author and year with links to [REFERENCES.md](REFERENCES.md).

## 1. Three rungs

Every task is studied at up to three rungs (Table 1). The hand-built network (HAND) has every weight set by a formula and is checked
exact on all inputs in range; it is an existence proof and a reference algorithm, not a claim about what training finds. In the PART
rung the hand-built parts are frozen and one stage is trained on top of them, so the experiment shows which frozen parts training
reads and which algorithm it builds from them. In the TRAIN rung the whole network is trained from scratch, which shows whether the
same algorithm appears without hand-built help. Final cells use 10 seeds; screening cells use 3 (sometimes 5).

*Table 1. The three rungs.*

| Rung | What is trained | What it shows |
|---|---|---|
| HAND | nothing: every weight is set by a formula and checked exact on all inputs in range | an existence proof and a reference algorithm |
| PART | one trained stage on top of frozen hand-built parts, 10 seeds (fewer in screening cells) | which frozen parts training reads, which algorithm it builds from them |
| TRAIN | a network trained from scratch, 10 seeds (fewer in screening cells) | whether the same algorithm appears without any hand-built help |

The first entries (the count and modular-addition tasks on the frozen model m2 of Section 2) are PART experiments with controls; most
later entries cover all three rungs, and the code release has the HAND networks and a single-network PART / TRAIN trainer for base
conversion (Section 7). For each trained network the analysis reports an ablation map (which frozen parts are read), the circuit type
(seeds clustered by mechanism, with seed counts), a closed form and its verification status (Section 9), and the learning curve with
its grokking step ([Power et al., 2022](REFERENCES.md#power2022)).

## 2. The frozen hand-built model ("m2")

The counting and modular-addition entries read their prompts with m2, a GPT-NeoX-shaped decoder written without training. It is built
by `code/m2.py` and evaluated by `code/neox.py`, a 40-line forward pass. The state dict uses the Hugging Face `GPTNeoXForCausalLM`
parameter names and loads there, and the two forward passes agree bit for bit on a test prompt. Table 2 lists its dimensions.

*Table 2. Dimensions and conventions of m2.*

| Item | Value |
|---|---|
| Layers / heads / d_model / d_head / MLP width | 6 / 4 / 528 / 132 / 512 |
| Activation, norm, positions | exact GELU; LayerNorm eps 1e-5, every gain a fixed constant, bias 0; no positional embedding and no rotary |
| Residual | sequential (attention, then MLP, each added to the stream) |
| Vocabulary | character level, 70 ids: 0 end-of-text, 1 unknown, 2..69 the 68 characters of the prompts (letters, digits, `( ) : = < > - + | _ \n` and space) |
| Residual layout | a fixed block of dimensions per quantity (`DIMS` in `m2.py`): token one-hot, prefix mean, 1/position, offsets, copy pointer, last letter, count X, depth ratio BAL, two numbers NA/NB, output logits, ... |
| Training | none; the weights are formulas, with only 0 and a few constants (recency 15 per position, class gate 5000, token match 1500, readout 30) |

Each brick of m2 is a head or a group of MLP neurons, and `BRICKS` in `m2.py` records what each writes (Table 3).

*Table 3. Bricks of m2 by layer.*

| Layer | Attention heads | MLP |
|---|---|---|
| 0 | prefix mean of tokens | first-token flags (task keyword) |
| 1 | prefix mean of the flag (gives 1/position) | position: piecewise-linear 1/p, 500 knots |
| 2 | latest `:` anchor; last lower-case letter; last newline | line offset (steps) |
| 3 | copy (token at fixed offset); key-value match; has-letter; bracket-balance prefix | negative-balance flag |
| 4 | contains-letter; count (X = 1/(c+1), uniform attention over the matching letters of the current line); previous digits 1 and 2 | router (copy / bigram / yes-no), count digit decoder (step ladder) |
| 5 | bracket segment (BAL = b/(L+1), ANY); compare number readers NA and NB | bracket-validity answer; compare answer (`=` is the default output) |

The MLPs are built from step units. A step is a pair of GELU neurons, `step(z; c) = (gelu(40(z-c)+8) - gelu(40(z-c)-8))/16`, which is
exactly 0 for z <= c - 0.4 and 1 for z >= c + 0.4 on integer-valued inputs. The exactness checks (`python run.py check`) confirm that
the count head with its digit decoder answers 0..9 correctly for lines of 9 letters, that the menu's one-hot is exact to 4e-6 and that
the scalar number readers are exact to 2e-4 (modular addition). The count of m2 is exact only up to L = 45 items.

LayerNorm is made harmless by fixing its scale: every LayerNorm gain is a constant and each reader row sums to zero over the residual
(a dimension that is always 0 absorbs the sum), so each LayerNorm acts as the identity map up to 1e-5.

## 3. The frozen menu and the trained stage

### 3.1 Menu

At the answer position (the final space after `ANSWER:`) the residual of m2 holds the task's integers as scalars, written by the
number readers. A hand-written frozen menu layer re-encodes each integer in three ways, using the same step units (Table 4). Every
block is centred (mean over all cells) and divided by its rms row norm, so that each block has rms 1 (constant blocks become 0). The
total menu width is 310 (modular addition), 326 (count tasks) or 416 (open-bracket count).

*Table 4. Encodings of the menu.*

| Encoding | Block | Dimensions |
|---|---|---|
| scalar | raw value from m2 (x/100, 1/(c+1), depth ratio, position) | 1 per integer |
| one-hot | `step(z; k-1/2) - step(z; k+1/2)`, one unit per value | one per value |
| thermometer | `[z >= k]`, `step(z; k-1/2)` | one per threshold |
| distractors | other m2 blocks at the slot: copy pointer (70), last letter (70), bracket balance, count, position; those constant on a task are 0 | varies |

### 3.2 Trained stage

The trained stage (`code/train.py`) has one hidden layer, `logits = W_out ReLU(W_in x + b_in) + b_out`, with hidden width 256, one
output per class and the default PyTorch initialisation (seeded). Only this MLP is trained; m2 and the menu are frozen and are cached
once, since they do not change. Table 5 gives the training settings.

*Table 5. Training settings of the trained stage.*

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 1e-3, weight decay 1.0 (decoupled, a very strong value that drives grokking), betas (0.9, 0.98), eps default |
| Batch | full batch (all training cells every step), cross-entropy loss, fp32 |
| Steps | modular addition 300,000; count tasks 200,000. No early stopping: every run uses its full budget |
| Evaluation | every 500 steps: train and test accuracy and loss, weight norm, per-block input-weight norm |
| Seeds | seed s sets both the initialisation and the train/test split; seeds 0..9 |
| Split | a random permutation of the cells; the first `int(frac * N)` are training cells, the rest test. Modular addition frac 0.4 of 59^2 = 3481 pairs. Count tasks fracs 0.4 / 0.6 / 0.8 of 1080 cells |
| Controls | `random`: the frozen base replaced by a random-weight model of the same shape (weights N(0, 0.02), seeds 1000+s), its final residual z-scored per dimension and rescaled to the menu's total norm (3 seeds); `scalar`: the menu reduced to the two raw scalars (3 seeds) |

### 3.3 Packing and hardware

The published runs were executed in two ways with identical settings. Modular addition and the 0.4 fraction of the count tasks were
trained one network per process on consumer CPU machines (14 to 29 minutes per network at 300,000 steps). The 0.6 and 0.8 fractions
were trained with all networks of a batch in one process on one consumer 12 GB GPU, stacked as batched matrix multiplies with padded
inputs (extra input weights start at 0 and receive no gradient; absent classes are masked; the loss is the sum of each network's own
mean cross-entropy; fused AdamW; fp32, TF32 not enabled). This is exactly independent training: batched and unbatched runs agree to 3
decimals. `code/` trains one network per process (`train.py`, and `convert.py` for base conversion) and needs no GPU; a 200,000-step
count network takes about 5 minutes on 4 CPU threads, an open-bracket network at fraction .8 about 17 minutes on a busy machine, plus
about a minute to build the menu.

## 4. Tasks and prompt generation

All prompts are generated, not sampled from a corpus. There is one prompt per cell, and the split is over cells, so a test cell is a
never-seen (c, L) or (x, y) combination. The modular-addition prompt is `SUM: x y\nANSWER: ` with x, y in 0..58 and the answer
(x + y) mod 59 (59 classes). The count prompts are `COUNT b: a b\nANSWER: 1\nCOUNT a: <line>\nANSWER: `, where the line holds c letters
`a` and L - c letters `b`, space-separated and shuffled (Python `random.Random(0)`, consumed once per cell in cell order), with
L = 1..45 and c = 0..L, which gives 1080 cells. Their labels are the count mod 5 (5 classes) and the comparison (a if 2c > L, b if
2c < L, `=` if 2c = L: 3 classes, 22 `=` cells), with (u, v) = (c, L). The open-bracket prompt is
`VALID_BRACKETS: ()\nANSWER: YES\nVALID_BRACKETS: <line>\nANSWER: `, with a shuffled line of (L+b)/2 `(` and (L-b)/2 `)`, L = 1..45 and
b = -L..L in steps of 2 (1080 cells); the label is the number of `(`, (L+b)/2 (46 classes), with (u, v) = (b, L).

The fixed worked-example line (the shot) before the question is needed: without a preceding newline the has-letter head of m2 misses the question letter.

## 5. Instrumentation (the analysis code)

This section describes the analysis of the trained stage on m2 (`analyze_atlas.py`, `analyze_modadd.py`). Accuracy is always arg-max
accuracy (the class with the highest logit is the answer) on the true task labels; "test" means held-out cells and "all" means every cell.

### 5.1 Learning curve

The curve is logged every 500 steps. The fit step is the first step with train accuracy >= 0.99. The grok step is the first step with
test accuracy >= a threshold (the entries quote 50%, 90% and 99%). "No grokking" means that test accuracy equals train accuracy at the
first evaluation.

### 5.2 Encoding-read ablations

For a block of the menu, or an encoding pair such as both one-hots (`usage`, and pair versions), *ablate* replaces its columns by their
mean over all cells and reports test accuracy; *keep only* ablates everything else; *resample* replaces the block by that of a random
other cell. *contrib* is the rms of that block's input to the hidden pre-activations.

### 5.3 Separable effective embeddings

The hidden pre-activation of neuron n is fitted over cells as `c_n + E_u[n][u] + E_v[n][v]` (least squares with one free value per
integer); the unexplained share is `sep_resid`. Substituting this separable form for the trained first layer, with the trained W_out
kept, gives `form_sep`.

### 5.4 Substitution of a closed form

A closed form is a short formula for what the trained part computes. It is substituted for the trained part, with the trained output
layer kept unless stated. *Accuracy* is the arg-max accuracy of the substituted network on all cells against the true labels;
*agreement* is the fraction of cells where the substituted and the trained network give the same arg-max. The variants used are:

- `form_u_only`: the second integer's embedding replaced by its mean (does the net ignore it).
- `form_res5` (count mod 5): the count's embedding replaced by its 5 residue-class means.
- `form_fourier` (count mod 5): logits replaced by `A cos(2pi(c-r)/5) + B cos(4pi(c-r)/5)`, with (A, B) fitted by least squares per network (`fourier_r2`).
- `form_line` (compare): the logit difference a - b fitted by `alpha c + beta L + gamma`; the closed form is the sign of the fit, and `=` is never output.
- `form_clock` (open-bracket count): each neuron's pre-activation replaced by `c_n + a cos(w b/2 + phi) + a' cos(w L/2 + psi)`, `w = 2pi k/92`, with k in 1..45 chosen per neuron by least squares.
- modular addition (`analyze_modadd.py`): the per-neuron frequency k from the DFT over x of the neuron's input-weight embedding; the neuron formula `ReLU(c_n + a cos(w_k x + phi) + b cos(w_k y + psi))` with the trained W_out kept (accuracy / agreement, "neuron formula"), and the fully analytic layer with `W_out[n, c] = A cos(w_k c + phi + psi)` ("analytic"). Phase rule: median |theta - phi - psi|.

Substitution alone shows that a formula is sufficient to read out the answer. Whether it counts as verified is decided by the rule of
Section 9.

### 5.5 Frequency and Fourier analysis (modular addition)

The separable embeddings E_x and E_y are transformed by a DFT over x (k = 1..29, `w = 2 pi k/59`) after removing their linear-in-x
part; key frequencies carry >= 5% of the power. The logits are fitted on the key frequencies with the clock basis `cos/sin(w_k(x+y-c))`
of [Nanda et al. (2023)](REFERENCES.md#nanda2023) and the pizza basis of [Zhong et al. (2023)](REFERENCES.md#zhong2023) (the same
times `|cos(w_k(x-y)/2)|`); `dd` is the share of the correct-logit variance explained by x - y (clock ~ 0). A network is typed as
*memorize* if test accuracy < 0.9, else *lookup* if the top-5 frequencies hold < 50% of the power, else *pizza* if the pizza fit beats
the clock fit by 0.05 and dd > 0.2, *clock* if dd < 0.1, and *mixed* otherwise.

### 5.6 Circuit-type labels

The labels (`typ` in `analyze_atlas.py`, thresholds set from a pilot) summarise the measured forms; they are not an independent
classifier. Count mod 5 is a *residue lookup* if `form_res5` and `form_u_only` >= 0.99. Compare is a *linear comparator, `=` memorized*
if `cmp_r2` >= 0.7 and `=` test accuracy < 0.5, and an *exact comparator* if test accuracy >= 0.99. Open-bracket count is a *low-k clock
sum* if test accuracy >= 0.5 and `form_clock` >= 0.85, and *memorize* if test accuracy < 0.2.

### 5.7 Ridge probes of the random-base control

From the z-scored residual of the random base, ridge regression (lambda 1e-2, intercept included) predicts integer quantities (c, L,
2c - L, 1/(c+1), c mod 5 = 0, b, (L+b)/2); it is fitted on the training cells of the network's split and scored by R^2 on its test
cells. The control's finding is R^2 near 1 for c, L and 2c - L and near 0 for `c mod 5 = 0`.

### 5.8 Causal use of a variable: interchange and DAS

A linear probe shows that a variable is decodable, not that the network uses it. *Interchange* replaces the probed component of a
variable in one input's residual by that of another input and counts how often the answer flips as the variable predicts
([Geiger et al., 2023](REFERENCES.md#geiger2023)). It under-reads variables with many values or a positional code: in binary addition
(carry-chain format) the carry scores .16 by interchange but 1.00 with distributed alignment search (DAS, a learned rotation), and some
position variables score 0 by interchange even on the hand-built net. A variable is therefore read as unused only when DAS is also low.
Between-seed comparisons use linear mode connectivity, the loss barrier on the interpolation path
([Frankle et al., 2020](REFERENCES.md#frankle2020)); for networks on frozen parts the permutation-only alignment is the valid arm, since
a general rotation breaks the shared frozen basis.

## 6. Reproduction cost and tolerances

Everything runs on CPU. Menu construction (an m2 forward pass over all cells) takes about a minute, and training time scales with the
steps (see each entry). Re-running a seed reproduces its headline numbers (accuracies, circuit type) but not bit for bit: count-mod-5
seed 0 on CPU matches the published run to 3 decimals, and open-bracket seed 0 at fraction .8 gives test .787 against .782, a 50% step
of 21.0k against 23.5k and `form_clock` .917 against .930. The published networks were trained on different machines (CPU and GPU),
and the random control uses a re-implemented initialiser with the same distribution. The smoke mode (`--steps 400 --every 100`) only
checks that the pipeline runs; at that length the networks have not converged and the analysis numbers are not those of the entries.

## 7. Small two-layer networks (base conversion and other whole-network entries)

Entries that train or hand-build the whole network, rather than a stage on m2, use smaller GPT-NeoX models; each entry's Section 2
gives its dimensions. The released example is `code/convert.py` for base conversion: 2 layers, d_model 256, 4 heads of 64, MLP 256,
ReLU, no LayerNorm, no positional embedding or rotary, a sequential residual, a vocabulary of 18 tokens (start, 0-9, a-f, `=`) and an
untied readout. Position is built from attention: a head on the start token gives 1/(r+1)-like values from which an MLP interpolates a
position one-hot. Parameter names are those of Hugging Face `GPTNeoXForCausalLM` without the LayerNorm keys. Table 6 gives the rungs.

*Table 6. Rungs of the base-conversion networks.*

| Rung | What is trained |
|---|---|
| HAND | nothing; exact on all 256 inputs (`python run.py convert check`) |
| PART | the layer-1 MLP only (re-initialised N(0, .02), biases 0); embedding, layer 0, layer-1 attention and readout are the frozen hand-built ones |
| TRAIN | every weight, from N(0, .02) (biases 0) |

The data are all 256 inputs, one sequence each; seed s sets the initialisation, the batch order and the split (the first
`int(frac * 256)` of a random permutation train, the rest test). The loss is cross-entropy on the output tokens, and the metric is
exact match (every output token right, teacher forced). Training uses AdamW with lr 1e-3, weight decay 1.0, betas (.9, .98), minibatch
64 and full fp32. Reduced-precision matrix multiplication (TF32) takes the hand-built converters from 100% to about 2% exact, so it must
stay off. The published runs packed 10 seeds per process on a GPU (vmap) with early stopping (all seeds exact, or no seed's best test
accuracy up by .005 in 5,000 steps after 10,000 steps); `code/` trains one network at a time on CPU with the same objective and stops
when train and test are both exact.

The instrumentation of these networks is as follows. *Best* is the evaluation (every 500 steps) with the highest test accuracy; final
weights can be far worse, because training is unstable at weight decay 1. *Substitution* (`convert.substitute`) replaces, at every
scored position, the residual after the last attention layer by its least-squares fit (with intercept, over all 256 inputs) on HAND's
variable at that position, and measures the exact-match rate of the rest of the network; the variable is taken as the one the network
uses if the rate stays within .01 of the network's own. The fit uses test inputs too, so it says which variable the stage needs, not
that the network computes it cleanly. The *threshold check* (`convert.thresholds`, decimal to binary, answer only) finds per answer bit
the best theta in `pred = [F >= theta]` over HAND's remainder F, and its agreement with the network on all inputs. Reported sizes use
mean ablation of heads and neurons: a head or MLP is used if replacing its output by its mean over inputs costs more than 5% exact, and
the neuron count is the smallest set, ranked by output spread, that keeps exact within .01 when the rest are mean-ablated. The
input-independent position frame (the layer-0 head and MLP) is not counted.

## 8. Diagnostics of whole networks

Every trained whole network (PART and TRAIN, best and final snapshot, failed cells included) receives one standard set of diagnostics,
computed in fp32 on held-out inputs and compared with the HAND network of the same architecture where one exists. The script that
computes them is not yet part of `code/`; its outputs are in each entry's Appendix B.

The ablation map sets each head and each MLP to zero or to its per-position batch mean (query-mean ablation); a component is used when
exact match drops by more than .05. Ridge probes read each HAND variable from the residual after every layer and report held-out R²
(and one-hot accuracy for integer variables). The representation is compared with HAND in three ways that do not depend on a shared
basis: HAND-PC R², the held-out R² of HAND's top 8 residual principal components from the network's residual; linear centred kernel
alignment (CKA); and the participation ratio and the ranks at 90% and 99% of the variance. The weight matrices are compared through the
QK and OV circuits of each head ([Elhage et al., 2021](REFERENCES.md#elhage2021)) by the overlap of their top singular subspaces, which
is meaningful only when the residual basis is shared, that is in PART. Attention patterns are compared by the mean Jensen-Shannon
divergence (in nats) to every HAND head.

Causal use is tested by activation patching on a corrupted batch, reported as drop and recovery for each component
([Zhang & Nanda, 2023](REFERENCES.md#zhang2024)), with attribution patching as a screen; by linear interchange of each variable's
subspace (flip, with nf the number of answer-fixing queries; nf 0 means untested, not unused); and by DAS
([Geiger et al., 2023](REFERENCES.md#geiger2023)) when interchange is low or ambiguous. Edge attribution with integrated gradients
(EAP-IG) scores every connection; its result is trusted only when the rank agreement with exact single-edge patching (spearman_exact)
is at least .8, and the trusted edge set is compared with HAND's by Jaccard overlap and hand_recall. A weight alignment (orthogonal
Procrustes on the residual, heads and neurons matched one to one) maps the network onto HAND or onto another seed; its fit R² is read
only when the aligned network computes the same function, and the loss barrier along the linear path between two aligned seeds measures
linear mode connectivity ([Frankle et al., 2020](REFERENCES.md#frankle2020)). Number-token embeddings are described by their Fourier
power and by representational similarity (RSA) with a line, a circle, a helix and HAND's embedding rows. Restricted and excluded losses
([Nanda et al., 2023](REFERENCES.md#nanda2023)) test whether the HAND-variable subspaces carry what the answer needs at each layer.

Since the method revision of 2026-11-08, every trained network also receives a discovery pass that does not start from HAND, so that a
circuit matching no hand-built design is found and counted as its own circuit type. Automatic circuit discovery ranks edges by EAP-IG and
prunes them in the manner of ACDC ([Conmy et al., 2023](REFERENCES.md#conmy2023)) while exact match stays within .05 of the network's
on the fit inputs; its faithfulness is the exact match kept on held-out inputs with everything outside the discovered circuit
resample-ablated. A small sparse autoencoder on one residual or MLP site ([Bricken et al., 2023](REFERENCES.md#bricken2023);
[Cunningham et al., 2023](REFERENCES.md#cunningham2023)) gives a dictionary of features, which count only when they beat the raw basis
of the site and when splicing the reconstruction back keeps exact match near the network's. Path patching, layer-level ablation (which
finds heads that matter only together) and interchange split by input bucket complete the set. A weight-decay perturbation probe and the
top Hessian eigenvalue (the sharpest curvature of the loss) are recorded as well; both are correlational and are compared only within one run.

## 9. Verification and the acceptance rule

A closed form for a trained part is called *verified* only when it passes two tests on held-out inputs. First, substituting the closed
form for the trained part keeps exact match at no less than .90 of the network's own exact match. Second, a resample ablation of the
complement, in which every connection outside the claimed circuit carries its value from another input while the circuit itself runs
on the true input, also keeps exact match at no less than .90 of the network's own. The second test follows causal scrubbing
([Chan et al., 2022](REFERENCES.md#chan2022)): substitution shows that the formula is sufficient to produce the answer, but only the
complement test shows that the rest of the network is not needed for it. The complement is resampled over 3 draws and read against
three controls: zero and mean ablation of the complement, resampling the circuit itself (if accuracy does not drop, the circuit is not
needed), and keeping a random edge set of the same size (if accuracy does not drop, the circuit is not specific). The test is trusted
only when dropping nothing reproduces the network and dropping everything reproduces the source input's predictions.

A closed form that keeps accuracy under substitution but whose complement was not tested is reported as "supported by substitution;
complement not tested", and its status is *partial*. A closed form that an audit found to fail the full bar is reported as a
*sufficient readout only*. The full bar, applied by the audit of 2026-11-07, requires exact match on fresh held-out inputs in both the
best and the final snapshot, matched native, zero and shuffle controls, and fewer fitted numbers than the part the formula replaces.
Further statuses are *rejected* (tested and failed) and *none* (no closed form found). The status of each entry is stated in its lede
and fact table and summarised in [INDEX.md](INDEX.md). Single-component substitutions often keep accuracy one at a time while a scrub of
the whole hypothesis fails, which is why the rule requires the complement test.

## Reading an entry

Each entry is a short article in the format of [TEMPLATE.md](TEMPLATE.md), written to the standard of [STYLE.md](STYLE.md). It opens
with a lede that states the task, the main finding with its seed count and the verdict, followed by a fact table with the closed-form
status, the date of the last run and links to the weights and the reproduction commands. The numbered sections then give the task and
data, the model with its training settings and reproduction, the hand-built solution with Figure 1, what training finds, the mechanism
and its verification, limitations, and the prediction recorded before training. Appendix A holds an exact excerpt of the archived
weights and Appendix B the detailed diagnostics. Terms used across entries (rungs, formats, seed counts, closed forms, diagnostics) are
defined in [GLOSSARY.md](GLOSSARY.md).

## Related work

Hand-built transformers already serve as ground truth in interpretability research. RASP describes transformer computations as programs
([Weiss et al., 2021](REFERENCES.md#weiss2021)); Tracr compiles such programs into transformer weights
([Lindner et al., 2023](REFERENCES.md#lindner2023)); and InterpBench trains models to implement known circuits so that
circuit-discovery methods can be scored ([Gupta et al., 2024](REFERENCES.md#gupta2024)). These works use a known circuit to evaluate
interpretability tools. Single-task reverse engineering across seeds also exists: the Fourier clock of modular addition
([Nanda et al., 2023](REFERENCES.md#nanda2023)), group operations ([Chughtai et al., 2023](REFERENCES.md#chughtai2023)), and the clock
and pizza algorithms ([Zhong et al., 2023](REFERENCES.md#zhong2023)); trained networks have also been constrained to be readable
programs ([Friedman et al., 2023](REFERENCES.md#friedman2023)). The atlas asks what gradient descent finds when an exact hand-built
solution is available, with one protocol across many tasks: three rungs, 10 seeds per final cell, circuit types counted per seed, and a
closed form called verified only under the rule of Section 9.

## How this was made

The atlas has one author. Most experiments, analyses and entry drafts were run by LLM coding agents under a written research protocol:
fixed seeds, predictions recorded before training, every run logged, and failures diagnosed rather than dropped. Every number comes from
a logged run, and the networks are public so that any claim can be checked against the weights.

## References

Full records are in [REFERENCES.md](REFERENCES.md): [bricken2023](REFERENCES.md#bricken2023), [chan2022](REFERENCES.md#chan2022),
[chughtai2023](REFERENCES.md#chughtai2023), [conmy2023](REFERENCES.md#conmy2023), [cunningham2023](REFERENCES.md#cunningham2023),
[elhage2021](REFERENCES.md#elhage2021), [frankle2020](REFERENCES.md#frankle2020), [friedman2023](REFERENCES.md#friedman2023),
[geiger2023](REFERENCES.md#geiger2023), [gupta2024](REFERENCES.md#gupta2024), [lindner2023](REFERENCES.md#lindner2023),
[nanda2023](REFERENCES.md#nanda2023), [power2022](REFERENCES.md#power2022), [weiss2021](REFERENCES.md#weiss2021),
[zhang2024](REFERENCES.md#zhang2024), [zhong2023](REFERENCES.md#zhong2023).
