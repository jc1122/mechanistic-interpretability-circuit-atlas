# Methods

This document describes the methods shared by all entries of the atlas: the three experimental rungs, the frozen hand-built model and
menu used by the counting and modular-addition entries, the small whole-network models, the instrumentation, the diagnostics applied
to every trained network, the rule by which a closed form is accepted (with the stricter specificity audit of the AlgZoo entries), and an estimator of a network's accuracy under a known
input law (Section 10). Each entry gives its task-specific values in its Section 2
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

### 9.1 Specificity audit of the AlgZoo entries (2026-11-14)

The external-zoo entries ([second-argmax](entries/algzoo-2nd-argmax.md), [median](entries/algzoo-median.md) and
[argmedian](entries/algzoo-argmedian.md)) count a network as proven only if its earlier proof also passes a stricter specificity
gate, applied by an audit on 2026-11-14 to every earlier proof in these entries ([longest-cycle](entries/algzoo-longest-cycle.md) had none). A same-size random control can be much weaker than the claimed circuit, so the audit matches each
random control to the claim by output effect. The matching rule was registered before any control was run on a target network. On
an independent 2,048-row calibration set with three donors, every omitted component gets the root-mean-square size of its effect on
the output, binned as zero (at most 1e−12) or by floor(log2(effect/1e−12)). A control omits the same number of components with the
same site, time-step and part-kind bins, inside the same original part, section or functional chart, so that matched nonzero effects
differ by less than a factor of two. Bins were never widened, and no alternative form, training or optimisation was tried. Finite
pools keep duplicates and masks identical to the claim. For continuous objects, 8,192 independent candidates are drawn and the first
64 matches are kept; the claim itself is never inserted.

Each defined test then uses 64 matched random masks, ten independent streams of 4,096 fresh rows (40,960 rows per network), three
donor permutations and the original .90 acceptance share. On every one of the ten streams the claimed circuit must keep at least .01
more of the network's accuracy or skill than the strongest of the 64 controls (the control with the highest retention, that is, the
smallest drop), and every original deciding gate of that proof must still pass on the same rows. The mean over the 64 controls is
also reported, but it cannot rescue a failure against the strongest control. The ten row seeds are diagnostic replicates, not new
model seeds; the networks are the frozen published ones.

An earlier proof becomes *read* (a named reading without a passing certificate) for one of three reasons. (1) *Strongest-control
failure*: on at least one stream the claim does not beat the strongest matched control by .01, or a network that passes the new gate
fails one of its original gates on the same rows. (2) *Algebraically null readout-kernel complement*: the discarded directions lie in
the kernel of every decision contrast of the final readout, Δ(I−P)=0 with Δ the readout's decision contrasts and P the projector onto
the kept directions, so resampling them cannot change the decision; complement and joint tests then repeat the native output and the
standalone substitution, and give no independent evidence of specificity. These proofs are read without a run. (3) *Insufficient
matched pool*: fewer than 64 matching random controls exist (for continuous objects, among the 8,192 candidates), so no control can
be built; this is neither a pass nor evidence that no other mechanism exists. Width-2 whole-network exceptions, whose claimed circuit
is the entire network so that nothing remains to resample, keep their earlier status; the exception covers width 2 only, so wider
empty complements become read. These are finite fresh-row specificity tests, not exhaustive population proofs, and networks that were
already read or unexplained were not retested.

## 10. Estimating a network's accuracy with rank-coordinate randomised quasi-Monte Carlo

This section describes an estimator of a trained network's accuracy under a known input distribution, and the preregistered study
that compared it with plain sampling at equal arithmetic cost. The estimator (arm R) changes the input coordinates so that the
boundaries of the task's label become parallel to the coordinate axes, then integrates in those coordinates with scrambled Sobol'
points. On 48 noisy copies of one public network, at the registered primary budget, its truth-corrected mean squared error (MSE) was
.380026 of that of plain Monte Carlo (MC) at equal full operation cost (95% CI .338419–.430555). The method samples the network's
forward pass and is therefore a hybrid estimator; it does not compute accuracy from the weights alone. It was slower than MC in wall
time at every tested budget. The limits are listed in Section 10.8 and apply wherever the result is quoted.

### 10.1 Problem

Let f(x) be 1 when a network classifies input x correctly and 0 otherwise, and let x follow a known law, here T independent
standard normal numbers. The estimand is the accuracy p = E f(X). Plain MC averages f over n independent inputs; its MSE is
p(1 − p)/n. ARC has proposed that methods which use a network's internal structure should be judged by whether they estimate such
expectations more accurately than sampling at comparable compute ([Neyman et al., 2025](REFERENCES.md#neyman2025)). The estimator
below does not meet that goal in its pure form, because it still evaluates the network on sampled inputs. It asks a narrower
question: whether the structure of the task, known before the network is seen, can be used to choose better sample points.

The network is ARC's public AlgZoo second-argmax example ([Alignment Research Center, n.d.](REFERENCES.md#ref-https-github-com-alignment-research-center-alg-zoo),
MIT-0 licence): a one-layer ReLU recurrent network with hidden size h = 16, sequence length T = 10 and no biases (432 weights). It
reads x_1, …, x_10 one per step, h_t = ReLU(W h_{t−1} + U x_t) from h_0 = 0, and outputs logits V h_10; the label J(x) is the
position of the second-largest input. All arithmetic in this section is float64 on CPU, with the float32 weights promoted; it is not
bitwise native float32 execution.

### 10.2 Rank coordinates

The label J(x) changes only on the 45 hyperplanes x_i = x_j, which are oblique to the axes of x. Randomised quasi-Monte Carlo
(RQMC) integrates an indicator faster when its discontinuities are parallel to the coordinate axes
([He & Wang, 2015](REFERENCES.md#he2015)), and work in computational finance chooses how a random path is generated so that a
payoff's discontinuities suit QMC ([Wang & Tan, 2013](REFERENCES.md#wang2013)). No rotation of x makes all 45
hyperplanes axis-parallel. R instead writes x as a sorted list of values and a permutation, and draws each from its own uniform
coordinates. It uses d = 2T − 1 = 19 uniforms: V_0, …, V_9 for the values and W_0, …, W_8 for the positions.

The values are the descending uniform order statistics, generated by ratios ([Devroye, 1986](REFERENCES.md#devroye1986)):

  P_0 = V_0^(1/T),  P_r = P_{r−1} · V_r^(1/(T−r))  (r = 1, …, T − 1),  s_r = Φ^{−1}(P_r),

where Φ is the standard normal distribution function, so that s_0 > s_1 > … > s_9. The positions are a Lehmer code (the factorial
number system for permutations): for r = 0, …, T − 2 the rank-r value goes to the k_r-th position, counted in increasing order, among
the positions not yet used, with

  k_r = ⌊(T − r) · W_r⌋ ∈ {0, …, T − r − 1},

and the last value takes the remaining position. The second-largest value s_1 is placed by k_0 and k_1 alone, so J is constant on the
axis-parallel boxes ⌊10 W_0⌋ = a, ⌊9 W_1⌋ = b, and every change of label lies on a face of one of them.

RQMC point sets are most evenly spread in their first coordinates, so the order in which the 19 uniforms are read from the point set
matters. It was frozen before any data: W_0, W_1, V_1, V_2, W_2, V_0, W_3, …, W_8, V_3, …, V_9. The first two place the label.
V_1 and V_2 set the gaps between the top three values on the uniform scale (P_1/P_0 = V_1^(1/9) and P_2/P_1 = V_2^(1/8)), so
near-ties among those values lie where V_1 or V_2 approaches 1. Nothing about the order was fitted to a network.

### 10.3 Point set and the estimator

Each estimate draws one fresh scrambled Sobol' point set of n = 2^m points in 19 dimensions ([Sobol', 1967](REFERENCES.md#sobol1967)),
with the direction numbers of [Joe and Kuo (2008)](REFERENCES.md#joe2008), using PyTorch's `SobolEngine` with `scramble=True`
([PyTorch, 2026](REFERENCES.md#pytorchsobol)). That engine applies a random lower-triangular linear matrix scramble and a random
digital shift ([Matoušek, 1998](REFERENCES.md#matousek1998); [Owen, 1998](REFERENCES.md#owen1998)), written LMS + shift below.
All 2^m points are used, with no skipping, thinning or optimisation of the point set, and the engine is built with float64 as the
default type so that the first point is not rounded. Each coordinate of each point is a 30-bit integer c ∈ {0, …, 2^30 − 1}; it is
jittered uniformly within its cell,

  u = (c + ε) / 2^30,  ε ~ U(0, 1) independent,

and points on the boundary of the cube are rejected, never clipped. With x_i = G(u_i), where G is the map of Section 10.2, the
estimate is

  p̂_R = (1/n) Σ_{i=1}^{n} f(G(u_i)).

Two controls isolate the two ingredients. RI uses the same map G with n independent uniform points (rank coordinates without
equidistribution). RAW uses the same scrambled, jittered Sobol' points in T = 10 dimensions with x = Φ^{−1}(u) coordinatewise (an
equidistributed point set in the raw coordinates, whose label boundaries stay oblique).

### 10.4 Why the estimator is unbiased

The argument holds in exact real arithmetic and uses three facts. First, if V_0, …, V_9 are independent U(0, 1), the maximum of T
independent uniforms has the law of V_0^(1/T), and given P_{r−1} the remaining T − r uniforms are independent U(0, P_{r−1}), whose
maximum has the law of P_{r−1} V_r^(1/(T−r)); so (P_0, …, P_9) has the law of the descending order statistics of T independent
uniforms ([Devroye, 1986](REFERENCES.md#devroye1986)). Second, if the W_r are independent uniforms, k_0, …, k_8 are independent and
uniform on their ranges, and the Lehmer code is a bijection between the ∏_r (T − r) = T! codes and the permutations, so the
permutation is uniform and independent of the values. Independent normal inputs are exactly a sorted list of normal order statistics
placed by an independent uniform permutation, and Φ^{−1} is increasing, so G(U) has independent N(0, 1) coordinates when
U ~ U(0, 1)^19. Third, a uniform random digital shift makes each point's 30-bit integer vector uniform over the 2^(30·19) cells
whatever the scrambling matrix is, and the uniform jitter then makes each u_i exactly U(0, 1)^19. Hence E f(G(u_i)) = p for every
i, and E p̂_R = p. The points are dependent, which changes the variance but not the mean. The same argument covers RI and RAW.

The implementation runs in float64 with a pseudo-random jitter, a library inverse normal and finite-precision powers, as plain MC
does; no rigorous bound on the numerical bias was derived. The variance reduction is not derived either. Scrambled nets reduce
variance for square-integrable integrands asymptotically ([Owen, 1997](REFERENCES.md#owen1997)), and the rate for an indicator
depends on how many axes its boundaries are not parallel to ([He & Wang, 2015](REFERENCES.md#he2015)). In R's coordinates the
label's boundaries are axis-parallel, but the network's own decision boundaries, and bands of fixed Gaussian gap between the top
values, are not; the study measures the variance instead of predicting it.

### 10.5 Full cost accounting

Cost is counted in operation units: 2 per multiply-accumulate (MAC), 1 per other scalar operation and 20 per call of a named special
function (power, inverse normal). One MC row costs 5,931 units at h = 16, T = 10: 2,880 MACs, 170 scalars for the ReLUs, the argmax
and the comparison with the label, and 1 for the mean. Each R or RAW estimate is charged its own scrambling setup,
d(30^3 + 30^2 + 30) units for one fresh engine, plus n(5d + 30) units to generate points and 4nd for jitter and endpoint checks; the
rank map adds T power and T inverse-normal calls per row, the products, the Lehmer placement and input validation. Nothing is
amortised across estimates, budgets or networks. The totals for one estimate of n rows are

| arm | full cost (units) |
|---|---|
| MC | 5931n |
| R | 6673n + 530680 |
| RI | 6510n + 10 |
| RAW | 6251n + 279300 |

*Table 10.1. Full operation cost per estimate at h = 16, T = 10. MC = plain Monte Carlo; R = rank coordinates with scrambled Sobol';
RI = rank coordinates with independent uniforms; RAW = scrambled Sobol' in raw coordinates.*

Each estimate of cost C is paired with an MC control of ⌊C/5931⌋ independent rows, so the control never costs more than the
estimate. The bit operations of the Sobol' generator are an explicit conservative model, not hardware FLOPs or an audit of the
library's native code. Random-number generation for MC and the label oracle are excluded from the arithmetic, as in the plain-MC
specification, and included in the measured times.

### 10.6 Evaluation design

The population was 48 noisy copies of the public example, generated for this study. Each copy adds independent Gaussian noise to each
weight tensor with standard deviation σ times that tensor's root mean square, 16 copies at each of σ = .01, .03 and .10. These are
copies of one trained network, not independently trained networks. Their seeds were reserved for the scored population, and none of
them coincided with the 37 networks (gate copies and the original) used in the earlier stages. The reference accuracy of each copy was measured from 2^23 independent
inputs in two independent halves of 2^22; it is a measurement, not an exact value. Table 10.2 gives the population.

| σ | copies | mean accuracy | range |
|---|---:|---:|---|
| .01 | 16 | .778408 | .619044–.893669 |
| .03 | 16 | .450281 | .276740–.613647 |
| .10 | 16 | .197891 | .145191–.258736 |

*Table 10.2. Scored population: noisy copies of the AlgZoo example, reference accuracy from 2^23 inputs each.*

Each arm (R, RI, RAW) ran at n = 1024, 4096 and 16384 rows, with 32 independent estimates per copy and budget, each paired with its
own independent full-cost MC control: 13,824 estimates and 13,824 controls in all. Call order and the order within each pair were
randomised from the seed. For a copy j with reference accuracy p̃_j from N = 2^23 rows, the truth-corrected MSE of its 32 estimates
is mean_k (p̂_jk − p̃_j)^2 − p̃_j(1 − p̃_j)/(N − 1), which removes the reference's own sampling variance because the estimates are
independent of it. The measured ratio divides the equal-weight mean of these over copies by the same quantity for the paired MC
controls. The theory ratio uses p̃_j(1 − p̃_j)N/(N − 1) · mean_k(1/m_jk) for MC, where m_jk is the control's row count, and the paired
excess is the difference of the two means. Uncertainty comes from 2000 bootstrap draws that resample copies with replacement within
each σ, keep all 32 repetitions of a copy together, and use the same draws for numerator and denominator
([Field & Welsh, 2007](REFERENCES.md#field2007)); intervals are 2.5% and 97.5% percentiles, pointwise for each row and not adjusted
for the number of rows.

The primary comparison, fixed before data, was pooled R at n = 4096. It counted as a gain only if the measured ratio's upper 95% bound
was below 1, the theory ratio's upper bound was below 1 and the paired excess MSE's upper bound was below 0. Every other row is
secondary and descriptive; RI and RAW are controls, not candidates.

Earlier stages are part of the record. Synthetic qualification checked the transform, the Lehmer bijection over all 720 codes at
T = 6, known-accuracy networks and the cost counters. Its first version left some internal checks uncharged; the counters were
repaired before any network data, and the repaired run then failed its two-order timing self-control (MC against MC, ratio .741926,
95% CI .588921–.861499). A new order-balanced timing protocol passed (wall-time ratio .967965, CI .895754–1.047123; CPU-time ratio
.973409, CI .905087–1.046009). A diagnostic
gate on 12 separate copies forecast R's in-cell disagreement at .622759 (CI .586693–.653928, bar ≤ .85); it was a screen, not an MSE
estimate. A second candidate, exact integration of f over the value of the label coordinate given the others (arm Q), failed its
preregistered cost gate (variance-cost forecast 5.328995, CI 5.097779–5.570444, bar < .8) and was not run on the scored population.

### 10.7 Results

The primary comparison passed all three bars (Table 10.3). R's truth-corrected MSE was 1.501955548e-5 against 3.952239394e-5 for
its MC controls, with paired excess −2.450283846e-5 (95% CI −2.808109419e-5 to −2.071539768e-5). RI, which keeps the rank map but
drops the point set, showed no pooled gain, and at n = 16384 its whole interval lay above 1 (a loss).
RAW, the point set without the rank map, also had pooled ratios below 1 at every budget.

| arm | n | measured ratio [95% CI] | theory ratio [95% CI] | paired excess MSE [95% CI] |
|---|---:|---|---|---|
| R | 1024 | .532490 [.467014, .602776] | .493806 [.451840, .535651] | −6.5447e-5 [−8.0975e-5, −5.1210e-5] |
| **R** | **4096** | **.380026 [.338419, .430555]** | **.376576 [.346184, .408093]** | **−2.4503e-5 [−2.8081e-5, −2.0715e-5]** |
| R | 16384 | .325068 [.287709, .369201] | .329276 [.299250, .357651] | −6.9145e-6 [−7.8312e-6, −5.9567e-6] |
| RI | 1024 | 1.029884 [.939514, 1.128201] | 1.024759 [.963749, 1.095666] | 4.9604e-6 [−1.0764e-5, 2.0161e-5] |
| RI | 4096 | 1.059524 [.980677, 1.142782] | 1.108701 [1.040734, 1.181027] | 2.5959e-6 [−8.5719e-7, 5.9711e-6] |
| RI | 16384 | 1.118785 [1.012815, 1.228283] | 1.111475 [1.033838, 1.187421] | 1.2294e-6 [1.3389e-7, 2.2709e-6] |
| RAW | 1024 | .868017 [.791726, .956633] | .857636 [.795361, .924145] | −2.1696e-5 [−3.5980e-5, −6.7537e-6] |
| RAW | 4096 | .767940 [.694642, .852093] | .757557 [.703834, .811744] | −9.8272e-6 [−1.3714e-5, −5.7903e-6] |
| RAW | 16384 | .653566 [.600788, .711311] | .656277 [.613059, .696426] | −3.7638e-6 [−4.6512e-6, −2.9179e-6] |

*Table 10.3. Pooled results over 48 copies, 32 estimates per copy and cell. Ratios are truth-corrected MSE of the arm over that of
equal-cost MC: measured against the paired MC controls, or against MC's theoretical MSE. Below 1 favours the arm. The bold row is the
registered primary comparison; all other rows are secondary. Excess MSE is arm minus paired MC.*

| σ | arm | n = 1024 | n = 4096 | n = 16384 |
|---|---|---|---|---|
| .01 | R | .490478 [.395959, .622136] | .379613 [.307064, .478779] | .300257 [.261393, .347490] |
| .01 | RI | 1.035941 [.903304, 1.204984] | .986928 [.860600, 1.131738] | 1.146330 [1.014524, 1.278641] |
| .01 | RAW | .843984 [.723497, 1.011534] | .786415 [.666799, .940301] | .603734 [.518273, .707393] |
| .03 | R | .567462 [.446743, .718572] | .345155 [.277134, .439061] | .322312 [.253059, .400974] |
| .03 | RI | 1.038749 [.902389, 1.198519] | 1.138783 [.982717, 1.310461] | 1.084838 [.909487, 1.275335] |
| .03 | RAW | .855100 [.730336, .982908] | .769202 [.648198, .923915] | .656780 [.571540, .773073] |
| .10 | R | .526236 [.439129, .618893] | .438677 [.361707, .523863] | .357791 [.284119, .459709] |
| .10 | RI | 1.011141 [.833147, 1.230784] | 1.034050 [.935231, 1.157597] | 1.146006 [.937331, 1.385545] |
| .10 | RAW | .913593 [.752126, 1.124016] | .746497 [.643143, .870981] | .705677 [.628100, .796487] |

*Table 10.4. Measured MSE ratio [95% CI] against paired equal-cost MC, per noise scale (16 copies each). Pointwise intervals,
secondary and descriptive; two RAW intervals at n = 1024 include 1.*

The gain is in MSE at equal operation count. In mean time per call on one CPU thread, R and RI were slower than their matched MC
controls at every budget, and RAW at n = 1024 and 4096; RAW at n = 16384 took 76.941 ms against 77.652 ms. At the primary budget a complete R call took 17.585 ms of
wall time against 7.679 ms for its MC control (CPU 17.027 ms against 7.459 ms), so the result is not a wall-time gain, and no
wall-time-matched comparison was made. Table 10.5 gives the costs.

| arm | n | full units | matched MC rows | method / MC wall time (ms) | method / MC CPU time (ms) |
|---|---:|---:|---:|---|---|
| R | 1024 | 7,363,832 | 1241 | 6.621 / 2.515 | 6.427 / 2.440 |
| R | 4096 | 27,863,288 | 4697 | 17.585 / 7.679 | 17.027 / 7.459 |
| R | 16384 | 109,861,112 | 18523 | 123.532 / 87.746 | 120.493 / 85.465 |
| RI | 1024 | 6,666,250 | 1123 | 5.017 / 2.249 | 4.843 / 2.200 |
| RI | 4096 | 26,664,970 | 4495 | 14.277 / 7.237 | 13.877 / 6.992 |
| RI | 16384 | 106,659,850 | 17983 | 110.415 / 81.545 | 107.259 / 79.317 |
| RAW | 1024 | 6,680,324 | 1126 | 4.205 / 2.293 | 4.060 / 2.240 |
| RAW | 4096 | 25,883,396 | 4364 | 9.840 / 7.126 | 9.434 / 6.898 |
| RAW | 16384 | 102,695,684 | 17315 | 76.941 / 77.652 | 74.850 / 75.665 |

*Table 10.5. Cost of one estimate and of its matched MC control. Times are means over all calls of a cell, including sampling, setup,
transform, labels, forward pass and mean; network loading, validation, reference sampling and file I/O are excluded. The hardware model
is not recorded.*

### 10.8 Limitations

R is a hybrid estimator: it evaluates the network on 2^m sampled inputs and uses no deduction from the weights, so it does not meet
ARC's objective of estimating accuracy mechanistically. The population consists of noisy copies of a single trained network; no
independently trained network, other training seed, other width or other task was tested, and the gain may not transfer. The gain
concerns MSE at equal operation count; every primary implementation was slower in wall time than its matched MC control, and no
claim about runtime, hardware FLOPs or untested budgets is made. RAW also beat MC, so part of R's gain may come from the point set in
any coordinates; no paired R-against-RAW or R-against-RI interval was registered or computed, and dividing the separately
MC-normalised ratios of Table 10.3 does not provide one. That axis-parallel label boundaries cause the gain is an interpretation:
the network's own boundaries and fixed-gap tie bands are not axis-parallel in R's coordinates, and no experiment isolated alignment
as the cause. The reference accuracies are measured from 2^23 inputs, not exact; the numerical bias of float64 execution is not
bounded; the intervals are pointwise. A validation on independently trained networks, registered in advance, has not been run.

### 10.9 Relation to published work

To our knowledge, a change of input coordinates that makes a task's label boundaries axis-parallel, combined with randomised
quasi-Monte Carlo, has not been used before to estimate a trained network's accuracy. Searches of the scite literature index, arXiv
listings and web search summaries on 2026-11-14 found the principle in computational finance, where a path construction is chosen so
that a payoff's discontinuity is axis-parallel ([Wang & Tan, 2013](REFERENCES.md#wang2013)), with the corresponding error rates for
RQMC on discontinuous integrands ([He & Wang, 2015](REFERENCES.md#he2015)). The closest use on networks is quasi-random sampling of
permutations for Shapley-value estimation ([Mitchell et al., 2022](REFERENCES.md#mitchell2022)), which maps low-discrepancy points on
a sphere to permutations; it targets attribution, not accuracy, and uses no order-statistic or Lehmer coordinates. Absence from these
searches is not proof that no such work exists, and the finance and permutation sources were read at abstract or selected-section
level.

### 10.10 Reproduction

The code is [code/rank_rqmc.py](code/rank_rqmc.py) (MIT licence), with tests in [code/test_rank_rqmc.py](code/test_rank_rqmc.py). It
downloads the public example network from ARC's storage (the file that AlgZoo's `example_2nd_argmax()` loads) and checks its SHA-256,
or reads a local copy given with `--weights` (`.pth`, or `.safetensors` with the original three tensor names). It was tested with
Python 3.12, NumPy 2.5.4 and PyTorch 2.15.0 on CPU; bitwise equality of the estimates depends on NumPy's PCG64 streams and PyTorch's
`SobolEngine`, so other versions may give different, equally valid, numbers. From the `code/` folder:

```
python -m unittest test_rank_rqmc                        # Lehmer bijection, order-statistic law, unbiasedness, costs, recorded outputs
python rank_rqmc.py reproduce                            # sigma .03, arm R, n 4096: 16 copies, 8 estimates each, 2^21 reference rows
python rank_rqmc.py reproduce --repeats 32 --truth-log2 23   # the study's budget for that row
python rank_rqmc.py reproduce --sigma-index 0 --arm RAW --n 1024   # any other row: sigma index 0, 1, 2; arm R, RI, RAW; n 1024, 4096, 16384
```

`reproduce` regenerates the copies from the study's seeds and prints the row's MSEs, both ratios and the paired excess with their 2000-draw
bootstrap intervals (output keys `ratio_measured`, `ratio_measured_ci95`, `ratio_theory_ci95`, `excess_ci95`). At the reduced default
the intervals are wider and the reference accuracies noisier: for σ = .03, R, n = 4096 it printed a measured ratio of .286411
(95% CI .197481–.425005) against .345155 (CI .277134–.439061) in Table 10.4, and took 523 s on one shared CPU thread. With
`--repeats 32 --truth-log2 23` it uses the study's budget and seeds. In a verification run (2026-11-14, one CPU thread, the versions
above), this code reproduced every one of the 13,824 recorded estimates and 13,824 MC controls exactly (value, cost and control
size), the full 2^23-row reference accuracy of one copy per scale, and all 36 rows of Tables 10.3 and 10.4 with zero difference in
every ratio, MSE and interval. That run rescored the study; it is not an independent replication with new seeds. Pooled rows of
Table 10.3 need all three scales and are not exposed by the command line.

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
[devroye1986](REFERENCES.md#devroye1986), [elhage2021](REFERENCES.md#elhage2021), [field2007](REFERENCES.md#field2007),
[frankle2020](REFERENCES.md#frankle2020), [friedman2023](REFERENCES.md#friedman2023), [geiger2023](REFERENCES.md#geiger2023),
[gupta2024](REFERENCES.md#gupta2024), [he2015](REFERENCES.md#he2015), [joe2008](REFERENCES.md#joe2008),
[lindner2023](REFERENCES.md#lindner2023), [matousek1998](REFERENCES.md#matousek1998), [mitchell2022](REFERENCES.md#mitchell2022),
[nanda2023](REFERENCES.md#nanda2023), [neyman2025](REFERENCES.md#neyman2025), [owen1997](REFERENCES.md#owen1997),
[owen1998](REFERENCES.md#owen1998), [power2022](REFERENCES.md#power2022), [pytorchsobol](REFERENCES.md#pytorchsobol),
[sobol1967](REFERENCES.md#sobol1967), [wang2013](REFERENCES.md#wang2013), [weiss2021](REFERENCES.md#weiss2021),
[zhang2024](REFERENCES.md#zhang2024), [zhong2023](REFERENCES.md#zhong2023).
