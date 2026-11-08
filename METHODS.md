# Methods: architecture, instrumentation and experimental conditions

Background for every entry. All code is in [code/](code/) (MIT); `code/run.py` reproduces each entry. The numbers below are those
of the code; each entry's Setup section lists its task-specific values and the command.

## 1. Three rungs

| Rung | What is trained | What it shows |
|---|---|---|
| HAND | nothing: every weight is set by a formula and checked exact on all inputs in range | an existence proof and a reference algorithm |
| PART | one trained stage on top of frozen hand-built parts, 10 seeds | which frozen parts training reads, which algorithm it builds from them |
| TRAIN | a network trained from scratch, 10 seeds | whether the same algorithm appears without any hand-built help |

Most entries in this atlas are PART experiments (plus controls); the base-conversion entry (section 7) covers all three rungs, with the HAND networks and a single-network PART / TRAIN trainer in the code release. For each
trained network the analysis reports: an ablation map (which frozen parts are read), the circuit type (clustered by mechanism, with seed
counts), a closed form accepted only if substituting it for the trained part keeps accuracy, and the learning curve (grokking step).

## 2. The frozen hand-built model ("m2")

A GPT-NeoX-shaped decoder written without training (`code/m2.py`, evaluated by `code/neox.py`, a 40-line forward pass; the state dict uses
Hugging Face `GPTNeoXForCausalLM` parameter names and loads there, and the two forward passes agree bit for bit on a test prompt).

| Item | Value |
|---|---|
| Layers / heads / d_model / d_head / MLP width | 6 / 4 / 528 / 132 / 512 |
| Activation, norm, positions | exact GELU; LayerNorm eps 1e-5, every gain a fixed constant, bias 0; no positional embedding and no rotary |
| Residual | sequential (attention, then MLP, each added to the stream) |
| Vocabulary | character level, 70 ids: 0 end-of-text, 1 unknown, 2..69 the 68 characters of the prompts (letters, digits, `( ) : = < > - + | _ \n` and space) |
| Residual layout | a fixed block of dimensions per quantity (`DIMS` in `m2.py`): token one-hot, prefix mean, 1/position, offsets, copy pointer, last letter, count X, depth ratio BAL, two numbers NA/NB, output logits, ... |
| Training | none; the weights are formulas, with only 0 and a few constants (recency 15 per position, class gate 5000, token match 1500, readout 30) |

Bricks by layer (each brick is a head or a group of MLP neurons; `BRICKS` in `m2.py` records what each writes):

| Layer | Attention heads | MLP |
|---|---|---|
| 0 | prefix mean of tokens | first-token flags (task keyword) |
| 1 | prefix mean of the flag (gives 1/position) | position: piecewise-linear 1/p, 500 knots |
| 2 | latest `:` anchor; last lower-case letter; last newline | line offset (steps) |
| 3 | copy (token at fixed offset); key-value match; has-letter; bracket-balance prefix | negative-balance flag |
| 4 | contains-letter; **count** (X = 1/(c+1), uniform attention over the matching letters of the current line); previous digits 1 and 2 | router (copy / bigram / yes-no), count digit decoder (step ladder) |
| 5 | bracket segment (BAL = b/(L+1), ANY); compare number readers NA and NB | bracket-validity answer; compare answer (`=` is the default output) |

Step units: a "step" is a pair of GELU neurons, `step(z; c) = (gelu(40(z-c)+8) - gelu(40(z-c)-8))/16`, exactly 0 for z <= c - 0.4 and 1 for
z >= c + 0.4 on integer-valued inputs. Exactness checks (`python run.py check`): the count head with its digit decoder answers 0..9
correctly for lines of 9 letters; the menu's one-hot is exact to 4e-6 and the scalar number readers to 2e-4 (modular addition).
m2's count is exact only up to L = 45 items.

Fixed-scale LayerNorm: every LayerNorm gain is a constant and each reader row sums to zero over the residual (a dimension that is always
0 absorbs the sum), so each LayerNorm acts as the identity map up to 1e-5.

## 3. The frozen menu and the trained stage

At the answer position (the final space after `ANSWER:`) the residual of m2 holds the task's integers (as scalars, via the number
readers). A hand-written frozen **menu** layer re-encodes each integer three ways, using the same step units:

| Encoding | Block | Dimensions |
|---|---|---|
| scalar | raw value from m2 (x/100, 1/(c+1), depth ratio, position) | 1 per integer |
| one-hot | `step(z; k-1/2) - step(z; k+1/2)`, one unit per value | one per value |
| thermometer | `[z >= k]`, `step(z; k-1/2)` | one per threshold |
| distractors | other m2 blocks at the slot: copy pointer (70), last letter (70), bracket balance, count, position; those constant on a task are 0 | varies |

Every block is centred (mean over all cells) and divided by its rms row norm, so each block has rms 1 (constant blocks become 0).
Total menu width: 310 (modular addition), 326 (count tasks), 416 (open-bracket count).

**Trained stage** (`code/train.py`): one hidden layer, `logits = W_out ReLU(W_in x + b_in) + b_out`, hidden width 256, one output per class,
default PyTorch initialisation (seeded). Only this MLP is trained; m2 and the menu are frozen (and are cached once, since they do not change).

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr 1e-3, weight decay 1.0 (decoupled, a very strong value that drives grokking), betas (0.9, 0.98), eps default |
| Batch | full batch (all training cells every step), cross-entropy loss, fp32 |
| Steps | modular addition 300,000; count tasks 200,000. No early stopping: every run uses its full budget |
| Evaluation | every 500 steps: train and test accuracy and loss, weight norm, per-block input-weight norm |
| Seeds | seed s sets both the initialisation and the train/test split; seeds 0..9 |
| Split | a random permutation of the cells; the first `int(frac * N)` are training cells, the rest test. Modular addition frac 0.4 of 59^2 = 3481 pairs. Count tasks fracs 0.4 / 0.6 / 0.8 of 1080 cells |
| Controls | `random`: the frozen base replaced by a random-weight model of the same shape (weights N(0, 0.02), seeds 1000+s), its final residual z-scored per dimension and rescaled to the menu's total norm (3 seeds); `scalar`: the menu reduced to the two raw scalars (3 seeds) |

**Packing and hardware.** The published runs were executed in two ways with identical settings. Modular addition and the 0.4 fraction of
the count tasks: one network per process on consumer CPU machines (14 to 29 minutes per network at 300,000 steps). The 0.6 and 0.8
fractions: all networks of a batch in one process on one consumer 12 GB GPU, stacked as batched matrix multiplies with padded inputs
(extra input weights start at 0 and receive no gradient; absent classes are masked; loss = sum of each network's own mean
cross-entropy; fused AdamW; fp32, TF32 not enabled), which is exactly independent training (batched and unbatched agree to 3 decimals).
`code/` contains the single-network trainer only; it needs no GPU (a 200,000-step count network takes about 5 minutes on 4 CPU threads, an open-bracket network at fraction .8 about 17 minutes on a busy machine, plus about a minute to build the menu).

## 4. Tasks and prompt generation

All prompts are generated, not sampled from a corpus. One prompt per cell; the split is over cells, so a test cell is a never-seen
(c, L) or (x, y) combination.

- **Modular addition.** Prompt `SUM: x y\nANSWER: `, x, y in 0..58, answer (x + y) mod 59 (59 classes).
- **Count tasks.** Prompt `COUNT b: a b\nANSWER: 1\nCOUNT a: <line>\nANSWER: ` where the line is c letters `a` and L - c letters `b`,
  space-separated and shuffled (Python `random.Random(0)`, consumed once per cell in cell order). L = 1..45, c = 0..L: 1080 cells.
  Labels: count mod 5 (5 classes); compare (a if 2c > L, b if 2c < L, `=` if 2c = L: 3 classes, 22 `=` cells); (u, v) = (c, L).
- **Open-bracket count.** Prompt `VALID_BRACKETS: ()\nANSWER: YES\nVALID_BRACKETS: <line>\nANSWER: `, a shuffled line of (L+b)/2 `(` and (L-b)/2 `)`;
  L = 1..45, b = -L..L in steps of 2: 1080 cells. Label: the number of `(` = (L+b)/2 (46 classes). (u, v) = (b, L).

The fixed shot line before the question is needed: without a preceding newline m2's has-letter head misses the question letter.

## 5. Instrumentation (the analysis code)

Accuracy is always arg-max accuracy on the true task labels. "Test" = held-out cells; "all" = every cell.

**Learning curve.** Logged every 500 steps. Fit step = first step with train accuracy >= 0.99. Grok step = first step with test accuracy
>= a threshold (the entries quote 50%, 90% and 99%). "No grokking" means test accuracy equals train accuracy at the first evaluation.

**Encoding-read ablations** (`usage`, and pair versions). For a block (or an encoding pair, e.g. both one-hots): *ablate* = replace its
columns by their mean over all cells, report test accuracy; *keep only* = ablate everything else; *resample* = replace by the block of a
random other cell. *contrib* = rms of that block's input to the hidden pre-activations.

**Separable effective embeddings.** The hidden pre-activation of neuron n is fitted over cells as `c_n + E_u[n][u] + E_v[n][v]`
(least squares with one free value per integer); the unexplained share is `sep_resid`. Substituting this separable form for the trained
first layer, keeping the trained W_out, gives `form_sep`.

**Substitution check (closed form).** A closed form for the trained part is *accepted* only if substituting it keeps accuracy. Always
with the trained output layer kept unless stated. *Accuracy* = arg-max accuracy of the substituted network on all cells against the true
labels. *Agreement* = fraction of cells where the substituted network and the trained network give the same arg-max. Variants used:
- `form_u_only`: the second integer's embedding replaced by its mean (does the net ignore it?).
- `form_res5` (count mod 5): the count's embedding replaced by its 5 residue-class means.
- `form_fourier` (count mod 5): logits replaced by `A cos(2pi(c-r)/5) + B cos(4pi(c-r)/5)`, (A, B) least squares per network (`fourier_r2`).
- `form_line` (compare): the logit difference a - b fitted by `alpha c + beta L + gamma`; closed form = sign of the fit, `=` never output.
- `form_clock` (open-bracket count): each neuron's pre-activation replaced by `c_n + a cos(w b/2 + phi) + a' cos(w L/2 + psi)`, `w = 2pi k/92`, k
  in 1..45 chosen per neuron by least squares.
- Modular addition (`analyze_modadd.py`): per-neuron frequency k from the DFT over x of the neuron's input-weight embedding; neuron
  formula `ReLU(c_n + a cos(w_k x + phi) + b cos(w_k y + psi))` with the trained W_out kept (accuracy / agreement, "neuron formula"), and the
  fully analytic layer with `W_out[n, c] = A cos(w_k c + phi + psi)` ("analytic"). Phase rule: median |theta - phi - psi|.

**Frequency / Fourier analysis (modular addition).** DFT of the separable embeddings E_x, E_y over x (k = 1..29, `w = 2 pi k/59`); the
linear-in-x part is removed first; key frequencies carry >= 5% of the power. Logits are fitted on the key frequencies with the *clock*
basis `cos/sin(w_k(x+y-c))` and the *pizza* basis (the same times `|cos(w_k(x-y)/2)|`); `dd` = share of the correct-logit variance
explained by x - y (clock ~ 0). Typing: *memorize* if test accuracy < 0.9, else *lookup* if the top-5 frequencies hold < 50% of the power, else *pizza* if the pizza fit beats the clock fit by 0.05 and dd > 0.2, *clock* if dd < 0.1, else *mixed*.

**Circuit-type labels** (`typ` in `analyze_atlas.py`; thresholds set from a pilot): count mod 5 *residue lookup* if `form_res5` and
`form_u_only` >= 0.99; compare *linear comparator, `=` memorized* if `cmp_r2` >= 0.7 and `=` test accuracy < 0.5, *exact comparator* if test
accuracy >= 0.99; open-bracket count *low-k clock sum* if test accuracy >= 0.5 and `form_clock` >= 0.85, *memorize* if test accuracy < 0.2.
Types are a summary of the measured forms, not an independent classifier.

**Ridge probes (random-base control).** From the z-scored residual of the random base, ridge regression (lambda 1e-2, intercept included)
predicts integer quantities (c, L, 2c - L, 1/(c+1), c mod 5 = 0, b, (L+b)/2); fitted on the training cells of the network's split, R^2
on its test cells. R^2 near 1 for c, L, 2c - L and near 0 for `c mod 5 = 0` is the control's finding.

**Causal use of a variable: interchange vs DAS.** A linear probe shows a variable is decodable, not that the net uses it. *Interchange*
replaces the probed component of a variable in one input's residual by that of another input and counts how often the answer flips as
the variable predicts. It under-reads variables with many values or a positional code: in binary addition (carry-chain format) the carry
scores .16 by interchange but 1.00 with distributed alignment search (DAS, a learned rotation), and some position variables score 0 by
interchange even on the hand-built net. We therefore read "unused" only when DAS is also low. Between-seed comparisons use linear mode
connectivity (loss barrier on the interpolation path); for nets on frozen parts the permutation-only alignment is the valid arm, since a
general rotation breaks the shared frozen basis.

## 6. Reproduction cost and tolerances

Everything runs on CPU. Menu construction (m2 forward over all cells) takes about a minute. Training time scales with steps; see each
entry. Re-running a seed reproduces its headline numbers (accuracies, circuit type; count-mod-5 seed 0 on CPU matches the published run to 3 decimals; open-bracket seed 0 at fraction .8 gives test .787 vs .782, 50% step 21.0k vs 23.5k, `form_clock` .917 vs .930) but not bit for bit: the published networks were
trained on different machines (CPU and GPU), and the random control uses a re-implemented initialiser with the same distribution. The
smoke mode (`--steps 400 --every 100`) only checks that the pipeline runs; at that length networks have not converged and the analysis
numbers are not those of the entries.

## 7. Small two-layer networks (base conversion and other whole-network entries)

Entries that train or hand-build the whole network (not a stage on the frozen m2) use a smaller GPT-NeoX model, `code/convert.py` for
base conversion: 2 layers, d_model 256, 4 heads of 64, MLP 256, ReLU, no LayerNorm, no positional embedding or rotary (position is built
from attention: a head on the start token gives 1/(r+1)-like values from which an MLP interpolates a position one-hot), sequential
residual, vocabulary of 18 tokens (start, 0-9, a-f, `=`), untied readout. Parameter names are Hugging Face GPTNeoXForCausalLM's without
LayerNorm keys.

| Rung | What is trained |
|---|---|
| HAND | nothing; exact on all 256 inputs (`python run.py convert check`) |
| PART | the layer-1 MLP only (re-initialised N(0, .02), biases 0); embedding, layer 0, layer-1 attention and readout are the frozen hand-built ones |
| TRAIN | every weight, from N(0, .02) (biases 0) |

Data: all 256 inputs, one sequence each; seed s sets the init, the batch order and the split (first `int(frac * 256)` of a random
permutation train, the rest test). Loss: cross-entropy on the output tokens; metric: exact match (every output token right, teacher
forced). AdamW lr 1e-3, weight decay 1.0, betas (.9, .98), minibatch 64, full fp32: reduced-precision matmuls (TF32) take the hand-built
converters from 100% to about 2% exact, so they must stay off. The published runs packed 10 seeds per process on a GPU (vmap) with early
stopping (all seeds exact, or no seed's best test accuracy up by .005 in 5,000 steps after 10,000 steps); `code/` trains one network at
a time on CPU with the same objective and stops when train and test are both exact.

Instrumentation: *best* = the evaluation (every 500 steps) with the highest test accuracy; final weights can be far worse (instability
at weight decay 1). *Substitution* (`convert.substitute`): at every scored position the residual after the last attention layer is
replaced by its least-squares fit (with intercept, over all 256 inputs) on HAND's variable at that position, and the exact-match rate
of the rest of the network is measured; the variable is accepted as the one the net uses if the rate stays within .01 of the net's own.
The fit uses test inputs too: it says which variable the stage needs, not that the net computes it cleanly. *Threshold check*
(`convert.thresholds`, decimal to binary, answer only): per answer bit the best theta in `pred = [F >= theta]` over HAND's remainder F,
and its agreement with the net on all inputs. Reported sizes use mean-ablation of heads and neurons (a head or MLP is "used" if
replacing its output by its mean over inputs costs more than 5% exact; neurons: the smallest set, ranked by output spread, that keeps
exact within .01 when the rest are mean-ablated); the input-independent position frame (layer-0 head and MLP) is not counted.

## Reading an entry

| Term | Meaning |
|---|---|
| **HAND** | every weight set by a formula; exact on every input in range |
| **PART** | frozen hand-built parts plus one trained stage (`PARTp`: the trained stage is an attention layer instead of an MLP) |
| **TRAIN** | trained from scratch |
| **Format** | `A` = answer only; other letters (`T`, `G`, `S`, `C`, ...) are scratchpad variants, defined in each entry's Task paragraph |
| **Seeds** | "6/10" = 6 of 10 seeds meet the criterion named in that cell (usually held-out exact match ≥ .99) |
| **Closed form verified** | substituting the formula for the trained weights keeps accuracy; "partial" or "rejected" says what failed |
| **menu** | a frozen hand-written layer that re-encodes integers (one-hot, thermometer, scalar); section 3 |

Every entry follows [TEMPLATE.md](TEMPLATE.md): Summary, Setup (with the reproduce command), Prediction before training, Hand-built reference,
What training finds, Closed form, Controls and limits.

## Related work

Hand-built transformers as ground truth exist: [Tracr](https://arxiv.org/abs/2301.05062) (paper) compiles RASP programs into transformer weights, and
[InterpBench](https://arxiv.org/abs/2407.14494) (paper) trains models to implement known circuits so that circuit-discovery methods can be scored.
Both evaluate interpretability tools. Single-task reverse engineering across seeds also exists: the modular-addition clock
([Nanda et al.](https://arxiv.org/abs/2301.05217), paper), group operations ([Chughtai et al.](https://arxiv.org/abs/2302.03025), paper), clock vs pizza
([Zhong et al.](https://arxiv.org/abs/2306.17844), paper); and trained nets constrained to be readable programs
([Learning Transformer Programs](https://arxiv.org/abs/2306.01128), paper). This atlas asks what gradient descent finds instead of an exact
hand-built solution, with one protocol across many tasks: three rungs, 10 seeds per final cell, circuit types counted per seed, and a closed
form accepted only if substituting it keeps accuracy, with zero and shuffle controls on the rest of the net.

## How this was made

One author, with most experiments, analyses and entry drafts run by LLM coding agents under a written research protocol (fixed seeds,
predictions recorded before training, every run logged, failures diagnosed rather than dropped). Every number comes from a logged run;
the networks are public so any claim can be checked.
