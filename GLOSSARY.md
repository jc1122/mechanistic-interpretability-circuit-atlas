# Glossary

Plain-language meanings of the terms used in the entries. Technical definitions and the analysis code: [METHODS.md](METHODS.md).

## The three rungs
- **HAND**: a hand-built network. Every weight is set by a formula, nothing is trained, and it is checked exact on every input in
  range. It is a reference algorithm, not a claim about what training finds. ([METHODS §1](METHODS.md#1-three-rungs))
- **PART**: "partly trained". Hand-built parts are frozen and one stage (usually an MLP) is trained on top. Shows which hand-built
  parts training chooses to read. **PARTp**: the same, but the trained stage is an attention layer.
- **TRAIN**: the whole network trained from scratch, with no hand-built help.

## Task formats
- **Answer only** (format `A` in most entries, `AO` in multiplication-base-order): the network writes only the final answer.
- **Scratchpad**: the network first writes intermediate steps (a running sum, each Euclid row, each matrix update, ...), then the
  answer. Entries name their scratchpad variants with letters (`T`, `G`, `S`, `O`, `AL`, `R`, ...); each letter is defined in that
  entry's Task paragraph and means nothing across entries.
- **MSB-first / LSB-first**: most / least significant digit written first.
- **Teacher-forced**: each output token is scored given the correct previous tokens, not the network's own earlier outputs.
- **Length (or size) generalisation**: accuracy on inputs longer or larger than any seen in training.
- **Router scratchpad / coupled id**: a scratchpad in which every operand token carries a column id, the same id on the digits of one
  column and on the step that consumes them (a coupled id, in the spirit of position coupling), so that each output step can look up
  its operands by id instead of by absolute position.
- **Carry-type cell**: a conversion cell in which an output digit depends on a value carried across several input digits (binary to
  decimal, decimal to binary), as opposed to hex to binary, where each output nibble depends on one input digit only.

## Counting results
- **Seed**: one independent training run (its own initialisation, data order and split).
- **k/n**: k of n seeds meet the criterion named in that cell, usually held-out **exact match** ≥ .99 (exact match: every output
  token right).
- **Screen / final**: a screen is a 3-seed (sometimes 5-seed) first pass; the final cell reruns the chosen setting with 10 seeds and
  is the one the entry is judged on.
- **r1, r2**: the first run and the retry of a cell; every run is kept and reported.
- **Circuit type**: seeds grouped by mechanism (what they compute and where), not by weight values.
- **Grok / grokking**: test accuracy rising long after training accuracy is perfect. The **grok step** is the first step where test
  accuracy crosses a threshold (the entries quote 50%, 90% and 99%). ([METHODS §5](METHODS.md#5-instrumentation-the-analysis-code))
- **Memorises**: perfect on training inputs, low on held-out inputs.
- **Best / final snapshot**: the saved network with the highest test accuracy during training / the network at the end of training.
- **First exact step**: the first evaluation at which test exact match reached 1 (entries that use .99 instead say so).

## Verifying a circuit
- **Closed form**: a short formula for what a trained part computes.
- **Substitution**: replace the trained part by its closed form and re-score. A closed form that keeps accuracy under substitution
  alone is "supported by substitution"; its status is partial until the complement is also tested.
- **Verified (atlas acceptance rule)**: a closed form is verified when substituting it for the trained part keeps exact match at no
  less than .90 of the network's own, and a resample ablation of everything outside the claimed circuit also keeps it at no less than
  .90 of the network's own. ([METHODS §9](METHODS.md#9-verification-and-the-acceptance-rule))
- **Sufficient readout**: a substitution keeps accuracy, but the full bar (best and final snapshots, zero and shuffle controls) is
  not passed. The formula is enough to read out the answer; it is not shown to be the mechanism.
- **Resample ablation of the complement (scrub)**: replace everything *outside* the claimed circuit with its values on other inputs.
  If accuracy survives, the rest of the network was not needed; a working substitution alone does not show that.
- **Full bar**: the acceptance rule an audit applies to a closed form: it must keep exact match on fresh held-out inputs in both the
  best and the final snapshot, be checked against matched native, zero and shuffle controls, and use fewer fitted numbers than the
  part it replaces. A closed form that passes substitution but not the full bar is a sufficient readout.
- **Native / zero / shuffle controls**: exact match of the unmodified (native) network, set next to the same test with the replaced
  part set to zero and with shuffled values put in its place.
- **Truth table (as a closed form)**: a part described by listing its output for every combination of a few discrete inputs, one
  entry (or one neuron) per combination; it is exact on its inputs but does not compress them into a formula.
- **Ablation**: switch a part off (set it to zero or to its mean) and see whether accuracy drops. A head or MLP is "used" when its
  ablation costs more than .05 of exact match.
- **Query-mean ablation**: a head's or MLP's output at each query position is replaced by its mean over the batch at that position
  (the "mean mode" of the ablation map), which removes input-specific information but keeps the average signal.
- **Decision surrogate**: a replacement for a trained part that reproduces the network's output decisions on the inputs the task can
  reach, without matching its continuous response; it supports a readout, and on a failing network it is a repair, not an explanation.
- **Semantic scrub**: a resample ablation in which each activation outside a candidate circuit (or each component inside it) is
  replaced by its value on another input that agrees on the variable the hypothesis assigns to it; the circuit passes if accuracy stays
  above the stated bar.

## Analysis measures
- **R²** (probe R², "at Lk" = after layer k): how well a linear (ridge) read-out of the residual stream recovers a hand-built variable
  (1 = perfectly). Decodable is not the same as used.
- **Interchange flip**: swap one variable's value between two inputs inside the network and count how often the answer changes as
  the variable predicts; a high flip means the variable is causally used. `nf` = number of flips tested; nf 0 means untested, not
  unused.
- **Interchange variable**: a hand-built variable (a fetched bit, a carry, a partial sum) whose value is swapped between two inputs
  in the interchange flip test.
- **DAS** (distributed alignment search): the same swap through a learned rotation; catches variables spread over many directions.
  "Unused" is read only when DAS is also low.
- **JS**: mean Jensen-Shannon divergence (in nats; ln 2 = disjoint) between a trained head's attention pattern and the
  closest hand-built head's (0 = identical).
- **PR, r90, r99 (rank99), effective rank**: PR is the participation ratio (the effective number of dimensions, also called effective
  rank) of the residual stream or of a weight matrix; r90 and r99 (written rank99 in some entries) are the number of principal
  directions needed to reach 90% / 99% of the variance or energy.
- **CKA** (`hand_cka`): linear centred kernel alignment between the trained network's residual stream and HAND's at the same layer over
  the scored positions; 1 means the same representational geometry up to rotation and scale. It is a basis-free similarity, not causal
  evidence.
- **HAND-PC R²** (`hand_pc_r2`): held-out R² with which a linear map from the trained network's residual predicts the top 8 principal
  components of HAND's residual at the same layer on the same inputs. Basis-free, so it is the comparison to read for TRAIN.
- **Align fit** (`fit_r2`): R² of HAND's residual from the trained network's residual after the best rotation and scaling (orthogonal
  Procrustes, with heads and neurons matched). Below 0 means no orthogonal map relates the two codes; the alignment is read only when its
  function error (logit change of the aligned network) is near 0.
- **RSA**: representational similarity analysis; the Spearman correlation between the pairwise-distance matrix of number embeddings and
  that of a reference geometry (a line, a circle, a helix, or HAND's embedding rows).
- **Activation patching, drop and recovery**: a component's activation is swapped between a clean and a corrupted input. Drop =
  (clean − noised) / (clean − corrupt), where the clean run receives the corrupted activation; recovery = (denoised − corrupt) /
  (clean − corrupt), where the corrupted run receives the clean activation. Drop 1 means the component is needed; recovery 1 means it
  is sufficient. Rows are sorted by drop.
- **Donor patch**: patching in an activation taken from another input, the donor; "same-token donors" share the current token, so
  that only the patched variable differs.
- **Path drop**: in path patching only the direct connections of one path carry corrupted values; the drop is the share of the full
  corruption effect on the mean log-probability that this path alone produces (1 = all of it).
- **Step Jaccard**: the overlap (intersection over union) of the neuron sets used at different scratchpad steps; near 1 means one
  shared circuit serves every step.
- **Edges (EAP-IG) / edge overlap (Jaccard)**: an attribution estimate of the circuit's connections (edge attribution patching with
  integrated gradients), and the share of top connections a trained net and the hand-built net have in common. Trusted only when its
  exactness check passes.
- **spearman_exact (EAP-IG faithfulness)**: the rank agreement between the EAP-IG edge scores and exact single-edge patching on the
  top-scoring and on random edges. Edge results are trusted at spearman_exact ≥ .8; below that bar the edge scores are not faithful.
- **hand_recall**: the share of HAND's edge set found among the trained network's top 2k edges, where k is the size of HAND's set.
- **Faith (discovered circuit)**: the exact match kept on held-out inputs when everything outside an automatically discovered circuit
  is resample-ablated.
- **Loss barrier (LMC barrier)**: linear mode connectivity; the largest rise of the loss above the straight line between the two end
  losses when interpolating between two seeds' weights. Near 0 means both sit in one basin.
- **Fourier "clock"**: numbers encoded as angles on a circle, so adding numbers is rotating (known from modular addition).
- **Pizza**: the variant of the Fourier clock in which the answer is read from the sum of the two angle vectors, so the logit also
  depends on x − y through |cos(w(x − y)/2)|; the clock does not depend on x − y.
- **Discrete log**: for a prime p and a generator g, the discrete log of a nonzero residue a is the k with g^k = a mod p. It turns
  multiplication into addition of logs, so a clock over the logs computes products.
- **Nyquist frequency**: the highest frequency a grid of integers can carry, one sign change per step: cos(πn) = (−1)^n.

## Hand-built machinery
- **m2**: the frozen hand-built 6-layer transformer that reads prompts for the counting and modular-addition entries.
  ([METHODS §2](METHODS.md#2-the-frozen-hand-built-model-m2))
- **Menu**: a frozen hand-written layer that re-encodes each integer several ways (scalar, one-hot, thermometer) for the trained stage
  to choose from. ([METHODS §3](METHODS.md#3-the-frozen-menu-and-the-trained-stage))
- **Index head / fetch head**: an attention head that copies the token at one computed position (an operand digit, the previous
  scratchpad bit) to the current position.
- **Parity staircase**: a layer of paired ReLU steps that reads one bit of a large integer sum: the output steps up and down at
  alternate thresholds, so its output alternates between 0 and 1 as the sum grows.
- **Recency head**: an attention head whose score grows with position, so it attends to the most recent matching token.
- **Match head**: an attention head that attends to the token whose key equals the query (a tag, an id or a value), used to fetch an
  operand by name.
- **Chain head**: an attention head that attends to the first lower column with a flag set (for example "this column decides the
  carry") and copies its value, which gives carry or borrow lookahead in constant depth.
- **fp32 / TF32**: full / reduced-precision matrix multiplication; hand-built parts are only exact in fp32.

## Standard machine-learning terms
- **Logit**: a raw output score, one per possible answer; the answer with the largest logit is the prediction.
- **One-hot code**: a vector with one unit per possible value, 1 at the value present and 0 elsewhere.
- **Thermometer code**: a vector whose k-th unit is 1 when the number is at least k, so the units fill up from the bottom like a
  thermometer.
- **ReLU**: the function max(0, x); a ReLU unit is off below its threshold and grows linearly above it.
- **GELU**: a smooth variant of ReLU that lets small negative inputs through slightly.
- **MLP**: a feed-forward block of two weight matrices with a ReLU or GELU between them; in a transformer it acts on each position
  separately.
- **Attention head (query, key, value)**: a unit that lets a position read from other positions. Each position emits a query; each
  position it may read emits a key and a value; the head averages the values, weighting each by how well its key matches the query.
- **Residual stream**: the running vector at each position that every layer reads from and adds its output to.
- **LayerNorm / rms norm**: rescaling a vector to a fixed size; LayerNorm also subtracts the mean. The rms (root mean square) norm of
  a vector is the square root of the mean of its squared entries.
- **GPT-NeoX-shaped decoder**: the transformer layout of the GPT-NeoX models: a stack of layers, each an attention block and an MLP,
  reading left to right so that each position sees only earlier ones.
- **d_model / d_head**: the width of the residual stream, and the width of each attention head's query, key and value vectors.
- **AdamW**: the standard optimiser for transformers; it adapts the step size per weight and applies weight decay separately.
- **Weight decay**: a pull of every weight towards zero at each step; larger values favour smaller, simpler solutions.
- **Warmup and cosine schedule**: the learning rate rises from zero over the first steps (warmup), then falls along half a cosine
  curve towards zero.
- **Gradient clipping**: capping the size of each update step (for example at norm 1) so that a single bad batch cannot throw
  training off.
- **Full batch**: every training example is used in every step.
- **OHEM**: online hard example mining; each step keeps the highest-loss rows of a larger batch (for example 256 of 1,024).
- **Early stop / plateau stop**: ending a run once it reaches the target accuracy (early stop), or once test accuracy has not improved
  for a set number of steps (plateau stop).
- **Chance level**: the accuracy of guessing, for example .2 with 5 equally likely answers.
- **safetensors**: a file format for network weights that stores only numbers, so loading it cannot run code.
- **sha256**: a fingerprint of a file; any change to the file changes it, so it confirms that a download is the archived file.
- **BOS token**: the "beginning of sequence" token placed before the prompt.
- **One-shot prefix**: a single solved example placed before the question, showing the format of the answer.
- **DTM**: depth to mate, the number of moves to checkmate under optimal play.
- **Distractor feature**: an input given to the trained stage that carries no information about the answer, to see whether training
  ignores it.
