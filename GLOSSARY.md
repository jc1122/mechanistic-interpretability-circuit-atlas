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

## Verifying a circuit
- **Closed form**: a short formula for what a trained part computes.
- **Substitution**: replace the trained part by its closed form and re-score; the closed form is accepted when accuracy is kept.
- **Sufficient readout**: a substitution keeps accuracy, but the full bar (best and final snapshots, zero and shuffle controls) is
  not passed. The formula is enough to read out the answer; it is not shown to be the mechanism.
- **Resample ablation of the complement (scrub)**: replace everything *outside* the claimed circuit with its values on other inputs.
  If accuracy survives, the rest of the network was not needed; a working substitution alone does not show that.
- **Ablation**: switch a part off (set it to zero or to its mean) and see whether accuracy drops. A head or MLP is "used" when its
  ablation costs more than .05 of exact match.

## Analysis measures
- **R²** (probe R², "at Lk" = after layer k): how well a linear (ridge) read-out of the residual stream recovers a hand-built variable
  (1 = perfectly). Decodable is not the same as used.
- **Interchange flip**: swap one variable's value between two inputs inside the network and count how often the answer changes as
  the variable predicts; a high flip means the variable is causally used. `nf` = number of flips tested; nf 0 means untested, not
  unused.
- **DAS** (distributed alignment search): the same swap through a learned rotation; catches variables spread over many directions.
  "Unused" is read only when DAS is also low.
- **JS**: mean Jensen-Shannon divergence (in nats; ln 2 = disjoint) between a trained head's attention pattern and the
  closest hand-built head's (0 = identical).
- **PR, r90, r99**: participation ratio (effective number of dimensions) of the residual stream or of a weight matrix, and its rank
  at 90% / 99% of the variance or energy.
- **Edges (EAP-IG) / edge overlap (Jaccard)**: an attribution estimate of the circuit's connections, and the share of top connections
  a trained net and the hand-built net have in common. Trusted only when its exactness check passes.
- **Loss barrier (LMC)**: the loss bump when interpolating between two seeds' weights; near 0 means both sit in one basin.
- **Fourier "clock"**: numbers encoded as angles on a circle, so adding numbers is rotating (known from modular addition).

## Hand-built machinery
- **m2**: the frozen hand-built 6-layer transformer that reads prompts for the counting and modular-addition entries.
  ([METHODS §2](METHODS.md#2-the-frozen-hand-built-model-m2))
- **Menu**: a frozen hand-written layer that re-encodes each integer several ways (scalar, one-hot, thermometer) for the trained stage
  to choose from. ([METHODS §3](METHODS.md#3-the-frozen-menu-and-the-trained-stage))
- **fp32 / TF32**: full / reduced-precision matrix multiplication; hand-built parts are only exact in fp32.
