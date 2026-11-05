# Entry template

```
# <Task name>

**Task.** Input/output format, ranges, train/test split.
**Status.** Seeds run, date, architecture, training budget.

## Summary
Two to four sentences: which circuit, how often, verified closed form.

## Setup
Link [METHODS.md](../METHODS.md) for the shared architecture and instrumentation; then only what is task-specific: prompt format and
generator, label rule, number of cells and classes, menu width, hidden width, optimiser values if they differ, steps, evaluation
interval, train fractions, seeds, controls, hardware class (no host names). End with the reproduce command(s), runnable from `code/`
(no paths outside the repository), and the output keys that carry the headline numbers.

    python run.py <command> --task <task> --seeds 0 1 2 ...

## Architecture as used
Required. The exact configuration of every network in the entry, as run: layers, d_model, heads (d_head), MLP width, activation,
vocabulary and positions; the frozen parts and the trained parts (with shapes, e.g. W_in [256, 326]); the input blocks of the trained
part and their index ranges. Name the archived files (safetensors + config JSON + manifest with seed, rung, accuracy, sha256) and say
where the full networks are published.

## Prediction before training
Required. What the hand-built design and earlier entries predicted the trained networks would do (circuit, parts read, closed form,
grokking), written before the TRAIN/PART runs, and whether it held. If no prediction was recorded in advance, say so; a retrospective
prediction must be labelled as such. This is the record on which the atlas is scored as a predictor of new circuits.

## Hand-built reference
Mechanism in words, then the weights as formulas (per layer / head / neuron block). Exactness check.

## What training finds
- Setups: fully trained; frozen hand-built parts + trained stage (name the frozen parts and the trained stage).
- Learning curve: memorization step, 50% / 90% test step, final accuracy.
- Circuit types, clustered by mechanism (not by weights), with seed counts.
- Which frozen parts are read (ablation per part).

## Closed form
Equation per circuit type; accuracy when substituted for the trained part; agreement with the network.

## Weights excerpt and reading
Required. The key matrices or vectors of the circuit, small and exact, taken from the archived weights: the hand-built formula weights
(constants and the non-zero pattern), the trained closed-form parameters per circuit type and seed (a table), and selected rows,
neurons or heads (a table or a short code block; full tensors stay in the archive). Then the reading: what each component computes and
why these numbers produce it. State how each number is computed (weight slice or fit) so it can be recomputed from the published weights.

## Comparison
Hand-built vs trained: same algorithm? Differences. Relation to published circuits.

## Controls and limits

## References
DOI / arXiv links only for works that were checked.
```
