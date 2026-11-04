# Entry template

```
# <Task name>

**Task.** Input/output format, ranges, train/test split.
**Status.** Seeds run, date, architecture, training budget.

## Summary
Two to four sentences: which circuit, how often, verified closed form.

## Hand-built reference
Mechanism in words, then the weights as formulas (per layer / head / neuron block). Exactness check.

## What training finds
- Setups: fully trained; frozen hand-built parts + trained stage (name the frozen parts and the trained stage).
- Learning curve: memorization step, 50% / 90% test step, final accuracy.
- Circuit types, clustered by mechanism (not by weights), with seed counts.
- Which frozen parts are read (ablation per part).

## Closed form
Equation per circuit type; accuracy when substituted for the trained part; agreement with the network.

## Comparison
Hand-built vs trained: same algorithm? Differences. Relation to published circuits.

## Controls and limits

## References
DOI / arXiv links only for works that were checked.
```
