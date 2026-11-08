# Entry template

Spelling: British (-ise, centre, behaviour, generalise, memorise) in prose; code, output keys and label names stay as they are.
Numbers: never rounded or recomputed when copied between sections. Terms used in more than one entry are defined once in
[GLOSSARY.md](GLOSSARY.md); link it, do not redefine. Entry-specific variables (e.g. LK_k, PA) are defined in words at their first
use in the top block and in the Summary.

```
# <Task name, not a claim>

> - **Task.** <one sentence>. Example: `<input>` → `<output>`.
> - **Hand-built net.** <the algorithm the exact hand-written network implements, one sentence>.
> - **Trained nets.** <what training from scratch and on frozen hand-built parts learns, with seed counts>.
> - **Verdict.** <Solved | Partly solved | Unsolved> (<cell, n/N seeds>). Closed form: <verified | partial | sufficient readout only | rejected | none> (<deciding number>).
> - **Links.** [Weights](<Hugging Face folder>) · Reproduce: <[commands](#setup) | not available yet> · [Terms](../GLOSSARY.md)

## Task and status
**Task.** Input/output format, ranges, train/test split.
**Formats.** One bullet per format letter used in this entry: `A` answer only; `<letter>` <what the scratchpad writes>.
**Status.** <date of the last run> · rungs run: <HAND, PART, TRAIN> · seeds: <per cell> · training budget · <audit caveat if any>.

## Summary
Two to four sentences: which circuit, how often, verified closed form. Expand each rung/format letter and diagnostic at its first use
here or link it to GLOSSARY.md.

## Setup
Link [METHODS.md](../METHODS.md) for the shared architecture and instrumentation; then only what is task-specific: prompt format and
generator, label rule, number of cells and classes, menu width, hidden width, optimiser values if they differ, steps, evaluation
interval, train fractions, seeds, controls, hardware class (no host names). Give the Hugging Face folder link, then end with the
reproduce command(s), runnable from `code/` (no paths outside the repository), and the output keys that carry the headline numbers.

    python run.py <command> --task <task> --seeds 0 1 2 ...

If the task is not in `code/` yet, write exactly: "**Reproduce:** not available yet: this task's training and analysis code is not in
`code/`. The weights, test splits and recorded accuracies are in the Hugging Face folder above."

## Architecture as used
Required. The exact configuration of every network in the entry, as run: layers, d_model, heads (d_head), MLP width, activation,
vocabulary and positions; the frozen parts and the trained parts (with shapes, e.g. W_in [256, 326]); the input blocks of the trained
part and their index ranges. Name the archived files (safetensors + config JSON + manifest with seed, rung, accuracy, sha256) and
point to the Hugging Face folder.

## Prediction before training
Required. What the hand-built design and earlier entries predicted the trained networks would do (circuit, parts read, closed form,
grokking), written before the TRAIN/PART runs, and whether it held. If no prediction was recorded in advance, say so; a retrospective
prediction must be labelled as such. This is the record on which the atlas is scored as a predictor of new circuits.

## Hand-built reference
Mechanism in words, then the weights as formulas (per layer / head / neuron block). Exactness check.

## What training finds
- Setups: fully trained; frozen hand-built parts + trained stage (name the frozen parts and the trained stage).
- Learning curve: memorisation step, 50% / 90% test step, final accuracy.
- Circuit types, clustered by mechanism (not by weights), with seed counts.
- Which frozen parts are read (ablation per part).
- Diagnostics (per setup): used heads + neurons, best linear-probe R² per hand-built variable (layer), residual effective rank vs the
  hand-built net, attention divergence (JS) to the matching hand-built head, best vs final test; failures explained from these.
  Each table gets a one-line key under it naming its abbreviations (or a link to GLOSSARY.md).

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

## Top-block rules

1. Five labelled lines, each a list item inside the blockquote (`> - **Label.** ...`), so that GitHub renders them on separate lines.
   At most 40 words each, except "Trained nets" (at most 60).
2. Every number in the block occurs verbatim in the entry body (same rounding, same range). No new numbers, no derived means.
3. No unexplained shorthand: write "trained from scratch", "trained on frozen hand-built parts", "answer only", "with a scratchpad
   (intermediate steps written out)". Rung and format letters may follow in parentheses, e.g. "(TRAIN, format O)", so the reader can
   find them in the tables below.
4. The example is copied from the entry's Task paragraph if it has one; otherwise it is built only from the stated prompt format and
   label rule, with values small enough to check by eye. If the format is not fully specified (operand bit order, separators), give the
   example in words or numbers, not as a token string. If even that is not possible, omit "Example:".
5. Verdict words: *Solved* = the entry's final cell meets its own criterion in all its seeds, or the entry calls it solved; *Partly
   solved* = some cells, seeds or ranges solved, others not; *Unsolved* = the entry's headline cell fails. The parenthesis names the
   cell and seed count exactly as the entry states it. The closed-form word follows the entry; if the entry carries an audit caveat,
   the caveat wins.
6. Links line: copy the entry's Hugging Face URL (folder names do not always match the entry name); Reproduce links to the Setup
   anchor only if a runnable command exists.
7. Then a `## Task and status` heading holding the Task and Status paragraphs, with format letters as a short list.

## Index and README lines

A new entry adds one line to [README.md](README.md#entries) and one row to [INDEX.md](INDEX.md), each in its domain group (the same
group headings in both files), after the group's last line.

README line:
```
- [<entry>](entries/<entry>.md)<repro>: <main result, at most 15 words>
```
`<repro>` is ` ⟳` when the entry's Setup has a `python run.py ...` command that runs from `code/`, else empty. Main result in plain
words, no format letters; lead with the circuit, then the contrast ("...; answer only never trains"); at most one seed count.

INDEX row:
```
| [<entry>](entries/<entry>.md) | <task, at most 8 words> | <circuit found, at most 15 words> | <[RUNG ]k/n[, RUNG k/n][ *(screen)*]> | <✓|partial|readout only|✗|—>[ <at most 6 words>] | <code|weights> |
```
Every number in both lines occurs in the entry; the closed-form word follows the entry's own audit caveat. Format letters stay out
of both lines.
