# Entry template

Write every entry to [STYLE.md](STYLE.md) (register, citations, figures, checks). Terms used in more than one entry are defined once
in [GLOSSARY.md](GLOSSARY.md); link them at first use and also define them in a clause. Numbers are never rounded or recomputed when
copied between sections. Worked examples: [count-mod-5](entries/count-mod-5.md), [multiplication](entries/multiplication.md).

````
# <Task name, not a claim>

<Lede: 3-5 sentences of prose. What was studied (task, model, what was trained); the main finding with its seed count; the verdict
(Solved | Partly solved | Unsolved) and the closed-form status, with any audit caveat in its own words.>

| | |
|---|---|
| Task | <one sentence> |
| Input → output example | <small example, checkable by eye> |
| Model | <architecture in one line; which part is trained> |
| Rungs and seeds | <HAND / PART / TRAIN cells with seeds per cell; retries> |
| Closed form status | <verified | partial | sufficient readout only | rejected | none> (<deciding number>; <audit caveat>) |
| Status | <Solved | Partly solved | Unsolved> (<cell, n/N>); last run <YYYY-MM-DD>[; audit <YYYY-MM-DD>] |
| Weights | [Hugging Face: <folder>](<Hugging Face folder URL>) |
| Reproduce | <[Commands](#24-reproduction), run from `code/`> or <Not available yet: training code for this task is not yet in `code/`> |

## 1. Task and data
Prompt format, label rule, ranges, number of cells or pairs, train/test split and fractions, the correctness criterion (exact
match of what), length or size tests. Output formats as a table with their letters (answer only, scratchpad variants).

## 2. Model
### 2.1 Architecture (or Frozen hand-built network)
Layers, d_model, heads (d_head), MLP width, activation, normalisation, positions, vocabulary, parameter counts.
### 2.2 Frozen parts and trained part
What is frozen, what is trained (with shapes, e.g. W_in [256, 326]), and which inputs the trained part reads (index ranges as a table).
### 2.3 Training
Optimiser, learning rate, weight decay, schedule, steps, batch, evaluation interval, precision, seeds, controls, hardware class (no
host names). Several runs: a table with one row per run.
### 2.4 Reproduction
Commands runnable from `code/` and the output keys that carry the headline numbers, or exactly: "Reproduce: not available yet:
training code for this task is not yet in `code/`; the weights, held-out splits and recorded accuracies are on Hugging Face at <link>."
### 2.5 Archive
The archived files (safetensors, config JSON, manifest with seed, rung, accuracy, sha256) and the round-trip check.

## 3. Hand-built solution
The algorithm in words, then the weights as formulas per layer, head or neuron block, and the exactness check. Figure 1 (Mermaid
flowchart of the hand-built circuit, trained part marked) with a numbered caption.

## 4. What training finds
Table of results per cell (seeds, solved k/n, best and final test accuracy, steps to criterion, length tests), then prose: learning
curve and grokking, circuit types clustered by mechanism with seed counts, which frozen parts are read (ablations), and failures
explained from diagnostics. Subsections per format when formats behave differently.

## 5. Mechanism and verification
Closed form per circuit type (equation; accuracy when substituted; agreement with the network), the verdict in the entry's own words,
resample ablation of the complement and controls (or a statement that none was reported), wiring and causal evidence, Figure 2 where
training finds a different circuit, and the relation to the hand-built design and to published work (author-year citations).

## 6. Limitations and open questions
What was not run, what the evidence cannot show, open questions.

## 7. Prediction record
What the hand-built design and earlier entries predicted before training, and whether it held. If nothing was recorded in advance,
say so; label any retrospective prediction as such.

## Appendix A. Weights excerpt and reading
Small exact excerpts from the archived weights (hand-built constants, trained closed-form parameters per seed, selected rows or
neurons), how each number is computed (weight slice or fit), and the reading.

## Appendix B. Detailed evidence
Full diagnostics tables (used heads and neurons, probe R² per variable and layer, effective rank, attention JS, patching,
interchange, edges), per-seed data, with a key under each table.

## References
- Author, A., & Author, B. (Year). Title. *Venue*. https://doi.org/... (what the work is cited for)
````

## Fact-table rules

1. Every number in the lede and the fact table occurs verbatim in the body or the appendix (same rounding, same range). No new
   numbers, no derived means.
2. The example is copied from Section 1 if it gives one; otherwise it is built from the stated prompt format and label rule, with
   values small enough to check by eye. If the format is not fully specified (operand bit order, separators), give the example in
   words or numbers, not as a token string; if even that is not possible, write "see Section 1".
3. Verdict words: *Solved* = the entry's final cell meets its own criterion in all its seeds, or the entry calls it solved; *Partly
   solved* = some cells, seeds or ranges solved, others not; *Unsolved* = the headline cell fails. The parenthesis names the cell and
   seed count as the entry states it. The closed-form word follows the entry; if the entry carries an audit caveat, the caveat wins.
4. Weights: copy the entry's Hugging Face URL (folder names do not always match the entry name). Reproduce links to Section 2.4 only
   if a runnable command exists.

## Index and README lines

A new entry adds one line to [README.md](README.md#entries) and one row to [INDEX.md](INDEX.md), each in its domain group (the same
group headings in both files), after the group's last line.

README line:
```
- [<entry>](entries/<entry>.md)<repro>: <main result, at most 15 words>
```
`<repro>` is ` ⟳` when the entry's Reproduce row links a `python run.py ...` command that runs from `code/`, else empty. Main result in plain
words, no format letters; lead with the circuit, then the contrast ("...; answer only never trains"); at most one seed count.

INDEX row:
```
| [<entry>](entries/<entry>.md) | <task, at most 8 words> | <circuit found, at most 15 words> | <[RUNG ]k/n[, RUNG k/n][ *(screen)*]> | <✓|partial|readout only|✗|—>[ <at most 6 words>] | <code|weights> |
```
Every number in both lines occurs in the entry; the closed-form word follows the entry's own audit caveat. Format letters stay out
of both lines.
