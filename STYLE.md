# Writing standard for atlas entries

An entry is a short encyclopedia article or scientific paper about one task. A reader who has never seen the project should learn
from the first screen what was analysed (task, data, model) and what the network computes (the algorithm or circuit), and should be
able to check every number against the archived weights. The section layout is in [TEMPLATE.md](TEMPLATE.md); this file sets the
register, the citation and figure conventions, and the checks an entry must pass.

## Register

- Scientific, plain English with British spelling (-ise, centre, behaviour, generalise, memorise). Code, output keys and label names
  stay as they are.
- Past tense for what was done ("ten seeds were trained"); present tense for what a network computes ("MLP3 reads the one-hot code").
- Full sentences in paragraphs. A paragraph opens with its point and then gives the evidence.
- Define each term in a clause at its first use and link the [glossary](GLOSSARY.md) entry as well (inside `entries/` the path is `../GLOSSARY.md`): "a scratchpad, in which the
  network writes intermediate results before the answer ([glossary](GLOSSARY.md#task-formats))". Entry-specific variables (LK_k, X,
  E_k) are defined in words where they first appear.
- Rung and format letters (HAND, PART, TRAIN; `A`, `B`, `T`) may be used once they are introduced; write "run 1" and "run 2", not
  r1/r2, and "seed 2", not s2, in prose. Tables may keep compact labels if a key under the table expands them.
- Every number carries its n and its criterion: "8/10 seeds reached exact match ≥ .99 on held-out pairs", not "8/10 solved".
- Numbers are copied, never rounded, merged into new ranges, averaged or recomputed. A range stays a range.
- Verdicts are never stronger than the evidence. An audit caveat (for example "sufficient readout, not a verified mechanism") is
  repeated wherever the closed form is named: lede, fact table, Section 5 and the figure caption.
- "Verified" is reserved for the atlas acceptance rule: substituting the closed form for the trained part keeps accuracy and a
  resample ablation of everything outside the claimed circuit also keeps it (within the acceptance share in METHODS.md). With
  substitution alone write "supported by substitution; complement not tested"; the Closed form status is then "partial".
- Paraphrase at the strength of the data: "nearly identical" when a table shows 14.25 against 13.40, not "identical".
- When the old entry gives two values for one quantity (for example .10–.19 in the text and .10–.20 in a diagnostics table), keep
  both and say where each comes from if the entry says so ("training log" vs "diagnostics run on the best snapshot"); if it does not,
  keep both, attach each to its own table, and list the pair in your report. Never pick one silently.
- Entry-specific jargon (parity staircase, index or fetch head, truth table, interchange variable, full bar, native/zero/shuffle
  controls, f(...) labels, μ[q, u], diagnostics keys such as spearman_exact) is defined in words at first use or linked to a GLOSSARY
  entry; add any missing GLOSSARY term to your report, not to GLOSSARY.md.

## What not to write

- Bold-label bullet lists (`- **Label.** text`). Use a paragraph, or a table if the content is per cell or per component.
- Contrast flourishes of the form "not X but Y", "not just X, it is Y".
- Em-dash chains or em dashes as sentence glue; use a full stop, a comma or parentheses.
- Marketing adjectives and intensifiers (remarkable, striking, elegant, powerful, crucial, novel, notably, importantly, key insight).
- Rhetorical questions. A question is allowed only inside a quoted prediction record.
- "Note:", "TL;DR", "In summary", emojis, and arrows used as prose ("A → memorises"). An arrow is allowed in the fact table's
  example row, in table cells that show a before/after value, and inside code and Mermaid blocks.
- Fragments and shorthand such as "t(.9) 500", "f(add) 10/10" in prose; spell them out once ("the first evaluation at which test
  accuracy reached .9") and keep the compact form for tables.

Bullets are for content that is genuinely a list (output formats, archive files, open questions when there are several). Per-cell
results, per-seed values and per-component evidence go in tables with a numbered caption and a key for abbreviations.

## Structure (why the template looks as it does)

The order follows a short paper: lede (abstract), fact table (the reader's "infobox"), task and data, model, the hand-built solution,
results, mechanism and verification, limitations, prediction record, appendix, references. Three departures from the plain paper
order, with reasons:

- A **Status** row was added to the fact table. The publishing pipeline needs the date of the last run and the verdict word
  (Solved / Partly solved / Unsolved); the lede states the verdict in prose, the table states it in a form a script can read.
- **Training settings and reproduction** sit in Section 2 (Model) as subsections, not in a separate Setup section, so that every
  setting stays next to the network it applies to. The fact table links to the reproduction subsection.
- The appendix is split into **A. Weights excerpt and reading** (required in every entry) and **B. Detailed evidence**, so that the
  weight excerpt, which readers use to check the closed form, is always at a fixed place.

Detailed evidence that supports but does not carry the argument (weight excerpts, full diagnostics tables, per-seed values) goes to
the appendix. Nothing is dropped in the move: every number of the earlier version must survive in the body or the appendix.

## Citations

- In text, author-year, linked to the entry's References section: `[Behrens et al. (2024)](#references)`, or in parentheses
  `([Dziri et al., 2023](#references))`. One author: the surname alone. Two authors: both surnames joined by "and" in running text and "&" in parentheses. Three or
  more authors: first author et al.
- References use one format, a bulleted list in alphabetical order of first author:
  `Author, A., & Author, B. (Year). Title. *Venue*. https://doi.org/...` (an arXiv link where there is no DOI) followed by a short parenthesis saying what the work is
  cited for. Take author lists, titles and venues from [REFERENCES.md](REFERENCES.md) at the repository root (it corrects earlier
  author errors); keep every link the entry already had. Long author lists follow REFERENCES.md (they may end in "et al.").
- Cite only works that were checked. Every work in References is cited at least once in the text.
- To add a reference, copy its line from REFERENCES.md into the entry's References at its alphabetical place and cite it in the text. A
  work that is not in REFERENCES.md is not cited; list it with its link in your report.

## Figures and tables

- Every entry has **Figure 1**: a Mermaid flowchart (a ```` ```mermaid ```` block, which GitHub renders) of the hand-built circuit, with
  the part that is trained (PART) marked by `classDef trained` and stated in the caption. If the task has no PART cell, mark no node
  and say so in the caption. Use the entry's own names for layers, heads
  and MLPs (L3, MLP3, "layer 4 head 1"), and its own variable names.
- Where training finds a different or more specific circuit, **Figure 2** shows it in the same style, and its caption repeats the
  closed-form status word of the fact table (verified, partial, sufficient readout only) with its audit caveat, and says which nodes are
  marked as trained.
- Captions are numbered and in italics directly below a figure and above a table: "*Figure 1. The hand-built network ... The shaded
  MLP is the only trained part.*" A caption states what is shown, not what to conclude.
- Mermaid syntax kept simple so that it parses: `flowchart TD` (top-down reads at every width; `LR` only for three nodes or fewer); every label in double quotes; no double quotes,
  backticks or `#` inside a label; ASCII operators in labels (`-`, `>=`, `xor`); colours as named CSS colours in one `classDef`.
  Every block must pass the Mermaid 11 parser (`await mermaid.parse(block)` from the `mermaid` npm package, or mermaid-cli) before an
  entry is published.

## Checks before publishing

1. Numbers: compare the multiset of numeric tokens of the old and new entry (after normalising minus signs and dashes). No token may
   be lost unless the entry merges a duplicated statement; list such merges in the commit message. A new entry has no old version:
   check instead that every number in it comes from the task's results and that the fact-table rules in TEMPLATE.md hold.
2. Mermaid: every block parses (see above).
3. No host names, local paths or accounts (the Hugging Face weights URL is the one allowed account name); every link is a public URL or a path inside this repository, and every relative link and
   `#anchor` resolves to an existing file or heading.
4. AI tells. Run on prose only (outside fenced blocks and table rows, before References) and expect no output (`E.md` is the entry file):

```
prose() { awk '/^## References/{exit} /^```/{f=!f;next} !f && !/^\|/' "$1"; }
prose E.md | grep -nE '^\s*[-*] \*\*[^*]+[.:]\*\*'                  # bold-label bullets
prose E.md | grep -n '—'                                            # em dashes
prose E.md | grep -niE '\bnot (just |only |merely )?[a-z-]+( [a-z-]+){0,3},? but\b'   # not X but Y
prose E.md | grep -niE '\b(remarkabl|striking|elegant|powerful|crucial|seamless|novel|fascinating|notably|importantly|key insight|delve)'
prose E.md | grep -nE '(^|[^"])\b[A-Z][^.?!"]*\?\s'                # questions outside quotes
prose E.md | grep -nE 'Note:|TL;DR|In summary'
prose E.md | grep -n '→'                                            # arrows in prose
prose E.md | grep -nP '[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]'       # emojis
```
