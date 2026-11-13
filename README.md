# Mechanistic Interpretability Circuit Atlas

This atlas asks what small neural networks compute, how we can check their circuits, and how those circuits compare with algorithms written by hand. Its 41 entries cover counting, arithmetic, number formats, automata, finite fields, symbolic mathematics, language, search, chess and external recurrent-model zoos. Each entry explains the task first, then gives the model, results, proposed circuit and tests that support or limit the explanation. Start with [count-mod-5](entries/count-mod-5.md), browse the [domain index](INDEX.md), or use the [glossary](GLOSSARY.md) when a term is unfamiliar.

The transformer studies compare hand-built weights, one stage trained on frozen hand-built parts, and training from scratch. A closed form is an equation-based replacement for a trained computation. It counts as verified only when both substitution and resampling everything outside the claimed circuit retain at least .90 of the network's accuracy under the entry's stated criterion. Several updated entries now pass those tests at a stated scope; others remain partial or sufficient readouts. The external zoos have their own sampled-certificate accounting, with limitations near the top of each page; “proven” there does not mean an all-input theorem or perfect task performance.

In the arithmetic comparisons, trained networks often compute the hand-built variables with different wiring: from-scratch networks share at most .15 of their top circuit edges with HAND (Jaccard), against .78–1.0 for networks trained on frozen parts. Some failed networks still carry the variables but fail to generalise. A scratchpad, which writes intermediate results before the answer, can shrink the hand-built circuit by about an order of magnitude (long division: 186 neurons without one, 14 with one). These are findings at the scopes linked below, not universal claims about every entry.

## Introduction

Hand-built transformers are already used as ground truth in interpretability. Tracr compiles programs into transformer weights
([Lindner et al., 2023](REFERENCES.md#lindner2023)), and InterpBench trains models to implement known circuits so that circuit-discovery
methods can be scored ([Gupta et al., 2024](REFERENCES.md#gupta2024)). Both use a known circuit to test interpretability tools. Small
algorithmic tasks are also where trained networks have been reverse-engineered in detail, most clearly the Fourier circuit that
appears when a network groks modular addition ([Power et al., 2022](REFERENCES.md#power2022);
[Nanda et al., 2023](REFERENCES.md#nanda2023)).

The atlas turns the question around. For each task it writes the exact circuit by hand and then asks what gradient descent finds in
its place: whether it reads the hand-built parts when they are frozen and available, whether it rediscovers the same algorithm from
scratch, and how often each solution appears over seeds. Because every hand-built variable is known exactly, a trained network can be
probed, patched and ablated against a reference instead of against a guess.

What is new is the protocol rather than any single task. The same three rungs, usually ten seeds per final cell, circuit types counted per
seed and one acceptance rule are applied across the transformer tasks, and negative results are kept. The acceptance rule follows causal scrubbing
([Chan et al., 2022](REFERENCES.md#chan2022)): a formula that reproduces the network when substituted is only a sufficient readout
until the rest of the network has also been shown to be unnecessary ([METHODS §9](METHODS.md#9-verification-and-the-acceptance-rule);
[related work](METHODS.md#related-work)).

## Main findings

Swapping parts one at a time does not verify a circuit. Earlier single-component substitutions kept accuracy while joint scrubs failed in five tasks; those failed tests remain failed even where a separately tested successor form now passes: [group automata](entries/dfa-group-automata.md) (71.42% and 1.69% on two seeds),
[base conversion](entries/base-conversion.md) (.556 / .923), [long division](entries/long-division-square-root.md) (2.9% and 1.8%),
[subtraction](entries/subtraction.md) (.938 / .933) and [decimal and hexadecimal addition](entries/decimal-hex-addition.md)
(.74–.82). A working replacement is therefore reported as a sufficient readout and not as a verified mechanism.

Trained networks compute the hand-built variables with different wiring. In [binary addition](entries/binary-addition.md) and
[decimal and hexadecimal addition](entries/decimal-hex-addition.md), networks trained from scratch share at most .15 of their top circuit
edges with the hand-built network (Jaccard), while networks trained on frozen hand-built parts share .78–1.0
([SURPRISES](SURPRISES.md#same-variables-different-wiring)).

Failed networks still use the right variables; they fail to generalise. Swapping a hand-built variable between two inputs flips the
answer in failed networks about as often as in solved ones, in [binary addition](entries/binary-addition.md),
[decimal and hexadecimal addition](entries/decimal-hex-addition.md) and
[long division and square root](entries/long-division-square-root.md).

A scratchpad shrinks the hand-built circuit by about an order of magnitude (Figure 1). Long division needs 186 neurons when the network
writes the answer only and 14 when it writes the steps, and long-hand square root 46 against 20. With a scratchpad of running sums, the
stage trained on frozen parts in [binary multiplication](entries/multiplication.md) reuses the adder's column-sum lookup in 10/10 seeds
(a sufficient readout, substitution 1.00), while answer-only multiplication never trains.

![Hand-built circuit size, answer only vs with a scratchpad: long division 186 vs 14 neurons, square root 46 vs 20 neurons, 6-bit multiplication 14 vs 7 heads](figures/scratchpad_size.svg)

*Figure 1. Size of the hand-built circuits with the answer only and with a scratchpad: long division 186 against 14 neurons, long-hand
square root 46 against 20 neurons, 6-bit binary multiplication 14 against 7 heads.*

These and other results that went against expectation are collected, each marked verified or preliminary, in
[SURPRISES.md](SURPRISES.md).

## How to read the atlas

Each task is studied at three rungs. HAND is the hand-built network, exact on every input in range; PART trains one stage on top of
frozen hand-built parts; TRAIN trains the whole network from scratch. A seed count such as 6/10 means that 6 of 10 seeds meet the
cell's criterion, usually held-out exact match ≥ .99. Each entry is a short article with a lede, a fact table, numbered sections, a
figure of the hand-built circuit and an appendix of weights and diagnostics; [METHODS.md](METHODS.md#reading-an-entry) describes the
layout, [GLOSSARY.md](GLOSSARY.md) defines the terms and [STYLE.md](STYLE.md) sets the writing and verification standard. To start,
read [count-mod-5](entries/count-mod-5.md), a complete and reproducible entry, and then scan [INDEX.md](INDEX.md), which gives one row
per entry with its circuit, seed counts and closed-form status.

## Entries

Main result per entry; the full row is in [INDEX.md](INDEX.md). ⟳ marks entries that rerun from [code/](code/README.md); the others
publish their networks, test sets and recorded accuracies on [Hugging Face](https://huggingface.co/jc1122/circuit-atlas-nets).

**Counting and modular arithmetic**
- [modular-addition-mlp](entries/modular-addition-mlp.md) ⟳: Fourier clock circuit in 10/10 seeds; test accuracy stays below .99
- [count-mod-5](entries/count-mod-5.md) ⟳: flat lookup from the count to its residue in 30/30
- [compare-counts](entries/compare-counts.md) ⟳: linear comparator with a tilted slope; ties are memorised, never generalised
- [open-bracket-count](entries/open-bracket-count.md) ⟳: low-frequency cosine sum plus a parity channel; memorises below 80% data
- [modular-arithmetic-mod-97](entries/modular-arithmetic-mod-97.md): token multiplication uses discrete-log circles; digit formats form no circles

**Arithmetic**
- [decimal-hex-addition](entries/decimal-hex-addition.md): column-sum lookup; trained networks never form the carry-lookahead head
- [binary-addition](entries/binary-addition.md): lookup on a + b + carry; position inputs prevent length generalisation until masked
- [router-addition](entries/router-addition.md): one shared column step; longer inputs work only without position input
- [subtraction](entries/subtraction.md): per-column lookup on a − b − borrow; no length generalisation from scratch
- [multiplication](entries/multiplication.md): the scratchpad step reuses the adder's lookup; answer only never trains
- [multiplication-base-order](entries/multiplication-base-order.md): least significant digit first is learnable, most significant first rarely; a scratchpad solves both
- [long-division-square-root](entries/long-division-square-root.md): one shared compare-and-subtract step; answer only needs 16–90x more data
- [gcd-euclid](entries/gcd-euclid.md): three-rule router on frozen Euclid parts; from scratch, seed-specific distributed circuits
- [quadratic-roots](entries/quadratic-roots.md): executor PART reduces to one verified AND neuron; from-scratch and parent formats fail
- [compose-arithmetic-registers](entries/compose-arithmetic-registers.md): compositions learned without register lookup; the bare step fails
- [integer-matvec](entries/integer-matvec.md): dense last-layer carry circuit reaches 8/10 exact with scratchpad; answer only needs longer training

**Number formats**
- [base-conversion](entries/base-conversion.md) ⟳: hex to binary groks as a nibble lookup; carry-type conversions are never learned
- [output-order-base-conversion](entries/output-order-base-conversion.md): most significant first is a threshold ladder; least significant first mostly fails

**Automata and finite fields**
- [dfa-group-automata](entries/dfa-group-automata.md): serial scratchpad MLP chain; answer-only S3 uses prefix doubling on frozen parts
- [gf2-matrix-product](entries/gf2-matrix-product.md): XOR-AND lookup per step; from scratch, the same variables with different wiring
- [gf2-elimination](entries/gf2-elimination.md): XOR-AND row update; from scratch, three circuit types given enough matrices
- [linear-systems-finite-fields](entries/linear-systems-finite-fields.md): microstep scratchpad learned from scratch; fused updates bypass the hand-built subtractor
- [crc-remainder](entries/crc-remainder.md): distributed shift-XOR step; readout and circuit checks pass, without an upstream formula
- [dyck-k-and-agreement](entries/dyck-k-and-agreement.md): agreement copies the first noun; bracket scratchpads learn a pop-pointer circuit with failed complement checks
- [counter-languages](entries/counter-languages.md): ratio counts and summed violations solve answer-only tasks; scratchpads solve only one language
- [binding-coreference](entries/binding-coreference.md): one head attends to the referent; substitution works but every type-level complement fails

**Symbolic mathematics**
- [polynomial-add-mul](entries/polynomial-add-mul.md): addition learned in 40/40 networks; multiplication never reaches exact match
- [polynomial-differentiation](entries/polynomial-differentiation.md): low-bit multiplier learned; the unseen multiplier bit never is
- [polynomial-integration](entries/polynomial-integration.md): only the scratchpad transfers to withheld fractions
- [chain-rule-differentiation](entries/chain-rule-differentiation.md): solved from scratch with a token-pair front end; frozen-parser setup bypasses the parser
- [symbolic-algebra-atoms](entries/symbolic-algebra-atoms.md): collecting terms learned at every rung; expanding and solving need a scratchpad
- [nested-expressions-mod-7](entries/nested-expressions-mod-7.md): verified residue table on a frozen parser; from-scratch nets fail held-out literals
- [euler-decay-ode](entries/euler-decay-ode.md): frozen-route update misses held-out first steps; from-scratch nets memorise states, with no solved cell

**Search and games**
- [grid-bfs](entries/grid-bfs.md): distributed reached-set scratchpad solves 10/10; answer only fails beyond distance 3
- [dpll-search](entries/dpll-search.md): verified local update completes the frozen solver in 10/10; from-scratch state updates fail
- [chess-krk](entries/chess-krk.md): rule router at .993–.999, never exact; trained networks code kings as coordinates
- [chess-mate-in-one](entries/chess-mate-in-one.md): the trained final MLP decodes the hand-built escape count
- [chess-attack-maps](entries/chess-attack-maps.md): hint scratchpad learns ray pointers, reaching 7/10 recorded-test exact; tested answer-only networks fail
- [chess-kbbk-mate](entries/chess-kbbk-mate.md): distributed lookup and candidate copying; 0/9 seeds solve held-out moves

**External model zoos**
- [algzoo-2nd-argmax](entries/algzoo-2nd-argmax.md): sampled forms cover 132 networks, including optimal-only flags; many longer networks remain unexplained
- [algzoo-median](entries/algzoo-median.md): sampled forms cover 100 nets, with empty-complement exceptions, inexact medians and uncompressed forms

## Quick start

```
cd code && pip install -r requirements.txt
python run.py check                                                         # the hand-built model and the menus are exact
python run.py atlas --task count_mod5 --seeds 0 --steps 400 --every 100    # smoke run, about a minute
```

The full commands per entry are in [code/README.md](code/README.md).

## Status

The atlas is work in progress. The entries include published updates through 2026-11-13. The two external-zoo entries are dated snapshots, not automatically updated counts: second-argmax ends at Sol iteration 12 and median at iteration 22. Each entry states its own last run, evaluation criterion, archive coverage and unresolved tests. Five entries rerun from `code/`; the others link archived networks and recorded checks, with code not yet ported. How the atlas was made (one author, with LLM agents working under a written protocol) is described in [METHODS.md](METHODS.md#how-this-was-made).

The earlier UX snapshot covered 31 transformer tasks, with three called solved, 27 partly solved and one unsolved, and last runs dated 2026-11-04 to 2026-11-08. It reported no entry meeting the full verification rule. Those are historical snapshot claims, not totals for the current 41-entry catalogue; later entries add accepted forms, explicit negative results and external zoos. Current verdicts and their scopes are listed per entry in [INDEX.md](INDEX.md).

## Cite and license

Cite the atlas via [CITATION.cff](CITATION.cff) (GitHub: "Cite this repository"). Text and networks are released under
[CC BY 4.0](LICENSE) and the code in `code/` under [MIT](code/LICENSE).
