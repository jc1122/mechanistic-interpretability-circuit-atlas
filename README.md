# Mechanistic Interpretability Circuit Atlas

This atlas asks which circuits small transformers learn on algorithmic tasks, and how those circuits compare with one written by hand.
For 31 tasks (counting, arithmetic, number formats, automata, finite fields, symbolic mathematics, search and chess) an exact
hand-built transformer was written, with every weight set by a formula, and networks were then trained over many seeds, both as a
single stage on top of frozen hand-built parts and from scratch. For every trained network the atlas reports which circuit it uses,
how many seeds share it, and whether a closed-form description of it survives substitution. Three tasks are solved, 27 are partly
solved and one is unsolved. Trained networks usually compute the hand-built variables, but with different wiring: networks trained
from scratch share 6–9% of their top circuit edges with the hand-built network, against 79–100% for networks trained on frozen parts.
Failed networks still carry the variables and fail to generalise rather than to compute them. A scratchpad shrinks the hand-built
circuit by about an order of magnitude (long division: 186 neurons without one, 14 with one). A closed form is called verified only if
substituting it and resampling everything outside the claimed circuit each keep at least .90 of the network's exact match; no entry
yet meets this rule in full, so the closed forms are reported as partial or as sufficient readouts.

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
seed and one acceptance rule are applied to all 31 tasks, and negative results are kept. The acceptance rule follows causal scrubbing
([Chan et al., 2022](REFERENCES.md#chan2022)): a formula that reproduces the network when substituted is only a sufficient readout
until the rest of the network has also been shown to be unnecessary ([METHODS §9](METHODS.md#9-verification-and-the-acceptance-rule);
[related work](METHODS.md#related-work)).

## Main findings

Swapping parts one at a time does not verify a circuit. Single-component substitutions usually keep accuracy, but scrubbing the whole
proposed circuit at once fails in five tasks: [group automata](entries/dfa-group-automata.md) (71.42% and 1.69% on two seeds),
[base conversion](entries/base-conversion.md) (.556 / .923), [long division](entries/long-division-square-root.md) (2.9% and 1.8%),
[subtraction](entries/subtraction.md) (.938 / .933) and [decimal and hexadecimal addition](entries/decimal-hex-addition.md)
(.74–.82). A working replacement is therefore reported as a sufficient readout and not as a verified mechanism.

Trained networks compute the hand-built variables with different wiring. In [binary addition](entries/binary-addition.md) and
[decimal and hexadecimal addition](entries/decimal-hex-addition.md), networks trained from scratch share only 6–9% of their top circuit
edges with the hand-built network, while networks trained on frozen hand-built parts share 79–100%
([SURPRISES](SURPRISES.md#same-variables-different-wiring)).

Failed networks still use the right variables; they fail to generalise. Swapping a hand-built variable between two inputs flips the
answer in failed networks about as often as in solved ones, in [binary addition](entries/binary-addition.md),
[decimal and hexadecimal addition](entries/decimal-hex-addition.md) and
[long division and square root](entries/long-division-square-root.md) (981 networks analysed across tasks).

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
- [gcd-euclid](entries/gcd-euclid.md): the trained stage is a three-rule router on frozen Euclid parts
- [quadratic-roots](entries/quadratic-roots.md): the register executor reduces to one AND neuron; the parent formats fail
- [compose-arithmetic-registers](entries/compose-arithmetic-registers.md): compositions learned without register lookup; the bare step fails

**Number formats**
- [base-conversion](entries/base-conversion.md) ⟳: hex to binary groks as a nibble lookup; carry-type conversions are never learned
- [output-order-base-conversion](entries/output-order-base-conversion.md): most significant first is a threshold ladder; least significant first mostly fails

**Automata and finite fields**
- [dfa-group-automata](entries/dfa-group-automata.md): transition table learned with a scratchpad; answer-only S3 uses prefix doubling on frozen parts
- [gf2-matrix-product](entries/gf2-matrix-product.md): XOR-AND lookup per step; from scratch, the same variables with different wiring
- [gf2-elimination](entries/gf2-elimination.md): XOR-AND row update; from scratch, three circuit types given enough matrices
- [linear-systems-finite-fields](entries/linear-systems-finite-fields.md): microstep scratchpad learned from scratch; fused updates bypass the hand-built subtractor
- [crc-remainder](entries/crc-remainder.md): shift-XOR step learned; answer only uses a different parity readout

**Symbolic mathematics**
- [polynomial-add-mul](entries/polynomial-add-mul.md): addition learned in 40/40 networks; multiplication never reaches exact match
- [polynomial-differentiation](entries/polynomial-differentiation.md): low-bit multiplier learned; the unseen multiplier bit never is
- [polynomial-integration](entries/polynomial-integration.md): only the scratchpad transfers to withheld fractions
- [chain-rule-differentiation](entries/chain-rule-differentiation.md): solved from scratch with a token-pair front end; frozen-parser setup bypasses the parser
- [symbolic-algebra-atoms](entries/symbolic-algebra-atoms.md): collecting terms learned at every rung; expanding and solving need a scratchpad

**Search and games**
- [grid-bfs](entries/grid-bfs.md): reached-set scratchpad learned from scratch; answer only is exact only up to distance 3
- [dpll-search](entries/dpll-search.md): a trained last MLP completes both hand-built solvers; from scratch the state update fails
- [chess-krk](entries/chess-krk.md): rule router at .993–.999, never exact; trained networks code kings as coordinates
- [chess-mate-in-one](entries/chess-mate-in-one.md): the trained final MLP decodes the hand-built escape count

## Quick start

```
cd code && pip install -r requirements.txt
python run.py check                                                         # the hand-built model and the menus are exact
python run.py atlas --task count_mod5 --seeds 0 --steps 400 --every 100    # smoke run, about a minute
```

The full commands per entry are in [code/README.md](code/README.md).

## Status

The atlas is work in progress. Entries are dated (last runs from 2026-11-04 to 2026-11-08) and state which cells are still open. Five
entries rerun from `code/`; the others publish their networks, test sets and recorded accuracies, and their code is not ported yet.
How the atlas was made (one author, with LLM agents working under a written protocol) is described in
[METHODS.md](METHODS.md#how-this-was-made).

## Cite and license

Cite the atlas via [CITATION.cff](CITATION.cff) (GitHub: "Cite this repository"). Text and networks are released under
[CC BY 4.0](LICENSE) and the code in `code/` under [MIT](code/LICENSE).
