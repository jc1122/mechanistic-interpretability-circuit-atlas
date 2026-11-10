# Mechanistic Interpretability Circuit Atlas

What circuits do small transformers actually learn on algorithmic tasks, and how do they compare with a circuit written by hand?

For each task the atlas builds an exact **hand-built** network, then trains networks over many seeds, both from scratch and on top of frozen
hand-built parts. It reports which circuit each seed converges to, how often, and whether a closed form of it survives being
**substituted back** into the network. Negative results are kept.
Unlike [Tracr](https://arxiv.org/abs/2301.05062) or [InterpBench](https://arxiv.org/abs/2407.14494), which use known circuits to test interpretability tools,
the atlas asks what gradient descent finds *instead of* the known circuit ([related work](METHODS.md#related-work)).

## Headline results

- **Swapping parts one at a time is not verification.** Single-component substitutions keep accuracy, yet scrubbing the whole proposed
  circuit at once fails in five tasks. A working replacement is a sufficient readout, not a verified mechanism.
- **Same variables, different wiring.** Nets trained from scratch compute the hand-built variables but share only 6–9% of its circuit edges.
- **Failed nets still compute the right variables; they fail to generalise** (981 nets, three arithmetic tasks).
- **A scratchpad makes the circuit up to 10x smaller**, and training reuses the adder's column-sum lookup as the multiplication step.

More, each marked verified or preliminary: [SURPRISES.md](SURPRISES.md).

## Entries

Full table with seed counts and closed-form results: [INDEX.md](INDEX.md). Reading guide: [METHODS.md](METHODS.md#reading-an-entry).

<!-- One line per entry, in its domain group. Stage 10 adds a line for a new entry; the full row goes to INDEX.md. -->
**Counting and modular arithmetic**
- [modular-addition-mlp](entries/modular-addition-mlp.md): Fourier "clock" in 10/10 seeds
- [count-mod-5](entries/count-mod-5.md): flat residue lookup in 30/30
- [compare-counts](entries/compare-counts.md): tilted linear comparator; `=` is only memorised
- [open-bracket-count](entries/open-bracket-count.md): low-frequency clock sum plus a parity channel
- [modular-arithmetic-mod-97](entries/modular-arithmetic-mod-97.md): multiplication by a discrete-log clock; digit formats never form circles

**Arithmetic**
- [decimal-hex-addition](entries/decimal-hex-addition.md): column-sum lookup; trained nets skip the carry-lookahead head
- [binary-addition](entries/binary-addition.md): lookup on a + b + carry; position inputs break longer inputs until masked
- [router-addition](entries/router-addition.md): one shared column step; longer inputs work only when the stage cannot read position
- [subtraction](entries/subtraction.md): per-column lookup on a − b − borrow, not the adder's key
- [multiplication](entries/multiplication.md): the scratchpad step is the adder's lookup; answer only never trains
- [multiplication-base-order](entries/multiplication-base-order.md): LSB-first answers learnable, MSB-first mostly not; a scratchpad fixes both
- [long-division-square-root](entries/long-division-square-root.md): one shared compare-and-subtract step; answer only needs 16–90x more data
- [gcd-euclid](entries/gcd-euclid.md): a learned three-rule router on frozen Euclid parts
- [quadratic-roots](entries/quadratic-roots.md): the register executor is one AND neuron; the parent tasks fail
- [compose-arithmetic-registers](entries/compose-arithmetic-registers.md): compositions learned without the register lookup; the bare primitive fails

**Number formats**
- [base-conversion](entries/base-conversion.md): hex to binary groks as a nibble lookup; carry-type conversions are never learned
- [output-order-base-conversion](entries/output-order-base-conversion.md): MSB-first is a threshold ladder; LSB-first needs a mod readout and mostly fails

**Automata and finite fields**
- [dfa-group-automata](entries/dfa-group-automata.md): transition table learned at once; S3 answer-only uses prefix doubling
- [gf2-matrix-product](entries/gf2-matrix-product.md): XOR-AND lookup; from scratch, same variables with different wiring
- [gf2-elimination](entries/gf2-elimination.md): XOR-AND row update; from scratch, three circuit types given enough data
- [linear-systems-finite-fields](entries/linear-systems-finite-fields.md): microstep scratchpad learned from scratch; fused stages bypass the hand-built subtractor
- [crc-remainder](entries/crc-remainder.md): shift-XOR step learned; answer only uses a parity readout instead

**Symbolic math**
- [polynomial-add-mul](entries/polynomial-add-mul.md): addition learned everywhere; multiplication only with frozen product gates
- [polynomial-differentiation](entries/polynomial-differentiation.md): low-bit multiplier learned, never the unseen bit
- [polynomial-integration](entries/polynomial-integration.md): only the scratchpad transfers to withheld fractions
- [chain-rule-differentiation](entries/chain-rule-differentiation.md): solved from scratch with a token-histogram front end; PART bypasses the frozen parser
- [symbolic-algebra-atoms](entries/symbolic-algebra-atoms.md): collecting terms learned at every rung; expanding and solving need a scratchpad
- [nested-expressions-mod-7](entries/nested-expressions-mod-7.md): a trained argmax residue table on the frozen parser; from-scratch nets fail held-out literals

**Search and games**
- [grid-bfs](entries/grid-bfs.md): reached-set scratchpad learned; answer only limited by depth
- [dpll-search](entries/dpll-search.md): trained MLP completes the hand-built solvers; from scratch the state update is not learned
- [chess-krk](entries/chess-krk.md): rule router at .993–.999, never exact; trained nets code kings as coordinates
- [chess-mate-in-one](entries/chess-mate-in-one.md): final MLP decodes the hand-built escape count

## Quick start

```
cd code && pip install -r requirements.txt
python run.py check                                              # the hand-built model and the menus are exact
python run.py atlas --task count_mod5 --seeds 0 --steps 400 --every 100    # smoke run, about a minute
```

Trained networks: [huggingface.co/jc1122/circuit-atlas-nets](https://huggingface.co/jc1122/circuit-atlas-nets) (safetensors).
How the atlas was made (one author, LLM agents under a written protocol): [METHODS.md](METHODS.md#how-this-was-made).

## License

Text: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code in `code/`: [MIT](code/LICENSE). Networks: CC BY 4.0.
