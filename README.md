# Mechanistic Interpretability Circuit Atlas

An encyclopedia of circuits that neural networks use to solve well-defined tasks.

Each entry describes one task and the circuit types that trained networks converge to. It has three parts:

- **Hand-built reference:** a network whose every weight is set by a formula, exact on the task.
- **What training finds:** many seeds, both fully trained from scratch and trained on top of frozen hand-built parts. Each
  entry gives how often each circuit type appears, which frozen parts training chooses to reuse, and how fast it generalizes (grokking).
- **Closed form:** a formula for the trained part. It is accepted only if *substituting* it for the trained weights keeps the accuracy.

Entries call the three network settings **HAND** (hand-built), **PART** (frozen hand-built parts plus one trained stage) and **TRAIN** (trained from scratch).

**Weights:** the trained networks and frozen inputs behind each entry are on Hugging Face: [https://huggingface.co/jc1122/circuit-atlas-nets](https://huggingface.co/jc1122/circuit-atlas-nets) (safetensors, CC BY 4.0).

Entries are written as encyclopedia articles: the mechanism in plain words, the equations, the evidence (ablations, accuracies,
seed counts) and the limits. [METHODS.md](METHODS.md) describes the architecture, the experimental conditions and the instrumentation common to all entries, and
[code/](code/) is a self-contained CPU reimplementation (Python, torch) that rebuilds the hand-built network, generates the synthetic data,
trains and runs the analysis; every entry has a Setup section with the command that reproduces it.

## Quick start

```
cd code && pip install -r requirements.txt
python run.py check                                              # the hand-built model and the menus are exact
python run.py atlas --task count_mod5 --seeds 0 --steps 400 --every 100    # smoke run (about a minute); drop --steps/--every for the full 200,000 steps
```

## Index

| Entry | Task | Circuit type(s) found | Seeds | Closed form verified |
|---|---|---|---|---|
| [modular-addition-mlp](entries/modular-addition-mlp.md) | (x + y) mod 59, MLP on a frozen menu | Fourier "clock" (Gromov MLP solution), 10/10 | 10 | yes, .996–1.000 |
| [count-mod-5](entries/count-mod-5.md) | count of a letter mod 5, MLP on a frozen menu | flat residue lookup, 30/30 | 30 | yes, 1.000 |
| [compare-counts](entries/compare-counts.md) | which of two letters is more frequent | tilted linear comparator, `=` memorized, 30/30 | 30 | partial, .93–.96 (no `=`) |
| [open-bracket-count](entries/open-bracket-count.md) | number of `(` in a bracket line | low-frequency clock sum (k = 1–3) + parity channel at 80% data, 10/10; memorization below | 30 | yes, .91–.93 |
| [base-conversion](entries/base-conversion.md) | hex to binary, binary to decimal, decimal to binary (x = 0..255), answer only and scratchpad; 2-layer net | hex to binary: nibble one-hot lookup, 38/40 TRAIN nets (grokks from scratch in 16/40); decimal to binary A (frozen heads): remainder threshold, 10/10; carry-type cells never learned | 10 per setting | yes, hex to binary 1.00 in 36/40; thresholds off by <= .06 |
| [decimal-hex-addition](entries/decimal-hex-addition.md) | a + b, 3 digits, base 10 and 16; answer only, carry token, column sums; 4-layer net | column-sum lookup f(X1 + X2 + carry) in 60/60 PART nets (retry runs); TRAIN solves 37/40 with 40k steps, with no carry-lookahead head | 10 per setting | yes, .95–1.00 (PART) |
| [binary-addition](entries/binary-addition.md) | a + b, 6 bits; answer only, carry token, column sums; 4-layer net | one-head carry lookahead by hand; every working trained stage is a lookup on a + b + carry; position inputs break length until masked (column sums exact at 12 bits, 10/10); TRAIN solves all three formats in the final runs but fails at 12 bits | 10 per setting | yes, 1.00 (A, C), 3/10 to 6/10 (B) |
| [subtraction](entries/subtraction.md) | a − b (a ≥ b), 3 digits base 10 and 6 bits base 2; answer only, borrow token, signed column differences; 4-layer net | per-column lookup on a − b − borrow (not the adder's a + b + c) in all 60 PART nets but one; TRAIN solves all six cells after one retry, same key in the last MLP in 84/100 nets; position shortcut breaks length unless masked | 10 per setting | yes, = net accuracy (PART .97–1.00) |
| [multiplication](entries/multiplication.md) | a · b, 6 bits; answer only and running-sum scratchpad; 4-layer net | scratchpad step = the adder's sum lookup on x + b_j a + carry (PART 10/10, TRAIN 5/10, closed form 8/10); answer only never trains (parity staircase not fitted, TRAIN memorises) | 10 per setting | yes, 1.00 (scratchpad PART) |
| [long-division-square-root](entries/long-division-square-root.md) | binary N / D and long-hand square root of N, 8-bit and wide ranges; answer only and per-step scratchpad; 3-layer net | scratchpad step = one shared compare-and-subtract [x ≥ T], x − bit·T (division B PART 10/10 exact, square root B .985–.992); answer-only PART never places the thresholds (≤ .29); TRAIN generalises only with 16–90x more data (division A 10/10 ≥ .99) | 10 per setting | yes, 1.00 (scratchpad PART; nets 1.00 / .985–.992) |
| [dfa-group-automata](entries/dfa-group-automata.md) | state of a parity, Z3 or S3 automaton after a word of 1–16 symbols; answer only and per-symbol state scratchpad; 6-layer net | scratchpad: transition table δ(q_prev, s) learned at once (PART and TRAIN 30/30 per format), length set by the previous-state fetch (PART n 128, TRAIN ~2x); answer-only: S3 PART = hand-built prefix-doubling tables (10/10); abelian stage f(S) misses large S; TRAIN count → residue in range (parity, Z3 10/10 with fresh words), S3 3/10 | 10 per setting | yes, δ table 1.00 (B); f(S) tables = net accuracy (A) |
| [gf2-matrix-product](entries/gf2-matrix-product.md) | C = A B over GF(2), 3×3 bit matrices; answer only and running-parity trace; 4-layer net | trace step = 3-bit XOR-AND lookup on (s_prev, A_ij, B_jk): PART reads the hand-built fetch heads (10/10); TRAIN trace 10/10 computes the same variables with fetch heads one layer later, different wiring; answer only 10/10 (13–14 heads); no length generalisation | 10 per setting | yes, 1.00 (trace PART) |

Planned: counting and bracket tasks. See [TEMPLATE.md](TEMPLATE.md) for the entry format.

## Cross-entry findings

Surprising results across entries, including early ones from tasks still running, are collected in [SURPRISES.md](SURPRISES.md).

- Frozen exact encodings can hurt: on compare-counts and open-bracket-count a *random* frozen base (which encodes the integers
  linearly) generalizes better than the exact one-hot/thermometer menu, which invites memorization. Only residue tasks need the exact codes.
- Equality never generalizes as its own learned class; a hand-built comparator that makes `=` the default ("neither greater nor less") is exact.
- Frequency use differs by domain: modular addition spreads over 22–27 frequencies; the non-modular count uses only the lowest 1–3.
- Circuit probes ("Circuit found in trained nets" sections): component substitutions on trained nets usually preserve accuracy one
  at a time, but scrubbing the whole proposed circuit at once fails in every probed transformer task, and greedily pruned circuits are
  rarely 99%-faithful on fresh inputs: a working replacement is a sufficient readout, not a verified mechanism.

## License

Text (entries, METHODS.md, README): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code in `code/`: [MIT](code/LICENSE).
