# Mechanistic Interpretability Circuit Atlas

An encyclopedia of circuits that neural networks use to solve well-defined tasks.

Each entry describes one task and the circuit types that trained networks converge to. It has three parts:

- **Hand-built reference:** a network whose every weight is set by a formula, exact on the task.
- **What training finds:** many seeds, both fully trained from scratch and trained on top of frozen hand-built parts. Each
  entry gives how often each circuit type appears, which frozen parts training chooses to reuse, and how fast it generalizes (grokking).
- **Closed form:** a formula for the trained part. It is accepted only if *substituting* it for the trained weights keeps the accuracy.

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

Planned: binary, decimal and hex addition (answer only, carries written, place-value split); base conversion; subtraction;
multiplication; long division; long-hand square root; counting and bracket tasks. See [TEMPLATE.md](TEMPLATE.md) for the entry format.

## Cross-entry findings

Surprising results across entries, including early ones from tasks still running, are collected in [SURPRISES.md](SURPRISES.md).

- Frozen exact encodings can hurt: on compare-counts and open-bracket-count a *random* frozen base (which encodes the integers
  linearly) generalizes better than the exact one-hot/thermometer menu, which invites memorization. Only residue tasks need the exact codes.
- Equality never generalizes as its own learned class; a hand-built comparator that makes `=` the default ("neither greater nor less") is exact.
- Frequency use differs by domain: modular addition spreads over 22–27 frequencies; the non-modular count uses only the lowest 1–3.

## License

Text (entries, METHODS.md, README): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code in `code/`: [MIT](code/LICENSE).
