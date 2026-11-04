# Mechanistic Interpretability Circuit Atlas

An encyclopedia of circuits that neural networks use to solve well-defined tasks.

Each entry describes one task and the circuit types that trained networks converge to. It has three parts:

- **Hand-built reference:** a network whose every weight is set by a formula, exact on the task.
- **What training finds:** many seeds, both fully trained from scratch and trained on top of frozen hand-built parts. Each
  entry gives how often each circuit type appears, which frozen parts training chooses to reuse, and how fast it generalizes (grokking).
- **Closed form:** a formula for the trained part. It is accepted only if *substituting* it for the trained weights keeps the accuracy.

Entries are written as encyclopedia articles: the mechanism in plain words, the equations, the evidence (ablations, accuracies,
seed counts) and the limits. Experiments are run elsewhere; this repository holds the results and descriptions.

## Index

| Entry | Task | Circuit type(s) found | Seeds | Closed form verified |
|---|---|---|---|---|
| [modular-addition-mlp](entries/modular-addition-mlp.md) | (x + y) mod 59, MLP on a frozen menu | Fourier "clock" (Gromov MLP solution), 10/10 | 10 | yes, .996–1.000 |

Planned: binary, decimal and hex addition (answer only, carries written, place-value split); base conversion; subtraction;
multiplication; long division; long-hand square root; counting and bracket tasks. See [TEMPLATE.md](TEMPLATE.md) for the entry format.

## License

Text: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
