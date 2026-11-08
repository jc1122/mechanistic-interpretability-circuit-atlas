# Code

This folder holds the code of the atlas under the MIT licence ([LICENSE](LICENSE)). It reruns the five entries below on CPU. The other entries publish their networks, test sets and
recorded accuracies on [Hugging Face](https://huggingface.co/jc1122/circuit-atlas-nets); their code is not ported yet.

## Install

```
pip install -r requirements.txt      # tested with Python 3.12 on CPU
```

## What reruns here

Run from this folder.

| Entry | Command (main cell; Section 2.4 of each entry has the rest) |
|---|---|
| all | `python run.py check`: the hand-built model and the menus are exact |
| [modular-addition-mlp](../entries/modular-addition-mlp.md) | `python run.py modadd --seeds 0 1 2 3 4 5 6 7 8 9` |
| [count-mod-5](../entries/count-mod-5.md) | `python run.py atlas --task count_mod5 --frac 0.4 --seeds 0 1 2 3 4 5 6 7 8 9` |
| [compare-counts](../entries/compare-counts.md) | `python run.py atlas --task compare_counts --frac 0.4 --seeds 0 1 2 3 4 5 6 7 8 9` |
| [open-bracket-count](../entries/open-bracket-count.md) | `python run.py atlas --task open_count --frac 0.8 --seeds 0 1 2 3 4 5 6 7 8 9` |
| [base-conversion](../entries/base-conversion.md) | `python run.py convert check`, then `python run.py convert --variant hex2bin --fmt A --rung part --seeds 0 1 2 3 4 5 6 7 8 9` |

Controls (`--kind random --probe`, `--kind scalar`) and the other fractions or variants are listed in Section 2.4 (Reproduction) of
each entry. Aggregate over seeds: `python run.py table out/<task>_*.json`. Smoke test (about a minute; the numbers are not the entries'):
`python run.py atlas --task count_mod5 --seeds 0 --steps 400 --every 100`.

## Output

Each run writes `out/<name>.json`: learning curve, ablation (usage) table and analysis. Weights are not saved.

## Cost and tolerance

A 200,000-step count network takes about 5 minutes on 4 CPU threads, plus about a minute to build the menu; a full 50,000-step
base-conversion TRAIN network takes on the order of an hour. Reruns match the headline numbers, not bit for bit (the published
networks were trained on other machines, CPU and GPU). Details: [METHODS.md §6](../METHODS.md#6-reproduction-cost-and-tolerances).

## Files

| File | Role |
|---|---|
| `run.py` | command line |
| `m2.py`, `neox.py` | the frozen hand-built 6-layer model and its 40-line GPT-NeoX forward pass |
| `menu.py`, `atlas.py` | frozen menus and task prompts for the m2-based entries |
| `train.py` | the single trained stage |
| `analyze_atlas.py`, `analyze_modadd.py` | circuit typing, substitution and probes |
| `convert.py` | base conversion: hand-built 2-layer networks, PART / TRAIN trainer, substitution |
