# Compare two counts: MLP on a frozen menu

**Task.** The same prompts as [count mod 5](count-mod-5.md) (a shuffled list of a's and b's). Answer `a` if c > L − c, `b` if c < L − c,
`=` if 2c = L (3 classes over 1080 cells (c, L), L = 1..45; 22 cells are `=`). The trained stage must compare c with L/2.
**Status.** 2026-11-04. 30 networks (10 seeds × train fractions .4/.6/.8) + 18 control runs; setup as in count mod 5.

## Setup
Same prompts, menu (width 326) and trained stage as [count mod 5](count-mod-5.md), with 3 outputs (`a`, `b`, `=`); see
[METHODS.md](../METHODS.md). Hidden width 256, 200,000 steps, evaluation every 500 steps, 10 seeds at train fractions .4 / .6 / .8,
controls 3 seeds each. Label rule: class 0 if 2c > L, 1 if 2c < L, 2 (`=`) if 2c = L.

**Reproduce** (from `code/`):
```
python run.py atlas --task compare_counts --frac 0.4 --seeds 0 1 2 3 4 5 6 7 8 9    # repeat with --frac 0.6 and 0.8
python run.py atlas --task compare_counts --kind random --seeds 0 1 2 --probe
python run.py atlas --task compare_counts --kind scalar --seeds 0 1 2
python run.py table out/compare_counts_*.json
```
Headline per network: `final`, `eq_test_acc` (test accuracy on `=` cells), `cmp_ratio` and `cmp_r2` (the linear comparator), `form_line` / `form_line_agree`.

## Summary
All 30 networks learn a linear comparator with a slightly wrong slope and memorize the `=` class, which never generalizes (0 correct
`=` test cells in every network). A random frozen base does better than the exact hand-built menu (.99–1.0 vs .91–.96).

## What training finds
- **Learning curve.** Test accuracy plateaus at the first evaluation (step 500) and never improves: .91–.94 (fraction .4) to .92–.96 (.8).
  Neither grokking nor a clean failure; the weight norm stays flat.
- **Circuit type: linear comparator, tilted boundary, `=` memorized — 30/30.** The a−b logit difference is 83–87% linear in (c, L) with an
  L/c slope ratio of −.56 to −.64 (the exact rule is −.50); errors sit at |2c − L| ≤ 2.
- **Encodings read (fraction .4):** the thermometer of c is the main input (ablating it: .59); one-hot and scalar contribute little.

## Closed form
answer = a if αc + βL + γ > 0 else b (never `=`), fitted per network: accuracy .93–.96 (networks .95–.96), agreement .92–.94.
A form built on the exact variable 2c − L scores lower (.90–.94): the networks do not use 2c − L.

## Comparison
The hand-built comparator makes `=` the default output and subtracts two steps on the difference (greater, less), so equality is
"neither" and is exact on unseen cells. The trained stage instead gives `=` its own logit. On one-hot or thermometer codes, equality is a
non-linear (XOR-like) feature, so it can only memorize it.

## Controls and limits
- Random frozen base: .99–1.000. Its residual encodes c, L and 2c − L linearly (ridge R² 1.000), so a few ReLUs give an exact comparator, `=` included.
- Scalar-only menu: .97–.99, also better than the full menu (`=` test cells .25–.44 correct).
- Hypothesis (untested): exact one-hot codes make `=` and boundary cells memorizable at a lower weight norm than learning the line 2c − L.
