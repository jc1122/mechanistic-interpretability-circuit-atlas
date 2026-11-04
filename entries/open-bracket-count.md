# Open-bracket count: MLP on a frozen menu

**Task.** Prompt `VALID_BRACKETS: ()\nANSWER: YES\nVALID_BRACKETS: <line>\nANSWER: `, answer the number of `(` = (L + b)/2 (46 classes),
where b is the final depth (#`(` − #`)`) and L the length, L = 1..45, b = −L..L in steps of 2 (1080 cells, one shuffled line each).
**Status.** 2026-11-04. 30 networks (10 seeds × train fractions .4/.6/.8) + 18 control runs; setup as in [count mod 5](count-mod-5.md),
with the frozen network supplying the depth ratio b/(L+1) and L.

## Setup
Frozen m2 and menu as in [count mod 5](count-mod-5.md) but on the bracket prompt: u = b (depth, from the bracket head's b/(L+1)), v = L; menu
width 416 (the b thermometer is a second step stage gated by the one-hot of L). Hidden width 256, 46 outputs, 200,000 steps, evaluation
every 500 steps, 10 seeds at train fractions .4 / .6 / .8, controls 3 seeds each; see [METHODS.md](../METHODS.md).

**Reproduce** (from `code/`; the clock sum appears at fraction .8):
```
python run.py atlas --task open_count --frac 0.8 --seeds 0 1 2 3 4 5 6 7 8 9    # also --frac 0.4 and 0.6
python run.py atlas --task open_count --kind random --seeds 0 1 2 --probe
python run.py atlas --task open_count --kind scalar --seeds 0 1 2
python run.py table out/open_count_*.json
```
Headline per network: `final`, `t50`, `form_clock` / `form_clock_agree`, `clock_k_top`.

## Summary
The only task in this set that groks, and only with 80% of cells in training: 10/10 networks build a low-frequency trigonometric sum
(frequencies k = 1–3 of 92), a Gromov-style clock on a non-modular domain. With 40% or 60% training data all networks memorize.

## What training finds
- **Learning curve.** Train 100% at step 500 everywhere. Fraction .4: final test .006–.019 (below chance; wrong answers are neighbouring
  cells' answers). Fraction .6: .04–.11. Fraction .8: test .02–.05 at 10k steps, 50% at 20.5k–28.5k, then flat at .78–.88 from ~150k while
  the weight norm falls 80 → 73. None reaches 90%.
- **Circuit types.** Low-frequency clock sum: 10/10 at fraction .8. Memorization: 20/20 at .4 and .6.
- **Encodings read (fraction .8):** all three encodings are used together, the one-hot as the main input; distractors have no effect.

## Closed form
pre_n = c_n + a_n cos(ω_n b/2 + φ_n) + a′_n cos(ω_n L/2 + ψ_n), ω_n = 2πk_n/92, k_n ∈ {1, 2, 3}, with the trained output weights:
accuracy .91–.93 (networks .96–.98), agreement .88–.92. The pre-activation separates into a b part plus an L part (substitution
.95–.98) but is not linear (a straight-line form scores .09–.11). Wavelengths of 31–92 exceed half the domain, so the cosines do not alias.

## Comparison
No hand-built open-count brick was given. The nearest hand-built design forms the sum linearly in the residual and reads it with step
staircases; the trained stage never forms the sum linearly and builds a trigonometric sum instead. Modular addition (x + y mod 59) used
22–27 frequencies; this non-modular task uses only k = 1–3.

## Controls and limits
- Random frozen base: .57–.60 / .72–.77 / .81–.86 (fractions .4 / .6 / .8), better than the menu at .4 and .6. Its residual holds b and L
  linearly (ridge R² 1.000).
- Scalar-only menu: .35–.56.
- Open: is a 90% clock reachable with more data or lower weight decay (more steps do not help: accuracy and norm are flat after ~150k)?

## References
- Gromov, A. (2023). Grokking modular arithmetic. https://doi.org/10.48550/arxiv.2301.02679
