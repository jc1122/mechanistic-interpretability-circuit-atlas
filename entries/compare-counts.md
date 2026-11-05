# Compare two counts: MLP on a frozen menu

**Task.** The same prompts as [count mod 5](count-mod-5.md) (a shuffled list of a's and b's). Answer `a` if c > L − c, `b` if c < L − c,
`=` if 2c = L (3 classes over 1080 cells (c, L), L = 1..45; 22 cells are `=`). The trained stage must compare c with L/2.
**Status.** 2026-11-04. 30 networks (10 seeds × train fractions .4/.6/.8) + 18 control runs; setup as in count mod 5.

## Setup
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/compare-counts](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/compare-counts) (hand-built base in `_shared/m2`).
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

## Architecture as used
- **Frozen base, hand-built m2** (no training): GPT-NeoX-shaped decoder, 6 layers, d_model 528, 4 heads (d_head 132), MLP width 512, exact
  GELU, fixed-gain LayerNorm (eps 1e-5), no positional embedding and no rotary, sequential residual, character vocabulary of 70 ids.
  10,041,408 parameters, of which 13,443 are non-zero; read at the last prompt position, residual before the final LayerNorm.
  `code/m2.py` rebuilds it bit for bit (safetensors sha256 `3dbb5226…cd9a6a`).
- **Frozen menu** (formula, no parameters learned) on m2's count X = 1/(c+1) and position scalar R (gives L). Blocks, index ranges of the
  326-wide input: su 0 (X), sv 1 (R), ou 2–47 (one-hot c = 0..45), ov 48–93 (one-hot L = 0..45), tu 94–138 ([c ≥ k], k = 1..45),
  tv 139–183 ([L ≥ k]), distractors brackets 184–185 (constant, so 0), copy pointer 186–255, last letter 256–325. Each block centred
  and divided by its rms row norm.
- **Hand-built count (frozen m2, layer 4 head 1).** The query is the question letter (from the has-letter head); every key token equal
  to it, and the newline that ends the counted line, scores 300 (+ a recency term 3 per position of the segment anchor, which selects the
  current `:` segment; centred so that tied keys score exactly 0). Attention is therefore uniform over the c matches + 1 newline. The
  value reads only the newline indicator, so the head writes X = 1/(c+1). Its non-zero entries: 95 in q, 52 in k, 2 in v/output.
- **Menu decode of X** (step pairs, `step(z; c, s) = (gelu(s(z − c) + 8) − gelu(s(z − c) − 8))/16`): [c ≥ k] = step(−X; −m_k, 32/g_k),
  m_k = (1/k + 1/(k+1))/2, g_k = 1/k − 1/(k+1); the one-hot is the difference of neighbouring thermometer units. Exact on all cells.
- **Trained stage:** `logits = W2 ReLU(W1 x + b1) + b2`, W1 [256, 326], b1 [256], W2 [3, 256], b2 [3] (84,483 parameters); classes a, b, `=`.
- **Archive.** Per network one safetensors file (`W1`, `b1`, `W2`, `b2`, `test_idx`), plus `config.json`, `inputs.safetensors` (the frozen
  menu rows X, the integers and the labels) and `manifest.json` (seed, rung, train fraction, accuracy, circuit type, sha256). Every file
  was re-loaded and its forward pass reproduces the recorded test accuracy exactly. Public hosting of the weights is pending. 30 networks (`s<seed>_f<40|60|80>`).

## Prediction before training
No prediction was recorded before these runs (the entry predates this section). Retrospective, labelled as such: the hand-built
comparator (`compare_answer`, below) predicts that `=` is learnable only as the default "neither greater nor less"; the trained stage
gives `=` its own logit instead and never generalizes it (0 correct `=` test cells in 30/30), so the hand-built design did not predict
the trained circuit.

## What training finds
- **Learning curve.** Test accuracy plateaus at the first evaluation (step 500) and never improves: .91–.94 (fraction .4) to .92–.96 (.8).
  Neither grokking nor a clean failure; the weight norm stays flat.
- **Circuit type: linear comparator, tilted boundary, `=` memorized — 30/30.** The a−b logit difference is 83–87% linear in (c, L) with an
  L/c slope ratio of −.56 to −.64 (the exact rule is −.50); errors sit at |2c − L| ≤ 2.
- **Encodings read (fraction .4):** the thermometer of c is the main input (ablating it: .59); one-hot and scalar contribute little.

## Closed form
answer = a if αc + βL + γ > 0 else b (never `=`), fitted per network: accuracy .93–.96 (networks .95–.96), agreement .92–.94.
A form built on the exact variable 2c − L scores lower (.90–.94): the networks do not use 2c − L.

## Weights excerpt and reading
Hand-built comparator (m2 layer 5 MLP, three step pairs, slope 40; gate g = 200·[task flag] + 200·[space token]): `=` unit fires at
g ≥ 300 (writes +`=`); `>` unit at 100(NA − NB) + g ≥ 400.5 (writes +`>`, −`=`); `<` unit symmetric. Equality is what remains when
neither difference unit fires.

Trained linear comparator: least-squares fit of the a − b logit difference to αc + βL + γ over all cells, fraction .4:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| α | 4.43 | 4.47 | 4.27 | 4.40 | 3.84 | 4.22 | 4.19 | 4.17 | 4.12 | 4.31 |
| β | −2.55 | −2.51 | −2.74 | −2.72 | −2.31 | −2.65 | −2.52 | −2.40 | −2.62 | −2.60 |
| β/α | −.58 | −.56 | −.64 | −.62 | −.60 | −.63 | −.60 | −.58 | −.64 | −.60 |

Seed 0 logits (a, b, `=`) at L = 20, all test cells: c = 8: (−1.0, 6.8, −10.1); c = 9: (−6.6, 10.9, −8.3); c = 10 (`=`): (6.4, −0.4,
−9.9); c = 11: (1.6, 0.7, −4.7); c = 12: (2.8, −1.7, −2.9). Margin of the `=` logit over the best other class on `=` cells: +13.7 on
training cells, −15.5 on test cells. W1 column norm per block (seed 0): tu 7.3, ov 7.0, ou 6.4, tv 4.1, sv 2.5, su 2.4.

Reading. The a − b difference is a tilted line: β/α ≈ −.6 instead of the exact −.5, so the boundary drifts away from 2c = L as L grows; the
errors sit at |2c − L| ≤ 2. The `=` logit is large only on the 22 × fraction training cells with 2c = L and very negative
on the held-out ones: it is a per-cell lookup on the one-hot codes, not a function of 2c − L. The hand-built unit avoids this by never
giving `=` a positive detector.

## Comparison
The hand-built comparator makes `=` the default output and subtracts two steps on the difference (greater, less), so equality is
"neither" and is exact on unseen cells. The trained stage instead gives `=` its own logit. On one-hot or thermometer codes, equality is a
non-linear (XOR-like) feature, so it can only memorize it.

## Controls and limits
- Random frozen base: .99–1.000. Its residual encodes c, L and 2c − L linearly (ridge R² 1.000), so a few ReLUs give an exact comparator, `=` included.
- Scalar-only menu: .97–.99, also better than the full menu (`=` test cells .17–.44 correct).
- Hypothesis (untested): exact one-hot codes make `=` and boundary cells memorizable at a lower weight norm than learning the line 2c − L.
