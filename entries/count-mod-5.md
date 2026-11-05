# Count mod 5: MLP on a frozen menu

**Task.** Prompt `COUNT b: a b\nANSWER: 1\nCOUNT a: <c a's and L−c b's, shuffled>\nANSWER: `, answer c mod 5 (5 classes). Cells (c, L)
with L = 1..45, c = 0..L (1080 cells, one prompt each); the train/test split is over cells, so every c value appears in training.
**Status.** 2026-11-04. 30 networks (10 seeds × train fractions .4/.6/.8) + 18 control runs. Trained part: one MLP layer, 256 ReLU,
AdamW lr 1e-3, weight decay 1.0, full batch, 200k steps.

## Summary
Every network learns a flat residue lookup on the count c and ignores the list length L. A two-cosine closed form with equal amplitudes
(equivalently, a delta on the residue) reproduces it exactly (1.000). There is no grokking delay: test accuracy equals train accuracy from
the first evaluation.

## Setup
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/count-mod-5](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/count-mod-5) (hand-built base in `_shared/m2`).
Full architecture, optimiser and instrumentation: [METHODS.md](../METHODS.md).
A frozen hand-built transformer (6 layers, every weight set by formula) computes the count c (a segment-match attention head gives
1/(c+1), an MLP step ladder decodes it) and the line length L. A frozen menu places three encodings of each integer at the answer
position, each centered and scaled to rms 1: the raw scalar, a one-hot and a thermometer, plus distractor features. Only the MLP on top is trained.

Menu width 326; hidden width 256 ReLU, 5 outputs; 200,000 steps, evaluation every 500 steps, no early stop; 10 seeds at each train fraction
.4 / .6 / .8; controls (random base, scalar-only menu) 3 seeds at each fraction.

**Reproduce** (from `code/`):
```
python run.py atlas --task count_mod5 --frac 0.4 --seeds 0 1 2 3 4 5 6 7 8 9     # repeat with --frac 0.6 and 0.8
python run.py atlas --task count_mod5 --kind random --seeds 0 1 2 --probe        # random-base control with ridge probes
python run.py atlas --task count_mod5 --kind scalar --seeds 0 1 2
python run.py table out/count_mod5_*.json                                         # min-max over seeds, as quoted here
```
Headline per network: `final` (test accuracy), `form_res5` and `form_u_only` (substitution accuracy), `form_fourier` / `fourier_r2`.

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
- **Trained stage:** `logits = W2 ReLU(W1 x + b1) + b2`, W1 [256, 326], b1 [256], W2 [5, 256], b2 [5] (84,997 parameters).
- **Archive.** Per network one safetensors file (`W1`, `b1`, `W2`, `b2`, `test_idx`), plus `config.json`, `inputs.safetensors` (the frozen
  menu rows X, the integers and the labels) and `manifest.json` (seed, rung, train fraction, accuracy, circuit type, sha256). Every file
  was re-loaded and its forward pass reproduces the recorded test accuracy exactly. Public hosting of the weights is pending. 30 networks (`s<seed>_f<40|60|80>`).

## Prediction before training
No prediction was recorded before these runs (the entry predates this section). Retrospective, labelled as such: the hand-built
decoder already turns X into exact one-hot and thermometer codes, so the cheapest solution is a linear pooling of the one-hot by
residue, with L ignored. This held in 30/30 networks. Not anticipated: that the thermometer alone also carries most of it (.86).

## What training finds
- **Learning curve.** Train 99% and test 99% both at step 500 for every fraction (final test .992–1.000). Not grokking.
- **Circuit type: residue lookup on c, L ignored — 30/30.** Setting the L input to its mean keeps .995–1.000; replacing each c embedding by
  its residue-class mean keeps 1.000.
- **Encodings read (fraction .4, test after ablation):** all c features ablated .16, all L features ablated .997; one-hot pair kept alone .997,
  thermometer alone .86, scalars alone .21. The one-hot is the primary input; the information is redundant across one-hot and thermometer.

## Closed form
logit_r(c) = A cos(2π(c − r)/5) + B cos(4π(c − r)/5), fitted per network: R² .989–.999, accuracy 1.000 in every network.
A ≈ B (e.g. 6.77 / 6.68); with A = B the profile is a delta (2 at c ≡ r, −1/2 elsewhere): a flat lookup, not a Fourier mechanism.

## Weights excerpt and reading
Seed 0, fraction .4: the network's logits (centred over the 5 classes, averaged over L) for c = 0..9 (c = 10..45 repeat the pattern):

| c | r = 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | **14.25** | −2.59 | −3.42 | −3.93 | −4.30 |
| 1 | −2.95 | **13.56** | −3.30 | −3.42 | −3.88 |
| 2 | −3.35 | −3.10 | **13.35** | −3.12 | −3.78 |
| 3 | −3.57 | −3.33 | −3.31 | **13.83** | −3.63 |
| 4 | −3.50 | −3.40 | −3.28 | −3.22 | **13.40** |
| 5 | **13.40** | −3.29 | −3.39 | −3.34 | −3.38 |
| 9 | −3.57 | −3.63 | −3.42 | −3.34 | **13.97** |

Two-cosine fit (A, B) of the logits, seeds 0–9 (fraction .4): A = 6.73–6.77, B = 6.61–6.68 (seed 0: 6.77, 6.68).
W1 column norm per input block (seed 0): ou 14.6, tu 6.5, su 4.2; ov 1.9, tv 4.0, sv 3.1; copy 3.3, last 2.9.

Reading. With A = B the closed form gives 2A ≈ 13.5 at r = c mod 5 and A(cos 72° + cos 144°) = −A/2 ≈ −3.4 elsewhere, which is the table:
a flat, equal-height delta on the residue, identical for c and c + 5. The first layer's largest weights sit on the one-hot of c (ou), so
each count value is mapped by its own column to the hidden pattern of its residue class; the L blocks get small weights whose effect
the network averages out (setting L to its mean keeps .995–1.000).

## Comparison
The hand-built count decoder is a step ladder on 1/(c+1), the same construction as the menu's one-hot and thermometer. The trained stage
reads the one-hot output and pools it by residue: one more linear map. No hand-built mod brick was given.

## Controls and limits
- Random frozen base (same shape): .24–.40 test. Its residual holds c linearly (ridge R² 1.000) but carries no residue feature.
- Scalar-only menu: .29–.34 (chance .2): 1/(c+1) cannot be cut into residues by 256 ReLUs at this weight decay.
- Open: holding out c values (not cells) to test period extrapolation was not run; why one-hot beats thermometer when both are exact.
