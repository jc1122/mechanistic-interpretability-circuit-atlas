# Modular arithmetic mod 97 in six number formats

> - **Task.** Compute a + b, a·b or x² modulo 97 when the numbers are written as residue tokens, binary, decimal, hex or a decimal scratchpad. Example: (96 + 2) mod 97 = 1.
> - **Hand-built net.** Residue tokens: addition by a Fourier pizza at frequencies 17, 25, 32, 43; multiplication by adding base-5 discrete logs and looking up 5^(la+lb mod 96). Digit formats: fetch digits, place-value sums, compare-subtract. 17 nets, all exact on 9409 inputs.
> - **Trained nets.** Token multiplication at fraction .9: embeddings are circles in discrete-log order in 10/10 seeds (K-high clock 8, K-low ramp-like clock 2), 6/10 reach ≥ .99 test; token addition is a Fourier clock in 10/10. Binary multiplication and hex squaring memorise (test .013–.077).
> - **Verdict.** Partly solved (token multiplication 6/10 seeds ≥ .99 at fraction .9, 0/10 at .3; digit multiplication memorises). Closed form: partial (output-level only: substitution keeps accuracy ≥ .98 on all 9216 nonzero pairs in 8/10 token-multiplication seeds).
> - **Links.** [Weights](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/modular-arithmetic-formats) · Reproduce: not available yet · [Terms](../GLOSSARY.md)

## Task and status
**Task.** a+b, a·b (a, b in 0..96), x² (x in 0..9408) mod 97, all 9409 inputs; formats `tok` (one token per residue), binary LSB/MSB first
(`b2l`/`b2m`), decimal `b10`, hex `b16`, decimal scratchpad `s10` (S ; S div 97 ; r). Answer r = S mod 97, exact match.
**Formats.**
- `tok`: one token per residue.
- `b2l` / `b2m`: binary, least / most significant bit first.
- `b10`, `b16`: decimal, hexadecimal digits.
- `s10`: decimal scratchpad (S ; S div 97 ; r).

**Status.** 2026-11-08, after the final run (10 seeds: mul tok x3 = ENTRY cell, add tok x3, add b16, add s10, mul b2l, sq b16).
## Summary
No binary/decimal/hex net trained from scratch (TRAIN) carries a circular code of a or r; only token formats do. Token multiplication (frac .9) embeds residues as
circles in discrete-log order in 10/10 seeds (K-high multi-frequency clock 8, K-low ramp-like clock 2; 6/10 >= .99 test). Its output is a
<= 6-frequency clock in d = log a + log b - log c (substitution keeps accuracy on all 8 seeds >= .98). Digit multiplication memorises.
## Setup
[METHODS.md](../METHODS.md). AdamW lr 1e-3, wd 1.0 (x3wd .1), betas (.9, .98), warmup 1000, cosine over 40k, clip 1, batch 512, TF32; early
stop at test 1.0 or a 5k plateau; best-training and final snapshots, checkpoints every 2500. Split per seed: frac .3 (2823 train) or .9
(8468 train, 941 test). Consumer/rented GPUs; diagnostics and substitution on CPU in fp32.
**Reproduce:** training code for this task is not yet in `code/`; the weights, held-out splits and recorded accuracies are on Hugging Face (link above).
## Architecture as used
TRAIN: GPT-NeoX 4 layers x 4 heads (d_head 64), d_model 256, MLP 512 ReLU, no LayerNorm, NoPE, untied; vocabulary 103 (BOS, 97 residues/
digits, + * ^ = ;); all parameters trained. HAND: 8 x 8, d 512, same vocabulary, every weight by formula. Archive `modular-arithmetic-formats/`:
hand_{mul,add}_tok, train_{mul,add}_tok_f90_{best,final}_s<seed> (mul s0,1,2,4,7,9; add s0-2), split_{mul,add}_tok (perm, ntrain, seeds),
config.json, manifest.json (seed, rung, circuit type, fp32 accuracy, sha256). Networks: huggingface.co/jc1122/circuit-atlas-nets/tree/main/modular-arithmetic-formats; earlier round: huggingface.co/jc1122/circuit-atlas-nets/tree/main/modular-arithmetic-mod-97.
## Prediction before training
Recorded before training: tok add Fourier, tok mul Fourier in log coordinates, no p-periodic code in digit formats, s10 beats b10. Held, except
that frac .3 was below the grokking data threshold for tok add (3/10) and tok mul (0/10); 3x data fixed both.
## Hand-built reference
tok add: pizza (Zhong 2023), frequencies 17, 25, 32, 43. tok mul: discrete-log adder: embedding log_5 v and [v = 0], one averaging head, MLP
lookup 5^(la+lb mod 96). Digit formats: index heads fetch digits, place-value sums; add: lookup on A+B; mul: thermometer partial products and
compare-subtract [P >= 97j] -> q, r; sq: compare-subtract on x, square lookup. All 17 exact on all 9409 inputs (logit margin >= 23.6).
## What training finds
| cell (final, 10 seeds) | exact / >= .99 | test range | circuit types (seeds) |
|---|---|---|---|
| mul tok x3 | 4 / 6 | .425-1 | K-high log clock 8 (s0,2-8), K-low ramp-clock 2 (s1,s9) |
| add tok x3 | 6 / 10 | .998-1 | Fourier clock on Z_97 10 |
| add b16 | 0 / 9 | .990-.999 | r read at L4 (DAS .47-.72): r-only 9, S+r 1 |
| add s10 | 7 / 9 | .988-1 | q-transfer route (3/3 measured) |
| mul b2l, sq b16 | 0 / 0 | .013-.077 | memorised (failed); 3-seed rounds: mul b10/b16 memorise at every data/wd |
mul tok te90 21-30.5k (6 seeds); no HAND variable reaches probe .9 at any checkpoint. Length generalisation: not defined (fixed domain).
Diagnostics (final snapshot, ranges over seeds; L0 = embedding; neuron counts unmeasured, whole MLPs only):
| cell | used heads + MLPs | best probe R² per HAND var @L | eff. rank (PR) vs HAND | attn JS | best / final test |
|---|---|---|---|---|---|
| mul tok x3 | 11-16 h + 4 M | A/B/S/q/r <= .12; la .62, lb .40, ls .78 (K-low) @3-4 | U vs 1.01 | .17-.37 | .425-1 / .425-1 |
| add tok x3 | 13-16 h + 4 M | A, B, S, q <= .04; r <= .31 @4 | 4.1-8.7 vs 4.41 | .26-.28 | .997-1 / .998-1 |
| add b16 | 9-16 h + 4 M | S .99, q .93, r .93 @4; A .88, B .80 | 3.1-6.4 vs 4.02 | .15-.26 | .993-.999 / .990-.999 |
| add s10 | 7-8 h + 4 M | S .86, q .89, r .57 @4 | 5.0-7.1 vs 7.04 | .18-.24 | .997-1 / .988-1 |
| mul b16 (r1/x3) | 0-4 h | A .80, B .72, S .80 @4; r .49 @0 (chance) | 37-43 vs 1 | .19-.26 | .02-.25 |
## Closed form
mul tok (both types): logit(c) = sum_{k in K} alpha_k cos(2 pi k d/96) + beta_k sin(2 pi k d/96), d = (log5 a + log5 b - log5 c) mod 96, c != 0;
answer 0 iff a or b = 0; K = top-m frequencies of the net's mean logit vs d, alpha, beta fit on train rows. Substituted for the network output:
accuracy kept on all 9216 nonzero pairs for every seed >= .98 (smallest m: s4 1, s5 1, s2 2, s8 2, s9 3, s0/s1/s3 6); agreement with the net =
its accuracy (>= .99 on the six >= .99 seeds); s6 (.838) agree .837, s7 none. Output-level only (true discrete log); internally MLP3 = lookup
T[ls] keeps accuracy in s0,2,4,5,6,9. add tok: the same form on Z_97 with d = (a + b - c) mod 97; substitution not yet measured.
## Weights excerpt and reading
Clock frequencies K and amplitudes sqrt(alpha²+beta²) per seed (final, clock_fit): s4 k 43 amp 23.5; s5 7 39.2; s2 38, 29 16.5, 8.3; s8 17,
16 9.5, 6.4; s9 6, 2, 9 28.8, 9.9, 9.3; s1 6, 4, 8, 2, 10, 1 52.4 ... 7.4. Embedding top log-order frequencies: `excerpts.py modp` (DFT over
residues sorted by log_5, power share). Reading: each seed rotates log a and log b on a few circles and reads cos(k(la + lb - lc)).
## Comparison
HAND tok mul and TRAIN share the isomorphism Z_97* = Z_96 but not the code: ramp + lookup vs circles (the Nanda 2023 / Zhong 2023 clock moved
by the discrete log). Digit formats never form circles; solved digit nets read r out as a categorical code at L4.
## Controls and limits
Fourier label checked on all 17 HANDs (tok add F, other 16 L). The closed form is output-level with the oracle log; seeds differ in weights
(lmc barrier 12-19). r1 add b16/s10 nets were lost (log numbers only); final-run nets exist. Neuron counts and TRAIN align are unmeasured.
## References
Nanda et al. 2023, https://doi.org/10.48550/arxiv.2301.05217; Zhong et al. 2023, https://doi.org/10.48550/arxiv.2306.17844; Gromov 2023,
https://doi.org/10.48550/arxiv.2301.02679; Howe 2026, https://doi.org/10.48550/arxiv.2607.04333 (preprints, verified via scite).
