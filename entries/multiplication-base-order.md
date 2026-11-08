# Multiplication: digit order, number base, sub-step curriculum and a scratchpad
**Task.** `x*y=` -> product, x, y in [0, 256), base 2 (8+8 -> 16 digits), 10 (3+3 -> 5) or 16 (2+2 -> 4), zero-padded; answer-only (AO) MSB- or LSB-first, or scratchpad (SP): sums S_j = x (y mod
B^(j+1)) LSB-first + `;`, then the product MSB-first. Test: 2000 held-out pairs; training draws fresh pairs from the other 63,536; exact = every output token. **Status.** 2026-11-08; 4 layers, d 384,
30k steps; AO 6 cells x 4 mixes x 5 seeds (120), retries R1 2x width (45), D3 3x data (25), aux (12); SP 18; final cell (binary LSB, anneal) 10 seeds.
## Summary
Digit order decides AO learnability: LSB-first solves base 10 (20/20) and base 2 (17/20; final cell 10/10), MSB-first 1/60. Width rescues decimal/hex MSB (7/10, 6/10), never binary MSB (0/26 over
width, data, aux). A scratchpad solves every MSB cell at 1x (18/18). Four circuit types (one per base, plus SP) read column variables at MLP3; closed form only for the SP Fourier readout (dec 6/6, hex 3/6).
## Setup
[METHODS.md](../METHODS.md). Mix f: a sample is a sub-problem with prob. f (digit x digit, y with one / two nonzero digits), else uniform; f = 0, .3, .6, anneal (.6 -> 0 by 15k). AdamW lr 1e-3, wd .1,
betas (.9, .98), warmup 1k, cosine to 30k, clip 1, batch 256, TF32, eval every 500; solved = best test >= .99. Final cell: 10 seeds, same recipe, plateau stop. Networks: huggingface.co/jc1122/circuit-atlas-nets/tree/main/multiplication-base-order
## Architecture as used
GPT-NeoX, 4 layers, d_model 384 (12 heads x 32), MLP 256 ReLU (W_in [256, 384], whole residual), no LayerNorm, no positions, untied; vocab 19 (BOS, `*`, `=`, digits 0-15; SP 20 with `;`); all trained.
R1 d 768 (12 x 64); D3 batch 768; aux = running-sum probe loss at resid_3. Files `<cell>_s<seed>[_final].safetensors`, config.json, manifest.json (sha256, fp32
re-score; round trip passes for all 43 nets incl. SP and final cell; HF copy re-downloaded, sha256-checked, 2 nets re-scored).
## Prediction before training
SPEC.md: b16/b10 solve, b2 MSB fails: half right (hex fails at 1x); LSB helps every base: right; curriculum speeds base 2/10, rescues nothing: right (anneal > .3, not ~); width does not rescue b2 MSB:
right. SP (MUL_STUDY): b10/b16 >= 2/3 per mix, faster than 2x: right; b2 >= 2/6: right, slower than b10/16: wrong. Final cell >= 9/10: right (10/10).
## Hand-built reference
None (explicit exception; no HAND/PART rung). Probed: digit p_k, column sum col_k = sum_{i+j=k} x_i y_j, carry_k, low_k / high_k (product mod/div B^k); SP lines: the same for S_j = S_{j-1} + x y_j B^j.
## What training finds
| base | order | f 0 | .3 | .6 | anneal | R1 2x width | D3 3x data | SP 1x f0 / anneal (first .99) |
|---|---|---|---|---|---|---|---|---|
| 2 | LSB | 3/5 | 4/5 | 5/5 | 5/5; final 10/10 | - | - | - |
| 16 | LSB | 1/5 | 1/5 | 2/5 | 0/5 | anneal 5/5 | - | - |
| 2 | MSB | 0/5 | 0/5 | 0/5 | 0/5 | f0 0/5, a 0/5 (.91-.96) | f0 0/5 (.90-.93) | 3/3 / 3/3 (8.5-10.5k) |
| 10 | MSB | 0/5 | 0/5 | 0/5 | 1/5 | f0 3/5, f.3 4/5 | f0 1/5, a 2/5 | 3/3 / 3/3 (11.5-16k) |
| 16 | MSB | 0/5 | 0/5 | 0/5 | 0/5 | f0 5/5, a 1/5 | f0 0/5, a 0/5 (.97-.99) | 3/3 / 3/3 (17.5-21.5k) |
b10 LSB 5/5 in every mix. Aux: b2 MSB 0/6 (.90-.98), b10 MSB 0/6 (.87-.98). Final cell: test .5 at 6-10k, .99 at 16-23.5k (median 18.75k), stop 28.5k, best = final = 1.0 (10/10); train ~ test
(fresh pairs). AO MSB retries grok at 25-28k of 30k (budget-limited); b2 MSB under-fits (train .95-.999), b10/b16 MSB at 1x memorise (train 1.0). SP best = final = 1.0.
Circuit types (per-seed flip at L4 / DAS, non-overlapping): A linear column/carry, binary (b2 LSB 17/17; final cell 10/10, all diagnosed): col_k flip .99-1.0, carry_k .87-1.0. B non-linear column, decimal
(b10 LSB 20/20, R1 1/1): col_k flip .16-.26, DAS .99-1.0. C per-digit table, hex (b16 LSB 4/4, R1 5/5): col_k flip .03-.15, DAS .21-.28, carry_k flip .99-1.0. D SP scratch-line column (b16 SP 6/6,
b10 SP 4/4 diagnosed; b2 SP undiagnosed): col_k flip 1.0 at the scratch queries. Unsolved AO nets carry their base's type.
Diagnostics (best snapshots, seed ranges; var = inter / DAS flip at L4, u untested; R^2 = best probe col/carry (layer); PR = L4 effective rank, no HAND; restrict = L4 exact, vars excluded; JS n/a; final ~ best):
| cell | used H+M of 52 | patch top (drop) | p_k | col_k | carry_k | low_k | high_k | R^2 col/carry | PR | restrict | best / final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b2 LSB 1x | 22-42 | MLP3 (MLP2) .94-1.0 | .87-.92/1.0 | 1.0/1.0 | 1.0/1.0 | .83-.88/1.0 | .41-.63/1.0 | .88-.95/.93-.95 (L4) | 1.5-3.1 | .01 | .63-1.0 |
| final cell (10 best, 3 final) | 22-32 | MLP3 (MLP0) .98-1.0 | .91/1.0 | 1.0/1.0 | 1.0/1.0 | .85/1.0 | .58/1.0 | .94/.94 (L4) | 2.4 / 2.9 | .01 | 1.0 / 1.0 |
| b2 MSB 1x / 2x / aux | 40-52 | MLP3 .95-.99 | .97-.99/.99 | 1.0/1.0 | .99-1.0/.99 | .97-.99/.99 | .91-.97/.99 | .83-.96/.83-.92 (L3-4) | 1.2-1.7 | .01 | .74-.98 |
| b10 LSB 1x | 23-37 | MLP3 .96-1.0 | .99/.99 | .22-.24/1.0 | u | .10-.12/.36-.44 | .15-.16/.36-.45 | .92-.93/.93-.96 (L3) | 3.6-4.5 | .00 | 1.0 |
| b10 MSB 1x / aux | 33-47 | MLP3 (MLP0) .97-1.0 | .96-.98/.95-.98 | .17-.24/.99 | u | .09-.11/.35-.45 | .15-.18/.35-.44 | .88-1.0/.89-.96 (L3) | 6.0-7.0 | .00 | .87-.99 |
| b16 LSB 1x | 25-34 | MLP3 (MLP0) .97-1.0 | .97-.98/.96-.97 | .10-.14/.27-.28 | 1.0/.98-.99 | .07-.08/.22-.25 | .08/.23-.25 | .92-.93/.96-.97 (L3) | 3.9-6.3 | .00-.01 | .84-1.0 |
| b16 MSB 1x | 33-46 | MLP3 (MLP0) .99-1.0 | .91-.97/.91-.96 | .10-.13/.26-.31 | .98-.99/.95-.97 | .06-.08/.22-.24 | .06-.08/.22-.24 | .90-.92/.89-.93 (L3) | 7.0-7.5 | .00 | .72-.98 |
| b16 2x (LSB+MSB) | 27-31 | MLP3 .98-1.0 | .99-1.0/.98-.99 | .09-.11/.21-.24 | 1.0/.97-.98 | .03-.06/.19-.23 | .05-.09/.18-.23 | .92-.95/.92-.96 (L3) | 4.9-8.5 | .00-.01 | .99-1.0 |
| SP b16 (6+1 final); b10 (4+1) | 23-30; 13-19 | MLP3 .99-1.0; .98-1.0 | 1.0/.99; .99-1.0/1.0 | 1.0/.99-1.0; 1.0/1.0 | u | .07-.10/.26-.29; .11-.16/.33-.43 | .08-.09/.26-.29; .13-.19/.33-.43 | .81-.83/.62-.66 (L4/L3); .88-.89/.67-.79 (L4) | 4.4-6.7; 4.9-7.7 | .00 | 1.0 / 1.0 |
Reading: col_k / carry_k fix the answer (L4 exclusion -> exact ~0); failures are in computing them: MSB nets use more components, lower-R^2 carry, errors at the longest carry chains (b2 bits 6-9, b10 digit 2, b16 digits 1-2).
## Closed form
SP final-MLP digit map E = W_out^T U_digits^T (+ bias) replaced by Fourier modes over digit d: Ehat[j,d] = c[j,0] + sum_{k in K} (c[j,k] cos 2pi kd/B + s[j,k] sin 2pi kd/B), coefficients = exact DFT
projections of the weights; K_dec = {1,2,3,5}, K_hex = {1,2,4,8}. Bar: exact >= .99 x native on 2048 fresh pairs: decimal 6/6 best, 6/6 final seeds (lossless 2/6, 4/6), hex 3/6, 4/6 (lossless 1/6);
full DFT keeps all, mean-only 0. Types A-C (AO): no accepted closed form; a scalar col_k/carry_k substitution (decimal, probe R^2 .998/.959) drops exact from 2041 to 3 of 2048: the code is not scalar.
## Weights excerpt and reading
f0 seed 0 best, final MLP neuron j: W_in[j, 0:2] -> E[j, digits 0-3] (full W_out and unembedding; rows by digit-contrast norm): decimal N123 (.0731, .1140) -> (1.431, -1.567, 1.452, -1.673); N8
(.1037, .0168) -> (-.967, 1.478, -1.314, 1.279); hex N215 (.0499, -.0325) -> (-2.169, -1.655, -.871, .071). N123 votes even digits, N8 odd (parity mode c5 = +1.518 / -1.307); hex N215 a broad
preference (c1 = -1.829, s1 = .462): outgoing votes, not how inputs activate them.
## Comparison
No HAND. Agrees with Lee et al. 2023 (reversed output) and Nogueira et al. 2021 (representation decides learnability); Pres et al. 2025's aux loss did not rescue binary MSB, a scratchpad did; decimal col_k's non-linear code fits Fu et al. 2026 (periodic number features). New: the base effect depends on digit order.
## Controls and limits
5 seeds per screening cell (aux/SP 3 per mix, final 10); base 2 diags on 1000 pairs. Length: SP padding or ranges [0, 512)-[0, 10000) 0/256 (one
net 2/256); final cell native 1.0 (20/20 nets), 9-bit padding .02-.29, [0, 512) .00-.10 (middle bits fail). b2 MSB AO 2x budget untested (run killed). Missing: SP b2 diags (OOM), SP b10 2 of 6 nets, 11 R1/D3 decimal diags.
## References
Lee et al. 2023, Teaching Arithmetic to Small Transformers, https://doi.org/10.48550/arxiv.2307.03381 (research preprint); Pres et al. 2025, Why Can't Transformers Learn Multiplication?, https://doi.org/10.48550/arxiv.2510.00184 (research preprint); Nogueira et al. 2021, Investigating the Limitations of Transformers with Simple Arithmetic Tasks, https://doi.org/10.48550/arxiv.2102.13019 (research preprint); Fu et al. 2026, Convergent Evolution: How Different Language Models Learn Similar Number Representations, https://doi.org/10.48550/arxiv.2604.20817 (research preprint)
