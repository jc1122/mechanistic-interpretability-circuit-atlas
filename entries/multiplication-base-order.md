# Multiplication: digit order and number base decide answer-only learnability; a scratchpad solves every cell; its readout is a sparse Fourier digit map

**Task.** `BOS x * y =` → product, x, y in [0, 256), written in base 2 (8 + 8 input digits → 16 product digits), 10 (3 + 3 → 5) or
16 (2 + 2 → 4), zero-padded. Answer-only (AO): the product MSB-first or LSB-first. Scratchpad (SP): after `=` the running sums
S_j = x · (y mod B^(j+1)) for j = 0…N_in−1, each LSB-first and closed by `;`, then the product MSB-first. Test: 2000 held-out pairs
(fixed seed); training draws fresh pairs from the other 63,536 every batch; exact = every output token correct (teacher-forced).
**Status.** 2026-11-07. AO 6 cells × 4 curriculum mixes × 5 seeds (120 runs); retries: 2× width (45), 3× batch data (25),
auxiliary running-sum probe loss (12); SP 18 runs (MSB-first, 3 seeds per mix); final cell (base 2 LSB, anneal) 10 seeds.
4 layers, d_model 384, 30k steps. No hand-built or PART rung (explicit exception, see Hand-built reference).

## Summary
Digit order decides whether AO multiplication is learned: LSB-first solves base 10 (20/20) and base 2 (17/20; final cell 10/10),
MSB-first 1/60. Doubling the width rescues decimal and hex MSB (7/10, 6/10) but never binary MSB (0/26 across width, data and the
auxiliary loss: the middle bits stay under-fit). A scratchpad solves every MSB cell at 1× width (18/18), binary fastest. Every AO net
carries the column variables at the readout, and MSB nets use 25–80 % more components. The only accepted closed form is the SP
readout: the final MLP's digit vote is a sparse Fourier map over digit values (decimal 6/6 seeds, hex 3/6 at the ≥ .99 × native bar).

## Setup
[METHODS.md](../METHODS.md) for shared conventions. Curriculum mix f: each sample is a sub-problem with probability f (digit × digit;
y with one nonzero digit; y with two), else a uniform pair; f = 0, .3, .6 or anneal (.6 → 0 by step 15k). AdamW lr 1e-3, wd .1,
betas (.9, .98), warmup 1,000, cosine to 30k, clip 1, batch 256, TF32, eval every 500 steps; solved = best test exact ≥ .99.
Seeds: 5 per AO screening cell, 3 per SP / auxiliary mix, 10 in the final cell. Public reproduction code for this entry is pending;
the archived networks below re-score to the recorded accuracies (manifest).

## Architecture as used
GPT-NeoX, 4 layers, d_model 384 (12 heads × d_head 32), MLP 256 ReLU (W_in [256, 384], reads the whole residual), no LayerNorm, no
positional encoding, untied embeddings; vocabulary 19 (BOS, `*`, `=`, digits 0–15; SP 20 with `;`); all weights trained (rung TRAIN).
Width retry R1: d_model 768 (12 × 64). Data retry D3: batch 768. Auxiliary arm: running-sum linear-probe loss on the layer-3 residual.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/multiplication-base-order](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/multiplication-base-order):
`<base><order>[_R1|_sp]_<mix>_s<seed>[_final].safetensors` (best-test snapshot, `_final` = last step), `test_<tag>.safetensors`
(held-out pairs), `config.json`, `manifest.json` (per net: base, order, mix, seed, step, fp32 re-score, recorded accuracy, sha256).
Archived: 12 AO nets (solved cells and failed controls), SP best + final for 6 decimal and 6 hex seeds and 2 binary packs, and
3 seeds of the final cell.

## Prediction before training
Recorded in the task specification before the runs: base 16 and 10 solve, binary MSB fails — half right (hex fails at 1× width);
LSB-first helps every base — right; the curriculum speeds up base 2/10 but rescues nothing — right (anneal > .3, otherwise ≈);
width does not rescue binary MSB — right. SP (recorded before the SP runs): decimal/hex ≥ 2/3 per mix and faster than the 2×-width
AO nets — right; binary ≥ 2/6 — right; binary slower than decimal/hex — wrong (binary is fastest). Final cell ≥ 9/10 — right.

## Hand-built reference
None: this task has no HAND or PART rung (explicit exception; the comparison is between formats, bases and orders). Probed
variables are those of schoolbook multiplication: output digit p_k, column sum col_k = Σ_{i+j=k} x_i y_j, carry_k, low_k / high_k
(product mod / div B^k); for SP lines the same quantities for S_j = S_{j−1} + x · y_j · B^j.

## What training finds
| base | order | f 0 | .3 | .6 | anneal | R1 2× width | D3 3× data | SP 1× f0 / anneal (first ≥ .99) |
|---|---|---|---|---|---|---|---|---|
| 2 | LSB | 3/5 | 4/5 | 5/5 | 5/5; final 10/10 | – | – | – |
| 16 | LSB | 1/5 | 1/5 | 2/5 | 0/5 | anneal 5/5 | – | – |
| 2 | MSB | 0/5 | 0/5 | 0/5 | 0/5 | f0 0/5, anneal 0/5 (.91–.96) | f0 0/5 (.90–.93) | 3/3 / 3/3 (8.5–10.5k) |
| 10 | MSB | 0/5 | 0/5 | 0/5 | 1/5 | f0 3/5, f.3 4/5 | f0 1/5, anneal 2/5 | 3/3 / 3/3 (11.5–16k) |
| 16 | MSB | 0/5 | 0/5 | 0/5 | 0/5 | f0 5/5, anneal 1/5 | f0 0/5, anneal 0/5 (.97–.99) | 3/3 / 3/3 (17.5–21.5k) |

Base 10 LSB solves 5/5 in every mix. Auxiliary loss: binary MSB 0/6 (.90–.98), decimal MSB 0/6 (.87–.98). Final cell: test .5 at
6–10k steps, .99 at 16–23.5k (median 18.75k), best = final = 1.0 in 10/10; train ≈ test (fresh pairs, no memorisation). AO MSB
retries reach exactness only at 25–28k of 30k (budget-limited); binary MSB under-fits (train .95–.999), decimal/hex MSB at 1× width
memorise (train 1.0). SP best = final = 1.0.

Diagnostics (best snapshots, ranges over seeds; variable columns = interchange / DAS flip at layer 4, u = untested; restrict = layer-4
exact with the variable subspace excluded):
| cell | used heads + MLPs of 52 | patch top (drop) | p_k | col_k | carry_k | low_k | high_k | restrict | best |
|---|---|---|---|---|---|---|---|---|---|
| b2 LSB 1× | 22–42 | MLP3 (MLP2) .94–1.0 | .87–.92/1.0 | 1.0/1.0 | 1.0/1.0 | .83–.88/1.0 | .41–.63/1.0 | .01 | .63–1.0 |
| b2 MSB 1× / 2× / aux | 40–52 | MLP3 .95–.99 | .97–.99/.99 | 1.0/1.0 | .99–1.0/.99 | .97–.99/.99 | .91–.97/.99 | .01 | .74–.98 |
| b10 LSB 1× | 23–37 | MLP3 .96–1.0 | .99/.99 | .22–.24/1.0 | u | .10–.12/.36–.44 | .15–.16/.36–.45 | .00 | 1.0 |
| b10 MSB 1× / aux | 33–47 | MLP3 (MLP0) .97–1.0 | .96–.98/.95–.98 | .17–.24/.99 | u | .09–.11/.35–.45 | .15–.18/.35–.44 | .00 | .87–.99 |
| b16 LSB 1× | 25–34 | MLP3 (MLP0) .97–1.0 | .97–.98/.96–.97 | .10–.14/.27–.28 | 1.0/.98–.99 | .07–.08/.22–.25 | .08/.23–.25 | .00–.01 | .84–1.0 |
| b16 MSB 1× | 33–46 | MLP3 (MLP0) .99–1.0 | .91–.97/.91–.96 | .10–.13/.26–.31 | .98–.99/.95–.97 | .06–.08/.22–.24 | .06–.08/.22–.24 | .00 | .72–.98 |
| b16 2× (LSB + MSB) | 27–31 | MLP3 .98–1.0 | .99–1.0/.98–.99 | .09–.11/.21–.24 | 1.0/.97–.98 | .03–.06/.19–.23 | .05–.09/.18–.23 | .00–.01 | .99–1.0 |
| final cell; SP b2/b10/b16 | open | open | open | open | open | open | open | open | 1.0 |

Reading: where an AO net is right, col_k / carry_k fix the answer (flip ≈ 1 in base 2, DAS .99 for decimal col_k), and excluding
their layer-4 subspace drops exact match to ≈ 0. Failures lie in computing them: MSB nets use more components, carry_k decodes
.83–.86 at layer 3 vs .95 in LSB nets, and errors sit at the longest carry chains (binary bits 6–9, decimal digit 2, hex digits 1–2).
Binary nets carry every column variable linearly; decimal col_k only non-linearly (interchange .22, DAS .99); hex has no col_k even
under DAS (.21–.31): its 16 × 16 digit table is read per digit.

## Closed form
SP readout only. The final MLP's digit map E = W_outᵀ U_digitsᵀ (+ bias) is replaced by Fourier modes over the digit value d:
Ê[j, d] = c[j, 0] + Σ_{k∈K} (c[j, k] cos 2πkd/B + s[j, k] sin 2πkd/B), coefficients = exact DFT projections of the weights;
K_dec = {1, 2, 3, 5}, K_hex = {1, 2, 4, 8}. Bar: exact ≥ .99 × native on 2048 fresh pairs: decimal 6/6 best and 6/6 final seeds
(lossless 2/6 and 4/6), hex 3/6 and 4/6 (lossless 1/6). Controls: the full DFT keeps every prediction, mean-only keeps none.
This covers the readout vote only; no compact circuit is accepted for the AO cells or for the SP column computation.

## Weights excerpt and reading
SP f0 seed 0 best, final-layer MLP neuron j: W_in[j, 0:2] → E[j, digits 0–3] (E from the full W_out and unembedding; neurons chosen
by digit-contrast norm). Decimal N123 (.0731, .1140) → (1.431, −1.567, 1.452, −1.673); N8 (.1037, .0168) → (−.967, 1.478, −1.314,
1.279); hex N215 (.0499, −.0325) → (−2.169, −1.655, −.871, .071). Reading: N123 votes for even digits and N8 for odd ones (parity
mode k = 5: c = +1.518 / −1.307); hex N215 carries a broad low-frequency preference (c1 = −1.829, s1 = .462). These are outgoing
votes (weight slices); they do not say how the inputs activate the neurons.

## Comparison
No hand-built net. Consistent with Lee et al. 2023 (reversed output order makes arithmetic learnable for small transformers) and
Nogueira et al. 2021 (the number representation decides learnability). The auxiliary running-sum loss that Pres et al. 2025 report
as a fix for answer-only multiplication did not rescue binary MSB here; a scratchpad (Nye et al. 2021) did. New here: the base effect
depends on digit order, and a curriculum only buys time where a cell is learnable at all.

## Controls and limits
5 seeds per screening cell (auxiliary/SP 3 per mix; final cell 10); base-2 diagnostics on 1000 pairs. SP length/range (n = 256):
padding or ranges [0, 512)–[0, 10000) score 0/256 (one net 2/256): no length generalisation. Open: default diagnostics of the
final cell and the SP nets (queued), 11 decimal R1/D3 diagnostics, binary MSB AO at 2× budget (run killed, untested), public code.

## References
[Lee et al., Teaching Arithmetic to Small Transformers](https://doi.org/10.48550/arxiv.2307.03381) — research preprint; digit-order effect.
[Nogueira et al., Investigating the Limitations of Transformers with Simple Arithmetic Tasks](https://doi.org/10.48550/arxiv.2102.13019) — research preprint; representation effect.
[Pres et al., Why Can't Transformers Learn Multiplication? Reverse-Engineering Reveals Long-Range Dependency Pitfalls](https://doi.org/10.48550/arxiv.2510.00184) — research preprint; auxiliary running-sum loss.
[Nye et al., Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/abs/2112.00114) — research preprint; scratchpad comparison.
