# Modular arithmetic mod 97 in five number formats: token multiplication learns a discrete-log clock at the output; digit multiplication memorises
**Task.** a + b and a · b mod 97 (a, b in 0..96) and x² mod 97 (x in 0..9408): 9409 inputs per cell, answer r = S mod 97, exact match.
Formats: `tok` (one token per residue; not for x²), binary LSB-first `b2l` / MSB-first `b2m`, decimal `b10`, hex `b16` (MSB first), and
`s10` (decimal intermediate outputs `S ; S div 97 ; r`). Sequence `BOS a op b = answer`. Per seed a random split of the 9409 inputs:
train fraction .3 (2823 train / 6586 test) or .9 (8468 / 941).
**Status.** 2026-11-06. HAND: 17 cells, all exact. TRAIN: 16 cells × 10 seeds at fraction .3; failed cells retried at .9 (and once more
with weight decay .1 for decimal/hex multiplication). Final cell: token multiplication at .9, 10 seeds. PART: a state-scratchpad variant
(7 cells × 3 seeds) fails everywhere and is reported as a localised failure.

## Summary
Does modular arithmetic in any number format converge to Fourier features? Only with residue tokens. Token multiplication (fraction .9)
solves 6/10 seeds; their output logits are a "clock" in discrete-log coordinates: logit(c) ≈ Σ_k α_k cos(2πk(log₅a + log₅b − log₅c)/96) +
β_k sin(..) with 1–6 frequencies reproduces 9409/9409 answers and agrees with the net on ≥ .9947 of held-out inputs in all six solved
seeds. This is an output-level surrogate, not a recovered internal circuit. The hand-built net computes the same group isomorphism
Z₉₇* ≅ Z₉₆ with a scalar log ramp and a lookup instead of circles. Decimal and hex multiplication memorise at every data and decay setting
tried: operands and the raw product are decodable, the residue is not.

## Setup
[METHODS.md](../METHODS.md) for the shared conventions. TRAIN: AdamW lr 1e-3, weight decay 1.0 (decimal/hex multiplication retry .1),
betas (.9, .98), warmup 1000, cosine over 40k steps, clip 1, batch 512, TF32; evaluation every 500 steps; stop at test 1.0 or no gain >
.005 over 5k steps after 20–30k (per cell); snapshots: best training accuracy ("best", not best held-out) and final, plus
checkpoints every 2500 steps. All seeds of a cell in one process on consumer GPUs. Archived nets were re-scored in fp32 on CPU: within
1 test row of the recorded TF32 accuracy for token multiplication, within .0012 for binary addition.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/modular-arithmetic-mod-97](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/modular-arithmetic-mod-97)
(35 files: hand-built token multiplier; token multiplication s0–9 best/final; binary addition s3/s4/s0 best/final; PART scratchpad
binary multiplication s0–2 best/final + its hand-built base; splits; `manifest.json` with sha256 and accuracies).
**Reproduce:** the training and analysis code for this task is not yet in `code/`; weights, splits and accuracies are in the folder above.

## Architecture as used
TRAIN (every cell): GPT-NeoX 4 layers × 4 heads (d_head 64), d_model 256, MLP 512 ReLU, sequential residuals, no LayerNorm, no positional
encoding, untied readout; vocabulary 103 (BOS, values 0..96 = residues or digits, `+ * ^ = ;`). HAND: 8 layers × 8 heads, d_model 512,
MLP 2304 (one config for all 17 cells); the token multiplier uses 1 head and 194 neurons of layer 0 (10,864 non-zero of 27.4M weights;
`hand_mul_tok` stores the non-zero tensors only). PART (scratchpad S, binary LSB multiplication): 3 layers × 9 heads (d_head 4),
d_model 36, MLP 130, vocabulary 395; only the layer-2 MLP is trained (`dense_h_to_4h` [130, 36], `dense_4h_to_h` [36, 130] + biases).

## Prediction before training
Recorded in the design file before training. Token addition 10/10 with Fourier embeddings: refuted at .3 (3/10), held at .9 (10/10).
Token multiplication 10/10 with Fourier in log coordinates: at .3 0/10, at .9 6/10 (+2 at ≥ .98), log-order code held. Binary: no
circular code: held (0 binary nets carry ≥ 2 circular key frequencies). Digit formats: lookup/compare, no p-periodic code: mostly held;
the heuristic Fourier screen flags decimal addition in 3/10 best (1/10 final) and decimal square in 2/10, which is not a causal clock
certificate. `s10` beats `b10`: held (square at .3: `s10` best .84–.99 vs `b10` .01).

## Hand-built reference
Token addition: the "pizza" construction (Zhong et al., 2023), 4 frequencies. Token multiplication: discrete-log adder; the embedding
holds log₅v and [v = 0]; one head averages the two operands (S = la + lb, zero flag), an MLP0 lookup writes 5^(S mod 96) (or 0 if a
zero flag is set). Digit formats: index heads fetch digits, place-value sums give A, B (or x); addition: lookup on A + B;
multiplication: thermometer [A ≥ k] and partial products relu(B − M(1 − [A ≥ k])), compare-subtract steps [P ≥ 97j] → q, r; square:
compare-subtract on x → x mod 97, then a square lookup. All 17 are exact on all 9409 inputs (minimum logit margin 23.6).

## What training finds
| Cell | Fraction | Solved (test ≥ .99) | Test, best snapshot |
|---|---|---|---|
| add tok | .3 → .9 | 0/10 (3 at .93–.98) → 10/10 | .067–.982 → .997–1 |
| add b10 / b16 / `s10` | .3 | 10/10 each | .993–1 |
| add b2l / b2m | .3 | 4/10 / 3/10 | .121–.999 / .947–.999 |
| mul tok | .3 → .9 | 0/10 → 6/10 | .033–.253 → .425–1 |
| mul b10 / b16 | .3; .9; .9 decay .1 (3 seeds) | 0 each | .9: .055–.247; decay .1: .124–.580 |
| mul b2m | .3; .9 | 0/10; 0/10 | .9: .133–.437 |
| mul `s10` | .9 (3 seeds) | 0/3 | .876–.911 |
| sq b10 / b16 | .3 → .9 | 0 → 9/10 / 4/10 | .9: .976–1 / .693–1 |
| sq b2l / b2m; sq `s10` | .3 and .9; .3 | 0 each; 1/10 | binary near chance (train ≈ .02: under-fit); `s10` .841–.990 |
Low-data `s10` and high-data b2l multiplication endpoints could not be verified from saved records and are excluded.
- Learning curve, token multiplication at .9: test rises with train (no memorisation phase at this fraction); first test ≥ .99 at
  21.5k, 22k, 26k, 28.5k, 35.5k, 36k steps (s0, s4, s8, s2, s3, s1); s5 .983, s9 .988, s6 .837, s7 .425 at the 40k budget.
- Circuit type, token multiplication: log-order clock in all 10 seeds' embeddings (see Weights excerpt); the output surrogate is
  certified in the 6 solved seeds. Embedding-level: replacing the residue embedding by its 24 strongest discrete-log Fourier modes keeps
  9409/9409 in best s4 (s0 8482/9409); other solved seeds are not certified at this level.
- Digit multiplication fails at the reduction: in decimal/hex multiplication (.9, 3 seeds each) A and S probe at R² .62–.80, B .37–.72 (layers 2–4),
  but r stays at .47–.49 (layer 0), and S→r distributed alignment ≤ .14. Binary square (train ≈ .02): x² mod 97 from 14 bits needs unit
  thresholds on x ~ 2^13 first.
- Binary LSB addition (.3): MLP0 is a causal prefix-count rectifier, m̂0(p) = ReLU(φ_p C) L_r R_r + b, φ_p = (onehot(t_p), c_p/(p+1), 1),
  c_p = counts of each token in the prefix (at `=`: popcount(a) + popcount(b)). Substituted for MLP0 at rank 4–8 on 1024 fresh test rows it
  keeps accuracy (≥ .99 × native) and agrees .9922–1 in 5/10 seeds (s2, 3, 5, 6, 8; best and final); 2 failed seeds pass only at full rank,
  3 reject. The later arithmetic is not resolved.
- PART (scratchpad S, binary LSB multiplication, 3 seeds): 0/3, exact ≤ .0085. Local states C, A, B reach layer 2 intact (probe R² 1);
  the trained MLP2 ranks first in activation patching (drop .49–.60). A fitted 10-coefficient affine writer fails substitution in all 6
  snapshots; replacing only this MLP by the hand-built modular update gives 941/941 in every snapshot: the failure is the writer.
- Scratchpad S vs answer only (multiplication, .9; S changes batch, decay, padding and target together, so not a format-only effect):
  TRAIN S solves 15/15 (5 formats × 3 seeds, first ≥ .99 at 13–28k) where answer only solves 6/10 token and 0 decimal/hex/MSB seeds; S nets depend on
  their written states but fail state-only restarts (.14–.73), and reach 0–.76 at padding 64 (trained 0–12). PART S 0/15.
- Diagnostics (token multiplication .9; best snapshot 9 seeds, final 10; used = heads with mean-ablation drop > .05 + whole MLP blocks):

| Snapshot | Used heads + MLPs | Best probe R² (layer): la / lb / la+lb mod 96 | A, B, r | Attention JS to HAND | Test |
|---|---|---|---|---|---|
| best | 7–16 + 4 | ≤ .62 / ≤ .40 / ≤ .78 (0–4) | ≤ .12 | .17–.37 | .425–1 |
| final | 11–16 + 4 | same range | ≤ .12 | .17–.37 | .425–1 |
Linear probes of the scalar logs stay low because the net stores them as angles; interchange was untested (no answer-fixing pairs).

## Closed form
Token multiplication, output replacement on nonzero a, b (a or b = 0 keeps answer 0): logit_hat(c) = Σ_{k∈K} [α_k cos(2πkδ/96) +
β_k sin(2πkδ/96)], δ = log₅a + log₅b − log₅c. K = the m strongest frequencies of the net's mean logit as a function of δ, α, β by least
squares on the seed's training rows; pass = held-out accuracy ≥ .99 and agreement ≥ .99 at some m ≤ 6.

| Seed | Net test (best / final) | m | K (best) | Fit R² | Surrogate, all 9409 | Agreement, test (best / final) |
|---|---|---|---|---|---|---|
| 4 | 1 / 1 | 1 | 43 | .49 | 1 | 1 / 1 |
| 8 | 1 / 1 | 2 | 17, 16 | .20 | 1 | 1 / 1 |
| 0 | 1 / 1 | 3 (final 6) | 10, 48, 21 | .52 | 1 | 1 / 1 |
| 2 | 1 / 1 | 3 (final 2) | 38, 36, 29 | .18 | 1 | 1 / 1 |
| 1 | .9947 / .9957 | 6 | 6, 4, 8, 2, 10, 1 | .42 | 1 | .9947 / .9957 |
| 3 | .9957 / .9957 | 6 | 6, 12, 18, 45, 48, 8 | .37 | 1 | .9957 / .9957 |
The argmax is unique only if gcd(K ∪ {96}) = 1; every passing set satisfies this. In failed s6 and s7 the top-6 sets are all even, so c and
−c (log₅(−1) = 48) tie exactly and the surrogate's accuracy is not meaningful there. The surrogate is fitted to the net's own logits on all
frequencies ranked over all inputs: it shows what the output computes, not how the layers compute it.

## Weights excerpt and reading
Residue embedding of the TRAIN nets (rows 1..96 of `gpt_neox.embed_in.weight`, residues 1..96): R² of the top-6 Fourier modes when the
rows are ordered by residue v vs by log₅v (final snapshot; fit over Z₉₆):

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| residue order | .21 | .17 | .20 | .26 | .15 | .29 | .24 | .23 | .22 | .19 |
| log order | .83 | .89 | .90 | .76 | .72 | .67 | .57 | .85 | .88 | .92 |
| top log frequencies (share) | 36 .36, 24 .18 | 2 .28, 4 .24 | 38 .48, 20 .25 | 6 .36, 12 .20 | 36 .44, 30 .22 | 37 .48, 22 .09 | 2 .37, 46 .32 | 40 .56, 16 .19 | 28 .39, 40 .27 | 1 .55, 2 .23 |
Reading: in every seed the embedding is periodic in the discrete log, not in the residue: the net transports the modular-addition clock
to Z₉₆ through log₅. The output frequency set (closed form) can differ from the embedding's leading modes (s4: output 43, embedding 36,
30, 43). Seed 4 final: one output frequency (k = 43, coprime to 96) with amplitude √(α²+β²) = 23.49 (best 8.43) carries every answer.
Hand-built token multiplier: embedding column LG = log₅v, column ZF = [v = 0]; layer-0 head 0 averages the two operand tokens
(key = value-token flag, query bias 240: uniform over a and b); MLP0 is a one-hot lookup on SL = la + lb (input weight 2, ZF weight −1000) writing Y_{5^s mod 97}.
Values: embedding fit by `fourier` (power of the rFFT of the embedding rows reordered), closed-form K, R², amplitudes by least squares on
the archived nets (recomputable from the HF files and `split_mul_x3`).

## Comparison
HAND and TRAIN share the isomorphism Z₉₇* ≅ Z₉₆ but not the code: scalar ramp + lookup vs circles in log order; TRAIN is the
modular-addition clock (Nanda et al., 2023; Zhong et al., 2023; Gromov, 2023) in log coordinates. Data decided token cells: fraction .3
memorises (multiplication 0/10), .9 groks, as in grokking on small algorithmic datasets with p = 97 (Power et al., 2022). Digit formats
never form circles; solved digit nets read r out categorically. Intermediate outputs help (S multiplication 15/15, `s10` square), in line
with scratchpads (Nye et al., 2021), but the trained S nets are not the hand-built autonomous recurrence.

## Controls and limits
The Fourier label was checked on all 17 HANDs (token addition F, the other 16 not). Best = best-training snapshot. Diagnostics cover
token multiplication fully (one best seed missing) and other cells partially; neuron-level sizes are measured only for best s4/s0
(12 heads + 1872 neurons / 14 + 1984 in deletion-local tests vs HAND 1 + 194). No length or range generalisation was tested (fixed
operand widths; padding shifts context only). Answer-only digit cells were run once per retry step; S cells have 3 seeds.

## References
- Nanda, N., et al. (2023). Progress measures for grokking via mechanistic interpretability. arXiv preprint. https://doi.org/10.48550/arxiv.2301.05217
- Zhong, Z., Liu, Z., Tegmark, M., & Andreas, J. (2023). The clock and the pizza. Conference paper (NeurIPS). https://doi.org/10.52202/075280-1186
- Gromov, A. (2023). Grokking modular arithmetic. arXiv preprint. https://doi.org/10.48550/arxiv.2301.02679
- Power, A., et al. (2022). Grokking: generalization beyond overfitting on small algorithmic datasets. arXiv preprint. https://doi.org/10.48550/arxiv.2201.02177
- Nye, M., et al. (2021). Show your work: scratchpads for intermediate computation with language models. arXiv preprint. https://doi.org/10.48550/arxiv.2112.00114
