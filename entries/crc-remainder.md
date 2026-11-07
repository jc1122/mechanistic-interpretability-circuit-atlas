# CRC remainder: the scratchpad step r = ((r ≪ 1) | a) XOR (q·D) is learned exactly (PART 10/10 in 500 steps, TRAIN 3/3); answer-only learns a divisor-conditioned parity readout that is not the hand-built one

**Task.** GF(2) polynomial remainder N mod D: N has n input bits (n 8/9/10 in training and test), D is monic of degree 4 (16–31;
D ∈ {19, 25, 28, 30} withheld). Prefix `BOS N_MSB…N_LSB / D_MSB…D_LSB =`. Answer-only A emits the 4 remainder bits; scratchpad T emits,
after each input bit, the quotient bit and the full 4-bit remainder, with `;` between states (5n bits + n − 1 separators). This is
algebraic division, not a standardised CRC encoding. Width-invariant (N, D) hash split 80/20 (no zero-padding leakage); fresh pairs every
batch. Exact = whole teacher-forced output correct; test = mean over 512 held-out pairs at each of n 8/9/10 (1,536 per seed).
**Status.** 2026-11-07. HAND A (4 layers) and T (3 layers) exact; PART T ten-seed final 10/10 (seeds 0–9); PART A, TRAIN A and TRAIN T at
3 screening seeds. Open: ten-seed finals for PART A and both TRAIN cells; matched zero controls for the PART T closed form on seeds 3–9;
the TRAIN A data retry (wider training widths, proposed, not run) and TRAIN A s2 checkpoint dynamics; greedy rollout; public code.

## Summary
With frozen hand-built routing, one trained MLP learns the hand-built one-step update on all ten seeds within 500 steps, and an
eight-ReLU table with 8 fitted coefficients replaces it (11,587/11,587 rows kept). From scratch the scratchpad is also learned (3/3, first
exact at 11.5–16k), with the quotient, shift and next-remainder variables causally used (interchange/DAS 1) but through an MLP → head →
MLP path rather than the hand-built direct fetches. Answer-only A is learned too (PART 3/3, TRAIN 2/3), but every A net reads the
remainder out only in the last layer through a final-MLP code that is not the hand-built parity staircase; the one failed TRAIN A seed
misses 3/256 rows at one answer bit for two divisors. Neither format generalises reliably to longer inputs from scratch.

## Setup
[METHODS.md](../METHODS.md) for shared conventions. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1,000, cosine, clip 1, batch 256 fresh
pairs, budget 30k, plateau 5,000 / minimum 22.5k, checkpoints every 2,500, eval every 500. PART fp32; TRAIN TF32 updates, fp32 evals and
OOD. Success = test exact 1. GPU accelerators for training, CPU fp32 for re-scoring. Test pairs are regenerated from the seed (see
`config.json`). Predictions were committed before any smoke run. Public reproduction code for this entry is pending; the archived
networks re-score to the recorded accuracies.

## Architecture as used
GPT-NeoX, ReLU, no LayerNorm, no positional encoding, untied embeddings, vocabulary 7, context 128, d_model 128. A: 4 layers × 16 heads
(d_head 8), MLP 1,024. T: 3 layers × 4 heads (d_head 32), MLP 256. PART trains only the stage MLP (A layer 3, T layer 2) with W_in masked
to the hand-built input coordinates; all routing stays frozen HAND. TRAIN: same shapes, all weights from random init.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/crc-remainder](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/crc-remainder):
`hand_{A,T}`, `part_T_{best,final}_s0…9`, `part_A_{best,final}_s0…2` (the 4 stage-MLP tensors; insert into `hand_<fmt>`),
`train_{A,T}_{best,final}_s0…2` (full state dicts), `config.json`, `manifest.json` (seed, steps, recorded and fp32 re-scored exact, sha256).

## Prediction before training
Recorded before the runs: PART T by 2.5k — **confirmed** (500). PART A by 15k with poor unseen-D / n 12 — **confirmed** (5.5–7k; unseen D
.09–.10). TRAIN T 3/3 by 15k — **mostly confirmed** (s0 took 16k). TRAIN A ≥ 1/3 by 30k — **confirmed** (2/3). TRAIN T fails n 12 —
**confirmed** (0/3).

## Hand-built reference
T (7 used heads + 236 neurons): heads fetch the previous state's shift bit, the input bit, the divisor bit and the quotient bit; one
XOR-AND stage computes r = ((r ≪ 1) | a) XOR (q·D). A (18 heads + 344 neurons): GF(2) coefficient-parity staircases per divisor. Both exact
on 9,568 sequences per format (n ≤ 6 exhaustive, n 7–12 stratified over all D), margin 30.

## What training finds
| Cell | Solved | Best → final test (s0 / s1 / s2) | First exact step (s0 / s1 / s2) | Stop |
|---|---|---|---|---|
| PART T, 10 seeds | 10/10 | 1 → 1 (all) | 500 (all) | solved 500 |
| PART A | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 | 5.5k / 5.5k / 7k | solved 7k |
| TRAIN T | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 | 16k / 13k / 15k | solved 18k |
| TRAIN A | 2/3 | .9954 / 1 / 1 → .9935 / 1 / 1 | 26k (best) / 23k / 24.5k | plateau 28k |
Scratchpad vs answer-only: PART 10/10 at 500 vs 3/3 at 5.5–7k; TRAIN 3/3 vs 2/3; the answer-only nets use about twice the heads (zero
ablation 16 vs 7 PART, 25–32 vs 12 TRAIN). No classic delayed grokking: test ≥ .9 never lags train ≥ .99. Length: PART T s0–2 pass n 1–12;
TRAIN T n 11 best → final .84 / .86 / .83 → .18 / .92 / .57 and n 12 ≈ 0; PART A n 12 seen-D ≈ 1/3, unseen D at n 10 .09–.10. Circuit
types: PART T — one type (hand-built table) 10/10; TRAIN T — one type by diagnostics, 3/3; A (PART and TRAIN) — one coarse cluster
(last-layer readout), 3/3 each; TRAIN A s2 additionally routes through a single dominant layer-3 head.

**Diagnostics** (n 10 batch of 256 teacher-forced rows; H = used heads, M = used MLPs, mean/zero ablation > 5 pp; probe = best linear
R² at the layer (T: shift, divisor, quotient, AND, next; A: count, next); inter → DAS = interchange flip, then distributed alignment;
restrict R / X = restricted / excluded exact on the probe subspace; edges ρ = EAP-IG check, J = edge Jaccard vs HAND; r99 = last-layer
residual rank; JS = attention divergence to HAND):
| Cell | E best / final | H; M | Best probe R² | Inter → DAS; R / X | Top patch (drop / rec) | Edges ρ; J | r99 (HAND); JS |
|---|---|---|---|---|---|---|---|
| PART T s0–2 | 1 / 1 | 4/7; 1/3 | .88–.91, 1, .90–.91, .66–.68, .88–.89 (3) | .92–1 → 1; 1 / 0 | MLP2 1 / 1 | .996; 1 | 9 (10); 0 |
| PART A s0–2 | 1, 1, .996 | 14/16; 2/4 | .52–.57 / .51–.60 (4) | .58–.73 → .97–1; .30–.42 / .13–.16 | MLP3 1 / 1 | .94–.97; .60–1 | 25; 0 |
| TRAIN T s0–2 | 1 / 1 | 4–5/12; 3/3 | .83–.88, .71–.88, .85–.95, .69–.90, .78–.91 (3) | .99–1 → 1; 1 / 0 | MLP2 1 / 1 | .81–.93; .29–.39 | 4–10 (10); ≤ .45 |
| TRAIN A s1 | 1 / 1 | 19–20/30–32; 4/4 | .74 / .69–.72 (4) | .98–1 → 1; 1 / .04–.05 | MLP0 1.01 / .86–.90 | .82–.87; .09 | 31–33; ≤ .47 |
| TRAIN A s2 | 1 / 1 | 16–18/25–28; 4/4 | .74 / .63–.68 (4) | .85–.99 → 1; .98–1 / .07–.08 | MLP3 1.03 / .45–.47 | .83–.86; .03 | 28–29; ≤ .46 |
| TRAIN A s0 (failed) | .988 / .988 | 20–21/26; 4/4 | .78 / .68–.70 (4) | 1 → 1; .98 / .09 | MLP3 1.01 / .73–.76 | .74–.83; .10 | 40; ≤ .45 |
T divisor / AND have no answer-fixing pairs and step / column / phase are constant at the query: untested, not unused. TRAIN A s0 final
edge ρ is .74 (< .8), so its edge reading is not trusted. Failure explained: TRAIN A s0 (train 1, test .9954 → .9935) gets 3/256 rows
wrong, all at answer bit 3, for D = 22 (1/24) and D = 29 (2/23); count and next are both used at layer 4 (inter 1 / .996, DAS 1), so no
hand-built variable is missing — the fault is a divisor-specific readout in MLP3 (patch rank 1). Checkpoint dynamics: s0 and solved s1
follow one trajectory to 20k (no memorisation alerts) and split only in the cosine decay (s1 reaches 1 at 25k; s0 peaks .992, ends
.988): a late residual, not a different path.

## Closed form
Accepted on fresh held-out screens (n 8/9/10, 768 rows per snapshot), best and final:
- **PART T s0–2:** an eight-ReLU table — SEP → `;`, QF → X, else (LOW ? A : X) XOR (DB AND QB) — with 8 fitted coefficients keeps
  11,587 → 11,587. Seeds 3–9 (the ten-seed final) solve identically but have no matched zero controls yet.
- **PART A s1–2:** a native ReLU ridge bank over the input bits NB, divisor and column (36,867 numbers vs 263,296 in the stage) reproduces
  the learned behaviour, errors included (768/768; s2 767/767). Projecting the ridges onto one hand-built low-bit count direction keeps
  only 191–237/768 (rejected).
- **TRAIN:** no accepted closed form.

## Weights excerpt and reading
PART T MLP2 bank: a_j = ReLU(β_j0 + β_j·[X, A, DB, QB, QF, LOW, SEP]), 256 units, 2,819 numbers. Its response clusters (159–184
units) are the 8 truth-table cells of the hand-built XOR-AND step; the substitute ReLUs are not identified native neurons.
PART A: a_j = ReLU(β_j0 + Σᵢ α_ji·NB_i + u_j[D] + v_j[col]); a dense per-divisor parity surrogate covers the seen divisors and the low ten
bits only, which is why unseen D and n 12 fail.

## Comparison
The scratchpad reaches the hand-built step 10–20× faster in PART and solves every TRAIN seed; answer-only needs about 2× the heads and
keeps a divisor-specific readout gap. Learning parities without intermediate supervision is hard and curriculum over widths helps
(Cornacchia & Mossel, 2023); length transfer of arithmetic transformers depends on the training widths (Jelassi et al., 2023);
scratchpads as intermediate computation follow Nye et al. (2021).

## Controls and limits
Zeroing the final MLP drives exact to 0; on shuffled tokens the circuit drops to chance and patching restores it 1 / 1. Native
donor-path tests agree on 6,155/6,155 T and 2,336/2,336 A queries. PART A's n 12 high-bit repair and a full hand-built-stage swap
(65,536/65,536) are repairs, not native generalisation. Scores are teacher-forced; neuron necessity was not measured; only PART T has
ten seeds.

## References
- Nye, M. et al. (2021). Show Your Work: Scratchpads for Intermediate Computation with Language Models. arXiv preprint.
  https://doi.org/10.48550/arXiv.2112.00114 (research preprint)
- Power, A. et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv preprint.
  https://doi.org/10.48550/arXiv.2201.02177 (research preprint)
- Jelassi, S. et al. (2023). Length Generalization in Arithmetic Transformers. arXiv preprint.
  https://doi.org/10.48550/arXiv.2306.15400 (research preprint)
- Cornacchia, E., & Mossel, E. (2023). A Mathematical Model for Curriculum Learning for Parities. arXiv preprint.
  https://doi.org/10.48550/arXiv.2301.13833 (research preprint)
