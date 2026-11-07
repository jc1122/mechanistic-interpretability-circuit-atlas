# Chess mate-in-one selection: a trained final MLP decodes the hand-built escape count (3/3 per format); long lists need coverage data; from-scratch training not yet measured

**Task.** A position with a bare black king (KQK/KRK-type, optionally one extra white piece) and K = 1–16 supplied legal checking
candidates; output the first candidate that mates, or NONE. This is candidate selection, not move generation. Answer-only A: one 17-way
index. Scratchpad T: `from to | count ;` per candidate (count = black-king escape squares, 0–8), then `! index`. Fresh boards every
batch, board-hash 80/20 split; exact = whole teacher-forced continuation correct.
**Status.** 2026-11-07. HAND A/T; PART A/T at r1 (K ≤ 8) and at the coverage retry r2 (K ≤ 16), 3 seeds each, all solved. Open:
**the TRAIN rung has no finished run** (earlier runs were cut by host loss at 7k / 4.5k steps; a relaunch is in training), so
scratchpad vs answer-only from scratch, grok step and TRAIN circuits are unmeasured; **no ten-seed final**; the K ≤ 16 stress replay
of T coverage-r2 has not landed; the T closed form is accepted only locally (K3/K5), not at K ≤ 16; public code pending.

## Summary
With every earlier stage frozen at HAND, a trained final MLP (layer 7) reads HAND's escape count (ESC) and first-mate slot (WIN) and
decodes the index in 2.5–8.5k steps in both formats. Trained on lists of at most 8 candidates (r1), the nets are exact on their
monitor but only 52–64% exact on a K ≤ 16 stress set; with coverage data (K ≤ 16, balanced first-mate index, escape counts 7–8) the A
nets are 608/608 on the same stress set. The decoder is not a copy of HAND: in T it also uses a direct ESC → MLP7 route next to HAND's
MLP6 → selector head → WIN route.

## Setup
[METHODS.md](../METHODS.md) for shared conventions. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1,000, cosine, clip 1, batch 8 fresh
boards, fp32, 10k budget, stop when all seeds are exact; eval every 500, checkpoints every 2,500. r1: K ≤ 8, 128 test sequences. r2
(retry rule: data): K ≤ 16, balanced first-zero index, extra-piece and escape-count 7–8 exposure, 512 test sequences, 64k / 68k
presentations per A / T seed. The best snapshot is selected on the test set (it doubles as monitor), so the K ≤ 16 stress set (608
legal sequences) is the independent check. Legality is cross-checked with python-chess. GPU accelerators for training, CPU fp32 for
re-scoring. Public reproduction code is pending; the archived networks re-score to the recorded accuracies.

## Architecture as used
GPT-NeoX, ReLU, no LayerNorm, no positional encoding, untied embeddings, vocabulary 928, context 128. HAND/PART: 8 layers, d_model 480,
24 heads × 20, MLP 640 (84 heads and ~1,270–1,290 neurons structurally active). PART trains only the layer-7 MLP (615,520 parameters),
input-masked to ONE, phase flags, from-/to-square coordinates, ESC and WIN. Planned TRAIN: 4 layers, d_model 384, 8 heads × 48, MLP 768.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-mate-in-one](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-mate-in-one):
`hand_{A,T}`, `part_{A,T}_{r1,r2}_{best,final}_s0…2` (the 4 MLP tensors; insert into `hand_<fmt>`), `test_<run>` (each run's test
sequences and scored-position masks), `config.json`, `manifest.json` (seed, steps, recorded and fp32 re-scored exact, sha256).

## Prediction before training
Recorded before the runs: PART 3/3 ≥ .99 by 2,000 steps — **solved, but later** (2.5–4k at r1). TRAIN: T ≥ 2/3, A ≤ 1/3 — **not yet
testable** (the cut runs were at A .97, T .49, pointing the other way).

## Hand-built reference
Source and capture terms delete the mover's old square and a virtual destination inserts it; nearest-live rays and leaper ORs test
each king-neighbour square with the old king square removed. MLP5 sums the escapes into ESC; the MLP6 gate MATE = ReLU(u) − ReLU(u − 1),
u = 2·TO − 2·EQS − 4·ESC − 1, feeds an earliest-prefix selector head (layer 7 head 0) that writes WIN. Verified: 339 stratified
sequences per format, K 1–16, counts 0–8, all exact; minimum gold logit margin A 30.0, T 29.9. Limit: mate is exact only for a bare
black king; with a black piece the count is escapes only (a capture of the checker can be legal at 0 escapes).

## What training finds
| Cell | Solved | Best → final test (s0 / s1 / s2) | First exact step | K ≤ 16 stress (608) |
|---|---|---|---|---|
| PART A r1 (K ≤ 8) | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 | 4k / 2.5k / 2.5k | 344–388 |
| PART T r1 (K ≤ 8) | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 | 2.5k / 2.5k / 4k | 317–328 |
| PART A r2 coverage | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 (512) | 6.5k / 8k / 8k | 608 on all 6 snapshots |
| PART T r2 coverage | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 (512) | 8–8.5k | not landed |
| TRAIN A / T | **no finished run** | | | |
Circuit types: one per PART cell, 3/3 seeds (frozen count + learned decoder). The r1 failure is length coverage: long lists and late
first-mate indices were rare in K ≤ 8 training; softmax over more attended tokens compresses logit gaps (Zsámboki et al., 2025, preprint), consistent with the small late-index margins.

**Diagnostics** (64 K = 3 rows per snapshot; used = component mean-ablation drop > .05 (heads + whole MLPs); R² = best linear probe
(layer); inter = linear interchange flip; r90/r99 = output-residual components; JS = attention divergence to HAND; ranges over seeds,
best and final):
| Cell | Used H+MLP | count R²; WIN R² | inter count; WIN | r90 / r99 | JS | Best / final |
|---|---|---|---|---|---|---|
| HAND A | 17+5 | 1.00 (6); .983 (8) | n/a; .90 | 11 / 34 | 0 | 1 / 1 |
| PART A r1 (5 of 6) | 14–17+5 | 1.00 (6); .992–.997 (8) | n/a; .82–1.00 | 9–10 / 31–32 | 0 | 1 / 1 |
| PART A r2 | 14–17+5 | 1.00 (6); .994–.997 (8) | n/a; .95–1.00 | 8–9 / 30–31 | 0 | 1 / 1 |
| HAND T | 38+5 | 1.00 (6); .979 (8) | 1.00; 1.00 | 20 / 91 | 0 | 1 / 1 |
| PART T r1 | 34–43+5 | 1.00 (6); .990–.994 (8) | .85–1.00; .80–1.00 | 16–17 / 72–75 | 0 | 1 / 1 |
| PART T r2 | 36–40+5 | 1.00 (6); .993–.995 (8) | .96–.98; .91–1.00 | 15–17 / 71–74 | 0 | 1 / 1 |
Reading: JS 0 and the layer-6 count probe come from the frozen parts by construction, not learned agreement. Top patch drop: A = layer 7
head 0 / MLP5 (1.0), T = MLP7 (1.0), then layer 2 head 3 (.66). Coverage makes T's count use more uniform (inter .96–.98 vs .85–1.00);
the learned output is lower-rank than HAND's. Edge-attribution checks fail (Spearman .35–.68 < .8), so edge wiring is not claimed.

## Closed form
**PART r1, accepted at the substitution bar (substituted ≥ .99 × native):** with tent features φ_j(v) = (1 − |v − j|)₊ and phase gates
g, MLP7 ≈ g_A Σ_j φ_j(WIN) θ_j + g_COUNT Σ_c φ_c(ESC) η_c + a from/to-gated coordinate lookup + syntax constants. A: 442 fitted scalars
(17-wide basis) replace 615,520 parameters on all 6 r1 snapshots and keep every prediction, including stress. T: 7,693 fitted scalars
keep 256/256 fresh K = 3 sequences on all 6 r1 snapshots (and 128/128 K3 / K5 whole-path for seed 0), but fail stress retention for
final s0/s1 — accepted locally only. Rejected: a WIN-only selector in T (interchange flip .82). Controls (T s0): zeroing the terminal
MLP → 0 exact; shuffled inputs → token accuracy .467. r2 nets have no substitution test yet.

## Weights excerpt and reading
Frozen MLP6 on (TO, EQS, ESC): input rows (2, −2, −4) twice, biases (−1, −2), output (1, −1) = the bounded MATE gate. Fitted T seed-0
response for WIN = 6 on output directions (YINDEX5, YINDEX6): best (.714, .579), final (.763, .734), unembedding scale 30; A seed-0
best (.953, .963). The small margin to the neighbouring index is a hypothesis for the late-index stress errors, not yet tested.

## Comparison
PART A and T both solve 3/3 with similar first-exact steps; T keeps more heads causally live (34–43 vs 14–17) because its count tokens
keep the escape stages in use. Scratchpad vs answer-only from scratch, TRAIN size and length generalisation: open (no finished TRAIN run).

## Controls and limits
Substitutions are on fixed K3 / K5 / stress sets, not all positions. Natural three-piece lists are short (enumerated KQK / KRK maxima 12
/ 6 distinct checks), so K 13–16 needs extra pieces. Seed counts are screening counts (3). Size counts are component-level single
ablations, not minimal circuits.

## References
- python-chess (software): https://github.com/niklasf/python-chess — legality cross-check
- Zsámboki et al. (2025). Post-Norm can Resharpen Attention. arXiv preprint. https://doi.org/10.48550/arXiv.2510.08341 (research preprint)
