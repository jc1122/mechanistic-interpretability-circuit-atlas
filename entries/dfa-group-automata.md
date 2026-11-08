# Group automata (parity, Z3, S3): a scratchpad state table is learned at once; answer-only learns count -> residue only in range
**Task.** Words s_1..s_n over a group (parity Z2: 2 symbols; Z3 counter: 3; S3: the 6 permutations), q_0 = identity, q_t = q_{t-1} * s_t.
A (answer only): `BOS s_1 .. s_n = Q(q_n)`. B (scratchpad): `BOS s_1 Q(q_1) .. s_n Q(q_n)`. n uniform 1..16; 20k unique train + 2k test words
per seed; exact match; length n = 32, 64, 128. **Status.** 2026-11-08; 10 seeds per cell; final run S3 B TRAIN wd .1: 10/10 solved.
## Summary
Scratchpad (B): PART and TRAIN learn the state table delta(q_prev, s) in <= 2500 steps (3 groups, 10/10); PART keeps HAND's recency head
(n 128); TRAIN fetches q_{t-1} its own way, breaks at ~2N; final S3 TRAIN: a serial MLP0-5 chain, delta table at MLP3-5, not HAND's one head.
Answer-only (A): S3 PART = HAND's composition table; abelian PART misses whole large S (staircase limit; closed after r3); A TRAIN fresh words:
parity, Z3 10/10 (count, then residue; length-bound), S3 3/10 (sign first; closed after r2, d 512 retry lost to infrastructure).
## Setup
[METHODS.md](../METHODS.md). AdamW lr 1e-3, wd .1, warmup 1000 + cosine, clip 1, batch 256, fp32, eval 500; PART 10k (Z3 A 30k), TRAIN 40k,
plateau stop; r2: fresh words per batch, PART 30k; r3: PART tail-balanced S (Dirichlet(1) symbol mix), S3 TRAIN d 512. Consumer GPUs.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/dfa-group-automata](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/dfa-group-automata) (safetensors, config.json, manifest with seed, rung, accuracy, sha256).
## Architecture as used
GPT-NeoX 6 layers x 4 heads x 64 (d 256), MLP 512 ReLU, no LayerNorm, NoPE, untied readout; vocab 7 / 9 / 15 (BOS, PAD, symbols, '=',
states). PART re-initialises and trains one MLP on frozen HAND (B: L1; A abelian: L1; A S3: L5); mask = its W_in on position dims held at 0.
TRAIN: all from HF init. Archive `dfa-group-automata/`: HAND x6, all PART seeds, best + worst TRAIN seed, final-run S3 B TRAIN s0-s3.
## Prediction before training
Written before training. 1 B PART exact, 8N masked: held. 2 B TRAIN >= 7/10, HAND circuit, half >= .9 at 2N: 10/10, HAND head 1/10 (final S3 0/10),
1-2/10 (final 2/10). 3 A abelian PART exact: failed (tail S). 4 A abelian TRAIN <= 5/10, length-bound: r2 10/10, length-bound. 5 A S3 PART = HAND
incl. length: held. 6 A S3 TRAIN <= 3/10, depth-bound: 0/10 exact, frontier n ~ 9-16 (held). 7 grok B < A abelian < A S3: B first.
## Hand-built reference
L0: BOS-sink position stage (P = 1/(j+1), hinges -> R = j/256). B: L1 recency head (score 100 [class] + 20 j) -> QP = q_{t-1}; L1 MLP G^2 neurons
relu(QP_a + TOK_b - 1) -> OUT[a*b]: 1 head + 4 / 9 / 36 neurons at any n. A abelian: BOS-sink head -> S = sum of symbol values; (G-1) n hinges
relu(S - k) read out as [S mod G = r]. A S3: prefix doubling, layer k fetches the block 2^(k-1) back, 36-neuron table: 5 heads + 180 neurons, n <= 31.
## What training finds
- B PART (all groups, mask or not): 10/10 exact at 500 steps; recency head kept (prev mass 1.0); stage = (q_prev, sym) table; exact n 128.
- B TRAIN: groks 500-2500 (10/10 per group); MLP3-5 = delta table; fetch: parity spread L2-L5 9/10 (HAND head 1/10), Z3 one head 7/10; n64 <= .14.
- Final S3 B TRAIN (10 seeds): grok 1500-2000; types: q done by L4-L5, MLP5 idle (6/10); q done in MLP5 (4/10). Discovered circuits 31-96
  edges (faith .94-.96), none matches HAND; n32 >= .9 in 2/10, n64 <= .18, n128 0.
- A S3 PART: 10/10 by 1000 steps; ST1..ST4 acc 1.0 at L2..L5 (HAND layers); n32 .18 = HAND. A abelian PART: S exact (R^2 1.00), stage =
  f(S), but whole large S wrong (parity S = 13, Z3 S >= 17-19); tail-balanced S: parity mask 7/10, parity 1/10, Z3 0/10 (not data).
- A TRAIN r1 (static words): parity no S mod 2 (acc .52), Z3 9/10, S3 only the sign (8/10). r2 (fresh): parity, Z3 10/10: S at L1-L2, MLPs =
  f(S), HAND's type over trained S only (parity n32 chance; Z3 n32 >= .95 in 3/10); S3 sign 1.0 (10/10), exact to n ~ 9-13, no doubling.
- Diagnostics (probe: median best layer, one-hot acc, R^2 for S; r90 of the last residual; attention JS ~0: HAND heads uniform):
| cell | used heads + MLPs | probes | r90 | best / final |
|---|---|---|---|---|
| B PART x3 | 1 + 1 | q_prev, q 1.00 (L2) | 2-10 | 1 / 1 |
| A S3 PART | 5 + 5 | ST1..ST4 1.00 (L2..L5), q .94 | 35 | 1 / 1 |
| B TRAIN parity / Z3 | 1-6 + 4-6 / 0-3 + 2-6 | q_prev, q 1.00 (L5-6 / L3), t R^2 .97 / .98 | 1.5 / 2 | 1 / 1 |
| B TRAIN S3 final (10) | 0-3 + 5-6 | sym 1.00 (L0), q_prev R^2 .87-1.00 (L3-L5), q .98-.99 (L5-L6); CKA vs HAND L6 .40-.53 | 2 | 1 / 1 |
| A parity PART r1 / r2 / r3 (+mask) | 1 + 1 | S 1.00 (L2), q .63 / .97 / .96-.99 | 1 | .96-.99 / .986-.999 / .988-1.0 |
| A Z3 PART r1 / r2 / r3; mask r1-r3 | 1 + 1 | S 1.00 (L2), q .80 / .76 / .79; mask .07 R^2 / .72 / .73 | 1 | .91-.98 / .92-.97 / .88-.97 |
| A parity TRAIN r1 / r2 | 0 + 0 / 3-10 + 4-6 | S 1.00 (L1), q .52 / 1.00 (L3) | 2 / 1 | .50-.52 / 1; final .48-.52 / 1 |
| A Z3 TRAIN r1 / r2 | 2-6 + 5-6 / 3-6 + 4-6 | S .95 / .98 (L1 / L2), q 1.00 (L4 / L2) | 2 | .998-1 / 1 (best = final) |
| A S3 TRAIN r1 / r2 | 0-4 + 0-5 / 4-20 + 4-6 | ST1 .37 / .55, ST2..4 .18-.23, q .34 / .64 | 2.5 / 4.5 | .21-.37 / .70-1.0 (best = final) |
## Closed form
B: q_t = delta(q_{t-1}, s_t) on the fetched state (substitution 1.0, PART 30/30, TRAIN 30/30); A S3 PART: MLP4 = f(ST4), MLP5 = f(q) 10/10;
A abelian: q = [S mod G] tables (= own accuracy, PART 40/40; TRAIN r2 parity 10/10, Z3 8/10 at L5, 9/10 at L3-L4). Final S3 B TRAIN:
MLP_l = T_l[q_{t-1}, s_t] (36 entries), l = 3, 4, 5: substitution 1.0 in 10/10; resampling all outside the discovered circuit keeps .94-1.0
(8/8 seeds with one): accepted 8/10. MLP0-5 + mean-ablation-used heads alone fails (.0-.53): "idle" heads still carry signal.
## Weights excerpt and reading
`excerpts.py dfa` (B PART mask, L1 stage). 65-85 % of W_in energy on QP + symbol dims. Neurons are group characters, not HAND's AND gates:
parity s0 n116 reads QP (-.77, .22), symbol (.73, -.26) = XOR; S3 s0 n106 reads QP and symbols with the permutation sign pattern (+ - - + + -).
## Comparison
Scratchpad: one G^2 table at any n; length set by the fetch (PART n 128, TRAIN ~2N). Answer-only: TRAIN finds length-bound shortcuts (Liu et al. 2022).
## Controls and limits
Substitution controls (zero, mean, shuffle, swap) per layer; resample complement; probes; d 256 only; abelian staircase ~16-18 teeth (Hong et al. 2022).
## References
Liu et al. 2022, https://doi.org/10.48550/arxiv.2210.10749 (arXiv preprint) · Hong et al. 2022, https://doi.org/10.48550/arxiv.2208.04924 (arXiv preprint) · Abbe & Sandon 2023, https://doi.org/10.1002/cpa.22121 (journal article, Comm. Pure Appl. Math.)
Buda et al. 2018, https://doi.org/10.1016/j.neunet.2018.07.011 (journal article, Neural Networks) · Parodi et al. 2026, https://doi.org/10.48550/arxiv.2604.14433 (arXiv preprint)
