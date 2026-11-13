# Quadratic roots: answer-only vs sub-result vs full trace vs scaffolded NAND scratchpad (HAND n* ≈ 3)
**Task.** `BOS sb |b| sc |c| =` → `f sr- |r-| sr+ |r+|` for x² + bx + c, |b|, |c| ≤ R (n 5: R 15 / 31; n 6: R 63, 16129 pairs); D = b² − 4c,
s = isqrt(max(D, 0)), r∓ = ⌊(−b ∓ s)/2⌋, flag N/I/R. A answer only; S writes D, s; T every step; **S-exec**: NAND updates on a register
file, one per step. Split by (b, c): train .5 (A/S/T), 80/20 (S-exec, 12903 pairs).
**Status 2026-11-12.** Final round, 10 seeds per cell: S-exec PART 10/10 solved; S-exec TRAIN (2x width) 0/10; A n6 R63 TRAIN 0/10 (cosine
decay from the 22.5k snapshot after a host loss; plateau, test .929-.944). Earlier rounds r1-r4 (3-10 seeds): every A/S/T cell 0/n.
## Summary
HAND: A ~6R neurons, S one layer fewer, compact T 62 neurons, S-exec 4 heads + 1 neuron (every n). Only the scratchpad PART solves (10/10,
one type: an AND-hinge bank read as NAND, one basin; HAND's one neuron substitutes exactly). Other cells memorise or underfit thresholds.
## Setup
[METHODS.md](../METHODS.md). CE on output tokens; score = exact match of all output tokens (teacher forced). AdamW 1e-3, wd .1, warmup 1000,
cosine, clip 1; 10 seeds per process; 40k, min 30k, plateau 5k, eval 500; PART fp32, TRAIN TF32 (GPU). S-exec: 128 fresh packets per step
(widths 3-6, register renaming). Reproduce: `launch_final.sh`, `test_hand.py`, `scratch/test_hand.py`, `subst_check.py`; keys `test`, `local`. Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/quadratic-roots](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/quadratic-roots).
## Architecture as used
A/S/T: GPT-NeoX, 4 L × 12 heads (d_head 64), d 768, MLP 512 ReLU (n 6: 16 heads, MLP 1024), no LayerNorm, NoPE, vocab 10. S-exec: 2 L,
2 heads × 64, d 128, MLP 128 ReLU, vocab 1029, positions unused. PART: frozen HAND + trained stage MLP (A L2, S/T L1; S-exec L1: W_in
[128, 128] masked to residual columns A, B, b_in [128], W_out [128, 128], b_out [128]). TRAIN w2: d 256, 4 heads, MLP 256. Archive
`quadratic-roots/`: hand_S_exec, part_S_exec_final_s0-9 (4 tensors), train_S_exec_final_s0-2, split, config.json, manifest.json (sha256).
## Prediction before training
Recorded before training: PART A 10/10 (wrong: r2 .58-.62); PART S ≥ 7/10 (wrong: .03-.06); PART T 10/10 (wrong: r3 .83 memorised); TRAIN
A ≤ 2/10, T ≤ 3/10 (right: 0); HAND T ~62 vs A ~6R (right); S-exec PART 3/3 by 2.5k (right); S-exec TRAIN ≥ 2/3 (wrong: 0/10).
## Hand-built reference
L0 frame: BOS head → 1/(1+r), MLP → POS one-hot. Table heads read a written number as a weighted token sum; readout THR: bit k of V =
[V − W ≥ 2^k] as relu(2u + 1.5) − relu(2u + .5), gated by −G per failed condition. A: L1 b², L2 isqrt [D ≥ m²], L3 roots. S: L1 b² + isqrt
of read |D|. T: all rules in L1. S-exec: L0 heads find the A/B pointer tokens, L1 heads fetch registers a, b by binary address match (wrong
code bit −40), one neuron h = ReLU(A + B − 1) → Y, readout bit0 = 10 h, bit1 = 5 (NAND). Exact on all pairs (A/S/T n ≤ 6 except T n 6;
S-exec n 3..8, margin 5.0). Sizes n 5 designed / used: A 6h 217 / 7h+194, S 9h 321 / 10h+260, T 9h 322 / 10h+187, T compact 9h 62 /
10h+44, S-exec 4h 1; n 6: A 6h 385, S 9h 507, S-exec 4h 1 (2296 steps).
## What training finds (final round, 10 seeds; earlier cells median of their seeds)
| cell | best / final test | train | stop | solved | verdict |
|---|---|---|---|---|---|
| S-exec PART | 1 / 1; n 7/8 rollout 1 | 1 | solved 500 | 10/10 | AND-hinge bank = NAND |
| S-exec TRAIN w2 | trace 0 / 0, local .58-.62 | .58-.62 | plateau 30k | 0/10 | constant output |
| A n6 R63 TRAIN | .929-.944 / same (decay from 22.5k) | .97-1 | plateau 13.5k | 0/10 | memorised, late lookup |
| earlier A/S/T r2-r3 (3-10 seeds) | A R31 TRAIN .88; S R31 PART .06; T R31 TRAIN .80; T R15 PART r3 .83 | 1; .13; 1; 1 | plateau/budget | 0 | memorised / underfit |
Circuit types (S-exec PART): one, 10/10, one basin (aligned lmc barrier .001-.007); bank sub-split 7/10 OR-pool, 3/10 mixed. Diagnostics:
| cell | used heads + neurons (ablation) | best probe R² (layer) | eff. rank99 vs HAND | attn JS vs HAND | best / final |
|---|---|---|---|---|---|
| S-exec PART (10) | 4h + 28-37 of 128 hinges (HAND 1) | a, b, sum 1, nand .998 (L2) | MLP1 2, L2 17 (HAND 1 / 18) | 0 (frozen) | 1 / 1 |
| S-exec TRAIN w2 (10) | none (any ablation ±.01) | a, b, sum, nand ≤ 0 | L1-L2 2-3 | .65 (ln 2 .69) | 0 / 0 |
| A n6 R63 decay (10) | 5-9 heads + 4 MLPs | s .95-.97, D .92-.96, b2 .93-.96 (L3) | MLP1 W_in 582 (s0; HAND ov ~0) | .38 (L0) | .945-.975 / .93-.98 (200 rows) |
| S R31 PART r2 (3) | stage only (patch drop 1.0) | D .97, s .97, b2 .88 (L2) | W_in 41 (HAND ov .03) | 0 (frozen) | .06 / .025-.035 |
Failures: S-exec TRAIN ignores the addressed registers (JS .65), outputs the majority bit; S PART misses D bits 2^5..2^0 (unit thresholds,
|D| ≤ 1085); A TRAIN: no early D → isqrt path; HAND-free circuit 38-55% of 1935 edges, .29-.42 edge overlap with HAND's; D causal only at L4.
## Closed form
S-exec PART, all 10 seeds: MLP1 := e_Y ReLU(A + B − 1) (HAND, 0 fitted scalars). Substituted for all four trained MLP1 tensors: 1792/1792
held-out packets per seed, 10/10 (native 1792/1792, agreement 1.0); predicted-state rollouts n 5-8 exact. Resample ablation of everything
outside HAND's 9 edges (of 26): exact 1.0 = base, 10/10; resampling those 9 instead: .49-.54, a random 9: .52-.55. HAND-free discovery
keeps the same 9 edges in 10/10 seeds (faith 1.0). Accepted. Others: none.
## Weights excerpt and reading
HAND: BIG 40, U 10, C_POS 128; THR step pair (2, 1.5) / (2, .5); S-exec neuron A 1, B 1, bias −1 → Y, readout (bit0, bit1) = (10 h, 5).
S-exec PART s0 top neuron by |bit-margin| (archive slice W_in[:, A/B], b_in, W_out[ONE/Y]): n25 A .124, B .093, c −.004, ONE −.056, Y .107 →
response 00/01/10/11 = 0 / .088 / .120 / .212, margin −1.35. Every seed's top hinges look the same (A, B .04-.16, c ~0, margin −1.05..−1.60):
each is a rescaled AND; ~33 of them sum to what HAND does with one. Recompute: excerpts.py `quad`.
## Comparison
HAND's NAND spread over a bank; from scratch the address fetch is not found; answer-only cells memorise. Prior tasks include fuzzy quadratics (Buckley and Eslami 1997), sequence-model polynomial roots (Saxton et al. 2019), approximate numerical roots (Freitas et al. 2021), worked quadratic solutions (Hendrycks et al. 2021), solving programs rather than NAND traces (Palermo et al. 2021), and quadratic root-type classification rather than root extraction (Thomas and Bani-Yaghoub 2026). Nanda et al. 2023 and Qiu et al. 2026 concern modular addition and integer arithmetic, respectively, not quadratic-root evidence.
## Controls and limits
HAND sizes: ablation minima on 200-300 rows. Rollouts sampled. A n6: decayed from s22500 (host lost); its diagnostics use 200 rows. Sub-split is causal, not separate algorithms.
## References
Buckley and Eslami 1997, Neural net solutions to fuzzy problems: the quadratic equation: <https://doi.org/10.1016/S0165-0114(95)00412-2> (journal article); Huang and Chi 2001, Complex polynomial roots by feedforward networks: <https://doi.org/10.1109/IJCNN.2001.1016716> (conference paper); Huang and Chi 2001, Problem decomposition for real polynomial roots: <https://doi.org/10.1109/IJCNN.2001.1016718> (conference paper); Buckley et al. 2002, Fuzzy equations using evolutionary algorithms and neural nets: <https://doi.org/10.1007/s005000100147> (journal article); Huang et al. 2003, Dilation for close polynomial roots: <https://doi.org/10.1016/S0375-9601(03)00216-0> (journal article); Huang et al. 2003, Ordered polynomial roots by constrained partitioning networks: <https://doi.org/10.1109/IJCNN.2003.1223844> (conference paper); Huang 2004, Constructive neural polynomial-root finding: <https://doi.org/10.1109/TNN.2004.824424> (journal article); Huang et al. 2004, Neural root finder based on root moments: <https://doi.org/10.1162/089976604774201668> (journal article); Huang et al. 2005, Constrained-learning neural root finders: <https://doi.org/10.1016/j.amc.2004.04.070> (journal article); Epitropakis and Vrahatis 2005, Neural root finding and approximation: <https://doi.org/10.1145/1140378.1140382> (journal article); Mourrain et al. 2006, Counting real polynomial roots through neural networks: <https://doi.org/10.1016/j.camwa.2005.07.012> (journal article); Dilanian et al. 2006, Neural polynomial-root finding for phase-retrieval diffractometry: <https://doi.org/10.1063/1.2201447> (journal article); Zhang et al. 2008, Multiple real roots using polynomial discrimination: <https://doi.org/10.1109/ICCIS.2008.4670918> (conference paper); Zhu et al. 2008, Fast neural extraction of all polynomial roots: <https://doi.org/10.1109/ICOSP.2008.4697750> (conference paper); Zhang and Zeng 2009, Neural simultaneous extraction of polynomial roots: <https://doi.org/10.1109/CIS.2009.78> (conference paper); Das and Seal 2012, Real polynomial roots with feedforward networks: <https://doi.org/10.1109/NCCCS.2012.6413004> (conference paper); Saxton et al. 2019, Analysing mathematical reasoning abilities of neural models: <https://doi.org/10.48550/arXiv.1904.01557> (arXiv preprint); Freitas et al. 2021, Neural approximation of arbitrary polynomial roots: <https://doi.org/10.3390/math9040317> (journal article); Hendrycks et al. 2021, Measuring mathematical problem solving with the MATH dataset: <https://doi.org/10.48550/arXiv.2103.03874> (arXiv preprint); Palermo et al. 2021, Mathematical reasoning via program synthesis: <https://doi.org/10.48550/arXiv.2107.07373> (arXiv preprint); Bukhsh et al. 2022, LSTM interpretation for approximating polynomial roots: <https://doi.org/10.1109/ACCESS.2022.3157306> (journal article); Nanda et al. 2023, Progress measures for grokking via mechanistic interpretability: <https://doi.org/10.48550/arXiv.2301.05217> (arXiv preprint); Thomas and Bani-Yaghoub 2026, Limits of interpretable learning in quintic root classification: <https://doi.org/10.48550/arXiv.2602.23467> (arXiv preprint); Qiu et al. 2026, Explaining and tuning arithmetic transformers with human strategies: <https://doi.org/10.48550/arXiv.2607.17166> (arXiv preprint); Nunley 2026, TANGO: Treating tokens as operators: <https://doi.org/10.48550/arXiv.2608.22117> (arXiv preprint); Wang et al. 2026, Function-conditioned neural Caputo King–Neta root estimator: <https://doi.org/10.3390/fractalfract10100669> (journal article).
