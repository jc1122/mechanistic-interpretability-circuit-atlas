# GCD by subtractive Euclid: frozen Euclid parts need only a three-rule router; from-scratch trace nets build their own distributed circuit

**Task.** `BOS GCD A:a B:b → R:g EOS` (A, answer only) or the T trace: rows `A:|a−b| B:min(a,b)` until b = 0, then `R:g EOS`. Typed scalar tokens (role × value 0–511). Tuple-hash split 80/20, 256 held-out tuples; exact = whole teacher-forced output incl. EOS.
**Status.** Final run (2026-11-10): T PART 10 seeds (6-bit, 30k steps), T TRAIN 10 seeds (7-bit, 60k budget; 3 from the r3 retry + 7 final). Screens: A/T PART and TRAIN, 3 seeds each; S (local scratchpad) TRAIN 0/3, closed.
## Summary
With the hand-built frame, fetch and Euclid-step layers frozen (PART), the trained last MLP is a router: one circuit type in 10/10 seeds (HAND's 7 edges + one), 7/10 exact. A three-rule closed form replaces it with 100% exact on all 4096 6-bit inputs and on 8-bit inputs where the nets score 0 (complement resample .974–1.0). From scratch (TRAIN) 5/10 seeds are exact; each builds its own distributed circuit (4–9 heads, all 4 MLPs) that decodes but does not causally use HAND's variables; no closed form.
## Setup
See [METHODS.md](../METHODS.md). Operands A 5 bits (retry 6), T PART 6 bits, T TRAIN 7 bits. AdamW lr 1e-3, wd .1 (the only wd run on the final cells), warmup 1000, cosine, clip 1, batch 16 fresh tuples, eval every 500, plateau stop 5000 after 75% of the budget; PART fp32, TRAIN TF32; seeds 0–9; one GPU class per run. Controls: answer-only A cells, S local-scratchpad cells, HAND net on the same split. Code is not in this repository; trainer and diagnostics are described here. Headline keys: manifest `acc_test_fp32`, `acc_fresh256`.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/gcd-euclid](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/gcd-euclid)
## Architecture as used
GPT-NeoX 4 layers × 4 heads (d_head 32), d_model 128, ReLU MLP (T 1056 wide, A 1026), no LayerNorm, no positional embedding, vocabulary 7174, untied unembedding. T builds position from the causal 1/position signal (frame 520 positions). PART: HAND frozen except layer-3 MLP (W_in [1056, 128], W_out [128, 1056] + biases), re-initialised and trained; its input is the whole residual (HAND's OLD/ABS/MIN/ZERO, role flags, position/clock blocks). TRAIN: all weights from default init in the same shape. Archive: `hand_gcd_T.safetensors`, `part_gcd_T_final10_s{0-9}` (4 tensors each), `train_gcd_T_{r3,final}_s*` (full), `config.json`, manifest (seed, rung, accuracy, sha256).
## Prediction before training
Written before training: (1) PART T ≥ 8/10 by 15k steps: count held (10/10 ≥ .99, 7/10 exact), time missed (≥ .99 at 21–26k). (2) TRAIN T ≥ 5/10: held (5/10, at 60k; "by 30k" missed: test .70–.95 at 30k). (3) TRAIN A ≤ 3/10: held (0/3, 1/3). (4) no 8-bit extrapolation: held. (5) T degrades less than A with size: refuted (8-bit A PART .31–.56, T PART .05–.17). (6) shared arithmetic across rows: held in structure for PART (one 8-edge circuit), not tested per row.
## Hand-built reference
A: L1H0 averages the two operand tokens; MLP2 is a 702-hinge pair-index lookup of gcd. T: L0H0 + MLP0/1 build three clocks (periods 16, 19, 23) from the 1/position signal; L2H0/H1 fetch the previous row's a, b (OLD) by clock offset; MLP2 five neurons: n0 = ReLU(a−b), n1 = ReLU(b−a), n2 = ReLU(a), n3 = ReLU(1.5−2b), n4 = ReLU(.5−2b), so ABS = n0+n1, MIN = n2−n0, ZERO = n3−n4; MLP3 seven neurons route by role. Size T 3 heads + 1572 neurons, A 1 + 705. Exact on every 5-bit input and stratified sets through 8 bits, margin ≥ 9. One HAND only (HAND-a); no alternative designed.
## What training finds
| Cell | Exact seeds (≥ .99) | Test best / final | 90% / 99% test step (k) | longer inputs |
|---|---|---|---|---|
| T PART final, 6 bit | 7/10 (10/10) | 1.0 ×8, .996 ×2 / 1.0 ×7, .996 ×3 | 13–23.5 / 21–26 | 7 bit .23–.47, 8 bit .05–.17 |
| T TRAIN final + r3, 7 bit | 5/10 (10/10) | 1.0 ×8, .996 ×2 / 1.0 ×5, .996 ×5 | 19–38.5 / 31.5–48.5 | 8 bit .20–.33 |
| A PART / A TRAIN r2 (screens) | 3/3 / 1/3 | 1 / .938–.996 | 1.5–6 / 9–12.5 | 8 bit .31–.56 / .13–.22 |
| S TRAIN w1 / w2 (screens) | 0/3 / 0/3 | full trace .03–.52 | — | — |
Unsolved seeds miss one token of one held-out tuple. No memorisation phase: test reaches 90% before train reaches 99%; no mem/stall alerts.
Circuit types: **P1 HAND + router** (PART, 10/10 seeds, 7 exact): discovered circuit = HAND's 7 edges + L2H1>MLP3; reads the frozen L2 heads and MLP2. **T1 own distributed Euclid** (TRAIN, 10/10, 5 exact): 54–84 of 207 edges (HAND jaccard ≤ .10), seed-specific heads, separate basins (loss barrier 8.8). Frozen parts read by PART: L2H0, L2H1, MLP2 (zero-ablation also L0H0, MLP0/1: the frame).
Diagnostics (n = 128 diag tuples, all exact; final snapshots; neurons = MLP0/1/2/3 units carrying 90% of activation mass; PR = residual participation ratio at L4, HAND 2.1; JS = max over used heads vs the best HAND head):
| Net | Used heads; MLPs | Neurons | Best probe R² (layer): a, b, d, m, next | PR | JS | inter (used?) | Best / final |
|---|---|---|---|---|---|---|---|
| PART s0–9 | L2H0, L2H1; M2, M3 | frozen + 176–196 | 1, 1, 1, 1 (L4); .94–.96 (L4) | 1.76–2.12 | 0 | m 1.0 (s8, nf 507) | 1.0 or .996 / 7 × 1.0 |
| TRAIN s3 | 9; M0–3 | 273/265/725/703 | .86 (L2), .84 (L2), .92, .87, .90 (L3) | 9.4 | .48 | d, m, n ≤ .04; a .45 | 1.0 / 1.0 |
| TRAIN s6 | 9; M0–3 | 176/367/579/749 | .86 (L3), .80 (L2), .93 (L3), .88 (L2), .92 (L3) | 6.7 | .49 | a, b .56 (nf 36–42) | 1.0 / 1.0 |
| TRAIN s7 | 7; M0–3 | 364/322/519/758 | .88, .81, .90, .86, .90 (L3) | 11.3 | .48 | all ≤ .13 | 1.0 / 1.0 |
| TRAIN s8 | 8; M0–3 | 171/311/613/706 | .88 (L2), .86 (L2), .86 (L3), .91 (L3), .92 (L4) | 5.8 | .49 | a .40, b .40 | 1.0 / 1.0 |
| TRAIN s9 | 5; M0–3 | 178/440/529/729 | .86, .80, .91, .82 (L3), .91 (L4) | 2.3 | .44 | a .55, b .50 | 1.0 / 1.0 |
Unsolved TRAIN s0–2, s4, s5: same type (4–9 heads, all MLPs, flips ≤ .43); their one error is a row-1 copy, B or R token.
## Closed form
**P1** (replaces the trained MLP3): after B emit `A:ABS` if ZERO = 0, else `R:OLD_a`; after A emit `B:MIN`; after R emit EOS. Substituted in all 10 seeds: 4096 6-bit inputs .9988–.9998 → 100%; 256 8-bit inputs 0% → 100%. Complement (everything outside the 8 edges resampled), 7 exact seeds: .974–1.0 (circuit resampled or random 8 edges: ≤ .021); path drops L2 heads→MLP2→MLP3 .70–.88, L2 heads→MLP3 direct .16–.31. Accepted (share ≥ .90).
**T1**: none. The discovered circuit passes the complement test in 1/5 exact seeds (.83–.94); HAND's old-A rule gives 57.8% exact; per-component MLP fits fail on 7-bit nets.
## Weights excerpt and reading
HAND T L2 (weight slice, layer-2 MLP rows with nonzero W_in): n0 OLD_a +1, OLD_b −1; n1 −1, +1; n2 OLD_a +1; n3 bias +1.5, OLD_b −2; n4 bias +.5, OLD_b −2. Trained PART MLP3 (best snapshot; share of |W_in| column mass, s0–9): closed-form inputs (ABS, MIN, OLD_a, ZERO, A/B/R flags) .276–.285, position/clock blocks .254–.270, rest .45–.46; |W_out| mass on the router outputs (Z/G rows of A, B, R; END) .620–.629. Reading: the router spends a quarter of its input weight on position, which the closed form does not need; this position dependence is where the trained router breaks on longer traces while the frozen parts stay exact.
## Comparison
PART keeps HAND's algorithm; the trained stage matches HAND's router at its interface, not neuron by neuron. TRAIN learns subtractive Euclid with a different, seed-specific circuit: Euclid variables are linearly decodable (R² .80–.98) but interchange does not move them, unlike the r1 screen where a 6-bit net used them causally. Answer-only A learns parity and divisibility-by-3 features as in Charton (2023); the local scratchpad S fails to learn its bit bricks. Rai et al. (2026): one circuit fitted over all rows can mix mechanisms, the likely reason T1 has no single closed form.
## Controls and limits
Teacher-forced scoring, not free rollouts; diag tuples are all exact, so the bucket test has one bucket. Interchange tested only compare on most PART seeds (nf 0 = untested); PART variable use rests on the circuit, complement and substitution tests. One wd (.1) in the final round; no HAND-b/-c. 8-bit sets are samples (256 tuples), not exhaustive. Inverse and fraction cells are side cells, not part of this entry's claims.
## References
- Charton, *Learning the greatest common divisor* (preprint, ICLR 2024) — https://arxiv.org/abs/2308.15594
- Power et al., *Grokking* (preprint) — https://arxiv.org/abs/2201.02177 ; Nye et al., *Show Your Work* (preprint) — https://arxiv.org/abs/2112.00114
- Rai, Geva, Yao, *Data-driven Circuit Discovery for Interpretability of Language Models* (preprint, 2026) — https://doi.org/10.48550/arxiv.2605.09129
