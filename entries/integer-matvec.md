# Integer matrix-vector product (m=3, entries 0..7): the scratchpad trains 8/10, answer-only needs 4x the budget
**Task.** y = Mv, square m=3, entries 0..7, each y_i as nine LSB-first bits (<= 196). **A** emits the m answers (46 tokens); **T** writes the row
accumulator after every shifted partial product (27 steps, 286 tokens). Fresh inputs every batch; disjoint 2,000-input test set per seed.
**Status.** Final run (2026-11-12): T TRAIN, 10 seeds, 30k steps; screening cells 3 seeds. 5L x 4H x 32, d_model 128, MLP 640, NoPE, no LN.
## Summary
T TRAIN solves 8/10 in the final 10-seed run (s7 .9995, s9 .815); T PART 3/3; A PART 0/3; A TRAIN needed three 2x budget retries (30k -> 120k)
to solve 1/3. Every solved T TRAIN net uses HAND-b's variables (accumulator bit x, carry, next bit) only at the last layer, with its own dense
circuit (no HAND edge match). No TRAIN closed form passes substitution: linear decoders of the operands miss. No trained T net works at m=2 or m=4.
## Setup
Shared methods: [METHODS.md](../METHODS.md). AdamW lr 1e-3, wd .1, betas (.9,.98), warmup 1000, cosine, clip 1, plateau 5000, ckpt 2500.
PART = HAND with the last MLP re-initialised (fp32); TRAIN from scratch (TF32 flag, fp32 master). T TRAIN 30k steps, batch 256 (7.68M examples/seed);
A TRAIN 120k planned (plateau stop 93k), batch 128; A PART 30k. Teacher-forced whole-output exact match. Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/integer-matvec](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/integer-matvec); public reproduction code is pending, the archived networks re-score to the recorded accuracies.
## Architecture as used
GPT-NeoX decoder, 5 layers, 4 heads x d_head 32, d_model 128, ReLU MLP 640, vocab 15 (BOS, unk, d0-d7, bits 0/1, `; , =`), no positional
encoding, no LayerNorm. PART trains only L4 MLP (W_in [640,128], W_out [128,640]); TRAIN trains all weights. Archived: `hand_A`, `hand_T`, TRAIN best + final of T r1 s0-2, A r3 s0-2 and the T final s0-9 (PART nets not archived), `manifest.json` (seed, round, circuit type, accuracy, sha256).
## Prediction before training
Committed: PART T 10/10 and A >= 8/10 by 10k; TRAIN T >= 7/10, A <= 3/10 by 30k; no TRAIN seed > .5 at m=4. Held: T PART (3/3 by 1k), T TRAIN
(9/10 >= .99, 8/10 exact by 30k), A TRAIN <= 3/10 at 30k (0/3), m=4 (0 in all 10 T seeds). Missed: A PART (0/3). Held-out-value arm not run.
## Hand-built reference
A (HAND-a): vector-bit fetch heads -> gated products relu(M_ij + 8 bit(v_j,b) - 8) -> uniform row attention (mean) -> rescale by m -> binary
staircase (6 heads, 769 neurons). T (HAND-b): position frame from BOS frequency, delimiter heads, operand fetch (L3H0) and accumulator bits
x, x' (L3H1/H2), selectors a, a', b (MLP3), 32-neuron SUM3 table (MLP4); 8 heads, 1,205 neurons. Both exact on 770/1,410/2,306 stratified
inputs at m=2/3/4, minimum logit margin ~29.7/30.0. No third HAND was designed.
## What training finds
| cell | solved (exact 1.0) | best / final test | grok step (first 1.0) | circuit type (seeds) |
|---|---|---|---|---|
| PART T | 3/3 | 1 / 1 | <= 1,000 | T1 frozen HAND-b routing + carry/parity voter bank (3/3) |
| PART A | 0/3 | .078-.12 / .078-.11 | none (22.5k) | none: staircase readout of bits 0-3 missing |
| TRAIN T (final) | 8/10 | 1 x9, .815 / 1 x8, .9995, .815 | median 17.5k (11.5k-29k) | T2 family (8/10): s0, s3, s4 three stitch-distinct types (1 each), 5 untested; s9 unsolved |
| TRAIN A (r3) | 1/3 | .9585/1/.9995 | s1 78.5k (gradual) | A1 exact products + 2-d nonlinear row-sum code (1/3; s2 .9995 same) |
## Diagnostics
| cell (seeds) | used heads; MLPs | best probe R² (L5 unless noted) | eff. rank (OV r90; MLP4 in r90) vs HAND | attn JS to HAND | best / final |
|---|---|---|---|---|---|
| PART T (s0-2) | 8 (= HAND); 5 | x,x',a,a' >= .9999 (L4); nextbit .86-.88 | OV = HAND; MLP4 95-96 vs 5 | 0 (frozen) | 1 / 1 |
| TRAIN T final s0,s4,s8 | 11, 9, 13; 5 | x .93/.91/.91, carry .66/.60/.62, nextbit .87/.84/.89, a .49/.44/.46 | 3-4; 39-46 vs 5 | 0-.63 | 1 / 1 |
| TRAIN T final s1-3,s5,s6 | 17-19; 5 | x .62-.90, carry .27-.61, nextbit .59-.86, a .35-.44 | 3-4; 35-39 vs 5 | 0-.63 | 1 / 1 |
| TRAIN T final s7 / s9 | 20 / 20; 5 | x .88/.51, carry .38/.35, nextbit .63/.57 | 3; 38/41 vs 5 | 0-.62 | 1/.9995; .815/.815 |
| TRAIN A s1 (r3) | 6: L3H0,L3H3,L4H0-3; 5 | vector .996 (L4), product .998 (L4), row_sum .70 | not measured | .28-.42 | 1 / 1 |
Causal (T final): x, nextbit, carry flip .92-1 at L5 only (2-d DAS .994-1); circuits 83-283 of 321 edges, all 5 MLPs, HAND-b edge Jaccard .03-.07;
faith >= .90 only s3 (.927); s0/s3/s4 do not stitch from L2 on (score 0, map R² .85-.99). A: row_sum flip .19-.39, 2-d DAS .997 at L5; HAND-a J .05.
## Closed form
T1: c=[x'+b·a'-s'>=1], y=(x+b·a+c) mod 2 substituted for MLP4: exact 1.0 on 2,000 saved m3 per seed (3/3); causal scrubbing 100%. Accepted.
T2 (final): the same formula on ridge-decoded x,x',a,a',b,s' (L5): exact 0-.006 vs net 1.0 on all 10 seeds (operand bits a/a'/b decode at
.69-.95, near the majority rate). Complement of the discovered circuit: s3 .932 (passes), s4 .839. Not accepted. A1: y_i = sum_j p_ij on decoded
products .52 (s1) vs 1.0, not accepted.
## Weights excerpt and reading
T PART s0 neuron 258 (fitted response): h=[.016 - .068 z_BB + .037 z_TERM - .028 z_R2 + .028 z_X + ...]+, vote +24.6: a threshold that
mixes clock and arithmetic coordinates, not an isolated carry gate. TRAIN (archived final nets, L4 direct bit vote): the top 5 of 640 neurons carry only .027-.077 of the |vote| in T (final s6: n6 -43.9, n25 +37.3), .098-.116 in A, against HAND T's 32 active neurons: the readout is spread, not a few carry gates.
## Comparison
T: HAND-b's local carry/parity variables, learned as a redundant bank (PART) or with fetch spread over L3-L4 and a 35-46-rank last MLP where HAND uses a 32-neuron table (TRAIN). A: products as HAND-a, aggregation not.
Charton 2021 trains real-valued matvec transformers with tolerance scoring; Getu 2021 bounds and trains ReLU matvec approximators; Ma et al. 2025 construct matvec transformer operators; Sworna et al. 2016 and Babu 2022 describe learned LUT-based matrix products.
Giannou et al. 2023 construct softmax-based matrix products; Mereghetti & Palano 2000 and Parekh et al. 2018 analyze threshold-circuit variants. Adjacent tasks/methods: Bai et al. 2025 report long-range dependency failures in scalar multiplication; Nye et al. 2021 study scratchpad supervision.
## Controls and limits
Final run only for T TRAIN (wd .1, one host, TF32 flag); other cells 3 seeds. Length: T TRAIN m=2 and m=4 exact 0 in all 10 seeds (token acc .42-.88);
lower/high-bit value slices of m3 .98-1 (s9 .32 high-bit). Teacher-forced metrics; not exhaustive at m=4 (8^20 inputs). Individual used neurons not
ablated. Failing linear decoders do not show a variable is absent (Geiger et al. 2021); a DAS-aligned substitution was not run.
## References
Charton 2021, Linear algebra with transformers: <https://arxiv.org/abs/2112.01898> (arXiv preprint); Getu 2021, Error bounds for a matrix-vector product approximation: <https://arxiv.org/abs/2111.12963> (arXiv preprint); Ma, Gao & Ng 2025, NLAFormer: <https://arxiv.org/abs/2508.19557> (arXiv preprint); Sworna, Ul Haque & Babu 2016, A LUT-based matrix multiplication using neural networks: <https://doi.org/10.1109/ISCAS.2016.7538964> (conference paper); Babu 2022, LUT-based matrix multiplication using neural networks: <https://doi.org/10.1201/9781003269182-15> (book chapter).
Giannou et al. 2023, Looped transformers as programmable computers: <https://arxiv.org/abs/2301.13196> (arXiv preprint); Mereghetti & Palano 2000, Threshold circuits for iterated matrix product and powering: <https://doi.org/10.1051/ita:2000105> (journal article); Parekh et al. 2018, Constant-depth and subcubic-size threshold circuits for matrix multiplication: <https://doi.org/10.1145/3210377.3210410> (conference paper).
Bai et al. 2025, Why can't transformers learn multiplication?: <https://arxiv.org/abs/2510.00184> (arXiv preprint); Nye et al. 2021, Show your work: <https://arxiv.org/abs/2112.00114> (arXiv preprint); Geiger, Lu, Icard & Potts 2021, Causal abstractions of neural networks: <https://arxiv.org/abs/2106.02997> (arXiv preprint); Telgarsky 2016, Benefits of depth in neural networks: <https://arxiv.org/abs/1602.04485> (arXiv preprint).
