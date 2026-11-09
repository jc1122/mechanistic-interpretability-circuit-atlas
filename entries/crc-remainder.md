# CRC remainder (GF(2) polynomial N mod D): local scratchpad step vs answer-only parity readout
**Task.** N = n input bits (n 8/9/10), D monic degree 4 (16-31; D 19, 25, 28, 30 withheld); held-out (N, D) pairs by a width-invariant 80/20 hash. Answer-only A emits the 4 remainder bits; scratchpad T writes the quotient bit and the full remainder after each input bit (5n bits + n-1 `;`).
**Status.** HAND A/T; PART T 10 seeds, PART A 3, TRAIN A 3; TRAIN T final run 10 seeds (plus 3 screening seeds); 30k budget; 2026-11-09.
## Summary
With frozen hand-built routing the trained stage reproduces the hand-built circuit edge for edge in both formats (PART T 10/10, A 3/3; discovered circuit = HAND's, Jaccard 1) and a table in HAND variables replaces it exactly. From scratch, T learns the scratchpad step in 10/10 final seeds (3/3 screening) on a 13-71-edge circuit that contains HAND's edges (recall .67-1); its readout closed form and circuit pass both acceptance checks in 10/10. Answer-only A from scratch (2/3) uses about half of all edges, matches no HAND, and has no accepted closed form.
## Setup
[METHODS.md](../METHODS.md) for shared conventions. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1,000, cosine, clip 1, batch 256 fresh pairs per step, 30k budget, plateau 5,000 / minimum 22.5k, eval every 500. PART fp32; TRAIN TF32 updates, fp32 evals. GPU training, CPU fp32 analysis. Diagnostics on 256 n10 held-out rows; substitution on 3 x 512 rows (n 8/9/10). Public reproduction code is pending; the archived networks re-score to the recorded accuracies.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/crc-remainder](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/crc-remainder) (safetensors, config.json, manifest with seed, rung, accuracy, sha256).
## Architecture as used
GPT-NeoX, ReLU, no LayerNorm, no positional encoding, vocabulary 7, d_model 128. A: 4 layers x 16 heads (d_head 8), MLP 1024. T: 3 layers x 4 heads (d_head 32), MLP 256. PART trains only the stage MLP (A layer 3, T layer 2; W_in [d_mlp, 128] masked to HAND's read coordinates), all else frozen HAND. TRAIN: same shapes from random init. Files: `hand_{A,T}`, `part_{fmt}_{best,final}_s{i}` (stage tensors only), `train_{fmt}_{best,final}_s{i}` (screening, full), `train_T10_{best,final}_s{0-9}` (T final run, full) safetensors + config.json + manifest (seed, rung, accuracy, sha256): https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/crc-remainder
## Prediction before training
Committed before any smoke run. T PART by 2.5k: held (500). A PART by 15k with poor unseen-D/n12: held. T TRAIN all seeds by 15k: solved 10/10 final, but first exact by 15k in only 2/10 (14-19.5k). A TRAIN >=1/3 by 30k: held (2/3). T TRAIN fails n12: held (max .26). Unseen D >= .95 in most seeds: held (7/10).
## Hand-built reference
T (3L, 7 used heads + 236 neurons): fetch the previous row's shift bit, input bit, divisor bit and quotient bit, then one MLP computes next = shift XOR (divisor_bit AND quotient) (10 stage neurons, W_in values +-1, -2). A (4L, 18 heads + 344 neurons): 12 L2 heads fetch input bits NB_i, 4 fetch divisor bits; MLP3 holds 262 parity-staircase neurons, one group per (D, bit j) reading S(D, j) = {i: bit j of x^i mod D} (W_in 1, gates 32). Exact on 9,568 sequences per format (n<=6 exhaustive, n7-12 stratified), margin 30.
## What training finds
- **T PART** (10/10, step 500): type *HAND XOR-AND stage* 3/3 tested: discovered circuit 9/118 edges = HAND's; resample of the complement keeps 1.0, resampling the circuit drops to 0.
- **A PART** (3/3, first exact 5.5-7k): type *HAND parity readout* 3/3: 30/1935 edges = HAND's; the top MLP3 neurons read exactly HAND's support sets (D 18, bit 3: NB 3, 6, 9). Probes are weak (R^2 .52-.57) yet the stage writes a pure bit code.
- **T TRAIN** (final 10/10, all exact at the joint stop 20.5k; first exact 14-19.5k): type *distributed scratchpad step* 10/10 (and 3/3 screening): shift, quotient and next used at L3 (interchange .98-1, DAS 1); L2 heads fetch, MLP2 writes the bit. Discovered 13-71/118 edges, HAND edge recall .67-1, Jaccard .10-.38. Size varies, mechanism does not: compact s1, s5, s6 (13-28 edges, at most 2 layer-0/1 heads) vs wide (42-71 edges). Path MLP1 -> L2 heads -> MLP2 (drop .32-.66, .68-1); MLP1 -> MLP2 directly <= .06. Held-out exact oscillated after the first exact (s1 1.0 -> .375) before all seeds were exact together.
- **A TRAIN** (2/3): type *dense non-HAND readout* (2/2 solved, failed s0 the same): 848-1000/1935 edges, Jaccard .03; count/next used only at L4 (DAS ~1). Failed s0 (train 1, test .99544 -> .99349): 3/256 rows wrong at bit 3 for D 22/29, a divisor-specific MLP3 readout gap.
- Diagnostics (final; E = 256-row exact; H; M = used heads / MLPs, mean ablation; n90 = neurons for 90% of ablation effect; probe = best R^2 per HAND var @layer; r99 = residual rank vs HAND; JS = attention JS vs HAND):

| Cell | E best/final | H; M (n90) | Best probe R^2 | r99 vs HAND; JS | Discovered edges; faith |
|---|---|---|---|---|---|
| T PART s0-2 | 1/1 | 4; 1 | shift .88-.91, Q .90-.91, next .88-.89 @L3 | 9 vs 10; 0 | 9; 1 |
| A PART s0-2 | 1, 1, .996 | 14; 2 | count .52-.57, next .51-.60 @L4 | 25; 0 | 30; .99-1 |
| T TRAIN final s0-9 | 1/1 (10/10) | 3-8; 2-3 (255-360) | shift .83-.97, divisor bit .50-.93, Q .83-.99, AND .55-.89, next .73-.95 @L3 | 3-13 vs 10; L0-1 <= .20, L2 .06-.50 | 13-71; .89-.99 |
| A TRAIN s1, s2 | 1/1 | 16-20; 4 | count .74, next .63-.72 @L4 | 28-33; <=.47 | 848, 1000; .92-.95 |
| A TRAIN s0 (fail) | .988/.988 | 20-21; 4 | count .78, next .68-.70 @L4 | 40; <=.45 | 989; .91 |
## Closed form
Substitution: the last MLP's output at every answer position replaced by a table in HAND variables (fit and test on separate batches); acceptance needs substitution exact and complement-resample exact >= .90 of the net's.
- T PART: MLP2 = W[phase, col, next], next = shift XOR (divisor_bit AND quotient): 1.0 (6/6 nets, also with the 35-key HAND-input table); resample 1.0. Accepted.
- A PART: MLP3 = W[j, next], next = XOR_{i in S(D, j)} NB_i: 1.0 (6/6); resample 1.0. Accepted.
- T TRAIN final: same readout table on MLP2: 1.0 in 19/20 best/final nets (final s7 .969, only n10 drops), same with the 35-key HAND-input table; resample of the complement of each seed's discovered circuit .915-.994, circuit resampled and random same-size circuit <= .003. Accepted 10/10 for the readout and circuit; layers 0-1 and the L2 heads stay trained (no formula).
- A TRAIN: table .84/.69 (s1/s2), claimed-component resample .09/.15. Rejected. The failed s0 keeps .998 under the table.
## Weights excerpt and reading
Weight slices of the stage MLP, final snapshot. HAND T: 10 stage neurons writing the answer, W_in values {-2, -1, 1}. PART T s0 top neurons by |W_out| x |W_in|, W_in on [X, A, DB, QB, QF, LOW, SEP] | bias -> W_out[0, 1]: n141 [-.06, -.08, -.08, .01, -.01, .02, -.04] | .05 -> [.09, -.09]; n194 [.04, .07, -.03, -.04, -.03, -.03, -.07] | .03 -> [-.12, .10]. Many small units share each truth-table cell. TRAIN T final: 168-194 of 256 MLP2 neurons hold 90% of the |W_out| x |W_in| mass (HAND: 10). HAND A: 262 neurons, W_in {1, 32}. PART A W_in on NB_0..9 (argmax divisor, bit):
| seed | neuron | NB weights ~ -.4 at | D, bit | HAND S(D, bit) within NB_0..9 |
|---|---|---|---|---|
| s0 | n730 | 3, 6, 9 | 18, 3 | 3, 6, 9 |
| s1 | n213 | 2, 5, 8 | 18, 2 | 2, 5, 8 |
| s2 | n524 | 1, 4, 6, 9 | 31, 1 | 1, 4, 6, 9 |
Each such neuron is a native parity-support detector: equal weights on exactly HAND's support set, read out per (D, bit).
## Comparison
| A <-> T | Solved | Circuit vs HAND | First E>=.99 | n11 (final) | n12 | unseen D |
|---|---|---|---|---|---|---|
| PART | 3/3 <-> 10/10 | = HAND <-> = HAND | 3.5-5.5k <-> .5k | .09 <-> 1 | .11-.14 <-> 1 | .07-.09 <-> 1 |
| TRAIN | 2/3 <-> 10/10 final | none <-> HAND superset | 20.5-23k <-> 14-19.5k | .13-.23 <-> .13-.98 | .09-.13 <-> 0-.26 | .12-.15 <-> 1 in 7/10 (.30-.75) |
In range (n 8-10) every solved net is exact; T TRAIN length transfer stays partial (n11 >= .95 in 3/10, short n4 <= .26, n6 <= .71). The scratchpad turns the task into a local step that every T seed learns and wires around HAND's edges; answer-only training spreads one parity readout over half the network. Dense trained nets spreading circuits matches Gao et al. 2025.
## Controls and limits
Complement checks pass their self-tests (full = base, empty = source) in every net. The readout table's key includes the emitted bit: it shows a pure bit code, not how next is computed; the HAND-input T key and the A support neurons carry the computation. HAND-free discovery on 256 n10 rows, final snapshot. One HAND per format (no alternative HAND tested). A TRAIN widths 8-12 retry: 0/3 (negative). T TRAIN final: one seed's dynamics only to step 7,500; device not logged. Open: public code.
## References
- Nye et al., *Show Your Work: Scratchpads for Intermediate Computation* (preprint) - https://arxiv.org/abs/2112.00114
- Jelassi et al., *Length Generalization in Arithmetic Transformers* (preprint) - https://doi.org/10.48550/arxiv.2306.15400
- Gao et al., *Weight-sparse transformers have interpretable circuits* (preprint) - https://doi.org/10.48550/arxiv.2511.13653
