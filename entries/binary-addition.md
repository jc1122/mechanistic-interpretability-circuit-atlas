# Binary addition: a one-head carry lookahead by hand; trained nets carry without it and stage MLPs read position unless it is masked
**Task.** `a+b=` → a + b, n-bit binary. A: s_n..s_0 (MSB first); B: s_0 k_1 s_1 .. k_n (LSB first, carry token k ∈ {`c`, `.`});
C: column sums d_{n-1}..d_0 ∈ {0,1,2}, `=`, s_n..s_0. Training data: all 4096 pairs at n = 6, 40% train / 60% test; exact match of all output tokens.
**Status.** 2026-11-05. HAND per format; PART 10 seeds × 3 formats (+ position-mask retry for B, C); TRAIN 10 seeds × 3 formats, failed cells retried (all runs kept).
## Summary
HAND carries in constant depth: one head attends to the first lower column with a_j = b_j (A) or d_j ≠ 1 (C) and copies the carry; B rips
through the written carry tokens. Every working trained stage computes the same sum-bit function f(a + b + c) (substitution = net accuracy,
A 10/10, C 10/10, B 3/10 → 6/10 with the mask). PART B/C stages also read absolute position, which breaks length; holding their position inputs
at 0 makes PART C exact at n = 12 in 10/10 (was 0). TRAIN solves all formats (A 10/10, B 10/10, C 9/10 and 10/10) but never forms the
lookahead head and fails at n = 12 in every seed.
## Setup
[METHODS.md](../METHODS.md). AdamW, betas (.9, .98), batch 256, CE on output tokens, eval every 500, seed = init + split. PART: stage MLP
trained 10k steps, lr 1e-3, wd 1 (TF32; mask retry fp32). TRAIN: run 1 20k, lr 1e-3, wd 1, final weights; run 2 30k + clip 1, best-train
snapshot; run 3 (A, C) lr 3e-4, wd .1, warmup 1000 + cosine, clip 1; run 4 (A) lr 1e-3, wd .1, warmup 1000 + cosine over 40k, clip 1.
Length: fp32 CPU, 1000 random pairs at n = 7, 8, 12 (TF32 breaks HAND from n = 7). One GPU class (consumer 10-12 GB).
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/binary-addition](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/binary-addition) (62: 3 HAND, 30 PART, 20 PART with position mask, 9 TRAIN).
**Reproduce:** the training and analysis code for this task is not yet in `code/` (`python run.py` has no addition task), so the learned cells cannot be rerun from this repository; the weights and the table of accuracies are in the folder above.
## Architecture as used
GPT-NeoX, 4 layers × 4 heads, d_model 256, d_head 64, MLP 256 ReLU, no LayerNorm, NoPE, vocabulary 9 (BOS, `0 1 2 + = . c`), 1,583,616 params,
same for every rung. PART trains only the stage MLP (A layer 3, B/C layer 2: dense_h_to_4h [256, 256] + bias, dense_4h_to_h [256, 256] + bias);
it reads TOK, X1/X2 (operand bits), X3 (C: d_i), CY (carry), PH (C phase) and the position blocks P, R, R2, RP, T, VT (zeroed columns in the mask retry).
Archive (Hugging Face folder above): `hand_{A,B,C}`, `part_<F>_s<seed>` and `partmask_<F>_s<seed>` (4 stage tensors + test_idx, all seeds), `train_<F>_s<seed>`
(full nets, 3 seeds per format: A 0/2/7, B 2/5/8, C 1/2/9), `config.json`, `manifest.json` (sha256); each net reloaded in fp32 and checked against its recorded test accuracy.
## Prediction before training
Recorded in the design notes before any run: sum = XOR3(a, b, c), carry = MAJ(a, b, c); A needs a carry-chain detector. Nothing was recorded for
PART/TRAIN (lookup vs sum, length, grokking); "PART stage = parity of a + b + c" is retrospective. Held for A, C (and B with the mask).
## Hand-built reference
Positions: uniform head → 1/(r+1), hinge MLP → r, anchor head on `+` → r_plus, hinge MLPs → r², thermometers T[m] = [r ≥ m], VT[m] = [r − r_plus ≥ m].
Index heads score −20(r_j − t)² (t linear in r, r_plus) and copy the bit at t. A: lookahead head on the first j < i with a_j = b_j copies a_j
(= carry into i; `=` sentinel = 0); MLP → parity(a_i + b_i + c). B: MLP → parity, MAJ; previous-token head copies MAJ into the carry token.
C: lookahead on the first d_j ≠ 1 copies [d_j = 2]; MLP → digit of a_i + b_i (phase 1), parity(d_i + c) (phase 2). Used 3 + 8 / 3 + 6 / 4 + 10 heads + neurons. Exact on all pairs n = 1..6 and 2000 pairs per n = 7..12, minimum logit margin 30.
## What training finds
- PART: test = train from step 500 (no grokking). A 10/10 f(sum), exact at n = 12. C 10/10 f(sum, phase) but n = 12 = 0; B lookup(a, b, c, pos)
  7/10. Mask retry: C n = 7/8/12 = 1.00 in 10/10 (was .17-.20 / .01-.02 / 0; 49-87 neurons); B n = 7 .99-1.00, n = 12 .53-1.00 (was .00-1.00), f(sum) 6/10. The frozen heads keep their roles (attn sub 1.00, 30/30).
- TRAIN A: r2 (wd 1) 4/10, r3 (lr 3e-4) 1/10 (memorizes: train 1.0, test .53-.98), r4 (default) 10/10, test .9 at 4.5-10.5k, ≥ .99 at 8-14k.
  9-13 heads + 254-591 neurons, heads anchored on `=`/BOS (8/10), a+0/b+0 index heads 2/10. TRAIN B r2 10/10 (≥ .99 at 3-8.5k), index heads
  a+0, b+0 in 6/10 (+ carry-token head 2/10). TRAIN C r2 9/10 (6.5-19.5k), index heads on a/b/d 7/9, `=2` phase anchors 2/9; r3 10/10, 8-16 heads.
- Length (TRAIN, all cells): n = 7 .00-.76, n = 8 ≤ .07, n = 12 ≤ .001.
- Diagnostics (used heads+MLPs; probe R² at L4 of a / b / carry-in c / sum bit (C: + d, PH); residual PR L4; JS of used heads to best HAND head; test best (final)):

| cell | HAND | PART | PART mask | TRAIN failed run | TRAIN final run |
|---|---|---|---|---|---|
| A | 3+2; 1/1/1/1; PR 5.5; 0; 1 | 3+2; 1/1/1/.94-1; 4.7; 0; 1 (1) | – | r3: 16+4; .38-.45/.36-.46/.49-.64/.53-.71; 1.4-3.4; .05-.18; .53-.98 | r4: 9-13+4; .58-.78/.57-.78/.71-.86/.84-.91; 1.2-2.6; .11-.24; .99-1 (= best) |
| B | 3+2; .83/.82/.72/1; 7.3; 0; 1 | 3+2; .81-.83/.81-.83/.75-.82/.95-.99; 5.3-5.8; .14-.16; 1 (1) | 3+2; .80-.84/.81-.84/.71-.83/.98-1; 5.5-5.7; .14-.16; 1 (1) | r1: 0-4+0-4; .24-.44/.19-.46/.75-.86/.50-.68; 1.1-1.6; 0-.29; ≥ .99 in 7/10 (.01-.19) | r2: 2-5+3-4; .44-.98/.45-.99/.73-.96/.90-.98; 2.0-3.0; .20-.40; 1 |
| C | 4+1; .62/.61/.60/.52, d .83, PH 1; 4.6; 0; 1 | 4+1; .62-.67/.64-.68/.60-.61/.53-.57; 4.1-4.3; 0; 1 (1) | 4+1; .64-.69/.66-.71/.60-.61/.53-.56; 4.1; 0; 1 (1) | r1: 0-16+0-4; .25-.41/.25-.49/.39-.59/.16-.42; 1.9-3.6; 0-.16; (.01-.62) | r2: 3-13+4; .38-.96/.38-.95/.46-.64/.47-.57, PH .97-1; 2.9-4.2; .19-.35; .91-1 |
- Failures explained: A r3 lacks the operand and carry bits at the query (a/b ≤ .46, c ≤ .64) and has full-rank W_in (159-184): memorization;
  r4's higher peak lr builds c (.71-.86) and generalizes. PART B/C length: stages read the T/VT thermometers as a phase/slot cue (C read
  subspace overlap with HAND .36-.39); masking them removes the failure in C. TRAIN length: NoPE with one training length.
## Closed form
Stage MLP replaced by its mean output given a + b + c (C: and phase): PART A 1.00 (10/10), C 1.00 (10/10, also mask 10/10), B 1.00 in 3/10
(mask 6/10); given (a, b, c, slot position) 1.00 in 30/30 PART nets. sum bit = (a + b + c) mod 2.
## Weights excerpt and reading
Numbers computed from the archived files (logit of output 0/1 at the sum-bit query, from the stage output through the unembedding, per value s = a + b + c):
HAND A [s=0: 30/−30; 1: −30/30; 2: 30/−30; 3: −30/30]; PART A s1 [18/−1; −1/18; 18/0; 0/18], s0 [20/7; 7/19; 19/7; 7/19]. Every PART A seed
writes the parity of s with margins 11-20 (HAND 60); the lookahead head (L3H0) has 96 non-zero query and 50 key weights, value 1 non-zero.
C seeds: phase-1 digit logits peak on the true digit (s=1: 7/19/4), phase-2 bits are parity of d + c. B's MAJ write rises gradually with s (s=0 about 0, s=3 .40-.74) instead of HAND's 0/1 step at s = 2.
## Comparison
Same arithmetic (f(sum)), different carry: HAND uses one lookahead head; trained nets spread the carry over several heads and MLPs, as in [decimal-hex-addition](decimal-hex-addition.md). The PART position read is a shortcut available only because one length was trained.
## Controls and limits
One training length; 1000-pair analysis subsets; TF32 TRAIN nets evaluated in fp32 (±1 pair). Final weights exist only for TRAIN r1 and r4 (r4 final = snapshot); r2/r3 report the best-train snapshot. TRAIN circuit types are attention-mass labels, not substitution-verified.

## References
- Quirke, P., & Barez, F. (2023). Understanding addition in transformers. https://doi.org/10.48550/arxiv.2310.13121 (per-digit streams, carry heads)
- McLeish, S., et al. (2024). Transformers can do arithmetic with the right embeddings. https://doi.org/10.48550/arxiv.2405.17399 (position inputs and length generalisation)
- Jelassi, S., et al. (2023). Length generalization in arithmetic transformers. https://doi.org/10.48550/arxiv.2306.15400
- Shen, R., et al. (2023). Positional description matters for transformers arithmetic. https://doi.org/10.48550/arxiv.2311.14737
