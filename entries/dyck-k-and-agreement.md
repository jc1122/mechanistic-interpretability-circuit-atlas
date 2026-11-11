# Dyck-k and subject-verb agreement

**Task.** Dyck-2 / Dyck-3 recognition (depth <= 4, n 1..40) and main-verb number agreement (<= 3 subjects, <= 2 PP attractors). A `BOS s_1..s_n = ANS`; B scratchpad `BOS s_1 S_1 .. s_n S_n = ANS`, S_t = X (dead), E (empty) or C<c> (top opener type / open subject number). Test = held-out words (hash split); extra evals n80, n160, depth 5-8 (d8), agree `wide` (4-5 subjects or 3-4 attractors).
**Status.** Screening round, 3 seeds per cell, plus a final 10-seed run of Dyck-2 scratchpad TRAIN at wd .1 (10/10 solved), 2026-11-11; 3 layers x 2 heads, d 64, MLP 64, NoPE, no LayerNorm; TRAIN 80k steps, PART 20k.

## Summary
Five circuit types. Agreement A nets copy the first noun (9/9 TRAIN, 2/3 PART; closed form accepted). Agreement B PART learns HAND-a's depth-match head (2/3; accepted). Dyck B nets learn a dense non-HAND state circuit; in the final run 8/10 seeds have a "pop pointer" head (exact .90-1.0 under substitution), but a second layer-2 head is also needed and is neither a previous-state nor a stack-top head, so no closed form is accepted. Dyck A nets solve 4/18 seeds with a count circuit without a stack state; dyck PART cells are unsolved (zero gradient through frozen step units).

## Setup
Shared architecture and instrumentation: [METHODS.md](../METHODS.md). Words drawn fresh every step (balanced / one-symbol mutation / prefix / uniform); agree sentences from a fixed pool (~80% train). AdamW lr 1e-3, warmup 1000, cosine, clip 1, batch 128, wd sweep .1/.3/1 (PART .1); TF32 TRAIN, fp32 PART and evals; GPU class: consumer CUDA/XPU cards. Cells: {dyck2, dyck3, agree} x {A, B} x {PART, TRAIN}. Networks: huggingface.co/jc1122/circuit-atlas-nets/tree/main/dyck-k-and-agreement.

## Architecture as used
GPT-NeoX names, d_model 64, 2 heads x 32, ReLU MLP 64, vocab 16 (0 BOS, 1 PAD, 2 `=`, 3 Y, 4 N, 5 X, 6 E, 7+c C<c>, 10+s symbol s). PART: HAND-a frozen, the match-head attention layer retrained (dyck layer 1, agree layer 2: query_key_value [192, 64] + dense [64, 64]), masked to HAND's input and output dims. TRAIN: all weights. Archive: HAND-a/-c per (lang, fmt), PART agree A/B and TRAIN best-wd cells, best + final s0-2, final-run dyck2 B best + final s0-9, safetensors + config.json + manifest.json (seed, rung, wd, circuit type, acc, sha256).

## Prediction before training
Written before r1. Held: agree A solved 3/3 via the first-noun shortcut, wide 1.0; dyck B TRAIN solved (predicted >= 2/3 dyck2, >= 1/3 dyck3; got 6/6 each at wd .1/.3), n80 .3-.9. Missed: dyck B PART predicted solved (0/6, can't fit); dyck A TRAIN predicted unsolved balance shortcut (4/18 solved, no shortcut: type swaps answered); agree B TRAIN predicted memorising (7/9 solved); d8 < .3 (got .41-.73); HAND depth match in dyck B (none).

## Hand-built reference
L0H0 BOS-sink count head -> MLP0 one-hot depth DEP_m and opener depth ODEP_m (step pairs); dyck L1H0 state head attends the latest opener whose depth equals b (HAND-a; HAND-c: <= b), score BETA [ODEP = target] + EPS T, value OT_c -> TOP_c; L1H1 check head (target b + 1) -> MT_c; MLP1 V = [closer c](1 - MT_c); L2H0 any-violation head NV; MLP2 DEAD, GC/GE readouts, OUTY = relu(DEP_0 - 4 NV + G EQ - G). agree: L1H0 latest-p head -> PP flag; L2H0 state head with PP nouns at -2 BETA. Constants BETA 1e4, EPS 20, G 2000, K 4, U 20. Exact on 113180 (dyck2), 83286 (dyck3), 62762 (agree) inputs, both formats, both variants.

## What training finds
- Setups: TRAIN (all weights); PART (HAND-a frozen + trained match-head attention layer).
- Learning curve: agree A solved by step 500-1000; agree B TRAIN test90 3500-6500, solve 5500-16500; dyck B test90 4000-12000, solve 13000-33000 (final run: test90 4000-7000, best 13000-22500); dyck A test90 1000-2000, solve 1000-15000.
- Circuit types: A-first 11/11 solved agree A seeds; B-HANDa 2/2 agree B PART; B-pop final dyck2 B 8/10 (screening: dyck2 3/3, dyck3 1/3 at wd .1); B-dense final 2/10 + screening dyck3 B 2/3, agree B TRAIN 7/7, 6 untested wd-.3 dyck seeds; A-count 4/4 solved dyck A seeds.
- Frozen parts read (PART): agree A reads only the token-1 path; agree B reads MLP1's PP flag (path MLP1>L2H0 drop .16); dyck PART reads nothing (constant answer, or TOP right and MT never learned).

| cell (best wd) | solved | best / final test | used comps | best probe R^2 (layer) | eff. rank (PCA r90, HAND pc R^2) | attn JS to HAND | discover edges, faith |
|---|---|---|---|---|---|---|---|
| agree A TRAIN .3 | 3/3 | 1 / 1 | 1-7 of 9 | answer .98 (L3); top .39 (L1) | 1, .39 (L3) | .26-.51 | 12-26, .96-1.0 |
| agree A PART | 2/3 | 1, 1, 0 | 1-2 | answer flip 1.0 (L3) | - | - | 3, 1.0 |
| agree B PART | 2/3 | 1, 1, 0 | 6 | top flip 1.0 (L3); HAND-a iia .91 | - | - | 10, .98 |
| agree B TRAIN .3 | 3/3 | 1 / 1 | 9 | top flip 1.0 (L3); HAND-a iia .33 | - | - | 36-41, .90-.92 |
| dyck2 B TRAIN .1 (final, 10 seeds) | 10/10 | 1 / 1 | 7-9 | top .93-.98, dead .84-.97 (L3); b .53-.80; MT at L0 value | - | HAND-a iia .19-.30 | 35-45, .865-.995 |
| dyck3 B TRAIN .1 | 3/3 | 1 / 1 | 9 | top .86, dead .95 (L3) | - | - | 45-47, .92-.97 |
| dyck2 A TRAIN .1 | 1/3 | .999-1 / .998-1 | 6-8 | answer .85 (L3); top .65 (L0) | - | - | 21-34, .92-.98 |
| dyck3 A TRAIN .1 | 2/3 | .999-1 / .999-1 | 6-8 | b .70 (L1), answer .71 (L1) | 1, .72 (L3) | .52-.61 | 27-31, .93-.96 |
| dyck2/3 A, B PART | 0/6 each | .47-.73 (= step 0 for A) | 0 (A), 7 (B) | B: top .99 (L2); MT at majority | - | - | 0-10, .04-.11 |

## Closed form
- A-first: ANS = verb of number(s_1); a layer-2 head at `=` attends one-hot to token 1. Substitution (both L2 heads forced to token 1): exact 1.0 in 11/11 seeds; complement (all but L2H1, MLP2): resample .945 of 1.0 -> accepted.
- B-HANDa: trained layer-2 attention = HAND-a state head. Substitution 1.0 (wide 1.0 vs trained .20-.24); complement 1.0 -> accepted.
- B-pop (partial): at an alive closer t the pointer head attends one-hot to S_(j-1), the state written before the matched opener j (BOS if j = 1), and copies it. Substitution .90-1.0 in 8/10 final seeds; complement without the other L2 head .17 / .06 (screening .27 / .14) -> not accepted. The other head is needed at every scored token; forcing it to the previous state or to the matched opener fails (pairs <= .88 in 9/10).
- B-dense, A-count: none found.

## Weights excerpt and reading
Printed by the archive's excerpts (`dyck`): HAND-a vs HAND-c query rows of the match head, and per archived net the direct type vote (U[o1] - U[o0]) W_O W_V (E[i1] - E[i0]) of each head (weight slices). HAND: the query row is one-hot in HAND-a (11 nonzero entries) and a thermometer in HAND-c (66), all at the same scale (56568.5) in dyck2, dyck3 and agree. Final run (dyck2 B, 10 seeds): the direct vote sits in layer 0 (largest head 1.3 to 12.3 in size per seed) while every layer-1 and layer-2 head is within 0.5 of zero, so the layer-2 pop pointer and its partner head act through where they attend (QK), not through a direct type vote; seeds of both circuit types look alike here (B-dense s0 and s4: L0 +2.1 and +8.1). r1 agree A (A-first): every head within 0.1 of zero, so the weight slice does not show the A-first vote; the complement and patching results above carry that claim. PART nets print 0.0 in every head (no type vote in this slice).

## Comparison
B-HANDa reproduces Yao et al. 2021's depth-match head inside the frozen scaffold. From scratch, no net learns it: the B nets build a state circuit in which one head can point to the pre-opener state (a recurrence through the scratchpad tokens, not a depth match), and agree A nets take the first-noun shortcut that is exact in and out of range here. Sharma & Raval 2026 find Dyck nets route to the stack top itself; our pointer reads the state before it instead. Dyck B length generalisation is weak (final n80 .31-.86, n160 .15-.22, d8 .55-.79, same for B-pop and B-dense), the in-range-only fit Bhattamishra et al. 2020 report for Transformers on several formal languages; dyck A count nets generalise better (n80 .94-1.0, d8 >= .98).

## Controls and limits
3 seeds per screening cell; 10 seeds only for dyck2 B wd .1. Probes and interchange are on 128 rows; A-count rests on 4 seeds and a small interchange (n_chg 18). The pop pointer was tested only at wd .1; the second L2 head of B-pop is unidentified; the final seeds are not one basin (lmc barrier .81-2.97). Single-head overrides in A-first leave unused heads unchanged, so specificity rests on complement and patching. agree PART s2 (A and B) is degenerate from init.

## References
- Yao et al. 2021, Self-attention networks can process bounded hierarchical languages: https://arxiv.org/abs/2105.11115 (arXiv)
- Bhattamishra et al. 2020, On the ability and limitations of Transformers to recognize formal languages: https://doi.org/10.18653/v1/2020.emnlp-main.576 (conference paper)
- Gan et al. 2026, survey citing Wen et al. 2023 on many attention patterns for bounded Dyck: https://doi.org/10.48550/arxiv.2601.02907 (arXiv)
- Sharma & Raval 2026, Dissociating decodability and causal use in bracket-sequence Transformers: https://doi.org/10.48550/arxiv.2604.22128 (arXiv)
- Bengio et al. 2013, straight-through estimator (PART failure): https://doi.org/10.48550/arxiv.1308.3432 (arXiv)
