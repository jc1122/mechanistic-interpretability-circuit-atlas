# Counter languages (Dyck-1, a^n b^n, a^n b^n c^n): answer-only nets count by ratio and sum violations in one or two heads
**Task.** Words over 2-3 symbols, n 1..40. A (answer only): `BOS s_1 .. s_n = Y|N` (member?). B (scratchpad): `BOS s_1 S_1 .. s_n S_n = Y|N`,
S_t = X (dead prefix) or D<d> (fewest symbols to append). Fresh words every step (member / one-symbol mutation / <= 3 runs / random); test =
2048 held-out words per seed (hash split); exact = every scored token. Length tests n 80 and 160. **Status.** 2026-11-11; screening r1-r3
(3 seeds per cell and wd); final run anbncn A wd .1, 10 seeds; 2 layers x 2 heads x 32, d 64, MLP 64, NoPE, no LayerNorm; 40k steps (B retries 80k).
## Summary
Answer-only TRAIN solves all three languages (r1 24/24 seeds at wd .1/.3, dyck1/anbn also wd 1; final anbncn wd .1 10/10): L0 heads average
the symbols (a ratio count), MLP0 writes violation flags, an L1 head sums them, MLP1 answers Y iff d = 0 and no violation (HAND's algorithm).
Two types: one any-head (final 7/10) or the sum split over both L1 heads (3/10). MLP1 = a 4-entry table T[Z0, DEAD]: substitution 1.0 in
10/10; resampling outside the discovered edges keeps >= .90 in 9/10. Scratchpad TRAIN: only anbn solves (1/3), via the previous state token.
## Setup
AdamW lr 1e-3, betas (.9, .98), warmup 1000 + cosine, clip 1, batch 128, eval every 500, plateau 5000 after 30k; wd sweep .1/.3/1 (final .1);
TF32 training, fp32 evals; PART = HAND with its L0 MLP retrained, 20k fp32; consumer GPUs. Training code is not public; keys `best_test`, `ext`. Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/counter-languages](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/counter-languages).
## Architecture as used
GPT-NeoX 2 layers x 2 heads x 32 (d 64), MLP 64 ReLU, NoPE, no LayerNorm, untied lm_head; vocab 6 + k + DMAX + 1 (k symbols, D0..D160/D320).
TRAIN: all weights trained from the HF init. PART: frozen HAND, trained L0 MLP W_in [64, 64] (masked to SYM and CNT dims), W_out [64, 64].
Archive `counter-languages/`: HAND per language x format, TRAIN best + final s0-2 of the solved r1 cells and s0-9 of the final run (safetensors).
## Prediction before training
Written before each round. r1: dyck1 A TRAIN not solved: failed (9/9 solved). anbn A >= 1/3: held (9/9). anbncn A not solved: failed (6/9).
B TRAIN >= 2/3 (dyck1, anbn): failed (anbn 1/3; dyck1 0/21 over r1-r3); anbncn B not solved: held (0/21). B PART 3/3: failed (0/3). Final:
8-10/10 solved held (10/10); grok median 3500-5000 missed (5750); A1, A2 >= 2/10 held (7/3); n80 .85-1.0 held. Surprises seen: ratio count (A), previous state token (B).
## Hand-built reference
L0H0 BOS-sink count head (score ln 2^23 on BOS, value 2^23 per symbol) -> CNT_e = #e. L0 MLP: per flag (token e, z >= 1) two step
neurons relu(2z - 1) - relu(2z - 2) gated by G = 2000 on the token; Z0 = hat(d) at 0 (3 neurons). L1H0 any head: NV = number of violating
positions. L1 MLP: DD, DEAD = clamp(NV, 0, 1), OUT_Y = relu(Z0 - 2 NV) at '='. lm_head D<k> = 2k DD - k^2 + 1/2 (quadratic readout), X, Y,
N. Neurons L0 2F + 3 (F = 1, 2, 6 flags), L1 5. Exact on every word of length 1..16 (k = 2) / 1..10 (k = 3), <= 3-run words to 40, 80 and
160 (183k-399k inputs, both formats, 0 mismatches). Only this HAND (HAND-a) was built.
## What training finds
- Fully trained A: grok 1000-12500 steps (final anbncn wd .1: 2500-10500, median 5750; wd 1 drifts, 0/3). PART: A .96-.997 (answers from
  final counts, skips the any-violation path), B .23-.61 (MLP0 writes past the frozen quadratic readout); 0/18.
- Circuit types (A): A1 one any-head (keep_in_layer .95-1.0): r1 13/24, final 7/10; A2 sum split over both L1 heads: r1 11/24, final 3/10.
  Bs (anbn B wd .1, 3/3 incl. the two .998 seeds): state from the previous state token (ref_state flip .91-1.0), no HAND built for it.
- B failures: dyck1 misses OUT_Y at '=' and rare deep states; anbncn misses the violation -> dead step (V, DEAD flips .35-.69 at L1).
- Diagnostics (final snapshots; probes best layer R^2; rank = r90 of the last residual; JS = attention divergence to HAND's count head):
| cell | used heads + MLPs / n90 neurons L0, L1 | probes | rank, HAND PC R^2 | JS L0 | best / final |
|---|---|---|---|---|---|
| anbncn A TRAIN wd .1 (final, 10 seeds) | 1-4 + 2 / 20-42, 18-35 | d .65-.89 (L1), NV .61-.76 (L2), OUT_Y .15-.97 | 1, .76-.87 | .40-.50 | 1 / 1 (10/10) |
| dyck1 A TRAIN wd .1/.3/1 | 1-4 + 2 / 35-51, 18-33 | d .55-.81 (L1), OUT_Y .66-.99 (L2), DEAD .42-.81 | 1, .85-.97 | .53-.58 | 1 / 1 (9/9) |
| anbn A TRAIN wd .1/.3/1 | 1-4 + 1-2 / 27-46, 4-39 | d .72-.88 (L1), V .71-.94, OUT_Y .98-1.0 | 1, .96-1.0 | .39-.41 | 1 / 1 (9/9) |
| anbncn A TRAIN wd .1/.3 | 3-4 + 2 / 24-46, 14-40 | d .71-.84 (L1), NV .53-.72, OUT_Y .45-.98 | 1, .69-.89 | .50 | 1 / 1 (6/6) |
| anbn B TRAIN wd .1 | 4 + 2 / 48-55, 34-46 | ref_state .89-.90, DD .81-.87, DEAD .86-.90, CNT0 <= .25 | 2, .72-.76 | .47-.50 | .998-1 / .998-1 (1/3) |
| dyck1 / anbncn B TRAIN (failed) | 4 + 2 / 18-48, 9-45 | DD .76-1.0, ref_state .84-1.0, CNT <= .7 | - | - | .715-.999 best (0/21 each) |
## Closed form
A1/A2: d = sum_e c_e #e/(t+1); V_t = [s_t = e] and [z_e(t) >= 1]; DEAD = [sum_t V_t > 0]; answer Y iff Z0 and not DEAD; MLP1(x_=) =
T[Z0, DEAD]. Final: substitution 1.0 (10/10; probe-decoded keys .96-1.0; shuffled keys .69-.82); complement on the discovered edges .88-.995
(accepted 9/10: A1 6/7, A2 3/3; one A1 seed .880). r1: subst. 1.0, complement .91-1.0 (24/24). Bs: MLP1 = T[S_t-1, s_t] .38-.74: rejected.
Length (n 80 / 160; N-everywhere ~.75): final anbncn A .91-1.0 / .86-.98; r1 dyck1 A .88-.98 / .75-.83; anbn A .86-.99 / .85-.99; anbn B .44-.56 / .18-.45.
## Weights excerpt and reading
`excerpts.py counter`: per archived net, |OV(sym0 - sym1)| and |OV BOS| of the L0 heads (a symbol-dependent value averaged with BOS = a
ratio count) and the five MLP1 neurons with the largest (U[Y] - U[N]) vote and their share. HAND: MLP0 2F + 3, MLP1 5 active neurons.
## Comparison
Same algorithm as HAND-a in A (count -> flags -> any -> readout), with a ratio count instead of HAND's de-normalised sum, so precision
falls with length (dyck1 n 160 near baseline). Matches the counter-attention construction of Bhattamishra et al. 2020. B nets take a
state recurrence instead, like the dfa scratchpad entry ([dfa-group-automata](https://github.com/jc1122/mechanistic-interpretability-circuit-atlas/blob/main/entries/dfa-group-automata.md)).
## Controls and limits
Resample complement with circuit_resampled / random controls; shuffled-key substitution; diag set fixed length 24 (128 words); only one HAND
built; final nets are one run (identical to a paired trial control); lmc barriers 37-71 (separate basins); NV flip untested in dyck1; B closed form open.
## References
Bhattamishra et al. 2020, https://doi.org/10.18653/v1/2020.emnlp-main.576 (conference paper, EMNLP) · Bayat Makou et al. 2026, https://doi.org/10.48550/arxiv.2606.06267 (arXiv preprint)
· Pracher et al. 2026, https://doi.org/10.48550/arxiv.2609.26679 (arXiv preprint) · Lagasse 2025, https://doi.org/10.48550/arxiv.2510.03282 (arXiv preprint)
