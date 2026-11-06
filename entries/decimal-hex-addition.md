# Decimal and hexadecimal addition: a column-sum lookup; trained nets carry without a lookahead head
**Task.** `a+b=` → a + b, 3-digit numbers, base 10 and 16. A: s_3..s_0 (MSB first); B: s_0 k_1 s_1 k_2 s_2 k_3 (LSB first, k = carry token); C: column sums d_2..d_0 ∈ 0..2b−2, `=`, s_3..s_0. Per seed 20k train / 2k test random pairs; exact match of all output tokens.
**Status.** 2026-11-05. HAND per base × format (6 cells); PART and TRAIN 10 seeds per cell; failed cells retried once (both runs kept).
## Summary
Every trained stage that works is a lookup on the column sum s = a + b + c, read as X1 + X2 + CY with equal weights, the HAND input (PART
f(sum) in all 60 nets of the final runs, substitution = net accuracy). PART A extrapolates to 4 and 6 digits; B and C do not. TRAIN needs late
grokking (t(.9) up to 37k): the first run (20k, 2k-plateau stop) solved none of A/C; the retry (40k, warmup + cosine + clip) solved
37/40 (b16 A 7/10), without a HAND-like carry-lookahead head and with length n + 1 = 0. B (carry written as a token) trains in 10/10 at both bases.
## Setup
[METHODS.md](../METHODS.md). AdamW lr 1e-3, wd 1, batch 256, eval every 500, best-train snapshot. Run 1 (v2): betas (.9, .98), TF32,
PART 10k, TRAIN 20k, stop on a 2k plateau after 5k. Retry: PART fp32 20k; TRAIN 40k, b2 .95, warmup 1000, cosine, clip 1, plateau stop
5k (b10 A) or 20k after 20k. Analyses fp32 CPU: mean ablation, probes, PCA/SVD, attention JS, length at 4 / 6 digits. Consumer GPUs.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/decimal-hex-addition](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/decimal-hex-addition) (42: 6 HAND, 36 PART/TRAIN).
**Reproduce:** the training and analysis code for this task is not yet in `code/` (`python run.py` has no addition task), so the learned cells cannot be rerun from this repository; the weights and the table of accuracies are in the folder above.
## Architecture as used
GPT-NeoX 4 layers × 4 heads, d_model 256, d_head 64, MLP 256 ReLU, no LayerNorm, NoPE; vocab 25 (b10) / 37 (b16): BOS, unk, values
0..2b−2, `+ = . c`. PART trains the stage MLP only (A layer 3, B/C layer 2: dense_h_to_4h [256, 256] + bias, dense_4h_to_h [256, 256] + bias)
reading TOK, X1/X2 (column digits), X3 (C: d_i), CY (carry), PH/EQS (C phase), T/VT (position thermometers); TRAIN: all 1.6M weights.
Archive (Hugging Face folder above): HAND, best and worst learned seed per cell x rung x run (`*_r2` = retry), `config.json`, `manifest.json` (sha256); each net reloaded in fp32 and checked against its recorded test accuracy.
## Prediction before training
Design notes, before any run: "mod-b clock or lookup per digit + carry bit". Lookup held (no clock neurons); the carry is used everywhere it is
needed, but TRAIN builds it without the HAND chain head. Nothing was predicted about grokking time or length (not recorded in advance).
## Hand-built reference
The binary-addition network with base-b tables. L0/L1: uniform head → 1/(r+1), hinge MLP → r, anchor head on `+`, thermometers T/VT. A: index heads copy a_i,
b_i (score −20(r_j − t)²); L2 MLP EQ = [a + b ≠ b−1], GEN = [a + b ≥ b]; chain head L3H0 attends to the first lower column with EQ and copies
GEN (= carry); L3 MLP one-hot of (X1 + X2 + CY) mod b. B: L2 MLP DIG = s mod b, MAJ = [s ≥ b]; L3H0 copies the previous MAJ into the carry
slot. C: chain head on the first d_j ≠ b−1; L2 MLP d = a + b, then (d + c) mod b (gate M = 4b). Exact for n = 1..12, margin 30.
## What training finds
- PART (10 seeds/cell): f(sum) lookup 10/10 in b10 A/B/C, b16 A, b16 C retry, b16 B retry; fp32 .99–1.00 (b10), .96–.99 (b16 A), .95–.98 (b16 C retry), b16 B retry 8/10 ≥ .99. No grokking delay (t(.9) 0.5–8.5k). Length n+1 / 2n: A .93–1.00 / .46–1.00; B, C ≤ .08.
- TRAIN v2 best (final): b10 A .09–.94 (.00–.59), b10 B .81–.99 (.01–.77), b10 C .14–1.00, b16 A .00–.84, b16 B .88–.99, b16 C .27–.93.
  Errors grow with carry-chain length in A/C (b16 A .56 → 0 at length 3), flat in B. Retry: b10 A, b10 C, b16 C 10/10 ≥ .998, b16 A 7/10
  (t(.9) 5.5–37k), final = best; 4–16 heads + 590–980 neurons, last MLP a sum lookup in 6–9/10 (else pair lookup); length n+1 ≤ .04.
- Diagnostics (used heads+MLPs; best-layer probe R² of a, carry-in c, digit; residual participation ratio at L4; JS (nats) of
  used heads to their best HAND head; test best (final); TRAIN v2 final = run v1; ‖ causal, medians over solved nets (exact ≥ .9): patch = component with the largest patching drop (nets);
  var inter/DAS = linear interchange / DAS flip, best layer (co = carry_out); edges = EAP-IG jaccard/hand_recall vs HAND, trusted nets (spearman_exact ≥ .8)):

| cell | HAND | PART | PART retry | TRAIN (v2) | TRAIN retry |
|---|---|---|---|---|---|
| b10 A | 3+2; a 1, c 1, dig 1; PR 2.2; JS .00; 1 ‖ patch MLP3; c 1, dig 1 (no DAS) | 3+2; a 1, c 1, dig .84–.87; PR 2.1–2.2; JS .00; .99–1 (.99–1) ‖ MLP3 10/10; c 1/1, dig .89/.95; edges .78/1 (10) | – | 4–16+4; a .57–.85, c .55–.81, dig .70–.92; PR 2.1–4.5; JS .08–.29; .09–.94 (.00–.93) ‖ 2 solved: MLP0, L3H3; c .97/.99, dig .97/.98; edges .13/.29 (2) | 4–12+4; a .67–.97, c .72–.94, dig .81–.95; PR 1.6–6.8; JS .27–.43; 1 (1) ‖ MLP0 8, MLP1 2; c .97/1, dig .97/1; edges .07/.12 (10) |
| b10 B | 3+2; a .86, c .62, dig 1; PR 3.8; JS .00; 1 ‖ MLP2; dig 1/.20, co 1/1 | 3+2; a .83–.89, c .62–.68, dig .76–.79; PR 2.6–2.9; JS .15–.16; .97–1 (.87–.98) ‖ MLP2 10/10; dig .96/.83, co .85/.99; edges .80/1 (10) | – | 1–16+4; a .46–.93, c .63–.84, dig .80–.96; PR 2.1–4.2; JS .07–.30; .81–.99 (.00–.99) ‖ 8 solved: MLP3 4, MLP0 4; dig .97/.99, co .96/1; edges .10/.33 (6) | – |
| b10 C | 4+1; a .73, c 1, dig 1; PR 2.7; JS .00; 1 ‖ MLP2; c 1/1, dig 1/1 | 4+1; a .75–.79, c 1, dig .64–.73; PR 3.0–3.2; JS .00; .99 ‖ MLP2 10/10; c .99/1, dig .64/.85; edges 1/1 (10) | – | 2–16+4; a .43–.71, c .55–.92, dig .74–.86; PR 2.0–5.0; JS .08–.32; .14–1 ‖ 4 solved: MLP3 2, MLP1 2; c .92/.99, dig .98/.99; edges 0/0 (3) | 4–15+4; a .48–.68, c .57–.96, dig .78–.92; PR 1.8–4.2; JS .18–.41; 1 (1) ‖ MLP0 5, MLP3 5; c 1/1, dig 1/1; edges 0/.11 (10) |
| b16 A | 3+2; a 1, c 1, dig 1; PR 1.9; JS .00; 1 ‖ MLP3; c 1, dig 1 (no DAS) | 3+2; a 1, c 1, dig .84–.88; PR 1.8–1.9; JS .00; .96–.99 (.93–.97) ‖ MLP3 10/10; c 1/1, dig .83/.78; edges .79/1 (10) | – | 0–14+0–4; a .65–.99, c .51–.81, dig .54–.93; PR 1.8–4.1; JS .00–.42; .00–.85 (.00–.20) ‖ no solved net | 0–16+0–4; a .54–.93, c .46–.88, dig .64–.96; PR 1.5–4.0; JS .00–.33; .03–1 (.03–1) ‖ 9 solved: MLP0 7, MLP3 2; c .91/1, dig .99/.99; edges .07/.29 (8) |
| b16 B | 3+2; a .86, c .61, dig 1; PR 3.2; JS .00; 1 ‖ MLP2; dig 1/.07, co 1/1 | 2–3+1–2; a .80–.86, c .62–.66, dig .68–.79; PR 2.1–2.4; JS .00–.16; .12–.98 (.02–.96) ‖ 7 solved: MLP2 7; dig .99/.27, co .77/1; edges .78/1 (7) | 3+2; a .84–.89, c .60–.64, dig .72–.79; PR 2.2–2.5; JS .13–.16; .64–1 (.64–1) ‖ 8 solved: MLP2 8; dig 1/.27, co .79/1; edges .78/1 (8) | 2–16+4; a .46–.94, c .57–.84, dig .81–.96; PR 1.8–3.8; JS .14–.39; .88–.99 (.00–.91) ‖ 8 solved: MLP0 6, MLP3 2; dig .97/.98, co .95/1; edges .13/.38 (8) | – |
| b16 C | 4+1; a .72, c 1, dig 1; PR 2.5; JS .00; 1 ‖ MLP2; c 1/1, dig 1/.99 | 4+1; a .75–.78, c 1, dig .63–.72; PR 2.6–2.7; JS .00; .85–.91 (.72–.90) ‖ 2 solved: MLP2 2; c .98/1, dig .41/.66; edges 1/1 (2) | 4+1; a .76–.78, c 1, dig .65–.73; PR 2.6–3.2; JS .00; .95–.98 (.95–.98) ‖ MLP2 10/10; c .90/1, dig .61/.64; edges 1/1 (10) | 3–16+4; a .44–.65, c .59–.90, dig .72–.92; PR 2.2–5.8; JS .09–.39; .28–.94 (.00–.82) ‖ 1 solved: MLP2; c .82/.97, dig .99/.99; edges 0/0 (1) | 5–16+4; a .38–.63, c .55–.94, dig .75–.92; PR 3.2–6.1; JS .17–.44; 1 (1) ‖ MLP3 6, MLP0 4; c .93/1, dig 1/1; edges 0/.22 (8) |
- Failures: PART b16 C / b16 B (2 seeds) missed the digit table with complete inputs (c R² 1, JS 0): under-trained, fixed by fp32 + 20k. TRAIN
  A/C v2 failed on long carry chains; neither c R² nor a chain-head match (1–3/10 nets, JS ≥ .34) separates them from the solved retries: the curves do.
- Causal: PART patch top = the stage MLP in 113/113 solved nets; PART edges keep HAND's wiring (hand_recall 1.0; jaccard .78-.80 in A/B, extra edges; 1.0 in C);
  TRAIN wiring differs (jaccard 0-.13; heads not aligned: a different-wiring flag, not a match score), patch top MLP0 or MLP3. No HAND var fails linear interchange while DAS finds it here; the reverse holds for the B digit (inter .96-1, DAS .07-.83, HAND .07-.20).
## Closed form
PART stage → mean output given s (C: given phase and its sum): substitution = net accuracy (.95–1.00; retry b16 B s7 .69). TRAIN: last MLP → f(sum) .88–.99 in B, .50–1.00 in the retry A/C nets (≥ .99 in about half); the earlier layers have no closed form.
## Weights excerpt and reading
Numbers computed from the archived files. HAND A L3 neurons read [X1, X2, CY] = [1, 1, 1] with bias −v (b10 n11–n15: −10..−14): hinges
relu(s − v). PART A (trained from noise): the top neurons read the same dims with equal weights, b10 s0 n196 [.04, .04, .04] bias −.03,
b16 s7 n114 [−.08, −.08, −.08] bias .04; |mean|/rms of the three weights has median 1.00 over the 92–122 active neurons of each archived A
net, so the stage is f(X1 + X2 + CY). C reads X1 = X2 (phase-1 sum, median .99–1.00); B reads X1 = X2 and the own-carry token with its own weight
(median .67–.84). TRAIN (no shared basis): digit-token embeddings are periodic in the value, top-2 DFT frequencies (1 and 2 or 3) hold .71–.97 of their power (v2 and retry nets alike).
## Comparison
PART = HAND's algorithm (sum lookup, ~100–200 hinges); TRAIN finds the digit table and a carry, not the lookahead head. As in binary addition, PART A extrapolates.
## Controls and limits
One training length (n = 3); length at 4 and 6 digits only. v2 PART nets are TF32 (up to .03 lower in fp32). Probes are linear, at scored positions.

## Circuit found in trained nets
Circuit probes on best snapshots (2026-11-06), on fresh pairs disjoint from all stored splits. **Verified** = causal intervention plus a
successful component substitution (not a whole-net algorithm); **supported** = causal only. Teacher-forced 3-digit whole-answer exact
unless stated; ablation numbers are accuracy retained; layers zero-based. s = round(X1 + X2 + CY) (A) or round(X1 + X2 + TOK[c]) (B).

| Component | Role | Evidence (verified) |
|---|---|---|
| b10 A TRAIN retry seed 1: MLP0 + digit embeddings | position/token table + Fourier k = 1, 2, 3 embeddings | MLP0 mean ablation .001; joint replacement 1024/1024 exact and agreement in both tested seeds (later layers native) |
| b10 A PART seed 0: L2H0 / L2H1 | fetch b / a | mean ablation .002 / .000; correct-variable substitutions 1.000 / 1.000 |
| b10 A PART seed 0: L3H0 | first decisive lower-column carry | mean ablation .127; carry-table substitution 1.000, R² = 1 |
| b10 A PART seed 0: MLP3 (121 neurons) | sum-response digit stage | population ablation .001; sum-response substitution 1.000, drop 0 |
| b16 B PART retry seed 1: L3H0 | analog carry state, not a Boolean bit | carry-bit means .949; compact hinge substitute 1.000, R² .9964 |
| b16 B TRAIN seed 4: L3H0 | copies the preceding arithmetic position | joint hard copies .993, agreement 1; bit-only / subtotal-only content .820 / .903 |
| b16 B TRAIN seed 4: MLP3 (231 neurons) | sum-coded digit lookup | ablation .001; substitution .993 = baseline, agreement .998 (not identity) |
| b10 C TRAIN retry seed 0: L3H0 | one head reused: b digits, then written column sums | head ablation .000; joint six-query hard substitution 1.000 |

- Closed forms (actual internal channels, other stage outputs zero): A PART `Y[(s mod b) + 2] = 1`; B PART `DIG = s mod b`, `MAJ = [s ≥ b]`;
  C PART `Y[round(X1 + X2) + 2] = 1` before the phase token, `Y[(round(X3 + CY) mod b) + 2] = 1` after. Replacing only the learned stage gives
  1.000 in 80/80 PART snapshots (40,960/40,960 pairs), failed seeds included (a repair, not their mechanism). In b16 B PART it also lifts
  4 / 6 digits from .066 / .032 to 1.000: the length failure is in the learned stage, not the frozen routing.
- b16 B PART analog carry: `MJP = a + b·s + c·ReLU(s − 15) + d·ReLU(s − 16)` on the live predecessor s; e.g. (a, b, c, d) at the first carry
  query = (−2.0873, .1209, .0511, −.1162); 1.000 on 1,024 pairs, while canonical 0/1 output gives .222. Not length-general.
- Supported: b10 A TRAIN uses a distributed stack (MLP0/1/2/3 ablations .001 / .003 / .047 / .007). Scrubbing every component within the
  local-variable classes at once gives .74–.80 (seed 1) and .81–.82 (seed 2); lower digits stay 1.000 and every error is the top carry,
  whose context-dependent readout is unresolved. b10 C TRAIN's 105-neuron sum population is necessary (.002) but a pure-sum model gives .933.
  Failed b16 A TRAIN: units accuracy .054 although carry interchange/DAS is .976 / .996: not a carry-only failure.
- Size (search-only local minima, not verified circuits): b10 A TRAIN seeds 1 / 2 keep 7 heads + 608 / 597 neurons, independent exact
  .980 / .985 (fail 99%); HAND under the same procedure 3 + 22 (.992).
- Seed census (180 snapshots, 131 at ≥ .90; signatures within .01 exact and token agreement): PART: frozen lookahead/two-phase routing +
  canonical stage in every solved seed; B carry head bit-sufficient / subtotal-required: b10 8/2, b16 3/5. TRAIN sum-table sufficient /
  main effects / unresolved: b10 A 4/0/6, b16 A 3/3/3, b10 B 6/0/3, b16 B 5/1/2, b10 C 3/2/5, b16 C 2/0/8; HAND-chain carry route in 0 A nets.
- Note: the Summary's "equal weights" holds at the decision level only: in b16 B PART, setting own-carry weights equal to operand weights
  drops exact to .938 (two top neurons) and .303 (all). The census's sufficient sum table is stricter than the Closed form's TRAIN f(sum)
  counts, hence fewer TRAIN nets pass here.
- Open: the A TRAIN top-carry state; compact joint forms for B/C TRAIN; why failed hex A loses the units digit.

## References
- Quirke, P., & Barez, F. (2023). Understanding addition in transformers. https://doi.org/10.48550/arxiv.2310.13121 (per-digit streams, carry heads)
- Quirke, P., Neo, C., & Barez, F. (2024). Understanding addition and subtraction in transformers. https://doi.org/10.48550/arxiv.2402.02619 (shared add/sub skeleton, borrow)
- Lee, N., et al. (2023). Teaching arithmetic to small transformers. https://doi.org/10.48550/arxiv.2307.03381 (output order, scratchpads)
- Zhou, T., et al. (2024). Pre-trained large language models use Fourier features to compute addition. https://doi.org/10.48550/arxiv.2406.03445 (contrasting mechanism in LLMs)
- Kantamneni, S., & Tegmark, M. (2025). Language models use trigonometry to do addition. https://doi.org/10.48550/arxiv.2502.00873 (contrasting mechanism in LLMs)
- Nikankin, Y., et al. (2024). Arithmetic without algorithms: language models solve math with a bag of heuristics. https://doi.org/10.48550/arxiv.2410.21272
