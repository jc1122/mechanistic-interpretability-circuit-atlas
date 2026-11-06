# Symbol-level algebra on term atoms: collecting like terms is learned at every rung; expansion is not, and solving a linear equation is learned from scratch only with a scratchpad

**Task.** Each term is one token (side, power, coefficient). Three operations: **collect** (sum like terms of a polynomial),
**expand** (binomial product), **solve** (`ax + b = cx + d` → a rational atom). Prompt `BOS <left terms> ; <right terms> =` with
independently permuted terms; two terms per side, coefficients −9…9 without ±8 (±8 held out). Outputs are integer/rational atoms, not
digit strings. Format **A** writes only the answer; format **T** writes a scratchpad (expand: distribution products then collection;
collect: copy/copy/sum triples; solve: `D N ; solution`). Split by canonical-input hash mod 97 (one residue = test); fresh training
sequences every batch; 1000 held-out sequences per seed. Exact = whole teacher-forced output.
**Status.** 2026-11-06. HAND exact for all six cells. Three-seed screening runs (30k steps) for PART and TRAIN in all six cells, plus
PART collect T seeds 3–9, PART expand A/T at 60k and TRAIN expand T at 60k. Pending: ten-seed finals (except PART collect T), most
best/final diagnostics, native-vs-closed-form agreement on a fresh batch, literal trained-weight excerpts, public code.

## Summary
Collection is easy: training only the last arithmetic MLP on frozen hand-built routing (PART) solves A and T in every seed (10/10 for T)
within 0.5–1k steps, and a three-number addition formula (A) or an eight-number phase-gated copy/copy/sum (T) replaces the trained MLP
with 4096/4096 exact sequences in all six fitted seeds. Training from scratch (TRAIN) also collects (A ≥ .996, T 1.0, 3/3). Expansion is
not learned in the answer-only format at either rung (PART ≤ .96 even at 60k, TRAIN ≤ .13), but TRAIN with the scratchpad solves it at
60k (3/3). Solving the equation needs the scratchpad from scratch (TRAIN T 3/3 exact, TRAIN A ≤ .97); PART fails both formats because its
quadratic stage lacks precision.

## Setup
[METHODS.md](../METHODS.md) for the shared conventions. AdamW lr 1e-3, wd .1, warmup 1000, cosine over the run (30k; retries 60k),
clip 1, batch 128 fresh sequences per step, eval every 500 steps on the 1000 held-out sequences (best snapshot = highest there), stop
when all seeds are exact or on a 5k plateau after 75% of the steps. PART fp32, TRAIN TF32; all seeds of a cell in one process on
consumer GPUs; re-scores fp32 on CPU.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/symbolic-algebra-atoms](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/symbolic-algebra-atoms)
(52 nets + 6 HAND; best snapshots; each run's held-out sequences in `test_<run>`). Reloaded in fp32, all 52 reproduce their recorded
held-out exact (TF32 nets within 3/1000) and every HAND scores 1.0.
**Reproduce:** the generator, HAND and trainer for this task are not yet in `code/`; weights, test sequences and accuracies are in the folder above.

## Architecture as used
GPT-NeoX, ReLU, no LayerNorm, no positional encoding (causal mask only), untied read-out, vocabulary 2328, d_model 64, 4 heads of 16,
MLP 256, positions up to 128. Solve A: 2 layers; all other cells: 3 layers. TRAIN learns every tensor. PART: HAND frozen; the stage MLP
(solve A layer 1, others layer 2: `dense_h_to_4h` [256, 64] + bias, `dense_4h_to_h` [64, 256] + bias) re-initialised and trained with
input columns and output rows masked to the HAND interface. Zero-based input columns: collect A {27, 28}; collect T adds 18…25 (phases);
expand A {27, 29…33, 34, 36…40, 18…25}; expand T {27, 29…33, 41, 18…25}; solve A {42, 43}; solve T {27, 28, 42, 43, 18…25}; output rows
{44} (solve A {45…47}, solve T {44…47}). Files: `hand_<op>_<fmt>`, `part_*` (trained tensors only; insert into the HAND), `train_*` (full
state dicts), `test_<run>`, `config.json`, `manifest.json` (seed, best step, recorded and re-scored exact, final exact, sha256).

## Prediction before training
Recorded before training (exact ≥ .99): collection PART 3/3 by 5k and TRAIN 3/3 by 15k; expansion PART ≥ 2/3 and expansion T TRAIN
≥ 2/3 by 15k; solve A TRAIN ≤ 1/3, solve T TRAIN ≥ 2/3. Collection and solve held (collect T TRAIN later than predicted: best at
17.5–22.5k); the expansion predictions failed at 30k (expansion T TRAIN succeeded only at 60k).

## Hand-built reference
Except solve A, L0 builds a reciprocal-position frame, L1 anchors phases at `=`, L2 heads fetch operands by squared side/power match
(score −24(side − s)² − 24(power − p)² + 256·is_term). Signed ReLU pairs add/subtract; multiplication sums bit-gated signed pairs with
weights (1, 2, 4, 8, −16). Expansion T reads the written products. Solve A fetches and subtracts coefficients in L0 and forms quadratics in
L1; solve T does the arithmetic in L2; the ratio score 2r·DN − r²D² (plus ALL/NONE scores) picks a bounded rational atom (not Euclid).
Checks: 34,454 sequences across the six HANDs (exhaustive local products, 3721 solve states, stratified full inputs, permutations,
greedy generation); minimum margin .0011 (solve), 1 elsewhere. Heads + neurons: collect A 4+22, T 4+147; expand A 6+41, T 5+60;
solve A 4+244, T 6+270.

## What training finds
Held-out exact, best snapshot (final in brackets where lower by > .005); 3 seeds unless marked.
| Cell | PART | TRAIN |
|---|---|---|
| collect A | 1, 1, 1 (step .5k) | 1, 1, .996 (11–20k) |
| collect T | 1 in 10/10 (1k) | 1, 1, 1 (17.5–22.5k) |
| expand A | .90, .90, .80; 60k: .95, .96, .88 | .11, .12, .13 [.09–.11] |
| expand T | .71, .74, .78; 60k: .78, .81, .86 | .11, .16, .99; 60k: 1, 1, 1 (29.5–36.5k) |
| solve A | .97, .96, .96 | .97, .29 [.24], .42 [.33] |
| solve T | .41, .40, .39 | 1, 1, 1 (25–26.5k) |
Learning curves show no separate memorisation phase: collection TRAIN A crosses test .5/.9 at 2–3.5k/4.5–5.5k, before train exact
(13.5–18k). Diagnostics (best snapshot, one seed each): PART collect A/T decode the sum with probe R² = 1 at L3, residual rank99 = HAND;
PART solve A decodes D, N exactly at L1 but D², N², DN only at R² .91/.61/.84 at L2; TRAIN collect A sum R² .92 at L3 (12 heads matter,
JS to HAND .16), collect T .42, solve T difference .998 at L3 and D/N .98 at L2. Failure localisation, solve A PART seed 0: exact D, N
reach the learned quadratic stage; substituting exact arithmetic there repairs .971 → 1, so the failure is precision of the quadratic.

## Closed form
**PART collection:** A: S = a + b; T: S = g0·a + g1·b + g2·(a + b) (phase gates g, separator 0), from the fetched operands. Replacing the
whole stage-MLP output with these response fits scores 4096/4096 fresh held-out sequences in each of the six seeds (exact-sum control
also 4096/4096); no fitting on the evaluation set. The original best-snapshot runs also record native → fitted exact 1 → 1 for all six.
Native-net agreement on the fresh batch is pending. No TRAIN closed form is accepted.

## Weights excerpt and reading
HAND collect A, input (a, b): neuron 0 (W_in, b_in, W_out) = ((1, 1), 0, 1), neuron 1 ((−1, −1), 0, −1): ReLU(a + b) − ReLU(−a − b).
Trained PART response-fit coefficients (fits of the stage's response, rounded; A on [a, b, 1], T on [g0·a, g1·b, g2·(a+b), g0, g1, g2, g3, 1]):
| Seed | Collect A | Collect T |
|---|---|---|
| 0 | (.998348, .999904, .020512) | (.999751, .999719, .999953, −.001477, .003338, −.002098, .000132, −.000105) |
| 1 | (.995752, .998575, −.009040) | (1.000926, 1.001168, 1.001172, −.001005, −.002591, −.002531, .004153, −.001974) |
| 2 | (.998967, 1.002262, .005735) | (1.000948, 1.001930, 1.002163, −.000087, .000327, −.001392, .002057, .000906) |
Reading: slopes ≈ 1 implement copy/add; offsets stay below half a unit over all 289 legal operand pairs per phase, so the integer
read-out 2tS − t² picks the right atom. In PART collect A, 193/256 learned neurons point within cosine .95 of signed-sum directions;
ablating them as a group drops exact to .004, i.e. the sum is spread over many hinge pairs rather than HAND's single pair.
Literal trained rows are now in the archive; their excerpt is pending.

## Comparison
PART collection computes HAND's variable (the sum) with a distributed implementation, not HAND's sparse weights. Scratchpads decide
the TRAIN outcome for expansion and solving, not for collection. Held-out ±8 coefficients: TRAIN collect A/T and solve T score 0;
PART collect at 6 terms and wide coefficients .66/.008. Progress-measure methods [Nanda et al.] are methodological context only.

## Controls and limits
HAND controls exact; substitution evaluated on fresh hash-held-out sequences, fits not tuned on them. Sampling is with replacement;
saved perfection is not exhaustive exactness. Not established: natural-notation parsing, learned fraction reduction, unseen rational
atoms, length or range transfer. Pending: ten-seed finals, best/final diagnostics for most nets, fresh native-vs-formula agreement,
trained-weight excerpts, sufficient-neuron counts, checkpoint dynamics, public code.

## References
[Nanda et al., Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217) — research preprint; methodological comparison.
[Wen et al., Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective](https://arxiv.org/abs/2410.05192) — research preprint; retry-budget motivation only.
