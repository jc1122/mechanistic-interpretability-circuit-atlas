# DPLL search on small 3-SAT

> - **Task.** Decide whether a 3-SAT formula (n = 4–6 variables, m = 10–25 clauses) is satisfiable, either answer only or after writing out the DPLL solver state at each step. Example (in words): a formula that some assignment satisfies → SAT; one that every assignment falsifies → UNSAT.
> - **Hand-built net.** Answer only: 1 head + 130 neurons check 128 witness assignments and threshold the count of survivors. Scratchpad: 14 heads + 71 neurons run unit propagation, decision and backtrack on the full solver state.
> - **Trained nets.** With every earlier stage frozen at the hand-built values, a trained last MLP solves both formats on 3/3 seeds (answer only in 0.5–1k steps, scratchpad in 3–5.5k). Trained from scratch, answer only reaches .72–.86 and the scratchpad .00–.02 complete traces: the state update is not learned.
> - **Verdict.** Partly solved (frozen-parts setup 3/3 in both formats; from scratch 0/3; ten-seed final not finished). Closed form: partial (T PART seven-field update keeps 52/52 traces per seed, zero/shuffle 0/52; A PART threshold not substituted).
> - **Links.** [Weights](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/dpll-search) · Reproduce: not available yet · [Terms](../GLOSSARY.md)

## Task and status
**Task.** Ordered 3-SAT, n = 4–6 variables, m = 10–25 clauses, outcome-balanced, UNSAT proposals core-enriched. Answer-only A: one
SAT/UNSAT token. Scratchpad T: after each DPLL update (unit propagation, decision, backtrack) the full solver state — assignment,
decision levels and tried flags, a clock, the clause in focus and the conflict flag — then SAT/UNSAT. Canonical clause-multiset hashing
gives train/monitor/test buckets 8:1:1, so clause-order or clock variants never cross splits. Test: 192 formulas per seed; exact = every
non-PAD teacher-forced output token correct (T: the complete trace).
**Formats.**
- **A**: answer only (one SAT/UNSAT token).
- **T**: scratchpad; the full solver state after each DPLL update, then SAT/UNSAT.

**Status.** 2026-11-07. HAND A/T; PART A/T and TRAIN A/T, 3 screening seeds each, all runs finished. Open: **the ten-seed final cell
(T PART, seeds 3–9) has not finished** (four launches lost to host failures; the current one is in training); the T PART learned-update
closed form is accepted with zero/shuffle controls (below), other component claims stay frozen-HAND identities; no admissible TRAIN retry (data already ≥ 3×; size needs train < 1,
unmeasured); A PART n = 7 replay and the A PART threshold substitution are not run; public code pending.

## Summary
With every earlier stage frozen at the hand-built values (HAND), a trained last MLP (PART) solves both formats on 3/3 seeds (A in 0.5–1k steps, T in 3–5.5k). In T the
learned MLP5 sits on HAND's retrieval and controller outputs and does the conditional copy/clear/flip of the state; in A the learned MLP1
reads HAND's witness count and thresholds it, on HAND's four-edge pooling → count → threshold path (edge check ρ = 1, Jaccard 1). From
scratch (TRAIN), A reaches only .72–.86 and T .00–.02 test exact at the same data; the trained nets decode the step clock and variable well but
not the new state or conflict, and every compact bottleneck fitted to them fails substitution.

## Setup
[METHODS.md](../METHODS.md) for shared conventions. AdamW lr 1e-3, wd .1, betas (.9, .98), warmup 1,000, cosine, clip 1, batch 8 fresh
formulas per step, 30k budget, plateau stop 5k after 22.5k; eval every 500, checkpoints every 2,500. PART fp32; TRAIN TF32 updates,
fp32 evals. Best is selected by a 96-formula monitor (exact, then CE), not by test. Success = best test exact ≥ .99. GPU accelerators
for training, CPU fp32 for re-scoring. Test formulas are regenerated from their seed (see `config.json`).
**Reproduce:** training code for this task is not yet in `code/`; the weights, held-out splits and recorded accuracies are on Hugging Face (link above). The archived networks re-score to the recorded accuracies.

## Architecture as used
GPT-NeoX, ReLU, no LayerNorm, no positional encoding (clocks are explicit tokens), untied embeddings, vocabulary 1,104, context 16,384.
A: 2 layers, d_model 272, 1 head (d_head 272), MLP 128. T: 6 layers, d_model 160, 5 heads × 32, MLP 96. PART A retrains only layer-1 MLP
(input masked to the witness-count channel, output to the answer channel); PART T retrains only layer-5 MLP (inputs: old state/level and
controller/comparison channels; outputs: new state/level). TRAIN: the same shapes, all weights trained.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/dpll-search](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/dpll-search):
`hand_{A,T}`, `part_{A,T}_{best,final}_s0…2` (the 4 MLP tensors; insert into `hand_<fmt>`), `train_{A,T}_{best,final}_s0…2` (full
state dicts), `config.json`, `manifest.json` (seed, steps, recorded and fp32 re-scored exact, sha256). T PART best = final (same bytes).

## Prediction before training
Recorded before the runs: A PART ≥ 2/3 at .99 — **confirmed** (3/3); A TRAIN ≤ 1/3 — **confirmed** (0/3); T PART ≥ 2/3 by 5k —
**partly** (first exact hits by 5k but unstable; all-seed exact at 5.5k); T TRAIN 1–2/3 by 30k — **refuted** (0/3, ≤ .016).

## Hand-built reference
T: layer 0 retrieves clock and header; layer 1 the previous state and clause operands; layer 2 copies the clause and selects the unknown
/ deepest-decision variable; layer 3 finds the first unit clause or conflict and the action; layer 4 copies the controller; layer 5 does
the conditional copy/clear/flip. Signed binary clock matching keeps stale conflicts from winning. A: clause falsification for 128 witness
assignments is embedded, one head averages over clauses, an MLP counts surviving witnesses and thresholds (> 0 → SAT). Structural size:
A 1 head + 130 neurons, T 14 heads + 71 neurons. Verified: A 128/128 formulas; T 264 traces (172,395 tokens) through 83 rows and all
actions plus a free 261-token rollout; all 216 local clause cells exhaustive; minimum logit margin A 30, T 15. Formula tests are
stratified, not exhaustive.

## What training finds
| Cell | Solved | Best → final test (s0 / s1 / s2) | Stop | First test-exact step |
|---|---|---|---|---|
| PART A | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 | solved 1k | 1k / .5k / .5k |
| PART T | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 | solved 5.5k | 4.5k / 3k / 5.5k (earlier hits regressed) |
| TRAIN A | 0/3 | .719 / .833 / .865 → .745 / .828 / .859 | plateau 28k | never |
| TRAIN T | 0/3 | .010 / .005 / .005 → .016 / 0 / .005 | budget 30k | never |
| T PART final, 10 seeds | **not finished** | | | |
Grok step is unestablished (training-set fit was not measured). Circuit types: PART A — one type, 3/3 (HAND's pooling → count → threshold
path); PART T — one type, 3/3 (frozen retrieval + learned local update); TRAIN A (pooling → dense classification) and TRAIN T (seed 0/1
retrieval-dominated, seed 2 late-update) are provisional diagnostic groupings, not verified algorithms.

**Diagnostics** (used = heads + neurons whose single mean-ablation drops exact by > .05, not joint pruning; R² = best linear probe of a
HAND variable (layer); r99 = residual components for 99% variance; JS = attention divergence to HAND; inter = linear interchange flip;
T batch 32 windows at n = 6, m = 20; A 128 answers; ranges over seeds, best and final):
| Cell | Used H+N | Best probe R² (layer) | r99 res. | JS | inter; DAS | Best / final |
|---|---|---|---|---|---|---|
| PART A | 1+0 each | witness count 1 (2); SAT ≤ .15 (thresholded) | 6–8 | 0 | 1; 1 | 1 / 1 |
| PART T | 11+23, 10+23, 9+21 | clock .82–.99; unit sign .55–1 (0); state/level ≤ .39 (6) | 2 (HAND 2) | 0 | 0–.56 (nf 1–9); joint 22-input 1 | 1 / 1 |
| TRAIN A | 1+0 each | SAT .16–.31; witness count ≤ .19 | 2–3 | .28–.44 | .64–.93; .87–.90 | .72–.86 |
| TRAIN T | 13–17 + 1–55 | step .95–.98 (3–4); variable .99 (0); action .87–.91; new state .37–.48; conflict ≤ .51 | 19–25 (HAND 2) | .19–.66 | 0–.8; .87–.90 | .00–.02 |
Failures explained: TRAIN T decodes where it is in the search (step, variable, action) but not what the update writes (new state ≤ .48,
conflict ≤ .51) — the missing variable is the state update itself. TRAIN A uses a dense code (MLP r99 93–102 vs HAND's threshold), with
no linear witness count (≤ .19). T PART aligns with HAND's weights up to rotation (overlap .84–1, fit R² 1); TRAIN overlap ≤ .04,
CKA ≤ .34; T TRAIN seeds sit in different basins (LMC barrier 6.5–6.8). T edge checks are below the faithfulness bar (ρ .38–.87).

## Closed form
**Accepted (T PART, 3/3 seeds; best and final are byte-identical).** The learned MLP5 equals a parameter-free seven-field update
F = (state, level, level²): conditional copy / clear / force / decide / backtrack of the old state. On 52 fresh complete teacher-forced
traces per seed ((n, m) = (4, 10), (5, 17), (6, 20), (6, 25), (7, 20)), substituting F for MLP5 — alone or with the whole frozen path
replaced by its HAND formulas — gives 52/52 per seed vs native 48 / 51 / 48 (all native errors repaired); zeroing or position-shuffling
MLP5's output gives 0/52. A 14-scalar affine fit of F's fields also keeps 52/52 (14 < 2,887 learned parameters). Rejected: a 14-scalar raw-activation fit of MLP5 (96/96 → 0/96 windows; 0/52 complete traces); an 817-scalar
count bottleneck for TRAIN A (.70–.84 → .47–.56); a 4,822-scalar update bottleneck for TRAIN T (.31–.68 → 0 on its windows). A PART's
threshold form is not yet substituted.

## Weights excerpt and reading
No trained-weight formula is promoted: the accepted seven-field form is decision-equivalent on measured inputs, not a reconstruction of
learned neurons, and its implementation thresholds flags, argmaxes the old state and clamps the old level. Frozen HAND identity in PART
is architectural, not evidence that the learned stage has HAND's weights.

## Comparison
Scratchpad vs answer-only: PART 3/3 vs 3/3; TRAIN 0/3 vs 0/3, but answer-only gets much closer (.72–.86 vs ≤ .016 complete traces;
T's whole-trace bar is far stricter). Size: PART A 1 head and no individually necessary neuron; PART T 9–11 heads + 21–23 neurons vs
HAND's 14 + 71 structural. Length (n = 7, m = 10–17, 24 formulas per seed): PART T .79 / .92 / .71; TRAIN A .79 / .63–.67 / .79; TRAIN T 0;
PART A not measured. Related: a hand-constructed transformer DPLL with chain of thought (Pan et al., 2024, preprint) and full-state
inductive scratchpads (Abbe et al., 2024, preprint).

## Controls and limits
HAND passes an independent SAT oracle, causal-prefix and right-padding checks, clock shifts and clause reversal; PART masks preserve HAND.
T scores are teacher-forced; greedy rollout was checked on one n = 4 formula per seed only (T PART and TRAIN A 1/1, TRAIN T 0/1). No
evidence beyond m = 25. Seed counts are screening counts (3); the ten-seed final is pending. Size counts are small-batch single
ablations, not minimal circuits.

## References
- Pan, L., Ganesh, V., Abernethy, J., Esposo, C., & Lee, W. (2024). Can Transformers Reason Logically? A Study in SAT Solving. arXiv
  preprint. https://arxiv.org/abs/2410.07432 (research preprint)
- Abbe, E., Bengio, S., Lotfi, A., Sandon, C., & Saremi, O. (2024). How Far Can Transformers Reason? The Globality Barrier and Inductive
  Scratchpad. arXiv preprint. https://arxiv.org/abs/2406.06467 (research preprint)
