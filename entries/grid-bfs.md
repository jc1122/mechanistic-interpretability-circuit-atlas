# Grid BFS: a scratchpad of reached-set maps is learned from scratch (3/3 with 3× data); answer-only fails with error growing with distance; the frozen-parts update MLP is a three-input hinge

**Task.** Four-neighbour 6×6 mazes, ≤ 14 walls, connected free start and target, shortest-path distance 0–10. Answer-only A: one
distance token. Scratchpad T: after each BFS step the reached-set map (every square: unreached / reached / wall), a CHECK of the target,
then the next step index or END. Square tokens encode coordinates. Split by wall-mask hash 80/10/10; fresh mazes every batch. Test: 16
mazes per distance (176 per seed); exact = whole teacher-forced output correct (T includes END).
**Status.** 2026-11-07. HAND T (3 layers) and A (38 layers); PART T 3 seeds; TRAIN A/T 3 seeds each at r1 (30k) and r2 (90k, 3× fresh
data). Open: **the ten-seed final cell (T PART + T TRAIN) has not run**; the TRAIN A rung is unsolved (0/3 after the data retry; the
depth-8 / distance-curriculum retry is proposed, not run); TRAIN A r2 diagnostics and 7 of the 12 r2 diagnostics have not landed; greedy
rollout and OOD (distance 11–20, 8×8) are unmeasured for TRAIN; no substitution test on the solved r2 TRAIN nets; public code pending.

## Summary
With 3× fresh data the 4-layer net learns the T scratchpad from scratch on 3/3 seeds (grok 63.5–72k steps); the answer-only format
fails at matched size and data (0/3, test .93–.95): exact for distance ≤ 3, worse with every further step (distance 10: .50–.88), the
signature of a fixed-depth net that cannot unroll 10 serial BFS steps (the hand-built A needs 38 layers). Frozen hand-built addressing
plus one trained MLP (PART T) solves 3/3 within 500 steps, and its whole output is replaced by a three-input hinge, Y = (1, a, w)·B,
with 176/176 kept on all seeds. Solved TRAIN T nets use the hand-built update variables (`or`, `next`) in their last layers but in a
high-rank code unrelated to the hand-built basis.

## Setup
[METHODS.md](../METHODS.md) for shared conventions. AdamW lr 1e-3, wd .1, warmup 1,000, cosine, clip 1, batch 8 fresh mazes per step
(memory bound on long maps); checkpoints every 2,500; eval every 500. PART fp32, 10k budget (stopped solved at 500). TRAIN TF32 updates,
fp32 evals; r1 30k; r2 (retry rule: data) 90k with 3× fresh data, T stopped solved at 79k, A ran to budget. Success = best test exact
≥ .99. GPU accelerators for training, CPU fp32 for re-scoring. Test mazes are regenerated from their seed (see `config.json`). Public
reproduction code for this entry is pending; the archived networks re-score to the recorded accuracies.

## Architecture as used
GPT-NeoX, ReLU, no LayerNorm, no positional encoding, untied embeddings, vocabulary 565, context 8192. HAND T: 3 layers, d_model 80,
5 heads, MLP 24 (7 heads + 10 neurons used). HAND A: 38 layers, d_model 384, 3 heads, MLP 144 (2 heads + 5,258 neurons used). PART T:
HAND T with layer-2 MLP retrained (W_in [24, 80], b_in [24], W_out [80, 24], b_out [80]), masked to six input bits (N, E, S, W, self,
wall) and three state outputs (unreached, reached, wall). TRAIN A/T: 4 layers, d_model 256, 8 heads, MLP 512, all weights trained.
Networks: [huggingface.co/jc1122/circuit-atlas-nets/tree/main/grid-bfs](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/grid-bfs):
`hand_{T,A}`, `part_T_{best,final}_s0…2` (the 4 MLP tensors; insert into `hand_T`), `train_{T,A}_{best,final}_s0…2` (r2 full state
dicts), `config.json`, `manifest.json` (seed, steps, recorded and fp32 re-scored exact, sha256). The failed r1 TRAIN nets are not
archived (superseded by r2).

## Prediction before training
Recorded before the runs: PART 3/3 within 2.5k — **confirmed** (500 steps). TRAIN T ≥ 2/3 within 5–25k — **refuted** at r1 (0/3 at
30k); r2 reached 3/3 only at 63.5–72k with 3× data. TRAIN A ≤ 1/3 — **confirmed** (0/3 at r1 and r2).

## Hand-built reference
T: layer 0 binds each map cell to its step clock; jump heads fetch the N/E/S/W/self cells of the previous map (BOS sink at borders);
three neurons compute `wall ? wall : OR(self, N, E, S, W)` (hinge rows (1, 1, 1, 1, 1, −5), biases 0 / −1, reached outputs +1 / −1, a wall
copy and an unreached bias 1); CHECK reads the target cell; a router emits the next step index or END. A: 35 synchronous updates over
38 layers, then d = 36 − hits. Verified: T 184/184 cases, A 41/41, minimum logit margin 30; stratified full-network tests plus
exhaustive local and tiny-grid checks, not an enumeration of all mazes.

## What training finds
| Cell | Solved | Best → final test (s0 / s1 / s2) | ≥ .99 step |
|---|---|---|---|
| PART T r1 | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 | 500 |
| TRAIN T r1, 30k | 0/3 | .716 / .381 / .761 → same | none |
| TRAIN A r1, 30k | 0/3 | .898 / .807 / .898 → .898 / .807 / .892 | none |
| TRAIN T r2, 90k, 3× data | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 (stop 79k) | 72.0k / 63.5k / 67.5k |
| TRAIN A r2, 90k, 3× data | 0/3 | .943 / .926 / .955 → .932 / .915 / .943 | none; d ≤ 3 exact, d 10 .62 / .50 / .88 |
| T final, 10 seeds | **not run** | | |
Scratchpad vs answer-only at r2: 3/3 vs 0/3 at the same size. Fresh data every batch: no memorisation step. Circuit types: PART — one
type (frozen fetch + learned local update), 3/3 seeds; TRAIN T r2 — one type by diagnostics (update variables decoded and causally
used in layers 3–4, last MLP needed), 3/3; TRAIN A — unsolved, distance-limited.

**Diagnostics** (used = mean-ablation drop > .05 of 32 heads + 4 MLPs; R² = best linear probe (layer); PR = residual participation
ratio; JS = attention divergence to the closest hand-built head; patch = shuffle-batch activation patching, component with the
largest drop; diagnostic batch n = 32 at one distance):
| Cell | Used | Best probe R² (layer) | PR; hand-built CKA | JS | Patch rank 1 | Best / final |
|---|---|---|---|---|---|---|
| PART T s0–2 r1 | 7/7 heads (= HAND) | clock .91 (1), square .92 (0) | 1–1.13; HAND-PC R² 1 | 0 | MLP2 .95 | 1 / 1 |
| TRAIN T s0–2 r1 | 24 / 15 / 25 of 32 heads | relative step .89–.94 (3), `or` .72–.80 (4), `next` .66–.77 (4) | 16–32; < .04 | .44–.65 | MLP3 .93–.96 | .38–.76 |
| TRAIN A s0 r1 | 9/32 | wall .85 (1), target .84 (3) | 1–3.5; .15–.57 | .08–.42 | MLP0 | .90 / .90 |
| TRAIN T r2 (s0 b/f, s1 b, s2 b/f) | 17–20 of 27–30 | `or` .78–.84 (4), `next` .74–.84 (4), self .66–.71 (4) | 18–32; ≤ .02 | .64–.67 | MLP3 .90–.98 | 1 / 1 (diag batch s1/s2 .94–.97) |
| TRAIN A r2 | not landed | | | | | .94 / .93 |
Failures explained: TRAIN T r1 errors are CHECK → step-index/END decisions; every map cell is correct, and in s0 interchange on `or` /
`next` at layer 4 flips 1 / .88 (nf 72 / 77), so the update variables are used — the failure is the stop/route decision. TRAIN A fails
by distance (exact to d = 3, falling to d = 10). TRAIN T r2 s1/s2 score 1.0 on the 176 test mazes but .94–.97 on the 32-maze diagnostic
batch: "solved" is at the ≥ .99 bar on the test set, not exactness on all mazes.

## Closed form
**PART T (accepted at the substitution bar):** round/clamp the six fetched inputs to bits r, compute s = Σr − 5w and a = ReLU(s) −
ReLU(s − 1), and replace the whole MLP2 output by Y = (1, a, w)·B in the three state channels (zero elsewhere); B (9 scalars) is fitted
to the MLP's outputs on all 64 local interfaces, never to maze answers. All three seeds, best and final: 176/176 → 176/176, agreement
176/176 (bar ≥ .99 × native). This is output equivalence on valid inputs: the learned local argmax matches the hand-built rule on 48/48
valid tuples (53 / 59 / 53 of all 64). Matched zero/shuffle controls are open. **TRAIN:** two candidates fitted on the r1 nets are
rejected: T as an OR of decoded bits with affine encoding (fresh 104 → 2, 70 → 0, 135 → 1 of 176) and A as a decoded-distance tent
(156 → 37, 149 → 56, 156 → 32). The solved r2 TRAIN nets have no substitution test yet.

## Weights excerpt and reading
Hand-built update rows (N, E, S, W, self, wall) = (1, 1, 1, 1, 1, −5), biases 0 / −1. Fitted PART B rows (unreached, reached, wall),
rounded to 4 decimals (least-squares fit, best = final within each seed):
| Seed | b (constant) | a (reached hinge) | w (wall) |
|---:|---|---|---|
| 0 | .2154, .0521, .0225 | −.4179, .3165, .0402 | −.4173, .1637, .2909 |
| 1 | .2118, .0417, −.0276 | −.3765, .2925, .1149 | −.5479, .2123, .3715 |
| 2 | .2034, .0461, .0129 | −.3393, .2704, .0555 | −.4075, .1781, .2830 |
Trained PART seed 0, layer-2 MLP (`part_T_best_s0.safetensors`; W_in[j, 34:40] over (N, E, S, W, self, wall), b_in[j], W_out[40:43, j]):
| Neuron | Input row | Bias | Output (unreached, reached, wall) |
|---:|---|---:|---|
| 2 | .0473, .0498, .0494, .0501, −.0517, .1399 | −.0002 | −.1453, −.0856, .1214 |
| 4 | .1252, .1286, .1081, .1362, .0936, .1428 | .0001 | −.1406, .0551, .1290 |
| 19 | .0762, .1075, .0851, .0831, .0490, .1213 | .1068 | .0878, .0926, .1203 |
Reading: the learned rows mix neighbour, self and wall evidence across neurons rather than reproducing the two hand-built hinges
(neuron 2 opposes self to the neighbours and favours the wall output; neuron 4 suppresses unreached; neuron 19 is a shared offset).
Only their sum is the rule: in B, a raises reached and lowers unreached by .27–.42, w raises wall, so argmax gives `wall ? wall :
OR(neighbours, self)` on every valid tuple.

## Comparison
PART keeps the hand-built algorithm in the hand-built basis (JS 0, PR ≈ 1) and learns the update as a different but equivalent hinge
sum. TRAIN T learns the same per-step update variables (decoded at layer 4, used by interchange) in a distributed code (hand-built CKA
≤ .02). The scratchpad supplies the serial BFS steps a fixed-depth answer-only net lacks, consistent with depth-bounded results for
graph connectivity (arXiv 2510.19753, 2025, preprint) and fixed-depth limits of log-precision transformers (Merrill & Sabharwal, 2023);
scratchpads as intermediate computation follow Nye et al. (2021, preprint).

## Controls and limits
T scores are teacher-forced; greedy rollout and OOD (distance 11–20, 8×8 grids) are unmeasured for TRAIN. No hand-built T+ or A PART
cell. The ten-seed final is not run, so seed counts are screening counts (3). The A retry proposal (depth 8 or distance curriculum)
is not run. Size counts are small-batch mean-ablation counts, not minimal circuits.

## References
- Merrill, W., & Sabharwal, A. (2023). The Parallelism Tradeoff: Limitations of Log-Precision Transformers. *Transactions of the
  Association for Computational Linguistics*, 11. https://doi.org/10.1162/tacl_a_00562 (journal article)
- Nye, M. et al. (2021). Show Your Work: Scratchpads for Intermediate Computation with Language Models. arXiv preprint.
  https://doi.org/10.48550/arXiv.2112.00114 (research preprint)
- (2025). Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data.
  arXiv preprint. https://doi.org/10.48550/arXiv.2510.19753 (research preprint)
