# Polynomial integration with reduced fractions (integ) — ENTRY 2026-11-08
**Task.** Prompt `BOS M:m C:c₀ … C:cₘ₋₁ =`; per term j emit `J:j N:c/g D:j/g`, g = gcd(|c_{j−1}|, j) (zero = 0/1). Typed integer
tokens. **A** = answer only; **T** (scratchpad, main) first writes every subtractive-Euclid row `A:a B:b` ((a,b)→(|a−b|,min)) until
b = 0, then `G:g N D`. Train: 2–4 terms, c ∈ −9…9 without ±8, fractions ±2/3, ±3/4 withheld; ~80/20 whole-polynomial hash split.
**Status.** HAND A/T exact. Screen 3 seeds × {A,T} × {PART,TRAIN} all solved; final **T TRAIN 9/10** (s4 .875 at the 30k budget).
## Summary
Scratchpad cuts PART stop 26,000 → 5,500 steps and is the only cell that transfers (withheld fractions 1.0, 5 terms .59–.88) but slows
TRAIN 14,500 → 27,000. TRAIN nets solve IID with a distributed MLP-heavy code, not HAND's (align fit R² < 0, CKA ≤ .27); their verified
pieces are L0 lexical prefix pooling (exact formula) and, in one seed, a coefficient reader. Euclid state is used by ≤ 2.5–5k steps, the
signed numerator last (flip 5–12.5k, 90% test 14.5–22k); the one unsolved seed fails exactly at numerator emission.
## Setup
AdamW lr 1e-3, wd .1, warmup 1,000, cosine 30k, clip 1, batch 16 fresh draws/step, eval 500, ckpt 2,500, plateau 5k after 22.5k;
PART fp32, TRAIN TF32, re-scored fp32. 256 held-out polys/seed, teacher-forced whole-output exact. Runs: PART `train-37a3d5`, A TRAIN
`train-36ab3a`, T TRAIN `train-b23cca` (s0–2) + `train-2c76ff` (s3–9). One GPU per pack; diagnostics and substitutions on CPU. Networks: huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-integration.
## Architecture as used
GPT-NeoX, 4 layers, d_model 64, 4 heads × 16, ReLU MLP 384 (A) / 448 (T), no LayerNorm, NoPE, vocab 130, context ≤ 218. TRAIN learns
all (A 16 heads + 1,536 neurons, T 16 + 1,792). PART: HAND frozen, L3 MLP re-initialised and trained (W_in masked to HAND columns).
Archive `hand_{A,T}`, `{part,train}_{fmt}_{best,final}_s<k>` safetensors + config/manifest (seed, rung, fp32 re-score, sha256) on HF.
## Prediction before training (commit 0e88b126)
PART T ≥ 2/3 by 10k: held (5.5k). PART A ≤ 1/3 by 30k: refuted (3/3, 26k). TRAIN T ≥ 2/3 by 25k: held (screen 2/3 at 24k; final 7/10
first-exact ≤ 24.5k). TRAIN A ≥ 1/3: held (3/3, 14.5k). PART length transfer ≥ .90: refuted (.59–.88); T > A on withheld fractions: held for PART only.
## Hand-built reference
A: reduction table as ReLU hinges on the (c, j) code; 4 heads + 428 neurons. T: coefficient/register fetches, ABS = ReLU(a−b)+ReLU(b−a),
MIN = a − ReLU(a−b), signed quotient staircases; 6 heads + 769 neurons (665 framing). Exact on 2,804 polys/format, margin ≥ 10.
## What training finds
| Cell | Solved | First ≥ 90% test (per seed) | First exact (per seed) | Stop | 5 terms | withheld fractions |
|---|---|---|---|---|---|---|
| A PART | 3/3 | 19.5k, 22.0k, 20.0k | 24.5k, 23.0k, 20.0k | 26.0k | .05–.12 | 0 |
| T PART | 3/3 | 2.0k, 2.0k, 2.5k | 3.5k, 2.0k, 3.0k | 5.5k | .59–.88 | 1.0 |
| A TRAIN | 3/3 | 8.0k, 7.5k, 11.5k | 11.5k, 10.0k, 14.0k | 14.5k | 0 | 0–.19 |
| **T TRAIN final** | **9/10** | s0–2 16.0k, 16.0k, 22.0k; s3–9 14.5k, —, 18.0k, 22.0k, 16.5k, 20.5k, 17.5k | 19.0k, 24.0k, 27.0k, 17.5k, —, 22.5k, 27.0k, 22.0k, 24.5k, 20.5k | 27k / 30k | 0 | 0 |
OOD (best, 64/seed): every TRAIN net 0 at 5/6/8 terms and forced ±8 (new M/J token ids, not a pure length test). No mem/stall alert
in any of 11 trajectories. Failed s4 (best = final = step 30k): exact .875, numerator token .972, every other role ≥ .999; numerator
inter .933 (nf 75), DAS .920, gcd DAS .984: state present, signed-numerator selection at G→N fails. Circuit types (seeds once): A PART 3/3
frozen reader + learned response; T PART 3/3 frozen Euclid interface + response; A TRAIN 3/3 distributed four-MLP (sign bottleneck
L3 s0/s2, L2H3 s1); T TRAIN 10/10 one MLP-heavy family (every MLP lesion → exact 0), sign-transport subtypes 1/1/1 in s0–2.
## Diagnostics (best snapshot; B/F agree unless noted; N = guarded activity upper bound, not a joint minimum)
| Cell | Heads mean/zero-sensitive; neurons | Probe R² (layer): num / den / gcd / next | Inter flip num / den / gcd | r90 res L4 (HAND 1); MLP3 in/out | L3 JS | Best / final |
|---|---|---|---|---|---|---|
| A PART | 1/4; ≤ 380–394 | .90–.91 / .84–.85 / — / .49–.53 (4) | 1 (s2) / .35–.44 / — | 1; 4–5/4 | 0 | 1 / 1 |
| T PART | 4/6; ≤ 667–687 | .86–.91 / .59–.67 / .42–.61 (4) | U– / .77–.98 / .56–.90 | 1; 10/10–11 | 0 | 1 / 1 |
| A TRAIN | 2–7/8–9; ≤ 1324–1358 | .49–.88 / .93–.95 / — / .39–.81 (4) | 1 / 1 / — | 7–8; 13–15/13–15 | .10–.56 | 1 / 1 |
| T TRAIN solved | 12–15/15–16; ≤ 1360–1573 (s0–2) | .27–.74 / .96–.99 / .84–.96 (3–4) | 1 / .89–1 / .29–1 | 9–15; 17–21/17–19 | .39–.60 | 1 / 1 |
| T TRAIN s4 | 16/16; ? | .42 / .97 / .80 (2–4) | .93 / .97 / .94 | 9; 15/17 | .47–.55 | .875 / .875 |
HAND controls: A num/den R² .999/.915, inter den 1; T HAND JS 0. Low TRAIN numerator R² with inter 1 = used, coded non-linearly.
## Closed form
T TRAIN layer 0: `P_h(t) = Σ_u N_u e^{S_h[t,u]} v_h[u] / Σ_u N_u e^{S_h[t,u]}`, S = Q(E_t)·K(E_u)/4, v = OV(E_u), N_u = prefix count:
substitution keeps every prediction, 20/20 s0–9 snapshots (error ≤ 1.8e-7). Layer-2 G-query reader `b_h + u_h[c] + v_h[j]`:
PASS s0 (512/512, also hard fetch `1[k=j+1]`), s1 H1 only; FAIL s2 (≤ 502/512); bar ≥ .99 × native on 512 fresh polys. No gcd/division form.
## Weights excerpt and reading (T TRAIN s0 best, L0 head 0, columns 0:3; raw float32 weight slice)
Q₀ −0.0403, −0.1027, −0.0268 (b .0550); K₀ −0.0468, −0.0292, −0.2562 (b .0424); V₀ .0778, .0074, −0.0265 (b −.0091).
S₀[G:1, C:{−3,0,3}] = [−.222, .185, −.345]: tokens count by prefix and lexical score, no scalar coordinate; MLP r90 17–21 vs HAND 3/6.
## Comparison
HAND: 6 heads + 769 neurons; T TRAIN 12–16 heads, ≤ 1.4–1.6k active neurons, different code. Scratchpad: fastest and only transferring
PART; slower TRAIN, 9/10 at 30k. Answer-only A TRAIN learns faster (14.5k) but never transfers.
## Controls and limits
Edges: A PART J .5 / recall 1, A TRAIN J .2 / recall .33 (gates pass); T edges fail the HAND gate (no wiring claim). LMC barriers TRAIN
3.5–9.4 nats (separate basins). Neuron counts are upper bounds; s3–9 sign subtypes unassigned; 8 of 19 trajectories lack dynamics.
## References
Lample & Charton 2019 (arXiv:1912.01412); Nye et al. 2021 (arXiv:2112.00114); Power et al. 2022 (arXiv:2201.02177).
