# Polynomial differentiation by the power rule (deriv)

This entry studies how a 3-layer transformer with one head per layer differentiates a degree-3 integer polynomial term by term, given one token per
integer, either writing each derivative term directly (answer only) or first copying the exponent and coefficient (a scratchpad). With
the addressing frozen from the hand-built network, one masked MLP learned the low-bit multiplier in all 10 seeds of each format within
2–3k steps, but never the multiplier bit it had not seen. Trained from scratch, the network needed width 2 and 60k steps and solved
6/10 seeds in the selected final cell (scratchpad format); the failed seeds miss only the position-indexed copy of a_k, stuck near
1/3. The task is partly solved (frozen parts 10/10 per format; from scratch 6/10). The closed form is partial: on the frozen parts,
MLP2 = k·a on the two observed bits keeps 1.000 on 40/40 snapshots when substituted, with the complement not tested; from scratch
there is only a numerical compression, checked in 4 of 6 solved seeds.

| Fact | Value |
|---|---|
| Task | Differentiate Σ a_k x^k (degree 3) term by term, one token per integer |
| Input → output example | Answer only: `BOS 5 4 1 1 =` (coefficients a_0 to a_3) → `4 0 2 1 3 2`, the pairs (k·a_k, k−1) |
| Model | GPT-NeoX-shaped, 3 layers × 1 head, d_model 16, ReLU MLP 48, NoPE, no LayerNorm; the partly trained networks train only a masked MLP2 |
| Rungs and seeds | HAND `A` and `T`; PART final 10 seeds per format; TRAIN 3-seed screens at width 1 and 2, then a 10-seed final (`T`, width 2, 60k steps) |
| Closed form status | Partial (PART: MLP2 = k·a on the two observed bits, 1.000 on 40/40 snapshots on substitution; complement not tested. TRAIN: numerical compression only, 4 of 6 solved final seeds) |
| Status | Partly solved (PART 10/10 per format; TRAIN `T` width 2 final 6/10); last run 2026-11-08 |
| Weights | [Hugging Face: polynomial-differentiation](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-differentiation) |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

## 1. Task and data

The input is `BOS a0 … a_d =`, and the output is the derivative of Σ a_k x^k in ascending k, with no zero trimming. Each integer is one
token. Two output formats were studied (Table 1); the [scratchpad](../GLOSSARY.md#task-formats) format writes the exponent and the
copied coefficient before each product.

*Table 1. Output formats.*

| Format | Output per term |
|---|---|
| `A` (answer only) | (k·a_k, k−1) |
| `T` (scratchpad) | (k, a_k, k·a_k, k−1) |

For the example `BOS 5 4 1 1 =`, format `A` writes `4 0 2 1 3 2` and format `T` writes `1 4 4 0 2 1 2 1 3 1 3 2`.

The hand-built network was checked on degree 1–6 and coefficients −15..15. Training used degree 3 and coefficients −9..9; the pairs
(k, a) = (2, ±3) and (3, ±2) were withheld as inputs, although their product labels still occur elsewhere. Training data were fresh
hash-disjoint draws with replacement from 104329 primary polynomials, split into 103251 for training and 1078 for testing. An output
counts as correct only on exact match of the whole teacher-forced output ([glossary](../GLOSSARY.md#task-formats)).

## 2. Model

The shared architecture and instrumentation are described in [METHODS.md](../METHODS.md). The three
[rungs](../GLOSSARY.md#the-three-rungs) are HAND (every weight set by formula), PART (frozen hand-built parts with one trained stage)
and TRAIN (all weights trained from scratch).

### 2.1 Architecture

All networks are GPT-NeoX-shaped with 3 layers × 1 head, d_model = d_head 16 and a ReLU MLP of width 48, a vocabulary of 183, context
64, an untied read-out, no LayerNorm and no positional encoding. TRAIN at width 2 doubles d_model and the MLP to 32/96.

### 2.2 Frozen parts and trained part

PART freezes everything except a masked MLP2 (the layer-2 MLP): W_in 48×16 reads only the input columns AA = 12 (the fetched
coefficient) and the multiplier bits 9–11, and the output writes only column S = 13. This leaves 289 free scalars.

### 2.3 Training

The optimiser was AdamW with learning rate 1e-3, weight decay .1, warmup 1000, a cosine schedule, gradient clip 1 and batch 256.
Evaluation used 1024 draws every 500 steps, with a checkpoint every 2500 steps. PART ran in fp32 with a 30k budget; TRAIN used TF32
updates and fp32 evaluations. The selected TRAIN final ran format `T` at width 2 for up to 60000 steps (minimum 45000, plateau stop
5000, seeds 3 … 12), which is 14.72M draws per seed; it stopped on the plateau at 57.5k. The PART final ran seeds 0–9.

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-differentiation](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/polynomial-differentiation).
The training command of the selected final, for reference, was
`train.py --fmts T --kinds train --width 2 --steps 60000 --min_steps 45000 --plateau 5000 --tf32 1 --seeds 3 … 12`
(`launch_final_train.sh`); the PART final was `launch_final_part.sh`, seeds 0–9.

### 2.5 Archive

All cells finished and were diagnosed: 108/108 trained best and final snapshots plus 2 HAND controls. The networks are on Hugging Face
at the link above.

## 3. Hand-built solution

The hand-built network builds a position frame, fetches each coefficient by position and multiplies it by k with signed bit gates
(Figure 1).

Uniform attention to BOS gives P = 1/(r+1) at position r. Reciprocal hinges recover r/64 and r²/4096. An anchor on `=` and
relative-step thresholds write the term and phase, the multiplier bits of k and the index of the coefficient to fetch.
Quadratic-score attention, with score 24[2(k+1)t − t²], fetches a_k. Six signed bit gates, each passing ±a only when bit j of k is
set, sum a·2^j into S = k·a_k, and an integer read-out with logits 30(2vS − v²) picks the integer closest to S. The sizes are 3 heads
+ 48 neurons in format `A` and 3H+82N in format `T`. The network is exact on 34320 polynomials per format (exhaustive at degree 1 and
2, stratified at 3–6) and on 217 primitive products, with minimum margin 30.0 (`A`) and 29.9998 (`T`).

```mermaid
flowchart TD
    P["Prompt: BOS a0 ... a_d ="] --> U["uniform BOS attention: P = 1/(r+1)"]
    U --> HG["reciprocal hinges: r/64, r^2/4096"]
    HG --> AN["anchor on = and step thresholds: term, phase, multiplier bits, index"]
    AN --> F["quadratic-score attention H2: fetch a_k"]
    F --> M["MLP2: signed bit gates, S = sum over set bits j of a times 2^j = k a_k"]
    AN --> M
    M --> R["integer read-out: logits 30(2vS - v^2)"]
    classDef trained fill:gold,stroke:darkgoldenrod,color:black
    class M trained
```

*Figure 1. The hand-built network. A frame from uniform attention and hinges locates each term, the layer-2 head fetches a_k, and the
bit gates of MLP2 form k·a_k for the integer read-out. The shaded MLP2 is the part PART replaces with a masked trained stage.*

## 4. What training finds

Table 2 gives every cell. "Grok" is the first evaluation at which held-out exact match reached ≥ .99
([grokking](../GLOSSARY.md#counting-results)). Training data were fresh, so train accuracy equals test accuracy and there is no
memorisation.

*Table 2. Results per cell. Seeds in brackets; best and final held-out (IID) exact match, per seed where three are listed.*

| Cell (seeds) | Solved | Best → final IID exact | grok (≥ .99), k steps |
|---|---|---|---|
| PART A / T final (0–9) | 10/10 each | 1 → 1 | 2–3 |
| TRAIN w1 A / T (0–2), 30k | 0/3 / 1/3 | A ≤ .09; T .007/.998/.073 | T s1 26 |
| TRAIN w2 A / T (0–2), 30k | 1/3 / 0/3 | A .29/.89/1; T .17/.83/.22 | A s2 14.5 |
| TRAIN w2 A / T (0–2), 60k | 2/3 / 2/3 | A .56/1/1; T .34/1/1 | A 19–22; T 21–27 |
| TRAIN w2 T final (3–12), 60k | 6/10 | solved 1; s3/4/9/11 final .33/.33/.11/.35 | 16.5–49 |

Key: w1, w2 = width 1 (d_model 16) and width 2 (d_model 32); s1 = seed 1.

Neither format won uniformly: at width 1 the scratchpad solved 1/3 against 0/3 for answer only, at width 2 with 30k steps 0/3 against
1/3, and at width 2 with 60k steps 2/3 each.

### 4.1 Circuit types

Circuit types were assigned by donor-flip query patches on the copied a_k. The products read the trace: for k = 2 and 3, the trace
flip is .56–1 against ≤ .14 for the input. PART has one [circuit type](../GLOSSARY.md#counting-results), a masked bit-gate
multiplier, in 10/10 seeds per format. In the TRAIN `T` final, 4 of the 6 solved seeds were analysed (the best snapshot has the same
type as the final one), and they split by copy routing (H1 and H2 are the attention heads of layers 1 and 2): split copies (a_1 by H1, a_2 and a_3 by H2) 1/4; first copy by H2, later
copies by H1 1/4; H1 copies with H2 mixed or idle 2/4. The screens add early-fetch and late-fetch types in format `A` and a late-copy
type in format `T`, so there is no single TRAIN circuit.

### 4.2 Why TRAIN fails

The failed TRAIN seeds learned the multiply and failed at the copy. In the final cell (err/tok_acc on the final 10), seeds 3, 4 and 11
are wrong only at token 1, the copy of a_1 (token accuracy .32–.35), with all 11 other tokens at 1.0; seed 9 also fails at token 5, the
copy of a_2 (.45). The failed width-2 60k `T` seed 0 shows the same pattern. Answer interchange stays at .81–.89 and patching needs all
six components, so the product and exponent path works; the missing variable is the a_k fetch at the copy slot. This is
consistent with the index–operand alignment bottleneck discussed by [Cho et al. (2024)](#references). The solved TRAIN networks carry k and
the exponent linearly (R² > .9 at L1) but a and the product only weakly (< .5): their code is categorical and of higher rank, unlike
HAND's scalar code (hand_cka, the CKA similarity to HAND's activations, ≤ .06). Full diagnostics are in Appendix B.

### 4.3 Length and range

On the length and out-of-distribution sets (final 10 TRAIN seeds, 256 rows), degree 1, 2, 4, 5 and 6, the unseen pairs and wide
coefficients all score 0–.016 exact at the final snapshot (best ≤ .062) in all ten seeds, solved ones included. Summarised over these sets (degree 1, 2
and 4–6, unseen pairs, wide coefficients), the solved TRAIN networks do not extrapolate (≤ .06). PART scores 1.0 at
degree 1 and 2 and 1.0 on the unseen pairs in 9/10 seeds, and 0 on wide coefficients. PART keeps the old terms exact under extension
and fails only the multiplier bit it never saw (degree 4 .05, degree 5 and 6 0).

## 5. Mechanism and verification

### 5.1 PART closed form

MLP2's whole write is ΔS = c + u·g(a, b0) + v·g(a, b1), with g = ReLU(a + 32b − 32) − ReLU(−a + 32b − 32) = a·b, and the fit gives
(c, u, v) ≈ (0, 1, 2). On the same 1024 held-out draws, in 40/40 snapshots, the substituted form scores 1.000, equal to the network.
The controls fail: zero or mean replacement 0, a row shuffle of (a, b0, b1) .002, and flipping u and v 0. Seed-swapped constants or
HAND's (0, 1, 2) also score 1.000, so ΔS = a·(b0 + 2b1) = k·a. The circuit is the frame plus 4 gates and 3 constants (3H+46N in `A`,
3H+80N in `T`), valid only in-distribution at degree 3, because bit b2 never occurs in training. The closed form is supported by
substitution; no [resample ablation of the complement](../GLOSSARY.md#verifying-a-circuit) was reported, so its status is partial.

### 5.2 TRAIN numerical substitute

For TRAIN, every head was replaced by A = causal_softmax((zQ_L)(zQ_R)ᵀ), ΔH = (A zO_L)O_R + b, and every MLP by
ΔM = ReLU(zI_L I_R)O_L O_R + b, with z = [x; 1] and low-rank factors fitted on internal activations. With all six components replaced
together, seed 7 keeps .997/.998, seed 8 .997/1, seed 10 .9985/1 and seed 12 1/1 (best/final, native 1; 2048 draws), with 8956–15908
of 31488 scalars. In the screens, 12/12 substitutions kept ≥ .99 × native. This is a compression and not a recovered power-rule algebra.

```mermaid
flowchart TD
    P["Prompt: BOS a0 ... a_d ="] --> H1["H1: copies a_k in some seeds"]
    P --> H2["H2: copies a_k in some seeds"]
    H1 --> C["trace slot: copied a_k"]
    H2 --> C
    C --> M["MLPs: categorical, higher-rank product code"]
    K["k and exponent, linear at L1"] --> M
    M --> Y["written k a_k and k-1"]
```

*Figure 2. The circuit training from scratch finds in format `T`: the copy of a_k is routed by H1, H2 or both depending on the seed
(4 of 6 solved final seeds analysed), and the product is read from the written trace. Only a numerical compression substitutes for
it; no closed form was recovered.*

### 5.3 Relation to the hand-built network

PART reproduces HAND's function in HAND's basis (JS 0, rank 3, probes equal to HAND's) and fails only the multiplier bit it never saw.
TRAIN finds a different, higher-rank code with seed-dependent copy routing; its hard part is addressing, and the arithmetic is learned.

## 6. Limitations and open questions

Scores in format `T` are teacher-forced; a greedy rollout was checked only for PART and for the solved screening `T` networks. There
are no checkpoint dynamics on the final 10 seeds. The TRAIN circuit types and the numerical substitute cover 4 of the 6 solved final
seeds; two jobs never started because of the compute queue. Repeated seed labels across cells are not independent replications.

## 7. Prediction record

The prediction was recorded before training (commit in Appendix B). PART 3/3 within 5k steps, keeping the old
terms under extension and failing the unseen multiplier bit, was confirmed. TRAIN ≥ 2/3 per format by 30k was refuted (`A` 0/3, `T`
1/3); TRAIN needed width 2 and 60k steps (screening `A` 2/3, `T` 2/3; `T` final 6/10).

## Appendix A. Weights excerpt and reading

HAND gate rows have (W_AA, W_bit, bias) = (±1, 32, −32) and outgoing S weights ±2^j; the fetch score is 24[2(k+1)t − t²] and the integer
logits are 30(2vS − v²). Gate j passes ±a only when bit j of k is set, S sums a·2^j = k·a, and the read-out picks the integer closest
to S.

In the PART final networks (MLP2 rows with the 4 largest |W_out[S]|, all 20 nets), W_AA is ±.06–.11, W_b0 and W_b1 are .0–1.16, W_b2
≈ 0 (|.04| at most), the bias is −.46…−.96 and the output to S is ±1.10–1.48, with sign(W_AA) = sign(out S). These are the same
±a·bit gates spread over many soft rows (a small a-slope, with the bias cancelling the bit weight); no row reads the unseen bit b2.

## Appendix B. Detailed evidence

*Table B1. Diagnostics per cell.*

| Cell | Used H+MLP | R² k / a / product / multiplier | rank | JS | best → final |
|---|---|---|---|---|---|
| HAND A / T (formula 3H) | 1+1 | 1/1/1/1 (A); 1/1/.87/.999 (T) | 3 | 0 | 1 → 1 |
| PART A, T (13 seeds each) | 1+1 | = HAND to 3 decimals | 3 | 0 | 1 → 1 |
| TRAIN w1 T (0–2) | 3+3 | .27–.90 / .10–.25 / .07–.17 / .31–.37 | 3–4 | .45–.66 | Table 2 |
| TRAIN w2-60k A (0–2) | 3+3 | .94–.98 / .13–.23 / .14–.20 / .85–.93 | 2–11 | .29–.47 | .56/1/1 → .56/1/1 |
| TRAIN w2-60k T (0–2) | 3+3 | .94–.96 / .24–.37 / .18–.29 / .46–.58 | 6–9 | .52–.60 | .34/1/1 → .32/1/1 |
| TRAIN w2 T final solved (5,6,7,8,10,12) | 3+3 | .91–.97 (L1) / .23–.49 / .16–.43 / .41–.59 | 2–12 | .53–.62 | 1 → 1 all |
| TRAIN w2 T final failed (3,4,9,11) | 3+3 | .89–.95 (L1) / .20–.32 / .17–.25 / .51–.62 | 5–13 | .54–.59 | Table 2 |

Key: R² = best linear-probe R² over layers ([glossary](../GLOSSARY.md#analysis-measures)); rank = residual r99 at L3 (HAND 3); JS =
L2H0 attention divergence to the same-index HAND head; Used = heads and MLPs whose mean-ablation drop is > .05 (neurons not counted).
The HAND and PART frames survive mean ablation, so only H2 and MLP2 show as used. TRAIN interchange on `answer` at L3 gives flip
.81–1.0, with DAS ≈ interchange. The EAP-IG edge check gives spearman .49–.99 (seed 8 .49: its edges are not trusted).

The prediction file was committed as 61df19ce in the project repository.

## References

- Cho, H., Cha, J., Awasthi, P., Bhojanapalli, S., Gupta, A., & Yun, C. (2024). Position coupling: Improving length generalization of
  arithmetic transformers using task structure. *arXiv preprint*. https://doi.org/10.48550/arXiv.2405.20671 (index–operand alignment
  as the arithmetic bottleneck)
