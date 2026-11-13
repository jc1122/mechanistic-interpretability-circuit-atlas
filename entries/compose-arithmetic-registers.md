# Composing arithmetic through scratchpad registers

A register is a named place to store a number; using the right name matters as much as doing the arithmetic. A small transformer was trained to execute register programs in binary and write each intermediate step, a [scratchpad](../GLOSSARY.md#task-formats). Across all 10 final independent training runs (seeds, [glossary](../GLOSSARY.md#counting-results)), familiar compositions passed the recorded full-trace accuracy screen of ≥ .9, yet freely named one-step operations failed; the task remained unsolved, with 0/10 final seeds solved and mean trained-cell accuracy .871-.897. Diagnostics support fetching operands by position rather than register name; no faithful register-tag mechanism was recovered. Freezing a hand-built tag router let the local arithmetic stage become exact in 3/3 screening seeds; its formula is partial, with replacement accuracy 1.0 in 18/18 tests, but tests resampling everything outside the claimed circuit (the complement, [glossary](../GLOSSARY.md#verifying-a-circuit)) were restricted to final snapshots and only one seed for shift and reverse.

| Fact | Value |
|---|---|
| Task | Execute named-register operations and compositions on operands up to 15 bits, writing the complete trace |
| Input → output example | In words: a STEP that ADDs registers holding 101 (5) and 11 (3) writes 1000 (8) |
| Model | 4 layers; register-line format: 16 attention heads, residual width 256, no positional encoding or LayerNorm. Frozen-router format: only the layer-3 feed-forward block (MLP3) is trained |
| Rungs and seeds | Formula-set reference (HAND), frozen-router stage (PART), from scratch (TRAIN). Final: TRAIN register-line format, full composition arm, 10 seeds. PART: 3 seeds per arm; TRAIN screens: 3 or 10 seeds by format |
| Closed form status | Partial (STEP-only PART screen: substitution 1.0 in 18/18 tests; complement 1.0 on final snapshots, ADD seeds 0-2 and SHL/REV seed 0 only; none for TRAIN) |
| Status | Unsolved (from-scratch register-line/full-composition cell, TRAIN O c, 0/10); solved frozen-router STEP-only cell (PART S a, 3/3, screen); last run 2026-11-08 |
| Weights | [Hugging Face: compose-arithmetic-registers](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/compose-arithmetic-registers) |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

## 1. Task and data

The machine works in base 2. A STEP line runs ADD (addition), SHL (shift left) or REV (reverse the digit order) on registers named by
tags in the prompt header, with operands up to 15 bits. For example, ADD on registers holding 101 (5) and 11 (3) writes 1000 (8).
SUM k (k operands), MUL n (n multiplier bits) and POW e compose STEP lines. Each header names a program counter and the register
tags, and every digit carries a significance tag. The network writes the whole trace, which is a
[scratchpad](../GLOSSARY.md#task-formats), a sequence of intermediate lines written before the answer. Two trace formats were used
(Table 1). Fresh samples were drawn at every training step; the trainer evaluation sets hold 128 samples per cell. The later
mechanism diagnostics used separate 400-row banks; their accuracies need not equal the trainer's. A trace counts as correct
only when every [teacher-forced](../GLOSSARY.md#task-formats) trace token is exact (each token scored given the correct previous
tokens).

*Table 1. Trace formats and training arms.*

| Label | Meaning |
|---|---|
| `O` | the trace writes register lines only |
| `S` | the trace also writes the gathered bits and the carry |
| arm `a` | STEP only |
| arm `b` | STEP plus SUM2-3, MUL1-2, POW2 |
| arm `c` | STEP plus SUM2-5, MUL1-4, POW2-3 |

The depth+1 cells SUM6, MUL5 and POW4, one composition step deeper than any trained cell, were never trained and test depth
generalisation.

## 2. Model

The shared architecture and instrumentation are described in [METHODS.md](../METHODS.md). The three
[rungs](../GLOSSARY.md#the-three-rungs) are HAND (every weight set by formula), PART (frozen hand-built parts with one trained stage)
and TRAIN (all weights trained from scratch).

### 2.1 Architecture

Format O (TRAIN): 4 layers, d_model 256, 16 heads × 16, MLP 1024, ReLU, no positional encoding (NoPE), no LayerNorm, untied readout,
vocabulary 296; all weights are trained. Format S (PART): 4 layers, d_model 208, 13 heads × 16, MLP 134, vocabulary 424.

Here d_model is the running-state width. An attention head reads information from earlier positions; the MLP is a feed-forward block at each position, and ReLU keeps positive responses and sets negative ones to zero. LayerNorm would centre and rescale the state; NoPE means no added position vectors. Untied readout means separate input-embedding and output weights ([glossary](../GLOSSARY.md#standard-machine-learning-terms)).

### 2.2 Frozen parts and trained part

In format S the hand-built network is frozen and only the layer-3 MLP is trained: W_in [134, 208] and W_out [208, 134], masked to
HAND's stage ports, which is 8,728 parameters. A port is a named input or output channel; the mask permits connections only through those channels. The stage reads the ports that HAND's own column stage reads (Section 3).

### 2.3 Training

The final run report was dated 2026-11-08; the later diagnostic backfill was dated 2026-11-09. The optimiser was AdamW with learning
rate 1e-3, weight decay 0.1 and betas (.9,.98), warmup 1000 steps, a cosine schedule and gradient
clipping at 1. TRAIN ran in TF32; PART and all evaluations ran in fp32. Each training step used one cell (STEP .4, compositions .6),
with a batch of ≈ 8000 tokens, and the networks were evaluated every 1000 steps. The final cell, O c, used seeds 0-9 and 30k steps;
the screens used 3 seeds (S) or 10 seeds (O). Each run used a single consumer or data-centre GPU.

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/compose-arithmetic-registers](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/compose-arithmetic-registers).

### 2.5 Archive

The folder holds `hand_O` and `hand_S` (hand-built networks), `part_S_a_{best,final}_s0-2` (the 4 stage tensors of each PART seed),
`train_O_c_final_{best,final}_s{0,1,8}`, `config.json` and a manifest (seed, rung, accuracy, sha256).

## 3. Hand-built solution

The hand-built network finds each operand by its register tag and adds one column per step (Figure 1). Layer 0 head 0 (L0H0) attends
uniformly and supplies a position code. Eleven layer-1 recency heads, which attend to the most recent matching token, copy the latest
P/A/S/GA/ST fields of the trace; among their outputs are PA and PS (the tags of the registers named in the current A and S fields,
that is, the register addresses), GA_RD (the gathered operand bit) and PC (the previous carry bit). Layer-2 match heads fetch operands
by tag equality: L2H0 gives XA1, L2H1 gives XS0 and L2H2 gives XR (the fetched addend bits), with match scores 30(2v·id − v²).

The layer-3 MLP is the column stage. It computes t = GA_RD + NUM + CI, where NUM is the fetched addend bit and CI = ReLU(PC + NF − 1)
is the carry-in (the previous carry PC, passed only when NF, a layer-1 flag set from the column index for every column but the
lowest, is on). It writes the digit d = t mod 2 and the carry
c' = ⌊t/2⌋; SHL and REV copy GA_RD. The network was exact on the tested examples (margin 30): every cell in base 2 used 200 examples,
and every cell in bases 8/10/16 used 50 examples. It uses 21/64 heads. The router is bounded by 16 digits and 40 register tags;
long-product sampling in bases 8/10/16 resampled examples to fit that tag table, so MUL4-5 there were biased towards small
multipliers. These finite tests do not enumerate all register programs. The set of ADD-stage neurons is shared by STEP, SUM and MUL (Jaccard 1.00), so one adder
serves every task.

In Figure 1, position and recency heads recover register addresses, the gathered operand bit and the previous carry. Tag-match heads fetch the other bit; MLP3 combines the bits and incoming carry to write a column, or copies a shifted or reversed bit. Only MLP3 is trained in PART. Status remains partial because complement coverage is restricted to the named final snapshots.

```mermaid
%%{init: {"themeVariables": {"fontSize": "24px"}, "flowchart": {"nodeSpacing": 20, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    accTitle: Hand-built tag fetch and arithmetic column stage
    accDescr: Position and recency heads recover register addresses, operand bits and the previous carry. Tag-match heads fetch addends, then the column MLP writes a digit and carry or copies the shifted or reversed operand bit. Only the column MLP is trained in PART. Status is partial because complement coverage is restricted to the named final snapshots.
    P["Prompt<br/>header:<br/>program<br/>counter,<br/>register<br/>tags; trace<br/>so far"] --> H0["L0H0:<br/>uniform<br/>attention,<br/>position<br/>code"]
    H0 --> R["11 L1<br/>recency<br/>heads: copy<br/>latest<br/>P/A/S/GA/ST<br/>fields"]
    R --> PA["PA, PS:<br/>register<br/>addresses"]
    R --> GA["GA_RD:<br/>gathered<br/>operand bit"]
    R --> PC["PC: previous<br/>carry"]
    PA --> M2["L2 match<br/>heads by tag<br/>equality:<br/>L2H0, L2H1,<br/>L2H2"]
    M2 --> NUM["NUM: fetched<br/>addend bit<br/>XA1, XS0 or<br/>XR"]
    GA --> S3["L3 MLP<br/>column<br/>stage: t =<br/>GA_RD + NUM<br/>+ CI"]
    NUM --> S3
    PC --> CI["CI = ReLU(PC<br/>+ NF - 1)"]
    CI --> S3
    S3 --> D["digit d = t<br/>mod 2, carry<br/>c' =<br/>floor(t/2);<br/>SHL, REV<br/>copy GA_RD"]
    classDef trained fill:gold,stroke:darkgoldenrod,color:black
    class S3 trained
    %% Invisible links constrain vertical layout only, not computation.
    PA ~~~ GA
    GA ~~~ PC
    PC ~~~ M2
    NUM ~~~ CI
```

*Figure 1. The hand-built register machine. Gold MLP3 is the only trained PART block in the gathered-bit format. The column formula
has partial status: substitution passed in all three screening seeds, while complement tests used final snapshots only, ADD seeds
0-2 and SHL/REV seed 0. No closed form was recovered for the final from-scratch cell.*

## 4. What training finds

Table 2 gives the results per cell. The mean is over the trained cells of each arm; "full" counts every trace token exact.

*Table 2. Results per cell (ranges over seeds; run 2 is the retry).*

| Rung / fmt / arm | Seeds | best → final (mean trained cells) | Outcome |
|---|---|---|---|
| TRAIN O c (final) | 10 | .877-.899 → .871-.897 | 0/10; compositions 1.00, STEP ADD/SHL/REV ≤ .59/.61/.54, depth+1 0/10 |
| TRAIN O a / O b (run 1) | 10 each | .51-.68 / .81-.83 | O a run lost; O b: SUM4/MUL3/POW3 never ≥ .9 |
| PART S a (run 1) | 3 | 1.00 → 1.00 | solved 3/3 (STEP primitives) |
| PART S b / S c (runs 1, 2) | 3 each | .968-.970 / .80-.83 | b: untrained depth+1 cells; c: budget-limited, still rising |
| TRAIN S a/b/c (runs 1, 2) | 3 each | .37-.48 | underfit (train .42-.56); 2× budget gave nothing |

### 4.1 Trained from scratch (TRAIN O c)

Trained from scratch on STEP plus SUM2-5/MUL1-4/POW2-3, all 10 seeds passed the composition sub-cell screen at full-trace exact
match ≥ .9, but all 10 failed the bare STEP primitive (final .54-.61). The cell is unsolved (final .871-.897 over the trained
cells). "Compositions 1.00" in Table 2 is the original trainer summary, not a claim that every independent re-score was exact: run 1
reported .97-1.00 across trained compositions, and the separate final SUM5 diagnostic bank gave .99-.9975. The grok step, here
the first evaluation at which full exact match reached .9 ([glossary](../GLOSSARY.md#counting-results)), was 17-18k for SUM5 and 14k
for MUL4; STEP never reached it. At depth+1 the best seed reached, in full and line-digit (exec) exact match, SUM6 .24/.81, MUL5
0/.17 and POW4 .07/.52.

The diagnostics support a failure to use register tags for fetching operands. HAND's PA/PS variables are the addresses named in the
current register fields; their linear readouts are weak ([probe](../GLOSSARY.md#analysis-measures) R² .29-.55 at L2, ~0 at L4;
HAND .99). Weak linear decoding alone would not prove that an address is absent. The diagnosis also uses head ablation, patching,
operand interchange and the contrast between fixed-convention compositions and free-register STEP lines. The carry is present and
causal (interchange flip .96-1.00, the share of answers that change as predicted when the carry is swapped between inputs). These
tests support operand fetching by trace position within the trained compositions' fixed register convention; in a bare STEP line
the registers are free. All 10 seeds fall into one diagnosed failure type, positional-convention fetch with an MLP3 adder:
compositions pass the screen, but a working tag fetch for STEP was not recovered.

### 4.2 Frozen tag router (PART S)

With HAND's tag router frozen and only the column-stage MLP trained (S a PART), the stage learns HAND's rule exactly in 3/3 seeds:
d = t mod 2, c' = ⌊t/2⌋. This is one circuit type, frozen tag router with HAND's column stage (3 of 3). Arm b reached .968-.970,
limited by its untrained depth+1 cells, and arm c reached .80-.83 and was still rising when its budget ended. Trained from scratch in
format S, all arms underfit (.37-.48, train .42-.56), and twice the budget gave nothing.

No alternative HAND was built. A later HAND-free discovery pass did run on the final from-scratch network, superseding the earlier
"not run" record: 13/20 sub-cell/seed reports were complete. It found no match to HAND, but usually did not isolate a small circuit
that retained the network's accuracy. No such backfill ran on the frozen-router networks. Coverage, controls and failures are in
Appendix B.

## 5. Mechanism and verification

### 5.1 Closed form of the PART stage

For the trained MLP3 of S a PART, per column: t = GA_RD + NUM + CI; d = t mod 2; c' = ⌊t/2⌋ (ADD); SHL and REV copy GA_RD with
c' = 0. The closed form has 0 fitted numbers. Substituting HAND's stage for the trained MLP3 gives 1.0 in 18/18 tests (best and final
snapshots, seeds 0-2, ADD/SHL/REV, 256 fresh traces each; native .992-1.0). Zeroing MLP3 gives 0, and HAND's MLP3 with shuffled
hidden-to-output wiring gives 0.

A [resample ablation of the complement](../GLOSSARY.md#verifying-a-circuit), which replaces every component outside the claimed
circuit with its values on other inputs, was run with the circuit L0H0, MLP0, L1H0/1/2/5/7/8/9/11/12, MLP1, L2H0/1/2, MLP2, MLP3. It
gives 1.0 = base (ADD seeds 0-2, SHL seed 0, REV seed 0). Resampling the circuit itself gives .15-.42, and a random set of the same
size gives .19-.54. The complement test used final snapshots only, with SHL and REV on one seed and substitution limited to 5-bit operands. The initial
REV set omitted L2H2, the fetched reverse operand XR: complement accuracy was .315. Adding that head restored 1.0, showing why the
routing claim needs the reverse-specific fetch. The entry's status stays partial: the formula covers only the STEP-only PART
screen, with incomplete complement coverage; the final cell TRAIN O c has none (unsolved).

### 5.2 The circuit trained from scratch

The networks trained from scratch use a different code from HAND (Figure 2). The centred kernel alignment (CKA) of layer 4 with HAND
is .01-.05, and 55-64 of 64 heads best-match HAND's uniform head. Operands come from the trace convention, not from HAND's tag lookup.
In the solved SUM5 sub-cell PA is also low (R² .40 at L2). The fetched operand XA1 is causally used under distributed alignment
search (DAS), which tests an interchange in a learned rotated subspace (.92-1.0;
[glossary](../GLOSSARY.md#analysis-measures)); MLP3 has the largest patching drop.

In Figure 2, trained attention fetches operand bits by their positions in the fixed trace convention, rather than a recovered header-tag route. The operand bits and a causal carry path feed the trained adder MLP3. No TRAIN closed form was recovered; the entry's partial status belongs only to the separate PART screen.

```mermaid
flowchart TD
    accTitle: Positional operand fetching in the failing TRAIN family
    accDescr: The fixed trace convention supplies operand bits by position, while header-tag fetching was not recovered. A causal carry path enters the trained adder MLP. Marked nodes are wholly trained components. No TRAIN closed form was recovered; the page's partial status applies only to the separate PART screen.
    T["trace so far,<br/>fixed register<br/>convention"] --> F["attention: operand<br/>fetch by position<br/>in the trace"]
    H["prompt header tags"] -.->|"tag fetch not recovered"| F
    F --> X["operand bits, XA1"]
    X --> M3["MLP3 adder"]
    CY["carry CY, causal"] --> M3
    M3 --> Y["register line<br/>digits"]
    classDef trained fill:gold,stroke:darkgoldenrod,color:black
    class F,M3 trained
    %% Invisible links constrain vertical layout only, not computation.
    T ~~~ H
    X ~~~ CY
```

*Figure 2. The diagnosed from-scratch failure type in 10 of 10 final seeds. Gold nodes are wholly trained components, not a PART
block. Fixed-convention operand fetching and a causal carry path support the diagnosis; the HAND-free pass matched no HAND and
usually retained no small faithful circuit. The entry's closed-form status is partial for the separate PART screen only; none was
recovered for this TRAIN circuit.*

### 5.3 Relation to the hand-built network and to published work

PART computes the same algorithm as HAND: the stage function is equal under substitution. TRAIN does not: it binds operands by order
in the trace rather than by tag. Dereferencing variables in symbolic programs was studied by
[Wu et al. (2025)](#references). [Cui et al. (2026)](#references) studies spatial variable binding in vision-language models; its
domain differs, and it is not direct evidence for this arithmetic network's order-based fetching. Positional shortcuts under low layout diversity were described by
[Nevermann and Gros (2026)](#references), and mixed-layout exposure was studied by [Shintani (2026)](#references). Position ids derived from task structure were proposed for arithmetic transformers by
[Cho et al. (2024)](#references), and scratchpads and output order in arithmetic were studied by [Lee et al. (2023)](#references).

## 6. Limitations and open questions

Scoring is teacher-forced, not a rollout of the network's own outputs. Depth and length are confounded at depth+1. Lengths beyond 16
digits lie outside the frozen router's id range, so S a PART has no longer-input test; its position shift (pad 256) is exact. The
complement test used final snapshots only, with SHL/REV on one seed. The later discovery, dictionary and bucket backfill completed
only 13/20 final TRAIN reports and did not run on the S networks; the missing reports cannot be treated as negative results.
The O c final has no pre-registered prediction. Lost runs (O a run 1, S b TRAIN run 2, S c PART run 2 final) are not counted.

## 7. Prediction record

Before run 1 of O c the prediction was: depth+1 full ≥ .9 in ≥ 8/10 seeds, with ADD-stage neurons shared across tasks as in HAND. For
format S: PART ≥ .99 in 3/3 by 10k steps, and c TRAIN depth+1 ≥ .9 in ≥ 1/3. All were falsified except S a PART (3/3, solved by 16k).
The O c final run had no pre-registered prediction and is reported without one.

## Appendix A. Weights excerpt and reading

In the hand-built stage the tag-match scores are 30(2v·id − v²) and the stage bias offsets are integers. HAND's stage has 134 live
neurons with W_in values in {-1000, 1, 10, 1000, 10000}. Each PART seed keeps 134 live neurons. Its top 3 neurons, ranked by |W_out|
times |W_in on ports| on the final snapshot and printed by the archive's excerpt script `compose`, read mostly GA_RD and CI (|w| up to
.85) and write the digit outputs YD[0]/YD[1] (|w| .5-.76) with real-valued, seed-specific weights. Reading: the substitution and
complement results show that the trained stage computes the same function as HAND's stage on its ports; no neuron-level identity is
claimed.

## Appendix B. Detailed evidence

*Table B1. Diagnostics per cell.*

| Cell (seeds) | exact | used heads / MLPs | R² CY / PA / PS / XA1 | inter flip (nf>0) | r90 L4 vs HAND 3 | JS vs HAND | patch |
|---|---|---|---|---|---|---|---|
| O c STEP_ADD final (s0-9) | .41-.535 (s8 .785) | 0-5 / 2-4 | .85-.89@4 / .29-.55@2 / .29-.55@2 / .65-.70 | CY .96-1.00@4 | 9-11 | .11-.13 | MLP0 / MLP3 |
| O c SUM5 final (s0-9) | .99-.9975 | 11-21 / 4 | .92@4 / .40@2 / .60@2 / .87@4 | XA1 DAS .92-1.0 | 26-29 | .19-.21 | MLP3 |
| S a PART STEP_ADD final (s0-2) | 1.0 | 7 (mean) / MLP2, MLP3 | carry .40-.44; operands .98-.99 | carry .52-.91 | 2 | 0 (frozen) | MLP3 |

Key: s = seed; CY = carry; R² is the best-layer ridge probe, with @k the layer; inter flip = interchange flip, counted only where the
number of tested flips nf > 0; DAS = distributed alignment search; r90 L4 = rank of the layer-4 residual stream at 90% of the variance,
against 3 for HAND; JS = Jensen-Shannon divergence of attention against HAND; patch = component with the largest patching drop. Terms
are defined in the [glossary](../GLOSSARY.md#analysis-measures). Best against final snapshot: O c best .877-.899, final .871-.897;
S a PART 1.00 at both snapshots.

### B.1 Later HAND-free discovery and its limits

The method-v2 backfill checked circuits without imposing HAND's edge set. It used 400 rows per sub-cell, shuffle corruption, and an
acceptance share of .90. Complete reports covered all 10 STEP_ADD seeds and SUM5 seeds 4, 5 and 8 (13/20). SUM5 seed 6 had discovery
only; seeds 0, 1, 2, 3, 7 and 9 lacked the later kinds. Earlier layer-group diagnostics still apply because this backfill did not
rerun them. Discovery, complement, learned dictionary features and exact/wrong-row buckets are distinct checks
([glossary](../GLOSSARY.md#analysis-measures)).

*Table B2. STEP_ADD discovery retained the acceptance share in only two of ten seeds, using nearly every head. "Faith" is held-out
accuracy with the discovered complement resampled, divided by native accuracy on that held-out bank. The independent complement
check has its own bank. Every complete check passed its full/empty controls.*

| Seed | Discovery criterion | Edges / 1935 | Heads / 64 | Held-out faith | Complement retained share |
|---|---|---|---|---|---|
| 4 | exact | 439 | 61 | .964 | .908 |
| 7 | exact | 686 | 63 | .942 | .947 |

Log-probability discovery in the other reported STEP seeds kept only .56-.80 of the network's held-out accuracy; seed 8's exact
criterion retained .811 held-out and .847 under the separate complement check. Thus even the successful circuit subsets in Table
B2 are large subsets of a failing network, not a recovered tag-address algorithm. Alignment to HAND was false in all 13 complete
reports. Resampling the retained circuits or equally sized random sets damaged performance; those controls establish specificity
of the reported sets, not a compact closed form.

*Table B3. SUM5 discovery. The three complete reports fell below the .90 held-out faith criterion even where their separate
complement check exceeded it. Seed 6's partial report passed held-out faith but has no complement result.*

| Seed | Edges | Heads | Held-out faith | Complement retained share |
|---|---|---|---|---|
| 4 | 288 | 62 | .855 | .873 |
| 5 | 279 | 53 | .894 | .907 |
| 8 | 235 | 49 | .876 | .903 |
| 6 | 266 | 54 | .905 | missing |

In STEP, the carry interchange flip remained .94-1.0 even on wrong rows, supporting a working carry path upstream of a wrong
answer. Address and operand swaps had nf 0 in the exact/wrong buckets: no answer-fixing queries were available, so those variables
are untested there, not unused. Dictionary address features reached correlation ≤ .27; this is weak decoding, not a proof that no
nonlinear address code exists. These tests sharpen the positional-fetch diagnosis without verifying a native algebraic formula.

## What to read next

Read [binding and coreference](binding-coreference.md) for explicit referent selection, or [multiplication and output order](multiplication-base-order.md) for another comparison of arithmetic with the way its intermediate results are written.

## References

- Cho, H., Cha, J., Awasthi, P., Bhojanapalli, S., Gupta, A., & Yun, C. (2024). Position coupling: Improving length generalization of
  arithmetic transformers using task structure. *arXiv preprint*. https://arxiv.org/abs/2405.20671 (position ids from task structure)
- Cui, K., Prakash, N., Messica, S., et al. (2026). The Dual Mechanisms of Spatial Variable Binding in Vision-Language Models.
  *arXiv preprint*. https://doi.org/10.48550/arxiv.2603.22278 (a spatial-binding comparison, not direct arithmetic evidence)
- Lee, N., Sreenivasan, K., Lee, J. D., Lee, K., & Papailiopoulos, D. (2023). Teaching arithmetic to small transformers.
  *arXiv preprint*. https://arxiv.org/abs/2307.03381 (scratchpads and output order in arithmetic)
- Nevermann, D. H., & Gros, C. (2026). Distance generalization in transformers: Why bother with positional encoding?. *arXiv
  preprint*. https://arxiv.org/abs/2609.11913 (low layout diversity, positional shortcuts)
- Shintani, S. A. (2026). Arithmetic OOD failure unfolds in stages in minimal GPTs. *arXiv preprint*.
  https://arxiv.org/abs/2603.26828 (mixed-layout exposure)
- Wu, Y., Geiger, A., & Millière, R. (2025). How do transformers learn variable binding in symbolic programs?. *arXiv preprint*.
  https://doi.org/10.48550/arxiv.2505.20896 (dereferencing in symbolic programs)
