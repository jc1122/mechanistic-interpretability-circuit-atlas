# Greatest common divisor by repeated subtraction

The largest number that divides both 6 and 4 is 2; repeated subtraction finds this greatest common divisor by keeping the pair's absolute difference and smaller value. Four-layer transformers either gave the answer directly or wrote each such step before answering. With the arithmetic and input-reading stages fixed by hand, 7/10 final runs produced exact whole traces on 256 held-out input tuples with correct earlier output tokens supplied; a three-rule selector that chooses which supplied value to write next was substituted into all 10 models and was exact on all 4096 six-bit pairs, repairing errors in every seed rather than reproducing each model's answers.

The task is partly solved and the formula status is [partial](../GLOSSARY.md#verifying-a-circuit): replacement and tests changing activity outside the circuit passed in the 7 exact fixed-scaffold runs, but 3 trained output selectors still made held-out errors, and no complete formula was accepted for the from-scratch models, of which 5/10 were exact. Neither training setting reliably transferred to eight-bit inputs.

| Fact | Value |
|---|---|
| Task | Compute gcd(a, b), directly or as a trace of subtractive-Euclid rows; inverse and fraction cells are historical side tasks |
| Input → output example | Answer only: prompt `BOS GCD A:6 B:4 →`, output `R:2 EOS` |
| Model | GPT-NeoX-shaped, 4 layers × 4 heads, d_model 128, ReLU MLP; PART trains only the layer-3 MLP |
| Rungs and seeds | GCD PART trace: 3 screening + 10 final seeds; TRAIN trace final: 10 seeds, including 3 from the budget retry; answer-only and side-cell screens: 3 seeds each |
| Closed form status | Partial (PART router substitution 1.0 in 10/10 on 4096 six-bit pairs, repairing native errors rather than matching every decision; complement .974–1.0 in 7/7 held-out-exact seeds on 128 tuples, not exhaustively replayed; no complete TRAIN form) |
| Status | Partly solved (PART trace 7/10 exact, TRAIN trace 5/10 exact on 256 held-out tuples; both 10/10 ≥ .99); last run 2026-11-10 |
| Weights | [Hugging Face: gcd-euclid](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/gcd-euclid), published revision `6ca84a561e25` |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

HAND sets every weight by formula; PART trains one stage on frozen HAND parts; TRAIN learns all weights ([rungs](../GLOSSARY.md#the-three-rungs)). A scratchpad is a written sequence of intermediate results ([task formats](../GLOSSARY.md#task-formats)). A router chooses the next token's role and value from supplied arithmetic channels. An MLP is a feed-forward block ([glossary](../GLOSSARY.md#standard-machine-learning-terms)). Complement resampling replaces everything outside a proposed circuit with values from other inputs ([verification](../GLOSSARY.md#verifying-a-circuit)).

## The circuit at a glance

<img src="../figures/circuits/gcd-euclid.svg" width="640" alt="Two frozen layer-2 heads read the previous row A:6 B:4 in parallel. A frozen layer-2 block supplies difference 2 and minimum 4. For the query asking for the next B token, the learned layer-3 router targets B:4 from that minimum; the role is supplied, and one token rather than a whole row is illustrated. Substitution can repair native decisions; acceptance is partial.">

*Figure S1. Final trace networks with only the router learned (seeds 0–9) use frozen hand-built arithmetic; status is partial because sampled paired checks cover seven held-out-exact seeds, substitution repairs native decisions in every seed and native eight-bit transfer fails.*

Figure S1 shows parallel operand reads and frozen arithmetic feeding the learned token router; for the illustrated next-B query, minimum 4 supplies the single token B:4.

## 1. Task and data

The main task is gcd(a, b), with gcd(0, 0) = 0. Numbers are typed scalar tokens (role × value 0–511, vocabulary 7174), not digit strings.
In the trace format the network writes subtractive-Euclid rows until b = 0, then the answer. For example, the answer-only prompt
`BOS GCD A:6 B:4 →` has output `R:2 EOS`.

*Table 1. Output formats.*

| Format | Sequence |
|---|---|
| `A` (answer only) | `BOS GCD A:a B:b → R:g EOS` |
| `T` (trace, a [scratchpad](../GLOSSARY.md#task-formats)) | after the prompt, rows `A:\|a−b\| B:min(a,b)` until b = 0, then `R:g EOS` |
| `S` (local scratchpad, screening only) | local register-machine rows; GCD has 4 fields per local step |

The split is a per-seed hash of the whole input tuple, 80/20, with 256 held-out tuples per seed. Exact match means the whole
teacher-forced output, including EOS, is right; teacher-forced means each token is scored given the correct previous tokens
([glossary](../GLOSSARY.md#task-formats)). Answer-only GCD used 5-bit operands, then 6 bits in the data retry; PART trace used 6 bits,
and the final TRAIN trace used 7 bits. Earlier magnitude checks used 64 tuples at 7 and 8 bits per seed (A PART: 6 and 8 bits). The
later router substitution uses a separate 256-tuple 8-bit set, not the earlier 64-tuple set.

The historical side tasks are the inverse of v mod a prime p (non-units give 0) and the reduced sum a/b + c/d. An inverse multiplies v to 1 modulo p; a non-unit has no such inverse. Reducing a fraction divides numerator and denominator by their greatest common divisor. The inverse trace adds
coefficient registers U and V; the fraction trace first writes products N, M and O. Inverse used 5 bits for A and 6 for T, with data
retries at A 6 and T 7; fraction used 3 bits initially. Inverse T also withholds p ∈ {37, 43, 53, 61}. These side cells are not part of
the current subtractive-Euclid claim.

## 2. Model

The shared conventions are described in [METHODS.md](../METHODS.md). The three [rungs](../GLOSSARY.md#the-three-rungs) are HAND
(every weight set by formula), PART (frozen hand-built parts with one trained stage) and TRAIN (all weights trained from scratch).

### 2.1 Architecture

All networks are GPT-NeoX-shaped with 4 layers × 4 heads (d_head 32), d_model 128, ReLU MLP, no LayerNorm, no positional encoding,
untied embedding and unembedding, and vocabulary 7174. The MLP width is 1026 for gcd A, 1056 for gcd T, 1026 for inverse A, 2096 for
inverse T and 4098 for fraction A. The trace networks locate rows with a frozen hand-built frame, a clock built from causal
attention's 1/position signal, with 520 / 1040 positions for gcd / inverse.

### 2.2 Frozen parts and trained part

PART freezes HAND and re-initialises the layer-3 MLP: `dense_h_to_4h` [W, 128] + bias and `dense_4h_to_h` [128, W] + bias, where W is
the MLP width. For GCD T these are W_in [1056, 128] and W_out [128, 1056]. The input is the whole residual: frozen OLD operand values,
ABS (absolute difference), MIN (minimum), ZERO (the stop flag), role flags and position/clock blocks. The four archived PART tensors
are inserted into `hand_<op>_<fmt>`. TRAIN learns all weights from default initialisation, keeping width 1056 for GCD T, also at 7 bits,
and 1026 for GCD A.

### 2.3 Training

The optimiser was AdamW with learning rate 1e-3, weight decay .1, betas (.9, .98), warmup 1000, cosine decay, clip 1 and a batch of 16
fresh tuples per step, with evaluation every 500 steps. The final cells used only weight decay .1, not a sweep. PART ran in fp32 and
TRAIN in TF32. Seeds of a cell ran together on a consumer-GPU class; analysis ran on CPU. No hardware identity is needed to reproduce
the scientific configuration.

The screening snapshot was selected by highest exact match on a 128-tuple training monitor (then lowest loss), not peak test accuracy.
Its budget was 30k, stopping at test 1.0 or when exact match gained no more than .005 over 5k steps after 22.5k. The final PART trace
used 10 seeds, 6-bit inputs and a 30k budget. The final TRAIN trace used 10 seeds, 7-bit inputs and a 60k budget: three from run 3,
the 2x-budget retry, and seven final seeds. The plateau window remained 5000 after 75% of the budget. The TRAIN final pack stopped at
54.5k; its run-3 seeds stopped at 52.5k. The older report's pending final and pending PART diagnostics are now superseded by these
completed results. Local-scratchpad TRAIN screens at both widths remained 0/3.

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/gcd-euclid](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/gcd-euclid).
The manifest's `acc_test_fp32` and `acc_fresh256` carry the re-scored headline and fresh-tuple accuracies.

### 2.5 Archive

The current publication at revision `6ca84a561e25` adds `hand_gcd_T.safetensors`, `part_gcd_T_final10_s{0-9}` (four tensors each) and
`train_gcd_T_{r3,final}_s*` (full networks), with `config.json` and a manifest recording seed, rung, accuracy and sha256. No new aggregate
archive count is supplied by the final publication. Safetensors stores numeric weights without executable objects; sha256 fingerprints file contents ([glossary](../GLOSSARY.md#standard-machine-learning-terms)).

The earlier archive inventory (2026-11-06) was 39 networks: 5 HAND, 25 PART and 9 TRAIN, with each run's test and monitor tuples in `test_<run>`.
Its files were `hand_*`, `part_<op>_<fmt>[_final10]_s<seed>` and `train_gcd_<fmt>_<r1|r2>_s<seed>` (safetensors), with operand bits, best
step, recorded/re-scored test accuracy and fresh-tuple accuracy in the manifest. After re-download all 39 sha256 matched; reloaded
networks reproduced recorded accuracy in fp32, and each had a 256-fresh-tuple check. This is the historical inventory, not a count of
the enlarged final archive. Run-3 networks were not retrieved at the time of that inventory; they are included in the final account.

## 3. Hand-built solution

The answer-only reference is a pair-index lookup, a lookup control rather than an algorithm. L1H0 averages the operand tokens and
MLP2 holds 702 pair-index hinges, writing gcd(a, b) as a scalar. Together with the last router, its size is 1 head + 705 neurons at
5 bits.

The trace reference computes one subtractive-Euclid step per row (Table F1). L0H0 and MLP0/1 build three position clocks with periods
16, 19 and 23 from the 1/position signal. L2H0 and L2H1 fetch the previous row's a and b into OLD channels by clock offset. MLP2 computes
the absolute difference, minimum and zero flag using five ReLU neurons (Appendix A); MLP3's seven neurons route the result to the token
being written. The earlier description called these fetch heads layer 1; the current component names are L2H0/H1. Inverse T adds the
coefficient updates (u − v) mod p; fraction T adds product and division macros.

Formula sizes (heads + neurons with non-zero weights) are gcd A 1 + 705, gcd T 3 + 1572, inverse A 1 + 496, inverse T 6 + 3154 and
fraction A 1 + 2557. Exactness was checked on all inputs of small domains (gcd A: 1024 pairs) plus stratified larger sets through
8 bits, not all larger inputs. The earlier harness reported minimum logit margin 10, greedy and teacher-forced, and archive checks
reported 1.000 on each run's 256 test tuples in all five published cells. The later compact reference gives the weaker bound margin
≥ 9; both reports are retained. There is one design, HAND-a; no HAND-b or HAND-c was designed.

Table F1 records the fixed arithmetic and trained routing interfaces. Frozen clocks address the preceding operand pair. MLP2 computes the absolute difference, minimum and stop flag; MLP3 then chooses which token role to write. Only that final router MLP is trained in PART, so successful routing does not show that the arithmetic was learned.

<details>
<summary>Technical fixed Euclid step and trained token router (Table F1)</summary>

*Table F1. The hand-built GCD trace network. Clocks address the previous row, MLP2 computes the Euclid step, and MLP3 routes it to the output. MLP3 is the only stage trained in PART.*

| Component | Input, operation and output |
|---|---|
| Fixed layer 0, head 0 (L0H0) and MLP0/1 | Read prompt and rows so far; form position clocks with periods 16, 19 and 23. |
| Fixed layer-2 heads 0 and 1 (L2H0, L2H1) | Use the clocks to fetch the preceding operands OLD_a and OLD_b. |
| Fixed MLP2 | Five ReLU neurons compute absolute difference ABS, minimum MIN and stop flag ZERO. |
| Trained MLP3 | Route by current token role to write the next A, B or R value, or EOS (end of sequence). |

</details>

## 4. What training finds

The final analysis dated 2026-11-10 finds both trace cells at ≥ .99 in all ten seeds, but exactness is lower (Table 2). Each non-exact final seed misses one
held-out tuple. The PART errors are an A/B/R trace in seed 3, a B token in seed 4 and a trace in seed 9. TRAIN errors are row-1 copies
in seeds 0–2, an R token in seed 4 and a B token in seed 5.

*Table 2. Final trace results on 256 held-out tuples per seed. The step columns are first evaluations reaching the stated criterion.*

| Cell | Exact seeds (≥ .99) | Best / final test | ≥ .9 / ≥ .99 step (k) | Magnitude checks |
|---|---|---|---|---|
| PART T, 6 bit | 7/10 (10/10) | 1.0 ×8, .996 ×2 / 1.0 ×7, .996 ×3 | 13–23.5 / 21–26 | 7 bit .23–.47; 8 bit .05–.17 |
| TRAIN T final + run 3, 7 bit | 5/10 (10/10) | 1.0 ×8, .996 ×2 / 1.0 ×5, .996 ×5 | 19–38.5 / 31.5–48.5 | 8 bit .20–.33 |
| S TRAIN width 1 / width 2 (screens) | 0/3 / 0/3 | full trace .03–.52 | never | not reported |

The earlier article called the PART final per-seed list in Table 3 “best snapshot”. The final analysis distinguishes eight exact best
snapshots from seven exact final snapshots as in Table 2. The earlier list is retained as recorded, not used to overwrite that
distinction. Likewise, the earlier run-3 log claim “3/3 solved” used ≥ .99: all three final snapshots were .996, not exact 1.0.

*Table 3. Historical screens and the earlier PART-final listing. “Seeds ≥ .99” includes exactly solved seeds in parentheses; 90% test
is the first test exact match ≥ .9. Fresh and 8-bit columns use 256 and 64 tuples respectively.*

| Cell | Seeds ≥ .99 (exact) | Test exact, earlier recorded snapshot | 90% test (k steps) | Fresh 256 tuples | 8-bit (64 tuples) |
|---|---|---|---|---|---|
| gcd A PART | 3/3 (3) | 1, 1, 1 | 2.5, 1.5, 2.5 | 1, 1, 1 | .31–.56 |
| gcd T PART screen | 3/3 (3) | 1, 1, 1 | 22.5, 21.5, 19.5 | 1, .992, .977 | .05–.11 |
| gcd T PART final, earlier list | 10/10 (7) | 1, 1, 1, .996, .996, 1, 1, 1, 1, .996 | 19.5, 18, 15.5, 17, 21, 13, 22, 22.5, 23.5, 23 | .996–1 | .05–.17 |
| gcd T TRAIN run 1 (6 bits) | 1/3 (0) | .992, .977, .988 | 13.5–18.5 | .992, .953, .984 | .03–.11 |
| gcd T TRAIN run 2 (7 bits) | 2/3 (1) | .988, .992, 1 | 19–22.5 | .988, 1, .996 | .19–.23 |
| gcd A TRAIN run 2 (6 bits) | 1/3 (0) | .938, .996, .949 | 3–6 | .930, .980, .965 | .13–.22 |
| inverse A / fraction A PART | 3/3, 3/3 | 1, 1, 1 / 1, 1, .992 | 2–2.5 / 18–20.5 | ≥ .992 | – |
| inverse T PART | 0/3 | .836, .832, .879 | never | .82–.87 | – |

Some earlier runs were not archived: gcd A TRAIN run 1 (5 bits) reached 0/3 at ≥ .99 (.898–.926); the first report of gcd T TRAIN run 3
(7 bits, 60k steps) gave 3/3 at .996 from its log before retrieval. Fraction T and inverse T TRAIN screens reached ≤ 2/3 (Section 6).
The compact published A PART / A TRAIN run-2 summary gives test 1 / .938–.996, ≥ .9 at 1.5–6k and ≥ .99 at 9–12.5k; the complete
per-cell values remain in Table 3.

### 4.1 Circuit types

PART has one type: frozen HAND-a plus a learned router. All 10 final seeds use HAND's 7 edges plus L2H1-to-MLP3, an 8-edge circuit;
L2H0, L2H1 and MLP2 provide the frozen computation. Zero ablation additionally needs L0H0 and MLP0/1, which build the frame. Including
screens, the earlier family count is PART A 3/3 and PART T 13/13 (3 screen + 10 final). This is an interface-level mechanism, not a
neuron-by-neuron reconstruction.

TRAIN has a distributed type in all 10 final seeds, with 4–9 heads and all 4 MLPs. The published discovered-circuit summary is 54–84 of
207 edges, HAND Jaccard ≤ .10, with seed-specific head sets and loss barrier 8.8 between tested seeds. Edge Jaccard is shared edges divided by their combined edge set. Representation-alignment fit R² ≤ .53 separates the tested codes without establishing different algorithms. A loss barrier measures whether
interpolating weights raises loss ([glossary](../GLOSSARY.md#analysis-measures)); it does not establish a different algorithm. The earlier
three-seed screens had three ≥ .99 seed-runs needing all four MLPs and 5–13 of 16 heads: one run-1 seed depended on layer-1 heads, and
two run-2 seeds on layer-0 and layer-1 heads. TRAIN A's one ≥ .99 seed used parity and divisible-by-3 features. These are ablation-defined
families, not basin identities.

### 4.2 Learning curve

There is no recorded memorise-then-generalise phase, the pattern called [grokking](../GLOSSARY.md#counting-results)
([Power et al., 2022](#references)). In the two successful run-2 TRAIN seeds, test ≥ .9 at 19k and 19.5k preceded training-monitor
≥ .99 at 23.5k and 24.5k. Final cells also reached test 90% before train 99%, with no mem/stall alerts. This ordering is evidence about
these monitors, not an absence of every possible memorisation process.

### 4.3 Diagnostics

Euclid variables are linearly decodable in TRAIN, but the tested interchange does not move difference, minimum or next value
(flip ≤ .04 with 571–1320 answer-fixing pairs under the tested alignment). Interchange replaces a candidate variable between examples to test causal use
([glossary](../GLOSSARY.md#analysis-measures)). Low flips do not exclude a nonlinear code; old a/b have larger flips in some seeds.
Earlier run 1 showed causal use of the probed variables, whereas run 2 had low linear interchange (Appendix B).

*Table 4. Final diagnostics on 128 teacher-forced tuples per network, all exact in this diagnostic bank. Concentration counts are the fewest MLP0/1/2/3 neurons covering 90% of squared input/output weight-norm products, not activation mass or ablated neuron counts. PR is residual participation ratio at L4 (HAND 2.1); JS is maximum used-head attention divergence to its best HAND match.*

| Net | Used heads; MLPs | Weight-concentration counts | Best probe R² (layer): a, b, d, m, next | PR | JS | Interchange | Best / final |
|---|---|---|---|---|---|---|---|
| PART seeds 0–9 | L2H0, L2H1; M2, M3 | frozen + 176–196 | 1, 1, 1, 1 (L4); .94–.96 (L4) | 1.76–2.12 | 0 | m 1.0 (seed 8, nf 507) | 1.0 or .996 / 7 × 1.0 |
| TRAIN seed 3 | 9; M0–3 | 273/265/725/703 | .86 (L2), .84 (L2), .92, .87, .90 (L3) | 9.4 | .48 | d, m, next ≤ .04; a .45 | 1.0 / 1.0 |
| TRAIN seed 6 | 9; M0–3 | 176/367/579/749 | .86 (L3), .80 (L2), .93 (L3), .88 (L2), .92 (L3) | 6.7 | .49 | a, b .56 (nf 36–42) | 1.0 / 1.0 |
| TRAIN seed 7 | 7; M0–3 | 364/322/519/758 | .88, .81, .90, .86, .90 (L3) | 11.3 | .48 | all ≤ .13 | 1.0 / 1.0 |
| TRAIN seed 8 | 8; M0–3 | 171/311/613/706 | .88 (L2), .86 (L2), .86 (L3), .91 (L3), .92 (L4) | 5.8 | .49 | a .40, b .40 | 1.0 / 1.0 |
| TRAIN seed 9 | 5; M0–3 | 178/440/529/729 | .86, .80, .91, .82 (L3), .91 (L4) | 2.3 | .44 | a .55, b .50 | 1.0 / 1.0 |

Key: d = difference, m = minimum; nf = answer-fixing query pairs under the reference rule, with nf 0 meaning untested. PR measures effective representation dimensionality ([analysis measures](../GLOSSARY.md#analysis-measures)). The source's “activation mass” label is corrected here: the recorded `insp.conc.n90` uses `n_i = ||W_in[i,:]|| ||W_out[:,i]||` and selects neurons covering 90% of `sum_i n_i^2`, a weight-concentration screen. Non-exact TRAIN seeds 0–2, 4 and 5
have the same type, 4–9 heads/all MLPs and flips ≤ .43 in the compact report. Across final TRAIN, Euclid variables have R² .80–.98;
that is decodability, not proof of use.

A greedy prune of solved run-2 seed 2 keeps 15 heads + 2940 neurons (HAND 3 + 1572) at 100% before and 99.2% after pruning on its
native set, but falls to 93.1% on fresh 7-bit tuples. It is neither a minimum nor a faithful circuit outside its selection set.

## 5. Mechanism and verification

### 5.1 The PART router

The three rules read only frozen OLD, ABS, MIN, ZERO and role channels. After a B token, emit `A:ABS` if ZERO = 0, else `R:OLD_a`;
after A, emit `B:MIN`; after R, emit EOS. In all 10 final seeds, replacing MLP3 gives 100% on all 4096 6-bit inputs (native
.9988–.9998) and on 256 sampled 8-bit inputs with maximum operand ≥ 128 (native 0%). The six-bit grid includes training inputs. The eight-bit intervention set differs from the trainer's eight-bit set (.05–.17); the 256-tuple count here belongs to the substitution test.

The exhaustive six-bit router substitution repairs 3, 4, 2, 2, 5, 2, 1, 2, 2 and 5 native input errors in final seeds 0–9, respectively;
every seed errs on (63, 63). The router is therefore task-exact but does not exactly reproduce the native router's errors. The
published complement remains its original sampled acceptance check.

A further check enumerates the complete eight-bit grid for seed 0: all 65,536 ordered tuples with operands 0..255, without filtering
out any trace lengths. Native accuracy is 5,453/65,536 and router substitution is 65,536/65,536, repairing 60,083 inputs and changing
1,060,347 scored tokens. This is task-exact arithmetic, not a faithful account of native errors. The earlier 0% native result refers
to the 256-row large-operand/length-filtered sample, not this complete grid. No exhaustive complement certificate is established by
these substitution checks.

Resampling everything outside the claimed 8 edges keeps .974–1.0 exact match against base 1.0 on 128 diagnostic tuples in each of the seven exact seeds; resampling
the circuit or random 8 edges gives ≤ .021. Path drops are .70–.88 for the L2-heads-to-MLP2-to-MLP3 route and .16–.31 for the direct
L2-heads-to-MLP3 route. Both acceptance shares are ≥ .90 on those seven seeds. The closed form is accepted in 7/7 exact seeds, but
partial at the ten-seed cell level because the other three lack this paired complement check. The full/base and empty/source self-checks passed. Path drops are shares of the clean-to-corrupted effect ([analysis measures](../GLOSSARY.md#analysis-measures)).

Earlier substitution-only checks remain distinct: A seed 0 stays at 100% on all 1024 inputs; T seed 5 rises from 99.95% to 100% on all
4096 6-bit inputs and from 10.6% to 100% on 226 stratified 8-bit inputs; the three non-exact PART finals become exact on their test
sets. These were repairs by substitution, not evidence that the native router extrapolates. Its position dependence breaks longer
traces while the frozen computation remains exact.

For the inverse A, fraction A and inverse T side cells, MLP3 was replaced by 3 / 5 / 17 hand-built router neurons with 2 / 4 / 10 fitted
output scalars. For inverse T these are f_A = |a−b|, f_B = min(a, b), f_U = (2s−1)(u−v) + pw, f_V = sv + (1−s)u and f_R = u·1[a = 1],
with s = 1[a ≥ b] and w the wrap flag. Across 18 seed × best/final tests, correct outputs rose from 8822 to 9216 of 9216; inverse T over
all lengths rose from 2614 to 3072 of 3072, with no newly wrong output in an independent re-run. These side-cell substitutions do not
extend the current GCD acceptance claim.

### 5.2 TRAIN: partial descriptions and rejected forms

No complete closed form is accepted for final TRAIN. Among the 5 exact seeds, only seed 9's discovered circuit passes the .90 complement share (.938). The compact .83–.94 range summarises all five, including that pass; Table B3 gives the individual values on 128 diagnostic tuples per seed. A complement pass alone is not a formula certificate. The HAND old-A rule gives 57.8% exact, and per-component MLP laws give 0/512 on 7-bit networks.

The front end of TRAIN T run 2, seed 2, pruned, has a finite token-table description (Table F2). Head h computes
T_h(q) = Σ_k softmax_k(Q_h(t_q)·K_h(t_k)/√32) O_h(t_k) with per-token tables Q, K and O; MLP0 neuron j computes
u_j = ReLU(b_j(t_q) + c_j·Σ_h T_h(q)). Replacing all four layer-0 heads and MLP0 by these tables gives 99.2% / 93.0% / 93.8% (native /
fresh 7-bit / all 1024 5-bit), identical to the pruned network on every input. This finite token-table compilation is a set of
content-weighted token histograms, not a compact Euclid law or a complete verified mechanism.

In TRAIN A run 2, seed 1, replacing 24 layer-0 neurons by α_j + β_j·1[3 | a and 3 | b] gives 99.66% before and 99.68% after on all
4096 inputs (mean ablation 96.0%). The period-2 embedding component E2(v) = (−1)^v mean_u[(−1)^u E(u)] keeps 99.66% when substituted
and leaves 74.0% when removed. These are parts, not a whole formula.

A categorical-table MLP3 for TRAIN T run 2 (≈ 29.6k fitted scalars) was rejected: it gives 505 / 512 correct (512 before) on seed 1
and 500 / 512 on seed 2, below the .99 bar. Jointly substituting all candidate TRAIN T components gives 1.2% native. The later
11 heads and MLP1–3 of that pruned network remain unexplained.

Table F2 records the historical front-end description, not a complete Euclid circuit. Token-table attention heads and MLP0 specify the measured front end of the historical pruned TRAIN network. Later heads and MLPs still lack a complete closed form. All marked components were trained. This finite token-table description is not a complete Euclid mechanism or paired-complement certificate; the entry's partial status belongs to PART.

<details>
<summary>Technical historical front end and unexplained later computation (Table F2)</summary>

*Table F2. The historical pruned TRAIN trace network, run 2 seed 2. All listed computational components were trained. The token-table substitution matches this network on the measured sets, but the complete closed-form status remains partial for the entry's PART claim; no complete TRAIN form or paired complement certificate is supplied by this front-end description.*

| Trained component | Input and recorded description |
|---|---|
| Four layer-0 heads | Read prompt and row tokens using per-token Q, K and O tables in run 2, seed 2. |
| MLP0 | ReLU of token bias plus the weighted head sum. |
| Later 11 heads and MLP1–3 | Produce the next token, but have no complete closed form. The finite front-end description does not recover a complete Euclid algorithm. |

</details>

### 5.3 Relation to the hand-built network and to published work

PART keeps the hand-built algorithm and learns its interface. TRAIN reaches decodable Euclid variables by a different route, with
content-weighted token histograms rather than hand-built position clocks in the analysed front end. Low tested linear interchanges
in the final pack do not rule out a nonlinear Euclid representation; unlike the earlier 6-bit run-1 screen, they do not establish
causal use of difference, minimum or next value.

[Charton (2023)](#references) found that GCD transformers predict the largest element of a learned divisor list (products of
number-base divisors and small primes). The answer-only network here likewise uses divisibility features (2, 3), on much smaller
numbers without a Euclid trace. Scratchpads help step computation ([Nye et al., 2021](#references)), but do not guarantee magnitude
transfer when a trained stage can read position ([Kazemnejad et al., 2023](#references)). The local scratchpad did not learn its bit
bricks. [Rai et al. (2026)](#references) motivate a possible explanation for the missing TRAIN form: a circuit fitted over all rows
can mix mechanisms. That is a hypothesis, not a measured cause here; per-role/per-depth discovery remains open.

## 6. Limitations and open questions

Scoring is teacher-forced, not free rollout. Diagnostics use 128 tuples in one length bucket and are exact on the final diagnostic
bank; this does not imply exactness on full held-out traces. Scalar tokens test magnitude transfer, not unseen digit-string lengths.
Earlier magnitude sets have 64 tuples; the original final-router 8-bit intervention set has 256 sampled tuples and is not exhaustive.
The later seed-0 check covers the complete eight-bit input grid, but only for substitution, not complement resampling. The earlier
226-tuple stress set is stratified. Neither native trained cell reliably extrapolates to 8 bits.

Interchange tested only compare on most PART seeds, with nf 0; other variables are untested except the recorded seed-8 minimum test.
PART's variable-use claim rests on circuit/complement/substitution evidence. The complement paired with the router covers seven exact
seeds, not ten. TRAIN has no complete accepted form; low probes or interchange do not prove a variable absent. Only weight decay .1
was run in the final cells; no alternative HAND design was tested. Earlier TRAIN counts are three-seed screens. Per-seed used-neuron
counts were not measured in those screens; final weight-concentration counts are a different measure, not causal minima. The final log-only dynamics summary is missing because no dynamics log was available. No alternative HAND or knowledge-distillation arm was run.

Inverse T TRAIN (runs 2 and 3: 0/3, best .59–.86) and fraction T TRAIN (run 1 2/3; 4-bit run 2 0/3) are outside this entry's current
claims; one inverse T TRAIN run lost its networks. The local register-machine variant (inverse PART 3/3 exact; all TRAIN 0/3) is not
published here. The public code remains pending. No per-row ablation establishes shared arithmetic on every row.

## 7. Prediction record

Five predictions were recorded before the first smoke run; the final published record also states the shared-arithmetic prediction
(Table 5). Counts at ≥ .99 are distinguished from exact seeds.

*Table 5. Prediction record, updated with the final run.*

| Prediction | Outcome |
|---|---|
| (1) PART T ≥ 8/10 GCD seeds ≥ .99 within 15k; ≥ 5/10 for inverse/fraction | GCD count held (10/10 ≥ .99, 7/10 exact), time missed (≥ .99 at 21–26k; earlier 90% test at 13k–23.5k); inverse/fraction T PART failed (0/3 each, .83–.92) |
| (2) TRAIN T GCD ≥ 5/10, inverse/fraction ≤ 4/10 by 30k | GCD exact count held at 60k (5/10), time missed (test .70–.95 at 30k); earlier screens were 1/3 at 6 bits, 2/3 at 7 bits and 3/3 ≥ .99 at 7 bits/60k |
| (3) TRAIN A ≤ 3/10 GCD/inverse, ≤ 2/10 fraction | Held for GCD: A run 1 0/3, run 2 1/3 at ≥ .99 |
| (4) No format reliably extrapolates to 8-bit operands | Held |
| (5) T degrades less with operand size than A | Refuted as stated: 8-bit A PART .31–.56, T PART .05–.17; A was trained on smaller numbers and its frozen part is a lookup |
| (6) Shared arithmetic across rows | Held in PART structure (one 8-edge circuit); not tested per row |

## Appendix A. Weights excerpt and reading

HAND gcd T, layer-2 MLP, the five used neurons (weight slice of `hand_gcd_T`, inputs OLD_a and OLD_b; all other weights 0):

```
n0 = ReLU( OLD_a − OLD_b)          n2 = ReLU(OLD_a)
n1 = ReLU(−OLD_a + OLD_b)          n3 = ReLU(1.5 − 2·OLD_b),   n4 = ReLU(.5 − 2·OLD_b)
ABS = n0 + n1 = |a − b|,   MIN = n2 − n0 = min(a, b),   ZERO = n3 − n4 = 1[b = 0]   (integer b)
```

Layer 3 (7 neurons) routes ABS, MIN, ZERO and role flags. The nonzero input rows are n0 (+1, −1), n1 (−1, +1), n2 (OLD_a +1),
n3 (bias +1.5, OLD_b −2) and n4 (bias +.5, OLD_b −2).

For PART MLP3 best snapshots, seeds 0–9, the closed-form inputs (ABS, MIN, OLD_a, ZERO and A/B/R flags) carry .276–.285 of absolute
W_in column mass; position/clock blocks carry .254–.270 and the rest .45–.46. Router outputs (Z/G rows of A, B, R and END) carry
.620–.629 of absolute W_out mass. About a quarter of input weight is on position, which the closed form does not need.

In learned A PART seed 0, neuron 439 is described by a least-squares activation fit, not a weight row:
≈ ReLU(.124·ANS·[current = B] + .066·[current = B] + .168·[current = R] + .234). Replacing neurons 439 and 472 by these fits keeps 100%
on all 1024 inputs; mean-ablating both leaves 91.7%. The router spreads one rule over many neurons.

## Appendix B. Detailed historical evidence

*Table B1. Earlier diagnostics on 128 teacher-forced tuples of one length per network, best and final snapshots. The earlier
“final seeds 3–9 pending” caveat is superseded only for the final measurements in Table 4.*

| Setup | Used heads (ablation drop > .05) | Best probe R² (layer) | Attention JS to HAND | Patching |
|---|---|---|---|---|
| gcd T PART | 2 / 3 heads, 2 / 4 MLP blocks (mean / zero ablation) | old a, b, a−b, min = 1 (L3); next value .94–.97 (L4) | 0 | MLP3 patch recovery 1.0 |
| gcd T TRAIN run 1 | 5–10 / 6–12 of 16; all 4 MLPs | a, b .83–.88 (L2–4); a−b .86–.94; min .87–.95; next .87–.97 | .003–.506 | next value, a−b, min interchange flip 1 |
| gcd T TRAIN run 2 | 8–11 / 9–13 of 16; all 4 MLPs | next .92–.96 (L2–4); a−b .92–.96 | – | next value interchange flip .01–.03 |
| gcd A TRAIN run 1 | 0–1 of 16; all 4 MLPs | next value .40–.60 | .23–.24 | interchange flip .84–.90 |

Key: R², JS, interchange flip and patching are defined in the [glossary](../GLOSSARY.md#analysis-measures). Earlier residual
participation ratio was 1.09–3.91 for HAND and 13.3–13.9 for failed inverse T TRAIN. These are not the final GCD TRAIN PR values.

*Table B2. Earlier compact screen summary, retained beside Table 3's per-cell results. Each A cell has 3 seeds; the TRAIN count is at ≥ .99, not exact 1.0.*

| Cell | Seeds ≥ .99 | Test best / final, compact summary | First .90 / .99 test step (k) | Longer inputs |
|---|---|---|---|---|
| A PART / A TRAIN run 2 | 3/3 / 1/3 | 1 / .938–.996 | 1.5–6 / 9–12.5 | 8 bit .31–.56 / .13–.22 |

*Table B3. Discovered-circuit complement tests in the 5 final TRAIN seeds exact on 256 held-out tuples. Each diagnostic test uses 128 tuples with native exact match 1.0; the acceptance share is .90. Passing this test alone does not supply a closed form.*

| Seed | Complement resample exact match | Meets .90 share of native 1.0 |
|---|---|---|
| 3 | .893 | No |
| 6 | .828 | No |
| 7 | .896 | No |
| 8 | .844 | No |
| 9 | .938 | Yes |

The compact .83–.94 range is retained as the older published summary; these per-seed values are copied separately from the final diagnostic record.

## What to read next

Compare [rounded Euler decay](euler-decay-ode.md) for a recurrence where learning the local numerical update remains a failure. [Arithmetic registers](compose-arithmetic-registers.md) separate arithmetic from selecting the right stored operands.

## References

- Charton, F. (2023). Learning the greatest common divisor: Explaining transformer predictions. *International Conference on Learning
  Representations (ICLR 2024)*; arXiv preprint 2023. https://arxiv.org/abs/2308.15594 (GCD in transformers, divisor lists)
- Kazemnejad, A., Padhi, I., Natesan Ramamurthy, K., Das, P., & Reddy, S. (2023). The impact of positional encoding on length
  generalization in transformers. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*.
  https://doi.org/10.52202/075280-1082 (position reading and length transfer)
- Nye, M., Andreassen, A. J., Gur-Ari, G., et al. (2021). Show your work: Scratchpads for intermediate computation with language
  models. *arXiv preprint*. https://arxiv.org/abs/2112.00114 (scratchpads)
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization beyond overfitting on small
  algorithmic datasets. *arXiv preprint*. https://arxiv.org/abs/2201.02177 (grokking)
- Rai, D., Geva, M., & Yao, Z. (2026). Data-driven circuit discovery for interpretability of language models. *arXiv preprint*.
  https://doi.org/10.48550/arxiv.2605.09129 (possible mixing of row-specific mechanisms in dataset-wide circuit discovery)
