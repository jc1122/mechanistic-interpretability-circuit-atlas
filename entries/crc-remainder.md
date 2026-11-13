# CRC remainder

Dividing `10000000` by `10001` leaves the 4-bit remainder `1000` in this version of long division, where adding and subtracting use XOR, addition without carries. Small transformers learned to write each shift-and-update step as a [scratchpad](../GLOSSARY.md#task-formats), reaching whole-output accuracy 1.0 in 10/10 final from-scratch runs at their joint stopping evaluation on 1,536 held-out pairs per run, with correct earlier output tokens supplied.

Their answer-reading formulas and circuits passed the original replacement and outside-circuit tests at ≥ .90 of unmodified-network accuracy; over all inputs of widths 1–12, however, 9/10 circuits failed that share and even the remaining pass changed the unmodified network's answers. The overall formula is [partial](../GLOSSARY.md#verifying-a-circuit): earlier layers remain unexplained, and the coarse answer table receives the reference answer bit itself, so a better answer can be a repair rather than an explanation. The task is partly solved because answer-only training was exact in 2/3 screening runs, while the final scratchpad networks scored only 0–.26 at input width 12; learning the step did not ensure reliable repetition on longer inputs.

| Fact | Value |
|---|---|
| Task | Polynomial remainder N mod D over GF(2), with n = 8/9/10 input bits and a monic degree-4 divisor; answer only or a per-bit scratchpad |
| Input → output example | Answer only: `BOS 10000000 / 10001 =` → `1000` (x^7 mod x^4 + 1 = x^3) |
| Model | GPT-NeoX-shaped, ReLU, no LayerNorm or positional encoding, d_model 128; answer only 4 layers × 16 heads; scratchpad 3 layers × 4 heads; PART trains only the stage MLP |
| Rungs and seeds | HAND fixes weights; PART trains one stage; TRAIN learns all weights ([rungs](../GLOSSARY.md#the-three-rungs)). HAND exact in both formats; PART scratchpad 10 seeds; PART answer only and TRAIN answer only 3 screening seeds each; TRAIN scratchpad 10 final plus 3 screening seeds |
| Closed form status | Partial overall (original sampled TRAIN scratchpad readout and circuit checks pass in 10/10 seeds at share ≥ .90; coarse table 1.0 in 19/20 best/final snapshots; fine table seed 7 best .9883, final .9694; wider-domain circuit share fails in 9/10 final seeds; coarse key includes the reference answer bit, not a recovered upstream computation; [domain and fidelity limits](#51-current-acceptance-and-scope)) |
| Status | Partly solved (TRAIN scratchpad final 10/10 exact, answer only 2/3 screening); last run 2026-11-09 |
| Weights | [Hugging Face: crc-remainder](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/crc-remainder), revision `58be416bb224` |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

## The circuit at a glance

<img src="../figures/circuits/crc-remainder.svg" width="640" alt="One residual stream connects an unverified early step code, layer-2 fetch heads and the native bit writer. Prior remainder 0100 and next input bit 0 target 1000. A separate dashed test table receives a true emitted-bit key from outside the network and can replace the writer and repair errors; the upstream formula is unknown.">

*Figure S1. Final scratchpad networks trained from scratch (seeds 0–9) differ from the compact hand-built stage; status is partial because the coarse table uses reference-answer keys and can repair errors, the upstream law is unverified, and widths 1–12 circuit checks retain below .90 of native accuracy in 9/10 seeds and change native decisions in all ten.*

Figure S1 separates the native code-to-bit route from the test-only table, whose emitted-bit keys are supplied by the reference computation.

## 1. Task and data

Each input bit is a polynomial coefficient: for example, `10001` denotes x^4 + 1. The target is N mod D over GF(2), arithmetic with
bit coefficients where addition and subtraction are XOR and multiplication of two coefficients is AND. N has n input bits
(n 8/9/10 in training and test). D is monic, meaning its leading coefficient is 1, and has degree 4 (16–31, with
D ∈ {19, 25, 28, 30} withheld from training). The prefix is `BOS N_MSB…N_LSB / D_MSB…D_LSB =`, most significant bit
first. This is algebraic division, not a standardised CRC encoding. For example, `BOS 10000000 / 10001 =` has remainder `1000`, since
x^7 mod (x^4 + 1) = x^3.

*Table 1. Output formats.*

| Format | Output |
|---|---|
| `A` (answer only) | the 4 remainder bits |
| `T` (scratchpad) | after each input bit, the quotient bit and the full 4-bit remainder, with `;` between states (5n bits + n − 1 separators) |

A scratchpad writes intermediate states before the answer ([glossary](../GLOSSARY.md#task-formats)). The split is a width-invariant
hash of (N, D), 80/20, so zero-padding cannot leak test pairs into training; fresh pairs are drawn every batch. An output counts as
exact when the whole teacher-forced output is correct: each token is scored given the correct earlier tokens
([glossary](../GLOSSARY.md#task-formats)). Test accuracy is the mean over 512 held-out pairs at each of n 8/9/10 (1,536 per seed).
Length was tested at n 11 and 12, short inputs at n 4 and 6, and unseen divisors at n 10.

## 2. Model

The shared conventions are in [METHODS.md](../METHODS.md). The three rungs are HAND (all weights set by formula), PART (one trained
stage on frozen hand-built parts) and TRAIN (all weights trained from scratch; [glossary](../GLOSSARY.md#the-three-rungs)).

### 2.1 Architecture

All networks are GPT-NeoX-shaped with ReLU, no LayerNorm, no positional encoding, untied embeddings, vocabulary 7, context 128 and
d_model 128. Format A uses 4 layers × 16 heads (d_head 8) and MLP 1024; format T uses 3 layers × 4 heads (d_head 32) and MLP 256.

### 2.2 Frozen parts and trained part

PART trains only the stage MLP (A layer 3, T layer 2), with W_in [d_mlp, 128] masked to the hand-built input coordinates; all routing
stays frozen HAND. TRAIN uses the same shapes with all weights from random initialisation.

### 2.3 Training

The optimiser was AdamW with learning rate 1e-3, weight decay .1 and betas (.9, .98), warmup 1,000 steps, a cosine schedule, clipping
at 1 and batches of 256 fresh pairs. The budget was 30k steps, with a plateau stop after 5,000 steps without improvement once past a
minimum of 22.5k, checkpoints every 2,500 steps and evaluation every 500 steps. PART ran in fp32; TRAIN used TF32 updates with fp32
evaluations and out-of-distribution tests ([precision glossary](../GLOSSARY.md#hand-built-machinery)). Success is test exact 1.
Training ran on GPU accelerators and re-scoring in fp32 on CPU. Test pairs are regenerated from the seed (see `config.json`).
The final TRAIN T run, dated 2026-11-09, used 10 seeds in addition to the 3 screening seeds.

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/crc-remainder](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/crc-remainder).

### 2.5 Archive

The folder holds `hand_{A,T}`, `part_T_{best,final}_s0…9` and `part_A_{best,final}_s0…2` (the 4 stage-MLP tensors, to be inserted into
`hand_<fmt>`), `train_{A,T}_{best,final}_s0…2` (screening full state dicts), `train_T10_{best,final}_s0…9` (final-run full networks),
`config.json` and `manifest.json` (seed, rung, steps, recorded and fp32 re-scored exact, sha256). The archived networks re-score to
the recorded accuracies. The final-run upload is revision `58be416bb224`; its round-trip check covered 58 networks, the download
check matched 60/60 file hashes, and 3/3 re-scored networks reproduced their recorded scores.

## 3. Hand-built solution

The scratchpad network T (7 used heads + 236 neurons) performs one step of long division per input bit (Table F1). Its heads fetch the
previous state's shift bit X, the input bit A, the divisor bit DB and the quotient bit QB. The layer-2 MLP computes
r = ((r ≪ 1) | a) XOR (q·D): it shifts the remainder left, appends input bit a and XORs the divisor in when quotient bit q is 1.
At each bit this is `next = shift XOR (divisor_bit AND quotient)`. The stage has 10 neurons, with W_in values ±1 and −2.

The answer-only network A (18 heads + 344 neurons) instead uses parity staircases, paired steps whose alternating output gives a
sum's parity ([glossary](../GLOSSARY.md#hand-built-machinery)). Its 12 layer-2 heads fetch NB_i, the input bit at power i, and 4 fetch divisor bits.
MLP3 holds 262 parity-staircase neurons, grouped by divisor D and answer bit j, reading
S(D, j) = {i: bit j of x^i mod D is 1}; here S is the set of input-bit positions contributing to that remainder bit. Input weights
are 1 and gates 32. Both formats are exact on 9,568 sequences per format (n ≤ 6 exhaustive, n 7–12 stratified over all D), with margin 30.

Table F1 records the fixed scratchpad and answer-only alternatives. Frozen scratchpad heads fetch the shift, input, divisor and quotient bits before the stage MLP shifts and XORs the conditional divisor. The separate answer-only route uses frozen heads and parity staircases. PART trains only the marked stage MLP in each format; its heads stay frozen.

<details>
<summary>Technical scratchpad and answer-only reference routes (Table F1)</summary>

*Table F1. The hand-built scratchpad (T) and answer-only (A) networks. Only the stage MLPs (T layer 2, A layer 3) are replaced by a trained stage in PART; the heads stay frozen.*

| Alternative route | Fixed reads, trained interface and output |
|---|---|
| Shared input | BOS N / D = and scratchpad so far. BOS is the beginning-of-sequence marker. |
| T: written division steps | Fixed heads fetch shift bit X, input bit A, divisor bit DB and quotient bit QB. The layer-2 MLP shifts r, appends a and XORs q times D. It writes a quotient bit and 4 remainder bits per step. |
| A: answer only | 18 frozen heads feed the layer-3 MLP's GF(2) parity staircases per divisor, producing 4 remainder bits. |

</details>

## 4. What training finds

*Table 2. Results per cell. Screening columns list seeds 0 / 1 / 2; final rows cover seeds 0–9. Exact means the whole test output.*

| Cell | Exact seeds | Best → final test | First exact step | Stop |
|---|---|---|---|---|
| PART T, 10 seeds | 10/10 | 1 → 1 (all) | 500 (all) | solved 500 |
| PART A, screening | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 | 5.5k / 5.5k / 7k | solved 7k |
| TRAIN T, screening | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 | 16k / 13k / 15k | solved 18k |
| TRAIN T, final | 10/10 | 1.0 → 1.0 (all) | 14–19.5k | joint solved stop 20.5k |
| TRAIN A, screening | 2/3 | .9954 / 1 / 1 → .9935 / 1 / 1 | 26k (best) / 23k / 24.5k | plateau 28k |

The earlier screening lede reported first exact at 11.5–16k for TRAIN T; Table 2 retains that article's per-seed values separately.
The final run reached first exact at 14–19.5k and all seeds were exact together at 20.5k. Exactness was not stable before that stop:
seed 1 fell from 1.0 to .375 after its first exact evaluation.

PART T is learned earlier than PART A: 10/10 at 500 against 3/3 at 5.5–7k, described in the screening comparison as 10–20× faster.
The answer-only screening networks use about twice the heads (zero ablation 16 against 7 in PART, 25–32 against 12 in TRAIN). There is
no classic delayed grokking, in which test accuracy rises after training accuracy is already high
([glossary](../GLOSSARY.md#counting-results); [Power et al., 2022](#references)): test ≥ .9 never lags train ≥ .99.

### 4.1 Circuit types

PART T is the HAND XOR-AND stage (10/10 by the earlier ablations, 3/3 directly tested by discovery): 9/118 edges, Jaccard 1 against
HAND. PART A is the HAND parity readout (3/3), 30/1935 edges, also Jaccard 1; weak probes (R² .52–.57) do not prevent its neurons from
reading HAND's bit-support sets (Appendix A). Earlier screening prose grouped PART and TRAIN A together as a non-staircase last-layer
readout (3/3 each); the later native-neuron support test distinguishes PART A's HAND parity readout from TRAIN A's dense code.

TRAIN T has one distributed scratchpad-step type in 10/10 final and 3/3 screening seeds. Shift, quotient and next remainder are
causally used at probe layer 3 (interchange .98–1, DAS 1), while layer-2 heads fetch and MLP2 writes the bit. Interchange swaps a
variable between inputs; DAS tests such swaps through a learned rotation ([glossary](../GLOSSARY.md#analysis-measures)). Discovered
circuits have 13–71 of 118 edges, HAND edge recall .67–1 and Jaccard .10–.38. They have more edges overall and overlap HAND, but do not contain every HAND edge in
every seed; the source's “superset” shorthand is too strong. Seeds 1, 5 and 6 are compact (13–28 edges, at most 2 layer-0/1
heads); the wide circuits have 42–71 edges. The path through MLP1, layer-2 heads and MLP2 carries drops .32–.66 and .68–1 in the
published summary; the detailed path measurement reaches 1.35. The direct MLP1-to-MLP2 drop is ≤ .06.

TRAIN A has one dense non-HAND readout type (2/2 solved seeds; failed seed 0 has the same type), using 848–1000/1935 edges, Jaccard
.03. Count and next remainder are used only at probe layer 4 (DAS ≈ 1). Seed 2 additionally routes through one dominant layer-3 head.

### 4.2 Failure and transfer

Failed TRAIN A seed 0 (train 1) scored .9954 best and .9935 final on the trainer test; the later account gives .99544 and .99349.
It misses 3/256 diagnostic rows, all at answer bit 3: D = 22 (1/24) and D = 29 (2/23). Count and next are both used at layer 4
(interchange flip 1 / .996, DAS 1). MLP3 ranks first by activation-patching drop, locating a divisor-specific readout gap rather than
a missing HAND variable. Seed 0 and solved seed 1 followed one trajectory to 20k without memorisation alerts, then separated during
cosine decay: seed 1 reached 1 at 25k, seed 0 peaked at .992 and ended at .988 on that diagnostic bank. A wider-data TRAIN A retry
(n 8–12) subsequently reached exact match in 0/3 seeds; it is no longer merely a proposed retry.

*Table 3. Final-run transfer and screening comparisons. Length scores use 512 rows; unseen divisors are tested at n 10.*

| Rung / format | First test ≥ .99 | n 11 final | n 12 | Unseen D |
|---|---|---|---|---|
| PART A | 3.5–5.5k | .09 | .11–.14 | .07–.09 |
| PART T | .5k | 1 | 1 | 1 |
| TRAIN A, screening | 20.5–23k | .13–.23 | .09–.13 | .12–.15 |
| TRAIN T, final | 14–19.5k | .13–.98 | 0–.26 | 1 in 7/10; other seeds .30–.75 |

PART T seeds 0–2 pass n 1–12. Final TRAIN T reaches ≥ .95 at n 11 in 3/10, but short-input exact is ≤ .26 at n 4 and ≤ .71 at n 6.
The older TRAIN T screen had n 11 best .84 / .86 / .83 and final .18 / .92 / .57, with n 12 ≈ 0. Older PART A tests reported n 12
with seen divisors ≈ 1/3 and unseen-D n 10 .09–.10; Table 3 retains the later comparison's .11–.14 and .07–.09 separately. These are
distinct recorded summaries; their precise evaluation difference was not specified. PART A's timing also needs its criterion kept:
Table 2 reports first exact match 1.0 at 5.5–7k, while Table 3 labels 3.5–5.5k as first test ≥ .99. The source does not establish whether
the criterion alone explains this difference; these ranges must not be used interchangeably.

## 5. Mechanism and verification

### 5.1 Current acceptance and scope

Substitution replaces the last MLP output at every answer query by a table fitted on separate batches. The original sampled tests scored
3 × 512 held-out rows at n 8/9/10; complement checks used 256 n 10 rows and 3 resample draws, not the same evaluation batch.
Acceptance requires both substitution and resample ablation of the complement to retain ≥ .90 of
the native network's exact match ([glossary](../GLOSSARY.md#verifying-a-circuit)). The complement test replaces everything outside
the claimed circuit with activations from other inputs.

*Table 4. Original sampled closed-form acceptance and controls. Best and final are tested separately. The input-exhaustive follow-up below qualifies domain coverage and fidelity without replacing these recorded scores.*

| Cell | Substitute | Substitution | Complement and controls | Scope |
|---|---|---|---|---|
| PART T | MLP2 = W[phase, col, next], next = shift XOR (divisor_bit AND quotient) | 1.0 in 6/6 best/final nets; also with the 35-key HAND-input table | resample 1.0; circuit resampled 0 | accepted on the 3 directly tested seeds |
| PART A | MLP3 = W[j, next], next = XOR of NB_i over S(D, j) | 1.0 in 6/6 | resample 1.0 | accepted on 3/3 seeds |
| TRAIN T final | same MLP2 readout, with earlier layers left trained | coarse table 1.0 in 19/20 best/final snapshots; fine 35-key table seed 7 best .9883 rather than 1.0; final seed 7 .9694 with either key | complement .915–.994; circuit resampled and random same-size ≤ .003 | verified at the readout/discovered-circuit scope, 10/10 at the .90 share; not a complete upstream formula |
| TRAIN A | table in answer bit and next remainder | .84 / .69 for solved seeds 1 / 2; failed seed 0 .998 | claimed-component resample .09 / .15 | rejected for the solved seeds |

Complement self-tests pass in every tested network (`full = base`, `empty = source`). The coarse readout key includes the emitted
bit: it establishes a pure bit code, not how the earlier layers compute next. The HAND-input T key and PART A's native support
neurons add computational evidence; no formula replaces TRAIN T layers 0–1 or its layer-2 heads. The overall status is therefore
partial, even though the stated readout and discovered circuit pass in 10/10 final seeds. The earlier published display gave seed 7
final as .969; the acceptance record gives .9694 and locates its loss at n 10 (.9082, displayed as .908 in the source summary). The
coarse table has 11 keys and the fine table 35, with none unseen in the evaluation; the fine best-snapshot exception cannot be folded
into the coarse 19/20 exact count.

An input-exhaustive follow-up checks every input in a stated finite domain, rather than a sample. For PART T seeds 0–9, both frozen
readout tables make exactly the same decisions as the native network and are task-exact on all 131,040 inputs per seed of the full specification domain. That domain
covers lengths 1–12, every degree-4 monic divisor and every bit string, including leading zeros and withheld divisors. The nine-edge
family circuit preserves every scored native decision under three registered full-domain same-length donor derangements, in which
each input receives activity from a different input of the same length. Necessity, random-size and endpoint controls pass.
Best and final tensors coincide within each seed. This is input-exhaustive decision equivalence for these substitutions and draws,
not enumeration of all donor pairs; seeds 3–9 use the historically extrapolated family mask, not separately recovered per-seed masks.

For PART A, Table 4 gives sampled acceptance scores on seen divisors, not fidelity to native errors over the whole domain.
Over all 28,672 primary inputs of lengths 8–10, including the four withheld divisors, the coarse table repairs 6,415/6,495 errors in
best/final seed 0, 6,461/6,466 in best/final seed 1, and 6,469 in seed 2. It remains task-exact but does not reproduce all native
decisions. The fine table, which also uses the divisor as a key, has no fitted entries for the four withheld divisors. Preserving
native outputs for absent keys is a fallback, not closed-form coverage.

All ten final TRAIN T seeds still meet the original accuracy-share gate on the exhaustive primary n 8–10 domain, including withheld
divisors, but every complementary circuit changes native decisions. On the wider 131,040-input specification domain (n 1–12), nine of ten original
complementary circuits fall below 90% of native accuracy. Seed 5 alone retains the share, not the predictions. The coarse tables
differ from native on 64,581–94,979 inputs per final seed; their keys use values supplied by HAND, the reference algorithm, and often
repair upstream errors rather than reproduce the trained computation. These results qualify scope and fidelity, not the original
sampled acceptance scores. No new complement test was run for best or screening snapshots.

Table F2 records the learned readout roles and the failed answer-only form. Trained scratchpad codes pass through fetch heads to the final bit write. The answer-only candidate instead uses a dense last-layer readout. All marked components were trained. Overall status remains partial: scoped scratchpad tests pass, but earlier layers lack a formula and the coarse key includes the emitted bit. The tested answer-only form is rejected.

<details>
<summary>Technical learned readouts and external-key boundary (Table F2)</summary>

*Table F2. The roles found from scratch. All listed network components are trained. The TRAIN T readout and discovered circuit pass substitution and complement checks at the .90 share in 10/10 seeds. Overall closed-form status is partial: earlier layers have no formula and the coarse table uses the emitted bit, so it is a readout rather than an upstream algorithm. TRAIN A's tested form is rejected.*

| Alternative route or test boundary | Input and recorded role |
|---|---|
| TRAIN T code and fetching | Input, divisor and previous state feed trained MLP1's code, then trained layer-2 fetch heads. The upstream computation has no formula. |
| TRAIN T MLP2 | The trained bit readout uses W (indexed by phase, column and next) to write quotient and remainder bits. |
| Coarse reference-keyed test | Its key includes the emitted bit; this test must not be drawn as native computation of that bit. Scoped readout acceptance does not identify the upstream algorithm. |
| TRAIN A | Input bits, divisor and column feed a trained last-layer MLP's dense non-HAND parity readout for 4 remainder bits. The tested form is rejected. |

</details>

### 5.2 Earlier substitution evidence

The earlier article's last run was dated 2026-11-07. That audit used fresh held-out screens of 768 rows per snapshot (n 8/9/10), best and final. For PART T seeds 0–2 an eight-ReLU
table with 8 fitted coefficients keeps 11,587 of 11,587 rows: write `;` at SEP, X at QF, otherwise
`(if LOW then A, else X) XOR (DB AND QB)`. SEP marks a separator, QF the quotient-bit slot and LOW the lowest remainder-bit slot.
At that audit seeds 3–9 lacked matched zero controls and no complement was reported. The later sampled complement evidence applies to the directly tested seeds; it does not fill those seven earlier control gaps.
The input-exhaustive follow-up reports new registered controls for all ten seeds, with the transferred-family-mask qualification above.

For PART A seeds 1–2, a native ReLU ridge bank over input bits NB, divisor and column used 36,867 numbers against 263,296 in the
stage and reproduced the learned behaviour, errors included (768/768; seed 2 767/767). Projection onto one HAND low-bit count
direction kept only 191–237/768, so that projection was rejected. Zeroing the final MLP gives exact 0; shuffled-token tests drop to
chance and patching restores 1 / 1. Native donor-path tests agree on 6,155/6,155 T and 2,336/2,336 A queries. The older claim that
TRAIN had no accepted form is superseded only for the scoped scratchpad readout and circuit, not a complete network formula.

### 5.3 Relation to published work

Learning parities without intermediate supervision is difficult, and curricula over widths can help
([Cornacchia & Mossel, 2023](#references)). Arithmetic-transformer length transfer depends on training widths
([Jelassi et al., 2023](#references)); scratchpads as intermediate computation follow [Nye et al. (2021)](#references).
The dense TRAIN circuits are compared with the weight-sparse circuits of [Gao et al. (2025)](#references), without claiming that
sparsity was trained or tested here.

## 6. Limitations and open questions

Scores are teacher-forced; greedy rollout was not run. Neuron necessity was not measured. Only the scratchpad has ten-seed PART and
TRAIN results; answer-only cells remain three-seed screens. PART A's n 12 high-bit repair and full hand-built-stage swap
(65,536/65,536) are repairs, not native generalisation. Only one HAND per format was tested, with no alternative designs.
Discovery used 256 n 10 rows from final snapshots. The TRAIN readout leaves earlier computation trained, and recall .67–1 does not
support a strict HAND-edge superset in every seed.

The earlier audit left ten-seed TRAIN T, TRAIN A's wider-data retry and scratchpad complement tests open; their outcomes are reported
above, including the negative retry. Still open are ten-seed PART A and TRAIN A, the historical matched-zero-control provenance for the earlier PART T table on seeds
3–9, TRAIN A seed 2 checkpoint dynamics and public code. The new full-domain PART T controls do not establish that missing historical
provenance; they are separately registered tests. TRAIN T's wider-domain circuit failures and native-decision changes also remain
limitations, even though primary-domain accuracy-share acceptance survives. One final-run dynamics record reaches only step 7,500, and the training
device was not logged. No missing diagnostic is treated as a passed check.

## 7. Prediction record

Predictions were committed before any smoke run. PART T by 2.5k held (500); PART A by 15k with poor unseen-D and n 12 accuracy held
(screen 5.5–7k; unseen D .09–.10). TRAIN T 3/3 by 15k mostly held in screening (seed 0 took 16k); the final run solved 10/10 but only
2/10 first reached exact by 15k (14–19.5k). TRAIN A ≥ 1/3 by 30k held (2/3). TRAIN T fails n 12 held in screening (0/3) and remained
limited in the final (maximum .26). Unseen-D accuracy ≥ .95 in most final seeds held (7/10).

## Appendix A. Weights excerpt and reading

The PART T MLP2 bank is a_j = ReLU(β_j0 + β_j·[X, A, DB, QB, QF, LOW, SEP]), 256 units, 2,819 numbers. Its response clusters
(159–184 units) are the 8 truth-table cells of the XOR-AND step; the substitute ReLUs are not identified native neurons. HAND T's
10 answer-writing stage neurons have W_in values {−2, −1, 1}. In PART T seed 0, the top neurons by |W_out| × |W_in| include:

*Table A1. PART T seed 0 weight slices on [X, A, DB, QB, QF, LOW, SEP].*

| Neuron | W_in | Bias | W_out[0, 1] |
|---|---|---|---|
| 141 | −.06, −.08, −.08, .01, −.01, .02, −.04 | .05 | .09, −.09 |
| 194 | .04, .07, −.03, −.04, −.03, −.03, −.07 | .03 | −.12, .10 |

Many small units share each truth-table cell. TRAIN T final uses 168–194 of 256 MLP2 neurons for 90% of |W_out| × |W_in| mass,
compared with HAND's 10.

The PART A bank is a_j = ReLU(β_j0 + Σᵢ α_ji·NB_i + u_j[D] + v_j[col]). A dense per-divisor parity surrogate covers seen divisors
and the low ten bits only, accounting for unseen-D and n 12 failures. HAND A uses 262 neurons with W_in {1, 32}.

*Table A2. Native PART A support detectors, final snapshots. Listed NB weights are approximately −.4.*

| Seed | Neuron | NB positions | D, bit | HAND S(D, bit) within NB_0..9 |
|---|---|---|---|---|
| 0 | 730 | 3, 6, 9 | 18, 3 | 3, 6, 9 |
| 1 | 213 | 2, 5, 8 | 18, 2 | 2, 5, 8 |
| 2 | 524 | 1, 4, 6, 9 | 31, 1 | 1, 4, 6, 9 |

These neurons read exactly the HAND support set for their divisor and bit, rather than one universal low-bit count direction.

## Appendix B. Detailed evidence

*Table B1. Earlier screening diagnostics, n 10 batch of 256 teacher-forced rows. These are not the ten-seed final-run ranges.*

| Cell | E best / final | H; M | Best probe R² | Inter → DAS; R / X | Top patch (drop / rec) | Edges ρ; J | r99 (HAND); JS |
|---|---|---|---|---|---|---|---|
| PART T s0–2 | 1 / 1 | 4/7; 1/3 | .88–.91, 1, .90–.91, .66–.68, .88–.89 (3) | .92–1 → 1; 1 / 0 | MLP2 1 / 1 | .996; 1 | 9 (10); 0 |
| PART A s0–2 | 1, 1, .996 | 14/16; 2/4 | .52–.57 / .51–.60 (4) | .58–.73 → .97–1; .30–.42 / .13–.16 | MLP3 1 / 1 | .94–.97; .60–1 | 25; 0 |
| TRAIN T s0–2 | 1 / 1 | 4–5/12; 3/3 | .83–.88, .71–.88, .85–.95, .69–.90, .78–.91 (3) | .99–1 → 1; 1 / 0 | MLP2 1 / 1 | .81–.93; .29–.39 | 4–10 (10); ≤ .45 |
| TRAIN A s1 | 1 / 1 | 19–20/30–32; 4/4 | .74 / .69–.72 (4) | .98–1 → 1; 1 / .04–.05 | MLP0 1.01 / .86–.90 | .82–.87; .09 | 31–33; ≤ .47 |
| TRAIN A s2 | 1 / 1 | 16–18/25–28; 4/4 | .74 / .63–.68 (4) | .85–.99 → 1; .98–1 / .07–.08 | MLP3 1.03 / .45–.47 | .83–.86; .03 | 28–29; ≤ .46 |
| TRAIN A s0 (failed) | .988 / .988 | 20–21/26; 4/4 | .78 / .68–.70 (4) | 1 → 1; .98 / .09 | MLP3 1.01 / .73–.76 | .74–.83; .10 | 40; ≤ .45 |

E = exact; H/M = used heads/MLPs, mean then zero ablation (> 5 pp loss); probe = best linear R² at the parenthesised layer
(T: shift, divisor, quotient, AND, next; A: count, next); R/X = restricted/excluded probe-subspace exact; ρ = edge-attribution
exactness check; J = edge Jaccard against HAND; r99 = last-residual rank; JS = attention divergence. These measures and patching
are defined in the [glossary](../GLOSSARY.md#analysis-measures). In T, divisor and AND have no answer-fixing pairs, and step, column
and phase are constant at the query: these variables are untested, not unused. TRAIN A seed 0's final edge ρ is .74 (< .8), so that
edge reading is not trusted.

*Table B2. Later diagnostics summary. Faith is held-out exact after resampling outside the discovered edges.*

| Cell | E best / final | Mean-used H; M (n90 neurons) | Best probe R² | r99; JS | Discovered edges; faith |
|---|---|---|---|---|---|
| PART T s0–2 | 1/1 | 4; 1 | shift .88–.91, Q .90–.91, next .88–.89 at L3 | 9 vs HAND 10; 0 | 9; 1 |
| PART A s0–2 | 1, 1, .996 | 14; 2 | count .52–.57, next .51–.60 at L4 | 25; 0 | 30; .99–1 |
| TRAIN T final s0–9 | 1/1 (10/10) | 3–8; 2–3 (255–360) | shift .83–.97, divisor .50–.93, Q .83–.99, AND .55–.89, next .73–.95 at L3 | 3–13 vs HAND 10; L0–1 ≤ .20, L2 .06–.50 | 13–71; .89–.99 |
| TRAIN A s1, s2 | 1/1 | 16–20; 4 | count .74, next .63–.72 at L4 | 28–33; ≤ .47 | 848, 1000; .92–.95 |
| TRAIN A s0 (failed) | .988/.988 | 20–21; 4 | count .78, next .68–.70 at L4 | 40; ≤ .45 | 989; .91 |

Q = quotient; n90 = neurons carrying 90% of the reported ablation effect. Probe layers count residual readout sites; MLP indices
are zero-based. The final-run attention and size ranges do not replace the earlier screening measurements in Table B1.

## What to read next

Compare [binary matrix product](gf2-matrix-product.md) for another XOR-and-AND update, or [binary elimination](gf2-elimination.md) for a row-update rule inside a larger frozen computation.

## References

- Cornacchia, E., & Mossel, E. (2023). A mathematical model for curriculum learning for parities. *ICML 2023 (arXiv preprint)*.
  https://doi.org/10.48550/arXiv.2301.13833 (parity learning and curriculum over widths)
- Gao et al. (2025). Weight-sparse transformers have interpretable circuits. *arXiv preprint*.
  https://doi.org/10.48550/arxiv.2511.13653 (comparison with dense learned circuits)
- Jelassi, S., d'Ascoli, S., Domingo-Enrich, C., Wu, Y., Li, Y., & Charton, F. (2023). Length generalization in arithmetic
  transformers. *arXiv preprint*. https://doi.org/10.48550/arXiv.2306.15400; https://doi.org/10.48550/arxiv.2306.15400
  (length transfer and training widths)
- Nye, M., Andreassen, A. J., Gur-Ari, G., et al. (2021). Show your work: Scratchpads for intermediate computation with language
  models. *arXiv preprint*. https://doi.org/10.48550/arXiv.2112.00114; https://arxiv.org/abs/2112.00114 (scratchpads)
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization beyond overfitting on small
  algorithmic datasets. *arXiv preprint*. https://doi.org/10.48550/arXiv.2201.02177 (grokking)
