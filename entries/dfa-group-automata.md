# Group automata (parity, Z3, S3)

Rearrange three objects, then rearrange them again: the final order depends on the sequence of moves. Six-layer transformers tracked these six possible arrangements, odd/even counts or remainders after division by 3, either answering directly or writing the current state after every symbol as a [scratchpad](../GLOSSARY.md#task-formats). All 10 final arrangement-tracking runs reached whole-output accuracy 1.0 on 2k held-out words with correct earlier output tokens supplied, but the computation ran through a chain of feed-forward blocks, which process each position, rather than the hand-built design's single state reader and update table.

The task is partly solved and the formulas are [partial](../GLOSSARY.md#verifying-a-circuit): table replacement passed in 10/10 final runs and tests changing activity outside the discovered circuits passed in all 8 measured runs, leaving 2 without that certificate; older failed joint tests remain failed. At 128 symbols every final trained run scored 0 on 1000 random words, while models learning one stage on the fixed hand-built scaffold were exact at that length.

| Fact | Value |
|---|---|
| Task | Give the final state of a parity, Z3 or S3 automaton after a word of 1..16 symbols, answer only or with the state written after every symbol |
| Input → output example | Parity, answer only: a word of five symbols with values 1 0 1 1 0 → state 1 (three symbols of value 1) |
| Model | GPT-NeoX-shaped, 6 layers × 4 heads, d_model 256, MLP 512 ReLU, no LayerNorm, no positional encoding; PART trains one MLP (layer 1, or layer 5 for answer-only S3) |
| Rungs and seeds | HAND exact in all 6 cells; PART and TRAIN 10 seeds per cell; failed cells retried (fresh words, then tail-balanced counts; answer-only S3 TRAIN one size step) |
| Closed form status | Partial (final S3 scratchpad accepted on 8/10: substitution 1.0 in 10/10, complement .94–1.0 in 8/8 with a discovered circuit; no universal or joint-modularity claim) |
| Status | Partly solved (scratchpad 10/10 per group; final S3 TRAIN 10/10 exact; answer-only S3 TRAIN 2/10 ≥ .99, 0/10 exact); last run 2026-11-08; earlier audit 2026-11-07 |
| Weights | [Hugging Face: dfa-group-automata](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/dfa-group-automata), revision `ababc8276281`; final archive note: 217 networks; earlier archive: 219 |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

HAND sets every weight by formula, PART trains one stage on those frozen parts and TRAIN learns all weights ([rungs](../GLOSSARY.md#the-three-rungs)). An MLP is the feed-forward block between attention stages ([glossary](../GLOSSARY.md#standard-machine-learning-terms)). A scratchpad writes intermediate results as output tokens, not hidden memory ([task formats](../GLOSSARY.md#task-formats)). Complement resampling replaces everything outside a proposed circuit using values from other inputs ([verification](../GLOSSARY.md#verifying-a-circuit)).

## The circuit at a glance

<img src="../figures/circuits/dfa-group-automata.svg" width="640" alt="For S3, the permutations of three items, an unverified early state-code and fetch branch feeds late feed-forward blocks, also called MLPs, on one residual stream. Their table tests use externally supplied true state and symbol keys. Some seeds bypass the last block. Identity with a first-pair swap targets that swap; upstream computation and the complete circuit remain partial.">

*Figure S1. Final scratchpad networks for S3 (permutations of three items), trained from scratch (seeds 0–9), differ from the hand-built reference through a serial feed-forward-block (MLP) path with an optional last block and late tables tested using externally supplied true state/symbol keys; status is partial, with paired acceptance on 8/10, two missing circuit certificates, upstream fetching unverified and older failed joint scrubs still failed.*

Figure S1 shows the serial state-code route and the last-block bypass; the table tests use true state and symbol keys rather than a recovered native fetch law.

## 1. Task and data

Parity records whether a count is odd or even. A permutation rearranges three objects; composing permutations means applying one rearrangement after another. Unlike adding counts, changing their order can change the final rearrangement. A transition table lists the next state for each current-state/symbol pair.

The identity is the state that leaves a symbol unchanged when combined with it; an abelian group is one in which order does not change the product. A word s_1..s_n is a sequence of group elements: parity (Z2, 2 symbols), a Z3 counter (3 symbols) or S3 (the 6 permutations of 3
objects). The automaton starts at q_0 = identity and updates q_t = q_{t-1} · s_t; the task is the state after the word. Two formats were
studied (Table 1); in the scratchpad format the network writes intermediate results before the answer
([glossary](../GLOSSARY.md#task-formats)).

*Table 1. Output formats. Q(q) is the token of state q.*

| Format | Sequence |
|---|---|
| `A` (answer only) | `BOS s_1 .. s_n = Q(q_n)` |
| `B` (scratchpad) | `BOS s_1 Q(q_1) .. s_n Q(q_n)`, teacher forced |

For parity, the five-symbol word with values 1 0 1 1 0 has state 1, because three symbols have value 1.

The word length n is uniform in 1..16. Run 1 used 20k unique training words per seed; later data retries used fresh words each batch, excluding test words, as detailed in Table 2. Each seed has 2k disjoint test words, and an output is correct only on exact match of every answer token. Format B is teacher-forced: each token is scored given the correct previous tokens
([glossary](../GLOSSARY.md#task-formats)). Length generalisation was tested on 1000 random words at n = 32, 64 and 128. For the
answer-only abelian cells, S denotes the sum of the symbol values, so that the answer is S mod G with G = 2 or 3.

## 2. Model

The shared architecture and instrumentation are described in [METHODS.md](../METHODS.md). The three
[rungs](../GLOSSARY.md#the-three-rungs) are HAND (every weight set by formula), PART (frozen hand-built parts with one trained stage)
and TRAIN (all weights trained from scratch).

### 2.1 Architecture

All networks are GPT-NeoX-shaped with 6 layers × 4 heads (d_head 64), d_model 256, MLP 512 ReLU, no LayerNorm, no positional encoding
(causal mask only) and an untied read-out. The vocabulary has 7 / 9 / 15 tokens for parity / Z3 / S3 (BOS, PAD, the G symbols, `=` and
the G states).

### 2.2 Frozen parts and trained part

PART re-initialises and trains one MLP on the frozen HAND: layer 1 in format B and in answer-only parity and Z3, layer 5 in answer-only
S3 (dense_h_to_4h [512, 256] + bias, dense_4h_to_h [256, 512] + bias). In the "mask" arms the PART stage's input weights on the position
dimensions are held at 0. TRAIN trains all weights from the default initialisation.

### 2.3 Training

All runs used AdamW with learning rate 1e-3, weight decay .1, betas (.9, .98), warmup 1000 and a cosine schedule, clipping at 1, batch
256 and fp32. Test accuracy was evaluated every 500 steps and the best-test snapshot was kept. Runs were on consumer GPUs (10–24 GB).
Table 2 gives the runs.

*Table 2. Runs. Each later run is a retry of the cells that failed before it.*

| Run | Data | Steps |
|---|---|---|
| run 1 | static 20k words | PART 10k (Z3 A 30k), TRAIN 40k; plateau stop 5k after 15k |
| run 2 (data rule) | fresh random words every batch (test words masked) | PART 30k, TRAIN 40k |
| run 3 | PART with tail-balanced symbol counts (per-word symbol probabilities ~ Dirichlet(1)) | as run 2 |
| run 3 (size step, A S3 TRAIN) | fresh words, d_model 512, 10 seeds | 40k |
| run 3b (size step, A S3 TRAIN) | fresh words, d_model 512, 3 seeds, 2× budget from scratch | 80k |
| final S3 B TRAIN (2026-11-08) | 10 seeds, weight decay .1 (the only final-run value) | 40k budget; solved stop 3500 |

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/dfa-group-automata](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/dfa-group-automata).
Held-out test words per group and seed are in `test_<group>.safetensors`. The earlier archive description counted 219 networks
(6 HAND, 200 PART, 13 TRAIN); the final publication note records 217. These are retained as dated archive inventories, not one
current total.

### 2.5 Archive

The archive holds `hand_<group>_<F>` (state dicts), `part_<run>_<F>_s<seed>` (stage tensors, all seeds), `train_<run>_<F>_s<seed>`
(full networks, best and worst seed per run), `config.json` and `manifest.json` (seed, rung, run arguments, fp32 test accuracy,
recorded accuracy, sha256). Every file was reloaded bit-identically and its test accuracy reproduced. Revision `ababc8276281`
includes final-run S3 B TRAIN seeds 0–3. This final pack saves best-step weights only: its best and final diagnostics use the same
weights, not two independent networks.

## 3. Hand-built solution

Layer 0 is a position stage with BOS as attention sink (a position that heads can attend to when nothing else is relevant): it computes P = 1/(j+1) at position j, and hinges give R = j/256. The later
layers differ by format (Table F1). G is the group size, QP holds the copied previous-state code, and a one-hot code gives each state its own coordinate ([glossary](../GLOSSARY.md#standard-machine-learning-terms)).

Scratchpad (B). A layer-1 recency head, with score 100·[state token] + 20 j, copies the previous state q_{t-1} into a block QP. The
layer-1 MLP has G² neurons relu(QP_a + TOK_b − 1), each writing OUT[a·b], where TOK_b is the one-hot of the current symbol. This is one
head and 4 / 9 / 36 neurons at any n.

Answer only, abelian groups. A BOS-sink head averages the symbols, which gives S, the sum of the symbol values. (G−1)·n hinge neurons
relu(S − k) (each is zero below the threshold k and rises linearly above it) are read out as [S mod G = r]. This is one head and 16 / 32 neurons at n = 16, and it grows with n.

Answer only, S3. The network uses prefix doubling: the word is covered by blocks of symbols whose states are composed pairwise, so the covered length doubles with each layer. In layer k an index head fetches the block 2^(k−1) back and a 36-neuron table composes
the two blocks into the block states ST1..ST4. This is 5 heads and 180 neurons, exact for n ≤ 31.

All 6 HAND cells are exact on 2000 words with n = 1..16.

Table F1 records the fixed reference alternatives. A frozen position stage feeds separate routes. Scratchpad B fetches the previous state and applies a transition table; answer-only parity and Z3 sum symbols and read a residue; answer-only S3 composes blocks by prefix doubling. PART trains only each route's named final MLP, leaving the other stages frozen.

<details>
<summary>Technical state-table, residue and prefix-doubling references (Table F1)</summary>

*Table F1. The hand-built networks. Format B copies the previous state with one recency head and applies the transition with a G²-neuron table; answer-only parity and Z3 sum the symbols and read the residue with hinges; answer-only S3 composes blocks by prefix doubling. Only the named final MLP in each route is replaced by a trained stage in PART.*

| Alternative route | Fixed computation and trained interface |
|---|---|
| Shared fixed position stage | Read word s_1 .. s_t. Layer 0 gives P = 1/(j+1) and R = j/256. |
| B: scratchpad | A fixed layer-1 recency head copies q_t-1 into QP. The layer-1 MLP, trained in PART, has G squared neurons ReLU(QP_a + TOK_b − 1). The output is state q_t = q_t-1 times s_t. |
| A: parity and Z3 | A fixed BOS-sink head supplies symbol sum S. The layer-1 MLP, trained in PART, uses (G−1) n hinges ReLU(S − k) and answers [S mod G = r]. |
| A: S3 | Fixed layers 1 to 4 fetch the block 2^(k−1) back with an index head and compose it with a 36-neuron table, storing ST1..ST4. The layer-5 MLP, trained in PART, composes the last blocks and answers q_n. |

</details>

## 4. What training finds

Table 3 summarises every cell; Appendix B gives the diagnostics per cell.

*Table 3. Results per cell (test exact match on held-out words, ranges over seeds; run numbers as in Table 2).*

| Cell | PART | TRAIN |
|---|---|---|
| B, all groups, earlier runs | 10/10 exact by step 500, masked or not; exact at n = 128 | groks at steps 500–2500 (10/10 per group); n = 32 .05–.99, n = 64 ≤ .14 |
| B S3, final run | no new PART cell | 10/10 exact; first exact 1500–2000; n = 32 ≥ .9 in 2/10, n = 64 ≤ .18, n = 128 0 |
| A S3 | 10/10 by step 1000; n = 32 .18 (HAND .18) | run 1 memorises (sign right in 8/10); run 2 best .70–.9995 (2/10 ≥ .99); size step: run 3 .487–.988, run 3b .36–.51 (not saved) |
| A parity | runs 1 / 2 / 3: best .96–.99 / .986–.999 / .988–1.0; run 3 masked 4/10 exact (7/10 on fresh batches), unmasked 1/10 | run 1 .50–.52; run 2 10/10 (grok at 13–21k) |
| A Z3 | runs 1 / 2 / 3: .91–.98 / .92–.97 / .88–.97; run 3 0/10 | run 1 9/10 exact (10/10 ≥ .99; rounded .998–1); run 2 10/10 (grok at 3.5–18k) |

### 4.1 Scratchpad (format B)

PART (all groups, masked or not) was exact on 10/10 seeds by step 500. It keeps the recency head (previous-state attention mass 1.0) and
the stage is a (q_prev, symbol) table, exact at n = 128.

TRAIN grokked at steps 500–2500 (10/10 per group); grokking means that test accuracy rises after training accuracy
([glossary](../GLOSSARY.md#counting-results)). The late MLPs are the transition (δ) table (substitution 1.0: parity MLP5, Z3 MLP3–5, 10/10). The fetch
of the previous state differs from HAND: parity spreads it over sharp heads in layers 2–5 (a single HAND-like head in 1/10), and Z3 has
one sharp head (mass .85–.95) in 7/10. Length generalisation is poor: n = 32 gives .05–.99, n = 64 ≤ .14, and token accuracy breaks at
t = 24–48.

### 4.2 Answer only, S3

PART reached 10/10 by step 1000 and reproduces the hand-built composition tables, exact to n = 31, the hand-built depth bound. The block states ST1..ST4 are decodable at accuracy 1.0 in layers 2..5 (the HAND layers), and at n =
32 PART scores .18, equal to HAND's .18 (HAND is exact only for n ≤ 31).

A permutation's sign records whether an odd or even number of swaps produces it; sign alone does not identify the full rearrangement. TRAIN run 1 (20k static words) memorised and learned only the sign of the permutation (right sign in 8/10 seeds). TRAIN run 2 (fresh
words) has sign accuracy 1.0 (10/10) and is exact up to n ≈ 9–13; its best test accuracy is .70–.9995 (2/10 ≥ .99), still rising at the
budget. ST2..ST4 are at chance under the linear probes: no linearly decoded doubling was found, but a different code was not ruled out.

The one size step of the retry order (d_model 512, fresh words) produced no saved network. Run 3 (10 seeds, 40k) reached test .487–.988,
still rising, but the networks were lost: the GPU ran out of memory in the final length evaluation, before the save. Run 3b (3 seeds,
80k from scratch, 2× budget) stopped at step 27.5k when the rented host's disk filled, at test .36–.51. No best snapshot exists, so
neither run is diagnosed or archived, and the answer-only S3 TRAIN result stands at 2/10 ≥ .99 (run 2). An audit on 2026-11-07 of the
unrounded curves found 2/10 seeds ≥ .99 in run 2 (maximum .9995, none exact). The later compact summary repeats 3/10 and a rounded
.70–1.0 range; those are the earlier rounded report, not a third seed meeting .99. The source audit records seed 0
.9910000562667847, seed 4 .999500036239624 and seed 3 .9895000457763672. Thus 0/10 exact and 2/10 ≥ .99 describe different criteria. The prediction summary's
frontier n ≈ 9–16 is retained alongside the exact-to-n ≈ 9–13 result above.

### 4.3 Answer only, parity and Z3

The PART stage computes S exactly (probe R² 1.00; [glossary](../GLOSSARY.md#analysis-measures)) and is a function f(S), but whole large
S values are wrong. The earlier lede described correct residues only for S ≤ 11–18; the count-specific failures below retain the
more detailed recorded scope. With fresh words (run 2), S = 13 is wrong for parity in 10/10 seeds, and every S ≥ 17–19 is wrong for Z3. With
tail-balanced S (run 3, large S now ~12% of words), parity masked was exact on the original test split in 4/10 (7/10 on fresh batches),
unmasked in 1/10, and Z3 in 0/10. Coverage of large S in the data is therefore not the cause.

TRAIN run 1 (20k static words): parity counts S but has no S mod 2 (test .50–.52, train also at chance); Z3 solved 9/10 (q decodable at
1.00 at L4). Here 9/10 counts exact seeds; the source audit records seed 7 .9975000619888306, so 10/10 meet ≥ .99, despite the
older .998–1 rounded range. TRAIN run 2 (fresh words): parity and Z3 10/10 (grok at 13–21k and 3.5–18k steps). S is decodable at L1–L2 and q at 1.00
from L2–L3, and the MLPs are functions of S. These networks use HAND's variables (interchange in Section 5) in a different code (no
orthogonal map to HAND's residual, CKA .11–.23), where centred kernel alignment (CKA) compares representation geometry rather than causal use ([glossary](../GLOSSARY.md#analysis-measures)), but the residue staircase covers only the trained S (parity at n = 32 at chance; Z3 at
n = 32 ≥ .95 in 3/10).

### 4.4 Causal evidence

The failed abelian PART stages use S (interchange flip .92–1; [glossary](../GLOSSARY.md#analysis-measures)) and carry q, which DAS
(distributed alignment search, which learns rotated directions for swapping a variable between inputs) recovers (.85–1) although it is only about half linearly readable (.48–.79). The miss is therefore the
large-S read-out, not the count. Deleting the S/q subspace drops every parity A PART and parity TRAIN run 2 network to chance
(.46–.52), and keeping only that subspace leaves accuracy unchanged. Parity TRAIN run 1 has S but no answer-controlling variable (all
flips at chance). Solved TRAIN networks use HAND's variables in their own basis (residual alignment fit R² < 0), and a high CKA does not
mark success (failed parity run 1 .99, solved run 2 ≤ .23). S3 TRAIN shows no linear trace of HAND's block states; whether some other
code carries them was not tested.

### 4.5 Final S3 scratchpad run

The final run solved 10/10 at the joint stop, step 3500, with first exact at 1500–2000. Every seed uses a serial MLP chain, with MLP0–4 needed in all seeds and MLP5 needed in only one subtype. Late transition tables and seed-dependent fetch heads carry the update; none of the 8 discovered circuits matches HAND. There are two subtypes:
state q completes by probe layer 4–5 with MLP5 idle (6/10), or q completes in MLP5 (4/10). The discovered circuits have 31–96 edges
with held-out faith .94–.96, where faith is exact match after resampling outside the kept circuit
([glossary](../GLOSSARY.md#analysis-measures)). Final length scores are n 32 .02–1.0 (≥ .9 in 2/10), n 64 ≤ .18 and n 128 0.

The final round uses one HAND design. Previous-state use is not established by interchange: it has no answer-fixing pairs, so
`nf = 0` means untested. A serial path is supported by path drops on seed 0: MLP0–5 to output .99, embedding to MLP0 .58,
MLP0 to MLP1 .44, MLP1 to MLP2 .44 and fetch head L2H0 .41. Low single-head mean-ablation effect does not establish that a head
carries no signal, a concern discussed by [Parodi et al. (2026)](#references).

Table F2 records the final trained family and its incomplete certificate coverage. Trained early MLPs and seed-dependent fetch heads feed the late transition-table writes. The final MLP is needed only in one subtype. All computational nodes were trained. Status remains partial: late-table substitution passes, complement certificates cover only measured discovered circuits, and upstream fetching is unverified. Earlier failed joint scrubs remain failed.

<details>
<summary>Technical final trained roles and missing certificates (Table F2)</summary>

*Table F2. The final S3 scratchpad TRAIN family. All listed computational components are trained. The status is partial: late-table substitution passes in 10/10 and the discovered-circuit complement passes in 8/8 tested seeds, giving acceptance on 8/10; two seeds have no discovered circuit. The upstream state-fetch formula is unverified. This does not reverse the older failed whole-hypothesis scrub or establish independent modules.*

| Trained components | Input and recorded role |
|---|---|
| MLP0, MLP1 and MLP2 | Read the word and written states so far; form a serial state code. |
| Seed-dependent fetch heads | Read the word/history and earlier MLP code; feed the late transition readouts. No upstream fetch formula is verified. |
| MLP3 and MLP4 | Delta tables on previous state and symbol write state q_t, directly or through MLP5. |
| MLP5 alternative | Needed in 4 of 10 seeds and idle in 6 of 10. The idle subtype retains the direct state-output route. |

</details>

## 5. Mechanism and verification

### 5.1 Final-run closed form and acceptance

For final S3 B TRAIN, `MLP_l(x_t) = T_l[q_{t-1}, s_t]` for l = 3, 4, 5: each late MLP is separately replaced by a 36-entry table in previous state and symbol, with the entries fitted on a disjoint half of a batch of 16-symbol words and scored on the held-out half. Substitution retains whole-output exact 1.0 in 10/10 seeds at each of these layers. The saved control rows omit the batch and half-split row counts; this evaluation is not the 2k-word training test. Zero, mean, shuffle and swap controls were run per layer. Zero/mean at layer 3 are ≤ .25 and shuffle is 0; zero/mean at MLP5 keep 1.0 in the 6/10 subtype
where it is idle, so MLP3–4 provide the informative checks.

The discovered-edge diagnostic files each record a 1000-word batch. Discovery reports faith on its held-out portion; the separate complement checks report resampling of the kept edge claim on the diagnostic batch, not the 2k-word training test. Resampling outside the discovered edges keeps .942–.963 in 8/8 seeds (0–2 and 4–8), summarised as .94–.96 in discovery and
.94–1.0 in the compact acceptance section. The larger node claims in seeds 0, 1 and 2 keep .993, 1.0 and .972. Circuit-resampled
and random same-size controls are 0, and the full/base and empty/source self-tests pass. Both acceptance shares exceed .90, so
acceptance covers 8/10 seeds. Seeds 3 and 9 have no discovered circuit; seed 9's narrower head-union claim keeps .716 and fails.
No missing discovery is counted as a pass.

The smaller claim consisting of MLP0–5 and mean-ablation-used heads fails (.0–.53; individual results .016, 0 and .53). Thus even
some heads called idle by that ablation screen carry signal under resampling. These results support the scoped final-run circuit
and late tables, not independent causal modules or a closed form for all earlier layers. The overall status is partial.

### 5.2 Earlier closed forms and audit

In format B the closed form is q_t = δ(q_{t-1}, s_t), a G × G table applied to the fetched previous state. Substituting the (q_prev,
symbol) mean table for the stage keeps accuracy 1.0 (PART 30/30; TRAIN parity and Z3 10/10) on the stated words. The later compact summary reports TRAIN 30/30 across all three groups; this
extends the reported substitution census, without changing the failed universal-table re-check below. As a universal
reduction it is rejected: on a re-check, parity seed 9 keeps 126/128 (< .99 of the network). In answer-only S3 PART, MLP5 = f(q) and
MLP4 = f(ST4) tables, 10/10. In the answer-only abelian cells each stage is a table f(S) (substitution = the network's own accuracy,
PART 40/40, TRAIN run 2 parity 10/10, Z3 8–9/10); replacing it by the exact table [S mod G] gives 1.0, so the closed form exists and the
trained stage misses only its large-S teeth.

These earlier forms are [sufficient readouts](../GLOSSARY.md#verifying-a-circuit), not verified mechanisms. The audit of 2026-11-07
found that no earlier closed form passes the full bar: every substitution then was on best snapshots and none on final snapshots.
The final S3 pack of Section 5.1 is a later, separate result; its saved best-step weights also serve as final weights.

### 5.3 Earlier circuit probes

Circuit probes were run on selected best snapshots on 2026-11-06 (Table 4). The probe records label a component verified when a causal intervention
plus an accuracy-preserving substitution hold on the stated finite domain, and supported with causal evidence only; neither label meets
the atlas acceptance rule, so Table 4 writes "causal + substitution" for the first, and the audit caveat above applies. H = heads, N = neurons; B is teacher-forced.

*Table 4. Components probed.*

| Component | Role | Evidence |
|---|---|---|
| S3 B TRAIN, all L0 heads | weighted prefix histogram | substitution 512/512 in both tested seeds (causal + substitution) |
| S3 B TRAIN, joint forms (histogram heads, ideal L2 routing, MLP2..5 tables) | answer-preserving reduction | seed 4 full net 100%; both pruned circuits 100% with refitted coefficients, 512 words (causal + substitution, answer reduction only) |
| S3 B PART, L1 fetch + MLP1 | displayed-state transport, local product | fetch zeroed: exact 0; changed-state patch 100%; 36-number substitution 256/256 (causal + substitution) |
| S3 A PART, MLP5 | composes the last frozen blocks | 36-number substitution 256/256, drop 0 (causal + substitution) |
| S3 A TRAIN (run 2), MLP5 | sign + inverse-image-of-zero readout | rank-4 projection 250/256, agreement 100%; sign-only 88/256, point-only 171/256 (causal + substitution, readout only) |
| Z3 A TRAIN (run 2), MLP5 | in-range periodic readout | seven-number cosine form 256/256, drop 0; MLP5 deletion 45.3% (causal + substitution at n = 16) |

The closed forms behind Table 4 are as follows. A histogram head computes H = Σ_t C_t exp(q·k_t/8) OV_t / Σ_t C_t exp(q·k_t/8), where C_t
are the prefix token counts. S3 PART writes the state logit T[MUL[q_a, q_b]] with 36 fitted numbers, where q_a and q_b are the fetched
and frozen-block states. In S3 A TRAIN the state write is δ_e = c(x) + u(x) sign(e) + v_{e⁻¹(0)}(x), which explains 99.96% of the
state-write variance; the coefficients are still neural (no input algorithm). In Z3 A TRAIN, with d = max(0, 1 − (S − 6)/29.5), the
write is δ_{6+e} = d a_e cos(2π(S − e)/3) and the other channels are d b_{S mod 3}, with a = (24.439, 27.467, 24.795) and b = (−62.595,
2.495, −38.438); the write R² is .764, and the form scores 228/256 at n = 32 and 65/256 at n = 64, so it is not faithful out of range.

Supported only: the S3 B TRAIN previous-state transport (whole-L2 changed-state patch 100%; ideal routing 512/512 in both seeds, but
seed 0 improves 115/128 to 128/128 on random-display words, which is a repair), and the S3 A TRAIN sign and point factors (DAS
transfers 256/256 and 181/187). The two late heads alone transfer the point in only 61/351 donor patches, so the upstream point-product
algorithm is unresolved.

A resample ablation of the whole S3 B TRAIN hypothesis at once ([glossary](../GLOSSARY.md#verifying-a-circuit)) fails: it gives
71.42% (seed 4) and 1.69% (seed 0) on the sufficient circuits (mean of 3 draws), although each is 100% unscrubbed and under joint
substitution. The replacements therefore do not prove independent causal modules.

Sizes are sufficient upper bounds, not minima. S3 B TRAIN seed 4 keeps 7H + 2202N and seed 0 8H + 1333N (of 24H + 3072N), 512/512 each;
HAND B has 2H + 292N (256 position hinges + 36 pair gates), not re-minimised. In the learned PART stage, 34/512 (B) and 22/512 (A)
neurons suffice, 256/256.

Table F3 retains the earlier rejected joint hypothesis separately from the final family. Trained prefix-histogram heads feed a redundant later fetch, then late state-table MLPs. These components are sufficient readouts separately, but their joint scrub fails. This earlier hypothesis has no full-bar certificate and is not retroactively verified by the later final run.

<details>
<summary>Technical earlier hypothesis and failed joint scrub (Table F3)</summary>

*Table F3. The earlier circuit hypothesis for format B (S3 B TRAIN, seeds 0 and 4). Prefix-histogram heads in layer 0 feed a later, redundant fetch of the previous state, and late MLPs apply the δ table. All components were trained. Each part is a sufficient readout; scrubbing the joint hypothesis fails (71.42% and 1.69%), and per the 2026-11-07 audit no closed form passes the full bar. The later final run does not retroactively verify this hypothesis.*

| Earlier trained component | Proposed role, not a jointly accepted formula |
|---|---|
| Layer-0 heads | Read the word and written states so far as weighted prefix histograms. |
| Later heads | Redundantly fetch the previous state using distributed pair and position thresholds. |
| Late MLPs | Apply a delta table to the fetched state and symbol to write state q_t. The joint hypothesis fails the scrub above. |

</details>

### 5.4 Relation to the hand-built network and to published work

Unlike HAND, S3 B TRAIN fetches later and redundantly, with distributed pair and position thresholds. PART keeps HAND's interfaces but
not its single-neuron AND gates (pair dependence R² .9993 against product-only .148 for the best-fitting neuron).

With a scratchpad, one G² table works at any length (HAND uses 4–36 neurons) and is learned at once; the length limit comes from the
previous-state fetch (PART n = 128, TRAIN ≈ 2× training length). With the answer only, HAND needs (G−1)·n neurons (abelian) or log2 n
layers (S3); PART reproduces the S3 depth bound. The abelian residue staircase has only ~16–18 correct teeth, a resolution limit of a
one-layer ReLU MLP learning a period-G sawtooth on a scalar, consistent with the spectral bias of ReLU networks
([Hong et al., 2022](#references)), and not a data limit. Tail-balanced counts tested the rare-class oversampling remedy associated
with [Buda et al. (2018)](#references); retaining those failures is part of the evidence for the resolution diagnosis. Transformers are known to learn logarithmic-depth "shortcut" solutions to
automata that are brittle out of distribution ([Liu et al., 2022](#references)); here S3 PART is exactly such a log-depth solution, and
the answer-only TRAIN networks are length-bound. Fresh samples per step rescue the parity count-then-residue cell (run 1 0/10,
run 2 10/10), while Z3 improves from the earlier 9/10 exact to 10/10, in line with the advantage of fresh-sample SGD on parities ([Abbe & Sandon, 2023](#references)). Scratchpads as an aid to
intermediate computation were introduced by [Nye et al. (2021)](#references).

## 6. Limitations and open questions

The earlier probes, ablation, activation patching, interchange and DAS used best snapshots, leaving their final snapshots unchecked.
The earlier article's last run was dated 2026-11-05; its audit followed on 2026-11-07. The later final S3 pack has substitution and
complement coverage as scoped in Section 5.1; this does not supply final checks for other cells. Final-run used-neuron counts were not measured; the earlier sufficient upper bounds remain in Section 5.3. The scratchpad is teacher-forced. The diagnosed networks have width d 256. Two attempts at the same d 512 size step (answer-only S3 TRAIN) ended without saved networks (out of memory at 40k, disk full at 27.5k of 80k), so size against solved is untested. Each cell has 10
seeds, and length was evaluated on random words. The masked and unmasked PART pairs control for the position shortcut. The abelian PART
run 3 retry rules out coverage; a structured (hinge) initialisation of the stage was not tried.

The weights excerpt (Appendix A) reads B PART neurons as group characters; the probes support pair dependence but not a unique
group-character role. The f(S) tables for Z3 A TRAIN hold on random test words, but on a balanced histogram grid MLP2..5 = f(S) gives
only 90.41% agreement, so a global f(S) reduction is refuted. The parity A PART run-3 networks that are exact on test words still miss 2
of 17 counts on that grid.

Open questions:

- final-snapshot substitutions for the earlier forms (the 2026-11-07 audit recorded 0 final-weight diagnostics);
- the d 512 S3 answer-only size final (not completed);
- six solved seeds (6/9) of a context-reset S TRAIN follow-up, mechanism unclassified;
- the S3 answer-only point product;
- what the failed whole-hypothesis scrub omits;
- the smallest sufficient circuits under length extension.

## 7. Prediction record

The predictions were recorded in the task specification before any PART or TRAIN run. Table 5 lists them with the outcome.

*Table 5. Predictions and outcomes.*

| # | Prediction | Outcome |
|---|---|---|
| 1 | B PART exact, masked exact at 8× length | held, unmasked too |
| 2 | B TRAIN ≥ 7/10, hand-built circuit, half the seeds ≥ .9 at 2× length | 10/10; hand-built single fetch head 1/10 (parity); ≥ .9 at 2× length only 1–2/10 |
| 3 | A abelian PART exact | failed (large S) |
| 4 | A abelian TRAIN ≤ 5/10 with a length-bound count shortcut | run 1 parity 0/10, Z3 9/10; run 2 10/10 and length-bound (shortcut held, count broken) |
| 5 | A S3 PART = HAND including length | held |
| 6 | A S3 TRAIN ≤ 3/10, depth-bound | run 2 2/10 (≥ .99), 0/10 exact, with a length frontier, held |
| 7 | grokking order B < A abelian < A S3 | B first held; at PART the order is inverted |

The final S3 run confirms the scratchpad success count (10/10) and early learning (1500–2000), but rejects the predicted HAND circuit
(0/10 with HAND's single head). Only 2/10 reach ≥ .9 at twice the training length, below the predicted half; n 128 fails in 10/10.
These are final-run outcomes, distinct from the earlier 1–2/10 length-transfer counts.

## Appendix A. Weights excerpt and reading

In the B PART (masked) layer-1 stage, 65–85% of the input-weight energy (squared norm of dense_h_to_4h columns) sits on the QP and
symbol dimensions. The source describes the neurons as group characters, patterns of values over group elements, rather than HAND's AND gates, which activate for one state-symbol pair. Parity seed 0 neuron 116 reads QP (−.77, .22) and the
symbol (.73, −.26), a ± pattern that fires on (q, s) of unequal parity, which is XOR. S3 seed 0 neuron 106 reads QP and the symbols with
the permutation-sign pattern (+ − − + + −). The values are dense_h_to_4h rows of `part_<group>_mask_B_s0`, restricted to the QP and
symbol blocks. The probes of Section 5.3 support pair dependence but not a unique group-character role.

## Appendix B. Detailed evidence

Table B1 gives medians over seeds. Residual label L0 means the input and L6 means after all six transformer blocks. The probe is the best layer, one-hot accuracy, with R² for S and position t; r90 = principal
components for 90% of the last residual's variance; attention divergence to HAND is ≈ 0 throughout because HAND's unused heads are
uniform; patch = activation patching on a shuffled batch, the component with the largest accuracy drop; uses = interchange flip, then
DAS flip of the HAND variable at its best layer, ranges over 167 seeds; "untested" = no answer-fixing pairs, not unused. Terms are
defined in the [glossary](../GLOSSARY.md#analysis-measures).

*Table B1. Diagnostics per cell.*

| cell | used heads + MLPs | probes | r90 | patch (drop) | uses (inter → DAS) | best / final |
|---|---|---|---|---|---|---|
| B PART (3 groups) | 1 + 1 | q_prev, q 1.00 (L2) | 2–10 | MLP1 1.0, fetch head L1H0 .94 | sym 1, q .93–1 → 1; q_prev untested | 1 / 1 |
| A S3 PART | 5 + 5 | ST1..ST4 1.00 (L2..L5), q 1.00 | 35 | index heads L1–L3 ≈ 1 | q .26–.54 → .78–1; ST1–4 untested | 1 / 1 |
| B TRAIN parity / S3 | 1–6 + 4–6 / 0–3 + 4–6 | q_prev, q 1.00 (L3–6), t R² .88–.99 | 1.5 / 2 | MLP0 (6/10, 7/9), late MLPs .84–.98 | sym 1, q .99–1 → 1 | 1 / 1 |
| A parity PART r1 / r2 / r3 (+mask) | 1 + 1 | S 1.00 (L2), q .84–.97 / .93–1 / .93–1 | 1 | L1H0 and MLP1 1.0 (55/55) | S .92–1; q .48–.79 → .85–1 | .96–.99 / .986–.999 / .988–1.0 |
| A Z3 PART r1 / r2 / r3 | 1 + 1 | S 1.00 (L2), q .80 / .76 / .79 | 1 | L1H0 and MLP1 1.0 (11/11) | S .82–.90 (2 seeds); q .64–.86 → .70–.88 | .91–.98 / .92–.97 / .88–.97 |
| A parity TRAIN r1 / r2 | 0 + 0 / 3–10 + 4–6 | S 1.00 (L1), q .52 / 1.00 (L3) | 2 / 1 | — / layer-0 heads + MLP0 1.0 | S, q .48–.53 → .48–.53 / S 1.00, q .85–1 → .97–1 | .50–.52 / 1 |
| A Z3 TRAIN r1 / r2 | 2–6 + 5–6 / 3–6 + 4–6 | S R² .95 / .98 (L1 / L2), q 1.00 (L4 / L2) | 2 | MLP5 or layer-0 heads ≈ 1 | q .94–.96 → 1.00 (2 seeds) | .998–1 / 1 |
| A S3 TRAIN r1 / r2 | 0–4 + 0–5 / 4–20 + 4–6 | ST1 .37 / .55, ST2..4 .18–.23, q .34 / .64 | 2.5 / 4.5 | layer-0 heads, MLP5 ≈ 1 | untested / q .62–.99 → .57–.99 (5/7); ST1–4 untested | .21–.37 / .70–1.0 |
| A S3 TRAIN r3 / r3b (d 512) | not diagnosed (no best snapshot) | — | — | — | — | .49–.99 at 40k / .36–.51 at 27.5k (logged test) |

*Key: r1, r2, r3, r3b = runs 1, 2, 3 and 3b (Table 2); best / final = test exact match of the best and final snapshot.*

Seed census (150 best snapshots in 17 cells; mechanism classes, not basins). Table B2 lists the classes per cell.

*Table B2. Seed census.*

| Cell | Mechanism classes |
|---|---|
| B PART parity / Z3 / S3 | 20 / 20 / 20 frozen fetch + product, compact substitution 100% |
| B TRAIN (10 each), display-local / history-sensitive | parity 3/7, Z3 7/3, S3 9/1; the strongest-attention layer is an exactly sufficient fetch in 0/10, 2/10, 6/10 |
| A S3 PART (20) | 14 faithful compact products, 6 one-error repairs, all on frozen doubling |
| A parity TRAIN run 2 | 10 late-count codes |
| A Z3 TRAIN run 1 / run 2 | 9 count + context / 1 faithful late count, 9 count + context |
| A S3 TRAIN run 2 | seed 4 has the faithful sign + point readout; seed 0 (also ≥ .99) and seed 3 (.99 only after rounding) need extra directions |
| A parity PART run 3 | 8 scalar-hinge codes |
| Z3 A PART | 0 solved |

*Table B3. Later compact diagnostics summary. These values are retained beside the earlier Table B1 rather than silently replacing
its different probe statistics.*

| Cell | Used heads + MLPs | Probes | r90 | Best / final |
|---|---|---|---|---|
| B PART, three groups | 1 + 1 | q_prev, q 1.00 (L2) | 2–10 | 1 / 1 |
| A S3 PART | 5 + 5 | ST1..ST4 accuracy 1.00 (L2..L5), q R² .94 | 35 | 1 / 1 |
| B TRAIN parity / Z3 | 1–6 + 4–6 / 0–3 + 2–6 | q_prev, q accuracy 1.00 (L5–6 / L3), t R² .97 / .98 | 1.5 / 2 | 1 / 1 |
| B TRAIN S3 final (10) | 0–3 + 5–6 | symbol 1.00 (L0), q_prev R² .87–1.00 (L3–L5), q R² .98–.99 (L5–L6); L6 CKA .40–.53 | 2 | 1 / 1 |
| A parity PART r1 / r2 / r3 (+mask) | 1 + 1 | S R² 1.00 (L2), q R² .63 / .97 / .96–.99 | 1 | .96–.99 / .986–.999 / .988–1.0 |
| A Z3 PART r1 / r2 / r3; mask r1–r3 | 1 + 1 | S R² 1.00 (L2), q .80 / .76 / .79; mask .07 R² / .72 / .73 | 1 | .91–.98 / .92–.97 / .88–.97 |
| A parity TRAIN r1 / r2 | 0 + 0 / 3–10 + 4–6 | S R² 1.00 (L1), q accuracy .52 / 1.00 (L3) | 2 / 1 | .50–.52 / 1; final .48–.52 / 1 |
| A Z3 TRAIN r1 / r2 | 2–6 + 5–6 / 3–6 + 4–6 | S R² .95 / .98 (L1 / L2), q accuracy 1.00 (L4 / L2) | 2 | .998–1 / 1 (best = final) |
| A S3 TRAIN r1 / r2 | 0–4 + 0–5 / 4–20 + 4–6 | ST1 .37 / .55, ST2..4 .18–.23, q .34 / .64 | 2.5 / 4.5 | .21–.37 / .70–1.0 (rounded summary; see Section 4.2) |

Key: r1–r3 = runs 1–3; probe accuracy and R² are different measures; r90, CKA and attention JS are defined in the
[glossary](../GLOSSARY.md#analysis-measures). Attention JS is approximately 0 in the compact summary because unused HAND heads are
uniform, not because every trained head has HAND's role. The masked Z3 .07 is explicitly an R² value. The earlier q values in
Table B1 are one-hot accuracies, while the .63 and .94 figures here are R², not contradictory accuracies.

## What to read next

Compare [counter languages](counter-languages.md) for count-and-violation checks, or [Dyck brackets and agreement](dyck-k-and-agreement.md) for a written-state pointer whose partner computation remains unidentified.

## References

- Abbe, E., & Sandon, C. (2023). Polynomial-time universality and limitations of deep learning. *Communications on Pure and Applied
  Mathematics, 76*. https://doi.org/10.1002/cpa.22121 (fresh-sample SGD on parities)
- Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. *Neural Networks*. https://doi.org/10.1016/j.neunet.2018.07.011 (journal article; rare-class oversampling in a different task, context for the tail-balanced retry)
- Hong, Q., Siegel, J. W., Tan, Q., & Xu, J. (2022). On the activation function dependence of the spectral bias of neural networks.
  *arXiv preprint*. https://doi.org/10.48550/arxiv.2208.04924 (spectral bias of ReLU networks)
- Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A., & Zhang, C. (2022). Transformers learn shortcuts to automata. *arXiv preprint*.
  https://doi.org/10.48550/arxiv.2210.10749 (logarithmic-depth shortcuts to automata)
- Nye, M., Andreassen, A. J., Gur-Ari, G., et al. (2021). Show your work: Scratchpads for intermediate computation with language
  models. *arXiv preprint*. https://doi.org/10.48550/arxiv.2112.00114 (scratchpads)
- Parodi, F., Matelsky, J., & Segado, M. (2026). Zero-ablation overstates register content dependence in DINO vision transformers. *arXiv preprint*. https://doi.org/10.48550/arxiv.2604.14433 (arXiv preprint; ablation baselines in vision transformers, a methods comparison rather than a DFA result)
