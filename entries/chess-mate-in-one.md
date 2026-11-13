# Chess mate-in-one selection

If only the second of three supplied checks leaves a lone black king no escape square, this task picks the second; it selects the first mate from a list, without generating moves itself. Learning one output stage on a fixed hand-built escape-square counter solved both formats in 3/3 runs, but learning on short lists left only 52–64% exact on a 608-sequence longer-list stress set. Later whole-network screens reached exact 1.0 on 128 held-out sequences in 3/3 answer-only runs and, after retries, 3/3 count-writing runs, with correct earlier output tokens supplied.

The task is partly solved: the count-writing lookup is [Verified](../GLOSSARY.md#verifying-a-circuit) in this three-run screen, but it receives escape counts and the winning index from the reference algorithm, so it does not explain how the network finds them. For the answer-only whole-network screen, the formula is Sufficient readout only: changing activity outside its circuit fails. For the learned stage on fixed hand-built machinery, the formulas are Partial because that test is unreported. Longer-list whole-network transfer is unreported and no ten-run final experiment was obtained.

| Fact | Value |
|---|---|
| Task | Given a bare black king and K = 1–16 supplied legal checking moves, output the first move that mates, or NONE |
| Input → output example | In words: if only the second of three supplied checks leaves the king no escape square, the answer is the second |
| Model | HAND/PART: 8 layers, width 480, 24 heads × 20, MLP 640; PART trains only MLP7 (615,520 parameters). TRAIN: 4 layers, width 384 or 768 after the retry, with all weights trained. |
| Rungs and seeds | HAND A/T; PART A/T, 3 seeds each in run 1 and coverage run 2; TRAIN A run 1: 3 seeds; T runs 1, 3 and 4: 3 seeds per setting, two weight decays in run 4. No completed ten-seed final. |
| Closed form status | Verified for count-writing TRAIN only (T run 4 wd .1, 3/3: reference-keyed lookup substitution 1.0 on the 256-row cohort, complement .917/.917/1.0 on 64 rows, meeting .90 × matched native). Answer-only TRAIN (A): Sufficient readout only; complement fails. Fixed-scaffold stage (PART): Partial; complement untested, T fits local only. |
| Status | Partly solved (PART 3/3 per format; A TRAIN 3/3, T TRAIN retry 3/3 at exact 1.0 on 128 sequences); last completed screen 2026-11-10. No ten-seed final result. |
| Weights | [Hugging Face: chess-mate-in-one](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-mate-in-one) (HAND/PART weights and test archive; public follow-up TRAIN weights not confirmed). |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

## The circuit at a glance

Figure S1 shows the frozen escape-count and first-zero stages selecting a supplied check before the learned stage decodes its slot.

<img src="../figures/circuits/chess-mate-in-one.svg" width="640" alt="In the answer-only partly trained network, three supplied checks fan into frozen virtual-board and escape-count stages. Task-example escape counts 2,0,1 enter the frozen earliest-zero selector, which writes slot 2. Trained layer-7 neurons decode that slot into answer 2. Only the decoder is dashed because the complement is untested. The board needs a bare black king, and candidates are supplied, not generated. This is not the later reference-keyed whole-network lookup; counts are illustrative, not a scored board.">

*Figure S1 shows a learned decoder turning the first zero-escape slot into the answer (answer-only partly trained network, run 1, three seeds, best and final snapshots; matches hand-built selection); status is Partial because substitution passes but the complement is untested, short-list training does not transfer reliably to longer lists, and the escape counts illustrate the task rather than a scored board.*

## 1. Task and data

Each position has a bare black king (KQK or KRK type, optionally with one extra white piece) and K = 1–16 supplied legal checking
candidates. The network outputs the first candidate that mates, or NONE, as a 17-way index. This is candidate selection, not move generation. Fresh boards
are drawn for every batch, with an 80/20 split by board hash. An output counts as correct on exact match of the whole
teacher-forced continuation, with the correct earlier output tokens supplied ([glossary](../GLOSSARY.md#task-formats)). Legality is cross-checked with python-chess ([Fiekas, n.d.](#references)).

Natural three-piece lists are short (the enumerated KQK and KRK maxima are 12 and 6 distinct checks), so K 13–16 needs extra pieces.
Run 1 used K ≤ 8 and 128 test sequences. Run 2 (retry rule: data) used K ≤ 16, a balanced index of the first zero count, exposure to
extra pieces and to escape counts 7–8, and 512 test sequences, with 64k / 68k presentations per A / T seed. The best snapshot is
selected on the test set, which doubles as monitor, so a K ≤ 16 stress set of 608 legal sequences is the independent check.

*Table 1. Output formats.*

| Format | Output |
|---|---|
| `A` (answer only) | one 17-way index |
| `T` (scratchpad: intermediate working before the answer; [glossary](../GLOSSARY.md#task-formats)) | `from to \| count ;` per candidate (count = black-king escape squares, 0–8), then `! index` |

## 2. Model

The shared conventions are described in [METHODS.md](../METHODS.md). The three [rungs](../GLOSSARY.md#the-three-rungs) are HAND (every
weight set by formula), PART (frozen hand-built parts with one trained stage) and TRAIN (all weights trained from scratch).

### 2.1 Architecture

All networks are GPT-NeoX-shaped decoders with attention heads that read earlier positions and feed-forward MLP blocks ([glossary](../GLOSSARY.md#standard-machine-learning-terms)). They use ReLU, no LayerNorm, no positional encoding and untied embeddings, with a vocabulary of 928 tokens
and a context of 128. HAND and PART have 8 layers, d_model 480, 24 heads × 20 and MLP width 640; 84 heads and ~1,270–1,290 neurons are
structurally active. The initial TRAIN network has 4 layers, d_model 384, 8 heads × 48 and MLP width 768. The run-4 T retry has 4 layers, d_model 768, 8 heads and MLP width 1536.

### 2.2 Frozen parts and trained part

PART trains only the layer-7 MLP (MLP7; 615,520 parameters), with its input masked to ONE (a constant channel), the phase flags, the
from- and to-square coordinates, ESC (the escape count) and WIN (the first-mate slot). Everything else is frozen at the hand-built
values.

### 2.3 Training

The original PART optimiser was AdamW with learning rate 1e-3, weight decay .1, betas (.9, .98), warmup 1,000 steps, a cosine schedule and gradient
clip 1. Each step used a batch of 8 fresh boards, in [fp32](../GLOSSARY.md#hand-built-machinery), with a 10k-step budget and a stop
once all seeds are exact. Test accuracy was evaluated every 500 steps and checkpoints were saved every 2,500. Training ran on GPU
accelerators, and re-scoring on CPU in fp32; the archived networks re-score to the recorded accuracies.

The later TRAIN screens used TF32 and fresh boards. Run 1 had a 30k budget; T run 3 increased it to 60k, batch 32, and plateau-stopped at 45k. Run 4 changed both width and batch: width 768, MLP 1536, batch 96, budget 60000, minimum 45000, with weight decay .03 and .1. The .1 pack stopped solved at 57000; .03 ended at 60000 without holding exact. Width and data presentations changed together, so this does not isolate width's causal contribution.

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded
accuracies are on Hugging Face at
[huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-mate-in-one](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/chess-mate-in-one).

### 2.5 Archive

The Hugging Face folder holds `hand_{A,T}`; `part_{A,T}_{r1,r2}_{best,final}_s0…2` (the 4 MLP tensors, to be inserted into
`hand_<fmt>`); `test_<run>` (each run's test sequences and scored-position masks); `config.json`; and `manifest.json` (seed, steps,
recorded and fp32 re-scored exact match, sha256). This is the original HAND/PART archive. The later TRAIN evidence is recorded in the task results, but this review did not confirm that those follow-up TRAIN weights were added to the public folder; no downloadable TRAIN replication is claimed.

## 3. Hand-built solution

The hand-built network plays each candidate on a virtual board and counts the king's escapes (Table F1). Source and capture terms
delete the mover's old square, and a virtual destination inserts it. Nearest-live rays and leaper ORs test each square next to the
black king, with the old king square removed. MLP5 sums the escapes into ESC. The MLP6 gate computes MATE = ReLU(u) − ReLU(u − 1) with
u = 2·TO − 2·EQS − 4·ESC − 1. TO marks a candidate destination token; EQS records the retrieved `=` marker, separating the prompt candidates from the answer phase. For binary inputs MATE is 1
exactly when TO = 1, EQS = 0 and ESC = 0.
It feeds an earliest-prefix selector head (layer 7 head 0) that writes WIN, the slot of the first mating candidate.

The network was verified on 339 stratified sequences per format, with K 1–16 and counts 0–8, all exact; the minimum gold logit margin
is 30.0 for A and 29.9 for T. Mate is exact only for a bare black king: with a black piece the count is of escapes only, because a
capture of the checker can be legal at 0 escapes.

Table F1 records the fixed candidate-selection route shown in the worked circuit above. Each supplied candidate creates a virtual board. Ray and leaper checks supply the escape count; a zero-count gate identifies mate, and the selector writes the first winning slot. MLP7 writes the output index, also reading escape counts in the scratchpad format. Only MLP7 is trained in PART.

<details>
<summary>Technical candidate selection and decoder input routes (Table F1)</summary>

*Table F1. The hand-built network. Frozen layers play each supplied candidate, count the black king's escape squares into ESC and gate mate on a zero count; the selector head writes the first mating slot WIN. MLP7 is the only part that PART retrains; upstream counting and selection remain frozen.*

| Component | Input, operation and output |
|---|---|
| Fixed virtual board | Read candidate move and board; delete the old square and insert the destination. Nearest-live rays and leaper ORs check each king-neighbour square. |
| Fixed MLP5 | Sum escapes into ESC, 0–8. |
| Fixed MLP6 | Read ESC; compute the mate gate MATE = ReLU(u) − ReLU(u − 1). |
| Fixed layer 7, head 0 | Read MATE; the earliest-prefix selector writes WIN, the first mating slot. |
| MLP7, trained in PART | Read WIN to write the output index. ESC also has a masked direct input from MLP5. |

</details>

## 4. What training finds

Table 2 gives every cell through the 2026-11-10 follow-up analysis. The first exact step is the first evaluation at which test exact match reached 1.

*Table 2. Results per cell. Test exact match, given for seeds 0 / 1 / 2; stress = sequences exact of the 608 K ≤ 16 stress set.*

| Cell | Solved | Best → final test (s0 / s1 / s2) | First exact step | K ≤ 16 stress (608) |
|---|---|---|---|---|
| PART A run 1 (K ≤ 8) | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 | 4k / 2.5k / 2.5k | 344–388 |
| PART T run 1 (K ≤ 8) | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 | 2.5k / 2.5k / 4k | 317–328 |
| PART A run 2 coverage | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 (512) | 6.5k / 8k / 8k | 608 on all 6 snapshots |
| PART T run 2 coverage | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 (512) | 8–8.5k | not landed |
| TRAIN A run 1 | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 (128) | 14k / 14.5k / 14.5k | not reported |
| TRAIN T run 1, width 384 | 0/3 | .922 / .961 / .938, best = final (128) | never | not reported |
| TRAIN T run 3, width 384 | 0/3 | .961 / .953 / .969 → .961 / .938 / .969 (128) | never | not reported |
| TRAIN T run 4, width 768, wd .1 | 3/3 | 1 / 1 / 1 → 1 / 1 / 1 (128) | 56k / 47k / 51k | not reported |
| TRAIN T run 4, width 768, wd .03 | 0/3 at final exact stop | .992 / 1.0 / 1.0 → .992 / .992 / .992 (128) | transient exact in two seeds only | not reported |

Key: s = seed; wd = weight decay. TRAIN follow-up rows come from the completed screens analysed after the original 2026-11-07 entry. The run-4 wd .03 cell is operationally 0/3: two seeds touched test 1.0 but did not retain it, and all final scores were .992.

With every earlier stage frozen, the trained MLP7 reads HAND's escape count ESC and first-mate slot WIN and decodes the index in
2.5–8.5k steps in both formats. Trained on lists of at most 8 candidates (run 1), the networks are exact on their monitor but only
52–64% exact on the K ≤ 16 stress set; with coverage data (K ≤ 16, balanced first-mate index, escape counts 7–8) the A networks are
608/608 on the same stress set. There is one circuit type per PART cell, 3/3 seeds: frozen count plus learned decoder.

The run-1 failure is one of length coverage: long lists and late first-mate indices were rare in K ≤ 8 training. Softmax over more
attended tokens compresses logit gaps ([Zsámboki et al., 2025](#references)), which is consistent with the small margins at late
indices (Appendix A).

PART A and T both solve 3/3 with similar first exact steps. T keeps more heads causally live (34–43 against 14–17) because its count
tokens keep the escape stages in use. Coverage makes T's use of the count more uniform (interchange .96–.98 against .85–1.00), and the
learned output has a lower rank than HAND's (Appendix B). The [grok step](../GLOSSARY.md#counting-results) and the from-scratch scratchpad versus answer-only comparison were initially unmeasured. The completed screens now show A solving in run 1 while T needed the combined width/data retry; TRAIN longer-list generalisation remains unreported.

## 5. Mechanism and verification

### 5.1 Closed forms

The run-1 closed forms were accepted at the substitution bar (substituted ≥ .99 × native). With tent features
φ_j(v) = (1 − |v − j|)₊ and phase gates g,

MLP7 ≈ g_A Σ_j φ_j(WIN) θ_j + g_COUNT Σ_c φ_c(ESC) η_c + a from/to-gated coordinate lookup + syntax constants.

For A, 442 fitted scalars (a 17-wide basis) replace 615,520 parameters on all 6 run-1 snapshots and keep every prediction, including
the stress set. For T, 7,693 fitted scalars keep 256/256 fresh K = 3 sequences on all 6 run-1 snapshots (and 128/128 K3 / K5 whole-path
for seed 0), but fail stress retention for the final snapshots of seeds 0 and 1, so the T form is accepted locally only. A WIN-only
selector in T was rejected (interchange flip .82). Controls on T seed 0: zeroing the terminal MLP gives 0 exact, and shuffled inputs
give token accuracy .467. The run-2 PART networks have no reported completed substitution test. No
resample ablation of the complement, which scrambles everything outside the claim using other inputs, was reported for these PART forms ([glossary](../GLOSSARY.md#verifying-a-circuit)), so both are supported by substitution only (status Partial).

### 5.2 Wiring

The decoder is not a copy of HAND. In T it also uses a direct route from ESC to MLP7, next to HAND's route from MLP6 through the
selector head to WIN (Table F2). The largest patching drop is at layer 7 head 0 and MLP5 for A (1.0), and at MLP7 for T (1.0), followed
by layer 2 head 3 (.66). JS 0 and the layer-6 count probe come from the frozen parts by construction, not from learned agreement.
Edge-attribution checks fail (Spearman .35–.68 < .8), so edge wiring is not claimed.

Table F2 records the learned decoder's extra direct-count input. The frozen escape count passes through the mate gate and winner selector to MLP7, with an additional direct count route in scratchpad T. The trained MLP uses tent features to write the index or count tokens. Status remains partial: substitution supports answer only and a local scratchpad decoder, with complement untested.

<details>
<summary>Technical learned decoder and direct-count bypass (Table F2)</summary>

*Table F2. The decoder training finds. MLP7 reads WIN through HAND's selector route and, in T, ESC directly. The tent closed form keeps every prediction for A and holds only locally for T; both are supported by substitution with the complement not tested (partial).*

| Decoder input route | Fixed or trained operation |
|---|---|
| Winner-slot route | Fixed MLP5 supplies ESC to the fixed MLP6 MATE gate, then the fixed layer 7, head 0 selector writes WIN. |
| Extra direct-count route in T | ESC reaches MLP7 directly from MLP5, bypassing the mate gate and selector. |
| Trained MLP7 | Phase-gated tent features on WIN and ESC write the index, or count tokens in T. This is the only trained PART stage; the scratchpad replacement holds locally only. |

</details>

### 5.3 Whole-network TRAIN follow-up

The count-writing T TRAIN run-4 networks use a late joint count: L3 heads read board tokens and MLP3 forms the count, with a second path through L2 heads and MLP2. The count tokens can be correct while the count is poorly recoverable by a linear probe (R² ≤ .33); this is evidence against a simple linear count code, not evidence that no count is computed. Winner-index interchange flips are 1.0 at the final layer. A TRAIN instead has a distributed answer-only matcher, with winner flips 1.0 at L3/L4 and count R² .16–.63.

At count positions, the last-MLP lookup is μ[ESC]; at the index it is μ[WIN], where WIN is the first zero-escape candidate or NONE. Here μ is the mean native last-MLP output for that reference-key value. These keys are supplied by the task/reference calculation for the substitution, not established as a linearly decoded internal count. The test used a 256-row substitution cohort and 64 complement rows.

*Table 3. Completed TRAIN follow-up verification, seeds 0 / 1 / 2. The acceptance share is .90 × the matched native exact match.*

| Cell | Native on held substitution rows | Lookup substitute | Complement resample | Verdict |
|---|---|---|---|---|
| T TRAIN run 4, wd .1 | 1.0 / .992 / 1.0 | 1.0 / 1.0 / 1.0 | .917 / .917 / 1.0 | Verified, 3/3 screened seeds. |
| A TRAIN run 1 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | .63 / .56 / .70 | Sufficient readout only. |

An additive fit on the eight individual escape bits gives .22/.31/.37 in T and is rejected. This lookup is per count value, not a linear sum circuit. The first complement claim, used components plus L3 heads, passed only seed 0; seeds 1/2 gave .651/.524 until the full L2 head family and MLP1 were retained. The final claim passed with full_eq_base true, empty_eq_source 1.0, circuit_resampled 0 and random_same_size .00–.38. A's failed complement claim was not replaced by a verified wider claim.

Table F3 records the separate whole-network screen and its externally supplied test keys. Board-token embeddings feed parallel head-family routes into the final joint count-and-winner write. All displayed components were trained. The reference-keyed local lookup is verified only in the screened cohort with the wider retained head and earlier-MLP claim; it does not establish extraction of the supplied keys. Additive escape-bit fits fail, and longer-list transfer is unreported.

<details>
<summary>Technical whole-network screen and external test keys (Table F3)</summary>

*Table F3. Verified local reference-keyed lookup with the widened T TRAIN screen, 3/3 seeds, when the retained L2/L3 head families and earlier MLPs are included. All listed network components are trained. Substitution and complement resampling pass locally; additive escape-bit fits fail, and longer-list transfer is unreported. The supplied test keys do not establish native extraction of the count or winner.*

| Network role or separate intervention input | Recorded scope |
|---|---|
| Trained board-token embedding | Supplies the parallel late-head routes. |
| Parallel trained layer-3 head family and layer-2 heads/MLP2 | Both routes feed MLP3's joint count and winner write in parallel, not as sequential stages of the PART selector. |
| Trained MLP3 | Writes count tokens and the first-mate index. The accepted local lookup uses the wider retained head and earlier-MLP claim. |
| External keys for the lookup test | ESC and WIN come from the task/reference calculation, as described above. The test does not establish extraction of those keys from native activity. |

</details>

## 6. Limitations and open questions

Substitutions are on fixed K3 / K5 / stress sets, not on all positions. Seed counts are screening counts (3;
[glossary](../GLOSSARY.md#counting-results)). Size counts are component-level single ablations, not minimal circuits.

Open questions:

- earlier TRAIN attempts were interrupted at 7k / 4.5k steps; these infrastructure losses are separate from the completed learning results above;
- the later ten-seed final was stopped by the owner before the replacement run produced new snapshots, so there is no ten-seed final result and no implied live training;
- longer-list TRAIN stress results remain unreported, and the run-4 changes confound width with data presentations;
- the K ≤ 16 stress replay of T coverage run 2 has not landed;
- the PART T run-1 closed form is accepted only locally (K3/K5), not at K ≤ 16;
- the small margin to the neighbouring index (Appendix A) is a hypothesis for the late-index stress errors, not yet tested;
- public code is pending.

## 7. Prediction record

The predictions were recorded before the runs. PART would reach 3/3 ≥ .99 by 2,000 steps: solved, but later (2.5–4k in run 1). TRAIN
would reach T ≥ 2/3 and A ≤ 1/3. The cut runs were at A .97 and T .49 and did not settle the prediction. Completed run 1 refuted both TRAIN predictions (T 0/3, A 3/3); T reached 3/3 only after the run-4 retry. The later run-4 prediction of at most 1/6 exact seed/weight-decay rows was also refuted: 3/6 held exact by the stop rule and 5/6 touched it at some evaluation.

## Appendix A. Weights excerpt and reading

The frozen MLP6 on (TO, EQS, ESC) has input rows (2, −2, −4) twice, biases (−1, −2) and output (1, −1), which is the bounded MATE gate.
The fitted T seed-0 response for WIN = 6 on the output directions (YINDEX5, YINDEX6) is (.714, .579) for the best snapshot and (.763,
.734) for the final one, with unembedding scale 30; for A seed 0 the best snapshot gives (.953, .963). The small margin to the
neighbouring index is a hypothesis for the late-index stress errors, not yet tested.

## Appendix B. Detailed evidence

*Table B1. Diagnostics per cell. 64 K = 3 rows per snapshot; ranges over seeds, best and final.*

| Cell | Used H+MLP | count R²; WIN R² | inter count; WIN | r90 / r99 | JS | Best / final |
|---|---|---|---|---|---|---|
| HAND A | 17+5 | 1.00 (6); .983 (8) | n/a; .90 | 11 / 34 | 0 | 1 / 1 |
| PART A run 1 (5 of 6) | 14–17+5 | 1.00 (6); .992–.997 (8) | n/a; .82–1.00 | 9–10 / 31–32 | 0 | 1 / 1 |
| PART A run 2 | 14–17+5 | 1.00 (6); .994–.997 (8) | n/a; .95–1.00 | 8–9 / 30–31 | 0 | 1 / 1 |
| HAND T | 38+5 | 1.00 (6); .979 (8) | 1.00; 1.00 | 20 / 91 | 0 | 1 / 1 |
| PART T run 1 | 34–43+5 | 1.00 (6); .990–.994 (8) | .85–1.00; .80–1.00 | 16–17 / 72–75 | 0 | 1 / 1 |
| PART T run 2 | 36–40+5 | 1.00 (6); .993–.995 (8) | .96–.98; .91–1.00 | 15–17 / 71–74 | 0 | 1 / 1 |

Key: used H+MLP = heads + whole MLPs whose mean ablation drops exact match by > .05; R² = best linear
[probe](../GLOSSARY.md#analysis-measures) (layer); inter = linear interchange flip; r90/r99 = output-residual components for 90% / 99%
variance; JS = attention divergence to HAND.

## What to read next

Read [chess attack maps](chess-attack-maps.md) for the square-threat computation behind escape counting. [KRK mate](chess-krk.md) shows why matching a local mating decision and learning an endgame policy are different tasks.

## References

- Fiekas, N. (n.d.). python-chess: A chess library for Python [Computer software; GPL-3.0; repository since 2012, PyPI package `chess`; version used not recorded]. *GitHub*. https://github.com/niklasf/python-chess (Legality cross-check.)
- Zsámboki, P., Levi, B., Smith, D. A. J., et al. (2025). Post-norm can resharpen attention.
  *arXiv preprint*. https://doi.org/10.48550/arXiv.2510.08341 ; https://arxiv.org/abs/2510.08341 (Softmax over more tokens compresses logit gaps.)
