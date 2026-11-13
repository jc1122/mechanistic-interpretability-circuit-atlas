# Nested expressions modulo seven

For (2+6), keeping the remainder after division by 7 gives 1. A hand-built parser keeps a stack, a last-in-first-out list: it reduces ready `2+6` entries to `1`, then removes the brackets. Only the arithmetic block of a four-layer network was trained.

All 10 final runs got 260/260 held-out working traces right, with correct preceding tokens supplied. Replacing the block with exact arithmetic kept that score. Its internal values differed from the hand-built version: only the largest answer score had to be right.

The final frozen-parser experiment is solved. Its [formula is verified](../GLOSSARY.md#verifying-a-circuit) within the supplied parser: swapping the other parts' signals between examples also passed on 32 diagnostic rows. There was no untouched test, and each formula stress set had only 16 expressions. Training entirely from scratch usually failed even held-out single numbers, so this does not show that a network learned the parser.

| Fact | Value |
|---|---|
| Task | Evaluate a fully parenthesised expression using addition and multiplication, keeping the remainder modulo 7 |
| Input → output example | The expression (2 + 6) has residue 1 modulo 7 |
| Model | 4-layer transformer, width 256, 8 heads per layer, MLP width 192; PART trains only 30 arithmetic neurons and their residue-output bias |
| Rungs and seeds | HAND scratchpad; PART scratchpad 3 screening and 10 final seeds; TRAIN scratchpad and answer-only screens with 3 seeds each, answer-only retried |
| Closed form status | Verified (trained block only: substitution 260/260 per seed; complement resample exact 1.0 on 32 rows, all 10 final seeds; frozen parser, no untouched test) |
| Status | Solved (final PART scratchpad, 10/10 at whole-trace exact 1.0); TRAIN screens 0/3 each; last run 2026-11-10 |
| Weights | [Hugging Face: nested-expressions-mod-7](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/nested-expressions-mod-7) |
| Reproduce | Not available yet: training code for this task is not yet in `code/` |

Read next: [Results](#4-what-training-finds) and [verification](#5-mechanism-and-verification), then [symbol-level algebra](symbolic-algebra-atoms.md) for structured inputs, or [finite-field linear systems](linear-systems-finite-fields.md) for written arithmetic microsteps.

## The circuit at a glance

Figure 3 shows the local arithmetic block inside the supplied parser ([mechanism and verification](#5-mechanism-and-verification)).

<img src="../figures/circuits/nested-expressions-mod-7.svg" width="640" alt="The parser writes the winning remainder after division by 7. One upward residual stream has separate read/write branches for Read top stack (layer 2 heads); Shift / copy (other parser actions); Score residues (trained arithmetic). Given stack: 2 + 6; arithmetic is modulo 7; Next stack: 1. Task values are illustrative; dashed boxes and branches are not verified.">

*Figure 3. Final partly trained seeds 0–9 keep the hand-built parser but use a different residue-score code whose winning remainder after division by 7 feeds its frozen stack writer; the local block is verified, not a learned parser, with no untouched test and small diagnostic/stress sets.*

## 1. Task and data

Modulo 7 means taking the remainder after division by 7. Expressions contain literals 0…6 and fully parenthesised addition or multiplication: `E := 0|…|6|(E+E)|(E*E)`. For example, (2 + 6) has residue 1. Each input token couples a symbol with its index, so position is part of the token identity.

The parser keeps a stack of unfinished subexpressions. It shifts an input symbol onto the stack or reduces a ready `a op b` to its residue; a ready `( v )` reduces to v. The scratchpad writes the whole parser state after every action. Its clock counts actions, its cursor identifies the next input symbol, its height counts stack entries and its origin sets the layout offset.

*Table 1. Output formats. A scratchpad is a written sequence of intermediate states before the answer ([glossary](../GLOSSARY.md#task-formats)).*

| Format | Output |
|---|---|
| `T`, scratchpad | Clock, cursor, height, origin and all 13 stack cells after every shift/reduce action; then `= value EOS` |
| `A`, answer only | `= value EOS` |

Training used 0–4 operators, depth ≤ 3 and fresh expressions every batch. A content hash held out approximately 20%; singleton expressions, consisting of one literal and no operators, used the rule `(literal+seed)%5==0`. Each seed had 260 held-out rows. Longer or deeper tests covered depth 4 with n = 4 operators, n = 5 or 7, and combinations of length and depth. Substitution stress sets contain 16 rows per setting; the trainer's other extension sets contain 16–32 rows as specified in the limitations.

The score requires every teacher-forced generated token to be exact: correct preceding tokens are supplied during scoring ([glossary](../GLOSSARY.md#task-formats)). The scratchpad answer alone copies the gold preceding state and is not a task metric. A solved seed has complete-trace exact accuracy 1.0. Validation rows were reused for best-snapshot selection, so there is no untouched test.

## 2. Model

The [shared methods](https://github.com/jc1122/mechanistic-interpretability-circuit-atlas/blob/main/METHODS.md) describe the rungs: HAND has every weight set by formula; PART freezes HAND except the arithmetic block; TRAIN learns all weights from scratch ([glossary](../GLOSSARY.md#the-three-rungs)). Entry layout follows the [atlas TEMPLATE](https://github.com/jc1122/mechanistic-interpretability-circuit-atlas/blob/main/TEMPLATE.md), a project specification.

### 2.1 Architecture

The GPT-NeoX-shaped decoder has 4 layers, d_model 256, 8 heads per layer with d_head 32, a ReLU MLP of width 192, vocabulary 595, context length 1024, no LayerNorm, no positional embedding and an untied output. It predicts the next token from earlier tokens. Here d_model is the running-state width and d_head each head's vector width. An MLP is a feed-forward block at each position, and an attention head reads information from earlier positions. ReLU keeps positive responses and sets negative ones to zero. LayerNorm would centre and rescale the state. No position vectors are added; the indexed input-token identities still carry position. Untied output means separate input and output weights ([glossary](../GLOSSARY.md#standard-machine-learning-terms)).

### 2.2 Frozen parts and trained part

PART uses HAND T with only L2 neurons 0–29 and the AR output bias trained. The first matrix's rows 0–29 are restricted to 16 inputs: seven residue channels for each of T1 and T3, plus the two operator channels of T2 (`+` and `*`). T1, T2 and T3 hold the three fetched stack entries. The output matrix has [7 AR rows, 30] plus AR bias, for 727 trained parameters. AR holds seven competing scores, one for each result residue. TRAIN learns all weights; no answer-only HAND was claimed.

### 2.3 Training

AdamW used learning rate 1e-3, weight decay .1, betas (.9, .98), warmup 1000, cosine decay and clipping at 1. Weight decay .1 was the only setting run; its final PART success rate was 10/10. Batch size was 256 fresh draws. Scratchpad batches were accumulated in 32 chunks of 8. Evaluation was every 500 steps, checkpoints every 2500, with a plateau window of 5000 after 75% of the budget. The usual budget was 30k, with a solved stop. The answer-only retry doubled the budget to 60k and stopped at 45k.

Screening used seeds 0–2; final PART used seeds 0–9. PART ran in fp32 and TRAIN in TF32, on GPU-class hardware ([glossary](../GLOSSARY.md#hand-built-machinery)).

### 2.4 Reproduction

Reproduce: not available yet: training code for this task is not yet in `code/`; the weights, held-out splits and recorded accuracies are on Hugging Face at [nested-expressions-mod-7](https://huggingface.co/jc1122/circuit-atlas-nets/tree/main/nested-expressions-mod-7).

### 2.5 Archive

Published revision `b293821` contains `hand_T`, PART's 4 trained tensors for screening seeds 0–2 and final seeds 0–9, full TRAIN T/A networks for seeds 0–2, best and final snapshots, `config.json` and a manifest with seed, rung, accuracy and sha256. The publication record reports 38 trained networks plus HAND, fp32 re-score equal to recorded accuracy in 38/38, and matching hashes in 39/39. The checked PART final seed 6 reproduced 1.0; TRAIN A retry final seed 1 reproduced .9962, a separately recorded precision from the .996 summary below. These are earlier publication checks, not new model evaluations. Acceptance checks below used the archived final PART blocks, not only training-log summaries.

## 3. Hand-built solution

The reference uses an eager shift/reduce parser: reduce a ready operation or bracketed value whenever possible; otherwise shift the next input symbol. L0 computes clock and size, and L1H0 reads the preceding counters. L2H0–2 fetch the top three stack cells, L2H3 reads the next input symbol and L2H4 fetches a copied cell. MLP2 performs the local arithmetic and sets routing flags; MLP3 writes the next line.

The 30 arithmetic neurons use triangular hinges on a+b for addition, and on log₃a+log₃b for multiplication, with a zero branch. The discrete logarithm turns nonzero multiplication into addition of exponents ([glossary](../GLOSSARY.md#analysis-measures)); zero is handled separately. HAND's 15 addition and 15 multiplication hinges each read one operator. Its full size is 8 heads and 216 neurons.

The test harness scored 1845/1845 exact with minimum logit margin 30. Coverage was exhaustive for at most 1 operator at each of 13 origins, and stratified up to n = 7 and depth 4; it was not exhaustive over every larger expression.

Table F1 records the arithmetic and copy routes into the fixed stack writer. Learning is confined to local arithmetic; the parser and its addressing are supplied.

<details>
<summary>Technical supplied parser, arithmetic block and copy bypass (Table F1)</summary>

*Table F1. The hand-built stack parser with a trained arithmetic block. The block's closed form is verified within the frozen-parser setting: substitution passed on 260 held-out expressions per seed and complement resampling on 32 diagnostic rows, in all ten final seeds. Validation rows were reused for selection; there is no untouched test, and substitution stress sets contain only 16 expressions each.*

| Part and training status | Input, role and destination |
|---|---|
| Expression tokens and written stack state | Supply the current expression and parser state. |
| Layer 1, head 0 (L1H0); frozen | Reads parser counters that direct stack retrieval. |
| Layer 2, heads 0–2 (L2H0–2); frozen | Fetch the top three stack entries for the local operation. |
| Layer-2 neurons 0–29; trained | Score the arithmetic result modulo 7, the remainder after division by 7. Only this arithmetic block is learned. |
| Layer 2, heads 3–4 (L2H3–4); frozen | Fetch the next input or a copied stack cell directly for the writer, bypassing arithmetic on copy/shift actions. |
| Layer-3 feed-forward block (MLP3); frozen | Combines arithmetic and copy inputs to write the next stack state, complete trace and final residue. Local verification does not verify a learned parser. |

</details>

## 4. What training finds

The final PART cell was solved in 10/10 seeds on 2026-11-10, with the shared stop at 5.5k steps. The learned arithmetic block gave the correct largest residue score in every one of its 98 local input cases. It did not reproduce HAND's numerical one-hot output, a vector with only one active channel ([glossary](../GLOSSARY.md#standard-machine-learning-terms)). The frozen parser needed only the identity of the largest channel, called argmax, so a different score code still gave the correct trace.

*Table 2. Whole-output exact accuracy. Best and final are the archived evaluation snapshots; the answer-only retry's separate log maxima are reported below. n counts operators; OOD means longer or deeper than training.*

| Cell | Held-out exact best / final | n ≥ 1 strata | OOD depth4 / n5 / n7 (final) | Circuit type (seeds) |
|---|---|---|---|---|
| PART T final (seeds 0–9) | 1.0 ×10 / 1.0 ×10 (stop 5500; first exact 2000–5500, median 2500) | 1.0 | 1.0 ×10 each | HAND-a parser + trained argmax residue block (10/10) |
| PART T screening (seeds 0–2) | 1/1/1 / 1/1/1 (stop 3500) | 1.0 | 1/1/1 each | Same (3/3) |
| TRAIN T | .800/.800/.808 / .800/.800/.796 | ~1.0 (n=1 seed 2: 1 row wrong) | 0 each | Dense scratchpad parser, no HAND match (3/3) |
| TRAIN A retry | .800/.992/.800 / .800/.996/.800 | 208/208, 207/208, 208/208 | .31-.56 / .22-.44 / .19-.31 | Dense value code read by MLP3 (3/3) |

Singleton performance needs its own account. In scratchpad TRAIN, all 52/260 held-out singleton rows failed in every seed; seed 2 also made one n = 1 error. In answer-only TRAIN retry, seeds 0 and 2 scored 0/52 on singletons and 208/208 on n = 1–4; seed 1 scored 52/52 on singletons but 207/208 elsewhere, including 51/52 at n = 4. Thus neither “both formats fail all singletons” nor “every non-singleton trace is exact” describes the evidence.

The answer-only retry's training-log maxima were .881/.996/.800, whereas the saved best-snapshot evaluation in Table 2 was .800/.992/.800. In particular, seed 1's saved best had 50/52 at n = 4, versus 51/52 at final. These are separate recorded snapshot/evaluation records, not a reason to select one silently.

Scratchpad TRAIN failed all longer/deeper complete-trace tests. Errors were in unseen counter roles, including stack height, clock and cursor; height probe R² was .75–.77 against .99 in the frozen reference. This does not establish an arithmetic failure in the local operation block. No sparse faithful circuit was recovered for either wholly trained format.

## 5. Mechanism and verification

The trained PART block has the description

`AR(a, op, b) = α·e_{(a op b) mod 7} + β + ε`,

where a is the largest T1 residue channel, b the largest T3 channel and op the largest T2 operator channel. The vector e selects the correct residue, α and β are fitted scale and offset, and ε is the remaining error. The downstream write reads the largest AR channel. This is an argmax description on the tested local inputs, not a claim that ε is small in value.

Replacing the block with `onehot((a op b) mod 7)`, using α = 1 and β = 0 with 0 fitted numbers, kept 260/260 held-out expressions and 16/16 in each depth4/n5/n7 substitution set, in all 10 final seeds. Fitting α and β gave the same scores. Removing the block left 52/260, only the singleton rows. These substitutions cover the arithmetic block, with the parser still frozen.

Resampling everything outside the claimed 22-edge circuit kept exact 1.0 of base 1.0 in 10/10 final seeds, on 32 IID rows with three draws. Resampling the circuit itself, a random same-size set or setting it to zero gave 0; the self-checks passed. A mean replacement retained 1.0 and is not the deciding resample control. Both retention shares were 1.0 ≥ .90, meeting the atlas [verification rule](../GLOSSARY.md#verifying-a-circuit) for the trained block in the frozen-parser setting. This does not verify an independently learned parser, and there is no untouched test.

Path patching, which corrupts only selected connections, broke exact to 0 in every seed for either L2H0–2 to MLP2 (operand fetch) or MLP2 to MLP3 (residue write), despite their small log-probability shares of approximately .05 / .22. Screening had the same accepted block in 3/3 seeds. Table F2 explains why a weak one-hot fit can coexist with exact arithmetic decisions.

Table F2 records why the winning score matters more than matching every score value. Different wrong-channel values can coexist with the correct local arithmetic decision.

<details>
<summary>Technical largest-score residue code and local verification (Table F2)</summary>

*Table F2. The learned argmax residue code, where argmax means the largest-score choice. Only the layer-2 block was trained. Its closed form is verified for the trained block within the frozen parser: all 98 local cases have the correct largest channel in every final seed, substitution passed on 260 expressions and complement resampling on 32 rows. There is no untouched test; the stress sets contain 16 expressions each. Different wrong-channel values on different inputs do not contradict correct largest-channel decisions on each input.*

| Part or decision | Role and scope |
|---|---|
| Fetched residues a and b, and operator | Supplied by the frozen stack readers. |
| Layer-2 arithmetic block; trained | Writes seven AR residue scores. The largest score names (a op b) mod 7; exact one-hot values are not required. |
| Largest-score selection | Identifies the local result even when the non-winning scores vary across inputs. |
| Layer-3 feed-forward block (MLP3); frozen | Writes that selected residue into the stack. |
| Verification boundary | The trained block is verified within the supplied parser only; validation was reused, the diagnostic and stress sets are small, and there is no untouched test. |

</details>

PART retains HAND-a's 22-edge parser circuit while changing its arithmetic code; it does not use a Fourier clock. TRAIN T has a dense scratchpad parser with HAND edge Jaccard .07, and its counters do not extend to unseen values. TRAIN A has a last-layer value code that beats T on the OOD sets but remains below .6. These dense, non-unique circuit observations are compared with [Chen et al. (2026)](#references); no failed discovery search proves that no other faithful circuit could be found.

Scratchpad supervision was studied by [Nye et al. (2021)](#references), and task-based position coupling by [Cho et al. (2024)](#references). [Jeong et al. (2026)](#references) studied held-out concept combinations in text-to-image diffusion. That is a data-design analogy, not direct evidence for the arithmetic parser; the singleton diagnosis here comes from the explicit split and per-stratum counts.

## 6. Limitations and open questions

Validation rows were reused for best-snapshot selection. There is no untouched test. The held-out singleton stratum is a split artefact and is reported separately; success on most other strata must not erase its failures or the documented exceptions.

Substitution covers the trained arithmetic block only; parser exactness comes from HAND. Complement and path tests used 32 IID rows, with three resample draws for the acceptance test. OOD sets contain only 16–32 rows per seed, with 16 per substitution set. The final-run permutation-only linear-mode-connectivity barrier was .03, consistent with one basin under the shared frozen parser ([glossary](../GLOSSARY.md#analysis-measures)); it does not establish end-to-end universality. Cyclic layout rotation is weaker than arbitrary permutation.

The only weight decay tested was .1. Dense TRAIN discovery sets were not faithful; untested operand interchanges with nf 0 do not establish that the operands are unused. The answer-only best-selection records and log maxima differ, as reported in Section 4.

## 7. Prediction record

The advance prediction was that PART would reach ≥ .99 in every seed. It held for screening 3/3 and final 10/10. The within-5k prediction held for screening at 3500 but missed in 1/10 final seeds: seed 6 first became exact at 5500, with median 2500. Depth4/n4 ≥ .99 held at 1.0 ×10.

The scratchpad TRAIN prediction of ≥ 2/3 solved failed on the full-trace rule (.80 due mainly to the singleton stratum); non-singleton performance was near exact, with the documented seed-2 exception. Answer-only TRAIN was predicted at ≤ 3/10 and worse than T; the observed screen was 0/3, and T was better in range but A was better on OOD. The mechanism prediction that PART would read two fetched residues and the operator held; the alternative of a low-frequency code did not. The learned code is an argmax table rather than a Fourier representation.

## Appendix A. Weights excerpt and reading

The archived trained block was evaluated on all 98 one-hot local (a, op, b) inputs. Fits use α = mean gold score minus mean other score and β = mean other score. Table A1 reports the final ten seeds; these are numerical score fits, not task accuracies.

*Table A1. Final arithmetic-block fits. “Gold min” is the smallest correct-channel score across cases and “other max” the largest incorrect-channel score across cases. They need not come from the same input.*

| PART final | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Seed 6 | Seed 7 | Seed 8 | Seed 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| α / β | .96/.13 | .95/.21 | .98/.16 | .64/.16 | .98/.20 | .91/.13 | .28/.39 | .91/.19 | .97/.14 | .91/.17 |
| Gold min / other max | .68/1.10 | .58/1.28 | .72/1.00 | .51/1.01 | .68/1.04 | .71/.90 | .42/.77 | .71/.95 | .69/1.02 | .71/.98 |
| R² vs affine one-hot | .19 | .13 | .26 | .23 | .18 | .30 | .24 | .28 | .17 | .29 |

Every seed gives the correct argmax in 98/98 cases, but HAND's threshold condition, correct channel > 0 and all other channels ≤ 0, holds in only 0–3 of 98. All 30 neurons are live. At |W_in| > .05 on each operator input, 12–25 of 30 read both `+` and `*` (screening 12/13/22; final 13–25), whereas HAND separates its 15 addition and 15 multiplication hinges. Overlap between the pooled correct/incorrect score ranges does not invalidate per-input argmax. The HAND router slice is `W_in[AR_v, AT, RED] = (1, 32, 32)`, with bias −64; AT marks the stack slot receiving a three-cell reduction, and RED marks a ready arithmetic reduction.

## Appendix B. Detailed evidence

Used components have mean-ablation drop > .05. JS is Jensen–Shannon attention divergence to the best-matching HAND head. R² is a best-layer linear probe, r99 the residual dimension count at 99% variance, and HAND-PC R² the fit to HAND's principal components ([glossary](../GLOSSARY.md#analysis-measures)).

*Table B1. Per-cell diagnostics. Layer labels describe residual depth; exact scores are diagnostic scores where separately stated, not the complete training-test maximum.*

| Cell | Used heads + neurons | Best probe R² per HAND variable (layer) | Residual r99 L0–L4 / attention JS | Causal evidence | Best / final |
|---|---|---|---|---|---|
| HAND T | 8 heads + 216 neurons | By construction | — | — | 1 / 1 |
| PART final seeds 0–9 | L1H0, L2H0–4 + MLP2/3 ×10 (as HAND) | cursor/clock/origin ~1.0 L2, height .99 L2–4, op .63–.75 L3, action .71–.73 L3, a .31–.42, b .29–.47 L3, value .49–.59 L4 | Frozen parser (HAND-PC R² 1) / 0 on HAND heads | discover = HAND's 22 edges ×10, faith 1.0; value inter .57–.93 L4 | 1 / 1 |
| TRAIN T seeds 0–2 | 25–28 heads + all MLPs (587–594 neurons) | cursor .96–.98, clock .97–.99, height .75–.77 (L4), value .40–.59 L4; a/b/op < 0 | 72–77/27–36/29–37/46–60/81–94 / ≥ .56 every head | discover 191–242 edges, faith .25–.65; DAS cursor/height 1.0 L4 | .80 / .80 |
| TRAIN A seeds 0–2 | 10–13 heads + all MLPs (547–566 neurons) | value −.02/.61/.17 L4 (no A HAND) | 1–2/10–15/15–26/25–36/31–34 / n/a | value inter .60–1.0 L4; MLP3 recovery .96–.97; discover 218–273, faith .50–.56 | See Table 2 |

The operand labels a and b and operator op refer to the fetched arithmetic inputs; action is the shift/reduce choice. A failed linear probe does not prove absence of a nonlinear code. TRAIN dictionary/discovery results and incomplete failing buckets do not license a more specific mechanism than the evidence reports.

## References

- Chen, X., Jin, M., Niu, J., et al. (2026). All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs. *arXiv preprint*. https://doi.org/10.48550/arxiv.2605.12671 (non-unique low-overlap circuit explanations)
- Cho, H., Cha, J., Awasthi, P., Bhojanapalli, S., Gupta, A., & Yun, C. (2024). Position coupling: Improving length generalization of arithmetic transformers using task structure. *arXiv preprint*. https://arxiv.org/abs/2405.20671 (task-structured position identities)
- Jeong, Y., Uselis, A., Laina, I., Oh, S. J., & Rohrbach, A. (2026). When Do Diffusion Models learn to Generate Multiple Objects? *arXiv preprint*. https://doi.org/10.48550/arxiv.2605.00273 (held-out-concept data-design analogy in image diffusion, not a parser result)
- Nye, M., Andreassen, A. J., Gur-Ari, G., et al. (2021). Show your work: Scratchpads for intermediate computation with language models. *arXiv preprint*. https://arxiv.org/abs/2112.00114 (scratchpad supervision)
