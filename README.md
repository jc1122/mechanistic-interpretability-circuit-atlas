# Circuit atlas

Explore how small neural networks count, multiply and search. This atlas compares hand-built solutions with what training learned,
keeping the tests and limits beside each explanation.

[<img src="figures/atlas-first-look.svg" width="640" alt="A fixed counter finds seven a's: the count-seven indicator is on (1), the count-two indicator off (0). A learned lookup groups two and seven under answer two, their remainder after division by five. Only the answer stage learns; its mechanism is unverified. Open at full size.">](figures/atlas-first-look.svg)

*Figure 1. [Count a's, then remove groups of five](entries/count-mod-5.md). The main count path is simplified; other supplied features
and individual neurons are omitted. Only the answer stage learns. The formula is partial; outside circuitry was not resampled.*

## Findings worth a look

- [A learned lookup largely ignores line length](entries/count-mod-5.md); the count itself is supplied.
- [Written steps let a small circuit do more](#why-writing-the-working-helps), in these hand-built designs.
- [A correct replacement can still miss the network's mechanism](entries/multiplication.md#53-circuit-probes).
- [An exact snapshot can fail by the end of training](entries/euler-decay-ode.md#4-what-training-finds).

## Start here

1. [Count modulo five](entries/count-mod-5.md). Follow a learned lookup; its test of outside circuitry is still missing.
2. [Binary multiplication](entries/multiplication.md). Writing running sums helps, yet a working readout does not verify the mechanism.
3. [Rounded decay](entries/euler-decay-ode.md). Every final run falls short of exact match on a simple shrinking-number rule.

Browse all [41 entries by topic](INDEX.md), look up a [plain-language term](GLOSSARY.md#first-words), or inspect the
[evidence and limits](#main-findings). For reuse, follow the [code and archive guide](#reuse-and-reproducibility).
This is a living research collection, not a peer-reviewed benchmark or a set of deployment-ready solvers.

## Why writing the working helps

A scratchpad is a written record of intermediate results ([glossary](GLOSSARY.md#task-formats)). In the hand-built division design,
one small circuit compares and subtracts at each step, reading the remainder already written. Without those written steps, the
answer-only design keeps many comparisons inside the network.

Neurons are small computing units; attention heads read information from other positions
([glossary](GLOSSARY.md#standard-machine-learning-terms)). The counts below measure these different parts, not training runs.

<img src="figures/scratchpad_size.svg" width="640" alt="Hand-built long division: an answer-only network keeps many comparisons inside, using 186 neurons. Four selected tests of 7 against multiples of 2 feed the quotient 3 and remainder 1. A scratchpad writes the next bit and remainder, letting the same 14-neuron circuit serve each step. With old remainder 1 and next bit 1, it compares 3 with 2, writes bit 1 and remainder 1, then reads that remainder again. These are design counts, not trained-network sizes.">

*Figure 2. Information flow in the named 8-bit long-division designs. Writing each step allows the compare-and-subtract circuit to
be reused. Four selected comparisons stand for the larger internal bank; the schematic omits other comparisons and bit-level wiring.
The complete [hand-built solution](entries/long-division-square-root.md#3-hand-built-solution) and the comparisons below give its scope.*

[Open the scratchpad diagram at full size](figures/scratchpad_size.svg).

Figure 2 follows division of 7 by 2. Above, tests against multiples of 2 feed an answer of quotient 3 and remainder 1. Below, a written
remainder of 1 and the next input bit 1 give a working value of 3. The arrow labelled “3, 2” carries that value and the divisor;
subtracting 2 writes decision bit 1 and remainder 1. The loop reads that remainder again. Both drawings show formula-set weights,
not what a trained network was shown to use.

*Table 1. Size of the named hand-built designs, with the answer only and with a scratchpad. These are not independent training
seeds, seed averages or a common order-of-magnitude reduction. Neurons and attention heads are different units.*

| Hand-built task | Answer only | With a scratchpad | Counted unit |
|---|---|---|---|
| [8-bit long division](entries/long-division-square-root.md#3-hand-built-solution) | 186 | 14 | Used neurons |
| [8-bit long-hand square root](entries/long-division-square-root.md#3-hand-built-solution) | 46 | 20 | Used neurons |
| [6-bit binary multiplication](entries/multiplication.md#3-hand-built-solution) | 14 | 7 | Attention heads |

A scratchpad can shrink a hand-built circuit (Figure 2). In the 8-bit designs, long division needs 186 neurons when the network
writes the answer only and 14 when it writes the steps, and long-hand square root 46 against 20. These are counts for named
hand-built designs, not seed averages or a common order-of-magnitude reduction. With a scratchpad of running sums, the
stage trained on frozen parts in [binary multiplication](entries/multiplication.md) reuses the adder's column-sum lookup in 10/10 seeds
(a sufficient readout, substitution 1.00), while answer-only multiplication failed in the tested runs. This does not rule out other
training settings, and the running-sum readout is not a verified whole-network mechanism.

That multiplication result uses held-out, whole-output exact match 1.0 on 1639 pairs per seed. Each next token is scored with the
correct previous tokens supplied, called teacher forcing ([glossary](GLOSSARY.md#task-formats)); it is not a test of autonomous rollout.

## How to read the atlas

A circuit is the set of network parts and connections that carry out a computation ([glossary](GLOSSARY.md#first-words)). In the
opening picture (Figure 1), hand-built weights supply the letter count. The count-seven indicator is on and the count-two indicator
is off; the learned answer stage groups those counts by remainder. Its main count path is shown, not every supplied feature. In
[count modulo five](entries/count-mod-5.md), 30/30 seeds reached held-out accuracy ≥ .99. The split held out count/length combinations,
not whole count values. Unseen count values were not tested, and a working formula does not replace the missing test of circuitry
outside it.

For readers, each entry starts with a concrete task and a short evidence summary. For reusers, [five entries have CPU training code](code/README.md);
others offer archived artifacts with entry-specific gaps. [Reuse and reproducibility](#reuse-and-reproducibility) explains the difference
between rerunning training, rescoring a model and checking a file.

Most transformer tasks compare three rungs; entries state missing rungs and exceptions. HAND is the formula-set reference network:
read its entry for the exactness argument, exhaustive or sampled checks and any numerical/design failure. PART trains one stage on
frozen hand-built parts; TRAIN trains the whole network from scratch. A seed count such as 6/10 means that 6 of 10 seeds meet the
cell's criterion, usually held-out exact match ≥ .99. Each entry is a short article with a lede, a fact table, numbered sections, a
figure of the hand-built circuit and an appendix of weights and diagnostics; [METHODS.md](METHODS.md#reading-an-entry) describes the
layout, [GLOSSARY.md](GLOSSARY.md) defines the terms and [STYLE.md](STYLE.md) sets the writing and verification standard. To start,
read [count-mod-5](entries/count-mod-5.md), a complete and reproducible entry, and then scan [INDEX.md](INDEX.md), which gives one row
per entry with its circuit, seed counts and closed-form status.

## Reader routes

| To understand | Start here | Keep in view |
|---|---|---|
| A small learned computation | [Count modulo five](entries/count-mod-5.md) | A correct replacement is not a complement test. |
| What a scaffold contributes | [Polynomial integration](entries/polynomial-integration.md) | Frozen routing is different from learning the whole algorithm. |
| A negative result | [Rounded decay](entries/euler-decay-ode.md) | Test-selected best snapshots need not stay exact. |
| External recurrent models | [Second-largest position](entries/algzoo-2nd-argmax.md) | Sampled certificates and their flags are not all-input theorems. |
| Reuse of a model or result | [Reuse guidance](#reuse-and-reproducibility) | Record entry, model and exact archive revision. |

## Entries

Main result per entry; the criterion, counted unit and mechanism status are in [INDEX.md](INDEX.md). ⟳ means CPU code is available
in [code/](code/README.md), not that this editorial revision independently reran it. Other entries link
[archived networks](https://huggingface.co/jc1122/circuit-atlas-nets) and recorded checks, sometimes with missing rungs, seeds or
splits. Read each entry's archive subsection before reuse.

### Counting and modular arithmetic
- [modular-addition-mlp](entries/modular-addition-mlp.md) ⟳: Fourier clock learned; 0/10 seeds reach 99% held-out accuracy; formula remains partial
- [count-mod-5](entries/count-mod-5.md) ⟳: count-to-remainder lookup; 30/30 seeds reached held-out accuracy ≥ .99
- [compare-counts](entries/compare-counts.md) ⟳: approximate comparison; every tested network fails on unseen ties
- [open-bracket-count](entries/open-bracket-count.md) ⟳: broad cosines plus odd/even features; 0/10 seeds reached .90 held-out accuracy
- [modular-arithmetic-mod-97](entries/modular-arithmetic-mod-97.md): discrete-log circles; output surrogate uses externally supplied logs, leaving the internal mechanism unresolved

### Arithmetic
- [decimal-hex-addition](entries/decimal-hex-addition.md): column-total readouts learned; from-scratch carry mechanism remains unresolved
- [binary-addition](entries/binary-addition.md): column-sum stage transfers when position is masked; from-scratch networks fail longer inputs
- [router-addition](entries/router-addition.md): shared column readout; masking improves length transfer, while native routing remains length-sensitive
- [subtraction](entries/subtraction.md): difference tables reproduce local answers; the proposed whole mechanism fails resampling
- [multiplication](entries/multiplication.md): running-sum readout resembles an adder; full mechanism remains unverified
- [multiplication-base-order](entries/multiplication-base-order.md): units-first products train more readily; running-sum scratchpads solve the screened fixed range
- [long-division-square-root](entries/long-division-square-root.md): shared compare-and-subtract readout; larger-data answer-only runs generalise
- [gcd-euclid](entries/gcd-euclid.md): three-rule router on frozen parts; from-scratch circuits remain distributed and without accepted formulas
- [quadratic-roots](entries/quadratic-roots.md): verified Boolean executor uses supplied register fetches; end-to-end root learning fails
- [compose-arithmetic-registers](entries/compose-arithmetic-registers.md): fixed-layout arithmetic learned; fetching freely named registers remains unsolved
- [integer-matvec](entries/integer-matvec.md): dense scratchpad circuit; 8/10 final seeds exact; answer-only evidence remains screening

### Number formats
- [base-conversion](entries/base-conversion.md) ⟳: digit-code readout; conversions combining digits failed from scratch in these runs
- [output-order-base-conversion](entries/output-order-base-conversion.md): magnitude comparisons train more readily than remainders; rounded stop labels overstate exact success

### Automata and finite fields
- [dfa-group-automata](entries/dfa-group-automata.md): serial state-update chain; final substitution and complement checks cover eight seeds, leaving two untested
- [gf2-matrix-product](entries/gf2-matrix-product.md): running-parity lookup; training moves operand retrieval one layer later; complement remains unreported
- [gf2-elimination](entries/gf2-elimination.md): binary row updates are relearned; diverse matrices yield three fixed-size circuit families
- [linear-systems-finite-fields](entries/linear-systems-finite-fields.md): microstep traces train; reference-keyed readout substitutes, while whole-circuit verification fails
- [crc-remainder](entries/crc-remainder.md): local bit-update readout and circuit pass checks; upstream formula and longer-input success remain absent
- [dyck-k-and-agreement](entries/dyck-k-and-agreement.md): bracket-state pointer needs an unidentified partner; agreement exploits first-noun placement
- [counter-languages](entries/counter-languages.md): ratio counts and prefix violations solve membership; one final circuit fails verification
- [binding-coreference](entries/binding-coreference.md): one head attends to the referent; every tested type-level circuit fails complement resampling

### Symbolic mathematics
- [polynomial-add-mul](entries/polynomial-add-mul.md): coefficient sums learned; multiplication improves with frozen product gates, without exact success
- [polynomial-differentiation](entries/polynomial-differentiation.md): local multiplier learned; final scratchpad failures copy the wrong coefficient
- [polynomial-integration](entries/polynomial-integration.md): distributed integration code; only frozen-scaffold scratchpad is exact on withheld fractions
- [chain-rule-differentiation](entries/chain-rule-differentiation.md): weighted token counts replace the front end; the complete rewrite remains unexplained
- [symbolic-algebra-atoms](entries/symbolic-algebra-atoms.md): categorical expansion and solving codes need intermediate results; expansion loses one exact final seed
- [nested-expressions-mod-7](entries/nested-expressions-mod-7.md): frozen-parser arithmetic verified on sampled checks; from-scratch networks usually fail held-out singletons
- [euler-decay-ode](entries/euler-decay-ode.md): rounded decay unsolved; test-selected exact snapshots do not stay exact

### Search and games
- [grid-bfs](entries/grid-bfs.md): reached-set maps learned from scratch; answer-only failures have no verified depth explanation
- [dpll-search](entries/dpll-search.md): a verified local update completes frozen search; whole-network training fails
- [chess-krk](entries/chess-krk.md): coordinate decoder; 7/7 completed seeds pass .99, none exact; three launches remain unfinished
- [chess-mate-in-one](entries/chess-mate-in-one.md): a late count lookup passes screening checks; long-list transfer and final replication remain incomplete
- [chess-attack-maps](entries/chess-attack-maps.md): writing each ray helps attack-map learning; no tested circuit formula accepted
- [chess-kbbk-mate](entries/chess-kbbk-mate.md): no solved network; candidate-list scratchpads copy a candidate whose generation remains inaccurate

### External model zoos
- [algzoo-2nd-argmax](entries/algzoo-2nd-argmax.md): short-list comparison and angle forms and longer-list readings; after the specificity audit most are read only (16 proven / 257 read / 7 unexplained); optimal-only certificates stay flagged
- [algzoo-median](entries/algzoo-median.md): sampled formulas explain median estimates; after the specificity audit only 18 width-2 empty-complement exceptions stay proven; near-network-sized forms remain flagged
- [algzoo-argmedian](entries/algzoo-argmedian.md): length-3 tables, paired registers and filter banks; 22 proven (20 width-2 exceptions) / 78 read
- [algzoo-longest-cycle](entries/algzoo-longest-cycle.md): no graph algorithm identified in two-layer attention nets; 0 proven / 94 read / 6 unexplained

## Main findings

These detailed results support the [opening stories](#findings-worth-a-look). The catalogue separates what works from what remains
unexplained; every entry states the task, proposed circuit and limits.

The transformer studies compare hand-built weights, one stage trained on frozen hand-built parts, and training from scratch. A closed form is an equation-based replacement for a trained computation. It counts as verified only when both substitution and resampling everything outside the claimed circuit retain at least .90 of the network's accuracy under the entry's stated criterion. Several updated entries now pass those tests at a stated scope; others remain partial or sufficient readouts. The external zoos have their own sampled-certificate accounting, with limitations near the top of each page; “proven” there does not mean an all-input theorem or perfect task performance.

Swapping parts one at a time does not verify a circuit. Earlier single-component substitutions kept accuracy while joint scrubs,
which replace components with states from other inputs matching the proposed variables, failed in five tasks. Those tests remain
failed even where a separately tested successor form now passes. A working replacement alone supports a readout; verification also
requires the complement test and the entry's stated controls.

| Earlier tested hypothesis | Recorded whole-output exact-match score and scope |
|---|---|
| [Group automata](entries/dfa-group-automata.md#53-earlier-circuit-probes) | 71.42% in seed 4 and 1.69% in seed 0; means of 3 donor draws, with native and joint substitution each 100%. |
| [Base conversion](entries/base-conversion.md#5-mechanism-and-verification) | .556 / .923 in seeds 1 / 4 under the selected-nibble-only joint scrub. |
| [Long division](entries/long-division-square-root.md#5-mechanism-and-verification) | 2.9% and 1.8% in answer-only TRAIN seeds 0 and 3 under the whole-role scrub. |
| [Subtraction](entries/subtraction.md#5-mechanism-and-verification) | .938 / .933 in binary borrow-token TRAIN seeds 0 / 3, below the .99 control bar; means of three donor draws on 416 evaluation pairs, also used to select the keep-sets. |
| [Decimal addition](entries/decimal-hex-addition.md#53-why-the-train-algorithm-remains-unresolved) | .74–.80 in seed 1 and .81–.82 in seed 2. The earlier overview's .74–.82 combined display is historical, not a newly pooled range. |

These are selected-seed audits, not failed-seed frequencies. Subtraction used 416 non-training fit pairs and 416 disjoint evaluation
pairs, with the evaluation pairs also used adaptively for keep-set selection; this was not an untouched validation set. Example counts
for the other scrub rows are unavailable in this overview and must not be borrowed from a neighbouring substitution batch.

Trained networks can use the hand-built variables with different wiring. In [binary addition](entries/binary-addition.md) and
[decimal and hexadecimal addition](entries/decimal-hex-addition.md), the earlier overview summarised edge overlap with HAND
(Jaccard) as at most .15 for training from scratch and .78–1.0 for training on frozen parts
([historical summary](SURPRISES.md#same-variables-different-wiring)). These are attribution screens, not causal circuit matches;
heads were not aligned in TRAIN. Each entry's Appendix B gives the cell-specific trusted-network counts and the attribution
exactness criterion, spearman_exact ≥ .8. No common pooled seed denominator is established by these summary ranges.

Some failed networks still use the tested hand-built variables. Swapping a hand-built variable between two inputs flips the
answer in the examined failed networks about as often as in solved ones, in [binary addition](entries/binary-addition.md),
[decimal and hexadecimal addition](entries/decimal-hex-addition.md) and
[long division and square root](entries/long-division-square-root.md).

These and other results that went against expectation are collected, each marked verified or preliminary, in
[SURPRISES.md](SURPRISES.md).

## Introduction

Hand-built transformers are already used as ground truth in interpretability. Tracr compiles programs into transformer weights
([Lindner et al., 2023](REFERENCES.md#lindner2023), preprint), and InterpBench trains models to implement known circuits so that circuit-discovery
methods can be scored ([Gupta et al., 2024](REFERENCES.md#gupta2024), preprint). Both use a known circuit to test interpretability tools. Small
algorithmic tasks are also where trained networks have been reverse-engineered in detail, most clearly the Fourier circuit that
appears when a network groks modular addition ([Power et al., 2022](REFERENCES.md#power2022), preprint;
[Nanda et al., 2023](REFERENCES.md#nanda2023), conference paper).

The atlas turns the question around. For each transformer task it specifies a hand-built reference, with the exactness scope stated in the entry, and asks what gradient descent finds in
its place: whether it reads the hand-built parts when they are frozen and available, whether it rediscovers the same algorithm from
scratch, and how often each solution appears over seeds. Because every hand-built variable is known exactly, a trained network can be
probed, patched and ablated against a reference instead of against a guess.

What is new is the protocol rather than any single task. The same three rungs, usually ten seeds per final cell, circuit types counted per
seed and one acceptance rule are applied across the transformer tasks, and negative results are kept. The acceptance rule follows causal scrubbing
([Chan et al., 2022](REFERENCES.md#chan2022), technical forum post): a formula that reproduces the network when substituted is only a sufficient readout
until the rest of the network has also been shown to be unnecessary ([METHODS §9](METHODS.md#9-verification-and-the-acceptance-rule);
[related work](METHODS.md#related-work)).

The methods also include [estimating accuracy with rank-coordinate RQMC](METHODS.md#10-estimating-a-networks-accuracy-with-rank-coordinate-randomised-quasi-monte-carlo):
a hybrid estimator of a network's accuracy that, on 48 noisy copies of one public AlgZoo network, had .380026 of plain sampling's
mean squared error at equal operation count (not faster in wall time).

## Reuse and reproducibility

The atlas adapts [NeurIPS reporting guidance](https://neurips.cc/public/guides/PaperChecklist) (conference checklist),
[model cards](REFERENCES.md#mitchell2019) (conference paper) and [datasheets](REFERENCES.md#gebru2021) (journal article).
These are reporting goals, not claims of independent replication or formal compliance.

| Reuse route | What is available | What it does not establish |
|---|---|---|
| Retrain | Five entries have CPU commands, dependencies and output descriptions in [code/](code/README.md). | A short smoke run does not reproduce the full seed study or every later diagnostic. |
| Rescore an archived model | Each entry links its task archive and states the published scope. [Training snapshots](https://huggingface.co/jc1122/circuit-atlas-snapshots) are a separate archive. | A folder's existence does not establish complete seed/split coverage. Task-specific tensors are not generic Transformers checkpoints. |
| Check integrity | Manifests record identifiers and hashes; entries describe recorded round trips. | Matching hashes do not establish task accuracy, a mechanism or independent replication. |
| Extend a study | Use the task/data/model description and limitations in the entry, then record any changed conditions. | A sampled check does not prove an all-input algorithm or justify pooling incompatible cohorts. |

Record the atlas commit, entry date, exact archive revision, task/cell, seed and best/final or step identifier. Check the data split,
evaluation criterion, precision and loading layout before comparing results. Teacher-forced token scores do not establish autonomous
rollout. Synthetic templates do not establish natural-language competence. Not all training code, controls, seeds or snapshots are
public; unavailable items must remain explicit rather than being inferred from another task. The [writing standard](STYLE.md#publication-and-reuse)
sets the reporting checks; material corrections belong in [CHANGELOG](CHANGELOG). Report a broken link or claim with the entry,
revision and relevant public evidence through the repository's issue tracker; never include private data or credentials.

## Quick start

```
cd code && pip install -r requirements.txt
python run.py check                                                         # the hand-built model and the menus are exact
python run.py atlas --task count_mod5 --seeds 0 --steps 400 --every 100    # smoke run, about a minute
```

The full commands per entry are in [code/README.md](code/README.md).

## Status

The atlas is work in progress. The entries include published updates through 2026-11-13. The two external-zoo entries are dated snapshots, not automatically updated counts: second-argmax ends at analysis iteration 61 and median at iteration 22, both with the 2026-11-14 specificity audit applied. Each entry states its own last run, evaluation criterion, archive coverage and unresolved tests. Five entries rerun from `code/`; the others link archived networks and recorded checks, with code not yet ported. How the atlas was made (one author, with LLM agents working under a written protocol) is described in [METHODS.md](METHODS.md#how-this-was-made).

The earlier UX snapshot covered 31 transformer tasks, with three called solved, 27 partly solved and one unsolved, and last runs dated 2026-11-04 to 2026-11-08. It reported no entry meeting the full verification rule. Those are historical snapshot claims, not totals for the current 41-entry catalogue; later entries add accepted forms, explicit negative results and external zoos. Current verdicts and their scopes are listed per entry in [INDEX.md](INDEX.md).

## Cite and license

The full collection name is Mechanistic Interpretability Circuit Atlas. Cite the atlas via [CITATION.cff](CITATION.cff) (GitHub: "Cite this repository"), and name the task entry and exact revisions used.
Archived on Zenodo: [10.5281/zenodo.24209450](https://doi.org/10.5281/zenodo.24209450) (all versions) and
[10.5281/zenodo.24209451](https://doi.org/10.5281/zenodo.24209451) (version 1.0.0). Do not invent a paper citation. The citation file's release metadata is not a last-run date for every entry; publication
changes are tracked in [CHANGELOG](CHANGELOG).

Original atlas text, diagrams and first-party networks use [CC BY 4.0](LICENSE); the code in `code/` uses [MIT](code/LICENSE).
For CC BY reuse, give credit, link the [licence](https://creativecommons.org/licenses/by/4.0/) (licence summary) and indicate changes.
Third-party models, code and other linked material retain their own licences and attribution requirements; the atlas does not
relicense them. Keep the original authors, public source links and source types when adapting an entry.
The AlgZoo model definitions follow [ARC's alg-zoo](https://github.com/alignment-research-center/alg-zoo) (code repository, MIT-0);
other AlgZoo interpretation repositories are cited, not redistributed. Each tagged release is archived on Zenodo with a DOI; cite
the release you used and the Hugging Face revision of any network you load.
