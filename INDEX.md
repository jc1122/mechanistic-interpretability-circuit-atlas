# Find an entry

Choose a task you recognise, then follow how the network computes and where the explanation stops. New to the atlas? Take the
[three-entry tour](README.md#start-here), or read the [first words](GLOSSARY.md#first-words).

## Browse by topic

- [Counting and modular arithmetic](#counting-and-modular-arithmetic): count letters, compare totals and wrap numbers around a fixed range.
- [Arithmetic](#arithmetic): add, subtract, multiply, divide and work with stored numbers.
- [Number formats](#number-formats): write the same number with different digits or in a different order.
- [Automata and finite fields](#automata-and-finite-fields): track a changing state, match brackets and solve small binary systems.
- [Symbolic mathematics](#symbolic-mathematics): manipulate expressions, differentiate, integrate and repeat a rounded decay step.
- [Search and games](#search-and-games): explore mazes, search for satisfying assignments and choose chess moves.
- [External model zoos](#external-model-zoos): inspect recurrent networks trained elsewhere to compare numbers or estimate a median.

## What the evidence columns mean

One row per entry, grouped as in the [README](README.md#entries). The atlas has 41 entries. Each linked article gives the task,
evaluation examples, criterion, archive coverage and limitations. PART trains a stage on frozen hand-built parts; TRAIN trains the
whole network from scratch ([rungs](GLOSSARY.md#the-three-rungs)). A seed is one training run; best/final snapshots are saved states,
not independent runs. A mechanism-family count is not a solved-seed count. *(Screen)* marks preliminary rather than final evidence.

In Closed form, ✓ means **verified at the named scope**: substitution and complement resampling pass under the
[acceptance rule](METHODS.md#9-verification-and-the-acceptance-rule). Partial means limited coverage or an untested requirement;
readout only means a replacement works but its mechanism claim fails the audit; ✗ means the tested form was rejected; — means no
form was tested or recovered, as stated. A flag or qualification is part of the claim. None of these is an all-input theorem.

In Reproduce, code means published CPU commands exist in [code/](code/README.md), not that this catalogue update reran them. Weights
means [archived artifacts](https://huggingface.co/jc1122/circuit-atlas-nets) and recorded checks, with entry-specific gaps; it does not promise every seed, rung or split. Model loading
is task-specific. See [reuse guidance](README.md#reuse-and-reproducibility), [glossary](GLOSSARY.md) and [corrections](CHANGELOG).

## Counting and modular arithmetic

| Entry | Task | Circuit found | Runs and criterion | Closed form | Reproduce |
|---|---|---|---|---|---|
| [modular-addition-mlp](entries/modular-addition-mlp.md) | Add two residues modulo 59 | Fourier clock using one-hot inputs and threshold encodings | clock 10/10 seeds; held-out .91–.97; 0/10 reaches 99% | partial: substitution; complement not tested | code |
| [count-mod-5](entries/count-mod-5.md) | Letter count modulo five | Count-to-remainder lookup; length ignored | PART 30/30 seeds ≥ .99 held-out | partial: 1.000 on 1080 all cells; no complement | code |
| [compare-counts](entries/compare-counts.md) | Name the more frequent letter | Tilted comparator; equality memorised rather than transferred | comparator family 30/30 networks; fraction .8: 10 seeds, held-out .917–.963; zero correct held-out ties | partial: approximate two-way comparator | code |
| [open-bracket-count](entries/open-bracket-count.md) | Count opening brackets | Broad cosines plus odd/even features; low frequencies most common | PART 0/10 ≥ .90 held-out at fraction .8 | partial: training-inclusive approximation; no complement | code |
| [modular-arithmetic-mod-97](entries/modular-arithmetic-mod-97.md) | Arithmetic modulo 97 in different formats | Token discrete-log circles; digit formats form no circles | TRAIN multiplication 6/10 ≥ .99; addition 10/10 ≥ .99; readout: 8 seeds with native ≥ .98 | partial: supplied logs; 9216 training-inclusive pairs | weights |

## Arithmetic

| Entry | Task | Circuit found | Runs and criterion | Closed form | Reproduce |
|---|---|---|---|---|---|
| [decimal-hex-addition](entries/decimal-hex-addition.md) | Three-digit decimal and hexadecimal addition | Column-total readout; carry computation unresolved from scratch | TRAIN retry 37/40 ≥ .99; readout family 60/60 | partial: local stage; complement not tested | weights |
| [binary-addition](entries/binary-addition.md) | Add two six-bit numbers | Column-sum stage; positional shortcuts hinder longer inputs | TRAIN reported 10/10 per format after retries; 12-bit fails | partial: local stage; complement not tested | weights |
| [router-addition](entries/router-addition.md) | Add labelled digit columns | Shared column readout; masking improves longer-input transfer | masked PART 10/10 exact; unmasked native routing remains length-sensitive | partial: addition lacks complement; subtraction final fails | weights |
| [subtraction](entries/subtraction.md) | Subtract decimal or binary numbers | Signed-difference readouts; complete mechanism fails resampling | PART readout counts 10/10, 9/10 or 8/10 by cell | readout only: whole hypothesis fails | weights |
| [multiplication](entries/multiplication.md) | Multiply binary integers | Adder-step readout; native context dependence remains unresolved | PART 10/10 exact; TRAIN retry 8/10 recorded 1.0, 9/10 ≥ .99 | readout only: full audit not passed | weights |
| [multiplication-base-order](entries/multiplication-base-order.md) | Multiply below 256 in three bases | Units-first outputs and running-sum scratchpads learn more readily | scratchpad 18/18 best ≥ .99 on 2000 pairs | partial: outgoing digit votes; no complement | weights |
| [long-division-square-root](entries/long-division-square-root.md) | Binary division and integer square root | Shared compare/subtract substitute; trained comparator code unresolved | division trace PART 10/10 exact; answer TRAIN 10/10 ≥ .99 | partial: substitute can repair native errors | weights |
| [gcd-euclid](entries/gcd-euclid.md) | Greatest common divisor by Euclid's algorithm | Three-rule router on frozen parts; distributed from-scratch circuits | PART 10/10 ≥ .99, 7/10 exact; TRAIN 5/10 exact | partial: PART complement on seven exact seeds | weights |
| [quadratic-roots](entries/quadratic-roots.md) | Integer quadratic outputs | Supplied-program Boolean executor; end-to-end root training fails | executor PART 10/10 exact; TRAIN 0/10; parent final 0/10 | ✓: supplied executor only | weights |
| [compose-arithmetic-registers](entries/compose-arithmetic-registers.md) | Execute arithmetic on named binary registers | Fixed-layout operand fetch; freely named register access fails | TRAIN 0/10 exact; PART 3/3 exact *(screen)* | partial: final-only complement; limited operation coverage | weights |
| [integer-matvec](entries/integer-matvec.md) | Integer matrix-vector product | Dense carry/next-bit circuit; neither tested size transfers | scratchpad TRAIN 8/10 final exact on 2000 examples; answer-only 1/3 *(screen)* | partial: PART complement on two seeds; PART weights absent | weights |

## Number formats

| Entry | Task | Circuit found | Runs and criterion | Closed form | Reproduce |
|---|---|---|---|---|---|
| [base-conversion](entries/base-conversion.md) | Convert hexadecimal, binary and decimal | Digit-code readout; echo and context affect native routing | hex-to-binary TRAIN 16/40 ≥ .99; decimal PART 6/10 ≥ .99; carry TRAIN 0/60 | partial: sufficient readout; digit-only routing fails | code |
| [output-order-base-conversion](entries/output-order-base-conversion.md) | Sixteen-bit decimal and binary conversion | Magnitude comparisons learn more readily than remainders; rounded stops overstate exactness | TRAIN decimal-to-binary 1/7 exact on 6554 integers; protocol 3/7 | partial: stage substitution only; no complement | weights |

## Automata and finite fields

| Entry | Task | Circuit found | Runs and criterion | Closed form | Reproduce |
|---|---|---|---|---|---|
| [dfa-group-automata](entries/dfa-group-automata.md) | Track parity or group-automaton state | Serial state-update chain; answer-only routing differs | final scratchpad TRAIN 10/10 exact; none at 128 symbols; answer-only 2/10 ≥ .99 | partial: final checks cover 8/10 seeds | weights |
| [gf2-matrix-product](entries/gf2-matrix-product.md) | Multiply binary matrices using XOR | Local Boolean lookup; trained operand retrieval one layer later | PART 10/10 exact; TRAIN 10/10 each format | partial: 20/20 PART snapshots; complement unreported | weights |
| [gf2-elimination](entries/gf2-elimination.md) | Solve binary linear systems | Row-update table; final whole-network solutions form three families | TRAIN answer only 10/10 exact on 2000 systems | partial: local substitutions; complement untested | weights |
| [linear-systems-finite-fields](entries/linear-systems-finite-fields.md) | Solve systems modulo five or seven | Reference-keyed late write substitutes; complement tests fail | TRAIN detailed trace 10/10 exact on 2000 systems | readout only: final TRAIN operand table | weights |
| [crc-remainder](entries/crc-remainder.md) | Compute a binary polynomial remainder | Distributed shift-XOR step; readout and discovered circuit pass separately | TRAIN final 10/10 exact | partial: readout and circuit, not upstream formula | weights |
| [dyck-k-and-agreement](entries/dyck-k-and-agreement.md) | Recognise nesting; choose verb number | Partial pre-opener-state pointer; first-noun agreement shortcut | bracket trace TRAIN 10/10 exact; agreement 9/9 exact | partial: pointer's partner head unidentified | weights |
| [counter-languages](entries/counter-languages.md) | Recognise three counting languages | Ratio counts and prefix violations; one final circuit fails complement | TRAIN answer only 10/10 exact on 2048 words; scratchpad 1/3 | partial: 9/10 verified; seed 7 readout only | weights |
| [binding-coreference](entries/binding-coreference.md) | Resolve reflexive and pronoun references | One head attends to referent; state-copy readout substitutes | TRAIN answer-only run 1 3/30, run 2 4/10 exact; no final *(screen)* | readout only: every type-level complement fails | weights |

## Symbolic mathematics

| Entry | Task | Circuit found | Runs and criterion | Closed form | Reproduce |
|---|---|---|---|---|---|
| [polynomial-add-mul](entries/polynomial-add-mul.md) | Add and multiply integer polynomials | Sum/readout ratio; multiplication improves with frozen product gates | addition 40/40 exact; split-sum multiplication 3/3 ≥ .99 *(screen)* | partial: 40/40 PART snapshots; no complement | weights |
| [polynomial-differentiation](entries/polynomial-differentiation.md) | Differentiate an integer polynomial | Local multiplier; final scratchpad failures fetch the wrong coefficient | PART 10/10 per format exact; TRAIN 6/10 final exact | partial: PART substitution; complement not tested | weights |
| [polynomial-integration](entries/polynomial-integration.md) | Integrate a polynomial to reduced fractions | Distributed integration code; only scaffolded scratchpad transfers exactly to withheld fractions | TRAIN scratchpad 9/10 final exact | partial: pooling 20/20 snapshots; no gcd/division form | weights |
| [chain-rule-differentiation](entries/chain-rule-differentiation.md) | Symbolic differentiation | Weighted prefix counts; complete derivative rewrite remains without compact form | TRAIN scratchpad 10/10 exact on 256 expressions; PART 0/12 | partial: front end only; no complement | weights |
| [symbolic-algebra-atoms](entries/symbolic-algebra-atoms.md) | Collect, expand and solve symbolic expressions | Categorical expansion and solving codes need intermediate results | TRAIN expansion 10/10 best, 9/10 final exact; solving 10/10 | partial: PART collection 13/13 substitutions; no complement | weights |
| [nested-expressions-mod-7](entries/nested-expressions-mod-7.md) | Evaluate parenthesised arithmetic modulo seven | Frozen parser with learned highest-score residue table | PART 10/10 exact; TRAIN 0/3 per format exact | ✓: arithmetic block on frozen parser | weights |
| [euler-decay-ode](entries/euler-decay-ode.md) | Repeat a rounded integer decay step | Hand-built route retained; held-out update thresholds fail | PART 0/10 final exact on 480 cases; .963–.988 | —: no closed form tested | weights |

## Search and games

| Entry | Task | Circuit found | Runs and criterion | Closed form | Reproduce |
|---|---|---|---|---|---|
| [grid-bfs](entries/grid-bfs.md) | Shortest-path distance on grid mazes | Reached-set maps learned; from-scratch circuit remains distributed | TRAIN scratchpad reported 10/10 exact, teacher-forced; answer-only trials fail | partial: PART complement checked in three seeds | weights |
| [dpll-search](entries/dpll-search.md) | Satisfiability of small Boolean formulas | Local state update inside frozen search; whole-network training fails | PART 10/10 exact on 192 traces; TRAIN 0/3 *(screen)* | ✓: local update only | weights |
| [chess-krk](entries/chess-krk.md) | King-and-rook endgame policy imitation | Coordinate decoder; destination rule remains unexplained | final TRAIN 0/7 exact, 7/7 ≥ .99; three unfinished | partial: source 7/7; no complement | weights |
| [chess-mate-in-one](entries/chess-mate-in-one.md) | First mate among supplied checks | Late count/winner lookup; answer-only complement fails | TRAIN count-writing 3/3 exact on 128 sequences *(screen)* | ✓: reference-keyed count-writing lookup only | weights |
| [chess-attack-maps](entries/chess-attack-maps.md) | Mark squares attacked by one colour | Weighted token counts, late pointers and feed-forward ray steps | armies 7/10 recorded 1.0; sliders 1/10 | ✗: tested circuit formulas fail | weights |
| [chess-kbbk-mate](entries/chess-kbbk-mate.md) | Predict king-and-two-bishops endgame policy moves | Distributed readout; scratchpad candidate generation fails | TRAIN 0/9 ≥ .99; best .627 on 2,048 rows | —: no closed form recovered | weights |

## External model zoos

| Entry | Task | Circuit found | Networks and snapshot criterion | Closed form | Reproduce |
|---|---|---|---|---|---|
| [algzoo-2nd-argmax](entries/algzoo-2nd-argmax.md) | Position of second-largest Gaussian input | Comparators, angle maps and clamp/fan forms; many longer networks unexplained | 280 frozen nets, five per cell; 132 proven / 33 read / 115 unexplained | partial: 117 resample; 15 optimal-only | weights |
| [algzoo-median](entries/algzoo-median.md) | Median of Gaussian scalars | Pair means, comparator insertion and inexact recurrences; wide forms do not compress | 100 frozen nets, five per cell; 100 proven / 0 read / 0 unexplained | partial: 20 empty complements; other flags | weights |

Both rows retain the published 2026-11-13 snapshots: second-argmax ends at Sol iteration 12; median at iteration 22. “Proven” is a
registered sampled-certificate category, not task accuracy or all-input equivalence. Median includes 49 accurate networks and
51 inexact approximators, 20 empty-complement exceptions and 11 other flagged proofs. Its widest forms retain 274/1056 coefficients
versus 288/1088 network parameters. No length-generalising median circuit is certified. Read each entry's near-top limitations
before interpreting the registry counts; the source networks are externally hosted, not all re-uploaded in the atlas archive.
