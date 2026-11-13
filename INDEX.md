# Index

One row per entry, grouped as in the [README](README.md#entries). The atlas has 41 entries: the full task, data, criterion and caveats are in each entry's fact table. PART trains one stage on frozen hand-built parts; TRAIN trains the whole network from scratch ([rungs](GLOSSARY.md#the-three-rungs)). In Seeds, k/n means k of n seeds meet the named criterion; *(screen)* marks screening evidence rather than a final cell. A circuit-type count among solved seeds is not a success rate across all seeds.

In Closed form, ✓ means substitution and complement resampling pass under the [acceptance rule](METHODS.md#9-verification-and-the-acceptance-rule), partial means only the stated rung, part or seed subset is covered, readout only means substitution works but the claimed mechanism fails its audit, ✗ means tested and rejected, and — means not tested. A scope or flag after the status is part of the claim. External-zoo “proven” counts are registered sampled-certificate categories, not task-accuracy counts or all-input theorems.

In Reproduce, code means a command runs from [code/](code/README.md); weights means the entry links archived networks and recorded checks on [Hugging Face](https://huggingface.co/jc1122/circuit-atlas-nets), with any archive limitations stated there. Other terms are in [GLOSSARY.md](GLOSSARY.md).

**Counting and modular arithmetic**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [modular-addition-mlp](entries/modular-addition-mlp.md) | (x + y) mod 59 on a frozen menu | Fourier clock of a wide MLP, 22–27 of 29 frequencies | clock 10/10; test .91–.97, none ≥ .99 | partial .996–1.000; complement not tested | code |
| [count-mod-5](entries/count-mod-5.md) | letter count mod 5 on a frozen menu | flat residue lookup on the count; line length ignored | 30/30 | partial 1.000; complement not tested | code |
| [compare-counts](entries/compare-counts.md) | which of two letters is more frequent | linear comparator in c and L; ties only memorised | greater/less learned in 30/30; `=` 0 correct test cells | partial .93–.96; never outputs `=` | code |
| [open-bracket-count](entries/open-bracket-count.md) | number of `(` in a bracket line | low-frequency cosine sum (k = 1–3) plus a parity channel | clock sum 10/10 at fraction .8, none reaches .9; memorises at .4 and .6 | partial .91–.93 (networks .96–.98) | code |
| [modular-arithmetic-mod-97](entries/modular-arithmetic-mod-97.md) | addition, multiplication, squaring mod 97; six number formats | token multiplication: discrete-log circles; digit formats form no circles | token multiplication 6/10, token addition 10/10 | partial: output-level clock, 8 seeds ≥ .98 | weights |

**Arithmetic**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [decimal-hex-addition](entries/decimal-hex-addition.md) | 3-digit addition in base 10 and 16 | column-sum lookup on X1 + X2 + carry; no carry-lookahead head from scratch | PART lookup in all 60 networks; TRAIN 37/40 after one retry | partial: PART stage lookup .95–1.00 | weights |
| [binary-addition](entries/binary-addition.md) | a + b, 6-bit binary | lookup on a + b + carry; position inputs block longer inputs until masked | TRAIN 10/10 per format after retries; all fail at 12 bits | partial: PART sum-bit lookup 10/10, carry tokens 3/10 | weights |
| [router-addition](entries/router-addition.md) | addition of 16 bits or 6 decimal digits | one shared column step; twice the length only without position input | PART 10/10; TRAIN router 10/10 in range, 0 at 2N | partial: PART step 1 on 60 snapshots | weights |
| [subtraction](entries/subtraction.md) | a − b, 3 decimal digits or 6 bits | per-column lookup on a − b − borrow | PART 8/10 to 10/10 per cell; retried TRAIN 10/10 | readout only: equals network accuracy | weights |
| [multiplication](entries/multiplication.md) | a · b, 6-bit binary | scratchpad step reuses the adder's sum lookup; answer only never trains | PART 10/10, TRAIN 8/10 (scratchpad) | readout only: 1.00 in 10/10 PART | weights |
| [multiplication-base-order](entries/multiplication-base-order.md) | multiplication below 256 in bases 2, 10, 16 | least significant first learnable, most significant first 1/60; scratchpad solves all | scratchpad 18/18; final cell 10/10 | partial: Fourier readout, decimal 6/6, hex 3/6 | weights |
| [long-division-square-root](entries/long-division-square-root.md) | binary division and square root, 8-bit and wide | one shared compare-and-subtract step; answer only needs 16–90x more data | division scratchpad PART 10/10; none exact on all inputs | partial: PART scratchpad step, exact | weights |
| [gcd-euclid](entries/gcd-euclid.md) | GCD by subtractive Euclid | three-rule router on frozen parts; seed-specific distributed circuits from scratch | PART 10/10 ≥ .99 (7/10 exact); TRAIN 5/10 exact | partial: PART complement on seven exact seeds | weights |
| [quadratic-roots](entries/quadratic-roots.md) | integer discriminant, square root and quadratic roots | executor PART pools hinges into NAND; from-scratch register fetch fails | executor PART 10/10 exact; TRAIN 0/10; parent final 0/10 | ✓: executor PART only | weights |
| [compose-arithmetic-registers](entries/compose-arithmetic-registers.md) | register machine: STEP composed into SUM, MUL, POW | compositions fetch operands by trace position; bare STEP fails | TRAIN 0/10; PART 3/3 *(screen)* | partial: PART 1.0 in 18/18, complement 1.0 | weights |
| [integer-matvec](entries/integer-matvec.md) | integer matrix-vector product, 3×3 | dense last-layer carry/next-bit circuit; answer only needs 4x budget | scratchpad TRAIN 8/10 exact; PART 3/3 *(screen)*; answer-only TRAIN 1/3 *(screen)* | partial: PART complement on two tested seeds | weights |

**Number formats**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [base-conversion](entries/base-conversion.md) | hex, binary and decimal conversions, x = 0..255 | hex to binary: nibble one-hot lookup; carry-type conversions never learned | hex to binary TRAIN 1/10 to 7/10; carry-type 0 of 80 | partial: 36 of 40; joint scrub fails | code |
| [output-order-base-conversion](entries/output-order-base-conversion.md) | 16-bit decimal ↔ binary, either digit order | most significant first is a threshold ladder; least significant first mostly fails | TRAIN decimal to binary 10/10; scratchpad PART 10/10 | partial: PART 12/12 snapshots; TRAIN template ✗ | weights |

**Automata and finite fields**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [dfa-group-automata](entries/dfa-group-automata.md) | parity, Z3 or S3 automaton state | final S3 scratchpad uses a serial MLP0–5 chain and late transition tables | scratchpad TRAIN 10/10 exact per group; final S3 10/10 | partial: final S3 accepted 8/10 | weights |
| [gf2-matrix-product](entries/gf2-matrix-product.md) | 3×3 bit matrix product over GF(2) | 3-bit XOR-AND lookup per step; from scratch same variables, different wiring | 10/10 in every cell | partial: PART 1.0 in 20/20 with controls | weights |
| [gf2-elimination](entries/gf2-elimination.md) | solve A x = b over GF(2) | XOR-AND row update; three circuit types from scratch given enough matrices | PART 10/10; TRAIN answer only 10/10 at m = 4; m = 3 5/10 | partial: PART 20/20 snapshots | weights |
| [linear-systems-finite-fields](entries/linear-systems-finite-fields.md) | solve A x = b over F₅ and F₇ | microsteps learned as a distributed lookup; fused update bypasses the subtractor | TRAIN microsteps, F₅, m = 2: 10/10; answer only 0/3 | partial: PART surrogate 1024/1024; complement fails | weights |
| [crc-remainder](entries/crc-remainder.md) | polynomial remainder over GF(2), degree-4 divisor | distributed scratchpad step; readout table and discovered circuit pass separately | PART 10/10; TRAIN final 10/10 exact; screening TRAIN 3/3 | partial: readout and circuit, not upstream formula | weights |
| [dyck-k-and-agreement](entries/dyck-k-and-agreement.md) | bracket recognition and main-verb number agreement | agreement copies first noun; bracket scratchpad learns a pop-pointer state circuit | answer-only agreement TRAIN 9/9 exact; bracket final TRAIN 10/10 exact, pointer substitution in 8/10 | partial: pop-pointer complement fails | weights |
| [counter-languages](entries/counter-languages.md) | recognise three counting languages | ratio counts, violation flags, sum head and four-entry answer table | answer-only final TRAIN 10/10 exact; scratchpad one language 1/3 exact | partial: final complement passes 9/10 | weights |
| [binding-coreference](entries/binding-coreference.md) | resolve reflexive and feature-marked pronoun references | one head attends to the referent; scratchpad SELF copies the state | answer-only TRAIN r1 3/30, r2 4/10 exact; no final *(screen)* | readout only: every type-level complement fails | weights |

**Symbolic mathematics**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [polynomial-add-mul](entries/polynomial-add-mul.md) | integer polynomial addition and multiplication, degree 3 | addition learned in every network; multiplication never exact | addition 40/40; multiplication 0/3 *(screen)* | partial: PART addition 40/40 | weights |
| [polynomial-differentiation](entries/polynomial-differentiation.md) | power-rule derivative, degree 3 | frozen fetch plus learned low-bit multiplier; from scratch the coefficient copy fails | PART 10/10, TRAIN 6/10 | partial: PART MLP2 = k·a, 40/40 snapshots | weights |
| [polynomial-integration](entries/polynomial-integration.md) | integrate a polynomial, reduced fractions c/j | only the scratchpad transfers to withheld fractions; distributed code from scratch | TRAIN scratchpad 9/10 | partial: layer-0 pooling 20/20; no gcd form | weights |
| [chain-rule-differentiation](entries/chain-rule-differentiation.md) | symbolic derivative of prefix expressions, depth ≤ 2 | token-pair front end from scratch; frozen-parser stage bypasses the parser | TRAIN 10/10 (scratchpad); PART 0/12 | partial: front end, 16/16 TRAIN nets | weights |
| [symbolic-algebra-atoms](entries/symbolic-algebra-atoms.md) | collect like terms, expand, solve ax+b=cx+d | collecting learned at every rung; expanding and solving need a scratchpad | scratchpad 10/10 in three cells; answer only 0/3 | partial: PART collect, 13/13 seeds | weights |
| [nested-expressions-mod-7](entries/nested-expressions-mod-7.md) | evaluate parenthesised expressions modulo 7 | frozen parser plus trained argmax residue table; from-scratch literal generalisation fails | PART final 10/10, 260/260 held-out, 16/16 each OOD set | ✓: PART arithmetic block only | weights |
| [euler-decay-ode](entries/euler-decay-ode.md) | fixed-point explicit Euler for exponential decay | PART keeps the hand-built five-edge route but misses held-out first steps | PART final 0/10 exact (.963–.988); no solved cell | —: no solved cell | weights |

**Search and games**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [grid-bfs](entries/grid-bfs.md) | shortest-path distance on 6×6 mazes | distributed reached-set scratchpad; PART keeps HAND; answer only fails beyond distance 3 | scratchpad TRAIN 10/10 exact; PART 10/10 exact; answer only 0/3 ≥ .99 *(screen)* | partial: PART complement on three tested seeds | weights |
| [dpll-search](entries/dpll-search.md) | DPLL on small 3-SAT | trained last MLP implements HAND's seven-field solver-state update; from scratch fails | scratchpad PART final 10/10 exact; TRAIN 0/3 *(screen)* | ✓: PART update and 36-edge circuit | weights |
| [chess-krk](entries/chess-krk.md) | king and rook vs king, 8-rule policy | rule router at .993–.999, never exact; trained nets code kings as coordinates | PART 60/60 runs; TRAIN 7/7 at .990–.997, none exact | partial: PART phase 20/20, TRAIN source 7/7 | weights |
| [chess-mate-in-one](entries/chess-mate-in-one.md) | first mate among up to 16 supplied checks | final MLP decodes the hand-built escape count; TRAIN not finished | 3/3 per format *(screen)* | partial: answer-only decoder, 442 scalars | weights |
| [chess-attack-maps](entries/chess-attack-maps.md) | predict a colour's attacked-square map | hint scratchpad learns contextual ray pointers; answer-only nearest-piece read fails | recorded-test TRAIN C3 7/10 exact, 10/10 ≥ .998; C2 1/10 exact | ✗: substitution and complement fail | weights |
| [chess-kbbk-mate](entries/chess-kbbk-mate.md) | king and two bishops versus king policy | distributed lookup and scratchpad candidate copy; no HAND reference | TRAIN 0/9 exact; three seeds per cell (width retry truncated) | —: no solved cell | weights |

**External model zoos**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [algzoo-2nd-argmax](entries/algzoo-2nd-argmax.md) | position of second largest Gaussian input | comparators, angle maps and clamp/fan forms; many longer networks unexplained | 280 frozen nets, five per cell; snapshot 132 proven / 33 read / 115 unexplained | partial: 117 resample, 15 optimal-only | weights |
| [algzoo-median](entries/algzoo-median.md) | median of Gaussian scalar inputs | pair means, comparator insertion and inexact level/renewal forms; wide forms do not compress | 100 frozen nets, five per cell; snapshot 100 proven / 0 read / 0 unexplained | partial: 20 empty-complement exceptions; other flags | weights |

Both external-zoo rows are the published 2026-11-13 snapshots (second-argmax: Sol iteration 12; median: Sol iteration 22). Median includes 49 accurate nets and 51 inexact approximators, plus 20 empty-complement and 11 other flagged proofs. Its widest forms retain 274/1056 coefficients versus 288/1088 network parameters. No length-generalising median circuit is certified. See each entry's near-top Limitations before interpreting “proven”.
