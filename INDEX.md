# Index

One row per entry, grouped as in the [README](README.md#entries). Full detail is in each entry's Summary and Status.
**PART** = one stage trained on frozen hand-built parts; **TRAIN** = trained from scratch; "k/n" = k of n seeds meet the cell's
criterion (usually held-out exact match ≥ .99); *(screen)* = 3-seed screening cell, no ten-seed final yet.
Closed form: ✓ = substituting the formula keeps accuracy · partial = holds for one rung or part only · readout only = substitution
keeps accuracy but the entry's audit says the full bar is not passed · ✗ = tested and rejected · — = not tested.
Reproduce: **code** = `code/run.py` reruns it ([code/README.md](code/README.md)) · **weights** = networks, test sets and recorded
accuracies on [Hugging Face](https://huggingface.co/jc1122/circuit-atlas-nets) only. Other terms: [GLOSSARY.md](GLOSSARY.md).

**Counting and modular arithmetic**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [modular-addition-mlp](entries/modular-addition-mlp.md) | (x + y) mod 59, MLP on a frozen menu | Fourier "clock" of a wide MLP (Gromov) | clock 10/10; test .91–.97, none ≥ .99 | ✓ .996–1.000 (per-neuron formula) | code |
| [count-mod-5](entries/count-mod-5.md) | count of a letter mod 5, MLP on a frozen menu | flat residue lookup on the count; length ignored | 30/30 | ✓ 1.000 | code |
| [compare-counts](entries/compare-counts.md) | which of two letters is more frequent | tilted linear comparator; `=` only memorised | 30/30 | partial: .93–.96, never outputs `=` | code |
| [open-bracket-count](entries/open-bracket-count.md) | number of `(` in a bracket line | low-frequency clock sum (k = 1–3) plus a parity channel; memorises below 80% data | 10/10 at fraction .8 | partial: clock form .91–.93 (nets .96–.98) | code |
| [modular-arithmetic-mod-97](entries/modular-arithmetic-mod-97.md) | a + b, a · b, x² mod 97 in six number formats | token multiplication: discrete-log clock; digit formats never form circles; digit multiplication memorises | multiplication 6/10, addition 10/10 (token format) | partial: output-level clock only, ≥ .98 in 8/10 seeds | weights |

**Arithmetic**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [decimal-hex-addition](entries/decimal-hex-addition.md) | a + b, 3 digits, base 10 and 16 | column-sum lookup on X1 + X2 + carry; from scratch, no carry-lookahead head | PART 60/60, TRAIN 37/40 | partial: PART stage lookup = net accuracy; none for TRAIN | weights |
| [binary-addition](entries/binary-addition.md) | a + b, 6-bit binary | lookup on a + b + carry; position inputs break longer inputs until masked | TRAIN 10/10 (answer only, carry token); masked PART 10/10 at 12 bits | partial: PART sum-bit lookup (A, C 10/10; B 3/10); TRAIN none general | weights |
| [router-addition](entries/router-addition.md) | a + b, 16 bits or 6 decimal digits | one shared column step; longer inputs work only when the stage cannot read position | PART 10/10; TRAIN never reaches 2N | partial: PART step stage 1.0 on 60 snapshots; none for TRAIN | weights |
| [subtraction](entries/subtraction.md) | a − b (a ≥ b), 3 digits base 10, 6 bits base 2 | per-column lookup on a − b − borrow, not the adder's key | PART 8/10 to 10/10 per cell | readout only: f(diff) keyed on task labels | weights |
| [multiplication](entries/multiplication.md) | a · b, 6-bit binary | scratchpad step = the adder's sum lookup; answer only never trains | PART 10/10, TRAIN 8/10 (scratchpad) | readout only: 1.00 (scratchpad PART) | weights |
| [multiplication-base-order](entries/multiplication-base-order.md) | x · y, x, y < 256, bases 2/10/16 | LSB-first answers learnable, MSB-first 1/60; a scratchpad solves every MSB cell | LSB binary final 10/10; scratchpad 18/18 | partial: scratchpad Fourier readout (decimal 6/6, hex 3/6) | weights |
| [long-division-square-root](entries/long-division-square-root.md) | binary N / D and square root of N, 8-bit and wide | one shared compare-and-subtract step; answer only needs 16–90x more data | division scratchpad PART 10/10; division answer-only TRAIN 10/10 (wide) | partial: scratchpad PART step 1.0; answer only and TRAIN none | weights |
| [gcd-euclid](entries/gcd-euclid.md) | gcd(a, b) by subtractive Euclid, plus inverse and fractions | learned three-rule router on frozen Euclid parts; from scratch a token-histogram code | PART final 10/10 ≥ .99 (7/10 exact); TRAIN 1/3 (6 bits), 2/3 (7 bits) *(screen)* | partial: PART router, all 10 final seeds exact | weights |
| [quadratic-roots](entries/quadratic-roots.md) | integer roots of x² + bx + c, incl. a NAND executor scratchpad | executor stage = one AND neuron; every parent cell memorises or underfits | executor PART 10/10, TRAIN 0/10 | partial: executor stage only, one neuron, 1,792/1,792 per seed | weights |
| [compose-arithmetic-registers](entries/compose-arithmetic-registers.md) | base-2 register machine: STEP composed into SUM, MUL, POW | compositions learned by trace position, no register lookup; bare STEP fails | TRAIN 0/10 solved (compositions 10/10, STEP fails); PART 3/3 *(screen)* | partial: frozen-router PART 1.0, complement resample 1.0; TRAIN none | weights |

**Number formats**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [base-conversion](entries/base-conversion.md) | hex, binary and decimal conversions, x = 0..255 | hex to binary: nibble one-hot lookup, groks from scratch; carry-type conversions never learned | hex to binary TRAIN 6/10, 7/10 (80% data) | partial: hex to binary 1.00 in 36/40; PART decimal to binary thresholds .94–1.06 | code |
| [output-order-base-conversion](entries/output-order-base-conversion.md) | 16-bit decimal ↔ binary, MSB- or LSB-first | MSB-first = threshold ladder, learned; LSB-first answer only needs a mod readout, mostly fails | TRAIN decimal to binary MSB-first 10/10; scratchpad PART 10/10 | partial: PART forms 12/12 tested snapshots (seeds 0–2); TRAIN template ✗ | weights |

**Automata and finite fields**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [dfa-group-automata](entries/dfa-group-automata.md) | parity, Z3 or S3 automaton state, words of 1–16 symbols | scratchpad: transition table learned at once; frozen-part S3 answer only uses prefix doubling | scratchpad PART, TRAIN 10/10; S3 answer-only TRAIN 2/10 | partial: δ table on best snapshots only, rejected as universal | weights |
| [gf2-matrix-product](entries/gf2-matrix-product.md) | C = A B over GF(2), 3×3 bit matrices | 3-bit XOR-AND lookup per trace step; from scratch, same variables, different wiring | 10/10 in every cell | partial: PART passes the full bar; none from scratch | weights |
| [gf2-elimination](entries/gf2-elimination.md) | solve A x = b over GF(2), m = 3 and 4 | XOR-AND row update; from scratch, three circuit types given enough matrices | PART 10/10; TRAIN answer only 10/10 (m = 4) | partial: PART 20/20 snapshots; from scratch one type only | weights |
| [linear-systems-finite-fields](entries/linear-systems-finite-fields.md) | solve A x = b over F₅/F₇, m = 2 and 3 | microstep scratchpad learned from scratch as one distributed lookup; fused update bypasses the hand-built subtractor | TRAIN microstep, F₅, m = 2: 10/10 | partial: PART surrogate 1024/1024; whole-net check fails | weights |
| [crc-remainder](entries/crc-remainder.md) | GF(2) polynomial remainder N mod D, degree-4 D | shift-XOR step learned; answer only uses a parity readout, not the hand-built staircase | PART 10/10 (scratchpad); TRAIN 3/3 *(screen)* | partial: PART eight-ReLU table 11,587/11,587; controls partly open | weights |

**Symbolic maths**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [polynomial-add-mul](entries/polynomial-add-mul.md) | integer polynomial addition and multiplication, degree 3 | addition learned in every seed; multiplication never exact | addition 40/40; multiplication none *(screen)* | partial: PART addition κ·HAND, 40/40; multiplication none | weights |
| [polynomial-differentiation](entries/polynomial-differentiation.md) | power-rule derivative, degree 3 | frozen fetch + learned low-bit multiplier; from scratch fails the coefficient copy | PART 10/10, TRAIN 6/10 | partial: PART MLP2 = k·a; TRAIN numerical compression only | weights |
| [polynomial-integration](entries/polynomial-integration.md) | integrate a polynomial, reduced fractions c/j | only the scratchpad transfers to withheld fractions; from scratch a distributed code | TRAIN scratchpad 9/10 | partial: layer-0 pooling; one seed's reader | weights |
| [chain-rule-differentiation](entries/chain-rule-differentiation.md) | symbolic derivative of prefix expressions, depth ≤ 2 | solved from scratch with a token-histogram front end; PART bypasses the frozen parser | TRAIN 10/10 (scratchpad); PART fails 12/12 runs | partial: front end, 16/16 TRAIN nets | weights |
| [symbolic-algebra-atoms](entries/symbolic-algebra-atoms.md) | collect like terms, expand, solve ax+b=cx+d | collecting learned at every rung; expanding and solving need a scratchpad | scratchpad finals 10/10; answer only never exact | partial: PART collect 1.0, 13/13 seeds; TRAIN none | weights |

**Search and games**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [grid-bfs](entries/grid-bfs.md) | shortest-path distance, 6×6 mazes | reached-set scratchpad learned from scratch; answer only hits a depth limit | 3/3 *(screen)* | partial: PART update hinge; TRAIN ✗ | weights |
| [dpll-search](entries/dpll-search.md) | DPLL on small 3-SAT | trained last MLP completes both hand-built solvers; from scratch the state update is not learned | PART 3/3, TRAIN 0/3 *(screen)* | partial: scratchpad PART state update, 52/52 | weights |
| [chess-krk](entries/chess-krk.md) | king + rook vs king, a fixed 8-rule policy | rule router at .993–.999, never exact; trained nets code kings as coordinates | PART 60/60 runs, TRAIN 7/7 at .990–.997 | partial: phase output only (20/20) | weights |
| [chess-mate-in-one](entries/chess-mate-in-one.md) | first mating move among K ≤ 16 supplied checks | final MLP decodes the hand-built escape count; TRAIN not finished | 3/3 per format *(screen)* | partial: answer-only decoder, 442 scalars | weights |
