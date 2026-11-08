# Index

One row per entry, grouped as in the [README](README.md#entries); each row follows the entry's lede and fact table, where the full
detail is. In the Seeds column, PART is one stage trained on frozen hand-built parts and TRAIN is a network trained from scratch;
"k/n" means that k of n seeds meet the cell's criterion (usually held-out exact match ≥ .99), and *(screen)* marks a 3-seed screening
cell with no ten-seed final yet. In the Closed form column, ✓ means verified under the acceptance rule of
[METHODS §9](METHODS.md#9-verification-and-the-acceptance-rule) (substitution and a resample ablation of the complement both keep
accuracy; no entry has this status yet), partial means supported by substitution for one rung or part only or with the complement not
tested, readout only means that substitution keeps accuracy but the entry's audit found the full bar not passed, ✗ means tested and
rejected, and — means not tested. In the Reproduce column, code means that `code/run.py` reruns the entry
([code/README.md](code/README.md)), and weights means that networks, test sets and recorded accuracies are on
[Hugging Face](https://huggingface.co/jc1122/circuit-atlas-nets) only. Other terms are defined in [GLOSSARY.md](GLOSSARY.md).

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
| [decimal-hex-addition](entries/decimal-hex-addition.md) | 3-digit addition in base 10 and 16 | column-sum lookup on X1 + X2 + carry; no carry-lookahead head from scratch | PART 60/60; TRAIN 37/40 after one retry | partial: PART stage lookup .95–1.00 | weights |
| [binary-addition](entries/binary-addition.md) | a + b, 6-bit binary | lookup on a + b + carry; position inputs block longer inputs until masked | TRAIN 10/10 per format after retries; all fail at 12 bits | partial: PART sum-bit lookup 10/10, carry tokens 3/10 | weights |
| [router-addition](entries/router-addition.md) | addition of 16 bits or 6 decimal digits | one shared column step; twice the length only without position input | PART 10/10; TRAIN router 10/10 in range, 0 at 2N | partial: PART step 1 on 60 snapshots | weights |
| [subtraction](entries/subtraction.md) | a − b, 3 decimal digits or 6 bits | per-column lookup on a − b − borrow | PART 8/10 to 10/10 per cell; retried TRAIN 10/10 | readout only: equals network accuracy | weights |
| [multiplication](entries/multiplication.md) | a · b, 6-bit binary | scratchpad step reuses the adder's sum lookup; answer only never trains | PART 10/10, TRAIN 8/10 (scratchpad) | readout only: 1.00 in 10/10 PART | weights |
| [multiplication-base-order](entries/multiplication-base-order.md) | multiplication below 256 in bases 2, 10, 16 | least significant first learnable, most significant first 1/60; scratchpad solves all | scratchpad 18/18; final cell 10/10 | partial: Fourier readout, decimal 6/6, hex 3/6 | weights |
| [long-division-square-root](entries/long-division-square-root.md) | binary division and square root, 8-bit and wide | one shared compare-and-subtract step; answer only needs 16–90x more data | division scratchpad PART 10/10; none exact on all inputs | partial: PART scratchpad step, exact | weights |
| [gcd-euclid](entries/gcd-euclid.md) | gcd by subtractive Euclid, inverse, fraction reduction | three-rule router on frozen Euclid parts; distributed token table from scratch | PART 10/10 ≥ .99 (7/10 exact); TRAIN 1/3, 2/3 *(screen)* | partial: PART router, 10 final seeds exact | weights |
| [quadratic-roots](entries/quadratic-roots.md) | integer roots of x² + bx + c | executor stage is one AND neuron; parent formats memorise or underfit | executor PART 10/10, TRAIN 0/10 | partial: executor only, 1,792/1,792 per seed | weights |
| [compose-arithmetic-registers](entries/compose-arithmetic-registers.md) | register machine: STEP composed into SUM, MUL, POW | compositions fetch operands by trace position; bare STEP fails | TRAIN 0/10; PART 3/3 *(screen)* | partial: PART 1.0 in 18/18, complement 1.0 | weights |

**Number formats**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [base-conversion](entries/base-conversion.md) | hex, binary and decimal conversions, x = 0..255 | hex to binary: nibble one-hot lookup; carry-type conversions never learned | hex to binary TRAIN 1/10 to 7/10; carry-type 0 of 80 | partial: 36 of 40; joint scrub fails | code |
| [output-order-base-conversion](entries/output-order-base-conversion.md) | 16-bit decimal ↔ binary, either digit order | most significant first is a threshold ladder; least significant first mostly fails | TRAIN decimal to binary 10/10; scratchpad PART 10/10 | partial: PART 12/12 snapshots; TRAIN template ✗ | weights |

**Automata and finite fields**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [dfa-group-automata](entries/dfa-group-automata.md) | parity, Z3 or S3 automaton state | transition table with a scratchpad; S3 answer only on frozen parts uses prefix doubling | scratchpad 10/10 per group; S3 answer-only TRAIN 2/10 | readout only: δ table, rejected as universal | weights |
| [gf2-matrix-product](entries/gf2-matrix-product.md) | 3×3 bit matrix product over GF(2) | 3-bit XOR-AND lookup per step; from scratch same variables, different wiring | 10/10 in every cell | partial: PART 1.0 in 20/20 with controls | weights |
| [gf2-elimination](entries/gf2-elimination.md) | solve A x = b over GF(2) | XOR-AND row update; three circuit types from scratch given enough matrices | PART 10/10; TRAIN answer only 10/10 at m = 4; m = 3 5/10 | partial: PART 20/20 snapshots | weights |
| [linear-systems-finite-fields](entries/linear-systems-finite-fields.md) | solve A x = b over F₅ and F₇ | microsteps learned as a distributed lookup; fused update bypasses the subtractor | TRAIN microsteps, F₅, m = 2: 10/10; answer only 0/3 | partial: PART surrogate 1024/1024; complement fails | weights |
| [crc-remainder](entries/crc-remainder.md) | GF(2) polynomial remainder N mod D, degree-4 D | shift-XOR step learned; answer only uses a different parity readout | PART 10/10 (scratchpad); TRAIN 3/3 *(screen)* | partial: PART table 11,587/11,587; controls open | weights |

**Symbolic mathematics**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [polynomial-add-mul](entries/polynomial-add-mul.md) | integer polynomial addition and multiplication, degree 3 | addition learned in every network; multiplication never exact | addition 40/40; multiplication 0/3 *(screen)* | partial: PART addition 40/40 | weights |
| [polynomial-differentiation](entries/polynomial-differentiation.md) | power-rule derivative, degree 3 | frozen fetch plus learned low-bit multiplier; from scratch the coefficient copy fails | PART 10/10, TRAIN 6/10 | partial: PART MLP2 = k·a, 40/40 snapshots | weights |
| [polynomial-integration](entries/polynomial-integration.md) | integrate a polynomial, reduced fractions c/j | only the scratchpad transfers to withheld fractions; distributed code from scratch | TRAIN scratchpad 9/10 | partial: layer-0 pooling 20/20; no gcd form | weights |
| [chain-rule-differentiation](entries/chain-rule-differentiation.md) | symbolic derivative of prefix expressions, depth ≤ 2 | token-pair front end from scratch; frozen-parser stage bypasses the parser | TRAIN 10/10 (scratchpad); PART 0/12 | partial: front end, 16/16 TRAIN nets | weights |
| [symbolic-algebra-atoms](entries/symbolic-algebra-atoms.md) | collect like terms, expand, solve ax+b=cx+d | collecting learned at every rung; expanding and solving need a scratchpad | scratchpad 10/10 in three cells; answer only 0/3 | partial: PART collect, 13/13 seeds | weights |

**Search and games**

| Entry | Task | Circuit found | Seeds | Closed form | Reproduce |
|---|---|---|---|---|---|
| [grid-bfs](entries/grid-bfs.md) | shortest-path distance, 6×6 mazes | reached-set scratchpad learned from scratch; answer only exact up to distance 3 | scratchpad TRAIN 3/3, answer only 0/3 *(screen)* | partial: PART hinge 176/176; none for TRAIN | weights |
| [dpll-search](entries/dpll-search.md) | DPLL on small 3-SAT | trained last MLP completes both hand-built solvers; from scratch no state update | PART 3/3, TRAIN 0/3 *(screen)* | partial: PART state update, 52/52 | weights |
| [chess-krk](entries/chess-krk.md) | king and rook vs king, 8-rule policy | rule router at .993–.999, never exact; trained nets code kings as coordinates | PART 60/60 runs; TRAIN 7/7 at .990–.997, none exact | partial: PART phase 20/20, TRAIN source 7/7 | weights |
| [chess-mate-in-one](entries/chess-mate-in-one.md) | first mate among up to 16 supplied checks | final MLP decodes the hand-built escape count; TRAIN not finished | 3/3 per format *(screen)* | partial: answer-only decoder, 442 scalars | weights |
