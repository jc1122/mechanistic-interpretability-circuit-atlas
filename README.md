# Mechanistic Interpretability Circuit Atlas

What circuits do small transformers actually learn on algorithmic tasks, and how do they compare with a circuit written by hand?

For each of 30 tasks (arithmetic, base conversion, automata, GF(2) linear algebra, symbolic algebra, search, chess endgames), the
atlas builds an exact **hand-built** network, then trains networks over many seeds, both from scratch and on top of frozen hand-built parts. It reports
which circuit each seed converges to, how often, and whether a closed-form description of it survives being **substituted back** into the
network. Negative results (cells that never learn, closed forms that fail) are kept.

## Headline results

- **Swapping one part at a time is weak evidence that a circuit is right.** Replacing components of a trained net one at a time with their
  proposed closed form usually keeps accuracy. Scrubbing the whole proposed circuit at once fails in five tasks (automata,
  base conversion, division, subtraction, decimal/hex addition). Where it holds, it fails again on the greedily pruned circuit. A working
  replacement is a sufficient readout, not a verified mechanism.
- **Same variables, different wiring.** Nets trained from scratch compute the hand-built intermediate variables (interchange tests pass),
  but share only 6–9% of their top circuit edges with the hand-built net, and no rotation or permutation maps one onto the other. Nets
  trained on frozen hand-built parts share 79–100%.
- **Failed nets still compute the right variables; they fail to generalise.** This holds across 981 nets in three arithmetic tasks.
- **A scratchpad makes the circuit up to 10x smaller** and makes one shared step (e.g. compare-and-subtract) serve every step of division and square root.
  However, trained stages read absolute position and break on longer inputs until that input is masked.
- **Training reuses the adder.** With a running-sum scratchpad the trained multiplication step is the adder's column-sum lookup (10/10
  on frozen parts, 5/10 from scratch).
- **Exact hand-built circuits break under TF32 matmuls.** Their large weights and sharp thresholds need full fp32.

More, each marked verified or preliminary: [SURPRISES.md](SURPRISES.md).

## Relation to prior work

Hand-built transformers as ground truth exist: [Tracr](https://arxiv.org/abs/2301.05062) (paper) compiles RASP programs into transformer weights, and
[InterpBench](https://arxiv.org/abs/2407.14494) (paper) trains models to implement known circuits so that circuit-discovery methods can be scored.
Both target *evaluating interpretability tools*. Single-task reverse engineering across seeds also exists: the modular-addition clock
([Nanda et al.](https://arxiv.org/abs/2301.05217), paper), group operations ([Chughtai et al.](https://arxiv.org/abs/2302.03025), paper), clock vs pizza
([Zhong et al.](https://arxiv.org/abs/2306.17844), paper); and trained nets constrained to be readable programs
([Learning Transformer Programs](https://arxiv.org/abs/2306.01128), paper).

This atlas asks a different question: given an exact hand-built solution, *what does gradient descent find instead?* It uses one protocol across
many tasks: three rungs per task, 10 seeds per final cell, circuit types counted per seed, and a closed form accepted only if substituting it keeps
accuracy, with zero and shuffle controls on the rest of the net.

## How to read an entry

| Term | Meaning |
|---|---|
| **HAND** | every weight set by a formula; exact on every input in range |
| **PART** | frozen hand-built parts plus one trained stage (`PARTp`: the trained stage is an attention layer instead of an MLP) |
| **TRAIN** | trained from scratch |
| **Format** | `A` = answer only; other letters (`T`, `G`, `S`, `C`, ...) are scratchpad variants, defined in each entry's Task paragraph |
| **Seeds** | "6/10" = 6 of 10 seeds meet the criterion named in that cell (usually held-out exact match ≥ .99) |
| **Closed form verified** | substituting the formula for the trained weights keeps accuracy; "partial" or "rejected" says what failed |
| **menu** | a frozen hand-written layer that re-encodes integers (one-hot, thermometer, scalar); [METHODS.md](METHODS.md) §3 |

Each entry has the same sections ([TEMPLATE.md](TEMPLATE.md)): Summary, Setup (with the reproduce command), Prediction before training,
Hand-built reference, What training finds, Closed form, Controls and limits. Cells marked *pending*, *open* or *not run* are still in progress; the entry's Status line gives its date.

## How this was made

One author, with most experiments, analyses and entry drafts run by LLM coding agents under a written research protocol (fixed seeds,
predictions recorded before training, every run logged, failures diagnosed rather than dropped). Every number comes from a logged run;
the networks are public so any claim can be checked.

**Weights:** the trained networks and frozen inputs behind each entry are on Hugging Face: [https://huggingface.co/jc1122/circuit-atlas-nets](https://huggingface.co/jc1122/circuit-atlas-nets) (safetensors, CC BY 4.0).
[METHODS.md](METHODS.md) describes the architecture, the experimental conditions and the instrumentation common to all entries, and
[code/](code/) is a self-contained CPU reimplementation (Python, torch) that rebuilds the hand-built network, generates the synthetic data,
trains and runs the analysis.

## Quick start

```
cd code && pip install -r requirements.txt
python run.py check                                              # the hand-built model and the menus are exact
python run.py atlas --task count_mod5 --seeds 0 --steps 400 --every 100    # smoke run (about a minute); drop --steps/--every for the full 200,000 steps
```

## Index

| Entry | Task | Circuit type(s) found | Seeds | Closed form verified |
|---|---|---|---|---|
| [modular-addition-mlp](entries/modular-addition-mlp.md) | (x + y) mod 59, MLP on a frozen menu | Fourier "clock" (Gromov MLP solution), 10/10 | 10 | yes, .996–1.000 |
| [count-mod-5](entries/count-mod-5.md) | count of a letter mod 5, MLP on a frozen menu | flat residue lookup, 30/30 | 30 | yes, 1.000 |
| [compare-counts](entries/compare-counts.md) | which of two letters is more frequent | tilted linear comparator, `=` memorized, 30/30 | 30 | partial, .93–.96 (no `=`) |
| [open-bracket-count](entries/open-bracket-count.md) | number of `(` in a bracket line | low-frequency clock sum (k = 1–3) + parity channel at 80% data, 10/10; memorization below | 30 | yes, .91–.93 |
| [base-conversion](entries/base-conversion.md) | hex to binary, binary to decimal, decimal to binary (x = 0..255), answer only and scratchpad; 2-layer net | hex to binary: nibble one-hot lookup, 38/40 TRAIN nets (grokks from scratch in 16/40); decimal to binary A (frozen heads): remainder threshold, 10/10; carry-type cells never learned | 10 per setting | yes, hex to binary 1.00 in 36/40; thresholds off by <= .06 |
| [decimal-hex-addition](entries/decimal-hex-addition.md) | a + b, 3 digits, base 10 and 16; answer only, carry token, column sums; 4-layer net | column-sum lookup f(X1 + X2 + carry) in 60/60 PART nets (retry runs); TRAIN solves 37/40 with 40k steps, with no carry-lookahead head | 10 per setting | yes, .95–1.00 (PART) |
| [binary-addition](entries/binary-addition.md) | a + b, 6 bits; answer only, carry token, column sums; 4-layer net | one-head carry lookahead by hand; every working trained stage is a lookup on a + b + carry; position inputs break length until masked (column sums exact at 12 bits, 10/10); TRAIN solves all three formats in the final runs but fails at 12 bits | 10 per setting | yes, 1.00 (A, C), 3/10 to 6/10 (B) |
| [subtraction](entries/subtraction.md) | a − b (a ≥ b), 3 digits base 10 and 6 bits base 2; answer only, borrow token, signed column differences; 4-layer net | per-column lookup on a − b − borrow (not the adder's a + b + c) in 57 of 60 PART nets; TRAIN solves all six cells after one retry, same key in the last MLP in 84/100 nets; position shortcut breaks length unless masked | 10 per setting | yes, = net accuracy (PART .97–1.00) |
| [multiplication](entries/multiplication.md) | a · b, 6 bits; answer only and running-sum scratchpad; 4-layer net | scratchpad step = the adder's sum lookup on x + b_j a + carry (PART 10/10, TRAIN 5/10, closed form 8/10); answer only never trains (parity staircase not fitted, TRAIN memorises) | 10 per setting | yes, 1.00 (scratchpad PART) |
| [multiplication-base-order](entries/multiplication-base-order.md) | x · y, x, y < 256, bases 2/10/16, answer only MSB/LSB-first and running-sum scratchpad, curriculum mixes; 4-layer net (TRAIN only) | answer only: LSB-first solves base 10 (20/20) and base 2 (17/20, final 10/10), MSB-first 1/60; 2× width rescues decimal/hex MSB (7/10, 6/10), never binary MSB (0/26); scratchpad solves every MSB cell at 1× (18/18); AO readout fixed by column sum / carry | 5 per screening cell, 3 per scratchpad mix, 10 (final) | partial: scratchpad final-MLP digit vote = sparse Fourier map (decimal 6/6, hex 3/6 seeds at ≥ .99 × native); none for answer-only |
| [long-division-square-root](entries/long-division-square-root.md) | binary N / D and long-hand square root of N, 8-bit and wide ranges; answer only and per-step scratchpad; 3-layer net | scratchpad step = one shared compare-and-subtract [x ≥ T], x − bit·T (division B PART 10/10 exact, square root B .985–.992); answer-only PART never places the thresholds (≤ .29); TRAIN generalises only with 16–90x more data (division A 10/10 ≥ .99) | 10 per setting | yes, 1.00 (scratchpad PART; nets 1.00 / .985–.992) |
| [dfa-group-automata](entries/dfa-group-automata.md) | state of a parity, Z3 or S3 automaton after a word of 1–16 symbols; answer only and per-symbol state scratchpad; 6-layer net | scratchpad: transition table δ(q_prev, s) learned at once (PART and TRAIN 30/30 per format), length set by the previous-state fetch (PART n 128, TRAIN ~2x); answer-only: S3 PART = hand-built prefix-doubling tables (10/10); abelian stage f(S) misses large S; TRAIN count → residue in range (parity, Z3 10/10 with fresh words), S3 2/10 at ≥ .99 | 10 per setting | yes, δ table 1.00 (B); f(S) tables = net accuracy (A) |
| [gf2-matrix-product](entries/gf2-matrix-product.md) | C = A B over GF(2), 3×3 bit matrices; answer only and running-parity trace; 4-layer net | trace step = 3-bit XOR-AND lookup on (s_prev, A_ij, B_jk): PART reads the hand-built fetch heads (10/10); TRAIN trace 10/10 computes the same variables with fetch heads one layer later, different wiring; answer only 10/10 (13–14 heads); no length generalisation | 10 per setting | yes, 1.00 (trace PART) |
| [gf2-elimination](entries/gf2-elimination.md) | solve A x = b over GF(2) by Gauss-Jordan, m = 3 and 4; per-step scratchpad and answer only; 4-layer net | row update = XOR-AND lookup OLD ⊕ (K ∧ PB ∧ ¬EQ) on the hand-built fetches (PART and PARTp 10/10 each); TRAIN at m = 3 memorises the 118 training matrices (5/10 per format); with 6000 matrices at m = 4 answer only solves 10/10 in three types (L3-head readout 6/10, diffuse L3 2/10, L2-complete 2/10; closed form only for L2-complete); no size generalisation | 10 per setting | yes, 1.00 (PART, 20/20) |
| [router-addition](entries/router-addition.md) | a + b, 16 bits and 6 decimal digits; answer only (both digit orders), inductive scratchpad, router scratchpad (op tokens + coupled digit ids); 4-layer net | one shared column step (sum / carry / next-id groups); PART length-generalises when its stage cannot read position (R unmasked, I/A masked: 10/10 at 2N); TRAIN solves in range (I, R, AL; MSB-first answers 0/10 binary, 2/10 decimal) but reaches 2N in no format, router included; trained nets deliver operands by input-dependent attention mixtures | 10 per setting | yes, 1.00 in range (PART R stage; PART AL spline) |
| [gcd-euclid](entries/gcd-euclid.md) | gcd(a, b) by subtractive Euclid (plus modular inverse, fraction reduction), operands 5–7 bits as scalar tokens; answer only and Euclid trace; 4-layer net | PART on frozen Euclid parts = a learned 3-rule router (T 13/13 seeds ≥ .99, 7/10 exact in the final cell); TRAIN trace 1/3 (6 bits), 2/3 (7 bits) with a distributed token-histogram front end; answer-only TRAIN 1/3 with parity/divisible-by-3 features; no 8-bit transfer | 10 (T PART final), 3 per other cell | yes, router 1.00 (all PART cells incl. inverse/fraction); TRAIN front end only (.99/.93) |
| [modular-arithmetic-mod-97](entries/modular-arithmetic-mod-97.md) | a + b, a · b mod 97 and x² mod 97 in residue tokens, binary, decimal, hex and decimal scratchpad; 4-layer net | token multiplication: discrete-log clock at the output (6/10 at 90% data, 0/10 at 30%), embeddings periodic in log₅ v in 10/10, hand-built uses a log ramp + lookup instead; binary/decimal/hex never form circles; digit multiplication memorises (operands decodable, residue not); binary addition MLP0 = prefix-count rectifier (5/10); scratchpad PART fails at its state writer | 10 per setting (scratchpad 3) | yes, output surrogate 9409/9409 with 1–6 frequencies, agreement ≥ .9947 (6/6 solved seeds); internal circuit not recovered |
| [chain-rule-differentiation](entries/chain-rule-differentiation.md) | symbolic derivative of prefix expressions (x^1–x^8, sin, cos, exp, +, *), depth ≤ 2; answer only (15-layer net) and per-node register-rewrite scratchpad (5-layer net) | TRAIN solves both formats (A 3/3, T 3/3 and 10/10) with a content-weighted token-histogram front end; PART (trained emitter on the frozen hand-built parser) fails 12/12 seed-runs: it bypasses the frozen address interface; no depth-3 or power 7–8 transfer | 10 (T TRAIN final), 3 per other cell | front end only (histogram heads 1.00, 3/3); last-MLP state table exact but not compact (rejected) |
| [polynomial-add-mul](entries/polynomial-add-mul.md) | integer polynomial addition and multiplication (convolution), degree 3, signed coefficients as scalar tokens; answer only (3-layer net) and partial-product scratchpad (4-layer net) | addition learned in every seed (final runs, 10/10 per cell: PART and TRAIN, A and T): PART = HAND's sum with a free logit scale; TRAIN A sparse-fetch 2/10 vs distributed-fetch 8/10; TRAIN T reads its own trace 10/10; multiplication not learned (PART 0/3, TRAIN A 0/3, TRAIN T 2/3 non-exact only at 60k) unless the hand-built signed-bit product gates stay frozen (split-sum A 3/3 at .998-.999) | 10 (add PART/TRAIN finals), 3 per other cell | yes for PART add: (S, ONE) = κ·HAND substituted, exact 1.0 in 40/40 nets; TRAIN add A fitted MLP2 decoder .993-1.0 (lossy); TRAIN add T rejected |
| [output-order-base-conversion](entries/output-order-base-conversion.md) | 16-bit decimal to binary and binary to decimal, answer MSB-first or LSB-first; answer only and halving/doubling scratchpad; 3-layer net | MSB-first answer only = threshold ladder on the remaining magnitude, learned (PART all seeds ≥ .99, TRAIN decimal to binary 10/10 ≥ .996); LSB-first answer only needs a residue (mod) read-out and mostly fails (binary to decimal TRAIN ≤ .045); scratchpads solve both orders (PART 10/10) | 10 per A cell, 10 trained / 3 archived per B cell | yes, PART thermometer (366 thresholds) and failing parity spline reproduce every token, 12/12; TRAIN scalar template rejected 0/12 |
| [symbolic-algebra-atoms](entries/symbolic-algebra-atoms.md) | collect like terms, expand a binomial product, solve ax+b=cx+d, on (side, power, coefficient) term atoms; answer only and scratchpad; 2-3-layer net | collection learned at every rung (PART 10/10 in 1k steps, TRAIN 3/3); expansion not learned answer-only (PART ≤ .96, TRAIN ≤ .13), TRAIN with scratchpad solves it at 60k (3/3; ten-seed final 10/10); solving learned from scratch only with the scratchpad (3/3 exact; ten-seed final 10/10), PART quadratic stage lacks precision | 3 per cell (10: PART collect T, TRAIN expand T, TRAIN solve T) | yes, PART collection sum / phase-gated copy-copy-sum, 4096/4096 in 6/6 seeds; no TRAIN closed form |
| [chess-krk](entries/chess-krk.md) | king + rook vs king, white to move, label = a fixed 8-rule fence policy (not DTM); answer only, fence-state scratchpad, scratchpad + candidate list; 6-layer net | PART (trained router on frozen hand-built rule features) .993–.999 in 60/60 seed-runs, reads rook rays and king distances; TRAIN scratchpad reaches .993–.998 (3/3) at 120k steps, answer only .970–.977 (0/3); trained nets code kings as coordinates, last MLP not additive in the HAND variables | 10 (PART, TRAIN 30k), 3 (TRAIN 60k/120k) | phase output only (per-rule templates, 20/20); destination not found |
| [polynomial-differentiation](entries/polynomial-differentiation.md) | differentiate an integer polynomial by the power rule, degree 3, coefficients as tokens; answer only and scratchpad; 3-layer 1-head net | frozen fetch + one trained MLP learns the low-bit multiplier (PART 10/10 per format, never the unseen bit); from scratch 6/10 (T final, 2× width, 60k): failed seeds miss only the position-indexed coefficient copy; no length/value transfer | 10 (PART A/T and TRAIN T finals), 3 per other cell | yes (PART): MLP2 = k·a on the two seen bits, 1.000 on 40/40 snapshots (zero/shuffle ≤ .002); TRAIN none |
| [polynomial-integration](entries/polynomial-integration.md) | integrate an integer polynomial, coefficients as reduced fractions c/j; answer only and subtractive-Euclid scratchpad; 4-layer net | all screening cells solved 3/3, T TRAIN ten-seed final 9/10 (unsolved seed fails only on the signed numerator); scratchpad cuts PART steps 26k→5.5k and alone transfers to withheld fractions (T PART 1.0, others ≤ .19); TRAIN 0 beyond 4 terms; trained T nets: layer 0 = prefix pooling, one seed's layer-2 coefficient reader | 10 (T TRAIN final), 3 per other cell | partial: layer-0 pooling formula 6/6 snapshots, coefficient reader 1/3 seeds (512/512); no gcd closed form |
| [grid-bfs](entries/grid-bfs.md) | shortest-path distance on 6×6 mazes; answer only and reached-set-map scratchpad; 3-layer hand-built / 4-layer trained net | scratchpad learned from scratch 3/3 with 3× data (grok 63.5–72k), answer only 0/3 (exact to distance 3, falls with distance: depth limit); PART update MLP 3/3 in 500 steps; trained T uses the update variables in a distributed code | 3 per cell (ten-seed final not run) | partial: PART update = three-input hinge Y = (1, a, w)·B, 176/176 on 6/6 snapshots (controls open); TRAIN candidates rejected |
| [dpll-search](entries/dpll-search.md) | DPLL on 3-SAT (n 4–6, m 10–25); answer only and solver-state scratchpad; 2-/6-layer hand-built nets | trained last MLP completes both hand-built solvers 3/3 (A 1k, T 5.5k steps); from scratch 0/3 (A .72–.86, T ≤ .02 complete traces: state update not learned) | 3 per cell (ten-seed final not finished) | T PART learned MLP5 = parameter-free seven-field state update (52/52 traces per seed; zero/shuffle 0/52); 3 bottlenecks rejected; A PART threshold open |
| [chess-mate-in-one](entries/chess-mate-in-one.md) | first mating move among K ≤ 16 supplied checks, bare black king; answer only and escape-count scratchpad; 8-layer hand-built net | trained final MLP decodes the hand-built escape count 3/3 per format; K ≤ 8 training fails K ≤ 16 stress (52–64%), coverage retry fixes A (608/608); TRAIN not yet run to completion | 3 per cell (no ten-seed final) | partial: tent-basis decoder, A 442 scalars on 6/6 snapshots incl. stress; T 7,693 scalars local only (fails stress on 2 snapshots) |
| [linear-systems-finite-fields](entries/linear-systems-finite-fields.md) | solve A x = b over F₅/F₇, m = 2/3, by Gauss-Jordan; answer only, matrix-update trace and microstep scratchpad; 5-layer net | fused-update PART bypasses the hand-built subtractor (one family, 12/12 seed-runs; solved only F₅m3 3/3); microstep PART keeps the arithmetic-to-router split (4/6 solved); from scratch the microstep format solves every screen (3/3) and the F₅m2 ten-seed final (10/10), the update trace only after a budget retry at m = 2 (3/3, 2/3), answer only never | 10 (F₅m2 S TRAIN final), 3 per other cell | yes, T PART only: MLP3 categorical surrogate 1024/1024 in 24/24 snapshots; S PART three-map form 8/8 snapshots but no shuffle control; TRAIN formula rejected |
| [quadratic-roots](entries/quadratic-roots.md) | integer discriminant/isqrt/roots of x² + bx + c; answer only, sub-result, full trace and an addressed-NAND register-executor scratchpad; 4-layer parent / 2-layer executor nets | executor PART 10/10 in 500 steps (exact at widths 7/8 and in own rollouts): addressed AND-hinge bank with NAND readout; executor TRAIN 0/10 (never fetches registers); every parent PART/TRAIN cell 0/10 (memorises or underfits large-scalar thresholds) | 10 per final | yes: executor stage → one neuron ReLU(A + B − 1), 1,792/1,792 per seed, 10/10; parent cells none (answer-only n 6 final pending) |
| [crc-remainder](entries/crc-remainder.md) | GF(2) polynomial remainder N mod D (degree-4 divisors, 8–10 input bits); answer only and quotient/remainder scratchpad; 3-/4-layer nets | scratchpad step r = ((r ≪ 1) \| a) XOR (q·D) learned with frozen routing 10/10 in 500 steps and from scratch 3/3 (MLP → head → MLP path); answer only PART 3/3, TRAIN 2/3, last-layer divisor-conditioned parity readout, not the hand-built staircase | 10 (PART T), 3 elsewhere | yes: PART T eight-ReLU table 11,587/11,587 (s0–2; s3–9 zero controls open); PART A native ridge bank; TRAIN none |


Planned: counting and bracket tasks.

## Cross-entry findings

Surprising results across entries, including early ones from tasks still running, are collected in [SURPRISES.md](SURPRISES.md).

- Frozen exact encodings can hurt: on compare-counts and open-bracket-count a *random* frozen base (which encodes the integers
  linearly) generalizes better than the exact one-hot/thermometer menu, which invites memorization. Only residue tasks need the exact codes.
- Equality never generalizes as its own learned class; a hand-built comparator that makes `=` the default ("neither greater nor less") is exact.
- Frequency use differs by domain: modular addition spreads over 22–27 frequencies; the non-modular count uses only the lowest 1–3.
- Circuit probes ("Circuit found in trained nets" sections): component substitutions on trained nets usually preserve accuracy one
  at a time, but scrubbing the whole proposed circuit at once fails in the group-automata, base-conversion, division, subtraction and
  decimal/hex-addition probes; where it holds on the full net (binary addition, GF(2) trace) it fails on the greedily pruned circuit.
  A working replacement is a sufficient readout, not a verified mechanism.

## License

Text (entries, METHODS.md, README): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code in `code/`: [MIT](code/LICENSE).
