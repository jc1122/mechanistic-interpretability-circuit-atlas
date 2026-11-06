# Mechanistic Interpretability Circuit Atlas

An encyclopedia of circuits that neural networks use to solve well-defined tasks.

Each entry describes one task and the circuit types that trained networks converge to. It has three parts:

- **Hand-built reference:** a network whose every weight is set by a formula, exact on the task.
- **What training finds:** many seeds, both fully trained from scratch and trained on top of frozen hand-built parts. Each
  entry gives how often each circuit type appears, which frozen parts training chooses to reuse, and how fast it generalizes (grokking).
- **Closed form:** a formula for the trained part. It is accepted only if *substituting* it for the trained weights keeps the accuracy.

Entries call the three network settings **HAND** (hand-built), **PART** (frozen hand-built parts plus one trained stage) and **TRAIN** (trained from scratch).

**Weights:** the trained networks and frozen inputs behind each entry are on Hugging Face: [https://huggingface.co/jc1122/circuit-atlas-nets](https://huggingface.co/jc1122/circuit-atlas-nets) (safetensors, CC BY 4.0).

Entries are written as encyclopedia articles: the mechanism in plain words, the equations, the evidence (ablations, accuracies,
seed counts) and the limits. [METHODS.md](METHODS.md) describes the architecture, the experimental conditions and the instrumentation common to all entries, and
[code/](code/) is a self-contained CPU reimplementation (Python, torch) that rebuilds the hand-built network, generates the synthetic data,
trains and runs the analysis; every entry has a Setup section with the command that reproduces it.

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
| [subtraction](entries/subtraction.md) | a − b (a ≥ b), 3 digits base 10 and 6 bits base 2; answer only, borrow token, signed column differences; 4-layer net | per-column lookup on a − b − borrow (not the adder's a + b + c) in all 60 PART nets but one; TRAIN solves all six cells after one retry, same key in the last MLP in 84/100 nets; position shortcut breaks length unless masked | 10 per setting | yes, = net accuracy (PART .97–1.00) |
| [multiplication](entries/multiplication.md) | a · b, 6 bits; answer only and running-sum scratchpad; 4-layer net | scratchpad step = the adder's sum lookup on x + b_j a + carry (PART 10/10, TRAIN 5/10, closed form 8/10); answer only never trains (parity staircase not fitted, TRAIN memorises) | 10 per setting | yes, 1.00 (scratchpad PART) |
| [long-division-square-root](entries/long-division-square-root.md) | binary N / D and long-hand square root of N, 8-bit and wide ranges; answer only and per-step scratchpad; 3-layer net | scratchpad step = one shared compare-and-subtract [x ≥ T], x − bit·T (division B PART 10/10 exact, square root B .985–.992); answer-only PART never places the thresholds (≤ .29); TRAIN generalises only with 16–90x more data (division A 10/10 ≥ .99) | 10 per setting | yes, 1.00 (scratchpad PART; nets 1.00 / .985–.992) |
| [dfa-group-automata](entries/dfa-group-automata.md) | state of a parity, Z3 or S3 automaton after a word of 1–16 symbols; answer only and per-symbol state scratchpad; 6-layer net | scratchpad: transition table δ(q_prev, s) learned at once (PART and TRAIN 30/30 per format), length set by the previous-state fetch (PART n 128, TRAIN ~2x); answer-only: S3 PART = hand-built prefix-doubling tables (10/10); abelian stage f(S) misses large S; TRAIN count → residue in range (parity, Z3 10/10 with fresh words), S3 3/10 | 10 per setting | yes, δ table 1.00 (B); f(S) tables = net accuracy (A) |
| [gf2-matrix-product](entries/gf2-matrix-product.md) | C = A B over GF(2), 3×3 bit matrices; answer only and running-parity trace; 4-layer net | trace step = 3-bit XOR-AND lookup on (s_prev, A_ij, B_jk): PART reads the hand-built fetch heads (10/10); TRAIN trace 10/10 computes the same variables with fetch heads one layer later, different wiring; answer only 10/10 (13–14 heads); no length generalisation | 10 per setting | yes, 1.00 (trace PART) |
| [gf2-elimination](entries/gf2-elimination.md) | solve A x = b over GF(2) by Gauss-Jordan, m = 3 and 4; per-step scratchpad and answer only; 4-layer net | row update = XOR-AND lookup OLD ⊕ (K ∧ PB ∧ ¬EQ) on the hand-built fetches (PART and PARTp 10/10 each); TRAIN at m = 3 memorises the 118 training matrices (5/10 per format); with 6000 matrices at m = 4 answer only solves 10/10 with a distributed circuit; no size generalisation | 10 per setting | yes, 1.00 (PART, 20/20) |
| [router-addition](entries/router-addition.md) | a + b, 16 bits and 6 decimal digits; answer only (both digit orders), inductive scratchpad, router scratchpad (op tokens + coupled digit ids); 4-layer net | one shared column step (sum / carry / next-id groups); PART length-generalises when its stage cannot read position (R unmasked, I/A masked: 10/10 at 2N); TRAIN solves in range (all but binary MSB-first answers) but reaches 2N in no format, router included; trained nets deliver operands by input-dependent attention mixtures | 10 per setting | yes, 1.00 in range (PART R stage; PART AL spline) |
| [gcd-euclid](entries/gcd-euclid.md) | gcd(a, b) by subtractive Euclid (plus modular inverse, fraction reduction), operands 5–7 bits as scalar tokens; answer only and Euclid trace; 4-layer net | PART on frozen Euclid parts = a learned 3-rule router (T 13/13 seeds ≥ .99, 7/10 exact in the final cell); TRAIN trace 1/3 (6 bits), 2/3 (7 bits) with a distributed token-histogram front end; answer-only TRAIN 1/3 with parity/divisible-by-3 features; no 8-bit transfer | 10 (T PART final), 3 per other cell | yes, router 1.00 (all PART cells incl. inverse/fraction); TRAIN front end only (.99/.93) |
| [modular-arithmetic-mod-97](entries/modular-arithmetic-mod-97.md) | a + b, a · b mod 97 and x² mod 97 in residue tokens, binary, decimal, hex and decimal scratchpad; 4-layer net | token multiplication: discrete-log clock at the output (6/10 at 90% data, 0/10 at 30%), embeddings periodic in log₅ v in 10/10, hand-built uses a log ramp + lookup instead; binary/decimal/hex never form circles; digit multiplication memorises (operands decodable, residue not); binary addition MLP0 = prefix-count rectifier (5/10); scratchpad PART fails at its state writer | 10 per setting (scratchpad 3) | yes, output surrogate 9409/9409 with 1–6 frequencies, agreement ≥ .9947 (6/6 solved seeds); internal circuit not recovered |
| [chess-krk](entries/chess-krk.md) | king + rook vs king mate: move of a fixed 8-rule fence policy; answer only, rule-router scratchpad, scratchpad + candidate list; 6-layer net | PART (frozen hand-built rule features + trained router) .993–.999 in 60/60 seed-runs, never exact; TRAIN scratchpad .993–.998 (3/3 ≥ .99 at 120k steps), answer only .97 (0/3) and wins far fewer games; trained nets code kings as coordinates, last MLP not additive in the hand-built variables | 3–10 per setting | partial: PART phase decoder 20/20; destination not |
| [chain-rule-differentiation](entries/chain-rule-differentiation.md) | symbolic derivative of prefix expressions (x^1–x^8, sin, cos, exp, +, *), depth ≤ 2; answer only (15-layer net) and per-node register-rewrite scratchpad (5-layer net) | TRAIN solves both formats (A 3/3, T 3/3 and 10/10) with a content-weighted token-histogram front end; PART (trained emitter on the frozen hand-built parser) fails 12/12 seed-runs: it bypasses the frozen address interface; no depth-3 or power 7–8 transfer | 10 (T TRAIN final), 3 per other cell | front end only (histogram heads 1.00, 3/3); last-MLP state table exact but not compact (rejected) |
| [polynomial-add-mul](entries/polynomial-add-mul.md) | integer polynomial addition and multiplication (convolution), degree 3, signed coefficients as scalar tokens; answer only (3-layer net) and partial-product scratchpad (4-layer net) | addition learned in every seed (PART and TRAIN, A and T, 10/10); multiplication not learned (PART whole stage 0/3, TRAIN A 0/3, TRAIN T 2/3 non-exact only at 60k) unless the hand-built signed-bit product gates stay frozen (split-sum A 3/3 at .999); no degree transfer | 10 (add PART/TRAIN finals), 3 per other cell | yes, split-sum read-out 123/39 numbers, 12/12 no-drop; TRAIN add A last MLP lossy (≥ 99% of original, 6/6) |
| [output-order-base-conversion](entries/output-order-base-conversion.md) | 16-bit decimal to binary and binary to decimal, answer MSB-first or LSB-first; answer only and halving/doubling scratchpad; 3-layer net | MSB-first answer only = threshold ladder on the remaining magnitude, learned (PART all seeds ≥ .99, TRAIN decimal to binary 10/10 ≥ .996); LSB-first answer only needs a residue (mod) read-out and mostly fails (binary to decimal TRAIN ≤ .045); scratchpads solve both orders (PART 10/10) | 10 per A cell, 10 trained / 3 archived per B cell | yes, PART thermometer (366 thresholds) and failing parity spline reproduce every token, 12/12; TRAIN scalar template rejected 0/12 |
| [symbolic-algebra-atoms](entries/symbolic-algebra-atoms.md) | collect like terms, expand a binomial product, solve ax+b=cx+d, on (side, power, coefficient) term atoms; answer only and scratchpad; 2-3-layer net | collection learned at every rung (PART 10/10 in 1k steps, TRAIN 3/3); expansion not learned answer-only (PART ≤ .96, TRAIN ≤ .13), TRAIN with scratchpad solves it at 60k (3/3); solving learned from scratch only with the scratchpad (3/3 exact), PART quadratic stage lacks precision | 3 per cell (PART collect T 10) | yes, PART collection sum / phase-gated copy-copy-sum, 4096/4096 in 6/6 seeds; no TRAIN closed form |
| [chess-krk](entries/chess-krk.md) | king + rook vs king, white to move, label = a fixed 8-rule fence policy (not DTM); answer only, fence-state scratchpad, scratchpad + candidate list; 6-layer net | PART (trained router on frozen hand-built rule features) .993–.999 in 60/60 seed-runs, reads rook rays and king distances; TRAIN scratchpad reaches .993–.998 (3/3) at 120k steps, answer only .970–.977 (0/3); trained nets code kings as coordinates, last MLP not additive in the HAND variables | 10 (PART, TRAIN 30k), 3 (TRAIN 60k/120k) | phase output only (per-rule templates, 20/20); destination not found |

Planned: counting and bracket tasks. See [TEMPLATE.md](TEMPLATE.md) for the entry format.

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
