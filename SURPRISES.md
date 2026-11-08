# Findings that went against expectation

This file collects results that contradicted what the hand-built designs, earlier entries or the literature led us to expect. Each
finding is labelled *verified* when the observation was measured on all seeds of the named cells with the named check, or
*preliminary* when it rests on the first seeds or one format, in which case the confirming experiment is named. The label applies to
the observation. Where a finding involves a closed form, its status is given in the entry's own words (partial, sufficient readout),
because "verified" for a closed form is reserved for the acceptance rule of
[METHODS §9](METHODS.md#9-verification-and-the-acceptance-rule), which no closed form in the atlas yet meets in full. Entries are
updated as checks finish, and the numbers below follow the entries where both give them.

## Frozen bases and exact encodings

### A random frozen base can beat an exact hand-built encoding

*Verified; [compare-counts](entries/compare-counts.md), [open-bracket-count](entries/open-bracket-count.md).* A random frozen network
encodes the integers linearly, and a small trained stage reads that code well. In compare-counts the random base reached .99–1.000
against .91–.96 for the hand-built menu; in open-bracket-count it was better than the menu at train fractions .4 and .6. Exact one-hot
and thermometer codes invite memorisation instead.

### Equality is never learned as its own class

*Verified, 30/30 networks; [compare-counts](entries/compare-counts.md).* No trained network answered a single held-out `=` cell
correctly. Equality is exact only when it is built as "neither greater nor less".

### Exact hand-built circuits need full-precision matrix multiplication

*Verified on binary addition and base conversion.* With TF32 tensor-core matrix multiplication (a 10-bit mantissa), hand-built adders
lose exactness from 7 bits and the converters drop to 2% exact; in full fp32 they are exact. Large hand-set weights and sharp
thresholds need full precision, so results on frozen hand-built parts are reported in fp32.

## Scratchpads and length

### A scratchpad shrinks the hand-built circuit by about an order of magnitude

*Verified on the hand-built networks, ablation size on 500 inputs;
[long-division-square-root](entries/long-division-square-root.md), [multiplication](entries/multiplication.md).* Long division needs
186 neurons with the answer only and 14 with the steps written out, and long-hand square root 46 against 20. Binary multiplication
needs one attention head per input bit with the answer only and a constant 7 heads with a running-sum scratchpad. In the scratchpad
networks one compare-and-subtract stage serves every step, the same for division and square root.

### Stages with a scratchpad generalised worse to longer inputs than answer-only stages

*Preliminary; decimal subtraction trained on 3 digits, stage trained on frozen parts; [subtraction](entries/subtraction.md).*
Answer-only stages reached .80–.98 at 4 digits and .80–.96 at 6; the scratchpad formats reached at most .20 at 4 digits. The trained
stage reads absolute position to tell which step it is in, and holding that input at its mean drops accuracy from .99 to .01, whereas
the hand-built network reads only the position relative to the operator. Masking the stage's position inputs (absolute position,
operator anchor, relative offset) largely restores generalisation: answer only .99–1 at 4 digits and .98–1 at 6; the place-value
format .95–.99 at 4 digits (from at most .02) but only .52–.59 at 6; the borrow-chain format .87–.98 at 4 digits (from at most .20)
and .39–.60 at 6. Position is therefore the main cause but not the whole story; the remaining 6-digit gap is unexplained, since all
inputs stay in the trained range.

### The same position shortcut appears in binary addition

*Verified, 10 seeds per format, 6-bit stage trained on frozen hand-built parts; [binary-addition](entries/binary-addition.md).* With
the stage's position inputs held at zero, the place-value format becomes exact at 7, 8 and 12 bits in 10/10 seeds (unmasked .17–.20,
.01–.02 and 0), and its trained stage computes the hand-built function. The carry-chain format reaches .99–1.00 at 7 bits and
.53–1.00 at 12 (unmasked .00–1.00). The remaining 12-bit errors of the carry-chain format are unexplained.

## Number formats

### Only the most local conversion is learned from scratch

*Verified, 10 seeds per setting; [base-conversion](entries/base-conversion.md).* Hex to binary groks in a 2-layer network trained on
half of the 256 inputs (answer only 1/10 seeds, and 6/10 with 80% of the data; scratchpad 2/10 and 7/10). Every carry-type
conversion (binary to decimal, and decimal to binary in both formats) memorises its training inputs and stays far from test exact
match .99 in all 80 TRAIN networks and 30 PART networks. An optimiser sweep on decimal to binary (answer only, from scratch; 17
settings at 50% and the best 5–6 at 70% and 90%, varying learning rate, weight decay, warmup, cosine decay and clipping, 3 seeds each)
does not change this: with 50%, 70% or 90% of the inputs in training no seed reaches test .99 (best mean .40 at 90%, where the test set
is only about 26 inputs). The same sweep solves subtraction, binary addition and multiplication, so the limit here is the data rather
than the optimiser (verified on decimal to binary only).

### Trained networks carry the hand-built variable but are 50–100x larger

*Verified, 36 of 40 hex-to-binary networks; [base-conversion](entries/base-conversion.md).* Replacing the residual after layer 1 by
its fit on HAND's one-hot of the source hex digit keeps exact match at 1.00, yet the networks use 4–8 heads and 310–490 neurons where
the hand-built network uses 1 head and 4–7 neurons. The substitution supports the variable; a joint scrub of the nibble-only story
fails (.556 / .923), so the circuit is not verified.

### Given the exact remainder, the network learns a threshold near, not at, the true one

*Verified, decimal to binary answer only, frozen heads, 10/10; [base-conversion](entries/base-conversion.md).* Each answer bit is one
threshold with theta in [.94, 1.06] instead of 1; the errors are the unseen inputs whose remainder falls between theta and 1.

## Arithmetic

### Trained addition networks learn the column-sum lookup but not the carry-lookahead head

*Verified, 10 seeds per cell; [decimal-hex-addition](entries/decimal-hex-addition.md).* With frozen hand-built parts the trained stage
reads X1 + X2 + carry with equal weights in 60/60 networks and reaches .95–1.00. Trained from scratch, 37/40 networks of the retry
runs (40k steps, warmup, cosine schedule, clipping) reach exact match .998 or more, yet none of them is separated from the failed
first-run networks by a hand-built-like chain head, and none generalises to a fourth digit (n + 1 at most .04), while the answer-only
stage on frozen parts extrapolates (.93–1.00).

### More data, not a scratchpad, rescues answer-only long division

*Verified, 10 seeds per cell; [long-division-square-root](entries/long-division-square-root.md).* Binary long division with the answer
only, trained from scratch on 8-bit dividends (1920 training pairs), solved 0/10 seeds, with test .64–.81 at train 1.00
(memorisation). With 12-bit dividends (30720 pairs, 16x) it solved 10/10 at test ≥ .99 (.9947–.99997; exact on every test pair in
0/10). The prediction written before the rerun ("answer only stays below .5: no scratchpad") failed. Long-hand square root with the
answer only rose from .10–.28 at 8 bits (128 numbers) to .981–.993 at 14 bits (11468 numbers, 90x) in all 10 seeds (2/10 ≥ .99), and
division with the steps written out from .53–.69 to .877–.999 (7/10 ≥ .99). The same holds for an automaton: answer-only parity of a
word (lengths 1–16) trained on 20k fixed words stays at chance (.501–.524, 0/10), while with fresh words in every batch it is exact in
10/10, but only in the training range (32 symbols .34–.55, 64 at chance).

### The stage on frozen parts fails the same task on threshold precision

*Preliminary; one task, two runs of 10 seeds;
[long-division-square-root](entries/long-division-square-root.md).* On the wide-range square root the stage trained on frozen
hand-built parts receives every input it needs (R² 1.00 at its input) yet scores .000–.006 with a constant learning rate. With warmup,
cosine annealing and clipping it reaches only .12–.29, and the test accuracy rises only in the last 6k of 30k steps as the learning
rate goes to zero. The root bits are right (.99–1.00), but the remainder bits fall with significance (from .98 to .61): the stage must
place a threshold at unit resolution on inputs of size 2^14. Wide-range division on frozen parts likewise scores .000–.004. The
confirming experiment, a longer annealing tail, lr 1e-4 or a rescaled stage input, has not been run.

### With a running-sum scratchpad, the multiplication step is the adder's step

*Verified as an observation; closed form a sufficient readout only; [multiplication](entries/multiplication.md).* The stage trained on
frozen hand-built parts is a lookup on x + b_j a + carry in 10/10 networks (substitution 1.00). Trained from scratch (retry run), 8/10
networks reach exact match 1.0 and 9/10 ≥ .99, and the last MLP has a closed form in 8/10: the adder key itself in 5/10 (substitution
1.00) and a 6-input table or an (x, y, carry) key in 3/10. The audit of 2026-11-07 found that none of these closed forms passes the full
bar. Answer-only multiplication trains in neither setting (test at most .80 from scratch, memorised).

## Automata

### A network that fails on the permutation automaton learns its sign first

*Verified, 10 seeds; [dfa-group-automata](entries/dfa-group-automata.md).* Trained from scratch to give the final state of the
6-state S3 permutation automaton (answer only, words of 1–16 symbols, 20k fixed words), the network plateaus near .33 test accuracy
(best .21–.37). The predicted state has the right sign, even or odd permutation (the quotient of S3 onto Z2), in 8/10 seeds
(.99–1.00), learned at 4–10k steps (test rising from .17 to .33), and the rest of the training set is memorised. The sign is bound to
the training length (32 symbols .40–.71, 64 symbols .48–.52). The same architecture trained on the parity automaton, which is the same
Z2 count, never learns it (chance). With fresh words in every batch the sign is 1.00 in 10/10.

## Tasks without an entry yet

### In chess attack maps the nearest-blocker attention never forms

*Preliminary.* For attack maps (the squares a colour attacks) of one piece, of sliding pieces and of random positions, no trained head
matches a hand-built ray head: attention JS is .45–.66 to every hand-built head, and 0 of 12 heads fall below .2 in all 30 networks
trained on frozen parts. The nearest-blocker distance is not linear in the residual (R² .03–.38), and no network reaches exact maps
(best .37 on single pieces, at baseline on random positions). Only 3 networks trained from scratch have a single ray-like head. The
confirming experiments, a scratchpad that writes each ray step and 2x width, have not been reported yet.

### Weight decay erases hand-built circuits used as initialisation

*Preliminary; one task, binary to decimal answer only.* Starting from the hand-built network with 2x fresh units and training
everything with weight decay 1, the hand-built weights drift completely away (relative drift 1.0, 10/10 networks), and the fresh units
take over (output norm 1.6–2.5x that of the hand-built part). With decay on the fresh units only, the hand-built weights stay (drift
.012–.019). Neither version beats training the stage on frozen parts (.56–.76), because the added width memorises the 128 examples. The
confirming experiment is the same drift check on the square-root and multiplication seeds.

## Representations and wiring

### Digit embeddings trained from scratch are periodic, but no clock neurons compute the sum

*Preliminary; [decimal-hex-addition](entries/decimal-hex-addition.md).* In the networks trained from scratch the digit-token
embeddings are periodic in the digit value: the top two DFT frequencies (1 and 2 or 3) hold .71–.97 of their power. The column sum is
still assembled by a lookup in the last MLP, as on frozen parts. This was measured on the archived networks (best and worst seed per
cell and run); the confirming experiment is the same DFT on every seed and a substitution of the embeddings by their top two Fourier
terms.

### Failed networks trained from scratch still use the hand-built variables

*Verified on binary addition, decimal and hexadecimal addition and long division and square root; 981 networks across tasks.* A
variable counts as used when swapping it between two inputs (by linear interchange, or by DAS where interchange reads low) flips the
answer in at least half of the cases. Failed against solved networks, 116/127 against 135/166 variable tests pass in binary addition,
185/204 against 155/156 in decimal and hexadecimal addition, and 39/45 against 10/10 in division and square root. Removing the probed
variable subspace breaks both groups alike, and keeping only that subspace preserves both. Yet the failed networks end at test .33,
.04 and .28. The networks fail to generalise, not to compute the variable. Where a variable really is missing, the tools localise it:
in answer-only binary multiplication the middle partial-product bits, which have the largest fan-in, are the ones not fetched (use .91
for the first bit, .52 for the eighth).

### Same variables, different wiring

*Verified on binary addition and decimal and hexadecimal addition; edge attribution trusted in 155/184 networks.* Networks trained from
scratch share only 6–9% of their top circuit edges with the hand-built network (Jaccard), solved and failed alike, and no rotation or
permutation maps their weights onto it (mean overlap .03–.12); networks trained on frozen hand-built parts share 79–100%. Training
recovers the hand-built variables but not its heads, edges or weights.

### Each seed trained from scratch finds its own solution; seeds on frozen parts share one

*Preliminary; 22 seed pairs over six tasks.* Along the linear interpolation between two seeds of one cell, the loss barrier on frozen
parts is .0001–.09 after a permutation alignment. Between two solved networks trained from scratch it is 1.1–8.2 (median 1.8), and no
alignment lowers it, although both use the same variables (above). The confirming experiment is more pairs per cell and the same check
on the tasks without an entry yet.

### Swapping one part at a time does not verify a circuit

*Verified in the entries' mechanism sections.* Component substitutions on trained networks usually keep accuracy one at a time, but
scrubbing the whole proposed circuit at once fails in [group automata](entries/dfa-group-automata.md),
[base conversion](entries/base-conversion.md), [division](entries/long-division-square-root.md),
[subtraction](entries/subtraction.md) and [decimal and hexadecimal addition](entries/decimal-hex-addition.md). Where substitution
holds on the full network ([binary addition](entries/binary-addition.md), [GF(2) trace](entries/gf2-matrix-product.md)), it fails on
the greedily pruned circuit. A working replacement is a sufficient readout, not a verified mechanism.

### Frequency use differs by domain

*Verified; [modular-addition-mlp](entries/modular-addition-mlp.md), [open-bracket-count](entries/open-bracket-count.md).* Modular
addition spreads its neurons over 22–27 frequencies, while the non-modular count uses only the lowest 1–3.
