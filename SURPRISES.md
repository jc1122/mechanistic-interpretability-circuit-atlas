# Surprising results

Short notes on results that went against expectation. Each says whether it is **verified** (measured on all seeds, the check named) or
**preliminary** (first seeds or one format; the confirming experiment named). Entries are updated as checks finish.

- **A random frozen base beats an exact hand-built encoding** (verified; [compare-counts](entries/compare-counts.md),
  [open-bracket-count](entries/open-bracket-count.md)). A random frozen network encodes the integers linearly, which a small trained stage
  reads exactly; exact one-hot and thermometer codes invite memorization instead.
- **Equality is never learned as its own class** (verified, 30/30 networks): it is exact only when built as "neither greater nor less".
- **Exact hand-built circuits break under reduced-precision matrix multiplication** (verified on binary addition and base conversion). With
  TF32 tensor-core matmuls (10-bit mantissa) hand-built adders lose exactness from 7 bits and converters drop to 2% exact; in full fp32
  they are exact. Large hand-set weights and sharp thresholds need full precision, so results on frozen hand-built parts are reported in fp32.
- **A scratchpad shrinks the circuit by up to an order of magnitude** (verified on the hand-built networks, ablation size on 500 inputs).
  Long division: 186 neurons answer-only vs 14 with the steps written out; long-hand square root: 46 vs 20; binary multiplication: one
  attention head per input bit answer-only vs a constant 7 heads with a running-sum scratchpad. In the scratchpad networks one
  compare-and-subtract stage serves every step, the same for division and square root.
- **…but the scratchpad networks trained on top of frozen parts did not generalize to longer inputs, while answer-only did**
  (preliminary, decimal subtraction trained on 3 digits). Answer-only: .80–.98 at 4 and 6 digits; scratchpad formats: ≤ .20 at 4 digits.
  The trained stage reads absolute position to tell which step it is in; holding that input at its mean drops accuracy from .99 to .01.
  The hand-built network reads only position relative to the operator. Masking the trained stage's position inputs (absolute position,
  operator anchor, relative offset) largely restores generalization: answer-only .99–1.0 at 4 digits and .98–.99 at 6; the place-value
  format .95–.99 at 4 digits (from .01–.02) but only .52–.59 at 6, so position is the main cause, not the whole story; the borrow-chain
  format likewise .87–.98 at 4 digits (from ≤ .20) and .39–.60 at 6. The remaining 6-digit gap is unexplained (all inputs stay in the
  trained range).
- **The same position shortcut appears in binary addition** ([binary-addition](entries/binary-addition.md); verified, 10 seeds per format, trained 6-bit stage on frozen hand-built parts).
  With the stage's position inputs held at zero, the place-value format becomes exact at 7, 8 and 12 bits in 10/10 seeds (unmasked: .17–.20,
  .01–.02, 0) and its trained stage computes the hand-built function; the carry-chain format reaches .99–1.00 at 7 bits and .53–1.00 at 12
  (unmasked .00–1.00). The remaining 12-bit errors of the carry-chain format are unexplained.
- **Only the most local conversion is learned from scratch** (verified, 10 seeds per setting; [base-conversion](entries/base-conversion.md)). Hex to binary groks in
  a 2-layer net trained on half of the 256 inputs (answer only 1/10 seeds, with 80% of the data 6/10; scratchpad 2/10 and 7/10), while every carry-type conversion
  (binary to decimal, decimal to binary in both formats) memorizes the training inputs and stays far from test exact match .99 in all 80 nets and 30 PART nets.
  An optimizer sweep on decimal to binary (answer only, from scratch; 17 settings at 50% and the best 5–6 at 70% and 90%, varying learning rate, weight decay, warmup, cosine decay and
  clipping, 3 seeds each) does not change this: with 50%, 70% or 90% of the 256 inputs in training no seed reaches test .99 (best mean .40
  at 90%, where the test set is only about 26 inputs). The same sweep solves subtraction, binary addition and multiplication, so the
  limit here is the data, not the optimizer (verified on decimal to binary only).
- **Trained nets carry the hand-built variable but are 50-100x larger** (verified, 36 of 40 hex-to-binary nets). Replacing the residual after layer 1 by its fit on
  HAND's one-hot of the source hex digit keeps exact match at 1.00, yet the nets use 4-8 heads and 310-490 neurons where the hand-built net uses 1 head and 4-7 neurons.
- **Given the exact remainder, the net learns a threshold close to, but not at, the true one** (verified, decimal to binary answer only, frozen heads, 10/10): each answer bit is
  one threshold with theta in [.94, 1.06] instead of 1; the errors are the unseen inputs whose remainder falls between theta and 1.
- **Trained addition nets learn the column-sum lookup but not the carry-lookahead head** (verified, 10 seeds per cell; [decimal-hex-addition](entries/decimal-hex-addition.md)). With frozen hand-built parts the trained
  stage reads X1 + X2 + carry with equal weights (60/60 nets) and reaches .95-1.00; trained from scratch, 37/40 nets of the retry runs (40k steps, warmup, cosine, clip) reach exact match .998+, yet none of them is separated from the
  failed first-run nets by a HAND-like chain head, and none generalizes to a fourth digit (n + 1 <= .04), while the frozen-part answer-only stage extrapolates (.93-1.00).
- **More data, not a scratchpad, rescues answer-only long division** (verified, 10 seeds per cell; [long-division-square-root](entries/long-division-square-root.md)).
  Binary long division, answer only, trained from scratch: 8-bit dividends (1920 training pairs) 0/10, test .64–.81 with train 1.00 (memorisation);
  12-bit dividends (30720 pairs, 16x) 10/10 at test ≥ .99 (.9947–.99997; exact on every test pair in 0/10). The prediction written before the
  rerun ("answer only stays below .5: no scratchpad") failed. Long-hand square root, answer only: 8-bit (128 numbers) .10–.28 → 14-bit
  (11468 numbers, 90x) .981–.993 in all 10 seeds (2/10 ≥ .99). Division with the steps written out: .53–.69 → .877–.999 (7/10 ≥ .99). The same
  holds for an automaton: answer-only parity of a word (lengths 1–16) trained on 20k fixed words stays at chance (.501–.524, 0/10); with fresh
  words in every batch it is exact in 10/10, but only in the training range (32 symbols .34–.55, 64 chance).
- **…while the frozen-part version of the same task fails on threshold precision** (preliminary; one task, two runs, 10 seeds each). On the
  wide-range square root the stage trained on top of frozen hand-built parts gets every input it needs (R² 1.00 at its input) yet scores
  .000–.006 (constant learning rate); with warmup, cosine annealing and clipping it reaches only .12–.29, and the test accuracy rises only in the
  last 6k of 30k steps as the learning rate goes to zero. The root bits are right (.99–1.00); the remainder bits fall with significance (.98 → .61):
  the stage must place a threshold at unit resolution on inputs of size 2^14. Wide-range division on frozen parts likewise scores .000–.004.
  Confirming experiment (not run): a longer annealing tail or lr 1e-4, or a rescaled stage input.
- **With a running-sum scratchpad, the multiplication step is the adder's step** (verified; [multiplication](entries/multiplication.md)).
  The stage trained on frozen hand-built parts is a lookup on x + b_j a_{k−j} + carry in 10/10 nets (substitution 1.00); trained from scratch
  (retry run) 9/10 nets solve the task and the last MLP has a closed form in 8/10: the adder key itself in 5/10 (substitution 1.00), a 6-input
  table or (x, y, carry) key in 3/10. Answer-only multiplication trains in neither setting (best .80 from scratch, memorised).
- **A net that fails on the permutation automaton learns its sign first** (verified, 10 seeds; [dfa-group-automata](entries/dfa-group-automata.md)). Trained from
  scratch to give the final state of the 6-state S3 permutation automaton (answer only, words of 1–16 symbols, 20k fixed words), the test accuracy
  plateaus near .33 (best .21–.37); the predicted state has the right sign (even or odd permutation, the quotient S3 → Z2) in 8/10 seeds (.99–1.00), learned at
  4–10k steps (test .17 → .33), and the rest of the training set is memorised. The sign is length-bound (32 symbols .40–.71, 64 .48–.52). The same
  architecture trained on the parity automaton, which is that same Z2 count, never learns it (chance). With fresh words in every batch the sign is 1.00 in 10/10.
- **Chess attack maps: the nearest-blocker attention never forms** (preliminary; tasks without an entry yet). For attack maps (squares a colour
  attacks) of one piece, of sliders and of random positions, no trained head matches a hand-built ray head (attention JS .45–.66 to every hand-built
  head; 0 of 12 heads below .2 in all 30 nets trained on frozen parts); the nearest-blocker distance is not linear in the residual (R² .03–.38) and no
  net reaches exact maps (best .37 on single pieces, at baseline on random positions). Only 3 nets trained from scratch have a single ray-like head.
  Confirming experiments (a scratchpad that writes each ray step, and 2x width): not reported yet.
- **Weight decay erases hand-built circuits used as initialisation** (preliminary; one task, binary to decimal answer only). Starting from the hand-built
  net with 2x fresh units and training everything with weight decay 1, the hand-built weights drift completely away (relative drift 1.0, 10/10 nets) and the fresh units take
  over (output norm 1.6–2.5x the hand-built part's); with decay on the fresh units only, the hand-built weights stay (drift .012–.019). Neither version
  beats training the stage on frozen parts (.56–.76): seeding adds width that memorises 128 examples. Confirming experiment: the same drift
  check on the square-root and multiplication seeds.
- **Digit embeddings trained from scratch are periodic, but no clock neurons compute the sum** (preliminary; [decimal-hex-addition](entries/decimal-hex-addition.md)).
  In the nets trained from scratch the digit-token embeddings are periodic in the digit value: the top-2 DFT frequencies (1 and 2 or 3) hold .71–.97
  of their power. The column sum is still assembled by a lookup in the last MLP, as on frozen parts. Measured on the archived nets (best and worst
  seed per cell and run); confirming experiment: the same DFT on every seed and a substitution of the embeddings by their top-2 Fourier terms.
- **Failed nets trained from scratch still use the hand-built variables; they fail to generalise, not to compute a variable** (verified on
  binary addition, decimal/hex addition and long division/square root; 981 nets across tasks). A variable counts as used when swapping it
  between two inputs (linear interchange, or DAS where that reads low) flips the answer in >= half the cases. Failed vs solved nets: 116/127 vs
  135/166 variable tests pass in binary addition, 185/204 vs 155/156 in decimal/hex addition, 39/45 vs 10/10 in division/square root; removing
  the probed variable subspace breaks both groups alike, keeping only it preserves both. Yet the failed nets end at test .33 / .04 / .28.
  Where a variable really is missing, the tools localise it: in answer-only binary multiplication the middle partial-product bits (largest
  fan-in) are the ones not fetched (use .91 for the first bit, .52 for the eighth).
- **Same variables, different wiring** (verified on binary and decimal/hex addition, edge attribution trusted in 155/184 nets). Nets trained
  from scratch share only 6–9% of their top circuit edges with the hand-built net (Jaccard), solved and failed alike, and no rotation or
  permutation maps their weights onto it (mean overlap .03–.12); nets trained on frozen hand-built parts share 79–100%. Training recovers the
  hand-built variables but not its heads, edges or weights.
- **Each seed trained from scratch finds its own solution; seeds on frozen parts share one** (preliminary, 22 seed pairs over six tasks).
  Linear interpolation between two seeds of one cell: on frozen parts the loss barrier is .0001–.09 after a permutation alignment; between two
  solved nets trained from scratch it is 1.1–8.2 (median 1.8), and no alignment lowers it, although both use the same variables (above).
  Confirming experiment: more pairs per cell and the same check on the tasks without an entry yet.
- **Swapping one part at a time is not verification** (verified in the entries' "Circuit found in trained nets" sections). Component substitutions on
  trained nets usually keep accuracy one at a time, but scrubbing the whole proposed circuit at once fails in the [group-automata](entries/dfa-group-automata.md),
  [base-conversion](entries/base-conversion.md), [division](entries/long-division-square-root.md), [subtraction](entries/subtraction.md)
  and [decimal/hex-addition](entries/decimal-hex-addition.md) probes; where it holds on the full net ([binary addition](entries/binary-addition.md),
  [GF(2) trace](entries/gf2-matrix-product.md)) it fails on the greedily pruned circuit. A working replacement is a sufficient readout, not a verified mechanism.
- **Frequency use differs by domain** (verified; [modular-addition-mlp](entries/modular-addition-mlp.md), [open-bracket-count](entries/open-bracket-count.md)):
  modular addition spreads over 22–27 frequencies; the non-modular count uses only the lowest 1–3.
