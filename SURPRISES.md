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
- **The same position shortcut appears in binary addition** (verified, 10 seeds per format, trained 6-bit stage on frozen hand-built parts).
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
