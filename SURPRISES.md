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
