# Count mod 5: MLP on a frozen menu

**Task.** Prompt `COUNT b: a b\nANSWER: 1\nCOUNT a: <c a's and L−c b's, shuffled>\nANSWER: `, answer c mod 5 (5 classes). Cells (c, L)
with L = 1..45, c = 0..L (1080 cells, one prompt each); the train/test split is over cells, so every c value appears in training.
**Status.** 2026-11-04. 30 networks (10 seeds × train fractions .4/.6/.8) + 18 control runs. Trained part: one MLP layer, 256 ReLU,
AdamW lr 1e-3, weight decay 1.0, full batch, 200k steps.

## Summary
Every network learns a flat residue lookup on the count c and ignores the list length L. A two-cosine closed form with equal amplitudes
(equivalently, a delta on the residue) reproduces it exactly (1.000). There is no grokking delay: test accuracy equals train accuracy from
the first evaluation.

## Setup: frozen menu + trained stage
A frozen hand-built transformer (6 layers, every weight set by formula) computes the count c (a segment-match attention head gives
1/(c+1), an MLP step ladder decodes it) and the line length L. A frozen menu places three encodings of each integer at the answer
position, each centered and scaled to rms 1: the raw scalar, a one-hot and a thermometer, plus distractor features. Only the MLP on top is trained.

## What training finds
- **Learning curve.** Train 99% and test 99% both at step 500 for every fraction (final test .992–1.000). Not grokking.
- **Circuit type: residue lookup on c, L ignored — 30/30.** Setting the L input to its mean keeps .995–1.000; replacing each c embedding by
  its residue-class mean keeps 1.000.
- **Encodings read (fraction .4, test after ablation):** all c features ablated .16, all L features ablated .997; one-hot pair kept alone .997,
  thermometer alone .86, scalars alone .21. The one-hot is the primary input; the information is redundant across one-hot and thermometer.

## Closed form
logit_r(c) = A cos(2π(c − r)/5) + B cos(4π(c − r)/5), fitted per network: R² .989–.999, accuracy 1.000 in every network.
A ≈ B (e.g. 6.77 / 6.68); with A = B the profile is a delta (2 at c ≡ r, −1/2 elsewhere): a flat lookup, not a Fourier mechanism.

## Comparison
The hand-built count decoder is a step ladder on 1/(c+1), the same construction as the menu's one-hot and thermometer. The trained stage
reads the one-hot output and pools it by residue: one more linear map. No hand-built mod brick was given.

## Controls and limits
- Random frozen base (same shape): .24–.40 test. Its residual holds c linearly (ridge R² 1.000) but carries no residue feature.
- Scalar-only menu: .29–.34 (chance .2): 1/(c+1) cannot be cut into residues by 256 ReLUs at this weight decay.
- Open: holding out c values (not cells) to test period extrapolation was not run; why one-hot beats thermometer when both are exact.
