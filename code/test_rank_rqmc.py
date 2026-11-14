"""Tests of rank_rqmc.py (METHODS.md Section 10). SPDX-License-Identifier: MIT.

  python -m unittest test_rank_rqmc            # from this folder; the last test downloads the AlgZoo example
  RANK_RQMC_WEIGHTS=FILE python -m unittest test_rank_rqmc   # use a local copy instead (.pth or .safetensors)
"""
import itertools
import math
import os
import unittest

import numpy as np

import rank_rqmc as R

PHI = lambda z: .5 * (1 + math.erf(z / math.sqrt(2)))

# Recorded outputs of the study's frozen estimator, net 0 of each noise scale, n = 1024, repetitions 0 and 1:
# (arm, repetition, correct rows of the estimate, correct rows of its matched MC control).
FROZEN = {
    0: ("3ef708da66220157cce0a9f38da8c6ce1da6b627103373ab855a67f490e7a1ae",
        [("R", 0, 801, 1003), ("R", 1, 805, 992), ("RAW", 0, 830, 906), ("RAW", 1, 810, 893), ("RI", 0, 835, 898), ("RI", 1, 842, 895)]),
    1: ("14c368d230cc59f57e67ae21a377a479bee77df36df02437bb882350b32975b8",
        [("R", 0, 371, 453), ("R", 1, 369, 447), ("RAW", 0, 352, 399), ("RAW", 1, 374, 389), ("RI", 0, 342, 386), ("RI", 1, 365, 411)]),
    2: ("87513a075dbe5faa0bafd6413e659b17a2a6290150de5e5e65f098485295d13a",
        [("R", 0, 148, 192), ("R", 1, 155, 187), ("RAW", 0, 162, 157), ("RAW", 1, 170, 182), ("RI", 0, 142, 158), ("RI", 1, 150, 171)]),
}


def comparator(a):
    """Two-step, two-neuron ReLU RNN with known accuracy 1 - |atan(a) - pi/4| / pi on Gaussian pairs (a = 1: exact)."""
    return {"U": np.array([[1.], [-1.]]), "W": np.array([[-a, a], [a, -a]]), "V": np.array([[1., -1.], [-1., 1.]])}


def constant_net(T=10, h=4):
    """Zero readout: argmax is always position 0, so accuracy is exactly 1/T."""
    return {"U": np.ones((h, 1)), "W": np.zeros((h, h)), "V": np.zeros((T, h))}


class Lehmer(unittest.TestCase):
    def test_bijection_all_codes(self):
        for T in range(2, 7):
            codes = itertools.product(*[range(T - r) for r in range(T - 1)])
            perms = {tuple(R.lehmer_decode(c, T)) for c in codes}
            self.assertEqual(len(perms), math.factorial(T))
            self.assertEqual(perms, set(itertools.permutations(range(T))))

    def test_transform_places_ranks_by_code(self):
        for T in (3, 5):
            names = R.priority(T)
            for code in itertools.product(*[range(T - r) for r in range(T - 1)]):
                u = np.full((1, 2 * T - 1), .5)
                for r, k in enumerate(code):
                    u[0, names.index(("W", r))] = (k + .5) / (T - r)
                x = R.rank_transform(u, T)[0]
                descending = list(np.argsort(-x, kind="stable"))
                self.assertEqual(descending, R.lehmer_decode(code, T))


class OrderStatisticLaw(unittest.TestCase):
    def test_iid_gaussian_output(self):
        T, n = 10, 200_000
        x = R.rank_transform(np.random.default_rng(1).random((n, 2 * T - 1)), T)
        ks_bar = 1.95 / math.sqrt(n)  # about the 0.1% Kolmogorov-Smirnov level per coordinate
        phi = np.vectorize(PHI)
        for j in range(T):
            cdf = phi(np.sort(x[:, j]))
            i = np.arange(1, n + 1)
            self.assertLess(max(np.max(i / n - cdf), np.max(cdf - (i - 1) / n)), ks_bar)
        c = np.corrcoef(x.T)
        self.assertLess(np.max(np.abs(c - np.eye(T))), .015)
        # every position is equally likely to hold the second-largest value
        freq = np.bincount(np.argsort(x, axis=1)[:, -2], minlength=T) / n
        self.assertLess(np.max(np.abs(freq - 1 / T)), .004)

    def test_uniform_rank_permutation(self):
        T, n = 4, 240_000
        x = R.rank_transform(np.random.default_rng(2).random((n, 2 * T - 1)), T)
        perms = np.argsort(x, axis=1)
        keys = (perms * (T ** np.arange(T))).sum(1)
        counts = np.unique(keys, return_counts=True)[1]
        self.assertEqual(len(counts), 24)
        chi2 = float(((counts - n / 24) ** 2 / (n / 24)).sum())
        self.assertLess(chi2, 49.7)  # 0.1% upper point of chi-square with 23 degrees of freedom


class Unbiased(unittest.TestCase):
    def check(self, p, truth, n=256, reps=400):
        for arm in R.ARMS:
            vals = np.array([R.estimate(p, n, 1000 + k, arm)[0] for k in range(reps)])
            se = vals.std(ddof=1) / math.sqrt(reps)
            self.assertLessEqual(abs(vals.mean() - truth), 4 * se + 1e-10, arm)

    def test_exact_comparator(self):
        for arm in R.ARMS:
            self.assertEqual(R.estimate(comparator(1.), 256, 7, arm)[0], 1.)

    def test_known_p_comparator(self):
        self.check(comparator(2.), 1 - abs(math.atan(2.) - math.pi / 4) / math.pi)

    def test_known_p_constant_T10(self):
        self.check(constant_net(), .1)


class Cost(unittest.TestCase):
    def test_study_formulas(self):
        for n in (1, 1024, 4096, 16384):
            self.assertEqual(R.cost_units("R", n, 16, 10), 6673 * n + 530680)
            self.assertEqual(R.cost_units("RI", n, 16, 10), 6510 * n + 10)
            self.assertEqual(R.cost_units("RAW", n, 16, 10), 6251 * n + 279300)
            self.assertEqual(R.cost_units("MC", n, 16, 10), 5931 * n)


class FrozenEquality(unittest.TestCase):
    def test_recorded_study_outputs(self):
        try:
            base = R.load_example(os.environ.get("RANK_RQMC_WEIGHTS"))
        except OSError as exc:  # offline
            self.skipTest(f"example network unavailable: {exc}")
        self.assertEqual(R.params_sha256(base), R.EXAMPLE_PARAMS_SHA256)
        rows = R.mc_row_units(16, 10)
        for si, (sha, records) in FROZEN.items():
            p = R.perturb(base, R.SIGMAS[si], R.MASTERS["weights"], si, 0)
            self.assertEqual(R.params_sha256(p), sha)
            for arm, rep, k, k_mc in records:
                ai, bi = R.ARMS.index(arm), R.BUDGETS.index(1024)
                value, units = R.estimate(p, 1024, R.stream_seed(R.MASTERS["estimates"], "plugin", si, 0, ai, bi, rep), arm)
                self.assertEqual(value, k / 1024)
                m = units // rows
                self.assertEqual(R.mc(p, m, R.stream_seed(R.MASTERS["mc"], "sampling", si, 0, ai, bi, rep)), k_mc / m)


if __name__ == "__main__":
    unittest.main()
