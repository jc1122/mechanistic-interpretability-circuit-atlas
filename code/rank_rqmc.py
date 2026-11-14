"""Rank-coordinate randomized quasi-Monte Carlo (RQMC) accuracy estimation for an AlgZoo second-argmax RNN.

SPDX-License-Identifier: MIT (see LICENSE in this folder). Method: ../METHODS.md, Section 10.

The network is ARC's public AlgZoo example (hidden size 16, sequence length 10, no biases; MIT-0,
https://github.com/alignment-research-center/alg-zoo). Inputs are T iid N(0, 1) numbers; the label is the
position of the second-largest number; the estimand is the network's accuracy p under that law.

Four estimators, each the mean of n native forward passes on inputs from a different sampler:
  MC   plain Monte Carlo, n iid Gaussian inputs
  R    rank coordinates (order-statistic ratios and a Lehmer code), LMS + digital-shift scrambled Sobol', cell jitter
  RI   the same rank coordinates fed with iid uniforms (control: coordinates without equidistribution)
  RAW  scrambled Sobol' with cell jitter in the raw coordinates, x = Phi^-1(u) (control: equidistribution without rank coordinates)

Every arm also returns its full operation cost (2 * MAC + scalar + 20 * named special function), which sets the
size of its matched MC control. Seeds are derived exactly as in the study, so the same keys give the same numbers.

  python rank_rqmc.py reproduce                       # one table row (sigma .03, R, n 4096) at reduced budget, with a CI
  python rank_rqmc.py reproduce --repeats 32 --truth-log2 23   # the study's full budget for that row
  python rank_rqmc.py reproduce --weights FILE        # use a local copy (.pth or .safetensors) instead of downloading
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from urllib.request import urlopen

import numpy as np
import torch

EXAMPLE_URL = "https://storage.googleapis.com/arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth"
EXAMPLE_BLOB_SHA256 = "498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb"
EXAMPLE_PARAMS_SHA256 = "21921eb8e44d4d775bee76b039f56672aa34a9d2e0a8c56006dd36a4e2c0baa4"
KEYS = {"U": "rnn.weight_ih_l0", "W": "rnn.weight_hh_l0", "V": "linear.weight"}
DOMAIN = {"weights": 11, "truth": 23, "sampling": 37, "plugin": 41}  # independent seed domains
MASTERS = {"weights": 202611141931, "truth": 202611141932, "estimates": 202611141933,
           "mc": 202611141934, "bootstrap": 202611141935}  # the study's scored population
SIGMAS = (.01, .03, .10)
ARMS = ("R", "RI", "RAW")
BUDGETS = (1024, 4096, 16384)
BITS = 30          # bits per Sobol' coordinate in torch's SobolEngine
CHUNK = 65536      # rows per forward batch for truth and MC


# ---------------------------------------------------------------- seeds, network, forward pass

def stream_seed(master, domain, *keys):
    """64-bit seed of one independent stream, keyed by integers (numpy SeedSequence)."""
    seq = np.random.SeedSequence([int(master), DOMAIN[domain], *map(int, keys)])
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def load_example(path=None):
    """Weights {U (h,1), W (h,h), V (T,h)} in float64. Downloads the public example unless a local file is given."""
    if path is None:
        with urlopen(EXAMPLE_URL, timeout=60) as response:
            raw = response.read(1_000_001)
        if hashlib.sha256(raw).hexdigest() != EXAMPLE_BLOB_SHA256:
            raise ValueError("downloaded example differs from the one used in the study")
        sd = {k: v.numpy() for k, v in torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu").items()}
    elif str(path).endswith(".safetensors"):
        from safetensors.numpy import load_file
        sd = load_file(str(path))
    else:
        sd = {k: v.numpy() for k, v in torch.load(str(path), weights_only=True, map_location="cpu").items()}
    if set(sd) != set(KEYS.values()):
        raise ValueError("expected exactly the three AlgZoo DistRNN tensors (bias=False)")
    return {k: np.array(sd[name], dtype=np.float64) for k, name in KEYS.items()}


def params_sha256(p):
    h = hashlib.sha256()
    for key in ("U", "V", "W", "a", "b", "h0"):  # the study's hash also covers the three zero bias/start vectors
        a = p[key] if key in p else np.zeros(len(p["V"]) if key == "a" else len(p["W"]))
        a = np.ascontiguousarray(a, dtype="<f8")
        h.update(key.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def perturb(base, sigma, master, scale_index, member_index):
    """A noisy copy: each tensor gets iid Gaussian noise of sd sigma * (that tensor's RMS)."""
    rng = np.random.default_rng(stream_seed(master, "weights", scale_index, member_index))
    p = {k: v.copy() for k, v in base.items()}
    for k in ("U", "W", "V"):
        p[k] += sigma * float(np.sqrt(np.mean(base[k] ** 2))) * rng.standard_normal(p[k].shape)
    return p


def correct(p, x):
    """Boolean vector: does the RNN's argmax equal the position of the second-largest input? (fp64)"""
    h = np.zeros((len(x), len(p["W"])))
    for t in range(x.shape[1]):
        h = np.maximum(h @ p["W"].T + x[:, t, None] * p["U"][:, 0], 0)
    return (h @ p["V"].T).argmax(1) == np.argsort(x, axis=1, kind="stable")[:, -2]


def mc_row_units(h, T):
    """Cost of one MC row: T(h^2+h)+hT MACs (x2), Th ReLUs + T argmax/compare scalars, plus 1 for the mean."""
    return 2 * (T * (h * h + h) + h * T) + (T * h + T) + 1


# ---------------------------------------------------------------- rank coordinates

def priority(T):
    """Frozen order in which the 2T-1 uniforms are read: W = Lehmer digits, V = order-statistic ratios."""
    first = [("W", 0), ("W", 1), ("V", 1), ("V", 2), ("W", 2), ("V", 0)]
    names = [("W", r) for r in range(T - 1)] + [("V", r) for r in range(T)]
    return [v for v in first if v in names] + [v for v in names if v not in first]


def rank_transform(u, T):
    """Map points of (0,1)^(2T-1) to R^T so that iid uniform input gives iid N(0,1) output.

    Values: descending uniform order statistics by ratios, P_0 = V_0^(1/T), P_r = P_{r-1} V_r^(1/(T-r)),
    then s_r = Phi^-1(P_r). Positions: the rank-r value goes to the k_r-th still unused position,
    k_r = floor((T-r) W_r) (a Lehmer code); the last value takes the last position.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != 2 * T - 1 or not np.all((u > 0) & (u < 1)):
        raise ValueError("points strictly inside the unit cube are required (never clipped)")
    n = len(u)
    C = {name: u[:, i] for i, name in enumerate(priority(T))}
    v, ranks = np.ones(n), []
    for r in range(T):
        v = v * np.power(C["V", r], 1 / (T - r))
        ranks.append(v)
    P = np.stack(ranks, axis=1)
    if not np.all((P > 0) & (P < 1)):
        raise ArithmeticError("order statistic at an endpoint")
    s = torch.special.ndtri(torch.from_numpy(P)).numpy()
    rows = np.arange(n)
    free = np.tile(np.arange(T), (n, 1))  # unused positions, in increasing order
    x = np.empty((n, T))
    for r in range(T - 1):
        k = np.floor((T - r) * C["W", r]).astype(np.int64)
        x[rows, free[rows, k]] = s[:, r]
        keep = np.arange(T - r - 1)[None, :]
        free = free[rows[:, None], keep + (keep >= k[:, None])]  # delete column k
    x[rows, free[:, 0]] = s[:, -1]
    if not np.isfinite(x).all():
        raise ArithmeticError("nonfinite transformed input")
    return x


def lehmer_decode(code, T):
    """Permutation (position of rank r) from Lehmer digits k_0..k_{T-2}, k_r in {0..T-r-1}; used by the tests."""
    free, pos = list(range(T)), []
    for k in code:
        pos.append(free.pop(k))
    return pos + free


# ---------------------------------------------------------------- samplers

def unit_points(n, d, seed, sobol):
    """n points in (0,1)^d. Sobol': one fresh LMS + digital-shift scramble, all 2^m points, each 30-bit cell jittered."""
    rng = np.random.default_rng(stream_seed(seed, "plugin", 1))
    if not sobol:
        u = rng.random((n, d))
    else:
        if n < 1 or n & (n - 1):
            raise ValueError("Sobol' needs a power of two; points are never skipped or thinned")
        old = torch.get_default_dtype()
        try:  # float64 construction avoids a known rounding of the first point
            torch.set_default_dtype(torch.float64)
            engine = torch.quasirandom.SobolEngine(d, scramble=True, seed=stream_seed(seed, "plugin", 2) % 2**31)
        finally:
            torch.set_default_dtype(old)
        cells = np.rint(engine.draw_base2(n.bit_length() - 1, dtype=torch.float64).numpy() * 2**BITS).astype(np.int64)
        u = (cells + rng.random((n, d))) / 2**BITS
    if not np.all((u > 0) & (u < 1)):
        raise ArithmeticError("uniform at an endpoint; never clipped")
    return u


def cost_units(arm, n, h, T):
    """Full operation cost of one estimate (2 MAC + scalar + 20 special), every setup charged to this call."""
    units = n * mc_row_units(h, T)  # n forwards and the mean
    if arm == "MC":
        return units
    d = T if arm == "RAW" else 2 * T - 1
    if arm == "RI":
        units += 2 * n * d  # endpoint checks
    else:
        units += d * (BITS**3 + BITS**2 + BITS)  # scramble setup per fresh engine
        units += n * (5 * d + BITS) + 4 * n * d  # point generation, cell jitter and checks
    if arm == "RAW":
        return units + 20 * n * T  # T inverse-normal calls per row
    lehmer = sum(k + 1 for k in range(2, T + 1))
    return units + T + n * (2 * d + 3 * T) + 20 * 2 * n * T + n * T + n * lehmer  # divisors, checks, power+ndtri, products, Lehmer


def estimate(p, n, seed, arm):
    """(estimate, full cost units) for arm R, RI or RAW; n native forwards each."""
    T = len(p["V"])
    if arm in ("R", "RI"):
        x = rank_transform(unit_points(n, 2 * T - 1, seed, sobol=arm == "R"), T)
    elif arm == "RAW":
        x = torch.special.ndtri(torch.from_numpy(unit_points(n, T, seed, sobol=True))).numpy()
    else:
        raise ValueError("arm must be R, RI or RAW")
    return float(np.mean(correct(p, x))), cost_units(arm, n, len(p["W"]), T)


def mc(p, n, seed):
    """Plain Monte Carlo accuracy from n iid Gaussian inputs."""
    rng, k, T = np.random.default_rng(seed), 0, len(p["V"])
    for start in range(0, n, CHUNK):
        k += int(correct(p, rng.standard_normal((min(CHUNK, n - start), T))).sum())
    return k / n


def truth(p, n, master, si, mi):
    """Reference accuracy from n iid rows in two independent halves (checkpoint-sized chunks as in the study)."""
    counts, T = [0, 0], len(p["V"])
    for split in range(2):
        for done in range(0, n // 2, CHUNK):
            m = min(CHUNK, n // 2 - done)
            rng = np.random.default_rng(stream_seed(master, "truth", si, mi, split, done))
            counts[split] += int(correct(p, rng.standard_normal((m, T))).sum())
    acc = sum(counts) / n
    return {"accuracy": acc, "variance": acc * (1 - acc) / (n - 1), "pq_unbiased": acc * (1 - acc) * n / (n - 1),
            "split": [c / (n // 2) for c in counts], "n": n}


# ---------------------------------------------------------------- scoring one table row

def bootstrap_indices(scales, master, group, draws=2000):
    """Net-cluster bootstrap, stratified by noise scale; the same draws serve every arm and budget."""
    rng = np.random.default_rng(int(np.random.SeedSequence([master, 41, 77, group]).generate_state(1, dtype=np.uint64)[0]))
    groups = [np.flatnonzero(np.array(scales) == s) for s in sorted(set(scales))]
    return np.concatenate([g[rng.integers(0, len(g), (draws, len(g)))] for g in groups], axis=1)


def score_row(records, master, group):
    """records: per net, {truth, values, mc_values, mc_rows}. Truth-corrected MSE ratios with 95% percentile CIs."""
    a = np.array([np.mean((np.array(r["values"]) - r["truth"]["accuracy"]) ** 2) - r["truth"]["variance"] for r in records])
    b = np.array([np.mean((np.array(r["mc_values"]) - r["truth"]["accuracy"]) ** 2) - r["truth"]["variance"] for r in records])
    t = np.array([r["truth"]["pq_unbiased"] * np.mean([1 / m for m in r["mc_rows"]]) for r in records])
    ix = bootstrap_indices([r["scale"] for r in records], master, group)
    av, bv, tv = a[ix].mean(1), b[ix].mean(1), t[ix].mean(1)
    q = lambda z: np.quantile(z, [.025, .975]).tolist()
    return {"nets": len(records), "corrected_mse": a.mean(), "mc_corrected_mse": b.mean(), "theory_mse": t.mean(),
            "ratio_measured": a.mean() / b.mean(), "ratio_measured_ci95": q(av / bv),
            "ratio_theory": a.mean() / t.mean(), "ratio_theory_ci95": q(av / tv),
            "excess_mse": a.mean() - b.mean(), "excess_ci95": q(av - bv)}


def run_net(base, si, mi, arm, n, repeats, truth_n):
    """One net of the study population: truth, `repeats` estimates and their matched full-cost MC controls."""
    p = perturb(base, SIGMAS[si], MASTERS["weights"], si, mi)
    ai, bi = ARMS.index(arm), BUDGETS.index(n)
    rec = {"scale": si, "member": mi, "net_sha256": params_sha256(p), "values": [], "mc_values": [], "mc_rows": [],
           "truth": truth(p, truth_n, MASTERS["truth"], si, mi)}
    for rep in range(repeats):
        value, units = estimate(p, n, stream_seed(MASTERS["estimates"], "plugin", si, mi, ai, bi, rep), arm)
        rows = units // mc_row_units(len(p["W"]), len(p["V"]))  # floor: the control never costs more than the method
        rec["values"].append(value)
        rec["mc_rows"].append(rows)
        rec["mc_values"].append(mc(p, rows, stream_seed(MASTERS["mc"], "sampling", si, mi, ai, bi, rep)))
    return rec


def main():
    ap = argparse.ArgumentParser(description="Recompute one row of METHODS.md Table 10.4 (one noise scale) at a chosen budget.")
    ap.add_argument("command", choices=["reproduce"])
    ap.add_argument("--weights", help="local .pth or .safetensors copy of the AlgZoo example (default: download)")
    ap.add_argument("--sigma-index", type=int, default=1, choices=[0, 1, 2], help="0, 1, 2 = sigma .01, .03, .10")
    ap.add_argument("--arm", default="R", choices=ARMS)
    ap.add_argument("--n", type=int, default=4096, choices=BUDGETS)
    ap.add_argument("--nets", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=8, help="the study used 32")
    ap.add_argument("--truth-log2", type=int, default=21, help="the study used 23")
    a = ap.parse_args()
    base = load_example(a.weights)
    if params_sha256(base) != EXAMPLE_PARAMS_SHA256:
        raise ValueError("these are not the example weights used in the study")
    recs = [run_net(base, a.sigma_index, mi, a.arm, a.n, a.repeats, 2**a.truth_log2) for mi in range(a.nets)]
    row = score_row(recs, MASTERS["bootstrap"], a.sigma_index + 1)
    row.update(sigma=SIGMAS[a.sigma_index], arm=a.arm, n=a.n, repeats=a.repeats, truth_n=2**a.truth_log2,
               mean_accuracy=float(np.mean([r["truth"]["accuracy"] for r in recs])))
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
