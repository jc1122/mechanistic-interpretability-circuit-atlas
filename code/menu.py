"""Frozen menu for (x + y) mod p: hand-built m2 followed by one hand-written frozen MLP layer. Prompt "SUM: x y\\nANSWER: " is read at
the answer slot (the last ' '), where m2's compare-number reader already holds NA = x/100, NB = y/100 (the "ADD" keyword would break
m2's first-token detector: the first three tokens must differ). Bricks (every brick is centred and scaled to rms row norm 1 over all
p^2 prompts; constant bricks become 0):
  sx, sy   scalar      NA, NB                                                  1 + 1
  ox, oy   one-hot     o_k = step(100 NA; k - 1/2) - step(100 NA; k + 1/2), k = 0..p-1   p + p
  tx, ty   thermometer t_k = step(100 NA; k - 1/2), k = 1..p-1                  p-1 + p-1
  count    X = 1/(c+1)        brackets  BAL, ANY        copy  CP (V = 70)       pos  R = r/1024      (distractors)
step(z; c) = (gelu(40(z - c) + 8) - gelu(40(z - c) - 8))/16 (two GELU neurons; exactly 0 or 1 for |z - c| >= .4).
"""
import math
import torch
import torch.nn.functional as F
import m2 as B
import neox

DISTRACT = [("count", ["X"]), ("brackets", ["BAL", "ANY"]), ("copy", ["CP"]), ("pos", ["R"])]


def base(kind="m2", seed=0):
    """Frozen base weights: the hand-built m2, or (random control) a random-init model of the same shape and vocabulary."""
    return B.state_dict() if kind != "random" else neox.random_init(B.CFG, 1000 + seed)


def slot_resid(sd, texts):
    """Residual stream after the last block (before the final LayerNorm) at the last position of each prompt.
    Prompts of equal length are batched (no padding, so every prompt sees exactly its own tokens)."""
    out = [None] * len(texts); by = {}
    for i, t in enumerate(texts): by.setdefault(len(t), []).append(i)
    for n, idx in by.items():
        for s in range(0, len(idx), 64):
            ii = idx[s:s + 64]; ids = torch.tensor([[B.ID[c] for c in texts[i]] for i in ii])
            r = neox.forward(sd, B.CFG, ids)[0][:, -1]
            for i, row in zip(ii, r): out[i] = row
    return torch.stack(out)


def step(z, c, s=40.0):
    u = s * (z - c); return (F.gelu(u + 8) - F.gelu(u - 8)) / 16


def assemble(br, scale=None):
    """Concatenate bricks, each centred and divided by its rms centred row norm (bricks with rms < 1e-6, i.e. constant, become 0).
    Returns X, slices {brick: (lo, hi)}, (mean, scale)."""
    if scale is None:
        mu = {n: v.mean(0) for n, v in br.items()}; sd = {n: (v - mu[n]).pow(2).sum(1).mean().sqrt().item() for n, v in br.items()}
        scale = (mu, {n: s if s > 1e-6 else math.inf for n, s in sd.items()})
    mu, sd = scale; sl, i = {}, 0
    for n, v in br.items(): sl[n] = (i, i + v.shape[1]); i += v.shape[1]
    return torch.cat([(v - mu[n]) / sd[n] for n, v in br.items()], 1), sl, scale


def prompts(p): return [(x, y, f"SUM: {x} {y}\nANSWER: ") for x in range(p) for y in range(p)]


def menu_layer(r, p):
    na, nb = 100 * r[:, B.d1("NA"), None], 100 * r[:, B.d1("NB"), None]
    k = torch.arange(p, dtype=r.dtype)
    br = {"sx": na / 100, "sy": nb / 100, "ox": step(na, k - .5) - step(na, k + .5), "oy": step(nb, k - .5) - step(nb, k + .5),
          "tx": step(na, k[1:] - .5), "ty": step(nb, k[1:] - .5)}
    for n, blocks in DISTRACT: br[n] = r[:, [d for b in blocks for d in B.DIMS[b]]]
    return br


def cache(p, kind="m2", seed=0):
    """(X [p^2, D], y [p^2], (x, y) pairs, slices, info). kind 'm2' = full menu, 'scalar' = sx/sy only, 'random' = final residual of a
    random-init base (z-scored per dim, total rms norm = the menu's 10 unit bricks)."""
    P = prompts(p); xy = torch.tensor([(x, y) for x, y, _ in P]); lab = (xy[:, 0] + xy[:, 1]) % p
    r = slot_resid(base(kind, seed), [t for *_, t in P])
    if kind == "random":
        z = (r - r.mean(0)) / (r.std(0) + 1e-8); z = z * (10 / z.pow(2).sum(1).mean()).sqrt()
        return z, lab, xy, {"rand": (0, z.shape[1])}, {}
    br = menu_layer(r, p); full = br
    if kind == "scalar": br = {n: br[n] for n in ("sx", "sy")}
    X, sl, scale = assemble(br)
    info = {"na_err": (100 * r[:, B.d1("NA")] - xy[:, 0]).abs().max().item(), "nb_err": (100 * r[:, B.d1("NB")] - xy[:, 1]).abs().max().item(),
            "onehot_err": max((full[o] - F.one_hot(xy[:, i], p)).abs().max().item() for i, o in enumerate(("ox", "oy")))}
    return X, lab, xy, sl, info
