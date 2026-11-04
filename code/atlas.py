"""Circuit-atlas tasks on the frozen m2 + menu. Every task is a function of two integers (u, v) that m2 computes at the answer slot; one
prompt per cell (u, v), the train/test split is over cells. Every prompt starts with one fixed shot line (without a preceding '\\n',
m2's has_letter head misses the COUNT letter and gives X = 0).
  count     "COUNT a: <c a's and L-c b's, shuffled>\\nANSWER: "    u = c   (m2 head L4H1: X = 1/(c+1)),   v = L (position MLP: R)
  brackets  "VALID_BRACKETS: <line>\\nANSWER: "                     u = b = #'(' - #')' (m2 head L5H0: BAL = b/(L+1)), v = L
L = 1..LMAX, c = 0..L; b = -L..L in steps of 2 (LMAX 45: 1080 cells each; m2's count is exact only up to L = 45).
Tasks: count_mod5 = c mod 5 (5 classes); compare_counts = a / b / '=' (2c > L, 2c < L, 2c = L); open_count = #'(' = (L + b)/2 (46 classes).
Menu bricks: su, sv = m2's raw scalars (X or BAL; R); ou, ov one-hot; tu, tv thermometer [u >= k] (exact step pairs, thresholds halfway
between neighbouring integer values); b needs the product BAL (L + 1), so its thermometer is a second step stage gated by the one-hot of
L. Distractors = other m2 blocks at the slot (constant ones become 0). Kinds: m2 (full menu), scalar (su, sv only), random (random base)."""
import random
import torch
import torch.nn.functional as F
import m2 as B
import menu

LMAX = 45
SHOT = {"count": "COUNT b: a b\nANSWER: 1\n", "brackets": "VALID_BRACKETS: ()\nANSWER: YES\n"}
HEAD = {"count": "COUNT a: ", "brackets": "VALID_BRACKETS: "}
TASKS = {"count_mod5": ("count", lambda u, L: u % 5, lambda: 5),
         "compare_counts": ("count", lambda u, L: 0 if 2 * u > L else 1 if 2 * u < L else 2, lambda: 3),
         "open_count": ("brackets", lambda u, L: (L + u) // 2, lambda: LMAX + 1)}
DIST = {"count": [("brackets", ["BAL", "ANY"]), ("copy", ["CP"]), ("last", ["LL"])],
        "brackets": [("count", ["X"]), ("neg", ["ANY"]), ("copy", ["CP"]), ("last", ["LL"])]}


def cells(fam):
    return [(u, L) for L in range(1, LMAX + 1) for u in (range(L + 1) if fam == "count" else range(-L, L + 1, 2))]


def text(fam, u, L, rng):
    if fam == "count": it = ["a"] * u + ["b"] * (L - u); rng.shuffle(it); body = " ".join(it)
    else: it = ["("] * ((L + u) // 2) + [")"] * ((L - u) // 2); rng.shuffle(it); body = "".join(it)
    return SHOT[fam] + HEAD[fam] + body + "\nANSWER: "


def pos0(fam):
    """Slot position r (= R * 1024 = index + 1) of a line with L items is a + L * per."""
    return (len(SHOT[fam]) + len(HEAD[fam]) + 9 - 1, 2) if fam == "count" else (len(SHOT[fam]) + len(HEAD[fam]) + 9, 1)


def menu_layer(fam, r):
    R = r[:, B.d1("R")] * B.RS; a, per = pos0(fam); k = torch.arange(1, LMAX + 1, dtype=r.dtype)
    tv = menu.step(R[:, None], a + per * (k - .5), 40.0 / per)  # [L >= k], k = 1..LMAX
    ov = torch.cat([1 - tv[:, :1], tv[:, :-1] - tv[:, 1:], tv[:, -1:]], 1)  # one-hot L = 0..LMAX
    if fam == "count":
        X = r[:, B.d1("X")]; s = X
        mid = .5 * (1 / k + 1 / (k + 1)); gap = 1 / k - 1 / (k + 1)  # c >= k  <=>  X <= mid(1/k, 1/(k+1))
        tu = menu.step(-X[:, None], -mid, 32.0 / gap)
        ou = torch.cat([1 - tu[:, :1], tu[:, :-1] - tu[:, 1:], tu[:, -1:]], 1)  # c = 0..LMAX
    else:
        BAL = r[:, B.d1("BAL")]; s = BAL; kb = torch.arange(-LMAX + 1, LMAX + 1, dtype=r.dtype)
        Ls = torch.arange(LMAX + 1, dtype=r.dtype)  # gate: z_L = BAL + 100 (onehot_L - 1); [b >= k] = step(z_L; (k - 1/2)/(L+1), 40(L+1))
        z = BAL[:, None, None] + 100 * (ov[:, :, None] - 1)
        tu = menu.step(z, (kb[None, None] - .5) / (Ls[None, :, None] + 1), 40 * (Ls[None, :, None] + 1)).sum(1)  # b = -LMAX+1..LMAX
        ou = torch.cat([1 - tu[:, :1], tu[:, :-1] - tu[:, 1:], tu[:, -1:]], 1)  # b = -LMAX..LMAX
    br = {"su": s[:, None], "sv": r[:, B.d1("R"), None], "ou": ou, "ov": ov, "tu": tu, "tv": tv}
    for n, blocks in DIST[fam]: br[n] = r[:, [d for b in blocks for d in B.DIMS[b]]]
    return br


def cache(fam, kind="m2", seed=0):
    """(X [cells, D], uv [cells, 2], slices, info)."""
    C = cells(fam); rng = random.Random(0); P = [text(fam, u, L, rng) for u, L in C]; uv = torch.tensor(C)
    r = menu.slot_resid(menu.base(kind, seed), P)
    if kind == "random":
        z = (r - r.mean(0)) / (r.std(0) + 1e-8); z = z * (len(menu_layer(fam, r)) / z.pow(2).sum(1).mean()).sqrt()
        return z, uv, {"rand": (0, z.shape[1])}, {}
    br = menu_layer(fam, r); full = br
    if kind == "scalar": br = {n: br[n] for n in ("su", "sv")}
    X, sl, _ = menu.assemble(br)
    off = 0 if fam == "count" else LMAX
    info = {"u_onehot_err": (full["ou"] - F.one_hot(uv[:, 0] + off, full["ou"].shape[1])).abs().max().item(),
            "v_onehot_err": (full["ov"] - F.one_hot(uv[:, 1], LMAX + 1)).abs().max().item(),
            "raw_err": ((full["su"][:, 0] - (1 / (uv[:, 0] + 1) if fam == "count" else uv[:, 0] / (uv[:, 1] + 1))).abs().max().item()),
            "live": {n: v.var(0).sum().item() > 1e-10 for n, v in full.items()}}
    return X, uv, sl, info


@torch.no_grad()
def usage(m, X, lab, sl, te):
    """Per brick: test accuracy after mean-ablating it (ablate), after replacing it by another row's (resample), with all other bricks
    mean-ablated (keep_only); contrib = rms of the brick's input to the hidden pre-activations."""
    mu = X.mean(0); g = torch.Generator().manual_seed(0); perm = torch.randperm(len(X), generator=g); W1 = m[0].weight; out = {}
    acc = lambda Z: (m(Z)[te].argmax(1) == lab[te]).float().mean().item()
    for n, (i, j) in sl.items():
        Xa = X.clone(); Xa[:, i:j] = mu[i:j]; Xr = X.clone(); Xr[:, i:j] = X[perm, i:j]; Xk = mu.expand_as(X).clone(); Xk[:, i:j] = X[:, i:j]
        out[n] = dict(contrib=((X[:, i:j] - mu[i:j]) @ W1[:, i:j].T).pow(2).sum(1).mean().sqrt().item(), ablate=acc(Xa), resample=acc(Xr), keep_only=acc(Xk))
    return out
