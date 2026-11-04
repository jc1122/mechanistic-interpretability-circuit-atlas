"""Per-network analysis of a trained glue MLP on the frozen modular-addition menu (called by run.py modadd).
Usage per brick: mean-ablate (brick -> its mean), resample (brick from a shuffled row), keep-only (all other bricks mean-ablated),
contrib = rms of W_in[:, brick] (x_brick - mean); the same per encoding pair (scalar / one-hot / thermometer of x and y).
Circuit: preactivation pre(x, y) = E_x[x] + E_y[y] + c (separability residual reported); DFT of E_x, E_y over x (frequency k = 1..(p-1)/2);
logit fits on the key frequencies K:
  clock  L(x,y,c) = sum_k a_k cos(w_k(x+y-c)) + b_k sin(w_k(x+y-c))
  pizza  L(x,y,c) = sum_k |cos(w_k(x-y)/2)| (a_k cos(w_k(x+y-c)) + b_k sin(w_k(x+y-c)))      (w_k = 2 pi k / p)
neuron formula: each neuron n -> ReLU(c_n + E_x[n] and E_y[n] projected on its dominant frequency), trained W_out kept (substitution check).
Fully analytic layer: pre_n = c_n + a_n cos(w x + phi_n) + b_n cos(w y + psi_n), W_out[n, c] = A_n cos(w c + phi_n + psi_n).
dd = share of the correct-logit variance explained by d = x - y mod p (pizza ~ high, clock ~ 0).
"""
import math
import torch, torch.nn.functional as F


def acc(lo, lab, ix=None):
    ix = slice(None) if ix is None else ix; return (lo[ix].argmax(1) == lab[ix]).float().mean().item()


def fpow(E, p):
    """E [n, p] (function of x) -> power per frequency k = 1..(p-1)/2, [n, (p-1)//2]."""
    Fk = torch.fft.fft(E, dim=1); return 2 * Fk[:, 1:(p - 1) // 2 + 1].abs().pow(2)


def project(E, p, k):
    """Projection of each row of E (centered) on frequency k[n] (cos + sin)."""
    x = torch.arange(p, dtype=E.dtype); w = 2 * math.pi * k[:, None].to(E.dtype) / p
    c, s = torch.cos(w * x), torch.sin(w * x)
    return (2 / p) * ((E * c).sum(1, keepdim=True) * c + (E * s).sum(1, keepdim=True) * s)


def fit_logits(L, p, K, pizza):
    """Least squares of centered logits L [p, p, p] on the clock/pizza basis of K; returns (R^2, fitted logits [p^2, p])."""
    x, y, c = torch.meshgrid(*(torch.arange(p, dtype=torch.float64),) * 3, indexing="ij")
    cols = []
    for k in K:
        w = 2 * math.pi * k / p; amp = torch.cos(math.pi * k * (x - y) / p).abs() if pizza else 1.0
        cols += [(amp * torch.cos(w * (x + y - c))).flatten(), (amp * torch.sin(w * (x + y - c))).flatten()]
    A = torch.stack(cols, 1); t = L.double().flatten()
    coef = torch.linalg.lstsq(A, t[:, None]).solution; fit = (A @ coef)[:, 0]
    return 1 - (t - fit).pow(2).sum().item() / t.pow(2).sum().item(), fit.view(p * p, p).float()


@torch.no_grad()
def analyze(m, X, lab, xy, sl, p, te):
    W1, b1, W2, b2 = m[0].weight, m[0].bias, m[2].weight, m[2].bias
    head = lambda pre: F.relu(pre) @ W2.T + b2
    lo = m(X); out = dict(acc_all=acc(lo, lab), acc_test=acc(lo, lab, te)); mu = X.mean(0)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(len(X), generator=g); use = {}
    for n, (i, j) in sl.items():
        Xa = X.clone(); Xa[:, i:j] = mu[i:j]; Xr = X.clone(); Xr[:, i:j] = X[perm, i:j]
        Xk = mu.expand_as(X).clone(); Xk[:, i:j] = X[:, i:j]
        use[n] = dict(in_norm=W1[:, i:j].norm().item(), contrib=((X[:, i:j] - mu[i:j]) @ W1[:, i:j].T).pow(2).sum(1).mean().sqrt().item(),
                      ablate=acc(m(Xa), lab, te), resample=acc(m(Xr), lab, te), keep_only=acc(m(Xk), lab, te))
    out["usage"] = use
    enc = {"scalar": ("sx", "sy"), "onehot": ("ox", "oy"), "thermo": ("tx", "ty")}  # both operands of one encoding
    out["pair"] = {}
    for e, bs in enc.items():
        if all(b in sl for b in bs):
            Xa = X.clone(); Xk = mu.expand_as(X).clone()
            for b in bs: i, j = sl[b]; Xa[:, i:j] = mu[i:j]; Xk[:, i:j] = X[:, i:j]
            out["pair"][e] = dict(ablate=acc(m(Xa), lab, te), keep_only=acc(m(Xk), lab, te))
    # separable effective embeddings
    pre = (X @ W1.T + b1).view(p, p, -1)  # [x, y, n]
    c0 = pre.mean((0, 1)); Ex = (pre.mean(1) - c0).T; Ey = (pre.mean(0) - c0).T  # [n, p]
    sep = pre - c0 - Ex.T[:, None] - Ey.T[None]
    out["sep_resid"] = sep.pow(2).sum().item() / (pre - c0).pow(2).sum().item()
    xs = torch.arange(p, dtype=Ex.dtype) - (p - 1) / 2  # linear-in-x part of E_x (the scalar / thermometer sum route)
    lin = lambda E: (E @ xs)[:, None] * xs / xs.pow(2).sum()
    out["linear_share"] = ((lin(Ex).pow(2).sum() + lin(Ey).pow(2).sum()) / (Ex.pow(2).sum() + Ey.pow(2).sum())).item()
    P = fpow(Ex - lin(Ex), p) + fpow(Ey - lin(Ey), p); tot = P.sum(0); share = tot / tot.sum()  # spectrum of the non-linear part
    order = share.argsort(descending=True); ks = [int(i) + 1 for i in order]
    K = [k for k, s in zip(ks, share[order].tolist()) if s >= 0.05][:8]
    out["freq_share"] = {k: round(share[k - 1].item(), 4) for k in ks[:10]}; out["key_freqs"] = K
    out["top5_share"] = share[order[:5]].sum().item()
    for b in ("ox", "tx"):  # per-brick spectra: this brick's own contribution to pre as a function of x (mean over y)
        if b in sl:
            i, j = sl[b]; Eb = (X[:, i:j] @ W1[:, i:j].T).view(p, p, -1).mean(1).T; Eb = Eb - Eb.mean(1, keepdim=True)
            Pb = fpow(Eb, p).sum(0); sb = Pb / Pb.sum(); ob = sb.argsort(descending=True)
            out["freq_" + b] = {int(k) + 1: round(sb[k].item(), 3) for k in ob[:6]}
    nk = P.argmax(1) + 1; npow = P.sum(1); live = npow > 0.01 * npow.max()
    out["neuron_freq"] = {int(k): int(((nk == k) & live).sum()) for k in nk[live].unique()}
    out["neuron_dom_share"] = (P.max(1).values[live] / npow[live]).mean().item(); out["live_neurons"] = int(live.sum())
    # neuron formula: ReLU(c + a cos(w_k x + phi) + b cos(w_k y + psi)), k = the neuron's dominant frequency
    pre_f = c0 + project(Ex, p, nk).T[:, None] + project(Ey, p, nk).T[None]
    lf = head(pre_f.reshape(p * p, -1)); out["neuron_formula_acc"] = acc(lf, lab); out["neuron_formula_agree"] = (lf.argmax(1) == lo.argmax(1)).float().mean().item()
    lf2 = head((pre_f + lin(Ex).T[:, None] + lin(Ey).T[None]).reshape(p * p, -1)); out["neuron_formula_lin_acc"] = acc(lf2, lab)  # + linear ramps
    out["sep_formula_acc"] = acc(head((c0 + Ex.T[:, None] + Ey.T[None]).reshape(p * p, -1)), lab)  # separability alone
    # logits: clock vs pizza
    Lc = (lo - lo.mean(1, keepdim=True)).view(p, p, p)
    for name, pz in (("clock", False), ("pizza", True)):
        r2, fit = fit_logits(Lc, p, K or ks[:1], pz); out[name + "_r2"] = r2; out[name + "_formula_acc"] = acc(fit, lab); out[name + "_agree"] = (fit.argmax(1) == lo.argmax(1)).float().mean().item()
    cl = lo.gather(1, lab[:, None])[:, 0] - lo.mean(1); d = (xy[:, 0] - xy[:, 1]) % p
    md = torch.zeros(p).index_add_(0, d, cl) / torch.bincount(d, minlength=p)
    out["dd"] = (md[d] - cl.mean()).pow(2).mean().item() / cl.var().item()
    out["type"] = ("memorize" if out["acc_test"] < 0.9 else "lookup" if out["top5_share"] < 0.5
                   else "pizza" if out["pizza_r2"] > out["clock_r2"] + 0.05 and out["dd"] > 0.2 else "clock" if out["dd"] < 0.1 else "mixed")
    # logit spectrum over s - c (s = x + y): the clock basis on all frequencies is a complete basis of functions g((x+y-c) mod p)
    Kall = list(range(1, (p - 1) // 2 + 1))
    out["sum_r2"], gfit = fit_logits(Lc, p, Kall, False); out["sum_formula_acc"] = acc(gfit, lab); out["sum_agree"] = (gfit.argmax(1) == lo.argmax(1)).float().mean().item()
    x3, y3, c3 = torch.meshgrid(*(torch.arange(p, dtype=torch.float64),) * 3, indexing="ij"); Lf = Lc.double()
    amp = torch.tensor([sum(((Lf * f(2 * math.pi * k / p * (x3 + y3 - c3))).mean() * 2) ** 2 for f in (torch.cos, torch.sin)).sqrt().item() for k in Kall])
    oa = amp.argsort(descending=True); out["logit_freq_amp"] = {int(k) + 1: round(amp[k].item(), 3) for k in oa[:8]}
    out["logit_amp_top5_share"] = (amp[oa[:5]].pow(2).sum() / amp.pow(2).sum()).item()
    # Gromov-type check per live neuron (dominant k): E_x ~ a cos(w x + phi), E_y ~ b cos(w y + psi), W_out[n, c] ~ A cos(w c + theta);
    # ReLU is convex, so its cross term ~ +cos(w x + phi) cos(w y + psi) holds cos(w(x+y) + phi + psi); it peaks at c = x + y iff theta = phi + psi
    xg = torch.arange(p, dtype=Ex.dtype); w = 2 * math.pi * nk.to(Ex.dtype)[:, None] / p
    ph = lambda E: torch.atan2(-(E * torch.sin(w * xg)).sum(1), (E * torch.cos(w * xg)).sum(1))  # E ~ cos(w x + ph)
    amp_ = lambda E: 2 / p * torch.sqrt((E * torch.cos(w * xg)).sum(1) ** 2 + (E * torch.sin(w * xg)).sum(1) ** 2)
    Wo = W2.T - W2.T.mean(1, keepdim=True)  # [n, p]
    phx, phy, pho = ph(Ex), ph(Ey), ph(Wo)
    err = torch.remainder(pho - phx - phy + math.pi, 2 * math.pi) - math.pi  # theta - phi - psi (0 = the Gromov phase rule)
    wsh = amp_(Wo).pow(2) * p / 2 / Wo.pow(2).sum(1)  # share of W_out[n] at the neuron's own frequency
    out["phase_err_med"] = err[live].abs().median().item(); out["wout_own_freq_share"] = wsh[live].mean().item()
    # fully analytic layer: pre_n = c_n + a_n cos(w x + phi_n) + b_n cos(w y + psi_n); W_out[n, c] = A_n cos(w c + phi_n + psi_n)
    ax, ay, ao = amp_(Ex), amp_(Ey), amp_(Wo)
    preA = c0 + (ax[:, None] * torch.cos(w * xg + phx[:, None])).T[:, None] + (ay[:, None] * torch.cos(w * xg + phy[:, None])).T[None]
    WA = (ao[:, None] * torch.cos(w * xg + phx[:, None] + phy[:, None])) * live[:, None]
    la = F.relu(preA.reshape(p * p, -1)) @ WA; out["analytic_acc"] = acc(la, lab); out["analytic_agree"] = (la.argmax(1) == lo.argmax(1)).float().mean().item()
    out["attention"] = "none (MLP-only glue)"
    return out


def steps_to(log, thr):
    return next((r["step"] for r in log if r["test_acc"] >= thr), None)
