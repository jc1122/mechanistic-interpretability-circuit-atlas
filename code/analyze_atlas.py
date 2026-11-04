"""Mechanism analysis of the trained atlas nets (called by run.py atlas; works on the menu rows and the trained weights).
Random-base nets: curves only (no menu rows).
pre(u, v) = W1 x + b1 is fitted as c0 + E_u[u] + E_v[v] (least squares over the cells; sep = unexplained share = distractors and
non-separable terms). Forms (substituted for the trained first layer, trained W_out kept; accuracy on ALL cells):
  sep    c0 + E_u + E_v                      lin    E_u, E_v -> per-neuron straight lines in u and v
  u_only E_v -> its mean                     z1d    pre_n = c_n + a_n z (least squares), z = the task's 1-D variable (Z below)
  res5   (count_mod5) E_u -> its residue-class means (u mod 5), E_v -> its mean
Logit structure: share of the centred logit variance that is a function of the task's key variable alone (key_r2), and of the
translation g(key - class) (cyc_r2: mod 5 for count_mod5, signed for open_count). ratio = median over live neurons of slope_u/slope_v.
"""
import math
import torch, torch.nn.functional as F

TASK = {"count_mod5": ("count", lambda u, v: u % 5, 5), "compare_counts": ("count", lambda u, v: 0 if 2 * u > v else 1 if 2 * u < v else 2, 3),
        "open_count": ("brackets", lambda u, v: (u + v) // 2, None)}
Z = {"count_mod5": lambda u, v: u.double() % 5, "compare_counts": lambda u, v: 2 * u.double() - v, "open_count": lambda u, v: (u.double() + v) / 2}
PAIRS = {"scalar": ("su", "sv"), "onehot": ("ou", "ov"), "thermo": ("tu", "tv"), "u_all": ("su", "ou", "tu"), "v_all": ("sv", "ov", "tv")}


def acc(lo, lab, ix=None):
    ix = slice(None) if ix is None else ix; return (lo[ix].argmax(1) == lab[ix]).float().mean().item()


def table_r2(L, key, nk):
    """Share of the variance of L [N, C] (rows centred) explained by a table over the integer key [N] (per class column)."""
    L = L - L.mean(1, keepdim=True); k = key - key.min(); m = torch.zeros(int(k.max()) + 1, L.shape[1], dtype=L.dtype).index_add_(0, k, L)
    m = m / torch.bincount(k).clamp(min=1)[:, None].to(L.dtype); return 1 - (L - m[k]).pow(2).sum().item() / L.pow(2).sum().item()


def sep_fit(pre, uv):
    us, vs = sorted(set(uv[:, 0].tolist())), sorted(set(uv[:, 1].tolist()))
    iu = torch.tensor([us.index(x) for x in uv[:, 0].tolist()]); iv = torch.tensor([vs.index(x) for x in uv[:, 1].tolist()])
    D = torch.cat([F.one_hot(iu, len(us)), F.one_hot(iv, len(vs))[:, 1:]], 1).double()
    D = torch.cat([D, torch.ones(len(uv), 1, dtype=D.dtype)], 1); P = pre.double()
    coef = torch.linalg.lstsq(D, P).solution; fit = D @ coef
    res = (P - fit).pow(2).sum().item() / (P - P.mean(0)).pow(2).sum().item()
    Eu = coef[:len(us)][iu]; Ev = torch.cat([torch.zeros(1, P.shape[1], dtype=P.dtype), coef[len(us):-1]])[iv]  # per cell
    return Eu, Ev, coef[-1], res


def linfit(E, x):
    """Per-neuron least squares E [N, n] ~ a x + b over the cells."""
    D = torch.stack([x, torch.ones_like(x)], 1); c = torch.linalg.lstsq(D, E).solution; return D @ c, c[0]


def analyze(r, cache):
    task = r["task"]; fam, f, _ = TASK[task]; w = {k: torch.as_tensor(v) for k, v in r["w"].items()}
    X = torch.as_tensor(cache["X"]); uv = torch.as_tensor(cache["uv"]); u, v = uv[:, 0], uv[:, 1]
    lab = torch.tensor([f(a, b) for a, b in uv.tolist()])
    te = w["test"].long(); W1, b1, W2, b2 = w["W1"], w["b1"], w["W2"], w["b2"]
    hd = lambda pre: F.relu(pre.float()) @ W2.T + b2
    pre = X @ W1.T + b1; lo = hd(pre); out = dict(task=task, kind=r["kind"], seed=r["seed"], acc_all=acc(lo, lab), acc_test=acc(lo, lab, te))
    sl = cache["sl"]; mu = X.mean(0)
    for name, bs in PAIRS.items():
        if all(b in sl for b in bs):
            Xa = X.clone(); Xk = mu.expand_as(X).clone()
            for b in bs: i, j = sl[b]; Xa[:, i:j] = mu[i:j]; Xk[:, i:j] = X[:, i:j]
            out["pair_" + name] = (round(acc(hd(Xa @ W1.T + b1), lab, te), 3), round(acc(hd(Xk @ W1.T + b1), lab, te), 3))  # (ablate, keep-only)
    Eu, Ev, c0, res = sep_fit(pre, uv); out["sep_resid"] = res; P = pre.double()
    live = (F.relu(pre).sum(0) > 0)
    ud, vd = u.double(), v.double()
    Lu, au = linfit(Eu, ud); Lv, av = linfit(Ev, vd)
    lin_sh = lambda E, L: 1 - (E - L).pow(2).sum().item() / (E - E.mean(0)).pow(2).sum().item()
    out["lin_share_u"], out["lin_share_v"] = lin_sh(Eu, Lu), lin_sh(Ev, Lv)
    out["v_share"] = (Ev - Ev.mean(0)).pow(2).sum().item() / ((Eu - Eu.mean(0)).pow(2).sum().item() + (Ev - Ev.mean(0)).pow(2).sum().item())
    big = live & (au.abs() > 1e-3 * au.abs().max()) & (av.abs() > 1e-3 * av.abs().max())
    out["ratio_uv"] = (au[big] / av[big]).median().item() if big.any() else None
    out["form_sep"] = acc(hd(c0 + Eu + Ev), lab); out["form_lin"] = acc(hd(c0 + Lu + Lv), lab)
    out["form_u_only"] = acc(hd(c0 + Eu + Ev.mean(0)), lab)
    z = Z[task](u, v); Pz, _ = linfit(P, z); out["form_z1d"] = acc(hd(Pz), lab)
    if task == "count_mod5":
        k = u % 5; m = torch.zeros(5, Eu.shape[1], dtype=Eu.dtype).index_add_(0, k, Eu) / torch.bincount(k, minlength=5)[:, None]
        out["res5_share"] = 1 - (Eu - m[k]).pow(2).sum().item() / (Eu - Eu.mean(0)).pow(2).sum().item()
        out["form_res5"] = acc(hd(c0 + m[k] + Ev.mean(0)), lab)
    Lc = lo.double()
    key = {"count_mod5": u, "compare_counts": 2 * u - v, "open_count": u + v}[task]
    out["key_r2"] = table_r2(Lc, key, None)
    if task in ("count_mod5", "open_count"):  # translation g(key - class): columns re-indexed by class
        C = Lc.shape[1]; Lc0 = Lc - Lc.mean(1, keepdim=True); cl = torch.arange(C)
        d = (u[:, None] % 5 - cl[None]) % 5 if task == "count_mod5" else (u + v)[:, None] // 2 - cl[None]
        d = d - d.min(); fl, dl = Lc0.flatten(), d.flatten()
        m = torch.zeros(int(dl.max()) + 1, dtype=fl.dtype).index_add_(0, dl, fl) / torch.bincount(dl).clamp(min=1)
        out["cyc_r2"] = 1 - (fl - m[dl]).pow(2).sum().item() / fl.pow(2).sum().item()
        out["form_cyc"] = acc(m[d], lab)  # logits = g(key - class) alone (the net's own averaged profile)
    if task == "compare_counts":  # linear comparator: dlogit(a - b) ~ alpha u + beta L + gamma; '=' is a third logit
        dl = Lc[:, 0] - Lc[:, 1]; Al = torch.stack([ud, vd, torch.ones_like(ud)], 1); co = torch.linalg.lstsq(Al, dl[:, None]).solution[:, 0]; fit = Al @ co
        out["cmp_ratio"] = (co[1] / co[0]).item(); out["cmp_r2"] = 1 - (dl - fit).pow(2).sum().item() / (dl - dl.mean()).pow(2).sum().item()
        eq = (2 * u == v); out["eq_test_acc"] = acc(lo, lab, te[eq[te]]); out["neq_test_acc"] = acc(lo, lab, te[~eq[te]])
        cf = torch.stack([fit, -fit * 0, torch.full_like(fit, -1e9)], 1)  # closed form: sign of the fitted line, '=' never
        out["form_line"] = acc(cf, lab); out["form_line_agree"] = (cf.argmax(1) == lo.argmax(1)).float().mean().item()
    if task == "count_mod5":  # fully analytic: logit_c = A cos(w(u - c)) + B cos(2w(u - c)), w = 2 pi / 5, (A, B) fitted to the net's logits
        cl = torch.arange(5, dtype=torch.float64); dd = 2 * math.pi * (ud[:, None] - cl[None]) / 5; L0 = Lc - Lc.mean(1, keepdim=True)
        Bm = torch.stack([torch.cos(dd).flatten(), torch.cos(2 * dd).flatten()], 1); ab = torch.linalg.lstsq(Bm, L0.flatten()[:, None]).solution[:, 0]
        cf = (Bm @ ab[:, None]).view(-1, 5); out["fourier_AB"] = ab.tolist(); out["form_fourier"] = acc(cf, lab)
        out["fourier_r2"] = 1 - (L0.flatten() - (Bm @ ab[:, None])[:, 0]).pow(2).sum().item() / L0.pow(2).sum().item()
    if task == "open_count":  # clock test: one frequency w per neuron, E_b ~ a cos(w b/2 + phi), E_L ~ a' cos(w L/2 + psi) (sum s = b/2 + L/2)
        ws = 2 * math.pi * torch.arange(1, 46, dtype=torch.float64) / 92; xb, xl = ud / 2, vd / 2; best = torch.full((P.shape[1],), -1.0); fits = {}
        Euc, Evc = Eu - Eu.mean(0), Ev - Ev.mean(0); tot = Euc.pow(2).sum(0) + Evc.pow(2).sum(0)
        for wi, wv in enumerate(ws):
            Bb = torch.stack([torch.cos(wv * xb), torch.sin(wv * xb)], 1); Bl = torch.stack([torch.cos(wv * xl), torch.sin(wv * xl)], 1)
            fb = Bb @ torch.linalg.lstsq(Bb, Euc).solution; fl = Bl @ torch.linalg.lstsq(Bl, Evc).solution
            r2 = 1 - ((Euc - fb).pow(2).sum(0) + (Evc - fl).pow(2).sum(0)) / tot; better = r2 > best; best = torch.where(better, r2, best)
            for n in better.nonzero().flatten().tolist(): fits[n] = (wi + 1, fb[:, n], fl[:, n])
        k = torch.tensor([fits[n][0] for n in range(P.shape[1])]); out["clock_r2_med"] = best[live].median().item()
        out["clock_k_top"] = torch.bincount(k[live], minlength=46).argsort(descending=True)[:6].tolist()
        Pc = c0 + Eu.mean(0) + Ev.mean(0) + torch.stack([fits[n][1] + fits[n][2] for n in range(P.shape[1])], 1)
        out["form_clock"] = acc(hd(Pc), lab); out["form_clock_agree"] = (hd(Pc).argmax(1) == lo.argmax(1)).float().mean().item()
    out["live"] = int(live.sum())
    return out


def curve(r):
    L = r["log"]; st = lambda th: next((x["step"] for x in L if x["test_acc"] >= th), None)
    return dict(mem=r["fit_step"], t50=st(.5), t90=st(.9), t99=st(.99), final=r["final"]["test_acc"], train=r["final"]["train_acc"], wnorm=r["final"]["wnorm"],
                peak=max(x["test_acc"] for x in L))


def typ(r):
    """Mechanism type from the measured forms (thresholds set from the pilot and the first finished nets)."""
    if "form_sep" not in r: return "random base" if r["final"] < .9 else "random base (solves)"
    if r["task"] == "count_mod5":
        return "residue lookup (u only)" if r["form_res5"] >= .99 and r["form_u_only"] >= .99 else "other"
    if r["task"] == "compare_counts":
        if r["acc_test"] >= .99: return "exact comparator"
        return "linear comparator, '=' memorized" if r["cmp_r2"] >= .7 and r["eq_test_acc"] < .5 else "partial"
    if r["acc_test"] >= .5 and r.get("form_clock", 0) >= .85: return "low-k clock sum"
    return "memorize (no sum)" if r["acc_test"] < .2 else "partial"


def row(r, cache):
    """One flat result row for a trained net r (run.py's result dict) and its menu cache {X, uv, sl}."""
    x = dict(task=r["task"], kind=r["kind"], seed=r["seed"], **curve(r))
    if r["kind"] != "random":
        x.update(analyze(r, cache)); x["usage_ablate"] = {n: round(u["ablate"], 3) for n, u in r["usage"].items()}
    x["frac"] = r["args"]["frac"]; x["type"] = typ(x); return x


KEYS = ["mem", "t50", "t90", "final", "peak", "wnorm", "form_sep", "form_lin", "form_u_only", "form_z1d", "form_res5", "form_fourier", "fourier_r2",
        "form_line", "form_line_agree", "cmp_ratio", "cmp_r2", "eq_test_acc", "neq_test_acc", "key_r2", "cyc_r2", "lin_share_u", "lin_share_v", "v_share", "sep_resid"]


def table(rows):
    """Markdown table: one line per (task, kind, train fraction) with min-max over seeds; then mean encoding-ablation accuracies."""
    groups = {}
    for x in rows: groups.setdefault((x["task"], x["kind"], x["frac"]), []).append(x)
    out = ["| task | kind | frac | n | types | " + " | ".join(KEYS) + " |"]
    for (t, k, fr), xs in sorted(groups.items()):
        ty = {}
        for x in xs: ty[x["type"]] = ty.get(x["type"], 0) + 1
        def agg(q):
            v = [x[q] for x in xs if x.get(q) is not None]
            return "-" if not v else f"{min(v):.3g}-{max(v):.3g}" if len(set(v)) > 1 else f"{v[0]:.3g}"
        out.append(f"| {t} | {k} | {fr} | {len(xs)} | {ty} | " + " | ".join(agg(q) for q in KEYS) + " |")
        if "pair_onehot" in xs[0]:
            for q in ("pair_scalar", "pair_onehot", "pair_thermo", "pair_u_all", "pair_v_all"):
                if q in xs[0]: out.append(f"|  | {q} (ablate, keep-only) mean | {sum(x[q][0] for x in xs) / len(xs):.3f}, {sum(x[q][1] for x in xs) / len(xs):.3f} |")
            ua = {n: sum(x["usage_ablate"][n] for x in xs) / len(xs) for n in xs[0]["usage_ablate"]}
            out.append("|  | brick mean-ablate test acc | " + " ".join(f"{n} {v:.3f}" for n, v in ua.items()) + " |")
    return "\n".join(out)


def probe_random(fam, seed, X, uv, tr, te):
    """Ridge probes (lambda 1e-2, intercept free) of integer quantities from the z-scored residual of the random base, fit on the train
    cells, R^2 on the test cells."""
    X = X.double(); uv = uv.double(); u, L = uv[:, 0], uv[:, 1]; A = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    q = {"count": {"c": u, "L": L, "c/L": u / L, "2c-L": 2 * u - L, "1/(c+1)": 1 / (u + 1), "c mod 5 = 0": (u % 5 == 0).double()},
         "brackets": {"b": u, "L": L, "b/L": u / L, "#( = (L+b)/2": (L + u) / 2, "#(/L": (L + u) / 2 / L}}[fam]
    out = {}
    for n, y in q.items():
        w = torch.linalg.solve(A[tr].T @ A[tr] + 1e-2 * torch.eye(A.shape[1], dtype=A.dtype), A[tr].T @ y[tr]); p = A[te] @ w
        out[n] = round(1 - (p - y[te]).pow(2).sum().item() / (y[te] - y[te].mean()).pow(2).sum().item(), 3)
    return out
