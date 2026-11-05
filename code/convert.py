"""Base conversion (hex -> bin, bin -> dec, dec -> bin), x = 0..255: hand-built networks, data, PART / TRAIN and the key analyses.
Two-layer GPT-NeoX (NoPE, no LayerNorm, ReLU, d_model 256, 4 heads x 64, MLP 256). Parameter names follow Hugging Face's
GPTNeoXForCausalLM without LayerNorm keys. Everything is plain torch; run it through `python run.py convert ...` (see ../METHODS.md, section 7).

Sequences (token ids: BOS 0, '0'..'9' 1..10, 'a'..'f' 11..16, '=' 17; digits are shared by all bases), format A = answer only,
B = scratchpad:
  hex2bin  BOS h1 h0 '='   A: b7..b0                      B: h1 b7 b6 b5 b4 h0 b3 b2 b1 b0
  bin2dec  BOS b7..b0 '='  A: d2 d1 d0                    B: v1 .. v8, v_k = x >> (8-k) as 3 digits (Horner doubling; v8 = answer)
  dec2bin  BOS d2 d1 d0 '='  A: b7..b0                    B: (r_{k-1} q_k) for k = 1..8, '=', b7..b0 (halving: remainder bit, quotient)
Exact match = every output token correct (teacher forced). Must run in full fp32: reduced-precision (TF32) matmuls break the hand-set mixtures."""
import math
import torch
import torch.nn.functional as F

CH = "0123456789abcdef="; ID = {c: i + 1 for i, c in enumerate(CH)}; BOS = 0; V = len(CH) + 1
NUM = {ID[c]: float(i) for i, c in enumerate("0123456789")}; HEX = [ID[c] for c in "0123456789abcdef"]
C_POS, BIG, U = 64.0, 30.0, 10.0
NH, DH, DM, NL = 4, 64, 256, 2


def seqs(variant, fmt):
    """(ids (256, L), o): all inputs x = 0..255; outputs are ids[:, o:], predicted at positions o-1 .. L-2."""
    def s(x):
        h, b, d = f"{x:02x}", f"{x:08b}", f"{x:03d}"
        if variant == "hex2bin": return h + "=", (b if fmt == "A" else h[0] + b[:4] + h[1] + b[4:])
        if variant == "bin2dec": return b + "=", (d if fmt == "A" else "".join(f"{x >> (8 - k):03d}" for k in range(1, 9)))
        return d + "=", (b if fmt == "A" else "".join(f"{(x >> (k - 1)) & 1}{x >> k:03d}" for k in range(1, 9)) + "=" + b)
    return torch.tensor([[BOS] + [ID[c] for c in "".join(s(x))] for x in range(256)]), 1 + len(s(0)[0])


def shapes():
    sh = {"gpt_neox.embed_in.weight": (V, DM), "lm_head.weight": (V, DM)}
    for l in range(NL):
        p = f"gpt_neox.layers.{l}."
        for n, o, i in (("attention.query_key_value", 3 * DM, DM), ("attention.dense", DM, DM), ("mlp.dense_h_to_4h", DM, DM), ("mlp.dense_4h_to_h", DM, DM)):
            sh[p + n + ".weight"] = (o, i); sh[p + n + ".bias"] = (o,)
    return sh


# ---------------------------------------------------------------- the hand-built networks
class Net:
    """Residual dims are allocated by name; heads and neurons are written by formula (see the entry's Hand-built reference)."""
    def __init__(self, L):
        self.L, self.sq = L, math.sqrt(DH); self.sd = {k: torch.zeros(v) for k, v in shapes().items()}
        self.n, self.dims, self.nn = 0, {}, [0, 0]
        self.dim("TOK", V); self.dim("P"); self.dim("POS", L - 1)
        for t in range(V): self.sd["gpt_neox.embed_in.weight"][t, self.dims["TOK"][t]] = 1.0
        self.positions()

    def dim(self, name, k=1):
        self.dims[name] = list(range(self.n, self.n + k)); self.n += k; assert self.n <= DM; return self.dims[name]

    def d(self, name, i=0): return self.dims[name][i]

    def w(self, l, part, h):  # query / key / value block (part 0 / 1 / 2) of head h: (weight rows, bias)
        s = slice((3 * h + part) * DH, (3 * h + part + 1) * DH); p = f"gpt_neox.layers.{l}.attention.query_key_value."
        return self.sd[p + "weight"][s], self.sd[p + "bias"][s]

    def value(self, l, h, rows):
        """rows: list of ({resid dim: coef}, {dst dim: coef}): value row i reads sum coef * resid and writes to dst."""
        v, _ = self.w(l, 2, h); O = self.sd[f"gpt_neox.layers.{l}.attention.dense.weight"]
        for i, (src, dst) in enumerate(rows):
            for a, c in src.items(): v[i, a] = c
            for a, c in dst.items(): O[a, h * DH + i] = c

    def neuron(self, l, x, b, out):
        p = f"gpt_neox.layers.{l}.mlp."; j = self.nn[l]; self.nn[l] += 1
        for a, c in x.items(): self.sd[p + "dense_h_to_4h.weight"][j, a] = c
        self.sd[p + "dense_h_to_4h.bias"][j] = b
        for a, c in out.items(): self.sd[p + "dense_4h_to_h.weight"][a, j] += c

    def obias(self, l, dim, c): self.sd[f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.bias"][dim] += c

    def positions(self):
        """L0H0 attends BOS with score ln 64: P = 64/(64 + r). L0 MLP: one hinge per position interpolates the POS one-hot on the P grid."""
        q, bq = self.w(0, 0, 0); k, _ = self.w(0, 1, 0); bq[0] = math.log(C_POS) * self.sq; k[0, self.d("TOK", BOS)] = 1.0
        self.value(0, 0, [({self.d("TOK", BOS): 1.0}, {self.d("P"): 1.0})])
        R = self.L - 1; u = torch.tensor([C_POS / (C_POS + r) for r in range(R)], dtype=torch.float64)
        A = torch.cat([torch.relu(u[:, None] - u[None, 1:]), torch.ones(R, 1, dtype=torch.float64)], 1)
        coef = torch.linalg.solve(A, torch.eye(R, dtype=torch.float64))  # column m: POS_m = A coef[:, m]
        for kk in range(1, R):
            self.neuron(0, {self.d("P"): 1.0}, -u[kk].item(), {self.d("POS", m): coef[kk - 1, m].item() for m in range(R)})
        for m in range(R): self.obias(0, self.d("POS", m), coef[R - 1, m].item())

    def table(self, l, h, rows, vrows):
        """Table head: rows {query position r: {key position j: weight}} (weights of a row sum to 1; BOS = 0 is the sink): score BIG + ln w."""
        q, _ = self.w(l, 0, h); k, _ = self.w(l, 1, h)
        for m in range(self.L - 1): k[m, self.d("POS", m)] = 1.0
        for r, ws in rows.items():
            for j, wt in ws.items():
                if wt > 0: q[j, self.d("POS", r)] += (BIG + math.log(wt)) * self.sq
        self.value(l, h, vrows)

    def gate(self, rs): return {self.d("POS", r): 1.0 for r in rs}
    def readout(self, tok, dim, c): self.sd["lm_head.weight"][tok, dim] += c

    def step(self, x, a, width, out, gate=None, G=0.0):
        """out += [x >= a] for x on a grid of spacing > width: two ReLU neurons."""
        g = {k: G for k in (gate or {})}; z = {kk: c / width for kk, c in x.items()}
        self.neuron(1, {**z, **g}, -a / width + 1 - G, out); self.neuron(1, {**z, **g}, -a / width - G, {kk: -c for kk, c in out.items()})


NUMV = lambda net: {net.d("TOK", t): v for t, v in NUM.items()}


def hex2bin(fmt):
    L = 12 if fmt == "A" else 14; net = Net(L); X = net.dim("X", 16)
    if fmt == "A": src = {r: (1 if r <= 6 else 2) for r in range(3, 11)}; bitj = {r: (10 - r) % 4 for r in range(3, 11)}
    else:
        src = {r: (1 if r <= 7 else 2) for r in range(3, 13)}; copy = {3, 8}
        bitj = {r: (12 - r) % 4 if r > 8 else (7 - r) % 4 for r in range(3, 13) if r not in copy}
    Y = net.dim("Y", V)
    net.table(1, 0, {r: {j: 1.0} for r, j in src.items()},  # head 0 reads the source hex digit: X = its one-hot (B: also copies it to Y)
              [({net.d("TOK", t): 1.0}, {X[i]: 1.0, **({Y[t]: 1.0} if fmt == "B" else {})}) for i, t in enumerate(HEX)])
    for j in range(4):  # bit j of the nibble: relu(sum_v bit_j(v) X_v + gate - 1)
        rs = [r for r, jj in bitj.items() if jj == j]; xb = {X[v]: 1.0 for v in range(16) if (v >> j) & 1}
        net.neuron(1, {**xb, **net.gate(rs)}, -1.0, {Y[ID["1"]]: 2.0 if fmt == "A" else 3.0})
        if fmt == "B": net.neuron(1, {**{kk: -1.0 for kk in xb}, **net.gate(rs)}, 0.0, {Y[ID["0"]]: 3.0})
    if fmt == "A": net.obias(1, Y[ID["0"]], 1.0)
    for t in range(V): net.readout(t, Y[t], U)
    return net


def bin2dec(fmt):
    if fmt == "A":
        net = Net(13); F_ = net.dim("F")[0]; s = {9: 100, 10: 10, 11: 1}  # head 0: F = N / s(r) from the bits; head 1 subtracts the written digits
        rows0 = {r: {**{k: 2 ** (8 - k) / (255 * s[r]) for k in range(1, 9)}, 0: 1 - 1 / s[r]} for r in s}
        net.table(1, 0, rows0, [(NUMV(net), {F_: 255.0})])
        net.table(1, 1, {9: {0: 1.0}, 10: {10: 1 / 11, 0: 10 / 11}, 11: {10: 10 / 11, 11: 1 / 11}}, [(NUMV(net), {F_: -110.0})]); width, hi = 0.01, 9
    else:
        net = Net(34); F_ = net.dim("F")[0]; rows = {}  # F = 2 * previous value (digit) + new bit, per place
        for t in range(10, 34):
            k, p = (t - 10) // 3 + 1, (t - 10) % 3
            if p < 2: rows[t - 1] = {0: 1.0} if k == 1 else {t - 3: 2 / 3, t - 2: 0.2 / 3, 0: 0.8 / 3}
            else: rows[t - 1] = {k: 1 / 3, 0: 2 / 3} if k == 1 else {t - 3: 2 / 3, k: 1 / 3}
        net.table(1, 0, rows, [(NUMV(net), {F_: 3.0})]); width, hi = 0.1, 19
    Z = net.dim("Z", hi + 1); net.obias(1, Z[0], 1.0)  # Z[a] = [F >= a]
    for a in range(1, hi + 1): net.step({F_: 1.0}, a, width, {Z[a]: 1.0})
    for dd in range(10):
        for a, c in ((dd, 1), (dd + 1, -1), (dd + 10, 1), (dd + 11, -1)):
            if a <= hi: net.readout(ID[str(dd)], Z[a], c * U)
    return net


def dec2bin(fmt):
    if fmt == "A":
        net = Net(13); F_ = net.dim("F")[0]; Y = net.dim("Y", 2); i_of = {r: 11 - r for r in range(4, 12)}
        net.table(1, 0, {r: {1: 100 / (111 * 2 ** i), 2: 10 / (111 * 2 ** i), 3: 1 / (111 * 2 ** i), 0: 1 - 1 / 2 ** i} for r, i in i_of.items()},
                  [(NUMV(net), {F_: 111.0})])  # F = N / 2^i from the digits (mixture head)
        rows1 = {}
        for r, i in i_of.items():
            ws = {12 - j: 2 ** (j - i) / 254 for j in range(i + 1, 8)}; rows1[r] = {**ws, 0: 1 - sum(ws.values())}
        net.table(1, 1, rows1, [(NUMV(net), {F_: -254.0})])  # minus the already written higher bits: the remainder
        net.step({F_: 1.0}, 1.0, 1 / 128, {Y[1]: 2.0}); net.obias(1, Y[0], 1.0)  # bit = [F >= 1]
        net.readout(ID["0"], Y[0], U); net.readout(ID["1"], Y[1], U)
        return net
    net = Net(46); Cd, Xd, D, ONE = (net.dim(n)[0] for n in ("C", "X", "D", "ONE")); r0, r1, dg, bg = {}, {}, [], []
    for t in range(5, 46):
        r = t - 1
        if t <= 36:
            k, p = (t - 5) // 4 + 1, (t - 5) % 4 - 1
            if p < 0: r1[r] = {t - 1 if k > 1 else 3: 1.0}; r0[r] = {0: 1.0}; bg.append(r)
            else:
                cur = t - 4 if k > 1 else p + 1; r0[r] = {cur: 1.0}; r1[r] = {cur - 1: 1.0} if p > 0 else {0: 1.0}; dg.append(r)
        elif t == 37: r0[r] = r1[r] = {0: 1.0}
        else: r0[r] = {0: 1.0}; r1[r] = {5 + 4 * (45 - t): 1.0}; bg.append(r)
    net.table(1, 0, r0, [(NUMV(net), {Cd: 1.0})]); net.table(1, 1, r1, [(NUMV(net), {Xd: 1.0})])
    for rs, wt in ((dg, 5.0), (bg, 1.0)):  # D += wt * parity(X): hinges at 0..8, gated to rows rs
        for a in range(9): net.neuron(1, {Xd: 1.0, **{kk: 20.0 for kk in net.gate(rs)}}, -a - 20.0, {D: wt * (1 if a == 0 else 2 * (-1) ** a)})
    for a in range(1, 9): net.neuron(1, {Cd: 1.0}, -float(a), {D: (-1.0) ** (a + 1)})  # D += floor(C / 2)
    net.obias(1, ONE, 1.0)
    for dd in range(10): net.readout(ID[str(dd)], D, U * 2 * dd); net.readout(ID[str(dd)], ONE, U * (0.5 - dd * dd))
    net.readout(ID["="], net.d("POS", 36), U * 200)
    return net


def build(variant, fmt): return {"hex2bin": hex2bin, "bin2dec": bin2dec, "dec2bin": dec2bin}[variant](fmt)


# ---------------------------------------------------------------- forward pass, exactness
def forward(sd, ids, mid1=None, cache=None):
    """GPT-NeoX forward (sequential residual, ReLU MLP, no LayerNorm, no positions). mid1: optional fn replacing the residual after
    the last attention layer (substitution tests); cache (dict) receives it under 'mid1'. Returns logits (N, L, V)."""
    x = sd["gpt_neox.embed_in.weight"][ids]; N, L, D = x.shape; mask = torch.ones(L, L, dtype=torch.bool).tril()
    for l in range(NL):
        p = f"gpt_neox.layers.{l}."
        q, k, v = (x @ sd[p + "attention.query_key_value.weight"].T + sd[p + "attention.query_key_value.bias"]).view(N, L, NH, 3 * DH).transpose(1, 2).chunk(3, -1)
        A = ((q @ k.transpose(2, 3)) / math.sqrt(DH)).masked_fill(~mask, float("-inf")).softmax(-1)
        x = x + (A @ v).transpose(1, 2).reshape(N, L, D) @ sd[p + "attention.dense.weight"].T + sd[p + "attention.dense.bias"]
        if l == NL - 1:
            if cache is not None: cache["mid1"] = x
            if mid1 is not None: x = mid1(x)
        x = x + torch.relu(x @ sd[p + "mlp.dense_h_to_4h.weight"].T + sd[p + "mlp.dense_h_to_4h.bias"]) @ sd[p + "mlp.dense_4h_to_h.weight"].T + sd[p + "mlp.dense_4h_to_h.bias"]
    return x @ sd["lm_head.weight"].T


def hits(sd, ids, o, **kw):
    """Per input: all output tokens correct (teacher forced), (N,) bool."""
    with torch.no_grad(): return (forward(sd, ids, **kw)[:, o - 1:-1].argmax(-1) == ids[:, o:]).all(1)


def check(variant=None, fmt=None):
    """HAND exactness on all 256 inputs and the smallest logit margin, for every (variant, format) or the one given."""
    ok = True
    for v in ([variant] if variant else ["hex2bin", "bin2dec", "dec2bin"]):
        for f in ([fmt] if fmt else ["A", "B"]):
            ids, o = seqs(v, f); sd = build(v, f).sd
            with torch.no_grad(): lo = forward(sd, ids)[:, o - 1:-1]
            h = lo.argmax(-1) == ids[:, o:]; t2 = lo.topk(2, -1).values; m = (t2[..., 0] - t2[..., 1]).min().item()
            nz = sum(int((t != 0).sum()) for t in sd.values()); good = bool(h.all()) and m >= 1
            ok &= good; print(f"HAND {v} {f}: exact {int(h.all(1).sum())}/256, token acc {h.float().mean():.4f}, min margin {m:.2f}, {nz} non-zero weights", flush=True)
    return ok


# ---------------------------------------------------------------- PART / TRAIN
STAGE = 1  # the trained stage of PART: the layer-1 MLP
STAGE_KEYS = [f"gpt_neox.layers.{STAGE}.mlp.{a}.{b}" for a in ("dense_h_to_4h", "dense_4h_to_h") for b in ("weight", "bias")]


def split(n, frac, seed):
    g = torch.Generator().manual_seed(seed); p = torch.randperm(n, generator=g); k = int(frac * n); return p[:k], p[k:]


def init(variant, fmt, rung, seed):
    """PART: HAND with the layer-1 MLP re-initialised (weights N(0, .02), biases 0). TRAIN: every weight N(0, .02), biases 0 (HF initialiser)."""
    sd = build(variant, fmt).sd; g = torch.Generator().manual_seed(seed)
    for k in (STAGE_KEYS if rung == "part" else list(sd)):
        sd[k] = torch.randn(sd[k].shape, generator=g) * 0.02 if k.endswith("weight") else torch.zeros_like(sd[k])
    return sd


def train(variant, fmt, rung, seed, a):
    """One network. a: steps, every, frac, bs, lr, wd. AdamW lr 1e-3, weight decay 1.0, betas (.9, .98), minibatch bs, CE on output tokens.
    Returns (final sd, best-test sd, log, test indices). Seed sets the init, the split and the batch order."""
    ids, o = seqs(variant, fmt); tr, te = split(len(ids), a.frac, seed); sd = init(variant, fmt, rung, seed)
    keys = STAGE_KEYS if rung == "part" else list(sd)
    for k in keys: sd[k].requires_grad_()
    opt = torch.optim.AdamW([sd[k] for k in keys], lr=a.lr, weight_decay=a.wd, betas=(.9, .98)); g = torch.Generator().manual_seed(1234 + seed)
    log, best, bt = [], None, -1.0
    for t in range(a.steps + 1):
        if t % a.every == 0 or t == a.steps:
            rtr, rte = hits(sd, ids[tr], o).float().mean().item(), hits(sd, ids[te], o).float().mean().item(); log.append(dict(step=t, train=rtr, test=rte))
            if rte > bt: bt, best = rte, {k: v.detach().clone() for k, v in sd.items()}
            print(f"{rung} s{seed} step {t} train {rtr:.3f} test {rte:.3f}", flush=True)
            if rtr >= 1 and rte >= 1: break
        if t == a.steps: break
        b = ids[tr[torch.randint(0, len(tr), (a.bs,), generator=g)]]
        loss = F.cross_entropy(forward(sd, b)[:, o - 1:-1].reshape(-1, V), b[:, o:].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return {k: v.detach() for k, v in sd.items()}, best, log, te


# ---------------------------------------------------------------- analyses
def hand_variable(variant, fmt):
    """HAND's layer-1 variable at every position: the nibble one-hot X (hex2bin) or the remainder scalar F (bin2dec / dec2bin A)."""
    ids, o = seqs(variant, fmt); net = build(variant, fmt); c = {}
    with torch.no_grad(): forward(net.sd, ids, cache=c)
    return c["mid1"][..., net.dims["X" if variant == "hex2bin" else "F"]]


def substitute(sd, variant, fmt):
    """Substitution check. At every scored position the residual after the last attention layer is replaced by its least-squares fit
    (with intercept, all 256 inputs) on HAND's variable; returns (exact of the net, exact with the substitution) on all inputs.
    hex2bin: the one-hot of the source hex digit. dec2bin A / bin2dec A: the remainder scalar F."""
    ids, o = seqs(variant, fmt); var = hand_variable(variant, fmt)

    def fn(x):
        x = x.clone()
        for q in range(o - 1, ids.shape[1] - 1):
            Z = torch.cat([var[:, q], torch.ones(len(ids), 1)], 1).double(); B = torch.linalg.lstsq(Z, x[:, q].double(), driver="gelsd").solution
            x[:, q] = (Z @ B).float()
        return x
    return hits(sd, ids, o).float().mean().item(), hits(sd, ids, o, mid1=fn).float().mean().item()


def thresholds(sd):
    """dec2bin A: is each answer bit a single threshold on HAND's remainder F? Per answer position the best theta (pred = [F >= theta]),
    its agreement with the net over all 256 inputs, (theta = 1 is HAND's rule)."""
    ids, o = seqs("dec2bin", "A"); var = hand_variable("dec2bin", "A")
    with torch.no_grad(): pred = forward(sd, ids)[:, o - 1:-1].argmax(-1) == ID["1"]
    out = []
    for q in range(pred.shape[1]):
        Fq = var[:, o - 1 + q, 0]; ths = Fq.unique(); agr = torch.stack([((Fq >= t) == pred[:, q]).float().mean() for t in ths]); j = agr.argmax()
        out.append((round(ths[j].item(), 3), round(agr[j].item(), 3)))
    return out
