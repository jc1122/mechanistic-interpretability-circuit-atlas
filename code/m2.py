"""Hand-built GPT-NeoX-shaped transformer "m2": 6 layers, d_model 528, 4 heads, MLP width 512, every weight set by a formula.
One function per brick; each writes into a state dict (HF GPTNeoX parameter names) and records which residual blocks it writes
(BRICKS). The model is evaluated with neox.py (no external model library). Tasks it solves by itself: copy, key-value lookup, yes/no
membership, count of a letter (digit answer), bracket validity, compare two numbers.

    sd = state_dict()            # dict name -> float32 tensor
    python run.py check          # exactness checks
"""
import math
import torch
import neox

CH = " :ER\nASNWOedfacbTCL>-1PD()20VM|4953786I_YXBHKFzxy+GUkmgsrhjilntopq<="  # the 68 characters of the prompts; token id = index + 2
V = len(CH) + 2  # + <|endoftext|> (0) and <unk> (1)
ID = {c: i + 2 for i, c in enumerate(CH)}
LOW, DIG = [ID[c] for c in CH if c.islower()], [ID[c] for c in "0123456789"]
NL, NH, D, DM = 6, 4, 528, 512
DH = D // NH
K, RS, J, OMAX = 1000.0, 1024.0, 500, 63  # constant dim, r scale, position knots, max offset
A, G, B, M, U = 15.0, 5000.0, 1000.0, 1500.0, 30.0  # recency per position, class gate, offset match, token match, readout
AS, GS = 3.0, 300.0  # segment heads (rec RA): small constants keep the centered scores exact in fp32
SQ = math.sqrt(DH)  # cancels the 1/sqrt(d_head) score scaling

DIMS, _n = {}, 0
for name, n in [("ONE", 2), ("Z", 1), ("TOK", V), ("M", V), ("F", 1), ("FC", 1), ("FK", 1), ("FH", 1), ("P", 1), ("R", 1),
                ("RA", 1), ("O", 1), ("O2", 1), ("NLF", 1), ("LL", V), ("CP", V), ("KV", V), ("HQ", V), ("NO", 1), ("OUT", V),
                ("FN", 1), ("FM", 1), ("FB", 1), ("X", 1), ("Y", 1), ("NEG", 1), ("BAL", 1), ("ANY", 1), ("N", 1), ("PD", 1),
                ("NA", 1), ("NB", 1)]:
    DIMS[name] = list(range(_n, _n + n)); _n += n
assert _n <= D
d1 = lambda n: DIMS[n][0]
CFG = dict(vocab_size=V, hidden_size=D, num_hidden_layers=NL, num_attention_heads=NH, intermediate_size=DM, rotary_pct=0.0,
           use_parallel_residual=False, hidden_act="gelu", layer_norm_eps=1e-5)  # GPTNeoXConfig fields: no rotary, sequential residual
BRICKS = []  # {name, layer, unit: "embed" | "unembed" | ("head", h) | ("mlp", lo, hi), writes: [block]}
_mlp_next = {}
S0 = math.sqrt((2 * K * K + 1) / D - (1 / D) ** 2)  # LN std of the embedding row; every LN gain = S0, so LN(x) = x - mean


def pre(l): return f"gpt_neox.layers.{l}."


def head(sd, l, h, part):
    """(weight rows, bias) of q (0), k (1) or v (2) of head h (HF layout: per head [q | k | v], DH rows each)."""
    s = slice((3 * h + part) * DH, (3 * h + part + 1) * DH)
    return sd[pre(l) + "attention.query_key_value.weight"][s], sd[pre(l) + "attention.query_key_value.bias"][s]


def value(sd, l, h, src, dst, w=1.0):
    """Head h copies residual dims src -> dst (value row i reads src[i], dense writes dst[i])."""
    v, _ = head(sd, l, h, 2); o = sd[pre(l) + "attention.dense.weight"]
    for i, (a, b) in enumerate(zip(src, dst)):
        v[i, a] = w; o[b, h * DH + i] = 1.0


def vlin(sd, l, h, rows):
    """Head h value rows: rows = [({src_dim: coef}, dst_dim)], row i = sum coef*resid[src] written to dst."""
    v, _ = head(sd, l, h, 2); o = sd[pre(l) + "attention.dense.weight"]
    for i, (src, dst) in enumerate(rows):
        for a, w in src.items(): v[i, a] = w
        o[dst, h * DH + i] = 1.0


def center(sd, l, h, i, g):
    """Segment heads: q.k row i = -(g + A r_anchor(i)) on a constant key 1 (ONE/K), so the tied keys score exactly 0 and LN's
    per-key scale (1 + eps_j) cannot break the tie (scores of ~1e4 gave ~1e-3 weight errors)."""
    q, bq = head(sd, l, h, 0); k, _ = head(sd, l, h, 1)
    q[i, d1("RA")] = -AS * RS * SQ; bq[i] = -g * SQ; k[i, DIMS["ONE"][0]] = 1 / K


def neurons(l, n):
    lo = _mlp_next.get(l, 0); _mlp_next[l] = lo + n; assert lo + n <= DM
    return lo


def step(sd, l, n0, x, c, s, out):
    """Two GELU neurons: 1 if sum(coef*resid[dim]) >= c + 16/s, 0 if <= c - 16/s; written as out {dim: coef}."""
    W, b = sd[pre(l) + "mlp.dense_h_to_4h.weight"], sd[pre(l) + "mlp.dense_h_to_4h.bias"]
    Wo = sd[pre(l) + "mlp.dense_4h_to_h.weight"]
    for j, hh in ((n0, 8.0), (n0 + 1, -8.0)):
        for dim, cf in x.items(): W[j, dim] = s * cf
        b[j] = -s * c + hh
        for dim, cf in out.items(): Wo[dim, j] = cf / 16.0 * (1 if hh > 0 else -1)


# ---- bricks ----
def embed(sd):
    """Token one-hot in TOK + constant mirrored pair ONE = (+K, -K): the BOS/one dim that keeps every LN a fixed scale."""
    E = sd["gpt_neox.embed_in.weight"]
    for t in range(V): E[t, DIMS["TOK"][t]] = 1.0
    E[:, DIMS["ONE"][0]], E[:, DIMS["ONE"][1]] = K, -K
    BRICKS.append(dict(name="embed", layer=-1, unit="embed", writes=["TOK", "ONE"]))


def ln_guard(sd):
    """Every LN gain S0, bias 0; readers are zero-sum (finish()), so each LN acts as the fixed map x -> x (up to 1e-5)."""
    for k in sd:
        if "layernorm" in k or "layer_norm" in k:
            sd[k].fill_(S0 if k.endswith("weight") else 0.0)


def uniform_head(sd, l, h, src, dst, name):
    """q = k = 0: attend uniformly to all positions <= i; output = prefix mean of src."""
    value(sd, l, h, DIMS[src], DIMS[dst]); BRICKS.append(dict(name=name, layer=l, unit=("head", h), writes=[dst]))


def recent_head(sd, l, h, cls, src, dst, name, extra=None, rec="R", rows=None, writes=None):
    """Attend to the most recent key j <= i with token in cls (score G[tok_j in cls] + G*extra_j + A*r_j); copy src -> dst.
    rec="RA": recency by the key's anchor r instead (all keys of one ':' segment tie: uniform over the latest segment that
    holds a class key). rows: linear value rows (vlin) instead of the src -> dst copy; dst then names the written block."""
    q, bq = head(sd, l, h, 0); k, _ = head(sd, l, h, 1)
    bq[0] = bq[1] = SQ
    g, a = (GS, AS) if rec == "RA" else (G, A)
    for t in cls: k[0, DIMS["TOK"][t]] = g
    if extra: k[0, d1(extra)] = g
    k[1, d1(rec)] = a * RS
    if rec == "RA": center(sd, l, h, 2, g if cls else 0.0)
    vlin(sd, l, h, rows) if rows else value(sd, l, h, src, dst)
    BRICKS.append(dict(name=name, layer=l, unit=("head", h), writes=writes or [name_of(dst)]))


def name_of(dims): return next(n for n, v in DIMS.items() if v[0] == dims[0])


def first_flag(sd, l):
    """MLP: F = 1 iff the prefix mean of the own token is 1 (only position 0); task flags from header letters, one per task:
    FC 'P' minus 'M' (COPY, not COMPARE), FK 'K' minus 'B' (KV, not BRACKETS), FH 'H', FN 'U' (COUNT), FM 'M', FB 'B'."""
    fl = [({"P": 1, "M": -1}, "FC"), ({"K": 1, "B": -1}, "FK"), ({"H": 1}, "FH"), ({"U": 1}, "FN"), ({"M": 1}, "FM"), ({"B": 1}, "FB")]
    n0 = neurons(l, 2 * V + 2 * len(fl))
    for t in range(V):
        step(sd, l, n0 + 2 * t, {DIMS["M"][t]: 1, DIMS["TOK"][t]: 1}, 1.75, 64.0, {d1("F"): 1})
    for i, (x, f) in enumerate(fl):
        step(sd, l, n0 + 2 * V + 2 * i, {DIMS["M"][ID[c]]: w for c, w in x.items()}, 8e-4, 2e4, {d1(f): 1})
    BRICKS.append(dict(name="first_flag", layer=l, unit=("mlp", n0, n0 + 2 * V + 2 * len(fl)), writes=["F"] + [f for _, f in fl]))


def position(sd, l):
    """MLP: r = 1/p as a piecewise-linear interpolant with knots at p_j = 1/(j+1), j <= J (exact at knots): R = r/RS."""
    n0 = neurons(l, J); W, b = sd[pre(l) + "mlp.dense_h_to_4h.weight"], sd[pre(l) + "mlp.dense_h_to_4h.bias"]
    Wo, bo = sd[pre(l) + "mlp.dense_4h_to_h.weight"], sd[pre(l) + "mlp.dense_4h_to_h.bias"]
    for k in range(1, J + 1):
        pk1, pk = 1 / k, 1 / (k + 1); s = 8.0 / (pk1 - pk); j = n0 + k - 1
        W[j, d1("P")] = -s; b[j] = s * pk1; Wo[d1("R"), j] = 2 * k / s / RS  # hinge c_k = 2k on max(0, p_{k-1} - p)
    bo[d1("R")] = 1 / RS
    BRICKS.append(dict(name="position", layer=l, unit=("mlp", n0, n0 + J), writes=["R"]))


def offset(sd, l):
    """MLP: o = r - r_anchor snapped to integers 0..OMAX by steps at k - 1/2: O = o/64, O2 = o^2/4096."""
    n0 = neurons(l, 2 * OMAX)
    for k in range(1, OMAX + 1):
        step(sd, l, n0 + 2 * (k - 1), {d1("R"): RS, d1("RA"): -RS}, k - 0.5, 40.0, {d1("O"): 1 / 64, d1("O2"): (2 * k - 1) / 4096})
    BRICKS.append(dict(name="offset", layer=l, unit=("mlp", n0, n0 + 2 * OMAX), writes=["O", "O2"]))


def copy_head(sd, l, h, shift=1, rows=None, name="copy", writes=("CP",)):
    """Most recent key with o_j = o_i + shift: score B(2 o_j (o_i+shift) - o_j^2) + A r_j; copies its token -> CP (or rows)."""
    q, bq = head(sd, l, h, 0); k, _ = head(sd, l, h, 1)
    q[0, d1("O")] = 2 * B * 64 * SQ; bq[0] = 2 * B * shift * SQ; bq[1] = -B * SQ; bq[2] = A * SQ
    k[0, d1("O")] = 64; k[1, d1("O2")] = 4096; k[2, d1("R")] = RS
    vlin(sd, l, h, rows) if rows else value(sd, l, h, DIMS["TOK"], DIMS["CP"])
    BRICKS.append(dict(name=name, layer=l, unit=("head", h), writes=list(writes)))


def match_head(sd, l, h, qsrc, ksrc_tokens, gate, src, dst, name, rec="R"):
    """Most recent key whose token (or LL letter) equals the query letter qsrc: M[match] + G'[gate tokens] + A r_j.
    rec="RA": uniform over the matches and gate tokens of the latest ':' segment that holds any (count)."""
    q, bq = head(sd, l, h, 0); k, _ = head(sd, l, h, 1)
    mt, a = (GS, AS) if rec == "RA" else (M, A)
    for i, t in enumerate(LOW):
        q[i, DIMS[qsrc][t]] = mt * SQ; k[i, DIMS[ksrc_tokens][t]] = 1.0
        if rec == "RA": q[V, DIMS[qsrc][t]] = mt * SQ  # gate scaled like the match (qsrc is a soft one-hot)
    bq[V], bq[V + 1] = (0.0 if rec == "RA" else M * SQ), a * SQ
    for t in gate: k[V, DIMS["TOK"][t]] = 1.0
    k[V + 1, d1(rec)] = RS
    if rec == "RA": center(sd, l, h, V + 2, 0.0); q[V + 2, :] -= q[V]  # minus the (soft) gate: tied keys score 0
    value(sd, l, h, src, dst); BRICKS.append(dict(name=name, layer=l, unit=("head", h), writes=[name_of(dst)]))


BIGRAM = {":": " ", **{d: "\n" for d in "0123456789<=>"}, "Y": "E", "E": "S", "S": "\n", "N": "O", "O": "\n"}


def router(sd, l):
    """MLP: OUT = CP if copy; else at ' ' the kv digit or YES/NO first letter; else the BIGRAM of the current token."""
    n = 2 * V + 2 * len(DIG) + 4 + 2 * len(BIGRAM); n0 = neurons(l, n); j = n0; o = DIMS["OUT"]; T = DIMS["TOK"]; sp = T[ID[" "]]
    for t in range(V):
        step(sd, l, j, {DIMS["CP"][t]: 1, d1("FC"): 1}, 1.5, 32.0, {o[t]: 1}); j += 2
    for t in DIG:
        step(sd, l, j, {DIMS["KV"][t]: 1, d1("FK"): 1, sp: 1}, 2.5, 32.0, {o[t]: 1}); j += 2
    step(sd, l, j, {d1("FH"): 1, sp: 1, d1("NO"): -1}, 1.5, 32.0, {o[ID["Y"]]: 1}); j += 2
    step(sd, l, j, {d1("FH"): 1, sp: 1, d1("NO"): 1}, 2.5, 32.0, {o[ID["N"]]: 1}); j += 2
    for a, b in BIGRAM.items():
        step(sd, l, j, {T[ID[a]]: 1, d1("FC"): -1}, 0.5, 32.0, {o[ID[b]]: 1}); j += 2
    BRICKS.append(dict(name="router", layer=l, unit=("mlp", n0, n0 + n), writes=["OUT"]))


def mlp_brick(sd, l, name, writes, units):
    """MLP brick from step pairs: units = [(x, c, s, out)] (see step)."""
    n0 = neurons(l, 2 * len(units))
    for i, (x, c, s, out) in enumerate(units): step(sd, l, n0 + 2 * i, x, c, s, out)
    BRICKS.append(dict(name=name, layer=l, unit=("mlp", n0, n0 + 2 * len(units)), writes=writes))


def count_digit(sd, l):
    """X = 1/(c+1) -> digit c (c <= 9) at the answer slot of COUNT: T_c = [X >= mid(1/(c+1), 1/(c+2))] writes +OUT_c -OUT_(c+1)."""
    o, g = DIMS["OUT"], {d1("FN"): 2000, DIMS["TOK"][ID[" "]]: 2000}
    mlp_brick(sd, l, "count_digit", ["OUT"], [({d1("X"): 1000, **g}, 500 * (1 / (c + 1) + 1 / (c + 2)) + 4000, 6.0,
                                               {o[DIG[c]]: 1, **({o[DIG[c + 1]]: -1} if c < 9 else {})}) for c in range(10)])


def neg_flag(sd, l):
    """NEG = 1 iff the segment balance mean Y < 0 (|Y| >= 1/19 when nonzero)."""
    mlp_brick(sd, l, "neg_flag", ["NEG"], [({d1("Y"): -1}, 0.028, 2000.0, {d1("NEG"): 1})])


def brackets_answer(sd, l):
    """At the VALID_BRACKETS slot: NO iff some prefix balance < 0 (ANY > 0) or the final balance > 0 (BAL > 0); else YES."""
    o, g = DIMS["OUT"], {d1("FB"): 100, DIMS["TOK"][ID[" "]]: 100}; Y_, N_ = o[ID["Y"]], o[ID["N"]]
    mlp_brick(sd, l, "brackets_answer", ["OUT"], [(g, 150, 32.0, {Y_: 1}),
                                                  ({d1("ANY"): 40, **g}, 201, 32.0, {N_: 1, Y_: -1}),
                                                  ({d1("BAL"): 40, d1("ANY"): -4000, **g}, 201, 32.0, {N_: 1, Y_: -1})])


def compare_answer(sd, l):
    """At the COMPARE slot (NA, NB = numbers/100): '>' if a - b >= 1, '<' if <= -1, else '='."""
    o, g = DIMS["OUT"], {d1("FM"): 200, DIMS["TOK"][ID[" "]]: 200}; gt, lt, eq = o[ID[">"]], o[ID["<"]], o[ID["="]]
    mlp_brick(sd, l, "compare_answer", ["OUT"], [(g, 300, 40.0, {eq: 1}),
                                                 ({d1("NA"): 100, d1("NB"): -100, **g}, 400.5, 40.0, {gt: 1, eq: -1}),
                                                 ({d1("NA"): -100, d1("NB"): 100, **g}, 400.5, 40.0, {lt: 1, eq: -1})])


def readout(sd):
    """logit_t = U * OUT_t (large margin)."""
    W = sd["lm_head.weight"]
    for t in range(V): W[t, DIMS["OUT"][t]] = U
    BRICKS.append(dict(name="readout", layer=NL, unit="unembed", writes=[]))


def finish(sd):
    """Zero-sum readers: every row reading an LN output gets -(row sum) on Z (a dim that is always 0), so LN's mean drops out."""
    z = d1("Z")
    for k in sd:
        if k.endswith(("query_key_value.weight", "dense_h_to_4h.weight")) or k == "lm_head.weight":
            sd[k][:, z] -= sd[k].sum(1)


def state_dict():
    BRICKS.clear(); _mlp_next.clear()
    sd = neox.zeros(CFG)
    T = DIMS["TOK"]; br = {T[ID["("]]: 1, T[ID[")"]]: -1}; dig = lambda w: {T[t]: w * i for i, t in enumerate(DIG)}
    embed(sd); ln_guard(sd)
    uniform_head(sd, 0, 0, "TOK", "M", "prefix_mean"); first_flag(sd, 0)
    uniform_head(sd, 1, 0, "F", "P", "inv_pos"); position(sd, 1)
    recent_head(sd, 2, 0, [ID[":"]], DIMS["R"], DIMS["RA"], "anchor")
    recent_head(sd, 2, 1, LOW, DIMS["TOK"], DIMS["LL"], "last_lower")
    recent_head(sd, 2, 2, [ID[":"], ID["\n"]], [T[ID["\n"]]], DIMS["NLF"], "nl_last")
    offset(sd, 2)
    copy_head(sd, 3, 0)
    match_head(sd, 3, 1, "LL", "LL", DIG, DIMS["TOK"], DIMS["KV"], "kv")
    recent_head(sd, 3, 2, LOW, DIMS["TOK"], DIMS["HQ"], "has_letter", extra="NLF")
    recent_head(sd, 3, 3, [], None, DIMS["Y"], "bal_prefix", rec="RA", rows=[(br, d1("Y"))]); neg_flag(sd, 3)
    match_head(sd, 4, 0, "HQ", "TOK", LOW, DIMS["NLF"], DIMS["NO"], "contains")
    match_head(sd, 4, 1, "HQ", "TOK", [ID["\n"]], [T[ID["\n"]]], DIMS["X"], "count", rec="RA")
    copy_head(sd, 4, 2, -1, [(dig(.01), d1("N")), ({T[t]: 1 for t in DIG}, d1("PD"))], "prev1", ("N", "PD"))
    copy_head(sd, 4, 3, -2, [(dig(.1), d1("N"))], "prev2", ("N",))
    router(sd, 4); count_digit(sd, 4)
    recent_head(sd, 5, 0, [ID["("], ID[")"], ID["\n"]], None, DIMS["BAL"], "brackets_seg", rec="RA",
                rows=[(br, d1("BAL")), ({d1("NEG"): 1}, d1("ANY"))], writes=["BAL", "ANY"])
    recent_head(sd, 5, 1, [ID["\n"]], DIMS["N"], DIMS["NB"], "cmp_b")
    recent_head(sd, 5, 2, [ID[" "]], DIMS["N"], DIMS["NA"], "cmp_a", extra="PD")
    brackets_answer(sd, 5); compare_answer(sd, 5)
    readout(sd); finish(sd)
    return sd
