"""Command line for the circuit-atlas experiments (see ../METHODS.md). Run from this directory; CPU is enough.

  python run.py check                                   # exactness of the hand-built model and the menus
  python run.py modadd --seeds 0 1 --steps 300000       # (x + y) mod 59: train the glue on the frozen menu, analyse (entry modular-addition-mlp)
  python run.py atlas --task count_mod5 --frac .4 --seeds 0 1 --steps 200000      # also compare_counts, open_count
  python run.py atlas --task compare_counts --kind random --seeds 0 --probe       # random-base control (+ ridge probes)
  python run.py table out/*.json                        # aggregate atlas results over seeds
  python run.py convert check                           # base conversion: the 6 hand-built networks are exact on all 256 inputs
  python run.py convert --variant hex2bin --fmt A --rung part --seeds 0 1 --steps 10000   # PART / TRAIN + substitution check (entry base-conversion)

Options: --kind m2|scalar|random (menu / scalar-only / random frozen base), --every N (evaluation interval), --out DIR (default out/).
Each run writes DIR/<name>.json with the learning curve, the usage (ablation) table and the analysis; weights are not saved."""
import argparse, json, os, sys, time
from pathlib import Path
import torch
import analyze_atlas, analyze_modadd, atlas, convert, m2, menu, train as T


def check():
    sd = m2.state_dict(); rng = __import__("random").Random(0); ok = 0
    texts = [atlas.SHOT["count"] + "COUNT a: " + " ".join(rng.sample(["a"] * c + ["b"] * (9 - c), 9)) + "\nANSWER: " for c in range(10)]
    ids = torch.tensor([[m2.ID[ch] for ch in t] for t in texts if len(t) == len(texts[0])])
    lg = __import__("neox").forward(sd, m2.CFG, ids)[1][:, -1].argmax(-1).tolist()
    print("hand-built count head, c = 0..9 (L = 9): answers", [m2.CH[i - 2] for i in lg])
    print("modular-addition menu (p = 59):", menu.cache(59)[4])
    for fam in ("count", "brackets"):
        i = atlas.cache(fam)[3]; print(f"atlas menu {fam}:", {k: v for k, v in i.items() if k != "live"})


def save(out, name, res):
    out.mkdir(parents=True, exist_ok=True); (out / f"{name}.json").write_text(json.dumps(res))


def run_modadd(a):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4))); t0 = time.time(); a.frac = a.frac or 0.4
    for s in a.seeds:
        X, lab, xy, sl, info = menu.cache(a.p, a.kind, s)
        m, tr, te, log = T.train(X, lab, sl, a.p, s, a)
        res = dict(kind=a.kind, seed=s, args=vars(a), menu_info=info, slices=sl, fit_step=T.first(log, "train_acc"), grok_step=T.first(log, "test_acc"),
                   final={k: log[-1][k] for k in ("train_acc", "test_acc", "train_loss", "test_loss", "wnorm")}, log=log,
                   analysis=analyze_modadd.analyze(m, X, lab, xy, sl, a.p, te) if a.kind != "random" else {})
        save(Path(a.out), f"modadd_{a.kind}_s{s}", res); an = res["analysis"]
        print(f"modadd {a.kind} s{s}: train99 step {res['fit_step']}, test 50%/90% step {analyze_modadd.steps_to(log, .5)}/{analyze_modadd.steps_to(log, .9)}, "
              f"final test {res['final']['test_acc']:.3f}, wnorm {res['final']['wnorm']:.1f}, {time.time() - t0:.0f}s", flush=True)
        if an: print(f"  type {an['type']}, dd {an['dd']:.2f}, phase err {an['phase_err_med']:.2f}, neuron-formula acc {an['neuron_formula_acc']:.3f} "
                     f"(agree {an['neuron_formula_agree']:.3f}), analytic acc {an['analytic_acc']:.3f}, logit top-5 freqs {list(an['logit_freq_amp'])[:5]}", flush=True)


def run_convert(a):
    """Base conversion: train one network per seed (rung part | train) on a random half (--frac) of x = 0..255 and analyse it."""
    if a.files == ["check"]: sys.exit(0 if convert.check() else 1)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4))); a.frac = a.frac or 0.5; t0 = time.time()
    for s in a.seeds:
        sd, best, log, te = convert.train(a.variant, a.fmt, a.rung, s, a); r = {}
        for tag, w in (("final", sd), ("best", best)):
            full, sub = convert.substitute(w, a.variant, a.fmt); r[tag] = dict(exact_all=round(full, 3), exact_substituted=round(sub, 3))
            if (a.variant, a.fmt) == ("dec2bin", "A"): r[tag]["thresholds"] = convert.thresholds(w)
        res = dict(variant=a.variant, fmt=a.fmt, rung=a.rung, seed=s, args=vars(a), test_idx=te.tolist(), log=log, analysis=r,
                   best_test=max(e["test"] for e in log), final_test=log[-1]["test"], grok_step=next((e["step"] for e in log if e["test"] >= .99), None))
        save(Path(a.out), f"convert_{a.variant}_{a.fmt}_{a.rung}_s{s}", res)
        print(f"convert {a.variant} {a.fmt} {a.rung} s{s}: best test {res['best_test']:.3f} final {res['final_test']:.3f} grok step {res['grok_step']}, "
              f"substitution (best net) {r['best']}, {time.time() - t0:.0f}s", flush=True)


def run_atlas(a):
    atlas.LMAX = a.lmax; fam, f, ncls = atlas.TASKS[a.task]; ncls = ncls()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4))); t0 = time.time(); a.frac = a.frac or 0.4; bases = {}
    for s in a.seeds:
        key = s if a.kind == "random" else 0
        if key not in bases: bases[key] = atlas.cache(fam, a.kind, s)
        X, uv, sl, info = bases[key]; lab = torch.tensor([f(u, L) for u, L in uv.tolist()])
        m, tr, te, log = T.train(X, lab, sl, ncls, s, a)
        sd = m.state_dict()
        res = dict(task=a.task, kind=a.kind, seed=s, args=vars(a), menu_info=info, slices=sl, fit_step=T.first(log, "train_acc"), grok_step=T.first(log, "test_acc"),
                   final={k: log[-1][k] for k in ("train_acc", "test_acc", "train_loss", "test_loss", "wnorm")}, usage=atlas.usage(m, X, lab, sl, te), log=log,
                   w={"W1": sd["0.weight"], "b1": sd["0.bias"], "W2": sd["2.weight"], "b2": sd["2.bias"], "test": te})
        r = analyze_atlas.row(res, {"X": X, "uv": uv, "sl": sl}); res.pop("w")
        if a.probe and a.kind == "random": r["probe"] = analyze_atlas.probe_random(fam, s, X, uv, tr, te)
        res["row"] = r; name = f"{a.task}_{a.kind}_s{s}" + (f"_f{round(a.frac * 100)}" if a.frac != 0.4 else "")
        save(Path(a.out), name, res)
        print(name, f"{time.time() - t0:.0f}s", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["check", "modadd", "atlas", "table", "convert"]); ap.add_argument("files", nargs="*")
    ap.add_argument("--task", default="count_mod5", choices=list(atlas.TASKS)); ap.add_argument("--kind", default="m2", choices=["m2", "scalar", "random"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--steps", type=int, default=None); ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--frac", type=float, default=None); ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--p", type=int, default=59); ap.add_argument("--lmax", type=int, default=45); ap.add_argument("--probe", action="store_true")
    ap.add_argument("--out", default="out")
    ap.add_argument("--variant", default="hex2bin", choices=["hex2bin", "bin2dec", "dec2bin"]); ap.add_argument("--fmt", default="A", choices=["A", "B"])
    ap.add_argument("--rung", default="part", choices=["part", "train"]); ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "check": check()
    if a.cmd == "modadd": a.steps = a.steps or 300000; run_modadd(a)
    if a.cmd == "atlas": a.steps = a.steps or 200000; run_atlas(a)
    if a.cmd == "convert": a.steps = a.steps or 10000; run_convert(a)
    if a.cmd == "table": print(analyze_atlas.table([json.loads(Path(f).read_text())["row"] for f in a.files]))
