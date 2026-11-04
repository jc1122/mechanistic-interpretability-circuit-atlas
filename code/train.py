"""Train the glue on the frozen menu: logits = W_out ReLU(W_in x + b_in) + b_out (MLP only: the menu already brings the operands to the
answer slot). AdamW lr 1e-3, weight decay 1.0, betas (.9, .98), full batch, hidden width 256, fp32. Seed s sets both the init and the
train/test split (a random permutation of the cells; the first frac * N are training cells)."""
import torch, torch.nn as nn, torch.nn.functional as F


def net(d, p, w, seed):
    torch.manual_seed(seed); return nn.Sequential(nn.Linear(d, w), nn.ReLU(), nn.Linear(w, p))


def split(n, frac, seed):
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(n, generator=g); k = int(frac * n)
    return perm[:k], perm[k:]


def train(X, lab, sl, p, seed, a):
    """a: namespace with frac, width, lr, wd, steps, every. Logs train/test accuracy and loss and the weight norm every `every` steps."""
    tr, te = split(len(X), a.frac, seed); m = net(X.shape[1], p, a.width, seed)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98)); log = []
    for t in range(a.steps + 1):
        if t % a.every == 0:
            with torch.no_grad():
                lo = m(X); W = m[0].weight
                row = dict(step=t, wnorm=sum(q.pow(2).sum() for q in m.parameters()).sqrt().item(),
                           brick_norm={n: W[:, i:j].norm().item() for n, (i, j) in sl.items()})
                for s, ix in (("train", tr), ("test", te)):
                    row[s + "_loss"] = F.cross_entropy(lo[ix], lab[ix]).item(); row[s + "_acc"] = (lo[ix].argmax(1) == lab[ix]).float().mean().item()
                log.append(row)
        if t == a.steps: break
        loss = F.cross_entropy(m(X[tr]), lab[tr]); opt.zero_grad(); loss.backward(); opt.step()
    return m, tr, te, log


def first(log, key, thr=0.99):
    """First logged step with metric >= thr (fit step: train_acc >= .99; grok step: test_acc >= .99), else None."""
    return next((r["step"] for r in log if r[key] >= thr), None)
