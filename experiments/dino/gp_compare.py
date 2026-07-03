#!/usr/bin/env python3
"""Gardens Point day->night: DBoW2 (AirSLAM) vs DINO/AnyLoc, all on one plot.
GT: night frame i matches day frame j if |i-j| <= tol (frame-aligned traversals).

usage: gp_compare.py <feat_cache.npz> <dbow_matrix.txt> <out.png> [--tol 3]
  feat_cache = the <out>.feat.npz written by cross_session.py (day=A, night=B)
"""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("feat"); ap.add_argument("dbow"); ap.add_argument("out")
ap.add_argument("--tol", type=int, default=3)
ap.add_argument("--k", type=int, default=32)
a = ap.parse_args()

z = np.load(a.feat)
denseA, denseB = z["denseA"], z["denseB"]
NA, NB = len(denseA), len(denseB)
D = denseA.shape[-1]

def l2n(X, ax=1):
    return X / (np.linalg.norm(X, axis=ax, keepdims=True) + 1e-8)

# rebuild VLAD (vocab on day=A) to match cross_session
from sklearn.cluster import KMeans
flat = denseA.reshape(-1, D)
rng = np.random.default_rng(0)
sub = flat[rng.choice(len(flat), size=min(50000, len(flat)), replace=False)]
C = KMeans(n_clusters=a.k, n_init=4, random_state=0).fit(sub).cluster_centers_.astype(np.float32)

def vlad(dense):
    out = np.zeros((len(dense), a.k * D), np.float32)
    for n, F in enumerate(dense):
        asg = np.argmin(((F[:, None, :] - C[None]) ** 2).sum(-1), 1)
        V = np.zeros((a.k, D), np.float32)
        for c in range(a.k):
            if np.any(asg == c):
                V[c] = (F[asg == c] - C[c]).sum(0)
        out[n] = l2n(l2n(V, 1).ravel()[None])[0]
    return out

# similarity matrices (query night B x map day A)
sims = {}
sims["dbow(AirSLAM)"] = np.loadtxt(a.dbow, skiprows=1)           # (NB, NA)
sims["tiny"] = l2n(z["tinyB"]) @ l2n(z["tinyA"]).T
sims["cls"] = l2n(z["clsB"]) @ l2n(z["clsA"]).T
sims["mean"] = l2n(z["meanB"]) @ l2n(z["meanA"]).T
sims["vlad(AnyLoc)"] = vlad(denseB) @ vlad(denseA).T

# frame-index GT
gi = np.arange(NB)[:, None]; gj = np.arange(NA)[None]
G = np.abs(gi - gj) <= a.tol
nq = NB

def evaluate(S):
    order = np.argsort(-S, 1)
    rec = {K: sum(G[i, order[i, :K]].any() for i in range(NB)) / nq for K in (1, 5, 10)}
    top1 = order[:, 0]; score = S[np.arange(NB), top1]
    correct = G[np.arange(NB), top1]
    idx = np.argsort(-score); co = correct[idx]
    tp = np.cumsum(co); fp = np.cumsum(~co)
    prec = tp / np.maximum(tp + fp, 1); recl = tp / nq
    return rec, prec, recl, (2 * prec * recl / np.maximum(prec + recl, 1e-9)).max()

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
print(f"{'method':16} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'maxF1':>7}")
for name, S in sims.items():
    rec, prec, recl, f1 = evaluate(S)
    print(f"{name:16} {rec[1]:6.3f} {rec[5]:6.3f} {rec[10]:6.3f} {f1:7.3f}")
    ax[0].plot(recl, prec, label=f"{name} (F1={f1:.2f})", lw=2)
    ax[1].bar(name, rec[1])
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
ax[0].set_title("Gardens Point day->night: DBoW2 vs DINO/AnyLoc"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_ylabel("Recall@1"); ax[1].tick_params(axis="x", rotation=30); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(a.out, dpi=110); print("saved", a.out)
