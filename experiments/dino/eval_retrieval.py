#!/usr/bin/env python3
"""Evaluate place-recognition descriptors for loop closure on a manifest.

GT loop (i,j): time gap > --tgap s, position < --dist m, orientation < --rot deg.
Retrieval query i considers only earlier candidates j with t_j < t_i - tgap.

Reports Recall@K (VPR) and the precision-recall of top-1 loop proposals
(the metric that matters for loop closure: false loops are catastrophic).
Saves a PR-curve + Recall@K plot.

usage: eval_retrieval.py <manifest.npz> <desc.npz> <out.png> [--tgap 10 --dist 2 --rot 45]
"""
import sys, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("manifest"); ap.add_argument("desc"); ap.add_argument("out")
ap.add_argument("--tgap", type=float, default=10.0)
ap.add_argument("--dist", type=float, default=2.0)
ap.add_argument("--rot", type=float, default=45.0)
ap.add_argument("--methods", default="tiny,cls,mean,vlad")
a = ap.parse_args()

m = np.load(a.manifest, allow_pickle=True)
d = np.load(a.desc, allow_pickle=True)
t = m["t"]; pos = m["pos"]; quat = m["quat"]  # quat wxyz
N = len(t)

# pairwise position + orientation + time
pd = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
dot = np.abs(q @ q.T).clip(0, 1)
ang = np.degrees(2 * np.arccos(dot))           # geodesic angle between orientations
td = np.abs(t[:, None] - t[None])

is_loop = (pd < a.dist) & (ang < a.rot) & (td > a.tgap)   # symmetric GT loop matrix
# candidate mask: j strictly earlier than i by > tgap
earlier = (t[None] < (t[:, None] - a.tgap))

has_gt = (is_loop & earlier).any(1)      # queries that have a retrievable loop
nq = int(has_gt.sum())
print(f"N={N} frames; {nq} queries have a GT loop "
      f"(dist<{a.dist}m, rot<{a.rot}deg, tgap>{a.tgap}s)")

def eval_method(X):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = X @ X.T
    S = np.where(earlier, S, -np.inf)     # only earlier candidates
    order = np.argsort(-S, axis=1)        # ranked candidates per query
    # Recall@K
    rec = {}
    for K in (1, 5, 10):
        hit = 0
        for i in range(N):
            if not has_gt[i]:
                continue
            topk = order[i, :K]
            if is_loop[i, topk].any():
                hit += 1
        rec[K] = hit / max(nq, 1)
    # PR of top-1 proposals
    top1 = order[:, 0]
    score = S[np.arange(N), top1]
    valid = np.isfinite(score)            # query has at least one earlier candidate
    correct = np.array([is_loop[i, top1[i]] for i in range(N)]) & valid
    fires = valid
    # sweep threshold on score
    idx = np.argsort(-score[fires])
    sc = score[fires][idx]; co = correct[fires][idx]
    tp = np.cumsum(co); fp = np.cumsum(~co)
    prec = tp / np.maximum(tp + fp, 1)
    recl = tp / max(nq, 1)
    f1 = 2 * prec * recl / np.maximum(prec + recl, 1e-9)
    return rec, prec, recl, f1.max()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
methods = a.methods.split(",")
print(f"\n{'method':8} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'maxF1':>7}")
for name in methods:
    rec, prec, recl, maxf1 = eval_method(d[name])
    print(f"{name:8} {rec[1]:6.3f} {rec[5]:6.3f} {rec[10]:6.3f} {maxf1:7.3f}")
    ax[0].plot(recl, prec, label=f"{name} (F1={maxf1:.2f})")
    ax[1].bar(name, rec[1])
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
ax[0].set_title("Loop-proposal precision-recall (top-1)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_ylabel("Recall@1"); ax[1].set_title("Recall@1 by method"); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(a.out, dpi=110)
print("\nsaved plot:", a.out)
