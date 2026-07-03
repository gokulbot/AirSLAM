#!/usr/bin/env python3
"""Head-to-head: AirSLAM's DBoW2 vs DINO/AnyLoc place recognition, same keyframes,
same GT loops. DBoW2 similarity = the dumped BoW score matrix (manifest['bow']).

usage: compare_dbow.py <kf_manifest.npz> <desc.npz> <out.png> [--tgap 10 --dist 2 --rot 45]
"""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("manifest"); ap.add_argument("desc"); ap.add_argument("out")
ap.add_argument("--tgap", type=float, default=10.0)
ap.add_argument("--dist", type=float, default=2.0)
ap.add_argument("--rot", type=float, default=45.0)
a = ap.parse_args()

m = np.load(a.manifest, allow_pickle=True); d = np.load(a.desc, allow_pickle=True)
t, pos, quat, bow = m["t"], m["pos"], m["quat"], m["bow"]
N = len(t)
pd = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
ang = np.degrees(2 * np.arccos(np.abs(q @ q.T).clip(0, 1)))
td = np.abs(t[:, None] - t[None])
is_loop = (pd < a.dist) & (ang < a.rot) & (td > a.tgap)
earlier = (t[None] < (t[:, None] - a.tgap))
has_gt = (is_loop & earlier).any(1); nq = int(has_gt.sum())
print(f"N={N} keyframes; {nq} have a GT loop (dist<{a.dist}m rot<{a.rot} tgap>{a.tgap}s)")

def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

sims = {"dbow(AirSLAM)": bow.astype(float),
        "tiny": l2n(d["tiny"]) @ l2n(d["tiny"]).T,
        "cls": l2n(d["cls"]) @ l2n(d["cls"]).T,
        "mean": l2n(d["mean"]) @ l2n(d["mean"]).T,
        "vlad(AnyLoc)": l2n(d["vlad"]) @ l2n(d["vlad"]).T}

def evaluate(S):
    S = np.where(earlier, S, -np.inf)
    order = np.argsort(-S, 1)
    rec = {}
    for K in (1, 5, 10):
        rec[K] = sum(is_loop[i, order[i, :K]].any() for i in range(N) if has_gt[i]) / max(nq, 1)
    top1 = order[:, 0]; score = S[np.arange(N), top1]; valid = np.isfinite(score)
    correct = np.array([is_loop[i, top1[i]] for i in range(N)]) & valid
    idx = np.argsort(-score[valid]); co = correct[valid][idx]
    tp = np.cumsum(co); fp = np.cumsum(~co)
    prec = tp / np.maximum(tp + fp, 1); recl = tp / max(nq, 1)
    f1 = (2 * prec * recl / np.maximum(prec + recl, 1e-9)).max()
    return rec, prec, recl, f1

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
print(f"\n{'method':16} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'maxF1':>7}")
for name, S in sims.items():
    rec, prec, recl, f1 = evaluate(S)
    print(f"{name:16} {rec[1]:6.3f} {rec[5]:6.3f} {rec[10]:6.3f} {f1:7.3f}")
    ax[0].plot(recl, prec, label=f"{name} (F1={f1:.2f})", lw=2)
    ax[1].bar(name, rec[1])
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
ax[0].set_title("MH_04 loop proposals: DBoW2 vs DINO/AnyLoc"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_ylabel("Recall@1"); ax[1].tick_params(axis="x", rotation=30); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(a.out, dpi=110); print("\nsaved", a.out)
