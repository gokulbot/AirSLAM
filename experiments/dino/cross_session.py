#!/usr/bin/env python3
"""Cross-session place recognition: query session B against map session A.
Descriptors: DINOv2 cls / mean / VLAD (+ tiny baseline). VLAD vocab is built on
the MAP session A and applied to both (realistic: vocab from the map).

GT cross-match (b in B, a in A): position < --dist m AND orientation < --rot deg,
using each sequence's ground truth in the common OpenLORIS scene frame.

Full AnyLoc: --model dinov2_vitg14 --facet value --layer 31

usage: cross_session.py <mapA.npz> <queryB.npz> <out.png> [opts]
"""
import argparse, numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("mapA"); ap.add_argument("queryB"); ap.add_argument("out")
ap.add_argument("--model", default="dinov2_vitg14")
ap.add_argument("--facet", choices=["token", "value"], default="value")
ap.add_argument("--layer", type=int, default=31)
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--dist", type=float, default=1.0)
ap.add_argument("--rot", type=float, default=45.0)
ap.add_argument("--qgamma", type=float, default=1.0,
                help="nonlinear gamma applied to QUERY images (simulated lighting change)")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
half = dev == "cuda"
print(f"[{dev}] loading {args.model} facet={args.facet} layer={args.layer}")
model = torch.hub.load("facebookresearch/dinov2", args.model, verbose=False).to(dev).eval()
if half:
    model = model.half()
E = model.num_features
R = args.res; P = (R // 14) ** 2
cap = {}
if args.facet == "value":
    model.blocks[args.layer].attn.qkv.register_forward_hook(
        lambda m, i, o: cap.__setitem__("v", o[..., 2 * E:3 * E].float()))
mn = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
sd = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

def _apply_gamma(arr, g):
    if g == 1.0:
        return arr
    return (255.0 * (arr.astype(np.float32) / 255.0) ** g).astype(np.uint8)

def load(p, gamma=1.0):
    im = Image.open(p).convert("RGB").resize((R, R), Image.BILINEAR)
    a = _apply_gamma(np.asarray(im), gamma)
    return torch.from_numpy(a).float().permute(2, 0, 1) / 255.0

def extract(paths, gamma=1.0):
    cls_l, mean_l, dense_l, tiny_l = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(paths), args.batch):
            b = paths[s:s + args.batch]
            x = torch.stack([load(p, gamma) for p in b]).to(dev)
            x = (x - mn) / sd
            if half:
                x = x.half()
            out = model.forward_features(x)
            cls_l.append(out["x_norm_clstoken"].float().cpu().numpy())
            patch = out["x_norm_patchtokens"].float()
            mean_l.append(patch.mean(1).cpu().numpy())
            dense = cap["v"][:, -P:] if args.facet == "value" else patch
            dense_l.append(dense.cpu().numpy())
            for p in b:
                ti = np.asarray(Image.open(p).convert("L").resize((32, 32))).astype(np.float32)
                tiny_l.append(_apply_gamma(ti, gamma).astype(np.float32).ravel())
    return (np.concatenate(cls_l), np.concatenate(mean_l),
            np.concatenate(dense_l), np.stack(tiny_l))

A = np.load(args.mapA, allow_pickle=True); B = np.load(args.queryB, allow_pickle=True)
pA = [str(p) for p in A["paths"]]; pB = [str(p) for p in B["paths"]]
print(f"map A: {len(pA)} frames | query B: {len(pB)} frames")

import os
cache = args.out + ".feat.npz"
if os.path.exists(cache):
    print("loading cached features:", cache)
    z = np.load(cache)
    clsA, meanA, denseA, tinyA = z["clsA"], z["meanA"], z["denseA"], z["tinyA"]
    clsB, meanB, denseB, tinyB = z["clsB"], z["meanB"], z["denseB"], z["tinyB"]
else:
    clsA, meanA, denseA, tinyA = extract(pA)
    clsB, meanB, denseB, tinyB = extract(pB, gamma=args.qgamma)
    np.savez(cache, clsA=clsA, meanA=meanA, denseA=denseA, tinyA=tinyA,
             clsB=clsB, meanB=meanB, denseB=denseB, tinyB=tinyB)
D = denseA.shape[-1]

def l2n(a, ax=-1):
    return a / (np.linalg.norm(a, axis=ax, keepdims=True) + 1e-8)

from sklearn.cluster import KMeans
flat = denseA.reshape(-1, D)
rng = np.random.default_rng(0)
sub = flat[rng.choice(len(flat), size=min(50000, len(flat)), replace=False)]
C = KMeans(n_clusters=args.k, n_init=4, random_state=0).fit(sub).cluster_centers_.astype(np.float32)

def vlad(dense):
    out = np.zeros((len(dense), args.k * D), np.float32)
    for n, F in enumerate(dense):
        a = np.argmin(((F[:, None, :] - C[None]) ** 2).sum(-1), 1)
        V = np.zeros((args.k, D), np.float32)
        for c in range(args.k):
            if np.any(a == c):
                V[c] = (F[a == c] - C[c]).sum(0)
        out[n] = l2n(l2n(V, 1).ravel())
    return out

descA = {"tiny": l2n(tinyA), "cls": l2n(clsA), "mean": l2n(meanA), "vlad": vlad(denseA)}
descB = {"tiny": l2n(tinyB), "cls": l2n(clsB), "mean": l2n(meanB), "vlad": vlad(denseB)}

# cross GT
posA, quatA = A["pos"], l2n(A["quat"]); posB, quatB = B["pos"], l2n(B["quat"])
pd = np.linalg.norm(posB[:, None] - posA[None], axis=-1)
ang = np.degrees(2 * np.arccos(np.abs(quatB @ quatA.T).clip(0, 1)))
G = (pd < args.dist) & (ang < args.rot)          # (Nb, Na)
has_gt = G.any(1); nq = int(has_gt.sum())
print(f"cross GT: {nq}/{len(pB)} query frames revisit a mapped place "
      f"(dist<{args.dist}m, rot<{args.rot}deg); min cross-dist median={np.median(pd.min(1)):.2f}m")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
print(f"\n{'method':6} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'maxF1':>7}")
for name in ("tiny", "cls", "mean", "vlad"):
    S = descB[name] @ descA[name].T
    order = np.argsort(-S, 1)
    rec = {}
    for K in (1, 5, 10):
        hit = sum(G[i, order[i, :K]].any() for i in range(len(pB)) if has_gt[i])
        rec[K] = hit / max(nq, 1)
    top1 = order[:, 0]; score = S[np.arange(len(pB)), top1]
    correct = np.array([G[i, top1[i]] for i in range(len(pB))])
    idx = np.argsort(-score); co = correct[idx]
    tp = np.cumsum(co); fp = np.cumsum(~co)
    prec = tp / np.maximum(tp + fp, 1); recl = tp / max(nq, 1)
    f1 = (2 * prec * recl / np.maximum(prec + recl, 1e-9)).max()
    print(f"{name:6} {rec[1]:6.3f} {rec[5]:6.3f} {rec[10]:6.3f} {f1:7.3f}")
    ax[0].plot(recl, prec, label=f"{name} (F1={f1:.2f})")
    ax[1].bar(name, rec[1])
ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
ax[0].set_title("Cross-session loop PR (office1-2 vs office1-1)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_ylabel("Recall@1"); ax[1].set_title("Cross-session Recall@1"); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(args.out, dpi=110); print("\nsaved", args.out)
