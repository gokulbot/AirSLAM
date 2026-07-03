#!/usr/bin/env python3
"""Compute place-recognition descriptors for a frame manifest.

  - cls   : DINOv2 CLS token
  - mean  : DINOv2 mean-pooled patch tokens (final layer)
  - vlad  : DINOv2 dense features + VLAD  == AnyLoc when --facet value on ViT-G
  - tiny  : 32x32 grayscale cosine baseline (trivial floor)

Full AnyLoc recipe:  --model dinov2_vitg14 --facet value --layer 31 --k 32
(value facet of layer 31 grabbed via a forward hook on that block's qkv.)

usage: compute_descriptors.py <manifest.npz> <out.npz>
       [--model dinov2_vits14] [--res 224] [--k 32]
       [--facet token|value] [--layer 31] [--batch 8] [--device auto]
"""
import argparse, numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("manifest"); ap.add_argument("out")
ap.add_argument("--model", default="dinov2_vits14")
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--facet", choices=["token", "value"], default="token")
ap.add_argument("--layer", type=int, default=31)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--device", default="auto")
args = ap.parse_args()

dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
half = dev == "cuda"
m = np.load(args.manifest, allow_pickle=True)
paths = [str(p) for p in m["paths"]]
N = len(paths); R = args.res; P = (R // 14) ** 2

print(f"[{dev}] loading {args.model} (facet={args.facet}, layer={args.layer}) ...")
model = torch.hub.load("facebookresearch/dinov2", args.model, verbose=False).to(dev).eval()
if half:
    model = model.half()
E = model.num_features   # embed dim

# hook the value facet of the target block: qkv output is (B, Ntok, 3E), value = last E
captured = {}
if args.facet == "value":
    def hook(_mod, _inp, out):
        captured["v"] = out[..., 2 * E:3 * E].float()   # (B, Ntok, E)
    model.blocks[args.layer].attn.qkv.register_forward_hook(hook)

mean_n = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std_n = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

def load(p):
    im = Image.open(p).convert("RGB").resize((R, R), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0

cls_l, mean_l, dense_l, tiny_l = [], [], [], []
with torch.no_grad():
    for s in range(0, N, args.batch):
        batch = paths[s:s + args.batch]
        x = torch.stack([load(p) for p in batch]).to(dev)
        x = (x - mean_n) / std_n
        if half:
            x = x.half()
        out = model.forward_features(x)
        cls_l.append(out["x_norm_clstoken"].float().cpu().numpy())
        patch = out["x_norm_patchtokens"].float()          # (B, P, E) final layer
        mean_l.append(patch.mean(1).cpu().numpy())
        dense = captured["v"][:, -P:] if args.facet == "value" else patch
        dense_l.append(dense.cpu().numpy())
        for p in batch:
            tiny_l.append(np.asarray(Image.open(p).convert("L").resize((32, 32))).astype(np.float32).ravel())
        print(f"  {min(s+args.batch,N)}/{N}")

cls = np.concatenate(cls_l); meanp = np.concatenate(mean_l)
dense = np.concatenate(dense_l); tiny = np.stack(tiny_l)
D = dense.shape[-1]

def l2n(a, ax=-1, eps=1e-8):
    return a / (np.linalg.norm(a, axis=ax, keepdims=True) + eps)

from sklearn.cluster import KMeans
flat = dense.reshape(-1, D)
rng = np.random.default_rng(0)
sub = flat[rng.choice(len(flat), size=min(50000, len(flat)), replace=False)]
print(f"VLAD vocab k={args.k} on {len(sub)} vecs ...")
C = KMeans(n_clusters=args.k, n_init=4, random_state=0).fit(sub).cluster_centers_.astype(np.float32)

def vlad(F):
    a = np.argmin(((F[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
    V = np.zeros((args.k, D), np.float32)
    for c in range(args.k):
        if np.any(a == c):
            V[c] = (F[a == c] - C[c]).sum(0)
    V = l2n(V, ax=1)
    return l2n(V.ravel())

vlad_desc = np.stack([vlad(dense[i]) for i in range(N)])
np.savez(args.out, cls=l2n(cls), mean=l2n(meanp), vlad=vlad_desc, tiny=l2n(tiny), paths=np.array(paths))
print("saved", args.out, "| vlad dim", vlad_desc.shape[1], "| facet", args.facet)
