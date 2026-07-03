#!/usr/bin/env python3
"""Build a frame manifest for an OpenLORIS sequence (fisheye cam0 + TUM groundtruth).
Filenames are timestamps in seconds; groundtruth.txt is: #Time px py pz qx qy qz qw.

usage: build_ol_manifest.py <seq_dir> <out.npz> [--stride N]
  <seq_dir> must contain mav0/cam0/data/*.png and groundtruth.txt
"""
import sys, os, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("seq"); ap.add_argument("out")
ap.add_argument("--stride", type=int, default=3)
a = ap.parse_args()

img_dir = os.path.join(a.seq, "mav0", "cam0", "data")
names = sorted(os.listdir(img_dir))[:: a.stride]
ts = np.array([float(os.path.splitext(n)[0]) for n in names])   # seconds

gt = np.loadtxt(os.path.join(a.seq, "groundtruth.txt"))          # t px py pz qx qy qz qw
gts = gt[:, 0]; gpos = gt[:, 1:4]; gquat = gt[:, 4:8]            # quat xyzw

idx = np.searchsorted(gts, ts)
idx = np.clip(idx, 1, len(gts) - 1)
left = np.abs(gts[idx - 1] - ts) < np.abs(gts[idx] - ts)
idx = idx - left.astype(int)
ok = np.abs(gts[idx] - ts) < 0.05                                # within 50 ms of a GT sample
names = [n for n, k in zip(names, ok) if k]
ts, idx = ts[ok], idx[ok]
paths = [os.path.join(img_dir, n) for n in names]

np.savez(a.out, paths=np.array(paths), t=ts.astype(np.float64),
         pos=gpos[idx].astype(np.float64), quat=gquat[idx].astype(np.float64))
print(f"{len(paths)} frames, stride {a.stride}, span {ts.max()-ts.min():.1f}s")
print("pos extent:", np.round(gpos[idx].max(0) - gpos[idx].min(0), 2), "m")
