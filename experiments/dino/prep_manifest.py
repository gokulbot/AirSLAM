#!/usr/bin/env python3
"""Sample frames from a EuRoC sequence and attach nearest ground-truth poses.
Output: an .npz manifest (paths, timestamps[s], positions Nx3, quats Nx4 wxyz).

usage: prep_manifest.py <mav0_dir> <out.npz> [--stride N]
"""
import sys, os, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("mav0"); ap.add_argument("out")
ap.add_argument("--stride", type=int, default=5)
a = ap.parse_args()

img_dir = os.path.join(a.mav0, "cam0", "data")
names = sorted(os.listdir(img_dir))[:: a.stride]
ts = np.array([int(n.split(".")[0]) for n in names], dtype=np.int64)  # ns

# ground truth
gt = np.loadtxt(os.path.join(a.mav0, "state_groundtruth_estimate0", "data.csv"),
                delimiter=",", skiprows=1)
gts = gt[:, 0].astype(np.int64)          # ns
gpos = gt[:, 1:4]                          # x,y,z
gquat = gt[:, 4:8]                         # w,x,y,z

# nearest-GT assignment (GT ~200Hz, frames ~20Hz -> within a few ms)
idx = np.searchsorted(gts, ts)
idx = np.clip(idx, 1, len(gts) - 1)
left = np.abs(gts[idx - 1] - ts) < np.abs(gts[idx] - ts)
idx = idx - left.astype(int)

# keep only frames whose nearest GT is within 50 ms (drop uncovered ends)
ok = np.abs(gts[idx] - ts) < 50_000_000
names = [n for n, k in zip(names, ok) if k]
ts, idx = ts[ok], idx[ok]
paths = [os.path.join(img_dir, n) for n in names]

np.savez(a.out,
         paths=np.array(paths),
         t=ts.astype(np.float64) / 1e9,
         pos=gpos[idx].astype(np.float64),
         quat=gquat[idx].astype(np.float64))
print(f"{len(paths)} frames, stride {a.stride}, span {(ts.max()-ts.min())/1e9:.1f}s")
print("pos extent:", np.round(gpos[idx].max(0) - gpos[idx].min(0), 2), "m")
