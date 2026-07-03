#!/usr/bin/env python3
"""Build a manifest for the keyframes in a dump_bow output: match each keyframe
timestamp to its EuRoC cam0 image + nearest GT pose, and carry the DBoW2 matrix.

usage: kf_manifest.py <bow.txt> <mav0_dir> <out.npz>
"""
import sys, os, numpy as np

bow_txt, mav0, out = sys.argv[1:4]
lines = open(bow_txt).read().splitlines()
N = int(lines[0])
ids = []; kt = []
for i in range(1, 1 + N):
    a, b = lines[i].split()
    ids.append(int(a)); kt.append(float(b))          # timestamp in seconds
kt = np.array(kt)
S = np.array([[float(x) for x in lines[1 + N + r].split()] for r in range(N)])  # NxN DBoW2 scores

# cam0 images (filenames = ns timestamps)
img_dir = os.path.join(mav0, "cam0", "data")
inames = sorted(os.listdir(img_dir))
its = np.array([int(n.split(".")[0]) for n in inames]) / 1e9    # seconds
ii = np.abs(its[:, None] - kt[None]).argmin(0)
paths = [os.path.join(img_dir, inames[j]) for j in ii]
img_err = np.abs(its[ii] - kt).max()

# GT poses
gt = np.loadtxt(os.path.join(mav0, "state_groundtruth_estimate0", "data.csv"),
                delimiter=",", skiprows=1)
gts = gt[:, 0] / 1e9; gpos = gt[:, 1:4]; gquat = gt[:, 4:8]     # quat wxyz
gi = np.abs(gts[:, None] - kt[None]).argmin(0)
gt_err = np.abs(gts[gi] - kt)

# keep only keyframes whose GT is within 50 ms (drop those outside GT coverage),
# and subset the DBoW2 matrix to match so indexing stays consistent.
m = gt_err < 0.05
paths = [p for p, k in zip(paths, m) if k]
S = S[m][:, m]
np.savez(out, paths=np.array(paths), t=kt[m], pos=gpos[gi][m], quat=gquat[gi][m],
         bow=S, ids=np.array(ids)[m])
print(f"{N} keyframes -> {int(m.sum())} with valid GT (dropped {int((~m).sum())} outside coverage)")
print("pos extent:", np.round(gpos[gi][m].max(0) - gpos[gi][m].min(0), 2), "m")
