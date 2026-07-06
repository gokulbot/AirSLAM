#!/usr/bin/env python3
# Convert Glasgow extreme-lighting SLAM sequences (KITTI layout) to AirSLAM's ASL/mav0
# layout via symlinks (no copy), and export ground-truth poses to TUM for evo ATE.
#   sequences/<route>/image_2|image_3/NNNNNN.png + times.txt  ->  mav0/cam{0,1}/data/<ns>.png
#   poses/<route>.txt (3x4 [R|t] per line)                    ->  <route>.tum
import os, glob, numpy as np

BASE = "/catkin_ws/src/air_slam/datasets/glasgow"

def mat2quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25*S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw, qx, qy, qz = (R[2,1]-R[1,2])/S, 0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw, qx, qy, qz = (R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw, qx, qy, qz = (R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S
    return qx, qy, qz, qw

def to_mav0(route, out_mav0, stereo=True):
    seqdir = f"{BASE}/sequences/{route}"
    times = [float(x) for x in open(f"{seqdir}/times.txt").read().split()]
    imgs = sorted(glob.glob(f"{seqdir}/image_2/*.png"))
    n = min(len(imgs), len(times))
    os.makedirs(f"{out_mav0}/cam0/data", exist_ok=True)
    if stereo: os.makedirs(f"{out_mav0}/cam1/data", exist_ok=True)
    for i in range(n):
        ns = round(times[i] * 1e9); name = f"{ns:019d}.png"
        for cam, sub in ([("cam0", "image_2")] + ([("cam1", "image_3")] if stereo else [])):
            src = os.path.abspath(imgs[i].replace("/image_2/", f"/{sub}/"))
            dst = f"{out_mav0}/{cam}/data/{name}"
            if os.path.lexists(dst): os.remove(dst)
            os.symlink(src, dst)
    print(f"{route}: {n} frames -> {out_mav0} (stereo={stereo})")

def to_tum(route, out_tum):
    P = np.loadtxt(f"{BASE}/poses/{route}.txt")
    times = [float(x) for x in open(f"{BASE}/sequences/{route}/times.txt").read().split()]
    n = min(len(P), len(times))
    lines = []
    for i in range(n):
        M = P[i].reshape(3, 4); R = M[:, :3]; t = M[:, 3]
        qx, qy, qz, qw = mat2quat(R)
        lines.append(f"{times[i]:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")
    open(out_tum, "w").write("\n".join(lines) + "\n")
    print(f"{route}: {n} GT poses -> {out_tum}")

if __name__ == "__main__":
    to_mav0("fig8_1_rl", f"{BASE}/mav0_fig8_rl", stereo=True)     # map: regular light (stereo)
    to_tum("fig8_1_rl", f"{BASE}/gt_fig8_rl.tum")
    to_mav0("fig8_1_offon", f"{BASE}/mav0_fig8_offon", stereo=False)  # query: lights off->on (mono)
    to_tum("fig8_1_offon", f"{BASE}/gt_fig8_offon.tum")
