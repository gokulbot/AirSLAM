#!/usr/bin/env python3
# Post-hoc Rerun logger for an AirSLAM run -> .rrd (open with: rerun <file>.rrd)
#   viode_rerun.py <run_dir> <mav0_dir> <gt.txt> <out.rrd>
# Shows: est(v1) vs GT trajectory (Umeyama-aligned) in 3D + keyframe images w/ YOLO vehicle boxes on a timeline.
import sys, os, glob
import numpy as np
import cv2
import rerun as rr

run_dir, data_dir, gt_path, out_rrd = sys.argv[1:5]

def load_tum(p):
    d = np.loadtxt(p, comments='#')
    return d[:, 0], d[:, 1:4]

t_est, xyz_est = load_tum(os.path.join(run_dir, "trajectory_v1.txt"))
t_gt, xyz_gt = load_tum(gt_path)

# associate est<->gt by nearest time, then Umeyama (sim3) align est -> gt
def umeyama(A, B):
    ma, mb = A.mean(0), B.mean(0)
    Ac, Bc = A - ma, B - mb
    H = Ac.T @ Bc / len(A)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1, 1, d])
    R = Vt.T @ Dm @ U.T
    s = np.trace(np.diag(S) @ Dm) / ((Ac ** 2).sum() / len(A))
    t = mb - s * R @ ma
    return s, R, t

pairs = [(i, int(np.argmin(np.abs(t_gt - t)))) for i, t in enumerate(t_est)]
pairs = [(i, j) for i, j in pairs if abs(t_gt[j] - t_est[i]) < 0.05]
A = xyz_est[[i for i, _ in pairs]]
B = xyz_gt[[j for _, j in pairs]]
s, R, tt = umeyama(A, B)
xyz_est_al = (s * (R @ xyz_est.T).T) + tt

rr.init("airslam_viode")
rr.save(out_rrd)
rr.log("world/gt", rr.LineStrips3D([xyz_gt], colors=[0, 230, 0], radii=0.03), static=True)
rr.log("world/est", rr.LineStrips3D([xyz_est_al], colors=[0, 170, 255], radii=0.03), static=True)

# YOLO vehicle boxes per keyframe
try:
    from ultralytics import YOLO
    model = YOLO("/tmp/yolov8m.pt")
    have_yolo = True
except Exception as e:
    print("YOLO unavailable:", e); have_yolo = False

imgs = sorted(glob.glob(os.path.join(data_dir, "cam0/data", "*.png")))
img_ns = np.array([int(os.path.basename(f).split('.')[0]) for f in imgs])
for i, t in enumerate(t_est):
    rr.set_time_seconds("t", t - t_est[0])
    rr.log("world/est/cam", rr.Points3D([xyz_est_al[i]], colors=[255, 230, 0], radii=0.08))
    j = int(np.argmin(np.abs(img_ns - t * 1e9)))
    img = cv2.cvtColor(cv2.imread(imgs[j]), cv2.COLOR_BGR2RGB)
    rr.log("cam0", rr.Image(img))
    if have_yolo:
        res = model(imgs[j], classes=[2, 3, 5, 7], conf=0.35, verbose=False)[0]
        if len(res.boxes):
            xy = res.boxes.xyxy.cpu().numpy()
            rr.log("cam0/vehicles", rr.Boxes2D(
                array=np.column_stack([xy[:, 0], xy[:, 1], xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1]]),
                array_format=rr.Box2DFormat.XYWH, colors=[255, 40, 40]))
print("wrote", out_rrd, "| est-vs-gt aligned, ", len(t_est), "keyframes")
