#!/bin/bash
# Backend-migration regression test (g2o -> GTSAM). Run after every OptimizerBackend change.
#
# Builds a fixed EuRoC map once (cached), then re-refines it — exercising GlobalBA +
# PoseGraphOptimization (the batch optimizers migrated first) — and checks the refined-map ATE
# against ground truth. Also reports the VO-baseline ATE (Frame/Localmap/IMU optimizers). A
# regression (broken optimizer) makes ATE explode past the threshold and the test exits non-zero.
#
#   usage: test/backend_regression.sh [ate_threshold_m]   (default 0.15; g2o baseline ~0.066)
# NB: no `set -u` — ROS's setup.bash references unset vars and would abort the script.
SEQ=MH_05_difficult
THRESH=${1:-0.15}
REPO=/catkin_ws/src/air_slam
DATA=$REPO/datasets/euroc/$SEQ/mav0
GT=$DATA/state_groundtruth_estimate0/data.csv
CACHE=/tmp/backend_regress_map          # cached mapv0 (built once)
WORK=/tmp/backend_regress_run           # per-run refinement output

source /opt/ros/noetic/setup.bash >/dev/null 2>&1
source /catkin_ws/devel/setup.bash >/dev/null 2>&1
cd $REPO
cleanup(){ pkill -9 -f rosmaster 2>/dev/null; pkill -9 -f roscore 2>/dev/null; pkill -9 -x visual_odometry 2>/dev/null; pkill -9 -x map_refinement 2>/dev/null; pkill -9 -f "euroc.launch" 2>/dev/null; sleep 4; }

# --- build the map once (self-contained) ---
if [ ! -f $CACHE/AirSLAM_mapv0.bin ]; then
  echo "[regress] building map cache via VO (one-time, ~2 min) ..."
  cleanup; mkdir -p $CACHE
  roslaunch air_slam vo_euroc.launch config_path:=$REPO/configs/visual_odometry/vo_euroc.yaml \
    camera_config_path:=$REPO/configs/camera/euroc.yaml dataroot:=$DATA saving_dir:=$CACHE \
    visualization:=false > $CACHE/vo.log 2>&1 &
  until [ -f $CACHE/AirSLAM_mapv0.bin ] || grep -qiE "terminate|Segmentation" $CACHE/vo.log; do sleep 3; done
  sleep 2; cleanup
  [ -f $CACHE/AirSLAM_mapv0.bin ] || { echo "[regress] FAIL: VO did not produce a map"; exit 2; }
fi

# --- refine (GlobalBA + PoseGraph) ---
cleanup; rm -rf $WORK; mkdir -p $WORK; cp $CACHE/AirSLAM_mapv0.bin $WORK/AirSLAM_mapv0.bin
roslaunch air_slam mr_euroc.launch config_path:=$REPO/configs/map_refinement/mr_euroc.yaml \
  map_root:=$WORK voc_path:=$REPO/voc/point_voc_L4.bin visualization:=false > $WORK/mr.log 2>&1 &
until [ -f $WORK/trajectory_v1.txt ] || grep -qiE "terminate|Segmentation|core dumped" $WORK/mr.log; do sleep 3; done
sleep 2; cleanup
[ -f $WORK/trajectory_v1.txt ] || { echo "[regress] FAIL: refinement produced no trajectory"; exit 3; }

# --- ATE (SE3-aligned) for VO baseline + refined map ---
python3 - "$GT" "$CACHE/trajectory_v0.txt" "$WORK/trajectory_v1.txt" "$THRESH" "$WORK/mr.log" <<'PY'
import subprocess, re, sys
gt, vo, refined, thresh, mrlog = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
L=[]
for l in open(gt):
    if l.startswith("#"): continue
    p=l.strip().split(",")
    L.append(f"{int(p[0])/1e9:.9f} {p[1]} {p[2]} {p[3]} {p[5]} {p[6]} {p[7]} {p[4]}")
open("/tmp/_regress_gt.tum","w").write("\n".join(L)+"\n")
def ate(traj):
    r=subprocess.run(["evo_ape","tum","/tmp/_regress_gt.tum",traj,"-a","--t_max_diff","0.05"],capture_output=True,text=True)
    m=re.search(r"\brmse\s+([\d.]+)",r.stdout); return float(m.group(1)) if m else 999.0
loops=""; backend="g2o"
import os
if os.path.exists(mrlog):
    for l in open(mrlog):
        if "loop pairs are found" in l: loops=l.strip()
        if "OptimizerBackend" in l and "'" in l: backend=l.split("'")[1]
vo_ate=ate(vo) if os.path.exists(vo) else -1
ref_ate=ate(refined)
print(f"[regress] backend = {backend}")
print(f"[regress] VO-baseline ATE   = {vo_ate:.4f} m  (Frame/Localmap/IMU optimizers)")
print(f"[regress] refined-map ATE   = {ref_ate:.4f} m  (GlobalBA + PoseGraph)  {loops}")
if ref_ate < thresh:
    print(f"[regress] PASS  (refined ATE {ref_ate:.4f} < {thresh})"); sys.exit(0)
else:
    print(f"[regress] FAIL  (refined ATE {ref_ate:.4f} >= {thresh})"); sys.exit(1)
PY
