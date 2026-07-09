# AirSLAM — New-Machine Setup

Reproduce this dev environment on a fresh machine. The setup is: a **Docker image** (CUDA 12.8 / TensorRT 10 / ROS Noetic / OpenCV 4.2 / g2o) with the **repo mounted in at runtime** and built with `catkin_make`.

> ⚠️ **Important:** `docker/Dockerfile` builds only the *base* environment. **GTSAM and the Python tooling were installed ad-hoc** into the running container and are **not** in the Dockerfile — `docker/setup_extras.sh` closes that gap. Don't skip it.

---

## Prerequisites (host)
- NVIDIA GPU + driver new enough for **CUDA 12.8** (Blackwell/sm_120 needs this; any recent RTX works too)
- **Docker** + **NVIDIA Container Toolkit** (`nvidia-ctk`) so `--gpus all` works
- Disk: ~50 GB for the image + however much for datasets (EuRoC 2.8G, OpenLORIS 18G, VIODE 1.6G)

Quick GPU check on the host: `nvidia-smi` (note the driver/CUDA version).

---

## Two paths

### Path A — Rebuild clean (reproducible, recommended)
```bash
# 1. Clone the fork
git clone https://github.com/gokulbot/AirSLAM.git
cd AirSLAM
git checkout feature/semantic-nav        # or your working branch

# 2. Build the base image (CUDA/TRT/ROS/g2o) — ~20-40 min
bash docker/build.sh                      # -> image: airslam:blackwell

# 3. Run the container (GPU + repo mounted + datasets at /datasets)
DATA_DIR=$HOME/datasets bash docker/run.sh

# 4. INSIDE the container: install the extras NOT in the Dockerfile (GTSAM + python)
bash /catkin_ws/src/air_slam/docker/setup_extras.sh     # ~15-30 min (GTSAM build)

# 5. INSIDE the container: build AirSLAM
cd /catkin_ws && catkin_make && source devel/setup.bash
```

### Path B — Transfer the snapshot (fastest, exact)
If the **old machine is reachable**, ship the fully-configured image instead of rebuilding:
```bash
# on OLD machine:
docker save airslam_snap:latest | gzip > airslam_snap.tar.gz     # ~15 GB compressed
# copy the tarball over, then on NEW machine:
gunzip -c airslam_snap.tar.gz | docker load
# run it with the same flags docker/run.sh uses (edit IMAGE=airslam_snap:latest)
IMAGE=airslam_snap:latest DATA_DIR=$HOME/datasets bash docker/run.sh
```
This bakes in GTSAM + all Python extras, so you skip step 4. **Still need models + datasets (below).**

---

## Models (ONNX + vocab) — required, gitignored
The neural weights live in `output/` and `voc/` (not in git). **TensorRT `.engine` files are GPU-specific and auto-rebuild on first run — do NOT copy them; copy the portable `.onnx`/`.bin`.**

Fastest — rsync from the old machine:
```bash
rsync -av OLD:/path/to/AirSLAM/output/  ./output/     # exclude *.engine if you like
rsync -av OLD:/path/to/AirSLAM/voc/     ./voc/
```
Files that must be present (portable):
- `output/plnet_s0.onnx`, `plnet_s1.onnx`, `superpoint_v1_sim_int32.onnx`, `superpoint_lightglue.onnx`, `superglue_{indoor,outdoor}_sim_int32.onnx` — **from AirSLAM upstream** (or regenerate)
- `output/dinov2_vits14.onnx`, `anyloc_vitg14.onnx` (+ `.data`), `anyloc_vocab.bin`, `anyloc_ref.bin` — regenerable via `experiments/dino/`
- `voc/point_voc_L4.bin` — DBoW vocabulary
- YOLO: `yolov8m.onnx` (dynamic rejection) — regenerate via `experiments/openloris_yolo_eval/export_yolo_onnx.py`

First `catkin_make` + run rebuilds every engine from the ONNX (slow once, cached after).

---

## Datasets — gitignored, transfer or download
rsync from the old machine (fastest), or fetch:
| dataset | how |
|---|---|
| **EuRoC** | https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets (mav0 folders) |
| **OpenLORIS** | form → GDrive: https://lifelong-robotic-vision.github.io/dataset/scene.html (rosbags → extract to mav0) |
| **VIODE** | Zenodo https://zenodo.org/records/4493401 → `tools/viode/extract_viode_bag.py <bag> datasets/viode/<seq>` |

Layout AirSLAM expects: `datasets/<name>/<seq>/mav0/{cam0,cam1}/data/*.png + imu0/data.csv` and `datasets/<name>/<seq>/groundtruth.txt` (TUM).

---

## Verify (inside the container)
```bash
source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash
roslaunch air_slam vo_euroc.launch \
  config_path:=/catkin_ws/src/air_slam/configs/visual_odometry/vo_euroc.yaml \
  camera_config_path:=/catkin_ws/src/air_slam/configs/camera/euroc.yaml \
  dataroot:=/catkin_ws/src/air_slam/datasets/euroc/MH_05_difficult/mav0 \
  saving_dir:=/tmp/vo_check visualization:=false
# expected: "Initialization is done!", ~183 keyframes, MH_05 VO ATE ~0.115 m
```

---

## Runtime env-var reference (features toggled via env, no rebuild)
| var | effect |
|---|---|
| `AIRSLAM_BACKEND` | `g2o` (default) or `gtsam` (parked reference backend) |
| `AIRSLAM_YOLO_ONNX` | path to yolo `.onnx` → enables dynamic rejection (else OFF) |
| `AIRSLAM_YOLO_CLASSES` | dynamic classes, e.g. `0`=person (default), `2,3,5,7`=vehicles |
| `AIRSLAM_MOTION_PX` | motion-gate reprojection threshold, px (default 4) |
| `AIRSLAM_NO_DYNREJ` | refine with dynamic hooks OFF (for clean A/B) |
| `AIRSLAM_DYN_DEBUG` | per-keyframe dynamic-flag logging |
| `AIRSLAM_NO_COV` | skip the uncertainty-aware-map `computeMarginals` |
| `AIRSLAM_STAGE_DUMP` | dump per-stage refine trajectories |
| `AIRSLAM_ISAM_SMOOTHER` | live iSAM2 smoother (opt-in, reference) |
| `AIRSLAM_FAKE_DYNAMIC` | inject synthetic dynamic points (hook test) |

---

## Gotchas (learned the hard way)
- **`~/.gitconfig` on the host is a *directory*** and breaks git. Prefix git with `GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1`.
- **TensorRT engines are per-GPU/per-TRT** — never copy `.engine` between machines; let them rebuild from ONNX.
- **GTSAM must use the *system* Eigen** (`GTSAM_USE_SYSTEM_EIGEN=ON`) to stay ABI-compatible with g2o/AirSLAM. `setup_extras.sh` does this.
- **`ultralytics` install** may hit a distutils/`psutil` conflict → `pip install --ignore-installed psutil` first (handled by `setup_extras.sh`).
- **`rerun-sdk`** must be `0.16.1` on Python 3.8 (newer rerun needs 3.9+).
- The dev container is Python 3.8; Blackwell + py3.8 caps torch at CPU 2.4.1 — fine (YOLO runs via ONNX→TRT, not torch).
