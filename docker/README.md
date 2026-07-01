# AirSLAM dev container (Blackwell / RTX 50-series)

Stock AirSLAM ships a CUDA 12.1 / TensorRT 8.6 image (`xukuanhit/air_slam:v4`), which
**cannot** drive a Blackwell GPU (sm_120) — Blackwell needs CUDA ≥ 12.8 / TensorRT 10.
This image rebuilds the environment on **CUDA 12.8 + TensorRT 10 + Ubuntu 20.04 (Focal)**.

Focal is deliberate: it provides **ROS Noetic** and **OpenCV 4.2** (AirSLAM's pinned
version) straight from apt, while we layer CUDA 12.8 + TensorRT 10 on top for Blackwell.

## Build the image
```bash
./docker/build.sh          # tags airslam:blackwell
```

## Run + build AirSLAM
```bash
DATA_DIR=~/datasets ./docker/run.sh
# then inside the container:
cd /catkin_ws && catkin_make
source devel/setup.bash
```
The repo is mounted at `/catkin_ws/src/air_slam`, so edits on the host (e.g. the
TensorRT 8→10 port) are picked up by a re-`catkin_make` without rebuilding the image.

## Known follow-up: TensorRT 8 → 10 code port
Moving to TRT 10 removes APIs AirSLAM uses (`enqueueV2`, binding indices, etc.).
Expect to update `src/super_point.cpp`, `src/plnet.cpp`, `src/super_glue.cpp`, and
`3rdparty/tensorrtbuffer` to the TRT 10 API (named tensors, `enqueueV3`,
`buildSerializedNetwork`). Tracked on branch `feature/docker-blackwell`.

## Notes
- If TRT didn't resolve to ≥ 10.8, pin versions in the Dockerfile (see the comment there).
- X11 GUI (rviz): `xhost +local:docker` is run automatically by `run.sh`.
