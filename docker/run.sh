#!/usr/bin/env bash
# Run the AirSLAM dev container: GPU + X11, with the repo mounted into the catkin ws.
#
#   Inside the container, build with:
#       cd /catkin_ws && catkin_make
#       source devel/setup.bash
#
# Env overrides: IMAGE, DATA_DIR, CONTAINER
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # the AirSLAM repo
IMAGE="${IMAGE:-airslam:blackwell}"
DATA_DIR="${DATA_DIR:-$HOME/datasets}"            # host folder holding EuRoC etc.
CONTAINER="${CONTAINER:-airslam}"

mkdir -p "$DATA_DIR"
xhost +local:docker >/dev/null 2>&1 || true

docker run -it --rm \
  --gpus all \
  --env DISPLAY="${DISPLAY:-}" \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --volume "$REPO_DIR":/catkin_ws/src/air_slam \
  --volume "$DATA_DIR":/datasets \
  --net host \
  --privileged \
  --name "$CONTAINER" \
  "$IMAGE" "${@:-/bin/bash}"
