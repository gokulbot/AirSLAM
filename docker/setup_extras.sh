#!/usr/bin/env bash
# Extras that are NOT baked into docker/Dockerfile — they were installed ad-hoc into the running
# dev container (the airslam_snap image). Run this INSIDE the container, once, after docker/run.sh:
#     bash /catkin_ws/src/air_slam/docker/setup_extras.sh
# Then: cd /catkin_ws && catkin_make && source devel/setup.bash
set -euo pipefail

echo "==> [1/2] GTSAM 4.2  (REQUIRED by CMakeLists find_package(GTSAM); NOT in the Dockerfile)"
if [ -f /usr/local/lib/libgtsam.so ]; then
  echo "    GTSAM already present — skipping."
else
  git clone https://github.com/borglab/gtsam.git /opt/gtsam
  cd /opt/gtsam && git checkout 4.2
  mkdir -p build && cd build
  # USE_SYSTEM_EIGEN=ON is critical: keeps GTSAM ABI-compatible with g2o/AirSLAM (same Eigen).
  # BUILD_UNSTABLE=ON: AirSLAM links gtsam_unstable. WITH_TBB=OFF for portability (kept consistent
  # because AirSLAM is recompiled against this same GTSAM).
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_BUILD_UNSTABLE=ON \
    -DGTSAM_WITH_TBB=OFF \
    -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF
  make -j"$(nproc)"
  make install
  ldconfig
  echo "    GTSAM 4.2 installed to /usr/local."
fi

echo "==> [2/2] Python tooling  (YOLO eval, Rerun viz, ATE metrics — all off the critical path)"
# distutils/psutil conflict blocks ultralytics unless psutil is reinstalled first:
pip install --ignore-installed psutil
# Blackwell + py3.8 caps torch at CPU 2.4.1 — fine, YOLO deploys via ONNX->TRT, not torch:
pip install torch==2.4.1+cpu torchvision==0.19.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install \
  ultralytics==8.4.90 \
  rerun-sdk==0.16.1 \
  evo==1.31.1 \
  onnx==1.17.0 onnxruntime==1.19.2 onnxslim

echo ""
echo "==> Extras done. Next:"
echo "      cd /catkin_ws && catkin_make && source devel/setup.bash"
