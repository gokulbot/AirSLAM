# AirSLAM dev image — Blackwell-capable (RTX 50-series, sm_120)
#
# Why this base: ROS Noetic only supports Ubuntu 20.04 (Focal); Blackwell (sm_120)
# needs CUDA >= 12.8 / TensorRT 10. Ubuntu 20.04 also ships OpenCV 4.2 via apt — the
# exact version AirSLAM pins. So Focal gives us ROS Noetic + OpenCV 4.2 for free, and
# we layer CUDA 12.8 + TensorRT 10 on top.
#
# This image is an ENVIRONMENT only — the AirSLAM source is mounted at runtime
# (see docker/run.sh) and built with catkin_make inside the container, so the
# TensorRT 8 -> 10 port can be iterated without rebuilding the image.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# Base tooling + AirSLAM apt dependencies
#   OpenCV 4.2 (libopencv-dev on Focal), Eigen3, Boost, glog, gflags, yaml-cpp,
#   SuiteSparse (for g2o's cholmod/csparse solvers)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git wget curl ca-certificates gnupg2 lsb-release \
      libeigen3-dev libboost-all-dev libyaml-cpp-dev \
      libgoogle-glog-dev libgflags-dev \
      libopencv-dev python3-opencv \
      libsuitesparse-dev libceres-dev \
 && rm -rf /var/lib/apt/lists/*
# libceres-dev: g2o's base_fixed_sized_edge.h includes <ceres/internal/fixed_array.h>
# on Linux (G2O_USE_VENDORED_CERES not set), so system Ceres is required.

# ---------------------------------------------------------------------------
# ROS Noetic (Focal)
# ---------------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" \
      > /etc/apt/sources.list.d/ros1.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      ros-noetic-desktop \
      ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-tf \
      python3-catkin-tools python3-rosdep python3-rosinstall python3-osrf-pycommon \
 && rm -rf /var/lib/apt/lists/* \
 && rosdep init && rosdep update

# ---------------------------------------------------------------------------
# TensorRT 10 (Blackwell-capable) from the CUDA ubuntu2004 apt repo
# (already configured in the nvidia/cuda base image).
#
# NOTE: We want TRT >= 10.8 for full Blackwell support. If the meta package below
# doesn't resolve to 10.8+, pin explicitly once we see `apt-cache policy libnvinfer-dev`:
#   apt-get install -y libnvinfer-dev=10.8.* libnvinfer-plugin-dev=10.8.* \
#                      libnvonnxparsers-dev=10.8.* libnvinfer-headers-dev=10.8.*
# Fallback if the CUDA repo lacks TRT: add the nv-tensorrt local/network repo.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev \
 && rm -rf /var/lib/apt/lists/* \
 && echo "TensorRT installed:" && dpkg -l | grep -i nvinfer || true

# ---------------------------------------------------------------------------
# g2o (AirSLAM pins tag 20230223_git) — built from source
# ---------------------------------------------------------------------------
RUN git clone https://github.com/RainerKuemmerle/g2o.git /opt/g2o \
 && cd /opt/g2o && git checkout 20230223_git \
 && mkdir build && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DG2O_BUILD_EXAMPLES=ON -DG2O_BUILD_APPS=ON -DBUILD_SHARED_LIBS=ON \
 && make -j"$(nproc)" && make install && ldconfig
# APPS=ON is REQUIRED: AirSLAM's cmake/FindG2O.cmake lists libg2o_cli/parser/simulator
# in G2O_LIBRARIES, so a core-only g2o build fails CMake configure with NOTFOUND.

# Ceres is intentionally NOT built: README lists it but CMakeLists.txt never calls
# find_package(Ceres) and nothing links it. Add it here only if a build error demands it.

# ---------------------------------------------------------------------------
# catkin workspace (AirSLAM source is mounted into src/ at runtime)
# ---------------------------------------------------------------------------
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc \
 && echo "[ -f /catkin_ws/devel/setup.bash ] && source /catkin_ws/devel/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
