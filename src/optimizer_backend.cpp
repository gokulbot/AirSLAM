#include "optimizer_backend.h"
#include "gtsam_backend.h"
#include "g2o_optimization/g2o_optimization.h"   // the existing free functions we forward to
#include <cstdlib>
#include <iostream>

void G2oBackend::LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines,
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list,
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
    VectorOfIMUConstraints& imu_constraints, const Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg) {
  ::LocalmapOptimization(poses, points, lines, velocities, biases, camera_list,
      mono_point_constraints, stereo_point_constraints, mono_line_constraints, stereo_line_constraints,
      imu_constraints, Rwg, cfg);
}

int G2oBackend::FrameOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines,
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list,
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg) {
  return ::FrameOptimization(poses, points, lines, velocities, biases, camera_list,
      mono_point_constraints, stereo_point_constraints, mono_line_constraints, stereo_line_constraints,
      imu_constraints, Rwg, cfg);
}

bool G2oBackend::IMUInitialization(MapOfPoses& poses, MapOfVelocity& velocities, Bias& bias,
    std::vector<CameraPtr>& camera_list, VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg) {
  return ::IMUInitialization(poses, velocities, bias, camera_list, imu_constraints, Rwg);
}

void G2oBackend::PoseGraphOptimization(MapOfPoses& poses, std::vector<CameraPtr>& camera_list,
    VectorOfRelativePoseConstraints& relative_pose_constraints) {
  ::PoseGraphOptimization(poses, camera_list, relative_pose_constraints);
}

void G2oBackend::GlobalBA(std::shared_ptr<Map> map, const OptimizationConfig& cfg, bool point_outlier_rejection,
    bool line_outlier_rejection, int first_iterations, int second_iterations) {
  ::GlobalBA(map, cfg, point_outlier_rejection, line_outlier_rejection, first_iterations, second_iterations);
}

OptimizerBackendPtr MakeOptimizerBackend(const std::string& fallback) {
  const char* env = std::getenv("AIRSLAM_BACKEND");
  std::string name = env ? env : fallback;
  OptimizerBackendPtr backend =
      (name == "gtsam") ? OptimizerBackendPtr(std::make_shared<GtsamBackend>())
                        : OptimizerBackendPtr(std::make_shared<G2oBackend>());
  std::cout << "[OptimizerBackend] using '" << backend->Name() << "'" << std::endl;
  return backend;
}
