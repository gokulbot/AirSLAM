#ifndef GTSAM_BACKEND_H_
#define GTSAM_BACKEND_H_

#include "optimizer_backend.h"

// GTSAM optimizer backend, migrated from g2o incrementally. Methods that have been re-expressed
// in GTSAM run native factor graphs; the rest forward to the existing g2o free functions until
// ported, so GtsamBackend is always a drop-in replacement for G2oBackend.
//
// Migrated so far:  PoseGraphOptimization  (BetweenFactor<Pose3>)
// Still g2o:        FrameOptimization, LocalmapOptimization, GlobalBA, IMUInitialization
class GtsamBackend : public OptimizerBackend {
 public:
  std::string Name() const override { return "gtsam"; }

  // --- native GTSAM ---
  void PoseGraphOptimization(MapOfPoses& poses, std::vector<CameraPtr>& camera_list,
      VectorOfRelativePoseConstraints& relative_pose_constraints) override;

  // --- not yet migrated: forward to g2o (identical to G2oBackend) ---
  void LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines,
      MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list,
      VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
      VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
      VectorOfIMUConstraints& imu_constraints, const Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg) override;

  int FrameOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines,
      MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list,
      VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
      VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
      VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg) override;

  bool IMUInitialization(MapOfPoses& poses, MapOfVelocity& velocities, Bias& bias,
      std::vector<CameraPtr>& camera_list, VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg) override;

  void GlobalBA(std::shared_ptr<Map> map, const OptimizationConfig& cfg, bool point_outlier_rejection,
      bool line_outlier_rejection, int first_iterations, int second_iterations) override;
};

#endif  // GTSAM_BACKEND_H_
