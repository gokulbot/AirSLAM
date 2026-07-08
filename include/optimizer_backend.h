#ifndef OPTIMIZER_BACKEND_H_
#define OPTIMIZER_BACKEND_H_

#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>

#include "g2o_optimization/types.h"   // MapOfPoses/Points3d/Line3d/Velocity/Bias, VectorOf*Constraints, Bias
#include "read_configs.h"             // OptimizationConfig
#include "camera.h"                   // CameraPtr

class Map;   // fwd-decl (GlobalBA takes shared_ptr<Map>); avoids a cycle with map.h -> optimizer_backend.h

// Swappable optimizer backend. The frontend fills the backend-agnostic constraint structs
// (types.h); a backend turns those into a factor graph and solves. G2oBackend wraps AirSLAM's
// existing g2o optimizers unchanged; GtsamBackend (Phase 3) re-expresses the same constraints as
// GTSAM factors. Chosen once and injected into Map / MapRefiner / MapUser.
class OptimizerBackend {
 public:
  virtual ~OptimizerBackend() = default;
  virtual std::string Name() const = 0;

  virtual void LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines,
      MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list,
      VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
      VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
      VectorOfIMUConstraints& imu_constraints, const Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg) = 0;

  virtual int FrameOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines,
      MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list,
      VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
      VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
      VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg) = 0;

  virtual bool IMUInitialization(MapOfPoses& poses, MapOfVelocity& velocities, Bias& bias,
      std::vector<CameraPtr>& camera_list, VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg) = 0;

  virtual void PoseGraphOptimization(MapOfPoses& poses, std::vector<CameraPtr>& camera_list,
      VectorOfRelativePoseConstraints& relative_pose_constraints) = 0;

  virtual void GlobalBA(std::shared_ptr<Map> map, const OptimizationConfig& cfg, bool point_outlier_rejection,
      bool line_outlier_rejection, int first_iterations, int second_iterations) = 0;
};
typedef std::shared_ptr<OptimizerBackend> OptimizerBackendPtr;

// g2o backend: forwards to AirSLAM's existing g2o_optimization.cc free functions (zero behaviour change).
class G2oBackend : public OptimizerBackend {
 public:
  std::string Name() const override { return "g2o"; }

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

  void PoseGraphOptimization(MapOfPoses& poses, std::vector<CameraPtr>& camera_list,
      VectorOfRelativePoseConstraints& relative_pose_constraints) override;

  void GlobalBA(std::shared_ptr<Map> map, const OptimizationConfig& cfg, bool point_outlier_rejection,
      bool line_outlier_rejection, int first_iterations, int second_iterations) override;
};

// Factory: picks the backend from the AIRSLAM_BACKEND env var ("g2o" default | "gtsam").
// Lets the regression test flip backends without a rebuild; a config field can replace it later.
// `fallback` is the per-role default backend (Map/online -> "g2o", MapRefiner/offline -> "gtsam").
// The AIRSLAM_BACKEND env var, when set, overrides it (used to force one backend for parity tests).
OptimizerBackendPtr MakeOptimizerBackend(const std::string& fallback = "g2o");

#endif  // OPTIMIZER_BACKEND_H_
