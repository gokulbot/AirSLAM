#include "gtsam_backend.h"
#include "g2o_optimization/g2o_optimization.h"   // forward the un-migrated methods

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>

// PoseGraphOptimization re-expressed in GTSAM, mirroring g2o's PoseGraphOptimization exactly:
//   variables    = camera poses Twc  (kv.second = (Rwc, pwc), i.e. camera pose in world)
//   measurement  (Rc1c2, tc1c2) = T_{c1c2} = X1^{-1} X2   =>  BetweenFactor<Pose3>(id1, id2, Tc1c2)
//   information  = Identity  => unit noise;  fixed poses anchored with a near-rigid prior;  LM, 20 iters.
void GtsamBackend::PoseGraphOptimization(MapOfPoses& poses, std::vector<CameraPtr>& camera_list,
    VectorOfRelativePoseConstraints& relative_pose_constraints) {
  if (poses.empty() || camera_list.empty() || relative_pose_constraints.empty()) return;

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial;

  // 1. pose variables (Twc); g2o's setFixed(true) -> a near-rigid prior in GTSAM
  auto fixed_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-9);
  for (auto& kv : poses) {
    gtsam::Pose3 Twc(gtsam::Rot3(kv.second.R), gtsam::Point3(kv.second.p));
    initial.insert(static_cast<gtsam::Key>(kv.first), Twc);
    if (kv.second.fixed) graph.addPrior(static_cast<gtsam::Key>(kv.first), Twc, fixed_noise);
  }

  // 2. relative-pose constraints (g2o uses Identity information -> unit noise)
  auto rel_noise = gtsam::noiseModel::Unit::Create(6);
  for (RelativePoseConstraintPtr& rpc : relative_pose_constraints) {
    gtsam::Pose3 Tc1c2(gtsam::Rot3(rpc->Rc1c2), gtsam::Point3(rpc->tc1c2));
    graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        static_cast<gtsam::Key>(rpc->id_pose1), static_cast<gtsam::Key>(rpc->id_pose2), Tc1c2, rel_noise);
  }

  // 3. solve (Levenberg-Marquardt, matching g2o's 20 iterations)
  gtsam::LevenbergMarquardtParams params;
  params.setMaxIterations(20);
  gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graph, initial, params).optimize();

  // 4. recover optimized camera poses Twc
  for (MapOfPoses::iterator it = poses.begin(); it != poses.end(); ++it) {
    gtsam::Pose3 Twc = result.at<gtsam::Pose3>(static_cast<gtsam::Key>(it->first));
    it->second.p = Twc.translation();
    it->second.R = Twc.rotation().matrix();
  }
}

// ---------------------------------------------------------------------------
// Not yet migrated: forward to the existing g2o free functions (same as G2oBackend).
// ---------------------------------------------------------------------------
void GtsamBackend::LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines,
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list,
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
    VectorOfIMUConstraints& imu_constraints, const Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg) {
  ::LocalmapOptimization(poses, points, lines, velocities, biases, camera_list,
      mono_point_constraints, stereo_point_constraints, mono_line_constraints, stereo_line_constraints,
      imu_constraints, Rwg, cfg);
}

int GtsamBackend::FrameOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines,
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list,
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg) {
  return ::FrameOptimization(poses, points, lines, velocities, biases, camera_list,
      mono_point_constraints, stereo_point_constraints, mono_line_constraints, stereo_line_constraints,
      imu_constraints, Rwg, cfg);
}

bool GtsamBackend::IMUInitialization(MapOfPoses& poses, MapOfVelocity& velocities, Bias& bias,
    std::vector<CameraPtr>& camera_list, VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg) {
  return ::IMUInitialization(poses, velocities, bias, camera_list, imu_constraints, Rwg);
}

void GtsamBackend::GlobalBA(std::shared_ptr<Map> map, const OptimizationConfig& cfg, bool point_outlier_rejection,
    bool line_outlier_rejection, int first_iterations, int second_iterations) {
  ::GlobalBA(map, cfg, point_outlier_rejection, line_outlier_rejection, first_iterations, second_iterations);
}
