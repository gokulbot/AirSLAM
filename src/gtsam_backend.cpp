#include "gtsam_backend.h"
#include "g2o_optimization/g2o_optimization.h"   // forward the un-migrated methods

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include "gtsam_line_factor.h"
#include <cmath>

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

// Single-frame robust pose optimization re-expressed in GTSAM. Mirrors g2o's FrameOptimization:
// free body pose(s), FIXED landmarks (unary factors), Huber robust noise, and a 3-pass chi2 outlier
// loop (reset pose -> optimize inliers -> reclassify; drop Huber on the last pass). Point info = I;
// line info = 0.1*I (=> sigma sqrt(10)). The visual-inertial case is still forwarded to g2o.
int GtsamBackend::FrameOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines,
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list,
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg) {
  using namespace gtsam;
  using symbol_shorthand::X;

  if (!velocities.empty() || !biases.empty() || !imu_constraints.empty()) {   // VI not migrated yet
    return ::FrameOptimization(poses, points, lines, velocities, biases, camera_list,
        mono_point_constraints, stereo_point_constraints, mono_line_constraints, stereo_line_constraints,
        imu_constraints, Rwg, cfg);
  }
  const size_t n_c = mono_point_constraints.size() + stereo_point_constraints.size() +
                     mono_line_constraints.size() + stereo_line_constraints.size();
  if (n_c == 0) return 0;

  auto Tcb_of = [&](int cam) {
    Eigen::Matrix4d T = camera_list[cam]->BodyToCamera();
    return Pose3(Rot3(T.block<3, 3>(0, 0)), Point3(T.block<3, 1>(0, 3)));
  };
  std::map<int, Pose3> Twb_init, Tbc;   // per-pose body pose + body_P_sensor
  for (auto& kv : poses) {
    Pose3 Twc(Rot3(kv.second.R), Point3(kv.second.p));
    Pose3 Tcb = Tcb_of(kv.second.id_camera);
    Twb_init[kv.first] = Twc * Tcb;
    Tbc[kv.first] = Tcb.inverse();
  }
  auto Kv_of = [&](CameraPtr c) {
    return Eigen::Vector3d(-c->Cx() * c->Fy(), -c->Fx() * c->Cy(), c->Fx() * c->Fy());
  };
  const double hMp = std::sqrt(cfg.mono_point), hSp = std::sqrt(cfg.stereo_point);
  const double hMl = std::sqrt(cfg.mono_line), hSl = std::sqrt(cfg.stereo_line);
  const double lineSig = std::sqrt(10.0);                 // line info 0.1*I -> sigma sqrt(10)
  auto fixedNoise = noiseModel::Isotropic::Sigma(6, 1e-9);
  auto robustify = [](double k, const SharedNoiseModel& base) -> SharedNoiseModel {
    return noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(k), base);
  };

  std::vector<bool> mp_in(mono_point_constraints.size(), true), sp_in(stereo_point_constraints.size(), true);
  std::vector<bool> ml_in(mono_line_constraints.size(), true), sl_in(stereo_line_constraints.size(), true);

  Values result;
  int num_outlier = 0;
  for (int pass = 0; pass < 3; ++pass) {
    const bool robust = (pass < 2);                       // last pass drops the kernel
    NonlinearFactorGraph graph;
    Values init;
    for (auto& kv : poses) {
      init.insert(X(kv.first), Twb_init[kv.first]);
      if (kv.second.fixed) graph.addPrior(X(kv.first), Twb_init[kv.first], fixedNoise);
    }
    for (size_t i = 0; i < mono_point_constraints.size(); ++i) {
      if (!mp_in[i]) continue;
      auto& c = mono_point_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      SharedNoiseModel base = noiseModel::Unit::Create(2);
      graph.emplace_shared<GtsamMonoPointFactor>(X(c->id_pose), Point3(points[c->id_point].p), Point2(c->keypoint),
          cam->Fx(), cam->Fy(), cam->Cx(), cam->Cy(), Tbc[c->id_pose], robust ? robustify(hMp, base) : base);
    }
    for (size_t i = 0; i < stereo_point_constraints.size(); ++i) {
      if (!sp_in[i]) continue;
      auto& c = stereo_point_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      SharedNoiseModel base = noiseModel::Unit::Create(3);
      graph.emplace_shared<GtsamStereoPointFactor>(X(c->id_pose), Point3(points[c->id_point].p), Vector3(c->keypoint),
          cam->Fx(), cam->Fy(), cam->Cx(), cam->Cy(), cam->BF(), Tbc[c->id_pose], robust ? robustify(hSp, base) : base);
    }
    for (size_t i = 0; i < mono_line_constraints.size(); ++i) {
      if (!ml_in[i]) continue;
      auto& c = mono_line_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      SharedNoiseModel base = noiseModel::Isotropic::Sigma(2, lineSig);
      graph.emplace_shared<GtsamMonoLineFactor>(X(c->id_pose), lines[c->id_line].line_3d, c->line_2d,
          cam->Fx(), cam->Fy(), Kv_of(cam), Tbc[c->id_pose], robust ? robustify(hMl, base) : base);
    }
    for (size_t i = 0; i < stereo_line_constraints.size(); ++i) {
      if (!sl_in[i]) continue;
      auto& c = stereo_line_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      SharedNoiseModel base = noiseModel::Isotropic::Sigma(4, lineSig);
      graph.emplace_shared<GtsamStereoLineFactor>(X(c->id_pose), lines[c->id_line].line_3d, c->line_2d,
          cam->Fx(), cam->Fy(), cam->BF() / cam->Fx(), Kv_of(cam), Tbc[c->id_pose], robust ? robustify(hSl, base) : base);
    }

    LevenbergMarquardtParams params; params.setMaxIterations(10);
    try {
      result = LevenbergMarquardtOptimizer(graph, init, params).optimize();
    } catch (const std::exception&) {
      result = init;   // keep the incoming pose if the solve is degenerate (mirrors a failed g2o pass)
    }

    // reclassify by raw (information-weighted) chi2
    num_outlier = 0;
    for (size_t i = 0; i < mono_point_constraints.size(); ++i) {
      auto& c = mono_point_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      GtsamMonoPointFactor f(X(c->id_pose), Point3(points[c->id_point].p), Point2(c->keypoint),
          cam->Fx(), cam->Fy(), cam->Cx(), cam->Cy(), Tbc[c->id_pose], noiseModel::Unit::Create(2));
      c->inlier = mp_in[i] = f.residual(result.at<Pose3>(X(c->id_pose))).squaredNorm() <= cfg.mono_point;
      if (!mp_in[i]) num_outlier++;
    }
    for (size_t i = 0; i < stereo_point_constraints.size(); ++i) {
      auto& c = stereo_point_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      GtsamStereoPointFactor f(X(c->id_pose), Point3(points[c->id_point].p), Vector3(c->keypoint),
          cam->Fx(), cam->Fy(), cam->Cx(), cam->Cy(), cam->BF(), Tbc[c->id_pose], noiseModel::Unit::Create(3));
      c->inlier = sp_in[i] = f.residual(result.at<Pose3>(X(c->id_pose))).squaredNorm() <= cfg.stereo_point;
      if (!sp_in[i]) num_outlier++;
    }
    for (size_t i = 0; i < mono_line_constraints.size(); ++i) {
      auto& c = mono_line_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      GtsamMonoLineFactor f(X(c->id_pose), lines[c->id_line].line_3d, c->line_2d,
          cam->Fx(), cam->Fy(), Kv_of(cam), Tbc[c->id_pose], noiseModel::Isotropic::Sigma(2, lineSig));
      c->inlier = ml_in[i] = 0.1 * f.residual(result.at<Pose3>(X(c->id_pose))).squaredNorm() <= cfg.mono_line;
      if (!ml_in[i]) num_outlier++;
    }
    for (size_t i = 0; i < stereo_line_constraints.size(); ++i) {
      auto& c = stereo_line_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      GtsamStereoLineFactor f(X(c->id_pose), lines[c->id_line].line_3d, c->line_2d,
          cam->Fx(), cam->Fy(), cam->BF() / cam->Fx(), Kv_of(cam), Tbc[c->id_pose], noiseModel::Isotropic::Sigma(4, lineSig));
      c->inlier = sl_in[i] = 0.1 * f.residual(result.at<Pose3>(X(c->id_pose))).squaredNorm() <= cfg.stereo_line;
      if (!sl_in[i]) num_outlier++;
    }
    if (graph.size() < 10) break;   // too few factors to keep iterating (mirrors g2o's edges<10 guard)
  }

  for (auto& kv : poses) {          // recover camera poses Twc = Twb * Tbc
    Pose3 Twc = result.at<Pose3>(X(kv.first)) * Tbc[kv.first];
    kv.second.p = Twc.translation();
    kv.second.R = Twc.rotation().matrix();
  }
  return (int)n_c - num_outlier;
}

bool GtsamBackend::IMUInitialization(MapOfPoses& poses, MapOfVelocity& velocities, Bias& bias,
    std::vector<CameraPtr>& camera_list, VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg) {
  return ::IMUInitialization(poses, velocities, bias, camera_list, imu_constraints, Rwg);
}

void GtsamBackend::GlobalBA(std::shared_ptr<Map> map, const OptimizationConfig& cfg, bool point_outlier_rejection,
    bool line_outlier_rejection, int first_iterations, int second_iterations) {
  ::GlobalBA(map, cfg, point_outlier_rejection, line_outlier_rejection, first_iterations, second_iterations);
}
