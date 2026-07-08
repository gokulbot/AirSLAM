#include "gtsam_backend.h"
#include "g2o_optimization/g2o_optimization.h"   // forward the un-migrated methods

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/PriorFactor.h>
#include "gtsam_line_factor.h"
#include "gtsam_plucker_line.h"
#include "gtsam_imu_factor.h"
#include "map.h"
#include "frame.h"
#include "mappoint.h"
#include "mapline.h"
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
// Windowed bundle adjustment re-expressed in GTSAM, mirroring g2o's LocalmapOptimization: free
// body poses + free points (Point3) + free lines (PluckerLine); fixed ones pinned by tight priors.
// 2-pass robust loop: optimize(5) with Huber -> classify (chi2 vs cfg.*, points also need depth>0)
// -> drop the kernel -> optimize(15) inliers-only (warm start) -> final classify. Point info = I;
// line info = pixel_sigma*I (=> sigma 1/sqrt(pixel_sigma)). VI case forwarded to g2o.
void GtsamBackend::LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines,
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list,
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
    VectorOfIMUConstraints& imu_constraints, const Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg) {
  using namespace gtsam;
  using symbol_shorthand::X; using symbol_shorthand::P; using symbol_shorthand::L;

  if (!velocities.empty() || !biases.empty() || !imu_constraints.empty()) {
    ::LocalmapOptimization(poses, points, lines, velocities, biases, camera_list,
        mono_point_constraints, stereo_point_constraints, mono_line_constraints, stereo_line_constraints,
        imu_constraints, Rwg, cfg);
    return;
  }

  auto Tcb_of = [&](int cam) {
    Eigen::Matrix4d T = camera_list[cam]->BodyToCamera();
    return Pose3(Rot3(T.block<3, 3>(0, 0)), Point3(T.block<3, 1>(0, 3)));
  };
  std::map<int, Pose3> Twb_init, Tbc;
  for (auto& kv : poses) {
    Pose3 Twc(Rot3(kv.second.R), Point3(kv.second.p));
    Pose3 tcb = Tcb_of(kv.second.id_camera);
    Twb_init[kv.first] = Twc * tcb;
    Tbc[kv.first] = tcb.inverse();
  }
  auto Kv_of = [&](CameraPtr c) {
    return Eigen::Vector3d(-c->Cx() * c->Fy(), -c->Fx() * c->Cy(), c->Fx() * c->Fy());
  };
  const double hMp = std::sqrt(cfg.mono_point), hSp = std::sqrt(cfg.stereo_point);
  const double hMl = std::sqrt(cfg.mono_line), hSl = std::sqrt(cfg.stereo_line);
  auto robustify = [](double k, const SharedNoiseModel& b) {
    return noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(k), b);
  };
  auto lineNoise = [](double pixel_sigma, int d) {                 // info = pixel_sigma*I
    return noiseModel::Isotropic::Sigma(d, 1.0 / std::sqrt(pixel_sigma));
  };

  Values values;
  for (auto& kv : poses) values.insert(X(kv.first), Twb_init[kv.first]);
  for (auto& kv : points) values.insert(P(kv.first), Point3(kv.second.p));
  for (auto& kv : lines) values.insert(L(kv.first), PluckerLine(kv.second.line_3d));

  std::vector<bool> mp_in(mono_point_constraints.size(), true), sp_in(stereo_point_constraints.size(), true);
  std::vector<bool> ml_in(mono_line_constraints.size(), true), sl_in(stereo_line_constraints.size(), true);

  auto buildGraph = [&](bool robust) {
    NonlinearFactorGraph g;
    for (auto& kv : poses) if (kv.second.fixed) g.addPrior(X(kv.first), Twb_init[kv.first], noiseModel::Isotropic::Sigma(6, 1e-9));
    for (auto& kv : points) if (kv.second.fixed) g.addPrior(P(kv.first), Point3(kv.second.p), noiseModel::Isotropic::Sigma(3, 1e-9));
    for (auto& kv : lines) if (kv.second.fixed) g.emplace_shared<PriorFactor<PluckerLine>>(L(kv.first), PluckerLine(kv.second.line_3d), noiseModel::Isotropic::Sigma(4, 1e-9));
    for (size_t i = 0; i < mono_point_constraints.size(); ++i) {
      if (!mp_in[i]) continue; auto& c = mono_point_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      SharedNoiseModel base = noiseModel::Unit::Create(2);
      g.emplace_shared<GtsamMonoPointBAFactor>(X(c->id_pose), P(c->id_point), Point2(c->keypoint),
          cam->Fx(), cam->Fy(), cam->Cx(), cam->Cy(), Tbc[c->id_pose], robust ? robustify(hMp, base) : base);
    }
    for (size_t i = 0; i < stereo_point_constraints.size(); ++i) {
      if (!sp_in[i]) continue; auto& c = stereo_point_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      SharedNoiseModel base = noiseModel::Unit::Create(3);
      g.emplace_shared<GtsamStereoPointBAFactor>(X(c->id_pose), P(c->id_point), Vector3(c->keypoint),
          cam->Fx(), cam->Fy(), cam->Cx(), cam->Cy(), cam->BF(), Tbc[c->id_pose], robust ? robustify(hSp, base) : base);
    }
    for (size_t i = 0; i < mono_line_constraints.size(); ++i) {
      if (!ml_in[i]) continue; auto& c = mono_line_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      SharedNoiseModel base = lineNoise(c->pixel_sigma, 2);
      g.emplace_shared<GtsamMonoLineBAFactor>(X(c->id_pose), L(c->id_line), c->line_2d,
          cam->Fx(), cam->Fy(), Kv_of(cam), Tbc[c->id_pose], robust ? robustify(hMl, base) : base);
    }
    for (size_t i = 0; i < stereo_line_constraints.size(); ++i) {
      if (!sl_in[i]) continue; auto& c = stereo_line_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      SharedNoiseModel base = lineNoise(c->pixel_sigma, 4);
      g.emplace_shared<GtsamStereoLineBAFactor>(X(c->id_pose), L(c->id_line), c->line_2d,
          cam->Fx(), cam->Fy(), cam->BF() / cam->Fx(), Kv_of(cam), Tbc[c->id_pose], robust ? robustify(hSl, base) : base);
    }
    return g;
  };

  auto depthPos = [&](const Values& v, int pose, const Point3& Xw) {
    return ((v.at<Pose3>(X(pose)) * Tbc[pose]).inverse() * Xw).z() > 0;
  };
  auto classify = [&](const Values& v) {
    for (size_t i = 0; i < mono_point_constraints.size(); ++i) {
      auto& c = mono_point_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      GtsamMonoPointBAFactor f(X(c->id_pose), P(c->id_point), Point2(c->keypoint),
          cam->Fx(), cam->Fy(), cam->Cx(), cam->Cy(), Tbc[c->id_pose], noiseModel::Unit::Create(2));
      Point3 Xw = v.at<Point3>(P(c->id_point));
      c->inlier = mp_in[i] = f.residual(v.at<Pose3>(X(c->id_pose)), Xw).squaredNorm() <= cfg.mono_point && depthPos(v, c->id_pose, Xw);
    }
    for (size_t i = 0; i < stereo_point_constraints.size(); ++i) {
      auto& c = stereo_point_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      GtsamStereoPointBAFactor f(X(c->id_pose), P(c->id_point), Vector3(c->keypoint),
          cam->Fx(), cam->Fy(), cam->Cx(), cam->Cy(), cam->BF(), Tbc[c->id_pose], noiseModel::Unit::Create(3));
      Point3 Xw = v.at<Point3>(P(c->id_point));
      c->inlier = sp_in[i] = f.residual(v.at<Pose3>(X(c->id_pose)), Xw).squaredNorm() <= cfg.stereo_point && depthPos(v, c->id_pose, Xw);
    }
    for (size_t i = 0; i < mono_line_constraints.size(); ++i) {
      auto& c = mono_line_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      GtsamMonoLineBAFactor f(X(c->id_pose), L(c->id_line), c->line_2d, cam->Fx(), cam->Fy(), Kv_of(cam), Tbc[c->id_pose], lineNoise(c->pixel_sigma, 2));
      c->inlier = ml_in[i] = c->pixel_sigma * f.residual(v.at<Pose3>(X(c->id_pose)), v.at<PluckerLine>(L(c->id_line))).squaredNorm() <= cfg.mono_line;
    }
    for (size_t i = 0; i < stereo_line_constraints.size(); ++i) {
      auto& c = stereo_line_constraints[i]; CameraPtr cam = camera_list[c->id_camera];
      GtsamStereoLineBAFactor f(X(c->id_pose), L(c->id_line), c->line_2d, cam->Fx(), cam->Fy(), cam->BF() / cam->Fx(), Kv_of(cam), Tbc[c->id_pose], lineNoise(c->pixel_sigma, 4));
      c->inlier = sl_in[i] = c->pixel_sigma * f.residual(v.at<Pose3>(X(c->id_pose)), v.at<PluckerLine>(L(c->id_line))).squaredNorm() <= cfg.stereo_line;
    }
  };

  for (int pass = 0; pass < 2; ++pass) {                 // g2o: optimize(5) robust, then optimize(15) plain
    NonlinearFactorGraph g = buildGraph(pass == 0);
    LevenbergMarquardtParams params; params.setMaxIterations(pass == 0 ? 5 : 15);
    try { values = LevenbergMarquardtOptimizer(g, values, params).optimize(); }
    catch (const std::exception&) {}
    classify(values);
  }

  for (auto& kv : poses) {
    Pose3 Twc = values.at<Pose3>(X(kv.first)) * Tbc[kv.first];
    kv.second.p = Twc.translation();
    kv.second.R = Twc.rotation().matrix();
  }
  for (auto& kv : points) kv.second.p = values.at<Point3>(P(kv.first));
  for (auto& kv : lines) kv.second.line_3d = values.at<PluckerLine>(L(kv.first)).g2o_line();
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

// Visual-inertial initialization re-expressed in GTSAM, mirroring g2o's IMUInitialization: poses
// fixed from vision; estimate per-keyframe velocities + a single shared gyro/acc bias + the gravity
// direction Rwg, linked by AirSLAM's EdgeIMU preintegration constraint (custom GtsamImuFactor) plus
// soft bias priors (g2o's EdgeGyr/EdgeAcc with info 1e2 / 1e5). Batch Levenberg-Marquardt.
bool GtsamBackend::IMUInitialization(MapOfPoses& poses, MapOfVelocity& velocities, Bias& bias,
    std::vector<CameraPtr>& camera_list, VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg) {
  using namespace gtsam;
  using symbol_shorthand::V;
  const Key Bg = Symbol('b', 0), Ba = Symbol('c', 0), Gk = Symbol('r', 0);
  if (imu_constraints.empty() || velocities.empty()) return false;

  NonlinearFactorGraph graph; Values values;
  for (auto& kv : velocities) values.insert(V(kv.first), Vector3(kv.second.velocity));
  values.insert(Bg, Vector3(bias.gyr_bias));
  values.insert(Ba, Vector3(bias.acc_bias));
  values.insert(Gk, GravityDir(Rwg));
  graph.addPrior(Bg, Vector3(bias.gyr_bias), noiseModel::Isotropic::Sigma(3, 1.0 / std::sqrt(1e2)));   // EdgeGyr prior
  graph.addPrior(Ba, Vector3(bias.acc_bias), noiseModel::Isotropic::Sigma(3, 1.0 / std::sqrt(1e5)));   // EdgeAcc prior

  for (auto& ipc : imu_constraints) {
    if (!velocities.count(ipc->id_pose1) || !velocities.count(ipc->id_pose2)) continue;
    Pose3d& p1 = poses[ipc->id_pose1]; Pose3d& p2 = poses[ipc->id_pose2];
    Eigen::Matrix4d T1 = camera_list[p1.id_camera]->BodyToCamera(), T2 = camera_list[p2.id_camera]->BodyToCamera();
    Eigen::Matrix3d Rwb1 = p1.R * T1.block<3, 3>(0, 0), Rwb2 = p2.R * T2.block<3, 3>(0, 0);
    Eigen::Vector3d twb1 = p1.p + p1.R * T1.block<3, 1>(0, 3), twb2 = p2.p + p2.R * T2.block<3, 1>(0, 3);
    SharedNoiseModel noise;
    try { noise = noiseModel::Gaussian::Covariance(ipc->preinteration->Cov.block<9, 9>(0, 0)); }
    catch (const std::exception&) { continue; }
    graph.emplace_shared<GtsamImuFactor>(V(ipc->id_pose1), Bg, Ba, V(ipc->id_pose2), Gk,
        ipc->preinteration, Rwb1, twb1, Rwb2, twb2, noise);
  }

  Values result;
  try { result = LevenbergMarquardtOptimizer(graph, values).optimize(); }
  catch (const std::exception&) { return false; }

  for (auto& kv : velocities) kv.second.velocity = result.at<Vector3>(V(kv.first));
  bias.gyr_bias = result.at<Vector3>(Bg);
  bias.acc_bias = result.at<Vector3>(Ba);
  Rwg = result.at<GravityDir>(Gk).Rwg();
  return true;
}

// Full-map bundle adjustment re-expressed in GTSAM, mirroring g2o's GlobalBA: extract the whole map
// (first keyframe anchored, all others + points + lines free), optimize(first_iterations) with
// Huber, optionally reject point/line outliers (chi2 vs cfg.*, points also need depth>0), drop the
// kernel, optimize(second_iterations), remove outlier observations from the map, write back
// SetPose/SetPosition/SetLine3D. Point info=I; line info=cov*I (cov=0.1 if >3 observers else 0.001).
// VI maps are forwarded to g2o (IMU factors not migrated).
void GtsamBackend::GlobalBA(std::shared_ptr<Map> map, const OptimizationConfig& cfg, bool point_outlier_rejection,
    bool line_outlier_rejection, int first_iterations, int second_iterations) {
  using namespace gtsam;
  using symbol_shorthand::X; using symbol_shorthand::P; using symbol_shorthand::L;

  if (map->IMUInit()) {
    ::GlobalBA(map, cfg, point_outlier_rejection, line_outlier_rejection, first_iterations, second_iterations);
    return;
  }

  std::map<int, FramePtr>& keyframes = map->GetAllKeyframes();
  std::map<int, MappointPtr>& mappoints = map->GetAllMappoints();
  std::map<int, MaplinePtr>& maplines = map->GetAllMaplines();
  if (keyframes.empty()) return;
  CameraPtr cam = map->GetCameraPtr();
  double fx = cam->Fx(), fy = cam->Fy(), cx = cam->Cx(), cy = cam->Cy(), bf = cam->BF();
  Eigen::Vector3d Kv(-cx * fy, -fx * cy, fx * fy);
  Pose3 Tcb(Rot3(cam->BodyToCamera().block<3, 3>(0, 0)), Point3(cam->BodyToCamera().block<3, 1>(0, 3)));
  Pose3 Tbc = Tcb.inverse();
  const int anchor = keyframes.begin()->first;

  Values values;
  for (auto& kv : keyframes) {
    Eigen::Matrix4d Twc = kv.second->GetPose();
    values.insert(X(kv.first), Pose3(Rot3(Twc.block<3, 3>(0, 0)), Point3(Twc.block<3, 1>(0, 3))) * Tcb);
  }
  for (auto& kv : mappoints) if (kv.second && kv.second->IsValid()) values.insert(P(kv.first), Point3(kv.second->GetPosition()));
  for (auto& kv : maplines) if (kv.second && kv.second->IsValid()) values.insert(L(kv.first), PluckerLine(kv.second->GetLine3D()));

  struct PtObs { int kf, lm; bool stereo; Eigen::Vector3d kp; };
  struct LnObs { int kf, lm; bool stereo; Vector8d obs; double cov; };
  std::vector<PtObs> pts; std::vector<LnObs> lns;
  for (auto& kv : keyframes) {
    FramePtr frame = kv.second; int fid = kv.first;
    std::vector<MappointPtr>& fmpts = frame->GetAllMappoints();
    for (size_t i = 0; i < fmpts.size(); i++) {
      MappointPtr mpt = fmpts[i]; if (!mpt || !mpt->IsValid()) continue;
      Eigen::Vector3d kp; if (!frame->GetKeypointPosition(i, kp)) continue;
      if (!values.exists(P(mpt->GetId()))) continue;
      pts.push_back({fid, mpt->GetId(), kp(2) >= 0, kp});
    }
    std::vector<MaplinePtr>& fmpls = frame->GetAllMaplines();
    for (size_t i = 0; i < fmpls.size(); i++) {
      MaplinePtr mpl = fmpls[i]; if (!mpl || !mpl->IsValid()) continue;
      Eigen::Vector4d ll, lr; if (!frame->GetLine(i, ll)) continue;
      if (!values.exists(L(mpl->GetId()))) continue;
      double cov = mpl->ObverserNum() > 3 ? 0.1 : 0.001;
      Vector8d o; o.setZero(); o.head<4>() = ll;
      bool st = frame->GetLineRight(i, lr); if (st) o.tail<4>() = lr;
      lns.push_back({fid, mpl->GetId(), st, o, cov});
    }
  }

  std::vector<bool> pt_in(pts.size(), true), ln_in(lns.size(), true);
  const double hMp = std::sqrt(cfg.mono_point), hSp = std::sqrt(cfg.stereo_point);
  const double hMl = std::sqrt(cfg.mono_line), hSl = std::sqrt(cfg.stereo_line);
  auto robustify = [](double k, const SharedNoiseModel& b) {
    return noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(k), b);
  };
  auto lineNoise = [](double cov, int d) { return noiseModel::Isotropic::Sigma(d, 1.0 / std::sqrt(cov)); };

  auto buildGraph = [&](bool robust) {
    NonlinearFactorGraph g;
    g.addPrior(X(anchor), values.at<Pose3>(X(anchor)), noiseModel::Isotropic::Sigma(6, 1e-4));  // gauge; 1e-9 (info 1e18) wrecked Cholesky conditioning
    for (size_t i = 0; i < pts.size(); i++) { if (!pt_in[i]) continue; auto& o = pts[i];
      if (o.stereo) { SharedNoiseModel b = noiseModel::Unit::Create(3);
        g.emplace_shared<GtsamStereoPointBAFactor>(X(o.kf), P(o.lm), Vector3(o.kp), fx, fy, cx, cy, bf, Tbc, robust ? robustify(hSp, b) : b); }
      else { SharedNoiseModel b = noiseModel::Unit::Create(2);
        g.emplace_shared<GtsamMonoPointBAFactor>(X(o.kf), P(o.lm), Point2(o.kp.head<2>()), fx, fy, cx, cy, Tbc, robust ? robustify(hMp, b) : b); } }
    for (size_t i = 0; i < lns.size(); i++) { if (!ln_in[i]) continue; auto& o = lns[i];
      if (o.stereo) { SharedNoiseModel b = lineNoise(o.cov, 4);
        g.emplace_shared<GtsamStereoLineBAFactor>(X(o.kf), L(o.lm), o.obs, fx, fy, bf / fx, Kv, Tbc, robust ? robustify(hSl, b) : b); }
      else { SharedNoiseModel b = lineNoise(o.cov, 2);
        g.emplace_shared<GtsamMonoLineBAFactor>(X(o.kf), L(o.lm), o.obs.head<4>(), fx, fy, Kv, Tbc, robust ? robustify(hMl, b) : b); } }
    return g;
  };
  auto depthPos = [&](const Values& v, int kf, const Point3& Xw) { return ((v.at<Pose3>(X(kf)) * Tbc).inverse() * Xw).z() > 0; };
  auto ptChi2 = [&](const Values& v, const PtObs& o) {
    Point3 Xw = v.at<Point3>(P(o.lm));
    if (o.stereo) return GtsamStereoPointBAFactor(X(o.kf), P(o.lm), Vector3(o.kp), fx, fy, cx, cy, bf, Tbc, noiseModel::Unit::Create(3)).residual(v.at<Pose3>(X(o.kf)), Xw).squaredNorm();
    return GtsamMonoPointBAFactor(X(o.kf), P(o.lm), Point2(o.kp.head<2>()), fx, fy, cx, cy, Tbc, noiseModel::Unit::Create(2)).residual(v.at<Pose3>(X(o.kf)), Xw).squaredNorm();
  };
  auto lnChi2 = [&](const Values& v, const LnObs& o) {
    PluckerLine Ln = v.at<PluckerLine>(L(o.lm));
    if (o.stereo) return o.cov * GtsamStereoLineBAFactor(X(o.kf), L(o.lm), o.obs, fx, fy, bf / fx, Kv, Tbc, lineNoise(o.cov, 4)).residual(v.at<Pose3>(X(o.kf)), Ln).squaredNorm();
    return o.cov * GtsamMonoLineBAFactor(X(o.kf), L(o.lm), o.obs.head<4>(), fx, fy, Kv, Tbc, lineNoise(o.cov, 2)).residual(v.at<Pose3>(X(o.kf)), Ln).squaredNorm();
  };

  { LevenbergMarquardtParams p; p.setMaxIterations(first_iterations);
    try { values = LevenbergMarquardtOptimizer(buildGraph(true), values, p).optimize(); } catch (const std::exception&) {} }
  if (point_outlier_rejection)
    for (size_t i = 0; i < pts.size(); i++)
      pt_in[i] = ptChi2(values, pts[i]) <= (pts[i].stereo ? cfg.stereo_point : cfg.mono_point) && depthPos(values, pts[i].kf, values.at<Point3>(P(pts[i].lm)));
  if (line_outlier_rejection)
    for (size_t i = 0; i < lns.size(); i++)
      ln_in[i] = lnChi2(values, lns[i]) <= (lns[i].stereo ? cfg.stereo_line : cfg.mono_line);

  if (second_iterations > 0) {
    LevenbergMarquardtParams p; p.setMaxIterations(second_iterations);
    try { values = LevenbergMarquardtOptimizer(buildGraph(false), values, p).optimize(); } catch (const std::exception&) {}
    std::vector<std::pair<FramePtr, MappointPtr>> mpt_out;
    std::vector<std::pair<FramePtr, MaplinePtr>> mpl_out;
    for (size_t i = 0; i < pts.size(); i++)
      if (!(ptChi2(values, pts[i]) <= (pts[i].stereo ? cfg.stereo_point : cfg.mono_point) && depthPos(values, pts[i].kf, values.at<Point3>(P(pts[i].lm)))))
        mpt_out.emplace_back(keyframes[pts[i].kf], mappoints[pts[i].lm]);
    for (size_t i = 0; i < lns.size(); i++)
      if (!(lnChi2(values, lns[i]) <= (lns[i].stereo ? cfg.stereo_line : cfg.mono_line)))
        mpl_out.emplace_back(keyframes[lns[i].kf], maplines[lns[i].lm]);
    map->RemoveOutliers(mpt_out);
    map->RemoveLineOutliers(mpl_out);
  }

  for (auto& kv : keyframes) {
    Pose3 Twc = values.at<Pose3>(X(kv.first)) * Tbc;
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M.block<3, 3>(0, 0) = Twc.rotation().matrix(); M.block<3, 1>(0, 3) = Twc.translation();
    kv.second->SetPose(M);
  }
  for (auto& kv : mappoints) if (kv.second && kv.second->IsValid() && values.exists(P(kv.first))) kv.second->SetPosition(values.at<Point3>(P(kv.first)));
  for (auto& kv : maplines) if (kv.second && kv.second->IsValid() && values.exists(L(kv.first))) {
    kv.second->SetLine3D(values.at<PluckerLine>(L(kv.first)).g2o_line());
    kv.second->SetEndpointsValidStatus(map->UppdateMapline(kv.second));
  }
}
