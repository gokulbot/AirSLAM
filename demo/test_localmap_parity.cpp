// Standalone parity test for LocalmapOptimization: build a synthetic windowed-BA scene, run g2o's
// ::LocalmapOptimization and GtsamBackend::LocalmapOptimization on independent copies, and compare
// the optimized free poses / points / lines. Vision-only (no IMU), so the GTSAM path is exercised.
#include <iostream>
#include <iomanip>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "camera.h"
#include "g2o_optimization/g2o_optimization.h"   // ::LocalmapOptimization, types, OptimizationConfig
#include "gtsam_backend.h"

static Eigen::Vector3d projStereo(const Eigen::Matrix3d& Rwc, const Eigen::Vector3d& pwc,
                                  const Eigen::Vector3d& Xw, double fx, double fy, double cx, double cy, double bf) {
  Eigen::Vector3d Xc = Rwc.transpose() * (Xw - pwc);   // = Rcw*Xw + tcw
  double zi = 1.0 / Xc.z();
  double u = Xc.x() * zi * fx + cx, v = Xc.y() * zi * fy + cy;
  return Eigen::Vector3d(u, v, u - bf * zi);
}

int main(int argc, char** argv) {
  std::string cam_file = argc > 1 ? argv[1] : "configs/camera/glasgow.yaml";
  CameraPtr cam = std::make_shared<Camera>(cam_file);
  std::vector<CameraPtr> camera_list = {cam};
  double fx = cam->Fx(), fy = cam->Fy(), cx = cam->Cx(), cy = cam->Cy(), bf = cam->BF();

  const int NP = 4;                              // poses (first 2 fixed)
  std::vector<Eigen::Matrix3d> R_true(NP);
  std::vector<Eigen::Vector3d> p_true(NP);
  for (int i = 0; i < NP; i++) {
    p_true[i] = Eigen::Vector3d(i * 0.15, 0, 0);
    R_true[i] = Eigen::AngleAxisd(i * 0.02, Eigen::Vector3d::UnitY()).toRotationMatrix();
  }
  std::vector<Eigen::Vector3d> X_true;
  for (int i = 0; i < 40; i++)
    X_true.push_back(Eigen::Vector3d(((i % 8) - 4) * 0.25, ((i / 8) - 2) * 0.25, 3.0 + (i % 5) * 0.4));

  MapOfPoses poses; MapOfPoints3d points; MapOfLine3d lines;
  MapOfVelocity velocities; MapOfBias biases;
  VectorOfMonoPointConstraints mpc; VectorOfStereoPointConstraints spc;
  VectorOfMonoLineConstraints mlc; VectorOfStereoLineConstraints slc;
  VectorOfIMUConstraints imu;
  Eigen::Matrix3d Rwg = Eigen::Matrix3d::Identity();

  for (int i = 0; i < NP; i++) {
    Pose3d ps; ps.id_camera = 0; ps.fixed = (i < 2); ps.R = R_true[i]; ps.p = p_true[i];
    if (!ps.fixed) { ps.p += Eigen::Vector3d(0.03, -0.02, 0.02);
      ps.R = R_true[i] * Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitZ()).toRotationMatrix(); }
    poses[i] = ps;
  }
  for (size_t j = 0; j < X_true.size(); j++) {
    Position3d pt; pt.fixed = false; pt.p = X_true[j] + Eigen::Vector3d(0.02, -0.01, 0.03);
    points[j] = pt;
    for (int i = 0; i < NP; i++) {
      if ((R_true[i].transpose() * (X_true[j] - p_true[i])).z() < 0.5) continue;
      auto c = std::make_shared<StereoPointConstraint>();
      c->id_pose = i; c->id_point = (int)j; c->id_camera = 0; c->inlier = true;
      c->keypoint = projStereo(R_true[i], p_true[i], X_true[j], fx, fy, cx, cy, bf);
      spc.push_back(c);
    }
  }
  for (int li = 0; li < 4; li++) {
    Eigen::Vector3d A((li - 2) * 0.5, -0.5, 3.5), B((li - 2) * 0.5 + 0.3, 0.5, 3.7);
    Eigen::Vector3d d = B - A, w = A.cross(d);
    Eigen::Matrix<double, 6, 1> pl; pl.head<3>() = w; pl.tail<3>() = d;
    g2o::Line3D L_true(pl), L_init = L_true;
    L_init.oplus(Eigen::Vector4d(0.01, -0.01, 0.008, 0.02));
    Line3d ld; ld.fixed = false; ld.line_3d = L_init; lines[li] = ld;
    for (int i = 0; i < NP; i++) {
      Eigen::Vector3d kA = projStereo(R_true[i], p_true[i], A, fx, fy, cx, cy, bf);
      Eigen::Vector3d kB = projStereo(R_true[i], p_true[i], B, fx, fy, cx, cy, bf);
      auto c = std::make_shared<StereoLineConstraint>();
      c->id_pose = i; c->id_line = li; c->id_camera = 0; c->inlier = true; c->pixel_sigma = 1.0;
      Vector8d obs; obs << kA(0), kA(1), kB(0), kB(1), kA(2), kA(1), kB(2), kB(1);
      c->line_2d = obs; slc.push_back(c);
    }
  }

  OptimizationConfig cfg;
  cfg.mono_point = 5.991; cfg.stereo_point = 7.815; cfg.mono_line = 5.991; cfg.stereo_line = 9.488;

  MapOfPoses poses_g = poses, poses_t = poses;
  MapOfPoints3d points_g = points, points_t = points;
  MapOfLine3d lines_g = lines, lines_t = lines;

  G2oBackend g2o; GtsamBackend gt;
  g2o.LocalmapOptimization(poses_g, points_g, lines_g, velocities, biases, camera_list, mpc, spc, mlc, slc, imu, Rwg, cfg);
  gt.LocalmapOptimization(poses_t, points_t, lines_t, velocities, biases, camera_list, mpc, spc, mlc, slc, imu, Rwg, cfg);

  double max_pose = 0, max_pt = 0, max_ln = 0, err_g = 0, err_t = 0;
  for (int i = 0; i < NP; i++) if (!poses[i].fixed) {
    max_pose = std::max(max_pose, (poses_g[i].p - poses_t[i].p).norm());
    err_g = std::max(err_g, (poses_g[i].p - p_true[i]).norm());
    err_t = std::max(err_t, (poses_t[i].p - p_true[i]).norm());
  }
  for (size_t j = 0; j < X_true.size(); j++)
    max_pt = std::max(max_pt, (points_g[j].p - points_t[j].p).norm());
  for (int li = 0; li < 4; li++)
    max_ln = std::max(max_ln, (static_cast<const Eigen::Matrix<double, 6, 1>&>(lines_g[li].line_3d).normalized() -
                               static_cast<const Eigen::Matrix<double, 6, 1>&>(lines_t[li].line_3d).normalized()).norm());

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "free-pose recovery vs truth:  g2o=" << err_g * 1000 << "mm  gtsam=" << err_t * 1000 << "mm\n";
  std::cout << "g2o-vs-gtsam  pose_max=" << max_pose * 1000 << "mm  point_max=" << max_pt * 1000
            << "mm  line_max=" << max_ln << "\n";
  bool ok = err_g < 0.02 && err_t < 0.02 && max_pose < 0.05 && max_pt < 0.05;
  std::cout << (ok ? "PASS: both recover truth and agree\n" : "FAIL\n");
  return ok ? 0 : 1;
}
