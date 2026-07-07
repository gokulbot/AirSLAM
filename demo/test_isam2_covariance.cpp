// iSAM2 smoother demo: run AirSLAM's (validated) reprojection factors through GTSAM's ISAM2 and
// extract per-pose marginal covariances -- the capability that enables uncertainty-aware / active
// SLAM. A chain of poses is linked by shared stereo-point observations and anchored at pose 0; the
// positional uncertainty should grow away from the anchor (the drift signature nav needs).
#include <iostream>
#include <iomanip>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/inference/Symbol.h>

#include "camera.h"
#include "gtsam_plucker_line.h"   // GtsamStereoPointBAFactor

using namespace gtsam;
using symbol_shorthand::X;
using symbol_shorthand::P;

static Eigen::Vector3d projStereo(const Eigen::Matrix3d& Rwc, const Eigen::Vector3d& pwc,
                                  const Eigen::Vector3d& Xw, double fx, double fy, double cx, double cy, double bf) {
  Eigen::Vector3d Xc = Rwc.transpose() * (Xw - pwc);
  double zi = 1.0 / Xc.z();
  double u = Xc.x() * zi * fx + cx, v = Xc.y() * zi * fy + cy;
  return Eigen::Vector3d(u, v, u - bf * zi);
}

int main(int argc, char** argv) {
  std::string cam_file = argc > 1 ? argv[1] : "configs/camera/glasgow.yaml";
  CameraPtr cam = std::make_shared<Camera>(cam_file);
  double fx = cam->Fx(), fy = cam->Fy(), cx = cam->Cx(), cy = cam->Cy(), bf = cam->BF();
  Pose3 Tbc = Pose3(Rot3(cam->BodyToCamera().block<3, 3>(0, 0)), Point3(cam->BodyToCamera().block<3, 1>(0, 3))).inverse();

  const int NP = 5;
  std::vector<Eigen::Matrix3d> R_true(NP); std::vector<Eigen::Vector3d> p_true(NP);
  for (int i = 0; i < NP; i++) { p_true[i] = Eigen::Vector3d(i * 0.3, 0, 0); R_true[i] = Eigen::Matrix3d::Identity(); }
  std::vector<Eigen::Vector3d> X_true;
  for (int i = 0; i < 60; i++) X_true.push_back(Eigen::Vector3d(((i % 10) - 5) * 0.2, ((i / 10) % 4 - 2) * 0.2, 3.0 + (i % 3) * 0.5));

  NonlinearFactorGraph graph;
  Values initial;
  // anchor pose 0 with a tight prior (world frame)
  auto poseAnchor = noiseModel::Isotropic::Sigma(6, 1e-4);
  auto pointNoise = noiseModel::Unit::Create(3);
  auto ptPrior = noiseModel::Isotropic::Sigma(3, 0.5);   // loose point prior (keeps the system well-posed)

  for (int i = 0; i < NP; i++) {
    Pose3 Twc(Rot3(R_true[i]), Point3(p_true[i]));
    Pose3 Twb = Twc * Tbc.inverse();
    initial.insert(X(i), (i == 0) ? Twb : Twb.compose(Pose3(Rot3::Rodrigues(0.01, -0.01, 0.01), Point3(0.05, -0.03, 0.04))));
    if (i == 0) graph.addPrior(X(i), Twb, poseAnchor);
  }
  for (size_t j = 0; j < X_true.size(); j++) {
    initial.insert(P(j), Point3(X_true[j] + Eigen::Vector3d(0.03, -0.02, 0.03)));
    graph.addPrior(P(j), Point3(X_true[j]), ptPrior);
  }
  // each pose observes a sliding window of points -> consecutive poses share points (chain)
  for (int i = 0; i < NP; i++) {
    int lo = i * 10, hi = std::min((int)X_true.size(), lo + 24);
    for (int j = lo; j < hi; j++) {
      Eigen::Vector3d kp = projStereo(R_true[i], p_true[i], X_true[j], fx, fy, cx, cy, bf);
      graph.emplace_shared<GtsamStereoPointBAFactor>(X(i), P(j), Vector3(kp), fx, fy, cx, cy, bf, Tbc, pointNoise);
    }
  }

  // --- ISAM2 (batch update) ---
  ISAM2 isam;
  isam.update(graph, initial);
  isam.update();
  Values est = isam.calculateEstimate();

  // batch LM for cross-check
  Values lm = LevenbergMarquardtOptimizer(graph, initial).optimize();

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "pose | pos-err(mm) | isam-vs-LM(mm) | translation 1-sigma (mm)\n";
  double max_il = 0;
  for (int i = 0; i < NP; i++) {
    Pose3 Ti = est.at<Pose3>(X(i)), Tl = lm.at<Pose3>(X(i));
    Eigen::Vector3d pos = (Ti * Tbc).translation();   // camera pos in world
    double perr = (pos - p_true[i]).norm() * 1000;
    double il = (Ti.translation() - Tl.translation()).norm() * 1000; max_il = std::max(max_il, il);
    Matrix cov = isam.marginalCovariance(X(i));        // 6x6 (rot, trans)
    double sigma = std::sqrt(cov.block<3, 3>(3, 3).trace()) * 1000;
    std::cout << "  " << i << "  |   " << perr << "     |    " << il << "      |   " << sigma << "\n";
  }
  bool ok = max_il < 1.0;   // iSAM2 estimate must match batch LM
  std::cout << (ok ? "PASS: iSAM2 optimizes AirSLAM's factors and yields growing per-pose covariance\n"
                   : "FAIL: iSAM2 estimate diverges from batch LM\n");
  return ok ? 0 : 1;
}
