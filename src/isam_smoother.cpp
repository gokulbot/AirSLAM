#include "isam_smoother.h"

#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "gtsam_plucker_line.h"   // GtsamStereoPointBAFactor
#include "gtsam_imu_factor.h"     // GtsamImuFactorVIO, GravityDir

using namespace gtsam;
using symbol_shorthand::P;
using symbol_shorthand::V;
using symbol_shorthand::X;

static const Key kBiasKey = Symbol('m', 0);      // shared IMU bias (Vector6 [gyr;acc])
static const Key kGravityKey = Symbol('r', 0);   // shared gravity direction

IsamSmoother::IsamSmoother(const gtsam::Pose3& body_P_camera, double fx, double fy, double cx, double cy, double bf)
    : Tbc_(body_P_camera), fx_(fx), fy_(fy), cx_(cx), cy_(cy), bf_(bf) {}

void IsamSmoother::AddKeyframe(int kf_id, const gtsam::Pose3& Twb_init, bool anchor,
                               const std::vector<StereoObservation>& obs, const ImuKeyframeData& imu) {
  NonlinearFactorGraph graph;
  Values new_values;
  new_values.insert(X(kf_id), Twb_init);
  if (anchor) graph.addPrior(X(kf_id), Twb_init, noiseModel::Isotropic::Sigma(6, 1e-4));

  auto ptNoise = noiseModel::Unit::Create(3);
  auto ptPrior = noiseModel::Isotropic::Sigma(3, 0.5);   // loose prior keeps each new landmark well-posed
  for (const auto& o : obs) {
    if (!known_landmarks_.count(o.landmark_id)) {         // add the landmark variable only the first time
      known_landmarks_.insert(o.landmark_id);
      new_values.insert(P(o.landmark_id), Point3(o.Xw_init));
      graph.addPrior(P(o.landmark_id), Point3(o.Xw_init), ptPrior);
    }
    graph.emplace_shared<GtsamStereoPointBAFactor>(X(kf_id), P(o.landmark_id), Vector3(o.keypoint),
                                                   fx_, fy_, cx_, cy_, bf_, Tbc_, ptNoise);
  }

  // --- VIO: velocity + shared bias/gravity + IMU factor to the previous keyframe ---
  if (imu.active && imu.preint) {
    if (!vi_initialized_) {                              // create the shared bias + gravity once
      Vector6 bias0; bias0 << imu.gyr_bias, imu.acc_bias;
      new_values.insert(kBiasKey, bias0);
      new_values.insert(kGravityKey, GravityDir(imu.Rwg));
      Vector6 biasSig; biasSig << Vector3::Constant(0.1), Vector3::Constant(0.01);   // soft bias prior
      graph.addPrior(kBiasKey, bias0, noiseModel::Diagonal::Sigmas(biasSig));
      graph.addPrior(kGravityKey, GravityDir(imu.Rwg), noiseModel::Isotropic::Sigma(2, 0.5));
      vi_initialized_ = true;
    }
    new_values.insert(V(kf_id), Vector3(imu.vel_init));
    graph.addPrior(V(kf_id), Vector3(imu.vel_init), noiseModel::Isotropic::Sigma(3, 10.0));   // loose, keeps iSAM2 well-posed
    kf_with_velocity_.insert(kf_id);
    if (imu.prev_kf_id >= 0 && kf_with_velocity_.count(imu.prev_kf_id)) {
      SharedNoiseModel imuNoise;
      try { imuNoise = noiseModel::Gaussian::Covariance(imu.preint->Cov.block<9, 9>(0, 0)); }
      catch (const std::exception&) { imuNoise = SharedNoiseModel(); }
      if (imuNoise)
        graph.emplace_shared<GtsamImuFactorVIO>(X(imu.prev_kf_id), V(imu.prev_kf_id), X(kf_id), V(kf_id),
                                                kBiasKey, kGravityKey, imu.preint, imuNoise);
    }
  }

  isam_.update(graph, new_values);      // incremental Bayes-tree update
  estimate_ = isam_.calculateEstimate();
  kf_ids_.push_back(kf_id);
}

gtsam::Pose3 IsamSmoother::GetBodyPose(int kf_id) const { return estimate_.at<Pose3>(X(kf_id)); }

Eigen::Matrix<double, 6, 6> IsamSmoother::GetCovariance(int kf_id) { return isam_.marginalCovariance(X(kf_id)); }

double IsamSmoother::PositionSigma(int kf_id) {
  return std::sqrt(isam_.marginalCovariance(X(kf_id)).block<3, 3>(3, 3).trace());
}
