#include "isam_smoother.h"

#include <cmath>

#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2Params.h>

#include "gtsam_plucker_line.h"   // GtsamStereoPointBAFactor
#include "gtsam_imu_factor.h"     // GtsamImuFactorVIO, GravityDir

using namespace gtsam;
using symbol_shorthand::P;
using symbol_shorthand::V;
using symbol_shorthand::X;

static const Key kBiasKey = Symbol('m', 0);      // shared IMU bias (Vector6 [gyr;acc])
static const Key kGravityKey = Symbol('r', 0);   // shared gravity direction

IsamSmoother::IsamSmoother(const gtsam::Pose3& body_P_camera, double fx, double fy, double cx, double cy, double bf)
    : Tbc_(body_P_camera), fx_(fx), fy_(fy), cx_(cx), cy_(cy), bf_(bf) {
  ISAM2Params params;
  params.relinearizeThreshold = 0.01;
  params.relinearizeSkip = 1;
  const double lag = 30.0;   // keep ~30 keyframes; older ones are marginalized into a boundary prior
  smoother_ = std::make_shared<IncrementalFixedLagSmoother>(lag, params);
}

void IsamSmoother::AddKeyframe(int kf_id, const gtsam::Pose3& Twb_init, bool anchor,
                               const std::vector<StereoObservation>& obs, const ImuKeyframeData& imu) {
  time_ += 1.0;
  NonlinearFactorGraph graph;
  Values new_values;
  FixedLagSmoother::KeyTimestampMap ts;

  new_values.insert(X(kf_id), Twb_init);
  ts[X(kf_id)] = time_;
  if (anchor) graph.addPrior(X(kf_id), Twb_init, noiseModel::Isotropic::Sigma(6, 1e-4));

  auto ptNoise = noiseModel::Unit::Create(3);
  auto inWindow = [&](int kf) { return kf == kf_id || estimate_.exists(X(kf)); };
  for (const auto& o : obs) {
    if (mature_.count(o.landmark_id)) {                         // existing landmark: add this observation
      graph.emplace_shared<GtsamStereoPointBAFactor>(X(kf_id), P(o.landmark_id), Vector3(o.keypoint),
                                                     fx_, fy_, cx_, cy_, bf_, Tbc_, ptNoise);
      ts[P(o.landmark_id)] = time_;                             // keep the landmark alive while observed
    } else {                                                    // fix #1: promote only at >=2 in-window views (no loose prior)
      pending_[o.landmark_id].push_back({kf_id, o.keypoint});
      std::vector<std::pair<int, Eigen::Vector3d>> inwin;
      for (auto& pk : pending_[o.landmark_id]) if (inWindow(pk.first)) inwin.push_back(pk);
      if (inwin.size() >= 2) {
        new_values.insert(P(o.landmark_id), Point3(o.Xw_init));
        ts[P(o.landmark_id)] = time_;
        for (auto& pk : inwin) {
          graph.emplace_shared<GtsamStereoPointBAFactor>(X(pk.first), P(o.landmark_id), Vector3(pk.second),
                                                         fx_, fy_, cx_, cy_, bf_, Tbc_, ptNoise);
          ts[X(pk.first)] = time_;                              // refresh the co-visible keyframe
        }
        mature_.insert(o.landmark_id);
        pending_.erase(o.landmark_id);
      }
    }
  }

  // --- VIO: velocity per keyframe + shared bias/gravity + IMU factor to the previous keyframe ---
  if (imu.active && imu.preint) {
    if (!vi_initialized_) {
      Vector6 bias0; bias0 << imu.gyr_bias, imu.acc_bias;
      new_values.insert(kBiasKey, bias0);
      new_values.insert(kGravityKey, GravityDir(imu.Rwg));
      Vector6 biasSig; biasSig << Vector3::Constant(0.1), Vector3::Constant(0.01);   // soft bias prior
      graph.addPrior(kBiasKey, bias0, noiseModel::Diagonal::Sigmas(biasSig));
      graph.addPrior(kGravityKey, GravityDir(imu.Rwg), noiseModel::Isotropic::Sigma(2, 0.5));
      vi_initialized_ = true;
    }
    ts[kBiasKey] = time_; ts[kGravityKey] = time_;              // shared vars never age out
    new_values.insert(V(kf_id), Vector3(imu.vel_init));
    ts[V(kf_id)] = time_;
    graph.addPrior(V(kf_id), Vector3(imu.vel_init), noiseModel::Isotropic::Sigma(3, 10.0));   // loose
    kf_with_velocity_.insert(kf_id);
    if (imu.prev_kf_id >= 0 && kf_with_velocity_.count(imu.prev_kf_id) && inWindow(imu.prev_kf_id)) {
      SharedNoiseModel imuNoise;
      try { imuNoise = noiseModel::Gaussian::Covariance(imu.preint->Cov.block<9, 9>(0, 0)); }
      catch (const std::exception&) { imuNoise = SharedNoiseModel(); }
      if (imuNoise) {
        graph.emplace_shared<GtsamImuFactorVIO>(X(imu.prev_kf_id), V(imu.prev_kf_id), X(kf_id), V(kf_id),
                                                kBiasKey, kGravityKey, imu.preint, imuNoise);
        ts[X(imu.prev_kf_id)] = time_; ts[V(imu.prev_kf_id)] = time_;
      }
    }
  }

  smoother_->update(graph, new_values, ts);
  estimate_ = smoother_->calculateEstimate();
  kf_ids_.push_back(kf_id);
}

gtsam::Pose3 IsamSmoother::GetBodyPose(int kf_id) const { return estimate_.at<Pose3>(X(kf_id)); }

Eigen::Matrix<double, 6, 6> IsamSmoother::GetCovariance(int kf_id) {
  return smoother_->getISAM2().marginalCovariance(X(kf_id));
}

double IsamSmoother::PositionSigma(int kf_id) {
  return std::sqrt(smoother_->getISAM2().marginalCovariance(X(kf_id)).block<3, 3>(3, 3).trace());
}
