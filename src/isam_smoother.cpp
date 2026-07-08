#include "isam_smoother.h"

#include <cmath>

#include <boost/make_shared.hpp>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "gtsam_imu_factor.h"     // GtsamImuFactorVIO, GravityDir

using namespace gtsam;
using symbol_shorthand::V;
using symbol_shorthand::X;

static const Key kBiasKey = Symbol('m', 0);      // shared IMU bias (Vector6 [gyr;acc])
static const Key kGravityKey = Symbol('r', 0);   // shared gravity direction

IsamSmoother::IsamSmoother(const gtsam::Pose3& body_P_camera, double fx, double fy, double cx, double cy, double bf)
    : Tbc_(body_P_camera) {
  K_ = boost::make_shared<Cal3_S2Stereo>(fx, fy, 0.0, cx, cy, bf / fx);   // baseline = bf/fx
  sparams_.setDegeneracyMode(gtsam::ZERO_ON_DEGENERACY);                  // single-view/degenerate -> zero, not throw
  sparams_.setEnableEPI(false);
  // NB: iSAM2 dogleg was tried here and made conditioning WORSE with incremental smart-factor
  // re-adds -- reverted. Robustness comes from the loose per-pose prior in AddKeyframe instead.
}

void IsamSmoother::AddKeyframe(int kf_id, const gtsam::Pose3& Twb_init, bool anchor,
                               const std::vector<StereoObservation>& obs, const ImuKeyframeData& imu) {
  NonlinearFactorGraph new_factors;
  Values new_values;
  FactorIndices to_remove;
  new_values.insert(X(kf_id), Twb_init);
  // anchor: tight (gauge). Every other pose: a LOOSE regularizing prior so a momentarily
  // vision-degenerate keyframe can't make iSAM2 indeterminant. Floor ~2 m / 0.5 rad -- far above the
  // cm-dm drift signal nav consumes, so it only bites otherwise-singular poses.
  if (anchor) {
    new_factors.addPrior(X(kf_id), Twb_init, noiseModel::Isotropic::Sigma(6, 1e-4));
  } else {
    Vector6 rs; rs << Vector3::Constant(0.5), Vector3::Constant(2.0);
    new_factors.addPrior(X(kf_id), Twb_init, noiseModel::Diagonal::Sigmas(rs));
  }

  auto pixNoise = noiseModel::Isotropic::Sigma(3, 1.0);
  std::vector<std::pair<int, size_t>> smart_pos;   // (landmark, position of its factor in new_factors)
  for (const auto& o : obs) {
    obs_[o.landmark_id].push_back({kf_id, o.keypoint});
    // Rebuild a FRESH smart factor with all observations. Never mutate a factor already in iSAM2 --
    // that desyncs its cached keys and breaks removal-by-index; the old one is removed cleanly.
    auto sf = boost::make_shared<SmartFactor>(pixNoise, sparams_, Tbc_);
    for (const auto& ob : obs_[o.landmark_id]) {
      StereoPoint2 sp(ob.second(0), ob.second(2), ob.second(1));   // (uL, uR, v)
      sf->add(sp, X(ob.first), K_);
    }
    if (smart_idx_.count(o.landmark_id)) to_remove.push_back(smart_idx_[o.landmark_id]);
    smart_pos.push_back({o.landmark_id, new_factors.size()});
    new_factors.push_back(sf);
  }

  // --- VIO: velocity per keyframe + shared bias/gravity + IMU factor to the previous keyframe ---
  if (imu.active && imu.preint) {
    if (!vi_initialized_) {
      Vector6 bias0; bias0 << imu.gyr_bias, imu.acc_bias;
      new_values.insert(kBiasKey, bias0);
      new_values.insert(kGravityKey, GravityDir(imu.Rwg));
      Vector6 biasSig; biasSig << Vector3::Constant(0.1), Vector3::Constant(0.01);
      new_factors.addPrior(kBiasKey, bias0, noiseModel::Diagonal::Sigmas(biasSig));
      new_factors.emplace_shared<PriorFactor<GravityDir>>(kGravityKey, GravityDir(imu.Rwg), noiseModel::Isotropic::Sigma(2, 0.5));
      vi_initialized_ = true;
    }
    new_values.insert(V(kf_id), Vector3(imu.vel_init));
    // Regularize the velocity meaningfully: sigma 10 (info 0.01) was below iSAM2's numerical floor,
    // so a keyframe whose IMU factor didn't fully pin V(kf) went indeterminant (was v737). sigma 0.5
    // (info 4) holds it; the IMU factors still dominate where the preintegration is good.
    new_factors.addPrior(V(kf_id), Vector3(imu.vel_init), noiseModel::Isotropic::Sigma(3, 0.5));
    kf_with_velocity_.insert(kf_id);
    if (imu.prev_kf_id >= 0 && kf_with_velocity_.count(imu.prev_kf_id)) {
      SharedNoiseModel imuNoise;
      try { imuNoise = noiseModel::Gaussian::Covariance(imu.preint->Cov.block<9, 9>(0, 0)); }
      catch (const std::exception&) { imuNoise = SharedNoiseModel(); }
      if (imuNoise) {
        new_factors.emplace_shared<GtsamImuFactorVIO>(X(imu.prev_kf_id), V(imu.prev_kf_id), X(kf_id), V(kf_id),
                                                      kBiasKey, kGravityKey, imu.preint, imuNoise);
        n_imu_factors_++;
      } else {
        std::cout << "[iSAM2] IMU factor SKIPPED kf " << kf_id << " (non-PSD preint cov)\n";
      }
    } else {
      std::cout << "[iSAM2] IMU factor SKIPPED kf " << kf_id << " (prev " << imu.prev_kf_id
                << " not in velocity set)\n";
    }
  }

  ISAM2Result result = isam_.update(new_factors, new_values, to_remove);
  for (const auto& sp : smart_pos) smart_idx_[sp.first] = result.newFactorsIndices[sp.second];
  estimate_ = isam_.calculateEstimate();
  kf_ids_.push_back(kf_id);
}

gtsam::Pose3 IsamSmoother::GetBodyPose(int kf_id) const { return estimate_.at<Pose3>(X(kf_id)); }

Eigen::Matrix<double, 6, 6> IsamSmoother::GetCovariance(int kf_id) { return isam_.marginalCovariance(X(kf_id)); }

double IsamSmoother::PositionSigma(int kf_id) {
  return std::sqrt(isam_.marginalCovariance(X(kf_id)).block<3, 3>(3, 3).trace());
}
