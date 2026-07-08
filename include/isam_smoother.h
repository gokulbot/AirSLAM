#ifndef ISAM_SMOOTHER_H_
#define ISAM_SMOOTHER_H_

#include <map>
#include <memory>
#include <set>
#include <vector>
#include <Eigen/Core>

#include <boost/shared_ptr.hpp>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

#include "imu.h"   // PreinterationPtr

// One stereo-point observation from a keyframe (u, v, u_right).
struct StereoObservation {
  int landmark_id;
  Eigen::Vector3d Xw_init;      // (unused by smart factors, kept for API compatibility)
  Eigen::Vector3d keypoint;
};

// Optional per-keyframe IMU data (VIO). When active, the smoother adds a velocity variable and an
// IMU factor to prev_kf_id; a shared bias + gravity are created on the first VI keyframe.
struct ImuKeyframeData {
  bool active = false;
  int prev_kf_id = -1;
  Eigen::Vector3d vel_init = Eigen::Vector3d::Zero();
  PreinterationPtr preint = nullptr;
  Eigen::Matrix3d Rwg = Eigen::Matrix3d::Identity();
  Eigen::Vector3d gyr_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();
};

// Incremental iSAM2 VIO smoother using STRUCTURELESS smart stereo factors: landmarks stay FREE but
// are marginalized on-the-fly inside the factor (Schur) -- no explicit landmark variables, so a
// landmark can never become an ill-conditioned variable at a window/marginalization boundary.
// Degenerate landmarks (too few views / small baseline) contribute zero (ZERO_ON_DEGENERACY) rather
// than throwing. This is the standard robust fixed-lag-VIO representation. The online job is the
// live pose estimate + marginal covariances; GlobalBA owns the global map.
class IsamSmoother {
 public:
  IsamSmoother(const gtsam::Pose3& body_P_camera, double fx, double fy, double cx, double cy, double bf);

  void AddKeyframe(int kf_id, const gtsam::Pose3& Twb_init, bool anchor,
                   const std::vector<StereoObservation>& obs, const ImuKeyframeData& imu = ImuKeyframeData());

  gtsam::Pose3 GetBodyPose(int kf_id) const;
  Eigen::Matrix<double, 6, 6> GetCovariance(int kf_id);
  double PositionSigma(int kf_id);         // sqrt(trace of translation covariance), metres
  int NumKeyframes() const { return static_cast<int>(kf_ids_.size()); }
  bool IsVisualInertial() const { return vi_initialized_; }

 private:
  typedef gtsam::SmartStereoProjectionPoseFactor SmartFactor;

  gtsam::ISAM2 isam_;
  gtsam::Values estimate_;
  gtsam::Pose3 Tbc_;
  boost::shared_ptr<gtsam::Cal3_S2Stereo> K_;      // shared stereo calibration
  gtsam::SmartStereoProjectionParams sparams_;
  std::map<int, std::vector<std::pair<int, Eigen::Vector3d>>> obs_;   // landmark -> [(kf_id, keypoint)]
  std::map<int, size_t> smart_idx_;                                   // landmark -> current iSAM2 factor index
  std::vector<int> kf_ids_;
  bool vi_initialized_ = false;
  std::set<int> kf_with_velocity_;
};
typedef std::shared_ptr<IsamSmoother> IsamSmootherPtr;

#endif  // ISAM_SMOOTHER_H_
