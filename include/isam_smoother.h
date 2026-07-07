#ifndef ISAM_SMOOTHER_H_
#define ISAM_SMOOTHER_H_

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <Eigen/Core>

#include <gtsam/geometry/Pose3.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include "imu.h"   // PreinterationPtr

// One stereo-point observation from a keyframe (u, v, u_right); Xw_init is the landmark's initial
// 3D position, used only the first time the landmark is seen.
struct StereoObservation {
  int landmark_id;
  Eigen::Vector3d Xw_init;
  Eigen::Vector3d keypoint;
};

// Optional per-keyframe IMU data (VIO). When active, the smoother adds a velocity variable for this
// keyframe and an IMU factor linking it to prev_kf_id via the preintegration; a shared bias +
// gravity are created on the first VI keyframe. Empty (active=false) => vision-only keyframe.
struct ImuKeyframeData {
  bool active = false;
  int prev_kf_id = -1;
  Eigen::Vector3d vel_init = Eigen::Vector3d::Zero();
  PreinterationPtr preint = nullptr;
  Eigen::Matrix3d Rwg = Eigen::Matrix3d::Identity();
  Eigen::Vector3d gyr_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();
};

// Incremental iSAM2 smoother for the online pipeline. Fed one keyframe at a time as VO produces
// them, it maintains the global factor graph via the Bayes tree (relinearizing only what changed)
// and exposes live per-keyframe marginal covariances -- the uncertainty signal that active /
// uncertainty-aware navigation consumes. Runs ALONGSIDE the existing sliding-window BA; it is the
// substrate for replacing the online-window + offline-global split with one online smoother.
//
// Points-only for now (the reprojection factors are the validated GtsamStereoPointBAFactor); lines
// and IMU factors slot in the same way. Growth is unbounded here (fine for a demo / short runs);
// a fixed-lag / marginalization policy is the production follow-up.
class IsamSmoother {
 public:
  IsamSmoother(const gtsam::Pose3& body_P_camera, double fx, double fy, double cx, double cy, double bf);

  // Add a keyframe: its body pose Twb (initial guess) + its stereo observations. The first keyframe
  // should be the anchor (adds a pose prior to fix the gauge).
  void AddKeyframe(int kf_id, const gtsam::Pose3& Twb_init, bool anchor,
                   const std::vector<StereoObservation>& obs, const ImuKeyframeData& imu = ImuKeyframeData());

  gtsam::Pose3 GetBodyPose(int kf_id) const;
  Eigen::Matrix<double, 6, 6> GetCovariance(int kf_id);
  double PositionSigma(int kf_id);         // sqrt(trace of translation covariance), metres
  int NumKeyframes() const { return static_cast<int>(kf_ids_.size()); }
  bool IsVisualInertial() const { return vi_initialized_; }
  const std::vector<int>& KeyframeIds() const { return kf_ids_; }

 private:
  std::shared_ptr<gtsam::IncrementalFixedLagSmoother> smoother_;   // iSAM2 + fixed-lag marginalization
  gtsam::Values estimate_;
  gtsam::Pose3 Tbc_;
  double fx_, fy_, cx_, cy_, bf_;
  double time_ = 0.0;                // keyframe counter, used as the fixed-lag "time"
  // fix #1 (like g2o's mature map points): a landmark enters the smoother only once >=2 keyframes
  // observe it (so it genuinely connects poses); no loose-prior crutch.
  std::map<int, std::vector<std::pair<int, Eigen::Vector3d>>> pending_;   // immature: lm -> [(kf, keypoint)]
  std::set<int> mature_;             // landmarks currently in the smoother
  std::vector<int> kf_ids_;
  bool vi_initialized_ = false;      // shared bias + gravity added?
  std::set<int> kf_with_velocity_;   // keyframes that have a velocity variable (for IMU-factor linking)
};
typedef std::shared_ptr<IsamSmoother> IsamSmootherPtr;

#endif  // ISAM_SMOOTHER_H_
