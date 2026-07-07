#ifndef ISAM_SMOOTHER_H_
#define ISAM_SMOOTHER_H_

#include <memory>
#include <set>
#include <vector>
#include <Eigen/Core>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>

// One stereo-point observation from a keyframe (u, v, u_right); Xw_init is the landmark's initial
// 3D position, used only the first time the landmark is seen.
struct StereoObservation {
  int landmark_id;
  Eigen::Vector3d Xw_init;
  Eigen::Vector3d keypoint;
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
                   const std::vector<StereoObservation>& obs);

  gtsam::Pose3 GetBodyPose(int kf_id) const;
  Eigen::Matrix<double, 6, 6> GetCovariance(int kf_id);
  double PositionSigma(int kf_id);         // sqrt(trace of translation covariance), metres
  int NumKeyframes() const { return static_cast<int>(kf_ids_.size()); }
  const std::vector<int>& KeyframeIds() const { return kf_ids_; }

 private:
  gtsam::ISAM2 isam_;
  gtsam::Values estimate_;
  gtsam::Pose3 Tbc_;
  double fx_, fy_, cx_, cy_, bf_;
  std::set<int> known_landmarks_;
  std::vector<int> kf_ids_;
};
typedef std::shared_ptr<IsamSmoother> IsamSmootherPtr;

#endif  // ISAM_SMOOTHER_H_
