#ifndef GTSAM_LINE_FACTOR_H_
#define GTSAM_LINE_FACTOR_H_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/numericalDerivative.h>

#include <g2o/types/slam3d_addons/line3d.h>
#include <g2o/types/slam3d/isometry3d_mappings.h>
#include "g2o_optimization/vertex_vi_pose.h"   // g2o::SE3Quat
#include "utils.h"                              // Vector8d

// Custom GTSAM factors for AirSLAM's 3D-line (Plücker) reprojection, mirroring EdgeSE3ProjectLine /
// EdgeStereoSE3ProjectLine. In FrameOptimization the lines are FIXED, so these are UNARY factors on
// the body pose Twb (camera pose Tcw = (Twb * Tbc)^-1). The Plücker transform + 2D-line projection
// reuse g2o::Line3D to stay bit-exact with g2o; the Jacobian is numeric (convergence-speed only).
//
// GTSAM has no line factor, so this is the "write a custom factor" piece that Phase 6 (semantic
// factors) builds on. When lines become free variables (GlobalBA / iSAM2) this generalises to a
// binary factor over (pose, line-manifold).

// project a camera-frame Plücker line to its 2D image line [a,b,c]
inline Eigen::Vector3d ProjectLine2D(const g2o::Line3D& line_cam, double fx, double fy,
                                     const Eigen::Vector3d& Kv) {
  Eigen::Vector3d w = line_cam.w();
  return Eigen::Vector3d(fy * w(0), fx * w(1), Kv.dot(w));
}

// camera pose Tcw (world->camera) as a g2o Isometry3, from the GTSAM body pose Twb
inline g2o::Isometry3 CameraTcwIso(const gtsam::Pose3& Twb, const gtsam::Pose3& Tbc) {
  gtsam::Pose3 Tcw = (Twb * Tbc).inverse();
  Eigen::Quaterniond q(Tcw.rotation().matrix());
  g2o::SE3Quat Tcw_se3(q, Eigen::Vector3d(Tcw.translation()));
  return g2o::internal::fromSE3Quat(Tcw_se3);
}

class GtsamMonoLineFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
  g2o::Line3D line_w_;         // fixed 3D line (world)
  Eigen::Vector4d obs_;        // observed 2D endpoints [x1,y1,x2,y2]
  double fx_, fy_;
  Eigen::Vector3d Kv_;         // [-cx*fy, -fx*cy, fx*fy]
  gtsam::Pose3 Tbc_;           // body_P_sensor (camera-in-body)

 public:
  GtsamMonoLineFactor(gtsam::Key poseKey, const g2o::Line3D& line_w, const Eigen::Vector4d& obs,
                      double fx, double fy, const Eigen::Vector3d& Kv, const gtsam::Pose3& Tbc,
                      const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, poseKey),
        line_w_(line_w), obs_(obs), fx_(fx), fy_(fy), Kv_(Kv), Tbc_(Tbc) {}

  gtsam::Vector2 residual(const gtsam::Pose3& Twb) const {
    g2o::Line3D lc = CameraTcwIso(Twb, Tbc_) * line_w_;
    Eigen::Vector3d l = ProjectLine2D(lc, fx_, fy_, Kv_);
    double n = l.head<2>().norm();
    gtsam::Vector2 e;
    e(0) = (obs_(0) * l(0) + obs_(1) * l(1) + l(2)) / n;
    e(1) = (obs_(2) * l(0) + obs_(3) * l(1) + l(2)) / n;
    return e;
  }

  gtsam::Vector evaluateError(const gtsam::Pose3& Twb,
                              boost::optional<gtsam::Matrix&> H = boost::none) const override {
    if (H) *H = gtsam::numericalDerivative11<gtsam::Vector2, gtsam::Pose3>(
        [this](const gtsam::Pose3& p) { return this->residual(p); }, Twb);
    return residual(Twb);
  }
};

class GtsamStereoLineFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
  g2o::Line3D line_w_;
  Vector8d obs_;               // [x1l,y1l,x2l,y2l, x1r,y1r,x2r,y2r]
  double fx_, fy_, b_;         // b = stereo baseline (metres)
  Eigen::Vector3d Kv_;
  gtsam::Pose3 Tbc_;

 public:
  GtsamStereoLineFactor(gtsam::Key poseKey, const g2o::Line3D& line_w, const Vector8d& obs,
                        double fx, double fy, double b, const Eigen::Vector3d& Kv,
                        const gtsam::Pose3& Tbc, const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, poseKey),
        line_w_(line_w), obs_(obs), fx_(fx), fy_(fy), b_(b), Kv_(Kv), Tbc_(Tbc) {}

  gtsam::Vector4 residual(const gtsam::Pose3& Twb) const {
    g2o::Isometry3 T_left = CameraTcwIso(Twb, Tbc_);
    Eigen::Vector3d ll = ProjectLine2D(T_left * line_w_, fx_, fy_, Kv_);
    double nl = ll.head<2>().norm();
    g2o::Isometry3 T_right = T_left;
    T_right(0, 3) -= b_;                       // right camera: shift x by baseline
    Eigen::Vector3d lr = ProjectLine2D(T_right * line_w_, fx_, fy_, Kv_);
    double nr = lr.head<2>().norm();
    gtsam::Vector4 e;
    e(0) = (obs_(0) * ll(0) + obs_(1) * ll(1) + ll(2)) / nl;
    e(1) = (obs_(2) * ll(0) + obs_(3) * ll(1) + ll(2)) / nl;
    e(2) = (obs_(4) * lr(0) + obs_(5) * lr(1) + lr(2)) / nr;
    e(3) = (obs_(6) * lr(0) + obs_(7) * lr(1) + lr(2)) / nr;
    return e;
  }

  gtsam::Vector evaluateError(const gtsam::Pose3& Twb,
                              boost::optional<gtsam::Matrix&> H = boost::none) const override {
    if (H) *H = gtsam::numericalDerivative11<gtsam::Vector4, gtsam::Pose3>(
        [this](const gtsam::Pose3& p) { return this->residual(p); }, Twb);
    return residual(Twb);
  }
};

#endif  // GTSAM_LINE_FACTOR_H_
