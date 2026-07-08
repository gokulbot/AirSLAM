#ifndef GTSAM_PLUCKER_LINE_H_
#define GTSAM_PLUCKER_LINE_H_

#include <iostream>
#include <Eigen/Core>

#include <gtsam/base/Manifold.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/numericalDerivative.h>

#include <g2o/types/slam3d_addons/line3d.h>
#include "gtsam_line_factor.h"   // ProjectLine2D, CameraTcwIso

// GTSAM manifold wrapping g2o::Line3D (Plücker 3D line, 4-DOF). retract = g2o's oplus and
// localCoordinates = g2o's ominus, so a free line optimizes exactly like g2o's VertexLine3D.
// This is what lets LocalmapOptimization / GlobalBA / iSAM2 carry FREE 3D lines.
class PluckerLine {
  g2o::Line3D line_;

 public:
  enum { dimension = 4 };
  typedef Eigen::Matrix<double, 4, 1> TangentVector;

  PluckerLine() {}
  explicit PluckerLine(const g2o::Line3D& l) : line_(l) {}
  const g2o::Line3D& g2o_line() const { return line_; }

  PluckerLine retract(const TangentVector& v) const {
    g2o::Line3D l = line_;
    l.oplus(v);
    return PluckerLine(l);
  }
  TangentVector localCoordinates(const PluckerLine& other) const {
    g2o::Line3D l = line_;
    return l.ominus(const_cast<g2o::Line3D&>(other.line_));
  }
  void print(const std::string& s = "") const {
    std::cout << s << "PluckerLine w=[" << line_.w().transpose() << "] d=[" << line_.d().transpose() << "]\n";
  }
  bool equals(const PluckerLine& o, double tol = 1e-8) const {
    return (static_cast<const Eigen::Matrix<double, 6, 1>&>(line_) -
            static_cast<const Eigen::Matrix<double, 6, 1>&>(o.line_)).norm() < tol;
  }
};

namespace gtsam {
template <>
struct traits<PluckerLine> : public internal::Manifold<PluckerLine> {};
}  // namespace gtsam

// Binary point reprojection factors for a FREE point (pose + Point3), for windowed BA / GlobalBA.
// Manual g2o cam_project (depth-agnostic, matching g2o's edge which projects even behind the camera).
class GtsamMonoPointBAFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
  gtsam::Point2 obs_;
  double fx_, fy_, cx_, cy_;
  gtsam::Pose3 Tbc_;

 public:
  GtsamMonoPointBAFactor(gtsam::Key poseKey, gtsam::Key pointKey, const gtsam::Point2& obs,
                         double fx, double fy, double cx, double cy, const gtsam::Pose3& Tbc,
                         const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>(model, poseKey, pointKey),
        obs_(obs), fx_(fx), fy_(fy), cx_(cx), cy_(cy), Tbc_(Tbc) {}

  gtsam::Vector2 residual(const gtsam::Pose3& Twb, const gtsam::Point3& Xw) const {
    gtsam::Point3 Xc = (Twb * Tbc_).inverse() * Xw;
    double zi = 1.0 / Xc.z();
    return gtsam::Vector2(Xc.x() * zi * fx_ + cx_ - obs_.x(), Xc.y() * zi * fy_ + cy_ - obs_.y());
  }
  gtsam::Vector evaluateError(const gtsam::Pose3& Twb, const gtsam::Point3& Xw,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
    if (!H1 && !H2) return residual(Twb, Xw);
    // Analytic Jacobians by chaining GTSAM primitives (each returns its Jacobian in GTSAM's tangent
    // convention): Twc = Twb*Tbc; Xc = Twc^-1*Xw; then a trivial projection Jacobian J = d(u,v)/dXc.
    Eigen::Matrix<double, 6, 6> Hc;
    Eigen::Matrix<double, 3, 6> Hp;
    Eigen::Matrix<double, 3, 3> Hx;
    gtsam::Pose3 Twc = Twb.compose(Tbc_, Hc);
    gtsam::Point3 Xc = Twc.transformTo(Xw, Hp, Hx);
    double zi = 1.0 / Xc.z(), zi2 = zi * zi;
    Eigen::Matrix<double, 2, 3> J;
    J << fx_ * zi, 0.0, -fx_ * Xc.x() * zi2,
         0.0, fy_ * zi, -fy_ * Xc.y() * zi2;
    if (H1) *H1 = J * Hp * Hc;
    if (H2) *H2 = J * Hx;
    return gtsam::Vector2(Xc.x() * zi * fx_ + cx_ - obs_.x(), Xc.y() * zi * fy_ + cy_ - obs_.y());
  }
};

class GtsamStereoPointBAFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
  gtsam::Vector3 obs_;   // (u, v, u_right)
  double fx_, fy_, cx_, cy_, bf_;
  gtsam::Pose3 Tbc_;

 public:
  GtsamStereoPointBAFactor(gtsam::Key poseKey, gtsam::Key pointKey, const gtsam::Vector3& obs,
                           double fx, double fy, double cx, double cy, double bf, const gtsam::Pose3& Tbc,
                           const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>(model, poseKey, pointKey),
        obs_(obs), fx_(fx), fy_(fy), cx_(cx), cy_(cy), bf_(bf), Tbc_(Tbc) {}

  gtsam::Vector3 residual(const gtsam::Pose3& Twb, const gtsam::Point3& Xw) const {
    gtsam::Point3 Xc = (Twb * Tbc_).inverse() * Xw;
    double zi = 1.0 / Xc.z();
    double u = Xc.x() * zi * fx_ + cx_, v = Xc.y() * zi * fy_ + cy_;
    return gtsam::Vector3(u - obs_(0), v - obs_(1), (u - bf_ * zi) - obs_(2));
  }
  gtsam::Vector evaluateError(const gtsam::Pose3& Twb, const gtsam::Point3& Xw,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
    if (!H1 && !H2) return residual(Twb, Xw);
    // Analytic Jacobians: Twc = Twb*Tbc; Xc = Twc^-1*Xw (GTSAM primitives supply d/dTwb, d/dXw);
    // J = d(u,v,uR)/dXc. u_right = u - bf/z adds +bf/z^2 to the z-column of the 3rd row.
    Eigen::Matrix<double, 6, 6> Hc;
    Eigen::Matrix<double, 3, 6> Hp;
    Eigen::Matrix<double, 3, 3> Hx;
    gtsam::Pose3 Twc = Twb.compose(Tbc_, Hc);
    gtsam::Point3 Xc = Twc.transformTo(Xw, Hp, Hx);
    double zi = 1.0 / Xc.z(), zi2 = zi * zi;
    double u = Xc.x() * zi * fx_ + cx_, v = Xc.y() * zi * fy_ + cy_;
    Eigen::Matrix<double, 3, 3> J;
    J << fx_ * zi, 0.0, -fx_ * Xc.x() * zi2,
         0.0, fy_ * zi, -fy_ * Xc.y() * zi2,
         fx_ * zi, 0.0, -fx_ * Xc.x() * zi2 + bf_ * zi2;
    if (H1) *H1 = J * Hp * Hc;
    if (H2) *H2 = J * Hx;
    return gtsam::Vector3(u - obs_(0), v - obs_(1), (u - bf_ * zi) - obs_(2));
  }
};

// Binary line reprojection factors for a FREE line (pose + PluckerLine), for windowed BA / GlobalBA.
// Same projection as the unary factors; Jacobians are numeric (wrt both pose and line manifold).
class GtsamMonoLineBAFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, PluckerLine> {
  Eigen::Vector4d obs_;
  double fx_, fy_;
  Eigen::Vector3d Kv_;
  gtsam::Pose3 Tbc_;

 public:
  GtsamMonoLineBAFactor(gtsam::Key poseKey, gtsam::Key lineKey, const Eigen::Vector4d& obs,
                        double fx, double fy, const Eigen::Vector3d& Kv, const gtsam::Pose3& Tbc,
                        const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, PluckerLine>(model, poseKey, lineKey),
        obs_(obs), fx_(fx), fy_(fy), Kv_(Kv), Tbc_(Tbc) {}

  gtsam::Vector2 residual(const gtsam::Pose3& Twb, const PluckerLine& line) const {
    g2o::Line3D lc = CameraTcwIso(Twb, Tbc_) * line.g2o_line();
    Eigen::Vector3d l = ProjectLine2D(lc, fx_, fy_, Kv_);
    double n = l.head<2>().norm();
    gtsam::Vector2 e;
    e(0) = (obs_(0) * l(0) + obs_(1) * l(1) + l(2)) / n;
    e(1) = (obs_(2) * l(0) + obs_(3) * l(1) + l(2)) / n;
    return e;
  }
  gtsam::Vector evaluateError(const gtsam::Pose3& Twb, const PluckerLine& line,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
    auto f = [this](const gtsam::Pose3& p, const PluckerLine& l) { return this->residual(p, l); };
    if (H1) *H1 = gtsam::numericalDerivative21<gtsam::Vector2, gtsam::Pose3, PluckerLine>(f, Twb, line);
    if (H2) *H2 = gtsam::numericalDerivative22<gtsam::Vector2, gtsam::Pose3, PluckerLine>(f, Twb, line);
    return residual(Twb, line);
  }
};

class GtsamStereoLineBAFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, PluckerLine> {
  Vector8d obs_;
  double fx_, fy_, b_;
  Eigen::Vector3d Kv_;
  gtsam::Pose3 Tbc_;

 public:
  GtsamStereoLineBAFactor(gtsam::Key poseKey, gtsam::Key lineKey, const Vector8d& obs,
                          double fx, double fy, double b, const Eigen::Vector3d& Kv,
                          const gtsam::Pose3& Tbc, const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, PluckerLine>(model, poseKey, lineKey),
        obs_(obs), fx_(fx), fy_(fy), b_(b), Kv_(Kv), Tbc_(Tbc) {}

  gtsam::Vector4 residual(const gtsam::Pose3& Twb, const PluckerLine& line) const {
    g2o::Isometry3 T_left = CameraTcwIso(Twb, Tbc_);
    Eigen::Vector3d ll = ProjectLine2D(T_left * line.g2o_line(), fx_, fy_, Kv_);
    double nl = ll.head<2>().norm();
    g2o::Isometry3 T_right = T_left;
    T_right(0, 3) -= b_;
    Eigen::Vector3d lr = ProjectLine2D(T_right * line.g2o_line(), fx_, fy_, Kv_);
    double nr = lr.head<2>().norm();
    gtsam::Vector4 e;
    e(0) = (obs_(0) * ll(0) + obs_(1) * ll(1) + ll(2)) / nl;
    e(1) = (obs_(2) * ll(0) + obs_(3) * ll(1) + ll(2)) / nl;
    e(2) = (obs_(4) * lr(0) + obs_(5) * lr(1) + lr(2)) / nr;
    e(3) = (obs_(6) * lr(0) + obs_(7) * lr(1) + lr(2)) / nr;
    return e;
  }
  gtsam::Vector evaluateError(const gtsam::Pose3& Twb, const PluckerLine& line,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
    auto f = [this](const gtsam::Pose3& p, const PluckerLine& l) { return this->residual(p, l); };
    if (H1) *H1 = gtsam::numericalDerivative21<gtsam::Vector4, gtsam::Pose3, PluckerLine>(f, Twb, line);
    if (H2) *H2 = gtsam::numericalDerivative22<gtsam::Vector4, gtsam::Pose3, PluckerLine>(f, Twb, line);
    return residual(Twb, line);
  }
};

#endif  // GTSAM_PLUCKER_LINE_H_
