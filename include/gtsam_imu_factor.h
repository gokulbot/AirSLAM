#ifndef GTSAM_IMU_FACTOR_H_
#define GTSAM_IMU_FACTOR_H_

#include <iostream>
#include <Eigen/Core>

#include <gtsam/base/Manifold.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/numericalDerivative.h>

#include "imu.h"       // SO3Exp, SO3Log, Preinteration/PreinterationPtr
#include "camera.h"    // Camera::IMU_G_VALUE

// 2-DOF gravity-direction manifold wrapping AirSLAM's GDirection: retract([u0,u1]) = Rwg*Exp([u0,u1,0]),
// so the gravity direction optimizes exactly like g2o's VertexGDirection (keeps |g| fixed, 2 DOF).
class GravityDir {
  Eigen::Matrix3d Rwg_;
 public:
  enum { dimension = 2 };
  typedef Eigen::Matrix<double, 2, 1> TangentVector;
  GravityDir() { Rwg_.setIdentity(); }
  explicit GravityDir(const Eigen::Matrix3d& R) : Rwg_(R) {}
  const Eigen::Matrix3d& Rwg() const { return Rwg_; }

  GravityDir retract(const TangentVector& v) const {
    Eigen::Vector3d u(v(0), v(1), 0.0); Eigen::Matrix3d dR; SO3Exp(u, dR);
    return GravityDir(Rwg_ * dR);
  }
  TangentVector localCoordinates(const GravityDir& o) const {
    Eigen::Vector3d w; SO3Log(Rwg_.transpose() * o.Rwg_, w); return TangentVector(w(0), w(1));
  }
  void print(const std::string& s = "") const { std::cout << s << "GravityDir Rwg=\n" << Rwg_ << "\n"; }
  bool equals(const GravityDir& o, double tol = 1e-8) const { return (Rwg_ - o.Rwg_).norm() < tol; }
};

namespace gtsam {
template <> struct traits<GravityDir> : public internal::Manifold<GravityDir> {};
}  // namespace gtsam

// IMU preintegration factor (AirSLAM's EdgeIMU). Poses are fixed (from vision) so they are baked in;
// variables are vel1, gyr_bias, acc_bias, vel2, gravity. 9-dim residual [er, ev, ep]; numeric Jacobians.
class GtsamImuFactor
    : public gtsam::NoiseModelFactor5<gtsam::Vector3, gtsam::Vector3, gtsam::Vector3, gtsam::Vector3, GravityDir> {
  PreinterationPtr preint_;
  Eigen::Matrix3d Rwb1_, Rwb2_;
  Eigen::Vector3d twb1_, twb2_;

 public:
  GtsamImuFactor(gtsam::Key v1, gtsam::Key gb, gtsam::Key ab, gtsam::Key v2, gtsam::Key g,
                 PreinterationPtr preint, const Eigen::Matrix3d& Rwb1, const Eigen::Vector3d& twb1,
                 const Eigen::Matrix3d& Rwb2, const Eigen::Vector3d& twb2, const gtsam::SharedNoiseModel& model)
      : gtsam::NoiseModelFactor5<gtsam::Vector3, gtsam::Vector3, gtsam::Vector3, gtsam::Vector3, GravityDir>(
            model, v1, gb, ab, v2, g),
        preint_(preint), Rwb1_(Rwb1), Rwb2_(Rwb2), twb1_(twb1), twb2_(twb2) {}

  gtsam::Vector9 residual(const gtsam::Vector3& vel1, const gtsam::Vector3& gb, const gtsam::Vector3& ab,
                          const gtsam::Vector3& vel2, const GravityDir& grav) const {
    Eigen::Matrix3d dR = preint_->GetDeltaRotation(gb);
    Eigen::Vector3d dV = preint_->GetDeltaVelocity(gb, ab);
    Eigen::Vector3d dP = preint_->GetDeltaPosition(gb, ab);
    Eigen::Vector3d g(0, 0, -Camera::IMU_G_VALUE);
    Eigen::Vector3d gw = grav.Rwg() * g;
    double dt = preint_->dT;
    Eigen::Vector3d er; SO3Log(dR.transpose() * Rwb1_.transpose() * Rwb2_, er);
    Eigen::Vector3d ev = Rwb1_.transpose() * (vel2 - vel1 - gw * dt) - dV;
    Eigen::Vector3d ep = Rwb1_.transpose() * (twb2_ - twb1_ - vel1 * dt - gw * dt * dt / 2) - dP;
    gtsam::Vector9 e; e << er, ev, ep; return e;
  }

  gtsam::Vector evaluateError(const gtsam::Vector3& v1, const gtsam::Vector3& gb, const gtsam::Vector3& ab,
                              const gtsam::Vector3& v2, const GravityDir& g,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none,
                              boost::optional<gtsam::Matrix&> H3 = boost::none,
                              boost::optional<gtsam::Matrix&> H4 = boost::none,
                              boost::optional<gtsam::Matrix&> H5 = boost::none) const override {
    if (H1) *H1 = gtsam::numericalDerivative11<gtsam::Vector9, gtsam::Vector3>([&](const gtsam::Vector3& x) { return residual(x, gb, ab, v2, g); }, v1);
    if (H2) *H2 = gtsam::numericalDerivative11<gtsam::Vector9, gtsam::Vector3>([&](const gtsam::Vector3& x) { return residual(v1, x, ab, v2, g); }, gb);
    if (H3) *H3 = gtsam::numericalDerivative11<gtsam::Vector9, gtsam::Vector3>([&](const gtsam::Vector3& x) { return residual(v1, gb, x, v2, g); }, ab);
    if (H4) *H4 = gtsam::numericalDerivative11<gtsam::Vector9, gtsam::Vector3>([&](const gtsam::Vector3& x) { return residual(v1, gb, ab, x, g); }, v2);
    if (H5) *H5 = gtsam::numericalDerivative11<gtsam::Vector9, GravityDir>([&](const GravityDir& x) { return residual(v1, gb, ab, v2, x); }, g);
    return residual(v1, gb, ab, v2, g);
  }
};

#endif  // GTSAM_IMU_FACTOR_H_
