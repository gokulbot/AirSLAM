// Numerical parity test: GTSAM reprojection factors vs AirSLAM's g2o edges.
// Validates the point factors (mono + stereo) that FrameOptimization / GlobalBA / iSAM2 will use.
// Convention: g2o optimizes the BODY pose Twb, camera pose Tcw derived via extrinsic Tcb.
// GTSAM equivalent: GenericProjectionFactor with body_P_sensor = Tbc (= Tcb^-1), pose var = Twb.
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/make_shared.hpp>

#include "g2o_optimization/edge_project_point.h"
#include "g2o_optimization/edge_project_line.h"
#include "g2o_optimization/vertex_vi_pose.h"
#include "g2o_optimization/vertex_line3d.h"
#include "gtsam_line_factor.h"
#include <g2o/types/slam3d/vertex_pointxyz.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/StereoFactor.h>

using namespace gtsam;

int main() {
  // EuRoC-like intrinsics
  const double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
  const double baseline = 0.11, bf = fx * baseline;

  // camera pose world->camera (Tcw)
  Eigen::Matrix3d Rcw = (Eigen::AngleAxisd(0.15, Eigen::Vector3d(0.2, 1.0, 0.1).normalized())).toRotationMatrix();
  Eigen::Vector3d tcw(0.05, -0.02, 0.30);
  // body->camera extrinsic (non-trivial, to exercise body_P_sensor)
  Eigen::Matrix3d Rcb = (Eigen::AngleAxisd(-0.05, Eigen::Vector3d::UnitZ())).toRotationMatrix();
  Eigen::Vector3d tcb(0.02, 0.0, -0.01);
  // 3D world point in front of the camera
  Eigen::Vector3d Xw(0.4, 0.1, 2.0);
  Eigen::Vector3d Xc = Rcw * Xw + tcw;
  // measurements = true projection + a small residual, so the errors are non-trivial but finite
  Eigen::Vector2d kp(Xc(0) / Xc(2) * fx + cx + 0.7, Xc(1) / Xc(2) * fy + cy - 0.5);
  Eigen::Vector3d kp3(kp(0), kp(1), kp(0) - bf / Xc(2) + 0.3);  // (u, v, u_right)

  // ---------- g2o edges (ground truth) ----------
  g2o::VertexPointXYZ vp; vp.setEstimate(Xw); vp.setId(0);
  VertexVIPose vpose; vpose.setEstimate(VIPose(Rcw, tcw, Rcb, tcb)); vpose.setId(1);

  EdgeSE3ProjectPoint em; em.fx = fx; em.fy = fy; em.cx = cx; em.cy = cy;
  em.setVertex(0, &vp); em.setVertex(1, &vpose); em.setMeasurement(kp); em.computeError();
  Eigen::Vector2d err_g2o_mono = em.error();               // (eu, ev)

  EdgeSE3ProjectStereoPoint es; es.fx = fx; es.fy = fy; es.cx = cx; es.cy = cy; es.bf = bf;
  es.setVertex(0, &vp); es.setVertex(1, &vpose); es.setMeasurement(kp3); es.computeError();
  Eigen::Vector3d err_g2o_stereo = es.error();             // (eu, ev, euR)

  // ---------- GTSAM factors ----------
  Pose3 Tcw(Rot3(Rcw), tcw);
  Pose3 Twc = Tcw.inverse();
  Pose3 Tcb(Rot3(Rcb), tcb);              // BodyToCamera
  Pose3 Tbc = Tcb.inverse();              // body_P_sensor (camera-in-body)
  Pose3 Twb = Twc * Tcb;                  // pose variable; Twb * Tbc = Twc
  const Key kPose = 1, kPoint = 0;

  auto Kmono = boost::make_shared<Cal3_S2>(fx, fy, 0.0, cx, cy);
  GenericProjectionFactor<Pose3, Point3, Cal3_S2> pf(
      Point2(kp), noiseModel::Isotropic::Sigma(2, 1.0), kPose, kPoint, Kmono, Tbc);
  Vector err_gtsam_mono = pf.evaluateError(Twb, Point3(Xw));  // predicted - measured (opposite sign to g2o)

  auto Kstereo = boost::make_shared<Cal3_S2Stereo>(fx, fy, 0.0, cx, cy, baseline);
  GenericStereoFactor<Pose3, Point3> sf(
      StereoPoint2(kp3(0), kp3(2), kp3(1)), noiseModel::Isotropic::Sigma(3, 1.0), kPose, kPoint, Kstereo, Tbc);
  Vector err_gtsam_stereo = sf.evaluateError(Twb, Point3(Xw));  // order (uL, uR, v)

  // ---------- line factors (fixed Plucker line, unary on pose) ----------
  Eigen::Vector3d Kv(-cx * fy, -fx * cy, fx * fy);
  Eigen::Vector3d LP0(0.1, 0.2, 2.5), LP1(0.6, -0.1, 2.8);        // world line through two points
  Eigen::Vector3d Ld = LP1 - LP0, Lw = LP0.cross(Ld);
  Eigen::Matrix<double, 6, 1> pluecker; pluecker.head<3>() = Lw; pluecker.tail<3>() = Ld;
  g2o::Line3D line_w(pluecker);
  Eigen::Vector4d obs_line(360.0, 250.0, 380.0, 300.0);          // observed 2D endpoints
  Vector8d obs_line_st; obs_line_st << 360, 250, 380, 300, 350, 250, 370, 300;

  g2o::VertexLine3D vl; vl.setEstimate(line_w); vl.setId(2);
  EdgeSE3ProjectLine el; el.fx = fx; el.fy = fy; el.Kv = Kv;
  el.setVertex(0, &vl); el.setVertex(1, &vpose); el.setMeasurement(obs_line); el.computeError();
  Eigen::Vector2d err_g2o_lmono = el.error();
  GtsamMonoLineFactor lf(kPose, line_w, obs_line, fx, fy, Kv, Tbc, noiseModel::Isotropic::Sigma(2, 1.0));
  Vector2 err_gtsam_lmono = lf.residual(Twb);

  EdgeStereoSE3ProjectLine els; els.fx = fx; els.fy = fy; els.b = baseline; els.Kv = Kv;
  els.setVertex(0, &vl); els.setVertex(1, &vpose); els.setMeasurement(obs_line_st); els.computeError();
  Eigen::Vector4d err_g2o_lstereo = els.error();
  GtsamStereoLineFactor lfs(kPose, line_w, obs_line_st, fx, fy, baseline, Kv, Tbc, noiseModel::Isotropic::Sigma(4, 1.0));
  Vector4 err_gtsam_lstereo = lfs.residual(Twb);

  // ---------- compare (points: g2o err = -gtsam err; lines: same sign) ----------
  double d_mono = (err_g2o_mono + err_gtsam_mono).norm();
  Eigen::Vector3d g2o_stereo_reord(err_g2o_stereo(0), err_g2o_stereo(2), err_g2o_stereo(1));  // ->(uL,uR,v)
  double d_stereo = (g2o_stereo_reord + err_gtsam_stereo).norm();
  double d_lmono = (err_g2o_lmono - err_gtsam_lmono).norm();
  double d_lstereo = (err_g2o_lstereo - err_gtsam_lstereo).norm();

  std::cout << std::fixed << std::setprecision(8);
  std::cout << "MONO POINT    g2o=[" << err_g2o_mono.transpose() << "]  gtsam=[" << err_gtsam_mono.transpose()
            << "]  residual=" << d_mono << "\n";
  std::cout << "STEREO POINT  g2o=[" << g2o_stereo_reord.transpose() << "]  gtsam=[" << err_gtsam_stereo.transpose()
            << "]  residual=" << d_stereo << "\n";
  std::cout << "MONO LINE     g2o=[" << err_g2o_lmono.transpose() << "]  gtsam=[" << err_gtsam_lmono.transpose()
            << "]  residual=" << d_lmono << "\n";
  std::cout << "STEREO LINE   g2o=[" << err_g2o_lstereo.transpose() << "]  gtsam=[" << err_gtsam_lstereo.transpose()
            << "]  residual=" << d_lstereo << "\n";

  bool ok = d_mono < 1e-6 && d_stereo < 1e-6 && d_lmono < 1e-6 && d_lstereo < 1e-6;
  std::cout << (ok ? "PASS: GTSAM point + line factors match g2o edges\n" : "FAIL: mismatch\n");
  return ok ? 0 : 1;
}
