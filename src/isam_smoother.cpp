#include "isam_smoother.h"

#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "gtsam_plucker_line.h"   // GtsamStereoPointBAFactor

using namespace gtsam;
using symbol_shorthand::P;
using symbol_shorthand::X;

IsamSmoother::IsamSmoother(const gtsam::Pose3& body_P_camera, double fx, double fy, double cx, double cy, double bf)
    : Tbc_(body_P_camera), fx_(fx), fy_(fy), cx_(cx), cy_(cy), bf_(bf) {}

void IsamSmoother::AddKeyframe(int kf_id, const gtsam::Pose3& Twb_init, bool anchor,
                               const std::vector<StereoObservation>& obs) {
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
  isam_.update(graph, new_values);      // incremental Bayes-tree update
  estimate_ = isam_.calculateEstimate();
  kf_ids_.push_back(kf_id);
}

gtsam::Pose3 IsamSmoother::GetBodyPose(int kf_id) const { return estimate_.at<Pose3>(X(kf_id)); }

Eigen::Matrix<double, 6, 6> IsamSmoother::GetCovariance(int kf_id) { return isam_.marginalCovariance(X(kf_id)); }

double IsamSmoother::PositionSigma(int kf_id) {
  return std::sqrt(isam_.marginalCovariance(X(kf_id)).block<3, 3>(3, 3).trace());
}
