#ifndef MAPPOINT_H_
#define MAPPOINT_H_

#include <limits>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/version.hpp>

class Mappoint{
public:
  enum Type {
    UnTriangulated = 0,
    Good = 1,
    Bad = 2,
  };

  Mappoint();
  Mappoint(int& mappoint_id);
  Mappoint(int& mappoint_id, Eigen::Vector3d& p);
  Mappoint(int& mappoint_id, Eigen::Vector3d& p, Eigen::Matrix<float, 256, 1>& d);
  void SetId(int id);
  int GetId();
  void SetType(const Type& type);
  Type GetType();
  void SetBad();
  bool IsBad();
  void SetGood();
  bool IsValid();

  void SetPosition(const Eigen::Vector3d& p);
  Eigen::Vector3d GetPosition();
  void SetDescriptor(const Eigen::Matrix<float, 256, 1>& descriptor);
  Eigen::Matrix<float, 256, 1>& GetDescriptor(); 

  void AddObverser(const int& frame_id, const int& keypoint_index);
  void RemoveObverser(const int& frame_id);
  int ObverserNum();
  std::map<int, int>& GetAllObversers();
  int GetKeypointIdx(int frame_id);

  // Dynamic-object evidence: fraction of this point's observations that landed on a dynamic-class
  // (e.g. person) mask. One score, two consumers: soft BA down-weighting (w = 1 - DynamicScore) and
  // the hard prune for the nav map (drop if DynamicScore > tau). Later augmented with motion evidence.
  void AddDynamicObservation(bool on_dynamic) { _n_obs_dyn++; if (on_dynamic) _n_dyn++; }
  float DynamicScore() const { return _n_obs_dyn ? static_cast<float>(_n_dyn) / _n_obs_dyn : 0.0f; }

public:
  int tracking_frame_id;
  int last_frame_seen;
  int local_map_optimization_frame_id;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version){
    ar & _id;
    ar & _type;
    ar & boost::serialization::make_array(_position.data(), _position.size());
    ar & _obversers;
    // ar & boost::serialization::make_array(_descriptor.data(), _descriptor.size());
    // v1+: dynamic-object evidence (defaults to 0 on older maps)
    if (version >= 1) {
      ar & _n_obs_dyn;
      ar & _n_dyn;
    }
  }

private:
  int _id;
  Type _type;
  Eigen::Vector3d _position;
  Eigen::Matrix<float, 256, 1> _descriptor;
  std::map<int, int> _obversers;  // frame_id - keypoint_index
  int _n_obs_dyn = 0;             // observations counted for dynamic evidence
  int _n_dyn = 0;                 // ...of which landed on a dynamic-class mask
};

BOOST_CLASS_VERSION(Mappoint, 1)

typedef std::shared_ptr<Mappoint> MappointPtr;

#endif  // MAPPOINT_H