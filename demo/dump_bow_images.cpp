// Run AirSLAM's SuperPoint + DBoW2 on two folders of images (map, query) and dump
// the cross BoW-similarity matrix (query x map). For comparing DBoW2 vs AnyLoc VPR
// on datasets that aren't AirSLAM maps (e.g. Gardens Point day/night).
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>

#include "read_configs.h"
#include "plnet.h"
#include "feature_detector.h"
#include "bow/database.h"
#include "utils.h"

static void bow_folder(const std::string& dir, FeatureDetectorPtr fd, DatabasePtr db,
                       std::vector<DBoW2::BowVector>& bows) {
  std::vector<std::string> names;
  GetFileNames(dir, names);
  std::sort(names.begin(), names.end());
  for (const auto& n : names) {
    cv::Mat img = cv::imread(ConcatenateFolderAndFileName(dir, n), 0);
    if (img.empty()) continue;
    Eigen::Matrix<float, 259, Eigen::Dynamic> feat;
    fd->Detect(img, feat);
    DBoW2::WordIdToFeatures wf;
    DBoW2::BowVector bv;
    db->FrameToBow(feat, wf, bv);
    bows.push_back(bv);
  }
  std::cout << dir << " -> " << bows.size() << " images" << std::endl;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "air_slam");
  std::string model_dir, map_dir, query_dir, voc_path, out_path;
  ros::param::get("~model_dir", model_dir);
  ros::param::get("~map_dir", map_dir);
  ros::param::get("~query_dir", query_dir);
  ros::param::get("~voc_path", voc_path);
  ros::param::get("~out_path", out_path);

  PLNetConfig plnet_config;
  plnet_config.use_superpoint = 1;
  plnet_config.max_keypoints = 400;
  plnet_config.keypoint_threshold = 0.04;
  plnet_config.remove_borders = 4;
  plnet_config.line_threshold = 0.5;
  plnet_config.line_length_threshold = 50;
  plnet_config.SetModelPath(model_dir);

  FeatureDetectorPtr fd = std::make_shared<FeatureDetector>(plnet_config);
  DatabasePtr db = std::make_shared<Database>(voc_path);

  std::vector<DBoW2::BowVector> mapb, qb;
  bow_folder(map_dir, fd, db, mapb);
  bow_folder(query_dir, fd, db, qb);

  std::ofstream ofs(out_path);
  ofs << qb.size() << " " << mapb.size() << "\n";              // nQuery nMap
  for (size_t i = 0; i < qb.size(); i++)
    for (size_t j = 0; j < mapb.size(); j++)
      ofs << std::setprecision(6) << db->Score(qb[i], mapb[j]) << (j + 1 < mapb.size() ? " " : "\n");
  ofs.close();
  std::cout << "dumped " << qb.size() << "x" << mapb.size() << " DBoW2 cross matrix to " << out_path << std::endl;
  exit(0);
  return 0;
}
