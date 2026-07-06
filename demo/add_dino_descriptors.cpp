// Offline pass: add a DINOv2 global descriptor to every keyframe of a saved map by
// re-reading its image (matched by timestamp) and re-serialize the map. Keeps the
// online VO / map_refinement paths untouched; the descriptors ride along in the map
// (Frame serialization v1) for cross-condition relocalization in MapUser.
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "map.h"
#include "frame.h"
#include "dino_extractor.h"
#include "utils.h"

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: add_dino_descriptors <dino_onnx> <dino_engine> <map_in.bin> <map_out.bin> <mav0_dir>" << std::endl;
    return 1;
  }
  const std::string onnx = argv[1], engine = argv[2], map_in = argv[3], map_out = argv[4], mav0 = argv[5];

  DinoExtractor dino(onnx, engine);
  if (!dino.build()) { std::cerr << "DINO engine build/load FAILED" << std::endl; return 1; }

  std::shared_ptr<Map> map(new Map());
  {
    std::ifstream ifs(map_in, std::ios::binary);
    if (!ifs) { std::cerr << "cannot open map " << map_in << std::endl; return 1; }
    boost::archive::binary_iarchive ia(ifs);
    ia >> map;
  }

  // list cam0 images and their timestamps (filenames are timestamps)
  const std::string img_dir = ConcatenateFolderAndFileName(mav0, "cam0/data");
  std::vector<std::string> names;
  GetFileNames(img_dir, names);
  std::sort(names.begin(), names.end());
  std::vector<double> img_ts(names.size());
  for (size_t i = 0; i < names.size(); ++i) img_ts[i] = ImageNameToTime(names[i]);

  std::map<int, FramePtr>& keyframes = map->GetAllKeyframes();
  int done = 0, skipped = 0;
  float first_norm = -1.f;
  for (auto& kv : keyframes) {
    FramePtr f = kv.second;
    const double ts = f->GetTimestamp();
    int best = -1; double best_d = 1e18;
    for (size_t i = 0; i < img_ts.size(); ++i) {
      double d = std::fabs(img_ts[i] - ts);
      if (d < best_d) { best_d = d; best = static_cast<int>(i); }
    }
    if (best < 0 || best_d > 0.05) { ++skipped; continue; }
    // grayscale to match relocalization.cpp query loading (cv::imread(...,0)) so map
    // and query DINO descriptors use identical preprocessing (matters for color datasets)
    cv::Mat img = cv::imread(ConcatenateFolderAndFileName(img_dir, names[best]), cv::IMREAD_GRAYSCALE);
    if (img.empty()) { ++skipped; continue; }
    Eigen::VectorXf desc;
    if (!dino.infer(img, desc)) { ++skipped; continue; }
    f->SetDinoDescriptor(desc);
    if (first_norm < 0) first_norm = desc.norm();
    ++done;
  }
  std::cout << "DINO descriptors added to " << done << " keyframes (skipped " << skipped
            << "), first |desc|=" << first_norm << std::endl;

  {
    std::ofstream ofs(map_out, std::ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << map;
  }
  std::cout << "saved map with descriptors -> " << map_out << std::endl;

  // round-trip verify: reload and confirm descriptors persisted through serialization
  std::shared_ptr<Map> check(new Map());
  {
    std::ifstream ifs(map_out, std::ios::binary);
    boost::archive::binary_iarchive ia(ifs);
    ia >> check;
  }
  int with = 0; int dim = 0;
  for (auto& kv : check->GetAllKeyframes()) {
    const Eigen::VectorXf& d = kv.second->GetDinoDescriptor();
    if (d.size() > 0) { ++with; dim = static_cast<int>(d.size()); }
  }
  std::cout << "verify: reloaded map has " << with << " keyframes with a "
            << dim << "-d descriptor" << std::endl;
  return 0;
}
