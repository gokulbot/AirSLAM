// Bridge externally-computed global descriptors (e.g. full AnyLoc ViT-G/VLAD from
// Python) into a saved map's per-keyframe _dino_descriptor, so loop closure can use a
// heavier descriptor than the deployed C++ DinoExtractor. Two modes:
//   dump:   dino_descriptor_io dump   <map_in.bin> <mav0_dir> <manifest_out.txt>
//           -> one keyframe-image path per line, in keyframe (frame_id) order
//   inject: dino_descriptor_io inject <map_in.bin> <desc.bin> <map_out.bin>
//           -> desc.bin = [int32 N][int32 D][float32 N*D], row i -> keyframe i's descriptor
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "map.h"
#include "frame.h"
#include "utils.h"

static std::shared_ptr<Map> load_map(const std::string& path) {
  std::shared_ptr<Map> map(new Map());
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) { std::cerr << "cannot open map " << path << std::endl; std::exit(1); }
  boost::archive::binary_iarchive ia(ifs); ia >> map;
  return map;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: dino_descriptor_io dump   <map_in> <mav0> <manifest_out>\n"
              << "       dino_descriptor_io inject <map_in> <desc.bin> <map_out>" << std::endl;
    return 1;
  }
  const std::string mode = argv[1];
  std::shared_ptr<Map> map = load_map(argv[2]);
  std::map<int, FramePtr>& kfs = map->GetAllKeyframes();   // iterates in frame_id order

  if (mode == "dump") {
    const std::string img_dir = ConcatenateFolderAndFileName(argv[3], "cam0/data");
    std::vector<std::string> names; GetFileNames(img_dir, names); std::sort(names.begin(), names.end());
    std::vector<double> img_ts(names.size());
    for (size_t i = 0; i < names.size(); ++i) img_ts[i] = ImageNameToTime(names[i]);
    std::ofstream mf(argv[4]);
    int n = 0;
    for (auto& kv : kfs) {
      const double ts = kv.second->GetTimestamp();
      int best = -1; double bd = 1e18;
      for (size_t i = 0; i < img_ts.size(); ++i) { double d = std::fabs(img_ts[i] - ts); if (d < bd) { bd = d; best = (int)i; } }
      mf << ConcatenateFolderAndFileName(img_dir, names[best]) << "\n";
      ++n;
    }
    std::cout << "dumped " << n << " keyframe image paths -> " << argv[4] << std::endl;
    return 0;
  }

  if (mode == "inject") {
    std::ifstream df(argv[3], std::ios::binary);
    if (!df) { std::cerr << "cannot open desc " << argv[3] << std::endl; return 1; }
    int32_t N = 0, D = 0;
    df.read(reinterpret_cast<char*>(&N), 4); df.read(reinterpret_cast<char*>(&D), 4);
    if ((int)kfs.size() != N) {
      std::cerr << "mismatch: map has " << kfs.size() << " keyframes but desc has " << N << " rows" << std::endl;
      return 1;
    }
    std::vector<float> row(D);
    int i = 0;
    for (auto& kv : kfs) {
      df.read(reinterpret_cast<char*>(row.data()), (std::streamsize)D * 4);
      Eigen::VectorXf desc(D);
      for (int j = 0; j < D; ++j) desc(j) = row[j];
      kv.second->SetDinoDescriptor(desc);
      ++i;
    }
    std::ofstream ofs(argv[4], std::ios::binary);
    boost::archive::binary_oarchive oa(ofs); oa << map;
    std::cout << "injected " << i << " descriptors (dim " << D << ") -> " << argv[4] << std::endl;
    return 0;
  }

  std::cerr << "unknown mode " << mode << std::endl;
  return 1;
}
