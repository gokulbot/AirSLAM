// Dump AirSLAM's DBoW2 (SuperPoint vocabulary) pairwise BoW-similarity matrix over
// the keyframes of a saved map, for comparing DBoW2 vs DINO/AnyLoc place recognition.
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <ros/ros.h>

#include "read_configs.h"
#include "map.h"
#include "map_refiner.h"
#include "bow/database.h"
#include "frame.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "air_slam");
  ros::NodeHandle nh;

  std::string config_path, model_dir, map_root, voc_path, out_path;
  ros::param::get("~config_path", config_path);
  ros::param::get("~model_dir", model_dir);
  ros::param::get("~map_root", map_root);
  ros::param::get("~voc_path", voc_path);
  ros::param::get("~out_path", out_path);

  MapRefinementConfigs configs(config_path, model_dir);
  MapRefiner map_refiner(configs, nh);
  map_refiner.LoadMap(map_root);
  map_refiner.LoadVocabulary(voc_path);

  MapPtr map = map_refiner.GetMap();
  DatabasePtr db = map_refiner.GetDatabase();
  std::map<int, FramePtr>& keyframes = map->GetAllKeyframes();

  std::vector<int> ids;
  std::vector<double> times;
  std::vector<DBoW2::BowVector> bows;
  for (auto& kv : keyframes) {
    FramePtr f = kv.second;
    DBoW2::WordIdToFeatures wf;
    DBoW2::BowVector bv;
    db->FrameToBow(f, wf, bv);
    ids.push_back(f->GetFrameId());
    times.push_back(f->GetTimestamp());
    bows.push_back(bv);
  }
  int N = (int)bows.size();
  std::cout << "computing " << N << "x" << N << " BoW score matrix" << std::endl;

  std::ofstream ofs(out_path);
  ofs << N << "\n";
  for (int i = 0; i < N; i++)
    ofs << ids[i] << " " << std::fixed << std::setprecision(9) << times[i] << "\n";
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double s = db->Score(bows[i], bows[j]);
      ofs << std::setprecision(6) << s << (j + 1 < N ? " " : "\n");
    }
  }
  ofs.close();
  std::cout << "dumped BoW matrix (" << N << " keyframes) to " << out_path << std::endl;
  exit(0);
  return 0;
}
