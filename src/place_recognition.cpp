#include "place_recognition.h"

#include <fstream>
#include <vector>
#include <algorithm>
#include <utility>

bool ExternalGlobalDescriptor::Compute(const cv::Mat& image, const std::string& name, Eigen::VectorXf& desc) {
  (void)image;
  std::string base = name;
  size_t dot = base.find_last_of('.');
  if (dot != std::string::npos) base = base.substr(0, dot);
  std::ifstream f(_dir + "/" + base + ".f32", std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  std::streamoff sz = f.tellg();
  f.seekg(0, std::ios::beg);
  desc.resize(static_cast<int>(sz / 4));
  f.read(reinterpret_cast<char*>(desc.data()), sz);
  return desc.size() > 0;
}

void BowPlaceRecognizer::Retrieve(FramePtr query, int topk, std::map<FramePtr, double>& candidates) {
  (void)topk;   // DBoW2 selects by shared-word thresholding, not a fixed K
  DBoW2::WordIdToFeatures word_features;
  DBoW2::BowVector bow_vector;
  std::vector<DBoW2::WordId> word_of_features;
  _database->FrameToBow(query->GetAllFeatures(), word_features, bow_vector, word_of_features);

  std::map<FramePtr, int> frame_sharing_words;
  _database->Query(bow_vector, frame_sharing_words);
  if (frame_sharing_words.empty()) return;

  int max_sharing_words = 0;
  for (const auto& kv : frame_sharing_words) {
    max_sharing_words = kv.second > max_sharing_words ? kv.second : max_sharing_words;
  }
  int sharing_words_num_thr = std::max(static_cast<int>(max_sharing_words * 0.3f), 8);
  for (auto it = frame_sharing_words.begin(); it != frame_sharing_words.end();) {
    if (it->second < sharing_words_num_thr) it = frame_sharing_words.erase(it);
    else ++it;
  }
  for (const auto& kv : frame_sharing_words) {
    candidates[kv.first] = _database->Score(_database->_frame_bow_vectors[kv.first], bow_vector);
  }
}

void DescriptorPlaceRecognizer::Retrieve(FramePtr query, int topk, std::map<FramePtr, double>& candidates) {
  const Eigen::VectorXf& qd = query->GetDinoDescriptor();
  if (qd.size() == 0) return;
  std::vector<std::pair<double, FramePtr>> ranked;
  for (auto& kv : _map->GetAllKeyframes()) {
    const Eigen::VectorXf& d = kv.second->GetDinoDescriptor();
    if (d.size() == qd.size() && d.size() > 0) {
      ranked.emplace_back(static_cast<double>(qd.dot(d)), kv.second);   // both unit-norm -> cosine
    }
  }
  if (ranked.empty()) return;
  size_t K = std::min(static_cast<size_t>(topk), ranked.size());
  std::partial_sort(ranked.begin(), ranked.begin() + K, ranked.end(),
      [](const std::pair<double, FramePtr>& a, const std::pair<double, FramePtr>& b) { return a.first > b.first; });
  for (size_t i = 0; i < K; ++i) candidates[ranked[i].second] = ranked[i].first;
}
