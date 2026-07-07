#ifndef PLACE_RECOGNITION_H_
#define PLACE_RECOGNITION_H_

// Clean interface subsystem for place recognition, decoupling the two independent axes
// the Phase-1 experiments exercised:
//   GlobalDescriptor  -- HOW a query's global descriptor is produced (ViT-S engine / external AnyLoc)
//   PlaceRecognizer   -- HOW candidate map keyframes are retrieved (DBoW2 bag-of-words / descriptor cosine)
// MapUser (relocalization) composes one of each from config; the covisibility grouping +
// LightGlue geometric verification downstream are shared and unchanged.
#include <string>
#include <map>
#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "frame.h"
#include "map.h"
#include "bow/database.h"

// ---------------- GlobalDescriptor: image -> place-recognition descriptor ----------------
class GlobalDescriptor {
 public:
  virtual ~GlobalDescriptor() = default;
  virtual std::string Name() const = 0;
  // `name` lets file-backed implementations key by frame; `image` for on-the-fly ones.
  virtual bool Compute(const cv::Mat& image, const std::string& name, Eigen::VectorXf& desc) = 0;
};
typedef std::shared_ptr<GlobalDescriptor> GlobalDescriptorPtr;

// Descriptors precomputed offline (e.g. full AnyLoc ViT-G/VLAD from Python), stored as
// <dir>/<name-without-ext>.f32 raw little-endian float32.
class ExternalGlobalDescriptor : public GlobalDescriptor {
 public:
  explicit ExternalGlobalDescriptor(std::string dir) : _dir(std::move(dir)) {}
  std::string Name() const override { return "external"; }
  bool Compute(const cv::Mat& image, const std::string& name, Eigen::VectorXf& desc) override;

 private:
  std::string _dir;
};

// ---------------- PlaceRecognizer: query frame -> scored candidate keyframes ----------------
class PlaceRecognizer {
 public:
  virtual ~PlaceRecognizer() = default;
  virtual std::string Name() const = 0;
  // `query` must carry whatever the recognizer needs: point features (BoW) or _dino_descriptor.
  virtual void Retrieve(FramePtr query, int topk, std::map<FramePtr, double>& candidates) = 0;
};
typedef std::shared_ptr<PlaceRecognizer> PlaceRecognizerPtr;

// DBoW2 bag-of-words retrieval over SuperPoint words.
class BowPlaceRecognizer : public PlaceRecognizer {
 public:
  explicit BowPlaceRecognizer(DatabasePtr database) : _database(std::move(database)) {}
  std::string Name() const override { return "dbow2"; }
  void Retrieve(FramePtr query, int topk, std::map<FramePtr, double>& candidates) override;

 private:
  DatabasePtr _database;
};

// Global-descriptor cosine retrieval over the map keyframes' stored _dino_descriptor.
class DescriptorPlaceRecognizer : public PlaceRecognizer {
 public:
  explicit DescriptorPlaceRecognizer(MapPtr map) : _map(std::move(map)) {}
  std::string Name() const override { return "descriptor"; }
  void Retrieve(FramePtr query, int topk, std::map<FramePtr, double>& candidates) override;

 private:
  MapPtr _map;
};

#endif  // PLACE_RECOGNITION_H_
