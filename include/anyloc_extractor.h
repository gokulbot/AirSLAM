#ifndef ANYLOC_EXTRACTOR_H_
#define ANYLOC_EXTRACTOR_H_

#include <string>
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "trt_module.h"
#include "place_recognition.h"

// Full AnyLoc as a native C++ GlobalDescriptor: DINOv2 ViT-G/14 value-facet (layer 31) via
// TensorRT outputs dense value tokens; VLAD-32 aggregation (with a loaded vocabulary) runs in
// C++. A drop-in strategy alongside DinoExtractor (ViT-S) -- MapUser picks by config.
class AnyLocExtractor : public TRTModule, public GlobalDescriptor {
 public:
  AnyLocExtractor(const std::string& onnx_file, const std::string& engine_file,
                  const std::string& vocab_file, int res = 224);

  bool infer(const cv::Mat& image, Eigen::VectorXf& descriptor);   // image -> K*E VLAD descriptor
  bool valid() const { return !_vocab.empty(); }

  // GlobalDescriptor interface
  std::string Name() const override { return "anyloc_vitg14"; }
  bool Compute(const cv::Mat& image, const std::string& name, Eigen::VectorXf& desc) override {
    (void)name; return infer(image, desc);
  }

 protected:
  bool configure_network(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                         TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                         TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                         TensorRTUniquePtr<nvonnxparser::IParser>& parser) override;

 private:
  bool process_input(const tensorrt_buffer::BufferManager& buffers, const cv::Mat& image);
  bool load_vocab(const std::string& path);
  Eigen::VectorXf vlad(const float* dense) const;

  int _res, _P, _E = 0, _K = 0;   // patches, embed dim, clusters
  std::vector<float> _vocab;      // K*E, row-major
  std::string _input_name = "image";
  std::string _output_name = "dense";
};

typedef std::shared_ptr<AnyLocExtractor> AnyLocExtractorPtr;

#endif  // ANYLOC_EXTRACTOR_H_
