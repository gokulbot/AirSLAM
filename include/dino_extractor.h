#ifndef DINO_EXTRACTOR_H_
#define DINO_EXTRACTOR_H_

#include <string>
#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "trt_module.h"
#include "place_recognition.h"

// DINOv2 global-descriptor extractor for cross-condition relocalization.
// The exported ONNX bakes in resize->/255->ImageNet-normalize->backbone->mean-pool
// ->L2-normalize, so we only feed a resized 3-channel RGB image in [0,255] and read
// back a unit-norm descriptor.
class DinoExtractor : public TRTModule, public GlobalDescriptor {
 public:
  DinoExtractor(const std::string& onnx_file, const std::string& engine_file,
                int res = 224, int desc_dim = 384);

  // image (any size, BGR or gray) -> L2-normalized descriptor (desc_dim).
  bool infer(const cv::Mat& image, Eigen::VectorXf& descriptor);

  // GlobalDescriptor interface: on-the-fly ViT-S descriptor (the name is unused here).
  std::string Name() const override { return "dino_vits14"; }
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

  int res_;
  int desc_dim_;
  std::string input_name_ = "image";
  std::string output_name_ = "descriptor";
};

typedef std::shared_ptr<DinoExtractor> DinoExtractorPtr;

#endif  // DINO_EXTRACTOR_H_
