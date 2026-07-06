#include "dino_extractor.h"

using namespace tensorrt_log;
using namespace tensorrt_buffer;

DinoExtractor::DinoExtractor(const std::string& onnx_file, const std::string& engine_file,
                             int res, int desc_dim)
    : TRTModule(onnx_file, engine_file), res_(res), desc_dim_(desc_dim) {
  setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

bool DinoExtractor::configure_network(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                                      TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                                      TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                                      TensorRTUniquePtr<nvonnxparser::IParser>& parser) {
  if (!parser->parseFromFile(onnx_file_.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
    return false;
  }
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  // input shape is fixed (1x3xRxR) -> no optimization profile needed.
  (void)builder;
  (void)network;
  return true;
}

bool DinoExtractor::process_input(const BufferManager& buffers, const cv::Mat& image) {
  cv::Mat rgb;
  if (image.channels() == 1) {
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  } else {
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  }
  auto* host = static_cast<float*>(buffers.getHostBuffer(input_name_));
  const int H = rgb.rows;
  const int W = rgb.cols;
  const int HW = H * W;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const cv::Vec3b& px = rgb.at<cv::Vec3b>(y, x);   // pixel already in [0,255]; ONNX normalizes
      host[0 * HW + y * W + x] = static_cast<float>(px[0]);
      host[1 * HW + y * W + x] = static_cast<float>(px[1]);
      host[2 * HW + y * W + x] = static_cast<float>(px[2]);
    }
  }
  return true;
}

bool DinoExtractor::infer(const cv::Mat& image_, Eigen::VectorXf& descriptor) {
  if (!engine_) return false;
  if (!context_) {
    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) return false;
  }

  cv::Mat image;
  cv::resize(image_, image, cv::Size(res_, res_));

  BufferManager buffers(engine_, 0, context_.get());
  if (!process_input(buffers, image)) return false;
  buffers.copyInputToDevice();
  buffers.setTensorAddresses(context_.get());
  if (!run(context_.get())) return false;
  buffers.copyOutputToHost();

  const auto* out = static_cast<const float*>(buffers.getHostBuffer(output_name_));
  descriptor.resize(desc_dim_);
  for (int i = 0; i < desc_dim_; ++i) descriptor(i) = out[i];
  return true;
}
