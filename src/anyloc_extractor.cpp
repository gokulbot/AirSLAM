#include "anyloc_extractor.h"

#include <fstream>
#include <cmath>
#include <cstdint>

using namespace tensorrt_log;
using namespace tensorrt_buffer;

AnyLocExtractor::AnyLocExtractor(const std::string& onnx_file, const std::string& engine_file,
                                 const std::string& vocab_file, int res)
    : TRTModule(onnx_file, engine_file), _res(res) {
  setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
  _P = (res / 14) * (res / 14);
  load_vocab(vocab_file);
}

bool AnyLocExtractor::load_vocab(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::cerr << "AnyLoc: cannot open vocab " << path << std::endl; return false; }
  int32_t K = 0, D = 0;
  f.read(reinterpret_cast<char*>(&K), 4);
  f.read(reinterpret_cast<char*>(&D), 4);
  _K = K; _E = D;
  _vocab.resize(static_cast<size_t>(K) * D);
  f.read(reinterpret_cast<char*>(_vocab.data()), static_cast<std::streamsize>(_vocab.size()) * 4);
  return f.good() || f.eof();
}

bool AnyLocExtractor::configure_network(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                                        TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                                        TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                                        TensorRTUniquePtr<nvonnxparser::IParser>& parser) {
  if (!parser->parseFromFile(onnx_file_.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
    return false;
  }
  config->setFlag(nvinfer1::BuilderFlag::kFP16);   // fp16==fp32 here (VLAD gap is pipeline, not precision)
  (void)builder; (void)network;
  return true;
}

bool AnyLocExtractor::process_input(const BufferManager& buffers, const cv::Mat& image) {
  cv::Mat rgb;
  if (image.channels() == 1) cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  else cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  auto* host = static_cast<float*>(buffers.getHostBuffer(_input_name));
  const int H = rgb.rows, W = rgb.cols, HW = H * W;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const cv::Vec3b& px = rgb.at<cv::Vec3b>(y, x);   // [0,255]; normalization baked into ONNX
      host[0 * HW + y * W + x] = static_cast<float>(px[0]);
      host[1 * HW + y * W + x] = static_cast<float>(px[1]);
      host[2 * HW + y * W + x] = static_cast<float>(px[2]);
    }
  }
  return true;
}

Eigen::VectorXf AnyLocExtractor::vlad(const float* dense) const {
  // dense: (_P tokens x _E channels) row-major; _vocab: (_K centers x _E) row-major.
  std::vector<float> V(static_cast<size_t>(_K) * _E, 0.0f);
  for (int p = 0; p < _P; ++p) {
    const float* tok = dense + static_cast<size_t>(p) * _E;
    int best = 0; float best_d = 1e30f;
    for (int c = 0; c < _K; ++c) {
      const float* ctr = _vocab.data() + static_cast<size_t>(c) * _E;
      float d = 0.f;
      for (int e = 0; e < _E; ++e) { float diff = tok[e] - ctr[e]; d += diff * diff; }
      if (d < best_d) { best_d = d; best = c; }
    }
    const float* ctr = _vocab.data() + static_cast<size_t>(best) * _E;
    float* v = V.data() + static_cast<size_t>(best) * _E;
    for (int e = 0; e < _E; ++e) v[e] += tok[e] - ctr[e];
  }
  // intra-normalize each cluster's residual
  for (int c = 0; c < _K; ++c) {
    float* v = V.data() + static_cast<size_t>(c) * _E;
    float nrm = 0.f; for (int e = 0; e < _E; ++e) nrm += v[e] * v[e];
    nrm = std::sqrt(nrm) + 1e-8f;
    for (int e = 0; e < _E; ++e) v[e] /= nrm;
  }
  // flatten (row-major) + global L2
  Eigen::VectorXf desc(static_cast<int>(V.size()));
  double nrm = 0.0;
  for (size_t i = 0; i < V.size(); ++i) { desc(static_cast<int>(i)) = V[i]; nrm += static_cast<double>(V[i]) * V[i]; }
  desc /= (static_cast<float>(std::sqrt(nrm)) + 1e-8f);
  return desc;
}

bool AnyLocExtractor::infer(const cv::Mat& image_, Eigen::VectorXf& descriptor) {
  if (!engine_ || _vocab.empty()) return false;
  if (!context_) {
    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) return false;
  }
  cv::Mat image;
  cv::resize(image_, image, cv::Size(_res, _res));
  BufferManager buffers(engine_, 0, context_.get());
  if (!process_input(buffers, image)) return false;
  buffers.copyInputToDevice();
  buffers.setTensorAddresses(context_.get());
  if (!run(context_.get())) return false;
  buffers.copyOutputToHost();
  const auto* dense = static_cast<const float*>(buffers.getHostBuffer(_output_name));
  descriptor = vlad(dense);
  return true;
}
