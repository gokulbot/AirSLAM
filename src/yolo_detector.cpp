#include "yolo_detector.h"

#include <algorithm>
#include <cmath>
#include <opencv2/dnn.hpp>

using namespace tensorrt_log;
using namespace tensorrt_buffer;

YoloDetector::YoloDetector(const std::string& onnx_file, const std::string& engine_file,
                           float conf_thr, float iou_thr)
    : TRTModule(onnx_file, engine_file), conf_thr_(conf_thr), iou_thr_(iou_thr) {
  setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

bool YoloDetector::configure_network(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                                     TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                                     TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                                     TensorRTUniquePtr<nvonnxparser::IParser>& parser) {
  if (!parser->parseFromFile(onnx_file_.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
    return false;
  }
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  (void)builder;   // fixed 1x3x640x640 input -> no optimization profile needed
  (void)network;
  return true;
}

bool YoloDetector::process_input(const BufferManager& buffers, const cv::Mat& lb) {
  // lb: letterboxed RGB res_ x res_ (uint8). YOLOv8 expects RGB CHW float in [0,1].
  auto* host = static_cast<float*>(buffers.getHostBuffer(input_name_));
  const int H = lb.rows, W = lb.cols, HW = H * W;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const cv::Vec3b& px = lb.at<cv::Vec3b>(y, x);
      host[0 * HW + y * W + x] = px[0] / 255.0f;
      host[1 * HW + y * W + x] = px[1] / 255.0f;
      host[2 * HW + y * W + x] = px[2] / 255.0f;
    }
  }
  return true;
}

bool YoloDetector::Detect(const cv::Mat& image_, std::vector<cv::Rect>& boxes) {
  boxes.clear();
  if (!engine_) return false;
  if (!context_) {
    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) return false;
  }

  cv::Mat rgb;
  if (image_.channels() == 1) cv::cvtColor(image_, rgb, cv::COLOR_GRAY2RGB);
  else cv::cvtColor(image_, rgb, cv::COLOR_BGR2RGB);

  // letterbox to res_ x res_ (preserve aspect, pad with 114)
  const int W0 = rgb.cols, H0 = rgb.rows;
  const float scale = std::min(res_ / static_cast<float>(W0), res_ / static_cast<float>(H0));
  const int nw = static_cast<int>(std::round(W0 * scale));
  const int nh = static_cast<int>(std::round(H0 * scale));
  const int padx = (res_ - nw) / 2, pady = (res_ - nh) / 2;
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(nw, nh));
  cv::Mat lb(res_, res_, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(lb(cv::Rect(padx, pady, nw, nh)));

  BufferManager buffers(engine_, 0, context_.get());
  if (!process_input(buffers, lb)) return false;
  buffers.copyInputToDevice();
  buffers.setTensorAddresses(context_.get());
  if (!run(context_.get())) return false;
  buffers.copyOutputToHost();

  // output0: 1 x 84 x 8400, row-major -> out[c*na + a]. rows 0-3 = cx,cy,w,h; 4-83 = class scores.
  const float* out = static_cast<const float*>(buffers.getHostBuffer(output_name_));
  const int na = 8400;
  std::vector<cv::Rect> cand;
  std::vector<float> scores;
  for (int a = 0; a < na; ++a) {
    float best = 0.0f; int bestc = -1;
    for (int c : dynamic_classes_) {
      const float s = out[(4 + c) * na + a];
      if (s > best) { best = s; bestc = c; }
    }
    if (bestc < 0 || best < conf_thr_) continue;
    const float cx = out[0 * na + a], cy = out[1 * na + a], ww = out[2 * na + a], hh = out[3 * na + a];
    // xyxy in letterbox coords -> un-letterbox to original pixel coords
    const float x1 = (cx - ww * 0.5f - padx) / scale;
    const float y1 = (cy - hh * 0.5f - pady) / scale;
    const float x2 = (cx + ww * 0.5f - padx) / scale;
    const float y2 = (cy + hh * 0.5f - pady) / scale;
    cand.emplace_back(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                      cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
    scores.push_back(best);
  }

  std::vector<int> keep;
  cv::dnn::NMSBoxes(cand, scores, conf_thr_, iou_thr_, keep);
  for (int i : keep) boxes.push_back(cand[i] & cv::Rect(0, 0, W0, H0));   // clamp to image
  return true;
}
