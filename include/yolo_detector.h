#ifndef YOLO_DETECTOR_H_
#define YOLO_DETECTOR_H_

#include <string>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#include "trt_module.h"

// YOLOv8 detector (TensorRT) -> boxes of dynamic-class objects (default: COCO person).
// Runs at KEYFRAME rate for dynamic-object rejection: a keypoint inside a returned box drives
// Mappoint::AddDynamicObservation(true). Exported ONNX is a plain YOLOv8 detect head
// (input images 1x3x640x640, output0 1x84x8400 = [cx,cy,w,h, 80 class scores] x 8400 anchors);
// NMS + un-letterbox are done here.
class YoloDetector : public TRTModule {
 public:
  YoloDetector(const std::string& onnx_file, const std::string& engine_file,
               float conf_thr = 0.35f, float iou_thr = 0.45f);

  // image (any size, BGR or gray) -> dynamic-class boxes in the image's ORIGINAL pixel coords.
  bool Detect(const cv::Mat& image, std::vector<cv::Rect>& boxes);

 protected:
  bool configure_network(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                         TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                         TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                         TensorRTUniquePtr<nvonnxparser::IParser>& parser) override;

 private:
  bool process_input(const tensorrt_buffer::BufferManager& buffers, const cv::Mat& letterboxed);

  int res_ = 640;
  float conf_thr_, iou_thr_;
  std::string input_name_ = "images";
  std::string output_name_ = "output0";
  std::vector<int> dynamic_classes_ = {0};   // COCO 'person' (extend outdoors: 1 bicycle,2 car,3 motorcycle...)
};
typedef std::shared_ptr<YoloDetector> YoloDetectorPtr;

#endif  // YOLO_DETECTOR_H_
