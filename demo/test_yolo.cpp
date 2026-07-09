// Standalone check of the C++ TensorRT YoloDetector before integrating into the frontend.
//   test_yolo <onnx> <engine> <image> [out.jpg]
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "yolo_detector.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "usage: test_yolo <onnx> <engine> <image> [out.jpg]\n";
    return 1;
  }
  YoloDetector det(argv[1], argv[2], 0.35f, 0.45f);
  if (!det.build()) { std::cout << "engine build/load FAILED\n"; return 2; }

  cv::Mat img = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);   // T265 fisheye is mono
  if (img.empty()) { std::cout << "could not read image " << argv[3] << "\n"; return 3; }

  std::vector<cv::Rect> boxes;
  if (!det.Detect(img, boxes)) { std::cout << "Detect FAILED\n"; return 4; }

  std::cout << "person boxes: " << boxes.size() << "\n";
  cv::Mat vis; cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);
  for (const auto& b : boxes) {
    std::cout << "  [" << b.x << "," << b.y << " " << b.width << "x" << b.height << "]\n";
    cv::rectangle(vis, b, cv::Scalar(0, 0, 255), 2);
  }
  if (argc >= 5) cv::imwrite(argv[4], vis);
  std::cout << "TESTYOLODONE\n";
  return 0;
}
