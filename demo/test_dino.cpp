// Validate the DINOv2 TensorRT engine end-to-end in C++: build/load the engine,
// run it on an image, dump the descriptor (compare to the Python/ONNX reference).
#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "dino_extractor.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: test_dino <onnx> <engine> <image> <out_txt>" << std::endl;
    return 1;
  }
  DinoExtractor dino(argv[1], argv[2]);
  if (!dino.build()) {
    std::cerr << "engine build/load FAILED" << std::endl;
    return 1;
  }
  cv::Mat img = cv::imread(argv[3]);
  if (img.empty()) {
    std::cerr << "cannot read image " << argv[3] << std::endl;
    return 1;
  }
  Eigen::VectorXf desc;
  if (!dino.infer(img, desc)) {
    std::cerr << "infer FAILED" << std::endl;
    return 1;
  }
  std::ofstream f(argv[4]);
  f << std::setprecision(9);
  for (int i = 0; i < desc.size(); ++i) f << desc(i) << (i + 1 < desc.size() ? " " : "\n");
  std::cout << "dino descriptor: dim=" << desc.size() << " norm=" << desc.norm() << std::endl;
  return 0;
}
