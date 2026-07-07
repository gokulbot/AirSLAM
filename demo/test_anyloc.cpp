// Validate the native C++ AnyLoc (ViT-G TensorRT + C++ VLAD) against the Python reference.
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "anyloc_extractor.h"

int main(int argc, char** argv) {
  if (argc < 6) { std::cerr << "usage: test_anyloc <onnx> <engine> <vocab> <image> <ref.bin>" << std::endl; return 1; }
  AnyLocExtractor ex(argv[1], argv[2], argv[3]);
  if (!ex.valid()) { std::cerr << "vocab load FAILED" << std::endl; return 1; }
  if (!ex.build()) { std::cerr << "engine build/load FAILED" << std::endl; return 1; }
  cv::Mat img = cv::imread(argv[4]);   // colour, matches the Python reference preprocessing
  if (img.empty()) { std::cerr << "cannot read " << argv[4] << std::endl; return 1; }
  Eigen::VectorXf desc;
  if (!ex.infer(img, desc)) { std::cerr << "infer FAILED" << std::endl; return 1; }
  std::ifstream rf(argv[5], std::ios::binary);
  int32_t D = 0; rf.read(reinterpret_cast<char*>(&D), 4);
  Eigen::VectorXf ref(D); rf.read(reinterpret_cast<char*>(ref.data()), (std::streamsize)D * 4);
  double cos = desc.dot(ref) / (desc.norm() * ref.norm() + 1e-12);
  std::cout << "anyloc C++: dim=" << desc.size() << " norm=" << desc.norm()
            << " | cosine(C++ TRT-fp16, Python) = " << cos << std::endl;
  return 0;
}
