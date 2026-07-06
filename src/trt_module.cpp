#include "trt_module.h"

#include <fstream>
#include <vector>
#include <utility>
#include <cuda_runtime_api.h>

using namespace tensorrt_log;
using namespace tensorrt_buffer;

TRTModule::TRTModule(std::string onnx_file, std::string engine_file)
    : onnx_file_(std::move(onnx_file)), engine_file_(std::move(engine_file)), engine_(nullptr) {}

bool TRTModule::build() {
  if (deserialize_engine()) {
    return true;
  }
  auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
  if (!builder) return false;
  const auto explicit_batch =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
  if (!network) return false;
  auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) return false;
  auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
  if (!parser) return false;

  if (!configure_network(builder, network, config, parser)) return false;

  auto profile_stream = makeCudaStream();
  if (!profile_stream) return false;
  config->setProfileStream(*profile_stream);

  TensorRTUniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan) return false;
  TensorRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
  if (!runtime) return false;
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine_) return false;
  save_engine();
  return true;
}

bool TRTModule::run(nvinfer1::IExecutionContext* context) {
  cudaStream_t stream;
  if (cudaStreamCreate(&stream) != cudaSuccess) return false;
  bool status = context->enqueueV3(stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  return status;
}

void TRTModule::save_engine() {
  if (engine_file_.empty() || engine_ == nullptr) return;
  nvinfer1::IHostMemory* data = engine_->serialize();
  std::ofstream file(engine_file_, std::ios::binary);
  if (!file) return;
  file.write(reinterpret_cast<const char*>(data->data()), data->size());
}

bool TRTModule::deserialize_engine() {
  std::ifstream file(engine_file_.c_str(), std::ios::binary);
  if (!file.is_open()) return false;
  file.seekg(0, std::ifstream::end);
  size_t size = file.tellg();
  file.seekg(0, std::ifstream::beg);
  std::vector<char> model_stream(size);
  file.read(model_stream.data(), size);
  file.close();
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
  if (runtime == nullptr) return false;
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream.data(), size));
  return engine_ != nullptr;
}
