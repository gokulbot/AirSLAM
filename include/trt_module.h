#ifndef TRT_MODULE_H_
#define TRT_MODULE_H_

#include <string>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "3rdparty/tensorrtbuffer/include/buffers.h"

using tensorrt_buffer::TensorRTUniquePtr;

// Base class owning the TensorRT engine lifecycle shared by every learned module
// (build/deserialize/save + a run helper). Subclasses implement configure_network()
// (ONNX parse + optimization profile / builder flags) and their own infer() with
// model-specific pre/post-processing. Existing modules (SuperPoint, PLNet, ...) can
// migrate onto this later; for now only DinoExtractor uses it.
class TRTModule {
 public:
  TRTModule(std::string onnx_file, std::string engine_file);
  virtual ~TRTModule() = default;

  // Load the cached engine if present, otherwise build it from the ONNX and cache it.
  bool build();

 protected:
  // Per-model: parse the ONNX into `network` and set profile / builder flags on `config`.
  virtual bool configure_network(TensorRTUniquePtr<nvinfer1::IBuilder>& builder,
                                 TensorRTUniquePtr<nvinfer1::INetworkDefinition>& network,
                                 TensorRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                                 TensorRTUniquePtr<nvonnxparser::IParser>& parser) = 0;

  // Common: create a stream, enqueueV3, synchronize. Tensor addresses must already
  // be bound on `context` (via BufferManager::setTensorAddresses) by the caller.
  bool run(nvinfer1::IExecutionContext* context);

  bool deserialize_engine();
  void save_engine();

  std::string onnx_file_;
  std::string engine_file_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
};

#endif  // TRT_MODULE_H_
