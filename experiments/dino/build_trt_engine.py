#!/usr/bin/env python3
"""Build a TensorRT engine from an ONNX and (if a CUDA mem binding is available)
run it on a reference input to validate numerics vs the ONNX. Dev/validation tool
for the DINOv2 reloc engine — deployment builds the engine from C++.

usage: build_trt_engine.py <onnx> <engine_out> [--fp16] [--ref_in x.npy --ref_desc d.npy]
"""
import argparse, numpy as np, tensorrt as trt

ap = argparse.ArgumentParser()
ap.add_argument("onnx"); ap.add_argument("engine")
ap.add_argument("--fp16", action="store_true")
ap.add_argument("--ref_in"); ap.add_argument("--ref_desc")
a = ap.parse_args()

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)          # TRT10: explicit batch by default
parser = trt.OnnxParser(network, logger)
with open(a.onnx, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print("PARSE ERROR:", parser.get_error(i))
        raise SystemExit(1)
print(f"parsed OK | inputs={[network.get_input(i).name+str(network.get_input(i).shape) for i in range(network.num_inputs)]}"
      f" outputs={[network.get_output(i).name+str(network.get_output(i).shape) for i in range(network.num_outputs)]}")

config = builder.create_builder_config()
if a.fp16:
    config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    raise SystemExit("build FAILED")
with open(a.engine, "wb") as f:
    f.write(memoryview(serialized))
print(f"ENGINE BUILT: {a.engine}  ({serialized.nbytes/1e6:.1f} MB, fp16={a.fp16})")

if not (a.ref_in and a.ref_desc):
    raise SystemExit(0)

# --- numerical validation (pycuda for device buffers) ---
try:
    import pycuda.autoinit, pycuda.driver as drv
except Exception as e:
    print("no pycuda -> skipped numerical run:", e); raise SystemExit(0)

rt = trt.Runtime(logger); engine = rt.deserialize_cuda_engine(serialized)
ctx = engine.create_execution_context()
xin = np.ascontiguousarray(np.load(a.ref_in), np.float32)
ref = np.load(a.ref_desc).astype(np.float32).reshape(-1)
out = np.empty((1, ref.shape[-1]), np.float32)
din, dout = drv.mem_alloc(xin.nbytes), drv.mem_alloc(out.nbytes)
drv.memcpy_htod(din, xin)
ctx.set_tensor_address(engine.get_tensor_name(0), int(din))
ctx.set_tensor_address(engine.get_tensor_name(1), int(dout))
ctx.execute_v2([int(din), int(dout)])
drv.memcpy_dtoh(out, dout)
cos = float((out[0] * ref).sum() / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-9))
print(f"TRT(fp16) vs ONNX descriptor: cosine={cos:.6f}")
