# ONNX → TensorRT 10 compatibility

The pre-exported ONNX models in `output/` were exported for **TensorRT 8**. TensorRT 10's
ONNX parser is stricter about integer types on shape-computation ops (elementwise, `Concat`,
`Where`, comparisons): it rejects mixing `Int64` (from `Shape`/`Unsqueeze`) with `Int32`
constants, whereas TRT 8 silently coerced them. Symptom:

```
[E] [TRT] ... incompatible input types Int64 and Int32 ...  Invalid Node - Concat_43
Error in SuperPoint building
```

`fix_int_types.py` patches this by casting the int32 inputs **up to int64** at each
mismatched op — aligning with TRT 10's native int64 shape handling. It preserves each
model's exact input/output tensor names (the contract the C++ relies on) and works for
dynamic-shape models, so no re-export or C++ change is needed.

In practice only the opset-12 `superpoint_v1_sim_int32.onnx` needs it; the opset-17 PLNet
models are already clean, but running the tool over all of them is harmless (0 casts).

## Apply

Requires `onnx` (`pip install onnx`). From the repo root:

```bash
python3 tools/onnx_trt10/fix_int_types.py --backup=output/orig_onnx output/*.onnx
```

Originals are backed up to `output/orig_onnx/` on first run. Delete any stale `output/*.engine`
afterward so the engines rebuild from the fixed ONNX:

```bash
rm -f output/*.engine
```
