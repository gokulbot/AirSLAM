# model_export

Source for AirSLAM's learned front-end, kept in-tree so the ONNX models under
[`../output/`](../output/) are reproducible rather than opaque binaries.

## Contents

- **`PLNet/`** — official [PLNet](https://github.com/sair-lab/PLNet) implementation:
  the CNN that jointly detects keypoints and structural lines (AirSLAM's feature
  detector). Includes the training/eval code, configs, and the pretrained
  point-model weights (`PLNet/hawp/fsl/point_model/point_model.pth`).

SuperPoint / SuperGlue / LightGlue ONNX graphs come from their upstream
exporters; only PLNet is vendored here since it is the AirSLAM-specific piece.

## From weights to a running engine

1. **Export to ONNX** — follow `PLNet/readme.md` to run the model and export an
   ONNX graph (needs the PyTorch env described there; `PLNet/downloads.sh` fetches
   the datasets/weights if the bundled `.pth` is not enough).
2. **Make it TensorRT-10 safe** — run the graph through
   [`../tools/onnx_trt10/fix_int_types.py`](../tools/onnx_trt10/), which casts
   `int32 → int64` at mixed-type ops so the TRT-10 parser accepts it. See that
   tool's README for why TRT-10 is stricter than TRT-8.
3. **Build the engine** — AirSLAM builds the `.engine` from the `.onnx` on first
   run for the local GPU (SM arch); engines are gitignored.

## Notes

- The nested PLNet `.git` was removed when vendoring (no repo-in-repo).
- `PLNet/docs/figures/` (paper images, ~13 MB) is gitignored to keep the repo
  lean — a few README image links will 404 on GitHub as a result. The files are
  still present locally; drop the ignore rule in `../.gitignore` to track them.
