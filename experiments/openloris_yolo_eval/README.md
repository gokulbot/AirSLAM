# OpenLORIS × YOLO-seg — Stage-1 segmentation eval (Fork #1: fisheye)

Semantic-navigation plan → **Stage 1 (segmentation eval) / Fork #1 (fisheye handling).**
Question: does an off-the-shelf COCO detector work on OpenLORIS **T265 grayscale fisheye**, or do we
need undistortion / fine-tuning before we can do dynamic rejection + a semantic map?

## Setup (container, CPU is fine for an eval)
The C++ container has no Python ML stack. Installed CPU torch + ultralytics (Python 3.8 → torch 2.4.1):

```bash
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --ignore-installed psutil      # system distutils psutil blocks the uninstall
python3 -m pip install ultralytics
```

Model: **yolov8m-seg** (COCO, 80 classes) — medium, so a *fair* test of whether the *data* is the limiter,
not the model size. ~3–5 s/frame on CPU (fine; this is an offline eval).

## What was run
- `yolo_fisheye_eval.py <frames...>` — YOLO-seg on **RAW fisheye vs UNDISTORTED**. Kannala-Brandt fisheye
  undistort uses cam0 K/D from `configs/camera/openloris_corridor1-1_vio.yaml`. Prints detections; annotated
  images → `/tmp/yolo_eval/`.
- `mask_viz.py <frames...>` — clean **3-panel** viz (original | all masks | person-mask) → `/tmp/mask_viz/`.
- Input: 6 `corridor1-1` cam0 frames.

## Findings — Fork #1 RESOLVED (favorably, simpler than feared)
- ✅ **Raw grayscale fisheye WORKS.** COCO YOLOv8m-seg detects `chair` (0.91), stool, `tv`, `refrigerator`,
  `bed`, and **`person`** with clean, usable masks directly on the raw fisheye. Furniture masks are tight.
- ❌ **Undistortion HURTS.** A quick `estimateNewCameraMatrixForUndistortRectify` (balance 0.3) over-zoomed →
  radial smear → **zero** detections. Raw ≫ undistorted. (Fixable with a better `Knew`, but *unnecessary*.)
- **Grayscale** lowers confidence (person ~0.49 vs ~0.9 in color) but does **not** block detection.
- **No fine-tuning needed** for common indoor classes — COCO covers them.

**Decision:** use the **raw fisheye as-is** for semantics — no undistortion, no fine-tuning, no color camera
for a first cut. Perception is *not* the bottleneck here.

## Dynamic-rejection lesson (the false positive)
At a low `person` threshold (0.25), a **bag/jacket on a bench** was detected as `person 0.27` (a false
positive). So "reject everything labelled person" is not enough:
1. **Sensible threshold** (~0.35–0.4 for `person`) kills the FP while keeping the real (grayscale) person.
2. **Class + MOTION** (DynaSLAM-style) is the robust version — a real person moves across keyframes, a bag
   doesn't; confirm "dynamic" from cross-view geometric inconsistency, not class alone.

## Images
The annotated / 3-panel JPGs are eval artifacts (**gitignored**, regenerable via the scripts):
`masks_3panel_{person,office}_scene.jpg`, `detect_rawfisheye_{scene0,people}.jpg`,
`detect_undistorted_BROKEN_smear.jpg`.

## Next
- Threshold/motion policy for dynamic rejection, then wire the person mask into AirSLAM's
  **feature→mappoint creation** gate (Stage 1 → Stage 2). See the roadmap "🎯 ACTIVE PLAN" section.
