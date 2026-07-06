# Phase 1 · A2.2 step 4 — DINO vs DBoW2 relocalization under illumination change

Full-pipeline evaluation of the C++ DINO relocalization (steps 1–3) against the DBoW2
baseline, on **real day/night stereo data with ground truth**.

## Dataset
[**Glasgow extreme-lighting SLAM**](https://huggingface.co/datasets/joe3012/glasgow-extreme-lighting-slam-dataset)
(HuggingFace, `joe3012/...`, MIT). Stereo colour 1280×720, KITTI layout
(`sequences/<route>/image_{2,3}` + `times.txt`, `poses/<route>.txt`), ground-truth poses
from Vive trackers. The `fig8_1` route is captured under 4 lighting conditions
(`rl` regular light, `offon` lights off→on, `strobe`, `strobe2`).

- **Map:** `fig8_1_rl` (regular light) — full stereo → VO → refine → DINO descriptors.
- **Query:** `fig8_1_offon`, first 1660 frames (a *different* traversal of the same room).

No calibration ships with the dataset → estimated rectified intrinsics (`fx=fy=700`,
`cx=640`, `cy=360`, baseline 0.12 m). VO built a 197-keyframe / 34k-point map that
Sim(3)-aligns to GT at **1.17 m RMSE over a 25 m path (~4.7% drift)** — normal stereo-VO,
so the estimate is sound. Reproduce the layout with `kitti_to_mav0.py`.

## Reality check on the dataset
The dataset's "extreme lighting" is **temporal flicker, not darkness** — every real query
frame is 44–139 mean brightness, comparable to the map. So it does not challenge the
illumination-robust SuperPoint/DBoW2 at all (DBoW2 scores 97.9% on the raw query). To get
a genuine night condition we **synthetically darkened the real query** (the same
methodology the AirVO/AirSLAM authors used to build `dark_euroc`), at two levels:
mean 32 (dim) and mean 10 (night; min 4.2).

## Results (recall + Sim(3)-aligned ATE vs Vive GT)

| Method | Query (mean brightness) | Recall | ATE RMSE |
|--------|-------------------------|--------|----------|
| DBoW2  | normal (78)             | **97.9%** | 1.21 m |
| DINO   | normal (78)             | 97.2%     | 1.20 m |
| DBoW2  | dark (32)               | **94.9%** | 1.21 m |
| DINO   | dark (32)               | 90.4%     | 1.18 m |
| DBoW2  | very dark (10)          | **69.9%** | 1.16 m |
| DINO   | very dark (10)          | **44.9%** | 1.16 m |

`use_dino:0` → DBoW2, `use_dino:1` → DINO (`reloc_glasgow_{dbow,dino}.yaml`).

## Verdict — negative result
**The deployment-grade DINO relocalization does not beat DBoW2 at any lighting level, and
the gap widens with darkness (69.9% vs 44.9% at night).** This does *not* contradict the
Python retrieval result (AnyLoc R@1 0.99 vs DBoW2 0.34 day→night); it exposes the gap
between research and deployment:

1. **Weaker descriptor.** Python AnyLoc = DINOv2 **ViT-G/14 + VLAD** on the value facet —
   deliberately illumination-*invariant*. What ships in C++ for 40 Hz is DINOv2
   **ViT-S/14 + mean-pool** — small and brightness-*sensitive*. Matching a mean-76 map
   descriptor to a mean-10 query descriptor, the little vector collapses.
2. **Shared verification ceiling.** Both paths end in the same SuperPoint + LightGlue +
   PnP check. Better candidate *selection* cannot rescue a frame SuperPoint cannot verify.
3. **Global-descriptor aliasing.** A small repetitive room (fig-8) is exactly where one
   global vector is *less* discriminative than DBoW2's local keypoint constellations.

ATE is ~1.2 m across the board because it is **map-drift-dominated** — every method
localizes against the same 1.17 m map (the single fig-8 gave 0 loop closures), so ATE
measures map quality, not the reloc method.

**Recommendation:** keep DBoW2 as the relocalization default (better *and* cheaper); DINO
stays an opt-in flag. To actually capture AnyLoc's night advantage would require either the
full ViT-G/VLAD descriptor for *offline* relocalization (too slow for the online path), or
an illumination-robust *local* feature to lift the shared verification ceiling.

## Files
- `kitti_to_mav0.py` — KITTI→ASL/mav0 symlink conversion (19-digit-ns filenames) + GT→TUM.
- `configs/camera/glasgow.yaml`, `configs/visual_odometry/vo_glasgow.yaml`,
  `configs/relocalization/reloc_glasgow_{dbow,dino}.yaml`.
- `demo/add_dino_descriptors.cpp` reads keyframe images **grayscale** to match the reloc
  query loader (`cv::imread(...,0)`) — critical for colour datasets.
