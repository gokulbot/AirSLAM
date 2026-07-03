# DINO loop-closure experiments (Phase 1)

Prototype for replacing AirSLAM's DBoW2 place recognition (step A of
`MapRefiner::LoopDetection`, `src/map_refiner.cc:65`) with DINOv2/AnyLoc global
descriptors. Python-first, per the roadmap — validate retrieval before any C++.

## Pipeline
1. `prep_manifest.py <mav0> <out.npz> --stride N` — sample frames, attach nearest GT pose.
2. `compute_descriptors.py <manifest> <desc.npz> [...]` — DINOv2 descriptors:
   - `cls` / `mean` (global tokens), `vlad` (VLAD over dense features), `tiny` (baseline).
   - **Full AnyLoc:** `--model dinov2_vitg14 --facet value --layer 31 --k 32`
     (value facet grabbed via a forward hook on that block's `qkv`).
3. `eval_retrieval.py <manifest> <desc.npz> <out.png> --tgap 10 --dist 2 --rot 45`
   — GT loop = position<dist, orientation<rot, time-gap>tgap; reports Recall@K and
   the top-1 loop-proposal precision-recall (the loop-closure-critical metric).

## Result 1 — EuRoC MH_04 (single session, 396 frames @ stride 5, 149 GT loops)

| method | R@1 | R@5 | R@10 | maxF1 |
|---|---|---|---|---|
| tiny-image | 0.664 | 0.765 | 0.832 | 0.605 |
| DINOv2-S VLAD | 0.893 | 0.926 | 0.953 | 0.877 |
| DINOv2-G CLS | 0.826 | 0.906 | 0.926 | 0.799 |
| DINOv2-G mean | 0.872 | 0.919 | 0.926 | 0.853 |
| **AnyLoc (ViT-G value-facet VLAD)** | 0.879 | **0.953** | **0.973** | **0.891** |

Takeaways:
- Any DINOv2 descriptor >> the tiny-image floor; VLAD > global token within a backbone.
- AnyLoc (ViT-G) wins on R@5/R@10 and the loop-critical maxF1 (0.891).
- ViT-S ≈ ViT-G *here* — MH_04 has no appearance change, so the big model's
  robustness isn't stressed. The advantage of AnyLoc should show on
  cross-session / illumination-change data (next experiment), and vs DBoW2.

## Environment
- CPU venv (`dino-exp/venv`) for ViT-S; GPU venv (`dino-exp/venv-gpu`, torch cu128)
  runs ViT-G on the Blackwell GPU (sm_120). Descriptors/plots live in `dino-exp/`
  (outside the repo).

## Result 3 — DBoW2 (AirSLAM's actual) vs DINO/AnyLoc, MH_04 (174 keyframes, 51 loops)

`demo/dump_bow.cpp` (via `dump_bow.launch`) dumps AirSLAM's DBoW2 (SuperPoint
vocabulary) pairwise BoW-similarity matrix over a saved map's keyframes;
`kf_manifest.py` attaches images+GT; `compare_dbow.py` scores both.

| method | R@1 | R@10 | maxF1 |
|---|---|---|---|
| **DBoW2 (AirSLAM)** | **0.863** | 1.000 | **0.831** |
| AnyLoc (ViT-G VLAD, res 224) | 0.765 | 0.980 | 0.787 |
| AnyLoc (ViT-G VLAD, res 448) | 0.765 | 0.941 | 0.783 |
| DINOv2 mean / cls | 0.745 / 0.686-0.725 | | 0.69-0.72 / 0.64-0.67 |
| tiny-image | 0.451 | 0.647 | 0.387 |

**DBoW2 BEATS AnyLoc on clean single-session loops** (~10% higher R@1 and maxF1),
robustly across resolution. This is the expected VPR result: local SuperPoint-BoW
is very precise under consistent appearance; holistic foundation descriptors win
only under *appearance change*. **Consequence: DINO/AnyLoc is not a free upgrade —
it REGRESSES easy-case loop closure. The entire case for replacing DBoW2 rests on
the appearance-change scenario**, which makes the day/night illumination test the
decisive, make-or-break experiment for Phase 1.

## Result 2 — OpenLORIS office cross-session (map=office1-1, query=office1-N)

Goal: test illumination/appearance robustness across sessions. **Finding: OpenLORIS
office is the wrong vehicle for this** — the cross-session variation is viewpoint +
dynamic objects, not illumination, and the room is a tiny 3.6x1.5m.

Using GT *only* (no compute), heading difference at matched positions vs office1-1:
`office1-2/4/7 ~167deg (opposite)`, `office1-5 143deg`, `office1-3 61deg`,
`office1-6 4deg (co-directional)`. So the pairs are either:
- **opposite-heading** (1-2): appearance VPR is *impossible* (can't recognize a
  place seen from behind) — all methods ~0.04 R@1, incl. AnyLoc. A real viewpoint
  limit; this is why AirSLAM does geometric verification + a loop-distance gate.
- **co-directional, same lighting** (1-6): near-duplicate frames → *trivial*, all
  methods (incl. tiny-image) hit R@1=1.0, even under gamma-2.5 query darkening
  (gamma is monotonic; the near-duplicate dominates).

office1-1..1-7 all share global lighting (mean brightness 73/255), so no pair
isolates illumination. **Conclusion: a clean illumination test needs a day/night
or seasonal VPR benchmark** (SVOX, Nordland, Oxford RobotCar, Tokyo 24/7), where
the same route is traversed under genuinely different conditions with no near-dups.

## TODO
- Illumination robustness on a real **day/night VPR benchmark** (SVOX/Nordland).
- Baseline vs **actual DBoW2** (instrument AirSLAM) — the real "replace it?" number.
- Moderate viewpoint-change pair (office1-3 @ 61deg) — the interesting middle ground.
