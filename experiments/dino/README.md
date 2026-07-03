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

## TODO
- Baseline vs **actual DBoW2** (instrument AirSLAM or reimplement) — the real comparison.
- **Cross-session** test (OpenLORIS office1-1 vs 1-2, or day/night) — where AnyLoc should win big.
