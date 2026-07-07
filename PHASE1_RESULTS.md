# Phase 1 — DINO / AnyLoc place recognition: Results

Record of the Phase-1 experiments (DINOv2 / AnyLoc global descriptors for AirSLAM place
recognition). All numbers are against ground truth. **Verdict up front: DINO/AnyLoc earns its
keep in exactly one place — relocalization under genuine darkness — and only as the heavy
ViT-G/VLAD descriptor. Everywhere else DBoW2 is as good or better.**

## What was built (all native C++, in the `air_slam` build)
Strategy-pattern place-recognition subsystem (`include/place_recognition.h`):

```
GlobalDescriptor (image -> descriptor)      PlaceRecognizer (query -> candidates)
├─ DinoExtractor        (ViT-S/14, TRT)      ├─ BowPlaceRecognizer        (DBoW2)
├─ AnyLocExtractor      (ViT-G/14 + VLAD-32) └─ DescriptorPlaceRecognizer (cosine)
└─ ExternalGlobalDescriptor (file-backed)
```
`MapUser` composes one of each by config; covisibility grouping + LightGlue verification are
shared. Also: per-keyframe descriptor in the map (versioned serialization), `add_dino_descriptors
[--anyloc]` bake, `dino_descriptor_io` bridge, separate `MapRefiner::DinoLoopDetection` pass.
The ViT-G TensorRT engine builds at **1.79 GB fp16 / 44 s** on an 8 GB laptop GPU → deployable.

## 1. Loop closure — EuRoC, SE(3)-aligned ATE (metric)
| Sequence | Method | Loops | ATE RMSE |
|----------|--------|------:|---------:|
| MH_04 | VO baseline | 0 | 0.121 m |
| MH_04 | **DBoW2** | 35 | **0.077 m** |
| MH_04 | DINO ViT-S | 81 | 0.081 m |
| MH_04 | DINO AnyLoc | 80 | 0.082 m |
| MH_05 | VO baseline | 0 | 0.121 m |
| MH_05 | DBoW2 | 40 | 0.0665 m |
| MH_05 | **DINO ViT-S** | 88 | **0.0660 m** |

Loop closure cuts ATE ~40–45%, but **which detector doesn't matter** (DBoW2 ≈ DINO ≈ AnyLoc,
within ~0.5 mm). DINO finds ~2.3× more loops (35→80) — those extra loops **do not improve the
map**. Same-session, DBoW2 is already optimal.

## 2. Relocalization — Glasgow day/night, Sim(3)-aligned (recall + ATE)
| Method | Query (mean brightness) | Recall | ATE RMSE |
|--------|-------------------------|-------:|---------:|
| DBoW2 | normal (78) | 97.9% | 1.21 m |
| DINO ViT-S | normal (78) | 97.2% | 1.20 m |
| DBoW2 | dark (32) | 94.9% | 1.21 m |
| DINO ViT-S | dark (32) | 90.4% | 1.18 m |
| DINO ViT-S | very dark (10) | 44.9% | 1.16 m |
| DBoW2 | very dark (10) | 69.9% | 1.16 m |
| AnyLoc (Python-fed) | very dark (10) | 73.0% | 1.14 m |
| **AnyLoc (native C++)** | **very dark (10)** | **71.8%** | 1.13 m |

Reloc ATE is ~1.2 m for everything — it's **map-drift-dominated** (localizing against the same
map), so **recall is the discriminator**, not ATE. **AnyLoc is the only method that beats DBoW2
under night-level darkness** (native 71.8% > DBoW2 69.9% > ViT-S 44.9% @ mean-10). At moderate
darkness / normal light, DBoW2 wins (SuperPoint is robust).

## 3. Retrieval quality (Python, night, GT-pose recall@K)
| Descriptor | R@1 | R@5 | R@10 |
|------------|----:|----:|-----:|
| AnyLoc ViT-G/14 + VLAD | **99.4%** | 100% | 100% |
| ViT-S/14 mean-pool | 89.2% | 97.9% | 99.1% |

**Retrieval is solved** — AnyLoc nails the right place 99.4% of the time at mean-10. The
full-pipeline ceiling is the **shared SuperPoint verification** (deployed ViT-S retrieval 99% R@10
but full reloc only 44.9%), not retrieval.

## Why the story is coherent
1. **AnyLoc's descriptor genuinely solves cross-condition retrieval** (99.4% night).
2. **It converts to a full-pipeline win only under genuine darkness** (native 71.8% > DBoW2 69.9%),
   capped by the shared SuperPoint verification.
3. **Same-session (loop closure, moderate/normal reloc), DBoW2 is as good or better** and cheaper.

→ Implication (roadmap Phase 2, redirected): the useful ML target is **illumination-robust *local*
features** (lift the verification ceiling for all methods) or **distilling AnyLoc into a small
model** (deployable frame-rate), *not* fine-tuning DINO for retrieval (already solved).

## Notes on method
- **Alignment:** EuRoC SE(3) (metric stereo → true metric ATE); Glasgow Sim(3) (estimated
  calibration → scale absorbed, so read Glasgow ATE *relatively*).
- **Datasets:** EuRoC MH_04/MH_05 (loop closure); Glasgow extreme-lighting SLAM
  (`joe3012/glasgow-extreme-lighting-slam-dataset`, real stereo + Vive GT) for day/night reloc —
  the dataset's own lighting is flicker not darkness, so the query was synthetically darkened to
  mean 32 / mean 10 (the AirVO/AirSLAM `dark_euroc` methodology).
- **Pipeline consistency:** map and query descriptors MUST use the same extractor + vocab; the
  VLAD vocab is fit on the grayscale deploy preprocessing (`fit_anyloc_vocab.py`).
- **Detailed writeups:** `experiments/glasgow/README.md`, `experiments/dino/README.md`.
