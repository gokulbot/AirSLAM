# AirSLAM Extensions — Personal Research Roadmap

> Personal, go-all-out project. Goal: deeply understand and extend a real SLAM system,
> learning foundation models, fine-tuning, factor-graph optimization, GPU acceleration,
> and semantic SLAM along the way. Open-source friendly. Not chasing novelty — chasing mastery.

## Vision

Take **AirSLAM** (illumination-robust point-line visual SLAM) and grow it into a
**semantic, foundation-model-powered SLAM system on a GTSAM incremental factor-graph backend.**
The backend is **GTSAM, extended where it counts** — custom **semantic factors** and a **multi-hypothesis
(MH-iSAM2) layer** built on GTSAM's iSAM2 — *not* a from-scratch clone. Effort goes into what doesn't exist
yet (semantics + MHT), not into re-deriving what GTSAM already does well.
The pieces are not separate hacks — they converge on one architecture and each phase de-risks the next.

## Tempo & depth (no deadline)

Time is not a constraint — so optimize for **understanding, not delivery speed**. That flips several defaults:
- **Understanding, not reinvention.** I already understand batch factor-graph optimization (L0–L2); the gap to close is *incremental smoothing (iSAM2)*. So understand iSAM2 by reading Kaess + studying GTSAM's `ISAM2` source and re-expressing AirSLAM's factors in GTSAM (Phases 3–4) — not by retyping a GTSAM clone. GTSAM is the backend and oracle both.
- **The real from-scratch work is the *extensions***, and they're first-class: **semantic factors** (Phase 6) and the **MH-iSAM2 multi-hypothesis layer** on GTSAM's iSAM2 (Phase 7) — plus implementing VLAD/metric-learning by hand (Phases 1–2). CUDA (Phase 8) is optional and out of the critical path.
- **Read the papers fully**, reproduce a key figure from each before building on it.
- **Write up each phase** (blog/notes) — teaching it back is the deepest test of understanding.
- Still ship per-phase milestones — depth-first ≠ never-finishing. Each phase ends with a working system + a writeup.

## The converged architecture

```
  stereo images
       │
       ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  MULTI-TASK FRONT-END NET   — one backbone, three heads        │   Phase 5 GO/NO-GO decides:
 │   ┌────────┐    ┌────────┐    ┌───────────────────┐            │   unified net  vs
 │   │ point  │    │ line   │    │ semantic head      │            │   geo-CNN + DINOv2 (split)
 │   │ head   │    │ head   │    │ (+ semantic feats) │            │
 │   └───┬────┘    └───┬────┘    └─────────┬─────────┘            │   (DINOv2 backbone, if used,
 └───────┼─────────────┼───────────────────┼──────────────────────┘    reused for loop closure —
    keypts+desc     line segs      per-pixel sem + sem feats            one forward pass)
         │             │                    │
   ┌─────▼─────┐  ┌────▼─────┐     ┌────────▼─────────┐
   │ point     │  │ line     │     │ semantic feature │
   │ matching  │  │ assoc    │     │ matching          │
   │(LightGlue)│  │(via pts) │     └────────┬─────────┘
   └─────┬─────┘  └────┬─────┘        ┌─────▼──────┐
         │             │              │  diff-PnP  │  soft matches → pose + covariance
         │             │              │ (EPro-PnP) │  (trainable end-to-end)
         │             │              └─────┬──────┘
         │             │                    │
   point reproj   line reproj    ┌──────────▼────────────┐
     factors        factors      │ SEMANTIC CHANNEL:      │
         │             │         │  • data-assoc GATING   │  decides which factors
         │             │         │  • dynamic rejection   │  exist → no double-count
         │             │         │  • [opt] pose factor   │  (down-weighted: shares pixels)
         │             │         │  • [opt] object lmk    │
         └──────┬──────┴──────────────────┬──────────────┘
                ▼                          ▼
 ┌──────────────────────────────────────────────────────────────┐
 │       FACTOR GRAPH  —  GTSAM  (+ your extensions)             │
 │   GTSAM: factors · LM · Bayes tree · iSAM2  (CPU, sequential)  │ ◀── loop candidates
 │   YOUR: semantic factors (Phase 6)                            │      (ambiguous → hypotheses;
 │   YOUR: MH-iSAM2 multi-hypothesis layer (Phase 7)            │       semantics PRUNE them)
 │   [optional] CUDA batched BA / hyp-eval (Phase 8)            │
 └───────────────────────┬──────────────────────────▲───────────┘
                         │                           │
              trajectory + semantic map    ┌──────────┴─────────────┐
                         │                 │ LOOP CLOSURE / RELOC    │
                         ▼                 │ DINOv2/AnyLoc global     │ ← DINO for cross-
              ┌────────────────────┐       │ descriptors → ANN(FAISS) │
              │ offline map refine │       │  + geometric verify      │
              │   → reusable map   │       └─────────────────────────┘
              └────────────────────┘
```

**Load-bearing principles:**
1. **One representation, reused everywhere** — the shared backbone feeds semantics *and* loop-closure retrieval in a single forward pass (the core efficiency bet).
2. **Semantics enter asymmetrically** — points/lines are landmarks (reprojection factors); semantics are a *gate/pose channel*. **Never double-count** the semantic-PnP pose against the geometric factors (shared pixels ≠ independent → overconfidence).
3. **CPU/GPU split** — iSAM2 + hypothesis-tree management stay sequential CPU; only the *batched* BA/hypothesis numerics go to GPU. GPU never lives *inside* iSAM2.
4. **Ambiguity is a feature** — semantic ambiguity → pose uncertainty (EPro-PnP) → multiple hypotheses (MH-iSAM2) → resolved by geometry + semantics.

> This is the north-star **target**. Everything upstream of the FACTOR GRAPH is provisional until **Phase 5** measures whether one representation can serve both fine geometry and coarse semantics (unified net vs. split backbones).

## Why extend GTSAM instead of rebuilding it

Earlier drafts built a from-scratch GTSAM clone (Phases 3/4/7/8). **Reconsidered and dropped.** I already
understand batch factor-graph optimization (L0–L2), so retyping it as a clone adds nothing; the real gap is
**incremental smoothing (iSAM2)**, and the real *new* work is **semantic factors** and **multi-hypothesis** —
neither of which is a from-scratch-optimizer problem:

- **Semantic factors are just new factor *types*.** Subclass a GTSAM factor, implement `evaluateError()` (residual + Jacobian) — this is what GTSAM is *for*. You don't own the machinery to add a factor. → **Phase 6**
- **MHT is *not* a factor — it's solver surgery.** Multi-hypothesis lives inside the Bayes-tree elimination (MH-iSAM2's "hypo-tree"); you extend GTSAM's iSAM2, you don't add an edge. This is the one genuinely deep piece — and it *requires* understanding incremental smoothing (the gap above), which is why closing that gap is a prerequisite, not a clone. → **Phase 7**

**GTSAM is the backend.** AirSLAM currently uses **g2o**, so the first backend task is a **g2o → GTSAM migration** (re-express point + line reprojection factors, hit ATE parity) — which is also *how* the factor graph is understood: by re-expressing it, not rebuilding it. Incremental smoothing is understood by reading Kaess iSAM2 + studying/using GTSAM's `ISAM2`, not reimplementing it. GTSAM doubles as the numerical **oracle** (regression-diff every phase). Everything stays inside the AirSLAM fork; the custom pieces (semantic factors, MHT layer) are in-tree, cleanly-separable modules.

**The one thing that would flip this back: CUDA.** GPU batching needs Struct-of-Arrays factor storage that GTSAM's OOP/virtual factors can't provide — so *if* speed demands it, CUDA forces owning the factor layout, as a **separate batched-BA / batched-hypothesis track** (not a GTSAM extension). It only accelerates the *batchable* numerics (parallel BA, evaluating many MHT hypotheses at once), never sequential iSAM2. Currently **optional / out of the critical path** (Phase 8) — reach for it when a real bottleneck (esp. many-hypothesis MHT) demands it, not by default.

**Open decisions:**
- *Migrate vs alongside* — swap AirSLAM's backend to GTSAM outright, or run GTSAM beside g2o during the transition. *Lean: migrate once point/line factors hit parity.*
- *iSAM2 flavor for MHT* — incremental MH-iSAM2 (extend GTSAM's iSAM2, CPU) vs a lighter batched/pruned scheme (GPU-friendly). *Decide at Phase 7 with the workload in hand.*

---

## Build order (each phase is independently useful)

### Phase 0 — Foundations *(do not skip)*
- **Goal:** Build & run stock AirSLAM, reproduce its published results, stand up an eval harness.
- **Do:** clone `sair-lab/AirSLAM`; get the deps working (ROS noetic, OpenCV 4.2, Eigen3, Ceres 2.0.0, g2o, TensorRT 8.6, CUDA 12.1); run on a couple of sequences; script ATE/RPE + runtime logging.
- **Learn:** the codebase layout — frontend (PLNet), tracking, backend (g2o/Ceres), offline map opt (DBoW2 in `/voc`), relocalization.
- **Done when:** you can reproduce baseline ATE on ≥2 datasets and have a one-command eval script.
- **Effort:** ~1 week (mostly build/deps pain).

### Phase 1 — DINO relocalization *(the quick win that became the testbed)*
- **Goal:** DINOv2/AnyLoc global descriptors for **cross-condition relocalization** — *not* a blanket DBoW2 replacement. (Finding: DBoW2 wins clean same-session loops; DINO/AnyLoc dominates day↔night. So DINO owns the illumination-change / reloc case; DBoW2 keeps same-session loops.)
- **Do:** ✅ **Python prototype done** — AnyLoc (ViT-G, value-facet, VLAD) vs AirSLAM's *actual* DBoW2 (`dump_bow` tools): clean loops DBoW2 0.86 > AnyLoc 0.77; Gardens Point day→night AnyLoc **0.99** ≫ DBoW2 0.34. **Next:** wire into C++ (DINOv2 ViT-S/B → ONNX→TensorRT engine, per-keyframe descriptor stored in the map, retrieval in the **relocalization** stage + geometric verify).
- **Learn:** VPR, VLAD aggregation, foundation-model features, ANN retrieval (FAISS).
- **Read:** AnyLoc; SALAD; "Loop Closure using AnyLoc VPR in DPV-SLAM" (arXiv 2601.02723).
- **Done when:** cross-condition reloc success beats DBoW2 on a day↔night / lighting-change sequence; ATE holds. *(Python retrieval bar already met.)*
- **Effort:** weekend (Python compare ✅) → 1–2 weeks (full C++).

### Phase 2 — Fine-tune DINOv2 for VPR *(the ML-skills detour)*
- **Goal:** Learn fine-tuning itself, using Phase 1 as the testbed.
- **Do:** ladder — frozen AnyLoc → head-only metric learning (GSV-Cities + Multi-Similarity loss, cache backbone feats for fast iteration) → LoRA/PEFT on the ViT → partial unfreeze.
- **Learn:** metric learning, hard mining, PEFT/LoRA, evaluation protocol.
- **Read:** GSV-Cities (amaralibey), SALAD (serizba), HF `peft`.
- **Done when:** fine-tuned ViT-S/B improves day-night Recall@5 vs frozen on Tokyo 24/7 / SVOX; dropped back into Phase-1 retrieval.
- **Effort:** 1–2 weeks.

### Phase 3 — Migrate backend g2o → GTSAM *(adopt, don't rebuild)*
- **Goal:** Move AirSLAM's optimization from **g2o** to **GTSAM** and hit ATE parity. Understand the factor graph by *re-expressing* AirSLAM's factors in GTSAM, not by cloning it.
- **Do:** re-express stereo point + line reprojection (and pose-graph/BA) as GTSAM factors; swap the backend; diff ATE against the stock g2o build on every eval sequence. GTSAM stays in as the numerical oracle.
- **Learn:** factor graphs, the SAM formulation, GTSAM's Factor/Values/graph API, custom factors, SE(3) Jacobians.
- **Read:** Dellaert "Factor Graphs for Robot Perception"; GTSAM examples + custom-factor docs.
- **Done when:** the GTSAM-backed build's ATE is within noise of stock AirSLAM on all eval sequences.
- **Effort:** 2–4 weeks (migration + parity, not from-scratch).

### Phase 4 — Master GTSAM's iSAM2 *(close the incremental-smoothing gap)*
- **Goal:** Close the one real knowledge gap — *incremental smoothing* — by reading, using, and dissecting GTSAM's iSAM2, not reimplementing it. This is the prerequisite for the MHT extension (Phase 7).
- **Do:** switch the Phase-3 backend to `ISAM2`; understand variable elimination → Bayes tree → fluid relinearization + partial re-elimination by tracing GTSAM's implementation and instrumenting it; confirm lower per-step latency vs batch.
- **Learn:** incremental inference, the Bayes tree, variable ordering, fluid relinearization — deeply enough to *modify* it in Phase 7.
- **Read:** Kaess et al. iSAM2 (IJRR 2012); GTSAM's `ISAM2` source (read to modify, not to copy).
- **Done when:** the incremental backend matches batch accuracy at lower per-step latency, and I can trace/explain every step of a Bayes-tree update.
- **Effort:** 2–4 weeks (reading + using the hard core, not rebuilding it).

### Phase 5 — Semantic head (SAM-distilled) *(frozen-backbone MVP first)*
- **Goal:** Get per-pixel semantics into the system cheaply, real-time.
- **Do:** auto-label sequences with **Grounded-SAM** (Grounding DINO + SAM for classes; SAM2 for temporal propagation) → train a seg head. **Start with PLNet backbone FROZEN + seg head only** to answer "are PLNet feats semantic enough?" If weak → DINOv2 shared backbone or separate fast seg net on keyframes.
- **Learn:** auto-labeling / distillation, multi-task heads, loss balancing (uncertainty weighting/GradNorm).
- **Read:** SAM / SAM2, Grounded-SAM, Semantic-Segment-Anything; Kendall multi-task uncertainty weighting.
- **Done when:** usable masks at frame/keyframe rate; documented go/no-go on integrated-head vs DINOv2-backbone vs separate-net.
- **Effort:** 2–4 weeks.

### Phase 6 — Semantic factors + data-association gating
- **Goal:** Add semantics to the factor graph (real semantic SLAM), at lower rate.
- **Do:** start with **semantic-labeled map points → data-association gating** (reject cross-class matches) + **dynamic-class rejection**. Optionally object landmarks (QuadricSLAM/CubeSLAM) later.
- **Learn:** semantic SLAM, custom factors (now in my own lib), multi-rate fusion.
- **Read:** DynaSLAM, DS-SLAM, Kimera; QuadricSLAM/CubeSLAM.
- **Done when:** ATE improves in dynamic-scene sequences (TUM/Bonn dynamic) vs the geometric-only system.
- **Effort:** 3–5 weeks.

### Phase 7 — MHT on GTSAM's iSAM2 *(the deep, novel contribution)*
- **Goal:** Add robustness + multi-hypothesis *on top of GTSAM's iSAM2*: keep multiple ambiguous loop/association hypotheses; let optimization + semantics resolve them. The one genuinely from-scratch backend piece — solver surgery, not a factor.
- **Do:** robust factors first (max-mixtures / switchable constraints / GNC — these *are* just factors) → then the **MH-iSAM2 "hypo-tree"** (shared subtrees across hypotheses) built by extending GTSAM's `ISAM2` internals; semantics (Phase 6) prune hypotheses. The ambiguous cross-condition matches AnyLoc surfaces (Phase 1) are exactly what this resolves.
- **Read:** Olson & Agarwal max-mixtures; Sünderhauf switchable constraints; **MH-iSAM2 (Hsiao & Kaess 2019)** — the reference structure to build on GTSAM.
- **Done when:** robust to perceptual-aliasing loops that break the single-hypothesis backend.
- **Effort:** 6–10 weeks (research-grade — extending GTSAM's iSAM2, not from scratch).

### Phase 8 — CUDA acceleration *(optional / when a bottleneck demands it)*
- **Goal:** GPU-accelerate the **batchable** numerics — batched hypothesis evaluation / BA — when the many-hypothesis MHT workload (Phase 7) becomes the bottleneck. NOT iSAM2 itself (sequential, stays CPU). Optional, off the critical path.
- **Do:** prototype with **Theseus** (batched differentiable NLS on GPU) to prove the throughput win; if it pays off, write a CUDA batched-BA path against a **Struct-of-Arrays factor layout** — the *one* place that forces owning the factor storage (a separate track, not a GTSAM extension): batched dense blocks → Schur complement → matrix-free PCG.
- **Learn:** GPU NLS, batched solves, Schur/matrix-free PCG, CUDA.
- **Read:** PBA/Multicore BA (Wu 2011), MegBA, DeepLM, Theseus, Ceres-CUDA.
- **Done when:** measurable speedup on the many-hypothesis workload vs the CPU path.
- **Effort:** open-ended (only if pursued).

---

## Cross-cutting decisions
- **Which shared backbone?** PLNet's (cheap, geometric, weak semantics) vs DINOv2 (semantic, heavy ViT, real-time risk). Phase 5's frozen-head MVP answers this empirically — decide then.
- **Real-time budget.** Stock AirSLAM: 73 Hz PC / 40 Hz embedded. Keep semantics/DINO on keyframes-or-lower; precompute offline where possible.
- **CPU vs GPU split.** iSAM2 + hypothesis-tree management → CPU. Batched hypothesis/BA numerics, retrieval, network inference → GPU.
- **Optimizer build choices.** Replace-vs-alongside GTSAM, Eigen-vs-fully-from-scratch linear algebra, full-API-mirror-vs-subset — see "Why build the optimizer." Lock the **SoA factor layout** at L1 (it gates GPU later).

## Candidate techniques (parking lot — try if needed)
- **Semantic feature matching** — *the idea: associate features by what they ARE (semantic identity), not by low-level texture* — so matching survives illumination change and texture loss that break photometric matching. Source of semantics = DINOv2 patch features (semantic; give coarse cross-image correspondence for free) and/or the Phase-5 seg head. This is the active-matching cousin of Phase-6 data-association gating.
  - *Where it wins:* robust association / loop closure / relocalization under lighting & texture change; rejecting cross-class matches.
  - *Hard limits (why it can't be the sole matcher):* (1) semantic features are geometrically **coarse** → poor sub-pixel localization; good for *association*, bad for the *triangulation/pose solve* that needs pixel-precise matches; (2) semantically-uniform regions (many identical "wall" patches) are **ambiguous** → perceptual aliasing on repeated structure; (3) a region that is textureless **and** semantically uniform has *no* distinguishing signal for any matcher — that's what PLNet **line** features are for.
  - *Verdict:* use semantics as a **matching prior / gate / disambiguator layered on geometric matching + lines**, not a standalone matcher. Lives in Phases 5–6; the shared-DINOv2 backbone is what makes it cheap.
  - *Considered & dropped: LoFTR / detector-free dense matching* — matches appearance, not semantics, so it doesn't serve the semantic-matching goal. (Note for the record: LoFTR is actually relatively *strong* in low texture, so "fails without texture" isn't the reason to skip it — "not semantic" is.)
- **Differentiable PnP for semantic/soft correspondences** — *how to get pose out of coarse semantic matches.* Standard PnP needs hard, pixel-precise 2D–3D matches; **differentiable/probabilistic PnP (EPro-PnP, BPnP; DSAC for reloc)** instead consumes *weighted / soft / distributional* correspondences and learns the weights end-to-end from a pose loss — a direct answer to "semantic features are geometrically coarse."
  - *Key reframe:* this is a **training-time** tool (learn a geometry-aware semantic front-end), **not** a runtime backend — the online solve stays classical BA/iSAM2 (own lib). At inference you deploy the learned weighting, not the differentiable solver.
  - *Best fits:* (a) learn to weight semantic correspondences for pose; (b) **object-level pose → object landmarks** (Phase 6, QuadricSLAM/CubeSLAM); (c) EPro-PnP emits a *pose distribution* whose multimodality on ambiguous semantic matches feeds **multi-hypothesis (Phase 7)** — semantic ambiguity → pose uncertainty → hypotheses.
  - *Limits:* doesn't manufacture info in ambiguous/blank regions (only represents the uncertainty); integration cost = train in PyTorch, distill into the real-time C++/TensorRT path.
- **Unified multi-task front-end (points + lines + semantics) — the "converged front-end" bet.** One shared backbone, three heads → geometric primitives (points, lines) *and* semantic features in a single forward pass; points/lines feed reprojection factors, semantic matches feed diff-PnP. Optionally trainable **end-to-end**: multi-task net → semantic matches → diff-PnP → pose loss → backprop into the semantic head (a geometry-aware semantic front-end). This is the north-star front-end that unifies the two entries above.
  - *Not symmetric in the graph:* points/lines → landmark **reprojection** factors; semantic-matches-via-diff-PnP → a **pose / relative-pose** factor whose information matrix = the PnP covariance (principled). **Watch double-counting:** the semantic-PnP pose and the geometric factors share pixels → not independent → naively summing over-counts information (overconfident estimate). Prefer semantics for **gating** (decides which reprojection factors exist, adds no independent measurement), or deliberately down-weight the semantic-pose factor.
  - *The hard tension (why it's a bet, not a given):* precise **keypoint/line localization** wants fine, local, high-res, sub-pixel features (CNN-friendly); **semantics** want coarse, global, invariant features (ViT-friendly). A plain patch-resolution ViT localizes keypoints poorly. Options: (a) **unified** net w/ hybrid CNN-stem + transformer + high-res heads (cheapest inference, highest negative-transfer risk); (b) **two backbones** (geometric CNN at frame rate + DINOv2 semantics at keyframe rate); (c) **shared stem, split pathways**. Plus multi-task loss balancing / negative-transfer risk and the 73 Hz budget.
  - *Decision gate:* this is precisely **Phase 5's go/no-go** — build the frozen-backbone seg-head MVP first, *measure* whether one representation serves both geometry and semantics, then choose unified-vs-split. Do not build the mega-net before that answer.
  - *Two architectures kept as live options (both just notes until Phase 5):* **(A) unified per-frame multi-task net** — one pass, max compute-sharing, negative-transfer risk; **(B) multi-rate split** — fast geometric net @ frame rate + semantic net @ **keyframe rate** (full semantic inference only at keyframes; cheap propagation between via SAM2 mask warping / reprojection for per-frame needs like dynamic rejection). (B) protects the 73 Hz budget and makes semantic factors sparse & complementary (less double-counting); (A) is the compute-optimal ideal *if* one backbone proves it can serve both.
- **Dense RGB-D mapping + learned depth — "a map you can navigate," and mono-as-RGB-D.** *The idea: give AirSLAM a per-pixel depth input and build a dense volumetric map (octree/OctoMap occupancy or TSDF/surfel) alongside the sparse point-line map.* Depth from either (a) a **real RGB-D / ToF / structured-light sensor** (TUM-RGBD, OpenLORIS D435i) or (b) a **learned depth head on the converged front-end** — one more head beside points/lines/semantics → metric depth from a single camera ("RGB-D from mono").
  - *What depth buys the SLAM:* instant **metric scale** and **single-view landmark init** (no triangulation baseline needed) → helps exactly the low-parallax / near-pure-rotation motion that starves stereo triangulation; plus a **dense reconstruction** a sparse cloud can't give — occupancy for planning/obstacle avoidance, surfaces for AR.
  - *Keep dense geometry OUT of the optimizer.* SLAM state stays the sparse factor graph; the dense octree/TSDF is a **downstream product** fused from optimized keyframe poses + depth and **re-deformed/regenerated** after loop closure & global BA move the poses. Keeps BA/iSAM2 cheap and the map globally consistent — the "sparse solve, dense render" split of ElasticFusion / BundleFusion / Kimera. **OctoMap** = memory-light multi-res occupancy for planning; **TSDF/surfel** = better surfaces. Likely an **early, self-contained module** (bolts onto keyframe poses, touches neither front-end nor optimizer) → a stop-anywhere win that predates the ML phases.
  - *Learned mono-depth is a prior, not a measurement.* Monocular depth is **scale-ambiguous & per-scene biased**; feeding it as a hard depth factor is dangerous. Use it (1) as a **down-weighted prior / initializer** for landmark depth, (2) for the **dense map only** (viz / planning), never as the pose-critical constraint. On a stereo or true-RGB-D rig, sensor depth supersedes it — learned depth is the play for **mono** deployment. Same **double-counting** caveat as diff-PnP: depth predicted from the same pixels feeding the reprojection factors isn't independent → soft prior, not additive evidence.
  - *Where it fits:* the **depth head** = a Phase-5-style multi-task extension (self-supervised photometric / stereo consistency, or supervised on RGB-D GT); the **dense octree map** = an independent bolt-on usable much earlier; **RGB-D input mode** = a small dataset/config change (AirSLAM's RealSense config already carries `depth_lower/upper_thr`). Ties into the unified multi-task front-end above (depth = 4th head) and Phase 6 (dense map + semantics → **semantic octree / object-level maps**).
  - *Early RGB-D-input track (recommended first depth step):* on **OpenLORIS D435i** (color+depth) or TUM-RGBD, add a depth-image reader + **single-view metric landmark init** — a small, self-contained module, independent of the ML and optimizer tracks. **Composes directly with Phase 1:** a DINO-relocalized frame can seed landmarks from its own depth with *no triangulation baseline*, so depth + cross-condition reloc reinforce each other — this is the concrete "**depth → localization**" payoff. A stop-anywhere win; do it whenever a depth experiment is wanted.
  - *DESIGN FORK — dense IN the factor graph vs sparse-solve/dense-render (Kaess).* The "keep dense OUT" bullet above is one school; the other (the "**everything in one factor graph**" ethos, Kaess / iSAM2 lineage) puts dense/depth constraints **into** the graph as factors so they *constrain* the poses — jointly optimal, and a superb custom-factor playground for the Phase 3–7 GTSAM work. Cost: per-pixel photometric factors blow the graph up (many, badly nonlinear). **Middle path if pursued: keyframe-based *semi-dense*** — depth / **plane** / surfel factors only at keyframes, not per-pixel — keeps iSAM2 tractable while staying all-in-one-graph. **Gate behind Phases 3–6** (GTSAM migrated + custom factors working): it's a capstone playground, not a starting point. *(If a specific Kaess paper is the target, slot it here.)*

## Datasets & evaluation
- **SLAM accuracy:** EuRoC, TUM-RGBD, OIVIO, UMA-VI, TartanAir, + AirSLAM's own illumination sets. Metrics: ATE, RPE, runtime/Hz.
- **VPR:** GSV-Cities (train), Pitts30k, MSLS, Tokyo 24/7, SVOX (day-night). Metrics: Recall@K, PR.
- **Dynamic / semantic:** TUM dynamic, Bonn dynamic. Metric: ATE vs geometric-only.
- **Dense mapping / depth:** TUM-RGBD & OpenLORIS (D435i depth), ICL-NUIM, ScanNet, ETH3D. Metrics: reconstruction accuracy/completeness, learned-depth AbsRel/RMSE, map memory footprint.
- **Loop closure:** precision-recall of proposals (false loops are catastrophic).
- **Optimizer:** per-layer diff vs GTSAM oracle (same graph → same solution within tolerance); per-step latency vs batch.

## Risks & mitigations
- *Breaking stock accuracy when swapping components* → keep Phase 0 baseline + regression eval after every phase.
- *Multi-task retrain degrading PLNet features* → frozen-backbone MVP first; re-validate SLAM after any joint training.
- *Auto-label noise* → confidence thresholds, manual spot-check, low-bar tasks (dynamic masking) first.
- *Custom optimizer drifting from correct* → diff against GTSAM oracle per layer; numeric-Jacobian checks at L0; never advance a layer until it matches.
- *Optimizer scope explosion* → build the thin vertical slice first (L0 + 2 factors + LM), hit Phase-3 parity, then deepen; resist building the Bayes tree before the batch solver works.
- *Scope/burnout* → every phase ships standalone; stop-anywhere is a feature.

## North star (stretch)
A clean open-source fork: **DINOv2-backbone, semantic, illumination-robust point-line SLAM on a from-scratch
GTSAM-like backend (own iSAM2 + multi-hypothesis + CUDA)**, with a reproducible benchmark suite — and a
writeup per phase. The optimization library is a cleanly separable in-tree module of the AirSLAM fork
(its own lib target) — not a separate repo; all work lands in the one AirSLAM tree.
