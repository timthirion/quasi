# Sponza baseline (PT-sponza-baseline)

- **Status:** draft
- **Last updated:** 2026-06-07
- **Last touched on:** drafting the first step toward complex-scene rendering

## Goal

Pick one concrete reference scene, get it to render at any
quality, characterise the bottlenecks. The path tracer at HEAD
ships across plans 0001-0021 with the Cornell Box (32 tris) +
the Stanford bunny (~5K tris) as its working set. The goal
across the **next several plans** is to move Quasi toward the
class of scenes that established rendering pipelines treat as
benchmarks — hundreds of thousands to millions of triangles,
dozens to hundreds of emitters, full PBR texture stacks across
the model.

This plan is the first concrete step: pick **Crytek Sponza**
as a goldilocks first target, load it, render it, and surface
every failure mode as a measurable bottleneck **rather than
fix anything mid-flight**. The deliberate framing is
*diagnose before optimise* — optimising before measuring is
the canonical mistake. The `Findings` section accretes the
actual bottlenecks observed; subsequent plans inherit a
priority order driven by data, not guesses.

## Context

What's already in:

* glTF ingest with PBR materials (baseColor + MR + normal +
  per-vertex tangents), embedded texture arrays, smooth
  normals, SAH BVH built at load. Last exercised at the bunny
  scale (~5K tris); Cornell scale (~32 tris) is the daily
  test loop.
* WebGPU storage-buffer layout sits at the **8 storage buffer
  cap** (flagged in plan 0014). `Material` is 96 bytes,
  `Vertex` is 64 bytes (post-plan-0019 per-vertex tangents).
* Render entry: `cargo run --release -- render --scene <gltf>
  --width N --height N --spp N --out <basename>` produces a
  tonemapped PNG + a raw EXR. Reference-render convention:
  hero scenes at 768²/2048 spp; smoke renders at 256²/64 spp.
* Profiling instrumentation: nothing beyond `log::info!`
  lines in `run_render` covering scene-load + render time.
  No GPU timing, no per-stage breakdown, no memory snapshot.

What's specifically **not** in:

* **Geometry instancing.** Every glTF node with a mesh produces
  its own vertex/index buffer entries — repeated columns +
  arches inflate the vertex budget linearly.
* GPU profiling (wgpu timestamp queries, occlusion queries).
* CPU profiling around BVH build, glTF parse, texture upload.
* Streaming / progressive / out-of-core asset loading.

What this plan is **not**:

* **Not a fix-it plan.** The whole point is *don't fix anything
  during this plan*. Bugs go into `Findings`; fixes go into
  follow-up plans. Discipline around this is what stops the
  baseline from sprawling.
* Not a feature plan. No new BSDFs, samplers, materials, light
  types, denoiser variants.
* Not a hero render. The output may be ugly, slow, or partly
  broken. The deliverable is the **diagnostic report**, not
  the visual.
* Not a benchmark comparison against another renderer. We
  measure Quasi against Quasi's own ambitions.

## Design

### Why Sponza first

Crytek Sponza is the right first target because:

1. **Triangle count is in the goldilocks band.** ~262K
   triangles — large enough to surface scale issues that
   don't show up on 5K tris, small enough to fit in WebGPU's
   default storage-buffer ceilings without first needing
   vertex compression.
2. **A glTF distribution exists** at
   `KhronosGroup/glTF-Sample-Models/2.0/Sponza` — native to
   the existing ingest pipeline. No Assimp dependency, no
   format adapter, no external HDR fetch.
3. **PBR texture stack is non-trivial.** ~25 textures across
   baseColor / metallic-roughness / normal channels. Exercises
   the texture-array path that plan 0015 PT-pbr-maps shipped.
4. **Recognisable.** Once it renders at any quality, the
   README's hero gallery gains a scene every reader will
   identify on sight.
5. **A natural predecessor.** If Sponza loads cleanly + renders
   in some reasonable time, the path to harder targets is
   incremental — multi-million-tri interiors, scenes with
   hundreds of practical lights, production-grade asset
   stacks. If it doesn't, this plan's Findings tell us
   exactly which plumbing to fix first.

### Asset acquisition

The Sponza glTF lives under the Khronos sample-models repo.
The asset is large enough (~100 MB inc. textures) that
committing it would re-inflate the repo by an order of
magnitude — exactly what the EXR-untracking work from commit
`5e527a1` just removed.

Convention:

* `scripts/fetch_sponza.py` (pure-stdlib Python, matches the
  existing `scripts/` Python convention) one-shot downloads
  the .gltf + .bin + texture files into `data/gltf/sponza/`,
  verifying expected SHAs against a manifest committed
  alongside the script.
* `.gitignore` excludes `data/gltf/sponza/` so the asset
  lives only on the contributor's disk + CI's runner.
* The manifest itself (slugs + SHAs + Khronos source URLs) is
  small + auditable, so it commits.

### Profiling instrumentation

Minimal cycle-accurate timing without adopting a profiling
crate dependency. Behind a `profile` cargo feature so the
hot path stays unaffected by default.

Brackets:

* `std::time::Instant` start/stop around:
  - glTF parse (`load_glb`)
  - BVH build (`pathtrace::bvh::Bvh::build`)
  - Texture upload to GPU
  - Scene-buffer assembly (`build_scene_buffers_*`)
  - Per-frame render (the existing `start = Instant::now()`
    bracket already captures this; just structured-log it)
  - PNG + EXR encode
* Log each as `log::info!("profile: stage=X duration_ms=Y")`
  so the output can be `grep "profile:" | awk` into a CSV.

Peak memory: try `proc/self/status` on Linux, `mach_task_info`
on macOS; if either turns into its own rabbit hole, defer to
a future plan and record eyeballed peak from Activity Monitor
/ htop in the plan's Findings. The point is *some* number,
not perfect numbers.

### Render protocol

Three attempts, each gated on the previous one completing.
The protocol is **load → render → record → stop**; do not
debug during the protocol.

1. **Smoke render:** 256² / 32 spp. Cheapest meaningful run.
   If this fails, we have our first Finding and we stop.
2. **Scaling pass** (if smoke succeeded): 512² / 256 spp.
   Tells us how render time scales with spp + resolution.
   If memory pinches here, Finding + stop.
3. **Reference attempt** (if scaling succeeded): 768² / 2048
   spp. The hero-class quality. If this completes, the
   resulting PNG ships as `data/output/sponza_reference.png`
   (the EXR stays gitignored per `data/output/README.md`).

If *any* attempt produces visible artefacts — clipped
geometry, missing textures, blown-out lights, NaN pixels —
record the artefact in Findings with a screenshot + the
specific input that triggered it, and **proceed without
fixing**.

### Findings shape

Same pattern as the research plans under `plans/research/`:
each finding is its own bullet, prefixed with the date
observed, with three lines of structure:

```
- **YYYY-MM-DD** — Observation. <one or two concrete sentences
  with numbers / file paths / error messages>.
  Interpretation. <what this implies about the bottleneck>.
  Drives follow-up plan: <candidate plan name, sketched>.
```

The Findings drive the close-plan pass's "Followups" section
+ the next several plan numbers.

## Milestones

1. [ ] `scripts/fetch_sponza.py` + manifest committed.
       Downloads .gltf + .bin + textures into
       `data/gltf/sponza/`, verifies SHAs. Tested locally end
       to end. `data/gltf/sponza/` added to `.gitignore`.
2. [ ] Profiling instrumentation behind `--features profile`:
       structured `log::info!` lines around glTF parse, BVH
       build, texture upload, scene-buffer assembly, render,
       encode. Memory snapshot best-effort.
3. [ ] **Smoke render**: 256² / 32 spp on Sponza. Result
       recorded — image (if any) + per-stage timings — in
       Findings.
4. [ ] **Scaling pass**: 512² / 256 spp. Findings updated.
5. [ ] **Reference attempt**: 768² / 2048 spp. If completes,
       ships as `data/output/sponza_reference.png` (PNG only;
       EXR stays gitignored).
6. [ ] `Findings` section carries 3+ concrete observations,
       each with date + interpretation + driving follow-up
       plan candidate.
7. [ ] `Followups` section names 3+ concrete next-plan
       candidates, ordered by Finding-driven priority.
8. [ ] `close-plan 0022` returns clean.

## Open questions

* **Which Sponza variant?** Khronos hosts the classic Crytek
  Sponza. Intel ships an enhanced variant ("Sponza-New") with
  richer materials + more lights. Default to the classic;
  Intel's variant is a natural follow-up once the classic
  renders. Decision pinned at draft time so render-attacker /
  defender pairs at close don't drift on it.
* **Commit a baseline PNG to the repo?** Yes if the reference
  attempt completes. `denoise_comparison.png` set the
  precedent for committing post-process PNGs alongside the
  hero gallery; a 768² Sponza PNG (~1 MB) is in the same
  budget. The EXR stays out per `data/output/README.md`.
* **Do we need a memory-profiling crate (dhat / jemalloc)?**
  Defer. Eyeballed peak + per-stage timings are enough to
  surface load-bearing bottlenecks at this stage. Promote to
  a plan if + when we can't otherwise localise a memory
  failure.
* **Plan-level discipline around "don't fix during the run."**
  Inevitable temptation: a tiny bug surfaces, a 5-line fix
  closes it, render proceeds. The plan's Goal pins the
  diagnose-before-optimise rule explicitly so the close-plan
  pass can flag drift if it surfaces during the diff review.

## Done when

* `cargo run --release -- render --scene data/gltf/sponza/Sponza.gltf
  ...` either produces an image *or* surfaces a specific
  failure mode that's recorded in `Findings` with enough
  detail to drive the next plan.
* The `profile` cargo feature produces structured per-stage
  timing log lines on demand. Default-off build is unchanged
  performance-wise (modulo `cfg`-gating overhead, which should
  be zero in optimised release builds).
* `Findings` section carries 3+ concrete observations.
* `Followups` section names 3+ concrete next-plan candidates,
  prioritised by what actually pinches.
* `close-plan 0022` orchestration returns clean: plan-skeptic
  raises no unaddressed P0, code-attacker / defender pair
  resolves all P0 attacks (refute or accept-with-fix), and
  render-attacker / defender pair runs on
  `sponza_reference.png` if it ships.
* CI green at HEAD.

## Findings

*(empty at draft — accretes during the render protocol.)*

## Followups

Educated guesses at draft time, ordered by likely-to-pinch-first.
The actual order after the Findings come in may differ
substantially.

* **PT-instancing** — glTF node-instancing → per-instance
  transform buffer + WGSL instance lookup. Without this,
  every repeated column / arch / chair in a Sponza-class
  scene re-uploads identical geometry.
* **PT-vertex-compression** — half-float positions, quantised
  normals, packed UVs. Cuts `Vertex` from 64 → ~32 bytes,
  doubles the in-budget triangle count for the same
  storage-buffer footprint.
* **PT-bvh-scale** — if SAH build is too slow at 262K tris
  (and especially at 1M+), switch to binned-SAH or LBVH for
  fast build; profile traversal cache behaviour against the
  larger node set.
* **PT-tlas-blas** — two-level BVH (TLAS over instance
  transforms → BLAS over per-mesh geometry). Pairs with
  PT-instancing for the structural win on repeated geometry.
* **PT-texture-lod** — ray-differential-driven mip selection
  via `textureSampleGrad`. Cuts texture noise + aliasing on
  distant geometry; matters more at higher resolutions where
  texel-to-pixel ratios get tight.
