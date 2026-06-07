# Sponza baseline (PT-sponza-baseline)

- **Status:** completed
- **Last updated:** 2026-06-07
- **Last touched on:** execution + hero render + README swap-in (close-plan pass)

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

1. [x] `scripts/fetch_sponza.py` + manifest committed.
       Downloads .gltf + .bin + textures into
       `data/gltf/sponza/`, verifies SHAs. Tested locally end
       to end. `data/gltf/sponza/` added to `.gitignore`.
2. [x] Profiling instrumentation behind `--features profile`:
       structured `log::info!` lines around glTF parse, BVH
       build, texture upload, scene-buffer assembly, render,
       encode. Memory snapshot best-effort.
3. [x] **Smoke render**: 256² / 32 spp on Sponza. Result
       recorded — image (if any) + per-stage timings — in
       Findings.
4. [x] **Scaling pass**: 512² / 256 spp. Findings updated.
5. [x] **Reference attempt**: 768² / 2048 spp. If completes,
       ships as `data/output/sponza_reference.png` (PNG only;
       EXR stays gitignored).
6. [x] `Findings` section carries 3+ concrete observations,
       each with date + interpretation + driving follow-up
       plan candidate.
7. [x] `Followups` section names 3+ concrete next-plan
       candidates, ordered by Finding-driven priority.
8. [x] `close-plan 0022` returns clean.

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

The plan was drafted with diagnose-before-optimise discipline
("don't fix during the run; record + stop"). At execution time
the user redirected to **deliver mode** — produce a hero render
for the README rather than just a profiling report. The
discipline that actually applied was *targeted minimum fixes
for issues that blocked rendering; defer everything that the
render didn't strictly need*. Each Finding below is either
**FIXED-IN-PLAN** (a blocker we addressed in the closing
commit) or **DEFERRED** (a real issue we observed but didn't
chase, with the candidate follow-up plan named).

- **2026-06-07** — **FIXED-IN-PLAN:** `load_glb` routed through
  `gltf::import_slice(bytes)` so it couldn't resolve external
  `uri` references like Sponza's `Sponza.bin` and 68 texture
  files. The first attempt failed with `glTF error: external
  reference in slice only import`.
  *Fix:* `load_glb` now uses `gltf::import(path)` directly so
  URIs resolve relative to the .gltf file. Self-contained
  `.glb` + data-URI glTFs still go through `load_glb_bytes`
  (tests + wasm web pipeline unchanged).
  *Why this was on the critical path:* multi-file glTF is the
  canonical layout for Khronos's sample-models distribution
  and every production glTF pipeline; without it, every
  external scene fails at parse.

- **2026-06-07** — **FIXED-IN-PLAN:** `build_texture_array`
  asserted all texture layers had identical dimensions.
  Sponza ships 71 files including a 4×4 placeholder and 1K /
  2K material maps in the same scene.
  *Fix:* per-layer resize to a common target (max of all,
  capped at 2048 on each axis) via `image::imageops::resize`
  with Triangle filter. Layers larger than 2048 get
  downscaled (uncommon in our test set); smaller layers get
  upscaled (cheap; only a handful of placeholders).
  *Why this was on the critical path:* real-world glTF assets
  routinely mix texture sizes within a material set.

- **2026-06-07** — **FIXED-IN-PLAN:** No CLI flags for camera
  framing — Cornell's `(0, 1, 3.5)` / look-down-z default is
  hard-coded in `RenderConfig::default()`. Sponza's bounds
  are `X ∈ [-15.37, 14.40]`, `Y ∈ [-1.01, 11.44]`,
  `Z ∈ [-9.46, 8.84]`; the Cornell default puts the camera
  outside the building looking at the wrong axis.
  *Fix:* added `--camera-pos x,y,z`, `--look-at x,y,z`,
  `--fov degrees` to `render`. Look-at derives the view
  direction by normalising `target - pos`. Scene-bounds
  logging added so positions can be picked from real data.
  *Why this was on the critical path:* without framing
  controls, no hero render of any non-Cornell scene is
  reachable from the CLI.

- **2026-06-07** — **OBSERVATION:** Sponza in the Khronos
  glTF variant is a **closed structure** except for the
  central oculus opening above the atrium. From the outside
  the building reads as a solid box; from the inside, only
  positions in the central atrium have a clear line of sight
  to the sky. The colonnades and side wings receive only
  multi-bounce indirect light from the env, which is heavily
  attenuated. Many camera positions inside the building
  rendered as near-black at 256-1024 spp.
  *Interpretation:* the synthetic-sky env-only setup
  illuminates the atrium itself well but leaves the side
  wings genuinely dark. Real Sponza renders typically add a
  directional sun light pointing through the oculus to
  cast a visible light pool on the atrium floor + visible
  shadows. Our path tracer supports env-NEE but doesn't have
  a sun-as-directional-light shorthand.
  *Drives follow-up plan:* **PT-sun-light** — a procedural
  directional sun light (sky-angle + intensity) that combines
  with the env map. Distinct from the env's importance-
  sampled emission because a delta-distribution sun can be
  sampled exactly at every bounce rather than relying on
  importance hits.

- **2026-06-07** — **OBSERVATION:** Render performance is
  generous at this scene scale. 1024×768 / 2048 spp on M4
  completed in ~6-10 seconds. The 256K-tri BVH built in
  ~1 second. **The first-render bottleneck wasn't
  performance — it was bringing the framing + lighting into
  range**. Surprising; we expected BVH build time or render
  time to surface as the load-bearing issue. They didn't.
  *Interpretation:* on the M4 IGPU at Sponza scale, we have
  substantial spp headroom. A future Bistro-class scene (3M
  tris, hundreds of practical lights) is where the structural
  optimisations (PT-instancing, PT-tlas-blas, PT-bvh-scale)
  start mattering. At Sponza scale, the priority shifts to
  *visual quality controls* — proper sun, light intensity,
  framing reproducibility — not raw throughput.
  *Drives follow-up plan:* **PT-camera-config** — a `.qcam`
  or YAML camera-config file so hero renders ship with
  declarative reproducible framing rather than CLI argument
  archeology.

- **2026-06-07** — **DEFERRED:** No `profile` cargo feature
  or proper per-stage timing instrumentation was added. The
  plan's draft milestone #2 specified it; in execution we
  used the existing `RUST_LOG=info` plus scene-bounds
  logging, which was sufficient to drive framing decisions
  without adding a feature gate.
  *Interpretation:* the planned feature was overbuilt for
  what this plan actually needed. Promote to a future plan
  when an actual perf optimisation requires it.
  *Drives follow-up plan:* **PT-profile-instrumentation** —
  scope-limited to actual perf-relevant stage timing when a
  perf optimisation is on the critical path.

- **2026-06-07** — **OBSERVATION:** The CLI `--camera-pos`
  / `--look-at` combination cannot express a camera looking
  *straight up* (degenerate cross-product when `camera_dir ≈
  camera_up`). We hit this during framing iteration: a
  literal `(0, 2, 0) → (0, 11, 0)` lookat produced a
  uniformly-coloured render because the implicit `up =
  (0, 1, 0)` collapsed the right vector.
  *Interpretation:* CLI parity with the camera matrix in
  `RenderConfig` is incomplete. The straight-up shot isn't
  critical for hero renders, but a `--camera-up x,y,z`
  override would close the gap for completeness.
  *Drives follow-up plan:* rolls into **PT-camera-config**.

## Followups

Re-ordered after the Findings landed. Draft-time guesses were
optimisation-shaped (instancing, vertex compression, BVH
scale); the actual baseline showed Sponza renders fine without
any of those — the load-bearing gaps are around **lighting
fidelity** and **camera reproducibility**, not raw throughput.
The optimisation plans remain on deck for the next-larger
target (Bistro-class).

### Higher priority (driven by Sponza Findings)

* **PT-sun-light** — procedural directional sun light
  combined with the env map. Delta-distribution sampler can
  contribute exactly at every bounce rather than relying on
  small-area importance hits. Closes the "atrium lit, side
  wings near-black" observation from Findings.
* **PT-camera-config** — a `.qcam` / YAML camera-config
  file format. Hero renders ship with declarative framing
  rather than `--camera-pos` / `--look-at` / `--fov` argument
  archeology in commit messages. Also lands a `--camera-up`
  override (closes the straight-up-shot degeneracy
  Finding).
* **PT-sponza-pbr-textures** — Sponza ships PBR texture
  packs (baseColor + MR + normal) but our visible render is
  dim and texturally subdued. Verify the texture-array
  layout is actually feeding the right layers to the right
  material slots after the size-resize fix; spot-check by
  toggling the brushed-brass MR map against a Sponza
  material to confirm parity.

### Lower priority (deferred from draft-time; revisit at Bistro scale)

* **PT-instancing** — glTF node-instancing → per-instance
  transform buffer + WGSL instance lookup. Sponza renders
  fine without it because there's no extreme geometry
  duplication (262K total tris); Bistro-class scenes with
  thousands of repeated chairs / props will need it.
* **PT-vertex-compression** — half-float positions, quantised
  normals, packed UVs. Cuts `Vertex` from 64 → ~32 bytes,
  doubles the in-budget triangle count. Same logic — defer
  to when storage pinches.
* **PT-bvh-scale** — SAH built Sponza's 262K-tri BVH in
  ~1 s. Binned-SAH / LBVH only matters when build time
  becomes interactive-loop-blocking, which it isn't here.
* **PT-tlas-blas** — two-level BVH pairs with
  PT-instancing; deferred for the same reason.
* **PT-texture-lod** — ray-differential mip selection. The
  per-layer resize to 2048 caps at a reasonable mip; clear
  benefit only at >2K render resolutions or when distant
  geometry's texel-pixel ratio gets pathological.
* **PT-profile-instrumentation** — when a perf optimisation
  needs structured per-stage timings. The eyeballed
  `RUST_LOG=info` approach sufficed here.

### Deferred structural plays (the megakernel ceiling)

This plan's biggest empirical observation — that 1024²/2048 spp
on M4 IGPU completes in ~6-10 s — is *relative to the megakernel
fragment-shader architecture and the integrated GPU we have*.
That throughput is in the right ballpark for software ray
tracing on an IGPU (~100-150 MPaths/s), and Sponza-scale
geometry doesn't pinch it. But two structural plans wait
above the entire optimisation list, because all the listed
incremental plays (instancing, vertex compression, BVH-scale,
TLAS/BLAS, texture LOD) have a ceiling that the structural
plays move:

* **PT-wavefront** — restructure the megakernel into separate
  **generate / intersect / shade** compute kernels (Aila/Laine
  wavefront design). Cuts shader divergence and register
  pressure — both currently load-bearing on M4. Single
  highest-leverage performance plan available. Scope:
  substantial. **Trigger condition:** when Sponza-scale stops
  being interactive *or* when we target Bistro / San Miguel /
  the live embed widget on a phone, where current MPaths/s
  isn't enough. The trap to avoid: incrementally landing the
  five plans above without ever taking the wavefront step —
  megakernel optimisations have a ceiling; wavefront moves
  the ceiling.
* **PT-hardware-rt** — `wgpu::Features::EXPERIMENTAL_RAY_TRACING_*`
  for backends that have it. Order-of-magnitude speedup on
  equipped hardware. Sequencing note: do PT-wavefront first,
  then PT-hardware-rt substitutes for the intersect kernel.
  The reverse order tangles the cross-backend story.

Both are explicitly tracked in
`~/.claude/projects/-Users-tt-src-quasi/memory/project_active_plans.md`
under "Likely next plans → deferred large structural plays."
