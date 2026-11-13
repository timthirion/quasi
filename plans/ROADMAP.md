# Quasi Roadmap (Rust)

## Mission

Build a high-quality global illumination renderer whose output is worth writing
up — polished technical blog posts and, ideally, novel research. Every feature is
chosen to be correct, measurable, and explainable.

This Rust implementation has one defining constraint that shapes everything:
**it runs in the browser.** Via `wgpu` → WebGPU and `wasm-pack`, the same renderer
that produces reference-quality images natively also drops into a blog post as a
**live, interactive widget**. A reader can orbit the Cornell Box, switch the
integrator, and watch the noise fall as samples accumulate. That interactivity is
the differentiator and the reason this implementation exists.

Design bias:
- **Correctness over features** — a result we can defend against a reference.
- **Measurability** — convergence, variance, MSE-vs-reference, timing are
  first-class (native harness).
- **One source, two targets** — native and web stay in lockstep; WebGPU is both
  the delivery vehicle and a subject worth writing about.

## Where we are today

**29 plans closed** (0001–0029). The renderer is a production-scale path tracer
with a full PBR surface + volume stack, running natively and in the browser, plus
a parallel real-time rasterizer driving motum's in-browser planner widget. 260
CPU-side tests pin the math; CI and Pages deploy are green on every push.

**Surfaces + materials.** Textured Lambertian, GGX microfacet conductors (Smith
G + Schlick Fresnel), smooth dielectrics with full Snell + unpolarised Fresnel +
TIR, and distance-modulated Beer-Lambert absorption inside glass. Normal /
roughness / metallic map ingest, MikkTSpace per-face tangents in WGSL, and a
KTX2 / Basis Universal decode path (`KHR_texture_basisu`).

**Participating media.** Homogeneous single-scattering fog with
NEE-through-volume shadow rays; heterogeneous clouds on 3-D density grids via
delta + ratio tracking with Henyey-Greenstein anisotropy; an OpenVDB ingest
pipeline so production cloud data (e.g. the Disney cloud dataset) drops straight
in.

**Light transport.** MIS + NEE from the start, multi-emitter NEE with a
power-weighted CDF, power-weighted env-vs-triangle emitter selection, HDR
equirectangular environment lighting, and a delta-distribution directional sun.
PCG / Halton / padded high-dimensional Sobol samplers, runtime-switchable.

**Sampling + post-process.** Variance-driven per-pixel adaptive sample
allocation; an analytic à-trous wavelet denoiser with edge-stopping on the
albedo / normal / depth AOVs, a settled tonemap-then-denoise ordering, and a
quantified halo metric guarding it; HDR bloom via a Kawase dual-filter mip chain
with a Unity-correct soft-knee, composited pre-tonemap.

**Scale.** SAH binned BVH on CPU with WGSL stack-walked traversal (~350× over
linear scan at 20 K triangles). Shipped scenes run to Crytek Sponza (~262 K
triangles, 68 PBR textures) and the Amazon Lumberyard Bistro Exterior (~2.8 M
triangles, 582 PBR textures), plus the Khronos ABeautifulGame chess scene.

**Delivery.** Embeddable as `create(host_id)` (default chrome with sampler /
integrator toggles + sample readout + reset) or `createHeadless(host_id)` (bare
canvas; embedder supplies UI). Per-instance renderers with observed canvas
sizing; converged, idle, and off-screen instances pause.

**Rasterizer track.** Forward-shaded instanced triangle pipeline, a small
geometry library (cube / sphere / cylinder), line + point overlays (depth-tested
or on-top), and a motum-shaped JSON scene API (`setWorldState` /
`setTrajectory` / `setTreeOverlay` / `setGoal` + `onGoalChanged`) with a
draggable goal handle that mouse-ray-casts onto the floor plane.

### Known gaps

- **Bloom is native-only.** Plan 0029 shipped it offline and never wired the
  widget, breaking the "both targets are first-class" rule. Tracked in
  [`0035`](0035-pt-bloom-verification.md) along with 0029's unmet verification
  riders.
- **PT-sky is blocked on a human.** The Hosek-Wilkie CPU math and scripts are
  in; vendoring the coefficient dataset and validating against it needs a local
  run. See [`0030`](0030-pt-sky.md).
- **The rasterizer track has been idle since 0002.** Three drafts
  ([`0032`](0032-rt-cluster-cull.md)–[`0034`](0034-rt-virtualized.md)) describe
  a virtualized-geometry arc that hasn't started.

## Plan + milestone conventions

- One `plans/NNNN-*.md` per concrete piece of work, zero-padded and globally
  incrementing across both tracks (**next free number: `0036`**).
- Within a plan, milestones use a **track prefix + a short semantic slug**:
  - **`PT-<topic>`** for path-tracer milestones (any plan whose work
    advances the offline path-traced renderer): e.g. `PT-bvh`, `PT-ggx`,
    `PT-cloud`, `PT-sobol-padded`.
  - **`RT-<topic>`** for real-time / rasterizer milestones: e.g.
    `RT-overlays`, `RT-motum-wire`.
  - Sequencing within a plan comes from the order of checkboxes in the
    plan doc; cross-plan ordering is the ROADMAP's job. The slugs
    themselves carry no ordinal — `PT-cloud` doesn't imply it happened
    after `PT-ggx`, only that both belong to the path-tracer track.
  - Pick clear topical names up front. Renaming a milestone after work
    starts pollutes the git log; if scope genuinely drifts, split into
    two milestones rather than rename one.
- The historical prefixes in plans `0001` (`M0–M4`), `0002` (`R0–R4`), and
  `0003` (`T0–T4`) stay as they were when those plans shipped — renaming
  shipped history doesn't earn its confusion cost. The `PT-<topic>` /
  `RT-<topic>` convention applies to plans `0004` onward.
- **Status vocabulary:** `proposed` → `active` → `completed`, plus `blocked`
  and `abandoned`. (`done` appears in plans 0001–0013, which shipped before the
  vocabulary settled; treat it as a synonym for `completed` and don't rewrite
  those headers.)
- A plan isn't done until it works **natively and on web**, unless the plan
  explicitly scopes itself native-only (e.g. the verification harness). Plan
  0029 is the cautionary example: it closed with a native-only feature and cost
  a follow-up plan.

## Phases

Phases are roughly ordered; boundaries are soft. Each becomes one or more
`plans/NNNN-*.md` as work starts.

### Phase 0 — Foundation: pixels on screen, native + web  ✅ done
`wgpu` device/queue, a fullscreen pass, and a render loop that runs both in a
native `winit` window and on an HTML canvas via `wasm-pack`. Proves the
dual-target pipeline before any rendering complexity. → plan 0001 (M0)

### Phase 1 — Cornell Box path tracer  ✅ done
A WGSL megakernel path tracer over an analytic Cornell Box (quads + emissive
light): progressive accumulation, Reinhard tonemap, orbit camera. Built **with
next-event estimation + MIS from the start** (the correct, low-variance baseline)
and selectable QMC samplers (PCG / Halton / Sobol). → plan 0001 (M1)

### Phase 2 — Output & measurement  ✅ done
AOVs (albedo / normal / depth), native image output (PNG + HDR EXR), and the
verification harness: image metrics (MSE / RMSE / rel-MSE) and a convergence
study (error vs. spp per sampler/integrator). This is the backbone for every
"how noisy / how converged" claim in a post. → plan 0001 (M2–M3)

### Phase 3 — Interactive blog demo  ✅ done
Package the renderer with `wasm-pack` into an embeddable widget: orbit camera,
sample-count readout, sampler/integrator toggles, live progressive refinement.
The first publishable artifact. → plans 0001 (M4), 0013

### Phase 3.5 — Real triangle geometry + acceleration  ✅ done
glTF-loaded triangle meshes + a SAH binned BVH with WGSL stack traversal. BVH
speedup measured at 348× over linear scan at 20 K triangles — the data point
later scenes extrapolate from. → plan 0003

### Phase 4 — Advanced transport  ✅ done
All five sub-phases shipped:

- **4a — PBR surface BSDFs** (GGX + dielectrics) → plan 0004, extended by 0015
  (normal/roughness/metallic maps), 0019 + 0027 (tangents), 0026 (KTX2).
- **4b — Participating media** → plans 0005 (homogeneous absorption + fog),
  0007 (Henyey-Greenstein), 0006 + 0008 (heterogeneous clouds, delta + ratio
  tracking), 0009 (VDB ingest).
- **4c — Padded high-dimensional Sobol** → plan 0012. Sobol now beats PCG in the
  convergence CSV instead of plateauing at 64 spp.
- **4d — Many-light sampling** → plans 0016 (power-weighted multi-emitter NEE),
  0020 (env-vs-triangle pick).
- **4e — Denoising** → plans 0017 (à-trous), 0018 (tonemap ordering), 0021
  (quantified halo metric).

### Phase 5 — Production scale, assets, and post-process  ✅ done
Not in the original plan; emerged once Phase 4 made big scenes worth rendering.
Environment-map illumination (0014), a delta-distribution sun (0023), the Sponza
baseline (0022), the chess (0024) and Bistro (0025) hero scenes with their
asset-pipeline prerequisites, adaptive sampling (0028), and HDR bloom (0029).
CI (0010) and the README rewrite (0011) landed here too.

### Phase 6 — Convergence quality, sky, and the real-time track  ← current
Three independent threads, in rough priority order:

1. **Finish what's open.** [`0030-pt-sky`](0030-pt-sky.md) is 3/5 and blocked on
   a local run; [`0035`](0035-pt-bloom-verification.md) closes 0029's
   verification debt and gets bloom onto the web target.
2. **Better denoising.** [`0031-pt-nfor`](0031-pt-nfor.md) adds NFOR alongside
   the à-trous denoiser — a second, stronger path now that specular caustics and
   volumetric noise exist to clean up. Pairs with research plan
   [`R0001`](research/R0001-tonemap-halo-bound.md).
3. **Wake the rasterizer track.** Plans 0032–0034 describe a virtualized-geometry
   arc (GPU cluster culling → software micropoly rasterization → cluster DAG +
   LOD + page streaming). Idle since 0002; the most ambitious remaining work and
   the strongest blog-post material on the RT side.

Also open: two research plans at hypothesis stage —
[`R0001`](research/R0001-tonemap-halo-bound.md) (tonemap halo bound) and
[`R0002`](research/R0002-param-driven-sampling.md) (parameter-driven sampling).
See [`research/README.md`](research/README.md).

## Active plans

- [`0030-pt-sky.md`](0030-pt-sky.md) — procedural Hosek-Wilkie sky
  (`PT-sky/*`). **blocked** — CPU math, bake, CLI wiring and the
  bit-identical-bake tripwire are in; coefficient-dataset vendoring +
  held-out validation need a local run. Remaining: `PT-sky/time-of-day`
  (Sponza dawn/noon/sunset triptych), `PT-sky/widget`.

- [`0035-pt-bloom-verification.md`](0035-pt-bloom-verification.md) —
  **active**, 1/6. `widget-smoke` shipped: CI now builds the interactive
  renderer on lavapipe and fails if it draws black. Remaining: Kawase kernel
  pin, bloom-off inertness, denoise × bloom halo interaction, the browser
  widget surface 0029 skipped, attacker pass.

## Proposed

- [`0031-pt-nfor.md`](0031-pt-nfor.md) — NFOR feature-weighted denoiser
  (Rousselle et al. 2016) alongside the à-trous path.
- [`0032-rt-cluster-cull.md`](0032-rt-cluster-cull.md) — GPU-driven cluster
  culling for the raster track.
- [`0033-rt-micropoly.md`](0033-rt-micropoly.md) — software rasterizer for
  sub-pixel triangles.
- [`0034-rt-virtualized.md`](0034-rt-virtualized.md) — full virtualized geometry
  (cluster DAG + LOD + page streaming).

## Done

Path-tracer track:

- [`0001-foundation.md`](0001-foundation.md) — Interactive Cornell Box path
  tracer (M0–M4). **2026-06-04**
- [`0003-triangle-meshes.md`](0003-triangle-meshes.md) — glTF meshes + SAH
  binned BVH (T0–T4). **2026-06-04**
- [`0004-pbr-and-textures.md`](0004-pbr-and-textures.md) — PBR materials +
  textures. **2026-06-04**
- [`0005-participating-media.md`](0005-participating-media.md) — Homogeneous
  media + fog. **2026-06-04**
- [`0006-clouds.md`](0006-clouds.md) — Heterogeneous media / clouds.
  **2026-06-04**
- [`0007-henyey-greenstein.md`](0007-henyey-greenstein.md) — HG phase function.
  **2026-06-05**
- [`0008-density-grids.md`](0008-density-grids.md) — Density-grid clouds.
  **2026-06-05**
- [`0009-vdb-ingest.md`](0009-vdb-ingest.md) — VDB ingest pipeline.
  **2026-06-05**
- [`0012-padded-sobol.md`](0012-padded-sobol.md) — Padded high-dimensional
  Sobol. **2026-06-05**
- [`0014-environment-map.md`](0014-environment-map.md) — Environment-map
  illumination. **2026-06-05**
- [`0015-pbr-maps.md`](0015-pbr-maps.md) — Normal / roughness / metallic maps.
  **2026-06-06**
- [`0016-many-lights.md`](0016-many-lights.md) — Multi-emitter NEE.
  **2026-06-06**
- [`0017-denoising.md`](0017-denoising.md) — Edge-aware à-trous denoiser.
  **2026-06-06**
- [`0018-denoise-tonemap.md`](0018-denoise-tonemap.md) — Tonemap-then-denoise
  ordering. **2026-06-06**
- [`0019-mikktspace.md`](0019-mikktspace.md) — Per-vertex tangents.
  **2026-06-06**
- [`0020-light-vs-env.md`](0020-light-vs-env.md) — Power-weighted
  env-vs-triangle pick. **2026-06-06**
- [`0021-denoise-halo-metric.md`](0021-denoise-halo-metric.md) — Quantified halo
  metric. **2026-06-07**
- [`0022-sponza-baseline.md`](0022-sponza-baseline.md) — Sponza baseline
  (~262 K tris). **2026-06-07**
- [`0023-pt-sun-light.md`](0023-pt-sun-light.md) — Delta-distribution sun.
  **2026-06-07**
- [`0024-pt-chess-showcase.md`](0024-pt-chess-showcase.md) — Khronos
  ABeautifulGame chess scene. **2026-06-07**
- [`0025-pt-bistro.md`](0025-pt-bistro.md) — Lumberyard Bistro hero (~2.8 M
  tris, 582 textures). **2026-06-07**
- [`0026-pt-ktx2.md`](0026-pt-ktx2.md) — KTX2 / Basis Universal decode.
  **2026-06-07**
- [`0027-pt-mikktspace.md`](0027-pt-mikktspace.md) — Per-face tangents in WGSL.
  **2026-06-07**
- [`0028-pt-adaptive.md`](0028-pt-adaptive.md) — Per-pixel adaptive sampling.
  **2026-06-15**
- [`0029-pt-bloom.md`](0029-pt-bloom.md) — HDR bloom post-process (Kawase
  dual-filter, Unity-correct soft-knee). **2026-07-30** — closed with 7/9
  milestones; remainder in [`0035`](0035-pt-bloom-verification.md).

Real-time / rasterizer track:

- [`0002-realtime-rasterization.md`](0002-realtime-rasterization.md) — Dual-
  pipeline split + real-time rasterizer (R0–R4). **2026-06-04**

Infrastructure:

- [`0010-continuous-integration.md`](0010-continuous-integration.md) — CI
  workflows. **2026-06-05**
- [`0011-readme-aesthetic.md`](0011-readme-aesthetic.md) — README rewrite.
  **2026-06-05**
- [`0013-browser-embed.md`](0013-browser-embed.md) — Blog widget embed
  pipeline. **2026-06-05**
