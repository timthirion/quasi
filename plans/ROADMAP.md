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

The renderer is a complete, interactive Cornell Box path tracer running natively
and in the browser, plus a parallel real-time rasterizer driving motum's
in-browser planner widget. Key pieces shipped:

- **Path tracer**: NEE + MIS over triangle meshes (glTF-loaded), SAH binned BVH
  with WGSL stack traversal, selectable samplers (PCG / Halton / Sobol),
  selectable integrators (MIS+NEE / pure BSDF), full AOV output (radiance /
  albedo / normal / depth), native PNG + multi-channel HDR EXR via parallel
  scoped-thread encoding, and a verification harness (image metrics +
  convergence sweep CSV).
- **Publishable artifact**: a Stanford bunny inside a Cornell box, shipped at
  512² / 1024 spp with a fresh convergence CSV. Plus a procedural icosphere
  scene that demonstrates 348× BVH speedup at 20k triangles.
- **Browser widget**: embeddable as `create(host_id)` (default chrome with
  sampler / integrator toggles + sample readout + reset) or
  `createHeadless(host_id)` (bare canvas; embedder provides UI).
- **Rasterizer**: forward-shaded instanced triangle pipeline with a small
  geometry library (cube / sphere / cylinder), line + point overlays
  (depth-tested or on-top), and a motum-shaped JSON scene API
  (`setWorldState` / `setTrajectory` / `setTreeOverlay` / `setGoal` +
  `onGoalChanged`) with a draggable goal handle that mouse-ray-casts onto
  the floor plane.

Plans done: [`0001-foundation`](0001-foundation.md) (M0–M4),
[`0002-realtime-rasterization`](0002-realtime-rasterization.md) (R0–R4),
[`0003-triangle-meshes`](0003-triangle-meshes.md) (T0–T4). No plans are
currently in flight; Phase 4 is open.

## Plan + milestone conventions

- One `plans/NNNN-*.md` per concrete piece of work, zero-padded and globally
  incrementing across both tracks (next free number: `0004`).
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

## Phases

Phases are roughly ordered; boundaries are soft. Each becomes one or more
`plans/NNNN-*.md` as work starts.

### Phase 0 — Foundation: pixels on screen, native + web  ✅ done
`wgpu` device/queue, a fullscreen pass, and a render loop that runs both in a
native `winit` window and on an HTML canvas via `wasm-pack`. Proves the
dual-target pipeline before any rendering complexity.

### Phase 1 — Cornell Box path tracer  ✅ done
A WGSL megakernel path tracer over an analytic Cornell Box (quads + emissive
light): progressive accumulation, Reinhard tonemap, orbit camera. Built **with
next-event estimation + MIS from the start** (the correct, low-variance baseline)
and selectable QMC samplers (PCG / Halton / Sobol).

### Phase 2 — Output & measurement  ✅ done
AOVs (albedo / normal / depth), native image output (PNG + HDR EXR), and the
verification harness: image metrics (MSE / RMSE / rel-MSE) and a convergence
study (error vs. spp per sampler/integrator). This is the backbone for every
"how noisy / how converged" claim in a post.

### Phase 3 — Interactive blog demo  ✅ done
Package the renderer with `wasm-pack` into an embeddable widget: orbit camera,
sample-count readout, sampler/integrator toggles, live progressive refinement.
The first publishable artifact.

### Phase 3.5 — Real triangle geometry + acceleration  ✅ done (plan 0003)
glTF-loaded triangle meshes + a SAH binned BVH with WGSL stack traversal. The
Cornell Box is now shipped as a glTF; the Stanford bunny renders at 4,968
triangles, the procedural icosphere stress test at 20,492. BVH speedup measured
at 348× over linear scan at 20k triangles — the data point future scenes
extrapolate from.

### Phase 4 — Advanced transport

Plan 0001's "Phase 4+" placeholder is now broken into focused sub-phases. Each
sub-phase is intended to become one `plans/000N-*.md` and one blog post. The
order below is the recommended sequence (visible payoff → convergence quality →
scale → polish); 4a and 4b are independent and either can lead.

#### Phase 4a — PBR surface BSDFs (GGX + dielectrics)
The Lambertian world hits its ceiling fast. GGX microfacet metals + dielectrics
turn the bunny into brushed steel or glass — every existing test scene gives a
fresh publishable image without adding any geometry. Materials' `roughness` /
`metallic` fields are already parsed at glTF load (0003 T0); the shader just
doesn't read them yet. NEE through specular bounces (light-bsdf MIS) gets
non-trivial in the rough-mirror limit; this is the right plan to nail it in.

**Publishable artifact:** the bunny rendered as brushed steel, as rough gold,
and as a single glass scattering volume — a 3-pane "material study" image.

#### Phase 4b — Participating media (the "cloud" track)
The biggest new visual capability after 4a. Volumetric path tracing renders
absorption / scattering / emission **between** surfaces — fog, smoke, clouds.
Staged:

1. **Homogeneous absorbing medium** — Beer-Lambert attenuation through a
   bounded region. Looks like a tinted shadow; cheapest correct step.
2. **Phase function + isotropic single scattering** — Henyey-Greenstein
   anisotropy parameter `g`. Volumetric NEE samples a point inside the medium
   and connects to a light through the surrounding medium.
3. **Heterogeneous medium** — density defined by a 3-D field (procedural
   fbm noise to start; an external `.vdb` ingest follows if useful).
   Delta tracking / ratio tracking gives unbiased free-flight sampling
   without inverting the optical depth integral.
4. **Multi-scattering convergence** — proper bounce-through-volume paths;
   this is where the canonical visual lands (silver-lining edges + soft
   ambient interior).

**Publishable artifact:** a Cornell box with a small cloud in it, backlit by
the ceiling area light, plus a separate "cloud on black" study showing the
silver-lining + ambient-interior story isolated.

#### Phase 4c — Padded high-dimensional Sobol
The 2-dim Sobol shipped in 0003 plateaus by 64 spp because consecutive
`next_2d` calls within one path read consecutive (correlated) Sobol points.
Joe-Kuo direction numbers for ≥16 dimensions, with a per-call dimension index
threaded through the integrator, turns Sobol from "trails PCG" to "beats it"
in the convergence CSV — and gives the convergence-blog post a clean
"low-discrepancy actually wins" panel.

#### Phase 4d — Many-light sampling
Becomes interesting once scenes grow past a single area light. Power-weighted
random pick + light BVH; ReSTIR / spatial reuse if the gains justify them.

#### Phase 4e — Denoising
A learned or analytic denoiser working from the albedo / normal / depth AOVs
already shipped in M2. Mostly matters once 4a/4b create variance worth
denoising — until there's specular caustics or volumetric noise to clean up,
brute-force converges fast enough.

## Active plans

_(none — Phase 4 sub-plans drafted as they start)_

## Done

- [`0001-foundation.md`](0001-foundation.md) — Interactive Cornell Box path
  tracer (M0–M4: pixels native+web, NEE+MIS over analytic quads, samplers +
  AOVs + native output, verification harness, embeddable widget). **done 2026-06-04**
- [`0002-realtime-rasterization.md`](0002-realtime-rasterization.md) — Dual-
  pipeline split + real-time rasterizer (R0–R4: module split, forward triangle
  pipeline, instanced scene, line/point overlays, motum-shaped JSON API +
  draggable goal handle). **done 2026-06-04**
- [`0003-triangle-meshes.md`](0003-triangle-meshes.md) — glTF-loaded triangle
  meshes + SAH binned BVH on the path tracer (T0–T4: glTF ingest, triangle
  WGSL over storage buffers, CPU BVH + WGSL stack traversal, 20k-triangle
  icosphere stress test + Stanford bunny publishable artifact). **done 2026-06-04**
