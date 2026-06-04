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

An empty repository. The roadmap below describes building up to a complete,
interactive Cornell Box path tracer and the measurement tooling around it.

## Phases

Phases are roughly ordered; boundaries are soft. Each becomes one or more
`plans/NNNN-*.md` as work starts.

### Phase 0 — Foundation: pixels on screen, native + web
`wgpu` device/queue, a fullscreen pass, and a render loop that runs both in a
native `winit` window and on an HTML canvas via `wasm-pack`. Proves the
dual-target pipeline before any rendering complexity.

### Phase 1 — Cornell Box path tracer
A WGSL megakernel path tracer over an analytic Cornell Box (quads + emissive
light): progressive accumulation, Reinhard tonemap, orbit camera. Built **with
next-event estimation + MIS from the start** (the correct, low-variance baseline)
and selectable QMC samplers (PCG / Halton / Sobol).

### Phase 2 — Output & measurement
AOVs (albedo / normal / depth), native image output (PNG + HDR EXR), and the
verification harness: image metrics (MSE / RMSE / rel-MSE) and a convergence
study (error vs. spp per sampler/integrator). This is the backbone for every
"how noisy / how converged" claim in a post.

### Phase 3 — Interactive blog demo
Package the renderer with `wasm-pack` into an embeddable widget: orbit camera,
sample-count readout, sampler/integrator toggles, live progressive refinement.
The first publishable artifact.

### Phase 4+ — Advanced transport (shared with the broader vision)
Real geometry (triangle meshes + acceleration / hardware ray tracing where
available), richer BSDFs (GGX, dielectrics), many-light sampling, denoising. Each
becomes a post in its own right.

## Parallel track — real-time rasterization

A second renderer track runs in parallel with the path tracer. Quasi grows
from a single-pipeline path tracer into a **dual-pipeline renderer**: a
path-traced pipeline (above) for offline-quality stills and the convergence
story, plus a **real-time rasterized pipeline** for 60fps interactive scenes
that the path tracer can't serve.

The two pipelines stay **wholly separate** at the renderer layer — different
scenes, different shaders, different draw paths — and share only the
platform plumbing (`gpu` module: wgpu device / queue / surface, frame loop,
canvas attachment). Each pipeline ships its own browser instance type;
each widget picks one.

The driving consumer is [`motum`](https://github.com/timthirion/motum)'s
Phase 4: live in-browser planning demos (drag a goal, watch RRT-Connect
solve, scrub the resulting trajectory). The path-tracer track keeps its
own audience (the Cornell Box / convergence widget at `0001`'s M4).

## Active plans

- [`0002-realtime-rasterization.md`](0002-realtime-rasterization.md) — Dual-
  pipeline split + a real-time rasterized renderer (milestones R0–R4). **active**

## Done

- [`0001-foundation.md`](0001-foundation.md) — Interactive Cornell Box path
  tracer (M0–M4: pixels native+web, NEE+MIS over analytic quads, samplers +
  AOVs + native output, verification harness, embeddable widget). **done 2026-06-04**
- [`0003-triangle-meshes.md`](0003-triangle-meshes.md) — glTF-loaded triangle
  meshes + SAH binned BVH on the path tracer (T0–T4: glTF ingest, triangle
  WGSL over storage buffers, CPU BVH + WGSL stack traversal, 20k-triangle
  Cornell publishable artifact with 348× BVH speedup). **done 2026-06-04**
