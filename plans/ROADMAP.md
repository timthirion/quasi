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

## Active plans

- [`0001-foundation.md`](0001-foundation.md) — Bring the renderer up to a complete
  interactive Cornell Box path tracer (Phases 0–3, staged as milestones). **active**
