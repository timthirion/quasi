# Multi-emitter NEE (PT-many-lights)

- **Status:** draft
- **Last updated:** 2026-06-06
- **Last touched on:** planning

## Goal

Replace the uniform-random pick over `emissive_triangles` with a
**power-weighted Bernoulli pick** so bright, large triangles attract
proportionally more NEE samples than dim, small ones. Closes the
Phase 4d hole in the ROADMAP and is what unlocks scenes with
multiple emitters of mismatched intensity without one of them
silently driving the variance.

Pairs naturally with PT-env (additive multi-light already in
place — env + triangle NEE both fire every step). The env half
already uses an importance distribution; the triangle half stops
being the dumb uniform pick that drags variance down at low spp.

## Context

What's already in:

* `sample_light` in `pathtrace.wgsl` uniformly picks an index from
  `emissive_triangles` (1 / N each), samples a barycentric point
  on the picked triangle, and returns `pdf_w = d² / (cos · A · N)`.
* `emissive_triangles: array<u32>` carries the GPU-side indices.
  CPU mirror: `TriangleScene::emissive_triangles`.
* The MIS pattern (`power_heuristic(pdf_light, pdf_bsdf)`) already
  treats `pdf_w` as the light-sample pdf in solid-angle measure —
  swapping the pick distribution just rewrites the `1/N` factor.

What this plan is **not**:

* Light BVH or hierarchical importance sampling. Flat CDF is fine
  until scenes carry >>100 light triangles; the current showcase
  scenes carry 2 (Cornell ceiling = single quad = 2 triangles).
* ReSTIR / spatial-temporal reuse. Variance reduction past the
  baseline power-weighted pick wants its own plan.
* Light groups. We treat every emissive triangle as an
  independent emitter; aggregating two co-planar triangles into
  one "panel emitter" with its own quad sampler is an
  optimisation past the scope here.

## Design

### Power-weighted CDF

Compute at scene-build time:

```rust
pub struct LightCdf {
    /// One entry per emissive triangle. `cdf[i]` = cumulative
    /// power through triangle `i`, normalised so `cdf[N-1] == 1`.
    pub cdf: Vec<f32>,
    /// Total emitted power, before normalisation. Used by the
    /// PT-env light-vs-env pick (a follow-up; not in this plan).
    pub total_power: f32,
}
```

Per-triangle power = `area · max(emission_r, emission_g,
emission_b)`. Area is `0.5 · |edge1 × edge2|` (the same formula
already in WGSL's `triangle_area`). Cumulative sum left-to-right,
normalise.

Upload as a new storage buffer `light_cdf: array<f32, N>`. Bind
group entry is — *carefully* — the 9th storage buffer if we don't
unpack any existing ones. Plan 0014 noted we sit at 8 (the
WebGPU default cap). Either:

1. Pack the `f32` CDF into the existing `emissive_triangles`
   buffer by widening the array element to `vec2<u32>` (tri
   index + cdf bits) → 1 buffer instead of 2. **Going with this.**
2. Or relax the device limit on native + fall back to uniform
   pick on wasm. Loses cross-target parity — reject.

Layout becomes `emissive_lights: array<EmissiveLight>` where
`EmissiveLight = { tri: u32, _pad: u32, cdf: f32, _pad2: f32 }`
(16 bytes, std430 friendly, no extra buffers).

### WGSL inverse-CDF sample

```wgsl
fn pick_emissive(xi: f32) -> u32 {
    if (U.emissive_count == 0u) { return 0u; }
    if (U.emissive_count == 1u) { return 0u; }
    var lo: u32 = 0u;
    var hi: u32 = U.emissive_count - 1u;
    loop {
        if (hi - lo <= 1u) { break; }
        let mid = (lo + hi) >> 1u;
        if (emissive_lights[mid].cdf <= xi) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // Largest index with cdf ≤ xi → pick the *next* one (the
    // bin xi falls into).
    if (emissive_lights[lo].cdf <= xi) { return lo + 1u; }
    return lo;
}
```

`sample_light` consumes the picked index and `picked_prob =
power[i] / total_power = cdf[i] - cdf[i-1]`. The `pdf_w` then
becomes `(d² / (cos · A)) · (1 / picked_prob)` instead of
`(d² / (cos · A · N))`.

### CPU mirror

`TriangleScene::build_light_cdf(&self) -> LightCdf` mirrors the
WGSL pick exactly (so tests can pin "scene with two lights of 10:1
power ratio attracts samples in a 10:1 ratio within MC
tolerance"). Tests live in `tests/lights.rs`.

### Sentinel: no lights

When `emissive_count == 0` the existing scenes skip NEE entirely
(early return in `sample_light`). Nothing changes there. When
`emissive_count == 1` (a single emitter), the CDF degenerates to a
single bin with cdf=1 and the pick is always 0 — equivalent to
the old uniform pick. No regression at low light counts.

### Cornell scenes are byte-stable

Every Cornell scene has exactly **one ceiling quad** = 2 emissive
triangles. With 2 triangles of identical area + identical
emission, the power-weighted pick is `(0.5, 1.0)` cdf — same as
uniform `1/2`. Cornell renders byte-stably modulo the floating-
point ε between `f * 0.5` and `f * (1/2)`.

## Milestones

### PT-power-pick
- [ ] `pathtrace::lights::LightCdf` struct + builder on
      `TriangleScene` (CPU). Per-triangle power = area × max
      emission channel.
- [ ] `EmissiveLight` struct (16 bytes std430). Replace the
      `emissive_triangles: Vec<u32>` field on `TriangleScene`
      with `emissive_lights: Vec<EmissiveLight>`.
- [ ] `Hit::tri` already there → just plumb the new buffer into
      the existing `@binding(5)` slot. No bind-group resize; the
      element type changes from `u32` to `EmissiveLight`.
- [ ] WGSL `pick_emissive(xi)` binary-search + `sample_light`
      pdf correction.
- [ ] CPU mirror test: CDF over a synthetic 3-light scene is
      monotone, ends at 1; pick frequency at 10k samples
      matches the analytic distribution within MC tolerance.
- [ ] Existing Cornell scenes render byte-stably (or within the
      power-of-two ε noise from the cdf rounding).

### PT-many-lights-scene
- [ ] New `cornell_many_lights.gltf` — Cornell room with the
      ceiling light replaced by **three** mismatched panels:
      one large dim panel + two small bright spots. Power ratio
      ≈ (3, 1, 1) with size ratio reversed so power-weighting
      and area-weighting are clearly distinguishable.
- [ ] Reference render at 512² / 1024 spp →
      `data/output/cornell_many_lights_reference.png`. The
      shadow geometry should clearly show three light sources.
- [ ] Convergence script (`scripts/converge_lights.py` or
      similar) compares uniform-pick spp-to-target-RMSE with
      power-pick at the same target. Headline number lands in
      the plan's "Done when" block.

## Open questions

- **Mixed env + triangle pick.** Already additive (independent
  env NEE + triangle NEE every step). A power-weighted pick
  between the two ("env vs triangles") would consolidate that —
  but with our scenes carrying either env *or* triangle, not
  both, the additive cost is two cheap samples and we don't
  push it now.
- **CDF in std430.** WebGPU promotes `f32` to a 16-byte slot
  inside a `vec4`-friendly storage buffer. Packing as
  `EmissiveLight { tri, _pad, cdf, _pad }` is wasteful but
  std430-aligned. Tighter layouts (8-byte struct with
  `tri: u32, cdf: f32`) work on most adapters but need device
  capability checks; defer.
- **Power = area × max emission channel.** Strictly we should
  use `area × emission_luminance` (Rec. 709 weights). For our
  RGB lights with roughly balanced channels, max-channel is
  within 5% of luminance and avoids a coefficient table. If a
  scene with strongly tinted lights regresses, switch to
  luminance.

## Done when

- A scene with mismatched-intensity emitters reaches a target
  RMSE faster under power-pick than uniform-pick (target: 2×
  speedup at 64 spp on `cornell_many_lights.gltf`).
- Existing Cornell scenes render byte-stably (single-emitter
  pick collapses to the old uniform behaviour).
- CPU mirror test pins the pick distribution within MC
  tolerance.
- Naga, native cargo test, fmt, clippy, wasm32 `cargo check`,
  Python unittests, CI, Pages-deploy all stay green at HEAD.
