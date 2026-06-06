# Tonemap-then-denoise (PT-denoise-tonemap)

- **Status:** completed
- **Last updated:** 2026-06-06
- **Last touched on:** tonemap insertion + halo test + comparison re-render

## Goal

Kill the **HDR halos** around the ceiling light in the
PT-denoise output. The à-trous wavelet's colour edge stop
`exp(-‖c_p - c_q‖² / σ_c²)` is computed on raw radiance, but
HDR scenes carry luminance variation across one or two orders
of magnitude — a ceiling-light pixel at L ≈ 30 next to a wall
pixel at L ≈ 1 produces a `‖c_p - c_q‖² ≈ 900` term that the
σ_c = 0.5 default can't tame, so the kernel pulls hard on a
handful of bright pixels and spreads a soft glow into the wall.

Fix: tonemap **before** the wavelet so the colour distance
becomes perceptual; untonemap after so the EXR write still
carries linear radiance. Same denoiser, same defaults; the
glow goes away because Reinhard squashes the HDR top end
into the same low-dynamic-range edge stop the algorithm was
built for.

## Context

What's already in:

* `pathtrace::denoise::denoise(...)` does demodulate → 5-pass
  à-trous → remodulate. The à-trous pass uses raw RGB squared
  difference for the colour weight.
* `pathtrace::output::write_tonemapped_png` does Reinhard +
  linear→sRGB encoding for the rendered PNG. The Reinhard
  curve `t = c / (1 + c)` is the standard place to land.

What this plan is **not**:

* Variance-adaptive σ_c (SVGF proper). That's the right
  long-term fix but needs a per-pixel variance AOV the path
  tracer doesn't write. Deferred to a future plan.
* Log-luminance colour stop. Equivalent fix; tonemap is just
  the more visually-grounded version of the same idea.
* Bilateral filter in tonemapped space. Same thing again with
  different machinery.

## Design

Two-line insertion inside `denoise(...)`:

```rust
// Existing: demod = radiance / max(albedo, ε)
let mut working = demod.clone();
for px in working.iter_mut() {
    px[0] = px[0] / (1.0 + px[0]);
    px[1] = px[1] / (1.0 + px[1]);
    px[2] = px[2] / (1.0 + px[2]);
}
// Run the 5 à-trous passes on `working` (tonemapped).
let buf = run_passes(working, normal, depth, params);
// Untonemap: c = t / max(1 - t, ε).
for px in buf.iter_mut() {
    px[0] = px[0] / (1.0 - px[0].min(1.0 - 1e-4)).max(1e-4);
    ...
}
// Existing: remodulate.
```

The demodulation step still runs first — pulling baseColor
detail out of the signal stays load-bearing even in
tonemapped space. The tonemap is **per-pixel-monotone** so
ordering is preserved; edges that the normal/depth stops were
catching are still caught.

`σ_c` default can stay at 0.5 because every pixel now lives
in `[0, 1)`, so squared-difference is well-bounded.

## Milestones

### PT-denoise-tonemap
- [x] `denoise(...)` inserts the tonemap + untonemap wrappers
      around the 5 à-trous passes. `DenoiseParams` unchanged;
      defaults still land.
- [x] 2 new CPU mirror tests:
      - Reinhard round-trip is identity on uniform HDR input
        (target `[2.5, 1.8, 0.7]`).
      - HDR-bright pixel (L=30) surrounded by dim (L=1)
        neighbours: ring at radius 8 stays within 1.5× the
        background — pre-tonemap this halo extended out to
        the ±16 px à-trous reach.
- [x] Re-rendered `denoise_comparison.png` — the ceiling-light
      halo on the glass bunny render is visibly gone.
- [x] Existing PT-denoise tests still green (4 → 6 tests in
      `denoise::tests`).

## Open questions

- **Tonemap operator.** Reinhard is cheapest, monotone, and
  invertible without numerical drama at our brightness range.
  ACES would be marginally more film-like but its bumpy curve
  loses monotonicity above L ≈ 5; we'd lose the cleanly
  reversible step. Going with Reinhard.
- **σ_c retune?** The defaults were tuned in linear space, but
  the visual character of the colour stop changes when the
  inputs live in `[0, 1)` rather than `[0, ∞)`. Default at
  0.5 should still be in range — the per-pair differences in
  tonemapped space typically land in `[0, 0.5]`, which the σ
  resolves correctly.

## Done when

- The PT-denoise comparison strip on `cornell_glass_bunny.gltf`
  at 64 spp no longer shows a visible halo around the ceiling
  light in the denoised half.
- Naga, native cargo test, fmt, clippy, wasm32 `cargo check`,
  Python unittests, CI, Pages-deploy all stay green at HEAD.
