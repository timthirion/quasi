# Edge-aware denoiser (PT-denoise)

- **Status:** draft
- **Last updated:** 2026-06-06
- **Last touched on:** planning

## Goal

Add a **CPU-side edge-aware à-trous wavelet denoiser** that runs
after `render_offscreen` and consumes the AOVs the path tracer
already writes (radiance, albedo, normal, depth). Closes the
Phase 4e hole in the ROADMAP. Pays off most at the embedded blog
demo, where browser-side spp is capped by latency and a clean
analytic filter pulls a 32–64 spp render into "looks like 1k
spp" territory for the hero patches without touching the
shader.

Pairs naturally with everything we've shipped — every
publishable reference render uses the same AOVs the denoiser
needs, so this is purely a post-process plumbing job (no new
samplers, no integrator changes).

## Context

What's already in:

* `pathtrace::offscreen` writes four AOVs per render:
  `radiance`, `albedo`, `normal`, `depth`. Layout pinned by
  `AOV_RADIANCE` / `AOV_ALBEDO` / `AOV_NORMAL` / `AOV_DEPTH`.
* `pathtrace::output::write_render` consumes the AOV pack and
  encodes PNG + EXR. We hook in **before** PNG encoding so the
  on-disk PNG is the denoised result (raw + denoised both
  written when `--denoise` is set).
* The CPU mirror infrastructure for analytic kernels already
  follows the pattern (see `pathtrace::env::ImportanceTables`)
  — module + tests, no GPU dependence.

What this plan is **not**:

* OIDN / Optix integration. Adds a hefty native dep, no wasm
  story. Out of scope here; could plug in alongside the
  analytic denoiser later.
* A learned denoiser (TF / Pytorch model weights). Same
  problem: dep weight, wasm story.
* GPU compute-shader implementation. The CPU side is fast
  enough for offline render passes (the 5-iteration à-trous on
  a 768² image takes ~50–100 ms in release).
* Temporal accumulation / motion-vector denoising. The widgets
  run offline samples, not temporal streams.

## Design

### Algorithm — edge-stopped à-trous wavelet

Five iterations of a 5×5 B3-spline kernel at increasing step
sizes: `1, 2, 4, 8, 16`. Per-pass weights at each pixel pair
`(p, q)`:

```
w_kernel  = h(i, j) · h(i', j')        // B3-spline: 1, 4, 6, 4, 1 / 16 per axis
w_color   = exp(-‖c_p - c_q‖² / σ_c²)
w_normal  = max(0, n_p · n_q) ^ σ_n
w_depth   = exp(-|z_p - z_q| / σ_z)
w_total   = w_kernel · w_color · w_normal · w_depth
output_p  = Σ_q w_total · c_q   /   Σ_q w_total
```

Defaults (tuned empirically — these work for the Cornell + env
showcase scenes without per-scene knobs):

* `σ_c = 0.50`
* `σ_n = 32.0`
* `σ_z = 0.10`

### Demodulation — divide by albedo

To avoid blurring across texture / colour detail, denoise the
**demodulated radiance** (radiance ÷ albedo) and remodulate
afterwards:

```
demod_p = radiance_p / max(albedo_p, ε)
denoised_demod = atrous(demod, normal, depth, ...)
output = denoised_demod · albedo
```

This keeps texture detail crisp (the albedo carries it, not the
denoiser) and stops the filter from washing out the
brushed-brass micro-streaks under HDR lighting.

For non-diffuse hits where `albedo` carries the F0 (PT-ggx) or
the emission colour, demodulation still produces a meaningful
quotient. Pixels with `albedo == 0` (camera ray that missed
everything in a no-env scene) are pass-through unchanged.

### CPU-only

The path tracer's GPU output is already read back into
`Vec<f32>` AOVs by `offscreen::render_offscreen`. The denoiser
operates on those slices directly:

```rust
pub fn denoise(
    radiance: &[f32],   // RGBA, length = w * h * 4
    albedo: &[f32],
    normal: &[f32],
    depth: &[f32],
    width: u32,
    height: u32,
    params: DenoiseParams,
) -> Vec<f32>;
```

Single-threaded for the first cut (rayon is a deps churn for a
~50 ms pass at 768²; revisit if profiling pulls the denoiser
onto the critical path).

### CLI integration

`render --denoise` runs the analytic à-trous pass after the
offscreen render and writes both `frame.png` (raw radiance)
and `frame_denoised.png` (denoised). The EXR write is always
the **raw** radiance — the EXR is the source-of-truth ground
data, never the post-processed one. README + docs note the
behaviour.

## Milestones

### PT-denoise
- [ ] `pathtrace::denoise` module with `atrous_pass(...)` (one
      iteration at step `k`) and `denoise(...)` (5-pass
      demodulate-denoise-remodulate). Pure CPU; pure
      `Vec<f32>`.
- [ ] `--denoise` flag on `render`. Writes
      `<basename>_denoised.png` alongside `<basename>.png` +
      `<basename>.exr`. Without the flag, no behaviour change.
- [ ] CPU mirror tests:
  - uniform input returns uniform (degenerate case);
  - edge with normal discontinuity preserves the edge (no
    blur across the seam);
  - flat patch with synthetic radiance noise has lower RMSE
    after the denoiser than before.
- [ ] Showcase: re-render `outdoor_normal_bunny.gltf` at 256
      spp **with** and **without** denoise. Output the side-
      by-side as `data/output/denoise_comparison_outdoor.png`
      (a 2×1 strip composed by the example).

## Open questions

- **Σ_c sensitivity.** Too aggressive and the bunny's brushed
  streaks blur out; too gentle and noise leaks through. The
  default `0.50` works for our HDR-illuminated scenes; expose
  on the CLI if a scene with extreme contrast needs a tweak.
- **Edge-weight composition.** Multiplying the three (color,
  normal, depth) weights is the SVGF-style baseline. Replacing
  the product with a max/min combination would chase specific
  artefacts; defer until we see one.
- **Albedo edges vs base-color textures.** Demodulation makes
  the denoiser blind to baseColor / albedo texture detail
  (which is the *intent*: that detail rides in the albedo
  AOV). PBR-map detail (normal + MR) shows up in `normal` /
  `radiance` and is preserved by the normal-edge stop.

## Done when

- `render --denoise --spp 256` on
  `cornell_glass_bunny.gltf` produces a perceptibly cleaner
  image than the raw output at the same spp.
- All pre-flight checks (fmt, clippy, test, wasm32, Python
  unittests) stay green at HEAD; CI + Pages deploy stay
  green.
