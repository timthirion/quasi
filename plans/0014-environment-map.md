# Environment-map illumination (PT-env)

- **Status:** completed
- **Last updated:** 2026-06-05
- **Last touched on:** outdoor bunny reference render + plan close

## Goal

Bring outdoor lighting into the path tracer. An **HDR equirectangular
environment map** replaces (or supplements) the existing
ceiling-light scenes — camera rays that miss all geometry return
the sky radiance, BSDF bounces that escape into the sky pick up
their colour from the dome, and NEE samples the environment via
**2-D importance sampling on the radiance PDF**. Closes the
single biggest gap between "renders inside a Cornell box" and
"renders that could be a postcard."

Pairs naturally with everything we've shipped: the Disney WDAS
cumulus rendered against a real polar sky becomes a striking
showcase render that ties PT-vdb, PT-hg, and PT-env together in
one image.

## Context

What's already in:

- `pathtrace.wgsl::trace_scene` returns `Hit { hit: false }` on a
  miss; the integrator currently breaks and contributes nothing
  for missed rays.
- NEE samples *triangle* emitters via `sample_light`. With
  environment lighting we add a second light source — the sky
  dome — that needs its own sample / MIS path.
- The `image` crate is already a build-time dep with the `png`
  feature; adding `hdr` (Radiance RGBE) is one-line.
- `wgpu` 3-D-texture infrastructure (from PT-vdb) shows the
  pattern for uploading a large per-scene data blob to a
  texture binding; we'll do the same for the 2-D equirectangular
  texture.

What this plan is **not**:

- Spherical harmonics / SH-projected environments.
- Real-time IBL from a probe (tangent-space normal maps, etc.).
- Sky models (Hosek-Wilkie, Preetham). We use a captured HDR.
- Multiple environment maps per scene. One per scene.

## Design

### File format: Radiance `.hdr`

Most CC-licensed HDR libraries (PolyHaven especially) default to
the Radiance RGBE `.hdr` format. ~5-10 MB for 2 K equirectangular
maps. The `image` crate decodes them via the `hdr` feature; output
is `Vec<[f32; 3]>` so we can keep precision.

`.exr` is a sensible alternative we can add later if a specific
asset needs it. The `exr` crate is already a dep for the AOV
output writer; reusing it for environment input is a follow-up
once a scene needs it.

### Material + scene wiring

A new optional field on the scene description, separate from the
glTF triangle materials:

```rust
pub struct EnvironmentMap {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 3]>,   // row-major, top-to-bottom
}
```

Lives on `TriangleScene` as `pub environment: Option<EnvironmentMap>`.
The path tracer loads it once at scene-build time and uploads to a
2-D `Rgba16Float` texture + builds the importance-sampling tables.

`Uniforms` grows a `has_environment: u32` flag so the WGSL miss
path knows whether to read the texture or fall back to black.

### Importance sampling — 2-D inverse-CDF

Standard PBRT-style hierarchical sampling. Build at scene load:

1. Marginal column PDF: `p_row[y] = sum_x f(x, y) * sin(θ_y)`
   where `f(x, y) = luminance(pixels[y * w + x])`. The `sin θ`
   weight accounts for the equirectangular distortion (poles are
   compressed in latitude-longitude).
2. Normalise the marginal so it integrates to 1.
3. For each row, normalise the conditional `p_col[x | y]`.

GPU side: upload `marginal_cdf: array<f32, H>` and
`conditional_cdf: array<f32, W * H>` as storage buffers. WGSL
inverse-CDF sample:

```wgsl
let row = inverse_cdf_1d(marginal_cdf, xi.y);    // y in [0, h)
let col = inverse_cdf_1d_row(conditional_cdf, row, xi.x);  // x in [0, w)
// Map (col, row) to direction on sphere
let phi   = (col + 0.5) / w * 2π;
let theta = (row + 0.5) / h * π;
let dir   = vec3(sin θ cos φ, cos θ, sin θ sin φ);
let pdf   = marginal_pdf[row] * conditional_pdf[row][col]
           / (2π² * sin θ);
```

### Miss-shader integration

`trace_scene` already returns `hit: false`. The integrator path:

```wgsl
let hit = trace_scene(ray);
if (!hit.hit) {
    if (U.has_environment != 0u) {
        let env = sample_env_at_dir(ray.dir);
        result.radiance += throughput * env * mis_weight_for_env_bsdf(...);
    }
    break;
}
```

`mis_weight` only kicks in when the previous bounce wasn't a
specular (δ-function) BSDF. The "first hit" case (camera ray
direct) gets the full unweighted environment.

### NEE adds an environment sample

The existing NEE picks a triangle emitter. We now also have an
environment emitter. Simplest: alternate sampling between the
two via a 1-D random pick weighted by their relative power
(scene-load time). Or: sample both and MIS-weight (more variance
reduction but more work). Going with **alternate-sampling**: pick
one with 50/50 (or power-weighted) probability, then use the
existing MIS pattern.

When the chosen "light" is the environment:

```wgsl
let dir, pdf = sample_env_importance(xi);
let f        = eval_bsdf(...wo, dir);
let bsdf_pdf = bsdf_pdf(...wo, dir);
let trans    = shadow_transmittance_no_geometry(p, dir, infinity);
let env      = sample_env_at_dir(dir);
let w_env    = power_heuristic(pdf, bsdf_pdf);
radiance    += throughput * f * trans * env * w_env / pdf;
```

`shadow_transmittance_no_geometry` is the existing shadow ray with
the target "at infinity" — we check whether the ray escapes the
scene (returns 1.0 attenuated by any media) or hits geometry
(returns 0). Easy reuse of the existing `shadow_transmittance`
with `dist = LARGE_FLOAT`.

### Scene definition

A new test scene `cornell_open_sky.gltf` — Cornell room with the
ceiling light replaced by a transparent / removed top, env map
contributes from above. Plus an "outdoor showcase" scene with no
walls: the Stanford bunny on a floor with the env map dome
overhead.

Both ride the existing glTF pipeline. The env map path lives
**outside** glTF (`--env-map <path.hdr>` CLI flag analogous to
`--cloud-grid`). The path tracer's `Scene` carries the optional
`EnvironmentMap`.

## Milestones

### PT-env
Single milestone — load + miss + NEE all together. The two are
tightly coupled (env without NEE is too noisy to ship as a
reference render), so cleanest is to land them in one diff.

- [x] `image` crate gains the `hdr` feature in `Cargo.toml`. New
      `pathtrace::env::EnvironmentMap` struct + a Radiance `.hdr`
      loader that returns `Vec<[f32; 3]>`.
- [x] `pathtrace::env::ImportanceTables` builds the marginal +
      conditional CDFs from luminance × sin θ weighting. CPU
      mirror of the inverse-CDF sample for testing.
- [x] `TriangleScene` grows `pub environment: Option<EnvironmentMap>`;
      the build path takes an optional `EnvironmentMap` arg.
- [x] `pathtrace::build_scene_buffers` (or a new helper) uploads
      the env texture (`Rgba16Float`) + the CDF storage buffers
      to new bind-group slots.
- [x] WGSL: env texture + CDF bindings; `sample_env_at_dir`
      (texture sample with equirectangular mapping); `sample_env_importance`
      (inverse-CDF); `env_pdf_at_dir` (for MIS). `Uniforms` grows
      `has_environment: u32`.
- [x] Integrator: miss-shader path returns weighted env emission;
      NEE picks env vs triangle by power-weighted Bernoulli, with
      MIS against BSDF.
- [x] `--env-map <path.hdr>` CLI flag on `render`. `offscreen::render_offscreen_with_grid_and_env`
      threads the optional env down.
- [x] CPU mirror tests in `pathtrace::env`: marginal+conditional
      CDFs integrate to 1; importance-sampled directions have
      empirical mean luminance matching the analytic full-sphere
      integral within MC tolerance; PDF round-trip
      (`sample → direction → eval_pdf` == sampled pdf) within
      tolerance.
- [x] At least one new test scene + a publishable reference
      render. Strongest candidate: the Disney cumulus against a
      PolyHaven HDR sky. Output gitignored if it's a Disney
      derivative.

## Resolved decisions

- **Sample pick:** went with **additive multi-light** — both
  triangle NEE and env NEE fire independently every step, each
  MIS-weighted against BSDF. Simpler than power-weighting the
  pick, and our scenes typically have only one light type active.
- **Bilinear env sampling:** the equirectangular texture uses
  `Linear` mag/min filters, so `env_radiance_at_dir` (miss path)
  is bilinear automatically. The inverse-CDF lookup snaps to
  a per-texel pixel by design.
- **Latitudinal sin θ wrap-around:** both `sample_env_importance`
  and `env_pdf_at_dir` clamp `sin θ < 1e-4 → pdf = 0`. The
  reference render at 2048 spp on the synthetic sky shows no
  pole-driven fireflies.

## Done when — all green

- [x] Rendering with `--env-map <path>.hdr` produces an image
      where the sky is visible on missed rays and lights the
      scene via NEE — see `data/output/outdoor_bunny_reference.png`.
- [x] The Cornell scenes still render unchanged when no
      `--env-map` is supplied (the `has_environment = 0` branch).
- [x] CPU mirror of the inverse-CDF + PDF tests stay green
      (6 tests in `tests/env.rs`).
- [x] Naga, native cargo test, fmt, clippy, wasm32 `cargo
      check`, Python unittests, CI, Pages-deploy all stay green
      at HEAD.

## Follow-ups (out of scope for this plan)

- Real PolyHaven HDR shipped with the repo for the showcase
  render. The synthetic procedural sky lands here as a
  deterministic, fetch-free placeholder; a one-time download +
  bake of `kloofendal_43d_clear_puresky_1k.hdr` (CC0) would
  replace it without code changes.
- Power-weighted Bernoulli light pick. Marginal value while
  scenes have either env or triangle emitters (not both); revisit
  if a future scene mixes the two.
