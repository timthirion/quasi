# Participating media (path tracer)

- **Status:** active
- **Last updated:** 2026-06-04
- **Last touched on:** PT-beer-lambert landed

## Goal

Bring absorbing + scattering media into the path tracer in two
focused milestones:

1. **PT-beer-lambert** — distance-modulated absorption inside
   dielectrics. Closes the deferred 0004 piece. Headline scene:
   the Stanford bunny in tinted green glass.

2. **PT-fog** — homogeneous medium volume with isotropic single
   scattering. Headline scene: Cornell box with a fog volume,
   visible god-rays from the ceiling light.

Heterogeneous media (clouds — the actual Phase 4 headline in
[`ROADMAP.md`](ROADMAP.md)) gets its own plan (likely `0006`).
Delta-tracking + a density grid is plenty of scope to stand alone.

## Context

What's already in (as of plan `0004`):

- The dielectric BSDF tracks entering / exiting via `Hit::front_face`
  (1 = entering, 0 = exiting). Perfect for medium boundaries.
- `Material` is 48 bytes. We have 2 × u32 trailing pad, but a `vec3`
  needs 16-byte alignment in std430 — adding `absorption: vec3<f32>`
  bumps the struct to **64 bytes**. Layout test pins the new size.
- `path_trace` already maintains `var throughput: vec3<f32>` —
  Beer-Lambert just multiplies into it per segment.

What this plan is **not**:

- VPL or photon mapping for caustics inside glass — pure PT will be
  noisy on tight caustic paths even at high spp; we live with it.
- Spectral rendering. We stay RGB; absorption is three independent
  coefficients.
- Coloured ambient air / atmospheric perspective.
- Nested media (glass cup half-full of orange juice). Single-level
  medium tracking only.

## Design

### Material grows to 64 bytes

```
offset 0..16:   albedo + roughness
offset 16..32:  emission + metallic
offset 32..48:  base_color_texture_idx + ior + 2 × u32 pad
offset 48..64:  absorption: vec3 + f32 pad     ← new in PT-beer-lambert
```

Sentinel: `absorption = (0, 0, 0)` means "no Beer-Lambert" — a
dielectric with zero absorption renders as clear glass.

### Tracking the current medium

`path_trace` carries a new local variable:

```wgsl
var current_medium: u32 = NO_MEDIUM;  // 0xFFFFFFFF
```

On a dielectric transmit:
- entering (`front_face == 1`)  ⇒  `current_medium = hit.mat`
- exiting  (`front_face == 0`)  ⇒  `current_medium = NO_MEDIUM`

(Reflect branches don't change the medium — the ray stays on the
side it came from.)

### Applying absorption per segment

Each iteration of the bounce loop has access to `hit.t` from the
trace. Before any other contribution, multiply throughput by the
medium attenuation **for the segment that just ended**:

```wgsl
if (current_medium != NO_MEDIUM) {
    let sigma_a = materials[current_medium].absorption;
    throughput *= exp(-sigma_a * hit.t);
}
```

NEE shadow rays don't get attenuated in PT-beer-lambert — the
demo bunny scene is constructed so no shadow ray crosses glass.
PT-fog brings shadow-ray attenuation in.

### Phase functions (PT-fog only)

Isotropic for the headline render. Henyey-Greenstein is a one-line
WGSL addition — slot it in at PT-fog if the scene benefits.

## Milestones

### PT-beer-lambert ✅
Distance-modulated absorption inside the existing dielectric BSDF.
The bunny in green glass is the visual.

- [x] `Material` gains `absorption: vec3<f32>` + 1 × f32 pad;
      total 64 bytes. CPU + WGSL layouts stay byte-identical;
      layout test pins size + offset 48.
- [x] glTF round-trips through `extras.absorption: [r, g, b]`
      (same path PT-dielectrics took for `ior`; the two share a
      `MaterialExtras` deserialise struct).
- [x] WGSL tracks `current_medium` in `path_trace` (sentinel
      `NO_MEDIUM = 0xFFFFFFFF`); multiplies throughput by
      `exp(-σ_a · hit.t)` per segment when inside a medium. The
      dielectric transmit branch toggles `current_medium` based on
      `Hit::front_face` — entering sets `current_medium = hit.mat`,
      exiting drops back to `NO_MEDIUM`. Reflect branches leave it
      alone.
- [x] New test scene: `cornell_glass_bunny.gltf` — the Stanford
      bunny with `ior = 1.5, absorption = (1.2, 0.1, 1.5)` (green
      glass; tuned so the body is unmistakably green-tinted at the
      ~0.5-unit thickness without going opaque). Reference render
      at 512² / 1024 spp is
      `data/output/cornell_glass_bunny_reference.png`.
- [x] CPU mirror in `pathtrace::medium`; tests in
      `tests/medium.rs` pin: zero σ → identity at every distance;
      attenuation at t = 0 → identity; positive σ → strict monotone
      decrease in t; chain rule across consecutive segments; 1-unit
      slab → `exp(-σ)`.

**NEE-through-glass note.** Shadow rays still don't get attenuated.
For the glass-bunny scene the body fully refracts NEE-bound paths,
so the BSDF-then-emission walk carries the visibility — NEE never
fires on a hit *inside* glass. PT-fog will need shadow attenuation
across volumes.

### PT-fog
Homogeneous medium volume with isotropic single scattering. The
god-rays Cornell room is the visual.

- [ ] Distance-sampling: `t_scatter = -ln(ξ) / σ_t` where
      `σ_t = σ_a + σ_s`. If `t_scatter` falls short of the next
      surface hit, scatter event; else just attenuate the segment.
- [ ] Isotropic phase function (uniform sphere sample). Wire HG up
      too if the headline render benefits from it (cap as a stretch
      goal).
- [ ] NEE through the medium: shadow ray attenuates by
      `exp(-σ_t · t_shadow)`.
- [ ] Mark the medium volume via a closed mesh whose material has
      `absorption > 0` AND `scatter > 0` AND `ior == 0` (no
      dielectric → not a surface, just a boundary).
- [ ] New test scene: `cornell_foggy_room.gltf` — a thin axis-
      aligned box covering the lower half of the Cornell room as
      the fog volume.
- [ ] Tests: NEE shadow-ray attenuation matches the analytic
      formula; distance-sampling inverse-CDF matches expected
      mean free path under Monte Carlo.

## Open questions

- **Beer-Lambert through NEE in PT-beer-lambert.** Geometry chosen
  to dodge the case for the bunny scene — but if the bunny's own
  geometry causes shadow rays to cross glass, we need to handle it.
  Likely the answer is "attenuate them, even at PT-beer-lambert" —
  decide during impl based on test renders.
- **Medium volume detection in PT-fog.** Easiest scheme: any closed
  mesh with `absorption > 0` AND no `ior` is a fog volume. But a
  scene with a glass sphere inside a fog room is ambiguous — does
  the medium swap inside the sphere? Single-level only for now;
  document the limitation.
- **σ_t = σ_a + σ_s split.** For PT-fog we need scattering
  coefficients too. Material grows another `vec3` (→ 80 bytes) or
  we infer `σ_s` from `albedo` × `σ_t`. Decide at PT-fog.

## Done when

- Stanford bunny renders as visibly green-tinted glass — thicker
  parts of the bunny appear darker green.
- Cornell room renders with visible god-rays from the ceiling light
  through a homogeneous fog volume.
- CPU mirrors of `attenuation(σ, t)` and the inverse-CDF distance
  sampler ship with pinned tests.
- Naga, native + wasm clippy, fmt, the unit + GPU-regression test
  suite all stay green.
