# Heterogeneous media — clouds (path tracer)

- **Status:** done
- **Last updated:** 2026-06-04
- **Last touched on:** PT-cloud landed — plan closed

## Goal

Render a procedural cloud — heterogeneous medium with varying
density — by extending the volumetric path tracer to handle
density-grid lookups via delta tracking. Closes the Phase 4
roadmap headline: "render a cloud in a scene." Visual goal: a
fluffy procedural cloud sphere inside the Cornell box, lit by the
ceiling light.

This is the natural next step after PT-fog (plan 0005). PT-fog
handles **homogeneous** media with a closed-form distance pdf;
PT-cloud handles **heterogeneous** media where σ_t varies with
position and no closed-form sampling exists.

## Context

What's already in (as of plan `0005`):

- `Material` is 80 bytes. Carries `absorption: vec3` and
  `scattering: vec3` for the homogeneous case.
- Path tracer dispatches on `is_medium_volume_material(m)` (ior = 0
  AND extinction > 0) → pass through the surface, toggle
  `current_medium`.
- `sample_volume_distance` does exponential inverse-CDF on a
  scalar σ_t majorant; works for homogeneous media.
- `shadow_transmittance` walks shadow rays through medium
  boundaries while accumulating `exp(-σ_t · t)` per segment.
- `aabb_box` helper emits closed boxes with outward-facing
  winding (the PT-fog winding-bug test pins this).

What this plan is **not**:

- OpenVDB density grids — wonderful but a separate-plan-sized
  scope (file format parsing + 3-D texture upload + smarter
  majorants). Procedural noise covers the visual story here.
- Multi-medium support — same single-level `current_medium`
  carry as PT-fog. Nested clouds (cloud inside fog) is out of
  scope.
- Henyey-Greenstein phase function — PT-fog's isotropic phase
  is kept; HG lands as a separate milestone (`PT-hg`) since it
  upgrades *both* fog and cloud renders.
- Multi-scattering convergence tricks (Manifold NEE,
  next-event estimation through media with control variates).

## Design

### Material grows to 96 bytes

```
offset 0..16:   albedo + roughness
offset 16..32:  emission + metallic
offset 32..48:  base_color_texture_idx + ior + 2 × u32 pad
offset 48..64:  absorption: vec3 + f32 pad
offset 64..80:  scattering: vec3 + f32 pad
offset 80..96:  cloud_center: vec3 + cloud_radius: f32     ← new
```

Sentinel: `cloud_radius == 0` means "homogeneous medium" (same
behaviour as PT-fog). `cloud_radius > 0` means "heterogeneous
cloud — sample the procedural density at the hit position."

### Procedural density

The cloud sphere is defined by `(cloud_center, cloud_radius)`.
Density at position `p`:

```
r = length(p - center) / radius
if r > 1: return 0                                    // outside sphere
edge_falloff = smoothstep(1, 0.5, r)                  // 1 at center, 0 at edge
noise = fbm_3d(p · noise_freq, octaves)               // 3-octave value noise fbm
return edge_falloff · max(0, threshold(noise))        // wispy clamp
```

`fbm_3d` is value noise (cheap, no gradient table) with smoothstep
trilinear interpolation. Hash function: 3-mix prime PCG-style.

### Delta tracking inside the cloud volume

For heterogeneous media, no closed-form distance distribution
exists. **Delta tracking** treats the entire volume as
homogeneous-with-majorant `σ_t_maj`, sampling fictitious
"interaction" points and rejecting them with probability
`1 - σ_t(x) / σ_t_maj` (null collisions). For visualization:

```
loop:
    t += -log(ξ) / σ_t_maj
    if t > t_max: surface hit
    pos = origin + t · dir
    σ_t_local = σ_t_maj · density(pos)
    if ξ < σ_t_local / σ_t_maj:
        real interaction → scatter (prob σ_s/σ_t) or absorb (1 - σ_s/σ_t)
    else: null collision, continue
```

Capped at a fixed iteration count to bound the worst-case
inside-cloud loop.

### Ratio tracking for shadow rays

Shadow rays through the cloud need an *unbiased transmittance
estimate*, not a binary "scattered or not." Ratio tracking is the
standard:

```
T = 1
t = 0
loop:
    t += -log(ξ) / σ_t_maj
    if t > t_max: return T
    σ_t_local = density(origin + t · dir) · σ_t_max
    T *= 1 - σ_t_local / σ_t_maj
```

Returns a per-channel transmittance vector. Russian-roulette
terminates if T falls below a threshold.

### Dispatch

`sample_volume_distance` and `shadow_transmittance` get a single
new branch each: when the current medium material has
`cloud_radius > 0`, fall into the delta/ratio-tracking path;
otherwise use the existing homogeneous path. PT-fog scenes are
byte-stable.

## Milestones

### PT-cloud ✅
Heterogeneous medium via delta tracking. Procedural fbm cloud as
the headline scene.

- [x] `Material` gains `cloud_center: vec3<f32>` (offset 80) +
      `cloud_radius: f32` (offset 92) → 96-byte stride. Layout test
      pins size + offsets. `GpuMaterial` mirrors. Round-trip via
      `extras.cloud_center` + `extras.cloud_radius`.
- [x] WGSL: value-noise + fbm helpers (`cloud_hash3`,
      `cloud_value_at`, `cloud_value_noise`, `cloud_fbm`). Cloud
      density function with `(center, radius)` sphere falloff +
      fbm modulation. 4 octaves, frequency 4.0, threshold + gain
      tunable.
- [x] WGSL: `sample_volume_distance` branches on `cloud_radius > 0`
      into `sample_volume_distance_heterogeneous` (delta tracking).
      Returns scatter / no-scatter / absorbed; iteration cap
      `HETERO_MAX_ITER = 256`.
- [x] WGSL: `shadow_transmittance` adds a `medium_segment_
      transmittance` helper that dispatches on `cloud_radius > 0`
      into ratio tracking (`T *= 1 - σ_t(x_i)/σ_t_maj` at each
      null collision) for cloud media; falls back to closed-form
      `exp(-σ_t · t)` for homogeneous media.
- [x] CPU mirror in `pathtrace::cloud` mirrors the WGSL hash,
      value noise, fbm, and density functions byte-for-byte.
- [x] New test scene: `cornell_cloud.gltf` — Cornell room
      (5 walls + ceiling light, no internal boxes) + 12-tri
      bounding box at `[-0.6, 0.4, -0.6] → [0.6, 1.6, 0.6]`. Cloud
      material: `cloud_center = (0, 1, 0)`, `cloud_radius = 0.5`,
      `absorption = 0.1`, `scattering = 10.0` (water-droplet
      single-scattering albedo ≈ 0.99). Reference render at
      512² / 1024 spp lands in
      `data/output/cornell_cloud_reference.png`.
- [x] Tests: `density` outside the sphere is 0; deterministic per
      position; non-negative and bounded (the σ_t majorant assumes
      density ≤ ~1.3 — pinned at < 2.0 for margin); sweep of a 32³
      grid shows >5% non-empty cells (the noise threshold isn't
      so strict the sphere is empty). Layout test updated.

**Tuning notes worth remembering.** Initial render at
`scattering = 4.0` looked too thin — almost no visible cloud.
Boosted to `scattering = 10.0` (single-scattering albedo stays at
~0.99) to get the puffy headline render. The noise threshold (0.2)
and gain (1.8) shape the "puffiness" — lower threshold + higher
gain = more cumulus-like; raising threshold = more wispy/sparse.
First debug pass replaced fbm with `smoothstep(1, 0.5, r)` as a
sanity-check to confirm the delta-tracking + ratio-tracking
infrastructure was working before tuning the noise.

**Out of scope here:** Henyey-Greenstein phase function (better
forward-scattering, the visible "silver lining" cloud look),
deferred to `PT-hg`. Volumetric importance sampling beyond
delta/ratio tracking. Spectral or wavelength-dependent media.

## Open questions

- **Iteration cap.** 256 is a guess; bound depends on σ_t_maj × t_max.
  For our cloud (σ_t ≈ 2, max t ≈ 1.5 inside cloud), expected
  iterations ≈ 3, hard-cap of 256 is comfortable. Verify during impl.
- **Per-channel σ_t in delta tracking.** Current scheme uses a single
  scalar majorant (max of σ_t.x/y/z). For mostly-grey clouds this is
  fine; for coloured smoke it would inflate variance — deal with it
  when we have a coloured-medium scene to motivate it.
- **Null-collision majorant choice.** Tight = fewer iterations but
  risk of going negative if σ_t exceeds majorant. Use a loose-but-
  safe majorant (e.g., 1.1 × analytic max) for safety.

## Done when

- `cornell_cloud.gltf` renders a recognisable fluffy cloud lit
  from above by the Cornell light. Surface darker on the bottom,
  brighter on the top.
- All four homogeneous scenes (Cornell quads, glass bunny, glass
  sphere, foggy room, metal bunny, textured floor) still render
  byte-stably — the heterogeneous branch only fires when
  `cloud_radius > 0`.
- CPU mirror of `cloud_density` ships with pinned tests.
- Naga, full unit test suite, fmt all green.
