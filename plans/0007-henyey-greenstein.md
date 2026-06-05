# Henyey-Greenstein phase function (path tracer)

- **Status:** done
- **Last updated:** 2026-06-05
- **Last touched on:** PT-hg landed — plan closed

## Goal

Replace the isotropic phase function with **Henyey-Greenstein**
(HG) — a one-parameter family that picks the asymmetry of the
scattering lobe via `g ∈ [-1, 1]`:

- `g = 0` → isotropic (current PT-fog behaviour)
- `g > 0` → forward scattering (typical for water clouds at
  g ≈ 0.7–0.85 — the "silver lining" effect)
- `g < 0` → backward scattering (rare, but useful for some
  pigments)

This is a single milestone (`PT-hg`). It improves **both** PT-fog
and PT-cloud renders, costs ~30 lines of WGSL, and unblocks more
realistic cloud appearance for the upcoming `PT-vdb` plan.

## Context

What's already in (as of plan `0006`):

- `Material` is 96 bytes with `cloud_center` + `cloud_radius` for
  PT-cloud's heterogeneous density.
- `path_trace`'s volume scatter event samples direction via
  `uniform_sphere_sample` (pdf = `1 / (4π) = PHASE_ISOTROPIC`),
  and NEE evaluates the phase function as the same constant.
- Both PT-fog and PT-cloud share this code path.

What this plan is **not**:

- Two-lobe / Double HG / Mie scattering — single-lobe HG is the
  industry-standard cheap approximation; multi-lobe lands when
  the headline render calls for it (it doesn't yet).
- Phase-function importance sampling beyond the analytic HG
  sample.
- Spectral / wavelength-dependent g.

## Design

### Material grows a single scalar

`Material` reuses the 4-byte pad already adjacent to
`scattering: vec3` for `phase_g: f32`. **No size change** — stays
96 bytes; only the trailing pad name flips. Sentinel `0.0` =
isotropic (the existing scenes' behaviour). The CPU layout test
gets one new offset assertion.

```
offset 64..80:  scattering: vec3<f32> + phase_g: f32     ← phase_g replaces _pad
```

`GpuMaterial` mirrors. glTF round-trips via `extras.phase_g`.

### HG sample + eval

The HG pdf:

    p(cos θ; g) = (1 - g²) / (4π · (1 + g² - 2g cos θ)^{3/2})

Inverse-CDF sampling on the cosine:

    if |g| < 1e-4:           # collapse to isotropic
        cos θ = 1 - 2ξ
    else:
        sqr = (1 - g²) / (1 - g + 2g·ξ)
        cos θ = (1 + g² - sqr²) / (2g)

A second uniform `ξ` picks the azimuth around the incoming
direction. The new direction is built in a local orthonormal frame
with the +z axis aligned to `ray.dir`. Same basis-vector trick as
`cosine_sample_hemisphere`.

### Dispatch in `path_trace`

The volume-scatter branch replaces:

```wgsl
let wi = uniform_sphere_sample(s);
prev_bsdf_pdf = PHASE_ISOTROPIC;
```

with:

```wgsl
let g = materials[current_medium].phase_g;
let wi = sample_hg_direction(ray.dir, g, s);
let cos_theta = dot(ray.dir, wi);
prev_bsdf_pdf = phase_hg_eval(cos_theta, g);
```

And the NEE phase evaluation switches from the `PHASE_ISOTROPIC`
constant to `phase_hg_eval(dot(ray.dir, ls.wi), g)`. With `g = 0`
both reduce to `1 / (4π)` — the existing PT-fog render stays
bit-identical.

## Milestones

### PT-hg ✅
Single milestone covering the whole thing.

- [x] `Material` reuses the trailing pad of the scattering slot for
      `phase_g: f32` at offset 76. **No size change** (96 bytes);
      layout test pins the offset. `GpuMaterial` mirrors; glTF
      round-trips via `extras.phase_g`.
- [x] WGSL: `phase_hg_eval(cos_theta, g)` and
      `sample_hg_direction(incoming, g, s)`. The `|g| < 1e-4`
      branch collapses to uniform-sphere sampling. The previous
      `uniform_sphere_sample` + `PHASE_ISOTROPIC` constant are
      retired (the new functions subsume them).
- [x] `path_trace` volume-scatter event uses
      `sample_hg_direction` for the bounce and `phase_hg_eval` for
      both `f` and the MIS BSDF pdf. NEE evaluates phase at
      `dot(ray.dir, ls.wi)`.
- [x] CPU mirror in `pathtrace::phase` (`eval` + `sample_cos_theta`).
- [x] `cornell_cloud.gltf` gets `phase_g = 0.4`. Re-shot reference
      render shows visibly anisotropic top-lighting (the top of
      the cloud picks up more direct ceiling light, multi-scatter
      softening through the body). `cornell_foggy_room.gltf` stays
      at `phase_g = 0` and visually matches the previous reference
      (g = 0 → exact isotropic by branch).
- [x] Tests: HG at `g = 0` equals `1 / (4π)` at every cos θ;
      forward (`g > 0`) peaks at cos θ = 1; backward at -1;
      pdf normalises to 1 over the sphere (numerical integration);
      importance-sample mean cos θ matches `g` within 1% (MC).
      Sample stays in `[-1, 1]`.

**Tuning note.** First pass used `g = 0.7` (water-droplet Mie
literature value). For the overhead-lit Cornell cloud this made
the rendered cloud look *dimmer* — forward scattering pushes
direct ceiling light *downward through the cloud, away from
camera*. The classic "silver lining" reads best when the light is
*behind* the cloud from camera view (sunset-style geometry); for
top-lit Cornell, milder anisotropy (`g = 0.4`) preserves cloud
body brightness while still adding the directional flavour.

**Carried-forward limitation.** The HG-with-`g = 0` branch
samples in a *local* frame around `ray.dir` (rotation by 2π) while
the retired `uniform_sphere_sample` returned a world-space
direction directly. Both are uniform on the sphere in expectation,
but per-pixel values shift due to different random consumption
patterns. PT-fog's render is *visually* identical, not bitwise.

## Open questions

- **Anisotropy for fog.** Real-world haze has g ≈ 0.3 ish. Keep
  PT-fog at g = 0 so its render stays the baseline, or bump to
  0.3? Decide during impl based on whether the visual is still
  recognisable.

## Done when

- Cornell cloud renders with a recognisable "silver lining" — the
  top edges where light comes through the thin part of the cloud
  read as much brighter than the body.
- Cornell foggy room render is bit-stable.
- CPU mirror of `phase_hg_eval` + `sample_hg_direction` ships
  with pinned tests.
- Naga, full unit test suite all green.
