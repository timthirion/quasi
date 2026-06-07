# PT-sun-light — delta-distribution sun

- **Status:** completed
- **Last updated:** 2026-06-07
- **Last touched on:** all five milestones ticked; Sponza hero swap-in shipped

## Goal

Add a single delta-distribution **directional sun light** to
the path tracer that combines with — does not replace — the
existing environment map.

Closes the headline Sponza finding from plan 0022: env-map-only
illumination through the oculus produces "atrium dim, side
wings near-black" framing because every NEE bounce has to
*hit* a small bright pixel near the top of the equirect
texture. A delta sun contributes deterministically at every
bounce, so a Sponza render with the sun positioned through the
oculus actually feels lit by direct sunlight.

## Context

What's already in:

* env map NEE with importance-sampled marginal + conditional
  CDF tables (PT-env, plan 0010+)
* triangle NEE with power-weighted picking + MIS (PT-many-
  lights, plan 0016)
* Bernoulli pick between env vs triangle NEE (PT-light-vs-env,
  plan 0020)
* `Uniforms` struct (`src/pathtrace/scene.rs` lines 77-122),
  96 bytes, mirrored byte-for-byte to WGSL

What's **not** in:

* Any analytic / parametric light type (delta point, delta
  dir, area-with-PDF beyond emissive triangles)
* Per-frame light settings beyond what flows through the
  scene file

## Design

### One sun per scene

A scene gets **zero or one** sun. No collection, no array,
no per-emitter switch — Sponza's plan is "the sun comes
through the oculus", and a singleton matches that. If a
later scene needs multiple deltas, lift to an array. Doing it
now would be premature.

Defined in `Uniforms` as two `vec4<f32>` fields:

```
sun_dir:   vec4<f32>   // xyz = unit direction TOWARD the sun
                       // w   = enabled flag (1.0 / 0.0)
sun_color: vec4<f32>   // xyz = emitted radiance (W/m²/sr units,
                       //       linear, > 1.0 allowed)
                       // w   = unused / padding
```

Cost: 32 bytes added to the uniform buffer; new total 128
bytes, still 16-aligned.

### Integrator change: deterministic per-bounce contribution

At each surface hit (every bounce, not just the camera-ray
hit) we add the sun's direct contribution:

```
if sun.enabled:
    L_sun = trace_shadow_ray(p, sun.dir)
    if unoccluded:
        wi = sun.dir
        f  = bsdf_eval(state, wi, wo)
        cosθ = max(0, dot(N, wi))
        radiance += throughput * f * sun.color * cosθ
```

Delta light → no PDF, no MIS weight against env / triangle
NEE. The sun's contribution is **additive** to whatever the
Bernoulli-picked env-or-triangle NEE produces this bounce.

### Sun does **not** compete in the Bernoulli pick

The current `p_env vs p_triangle` split (plan 0020) covers
the two **importance-sampled** NEE channels. A delta sun
doesn't need sampling — it's evaluated at zero cost per
bounce. Slotting it into the Bernoulli would waste the env-
or-triangle sample on bounces where the sun shadow ray
already lands the contribution.

### CLI surface

```
--sun-dir x,y,z          unit vector toward the sun (default: none)
--sun-color r,g,b        linear radiance (default: 1,1,1 when --sun-dir set)
--sun-intensity I        multiplier applied to --sun-color (default 1.0)
```

`--sun-dir` alone is enough to enable the sun; `--sun-color`
and `--sun-intensity` override the defaults. Disabled when
no `--sun-dir` provided — bit-identical render to today.

### What this plan is NOT

* Not a sky/sun *model* (no Hosek-Wilkie, no Preetham). Sky
  colour still comes from the env map; the sun just adds a
  parametric direct hit. A future PT-sky-model plan could
  replace both with an analytic dome.
* Not a soft-shadow sun. The angular radius of the real sun
  is ~0.27° — small enough that hard shadows look correct on
  Sponza-scale scenes. A future PT-sun-soft plan could sample
  the disc.
* Not multi-sun. Stays singleton until a scene demands more.

## Milestones

- [x] **[PT-sun-light/uniforms]** Add `sun_dir` + `sun_color`
  vec4 fields to `Uniforms` + WGSL mirror; zero-default
  preserves bit-identical behaviour for scenes without a
  sun. *Done when:* unit struct-size + offsets test passes;
  Cornell convergence test stays green to existing
  tolerance.

- [x] **[PT-sun-light/wgsl]** Per-bounce sun NEE in the path
  tracer: shadow ray, BSDF eval, additive contribution. Gated
  by `sun_dir.w > 0`. *Done when:* a sun-on render of a
  white quad facing straight up under a sun overhead matches
  the analytic Lambertian radiance to within 5 % RMSE.

- [x] **[PT-sun-light/cli]** `--sun-dir / --sun-color /
  --sun-intensity` flags route to `Uniforms`. *Done when:*
  `cargo run --release -- render --scene ... --sun-dir
  ...` produces a brighter render than the same command
  without `--sun-dir`.

- [x] **[PT-sun-light/test]** Convergence + smoke tests in
  `tests/sun_light.rs`: (a) sun-off scene equals pre-plan
  bit-identical (regression guard); (b) sun-on scene's mean
  luminance exceeds sun-off by the expected magnitude on a
  geometry where the analytic answer is computable.
  *Done when:* test file lands and both assertions pass.

- [x] **[PT-sun-light/sponza]** Re-render Sponza with the sun
  pointed through the oculus. *Done when:* hero PNG lands in
  `data/output/sponza_sunlit_reference.png` and visibly
  outperforms `sponza_reference.png` on side-wing
  illumination + atrium contrast.

## Done when

* All five milestones ticked
* `data/output/sponza_sunlit_reference.png` swapped into the
  README hero gallery, replacing the env-only Sponza card
* This plan moves to `Status: completed`

## Followups

Captured here, not implemented in-plan:

* **PT-sun-soft** — sample the sun's angular disc (sample a
  random direction within ~0.27° of `sun_dir`). Lands soft
  shadows for scenes where the hard-edge shadow reads as
  artificial.
* **PT-sky-model** — analytic Hosek-Wilkie sky to replace
  the env map for outdoor scenes. Closes the "sky is
  whatever HDR you point it at" framing in favour of a
  declarative sun-angle → sky-color model.
