# Power-weighted env-vs-triangle pick (PT-light-vs-env)

- **Status:** completed
- **Last updated:** 2026-06-06
- **Last touched on:** Bernoulli pick + uniform plumbing

## Goal

Stop double-spending NEE samples on env + triangle every step.
Plan 0014 + plan 0016 left the integrator with **additive
multi-light** — both NEE branches fire unconditionally on
every bounce — which is provably better variance per sample
but spends 2× the shadow rays. For scenes that carry either
env or triangle emitters (the typical case in our gallery),
the cheaper branch produces ~0 contribution every step.
Switch to a **power-weighted Bernoulli pick**: roll once per
NEE event, take the env or the triangle path proportional to
total power, divide by the pick probability.

Closes the "PT-light-vs-env power-weighted pick" follow-up
called out in plan 0016. Aligns with the way the per-bucket
pick already works inside the triangle CDF and inside the
env importance tables — this is the missing top-level
selector.

## Context

What's already in:

* `sample_light` (triangle NEE) computes per-triangle pick
  probability from `emissive_lights[i].cdf`. Total triangle
  power = `Σ area · max(emission)` — already implicitly
  encoded as the CDF's normalisation.
* `sample_env_importance` (env NEE) inverts the marginal +
  conditional CDFs from `env_data`. Total env power is
  reachable as the integral of `luminance × sin θ` over the
  env map; that quantity is already implicit in the
  importance tables but needs to be surfaced.
* The integrator's NEE block currently fires both branches
  whenever `mis_nee_mode && U.has_environment == 1u`.

What this plan is **not**:

* Multiple-importance-sampling across env + triangle. That
  would integrate both contributions every step with their
  joint MIS weight — variance lower than either branch alone
  but still spends both shadow rays. Out of scope; the cost
  story is what we're tightening.
* Per-pixel adaptive light pick (ReSTIR-style spatiotemporal
  reuse). Deferred.
* Power-weighted *triangle* picks (already in plan 0016).

## Design

### Total power, surfaced

Two new scalar uniform fields:

```wgsl
struct Uniforms {
    ...
    env_total_power: f32,
    triangle_total_power: f32,
};
```

Computed on the CPU at scene build time:

* `triangle_total_power = Σ area · max(emission)`.
  `recompute_emissive` (plan 0016) already sums this — expose
  the un-normalised total.
* `env_total_power = Σ luminance(pixel) · sin θ · dA` over the
  HDR equirectangular map. `ImportanceTables::build` already
  computes the unnormalised total — expose it on the struct.

When either is zero, the corresponding branch is skipped and
the surviving branch always wins the Bernoulli pick (the
probability collapses to 1 — no pdf correction needed).

### Bernoulli pick + pdf correction

Inside the integrator NEE block:

```wgsl
let total = U.env_total_power + U.triangle_total_power;
if (total <= 0.0) { /* no NEE */ }
let p_env = U.env_total_power / total;
let xi = next_1d(s);
if (xi < p_env) {
    // env branch — divide pdf by p_env so the contribution is
    // unbiased.
    let es = sample_env_importance(next_2d(s));
    if (es.valid) {
        ...
        ls.pdf_w = es.pdf * p_env;
    }
} else {
    // triangle branch — divide pdf by (1 - p_env).
    let ls = sample_light(hit.point, s);
    if (ls.valid) {
        ls.pdf_w = ls.pdf_w * (1.0 - p_env);
    }
}
```

The MIS weight against BSDF on the miss-shader path also
needs the same correction: when a BSDF ray escapes into the
env, the env pdf gets `p_env` baked in. The integrator
already calls `env_pdf_at_dir(ray.dir) * p_env` for the
weight.

### Existing-scene fallback

Triangle-only scenes have `env_total_power = 0` → `p_env = 0`
→ every NEE pick takes the triangle branch with `pick_prob =
1`, identical to the old behaviour. Same for env-only scenes
mirrored. The cornell_glass_bunny + outdoor_normal_bunny
reference renders should be byte-stable.

For scenes with BOTH env and triangle lights (e.g. a future
"Cornell room with an open ceiling looking at the sky"), the
new pick stops the additive double-cost without changing the
expected radiance.

## Milestones

### PT-light-vs-env
- [x] `Uniforms` grows `env_total_power: f32` +
      `triangle_total_power: f32` (+ 2 pad u32s for the std430
      16-byte alignment). Layout test pins the new total at
      112 bytes.
- [x] CPU: `TriangleScene.triangle_total_power` stores the
      pre-normalisation sum from `recompute_emissive`.
      `ImportanceTables.total_power` already existed; surface
      it via `SceneBuffers.env_total_power`. Offscreen +
      windowed renderers copy both into `Uniforms`.
- [x] WGSL: NEE block switches from additive (env NEE +
      triangle NEE every step) to **Bernoulli-picked** (one
      branch per step, sampled proportional to total power).
      Picked pdf is `raw_pdf × pick_prob`; the contribution
      divides by it. `power_heuristic` BSDF MIS sees the
      picked pdf, not the raw.
- [x] Miss-shader MIS weight uses the picked env pdf
      (`env_pdf_at_dir(dir) × p_env`), keeping unbiasedness
      when the BSDF ray escapes into the env.
- [x] BSDF-hit-on-emissive MIS weight uses the picked
      triangle pdf (`light_pdf_solid_angle × p_tri`).
- [x] Existing single-source scenes (cornell_glass_bunny,
      outdoor_normal_bunny) render byte-stably: triangle-only
      scenes get `p_env = 0` → always triangle branch;
      env-only scenes get `p_env = 1` → always env branch.
      Both collapses match the prior `1/N pick × single-branch
      NEE` behaviour.

## Open questions

- **Total power on env that uses sin θ weighting.** The
  importance tables build `luminance × sin θ` per pixel and
  normalise. The unnormalised sum is already there — just
  exposed.
- **Bernoulli pick over MIS.** MIS would lower variance but
  costs 2× the shadow rays; we're paying that today. If a
  scene with both env and triangles surfaces and the pick
  shows up as visible noise, revisit with proper MIS.
- **Uniform-size budget.** The current Uniforms is 96 bytes;
  +8 bytes lands at 104, well under the WebGPU 16 KB cap.
  No bind-group resizing required.

## Done when

- Single-source scenes (env-only or triangle-only) render
  byte-stably modulo ε.
- A mixed-light scene (test fixture) runs ~50% cheaper in
  shadow rays vs the additive path without visible variance
  increase at typical spp.
- Naga, native cargo test, fmt, clippy, wasm32 `cargo check`,
  Python unittests, CI, Pages-deploy all stay green at HEAD.
