# PT-sky — procedural atmospheric scattering sky

- **Status:** draft
- **Last updated:** 2026-06-15
- **Last touched on:** rev 2.1 — round-2 skeptic patches: corrects Hosek-Wilkie 2013 solar-radiance paper citation (rev-2 cited the wrong paper), specifies reference C++ fork explicitly, tightens calibration-constant procedure (was hand-wavy "eyeballed"), adds dawn/sunset RMSE checks to catch 180° azimuth flip

## Goal

Replace the HDR equirectangular env-map dependency for outdoor
scenes with an analytic procedural sky parameterised by **sun
elevation + sun azimuth + turbidity**. The same scene then
renders at dawn / noon / sunset / dusk by changing three floats
instead of swapping equirect HDRs. Quasi today renders Sponza
and Bistro with a baked HDR sky (`san_giuseppe_bridge_4k.hdr`)
that fixes time of day at noon-ish; "watch the scene at
different times of day" is a natural interactive widget story
that the current architecture doesn't support.

The Luz renderer ships "atmospheric simulation w/ scattering."
This plan brings the same to Quasi with an analytic model
calibrated against published reference data, routes the sky
*function* through the existing PT-env CDF pipeline for
importance sampling, and **leaves the sun disc to the existing
PT-sun-light delta-distribution path** (additive, no MIS
weighting — see plan 0023).

## Why analytic instead of ray-marched

Two viable model families:

1. **Analytic models** (Preetham 1999, Hosek-Wilkie 2012,
   Hošek-Wilkie 2013): closed-form expression for sky radiance
   as a function of zenith angle, sun zenith angle, and
   turbidity. Fitted against measured / simulated sky data.
2. **Ray-marched atmosphere** (Bruneton 2008, Hillaire 2020):
   physically march through the atmosphere computing
   Rayleigh + Mie + ozone scattering. Better at extreme
   conditions (sun below horizon, multi-scattering); requires
   precomputed transmittance + scattering LUTs.

**Choice: Hosek-Wilkie 2012.** Closed-form, one-pass
evaluation in CPU code (we never run this analytically inside
the WGSL miss shader — see "Baking, not in-shader eval"
below). The published accuracy gap vs Bruneton is small in
the regime that matters (sun elevation ≥ 5°). The poor low-
sun-elevation accuracy is explicitly out of scope for this
plan; the "sunset" milestone uses `elevation = 5°` as a hard
floor below which Hosek-Wilkie behaviour is documented as
"informative, not authoritative."

**Reference dataset:** the Hosek/Wilkie reference C++ code,
**specifically the cgg.mff.cuni.cz release fork** (NOT the
ART fork, NOT the Mitsuba fork — these differ in clamping
behaviour near `θ = π/2`). Source URL:
[cgg.mff.cuni.cz/projects/SkylightModelling](https://cgg.mff.cuni.cz/projects/SkylightModelling/),
files `ArHosekSkyModel.cpp` + `ArHosekSkyModelData_RGB.h`
+ `ArHosekSkyModelData_Spectral.h`. DOI `10.1145/2185520.2185591`
for the 2012 paper. We use the **RGB tables**
(`ArHosekSkyModelData_RGB.h`), not the spectral ones, because
Quasi is RGB-throughput. Both files have not been versioned
since the 2012 release; we vendor them verbatim at
`data/sky/ArHosekSkyModel.cpp` and `data/sky/ArHosekSkyModelData_RGB.h`
and treat any modification as a code change.

**Reference-output generation:** the validation CSV
(`tests/sky/reference_outputs.csv`) is generated **once**, at
plan-implementation time, by running the vendored reference
C++ compiled with `clang -O2 -fno-fast-math` (deterministic;
the reference code uses no RNG and no platform-specific
intrinsics). The generation command is committed as
`scripts/sky/gen_reference.sh` so the CSV is reproducible.

**Interpolation:** the reference C++ code (`ArHosekSkyModel.cpp`,
function `arhosekskymodel_radiance`) prescribes:
* **Turbidity:** linear between adjacent integer bins (2.0
  through 10.0).
* **Solar elevation:** quintic Bezier across 5 control-point
  splines (their splines, embedded in the data file).
* **Ground albedo:** linear between the two tabulated bins
  (0 and 1).

Departures from this interpolation are not "fitting the
paper"; they are a different model. PT-sky/hosek-cpu must
implement the reference scheme.

## Baking, not in-shader eval (and why disc lives elsewhere)

### The baking architecture

Quasi's NEE-on-env (PT-env, plan 0014) builds a 2D CDF over
the equirect HDR pixel grid via `ImportanceTables::build` in
`src/pathtrace/env.rs` (line 115). The CDF is used by the
WGSL path-trace shader for importance-sampling the
environment. To stay compatible, this plan bakes the
Hosek-Wilkie sky function to an equirect texture at render
start (or on widget slider release), then routes the baked
texture through the existing CDF pipeline. The miss shader
samples the baked equirect via the existing path; no analytic
sky in WGSL.

### Bake resolution: 1024×512 (not 512×256)

The rev-1 draft proposed 512×256. The plan-skeptic flagged
that the sun disc (0.27° angular diameter) doesn't fit in a
512×256 equirect (0.7°/texel at the equator) — the disc would
be a 1-texel smear. This plan resolves the issue by
**decoupling the sun disc from the baked equirect entirely.**
The baked equirect carries the sky *function* only (no disc);
the delta-distribution sun from PT-sun-light (plan 0023)
contributes additively to direct lighting; the analytic sun
*irradiance* (Hošek-Wilkie 2013) sets the sun-color value.

With the disc decoupled, the bake resolution choice is driven
by the sky function's spatial frequency, not the sun. The
sky function is smooth — turbidity-dependent gradients of a
few %/degree — so 1024×512 (0.35°/texel) is more than enough
for the CDF importance-sampling to work cleanly. The
0.35°/texel rate is also the same as a typical HDR captured
sky panorama at 4K (4096×2048 / 2π ≈ 0.5°/texel at the
equator, similar resolution after the singular pole region is
excluded).

### Sun-disc story in detail (load-bearing)

The plan-skeptic identified this as the strongest single
attack on the rev-1 draft. The resolution is:

1. **The baked equirect contains no sun disc** — at any
   direction within the sun's angular extent, the sky-
   function value is the bright-clear-sky aureole value
   that Hosek-Wilkie computes at that direction (which is
   itself ~10² brighter than the rest of the sky near the
   sun direction, but not the ~10⁵× brighter solar disc).
2. **The delta sun** (`--sun-dir`, `--sun-color`, plan 0023)
   contributes the disc as an additive direct-lighting
   term, with `cos(θ_sun)` falloff and shadow-ray
   visibility. With `--sky`, `--sun-dir` is auto-derived
   from `--sky-elevation` + `--sky-azimuth`; `--sun-color`
   is auto-set to the **Hošek-Wilkie 2013 analytic solar
   irradiance** (which uses the same coefficient tables for
   the atmospheric transmittance from sun-to-camera path).
3. **No double-counting** — the baked equirect does not
   contain the disc, so the env importance-sampling cannot
   accidentally sample "the disc" through the env channel.
   The delta sun's additive contribution is the only path
   the disc takes to the camera.

This design respects the existing plan 0023 contract: the
delta sun is **additive, not MIS-weighted, not part of the
env channel**. The skeptic's catch on rev-1's "still in MIS"
phrasing is fixed; the architecture above is the correct one.

**Citation for the analytic solar radiance:** **Hošek &
Wilkie 2013, "Adding a Solar-Radiance Function to the
Hošek-Wilkie Skylight Model"**, Pacific Graphics 2013, DOI
`10.1111/cgf.12244`. The rev-2 draft cited this as
"Hošek-Wilkie 2013" (wrong author order) and confused it
with a separate Wilkie/Hosek 2013 paper on Lambertian
reflectance prediction. The correct paper is by Hošek and
Wilkie (in that order), published at PG 2013, and provides
analytic limb-darkening + solar-spectrum data tables that
plug into the same coefficient framework as the 2012 sky
model. The plan vendors the reference implementation from
the **same cgg.mff.cuni.cz release** as the 2012 model.

## Coordinate convention

Pinned against `src/pathtrace/env.rs` line 16 + line 239:

```
dir = (sin θ cos φ, cos θ, sin θ sin φ)
```

Therefore:
* **+Y is up** (`cos θ` is the y-component of the direction at
  zenith).
* **+X corresponds to φ = 0** ("east" in the convention used
  by `env.rs`).
* **+Z corresponds to φ = π/2** ("north" — but with the
  caveat that the existing renderer does not commit to a
  cardinal-direction convention, only to the mathematical
  one).

Sky parameters use the same convention:
* `--sky-elevation` deg above horizon → θ = (90° - elevation).
* `--sky-azimuth` deg measured from +X axis toward +Z axis
  (i.e. `--sky-azimuth 90` = sun toward +Z).
* Derived sun direction:
  `sun_dir = (sin(θ) cos(φ), cos(θ), sin(θ) sin(φ))` —
  exactly the env.rs mapping.

This is the only convention statement in the plan. The
implementer must not invent a north/east/up mapping
elsewhere.

### PT-sky vs PT-sun-light parameter wiring

When `--sky` is set:
1. `--sky-elevation` + `--sky-azimuth` derive `sun_dir` via
   the formula above.
2. `--sun-dir` flag, if also present, **errors at CLI
   parse** — combining both is a user-intent ambiguity (do
   they want `--sun-dir` to override the sky-derived one, or
   should the sky direction win?). Force the user to choose.
3. `--sun-color`, if not present, defaults to the Hošek-
   Wilkie 2013 analytic solar irradiance computed for the
   sky-elevation + turbidity at render start. The units are
   consistent with the existing `--sun-color` (linear RGB,
   normalised against a unit "white" sun); the Hošek-Wilkie
   irradiance is divided by `(R + G + B) / 3` and scaled by
   a documented constant `k` (derived per the calibration
   procedure in PT-sky/sun-color) so the default render
   matches the existing Sponza sun-render mean atrium
   luminance. The value of `k` lands in `Findings`.
4. `--sun-intensity` continues to scale the final sun
   contribution; defaults to 1.0.

## Widget slider performance

The plan-skeptic flagged: a single browser thread baking a
512×256 equirect *plus* rebuilding the CDF *plus* re-uploading
to GPU is not "<100 ms" in the rev-1 draft, especially at
1024×512 (4× the work).

**Measured cost story (planned):** PT-sky/perf-measure
(milestone added) measures the end-to-end re-bake latency on
native + wasm before locking in the widget interaction model.
**Hard latency budget:** end-to-end (slider release → first
re-rendered frame) ≤ 250 ms on Apple M-series Safari at
1024×512 bake resolution. If not met, PT-sky/widget falls
back to one of:
* 512×256 bake (4× cheaper, marginal CDF-sampling quality
  loss on the sky function — verified separately).
* Async bake on a worker thread (browser-specific complexity).
* Debounce to slider-release only (no live preview).

**Slider debounce policy:** even with the budget met,
re-bake triggers only on slider `change` (drag end) events,
not `input` (drag in progress). This is the same policy used
in PT-bloom/widget for the bloom intensity slider.

## CLI surface

```
--sky                            enable procedural sky (default off; --env-map path remains supported)
--sky-elevation DEG              sun elevation above horizon (default 45)
--sky-azimuth DEG                sun azimuth from +X axis toward +Z axis (default 180 = -X direction)
--sky-turbidity T                atmospheric turbidity (default 2.5)
--sky-ground-albedo R,G,B        ground albedo for horizon tint (default 0.3,0.3,0.3)
```

`--sky` is mutually exclusive with `--env-map`; combining them
errors at CLI parse. `--sky` combined with explicit `--sun-dir`
also errors (per "PT-sky vs PT-sun-light parameter wiring"
above).

## Milestones

- [ ] **[PT-sky/hosek-cpu]** Pure-Rust implementation of
  Hosek-Wilkie 2012 in `src/pathtrace/sky.rs`. Uses the
  vendored `data/sky/ArHosekSkyModelData_RGB.h` coefficient
  tables. Interpolation matches the reference C++ code:
  linear in turbidity, quintic Bezier in solar elevation,
  linear in ground albedo. **Held-out validation test
  (the load-bearing correctness check):**
  * Generate 200 (sun-zenith, view-zenith, azimuth, turbidity)
    quadruples on a uniform grid that **does not** coincide
    with the tabulated dataset nodes (e.g. half-integer
    turbidity values and off-grid angles).
  * For each, compute the reference output by running the
    Hosek/Wilkie reference C++ code on the same inputs (we
    vendor this once into `tests/sky/reference_outputs.csv`
    so the test doesn't depend on a C++ build).
  * Assert ≤ 0.5% relative error on ≥ 95% of the held-out
    set. Failing samples must lie in the [0°, 5°] solar-
    elevation horizon band (documented poor-fit region).
  * Tabulated-node-only validation (the rev-1 spec) is
    **also** included but only as a secondary sanity check;
    pass criterion ≤ 0.05% on 100% of nodes (trivial — these
    are direct table lookups in the interpolation scheme).
- [ ] **[PT-sky/bake]** `Sky::bake_to_equirect(width, height)
  -> EnvironmentMap` produces a baked HDR equirect compatible
  with `src/pathtrace/env.rs`'s
  `EnvironmentMap { width, height, pixels: Vec<[f32;3]> }`
  (line 31). Bake runs at render start (CPU). The baked
  equirect carries the sky **function** only — no sun disc.
  Round-trip CPU test: bake → sample at a known direction →
  compare against `hosek_sky_radiance` at the same
  direction, within 0.5% (accounts for nearest-neighbour
  texel lookup vs analytic sample point).
- [ ] **[PT-sky/perf-measure]** Measure end-to-end re-bake
  latency on:
  * Native (Apple M-series, single-threaded CPU bake +
    sequential CDF build) at 512×256 and 1024×512.
  * Wasm (Apple M-series Safari, browser default
    single-threaded) at 512×256 and 1024×512.
  Numeric latency table lands in `Findings`. If the
  1024×512-on-Safari number exceeds 250 ms, this milestone
  forces the widget bake resolution down to 512×256 (and
  surfaces the trade-off explicitly: lower CDF resolution
  vs slider responsiveness).
- [ ] **[PT-sky/wire]** `--sky`, `--sky-elevation`,
  `--sky-azimuth`, `--sky-turbidity`, `--sky-ground-albedo`
  flags wired through `src/main.rs`. `--sky` baked equirect
  fed through the existing `render_offscreen_full` path's
  `env_map` argument (no new public function needed). CLI
  parse tests cover: `--sky` alone, `--sky` + `--env-map`
  (must error), `--sky` + `--sun-dir` (must error),
  `--sky-elevation` out of [0, 90] (must error).
- [ ] **[PT-sky/sun-color]** When `--sky` is set, derive
  `--sun-color` from Hošek-Wilkie 2013 analytic solar
  irradiance computed at the current
  (elevation, turbidity). **Calibration procedure (not
  eyeballed):**
  * Bounding box: directly-sunlit floor patch in
    `data/output/sponza_sunlit_reference.png`, pixel rect
    `(384, 480) – (640, 640)` — central atrium floor,
    shadow-free sun.
  * Compute mean linear RGB luminance over the bbox in the
    existing reference: `Y_ref`.
  * Render Sponza with `--sky --sky-elevation 75
    --sky-turbidity 2.5 --sun-intensity 1.0` at same
    resolution + spp; compute mean luminance over the same
    bbox: `Y_sky`.
  * Calibration constant `k = Y_ref / Y_sky`.
  * Numeric value of `k` + per-channel breakdown land in
    `Findings`.
- [ ] **[PT-sky/time-of-day]** Sponza rendered at three
  times of day:
  * Dawn: `--sky-elevation 8 --sky-azimuth 90 --sky-turbidity 3`
  * Noon: `--sky-elevation 75 --sky-azimuth 180 --sky-turbidity 2.5`
  * Sunset: `--sky-elevation 8 --sky-azimuth 270 --sky-turbidity 4`
  Three reference PNGs land at
  `data/output/sponza_dawn.png`, `_noon.png`, `_sunset.png`.
  **Reference-comparison tests (mandatory — catches 90° AND
  180° azimuth flips at low and high sun elevation):**
  * **Noon-vs-existing:** RMSE ≤ 0.05 on brightly-lit
    atrium region (luminance > 0.3 in existing reference).
  * **Dawn-vs-sunset asymmetry:** dawn shadows on the east
    wall (pixel column 200, row 300–500) must be significantly
    darker than sunset shadows on the east wall (sunset light
    comes from the opposite direction). Sun-direction
    asymmetry test: `mean_luminance(dawn, east_wall) <
    0.5 · mean_luminance(sunset, east_wall)`. This catches a
    180° azimuth flip that the bilaterally-symmetric noon
    case misses.
  * **Time-of-day visual distinguishability:** RMSE between
    any two of {dawn, noon, sunset} ≥ 0.15 on full frame.
    Catches the failure mode "three nearly-identical noon-ish
    renders, one with the sun slightly lower."
- [ ] **[PT-sky/noon-stability]** Bake at noon twice with
  identical parameters; produced equirect textures must be
  byte-identical. Catches accidental nondeterminism in the
  bake path (RNG seeded from clock, etc.).
- [ ] **[PT-sky/widget]** Browser widget gains sky-elevation
  + sky-azimuth sliders + a sky-turbidity slider. Sliders use
  the same `change`-not-`input` debounce as
  PT-bloom/widget. Slider release triggers re-bake + CDF
  rebuild + GPU re-upload + accumulation reset. The
  end-to-end latency is held to the PT-sky/perf-measure
  budget; widget test asserts the budget on first slider
  release.

## Done when

* All seven milestones ticked
* Held-out Hosek-Wilkie validation passing ≤ 0.5% on ≥ 95%
* Sponza dawn / noon / sunset triptych shipped as a new
  README hero panel
* Noon-vs-existing-Sponza reference comparison numerically
  green (RMSE ≤ 0.05)
* Widget time-of-day demo live; latency budget met
* Plan moves to `Status: completed`

## Findings

(Populated during execution: calibration constant for sun-
color derivation, perf-measure latency table, attacker findings
on the time-of-day triptych.)

## Followups (out of scope)

* **PT-sky-bruneton** — ray-marched physical atmosphere for
  sub-horizon sun, multi-scattering accuracy. Adds ~3 MB of
  precomputed LUTs (heavy for the wasm bundle).
* **PT-clouds-sky** — couple the existing volumetric cloud
  pipeline (plan 0006 PT-cloud, plan 0008 PT-vdb-ingest) to
  the sky shader so cloud shadows + cloud-tinted sky light
  cohere. Large art-direction surface; own plan.
* **PT-sky-night** — stars + moon + Milky Way. Aesthetic
  surface area large; unrelated to daytime story.
* **PT-sky-spectral** — spectral rendering through dielectric
  prisms / rainbows. Renderer-architecture-scale change.
* **PT-sky-analytic-shader** — drop the equirect bake and
  evaluate Hosek-Wilkie analytically in the WGSL miss
  shader. Faster per-pixel but loses the env CDF; only
  worth it if perf-measure motivates.
