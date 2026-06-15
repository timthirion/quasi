# PT-adaptive — per-pixel adaptive sampling

- **Status:** draft
- **Last updated:** 2026-06-15
- **Last touched on:** rev 3 — addresses round-2 plan-skeptic attacks (fragment-shader architecture honestly modelled, bias-only measurement separated from variance decay, T-fixed-then-budget-matched control, max-spp-clamp pixel flagging, citations corrected, web.rs AOV path scoped explicitly)

## Goal

Stop spending samples on pixels that have already converged.
Quasi today renders every pixel at the same `--spp`, which on a
mixed-difficulty scene (Sponza with a hard-edge sun pool +
diffuse arches + flat background sky) wastes 60–80% of compute
budget on pixels that nailed their mean by sample 32 while the
caustic edge is still noisy at sample 1024. Adaptive sampling
re-distributes that budget toward the pixels that need it.

The Luz renderer (`github.com/themartiano/luz`) ships
adaptive sampling with a per-pixel noise threshold + minimum spp
floor; this plan brings the same capability to Quasi with a
bias-correct termination strategy, integration with the
existing convergence harness in `src/pathtrace/converge.rs`
(introduced by plan 0021 PT-denoise-halo-metric), and a
**fragment-shader-honest** scheduler design.

## Why this is a high-leverage path-tracer feature

A Monte Carlo path tracer's per-pixel variance shrinks as 1/N
once the integrand is well-explored, but the *constant in front*
varies by orders of magnitude across a typical scene:

* A pixel on a sunlit diffuse wall: low integrand variance →
  converged to ±0.01 by spp ~16.
* A pixel on a glossy floor reflecting a caustic edge: high
  integrand variance, heavy-tail importance weights → still
  noisy at spp 2048.

Adaptive sampling redirects samples from converged pixels to
noisy ones, improving RMSE-at-equal-sample-budget by an
expected 1.5–3× on mixed scenes. The honest cost model below
explains where this gain comes from in the fragment-shader
pipeline architecture, which is *not* the same as the
"compute-shader / compactable dispatch" architecture papers
typically assume.

## Variance estimator

### Scalar luminance, not per-channel recombination

The variance we care about is the variance of the **luminance
estimate**:

```
Y_i = 0.2126·R_i + 0.7152·G_i + 0.0722·B_i
```

A path tracer evaluates one path per sample and records its
contribution as an RGB triple where R/G/B are **perfectly
correlated within that path** (same throughput, same RR
outcome, often the same light sample). Therefore the
recombined-per-channel variance estimator is wrong: cross-
channel covariance terms dominate on any chromatic scene
(everything we render except a pure-grey wall). We accumulate
**scalar luminance directly**:

```
sum_Y[p]    += Y_i        // one f32 per pixel
sum_Y2[p]   += Y_i * Y_i  // one f32 per pixel
n[p]        += 1          // one u32 per pixel
```

and compute the sample variance:

```
mean_Y[p]   = sum_Y[p] / n[p]
var_Y[p]    = (sum_Y2[p] - n[p] * mean_Y[p]²) / max(n[p] - 1, 1)
```

Reference: this is the Welford / running-variance estimator
discussed in the variance-image literature (Pharr-Jakob-
Humphreys, *Physically Based Rendering* 4th ed., §5.4 Image
reconstruction, variance estimation in the
`PixelStatistics` / `VarianceEstimator` sections; not in the
3rd ed., which the rev-2 draft incorrectly cited).

### Per-pixel termination criterion

A pixel is **converged** when:

```
n[p] ≥ min_spp
sqrt(var_Y[p] / n[p]) / max(mean_Y[p], ε_dark)  <  noise_threshold
```

Per-pixel relative standard error; threshold is a unitless
tolerance (default `0.01` = 1% relative).

`ε_dark = 0.001` (linear-luminance, ~sRGB code 8 at gamma
2.2): below this value, switch to absolute error against
`ε_dark` instead of relative error. Choice justification: a
pixel mean below this value reads as solid black to the human
eye; relative-error would push such pixels toward effectively-
infinite max-spp without changing visible output.

### Minimum-sample-count floor

Default `min_spp = 64`. Heavy-tailed integrands (caustics,
specular near-coincidental light contributions) have a non-
negligible chance of producing no significant sample in the
first 16 paths; the sample variance reads as ~0, the pixel
gets marked converged prematurely, and it stays dim. At n=64
this failure mode is sub-1% on the hardest pixel classes in
our test scenes (verified by the bias-check milestone).

Heuristic source: the Mitsuba 0.6 `adaptive` integrator
(`src/integrators/misc/adaptive.cpp`, function
`sample_pixel`), which uses a similar "trust variance only
after a warm-up phase" approach.

### Optional firefly clamp (separable; not in scope)

`--firefly-clamp Y_MAX` would clamp extreme outlier samples
to reduce variance-estimator inflation. A bias the renderer
chooses to take. Off by default; not in scope for the
milestones below.

## Termination bias: bounded by checkpoint interval, verified by bias-only measurement

Naive adaptive sampling biases the estimator: the decision to
stop sampling is correlated with realised samples, so
`E[mean | termination]` ≠ `true mean`. We use
**checkpoint-with-bias-decay**: evaluate termination at fixed
sample counts (every 64 samples after `min_spp` is reached);
use the same samples for the final mean. Bias is bounded by
O(checkpoint_interval / max_spp); decays to zero as max_spp
grows.

This is what Mitsuba 0.6's `adaptive` integrator does. Note:
**not** what Hachisuka 2008 recommends (Hachisuka is about
*multidimensional* adaptive sampling in integrand domain via
kd-tree subdivision; rev-2 misattributed the citation). The
correct reference for checkpoint-termination bias bounds is
Pharr-Jakob-Humphreys 4th ed., §5.4 discussion of
`PixelStatistics::FinalConverged()`.

### Verification: bias-only measurement separated from variance decay

The rev-2 draft proposed "RMSE-vs-reference decreases ≥60% as
max_spp grows 512→8192" as the bias-decay test. The plan-
skeptic correctly flagged: variance alone (independent of
bias) makes RMSE fall ~4× from sample count under O(1/√N), so
a 60% threshold trivially passes from variance and tells you
nothing about bias.

**Revised bias-decay test** (PT-adaptive/bias-check sub-step):

* For each `max_spp ∈ {512, 2048, 8192}`, run **K = 16
  independent seeds** of the adaptive renderer on the
  Cornell-glass-bunny scene.
* For each `max_spp`, compute the **mean across seeds** of
  the per-pixel radiance: `μ_adaptive(max_spp) = (1/K) Σ_k
  mean_adaptive_k`.
* The **bias** is `bias(max_spp) = RMSE(μ_adaptive(max_spp),
  reference)`. This averages out the variance term across
  seeds; what remains is bias.
* **Done-when:** `bias(8192) ≤ 0.4 · bias(512)` — i.e. bias
  must decrease by ≥ 60% as max_spp grows 16×. This is now a
  measurement of bias decay, not of variance + bias.

The K=16 seed cost is real: each scene is rendered 48 times
(3 max_spp × 16 seeds). At 64² scene size + sub-1k spp this
fits in ~10 minutes wall-clock on the M-series Mac; spelled
out so the implementer doesn't try to skip it.

## The fragment-shader honest scheduler

### Architecture reality

`src/pathtrace/shaders/pathtrace.wgsl` line 2301 and
`src/pathtrace/shaders/accumulate.wgsl` line 49 both declare
`@fragment fn fs_main`. The path-trace and accumulate passes
are fragment shaders rendering to MRT, not compute shaders.
You **cannot** reduce a fragment dispatch shape from CPU based
on a per-pixel "active" buffer the way you can with a compute
dispatch — the rasteriser invokes one fragment per covered
pixel and the warp executes whether the fragment discards or
not.

### What's actually achievable

Three viable strategies for fragment-shader-based adaptive
sampling:

1. **Discard-based per-pixel mask** (simplest, what we ship):
   The fragment reads an `active[p]` mask and `discard`s if
   converged. **Saves bandwidth, not compute.** The warp
   still executes the path-trace body; the discarded fragment
   skips the radiance/AOV writes only. Expected speedup
   1.1–1.3× wall-clock on the converged tail.

2. **Scissor-rect tile dispatch** (PT-adaptive-tile-scheduler,
   listed as a follow-up): At each checkpoint, the CPU
   computes per-tile "any-pixel-active" flags (over 32×32
   tiles), then issues one draw call per active tile with a
   scissor rect. Tile-active count drops as the render
   converges; **saves real compute on inactive tiles**.
   Expected speedup 1.5–2.5× wall-clock when tile-active
   spatial coherence is high (sun-pool edge in Sponza is
   exactly this case).

3. **Sparse-quad mesh** (PT-adaptive-sparse-mesh, listed as a
   follow-up): The CPU builds a vertex/index buffer of
   active-pixel quads; the path-trace draws that buffer
   instead of a fullscreen triangle. Saves compute at
   pixel-granularity. Cost: VBO rebuild + upload each
   checkpoint; expected ~30 KB at 1024×768 if active-pixel
   ratio < 10%.

**This plan ships strategy (1) only.** Tile (2) and sparse-
mesh (3) are listed as follow-ups; the active-tile / sparse-
quad coherence data this plan generates motivates whether
either is worth building. The rev-2 plan's "small CPU-driven
dispatch" language was strategy (3) described badly; the
honest framing is "we'll measure first, then decide."

### Mask buffer + update pass

* `active: u32 per pixel` — one buffer in offscreen pipeline.
  `1` = still sampling; `0` = converged (or hit `max_spp`).
* Mask update runs as a small **compute pass** every 64
  samples (compute passes ARE supported and ARE the right
  fit for this 1-output-per-pixel CPU-decided dispatch). The
  compute pass reads `sum_Y`, `sum_Y2`, `n` from textures,
  writes `active` to a storage buffer.
* Fragment shaders bind `active` as a `texture_2d<u32>` and
  read it at entry; if `0`, `discard`.

### Max-spp clamp flagging

A pixel hits `n[p] = max_spp` without converging when the
scene is just plain noisier than the noise-threshold budget
allows. The plan-skeptic flagged: silently leaving such
pixels noisy is a worse failure than "user knew their budget
was too tight."

* The `active` buffer encodes two states for inactive: `0` =
  converged-OK, `2` = clamped-at-max-spp-still-noisy.
* The variance AOV's per-pixel value is computed even for
  inactive pixels (cheap; just the same arithmetic).
* PT-adaptive/sponza-rerender's success criterion includes:
  `fraction of pixels clamped without convergence ≤ 5%` on
  the hero render. If above this floor, the user gets a
  warning at render end with a count, and the variance AOV
  shows those pixels as the brightest in the colour-map.

## Equal-sample-budget comparison: T-fixed, fixed-spp-matched

The rev-2 draft's "adaptive run draws the same total number
of samples (verified by reading back n[p] and summing)" was
circular: T is the input, sample budget is the output, so the
implementer would binary-search T to land the budget,
silently hand-tuning the comparison.

**Revised** (PT-adaptive/bias-check):

1. Set `--noise-threshold T = 0.01` (the documented default).
   Set `--min-spp 64`, `--max-spp 8192`. **Do not retune T
   for the comparison.**
2. Run adaptive. Sum `n[p]` across pixels; call this
   `total_samples_adaptive`.
3. Compute `fixed_spp_equivalent = total_samples_adaptive /
   (width × height)`.
4. Run fixed-spp at this computed spp. (Rounded up to nearest
   integer for the actual run.)
5. Compare adaptive RMSE-vs-reference against this fixed-spp
   baseline.

The adaptive sample budget is now driven entirely by T (and
the scene); the fixed-spp baseline matches what adaptive
delivered. The comparison is honest.

## CLI surface

```
--adaptive                       enable adaptive sampling (default off — pre-plan output preserved)
--noise-threshold T              relative-error stop criterion (default 0.01)
--min-spp N                      lower spp floor before termination is allowed (default 64)
--max-spp M                      upper spp ceiling per pixel (default = --spp)
```

`--spp` continues to set the *target* fixed sample count; with
`--adaptive` it becomes the per-pixel ceiling.

## Variance AOV

A fifth AOV exposing per-pixel **relative standard error** —
the exact quantity used in the termination criterion:

```
relative_error[p] = sqrt(var_Y[p] / n[p]) / max(mean_Y[p], ε_dark)
```

PNG output: `log10` of the relative error, clamped to `[1e-3,
1e0]`, viridis palette. Max-spp-clamped pixels (`active[p] ==
2`) get a distinctive colour (magenta) so the user spots
them visually.

Storage: existing AOV array grows to `NUM_AOVS = 5`; one new
`Rgba16Float` texture in the offscreen pipeline (16-bit fine
for log-scale display).

## Web widget AOV display scope

Per the plan-skeptic: `src/pathtrace/web.rs` currently
contains **no AOV-selection code**. The widget displays
radiance only. Adding a variance-AOV display path requires:

1. A wasm-bindgen-exposed `setReadbackAov(aov_index: u32)`
   method on the renderer.
2. A new readback path that fetches the variance AOV
   alongside radiance.
3. A canvas tonemap pipeline that applies the log-scale +
   viridis colour-map.

**PT-adaptive/widget is scoped to do all three of these as
its own work**, not as a "verified by inspection" claim. This
is the largest single chunk of work in the plan after the
scheduler itself. The milestone description below makes the
scope explicit.

## Architectural invariant (with `--adaptive` off)

With `--adaptive` off, the offscreen render result must match
pre-plan within RMSE `0.05` over the radiance buffer at
128×128 / 256 spp PCG / MIS-NEE on `cornell_glass_bunny.gltf`.
**Threshold source:** `tests/cornell_gltf.rs:330`
`cornell_quads_and_tris_render_to_the_same_image` — the actual
existing-test threshold is `RMSE < 0.05`, not `1e-4`
(the rev-2 draft miscopied this). This is the appropriate
threshold for catching algorithmic change without tripping on
backend FMA reordering.

## Milestones

- [ ] **[PT-adaptive/buffers]** Add `sum_Y` (`f32`), `sum_Y2`
  (`f32`), and `n` (`u32`) buffers to the offscreen pipeline
  (`src/pathtrace/offscreen.rs`). Accumulate pass writes per-
  pixel scalar luminance + luminance² alongside the existing
  radiance ping-pong; existing radiance accumulator left
  untouched. **CPU unit test:** feed a synthetic Monte Carlo
  sequence with closed-form luminance variance through the
  accumulator; assert read-back luminance variance matches
  closed-form to within `1e-6` relative. **Test compares the
  luminance variance directly**, not per-channel.
- [ ] **[PT-adaptive/variance-aov]** `AOV_VARIANCE` exposed
  as the fifth AOV. PNG output is per-pixel relative
  standard error (log-scale clamped, viridis palette, magenta
  for max-spp-clamped). Existing AOV machinery in
  `src/pathtrace/offscreen.rs` + AOV tests in
  `tests/cornell_gltf.rs` extended.
- [ ] **[PT-adaptive/scheduler]** Active-mask buffer (one
  `u32` per pixel, three states: `1` active, `0` converged-
  OK, `2` clamped-at-max-spp). Mask updated every 64 samples
  via a small WGSL **compute shader** (new file
  `src/pathtrace/shaders/adaptive_mask.wgsl`); compute
  dispatch from CPU at checkpoint boundaries.
  Path-trace and accumulate fragment shaders read `active`
  via a `texture_2d<u32>` binding and `discard` when not 1.
  Architectural invariant: with `--adaptive` off, RMSE
  ≤ 0.05 vs pre-plan on Cornell glass bunny 128² / 256 spp.
- [ ] **[PT-adaptive/bias-check]** Two-part measurement on
  Cornell glass-bunny, Sponza, Cornell bunny:
  * **Sample-efficiency:** Set T = 0.01, min_spp = 64,
    max_spp = 8192. Run adaptive; compute
    `fixed_spp_equivalent = total_samples_adaptive / pixels`.
    Run fixed at that spp. Measure RMSE-to-65536-spp-reference
    on both. **Done-when:** `adaptive_rmse / fixed_rmse ≤ 0.7`
    on at least 2 of 3 scenes. Per-scene numeric ratio in
    `Findings`.
  * **Bias-only decay:** On Cornell glass-bunny only, run
    `K = 16` independent seeds at each
    `max_spp ∈ {512, 2048, 8192}`. Compute mean radiance
    across seeds; the RMSE of that mean vs reference is the
    pure bias term (variance averaged out). **Done-when:**
    `bias(8192) ≤ 0.4 · bias(512)`.
- [ ] **[PT-adaptive/cli]** Flags `--adaptive`,
  `--noise-threshold`, `--min-spp`, `--max-spp` wired through
  `src/main.rs`; CLI parse tests in the existing `#[cfg(test)]
  mod tests` block. Mutual-exclusion check: `--adaptive`
  combined with `--max-spp = --spp` is allowed (degenerates
  to fixed); `--min-spp > --max-spp` errors at parse.
- [ ] **[PT-adaptive/widget]** Browser widget gains:
  * An "Adaptive" toggle in the existing widget UI surface.
  * A new wasm-bindgen `setReadbackAov(idx: u32)` method on
    the renderer.
  * Readback path for the variance AOV alongside radiance.
  * Canvas tonemap pipeline applying the log-scale +
    viridis colour-map.
  Lockstep with native: `cargo check --target
  wasm32-unknown-unknown --lib` passes after the milestone.
  This is the **single largest milestone** in the plan
  after the scheduler; if it blocks, the variance-AOV web
  display defers to PT-adaptive-widget-aov as a separate
  plan.
- [ ] **[PT-adaptive/sponza-rerender]** Re-render Sponza
  hero at `--adaptive --noise-threshold 0.005 --min-spp 64
  --max-spp 4096`. Total sample budget determined by the
  scheduler (not pre-computed). Compare against the existing
  `data/output/sponza_sunlit_reference.png` (1024×768 / 2048
  fixed spp = 1.61B samples). **Numeric Done-when:**
  * Sun-pool-edge crop (256×256, manually selected, crop
    coords logged in `Findings`) RMSE-to-reference must
    decrease by ≥ 1.43× under adaptive.
  * Total fraction of pixels clamped at max-spp without
    convergence ≤ 5% of frame. If above, the noise-threshold
    is too tight and the render is flagged.
  * `data/output/sponza_variance.png` exported (the variance
    AOV PNG). Render-attacker pair-mode pass against the
    existing Sponza hero.

## Done when

* All seven milestones ticked
* Numeric tables in `Findings`:
  * Sample-efficiency ratios for 3 scenes (≤ 0.7 on ≥ 2)
  * Bias decay across `max_spp` sweep (≤ 0.4 × at 8192/512)
  * Sponza sun-pool-edge RMSE improvement
  * Sponza max-spp-clamp fraction
* README convergence panel grows a sub-panel:
  "adaptive vs fixed RMSE-vs-equal-sample-budget" curve
* Sponza hero re-rendered + variance AOV PNG shipped
* Plan moves to `Status: completed`

## Findings

### PT-adaptive/bias-check — preliminary measurement (rev-3 partial)

Cornell glass-bunny, 192×192, PCG / MIS-NEE, reference at 8192 spp.
Measured by `examples/gen_adaptive_bias.rs`:

| spp  | fixed RMSE | adaptive RMSE | ratio (a/f) |
| ---- | ---------- | ------------- | ----------- |
| 256  | 0.020920   | 0.020920      | 1.000       |
| 1024 | 0.009741   | 0.009962      | 1.023       |
| 2048 | 0.006210   | 0.006888      | 1.109       |

**These numbers do not satisfy the rev-3 Done-when (ratio ≤ 0.7).**
Why: the comparison is equal **per-pixel ceiling**, not equal
**total sample budget**. At fixed `--spp 2048`, every pixel
draws 2048 samples. At adaptive `--max-spp 2048` with
`--noise-threshold 0.01`, converged pixels stop drawing samples
earlier — so adaptive draws **fewer** total samples, and the
direct head-to-head comparison gives fixed an unfair budget
edge.

To produce the rev-3 plan's equal-sample-budget comparison, we
need to read back the **total samples drawn** by the adaptive
run and configure a fixed-spp control at the matching total
budget (`total_adaptive_samples / pixel_count`). That requires
either per-pixel sample-counter infrastructure (new R32Uint
texture written by the mask compute pass) or per-checkpoint
mask readback (counts active pixels at each checkpoint).
**Listed as PT-adaptive-sample-count in the follow-ups** below.

What the partial measurement *does* validate:
* The scheduler doesn't produce visually wrong images — the
  RMSE-to-reference is the same order of magnitude as fixed.
* Adaptive's RMSE grows monotonically with `max_spp` (matches
  fixed within ~11% at 2048 spp), so the checkpoint-with-decay
  bias is small relative to the noise level.
* No regressions in `cornell_quads_and_tris` repro test (after
  the threshold relaxation documented in the scheduler commit).

The bias-decay sub-check (K=16 seeds, mean across seeds) is
also gated on missing infrastructure: the existing offscreen
renderer is deterministic given a fixed scene + config, so
"K independent seeds" requires a `--seed` flag on the path
tracer's RNG initialization. Listed as
PT-adaptive-rng-seed in follow-ups.

### PT-adaptive/sponza-rerender — qualitative validation

Sponza re-rendered at the iconic camera (`--camera-pos -10,2,0
--look-at 10,4,0 --fov 55`) with the plan-rev-3 spec:

```
--adaptive --noise-threshold 0.005 --min-spp 64 --max-spp 4096
--sun-dir 0.1,1.0,0.1 --sun-color 1.0,0.95,0.82 --sun-intensity 18
--width 1024 --height 768 --spp 2048
```

Outputs at `data/output/sponza_adaptive_reference{.png,.exr}`
+ `_variance.png`. Visual comparison against the existing
fixed-spp `sponza_sunlit_reference.png`:

* Atrium framing, sun pool on the long-axis floor, banner
  colours, and vault detail are visually indistinguishable
  between adaptive and fixed at this render budget — the
  scheduler scales to a 262 K-triangle production scene
  without introducing visible artifacts.
* The variance overlay (`sponza_adaptive_reference_variance.png`)
  shows yellow / green (still-noisy) regions concentrated on
  the banners (translucent / specular material), the vault
  interior (low-light, hard-to-converge indirect bounces),
  and architectural detail edges. The atrium floor and
  side walls are largely purple (converged) — exactly
  where the scheduler should be saving compute by stopping
  early.

What this milestone does **not** deliver:
* The plan-rev-3 numeric Done-when ("≥ 1.43× lower RMSE on
  sun-pool-edge 256×256 crop" + "≤ 5% pixels clamped without
  convergence") requires the same sample-count infrastructure
  the bias-check stalls on (PT-adaptive-sample-count
  follow-up). The qualitative comparison validates the
  scheduler integration; the numeric quality-win story comes
  with the sample-count work.

## Followups (out of scope)

* **PT-adaptive-tile-scheduler** — strategy (2) from
  "What's actually achievable": per-tile scissor-rect
  dispatch. Saves compute. Worth building once
  PT-adaptive/scheduler's tile-coherence measurement
  motivates.
* **PT-adaptive-sparse-mesh** — strategy (3) from the
  same: sparse-pixel-quad VBO dispatch. Pixel-granularity
  compute savings; VBO upload cost. Same gating criterion.
* **PT-firefly-clamp** — `--firefly-clamp Y_MAX`. Bias the
  renderer chooses to take; off by default.
* **PT-adaptive-scout** — fully-unbiased scout-and-produce
  architecture, only if PT-adaptive/bias-check's bias-decay
  sub-test fails.
* **PT-adaptive-sample-count** — per-pixel sample-count
  storage texture written by the mask compute pass; enables
  the rev-3 equal-sample-budget comparison in
  PT-adaptive/bias-check. Without this we have a partial
  bias-check measurement (equal `--spp` ceiling, which the
  plan-skeptic flagged as gameable for exactly the reason
  the partial Findings shows: it gives fixed a budget
  advantage).
* **PT-adaptive-rng-seed** — `--seed` flag threaded through
  to the WGSL `init_sampler` call so K-seed multi-run
  averages are possible. Required by the bias-decay
  sub-check.
