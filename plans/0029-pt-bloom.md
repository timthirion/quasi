# PT-bloom — HDR bloom post-process

- **Status:** draft
- **Last updated:** 2026-06-15
- **Last touched on:** rev 2.1 — round-2 skeptic patches: pivots architecture to GPU-pass-before-CPU-readback (no `OffscreenPipeline` struct; tonemap is CPU per-pixel), corrects Karis→Jimenez bloom-reference citation, fixes RMSE threshold (0.05 not 1e-4), re-baselines halo-metric-noregression

## Goal

Add a physically-motivated bloom post-process pass that runs on
the HDR radiance buffer before tonemap, so bright pixels (sun
glint, emissive lamps, sun-pool highlights) bleed into their
neighbours the way real camera lenses scatter bright light.
Quasi today tonemaps directly from the accumulated radiance;
the sun is the brightest pixel on screen and renders as a
hard-edged disc instead of a glow. The Bistro courtyard hero is
the worst-affected scene in the existing gallery: the sunlit
gothic façade through the archway should have a soft halo
around it, not a hard transition.

The Luz renderer (`github.com/themartiano/luz`, README
post-process section) lists "bloom" alongside DOF, exposure,
contrast, tonemap, and gamma. Quasi has tonemap + gamma,
neither of the others.

## Why pre-tonemap HDR bloom

Real cameras and real eyes both spread bright light into halos
(lens-element scattering, iris diffraction, intraocular
scatter). Real-time rendering convention — Unity HDRP's
`Bloom.shader`, Unreal's `PostProcessBloom.usf`, Frostbite's
"Moving Frostbite to PBR" (Lagarde & de Rousiers, 2014, slides
142–156) — runs bloom **before** tonemap so the convolution
sees the original HDR luminance ratios (a pixel at radiance
1000 spreads visibly larger than a pixel at radiance 10, even
when both clamp to sRGB white after tonemap). This is the
industry-standard choice; it is *not* the only physically
defensible choice (intraocular bloom is technically post-
retinal-response and would belong post-tonemap) but it matches
the existing post-process literature Quasi blog readers will
expect.

## Design

### Algorithm: Kawase dual-filter downsample/upsample

Standard runtime-bloom choice: successive 2× downsamples with
a 4-tap Kawase kernel, then successive 2× upsamples that blend
back into the higher mip. Each upsample adds a wider Gaussian
kernel by virtue of operating on a coarser mip, so the
composite is a *sum of Gaussians at varying scales* — which
matches how real lens-flare PSFs decompose.

**Reference:** Marius Bjørge, "Bandwidth-Efficient Rendering"
(SIGGRAPH 2015), §3.4 "Dual filter blur," is the canonical
write-up. The Frostbite slides reference Kawase's original
talk (CEDEC 2003) for the kernel weights.

### Mip-chain depth

Generate mips until `min(width, height) < 16` or the mip count
reaches 5, whichever comes first. At 1024×768 this gives the
expected 5 levels (smallest mip 32×24); at the wasm widget's
typical 384×288 it caps at 4 levels (smallest 24×18); at the
Cornell test scenes' 256×256 it caps at 4 (smallest 16×16).
The `< 16` floor protects against the 4-tap kernel sampling
beyond the mip's extent.

### Soft-knee threshold (Unity-correct)

Pure HDR bloom that convolves every pixel over-blooms the
midtones into a milky look. The fix is a soft-knee threshold
that extracts only the radiance *above* the threshold for the
bloom chain. A linear hard threshold bands; a quadratic soft
knee in `[threshold - knee, threshold + knee]` avoids the
banding. This is the standard Unity / Unreal / Frostbite
approach.

**Source:** Unity HDRP `Runtime/PostProcessing/Shaders/Builtins/Bloom.shader`,
`fragPrefilter4` function (Unity 2023.3 source). The formula
**must** handle the sub-threshold case as a zero — the
draft-revision-1 form of this plan had a buggy implementation
that produced negative weights below threshold, which would
have caused the bloom pass to *darken* midtones around bright
sources. The Unity-correct form:

```wgsl
fn soft_knee_extract(rgb: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    // Guard against NaN/Inf in the radiance buffer (fireflies).
    let safe = select(rgb, vec3<f32>(0.0), !all(rgb == rgb) || any(rgb > vec3<f32>(1e6)));
    let brightness = max(safe.r, max(safe.g, safe.b));
    let b_safe = max(brightness, 1e-6);

    // Quadratic curve over [threshold - knee, threshold + knee]:
    let curve_x = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    let curve = curve_x * curve_x * 0.25 / max(knee, 1e-6);

    // Linear above threshold:
    let linear = brightness - threshold;

    // Below (threshold - knee): both terms ≤ 0; clamp final weight to 0.
    let weight = max(max(curve, linear), 0.0) / b_safe;
    return safe * weight;
}
```

Key changes vs naive form: `clamp` (not `max`) bounds the
quadratic input on both sides; final `weight = max(..., 0.0)`
forces zero below the knee. The `safe` vector guards against
firefly pixels (NaN, Inf, > 1e6 radiance) which would
otherwise propagate through the entire mip chain and turn
every composited pixel black.

### Default intensity

`--bloom-intensity 0.04` is the default. **This default is
not arbitrary** — PT-bloom/intensity-sweep (milestone 4)
measures the per-pixel ratio of bloom-on to bloom-off
luminance in an annular ring 8–16 px from a single bright
Cornell light source, sweeping intensity in
`{0.01, 0.02, 0.04, 0.06, 0.08, 0.12}`. **Operational
definition of "right":** the locked default is the intensity
where the mean annular-ring luminance is
between 1.5× and 2.0× the bloom-off baseline (a numerical
band, not "matches a chart"). The Jimenez 2014 reference is
the canonical "Next Generation Post Processing in Call of
Duty: Advanced Warfare" SIGGRAPH 2014 Advances course
(Jimenez, not Karis — rev-2 cited the wrong author + paper).
The default is locked at the swept value (likely 0.04 ±
0.01 per the empirical band); the Bistro re-render uses the
locked default, not a separate value.

### Pass structure (GPU pass before CPU readback)

The actual codebase architecture (verified):
* `render_offscreen` (`src/pathtrace/offscreen.rs`) is a
  **free function** that produces an `Aovs` struct (no
  `OffscreenPipeline` struct exists; the rev-2 draft
  invented this).
* Tonemap is a **CPU per-pixel pass** post-readback,
  implemented in `src/pathtrace/output.rs` as
  `tonemap_pixel` (line 78), invoked by
  `write_tonemapped_png` (line 96).

Therefore bloom must run as a **GPU pass on the radiance
texture before the readback**, so the readback sees the
bloomed radiance and the CPU tonemap operates on that:

```
1. Existing path-trace + accumulate passes → radiance texture (Rgba32Float, GPU)
2. NEW: Extract pass:   radiance → bloom_mip0  (Unity-correct soft-knee)
3. NEW: Downsample × N: bloom_mip0 → mip1 → ... → mip_N      (N ≤ 5)
4. NEW: Upsample × N:   mip_N → mip_{N-1} → ... → mip0       (additive)
5. NEW: Composite pass: radiance += intensity * bloom_mip0   (in-place blend)
6. Existing readback → CPU
7. Existing CPU tonemap → PNG
```

The bloom mip-chain is one `Rgba16Float` texture
(half-precision fine). With `--bloom` off, the mip-chain
texture is not allocated and passes 2–5 are not invoked;
the radiance texture goes straight from step 1 to step 6
exactly as today. **Bypass is implemented inside
`render_offscreen_async`**, gated on a new
`bloom: Option<BloomParams>` field on `RenderConfig` — the
existing pre-plan code path is preserved with the
`Option::None` branch.

### Interaction with the à-trous denoiser (plan 0021)

Bloom and denoise both manipulate the radiance buffer; their
ordering matters. **Decision:** bloom runs **after** denoise;
the denoised image is the input to the bloom extract. The
plan 0021 halo metric measures luminance leakage in an
annular ring around bright features; bloom *by construction*
raises ring luminance, so a literal "no regression" test
against the bloom-off baseline would fail. **PT-bloom/halo-
metric-noregression instead compares two bloom-on
configurations**:
* baseline: denoise-off, bloom-on
* test: denoise-on, bloom-on

The denoise+bloom halo metric must not exceed the bloom-only
halo metric by more than 10%. This catches the failure
mode "denoise wakes up an unintended halo when bloom is in
the loop" without trivially flagging bloom's intended ring
luminance.

### Tonemap operator dependency

The default `--bloom-intensity 0.04` is tuned against the
Reinhard tonemap (Quasi's current default; see
`src/pathtrace/offscreen.rs` `tonemap_reinhard`). If a future
plan adds an ACES tonemap (compressive — heavily darkens
input radiance), the intensity default would need re-tuning
because the same intensity reads visually weaker under
compressive tonemap. The `Findings` section notes this
coupling.

### CLI surface

```
--bloom                          enable bloom (default: off — pre-plan output preserved)
--bloom-intensity I              composite multiplier (default: 0.04)
--bloom-threshold T              soft-knee threshold (default: 1.0 — slightly above tonemap-to-white)
--bloom-knee K                   soft-knee width (default: 0.5)
```

### Byte-equality invariant: scope and verification

With `--bloom` off, the offscreen render result must match
pre-plan within RMSE `0.05` over the radiance buffer at
128×128 / 256 spp PCG / MIS-NEE on `cornell_glass_bunny.gltf`.
**Threshold source:** the actual `tests/cornell_gltf.rs:330`
`cornell_quads_and_tris_render_to_the_same_image` assertion
is `rmse < 0.05`; the rev-2 draft miscopied this as `1e-4`.
The 0.05 threshold is appropriate for catching algorithmic
change without tripping on backend FMA reordering.

The bypass-when-off invariant is enforced by:
1. A new test in `tests/cornell_gltf.rs` that renders Cornell
   bunny with `RenderConfig { bloom: None, .. }` (the
   pre-plan code path) and asserts the radiance buffer is
   bit-identical to the same render with the bloom code
   path entirely deleted (gated via a `cfg(test)`
   `assert_no_bloom_state_touched()` hook that panics if any
   bloom code runs during the render).
2. Static-typing assertion: when `cfg.bloom.is_none()`, the
   `render_offscreen_async` body skips the bloom-state
   allocations and pass executions — verified by a
   compile-time `#[deny(unused_variables)]` on the bloom
   binding when the option is None (a runtime panic if any
   bloom buffer is accidentally allocated).

## Milestones

- [ ] **[PT-bloom/mip-chain]** Add a Kawase dual-filter mip-
  chain texture (`Rgba16Float`) to the offscreen pipeline
  behind an `Option<BloomChain>` field on `OffscreenPipeline`.
  WGSL downsample + upsample shaders. Mip count = `min(5,
  log2(min(w,h)/16))`. **CPU unit test:** a single-bright-
  pixel image at (128,128) in a 256×256 canvas, fed through
  the 4-level Kawase chain, produces a composite whose:
  * Total energy is within 10% of input total (Kawase
    bandwidth approximates energy-conserving Gaussian blur).
  * FWHM along the centre row is within `[14, 22]` pixels
    (matches the published Kawase 4-level Gaussian
    approximation FWHM at this canvas size).
- [ ] **[PT-bloom/extract]** Soft-knee threshold extract
  shader implementing the Unity-correct formula above. **CPU
  unit tests (all mandatory):**
  * Above-threshold: `extract([5, 0.5, 1.5], 1.0, 0.5)` →
    `rgb * 4.0 / 5.0` (`linear = 4.0`, `b = 5.0`).
  * **Sub-threshold (the bug-catch):** `extract([0.3, 0.2,
    0.1], 1.0, 0.5)` → `[0, 0, 0]` exactly (was the rev-1
    failure mode; this test must pass).
  * In-knee: `extract([0.7, 0.6, 0.5], 1.0, 0.5)` → a small
    positive weight, value matches the closed-form quadratic
    at `brightness = 0.7`.
  * NaN/Inf guard: `extract([NaN, 0.5, 1.5], 1.0, 0.5)` →
    `[0, 0, 0]`; `extract([1e7, 0.5, 1.5], 1.0, 0.5)` →
    `[0, 0, 0]` (firefly clamp).
- [ ] **[PT-bloom/composite]** Composite the bloom mip back
  into the radiance texture as an additive GPU pass running
  inside `render_offscreen_async`, before the existing CPU
  readback (steps 5→6 in the Pass structure diagram). With
  `cfg.bloom = None`, the composite pass is skipped; no GPU
  bloom resources are allocated; the radiance texture
  reaches the readback unchanged. Bypass-when-off test
  ships as part of this milestone, with the `assert_no_bloom
  _state_touched()` hook verifying the None-path doesn't
  invoke any bloom code.
- [ ] **[PT-bloom/intensity-sweep]** Render a Cornell box
  with a single area light at 4× emission at
  256×256 / 256 spp, six times, sweeping
  `--bloom-intensity ∈ {0.01, 0.02, 0.04, 0.06, 0.08,
  0.12}`. Measure luminance ratio in an annular ring
  (radii 8–16 px from the light centroid) vs the bloom-off
  baseline. Plot lands as
  `data/output/bloom_intensity_sweep.png`; the intensity
  value where the ring-to-centre ratio matches Karis 2014's
  reference is the locked default. Numeric ratio table lives
  in `Findings`.
- [ ] **[PT-bloom/cli]** `--bloom`, `--bloom-intensity`,
  `--bloom-threshold`, `--bloom-knee` flags wired through
  `src/main.rs`, tested via CLI parse tests in
  `src/main.rs`'s `#[cfg(test)] mod tests` (or
  `tests/cli_parse.rs` if it exists).
- [ ] **[PT-bloom/widget]** Browser widget gains a "Bloom"
  toggle + an intensity slider (range `[0.0, 0.15]`,
  centered on 0.04). **Hard performance budget:** ≤ 2 ms per
  composite on Apple M-series Safari at the widget's
  default 384×288 framebuffer. Measured via
  `performance.now()` around the bloom composite call;
  test asserts the budget. If exceeded, this milestone
  surfaces as a blocker — either drop the slider (toggle-
  only) or skip the wasm path for now. Slider debounce: re-
  composite only on `change` (drag end), not `input` (drag
  in progress), to keep the GPU command queue from
  saturating.
- [ ] **[PT-bloom/halo-metric-noregression]** Re-run plan
  0021 halo metric in two bloom-on configurations on the
  Cornell emissive-sphere scene:
  * baseline: `--denoise none --bloom`
  * test: `--denoise atrous --bloom`
  Halo metric (test) must not exceed halo metric (baseline)
  by more than 10%. This isolates the denoise+bloom
  interaction from bloom's intended ring-luminance signature.
- [ ] **[PT-bloom/cornell-comparison]** Side-by-side render
  of the Cornell-emission scene with bloom off vs default
  intensity. Numeric assertion: mean luminance in an
  annular ring 8–16 px from the light centroid is ≥ 1.5×
  the bloom-off baseline. Image lands as
  `data/output/cornell_bloom_comparison.png`.
- [ ] **[PT-bloom/bistro-rerender]** Re-render the Bistro
  hero with the locked default `--bloom`. Render-attacker
  pair-mode pass: compare new Bistro hero against the prior
  committed version (`HEAD~N:data/output/bistro_reference.png`).
  Attacker must surface ≥ 1 specific halo-improvement region
  on the sunlit gothic façade *and* ≥ 1 region where bloom
  did not soften an intended-sharp feature (cobblestones,
  awning edges). Both findings must land in attacker
  output. Swap into `data/output/bistro_reference.png`.

## Done when

* All nine milestones ticked
* Intensity-sweep numeric table in `Findings`; default
  locked at the measured value (likely 0.04, confirmed by
  sweep)
* Cornell bloom comparison shipped to README; annular-ring
  luminance ratio numerically asserted
* Bistro hero re-rendered with bloom-default; attacker pair-
  mode finding lands in `Findings`
* Halo-metric-noregression test green
* README features list gains "HDR bloom (Kawase dual-filter,
  Unity-correct soft-knee)"
* Plan moves to `Status: completed`

## Findings

(Populated during execution.)

## Followups (out of scope)

* **PT-lens-flare** — anamorphic streaks, ghost reflections.
  Reuses the bloom mip-chain; own plan because art-direction
  decisions (ghost vs streak vs star) are significant.
* **PT-exposure** — auto-exposure based on radiance
  histogram. Pairs with `PT-sky` to handle time-of-day
  intensity variation without re-tuning bloom defaults.
* **PT-bloom-physical** — measured / parameterised lens PSF
  instead of Kawase chain. Higher fidelity at offline
  budgets; unnecessary for the widget.
* **PT-bloom-aces** — re-tune bloom default if ACES tonemap
  ships. Coupling noted; not active until ACES lands.
