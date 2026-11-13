# PT-bloom-verification — close the gaps plan 0029 left open

- **Status:** active — 1 of 6 milestones shipped (`widget-smoke`)
- **Last updated:** 2026-11-13
- **Last touched on:** 2026-11-13 — added and shipped `[PT-bloomv/widget-smoke]`, the CI gate R0003's 2026-09-15 findings asked for (the widget rendered nothing for three months under a green CI). It comes first because `[PT-bloomv/widget]` is unverifiable without it. Originally created during the 0029 closing pass to hold the four verification riders and one whole milestone that 0029 shipped without. A second closure-pass run then ran the `close-plan` review gauntlet the first pass explicitly skipped (plan-skeptic + code-attacker/defender + render-attacker) and surfaced three P0 correctness bugs in the shipped bloom code that the inline verification never caught. Those were fixed in-place ahead of this plan's execution — see the `Prior P0 fixes` section below — so this plan's milestones now build on a known-good foundation.

## Goal

Plan [`0029-pt-bloom`](0029-pt-bloom.md) shipped HDR bloom and it went
straight into gallery-wide use — the Bistro hero, the chess hero, the
Sponza sunlit variant and both Cornell glass scenes are all
bloom-default renders. What it *didn't* ship was most of its own
verification, plus the browser widget surface entirely. This plan
closes that gap so bloom is defensible rather than merely
good-looking, and so it exists on both targets instead of one.

The single most important item is the **widget**. Quasi's stated
constraint — see `plans/README.md`, "Native and web builds are both
first-class: a plan isn't done until it works in both targets" — was
violated by 0029. Bloom is native-only today. Since the whole reason
this renderer exists is the live in-browser widget, a post-process
that only runs offline is a post-process the blog reader never sees.

## Prior P0 fixes (landed before this plan starts)

The 2026-07-30 close of 0029 explicitly skipped the `close-plan`
skill's review-agent gauntlet ("verification was inline"). A
2026-08-10 run of that gauntlet surfaced three P0 correctness
bugs in shipped bloom code, all fixed before this plan begins:

* **P0-A — Firefly guard vs Rgba16Float saturation.** The extract
  shader (`src/pathtrace/shaders/bloom_extract.wgsl`) and the CPU
  reference in `src/pathtrace/offscreen.rs` both guarded fireflies
  at `< 1e6`, while the mip chain storage is `Rgba16Float` (f16
  max ≈ 65504). Any pixel in `(65504, 1e6)` passed the extract
  guard and then rounded to `+Inf` on the store into mip 0,
  propagating through the entire chain via the tent-kernel
  upsample. The `--emission-scale 500` Vespa-night hero (commit
  `6486b93`) sits squarely in that band. Guard tightened to
  `< 6.5e4` in both places. Two new CPU tests pin the fix:
  `bloom_soft_knee_clamps_fireflies` now includes a `1.0e5` case
  (the previously-broken band); `bloom_soft_knee_admits_pixel_below_f16_bound`
  asserts a `6.0e4` pixel still extracts (guard is tight enough,
  not over-aggressive).

* **P0-B — Bloom silently degenerated at `min(w,h) < 32`.** The
  Kawase mip loop stopped at `< 16` per side, so below the 32-px
  floor `levels.len() == 1`, the downsample loop iterated `0..0`,
  and mip 0 stayed as the raw extract output. Composite then
  added `intensity * extracted_radiance` back into the radiance
  buffer — brightening pixels above threshold without any spread.
  Silent correctness failure; no test surfaced it. Fixed in
  `render_offscreen_async`: the bloom entry now
  `log::warn!` + `Option::filter`s the config to `None` when the
  min dim is below 32. Regression test in
  `tests/bloom.rs::bloom_below_min_dimension_is_skipped_not_wrong`
  (`#[ignore]` GPU test) renders 24×24 Cornell with and without
  bloom and asserts byte-identical radiance.

* **P0-C — NaN slipped past every bloom CLI clamp.** The guards
  used `x < 0.0`; `NaN < 0.0` is false in IEEE-754, so
  `--bloom-intensity NaN`, `--bloom-knee NaN`, and
  `--bloom-threshold anything` all parsed cleanly and NaN
  propagated into `composite = intensity * bloom`, turning every
  pixel into NaN. `--bloom-threshold` had no clamp at all. Fixed
  in `src/main.rs`: guards now `!x.is_finite() || x < 0.0` (or
  `<= 0.0` for knee); threshold gains a `!is_finite() || < 0.0`
  guard. `bloom_flags_reject_nan_and_infinity` pins the fix.

Detail on the fixes and their tests lives in the commit message
+ code comments (`P0-A` / `P0-B` / `P0-C` tags). The five
milestones below now execute against a bloom pipeline whose
firefly guard, dimensionality edge case, and CLI-input handling
are all known-correct — the widget budget measurement (in
particular) becomes a signal about the composite cost rather
than about latent bugs.

## Context

What shipped in 0029:

* `src/pathtrace/shaders/bloom_{extract,downsample,upsample,composite}.wgsl`
* the `Option<BloomConfig>` gate and pass sequencing in
  `src/pathtrace/offscreen.rs`
* `--bloom`, `--bloom-intensity`, `--bloom-threshold`, `--bloom-knee`
  in `src/main.rs`, with parse tests
* four CPU unit tests pinning `soft_knee_extract_reference`
* the intensity sweep: seven renders, `examples/analyze_bloom_sweep.rs`,
  `data/output/bloom_intensity_sweep.csv`, and the ratio table in
  0029's `Findings` (locked default `0.04` at 1.8145× ring luminance)

What did not:

* any numeric pin on the **Kawase kernel itself** — the mip chain is
  exercised by every render but its blur characteristics are
  unverified, so a refactor could change the kernel's spread without
  a single test failing
* any assertion that **`--bloom` off is genuinely inert**
* the **denoise × bloom** halo interaction check
* the **widget** toggle + slider

## Milestones

- [x] **[PT-bloomv/widget-smoke]** **Shipped 2026-11-13.** A CI gate
  that instantiates the interactive renderer and asserts it draws.
  `pathtrace::State` now presents through a `present::PresentTarget`
  (surface *or* offscreen texture); `State::new_headless` builds the
  exact widget pipelines against the offscreen target and
  `State::read_presented` reads the frame back. `tests/widget_smoke.rs`
  (not `#[ignore]`d) checks first frame, 8 frames, resize, camera
  restart and variance display mode against a `FrameStats` probe
  (`nonzero_frac > 0.5`, `mean > 10/255`). CI installs lavapipe and
  sets `QUASI_REQUIRE_GPU=1` so a missing adapter fails rather than
  skips. Mutation-checked locally: re-introducing the 40-byte AOV
  attachments, the `vec3<u32>` `PresentU` pad, or a black present
  shader turns all five tests red. **Not covered:** browser-only WGSL
  rejections (naga ≠ Safari — see R0003 `safari-black`) and the
  `web.rs` driver itself; the raster `State` has no equivalent yet.
- [ ] **[PT-bloomv/kernel-pin]** CPU test for the Kawase chain: a
  single bright pixel at (128,128) in a 256×256 buffer, run through
  the 4-level chain, asserting
  * total energy within 10% of the input total (the dual-filter
    kernel approximates an energy-conserving Gaussian), and
  * FWHM along the centre row within `[14, 22]` px.

  Needs a CPU reference implementation of the downsample/upsample
  taps mirroring the WGSL, in the same
  `soft_knee_extract_reference` style — the shader stays the source
  of truth for rendering, the CPU mirror is the source of truth for
  the test. Note the mirror can drift from the shader; pin the tap
  weights in one place and reference them from a comment in each.

- [ ] **[PT-bloomv/bypass-assert]** Prove `cfg.bloom == None` is
  inert. Render `cornell_glass_bunny.gltf` at 128×128 / 256 spp /
  PCG / MIS-NEE twice — once on the bloom-off path, once against a
  stored pre-bloom baseline — and assert RMSE < 0.05 over the
  radiance buffer (the threshold used by
  `tests/cornell_gltf.rs`'s `cornell_quads_and_tris_render_to_the_same_image`,
  chosen to catch algorithmic change without tripping on backend FMA
  reordering). Add the `assert_no_bloom_state_touched()` hook so the
  None path panics if any bloom allocation or pass executes.
  Follow the existing `#[ignore]` convention for GPU-requiring tests
  so CI stays deterministic.

- [ ] **[PT-bloomv/halo-interaction]** The check 0029 specified and
  skipped. On the Cornell emissive-sphere scene, run plan 0021's halo
  metric in two **bloom-on** configurations:
  * baseline: `--denoise none --bloom`
  * test: `--denoise atrous --bloom`

  Assert the test metric does not exceed baseline by more than 10%.
  Comparing bloom-on against bloom-on is the load-bearing detail: a
  naive comparison against bloom-off would fail trivially, because
  raising annular-ring luminance is exactly what bloom is *for*. The
  failure mode this actually catches is "the denoiser wakes up an
  unintended halo once bloom is in the loop." Machinery to reuse:
  `halo_intensity_at_ring` in `src/pathtrace/denoise.rs`.

- [ ] **[PT-bloomv/widget]** Bloom on the web target. Widget gains a
  "Bloom" toggle and an intensity slider over `[0.0, 0.15]` centred
  on the locked `0.04` default.
  * **Budget:** ≤ 2 ms per composite on Apple M-series Safari at the
    widget's default 384×288 framebuffer, measured with
    `performance.now()` around the composite call.
  * Debounce on `change` (drag end), not `input` (drag in progress),
    so dragging can't saturate the GPU command queue.
  * The interactive path composites per accumulation step, unlike
    the offline path's single composite before readback — confirm
    that ordering still reads correctly against progressive
    accumulation, and that toggling bloom mid-render doesn't require
    an accumulation reset (it shouldn't: bloom is a post-process on
    the accumulated buffer, not part of the estimator).
  * If the budget is missed, the documented fallback is toggle-only
    (drop the slider), and if it's missed badly, gate bloom off on
    web with the measurement recorded in `Findings` — a negative
    result recorded is a result; silently shipping a janky slider is
    not.

- [ ] **[PT-bloomv/attacker-pass]** Run the `close-plan`
  render-attacker/defender pair in pair-mode on the bloom-default
  gallery renders against their pre-bloom committed versions. Per
  0029's original criterion, the attacker must surface ≥ 1 specific
  halo-improvement region on the Bistro sunlit gothic façade **and**
  ≥ 1 region where bloom did *not* soften an intended-sharp feature
  (cobblestones, awning edges). Findings land here.

## Open questions

* **Does the CPU Kawase mirror earn its keep?** It's duplicated math
  that can drift from the shader. The alternative is a GPU-executed
  test behind `#[ignore]`, which tests the real kernel but doesn't
  run in CI. Leaning CPU mirror for the kernel-pin (cheap, always
  runs, and the tap weights are stable) — but if the mirror needs
  more than ~40 lines, prefer the GPU test.
* **Is 2 ms the right widget budget?** It was asserted in 0029
  without measurement. Measure first, then decide whether 2 ms is
  demanding or generous at 384×288; record the actual number
  regardless of whether it passes.
* **Should the sweep be re-run at widget resolution?** `0.04` was
  locked at 256×256. The mip count caps at 4 levels below 512 px, so
  384×288 gets a different chain depth and possibly a different
  visual weight for the same intensity. If the widget's bloom reads
  differently from the offline renders at the same setting, that's a
  finding worth writing up, not a bug to paper over.

## Done when

* All six milestones ticked
* Kawase kernel numerically pinned; bloom-off inertness asserted
* Halo interaction test green at the ≤ 10% ceiling
* Bloom live in the browser widget with its measured composite cost
  recorded in `Findings` (or a documented, measured decision not to
  ship it on web)
* Attacker pair-mode findings recorded
* Plan moves to `Status: completed`

## Findings

- **2026-11-13** — `[PT-bloomv/widget-smoke]`. Local Metal (Apple M4),
  64×48 default Cornell: first frame `nonzero_frac 0.709, mean 33.4,
  max 248`; after 8 frames `0.795 / 36.5 / 248`. The camera-restart
  frame reproduces the first frame bit-for-bit in these stats, so
  accumulation reset is exercised, not just "something drew".
  **CI (lavapipe, ubuntu-latest):** passes with an adapter required,
  including the exact-equality restart check, so lavapipe renders
  deterministically. Cost is the catch: five separate tests took
  443 s; merged into one `State` it is 234 s, versus 0.9 s on local
  Metal, so nearly all of it is lavapipe's LLVM compile of the
  megakernel. CI wall time went from ~3 min to ~5 min. If that hurts,
  move the smoke test to its own parallel CI job. Per-scenario
  `FrameStats` are printed but cargo hides stderr for passing tests;
  run with `--nocapture` to see them.

## Followups (out of scope)

Inherited from 0029, still out of scope here:

* **PT-lens-flare** — anamorphic streaks, ghost reflections; reuses
  the bloom mip chain.
* **PT-exposure** — auto-exposure from the radiance histogram; pairs
  with `PT-sky` so time-of-day changes don't need bloom re-tuning.
* **PT-bloom-physical** — measured/parameterised lens PSF instead of
  the Kawase chain.
* **PT-bloom-aces** — re-sweep the intensity default if an ACES
  tonemap ships. `0.04` is calibrated against Reinhard; a
  compressive tonemap makes the same intensity read weaker.
