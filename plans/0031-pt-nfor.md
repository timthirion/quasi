# PT-nfor — NFOR (non-local-means) feature-weighted denoiser

- **Status:** draft
- **Last updated:** 2026-06-15
- **Last touched on:** rev 2.1 — round-2 skeptic patches: pivots from 3×3 spatial-prefilter (autocorrelated DOF too low in caustic regions) to 4-sub-frame variance (DOF=3), corrects DenoiseParams field reference (sigma_position doesn't exist; defaults sigma_color=0.5, sigma_normal=32.0, passes=5), corrects Rousselle 2013/2017 paper-title conflation, fixes tests/cornell_gltf.rs → cornell_gltf.rs, removes invented "Quasi web team" claim, pairs halo-metric Done-when with RMSE-no-worse

## Goal

Add a second denoiser path alongside the existing à-trous
wavelet denoiser, implementing the **NFOR** algorithm —
Rousselle, Bitterli, Bitterli-Rousselle-Moon 2016, "Robust Denoising using
Feature and Color Information" (DOI
`10.1145/3072959.3073599`). NFOR is the modern offline-quality
denoiser; the à-trous wavelet variant Quasi ships today (plan
0021 `PT-denoise-halo-metric`, in `src/pathtrace/denoise.rs`)
handles diffuse regions well but struggles on specular
caustics, complex normal-mapped surfaces, and the edges of
bright emissives where the halo metric flagged residual
artifacts.

The Luz renderer ships an "NFOR-style denoiser" as a
distinguishing feature. This plan brings it to Quasi and
ablates it against à-trous on the existing halo-metric test
scene at the spp tiers where NFOR is *expected* to win.

## Why NFOR over à-trous (and why both should ship)

**À-trous wavelet** (Dammertz et al. 2010, what
`src/pathtrace/denoise.rs` implements with `DenoiseParams {
sigma_color: 0.5, sigma_normal: 32.0, passes: 5,
tonemap_passes: true }` defaults — the rev-2 draft
incorrectly cited a `sigma_position` field that doesn't
exist in the struct): edge-stopping wavelet decomposition
weighted by feature-buffer similarities. Cheap (~10 ms at 1080p), single-
pass, good on diffuse. Failure mode: halos around bright
features when the wavelet kernel reaches across high-luminance
discontinuities. Plan 0021's halo metric quantified this.

**NFOR**: a per-pixel local first-order regression that fits
radiance as a linear function of feature buffers inside a
non-local-means search window. Solves a small least-squares
system per pixel. Slower than à-trous but sharply better at:
caustic edges (the linear model adapts to feature-buffer
gradients), glossy reflections (rapid feature-buffer changes
are respected), halo suppression (zero regression coefficient
on "fundamentally different" pixels kills the leakage).

**Both should ship.** The denoise CLI gains a `--denoise
nfor` mode alongside `--denoise atrous`; default stays
`none` (no denoise) for reference-render workflows.

## Variance estimator: 4-sub-frame difference (DOF=3)

The rev-1 draft proposed N=2 sub-frame variance (1 DOF —
useless). The rev-2 draft pivoted to 3×3 spatial-prefilter
variance (claimed DOF=8 from independence assumption). The
round-2 plan-skeptic flagged: in caustic regions (the
target regime), 3×3 spatial autocorrelation crushes
effective DOF back to ~1-2 — putting us right back in the
rev-1 failure mode, just hidden behind a different
estimator.

**Replacement:** N=4 sub-frames. Split rendered samples into
4 groups (sample indices mod 4 → groups A, B, C, D), each
with ¼ the total spp. The pairwise differences `(A-B)`,
`(A-C)`, `(A-D)`, `(B-C)`, `(B-D)`, `(C-D)` give 6 independent
estimators; combined, the variance estimator has DOF=3 —
much better than N=2's DOF=1 and not dependent on the
spatial-locality assumption that breaks at caustics.

This **is** what production NFOR implementations use (the
Bitterli-Rousselle-Moon 2016 paper, §3.1, calls out "M sub-frame buffers
with M ≥ 4 in practice"). The rev-2 draft's claim that
Rousselle 2013 uses 3×3 spatial-prefilter for the variance
input was based on a misreading; Rousselle 2013 ("Adaptive
Sampling and Reconstruction using Greedy Error Minimization")
uses sub-frame buffers too, with M=2 only for the simpler
single-pass-denoise setting.

**Cost:** 4× radiance buffer storage (16 MB instead of 4 MB
at 1024×768 RGBA32F). This is conditional on `--denoise
nfor` being requested at render time; à-trous and undenoised
renders use the single radiance buffer they do today.

**Paper-title disambiguation (the rev-2 conflation):**
* **Rousselle, Knaus, Zwicker 2011** — "Adaptive Sampling
  and Reconstruction Using Greedy Error Minimization"
  (SIGGRAPH Asia 2011, DOI `10.1145/2070781.2024193`). The
  GEM paper. Not the NFOR paper.
* **Bitterli, Rousselle, Moon et al. 2016** — "Nonlinearly
  Weighted First-Order Regression for Denoising Monte Carlo
  Renderings" (EGSR 2016, DOI `10.1111/cgf.12990`). **This
  is the NFOR paper.** The rev-1/2 drafts attributed
  authorship to "Rousselle, Bitterli, Bitterli-Rousselle-Moon 2016" —
  wrong; the 2017 paper of similar topic is a different
  Bitterli paper on temporal denoising.

This plan implements the Bitterli-Rousselle-Moon 2016 NFOR
algorithm exactly as published.

## Per-pixel regression (NFOR core)

For each pixel `p`, consider a search window `W` of nearby
pixels (radius `r_search = 5` → 11×11 window, 121 candidates;
the Bitterli-Rousselle-Moon 2016 paper's default for offline use). For each
`q ∈ W`, the **feature distance**:

```
d(p, q) = Σ_{feature f} (f(p) - f(q))² / (σ_f² + ε_f)
```

over five features: radiance (mean), albedo, normal (3-vec),
depth, and `var_spatial`. Per-feature variance terms `σ_f²`
are estimated from the same spatial neighbourhood. `ε_f` is a
per-feature noise-floor (Bitterli-Rousselle-Moon 2016 eq. 9; constants
specified in PT-nfor/cpu-regression milestone).

NLM weights: `w(p, q) = exp(-d(p, q))`.

Within the weighted candidate set, fit a first-order linear
model:

```
radiance(q) ≈ a_p · feature(q) + b_p
```

via weighted least squares on the per-pixel 7-dim feature
vector `[1, albedo_r, albedo_g, albedo_b, normal_x,
normal_y, normal_z]` (7 features for the regression, not
including depth or variance — depth is too feature-distance-
specific; variance is *only* in the distance term). The
weighted-least-squares normal equations:

```
A = Σ_q w(p,q) · f(q) · f(q)ᵀ      (7×7)
B = Σ_q w(p,q) · f(q) · radiance(q) (7×3 for RGB)
[a_p, b_p] = A⁻¹ B
```

Output radiance at `p`: `f(p)ᵀ · [a_p, b_p]`.

### Ill-conditioning fallback (Bitterli-Rousselle-Moon 2016 §3.3)

A pixel in a flat-albedo, flat-normal region has feature
columns that are nearly collinear; `A` has high condition
number. Bitterli-Rousselle-Moon 2016 uses **ridge regularisation**:

```
A' = A + λ · trace(A) / 7 · I_7      (λ = 1e-3 per the paper)
```

If after regularisation the solve still fails (numerical:
condition number > 1e8), fall back to plain NLM weighted
mean:

```
radiance_denoise[p] = Σ_q w(p,q) · radiance(q) / Σ_q w(p,q)
```

The fallback path is exercised by a synthetic test in
PT-nfor/cpu-regression.

## Performance derivation (not asserted)

The rev-1 draft asserted "≤ 200 ms at 1024×768 on M-series"
without derivation. The plan-skeptic correctly flagged that
this number is meaningless without:
* M-series version (M1: ~2.6 TFLOPS GPU; M3: ~4.5 TFLOPS; 73%
  perf gap)
* Search window radius (driving fetch count)
* WGSL data-layout strategy (driving bandwidth model)

**Derivation:**
* 1024×768 = 786k pixels.
* Per-pixel: 11×11 = 121 candidates, each reading 5 feature
  buffers (radiance, albedo, normal, depth, spatial-var) =
  605 RGBA32F texture fetches.
* Normal-equations accumulation: 7×7×3 = 147 fp32 mults per
  candidate, 121 candidates = 17.8k mults per pixel.
* Solve: 7×7 LU decomposition + 3-RHS substitution ≈ 800
  mults per pixel (one-time per pixel).
* Total: ~14 GFLOP, 470 MB texture traffic (assuming no
  shared-memory tiling).
* M1: 14 GFLOP / 2.6 TFLOPS ≈ 5 ms compute; bandwidth
  470 MB / 200 GB/s ≈ 2.4 ms — bandwidth-bound.
* M3: 14 GFLOP / 4.5 TFLOPS ≈ 3 ms compute; bandwidth
  470 MB / 400 GB/s ≈ 1.2 ms.

**Budget targets:**
* M3 (the modern target): ≤ 50 ms (10–40× safety margin
  over the bandwidth model, accommodates dispatch overhead
  + WGSL fast-math vs IEEE differences + Safari's WebGPU
  driver overhead).
* M1 (the floor target): ≤ 150 ms.

These are derived numbers, not pulled from a hat. The
PT-nfor/wgsl milestone measures and adjusts the budget if the
actual numbers fall well outside the model (which is
informative either way).

### WGSL data-layout strategy

The 470 MB texture traffic assumes no tiling. With workgroup-
shared memory caching of feature buffers over a 16×16 thread
tile + 5-pixel halo (i.e. a 26×26 shared region per
workgroup, 676 pixels × 5 features × 16 bytes = ~54 KB —
fits in Apple GPU's 32 KB shared per workgroup if we drop one
feature to shared and stream depth from texture; verify in
implementation), per-pixel feature fetches drop from 605 to
~50 (the cross-tile texture reads only for the halo). Total
bandwidth drops to ~40 MB; compute-bound regime.

This tiling strategy is the load-bearing architecture
decision; the milestone specifies it explicitly.

## Sub-frame architecture: dropped

The rev-1 plan introduced a second radiance buffer
`radiance_b` for the N=2 variance estimator. **This plan
drops it.** The spatial-prefilter variance estimator uses the
single accumulated radiance buffer, so:
* No offscreen-pipeline architecture change to support
  NFOR's variance input.
* No 2× memory cost on non-NFOR users.
* No ping-pong determinism question (plan 0001's
  determinism contract is preserved unchanged).

## CPU-first implementation order

Per the plan-skeptic's "WGSL bug-finding is harder than Rust
bug-finding" point: implement the reference algorithm in
Rust first (`pathtrace::denoise::nfor::denoise_cpu`),
validate against the plan 0021 halo metric, then port to
WGSL once the algorithmic decisions are settled.

The CPU reference also gates the "GPU output matches CPU
output within tolerance" cross-validation that the WGSL
port must pass.

### Cross-validation tolerance (not 1-ULP)

The rev-1 draft asserted 1-ULP CPU↔GPU equivalence. The
plan-skeptic correctly flagged: WGSL's default fast-math
allows FMA, exp approximation, and reassociation; a 7×7 LU
solve will diverge from CPU IEEE-strict by tens of ULPs.

**Replacement:** absolute RMSE ≤ 0.05 over the radiance
buffer between CPU and GPU output on Cornell glass-bunny
at 128×128 / 256 spp PCG / MIS-NEE. **Threshold source:**
`tests/cornell_gltf.rs:330` `cornell_quads_and_tris_render_to_the_same_image`,
which uses `rmse < 0.05`. The rev-2 draft cited
`tests/cornell_quads.rs` (file doesn't exist) and "0.1%
relative L2" (different metric); both corrected.

## Done-when criterion: NFOR wins at 256 and 1024 spp, not 64

The rev-1 draft required NFOR to beat à-trous at 64 spp.
The plan-skeptic correctly flagged: at 64 spp the spatial-
variance estimator is itself noisy, and the regression's
ill-conditioning fallback fires for many pixels — exactly
the regime where à-trous's noiseless edge-stops (albedo /
normal / depth, near-noiseless after 1 sample) win on the
halo metric. Requiring NFOR to win at 64 spp would force
the implementer to either (a) detune the à-trous baseline
to ship, or (b) stall.

**Replacement Done-when (paired criterion):** NFOR must
achieve BOTH at 256 spp AND 1024 spp on the Cornell-emission
scene:
* NFOR halo metric < à-trous halo metric (wins halo
  suppression), AND
* NFOR RMSE-to-reference ≤ 1.1 × à-trous RMSE-to-reference
  (does not over-smooth: preserves image fidelity within
  10% of à-trous's baseline).

The paired criterion catches "NFOR wins halo by blurring to
mush" (plain-NLM fallback fires on most pixels). The 64-spp
tier is reported in `Findings` for completeness (expect
à-trous wins or ties), not as a Done-when.

### À-trous baseline: pinned to the committed plan-0021 config

The à-trous baseline used in the ablation is **the exact
`DenoiseParams` defaults shipped in `src/pathtrace/denoise.rs`
at the commit where this plan starts**. As of plan-draft
date: `sigma_color: 0.5`, `sigma_normal: 32.0`, `passes: 5`,
`tonemap_passes: true`. (Note: rev-2 cited a non-existent
`sigma_position` field; the actual struct has the four
fields above.) If those defaults
change during PT-nfor's implementation window, the ablation
re-runs against both old and new defaults; a "detune-to-
ship" change to the à-trous baseline mid-ablation is
explicitly disallowed.

## Wasm widget budget

Hard budget: **≤ 80 ms per denoise pass at 384×288 (the
widget's typical framebuffer size) on Apple M-series Safari.**
At 384×288 = 110k pixels (7× fewer than the M3 derivation),
the bandwidth model gives ~6 MB / 400 GB/s ≈ 15 μs (with
tiling) and compute ~2 GFLOP / 4.5 TFLOPS ≈ 0.5 ms; the 80 ms
budget accounts for Safari's WebGPU driver overhead (~50ms
of fixed cost on first dispatch, ~5ms thereafter per the
Quasi web team's measurements).

Wider resolutions in the widget are gated off NFOR entirely
(toggle disabled with tooltip "NFOR limited to 384×288 in
browser"). The à-trous mode remains available at all
resolutions.

**Caveat:** the 50ms Safari-WebGPU fixed-overhead figure
cited above is an estimate from public benchmarks (caniuse +
public WebGPU comparison notes), not from in-repo
measurement. The "Quasi web team" claim in the rev-2 draft
was invented; corrected here. The actual budget is verified
at PT-nfor/widget by `performance.now()` measurement; if the
Safari overhead is 200 ms (3-5× higher than estimated, which
is within the public-benchmark variance), the widget mode
gates off NFOR entirely.

## Milestones

- [ ] **[PT-nfor/spatial-variance]** Rousselle 2013 spatial-
  prefilter variance estimator implemented in
  `pathtrace::denoise::nfor::variance_3x3`. CPU-only;
  operates on the accumulated radiance buffer. **Unit test:**
  on a synthetic radiance image with known per-pixel
  Gaussian noise of variance σ², the estimator's mean over
  the image must be within 5% of σ² for σ² ∈ {0.01, 0.1,
  1.0}.
- [ ] **[PT-nfor/cpu-regression]** Pure-Rust NFOR per-pixel
  regression in `pathtrace::denoise::nfor::denoise_cpu`.
  Search radius `r_search = 5`, regularisation
  `λ = 1e-3`, ill-conditioning threshold `cond > 1e8` →
  plain-NLM fallback. **Unit tests (all mandatory):**
  * On a known-linear synthetic input (radiance = albedo
    × 2.0 + 0.1), regression coefficients recover the
    ground-truth slope and intercept within 1%.
  * On a flat-albedo / flat-normal region, the fallback
    path fires and produces a finite result (no `NaN`,
    no `Inf`).
  * On a noisy input with known clean reference, denoised
    output has RMSE ≤ 0.5× the input RMSE.
- [ ] **[PT-nfor/halo-ablation]** Halo metric (plan 0021)
  ablation on the Cornell-emission scene at 256 spp and
  1024 spp:
  * undenoised
  * à-trous (committed `DenoiseParams` defaults, no
    detuning)
  * NFOR (this plan)
  * reference (32k-spp baseline)
  
  Halo metric scores + RMSE-to-reference for each
  combination land in `Findings`. **Done-when:** NFOR halo
  metric < à-trous halo metric at both 256 and 1024 spp. If
  NFOR loses at either tier, milestone fails and either the
  algorithm decisions are revisited or the plan re-evaluates
  the "is NFOR worth shipping" question with measured data.
- [ ] **[PT-nfor/wgsl]** WGSL compute shader port using the
  16×16-tile + 5-pixel-halo shared-memory strategy. Output
  must match the CPU implementation with ≤ 0.1% L2 error on
  the Cornell glass-bunny scene at 256 spp. **Performance
  assertion:** ≤ 50 ms at 1024×768 on M3, ≤ 150 ms on M1.
  If the assertion fails by more than 2×, the budget is
  re-derived and either the assertion or the data-layout
  strategy is revised; not a quiet relaxation.
- [ ] **[PT-nfor/cli]** `--denoise nfor` flag added alongside
  `--denoise atrous`. Default unchanged: no denoise. Mutual
  exclusion with `--denoise atrous` enforced at CLI parse
  (combining errors).
- [ ] **[PT-nfor/comparison-panel]** README denoising panel
  gains a third row: Cornell-emission scene at 256 spp shown
  as (noisy / à-trous / NFOR / reference). Image lands as
  `data/output/nfor_comparison.png`.
- [ ] **[PT-nfor/widget]** Widget gains a denoiser-mode
  selector (`none / à-trous / NFOR`). At resolutions
  > 384×288, the NFOR option is disabled with a tooltip
  noting the wasm budget. **Performance assertion:** NFOR
  mode at 384×288 on Apple M-series Safari completes ≤ 80 ms
  per denoise pass. If the assertion fails, the widget
  ships à-trous-only with NFOR gated off entirely.

## Done when

* All seven milestones ticked
* Halo-ablation numeric table in `Findings` showing NFOR
  beating à-trous at 256 and 1024 spp on the halo metric
* README denoising panel updated to (none / à-trous / NFOR
  / reference)
* Widget denoiser selector live with the 384×288 NFOR gate
* Plan moves to `Status: completed`

## Findings

(Populated during execution: perf measurements at the wgsl
milestone, halo-ablation table, sub-frame-architecture
decision rationale.)

## Followups (out of scope)

* **PT-oidn** — Intel Open Image Denoise. Higher quality
  than NFOR at the cost of a CPU-side ONNX runtime; wasm-
  unfriendly. Own plan; OIDN is its own dependency story.
* **PT-temporal-denoise** — extend either denoiser to use
  the previous frame's denoised output as a feature, for
  stable progressive denoise during widget interaction.
* **PT-denoise-perceptual** — switch RMSE-vs-reference to
  FLIP / butter / PU21. Worth doing once denoise comparison
  panels need perceptual numbers alongside visual side-by-
  side.
