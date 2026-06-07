# Tonemap-then-denoise: an analytic halo bound for HDR à-trous wavelets

- **Status:** hypothesis
- **Last updated:** 2026-06-06
- **Last touched on:** drafting from the PT-denoise-tonemap fix in plan 0018
- **Paper target:** SIGGRAPH 2027 Short Papers OR Eurographics 2027 Short
  Papers; backup target EGSR 2027 work-in-progress.
- **Implementation foundation:** plan 0017 (PT-denoise) +
  plan 0018 (PT-denoise-tonemap).

## Hypothesis

For a 5×5 B3-spline à-trous wavelet with `k` iterations and
colour edge-stop `σ_c`, applied to a buffer where one pixel
carries HDR luminance `L` and its neighbours carry `ℓ ≪ L`, the
**halo radius** (the distance to which `|c_q - ℓ|` exceeds an
`ε`-tolerance after `k` passes) is bounded by a function
`H(L/ℓ, σ_c, k)` that grows **monotonically** in `L/ℓ`.

**Claim:** Applying a Reinhard pre-tonemap `t = c / (1 + c)`
before the wavelet bounds the halo by `H(1, σ_c, k)` —
**independent of `L/ℓ`** — at the cost of one division per
pixel per pass.

Intuitively, the colour edge stop `w_colour = exp(-‖c_p - c_q‖²
/ σ_c²)` collapses to ~0 around the bright pixel but the kernel
weight still pulls a small amount of `L`-mass into the
neighbour stencil. Pre-tonemap squashes `L` into `[0, 1)`,
removing the HDR-driven divergence.

## Related work

Cited positively:

* **Dammertz et al. 2010**, *Edge-Avoiding À-Trous Wavelet
  Transform for Fast Global Illumination Filtering*. The
  algorithm we're analysing.
* **Schied et al. 2017**, *Spatiotemporal Variance-Guided
  Filtering (SVGF)*. Solves the halo problem via a
  variance-adaptive `σ_c` driven by a per-pixel variance AOV.
  Our contribution is **a fix that doesn't require a variance
  AOV**.
* **Reinhard et al. 2002**, *Photographic Tone Reproduction
  for Digital Images*. The operator we exploit.
* **Bitterli & Jarosz 2015**, *Beyond Points and Beams:
  Higher-dimensional Photon Samples for Volumetric
  Appearance*. Side reference for edge-stop kernel analysis.

The gap our hypothesis addresses:

* Existing analysis of edge-aware à-trous wavelets is either
  (a) empirical (Dammertz) or (b) assumes variance is
  available (SVGF and follow-ups). **No closed-form bound on
  the halo radius as a function of the HDR ratio appears in
  the literature** that we could find — and consequently no
  analysis of how cheap pre-conditioners (tonemap, log-luminance)
  compare against variance-adaptive σ_c.

## Experimental design

### Phase 1 — analytic derivation

Working assumption: the per-pair weight reduces to a separable
form on a 1D bright-pixel-vs-dim-pixel boundary. Derive
`H(L/ℓ, σ_c, k)` by tracking, for each pass, the contribution
of the bright pixel to a target pixel at distance `r` (in
kernel-step units).

Open question: does the proof go through for general 2D edge
geometry or only for the 1D case? Expect the 1D case is
sufficient for a short paper; full 2D is a follow-up.

### Phase 2 — synthetic sweep

A grid:

* `L / ℓ ∈ {1, 3, 10, 30, 100, 300, 1000}`
* `σ_c ∈ {0.1, 0.2, 0.5, 1.0, 2.0}`
* `k ∈ {1, 2, 3, 5, 8}`

For each cell:

* Generate a 128×128 buffer with one centre-pixel at `L`,
  surroundings at `ℓ`.
* Run the raw à-trous denoiser.
* Run the tonemap-then-denoise variant.
* Measure: halo radius (largest `r` where `out(r) - ℓ > 0.1 · ℓ`).
* Plot measured vs predicted bound.

The denoiser code we'll use is `pathtrace::denoise` at HEAD,
with the tonemap wrap controlled by a feature flag.

### Phase 3 — real-scene validation

3–5 scenes with strong HDR features:

* `cornell_glass_bunny.gltf` (ceiling light + caustic).
* `outdoor_normal_bunny.gltf` (sun in env map).
* A scene with multiple emitters of mismatched intensity
  (`cornell_many_lights.gltf` is the obvious candidate but
  could synthesize a deliberately-adversarial scene with one
  L=1000 emitter).

For each: render at 32 / 64 / 128 / 256 spp. Apply both
denoisers. Compare against a 4096 spp reference. Report
PSNR / SSIM / FLIP for each (spp, denoiser) cell.

## Baselines

| Baseline | Why it's the right comparison |
|----------|-------------------------------|
| Raw à-trous (PT-denoise pre-0018) | The starting point. Shows what we're fixing. |
| Log-luminance edge stop | Equivalent fix via a different mechanism. If it matches us perceptually, we're choosing between two right answers; if it loses, our choice is justified. |
| SVGF (variance-adaptive σ_c) | The gold standard. Won't beat it in absolute quality but will be cheaper (no variance AOV) and provably bounded. |
| OIDN (learned) | The "what production renderers use" point of comparison. Not algorithmically comparable but a useful reality check. |

**Strongest baseline:** SVGF. Beating it is unlikely; matching
it at lower compute cost is the realistic goal.

## Milestones

1. **Analytic derivation** — `H` derived for 1D case;
   sufficient-condition proof for the pre-tonemap bound.
2. **Synthetic sweep** — full grid; plots; cross-validation of
   bound vs measurement at every cell.
3. **Real-scene validation** — 3–5 scenes; PSNR/SSIM/FLIP tables;
   visual figures of halos in raw vs tonemapped.
4. **Comparison with log-luminance + SVGF + OIDN** — same
   scenes, same metrics. Honest assessment of where we win,
   where we lose, where we tie.
5. **Write-up + figure crafting.**
6. **Submission.**

## Paper target

* **Primary:** SIGGRAPH 2027 Short Papers. Fits the "small
  complete result" mould.
* **Backup:** Eurographics 2027 Short Papers.
* **Fallback:** EGSR 2027 work-in-progress poster.

The contribution narrative is:
> "Edge-aware à-trous wavelets halo around HDR features. We
> derive a closed-form bound on the halo radius as a function
> of the HDR ratio and prove that a Reinhard pre-tonemap caps
> the halo at the LDR worst case, independent of the HDR
> ratio. Empirical validation across synthetic + real scenes
> confirms the bound and shows the fix matches SVGF
> perceptually at a fraction of the compute cost (no variance
> AOV required)."

## Done when

**Accept criteria** (move to `writing`):

* Analytic bound is derived for the 1D case.
* Synthetic sweep confirms the bound across all cells within
  a tight tolerance.
* Real-scene validation shows visible halo reduction on at
  least 4 of 5 scenes at 64 spp.
* At least one scene where tonemap-then-denoise visibly beats
  raw à-trous + matches SVGF within 1 dB PSNR.

**Abandon criteria** (move to `abandoned`, record why):

* Bound doesn't hold empirically — the colour edge stop's
  non-linearity in tonemapped space introduces a *different*
  failure mode (e.g. crushed-shadow halos) that's worse than
  the HDR halo.
* SVGF or OIDN beat us so badly across the board that the
  cheaper-than-SVGF angle has no headroom.

## Findings

*(none yet — move from `hypothesis` to `experimenting` when
the first analytic step is in hand.)*
