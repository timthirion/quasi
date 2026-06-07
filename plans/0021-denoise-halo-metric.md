# Quantified halo metric for PT-denoise (PT-denoise-halo-metric)

- **Status:** active
- **Last updated:** 2026-06-07
- **Last touched on:** drafting from the plan-skeptic audit of plan 0018

## Goal

Replace plan 0018's unfalsifiable Done-when criterion — *"no
longer shows a visible halo"* — with a quantitative regression
metric. Specifically: surface a per-test halo-intensity helper,
add a tonemap-ablation flag to `DenoiseParams`, and pin the
specific HDR-ratio-vs-halo-intensity relationship the
PT-denoise-tonemap fix is supposed to establish.

This directly closes the strongest single attack from the
2026-06-07 plan-skeptic dry-run on plan 0018 (recorded in
session history; not in the plan file itself). The point isn't
to relitigate plan 0018 — the implementation is correct and
the visible halo *is* gone — it's to make sure the next
denoiser change *can't quietly weaken the fix*. Today the unit
test passes for a 30%-effective regression; after this plan it
won't.

## Context

What's already in:

* `pathtrace::denoise` ships the 5-pass à-trous wavelet with
  Reinhard tonemap-then-denoise wrapping. Six tests in
  `denoise::tests` cover the kernel normalisation, uniform
  pass-through, normal-edge preservation, RMSE reduction on
  flat noise, Reinhard round-trip identity, and the existing
  single-pixel halo test (`tonemap_kills_hdr_halo_around_bright_pixel`).
* `DenoiseParams` carries `sigma_color`, `sigma_normal`,
  `sigma_depth`, `passes`. No ablation knob today — the
  tonemap wrap is unconditional inside `denoise()`.
* `gen_denoise_comparison` produces `data/output/denoise_comparison.png`
  but doesn't run in CI (a separate audit P1 from plan 0018
  that this plan does **not** address — see "Followups" below).

What the plan-skeptic audit on plan 0018 found that this plan
addresses:

* **P0:** "no longer shows a visible halo" is satisfiable by
  any subjective judgement. No numerical bound, no comparison
  metric, no committed before/after diff. A passing
  implementation could leave the halo at 80% of its original
  intensity and still tick the box.
* **P1:** The existing halo test's `1.5 × background` threshold
  is loose enough to pass for a 30%-effective fix.
* **P1:** The existing halo test uses `albedo = [1, 1, 1]`. The
  Goal section of plan 0018 argues that Reinhard pre-tonemap
  bounds the halo independent of the HDR ratio, but the design
  applies tonemap to the *demodulated* signal (`radiance /
  albedo`), not raw radiance — so the test can't actually
  exercise the case the design argument depends on.

What this plan is **not**:

* Not a re-design of the denoiser. Plan 0018's fix is correct
  empirically; this plan instruments the empirical claim.
* Not a `gen_denoise_comparison` CI regeneration — that's the
  fourth P1 from the audit and a separate, larger plan
  (would need GPU rendering in CI).
* Not SVGF variance-adaptive σ_c. That stays
  [`research/R0001-tonemap-halo-bound.md`](research/R0001-tonemap-halo-bound.md)'s
  territory.

## Design

### Ablation flag

Add `tonemap_passes: bool` to `DenoiseParams`, default `true`.
Production code unchanged behaviourally; the new tests flip it
to `false` for the tonemap-on-vs-tonemap-off ablation.

The default carries forward plan 0018's choice; switching it
off in production would silently restore the HDR-halo failure
mode. The field is documented accordingly.

### Halo-intensity helper

Extract a test-module helper:

```rust
fn halo_intensity_at_ring(
    out: &[[f32; 4]],
    width: u32,
    radius: i32,
    center: (i32, i32),
) -> f32
```

Returns the **maximum red-channel value** across the
Chebyshev ring at the given radius around `center` — same
geometry as the existing `tonemap_kills_hdr_halo_around_bright_pixel`
test (octagonal ring `|dx| ∨ |dy| == radius`), so the existing
test can be refactored onto the helper without behavioural
change.

### Three new tests

1. **`tonemap_ablation_at_hdr_ratios`** — sweep `L/ℓ ∈ {3, 10,
   30, 100, 300}`. For each ratio, run the denoiser **twice**:
   once with `tonemap_passes = true`, once with `false`.
   Measure halo intensity at radius 8 in each case. Assert
   the tonemap-on halo intensity is **strictly less than** the
   tonemap-off halo intensity at every HDR ratio ≥ 30 — the
   load-bearing claim plan 0018 makes about the fix.

   The specific bound is **discovered empirically by the test
   itself**, not assumed: the test prints the per-ratio
   intensities on failure so a denoiser change that flips the
   relationship surfaces with numbers, not a vague "looks
   wrong."

2. **`halo_with_realistic_albedo`** — single bright pixel at
   `L = 30`, all pixels carry `albedo = 0.7` (not unity). This
   exercises the demodulation pathway (the gap the audit
   named at `denoise.rs:135`). Assert halo intensity at
   radius 8 stays within 1.5× background — the same
   threshold as the existing test, now under realistic
   demodulation.

3. **`halo_from_bright_cluster`** — a `3 × 3` cluster of bright
   pixels at `L = 30` (closer to a real ceiling-light footprint
   than a single pixel), surrounded by dim. Assert halo
   intensity at radius 8 (measured from the cluster centre)
   stays within 2× background — the cluster's larger
   footprint allows more spillover by design; the bound is
   tighter than "no constraint at all" but looser than the
   single-pixel test.

The asserted thresholds are deliberately conservative — the
goal is to catch regressions where the tonemap fix becomes
30%-effective, not to pin down the asymptotic optimum. A
future PR that tightens the denoiser would be expected to
tighten the bounds.

## Milestones

1. [ ] `tonemap_passes: bool` added to `DenoiseParams` (default
       true). Existing tests + `denoise_comparison.png`
       regeneration both produce byte-stable output.
2. [ ] `halo_intensity_at_ring` helper extracted; existing
       `tonemap_kills_hdr_halo_around_bright_pixel` refactored
       onto it (byte-stable output, just deduplication).
3. [ ] `tonemap_ablation_at_hdr_ratios` test added. Verifies
       `tonemap < no-tonemap` at HDR ratios ≥ 30.
4. [ ] `halo_with_realistic_albedo` test added. Closes the
       demodulation-pathway audit gap.
5. [ ] `halo_from_bright_cluster` test added. Exercises a
       footprint closer to real ceiling lights.
6. [ ] `denoise::tests` count rises from 6 → 9; all pass.
7. [ ] `close-plan` skill orchestration returns clean
       (plan-skeptic + code-attacker/defender; no hero PNG
       changes, so render-attacker/defender skipped).

## Open questions

* **Does the tonemap-on-vs-tonemap-off relationship actually
  invert at very low HDR ratios (L/ℓ < 3)?** The Reinhard
  curve compresses the colour distance, which could in
  principle *increase* the halo when the source isn't HDR.
  The test deliberately starts at `L/ℓ = 3` rather than 1
  because of this. If the empirical sweep shows the
  inversion happening at higher ratios than expected, the
  test's `≥ 30` floor moves up — but we discover the
  number, not assume it.
* **Should the test use peak halo, mean halo, or integrated
  halo?** Going with peak (max over ring) — simplest,
  matches existing test geometry, easiest to interpret on
  failure. Mean / integrated are follow-ups if peak proves
  too noisy across CI runners.
* **Will the helper geometry (Chebyshev ring) bite us when
  the bright source is a cluster?** The ring is computed
  from the cluster centre, not the boundary. At radius 8
  with a 3×3 cluster, the actual distance from the cluster
  *edge* is 7 in one direction and 9 in another. The test
  acknowledges this with a looser 2× threshold; if it
  surfaces as a real measurement issue, the helper grows a
  `min_distance_from` parameter.

## Done when

* All five code milestones tick.
* `cargo test --lib denoise` reports 9 tests, all green
  (was 6 before this plan).
* The `tonemap_ablation_at_hdr_ratios` test's recorded
  per-ratio halo intensities are reproduced in this plan's
  body — once the empirical sweep runs, replace the "TBD"
  marker below with the numbers, making the structural
  claim auditable post-close.
* `close-plan 0021` returns clean — plan-skeptic raises no
  unaddressed P0; code-attacker/defender pair resolves all
  P0 attacks (accept-with-fix or refute-with-citation).
* CI green at HEAD.

### Empirical sweep results (filled at close time)

| L / ℓ | halo @ r=8 (tonemap on) | halo @ r=8 (tonemap off) | tonemap / no-tonemap |
|------:|------------------------:|-------------------------:|---------------------:|
|     3 |                     TBD |                      TBD |                  TBD |
|    10 |                     TBD |                      TBD |                  TBD |
|    30 |                     TBD |                      TBD |                  TBD |
|   100 |                     TBD |                      TBD |                  TBD |
|   300 |                     TBD |                      TBD |                  TBD |

## Followups (out of scope for this plan)

* **`gen_denoise_comparison` in CI.** The plan-skeptic audit's
  remaining unaddressed P1 — currently the committed
  `denoise_comparison.png` can stay stale if the denoiser
  regresses. Needs GPU rendering in CI (`actions/runner`
  has CPU-only `wgpu` adapter support via `wgpu`'s software
  fallback; would need a benchmark to confirm runtime is
  acceptable). Separate plan when prioritised.
* **Log-luminance vs Reinhard tonemap comparison.** The
  audit flagged that plan 0018's "Open questions" forecloses
  the log-luminance alternative. Belongs in research/R0001
  rather than here.
* **Plan-skeptic post-hoc audit note on plan 0018 itself.**
  Decided in session 2026-06-07 to leave plan 0018 untouched
  rather than retroactively edit closed plans. This plan
  carries the audit forward via the Goal section.
