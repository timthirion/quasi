# Gradient scatter and PRB on a browser-class compute API

- **Status:** hypothesis
- **Last updated:** 2026-09-15
- **Last touched on:** rev 2 — first `research-critic` pass. Rev 1 was
  drafted and attacked the same day; the critic returned 4×P0 and
  9×P1. Rev 2 re-poses the contribution entirely. The headline change:
  rev 1 claimed novelty from "WGSL has no autodiff," which is **false**
  (Slang compiles reverse-mode autodiff to WGSL; Brush trains 3D
  Gaussian splats in-browser on WebGPU via Burn). Rev 1 also missed
  that this repo's integrator is a `@fragment` shader with read-only
  storage bindings and therefore *cannot scatter gradients at all*.
  Rev 2 makes those two facts the subject rather than the blind spot.
- **Paper target:** Web3D 2027 (genre match — see *Venue*, and cf. the
  Web3D 2024 OpenPBR/WebGPU path-tracer implementation report,
  `10.1145/3665318.3677158`). Alternate: JCGT rolling, **only** if the
  gradient-scatter result stands as a technique article on its own.
  A negative envelope result is a **blog post, not a paper** — see
  *Done when*.
- **Implementation foundation:** plan 0001 (megakernel PT + NEE/MIS),
  0006 + 0008 (heterogeneous media, `Grid3D`), 0028 (adaptive sampling
  — the only existing compute shader in the PT path), 0013 (widget
  embed).

## Hypothesis

**On a compute API with no `f32` atomics, no `f64`, and no autodiff,
the binding constraint on physically-based inverse rendering is the
gradient scatter, not the derivative derivation — and there is a
fixed-point scatter scheme whose contention and precision behaviour
can be characterized well enough to make PRB viable in a browser.**

Three separately falsifiable sub-claims:

1. **Scatter.** A fixed-point u32 gradient accumulator with an
   adaptively-rescaled exponent achieves ≥ 99% agreement with an f64
   CPU reference across the optimization, at ≤ 2× the cost of a
   (hypothetical) native f32 atomic add, **and** beats a
   `atomicCompareExchangeWeak` CAS loop by ≥ 5× in the high-contention
   regime (few parameters, many samples).
2. **Replay fidelity.** A compute-ported primal integrator and its PRB
   replay pass reproduce identical path sequences, verified
   bit-exactly, across ≥ 10⁶ paths.
3. **Envelope.** There is a measured (parameters × resolution × spp)
   region in which a **plain-PRB** optimization step completes inside
   an interactive budget in a browser, and that region **contains the
   volume experiment E2's actual configuration** (not merely some
   non-empty region — see *Done when*).

   **Measured 2026-09-15 and already constrained.** `primal-timing`
   has run (see *Findings*). Native primal at 128² @ 4 spp is ~45 ms,
   so PRB at ≥ 2× lands at ~90 ms — inside a 100 ms budget with almost
   no margin, *natively*, before the browser's overhead. 256² @ 16 spp
   — rev 1's stated cap — is ~100–170 ms primal, i.e. **already busted
   by the primal alone.** Sub-claim (3) is therefore scoped to
   **≤ 128² at ≤ 4 spp** unless the submit-batching lever below
   recovers headroom, and the budget is stated as *"≥ 5 optimization
   steps/second,"* not a hard 100 ms.

Sub-claim (1) is the contribution. (2) is the correctness precondition
that makes (1) meaningful. (3) is the delivery result.

## What changed in rev 2, and why

Rev 1's framing was "WGSL has no autodiff, so hand-deriving adjoints is
the contribution." That is wrong twice over:

- **Slang already compiles reverse-mode autodiff to WGSL** (Bangaru et
  al., SIGGRAPH Asia 2023; `shader-slang.org/slang/user-guide/autodiff`),
  and its playground runs a differentiable 2D Gaussian-splatting
  *training loop* in-browser on WebGPU.
- **Brush** (`github.com/ArthurBrussee/brush`) trains full 3D Gaussian
  splats in a browser tab on WebGPU through Burn's autodiff over
  CubeCL-generated WGSL.

Rev 1 asserted that browser neural-field work "runs inference in the
browser but trains elsewhere." That sentence was written from memory,
is false, and would have been found by a reviewer in one search.

**The distinction that survives, and that rev 2 is built on:** PRB is
not autodiff. Naive reverse-mode over a path tracer tapes the whole
path graph — the memory blowup PRB exists to avoid. You cannot obtain
PRB by pointing an autodiff compiler at an integrator; the
constant-memory replay structure is a hand-designed algorithm. So
Slang and Burn are genuine baselines for *local derivative labor*
(BSDF and phase-function derivatives), and rev 2 plans to **use Slang
for exactly that** while hand-writing the path-level PRB structure.
What neither gives you is the gradient scatter, which on WebGPU is
unsolved because the API has no f32 atomics.

That scatter problem is not rendering-specific — it is the general
"reduction-heavy GPU compute on WebGPU" problem — which is what makes
it worth writing up.

## Prerequisite bill (read this before scheduling anything)

Most of this plan is renderer surgery that happens **before** any
research content. Rev 1 hid this; rev 2 states it up front. All items
verified in-tree on 2026-09-14/15.

1. **P0 — The integrator cannot scatter gradients.**
   `src/pathtrace/shaders/pathtrace.wgsl` has exactly two entry points,
   `@vertex` (:202) and `@fragment` (:2317); zero `@compute`. Every
   storage binding is `var<storage, read>` (:158–:188). The pipeline is
   `create_render_pipeline` (`src/pathtrace.rs:1281`) — there is no
   compute pipeline in the PT path at all; `adaptive_mask.wgsl` is the
   sole compute shader in the subsystem. The adjoint pass needs
   `read_write` storage plus atomics; writable fragment-stage storage
   is limit-gated in WebGPU (`maxStorageBuffersInFragmentStage` is 0 in
   Compatibility Mode) and fragment shaders have no workgroup memory,
   so there is no reduction tree before the scatter. **A compute port
   of the ~2 300-line megakernel is a hard prerequisite**, and it is
   larger than every other item here combined.
2. **P0 — No `f32` atomics in WebGPU.** WGSL core atomics are
   `i32`/`u32` only. This is sub-claim (1) and the plan's subject; it
   is listed here because it is also a *blocker*, not only a topic.
3. **P0 — `metallic` has no gradient.** `pathtrace.wgsl:1558` documents
   the shading branch as "`metallic > 0.5` picks GGX; else Lambertian"
   — a hard branch, so ∂L/∂metallic ≡ 0 almost everywhere, and any FD
   check on it passes vacuously. `:2262` additionally gates a
   mirror-collapse path on `metallic > 0.5 && roughness < 0.2`, putting
   a discontinuity in `roughness` too. **A continuous metallic-blend
   BSDF is a prerequisite**, and it changes existing render output, so
   it needs its own reference-comparison pass.
4. **P1 — `Grid3D` density is u8, and the GPU side is `R8Unorm`.**
   `grid.rs:61` stores `voxels: Vec<u8>`; `src/pathtrace.rs:858` uploads
   `R8Unorm` sampled with hardware trilinear (`pathtrace.wgsl:1265`).
   1/255 quantization swamps small gradient steps. `R32Float` is **not
   filterable** in WebGPU without the optional `float32-filterable`
   feature, and **atomics cannot target storage textures at all** — so
   the differentiable path needs a parallel f32 storage buffer plus a
   buffer→texture copy each iteration, or a hand-rolled 8-tap trilinear
   in the hot loop. Rev 1 called this "plus the WGSL sampler change,"
   which understated it.
5. **P1 — Native offscreen creates a device per call.**
   `src/pathtrace/offscreen.rs:378` (`render_offscreen_async`) calls
   `gpu::make_instance()` at :391 and `request_device` at :407 on every
   invocation. **Correction to rev 1:** this affects the *native*
   harness only. The browser path already holds a persistent device and
   queue in `State` (`src/pathtrace.rs:152–157`). Rev 1 claimed it
   blocked "any of the experiments, native or web"; that was wrong.
6. **Parked — the procedural sky.** See *Experiments*; sky recovery is
   cut from this plan.

## Related work

**The algorithm:**

- **Nimier-David, Speierer, Ruiz, Jakob 2020, "Radiative
  Backpropagation"** (SIGGRAPH). PRB's direct predecessor; rev 1
  omitted it, which is a visible hole when citing PRB.
- **Vicini, Speierer, Jakob 2021, "Path Replay Backpropagation"**
  (SIGGRAPH). Constant memory, linear time. The algorithm implemented
  here.
- **Zeltner, Speierer, Georgiev, Jakob 2021, "Monte Carlo Estimators
  for Differential Light Transport"** (SIGGRAPH). Enumerates attached
  vs. detached differential estimators and their variance
  consequences. Every adjoint written here silently picks a point in
  that design space; this paper is what lets the variance study
  explain its own numbers instead of merely reporting them.
- **Nimier-David, Vicini, Zeltner, Jakob 2022, "Unbiased Inverse Volume
  Rendering with Differential Trackers."** **Rev 1 mis-stated this.**
  The bias is in differentiating the **free-flight sampler**; the fix,
  *differential ratio tracking*, combines ratio tracking with
  **reservoir sampling** and samples distances proportional to
  unweighted transmittance. The paper also reports that optimizing
  toward dense volumes falls into local minima requiring bootstrapping
  from nonphysical emissive volumes — E2 must plan for this, not
  rediscover it.

**Variance and speed state-of-the-art (all absent from rev 1):**

- **Nicolet, Rousselle, Novák, Keller, Jakob, Müller 2023, "Recursive
  Control Variates for Inverse Rendering."** Up to 10× efficiency gain
  on PRB, on exactly the scene classes used here. **This forces a
  restatement of what the envelope measures:** the surface is a
  property of *plain PRB on WebGPU*, not of WebGPU. Stated that way in
  sub-claim (3) and in every figure caption.
- **Chang, Sivaram, Nowrouzezahrai, Hachisuka, Ramamoorthi, Li 2023,
  "Parameter-space ReSTIR for Differentiable and Inverse Rendering."**
  Reuses samples across optimization iterations — the published state
  of the art for the exact goal ("fast enough to be interactive") this
  plan claims.
- **Nicolet et al. 2021** (preconditioned parameterizations) — relevant
  to the optimizer choice below.
- **"A Survey on Physics-based Differentiable Rendering" (2025)**,
  `arxiv.org/abs/2504.01402`, plus 2025 PRB derivatives (radiance
  caching for differentiable path tracing, TOG 2025). PRB is a
  well-mined seam by 2026; "we implemented PRB" must be positioned
  against these.

**The browser/autodiff stack — baselines, not risks:**

- **Slang autodiff with a WGSL target** (Bangaru et al. 2023) — used
  here for local BSDF/phase derivatives, and evaluated as a full
  alternative (see *Baselines*).
- **Burn / CubeCL wgpu backend** and **Brush** — reverse-mode autodiff
  and in-browser *training* on WebGPU today. A reviewer will ask why
  this plan doesn't express the adjoint over Burn tensors; the answer
  must be measured, not asserted.
- **Web3D 2024 OpenPBR/WebGPU path tracer**, `10.1145/3665318.3677158`
  — the genre precedent for a WebGPU renderer implementation report,
  and venue intel.

**The gap, restated honestly:** there is no peer-reviewed
**physically-based** inverse rendering in a browser — splat training is
not light transport. But the *reason* it's unoccupied is not "no
autodiff on WGSL." It is that the scatter primitive is missing and
nobody has characterized the workaround.

## Experiments

Rev 1 had three; rev 2 has two, and the sky experiment is **cut**.

**E1 — Material + emission recovery (dim ~12).** Cornell Box, using
`tests/cornell_gltf.rs`'s existing scene. Recover per-wall albedo
(RGB × 3) plus light emission (RGB). **Requires prerequisite #3.**
This is the correctness experiment *and* the **worst-case contention
experiment for sub-claim (1)** — ~10⁶ samples scatter-adding into 12
words is the most brutal atomic contention in the whole plan.

**E2 — Volume density recovery (dim 32³ = 32 768).** Multi-view
recovery of a heterogeneous density field over the existing
delta/ratio-tracking path. Requires prerequisites #4 and differential
ratio tracking with reservoir sampling. This is the flagship: the
high-dimensional regime where gradients genuinely beat gradient-free
methods, and — note the inversion against E1 — the **low**-contention
scatter regime, since 32 768 accumulator words spread the atomics out.
E1 and E2 together bracket the contention axis, which is what makes
the scatter study a study rather than a single data point.

**Cut: sky recovery.** Rev 1 made it the flagship widget. It doesn't
survive. With the delta sun (plan 0023) enabled, moving `sun_dir` moves
every shadow boundary — a visibility discontinuity, explicitly out of
scope for this plan. With it disabled, the only signal is sky-dome
tint, and a closed-form Levenberg-Marquardt fit of the analytic
Hošek-Wilkie model to the sky pixels beats a differentiable renderer
outright, needing no transport simulation at all. Separately,
`turbidity` lerps between integer bins (`sky.rs:344–363`) so its
gradient is piecewise-constant with kinks at every integer and zero
outside `clamp(1.0, 10.0)` (`sky.rs:340`), and `ground_albedo` is a
two-endpoint lerp clamped to [0,1]. Sky-parameter recovery may still
make a **blog demo**; it is not a research claim and is removed from
this plan's milestones and done-when.

## Correctness testing (rev 1's gate was ill-posed)

Rev 1's go/no-go was "adjoint within 5% relative L2 of central finite
differences on ≥ 95% of parameters." That is broken in four ways, and
the fourth is fatal:

1. "Relative L2 per parameter" is a category error — the L2 norm of a
   scalar is its absolute value.
2. FD of a Monte Carlo estimator without common random numbers has
   variance O(σ/h); the gate is then satisfiable by raising spp, so it
   measures sample count, not correctness.
3. Relative error's denominator vanishes for near-zero gradients —
   exactly where failures concentrate — so a ≥ 95% threshold
   auto-passes by construction.
4. **With common random numbers it breaks at every discrete branch,
   and the volume case has one on the differentiated parameter
   itself:** `pathtrace.wgsl:1387` computes `let p_real = density;`
   then tests `if (next_1d(s) < p_real)`. The CRN finite difference in
   density is piecewise constant with jumps; its "derivative" is a sum
   of deltas. Nimier-David et al. 2022 demonstrate tracker bias by
   converged comparison and by optimization failure — **not** by an FD
   tolerance gate, for this reason.

Rev 2 replaces it with a three-part test:

- **Primary — Mitsuba 3 PRB parity.** Same scene, same parameters;
  gradients agree within sampling noise. A far stronger check than FD,
  and it validates the *algorithm*, not just the arithmetic.
- **Secondary — FD, restricted to smooth estimators.** Materials and
  emission only, with CRN, and only where no branch keys on the
  differentiated parameter. Explicitly *not* applied to density.
- **Volume — converged-reference + positive control.** Adjoint vs. a
  high-spp converged reference gradient, plus a demonstration that the
  *naive* (non-reservoir) tracker derivative **fails** the same test.
  A correctness test with no failing control is not a test.

## Baselines

- **Slang → WGSL autodiff, end-to-end.** *Strongest baseline.* If
  Slang can express the PRB adjoint pass and produce a working kernel
  in a week, the hand-derived recipe has no reason to exist. The plan
  must run this, not assume the answer. Expected outcome: Slang handles
  local derivatives well and does not give the path-level replay
  structure or the scatter — but that is a hypothesis, and
  `[R0003/slang-eval]` tests it before the hand-derivation work starts.
- **Burn / CubeCL adjoint over tensors.** The "why not just use the
  Rust autodiff stack that already runs in browsers" answer, measured.
- **Adam, not plain gradient descent**, for every optimization result;
  plus a preconditioned variant (Nicolet 2021) where applicable.
  Reporting plain-GD convergence would be rejected on sight.
- **Gradient-free at matched wall-clock:** CMA-ES at E1 (dim 12).
  **Not full-covariance CMA-ES at E2** — at n = 32 768 that is a 10⁹
  -entry covariance, so "E2 beats CMA-ES" would be satisfied by the
  baseline being inapplicable. E2 uses **sep-CMA / diagonal CMA or
  SPSA** instead, and says so.
- **Mitsuba 3 PRB on the LLVM/CPU backend**, alongside CUDA. The
  CUDA-on-NVIDIA vs WebGPU-on-M-series comparison confounds API,
  hardware and implementation; an equal-hardware point is needed for
  "how much of the gap is WebGPU?" to mean anything.
- **Recursive control variates (Nicolet 2023)** as the variance
  reference, so the envelope is reported against published practice
  rather than against plain PRB alone.

## Milestones

- [/] **[R0003/primal-timing]** *Milestone zero — decides sub-claim (3)
  before anything is built.* **Native half shipped 2026-09-15**; see
  *Findings* for the table. Result: the interactive region is roughly
  an order of magnitude smaller than rev 1 assumed, and the renderer is
  **submit-bound rather than shading-bound** at widget resolutions,
  which is a lever (batch samples per submit) and an argument for the
  compute port. **Remaining:** the same sweep in Chrome and Safari via
  the wasm build — native is only a lower bound on browser cost, and
  the Safari number is the one that decides whether the widget target
  is Chrome-only.
- [ ] **[R0003/slang-eval]** Evaluate Slang's WGSL autodiff on a
  single BSDF and on a toy transport loop. Decides whether local
  derivatives are hand-written or generated, and whether the whole
  hand-derivation premise survives. **Gates the derivation milestones.**
- [ ] **[R0003/scatter-study]** *The contribution.* Implement and
  characterize three gradient-scatter schemes on WebGPU: fixed-point
  u32 with adaptive exponent; `atomicCompareExchangeWeak` CAS loop;
  workgroup-reduction-then-single-atomic. Measure throughput vs.
  contention (parameter count from 1 to 10⁵ at fixed sample count) and
  precision vs. an f64 CPU reference. Sub-claim (1)'s gate.
- [ ] **[R0003/compute-port]** Port the megakernel integrator from
  `@fragment` to `@compute` with `read_write` bindings. Done when every
  existing reference render is **bit-identical** to its committed
  output (the repo's `bake_equirect_noon_is_deterministic` tripwire is
  the pattern to copy), and the fragment path is deleted rather than
  left to rot in parallel.
- [ ] **[R0003/replay-fidelity]** Sub-claim (2)'s gate: primal and PRB
  replay produce bit-identical path sequences over ≥ 10⁶ paths,
  asserted via a path-hash buffer.
- [ ] **[R0003/metallic-continuous]** Continuous metallic-blend BSDF
  replacing the `metallic > 0.5` branch, plus the `roughness < 0.2`
  mirror-collapse discontinuity. Includes a reference-comparison pass
  over the existing gallery, since this changes shipped output.
- [ ] **[R0003/device-reuse]** Persistent render context for the
  **native** offscreen harness (prerequisite #5).
- [ ] **[R0003/adjoint-material]** Material + emission adjoints, by
  whichever route `slang-eval` selected.
- [ ] **[R0003/mitsuba-parity]** Gradient parity against Mitsuba 3 PRB
  on E1's scene. Primary correctness gate.
- [ ] **[R0003/variance-study]** Gradient variance vs spp {1..64},
  interpreted against Zeltner 2021's attached/detached taxonomy, and
  reported against Nicolet 2023 as the efficiency reference.
- [ ] **[R0003/e1-materials]** Cornell recovery with Adam; CMA-ES at
  matched wall-clock. Records where gradients do and do not earn their
  keep at dim 12 — **a negative result here is expected and fine.**
- [ ] **[R0003/f32-density]** f32 density buffer + per-iteration
  buffer→texture copy, or hand-rolled trilinear (prerequisite #4).
- [ ] **[R0003/differential-tracker]** Differential ratio tracking with
  reservoir sampling per Nimier-David 2022. Done when the volume
  gradient passes the converged-reference test **and** the naive
  tracker is shown to fail it.
- [ ] **[R0003/e2-volume]** 32³ multi-view density recovery, with the
  emissive-bootstrap provision the 2022 paper's local-minima finding
  requires. Baseline: sep-CMA or SPSA.
- [ ] **[R0003/browser-envelope]** The envelope surface, Chrome and
  Safari, M-series — captioned as *plain PRB on WebGPU*. Must report
  Safari and Chrome separately; Brush documents Chrome-only support for
  in-browser splat training, so Safari viability is an open question,
  not an assumption.
- [ ] **[R0003/widget]** The live artifact. **Not** a small delta on
  plan 0035's debounce pattern — `src/pathtrace/web.rs` is a
  `requestAnimationFrame` loop (`Inner::tick`, :105) rendering one
  sample per frame to a canvas with a `SAMPLE_BUDGET` cap. It has no
  loss computation, no GPU→CPU readback, no optimizer state, no
  target-image ingest, no gradient buffers. A CPU-side optimizer also
  needs a `mapAsync` round-trip per iteration (≥ 1 frame of latency on
  top of the step budget) unless Adam is written in WGSL too. Size it
  accordingly.

## Done when

**Positive result** (→ `writing`):

- Sub-claim (1): the scatter study produces a scheme meeting its stated
  precision and contention targets, characterized across the full
  contention range that E1 and E2 bracket.
- Sub-claim (2): replay fidelity green at ≥ 10⁶ paths.
- Sub-claim (3): the interactive region of the envelope **contains E2's
  configuration** (32 768 parameters at the resolution and spp E2
  actually converges with). *Rev 1 said "some non-empty region," which
  a 1-parameter 64×64 1-spp point would satisfy.*
- Mitsuba parity green; E2 beats sep-CMA/SPSA decisively at matched
  wall-clock.

**Graceful abandonment** (→ `abandoned`, findings retained):

- `slang-eval` shows Slang generates the whole adjoint pass including a
  workable scatter → the engineering contribution is gone; write the
  envelope up as a blog post and abandon the paper.
- `primal-timing` shows 2× primal already busts the interactive budget
  → no browser story; either re-pose as native-only or stop.
- The scatter study finds no scheme better than CAS → record the
  measurements; this is a genuine negative result about WebGPU as a
  compute target and is worth a blog post, **but not a Web3D paper.**
- Replay fidelity cannot be achieved → PRB is not viable here; say so
  and stop.

**Explicitly not a paper:** "we measured that browser inverse rendering
is N× from interactive." Rev 1 claimed that was publishable. It is a
blog post. JCGT's scope is battle-tested techniques with analyzed
characteristics and usable code; a bare negative envelope measurement
does not clear it, and JCGT publishes PDFs and code — a live widget
carries no weight there.

## Findings

- **2026-09-15** — `[R0003/primal-timing]`, native half. Release build,
  M-series, default Cornell scene (`default_triangle_scene`), timing
  from the `render` subcommand's own instrumentation
  (`src/main.rs:751–760`), which covers the render loop only and
  **excludes** device creation. Median of 2–3 runs, first run per
  configuration discarded as shader-compile warmup.

  | Config | primal | PRB @ ≥2× |
  |---|---|---|
  | 128² @ 4 spp | ~45 ms | ~90 ms |
  | 128² @ 16 spp | ~80 ms | ~160 ms |
  | 128² @ 64 spp | ~220 ms | ~440 ms |
  | 256² @ 16 spp | ~100–170 ms | ~200–340 ms |
  | 256² @ 256 spp | ~615 ms | ~1.2 s |
  | 256² @ 1024 spp | ~1.54 s | ~3.1 s |

  Three consequences. **(a)** Rev 1's interactive cap (256², 16 spp,
  100 ms/step) is busted by the *primal* alone; the viable native
  region is ≤ 128² at ≤ 4 spp, and browser cost is strictly worse, so
  sub-claim (3) was restated. **(b)** Fitting the sweeps gives ~1.8 ms
  marginal per spp against ~50 ms (128²) / ~80 ms (256²) fixed
  overhead — and the marginal cost barely moves with a 4× pixel-count
  increase, so at widget resolutions the renderer is **submit-bound,
  not shading-bound.** The offscreen path submits once per sample
  (`src/pathtrace/offscreen.rs:923`, inside the `for frame in
  0..cfg.samples` loop). Batching samples per submit is untested
  headroom, and a compute port would naturally capture it — this
  strengthens the case for `[R0003/compute-port]` on grounds
  independent of gradient scatter. **(c)** Cold start (first process,
  shader compilation) is ~1.5 s; irrelevant to steady-state
  optimization but it dominates any single-shot measurement, and every
  later timing milestone must discard a warmup run.

- **2026-09-15** — Rev 1 attacked by `research-critic`; 4×P0, 9×P1.
  Two P0s were fatal to the framing and are now the plan's subject:
  the novelty claim ("no autodiff on WGSL") was falsified by Slang and
  Brush, and the integrator is a `@fragment` shader with read-only
  storage bindings that cannot scatter gradients without a full
  compute port. Two further P0s — `metallic` having an identically-zero
  gradient, and the FD correctness gate being ill-posed (worst case:
  the delta-tracking accept test thresholds on the very parameter being
  differentiated, `pathtrace.wgsl:1387`) — were fixed by adding a
  prerequisite and replacing the gate with Mitsuba parity plus a
  converged-reference test with a failing control. Sky recovery cut
  from the plan. All in-repo claims independently verified before
  acceptance; rev 1's claim that per-call device creation blocked the
  web path was **wrong** (`State` at `src/pathtrace.rs:152` already
  holds a persistent device) and is corrected.
