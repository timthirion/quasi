# Foundation: interactive Cornell Box path tracer (native + web)

- **Status:** active
- **Last updated:** 2026-06-04
- **Last touched on:** M3 landed — verification harness (metrics + convergence CSV)

## Goal

Take the renderer from an empty repo to a complete, physically-based Cornell Box
path tracer that runs both as a native desktop app and as an interactive
in-browser widget, with the measurement tooling to back up quality claims. This is
the foundation every later post builds on: by the end we can embed a live,
progressively-refining render in a blog and show converging error against a
reference.

Built as ordered milestones (M0–M4). Each is independently shippable and may be
split into its own `plans/NNNN-*.md` as it starts; this document is the spine that
ties them together.

## Context

Empty Rust repository (`cargo`, web-ready `.gitignore`). No code yet. Key
decisions, made in `AGENTS.md`: `wgpu` + WGSL (native + WebGPU), `winit`
windowing, `wasm-pack` for the web build. One core renderer crate driven by thin
native and web entry points.

The single most important architectural commitment: **native and web stay in
lockstep.** Every milestone is "done" only when it works in both targets (except
the verification harness, which is native-only by nature).

## Design

### Module shape
- `core` — owns the `wgpu` `Device`/`Queue`/`Surface`, the scene, the WGSL
  pipelines, and the frame loop. Platform-agnostic. No `winit`, no canvas.
- `bin` native entry — `winit` window + `pollster` to init `wgpu`, feeds events
  (orbit camera, key toggles) into `core`.
- web entry (`lib`, `cdylib`) — `wasm-bindgen` exports that attach to a canvas and
  drive `core`; input via canvas events.
- WGSL shaders in `.wgsl` files, included with `include_str!`.

### Rendering pipeline (mirrors the proven megakernel design)
1. **Path-trace pass** — fullscreen fragment shader writes one new sample/frame to
   an HDR texture (`Rgba16Float`; revisit if precision needs `Rgba32Float`).
2. **Accumulate** — progressive average into a ping-pong HDR texture
   (`weight = 1/(frame+1)`); reset when the camera moves.
3. **Tonemap pass** — Reinhard + gamma to the surface/canvas.

Compute vs. fragment for accumulation: start with a fragment/blit average to keep
WebGPU/WebGL2 portability simple; move to a compute kernel if needed.

### Integrator
Implement **NEE + MIS from the start** — direct area-light sampling combined with
BSDF sampling via the power heuristic. (Pure BSDF path tracing is kept only as a
selectable comparison mode for the convergence study, not the default.) Lambertian
diffuse first; the `material` carries roughness/metallic for later BSDF work.

### Scene
Analytic Cornell Box: 5 walls (white/red/green) + ceiling area light + two boxes,
all as quads with per-quad materials. A `look_at` orbit camera.

## Milestones

### M0 — Foundation: pixels, native + web ✅ DONE
- [x] Cargo project; single package as both native bin and wasm `cdylib` + `rlib`.
      (Kept as one crate for M0 rather than a separate `core` crate; split out when
      the renderer grows.)
- [x] `wgpu` init (adapter/device/queue/surface) — builds and links natively.
- [x] Fullscreen-triangle pass drawing a gradient (`src/shader.wgsl`).
- [x] `wasm-pack build --target web` succeeds; `index.html` attaches winit's
      canvas to `#quasi-canvas` and runs the same render.
- **Done when:** the gradient shows in both a desktop window and a browser tab.
  _Confirmed: gradient renders natively and in the browser (WebGPU). On the web,
  use winit's `EventLoopExtWebSys::spawn` rather than `run` so it doesn't unwind
  via the "control flow" exception._

**wgpu version note:** pin **current** wgpu (29), not an older release. An initial
0.20 pin compiled but failed at `requestDevice` in a 2025 browser, which rejects
the long-removed `maxInterStageShaderComponents` limit that old wgpu still sends.
Lesson for a browser-targeting renderer: track current wgpu. (winit kept at 0.29;
compatible via raw-window-handle 0.6.)

**Web architecture (post-M1):** the web path no longer uses winit. winit allows
only one event loop per process, which blocks **multiple instances on one page**.
Instead `State` is platform-agnostic and the web is driven per-instance by
`requestAnimationFrame` (`web` module: `QuasiInstance` + `create(host_id)`). Each
instance owns its canvas, rAF loop, pointer/wheel listeners, and a `ResizeObserver`
that sizes the canvas to the host element's `clientWidth/Height × devicePixelRatio`.
Native still uses a single winit window/event loop. This is the groundwork for M4.

**Web performance (multi-instance):** with N widgets the dominant cost is each
one path-tracing every rAF tick forever. Two pauses fix it: (1) **converged/idle
pause** — stop sampling once `frame_count` reaches `SAMPLE_BUDGET` (1024); camera
interaction/resize resets it; (2) **off-screen pause** — an `IntersectionObserver`
skips rendering while the host isn't in the viewport. The rAF loop keeps ticking
(cheap) but does no GPU work when idle or hidden. Observer-touched state lives in
`Cell`s (a `Shared` struct) separate from the `RefCell<Inner>` render borrow, so
a callback can never collide with a render. Deferred: sharing one GPU device
across instances (memory/startup win, but a shared device-loss failure domain).

### M1 — Cornell Box path tracer ✅ DONE (visually confirmed, native + web)
- [x] Scene structs (quad, material) + Cornell Box factory (`scene.rs`), with a
      GPU-packed `Uniforms` matching WGSL alignment (vec3 on 16-byte boundaries).
- [x] WGSL path tracer (`shaders/pathtrace.wgsl`): ray gen, quad intersection,
      Lambertian, NEE+MIS, PCG. Faithful port of the verified reference integrator.
- [x] Progressive accumulation: 3-pass pipeline (pathtrace → accumulate ping-pong
      HDR → present), `textureLoad` keeps passes pixel-aligned (no sampler/flips).
- [x] Orbit camera (drag/zoom), accumulation resets on movement.
- [x] WGSL validated headlessly via `naga` in `cargo test` (`tests/shaders.rs`).
- **Done when:** a recognizable, converging Cornell Box renders natively (and web).
  _Confirmed 2026-06-02: renders correctly in both targets, converges on hold,
  resets on camera move._

**WGSL note:** `from` and `target` are reserved keywords in WGSL — the naga test
caught both at `cargo test` time (no GPU needed). Worth keeping that test as the
first line of defense for shader changes.

**Uniform layout gotcha:** a `vec3<u32>` (or `vec3<f32>`) in a WGSL uniform forces
16-byte alignment, so `struct { u32, vec3<u32> }` is **32 bytes**, not 16 — which
silently mismatched the packed 16-byte Rust struct and failed only at runtime
("buffer too small"). The naga test can't catch this (it's a cross-language size
mismatch, not a WGSL error). Rule: keep uniform structs to scalar fields or full
16-byte quartets, and make the Rust and WGSL sizes obviously identical.

### M2 — Samplers, AOVs, output ✅ DONE
- [x] Selectable samplers: PCG / Halton / Sobol (`pathtrace::sampler`).
      All three dispatched at runtime by a `sampler_kind` uniform; the
      WGSL `next_2d(s)` branches on the kind. CPU mirrors exist for every
      sampler so canonical sequence values (van der Corput, Halton bases
      2/3, Sobol dims 0/1) are pinned in `cargo test`.
- [x] AOVs: radiance / albedo / normal / depth via 4-attachment MRT, each
      accumulated through the ping-pong pass. Accumulate bind group has
      9 entries (1 uniform + 8 textures). `max_color_attachments=4` on
      `downlevel_webgl2_defaults` is exactly enough; preserves web build.
- [x] Native image output (`pathtrace::output`): PNG (Reinhard + gamma
      1/2.2, matches the present shader) and multi-channel HDR EXR
      carrying RGB radiance plus `albedo.{R,G,B}`, `N.{X,Y,Z}`, `Z`
      (depth) in one layer. Encoders run on **scoped threads**
      (`std::thread::scope`) so PNG and EXR overlap — first exercise of
      AGENTS.md's "Use the language" guidance.
- [x] CLI: `cargo run -- render --out <base> [--width W --height H --spp N
      --sampler pcg|halton|sobol]` for headless renders. Offscreen path
      uses its own `Rgba32Float` targets (clean `bytemuck::cast_slice`
      readback, no f16 decode) and asks for `adapter.limits()` to clear
      the default `max_color_attachment_bytes_per_sample=32` cap.
- **Done when:** AOVs and an HDR EXR can be written from a native render.
  _Confirmed 2026-06-04: `cargo run -- render --width 256 --spp 128
  --sampler {pcg,halton,sobol}` produces a recognisable Cornell Box PNG
  and a 10-channel EXR; the EXR round-trips back through `exr::read_all_*`
  in the test suite._

**MRT format gotcha:** wgpu validates `BlendState::REPLACE` against the
texture format's blendability, even though "replace" is semantically
no-blend. `Rgba32Float` is non-blendable, so the unified `make_pipeline`
helper now passes `blend: None`. Switching all pipelines to `None` was
the right move — neither pipeline ever uses blending.

**Limit gotcha:** four `Rgba32Float` attachments add up to 64 bytes per
sample, which exceeds the WebGPU baseline of 32. The native offscreen
device requests `adapter.limits()` (always ≥ 64 on the backends we
target). The windowed renderer stays on `Rgba16Float` (32 bytes total)
and keeps `Limits::downlevel_webgl2_defaults()` on web.

**"Use the language" pattern:** image encoding is CPU-bound and trivially
splittable per format. `std::thread::scope` borrows `&Aovs` across the
PNG and EXR closures without an `Arc`, and the scope guarantees the
borrow doesn't outlive `write_render`. Clean fit; no `async` theatre.

### M3 — Verification harness (native) ✅ DONE
- [x] Image metrics (`pathtrace::metrics`): `mse_rgb`, `rmse_rgb`,
      `rel_mse_rgb` over the RGB channels of `[f32; 4]` AOV slices,
      accumulated in `f64`. Tests cover identical-images = 0, constant
      offset = exact MSE, black-reference rel-MSE, and a size-mismatch
      panic message. EXR round-trip is already pinned in M2 by
      `pathtrace::output::tests::exr_round_trip_preserves_values`.
- [x] Selectable integrator (`pathtrace::integrator::IntegratorKind`):
      `MisNee` (the M1/M2 default) and `Bsdf` (pure BSDF, no NEE, no MIS
      weighting). One uniform `integrator_kind` toggles between them
      inside the same WGSL shader.
- [x] Convergence runner (`pathtrace::converge`): renders a reference at
      `reference_spp` with PCG + MIS+NEE (the lowest-variance pair) then
      sweeps each (sampler, integrator) combination across doubling spp
      checkpoints, scoring each with RMSE + rel-MSE against the
      reference. CSV columns: `sampler,integrator,spp,rmse,rel_mse`.
- [x] CLI: `cargo run -- converge --out runs.csv [--width W --height H
      --max-spp N --reference-spp N]`. Render command also grew an
      `--integrator misnee|bsdf` flag for one-off pure-BSDF outputs.
- **Done when:** the CSV shows the path tracer converging; MIS beats
      pure BSDF at equal spp; metrics tests pass. _Confirmed 2026-06-04
      with `converge --width 64 --height 64 --max-spp 64 --reference-spp 256`:_
      - **Converges:** PCG and Halton hit the canonical √N rate — RMSE
        drops 7.1×–7.9× over 64× more samples (theoretical: 8×). Sobol
        converges, but slowly (see note below).
      - **MIS beats BSDF:** at 64 spp, MIS+NEE has 3–5× lower RMSE than
        pure BSDF for every sampler. PCG: 0.025 vs 0.124; Halton: 0.026
        vs 0.102; Sobol: 0.115 vs 0.354.
      - **Metrics tests pass:** `cargo test pathtrace::metrics` is green.

**Sobol caveat:** the M2 implementation provides direction vectors for
**two** dimensions, but the path tracer draws ~10–14 2D points per path
(camera jitter + light + BSDF × MAX_BOUNCES). Consecutive `next_2d`
calls within one path therefore read **consecutive** Sobol points,
which are correlated by design — Sobol's variance reduction depends on
draws coming from **distinct** dimensions. Result: Sobol converges
~2.2×/64×spp instead of the expected 8×, and trails PCG and Halton in
the CSV. The fix is **padded high-dim Sobol** (Joe-Kuo direction tables
for 16+ dimensions, advancing dimension per call within a path,
advancing index per path). Future work; the conventional canonical-
sequence tests (Sobol dim 0 ≡ van der Corput, 2-D mean → 0.5) still
pass, so the implementation is correct as far as it goes.

**rel-MSE ε note:** the relative-MSE denominator carries an ε of `1e-2`
to keep the metric defined for black reference pixels. Test
`rel_mse_grows_with_error_at_fixed_reference` pins the ε-stable region
(doubling per-pixel error ~ quadruples rel-MSE). Scale-invariance is
**not** an exact property of this form — at uniform 10× brighter
intensities the ε contribution shrinks, perturbing rel-MSE by a few
percent. That's the standard PBRT trade-off; documented here so a
future scale-invariance test isn't written against the wrong claim.

### M4 — Interactive blog demo
- [ ] `wasm-pack` package + embeddable HTML/JS harness.
- [ ] Live controls: orbit camera, sample-count readout, sampler + integrator
      toggles, progressive refinement that resets on interaction.
- **Done when:** the widget runs smoothly in a browser and is drop-in embeddable
      in a post.

## Open questions

- **HDR texture format on web:** `Rgba16Float` is broadly supported and filterable;
  confirm it's enough precision for accumulation, else use `Rgba32Float` (storage
  support varies). Decide during M1.
- **WebGPU vs. WebGL2 reach:** target WebGPU first (now widely available); decide
  whether a WebGL2 fallback is worth it based on the blog audience.
- **Compute or fragment accumulation:** start fragment for portability; measure and
  revisit (M1/M2).
- **EXR crate:** confirm the `exr` crate covers multi-channel AOV writing the way
  we want, or write a thin layer.

## Done when

- The Cornell Box path tracer (NEE+MIS, selectable samplers) renders natively and
  in the browser.
- The native harness demonstrates convergence with metrics + a CSV, and metrics
  tests pass under `cargo test`.
- An interactive, progressively-refining demo is embeddable in a blog post.
