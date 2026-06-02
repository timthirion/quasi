# Foundation: interactive Cornell Box path tracer (native + web)

- **Status:** active
- **Last updated:** 2026-06-02
- **Last touched on:** planning (no code yet)

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

### M1 — Cornell Box path tracer ✅ code complete (visual check pending)
- [x] Scene structs (quad, material) + Cornell Box factory (`scene.rs`), with a
      GPU-packed `Uniforms` matching WGSL alignment (vec3 on 16-byte boundaries).
- [x] WGSL path tracer (`shaders/pathtrace.wgsl`): ray gen, quad intersection,
      Lambertian, NEE+MIS, PCG. Faithful port of the verified reference integrator.
- [x] Progressive accumulation: 3-pass pipeline (pathtrace → accumulate ping-pong
      HDR → present), `textureLoad` keeps passes pixel-aligned (no sampler/flips).
- [x] Orbit camera (drag/zoom), accumulation resets on movement.
- [x] WGSL validated headlessly via `naga` in `cargo test` (`tests/shaders.rs`).
- **Done when:** a recognizable, converging Cornell Box renders natively (and web).
  _Builds native + wasm; clippy/fmt clean; shader tests pass. Visual confirmation
  pending — run `cargo run` / the web steps._

**WGSL note:** `from` and `target` are reserved keywords in WGSL — the naga test
caught both at `cargo test` time (no GPU needed). Worth keeping that test as the
first line of defense for shader changes.

### M2 — Samplers, AOVs, output
- [ ] Selectable samplers: PCG / Halton / Sobol; sampler test (sequences off-GPU).
- [ ] AOVs: albedo / normal / depth via MRT, each accumulated.
- [ ] Native image output: PNG (tonemapped) + EXR (HDR + AOVs).
- **Done when:** AOVs and an HDR EXR can be written from a native render.

### M3 — Verification harness (native)
- [ ] Image metrics (MSE / RMSE / rel-MSE) + `cargo test` on synthetic images and
      an EXR round-trip.
- [ ] A convergence runner: render error-vs-spp for each sampler/integrator
      against a high-spp reference; emit CSV.
- **Done when:** the CSV shows the path tracer converging; MIS beats pure BSDF at
      equal spp; metrics tests pass.

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
