# Real-time rasterization track

- **Status:** active
- **Last updated:** 2026-06-03
- **Last touched on:** kicked off on this machine; driven by motum's Phase 4
  (interactive in-browser planner demos).

## Goal

Grow Quasi from a single-pipeline path tracer into a **dual-pipeline
renderer** that supports both path-traced offline-quality stills and a
real-time rasterized pipeline suitable for 60fps interactive scenes. The
two pipelines stay **wholly separate** at the renderer layer — different
scenes, different shaders, different draw paths — and share only the
platform plumbing (wgpu device / queue / surface, frame loop, canvas
attachment).

The driving consumer is `motum`'s Phase 4: a browser widget renders a
robot, the user drags a goal, RRT-Connect runs in wasm, the planner's
search tree + the resulting animated trajectory render live. None of that
is realistic over a path tracer; it needs rasterization. The path tracer
keeps its own track for high-quality stills and the "watch convergence"
blog story.

## Context

Today (post-M1 of `0001-foundation`) Quasi is a Cornell-Box-only path
tracer. The renderer:

- Owns one `State` (`src/lib.rs`) that conflates wgpu device/queue/surface
  setup with the three-pass path-tracing pipeline (pathtrace → accumulate
  → present) and HDR ping-pong targets.
- Has a `web` module per-instance driver (rAF + `ResizeObserver` +
  `IntersectionObserver`) that's pipeline-agnostic by construction.
- Has shaders in `src/shaders/{pathtrace,accumulate,present}.wgsl`.
- Validates shaders headlessly via `naga` in `tests/shaders.rs`.

`scene.rs` is path-tracer-specific (quads + emissive light), with a fixed
`MAX_QUADS = 32` and a packed `Uniforms` matching the WGSL layout.

`0001-foundation` continues with M2–M4 (samplers, AOVs, verification,
blog widget) on the path-tracer track. This plan runs **in parallel** —
it doesn't block on the remaining 0001 milestones and they don't block
on it.

## Design

### Top-level module split

```
src/
├── lib.rs            // thin re-exports + crate-level docs
├── main.rs           // native entry
├── gpu/              // shared platform plumbing
│   ├── mod.rs        // wgpu instance/device/queue, surface config
│   ├── camera.rs     // OrbitCamera (currently in lib.rs)
│   └── web.rs        // browser per-instance driver (currently `mod web`)
├── pathtrace/        // existing renderer
│   ├── mod.rs        // path-tracer State + pipelines
│   ├── scene.rs      // current scene.rs
│   └── shaders/{pathtrace,accumulate,present}.wgsl
└── raster/           // new renderer
    ├── mod.rs        // raster State + pipelines
    ├── scene.rs      // mesh / instance / camera scene description
    ├── mesh.rs       // Mesh primitive + procedural helpers
    └── shaders/forward.wgsl
```

The `gpu` module is the only shared surface; **`pathtrace` and `raster`
never depend on each other**. Each owns its own `State`, its own bind
group layouts, and its own surface presentation path.

### Scene representations

The two pipelines use intentionally different scene shapes — the
abstractions a path tracer wants (light sources, BSDFs, area emitters)
are not the abstractions a rasterizer wants (instance lists, vertex
buffers, materials). No unified `Scene` trait; just two clear types.

**Path tracer (today):** `pathtrace::Scene` = quads + materials +
`light_index` (kept as-is).

**Rasterizer (new):**
```rust
pub struct RasterScene {
    pub meshes: Vec<Mesh>,        // geometry library
    pub instances: Vec<Instance>, // (mesh handle, transform, material)
    pub camera: Camera,
    pub lights: Vec<DirectionalLight>,
}
```

The rasterizer's wire format will eventually mirror motum's `WorldState`
(per-link pose + geometry handle), but R0 just defines the in-process
type. JSON ingestion lands in R4.

### Pipeline selection

Native binary takes a `--pipeline {pathtrace|raster}` flag (default:
`pathtrace`, the existing behavior). Web exports two independent
instance entry points: `PathTraceInstance::create(host_id)` (existing)
and `RasterInstance::create(host_id)` (new). Each browser widget picks
one renderer; nothing tries to be both at once.

### Camera

`OrbitCamera` is moved to `gpu::camera` and used unchanged by both
pipelines. Mouse drag / wheel zoom semantics stay the same.

## Milestones

This plan is staged like `0001-foundation`: each milestone is independently
shippable.

### R0 — Module split & shared GPU plumbing ✅ DONE
Refactor without behavior change. Path tracer keeps working native + web;
empty `raster` module compiles. Existing renderer now lives at
`quasi::pathtrace::*` and the shared bits at `quasi::gpu::*`.

- [x] Extract `OrbitCamera` to `gpu::camera`.
- [x] Extract the wgpu `Instance` factory (`make_instance`) to `gpu`.
      Device / adapter / queue creation stays inside the per-pipeline
      `State::new` since it's bound up with surface configuration; we'll
      revisit if both pipelines end up requesting them identically.
- [x] Move current `State`, `Targets`, accumulate / present passes, and
      `scene.rs` into `pathtrace`.
- [ ] ~~Move `mod web` into `gpu::web` (kept pipeline-agnostic by injecting
      the renderer)~~ — **deferred to R1.** The web driver touches a few
      path-tracer-specific bits (`camera.dirty`, `frame_count`,
      `SAMPLE_BUDGET`); the right abstraction emerges once we have a
      second consumer (a `RasterInstance`) to inform whether to generify
      via a `Renderer` trait or to keep two clean side-by-side drivers.
- [x] Empty `raster` module compiles and is wired into `lib.rs` re-exports.
- [x] Native `cargo build` + `cargo test` green; `cargo check` for
      `wasm32-unknown-unknown` green; clippy `-D warnings` clean; fmt
      clean. The Cornell Box widget on `index.html` keeps working
      unchanged because the wasm-bindgen `create()` entry is preserved.

**Done when:** all of M0 + M1's existing functionality keeps working
with no visible change; the new module boundaries are in place. ✓

### R1 — Forward triangle pipeline ✅ DONE
First raster pixel-on-screen. A single triangle mesh rendered with simple
forward shading native + web.

- [x] `Mesh` (positions, normals, colors, indices) + procedural
      `cube_mesh` (24 vertices, 36 indices, hard per-face normals).
- [x] `raster::State` with a TriangleList pipeline (vertex + fragment),
      back-face culling, `Depth32Float` depth attachment, and surface
      presentation. Camera math (look-at + perspective + multiply) is
      hand-rolled in-module.
- [x] `forward.wgsl`: lambert from a fixed directional sun + flat
      ambient, modulated by vertex color, gamma-encoded for the non-sRGB
      swapchain.
- [x] Naga validation test in `tests/shaders.rs`.
- [x] Native binary `--pipeline raster` flag (also `raster` bare) draws
      a single cube on a colored background.
- [x] CPU↔GPU struct layout assertion: `FrameUniforms` size = 112 bytes.
- [x] Web `RasterInstance::create_raster(host_id)` mirrors
      `QuasiInstance::create` but renders every frame (no convergence /
      sample budget); lives in `raster::web`. Required moving the
      existing path-tracer web driver out of `lib.rs` into
      `pathtrace::web` for symmetry; `lib.rs` now just holds the
      single `#[wasm_bindgen(start)]` shim.

**Done when:** a shaded cube renders native + web; both pass headless
validation. ✓ — native window works; wasm-bindgen exports compile for
`wasm32-unknown-unknown`. Browser visual confirmation will land with
the first motum demo widget (it'll be the first thing using
`create_raster`).

### R2 — Instanced scene
Many meshes, one draw path per geometry. Foundation for rendering whole
robots (skeletal sphere/cylinder primitives) or thousands of planner-tree
nodes.

- [ ] Geometry library inside `RasterScene`: register a mesh, get a
      `MeshHandle`.
- [ ] Per-instance buffer (transform matrix + material).
- [ ] Indexed indirect or instanced draw per `MeshHandle`.
- [ ] Procedural primitives: sphere, capsule, cylinder, box.
- [ ] Scene assembly API + tests for transform math.

**Done when:** a small assembled scene (ground plane + a few primitive
shapes at hand-chosen poses) renders correctly.

### R3 — Overlays for planner artifacts
Lines and points for visualizing planner search trees, end-effector
traces, and goal markers.

- [ ] Line primitive (instanced thin quads or `LineList` topology —
      decide during R3).
- [ ] Point sprite primitive.
- [ ] Throughput target: 10 000 lines @ 60fps in browser.
- [ ] Tests covering the line/point uniforms and a basic render.

**Done when:** an `Overlay` field on `RasterScene` carries thousands of
edges and points, rendered above (or with depth-tested against) the
instance scene.

### R4 — Motum-shaped scene API for embedders
Make the rasterizer drivable from a wasm host with motum's data types as
input. This is the seam motum's Phase 4 plugs into.

- [ ] JSON wire format compatible with motum's `WorldState`,
      `Trajectory`, `PlannerTree` (motum already serializes these).
- [ ] `RasterInstance` wasm-bindgen API:
      - `set_world_state(json)` — instance list + per-instance poses.
      - `set_trajectory(json)` + playhead controls.
      - `set_planner_tree(json)` — overlay.
      - `on_goal_changed(callback)`.
- [ ] Goal handle: a pickable, draggable marker entity with mouse → world
      ray casting against a manipulation plane.
- [ ] A minimal HTML harness in `index.html` that loads a hand-coded
      motum-shaped JSON and renders it.

**Done when:** a static motum scene round-trips through JSON into the
rasterizer and renders correctly in the browser; mouse drag of the goal
fires the callback with the new pose.

## Open questions

- **Where does PBR live?** Forward + simple lambert is enough for R1–R3.
  GGX / dielectrics belong on the path-tracer track (per 0001's later
  phases). If the raster pipeline wants PBR, it's a follow-up plan.
- **Depth-tested overlays vs. on-top?** Both have uses (planner tree
  occluded by robot vs. always visible). Decide during R3; likely
  expose as a per-overlay-batch flag.
- **One device across multiple browser instances?** Per `0001`'s note,
  shared-device is a future memory win but adds a shared failure
  domain. Defer to a separate plan once we have two widget types
  side-by-side.

## Done when

- Path tracer and rasterizer both render correctly native + web, with
  fully separate scene types, shaders, and pipelines, sharing only the
  `gpu` module.
- The rasterizer ingests motum's serialized scene + trajectory +
  planner-tree types and renders them in the browser.
- A goal-drag interaction round-trips from the renderer back to motum
  (via a wasm callback) cleanly enough to drive a planner demo widget.
