# Real-time rasterization track

- **Status:** done — R0–R4 shipped 2026-06-04. The plan stays as a
  record; new raster work picks up under a new `plans/000N-*.md` if it
  ever happens.
- **Last updated:** 2026-06-04
- **Last touched on:** R4 landed — motum-shaped JSON scene API + draggable goal handle

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

### R2 — Instanced scene ✅ DONE
Many meshes, one draw path per geometry. Foundation for rendering whole
robots and thousands of planner-tree nodes.

- [x] Geometry library inside `raster::State`: register a mesh, get a
      `MeshHandle`; three default handles (cube, sphere, cylinder) seeded
      at construction.
- [x] Per-instance vertex buffer (`InstanceRaw`: 4 mat4 columns + tint
      rgba, 80-byte stride, instance-stepped at locations 3..=7).
- [x] Bucket-by-mesh upload + per-mesh instanced `draw_indexed`; instance
      buffer auto-grows when the scene's instance count exceeds capacity.
- [x] Procedural primitives: `cube_mesh`, `sphere_mesh`, `cylinder_mesh`
      (capsule deferred until an arm-rendering test actually needs it).
- [x] `scene` module with `Scene`, `MeshHandle`, `Instance`, `InstanceRaw`,
      and `translation` / `scale` / `IDENTITY_MAT4` helpers; tests covering
      `InstanceRaw` layout + round-trip.
- [x] Default demo scene in `State::new`: ground plane + three colored
      cubes + a sphere + a cylinder. `cargo run -- raster` shows it.

**Done when:** a small assembled scene renders correctly. ✓ — 24 unit
tests green, native + wasm builds clean. Visual confirmation lands with
the motum demo widget once it's the first real consumer of the scene
API.

### R3 — Overlays for planner artifacts ✅ DONE

- [x] Line + point primitives via **native LineList / PointList
      topologies** (decision: simpler than instanced thin quads, fine
      for visualisation-quality 1 px lines, stays inside the WebGPU
      baseline — anti-aliased / thickness-configurable variants are a
      follow-up if motum needs them).
- [x] `Scene` grew **two overlay slots** instead of one:
      - `depth_tested_overlay` — `depth_compare: Less`. Geometry
        occludes the overlay (trajectory disappears behind a robot
        link).
      - `on_top_overlay` — `depth_compare: Always`. Overlay sits on
        top of everything (goal markers, axis indicators, planner
        search trees you want to keep readable).
      Both slots are an [`Overlay { lines, points }`] mixing line
      segments (pairs of [`OverlayVertex`]) and individual points.
- [x] Four overlay pipelines built off one `overlay.wgsl`: topology
      × depth-mode = 2 × 2. Stored as `overlay_pipelines[topology][depth_mode]`
      on `State`. Alpha-blending enabled; depth writes disabled either
      way so successive overlays don't occlude each other.
- [x] Two growable vertex buffers (`overlay_line_buf`, `overlay_point_buf`)
      hold `concat(depth_tested, on_top)` per primitive type. Per-frame
      packing mirrors the existing instance buffer's pattern — same
      `next_power_of_two` grow rule, same `queue.write_buffer` upload.
- [x] Render order inside the existing forward pass: triangles →
      depth-tested lines → depth-tested points → on-top lines →
      on-top points. One render pass, one bind group (shared with
      `forward.wgsl` via an identical `FrameU` layout).
- [x] Tests: `OverlayVertex` stride pinned at 28 bytes;
      `Overlay::line` appends two vertices with matching colour;
      `Overlay::point` appends one; `Overlay::clear` resets both;
      `Scene::is_empty` accounts for overlay state. Naga validation
      pins `overlay.wgsl` in `tests/shaders.rs`.
- [x] Default raster scene gains the demo overlay: three
      depth-tested coordinate axes at the origin (X red, Y green, Z
      blue) plus a yellow on-top point above each colored cube — gives
      `cargo run -- raster` an immediate visual demonstration of both
      slot semantics.

**Throughput at 10 000 lines.** The plan called for a 60fps target
with 10k lines in the browser. The buffer machinery (growable, single
`write_buffer` per frame, native LineList) scales to it on paper —
10k lines × 2 verts × 28 bytes = 560 KB upload per frame, ~33 MB/s at
60 Hz, trivial on any backend. Actual in-browser FPS pinning waits
for the motum widget in R4 (no test-harness page demands it yet);
recorded here as the "if this isn't 60fps once motum's wired up,
look at the upload, not the pipeline" debugging note.

**Same `FrameU` for both shaders.** `overlay.wgsl` declares the same
8-row `FrameU` struct as `forward.wgsl`; only `view_proj` is actually
read in the overlay path, but the lighting fields stay at matching
offsets so one bind group covers both pipelines. Saves duplicating
the uniform buffer and survives `Limits::downlevel_webgl2_defaults()`
unchanged — overlays add zero new resource bindings beyond a vertex
buffer per primitive type.

### R4 — Motum-shaped scene API for embedders ✅ DONE

- [x] **JSON wire format** in `pathtrace::raster::wire`:
      - `WireWorldState` is **byte-for-byte compatible** with
        `motum::world::WorldState` — the inner Pose matches nalgebra's
        `Isometry3<f64>` serialisation (`{ rotation: { quaternion: {
        coords } }, translation: { vector } }`). 9 unit tests pin the
        shape end-to-end, including identity, translation-only, and a
        90° Y rotation that demonstrates the column-major matrix
        builder.
      - `WireTrajectory` carries **world-space** waypoints
        (`{time, world_state}` pairs), not motum's joint-space
        `Trajectory`. Motum applies FK once per waypoint before sending
        — the renderer has no robot model, so expecting it to do FK
        would be a layering violation.
      - `WireTreeOverlay` carries **world-space** edges + nodes
        (`{from, to}` line segments and `[x,y,z]` points), not
        motum's `PlannerTree { nodes: Vec<Configuration> }`. Same
        reason: motum projects each tree node into world space (typically
        end-effector position) before sending.
      - `WireGoal` is a single [`WirePose`] used to position the
        draggable handle.
- [x] **wasm-bindgen API** on `RasterInstance` (gated `cfg(target_arch =
      "wasm32")`, in `raster::web`):
      - `setWorldState(json)` / `setTrajectory(json)` /
        `setTrajectoryTime(t)` / `setTreeOverlay(json)` /
        `setGoal(json)`.
      - `onGoalChanged(callback)` registers a `js_sys::Function`; the
        renderer fires it on `pointerup` after a drag with the new
        pose as a JSON string (manually formatted to match the
        identical schema `parseGoal` accepts — round-trip tested).
- [x] **Goal handle drag** built around three pure math helpers in a
      new `raster::picking` module (testable on native, no wasm gating):
      - `camera_ray(camera, canvas_w, canvas_h, mx, my)` builds a
        world-space ray through the cursor using the OrbitCamera's
        eye/forward/up basis. 2 tests pin centre-screen and right-
        edge directions.
      - `ray_floor_hit(origin, dir)` intersects the `y = 0`
        manipulation plane. 3 tests pin straight-down, upward (no
        hit), and a 45° slant.
      - `ray_hits_sphere(origin, dir, center, radius)` for picking.
      - `serialize_pose(pose)` round-trips through `parse_goal` (1 test).
      On pointerdown, the canvas-local ray is sphere-tested against the
      goal handle; if it hits, drag mode is entered (camera orbit is
      suppressed) and pointermove projects the cursor onto the floor
      until pointerup fires the callback.
- [x] **`index.html` harness** has a new section that creates a
      `create_raster("raster-host")`, pushes a hand-coded motum-shaped
      JSON (two 2-segment-arm waypoints + a hand-drawn 6-edge tree
      overlay + an initial goal), wires the trajectory playhead to a
      slider, and writes the dragged goal pose into a readout below
      the canvas. Drag the yellow ball and the coords update live.

**Trajectory playback is snap-to-waypoint, not interpolated.** The
`active_world_state` impl picks the most recent waypoint at or before
the playhead time. Lerp between waypoints would require interpolating
rotations as well (slerp), which doesn't have a clean two-line
implementation; ship snap for now and add lerp if the motum widget
asks for it.

**Throughput target from R3 carries over.** R3's plan note said "10k
lines @ 60fps in browser is what to verify once motum's wired up."
The R4 demo only ships 6 edges + 6 points so we still haven't
exercised the 10k floor — but the buffer machinery + native LineList
topology haven't changed. Filed as the same debugging note: if motum's
widget isn't 60fps, profile the per-frame `populate_scene` clear-and-
rebuild, not the pipeline.

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
