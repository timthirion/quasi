# Triangle meshes + BVH (path tracer)

- **Status:** active
- **Last updated:** 2026-06-04
- **Last touched on:** T2 landed — SAH binned BVH on CPU + WGSL stack traversal

## Goal

Replace the path tracer's analytic-quad scene with **triangle meshes**
loaded from glTF, accelerated by a CPU-built **SAH binned BVH** that the
GPU traverses with a stack-based loop. This unlocks every later
publishable artifact: render the Stanford bunny in the Cornell Box,
load arbitrary blog-test meshes, and stage the renderer for richer
materials (GGX, dielectrics) without rebuilding the geometry pipeline.

Plan 0001 took us to a complete Cornell-Box NEE+MIS path tracer; this
plan is the first **post-foundation** path-tracer plan, picking up
where 0001 closed. Built as ordered milestones (T0–T4). Each is
independently shippable and may be split into its own
`plans/000N-*.md` as it starts; this document is the spine.

## Context

Today the path tracer's scene is `pathtrace::scene::Scene` — at most 32
analytic `Quad`s with per-quad materials, stored inline in a uniform
buffer and intersected by a linear `for i in 0..quad_count` loop in
`pathtrace.wgsl`. That carries us through the Cornell Box and the M3
convergence study; it can't carry a triangulated mesh.

Other relevant existing surface:

- `pathtrace::offscreen` already creates a fresh device with
  `adapter.limits()` and renders to `Rgba32Float` AOVs, so adding
  larger storage buffers (mesh + BVH) is unblocked.
- `pathtrace::sampler` (PCG / Halton / Sobol) and `pathtrace::integrator`
  (MisNee / Bsdf) are orthogonal to geometry — they keep working with
  zero changes.
- The convergence runner (`pathtrace::converge`) treats radiance as a
  black box; it'll re-run unchanged against the new scene.

## Design

### Library choice

- **`gltf`** (the [`gltf` crate](https://crates.io/crates/gltf), glTF 2.0
  loader, Apache-2.0 / MIT). One ingest path. Carries hierarchical
  transforms, indexed positions / normals / UVs, and PBR-aligned
  materials (baseColorFactor, emissiveFactor, metallic / roughness
  scalars) — exactly the metadata the renderer will need as BSDFs
  expand in later plans. `.glb` (binary, self-contained) is the
  preferred on-disk form; `.gltf + .bin` also works.
- **Not bringing in a BVH crate.** SAH binned BVH is a classic
  ray-tracing component with measurable quality (build time, traversal
  cost), and we want full control over the linear GPU layout. Writing
  our own keeps the "use the language" promise (`AGENTS.md`) honest.

### Scene representation (replaces `pathtrace::scene`)

```rust
pub struct TriangleScene {
    pub vertices: Vec<Vertex>,     // position + normal (+ UV later)
    pub indices: Vec<u32>,         // triangle list, 3 indices per tri
    pub materials: Vec<Material>,  // small palette, indexed by triangle
    pub triangle_materials: Vec<u32>,
    pub bvh: Bvh,
    pub emissive_triangles: Vec<u32>, // for NEE
}

#[repr(C)]
pub struct Vertex { pub position: [f32; 3], pub normal: [f32; 3] }
```

The analytic Cornell Box gets a programmatic-glTF emitter (or just a
shipped `cornell.glb`) so M0–M3's reference image remains the
regression target. The quad shader is deleted; we never branch
geometry kinds at runtime.

### BVH layout

CPU build, GPU traverses. Inner nodes hold AABBs and two child
indices; leaves hold a triangle index range. Pack as **fixed-size
`Node`** (32 bytes; vec3 aabb_min, u32 left, vec3 aabb_max, u32 right)
with the leaf flag encoded in the MSB of `left` (and the low bits
holding the first triangle index; the count goes in the low bits of
`right`). The whole tree is one contiguous buffer (`Vec<Node>`),
uploaded as a storage buffer to the GPU.

- **Build:** SAH binned, 16 bins per axis, recursive split with a
  small-leaf cap (4 triangles). Single-threaded for T2; revisit with
  `rayon` once the convergence study calls for it.
- **Traversal:** WGSL function-local stack of `array<u32, 32>` — depth
  32 covers ~4 G triangles. Standard near-far ordered descent.
- **Triangle intersection:** Möller–Trumbore.

### Lights (T3)

The analytic Cornell light is currently a single quad; in the
triangulated scene it becomes 2 emissive triangles. NEE needs to
sample one of `N` emissive triangles uniformly, then sample a point
inside the chosen triangle by barycentric. MIS weight follows the
same power-heuristic formula with the per-triangle area PDF
(`pdf_w = dist² / (cos_l · area · N_emitters)`). Many-light sampling
(power-weighted, light BVH, ReSTIR) is **future work**, not in this
plan.

### Out of scope

- Triangle-mesh **instancing** (two-level BVH). glTF can encode it;
  we'll flatten transforms into world-space triangles at load. Add
  instancing in a follow-up if scenes get large enough to need it.
- Texture sampling. `baseColorTexture` is read from glTF but unused
  until UVs and a texture sampler are wired up — future plan.
- BSDFs beyond Lambertian. Materials' metallic / roughness fields
  get parsed and stored, but the shader still treats every surface
  as Lambertian. GGX / dielectrics live in a separate plan.

## Steps

### T0 — glTF ingest (CPU) ✅ DONE

- [x] Added `gltf = "1"` (1.4.1) as a shared dep — compiles for both
      native and `wasm32-unknown-unknown` via its `import_slice` API.
- [x] `pathtrace::mesh::load_glb(path)` (native, reads via `std::fs`)
      and `load_glb_bytes(&[u8])` (cross-target — the wasm side can
      pass `fetch`ed bytes).
- [x] Node hierarchy flattened by walking each scene's root nodes
      with an accumulated parent×local transform; per-vertex
      `transform_point` and per-vertex `transform_normal` (cofactor-
      based so non-uniform scale is correct). No instancing yet — each
      glTF "node referenced from two places" is currently re-walked,
      which would matter once we ship two-level BVH; T0 doesn't.
- [x] Materials: glTF's `baseColorFactor` → `albedo`,
      `emissiveFactor` → `emission`, `roughnessFactor` /
      `metallicFactor` stored alongside (unused by today's Lambertian).
      Slot 0 of `TriangleScene::materials` is always the default
      Lambertian white, so a primitive with no material binds to it.
- [x] Tests (12 in `pathtrace::mesh::tests`):
      Vertex / Material layouts pinned at 32 bytes (and Material =
      `scene::GpuMaterial` byte-for-byte); `Material::is_emissive`
      and `TriangleScene::recompute_emissive` correctness;
      `identity / translation / Y-rotation / non-uniform scale`
      transforms; `mat4_mul` identity; a programmatic 2-triangle
      glTF (inline JSON + base64-embedded buffer, no fixture file)
      round-trips through `load_glb_bytes` for vertex counts, material
      assignment, and the emissive triangle list; missing-NORMAL
      attribute returns `MeshError::NoNormals`.

**Vertex dedup decision.** The loader doesn't dedupe vertices across
primitives that share an accessor; each primitive's accessor read is
appended to the global vertex buffer with its own `vertex_offset`. The
2-triangle test catches this explicitly (12 vertices for 2 primitives
sharing one POSITION accessor) so the behaviour is intentional, not a
bug. Trades a small constant amount of memory for a simpler ingest
that matches real-world glTF files where primitives usually have
distinct attributes.

**Glb-via-data-URI test trick.** The round-trip test embeds the binary
buffer as a `data:application/octet-stream;base64,...` URI inside an
inline JSON glTF, so the entire test is self-contained — no
`gltf-json` dev-dep, no fixture file in the repo. A 24-line inline
base64 encoder lives next to the test.

**Implication for the testing principle.** This change also rewrote
`AGENTS.md`'s **Testing** section to spell out the discipline as
non-negotiable: no module ships without tests, a layout-pinning test
per uniform/buffer struct, cross-language constant pinning (Rust
discriminant ↔ WGSL `const`), naga validation, error-path tests, and
explicit honesty about GPU-only paths. Codifies what M0–M4 already
practised so future contributors don't drift.

### T1 — Triangle intersection, no BVH ✅ DONE

- [x] WGSL `pathtrace.wgsl` fully rewritten: Möller-Trumbore triangle
      intersection, linear scan over `U.triangle_count`, geometric
      normals flipped to face the ray, double-sided. NEE samples one
      emissive triangle uniformly + barycentric inside it; the
      light-pdf-solid-angle formula carries a `1/N_emitters` factor and
      the per-triangle area. MIS power-heuristic against the BSDF pdf
      unchanged from M3. Sampler / integrator dispatch is geometry-
      agnostic and kept intact.
- [x] Storage-buffer plumbing: `Uniforms` shrank from ~2.6 KB (camera
      + 32 inline quads + 32 materials) to 80 bytes (camera + 8 × u32).
      Five new fragment-stage storage buffers: `vertices`, `tri_indices`,
      `materials`, `tri_materials`, `emissive_triangles`. Shared helpers
      in `pathtrace.rs` (`build_pathtrace_bgl`, `build_scene_buffers`,
      `build_pathtrace_bg`) are used by both `State` and the offscreen
      renderer.
- [x] Replaced the M0-M3 quad scene at the API. `State::new()` and
      `offscreen::render_offscreen()` now take a `TriangleScene`; the
      default Cornell is `include_bytes!("../data/gltf/cornell_quads.gltf")`
      decoded via `mesh::load_glb_bytes`. CLI: `render --scene PATH`
      loads any glTF.
- [x] Cornell shipped as **two** committed glTFs in `data/gltf/`,
      generated by `cargo run --example gen_cornell`: `cornell_quads.gltf`
      (16 logical quads × 2 triangles = 32 triangles, the minimal
      tessellation) and `cornell_tris.gltf` (each quad subdivided 4×4 =
      512 triangles, stresses the linear scan a bit). Both files are
      under 25 KB.
- [x] **Regression** (gated, GPU-only): `tests/cornell_gltf.rs` renders
      `cornell_quads.gltf` and `cornell_tris.gltf` at 128×128 / 256 spp
      and asserts `rmse_rgb < 0.05` between the two radiance buffers.
      Measured on Apple M4: **rmse = 0.0049** — well inside the bound.
      Plus a fresh convergence sweep at 64×64 reproduces the M3
      numbers to within sampling noise (PCG MIS+NEE 64 spp: 0.025
      identical; MIS beats BSDF 3-5× at every checkpoint).

**Limits change.** Fragment-stage storage buffers aren't granted by
`wgpu::Limits::downlevel_webgl2_defaults()` (the WebGL2 fallback
profile). T1 switches the wasm device request to `Limits::default()`
(WebGPU baseline — `max_storage_buffers_per_shader_stage = 8`, we use
5). WebGL2 fallback is dropped on purpose; per `AGENTS.md` it was
always "WebGPU first, fallback maybe."

**The quad-vs-tris regression catches what we want.** Both files
describe the same Cornell geometry at different tessellations. A
geometry encoding bug (wrong winding, wrong normal direction, bad
material indexing, off-by-one in `tri_indices`) shows up as a large
RMSE between the two renders; sampling noise alone keeps RMSE around
~5e-3 at 256 spp.

**Test discipline (new categories).** Per the testing-discipline
refresh that went in with T0, T1 adds:
- A layout test for the new 80-byte `Uniforms` (offsets of
  `triangle_count` and `integrator_kind` pinned).
- An integration test that round-trips both committed glTFs through
  `load_glb_bytes` and checks topology (32 / 512 triangles, 5
  materials, 2 / 32 emissive triangles, identical material palettes).
- A GPU-only regression marked `#[cfg(not(target_arch = "wasm32"))]
  #[ignore]` so it doesn't break headless CI but can be invoked with
  `cargo test -- --ignored` locally.

### T2 — SAH binned BVH ✅ DONE

- [x] `pathtrace::bvh::Bvh::build(vertices, indices)` — recursive SAH
      binned split (16 bins/axis), leaf cap 4. Per-triangle AABBs +
      centroids precomputed; centroid-binning drives the SAH cost
      function (`left_count * left_area + right_count * right_area`).
      Degenerate inputs (all-colocated centroids, partition collapses
      to one side) fall back cleanly to a leaf. 9 unit tests cover
      Node layout, leaf bit pattern, single-triangle / empty / many-
      identical / 64-triangle-line builds, AABB tightness, child-
      index validity.
- [x] **Linear node layout** matches WGSL std430 byte-for-byte: 32
      bytes per `Node`, `vec3 + u32 + vec3 + u32` (vec3 size 12 bytes,
      so the u32 packs at offset 12 with no pad). Layout test
      `node_size_matches_wgsl_layout` pins it. The leaf flag is the
      MSB of `left_or_first`; low 31 bits index `triangle_indices`
      (for a leaf) or the left child (for an inner node). Same
      constants on both sides (`LEAF_FLAG = 0x80000000`).
- [x] WGSL stack traversal: function-local `array<u32, 32>` stack,
      slab-method `intersect_aabb` with inverse direction (handles
      axis-aligned rays via IEEE 754 inf propagation), both
      `trace_scene_bvh` and `occluded_bvh`. The linear-scan path
      survives as `trace_scene_linear` / `occluded_linear`; a runtime
      `U.use_bvh` flag picks between them. `--brute-force` on the
      `render` CLI flips it.
- [x] **Benchmark on cornell_tris.gltf (512 triangles, Apple M4)**:
      BVH = 458 ms; brute = 1218 ms; **speedup = 2.7×** (at 256×256 /
      64 spp). The plan's 10× target was scoped to the Stanford-bunny
      scene (T4, ~70k triangles); at 512 triangles the BVH's per-
      iteration stack-push overhead eats into the savings (brute
      force's inner loop is incredibly cheap). The regression test
      `bvh_is_faster_than_brute_force_at_512_triangles` enforces a no-
      regression floor of 2×; the 10× claim moves to T4.
- [x] BVH ↔ brute-force agreement: RMSE = **5.1e-5** on cornell_tris
      at 128×128 / 64 spp — well below `1e-3`. Same Möller-Trumbore,
      same RNG, same Monte-Carlo paths; the only divergence source is
      floating-point ordering of the closest-hit search.

**Bind group grew to 8 entries** (was 6 at T1): added `bvh_nodes`
(binding 6) and `bvh_tri_indices` (binding 7). Still within
`Limits::default()`'s `max_storage_buffers_per_shader_stage = 8`.
`SceneBuffers`, `build_pathtrace_bgl`, and `build_pathtrace_bg`
absorbed the two new buffers in one pass; the windowed `State` and
the offscreen renderer share both helpers.

**Uniforms got a `use_bvh: u32` field** in what used to be the
trailing pad slot. Total size unchanged at 80 bytes. Layout test now
pins `use_bvh`'s offset (48 + 7×4 = 76).

**TriangleScene grew a `bvh: Bvh` field** and `mesh::load_glb_bytes`
builds the BVH after triangle ingest. Manual construction via
`TriangleScene::default()` gives an empty leaf BVH (single zero-
volume node) — the WGSL traversal indexes `bvh_nodes[0]`
unconditionally, so a truly empty `nodes` Vec would be UB on GPU.

**WGSL stack depth = 32** (constant `STACK_DEPTH`), enough for a BVH
over ~4 G triangles. The push guard `sp <= STACK_DEPTH - 2` prevents
overflow at the cost of dropping deep traversals — only relevant for
pathological non-SAH inputs. Recorded here in case a future plan
ships a non-SAH builder.

### T3 — Emissive triangle area lights

- [ ] NEE samples one emissive triangle uniformly, then a barycentric
      point inside it; PDF carries the `1 / N_emitters` factor.
- [ ] MIS power heuristic against the BSDF pdf as before.
- [ ] Verification: the new Cornell render (light triangulated into 2
      triangles) still matches the M3 reference within 1e-3 RMSE.

### T4 — Stanford bunny in Cornell Box (publishable artifact)

- [ ] Ship `assets/bunny-in-cornell.glb` (bunny scaled into the box,
      placed on the floor).
- [ ] Run a fresh convergence sweep against this scene; emit
      `convergence-bunny.csv`. PCG + MIS+NEE should converge in
      roughly the same shape as the box-only study; rel-MSE numbers
      become the baseline for later GGX work.
- [ ] One reference PNG at 1024 spp and one HDR EXR shipped as a
      release artifact for the blog post.

## Open questions

- **Vertex normals vs. face normals.** Prefer glTF-supplied vertex
  normals; fall back to face normals if missing. Confirm during T0.
- **Storage-buffer feature on web.** Storage buffers need
  `BUFFER_BINDING_ARRAY` / appropriate WebGPU limits. The Cornell
  bunny mesh is small enough; verify the browser path during T1.
- **Cornell light as 2 triangles vs. 1 quad emitter.** Two triangles
  is the obvious encoding; sample-light needs to weight by area, not
  count. Resolved during T3 (no decision deferred).
- **`gltf` crate version churn.** Track 1.x; pin a specific minor
  initially and bump deliberately, mirroring the wgpu 29 lesson.

## Done when

- Path tracer renders glTF-loaded triangle meshes natively and in the
  browser, with a SAH binned BVH on CPU and stack traversal on GPU.
- The Cornell Box render is bit-similar (RMSE < 1e-3 vs. M3
  reference) at equal spp.
- The Stanford-bunny-in-Cornell scene renders cleanly, with a CSV +
  PNG + EXR shipped as the first publishable artifact of the
  triangle-mesh era.
