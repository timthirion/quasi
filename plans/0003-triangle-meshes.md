# Triangle meshes + BVH (path tracer)

- **Status:** proposed
- **Last updated:** 2026-06-04
- **Last touched on:** planning (no code yet)

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

### T0 — glTF ingest (CPU)

- [ ] Add `gltf` to native + wasm deps; pin to current 1.x.
- [ ] `pathtrace::mesh::load_glb(path) -> TriangleScene` for native and
      `load_glb_bytes(&[u8])` for the wasm side (browser can `fetch`
      bytes).
- [ ] Flatten the glTF node hierarchy into world-space triangles at
      load (premultiplied transforms; no instancing yet).
- [ ] Materials: `baseColorFactor` → `albedo`, `emissiveFactor` →
      `emission`, `roughnessFactor` / `metallicFactor` stored for
      later. Default material if absent.
- [ ] Unit tests: a programmatic 2-triangle glTF round-trips
      vertex counts, material indices, and an emissive triangle list.

### T1 — Triangle intersection, no BVH

- [ ] WGSL `triangle.wgsl` (or new entries in `pathtrace.wgsl`): linear
      scan over all triangles, Möller–Trumbore intersect, NEE+MIS
      against the now-multi-triangle light set.
- [ ] Storage-buffer plumbing: vertex / index / material / triangle-
      material / emissive-triangle buffers.
- [ ] Replace the M0–M3 quad scene at the API: `State::new` (and
      `offscreen::render_offscreen`) take a `TriangleScene` instead of
      synthesising the Cornell quads inline. Cornell Box is shipped
      as `assets/cornell.glb`.
- [ ] Regression: RMSE between the new triangulated Cornell render at
      1024 spp and the M3 reference under 1e-3.

### T2 — SAH binned BVH

- [ ] `pathtrace::bvh::Bvh::build(vertices, indices) -> Bvh` —
      recursive SAH binned split; 16 bins; leaf cap 4. Unit-tested
      against synthetic point sets (AABB tightness, leaf-triangle
      coverage, balanced builds on uniform inputs).
- [ ] Linear node layout per the design above; CPU-side struct
      asserts size / offsets vs. the WGSL `Node` struct.
- [ ] WGSL stack traversal in `triangle.wgsl`. The linear-scan path
      stays as a `--brute-force` flag for verification only.
- [ ] Benchmark: at the canonical Cornell + bunny scene, BVH
      traversal is at least 10× faster than the linear scan
      (measured by render time at 256×256 / 64 spp; recorded in plan
      notes).

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
