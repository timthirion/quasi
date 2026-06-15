# RT-cluster-cull — GPU-driven cluster culling for the raster track

- **Status:** draft
- **Last updated:** 2026-06-15
- **Last touched on:** initial draft from the Nanite-direction scoping note

## Goal

Add a **GPU-driven cluster culling pipeline** to Quasi's raster
track (`src/raster/`). Subdivide each mesh into ~128-triangle
clusters with bounding spheres + visibility cones; on every
frame, run a compute pass that frustum-culls, backface-cull-
cones, and Hi-Z-occludes clusters, then indirect-draws the
survivors through the existing forward shader.

This is the **tier 1** piece of a Nanite-like virtualized-
geometry track. Two follow-up plans target the harder pieces:
RT-micropoly (software rasterizer for sub-pixel triangles) and
RT-virtualized (cluster DAG + LOD + page streaming). This plan
delivers the GPU-driven-rendering substrate they all sit on,
sized for one focused milestone arc and one blog post.

## Why this is the right tier-1 piece

* **It's the GPU-driven rendering substrate.** Compute-pass
  culling + `dispatchWorkgroupsIndirect` + indirect-draw is
  the architecture every Nanite-style technique builds on.
  Ship this and the next two plans get to focus on their own
  concerns instead of re-inventing the pipeline.
* **It works on WebGPU today.** Compute shaders, atomics on
  storage buffers, and `drawIndexedIndirect` are all in the
  WebGPU MVP. No spec extensions, no Safari-only features, no
  bindless requirement.
* **It serves an actual measurement.** A million-triangle
  stress scene rendering at significantly lower frame time
  than naive draw-all is a publishable "GPU-driven rendering
  in the browser" demo without any of the deeper Nanite
  machinery.
* **It doesn't conflict with motum.** The existing motum-shaped
  JSON scene API continues working unchanged; clusters are
  built per registered mesh at the time of upload, and a
  scene with one cube + one cylinder degenerates to "two
  clusters, both pass culling every frame" — same as today
  in frame-time terms.

## Why this is the wrong place to start (and why we do it anyway)

Quasi's raster track exists to serve motum's in-browser
planner widgets — robots + obstacles, sub-100K-triangle
scenes. Motum will not exercise cluster culling at all in its
typical use case; this plan does not move motum's needle. The
case for shipping it anyway:

1. **The raster track has had R0–R4 done since 2026-06-04 and
   no new plans since.** The track's lifeline as something
   other than "the thing motum uses" needs a forward direction
   that's about *raster as a subject worth writing about*, not
   just *raster as a motum delivery channel*.
2. **The plan-skeptic for RT-micropoly will demand this be
   in place first** (the software rasterizer needs clustered
   input + per-cluster compute dispatch). Better to land it
   as a separable plan with its own validation than as a
   prerequisite buried inside a larger one.

The plan explicitly does **not** lay claim to performance on
small scenes; the stress-scene milestone shows the win at a
scale the existing raster track has not been measured at.

## Design

### Pipeline overview

```
existing per-frame raster (R4):
    ↓
    upload uniforms (camera + scene state)
    draw each registered mesh instance via fixed-function bind groups
    overlay pass (lines, points, goal handle)
    present pass

new per-frame raster:
    ↓
    upload uniforms (camera + scene state)
    [build Hi-Z pyramid from last frame's depth — once, post-resolve]
    cull compute pass: cluster_id → visible-cluster list
    compact visible list into indirect-draw args buffer
    drawIndexedIndirect against the visible-cluster index ranges
    overlay pass (unchanged)
    present pass
```

### Cluster representation

A cluster is **128 triangles** by default (configurable). For
a closed mesh of N triangles, expect ~N/128 clusters. Each
cluster carries:

```rust
#[repr(C)]
struct GpuCluster {
    bounding_sphere: [f32; 4],  // (cx, cy, cz, radius)
    normal_cone: [f32; 4],      // (nx, ny, nz, cos_half_angle)
    index_offset: u32,          // first index in the mesh's index buffer
    index_count: u32,           // always 128 * 3 == 384 except the last cluster
    material_id: u32,
    _pad: u32,
}
```

48 bytes per cluster. A 1 M-triangle mesh produces ~7800
clusters → ~370 KB of cluster metadata. Trivially fits.

### Cluster builder (CPU side)

Two-pass naive clusterer for v1:

1. **Sort triangles by Morton code** of their centroid. Groups
   triangles that are spatially close.
2. **Slide a 128-wide window**, emit each window as a cluster
   with its bounding sphere + normal cone computed over the
   triangles in the window.

This is significantly worse than METIS / Karis's edge-graph
clusterer, but it ships in ~200 LOC, has no external
dependencies, and is good enough for the v1 measurement. A
follow-up (RT-cluster-build-metis) can swap in a real
partitioner once the rest of the pipeline is proven.

Reference for the right clusterer: Brian Karis, "A Deep Dive
into Nanite Virtualized Geometry" (SIGGRAPH 2021), §3.1
"Cluster generation." We'll cite the paper but ship the
Morton-window approximation; the gap is documented in
`Findings`.

### Frustum + backface culling (GPU compute)

One workgroup per 32 clusters (subgroup-sized). Per cluster:

* **Frustum test:** classify the bounding sphere against the
  6 view-frustum planes; out → cull. Standard
  Ericson-style plane-sphere math.
* **Backface cone test:** if the cluster's normal cone's
  half-angle and the cone-from-camera-to-cluster-center
  don't overlap on the camera side, all triangles in the
  cluster face away → cull. Adapted from Karis 2021 §4.2,
  derived more accessibly in the Frostbite "Mesh Shaders"
  presentation (Achton, GDC 2019).

The cull pass writes a compacted visible-cluster index list
into a storage buffer using atomic increment of a counter.

### Hi-Z occlusion culling

Each frame's resolved depth buffer is downsampled into a
mip-chain depth pyramid via a compute pass. The next frame's
cull pass reads the pyramid: for each cluster, sample the
appropriate Hi-Z mip level for the cluster's projected
bounding box; if every sampled texel's depth is *closer to
camera* than the cluster's nearest depth, the cluster is
occluded.

**Standard caveat: this is temporally stable but one-frame
late.** Camera teleports or fast strafes cause one frame of
"too aggressive cull, things pop in." Mitigation: a small
expand-bounds margin (5%) at the cost of slightly less culling.
A more robust two-pass approach (cull-by-last-frame, draw,
cull-newly-visible, draw again) is documented as a follow-up.

### Indirect draw + the WebGPU-MVP API

The cull pass writes one `DrawIndexedIndirectArgs` entry per
surviving cluster into a storage buffer (with a separate
atomic counter for "how many clusters survived"). The CPU
issues a single `drawIndexedIndirect` call against this
buffer.

WebGPU's MVP **does not** have multi-draw-indirect; we issue N
separate draw calls in a loop from CPU, sourcing their args
from the GPU-resident buffer. This is sub-optimal vs Vulkan/D3D12
multi-draw but works. The "N separate indirect draws" cost is
mainly CPU-side overhead which scales with N. The cull pass's
output is the survivor count read back to CPU — a one-frame
round-trip we accept as the cost of WebGPU-MVP scope.

The multi-draw-indirect alternative is documented as a
follow-up (RT-multidraw-extension) once the wgpu side exposes
WGPUMultiDrawIndirect on backends that support it.

### Bindless workaround for materials

Nanite uses bindless. WebGPU doesn't. Workaround: a **texture
array** (already used by PT-textures in the path-trace track),
material indices in the cluster struct, sampled per fragment.
Same workaround Quasi already ships on the path-trace side.

This caps the renderer at ~256 distinct materials per scene
(texture-array layer limit per spec). Adequate for the
stress-scene milestone; constrained but documented for any
future scene-class push.

### Native + web lockstep

The cull pass, Hi-Z pyramid, and indirect-draw all use
WebGPU-MVP features. Native + wasm build the same WGSL +
storage-buffer + indirect-draw paths. No conditional code
between targets. The
[`feedback_native_web_lockstep`](../memory/feedback_native_web_lockstep.md)
rule is preserved.

### Backward compatibility with motum

The existing motum-shaped JSON scene API
(`setWorldState` / `setTrajectory` / etc.) registers a few
small meshes; each becomes one cluster (since its tri count
is < 128) and renders identically to today. Existing motum
tests are extended to assert the cluster pipeline doesn't
change the rendered output bit-for-bit (RMSE ≤ 0.05 vs
pre-plan).

## Milestones

- [ ] **[RT-cluster-cull/clusters]** CPU-side mesh clusterer
  in `src/raster/cluster.rs`. Morton-sort-and-window
  algorithm. Per cluster: bounding sphere (Ritter or
  Welzl) and normal cone (Karis 2021 §4.2 formula).
  **CPU unit tests:**
  * On a known unit cube (12 triangles → 1 cluster),
    bounding-sphere centre is origin, radius is `√3/2`,
    normal cone half-angle is `π`.
  * On a flat plane mesh (all normals coincide), normal
    cone half-angle is ~0.
  * On a procedural sphere (5184 triangles → ~41 clusters
    at 128 tri/cluster), every triangle index appears in
    exactly one cluster.
- [ ] **[RT-cluster-cull/buffers]** GPU upload path:
  `GpuCluster` storage buffer + per-cluster vertex/index
  buffers. New `RasterClusterBuffers` struct in
  `src/raster/cluster.rs` paralleling `SceneBuffers` for
  the path-trace side. Existing `upload_mesh` keeps its
  contract; the cluster builder runs **alongside** the
  current `MeshHandle` path so the motum-facing API is
  unaffected.
- [ ] **[RT-cluster-cull/frustum-cull]** WGSL compute shader
  `src/raster/shaders/cull.wgsl`. One workgroup per 32
  clusters; per-cluster plane-sphere classification against
  6 frustum planes. Writes `1` or `0` to a per-cluster
  visibility byte buffer. **Test:** known camera pointing
  at a procedural sphere, with another sphere placed off-
  screen → only the on-screen sphere's clusters survive,
  numeric count matches the closed-form expectation.
- [ ] **[RT-cluster-cull/backface-cone]** Adds backface-cone
  rejection to the cull shader. **Test:** a back-facing
  cluster (all triangles face away from the camera) gets
  culled; flipping the camera 180° flips which clusters
  survive. Tested on the procedural sphere.
- [ ] **[RT-cluster-cull/hi-z]** Hi-Z pyramid build pass
  (`src/raster/shaders/hiz.wgsl`) + occlusion test in
  `cull.wgsl`. Pyramid is a mip-chain of the depth buffer;
  each mip stores the **max** of its 4 source texels
  (reverse-Z assumed; depth=1 means far). **Test:** a small
  cluster placed behind a large occluder gets culled at the
  appropriate Hi-Z mip level. Numeric: the cull-rate on a
  stress scene with high overdraw is ≥ 50%.
- [ ] **[RT-cluster-cull/indirect-draw]** Compaction pass
  writes `DrawIndexedIndirectArgs` to a storage buffer;
  surviving-count read back to CPU; CPU issues N
  `drawIndexedIndirect` calls. **Test:** rendered frame
  pixel-matches the pre-plan "draw all clusters naively"
  output within RMSE ≤ 0.001 (essentially identical;
  difference is from indirect-draw command ordering only).
- [ ] **[RT-cluster-cull/motum-noregression]** Re-run the
  existing motum-API tests with the cluster pipeline
  enabled. **Done-when:** existing motum scene tests pass
  unchanged; rendered output matches the pre-plan
  rendering within RMSE ≤ 0.05.
- [ ] **[RT-cluster-cull/stress-scene]** Procedural scene
  with 1 M total triangles (e.g. 250 subdivided spheres at
  4000 triangles each, scattered through a viewing volume).
  **Numeric Done-when:**
  * Frame time with cluster culling enabled: ≤ 16 ms at
    1280×720 on Apple M-series native.
  * Frame time with naive draw-all (no culling): ≥ 80 ms
    on the same hardware (i.e. ≥ 5× speedup from culling).
  * Cull rate (fraction of clusters culled per frame): ≥ 60%
    on a typical viewpoint.
  * Same scene rendered in browser (Apple M-series Safari):
    ≤ 33 ms (30 fps target for the browser, which carries a
    WebGPU driver overhead the native path doesn't pay).

## Done when

* All seven milestones ticked
* Stress-scene numeric table in `Findings`: M-series native
  frame time, M-series Safari frame time, cull rate, on a
  reproducible procedural scene committed at
  `examples/gen_cluster_stress.rs`
* Motum existing tests pass; rendered output regression test
  green
* README features list gains "GPU-driven cluster culling
  (RT-cluster-cull)" under runtime
* Plan moves to `Status: completed`

## Findings

(Populated during execution.)

## Followups (out of scope)

* **RT-cluster-build-metis** — swap the Morton-window
  clusterer for a proper edge-graph partitioner (METIS or
  the Karis 2021 algorithm). Yields more spherical
  clusters → tighter bounds → higher cull rates.
* **RT-multidraw-indirect** — once wgpu surfaces
  multi-draw-indirect on Vulkan/D3D12 backends, collapse
  the N-iteration CPU loop into one call. WebGPU MVP gap;
  doesn't help wasm.
* **RT-twopass-occlusion** — Karis 2021's two-pass occlusion
  (draw-last-visible, build Hi-Z, cull-newly-visible, draw
  again) eliminates the one-frame-late artifact at the cost
  of one extra cull + draw pass per frame. Worth it once a
  scene with sufficient occlusion makes the artifact visible.
* **RT-cluster-lod** — per-cluster LOD selection at draw
  time. Prerequisite for RT-virtualized; meaningful standalone
  if scene-scale geometry warrants.
* **RT-micropoly** (plan 0033) — software rasterizer for
  sub-pixel triangles. Plugs into the indirect-draw arg
  buffer with a per-cluster "small or large" classifier.
* **RT-virtualized** (plan 0034) — cluster DAG + LOD +
  streaming. The full Nanite story.
