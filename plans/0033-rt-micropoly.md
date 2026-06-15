# RT-micropoly — software rasterizer for sub-pixel triangles

- **Status:** draft
- **Last updated:** 2026-06-15
- **Last touched on:** initial draft from the Nanite-direction scoping note

## Goal

Add a **compute-shader software rasterizer** for clusters
whose triangles project to ≤ 1 screen-space pixel each, with
the existing hardware-rasterized forward pipeline handling
larger clusters. Per-pixel visibility is resolved through an
**atomic visibility buffer** (cluster_id + tri_id packed in
u32, paired with an atomic-min depth buffer); shading happens
in a separate full-screen resolve pass.

This is the **tier 2** Nanite piece, the one that delivers
the "micropolygon rendering, every triangle is a pixel"
property. It sits on top of plan 0032 RT-cluster-cull (which
ships the GPU-driven culling substrate) and feeds into plan
0034 RT-virtualized (which adds the LOD selection that makes
sub-pixel triangles actually common).

## Prerequisite

**RT-cluster-cull (plan 0032) must be in flight or shipped.**
The classifier this plan adds runs on already-clustered
geometry; the indirect-draw infrastructure is reused. Without
0032 there's no cluster-id to write into the visibility
buffer and no obvious entry point for the per-cluster
"small or large" classifier.

## Why this is the right tier-2 piece

* It's the *one* Nanite technique that hardware rasterizers
  fundamentally can't replicate at sub-pixel scale (HW
  rasterizers always cost N cycles per triangle regardless of
  its projected size; SW rasterizer drops to amortised O(1)
  for sub-pixel triangles because there's nothing to scanline).
* The atomic-visibility-buffer architecture is the same on
  every modern GPU; the WebGPU implementation only differs
  in that we have to fake the 64-bit atomic.
* It's the natural blog-able piece: "Nanite's software
  rasterizer, in WebGPU." Self-contained, technically
  meaty, validates against a stress scene with known
  sub-pixel-rich geometry (e.g. a Boeing model, a Bistro
  in distance LOD, anything photogrammetric).

## The WebGPU 64-bit atomic problem

Nanite packs each visibility-buffer entry as one 64-bit value
(top 32 bits: depth as float bits; bottom 32 bits: cluster_id
+ triangle_id), updated with one `atomicMax` per fragment.
This gives "the depth-test and the visibility-write race
together; the closest-fragment wins" with one atomic op.

**WebGPU does not currently have 64-bit atomics on storage
buffers.** Atomics are u32-only per the WGSL spec. Three
viable workarounds, picked in order of decreasing speed and
increasing implementation complexity:

1. **Two-pass with separate depth + visibility buffers (the
   one we ship).** First pass: `atomicMin` on a u32 depth
   buffer (depth bit-cast from f32; the bit-pattern's
   monotonicity for positive floats means u32-min is
   equivalent to f32-min for ≥ 0). Second pass: compare each
   fragment's depth against the now-resolved depth buffer;
   if matched, write the cluster+tri id directly (no atomic
   needed — the depth pass already serialized them). The
   race is now between identical-depth fragments; we accept
   "one of them wins, arbitrarily" as the Nanite-spec-relaxation.

2. **Compare-exchange loop on packed u32.** Pack depth into
   the high 16 bits and cluster+tri into the low 16 bits;
   `atomicCompareExchangeWeak` in a loop until the depth
   bits don't change. Halves the depth precision (16-bit
   float-equivalent), which is borderline-acceptable for
   1080p but degrades at higher resolutions.

3. **Wait for WebGPU 64-bit atomics.** The spec working
   group has discussed this; Chrome's Dawn has prototyped
   it; no timeline. Out of scope.

We ship (1). The plan documents the relaxation (sub-pixel
race-resolution is arbitrary instead of "first wins")
explicitly so reviewers don't compare us to Karis 2021 and
notice the difference unannounced.

## The depth-bit-cast trick

For depths ≥ 0, IEEE-754 float bit-patterns are
monotonically ordered as u32s. So:

```wgsl
atomicMin(&depth_buffer[pixel], bitcast<u32>(fragment_depth))
```

is bit-equivalent to:

```
min(depth_buffer[pixel], fragment_depth)
```

as long as fragment_depth ≥ 0, which is true for all visible
geometry after the perspective divide. Source: this is the
standard "atomic depth in compute" trick used by every recent
software rasterizer (Wihlidal, "Optimizing the GPU…" 2017
slides; Karis 2021 §6.3 calls it out).

NaN and negative depths break this; we clamp incoming depths
to `[0, 1]` before the bitcast as a defensive measure.

## Cluster classifier

After RT-cluster-cull's cull pass produces the surviving-
cluster list, a new compute pass classifies each surviving
cluster as `SMALL` (avg projected triangle area ≤ 1 px²) or
`LARGE`. The classification metric:

```
projected_diameter = (cluster.bounding_sphere.w * 2) * focal_length / depth_to_camera
projected_tri_area_avg = (projected_diameter² / 4) / cluster.tri_count
classification = if projected_tri_area_avg <= 1.0 { SMALL } else { LARGE }
```

`focal_length` here is in pixels (derived from FOV + viewport
size). This is approximate — a cluster might have one big
triangle and a hundred small ones — but the average-driven
classifier is what Karis 2021 §5 reports they actually use,
and it works because clusters are spatially small enough
that triangles within them have similar projected sizes.

Two indirect-draw arg buffers are written: one for SMALL
clusters (compute-dispatched), one for LARGE clusters
(hardware-rasterized via the existing R4 pipeline).

## Software rasterizer (compute pass)

One thread per triangle. Per-triangle work:

1. Project the 3 vertices to screen space (perspective divide
   + viewport transform).
2. Compute the triangle's screen-space AABB.
3. **For sub-pixel triangles** (the SMALL-classifier case),
   the AABB is at most 2×2 pixels. Loop over those 1–4 pixels
   inline; for each:
   * Edge function test (Pineda 1988) for inside-triangle.
   * If inside, compute barycentric depth + the atomic depth
     write via the bitcast trick.
4. **For 2-4 pixel triangles**, the same 2×2 loop is fine.
5. The plan **explicitly does not** support arbitrary-size
   software raster — the SMALL classifier guarantees ≤ 4
   pixels per triangle, so no scanline walk is needed.

The cluster_id + triangle_id (packed into one u32) is held in
registers until depth-write succeeds; a second pass reads the
resolved depth buffer and writes the visibility id (no atomic
on the id buffer needed — depth-write order resolved the
race).

## Resolve / shading pass

A full-screen fragment-shader resolve:

* Per pixel: read visibility buffer → unpack cluster_id +
  tri_id.
* Fetch cluster metadata, fetch the three vertices, fetch
  material.
* Compute barycentrics, interpolate UVs / normals.
* Run the existing R4 forward shader code path.

This makes shading **decoupled from geometry** — every visible
pixel costs one fragment regardless of cluster size or
sub-pixel triangle count. Big shading win on
overdraw-heavy scenes.

## Hardware path stays for large clusters

Clusters that classify as LARGE go through R4's existing
forward pipeline, drawn via the same indirect-draw machinery
RT-cluster-cull set up. The HW path's output goes into the
**same** visibility buffer as the SW path: each HW-rasterized
fragment runs through a small fragment shader that performs
the atomic-min depth write + visibility-id write, identical
to the SW path. So the resolve pass sees a unified visibility
buffer regardless of which raster path produced each pixel.

This requires the HW fragment shader to discard its colour
output (or render to a dummy attachment) and only write the
visibility buffer atomically. Slightly weird but matches
Nanite's actual architecture — the visibility buffer is the
shared intermediate, not the colour buffer.

## Stress scene

`examples/gen_micropoly_stress.rs` — a procedural scene
designed to exercise sub-pixel raster: a viewing frustum
filled with ~100K cubes scattered at varying depths, with the
camera positioned so the per-cube projected size is ~1-3
pixels. Total triangle count ≥ 1.2 M; classifier reports
> 80% as SMALL.

## Native + web lockstep

All shaders are WGSL compute + fragment. All atomic ops are
`atomicMin` / `atomicAdd` on storage buffers — fully
supported in WebGPU MVP. The visibility buffer is a single
storage buffer; the depth buffer is a u32 storage buffer.
Resolve is a standard fragment shader.

No bindless, no 64-bit atomics, no subgroup ops required.
Browser-portable.

## Milestones

- [ ] **[RT-micropoly/classifier]** Compute pass classifies
  each surviving cluster (from RT-cluster-cull's visible-
  cluster list) as `SMALL` or `LARGE` per the
  projected-tri-area metric. Writes two indirect-draw arg
  buffers. **Test:** on a hand-crafted scene with known
  cluster projections, classification matches the closed-
  form expectation.
- [ ] **[RT-micropoly/depth-buffer]** Storage-buffer u32
  depth buffer + `atomicMin`-via-bitcast write helper.
  **CPU unit test:** the bitcast trick is monotone for
  positive floats (test 100 random positive floats: u32
  ordering matches f32 ordering).
- [ ] **[RT-micropoly/visibility-buffer]** Storage-buffer
  u32 visibility buffer (4 bits cluster-batch + 28 bits
  cluster_id + tri_id packed). **Test:** packing +
  unpacking round-trips on 1000 random ids.
- [ ] **[RT-micropoly/sw-raster]** WGSL compute shader.
  One thread per triangle in a SMALL cluster. Edge-function
  inside-test, atomic-min depth write, conditional
  visibility write. **Test:** render a single hand-placed
  triangle, validate the visibility buffer contains the
  expected cluster_id + tri_id at the expected pixels.
- [ ] **[RT-micropoly/sw-stress]** Render the
  `examples/gen_micropoly_stress.rs` scene through the SW
  path only (LARGE clusters disabled). **Test:** no
  visible holes (no `vis == 0` pixels where geometry should
  be present); frame time ≤ 33 ms native at 1280×720.
- [ ] **[RT-micropoly/hw-vis-fragment]** Modify the existing
  R4 forward pipeline's fragment shader: discard colour
  write, perform the atomic-min depth + visibility write
  instead. **Test:** rendered visibility buffer matches the
  SW path's output on a scene with identical geometry
  classified as both SMALL and LARGE under different camera
  configs (LARGE at close camera, SMALL at far camera).
- [ ] **[RT-micropoly/resolve]** Full-screen resolve fragment
  shader: visibility-id → cluster + tri → shaded pixel.
  **Test:** end-to-end render of the stress scene visually
  matches the existing R4 pipeline's render of the same
  scene (RMSE ≤ 0.05 on the tone-mapped image).
- [ ] **[RT-micropoly/hybrid-stress]** Render the stress scene
  through the hybrid pipeline (SW + HW classification). **Numeric
  Done-when:**
  * Frame time ≤ 16 ms native at 1280×720.
  * Frame time ≤ 50 ms in Apple M-series Safari at the same
    resolution.
  * Visible-triangle equivalence: the resolved image has
    > 99% of pixels matching the pure-HW R4 reference image
    within ΔE < 5 (perceptual colour distance).
- [ ] **[RT-micropoly/motum-noregression]** Motum's existing
  scenes go through the LARGE classifier exclusively (their
  triangles are not sub-pixel). Render output must match
  pre-plan within RMSE ≤ 0.05. Visibility-buffer-then-resolve
  path adds ~one fragment-shader pass per frame; latency
  measured and reported in `Findings`.

## Done when

* All nine milestones ticked
* `Findings` contains numeric frame-time table for the stress
  scene on M-series native + Safari, plus the motum-latency
  delta
* `examples/gen_micropoly_stress.rs` committed and produces
  a reproducible stress-scene render
* Visibility-buffer ↔ resolve architecture diagram in
  `Findings`
* README features list gains "Software rasterizer for sub-
  pixel triangles (RT-micropoly)" under runtime
* Plan moves to `Status: completed`

## Findings

(Populated during execution.)

## Followups (out of scope)

* **RT-micropoly-64bit** — when WebGPU adds 64-bit atomic
  storage-buffer ops (or `R64Uint` storage textures), replace
  the two-pass depth + vis architecture with the single-
  atomic Nanite-original design. Saves one full-screen pass.
* **RT-micropoly-subgroups** — use subgroup ops to coalesce
  the small-cluster compute dispatch (Karis 2021 §5.2:
  "wave-32 cluster batching"). 1.3-1.5× SW raster speedup
  reported; gated on uniform WebGPU subgroup support
  (Safari behind a flag as of writing).
* **RT-micropoly-msaa** — anti-aliasing for SW raster. The
  hardware raster path gets MSAA for free; the SW path
  needs explicit super-sampling or some kind of analytic
  edge AA (Toth 2019). Significant scope.
* **RT-virtualized** (plan 0034) — LOD selection that makes
  sub-pixel triangles the common case. Without virtualized
  geometry the classifier rarely hits SMALL on real scenes.
