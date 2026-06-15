# RT-virtualized — full virtualized geometry (cluster DAG + LOD + page streaming)

- **Status:** draft
- **Last updated:** 2026-06-15
- **Last touched on:** initial draft from the Nanite-direction scoping note

## Goal

Add the **virtualized-geometry** track to Quasi's raster
pipeline: a cluster **DAG** with multi-LOD selection per
cluster, a page-based GPU geometry cache, and a streaming
loader that brings pages in on demand. With this in place, a
scene with hundreds of millions of source triangles can render
at interactive rates, because only the LOD-selected clusters
ever touch GPU memory and the sub-pixel triangles among them
get the RT-micropoly software-rasterizer treatment.

This is the **tier 3** Nanite piece — the complete picture.
It is the **largest plan in the renderer's history** and the
one that absolutely cannot ship without RT-cluster-cull (0032)
and RT-micropoly (0033) underneath it.

## Prerequisites (both must be shipped, not just drafted)

* **RT-cluster-cull (0032)** — GPU-driven culling pipeline
  + indirect-draw substrate. This plan extends the cull pass
  with LOD selection and the cluster DAG; without 0032 there
  is no cull pass to extend.
* **RT-micropoly (0033)** — sub-pixel raster. The LOD
  selection in this plan deliberately targets clusters whose
  triangles project to ≤ 1 px; without 0033 the small-LOD
  clusters render slowly via the HW path, defeating much of
  the point.

If RT-micropoly is *drafted but not shipped*, this plan can
land its LOD + streaming pieces and degrade gracefully to
HW-only raster — but the perf story is materially worse and
the plan's Done-when criteria account for that case
explicitly.

## Why this plan is the biggest one in the renderer's history

Nanite is one of the most significant rendering papers of the
last decade (Karis et al., SIGGRAPH 2021, "A Deep Dive into
Nanite Virtualized Geometry," 31 pages). Reproducing its
core ideas — even at the small subset that maps to WebGPU —
is a real research-engineering project. The plan is **honest
about scope**:

* 11 milestones, the longest in any Quasi plan.
* Expected wall-clock implementation time: months, not weeks.
* Multiple correctness traps known in advance (DAG
  consistency, cluster-boundary crack avoidance, page-cache
  thrashing, streaming-latency hiding).

Why we ship the plan anyway, given the scope:

1. The track has a **clear blog series ahead of it** (cluster
   DAG construction is its own article; LOD selection is
   another; streaming is a third) — high publication-value
   density per implementation week.
2. The architecture has now been **published, dissected, and
   reproduced enough times** that there's actionable
   reference material at every step. We're not breaking new
   ground; we're porting known-good ideas to a new substrate.
3. The path tracer track has matured to "real production
   scenes work" (Bistro, Sponza). The raster track being
   stuck at "small motum scenes work" is a credibility gap
   that this plan closes.

If after RT-cluster-cull + RT-micropoly land the appetite for
this plan changes, deferring it is acceptable.

## What "Nanite-like, but in WebGPU" means here

We **do** ship:
* Cluster DAG with multi-LOD per cluster
* Screen-space-error LOD selection per cluster per frame
* Page-based GPU geometry cache
* Streaming loader (CPU-side, blocking IO for v1)
* Hierarchical culling (page → cluster → triangle)

We **do not** ship (deferred or out of scope):
* Bindless materials (WebGPU gap)
* 64-bit visibility-buffer atomic ops (handled in 0033 via
  the two-pass workaround; inherited here)
* Streaming over network for the wasm target — v1 ships
  with all pages bundled in the wasm binary; HTTP streaming
  is a separate plan (RT-virt-http-stream)
* Material LOD or texture virtualization
* Dynamic geometry (only static meshes)

## Architectural decisions, signed off up front

These four decisions are the load-bearing ones. The
plan-skeptic should focus its attack here.

### Cluster DAG construction algorithm

We implement the **Karis 2021 §3.2 "group-and-simplify"
algorithm** with explicit boundary preservation:

1. Start from the leaf-LOD clusters from RT-cluster-cull
   (Morton-window approximation; or, if RT-cluster-build-metis
   has shipped by then, the better partitioner).
2. Group **4 adjacent clusters** by minimizing inter-group
   edge weight (METIS-style edge cut). Each group becomes
   one cluster at the next-coarser LOD.
3. **Simplify the merged geometry** while pinning boundary
   vertices that are shared with neighbouring groups (this is
   the crack-avoidance constraint). Target tri count:
   half the source group's tri count.
4. Iterate. Stop when a level has ≤ 1 cluster or simplification
   stops reducing tri count.

This produces a DAG (not a tree): the boundary vertices'
positions are shared upward, so a coarse-LOD cluster's
boundary matches its fine-LOD descendants' boundary exactly.
This is what eliminates inter-LOD cracks.

**Citation:** Karis 2021 §3.2; also Mark Lee 2022, "Open
Source Nanite-style virtualised geometry" (Bevy engine
prototype, MIT-licensed) for an implementation reference.

### Screen-space error metric

Per cluster per frame:

```
projected_edge_error = world_space_simplification_error * focal_length / distance_to_camera
```

`world_space_simplification_error` is computed once at DAG
build time as the max vertex-distance between this cluster
and its parent-cluster's representation. `focal_length` is in
pixels.

LOD selection: traverse the DAG from the root. At each
cluster, if `projected_edge_error <= 1.0 px`, use this
cluster's geometry. Otherwise descend to children.

This guarantees the rendered geometry's screen-space
deviation from the source mesh is **bounded by 1 pixel**.

### Page layout + size

* **128 KB pages.** Big enough to amortise the per-page
  metadata overhead; small enough that the LRU eviction has
  reasonable granularity. Karis 2021 uses 128 KB; we
  inherit.
* Each page contains a flat array of clusters' vertex +
  index data. Cluster metadata (bounds, normal cone, parent
  cluster id) lives in a separate "cluster table" that's
  always GPU-resident.
* Page contents are positionally encoded with a 16-bit
  quantised offset relative to the cluster bounding sphere
  centre. Compression ~3.5× vs raw 32-bit floats.

### GPU page cache

* Fixed-size GPU storage buffer: **128 MB by default**
  (~1000 pages). Configurable via CLI / scene config.
* LRU eviction: each page has a "last frame referenced"
  timestamp; new pages evict the oldest-stamped resident
  page.
* When the cull pass selects a cluster whose page isn't
  resident, the cluster gets **skipped this frame** and the
  page is added to a request queue. CPU loads the page; it
  becomes available next frame. This is the "one frame late"
  artifact; mitigated by predictive prefetch (Karis §7) as
  a follow-up.

## Crack avoidance (the most important correctness trap)

The fundamental hazard: a cluster rendered at LOD N adjacent
to a cluster rendered at LOD N+1 (coarser). Their boundary
vertices must match exactly, or a one-pixel crack appears
along the boundary.

The DAG construction **pins boundary vertices** during
simplification (the "shared boundary" property of the DAG —
Karis §3.2). The boundary vertex positions are bit-identical
between a cluster and its parent. So **as long as both
clusters use the same DAG-level boundary**, no crack.

This is the load-bearing reason the cluster representation
isn't a tree of independent simplifications: it's a DAG
where boundary vertices are shared. The plan-skeptic should
attack the **implementation discipline** that maintains this
invariant — the simplifier needs to be specifically aware of
which vertices it cannot touch.

## Streaming loader

v1 ships with all pages bundled in the executable / wasm
binary. Async loading is from in-memory bytes (still adds a
one-frame latency due to the GPU upload going through
`queue.write_buffer` on a separate command). Real-time disk
or network IO is a follow-up.

The async path:

* Page request queue (per frame, drained by the loader).
* CPU loader has a budget of **8 pages per frame** (≤ 1 MB
  upload); throttled to keep the per-frame upload below the
  PCIe / browser-WebGPU bandwidth budget.
* New pages get a "frame N + 1" timestamp; available to
  cull next frame.

## Native + web lockstep

Streaming over file IO doesn't exist on wasm; the v1 bundles
everything. Native gets the same in-memory bundle for
simplicity. Both targets exercise the same loader code path,
just with `&'static [u8]` page sources instead of disk IO. The
[`feedback_native_web_lockstep`](../memory/feedback_native_web_lockstep.md)
rule is preserved.

## Milestones

- [ ] **[RT-virt/dag-cpu]** CPU-side cluster DAG builder in
  `src/raster/dag.rs`. Karis §3.2 group-and-simplify
  algorithm with boundary pinning. **CPU unit tests:**
  * On a known mesh, every leaf cluster has exactly one
    parent at LOD+1.
  * Boundary vertex positions are bit-identical between a
    cluster and its parent.
  * Total cluster count at LOD K halves between consecutive
    K's (within tolerance: simplification stops when it
    can't halve).
- [ ] **[RT-virt/error-metric]** Per-cluster
  `world_space_simplification_error` computed at DAG-build
  time. Stored in the cluster table. **Test:** for a
  hand-crafted DAG, the error metric matches a hand-
  computed reference for 10 known clusters.
- [ ] **[RT-virt/pages]** Page builder that lays out
  clusters' geometry into 128 KB pages with positional
  quantisation. **Test:** round-trip decode of a known
  page recovers the source geometry within the
  quantisation bound (max vertex error ≤ 1e-4 of
  bounding-sphere radius).
- [ ] **[RT-virt/cache]** GPU page cache: fixed-size storage
  buffer, LRU eviction tracked CPU-side with per-page
  timestamps. New `RasterPageCache` struct. **Test:** under
  a deterministic page-access pattern (a known list of page
  ids), eviction order matches the expected LRU order.
- [ ] **[RT-virt/loader]** CPU loader. Page request queue,
  drained at most 8 pages/frame. v1 sources pages from
  `&'static [u8]` (in-memory bundle). **Test:** request a
  page, frame N+1 finds it resident.
- [ ] **[RT-virt/lod-cull]** Extend RT-cluster-cull's cull
  compute pass with DAG traversal + LOD selection. Per
  cluster: compute screen-space error; if > 1 px, descend
  to children; if ≤ 1 px, use this cluster. **Test:** on a
  scene with a moving camera, the rendered cluster set at
  each frame matches a CPU-reference traversal of the same
  DAG.
- [ ] **[RT-virt/crack-test]** A dedicated test for the
  crack-avoidance invariant. A small mesh with a known
  cluster boundary; force the test camera to a position
  where one side renders at LOD K and the other at LOD K+1;
  assert the rendered image has no pixels of background
  colour along the boundary line.
- [ ] **[RT-virt/missing-page-skip]** Cluster selection
  encounters a non-resident page → cluster skipped, page
  enqueued, frame finishes. **Test:** scripted camera
  motion that triggers ~10 missing pages per frame; image
  is "mostly complete with sparse holes" the first
  problematic frame, fully complete the next, no crashes.
- [ ] **[RT-virt/stress-asset]** A dedicated stress asset.
  Most realistic option: the existing Bistro Exterior
  scene (~ 2.8 M source triangles after our PT-bistro
  ingest) rendered via the virtualized pipeline at a far
  camera viewpoint where ~80% of clusters are at sub-pixel
  LOD. **Numeric Done-when:**
  * Frame time ≤ 33 ms native at 1280×720.
  * Frame time ≤ 100 ms in Safari (LOD makes streaming the
    bottleneck — this is the wasm-budget reality).
  * Visible cluster count per frame ≤ 5% of total DAG
    cluster count.
  * Page cache hit rate after the first second of motion:
    ≥ 95%.
- [ ] **[RT-virt/sw-raster-integration]** With RT-micropoly
  shipped, the sub-pixel clusters route through the SW
  raster path. Frame time on the same stress asset drops to
  ≤ 20 ms native. (If RT-micropoly has not shipped at this
  plan's close-time, this milestone is skipped; Done-when
  is amended in `Findings`.)
- [ ] **[RT-virt/blog-asset]** A reproducible 2-3 minute
  demo: camera flythrough of the Bistro stress asset
  showing LOD transitions, with a frame-time HUD overlay
  and a cluster-count overlay. Lands as
  `examples/gen_virtualized_demo.rs` + a PNG / GIF / MP4
  in `data/output/`. The blog post can lift this directly.

## Done when

* All 11 milestones ticked (or 10 if RT-micropoly never
  shipped and the sw-raster-integration milestone is skipped)
* `Findings` contains the full frame-time table on the Bistro
  stress asset, plus the DAG-build statistics (cluster
  count per LOD, simplification ratio, build time, page
  count)
* Crack-avoidance test green
* `examples/gen_virtualized_demo.rs` runs cleanly native +
  wasm
* README features list gains "Virtualized geometry track
  (RT-virtualized): cluster DAG + LOD + page streaming"
* Plan moves to `Status: completed`

## Findings

(Populated during execution.)

## Followups (out of scope)

* **RT-virt-http-stream** — page streaming over HTTP for the
  wasm target. With this, the v1's "bundle everything in the
  wasm binary" can be replaced by on-demand page fetching.
* **RT-virt-prefetch** — Karis §7 predictive prefetch: from
  the current camera + velocity, predict which pages will be
  needed next frame and pre-issue their loads. Hides the
  one-frame-late artifact in normal camera motion.
* **RT-virt-disk-cache** — for native targets, a disk-backed
  page cache (mmap'd file) so the cold-start cost is paid
  once instead of every launch.
* **RT-virt-materials-vt** — virtual texturing for materials.
  Pair with RT-virtualized so material data also streams.
  Significant scope; own plan.
* **RT-virt-multilod-blend** — temporal cross-fade between
  LOD levels when a cluster transitions across the
  1-pixel boundary, to eliminate the "snap" some viewers
  notice. Often considered cosmetic; deferred until measured.
* **RT-virt-dynamic-geom** — extending to dynamic (skinned,
  animated) geometry. Different DAG construction strategy;
  separate plan.
* **RT-virt-bindless-when-ready** — when WebGPU adopts a
  bindless extension, swap the texture-array-with-indices
  workaround for true bindless.
