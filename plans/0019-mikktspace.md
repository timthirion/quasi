# Per-vertex tangents (PT-mikktspace)

- **Status:** completed
- **Last updated:** 2026-06-06
- **Last touched on:** vertex tangent + bumpy bunny showcase

## Goal

Grow the path tracer's `Vertex` by 16 bytes for a stored
**per-vertex tangent** so normal-mapped meshes get **smooth
shading across triangle seams**. Plan 0015 deliberately
deferred this and computed tangents per-hit from triangle UV
deltas, which works for the 2-triangle stone-tile floor but
would alias visibly on the bunny if the bunny ever wore a
normal map directly. Now that the bunny is our **core showcase
asset** across the gallery (its cylindrical UVs land in plan
0019b), per-vertex tangents become load-bearing for the next
round of bunny-centric scenes.

The plan name "PT-mikktspace" is aspirational — Mikktspace is
the gold-standard tangent-frame algorithm and would slot in
cleanly here later. For now we use the simpler tangent-from-UV-
delta path (the same formula `pathtrace::mesh::compute_tangents`
already implements + tests). Mikktspace itself becomes a
sub-milestone follow-up.

## Context

What's already in:

* `pathtrace::mesh::compute_tangents(positions, normals, uvs,
  indices)` returns per-vertex vec4 tangents (xyz = direction,
  w = bitangent sign per the glTF 2.0 convention). Tested
  against axis-aligned quad, flipped-V quad, degenerate UV.
  Currently unused by the loader — exposed as an internal
  utility.
* `pathtrace::mesh::orthonormalize_tangent` + `transform_dir`
  + `normalize_or_zero` helpers backing the computation.
* WGSL `triangle_tangent_frame(tri, normal)` derives the TBN
  per hit from triangle position + UV deltas. Replaceable.
* glTF ingest reads `POSITION`, `NORMAL`, `TEXCOORD_0` — no
  TANGENT yet.

What this plan is **not**:

* Mikktspace per se. The Mikktspace reference algorithm
  carries specific weighting + smoothing rules across
  triangle seams that produce gold-standard tangents but at
  significant implementation overhead. We use the simpler
  per-vertex average + Gram-Schmidt route (what we already
  test). Closing that gap is a future plan.
* Per-vertex bitangent storage. The .w bitangent sign is one
  byte of information; storing the bitangent itself would
  duplicate the cross product the WGSL does for free.
* Compressed tangents (Oct-encoding). The 16-byte vec4
  spelled out plainly is what's pinned by the layout test.

## Design

### Vertex grows the tangent

```rust
pub struct Vertex {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub normal: [f32; 3],
    pub _pad1: f32,
    pub uv: [f32; 2],
    pub _pad2: [f32; 2],
    pub tangent: [f32; 4],   // NEW — xyz dir, w bitangent sign
}
```

48 → 64 bytes. Layout test updated. Default tangent (no map)
is `(1, 0, 0, 1)` — costs no extra branches but stays well-
defined.

### glTF ingest reads TANGENT or derives

```rust
let tangents_local: Vec<[f32; 4]> = match reader.read_tangents() {
    Some(read) => read.collect(),
    None => compute_tangents(&positions, &normals, &uvs, &indices),
};
```

Then transform each xyz by the world matrix's upper-3x3 (plain
direction transform — not the cofactor used for normals). Re-
orthogonalise against the smoothed normal so the per-vertex
TBN stays unit + perpendicular.

### WGSL: interpolate tangent at hit, build TBN

Replace the per-hit `triangle_tangent_frame` with a barycentric
interpolation of the three vertex tangents, Gram-Schmidted
against the (geometric) normal:

```wgsl
fn vertex_tangent(tri: u32, u: f32, v: f32) -> vec4<f32> {
    let i0 = tri_indices[tri * 3u + 0u];
    let i1 = tri_indices[tri * 3u + 1u];
    let i2 = tri_indices[tri * 3u + 2u];
    let w = 1.0 - u - v;
    let t = vertices[i0].tangent * w
          + vertices[i1].tangent * u
          + vertices[i2].tangent * v;
    return vec4<f32>(normalize(t.xyz), sign(t.w));
}
```

This needs the barycentric coords at the hit. `record_hit`
already stashes `(u, v)` indirectly via `triangle_uv(tri, u,
v)`; we extend `Hit` with a `bary: vec2<f32>` field so the
shading-normal code can re-read them.

### Bunny showcase

The bunny under `cornell_normal_mapped.gltf` currently uses
the brushed-brass MR map but NOT a normal map (the normal map
goes on the floor). Bake a procedural multi-octave-fbm normal
map (low amplitude — just enough to read as "hammered metal"
under the cylindrical wrap), and add it to the bunny material
in a new scene `cornell_bumpy_bunny.gltf`. With per-vertex tangents in place
the normal-map perturbation reads smoothly across the bunny
silhouette rather than aliasing at triangle edges.

## Milestones

### PT-vertex-tangent
- [x] `Vertex` grows to 64 bytes; layout test asserts new size +
      tangent offset at 48.
- [x] `mesh.rs` ingest reads `TANGENT` when present, otherwise
      calls `compute_tangents`. Per-vertex tangent is
      world-transformed + Gram-Schmidted against the smoothed
      normal at upload time.
- [x] WGSL `Vertex` mirror gains `tangent: vec4<f32>`.
- [x] WGSL `Hit` gains `bary: vec2<f32>`; `record_hit` stores
      the barycentric `(u, v)`.
- [x] WGSL `vertex_tangent` + `interpolated_tbn` barycentrically
      blends the three per-vertex tangents and Gram-Schmidts
      against the geometric normal. Old per-hit
      `triangle_tangent_frame` deleted.
- [x] Existing PT-normal-map scenes (`cornell_normal_mapped`,
      `outdoor_normal_bunny`) render correctly; the 2-triangle
      stone-tile floor collapses to the same TBN under both
      paths.

### PT-bumpy-bunny — showcase
- [x] `examples/gen_pbr_maps` bakes a low-amplitude bunny-bumpy
      normal map (multi-octave fbm) into
      `data/textures/bunny_bumpy_normal.png`. Mean Z = 0.87.
- [x] `cornell_bumpy_bunny.gltf`: Cornell room with the
      brushed-brass bunny material gaining the bumpy normal map.
- [x] Reference at 768²/2048 spp →
      `data/output/cornell_bumpy_bunny_reference.png`. Visible
      bumpy texture reads smoothly across the bunny silhouette.

## Open questions

- **`bary` storage on `Hit`.** Eight extra bytes on the Hit
  struct only used by the normal-map path; could derive them
  again from `hit.uv` and per-vertex UVs, but that's a div on
  every shading lookup. The 8 bytes are cheaper.
- **Mikktspace gap.** Our derived tangents work but Mikktspace
  is the asset-pipeline standard. We'd see the difference if
  a DCC-authored normal map were used with our tangents vs
  with Mikktspace tangents — the bake / runtime would
  disagree about the tangent frame. For procedural maps we
  bake, the consumer + producer use the same maths so the
  disagreement is moot.
- **Per-vertex tangent on flat geometry.** A quad (floor) gets
  one tangent per vertex from the UV deltas — those are
  constant across the quad, so the barycentric blend collapses
  to the constant. Tests this is the case.
- **Surface UVs as a sampling prior, not just a texturing one.**
  Per-vertex UVs (and the parameterizations morsel can compute)
  are stored as decoration today. The hypothesis that they're
  also valuable as an *importance-sampling* substrate sits in
  [`research/R0002-param-driven-sampling.md`](research/R0002-param-driven-sampling.md).

## Done when

- The brushed-brass bunny under a fresh normal map (the bumpy
  scene) reads as smoothly-shaded across the body without
  triangle-edge aliasing in the perturbed normal.
- All existing scenes still render correctly.
- Naga, native cargo test, fmt, clippy, wasm32 `cargo check`,
  Python unittests, CI, Pages-deploy all stay green at HEAD.
