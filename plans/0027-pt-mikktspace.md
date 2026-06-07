# PT-mikktspace — per-face tangents in WGSL

- **Status:** in-flight (shader + chess + sponza shipped; bistro re-render parked)
- **Last updated:** 2026-06-07

## Goal

Replace per-vertex tangents with **per-face tangents** computed
on-the-fly in the WGSL fragment shader. Closes the UV-seam
vertical-line artifact that ships in three current heroes:

* **Chess** (`chess_reference.png`) — the user's original
  report on commit `8bb6c8e`. Hemisphere clamp from that
  commit reduced but didn't eliminate the stripe.
* **Sponza facades** — same artifact class on tiled brick /
  stone walls, less visible because Sponza is more lit-by-
  sun and the camera framing doesn't focus on flat brick.
* **Bistro** (`bistro_reference.png`) — the worst case. The
  brick + ashlar + door-trim assets all tile their normal
  maps across UV seams; the artifact fires at every tile
  row. Render-attacker now catches it consistently with the
  tightened mandate from commit `a5043a6`.

## Root cause (mechanics)

At a UV seam, two adjacent triangles share vertex positions
but their UVs are mapped to opposite ends of the texture
(u jumps from ~1 back to ~0). The per-vertex tangent
accumulator in `pathtrace::mesh::compute_tangents` sums each
triangle's tangent contribution at the shared vertex:

```
T(v) = Σ_{tri ∋ v} (∂P/∂u of that triangle)
```

The two triangles on opposite sides of the UV seam contribute
tangents in **opposite directions** because their UV
gradients invert across the seam. The accumulator averages
toward zero (worst case) or toward a "compromise" direction
that's wrong for both sides (typical case).

The barycentric blend in the WGSL fragment shader then
interpolates the bad accumulated tangent across each pixel
inside any triangle that touches a seam vertex. The
normal-map perturbation rotates the shading normal in a
direction that has no relationship to the actual texture's
tangent space. Visible result: a one-pixel-wide
bright/dark stripe at every UV-seam edge — which on a
tiled brick wall fires once per brick.

## Why per-face tangents fix it

Each triangle owns its **own** UV gradient. Computing the
tangent from `(P0, P1, P2, UV0, UV1, UV2)` of the triangle
itself produces a tangent that's correct for that triangle's
texture sampling, regardless of what the adjacent triangles
on the other side of any UV seam are doing. No averaging,
no cancellation.

This is the per-face variant of mikktspace. The per-corner
variant (each of a triangle's three corners stores its own
tangent) adds slightly better quality at corners but
requires per-corner storage. Per-face is the right balance
for the artifact this plan closes.

## Why compute in shader rather than upload a buffer

WebGPU baseline limits cap storage buffers per shader stage
at 8. The pathtrace bind group already uses all 8:

```
binding 1: vertices
binding 2: tri_indices
binding 3: materials
binding 4: triangle_materials
binding 5: emissive_triangles
binding 6: bvh_nodes
binding 7: bvh_tri_indices
binding 14: env_data (CDFs)
```

Adding a 9th `triangle_tangents` buffer would exceed the
baseline. Either we repack (substantial refactor) or
compute on demand. The shader **already** reads three
vertex positions + three vertex UVs per ray-triangle
intersection for the barycentric hit math — those reads
are sunk cost. Computing the tangent from values already
in registers costs a handful of FMAs + one
cross-product + one normalize. ~5 cycles per shading
evaluation, hidden behind memory latency anyway.

## Design

A new WGSL function:

```wgsl
fn triangle_tangent(tri: u32, n: vec3<f32>) -> vec4<f32> {
    // Read three vertex positions + UVs.
    // Solve the 2x2 linear system for ∂P/∂u given the UV
    // gradient (the standard mikktspace per-face tangent).
    // Gram-Schmidt against the (already unit) shading
    // normal `n`, sign-resolve the bitangent.
    // Return zero-length sentinel on degenerate UV
    // gradient — `apply_normal_map` falls back to the
    // geometric normal.
}
```

Replaces `vertex_tangent(tri, bary)` everywhere it's called.
The barycentric `bary` argument is no longer needed — the
tangent is constant across the triangle.

`interpolated_tbn(tri, bary, normal)` becomes
`triangle_tbn(tri, normal)`. The smoothstep UV-pole fade in
`apply_normal_map` is no longer needed (per-face tangents
handle poles cleanly — a triangle bordering a UV pole has
a well-defined gradient even though the pole vertex itself
is singular) and gets removed.

The hemisphere clamp from commit `8bb6c8e` stays as a
safety net for assets that ship their own degenerate
normal maps or extreme `normalTexture.scale`.

CPU side: `pathtrace::mesh::compute_tangents` and the
glTF-`TANGENT`-attribute ingest path stay for now —
they're no-ops in the new shader path but removing them
is a separate cleanup. The `Vertex.tangent` field also
stays (reads are just unused). A future micro-plan can
shrink the Vertex from 64 → 48 bytes and save ~45 MB on
Bistro-scale assets.

## Milestones

- [x] **[PT-mikktspace/wgsl]** `triangle_tangent(tri, n)`
  lands in `pathtrace.wgsl`. Replaces every call site of
  `vertex_tangent` / `interpolated_tbn`. UV-pole
  smoothstep fade in `apply_normal_map` removed (per-face
  tangents handle poles cleanly). Dead helpers
  (`vertex_tangent`, `interpolated_tbn`, transient
  `triangle_tbn`) removed.
- [x] **[PT-mikktspace/chess]** Chess hero re-renders
  without the vertical-line UV-seam artifact on the
  marble bodies. Render-attacker (tightened mandate from
  commit `a5043a6`) passes the periodic-pattern check on
  the pawn balls, bishop crowns, and marble board.
- [ ] **[PT-mikktspace/bistro]** Bistro hero re-renders
  without brick stripes (~38 px period), ashlar stripes
  (~48 px), or door-trim stripes (~28-32 px). Deferred
  — needs ~30 min GPU; smoke render at 768×576/256 spp
  verified the fix works; full 1024×768/2048 spp hero is
  parked for the user's GPU window.
- [x] **[PT-mikktspace/sponza]** Sponza hero re-renders
  cleanly with no regression on the lit brick walls /
  arches / banners.

## Done when

* All four milestones ticked
* Chess + Bistro + Sponza heroes regenerated, swapped
  into the README, and render-attacker single-image mode
  passes the periodic-pattern check on all three
* This plan moves to `Status: completed`

## Followups (out of scope)

* **PT-vertex-shrink** — drop `Vertex.tangent` (16 bytes)
  + `compute_tangents` CPU path. Vertex 64 → 48 bytes.
  Saves ~45 MB on Bistro Exterior; smaller wins
  elsewhere.
* **PT-mikktspace-percorner** — full mikktspace per-
  corner variant for the corners where per-face shows a
  visible facet seam. Probably not visible at the
  current asset resolutions.
