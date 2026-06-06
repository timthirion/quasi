# PBR maps — normal / roughness / metallic (PT-pbr-maps)

- **Status:** draft
- **Last updated:** 2026-06-06
- **Last touched on:** planning

## Goal

Lift the path tracer from "per-material constant" PBR to **per-pixel
PBR**. Today every material carries a single `roughness`,
`metallic`, and geometric normal; this plan adds the three texture
maps glTF 2.0 calls out (normal, metallic-roughness, optionally
emissive) and threads them through the same WGSL BSDF code without
disturbing the analytic path tracers.

Pairs naturally with PT-textures (already in — baseColor only) and
with PT-env (just landed — HDR illumination is what makes normal
maps actually shine). Normal-mapped GGX surfaces under an HDR sky
are the cheapest 10× visual upgrade we can ship to the blog gallery.

## Context

What's already in:

- `material_albedo(m, hit.uv)` in `pathtrace.wgsl` does the
  baseColor texture sample. `Material::base_color_texture_idx`
  drives it; `NO_TEXTURE = 0xFFFFFFFF` is the "use scalar"
  sentinel.
- A single `texture_2d_array<f32>` at `@binding(8)` holds all the
  textures across the scene. Layers are uniformly sized (the glTF
  ingest path resizes per-layer). Bind-group budget: we sit at 8
  storage buffers (cap) + 3 sampled textures + 3 samplers; adding
  normal/MR maps **does not** consume new bindings — they slot in
  as extra layers of the existing array.
- `Vertex` is 48 bytes (`position: vec3` + `_pad: f32` + `uv: vec2`
  + more pad to align). UVs are read from glTF `TEXCOORD_0`;
  missing UVs default to (0, 0).
- `Material` is 96 bytes (PT-hg final size). 32-byte alignment;
  growing it stays a `repr(C)` mirror against the WGSL struct.
- GGX BSDF (`pathtrace/ggx.rs` + WGSL `bsdf_ggx_*`) reads
  `m.roughness` + `m.metallic` directly. Dielectric BSDF (`m.ior >
  0`) ignores both.

What this plan is **not**:

- Mikktspace tangents — out of scope; we derive tangents from
  position + UV at glTF load (simple cross-product formula). Fine
  for our use cases (smooth meshes + uniform UV stretching).
  Adding a Mikktspace implementation later doesn't change any
  call sites.
- Per-texture UV channels — glTF allows it; we assume TEXCOORD_0
  for all maps and error if a material asks for a different
  channel.
- Bump / displacement maps. The displacement story is the same
  trap-door that DCC tools have always struggled with; skip it.
- Occlusion maps — minimal payoff under HDR sky; the multi-bounce
  GI already approximates it.

## Design

### Material grows two texture indices

```rust
pub struct Material {
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub emission: [f32; 3],
    pub metallic: f32,
    pub absorption: [f32; 3],
    pub ior: f32,
    pub scattering: [f32; 3],
    pub phase_g: f32,
    pub cloud_center: [f32; 3],
    pub cloud_radius: f32,
    pub base_color_texture_idx: u32,
    pub metallic_roughness_texture_idx: u32,  // NEW
    pub normal_texture_idx: u32,              // NEW
    pub _pad: u32,                            // NEW (alignment)
}
```

Bumps `Material` 96 → 112 bytes. `gpu_struct_layout_matches_wgsl`
guards the new offsets.

`metallic_roughness_texture_idx` follows the glTF 2.0 convention:
the texture's **G channel is roughness, B channel is metallic**.
R is unused (sometimes occlusion in some asset pipelines — we
ignore it). `material_pbr_at_hit(m, uv)` returns the effective
scalars by `m.roughness * texel.g`, `m.metallic * texel.b`. When
the index is `NO_TEXTURE`, the scalars pass through.

### Vertex grows a tangent

```rust
pub struct Vertex {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub uv: [f32; 2],
    pub _pad1: [f32; 2],
    pub tangent: [f32; 4],   // NEW: xyz = tangent dir, w = bitangent sign
}
```

48 → 64 bytes. The bitangent comes from `cross(normal, tangent.xyz) *
tangent.w` per the glTF spec. `tangent` is computed at glTF load
from position + UV deltas across each triangle, then averaged per
vertex and re-orthogonalised against the smooth normal (Gram-
Schmidt).

When a mesh has explicit glTF tangents in `TANGENT` we use those.
When it doesn't, we derive them. CPU code lives in
`pathtrace::mesh::compute_tangents` with the same shape as
`compute_smooth_normals`.

### WGSL: normal mapping at the hit

```wgsl
fn apply_normal_map(
    m: Material,
    hit: Hit,
    tangent: vec3<f32>,
    bitangent_sign: f32,
) -> vec3<f32> {
    if (m.normal_texture_idx == NO_TEXTURE) {
        return hit.normal;
    }
    let n_tex = textureSampleLevel(
        textures, sampler_obj, hit.uv, f32(m.normal_texture_idx), 0.0
    ).rgb;
    let n_ts = normalize(n_tex * 2.0 - 1.0);          // [0,1] → [-1,1]
    let n_w  = normalize(hit.normal);
    let t_w  = normalize(tangent - n_w * dot(tangent, n_w));
    let b_w  = cross(n_w, t_w) * bitangent_sign;
    return normalize(t_w * n_ts.x + b_w * n_ts.y + n_w * n_ts.z);
}
```

Called once per hit, before any BSDF evaluation. Returns the
**shading normal**; the geometric normal stays available on
`Hit.geom_normal` for self-intersection offset (avoids "smooth
normal points into the surface" artefacts).

### glTF ingest

`mesh.rs::load_glb_bytes` already pulls baseColor; mirror the same
pattern for the other two:

- `material.pbr_metallic_roughness().metallic_roughness_texture()`
  → push into the texture array, store the layer index.
- `material.normal_texture()` → ditto. (Normal textures have a
  `scale: f32` we honour as a per-material parameter — we fold
  it into the tangent-space sampling step.)

The texture-array sizing step (currently picks max width × max
height across baseColor textures) needs to consider the new maps
too. Resizing happens at load; no runtime cost.

### Scenes

A new `cornell_normal_mapped.gltf`: the bunny scene with a brushed-
brass material on the bunny (metallic-roughness map for the
brushed streak + low-frequency dirt) and a stone-tile normal map
on the floor (random tile bumps + pebble noise). Under the
existing area-light Cornell Box, then again under env lighting via
`outdoor_normal_bunny.gltf` (floor + brass bunny, no walls).

Procedurally bake both maps in `examples/gen_pbr_maps.rs`: each
~~256² R8G8B8A8 PNG, deterministic, no external assets. Real DCC
textures can replace them later; the procedural ones keep the repo
self-contained.

## Milestones

### PT-mr-map
- [x] `Material` grows `metallic_roughness_texture_idx`; layout
      test updated. `NO_TEXTURE` sentinel reused. (Slots into the
      existing `_pad` field, so total size stays at 96 bytes.)
- [x] glTF ingest reads `pbrMetallicRoughness.metallicRoughnessTexture`
      and pushes it into the texture array. Same UV channel as
      baseColor (TEXCOORD_0).
- [x] WGSL `material_metallic_roughness(m, uv)` returns effective
      `(roughness, metallic)` after the texture multiply. Folded
      into the local Material copy in `path_trace` so every
      downstream BSDF call reads the effective values without
      changes.
- [x] `examples/gen_pbr_maps` bakes a brushed-metal MR map (256²,
      multi-octave anisotropic streak noise + dirt + tarnish) into
      `data/textures/brushed_brass_mr.png`.
- [x] `cornell_metal_bunny.gltf` regenerates with the brushed
      bunny material (brass F0 = (0.95, 0.78, 0.46)) referencing
      the MR map via planar XZ UVs. New reference at
      `data/output/cornell_metal_bunny_reference.png`.
- [x] CPU mirror tests: `Material::effective_metallic_roughness`
      passes scalars through with no texture, multiplies G/B
      channels correctly, clamps roughness to the 0.04 floor,
      and respects scalar = 0 sentinels.

### PT-normal-map
- [ ] `Vertex` grows the `tangent: vec4` field. Layout test
      updated. WGSL `Vertex` mirrors.
- [ ] `Material` grows `normal_texture_idx` + the alignment
      pad. Layout test updated.
- [ ] glTF ingest reads `TANGENT` when present; otherwise calls
      `compute_tangents(positions, uvs, indices)` (CPU mirror,
      with tests pinning known-good outputs against a unit
      tetrahedron + a cube).
- [ ] glTF ingest reads `normalTexture` into the texture array,
      honours `normalTexture.scale` by folding into the tangent-
      space sample.
- [ ] WGSL `apply_normal_map(m, hit, tangent, sign)` computes
      the world-space shading normal. The integrator routes BSDF
      evaluation through the shading normal; self-intersection
      offset uses geometric normal.
- [ ] `examples/gen_pbr_maps` bakes a stone-tile normal map
      (256², random tile heights + edge bevels) into
      `data/textures/stone_tile_normal.png`.
- [ ] New `cornell_normal_mapped.gltf`: Cornell room with the
      floor replaced by a stone-tile material (baseColor flat
      grey + normal map). Brushed brass bunny standing on it.
- [ ] CPU mirror test: `apply_normal_map` over a flat (0, 1, 0)
      surface with an identity normal-texture (RGB = 0.5/0.5/1)
      returns (0, 1, 0); rotated tangent-space samples rotate
      the world normal as expected.

### PT-env-pbr — showcase
- [ ] `outdoor_normal_bunny.gltf`: floor (stone-tile normal map)
      + brushed brass bunny, no walls. Pairs with the synthetic
      sky from plan 0014.
- [ ] Reference render at 768²/2048 spp → `data/output/outdoor_normal_bunny_reference.png`.
      The bunny's brushed streaks read clearly under the env's
      directional sun lighting; the stone tiles show normal-map
      relief without baked shadows.
- [ ] README hero gallery: swap one of the four hero images for
      the normal-mapped reference (the visual upgrade vs. the
      existing tile is the whole point).
- [ ] Plan 0015 status → completed.

## Open questions

- **Per-texture array sizing.** Today all texture-array layers
  use the same dimensions (the max across baseColor textures).
  Mixing 1024² baseColor with 256² normal/MR is wasteful in
  VRAM but trivially correct. Defer compaction until a scene
  with mismatched mip pyramids actually pinches.
- **Mikktspace vs derived tangents.** The Stanford bunny has no
  explicit tangents in our `.obj`, so we derive. For organic
  meshes with smooth UVs the difference is invisible. If a
  future asset (rigged character, hard-edge mechanical part)
  shows seam artefacts, fall back to Mikktspace there.
- **Normal-map handedness.** OpenGL vs DirectX-style normal maps
  flip the green channel. We declare **OpenGL convention** (+Y
  up in tangent space) because glTF mandates it. The procedural
  bakes match.
- **Roughness floor.** A normal map can perturb a flat surface
  enough that GGX with roughness=0 (mirror) starts producing
  fireflies — the perturbed micro-normal still acts like a
  specular spike. Clamping `effective_roughness = max(.., 0.04)`
  is the standard PBR practice and what we do.

## Done when

- Rendering with the new maps shows visibly different surface
  behaviour vs. constant-material baseline at the **same scene
  geometry** — normal map reads as bumps, MR map reads as
  brushed streaks + dirt.
- Every existing scene renders byte-stably (no maps → no
  effective behaviour change).
- glTF round-trip preserves the new texture references.
- CPU mirror tests for `compute_tangents` and
  `apply_normal_map` stay green.
- Naga, native cargo test, fmt, clippy, wasm32 `cargo check`,
  Python unittests, CI, Pages-deploy all stay green at HEAD.
