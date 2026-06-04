# PBR materials + textures (path tracer)

- **Status:** active
- **Last updated:** 2026-06-04
- **Last touched on:** PT-textures landed

## Goal

Bring physically-based materials online in the path tracer:
**textured albedo**, **GGX microfacet metals**, and **dielectric
refractors** (glass). Each is a focused milestone with its own
publishable artifact — together they push every existing test scene
from "Lambertian-only" into "looks like a real renderer." The Cornell
bunny becomes brushed steel; the icosphere becomes glass; a textured
checker floor demonstrates the UV-sampling code visually.

This is plan `0004` and the first plan to use the
[`PT-<topic>`](ROADMAP.md#plan--milestone-conventions) semantic
milestone convention.

## Context

What's already in the path tracer (as of plan `0003`):

- Triangle meshes loaded from glTF, with per-triangle material indices
  and per-vertex normals. `pathtrace::mesh::Vertex` is 32 bytes
  (position + pad + normal + pad) — **no UV attribute yet**.
- Materials parse `baseColorFactor`, `emissiveFactor`, `roughnessFactor`,
  `metallicFactor` at load time but the WGSL shader **only reads
  `albedo` and `emission`**. `roughness` and `metallic` sit inert.
- The OBJ ingest already parses `vt` lines (the Stanford bunny ships
  with morsel-generated UVs covering `[0, 1]²`).
- Two 1024×1024 UV-checker test textures sit in `data/textures/`
  (rainbow + monochrome, by Valle).
- `Limits::default()` on wasm gives us 8 storage buffers per shader
  stage; the current pathtrace bind group uses 8 (uniform + 5 storage
  + 2 BVH). Adding a texture array + sampler bumps to 10 entries
  total but **uses separate texture / sampler binding slots** — both
  baseline-allowed on WebGPU.

What this plan is **not**:

- Many-light sampling (separate plan).
- Padded high-dim Sobol (separate plan).
- Denoising.
- Texture compression (BC1/BC7/ASTC) — RGBA8 only for now.
- Non-base-color texture maps (normal maps, roughness maps,
  metallic-roughness maps). Those land as a `PT-pbr-maps` follow-up
  if the headline renders need them; `PT-textures` here ships the
  scaffolding for one map only (`baseColorTexture`).

## Design

### Vertex layout grows to 48 bytes

The current 32-byte `Vertex` (position + pad + normal + pad) becomes
48 bytes: `position: vec3<f32>` (12 + 4 pad) + `normal: vec3<f32>`
(12 + 4 pad) + `uv: vec2<f32>` (8) + 8 bytes trailing pad to keep the
struct aligned for storage-buffer arrays (std430 rounds struct size
up to the struct's alignment = 16). The CPU `#[repr(C)]` mirror
follows the same byte layout; a layout test pins it.

### Material layout grows by 16 bytes

`Material` gains a `base_color_texture_idx: u32` field. Value
`0xFFFFFFFF` means "no texture, use the constant `albedo`"; any other
value is a layer index into the texture array. Total Material size
goes from 32 → 48 bytes (pad to next 16-byte boundary). The
`Material` shape on the glTF side already carries the field (via
`pbr_metallic_roughness.base_color_texture.index`); T0 just wasn't
reading it.

### Texture storage: `texture_2d_array<f32>`

All scene textures live in a single 2-D texture array (one layer per
texture). Constraints:

- All layers share dimensions. The renderer picks the **max** of the
  input textures' dimensions and resizes smaller ones via the
  `image` crate's `Lanczos3` filter at load time.
- Layer 0 is **always** a 1×1 white texture, so the binding is
  well-defined even for scenes with no textures.
- Format is `Rgba8UnormSrgb` — base-color textures are gamma-encoded
  by convention, and the sRGB sampler does the right thing.

### One sampler

A single `sampler` (linear filtering, repeat wrap) shared by every
material. Per-material sampler state can land in a follow-up if any
scene actually needs it.

### Möller-Trumbore returns barycentric

`intersect_triangle` currently returns `t`. It changes to return
`vec3<f32>(t, u, v)` so the caller can compute interpolated UV at
the hit. `trace_scene` widens its `Hit` struct to carry the
interpolated UV.

### Sampling at hit

Inside `path_trace`, after `trace_scene`:

```wgsl
let mat = materials[hit.mat];
var albedo = mat.albedo;
if (mat.base_color_texture_idx != 0xffffffffu) {
    let tex = textureSampleLevel(
        textures, tex_sampler, hit.uv, mat.base_color_texture_idx, 0.0
    );
    albedo = albedo * tex.rgb;
}
```

We use `textureSampleLevel(..., 0.0)` — explicit mip level — because
WGSL only allows automatic mip derivatives in fragment shaders that
write to a single render target without barriers. Our fragment
shader is fine for that **today**, but explicit lookup keeps the
shader portable to compute-shader path tracers later.

## Milestones

### PT-textures ✅
First slice: texture-modulated Lambertian. Closes the UV/texture
scaffolding plus one demo render.

- [x] `Vertex` grows a `uv: [f32; 2]` field; 48-byte stride, layout
      test pins it.
- [x] `mesh::load_glb_bytes` reads the `TEXCOORD_0` attribute when
      present (defaults `(0, 0)` per-vertex when absent), plus
      `pbr_metallic_roughness.base_color_texture.index` on the
      material. Texture images come out of `gltf::import_slice`'s
      `images` vec (R8G8B8A8 / R8G8B8 → upcast to RGBA8). The OBJ
      side was left for a follow-up — the bunny model already loads
      fine with `(0, 0)` UVs as default.
- [x] `Material` gains `base_color_texture_idx: u32` (sentinel
      `0xFFFFFFFF`). Layout test pinned at 48 bytes with the index
      at offset 32.
- [x] `TriangleScene` gains `pub textures: Vec<TextureImage>` with
      RGBA8 data + dimensions.
- [x] GPU upload: `wgpu::Texture` array, `Rgba8UnormSrgb`, one layer
      per texture, with a 1×1 white default layer for scenes that
      have no textures. Single-texture for now — when a second
      texture lands we'll wire up the Lanczos3 resize.
- [x] WGSL: `texture_2d_array<f32>` at `@group(0) @binding(8)`,
      `sampler` at `@binding(9)`. `intersect_triangle` returns
      `vec3<f32>(t, u, v)`; `Hit` carries interpolated UV; albedo
      gets multiplied by sampled texture when material has one.
- [x] New test scene: `examples/gen_cornell.rs` emits
      `data/gltf/cornell_textured_floor.gltf` — the Cornell room
      with the floor sampled from `uv_checker_color.png`. The PNG
      is embedded **in the glb binary buffer** referenced via a
      bufferView with `image/png` mimeType (the gltf crate's
      `import_slice` rejects external `data:` URIs). Planar UV
      projection on the floor quad. Reference render at 512² /
      1024 spp is `data/output/cornell_textured_floor_reference.png`.
- [x] Tests: Vertex + Material layout pinned at 48 bytes;
      `cornell_textured_floor.gltf` round-trips through
      `load_glb_bytes` with 1 texture (1024×1024, mostly non-white)
      and the floor's four corner UVs are (0,0)/(1,0)/(0,1)/(1,1).
      Naga validates the new shader.

**Out of scope here:** sampling a `roughnessTexture` or
`metallicRoughnessTexture`. Those land in `PT-ggx` if needed.

**Landed bug** (worth recording — the WGSL fix that nearly didn't
happen): when the textured-floor render kept coming back grey, the
debug short-circuits all pointed at the texture sampling working
correctly. The path tracer's NEE evaluation and the throughput
update both kept using `m.albedo` directly instead of the
texture-multiplied `albedo` local. Only the AOV output had been
switched over. Fix was three characters per line, in
`pathtrace.wgsl` `path_trace`.

### PT-ggx
Microfacet metal BRDF. Turns the bunny into brushed steel.

- [ ] Material's `metallic` + `roughness` fields hook up; the WGSL
      branches on `metallic > 0.5` to dispatch GGX vs Lambertian
      (proper "metallic = lerp" comes later if needed).
- [ ] GGX importance sampling: half-vector sampled from the
      Trowbridge-Reitz GGX distribution; `f` evaluated with Smith
      separable masking-shadowing.
- [ ] Conductor Fresnel via Schlick approximation; F0 read from the
      material's `albedo` (PBR convention).
- [ ] NEE + MIS compatible: the BSDF pdf goes into the same
      power-heuristic weight that the existing Lambertian path uses.
- [ ] New test scene: `cornell_metal_bunny.gltf` — the bunny with
      `metallic = 1, roughness = 0.3`. Reference render.
- [ ] Convergence sweep updated to cover the new BSDF.
- [ ] Tests: GGX importance-sampling pdf matches the analytic
      formula (CPU-side numerical check); BSDF evaluates to the
      Lambertian answer when `roughness = 1, metallic = 0`
      (regression bridge to the existing rendering).

### PT-dielectrics
Glass + clear plastics. Snell + Fresnel, transmission allowed.

- [ ] Material grows an `ior: f32` (sentinel `0.0` = not a
      dielectric). When non-zero, the BSDF chooses reflection or
      refraction based on the dielectric Fresnel term, importance-
      sampled.
- [ ] Refraction direction via Snell's law; total internal
      reflection handled.
- [ ] Path tracer: throughput multiplies by `1.0` (no albedo
      attenuation for clear dielectrics by default); colored
      glass uses `albedo` as Beer-Lambert absorption coefficient
      along the medium path.
- [ ] New test scene: `cornell_glass_sphere.gltf` — the icosphere
      from `0003` PT-stress with `ior = 1.5`, sitting on the
      Cornell floor. Reference render.
- [ ] Tests: Snell vector matches the analytic formula; total
      internal reflection kicks in at the right angle.

## Open questions

- **Image upload size cap.** Should the loader refuse to upload a
  texture larger than e.g. 4 K? Decide during `PT-textures`; for now
  reject anything past `Limits::default().max_texture_dimension_2d`.
- **Texture array length cap.** WebGPU baseline allows 256 array
  layers; we likely never need more than ~16. Decide on a soft cap
  during `PT-textures`.
- **Hex sentinel vs `Option<u32>`.** `0xFFFFFFFF` as "no texture" is
  unambiguous and bytemuck-friendly. Document explicitly; consider
  exposing as a constant `pub const NO_TEXTURE: u32 = u32::MAX`.
- **GGX → Lambertian regression.** Does the GGX path at
  `roughness = 1` reproduce the Lambertian render exactly? If yes,
  the existing convergence numbers carry over. Verify during
  `PT-ggx`.

## Done when

- Bunny renders as: (a) textured Lambertian (`PT-textures`),
  (b) brushed metal (`PT-ggx`), (c) glass (`PT-dielectrics`). Each
  scene ships a published PNG + EXR + a convergence CSV.
- The path tracer's `Material` carries albedo, roughness, metallic,
  ior, and base_color_texture_idx end-to-end — no more
  "field-parsed-but-unused" rot.
- Naga, native + wasm clippy, fmt, and the unit + GPU-regression
  test suite all stay green.
