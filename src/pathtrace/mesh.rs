//! Triangle mesh ingest for the path tracer.
//!
//! Plan 0003 — T0. Loads glTF 2.0 (binary `.glb` or text `.gltf`) into a
//! flat [`TriangleScene`] of world-space triangles, ready for the WGSL
//! integrator to intersect (T1) and a SAH BVH to accelerate (T2).
//!
//! ## Why glTF (only)
//!
//! glTF is a strict superset of what an `.obj` import would give us
//! (hierarchical transforms, PBR-aligned materials, embeddable binary
//! payload via `.glb`), so a single ingest path serves the renderer
//! through M4 of plan 0001 and Phase 4+ of the roadmap. The `gltf`
//! crate compiles for both native and `wasm32-unknown-unknown` via its
//! existing `import_slice` API.
//!
//! ## Layout
//!
//! - [`Vertex`] — 32 bytes, `[f32; 3]` position + scalar pad + `[f32; 3]`
//!   normal + scalar pad. The padding makes the struct trivially
//!   uploadable as a WGSL `struct { position: vec3<f32>, normal:
//!   vec3<f32> }` in std140 / storage layouts (per the M1 uniform-
//!   layout lesson).
//! - [`Material`] — same 32-byte shape as the existing
//!   `scene::GpuMaterial`, so the WGSL `Material` struct from plan 0001
//!   stays valid byte-for-byte when T1 rewires the shader.
//! - [`TriangleScene`] — flat buffers of vertices, indices (3 per
//!   triangle), materials, per-triangle material indices, and the list
//!   of emissive triangle indices.
//!
//! ## Out of scope for T0
//!
//! Loading is CPU-only here; no GPU plumbing yet. The BVH field
//! mentioned in the plan design lands in T2 (and is currently absent
//! from `TriangleScene`).
//!
//! glTF features deliberately ignored at T0: texture sampling
//! (`baseColorTexture` etc.), animations, skins, sparse accessors, and
//! instancing — the node hierarchy is flattened into world-space
//! triangles eagerly. The PBR metallic/roughness scalars are *parsed*
//! and stored but unused by today's Lambertian shader; they're carried
//! forward so a follow-up BSDF plan doesn't need to re-touch ingest.

use bytemuck::{Pod, Zeroable};

use crate::pathtrace::bvh::Bvh;

/// Single mesh vertex, packed to match the WGSL storage-buffer layout
/// the T1 shader will read. The two `_pad*` slots make the size 32
/// bytes (8 floats), which is what `vec3 + vec3` rounds up to under
/// std140 — matching CPU↔GPU sizes pins the class of bug the M1 uniform
/// layout note warned about.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub normal: [f32; 3],
    pub _pad1: f32,
}

/// PBR-aligned material. Lambertian (M1) only reads `albedo` and
/// `emission`; `roughness` and `metallic` are stored for later BSDFs.
/// Same 32-byte shape as `scene::GpuMaterial` so the WGSL `Material`
/// struct from plan 0001 keeps working when T1 swaps the geometry path.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct Material {
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub emission: [f32; 3],
    pub metallic: f32,
}

impl Default for Material {
    fn default() -> Self {
        // Mirrors the glTF 2.0 default material (white Lambertian).
        Material {
            albedo: [1.0, 1.0, 1.0],
            roughness: 1.0,
            emission: [0.0, 0.0, 0.0],
            metallic: 1.0,
        }
    }
}

impl Material {
    /// True iff any emission component is positive — drives the
    /// `emissive_triangles` list and (later) NEE light sampling.
    pub fn is_emissive(&self) -> bool {
        self.emission[0] > 0.0 || self.emission[1] > 0.0 || self.emission[2] > 0.0
    }
}

/// A scene flattened from glTF into world-space triangles ready for
/// shader upload.
#[derive(Clone, Debug, Default)]
pub struct TriangleScene {
    pub vertices: Vec<Vertex>,
    /// 3 indices per triangle, into `vertices`.
    pub indices: Vec<u32>,
    /// Material palette. Slot 0 is always the default material; glTF
    /// materials slot in starting at index 1.
    pub materials: Vec<Material>,
    /// Per-triangle index into `materials`. `triangle_materials.len()
    /// == indices.len() / 3`.
    pub triangle_materials: Vec<u32>,
    /// Triangle indices (not vertex indices) whose material has
    /// non-zero emission. Populated by [`Self::recompute_emissive`].
    pub emissive_triangles: Vec<u32>,
    /// SAH BVH over the triangles in `indices`. Built at glTF load
    /// time. The WGSL fragment shader walks this via stack-based
    /// traversal; the linear-scan path also remains, gated behind
    /// `Uniforms::use_bvh`.
    pub bvh: Bvh,
}

impl TriangleScene {
    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Rebuilds [`Self::emissive_triangles`] from the current materials.
    /// Callers don't normally invoke this — the loader does it before
    /// returning.
    pub fn recompute_emissive(&mut self) {
        self.emissive_triangles.clear();
        for (tri_idx, &mat_idx) in self.triangle_materials.iter().enumerate() {
            if self.materials[mat_idx as usize].is_emissive() {
                self.emissive_triangles.push(tri_idx as u32);
            }
        }
    }
}

#[derive(Debug)]
pub enum MeshError {
    Gltf(gltf::Error),
    #[cfg(not(target_arch = "wasm32"))]
    Io(std::io::Error),
    NoScene,
    NoPositions,
    NoNormals,
    UnsupportedPrimitive(gltf::mesh::Mode),
}

impl core::fmt::Display for MeshError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Gltf(e) => write!(f, "glTF error: {e}"),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::NoScene => write!(f, "glTF file has no scenes"),
            Self::NoPositions => write!(f, "primitive missing POSITION attribute"),
            Self::NoNormals => write!(f, "primitive missing NORMAL attribute"),
            Self::UnsupportedPrimitive(m) => {
                write!(f, "unsupported primitive mode {m:?} (only Triangles)")
            }
        }
    }
}

impl std::error::Error for MeshError {}

impl From<gltf::Error> for MeshError {
    fn from(e: gltf::Error) -> Self {
        Self::Gltf(e)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<std::io::Error> for MeshError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Matrix helpers — column-major (glTF convention)
// ---------------------------------------------------------------------------

/// 4×4 column-major: `m[col][row]`.
pub type Mat4 = [[f32; 4]; 4];

pub fn identity_mat4() -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    m[0][0] = 1.0;
    m[1][1] = 1.0;
    m[2][2] = 1.0;
    m[3][3] = 1.0;
    m
}

pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut c = [[0.0; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[k][row] * b[col][k];
            }
            c[col][row] = s;
        }
    }
    c
}

/// `m * (p, 1)` — apply the full affine transform to a point.
pub fn transform_point(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

/// Three column vectors of the upper-left 3×3.
fn upper_3x3_columns(m: &Mat4) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Transforms a normal by `cofactor(M_3x3) * n`, then normalises. The
/// cofactor matrix is the inverse-transpose up to a scalar, so this is
/// correct under arbitrary non-uniform scaling — at the cost of three
/// cross products per vertex (one per node, since we precompute the
/// cofactor once per primitive).
pub fn transform_normal(cols: &[[f32; 3]; 3], n: [f32; 3]) -> [f32; 3] {
    let c12 = cross(cols[1], cols[2]);
    let c20 = cross(cols[2], cols[0]);
    let c01 = cross(cols[0], cols[1]);
    let x = c12[0] * n[0] + c20[0] * n[1] + c01[0] * n[2];
    let y = c12[1] * n[0] + c20[1] * n[1] + c01[1] * n[2];
    let z = c12[2] * n[0] + c20[2] * n[1] + c01[2] * n[2];
    let len = (x * x + y * y + z * z).sqrt().max(1e-12);
    [x / len, y / len, z / len]
}

// ---------------------------------------------------------------------------
// glTF loader
// ---------------------------------------------------------------------------

/// Reads a glTF file from disk (`.glb` or `.gltf` + sidecar `.bin`).
/// Native-only because the wasm target doesn't have `std::fs`.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_glb<P: AsRef<std::path::Path>>(path: P) -> Result<TriangleScene, MeshError> {
    let bytes = std::fs::read(path.as_ref())?;
    load_glb_bytes(&bytes)
}

/// Parses a glTF blob (binary `.glb` or JSON `.gltf` with embedded
/// data-URI buffers). Cross-target: callers on the web pass bytes from
/// `fetch`.
pub fn load_glb_bytes(bytes: &[u8]) -> Result<TriangleScene, MeshError> {
    let (document, buffers, _images) = gltf::import_slice(bytes)?;

    let mut scene = TriangleScene::default();
    scene.materials.push(Material::default());
    for material in document.materials() {
        scene.materials.push(extract_material(&material));
    }

    let active_scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or(MeshError::NoScene)?;

    let identity = identity_mat4();
    for node in active_scene.nodes() {
        walk_node(&node, &buffers, &identity, &mut scene)?;
    }

    scene.recompute_emissive();
    scene.bvh = Bvh::build(&scene.vertices, &scene.indices);
    Ok(scene)
}

fn extract_material(material: &gltf::Material) -> Material {
    let pbr = material.pbr_metallic_roughness();
    let base = pbr.base_color_factor();
    let emissive = material.emissive_factor();
    Material {
        albedo: [base[0], base[1], base[2]],
        roughness: pbr.roughness_factor(),
        emission: emissive,
        metallic: pbr.metallic_factor(),
    }
}

fn walk_node(
    node: &gltf::Node,
    buffers: &[gltf::buffer::Data],
    parent: &Mat4,
    scene: &mut TriangleScene,
) -> Result<(), MeshError> {
    let local = node.transform().matrix();
    let world = mat4_mul(parent, &local);

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            process_primitive(&primitive, buffers, &world, scene)?;
        }
    }
    for child in node.children() {
        walk_node(&child, buffers, &world, scene)?;
    }
    Ok(())
}

fn process_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    world: &Mat4,
    scene: &mut TriangleScene,
) -> Result<(), MeshError> {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Err(MeshError::UnsupportedPrimitive(primitive.mode()));
    }

    let reader = primitive.reader(|b| Some(&buffers[b.index()]));
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or(MeshError::NoPositions)?
        .collect();
    let normals: Vec<[f32; 3]> = reader.read_normals().ok_or(MeshError::NoNormals)?.collect();
    if normals.len() != positions.len() {
        return Err(MeshError::NoNormals);
    }
    let indices: Vec<u32> = match reader.read_indices() {
        Some(read) => read.into_u32().collect(),
        // Non-indexed primitive: emit a simple 0..N index list.
        None => (0..positions.len() as u32).collect(),
    };

    // Pre-compute the normal-transform 3x3 once per primitive — this is
    // the world-matrix cofactor, valid for arbitrary non-uniform scaling.
    let cols = upper_3x3_columns(world);

    let vertex_offset = scene.vertices.len() as u32;
    for i in 0..positions.len() {
        let p = transform_point(world, positions[i]);
        let n = transform_normal(&cols, normals[i]);
        scene.vertices.push(Vertex {
            position: p,
            _pad0: 0.0,
            normal: n,
            _pad1: 0.0,
        });
    }

    // Slot 0 is the default material; glTF materials live in slots 1+.
    let material_idx = primitive
        .material()
        .index()
        .map(|i| (i + 1) as u32)
        .unwrap_or(0);

    for tri in indices.chunks_exact(3) {
        scene.indices.push(tri[0] + vertex_offset);
        scene.indices.push(tri[1] + vertex_offset);
        scene.indices.push(tri[2] + vertex_offset);
        scene.triangle_materials.push(material_idx);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn close_arr(a: [f32; 3], b: [f32; 3]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
    }

    // ---- Layout ----

    #[test]
    fn vertex_is_32_bytes() {
        assert_eq!(std::mem::size_of::<Vertex>(), 32);
    }

    #[test]
    fn material_is_32_bytes_and_matches_gpu_material_layout() {
        // The existing scene::GpuMaterial has the same shape; both sides
        // upload as the same WGSL `Material` struct, so a mismatch here
        // (e.g. a missing pad) only fails at runtime.
        assert_eq!(std::mem::size_of::<Material>(), 32);
        assert_eq!(
            std::mem::size_of::<Material>(),
            std::mem::size_of::<crate::pathtrace::scene::GpuMaterial>(),
        );
    }

    // ---- Material ----

    #[test]
    fn default_material_is_white_lambertian() {
        let m = Material::default();
        assert_eq!(m.albedo, [1.0, 1.0, 1.0]);
        assert_eq!(m.emission, [0.0, 0.0, 0.0]);
        assert!(!m.is_emissive());
    }

    #[test]
    fn material_is_emissive_when_any_channel_positive() {
        let mut m = Material::default();
        assert!(!m.is_emissive());
        m.emission = [0.0, 0.5, 0.0];
        assert!(m.is_emissive());
        m.emission = [1e-9, 0.0, 0.0];
        assert!(m.is_emissive());
    }

    // ---- Math ----

    #[test]
    fn identity_transforms_are_no_ops() {
        let m = identity_mat4();
        assert!(close_arr(
            transform_point(&m, [3.0, -2.0, 1.0]),
            [3.0, -2.0, 1.0]
        ));
        let cols = upper_3x3_columns(&m);
        let n = transform_normal(&cols, [0.0, 1.0, 0.0]);
        assert!(close_arr(n, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn translation_only_affects_points_not_normals() {
        let mut m = identity_mat4();
        m[3][0] = 5.0;
        m[3][1] = -2.0;
        m[3][2] = 7.0;
        assert!(close_arr(
            transform_point(&m, [1.0, 1.0, 1.0]),
            [6.0, -1.0, 8.0]
        ));
        let cols = upper_3x3_columns(&m);
        // Upper 3x3 is still the identity, so the normal is unchanged.
        assert!(close_arr(
            transform_normal(&cols, [0.0, 1.0, 0.0]),
            [0.0, 1.0, 0.0]
        ));
    }

    #[test]
    fn ninety_degree_y_rotation_maps_x_to_neg_z() {
        // glTF column-major: m[col][row]. R_y(90°) columns:
        //   col 0 = (cos, 0, -sin) = (0, 0, -1)
        //   col 1 = (0, 1, 0)
        //   col 2 = (sin, 0, cos) = (1, 0, 0)
        let mut m = identity_mat4();
        m[0][0] = 0.0;
        m[0][2] = -1.0;
        m[2][0] = 1.0;
        m[2][2] = 0.0;
        let p = transform_point(&m, [1.0, 0.0, 0.0]);
        assert!(close_arr(p, [0.0, 0.0, -1.0]), "got {p:?}");
        let cols = upper_3x3_columns(&m);
        let n = transform_normal(&cols, [1.0, 0.0, 0.0]);
        assert!(close_arr(n, [0.0, 0.0, -1.0]), "got {n:?}");
    }

    #[test]
    fn nonuniform_scale_normal_stays_perpendicular_to_a_surface() {
        // Stretch X by 2, Y by 1, Z by 3. A surface in the XZ plane has
        // normal (0,1,0); under non-uniform scale the normal must stay
        // along Y (post-normalize) — anything else means we'd be
        // multiplying by M_3x3 instead of cofactor(M_3x3).
        let mut m = identity_mat4();
        m[0][0] = 2.0;
        m[1][1] = 1.0;
        m[2][2] = 3.0;
        let cols = upper_3x3_columns(&m);
        let n = transform_normal(&cols, [0.0, 1.0, 0.0]);
        assert!(close_arr(n, [0.0, 1.0, 0.0]), "got {n:?}");
    }

    #[test]
    fn mat4_mul_with_identity_is_a_noop() {
        let mut m = identity_mat4();
        m[0][1] = 0.5;
        m[2][3] = -1.0;
        m[3][0] = 2.0;
        let i = identity_mat4();
        let r1 = mat4_mul(&i, &m);
        let r2 = mat4_mul(&m, &i);
        for col in 0..4 {
            for row in 0..4 {
                assert!(close(r1[col][row], m[col][row]));
                assert!(close(r2[col][row], m[col][row]));
            }
        }
    }

    // ---- Emissive bookkeeping ----

    #[test]
    fn recompute_emissive_finds_emissive_triangles() {
        let mut s = TriangleScene::default();
        s.materials.push(Material::default()); // 0 — non-emissive
        s.materials.push(Material {
            emission: [3.0, 3.0, 3.0],
            ..Material::default()
        }); // 1 — emissive
        s.triangle_materials = vec![0, 1, 0, 1, 1];
        s.recompute_emissive();
        assert_eq!(s.emissive_triangles, vec![1, 3, 4]);
    }

    // ---- glTF round-trip ----

    /// Inline base64 encoder so the test can embed a binary buffer into
    /// a glTF data URI without a dev-dep.
    fn base64_encode(input: &[u8]) -> String {
        const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
        let mut i = 0;
        while i + 3 <= input.len() {
            let n = (u32::from(input[i]) << 16)
                | (u32::from(input[i + 1]) << 8)
                | u32::from(input[i + 2]);
            out.push(CHARS[((n >> 18) & 0x3F) as usize] as char);
            out.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
            out.push(CHARS[((n >> 6) & 0x3F) as usize] as char);
            out.push(CHARS[(n & 0x3F) as usize] as char);
            i += 3;
        }
        let rem = input.len() - i;
        if rem == 1 {
            let n = u32::from(input[i]) << 16;
            out.push(CHARS[((n >> 18) & 0x3F) as usize] as char);
            out.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
            out.push('=');
            out.push('=');
        } else if rem == 2 {
            let n = (u32::from(input[i]) << 16) | (u32::from(input[i + 1]) << 8);
            out.push(CHARS[((n >> 18) & 0x3F) as usize] as char);
            out.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
            out.push(CHARS[((n >> 6) & 0x3F) as usize] as char);
            out.push('=');
        }
        out
    }

    /// Builds a glTF JSON with a data-URI buffer holding two unindexed
    /// triangles. Primitive 0 uses material 0 (white Lambertian);
    /// primitive 1 uses material 1 (emissive). One node, one mesh, one
    /// scene.
    fn make_test_gltf() -> Vec<u8> {
        let positions: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 1.0],
        ];
        let normals: &[[f32; 3]] = &[
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        let indices: &[u16] = &[0, 1, 2, 3, 4, 5];

        let mut bin = Vec::new();
        for p in positions {
            for &v in p {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        for n in normals {
            for &v in n {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        let positions_byte_length = positions.len() * 12;
        let normals_byte_length = normals.len() * 12;
        let indices_byte_offset = positions_byte_length + normals_byte_length;
        for &i in indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        let indices_byte_length = indices.len() * 2;
        let total_byte_length = bin.len();

        let b64 = base64_encode(&bin);
        let json = format!(
            r#"{{
"asset": {{ "version": "2.0" }},
"scene": 0,
"scenes": [{{ "nodes": [0] }}],
"nodes": [{{ "mesh": 0 }}],
"meshes": [{{ "primitives": [
  {{ "attributes": {{ "POSITION": 0, "NORMAL": 1 }}, "indices": 2, "material": 0 }},
  {{ "attributes": {{ "POSITION": 0, "NORMAL": 1 }}, "indices": 3, "material": 1 }}
]}}],
"accessors": [
  {{ "bufferView": 0, "componentType": 5126, "count": 6, "type": "VEC3", "min": [0.0,0.0,0.0], "max": [1.0,2.0,1.0] }},
  {{ "bufferView": 1, "componentType": 5126, "count": 6, "type": "VEC3" }},
  {{ "bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR" }},
  {{ "bufferView": 2, "byteOffset": 6, "componentType": 5123, "count": 3, "type": "SCALAR" }}
],
"bufferViews": [
  {{ "buffer": 0, "byteOffset": 0, "byteLength": {p_len} }},
  {{ "buffer": 0, "byteOffset": {n_off}, "byteLength": {n_len} }},
  {{ "buffer": 0, "byteOffset": {i_off}, "byteLength": {i_len} }}
],
"buffers": [
  {{ "byteLength": {total}, "uri": "data:application/octet-stream;base64,{b64}" }}
],
"materials": [
  {{ "pbrMetallicRoughness": {{ "baseColorFactor": [0.8, 0.7, 0.6, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.9 }} }},
  {{ "emissiveFactor": [4.0, 4.0, 4.0] }}
]
}}"#,
            p_len = positions_byte_length,
            n_off = positions_byte_length,
            n_len = normals_byte_length,
            i_off = indices_byte_offset,
            i_len = indices_byte_length,
            total = total_byte_length,
        );
        json.into_bytes()
    }

    #[test]
    fn round_trips_two_triangles_and_emissive_set() {
        let gltf = make_test_gltf();
        let scene = load_glb_bytes(&gltf).expect("load");

        // Both primitives share the POSITION/NORMAL accessors, but the
        // loader doesn't dedupe across primitives — each primitive gets
        // its own copy in the global vertex buffer. So 6 + 6 = 12
        // vertices for 2 triangles. Trades a small amount of memory for
        // a much simpler ingest, and matches what real glTF files
        // produce (per-primitive distinct attributes).
        assert_eq!(scene.vertices.len(), 12, "vertex count");
        assert_eq!(scene.indices.len(), 6, "index count");
        assert_eq!(scene.triangle_count(), 2);
        // 1 default + 2 glTF materials.
        assert_eq!(scene.materials.len(), 3);

        // Material 1 (slot index 1) corresponds to glTF material 0 (white).
        assert!(close_arr(scene.materials[1].albedo, [0.8, 0.7, 0.6]));
        assert!(close(scene.materials[1].roughness, 0.9));
        // Material 2 corresponds to glTF material 1 (emissive).
        assert!(close_arr(scene.materials[2].emission, [4.0, 4.0, 4.0]));

        // Primitive 0 used glTF material 0 → palette slot 1.
        // Primitive 1 used glTF material 1 → palette slot 2.
        assert_eq!(scene.triangle_materials, vec![1, 2]);

        // Triangle 1 is the emissive one.
        assert_eq!(scene.emissive_triangles, vec![1]);

        // Both primitives reference the *same* POSITION/NORMAL accessors,
        // so both copies of the vertex buffer contain the same 6 normals:
        // (0,1,0) for vertices 0..3 and (0,-1,0) for 3..6. Each
        // primitive's copy starts at its own vertex_offset.
        assert!(close_arr(scene.vertices[0].position, [0.0, 0.0, 0.0]));
        assert!(close_arr(scene.vertices[5].position, [0.0, 2.0, 1.0]));
        // Primitive 0 copy: vertex 0 is up, vertex 3 is down.
        assert!(close_arr(scene.vertices[0].normal, [0.0, 1.0, 0.0]));
        assert!(close_arr(scene.vertices[3].normal, [0.0, -1.0, 0.0]));
        // Primitive 1 copy starts at vertex 6 — same pattern.
        assert!(close_arr(scene.vertices[6].normal, [0.0, 1.0, 0.0]));
        assert!(close_arr(scene.vertices[9].normal, [0.0, -1.0, 0.0]));

        // Index lists are offset by `vertex_offset` per primitive:
        // primitive 0 → [0,1,2] + 0 = [0,1,2];
        // primitive 1 → [3,4,5] + 6 = [9,10,11].
        assert_eq!(&scene.indices[..3], &[0, 1, 2]);
        assert_eq!(&scene.indices[3..], &[9, 10, 11]);
    }

    #[test]
    fn missing_normals_errors() {
        // Same scene without the NORMAL attribute / accessor.
        let positions: &[[f32; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let mut bin = Vec::new();
        for p in positions {
            for &v in p {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        let indices: &[u16] = &[0, 1, 2];
        let pos_len = positions.len() * 12;
        let idx_off = pos_len;
        for &i in indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        let total = bin.len();
        let b64 = base64_encode(&bin);
        let json = format!(
            r#"{{
"asset": {{ "version": "2.0" }},
"scene": 0,
"scenes": [{{ "nodes": [0] }}],
"nodes": [{{ "mesh": 0 }}],
"meshes": [{{ "primitives": [
  {{ "attributes": {{ "POSITION": 0 }}, "indices": 1 }}
]}}],
"accessors": [
  {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0.0,0.0,0.0], "max": [1.0,0.0,1.0] }},
  {{ "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }}
],
"bufferViews": [
  {{ "buffer": 0, "byteOffset": 0, "byteLength": {p_len} }},
  {{ "buffer": 0, "byteOffset": {i_off}, "byteLength": 6 }}
],
"buffers": [
  {{ "byteLength": {total}, "uri": "data:application/octet-stream;base64,{b64}" }}
]
}}"#,
            p_len = pos_len,
            i_off = idx_off,
            total = total,
        );
        match load_glb_bytes(json.as_bytes()) {
            Err(MeshError::NoNormals) => {}
            other => panic!("expected NoNormals, got {other:?}"),
        }
    }
}
