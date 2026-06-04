//! Generates `data/gltf/cornell_quads.gltf` and `data/gltf/cornell_tris.gltf`
//! from the analytic Cornell Box description in `pathtrace::scene`.
//!
//! Both files share the same room + boxes; they differ only in
//! tessellation density:
//!
//! - `cornell_quads.gltf` — each of the 16 quads becomes 2 triangles
//!   (32 triangles total). The minimal triangulated representation,
//!   used as the regression baseline against the analytic-quad M3
//!   reference.
//! - `cornell_tris.gltf` — each quad is subdivided 4×4 into 16 sub-
//!   quads (32 sub-triangles), giving 512 triangles total. Useful for
//!   stressing the linear-scan and (later) BVH paths against a non-
//!   trivial mesh while keeping the rendered image visually identical.
//!
//! Run: `cargo run --example gen_cornell` from the repo root. The files
//! are committed to the repo; embedded into the binary by
//! `pathtrace.rs` via `include_bytes!`.

use std::fs;
use std::path::Path;

use quasi::pathtrace::scene::{cornell_box, GpuMaterial, GpuQuad};

fn main() {
    let out_dir = Path::new("data/gltf");
    fs::create_dir_all(out_dir).expect("create data/gltf");

    let scene = cornell_box();
    let quads = scene.quads.clone();
    let materials = scene.materials.clone();

    // 1) Whole Cornell (room + two boxes) at two subdivision densities.
    let outputs = [("cornell_quads.gltf", 1), ("cornell_tris.gltf", 4)];
    for (filename, subdiv) in outputs {
        let bytes = build_gltf(&quads, &materials, subdiv);
        let path = out_dir.join(filename);
        fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        println!(
            "wrote {} ({} bytes, {} subdivisions/quad → {} triangles)",
            path.display(),
            bytes.len(),
            subdiv * subdiv,
            quads.len() * subdiv * subdiv * 2,
        );
    }

    // 2) Cornell room + level-5 icosphere — the T4 publishable scene.
    //    Room only (5 walls + 1 light = first 6 quads of cornell_box());
    //    the two internal boxes are removed so the sphere is the hero.
    let room_quads: Vec<GpuQuad> = quads.iter().take(6).copied().collect();
    let room_materials: Vec<GpuMaterial> = materials.iter().take(6).copied().collect();
    let sphere_mat = GpuMaterial {
        albedo: [0.4, 0.5, 0.7],
        roughness: 1.0,
        emission: [0.0, 0.0, 0.0],
        metallic: 0.0,
    };
    let (sphere_positions, sphere_normals, sphere_indices) = icosphere(5, [0.0, 0.5, 0.0], 0.5);
    let bytes = build_gltf_with_sphere(
        &room_quads,
        &room_materials,
        &sphere_positions,
        &sphere_normals,
        &sphere_indices,
        sphere_mat,
    );
    let path = out_dir.join("cornell_sphere.gltf");
    fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!(
        "wrote {} ({} bytes, room + level-5 icosphere → {} triangles)",
        path.display(),
        bytes.len(),
        room_quads.len() * 2 + sphere_indices.len() / 3,
    );
}

// ---------------------------------------------------------------------------
// Material palette
// ---------------------------------------------------------------------------

/// Per-quad materials get deduplicated by exact float equality into the
/// glTF materials table. The analytic Cornell only ever uses 4 distinct
/// materials (white, red, green, light) so this stays compact.
fn unique_materials(materials: &[GpuMaterial]) -> (Vec<GpuMaterial>, Vec<usize>) {
    let mut palette: Vec<GpuMaterial> = Vec::new();
    let mut per_quad: Vec<usize> = Vec::with_capacity(materials.len());
    for m in materials {
        let existing = palette.iter().position(|p| {
            p.albedo == m.albedo
                && p.emission == m.emission
                && p.roughness == m.roughness
                && p.metallic == m.metallic
        });
        let idx = match existing {
            Some(i) => i,
            None => {
                palette.push(*m);
                palette.len() - 1
            }
        };
        per_quad.push(idx);
    }
    (palette, per_quad)
}

fn material_label(m: &GpuMaterial, idx: usize) -> String {
    let emissive = m.emission.iter().any(|&e| e > 0.0);
    if emissive {
        "light".to_string()
    } else if m.albedo[0] > m.albedo[1] && m.albedo[0] > m.albedo[2] {
        "red".to_string()
    } else if m.albedo[1] > m.albedo[0] && m.albedo[1] > m.albedo[2] {
        "green".to_string()
    } else if (m.albedo[0] - m.albedo[1]).abs() < 1e-3 {
        "white".to_string()
    } else {
        format!("material_{idx}")
    }
}

// ---------------------------------------------------------------------------
// Triangulation
// ---------------------------------------------------------------------------

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}

/// One material's geometry contribution to the glTF, before serialization.
struct PrimitiveBatch {
    material_idx: usize,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

/// Subdivides each `quad` into `subdiv`×`subdiv` sub-quads (so
/// `2·subdiv²` triangles per quad). Groups all triangles by material
/// into a single primitive per material.
fn triangulate(
    quads: &[GpuQuad],
    quad_material: &[usize],
    palette_len: usize,
    subdiv: usize,
) -> Vec<PrimitiveBatch> {
    let mut batches: Vec<PrimitiveBatch> = (0..palette_len)
        .map(|mat_idx| PrimitiveBatch {
            material_idx: mat_idx,
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        })
        .collect();

    for (q, &mat_idx) in quads.iter().zip(quad_material.iter()) {
        let normal = normalize(cross(q.u, q.v));
        let batch = &mut batches[mat_idx];
        let base = batch.positions.len() as u32;

        // (subdiv + 1)² grid of vertices in face uv space.
        for j in 0..=subdiv {
            for i in 0..=subdiv {
                let s = i as f32 / subdiv as f32;
                let t = j as f32 / subdiv as f32;
                let p = add(add(q.origin, scale(q.u, s)), scale(q.v, t));
                batch.positions.push(p);
                batch.normals.push(normal);
            }
        }

        // Two triangles per sub-quad cell; winding (A, B, C), (A, C, D)
        // gives a face normal pointing along u × v.
        let row = (subdiv + 1) as u32;
        for j in 0..subdiv as u32 {
            for i in 0..subdiv as u32 {
                let v00 = base + j * row + i;
                let v10 = v00 + 1;
                let v01 = base + (j + 1) * row + i;
                let v11 = v01 + 1;
                batch.indices.push(v00);
                batch.indices.push(v10);
                batch.indices.push(v11);
                batch.indices.push(v00);
                batch.indices.push(v11);
                batch.indices.push(v01);
            }
        }
    }

    batches.retain(|b| !b.positions.is_empty());
    batches
}

// ---------------------------------------------------------------------------
// Icosphere — recursive midpoint subdivision starting from an icosahedron,
// reprojected to the unit sphere at each step. Vertex normals = direction
// from origin (smooth shading).
// ---------------------------------------------------------------------------

/// Returns `(positions, normals, indices)` for an icosphere of subdivision
/// level `level`, scaled to `radius` and translated to `center`.
///
/// Vertex count: `10 * 4^level + 2`. Triangle count: `20 * 4^level`.
/// Level 5 = 10,242 vertices / 20,480 triangles — bunny territory
/// without an external download.
fn icosphere(
    level: usize,
    center: [f32; 3],
    radius: f32,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    let phi = (1.0_f32 + 5.0_f32.sqrt()) * 0.5;
    let mut verts: Vec<[f32; 3]> = vec![
        [-1.0, phi, 0.0],
        [1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [1.0, -phi, 0.0],
        [0.0, -1.0, phi],
        [0.0, 1.0, phi],
        [0.0, -1.0, -phi],
        [0.0, 1.0, -phi],
        [phi, 0.0, -1.0],
        [phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
    ];
    for v in &mut verts {
        let n = normalize(*v);
        *v = n;
    }

    let mut tris: Vec<u32> = vec![
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7,
        1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9,
        8, 1,
    ];

    use std::collections::HashMap;
    let mut edge_cache: HashMap<(u32, u32), u32> = HashMap::new();
    for _ in 0..level {
        let mut new_tris = Vec::with_capacity(tris.len() * 4);
        edge_cache.clear();
        for chunk in tris.chunks_exact(3) {
            let a = chunk[0];
            let b = chunk[1];
            let c = chunk[2];
            let ab = midpoint_index(a, b, &mut verts, &mut edge_cache);
            let bc = midpoint_index(b, c, &mut verts, &mut edge_cache);
            let ca = midpoint_index(c, a, &mut verts, &mut edge_cache);
            new_tris.extend_from_slice(&[a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca]);
        }
        tris = new_tris;
    }

    let mut positions = Vec::with_capacity(verts.len());
    let mut normals = Vec::with_capacity(verts.len());
    for v in &verts {
        positions.push([
            center[0] + v[0] * radius,
            center[1] + v[1] * radius,
            center[2] + v[2] * radius,
        ]);
        // The un-translated vertex on the unit sphere IS its own outward
        // normal; translation doesn't change the direction.
        normals.push(*v);
    }
    (positions, normals, tris)
}

fn midpoint_index(
    a: u32,
    b: u32,
    verts: &mut Vec<[f32; 3]>,
    cache: &mut std::collections::HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let va = verts[a as usize];
    let vb = verts[b as usize];
    let mid = normalize([
        (va[0] + vb[0]) * 0.5,
        (va[1] + vb[1]) * 0.5,
        (va[2] + vb[2]) * 0.5,
    ]);
    let idx = verts.len() as u32;
    verts.push(mid);
    cache.insert(key, idx);
    idx
}

// ---------------------------------------------------------------------------
// glTF JSON + base64 binary emission
// ---------------------------------------------------------------------------

const COMPONENT_TYPE_F32: u32 = 5126;
const COMPONENT_TYPE_U32: u32 = 5125;
const TARGET_ARRAY_BUFFER: u32 = 34962;
const TARGET_ELEMENT_ARRAY_BUFFER: u32 = 34963;
const PRIMITIVE_MODE_TRIANGLES: u32 = 4;

fn bounds(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in positions {
        for k in 0..3 {
            if p[k] < lo[k] {
                lo[k] = p[k];
            }
            if p[k] > hi[k] {
                hi[k] = p[k];
            }
        }
    }
    (lo, hi)
}

/// Same as [`build_gltf`] but also appends a single triangle mesh
/// (positions + normals + indices, one shared material). Used to ship
/// the icosphere alongside the room.
fn build_gltf_with_sphere(
    quads: &[GpuQuad],
    materials: &[GpuMaterial],
    sphere_positions: &[[f32; 3]],
    sphere_normals: &[[f32; 3]],
    sphere_indices: &[u32],
    sphere_material: GpuMaterial,
) -> Vec<u8> {
    let (mut palette, quad_material) = unique_materials(materials);
    let sphere_mat_idx = palette.len();
    palette.push(sphere_material);

    let mut batches = triangulate(quads, &quad_material, palette.len(), 1);
    batches.push(PrimitiveBatch {
        material_idx: sphere_mat_idx,
        positions: sphere_positions.to_vec(),
        normals: sphere_normals.to_vec(),
        indices: sphere_indices.to_vec(),
    });
    emit_gltf(&palette, &batches)
}

fn build_gltf(quads: &[GpuQuad], materials: &[GpuMaterial], subdiv: usize) -> Vec<u8> {
    let (palette, quad_material) = unique_materials(materials);
    let batches = triangulate(quads, &quad_material, palette.len(), subdiv);
    emit_gltf(&palette, &batches)
}

fn emit_gltf(palette: &[GpuMaterial], batches: &[PrimitiveBatch]) -> Vec<u8> {
    // --- Binary buffer + accessors ---
    let mut bin: Vec<u8> = Vec::new();
    let mut accessors_json: Vec<String> = Vec::new();
    let mut buffer_views_json: Vec<String> = Vec::new();
    let mut primitives_json: Vec<String> = Vec::new();

    for batch in batches {
        let pos_offset = bin.len();
        for p in &batch.positions {
            for &v in p {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        let pos_len = bin.len() - pos_offset;
        let pos_bv = buffer_views_json.len();
        buffer_views_json.push(format!(
            r#"{{"buffer":0,"byteOffset":{pos_offset},"byteLength":{pos_len},"target":{tgt}}}"#,
            tgt = TARGET_ARRAY_BUFFER,
        ));
        let (lo, hi) = bounds(&batch.positions);
        let pos_acc = accessors_json.len();
        accessors_json.push(format!(
            r#"{{"bufferView":{pos_bv},"componentType":{ct},"count":{cnt},"type":"VEC3","min":[{lx},{ly},{lz}],"max":[{hx},{hy},{hz}]}}"#,
            ct = COMPONENT_TYPE_F32,
            cnt = batch.positions.len(),
            lx = lo[0], ly = lo[1], lz = lo[2],
            hx = hi[0], hy = hi[1], hz = hi[2],
        ));

        let nor_offset = bin.len();
        for n in &batch.normals {
            for &v in n {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        let nor_len = bin.len() - nor_offset;
        let nor_bv = buffer_views_json.len();
        buffer_views_json.push(format!(
            r#"{{"buffer":0,"byteOffset":{nor_offset},"byteLength":{nor_len},"target":{tgt}}}"#,
            tgt = TARGET_ARRAY_BUFFER,
        ));
        let nor_acc = accessors_json.len();
        accessors_json.push(format!(
            r#"{{"bufferView":{nor_bv},"componentType":{ct},"count":{cnt},"type":"VEC3"}}"#,
            ct = COMPONENT_TYPE_F32,
            cnt = batch.normals.len(),
        ));

        let idx_offset = bin.len();
        for &i in &batch.indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        let idx_len = bin.len() - idx_offset;
        let idx_bv = buffer_views_json.len();
        buffer_views_json.push(format!(
            r#"{{"buffer":0,"byteOffset":{idx_offset},"byteLength":{idx_len},"target":{tgt}}}"#,
            tgt = TARGET_ELEMENT_ARRAY_BUFFER,
        ));
        let idx_acc = accessors_json.len();
        accessors_json.push(format!(
            r#"{{"bufferView":{idx_bv},"componentType":{ct},"count":{cnt},"type":"SCALAR"}}"#,
            ct = COMPONENT_TYPE_U32,
            cnt = batch.indices.len(),
        ));

        primitives_json.push(format!(
            r#"{{"attributes":{{"POSITION":{pos_acc},"NORMAL":{nor_acc}}},"indices":{idx_acc},"material":{mat_idx},"mode":{mode}}}"#,
            mat_idx = batch.material_idx,
            mode = PRIMITIVE_MODE_TRIANGLES,
        ));
    }

    let total_bin_len = bin.len();
    let b64 = base64_encode(&bin);

    // --- Materials JSON ---
    let materials_json: Vec<String> = palette
        .iter()
        .enumerate()
        .map(|(i, m)| {
            format!(
                r#"{{"name":"{name}","pbrMetallicRoughness":{{"baseColorFactor":[{ar},{ag},{ab},1.0],"metallicFactor":{met},"roughnessFactor":{rough}}},"emissiveFactor":[{er},{eg},{eb}]}}"#,
                name = material_label(m, i),
                ar = m.albedo[0], ag = m.albedo[1], ab = m.albedo[2],
                met = m.metallic, rough = m.roughness,
                er = m.emission[0], eg = m.emission[1], eb = m.emission[2],
            )
        })
        .collect();

    // --- Assemble root document ---
    let json = format!(
        r#"{{
"asset":{{"version":"2.0","generator":"quasi/examples/gen_cornell.rs"}},
"scene":0,
"scenes":[{{"nodes":[0]}}],
"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{prims}]}}],
"materials":[{mats}],
"accessors":[{accs}],
"bufferViews":[{bvs}],
"buffers":[{{"byteLength":{blen},"uri":"data:application/octet-stream;base64,{b64}"}}]
}}
"#,
        prims = primitives_json.join(","),
        mats = materials_json.join(","),
        accs = accessors_json.join(","),
        bvs = buffer_views_json.join(","),
        blen = total_bin_len,
    );
    json.into_bytes()
}

// ---------------------------------------------------------------------------
// Tiny inline base64 — same as the one in `pathtrace::mesh::tests`, kept
// here so the example has no dev-dep on a base64 crate.
// ---------------------------------------------------------------------------

fn base64_encode(input: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= input.len() {
        let n =
            (u32::from(input[i]) << 16) | (u32::from(input[i + 1]) << 8) | u32::from(input[i + 2]);
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
