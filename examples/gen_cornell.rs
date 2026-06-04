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

    // (file, subdivision per quad axis)
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

fn build_gltf(quads: &[GpuQuad], materials: &[GpuMaterial], subdiv: usize) -> Vec<u8> {
    let (palette, quad_material) = unique_materials(materials);
    let batches = triangulate(quads, &quad_material, palette.len(), subdiv);

    // --- Binary buffer + accessors ---
    let mut bin: Vec<u8> = Vec::new();
    let mut accessors_json: Vec<String> = Vec::new();
    let mut buffer_views_json: Vec<String> = Vec::new();
    let mut primitives_json: Vec<String> = Vec::new();

    for batch in &batches {
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
