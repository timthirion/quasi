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

    // 2) Cornell room + level-5 icosphere — procedural test scene
    //    that doesn't require any external asset.
    //    Room only (5 walls + 1 light = first 6 quads of cornell_box());
    //    the two internal boxes are removed so the sphere is the hero.
    let room_quads: Vec<GpuQuad> = quads.iter().take(6).copied().collect();
    let room_materials: Vec<GpuMaterial> = materials.iter().take(6).copied().collect();
    let sphere_mat = GpuMaterial {
        albedo: [0.4, 0.5, 0.7],
        roughness: 1.0,
        emission: [0.0, 0.0, 0.0],
        metallic: 0.0,
        ior: 0.0,
        absorption: [0.0, 0.0, 0.0],
        scattering: [0.0, 0.0, 0.0],
    };
    let (sphere_positions, sphere_normals, sphere_indices) = icosphere(5, [0.0, 0.5, 0.0], 0.5);
    let bytes = build_gltf_with_extra_mesh(
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

    // 3) Cornell room + Stanford bunny — the **canonical** publishable
    //    scene. OBJ embedded at compile time from data/obj/.
    let bunny_obj = include_str!("../data/obj/stanford-bunny.obj");
    let (bunny_raw_positions, bunny_indices) = parse_obj(bunny_obj);
    // Bunny extent ~0.16 × 0.19 × 0.12, min_y = 0. Scale 5× → ~0.8 ×
    // 0.94 × 0.6 standing on the floor; horizontal offset centres it.
    let bunny_scale = 5.0_f32;
    let bunny_offset = [0.0825_f32, 0.0, 0.0075];
    let bunny_positions: Vec<[f32; 3]> = bunny_raw_positions
        .iter()
        .map(|p| {
            [
                p[0] * bunny_scale + bunny_offset[0],
                p[1] * bunny_scale + bunny_offset[1],
                p[2] * bunny_scale + bunny_offset[2],
            ]
        })
        .collect();
    let bunny_normals = compute_smooth_normals(&bunny_positions, &bunny_indices);
    let bunny_mat = GpuMaterial {
        // Warm clay — distinguishes the bunny from the white walls
        // without sliding off the "Lambertian reference scene" footing.
        albedo: [0.8, 0.65, 0.5],
        roughness: 1.0,
        emission: [0.0, 0.0, 0.0],
        metallic: 0.0,
        ior: 0.0,
        absorption: [0.0, 0.0, 0.0],
        scattering: [0.0, 0.0, 0.0],
    };
    let bytes = build_gltf_with_extra_mesh(
        &room_quads,
        &room_materials,
        &bunny_positions,
        &bunny_normals,
        &bunny_indices,
        bunny_mat,
    );
    let path = out_dir.join("cornell_bunny.gltf");
    fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!(
        "wrote {} ({} bytes, room + Stanford bunny → {} triangles)",
        path.display(),
        bytes.len(),
        room_quads.len() * 2 + bunny_indices.len() / 3,
    );

    // 4) Cornell with a metallic bunny — the PT-ggx publishable
    //    scene. Same geometry as cornell_bunny, but the bunny material
    //    is brushed steel (metallic = 1, roughness = 0.3, F0 from the
    //    Schlick conductor albedo).
    let metal_bunny_mat = GpuMaterial {
        // F0 for steel ≈ (0.56, 0.57, 0.58) in linear; we go a touch
        // warmer to make the colour bleed from the red wall pop.
        albedo: [0.60, 0.58, 0.55],
        roughness: 0.3,
        emission: [0.0, 0.0, 0.0],
        metallic: 1.0,
        ior: 0.0,
        absorption: [0.0, 0.0, 0.0],
        scattering: [0.0, 0.0, 0.0],
    };
    let bytes = build_gltf_with_extra_mesh(
        &room_quads,
        &room_materials,
        &bunny_positions,
        &bunny_normals,
        &bunny_indices,
        metal_bunny_mat,
    );
    let path = out_dir.join("cornell_metal_bunny.gltf");
    fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!(
        "wrote {} ({} bytes, room + brushed-steel bunny → {} triangles)",
        path.display(),
        bytes.len(),
        room_quads.len() * 2 + bunny_indices.len() / 3,
    );

    // 5a) Cornell with a green-glass Stanford bunny — the
    //     PT-beer-lambert publishable scene. Same geometry as
    //     cornell_bunny, but the bunny material is a green-tinted
    //     dielectric (ior=1.5, absorption tuned to look unmistakably
    //     "green glass" without going opaque at the bunny's thickest
    //     parts ≈ 0.5 unit).
    let glass_bunny_mat = GpuMaterial {
        albedo: [1.0, 1.0, 1.0],
        roughness: 0.0,
        emission: [0.0, 0.0, 0.0],
        metallic: 0.0,
        ior: 1.5,
        // Bunny scale ≈ 0.6-0.8 unit thick at the body. exp(-1.2 ·
        // 0.6) ≈ 0.49 → ~half the red light gets absorbed across the
        // body; tiny green absorption leaves the colour green-leaning.
        absorption: [1.2, 0.1, 1.5],
        scattering: [0.0, 0.0, 0.0],
    };
    let bytes = build_gltf_with_extra_mesh(
        &room_quads,
        &room_materials,
        &bunny_positions,
        &bunny_normals,
        &bunny_indices,
        glass_bunny_mat,
    );
    let path = out_dir.join("cornell_glass_bunny.gltf");
    fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!(
        "wrote {} ({} bytes, room + green-glass bunny → {} triangles)",
        path.display(),
        bytes.len(),
        room_quads.len() * 2 + bunny_indices.len() / 3,
    );

    // 5) Cornell with a clear-glass icosphere — the PT-dielectrics
    //    publishable scene. Same geometry as cornell_sphere, but the
    //    sphere material is smooth glass (ior=1.5, roughness=0,
    //    metallic=0). The path tracer routes hits with `ior > 0` onto
    //    the smooth-dielectric branch (Snell + Fresnel + TIR).
    let glass_sphere_mat = GpuMaterial {
        // Tiny tint to avoid sterility — the WGSL multiplies it into
        // the transmitted radiance, so we keep it close to (1,1,1)
        // and let the caustic do the visual work.
        albedo: [1.0, 1.0, 1.0],
        roughness: 0.0,
        emission: [0.0, 0.0, 0.0],
        metallic: 0.0,
        ior: 1.5,
        absorption: [0.0, 0.0, 0.0],
        scattering: [0.0, 0.0, 0.0],
    };
    let bytes = build_gltf_with_extra_mesh(
        &room_quads,
        &room_materials,
        &sphere_positions,
        &sphere_normals,
        &sphere_indices,
        glass_sphere_mat,
    );
    let path = out_dir.join("cornell_glass_sphere.gltf");
    fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!(
        "wrote {} ({} bytes, room + glass icosphere → {} triangles)",
        path.display(),
        bytes.len(),
        room_quads.len() * 2 + sphere_indices.len() / 3,
    );

    // 5b) Cornell with a fog volume — the PT-fog publishable scene.
    //     A 12-triangle axis-aligned box sized to fill most of the
    //     room as the medium volume. ior = 0 so the BSDF dispatch
    //     skips dielectric handling and the path tracer's medium-
    //     boundary detection passes the ray through, only swapping
    //     `current_medium`. Tuned for visible god-rays: small
    //     absorption + moderate scattering so the path through the
    //     volume is bright enough to see but shadowy enough that
    //     the light cone reads as distinct rays.
    let fog_mat = GpuMaterial {
        albedo: [1.0, 1.0, 1.0],
        roughness: 1.0,
        emission: [0.0, 0.0, 0.0],
        metallic: 0.0,
        ior: 0.0,
        // Moderate fog. Mean free path = 1 / (σ_a + σ_s) ≈ 1.8
        // unit. Beer-Lambert across the 2-unit room gives ≈ 33%
        // direct transmittance, so walls visibly dim but stay
        // recognisable; the remaining ~67% of camera rays scatter
        // inside the fog volume and pick up in-scattered light
        // from the ceiling cone. Scattering ten times absorption
        // keeps the room bright without a smoke-dense feel.
        absorption: [0.05, 0.05, 0.05],
        scattering: [0.5, 0.5, 0.5],
    };
    // Extends ABOVE the ceiling (max_y = 2.05 vs ceiling at 2.0) so
    // the entire room volume sits inside the fog — light tile,
    // ceiling, and all the walls. The fog box's +y face at y=2.05
    // is unreachable from any in-room ray (the ceiling at y=2.0
    // hits first), so it's effectively dead geometry. The
    // alternative — fog top *just below* the ceiling — runs us
    // into shadow-origin / fog-top coincidence (the WGSL ceiling
    // shadow offset of 0.001 would land directly on the fog top
    // if it sat at y=1.999). Going above the ceiling sidesteps
    // that whole class of degeneracy and visually nothing changes.
    let (fog_positions, fog_normals, fog_indices) = aabb_box(
        [-0.99, 0.01, -0.99],
        [0.99, 2.05, 0.99],
    );
    let bytes = build_gltf_with_extra_mesh(
        &room_quads,
        &room_materials,
        &fog_positions,
        &fog_normals,
        &fog_indices,
        fog_mat,
    );
    let path = out_dir.join("cornell_foggy_room.gltf");
    fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!(
        "wrote {} ({} bytes, room + fog volume → {} triangles)",
        path.display(),
        bytes.len(),
        room_quads.len() * 2 + fog_indices.len() / 3,
    );

    // 6) Cornell with a textured floor — the PT-textures publishable
    //    scene. Same geometry as cornell_quads, but the floor's
    //    material samples the embedded uv_checker_color.png.
    let bytes = build_cornell_textured_floor();
    let path = out_dir.join("cornell_textured_floor.gltf");
    fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!(
        "wrote {} ({} bytes, room + UV-checker floor)",
        path.display(),
        bytes.len(),
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
/// `uvs` may be empty (the glTF won't carry a `TEXCOORD_0` attribute in
/// that case) or the same length as `positions`.
struct PrimitiveBatch {
    material_idx: usize,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

/// Subdivides each `quad` into `subdiv`×`subdiv` sub-quads (so
/// `2·subdiv²` triangles per quad). Groups all triangles by material
/// into a single primitive per material. Per-vertex UVs are the
/// subdivision coordinates `(s, t)` in `[0, 1]²` — a single tiling of
/// any texture over the full quad face.
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
            uvs: Vec::new(),
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
                batch.uvs.push([s, t]);
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
/// Closed axis-aligned box as 8 vertices + 36 indices. Triangle
/// winding is CCW-from-the-outside on every face so the geometric
/// normal computed by `record_hit` (the cross product of two edges
/// in vertex order) points OUTWARD. This is load-bearing for
/// PT-fog: the path tracer's medium-boundary detection toggles
/// `current_medium` based on `Hit::front_face`, which is derived
/// from `dot(geom_n, ray.dir)`. If the geometric normals point
/// inward, a camera ray entering the box reads as "exiting the
/// medium" and the volume attenuation never fires.
///
/// Vertex index encoding (sign of each coordinate):
///   0 = ---, 1 = +--, 2 = ++-, 3 = -+-,
///   4 = --+, 5 = +-+, 6 = +++, 7 = -++
fn aabb_box(min: [f32; 3], max: [f32; 3]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    let p = [
        [min[0], min[1], min[2]], // 0: ---
        [max[0], min[1], min[2]], // 1: +--
        [max[0], max[1], min[2]], // 2: ++-
        [min[0], max[1], min[2]], // 3: -+-
        [min[0], min[1], max[2]], // 4: --+
        [max[0], min[1], max[2]], // 5: +-+
        [max[0], max[1], max[2]], // 6: +++
        [min[0], max[1], max[2]], // 7: -++
    ];
    let tris: Vec<u32> = vec![
        // -x face (normal -x). CCW viewed from -x.
        0, 7, 3,  0, 4, 7,
        // +x face (normal +x). CCW viewed from +x.
        1, 2, 6,  1, 6, 5,
        // -y face (normal -y, bottom). CCW viewed from -y.
        0, 1, 5,  0, 5, 4,
        // +y face (normal +y, top). CCW viewed from +y.
        2, 3, 7,  2, 7, 6,
        // -z face (normal -z, back). CCW viewed from -z.
        0, 3, 2,  0, 2, 1,
        // +z face (normal +z, front). CCW viewed from +z.
        4, 5, 6,  4, 6, 7,
    ];
    let positions: Vec<[f32; 3]> = p.to_vec();
    // Flat normals don't matter for a medium-volume boundary (the
    // path tracer reads only the geometric normal in
    // `is_medium_volume_material`), but emit_gltf still wants per-
    // vertex normals. Use a constant up-normal as a placeholder; the
    // BSDF dispatch never reads it for medium-volume materials.
    let normals: Vec<[f32; 3]> = positions.iter().map(|_| [0.0, 1.0, 0.0]).collect();
    (positions, normals, tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn aabb_box_geometric_normals_point_outward() {
        // Reproduction of the PT-fog gotcha: every face's geometric
        // normal must point AWAY from the box centroid, otherwise
        // path tracer reads boundary crossings backwards and
        // Beer-Lambert silently never fires.
        let min = [-2.0_f32, -3.0, -4.0];
        let max = [5.0_f32, 1.0, 2.0];
        let centroid = [
            0.5 * (min[0] + max[0]),
            0.5 * (min[1] + max[1]),
            0.5 * (min[2] + max[2]),
        ];
        let (positions, _normals, tris) = aabb_box(min, max);
        assert_eq!(tris.len() % 3, 0);
        assert_eq!(tris.len() / 3, 12, "box must have 12 triangles (6 faces × 2)");
        for tri in tris.chunks(3) {
            let v0 = positions[tri[0] as usize];
            let v1 = positions[tri[1] as usize];
            let v2 = positions[tri[2] as usize];
            let n = cross(sub(v1, v0), sub(v2, v0));
            // From any triangle vertex to centroid → "inward" direction.
            // Outward normal must have negative dot with that.
            let inward = sub(centroid, v0);
            assert!(
                dot(n, inward) < 0.0,
                "triangle {tri:?} has inward-pointing normal (geom_n={n:?}, inward={inward:?})",
            );
        }
    }
}

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
// OBJ parsing + smooth vertex normals
// ---------------------------------------------------------------------------

/// Bare-bones OBJ parser — just positions and triangle faces, enough
/// to ingest the Stanford bunny. Ignores UVs and per-vertex normals
/// (we recompute smooth normals from positions below).
fn parse_obj(src: &str) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for line in src.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("v ") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() >= 3 {
                positions.push([
                    parts[0].parse().expect("OBJ position component"),
                    parts[1].parse().expect("OBJ position component"),
                    parts[2].parse().expect("OBJ position component"),
                ]);
            }
        } else if let Some(rest) = line.strip_prefix("f ") {
            // Faces can be `a b c` or `a/b c/d e/f` or `a/b/c d/e/f g/h/i`.
            // We only need the vertex index (first / -separated field).
            let parts: Vec<&str> = rest.split_whitespace().collect();
            let mut tri_idx: Vec<u32> = Vec::with_capacity(parts.len());
            for p in &parts {
                let v: i32 = p
                    .split('/')
                    .next()
                    .unwrap()
                    .parse()
                    .expect("OBJ face index");
                // OBJ is 1-indexed; convert to 0-indexed.
                let idx = if v > 0 {
                    v as u32 - 1
                } else {
                    // Negative = relative to current vertex list.
                    (positions.len() as i32 + v) as u32
                };
                tri_idx.push(idx);
            }
            // Triangulate (fan) any face with >3 verts.
            for i in 1..(tri_idx.len() - 1) {
                indices.push(tri_idx[0]);
                indices.push(tri_idx[i]);
                indices.push(tri_idx[i + 1]);
            }
        }
    }
    (positions, indices)
}

/// Computes per-vertex smooth normals by accumulating face normals
/// (unnormalised — area-weighted) over all triangles touching each
/// vertex, then normalising. Standard smooth-shading approach.
fn compute_smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals: Vec<[f32; 3]> = vec![[0.0_f32; 3]; positions.len()];
    for chunk in indices.chunks_exact(3) {
        let v0 = positions[chunk[0] as usize];
        let v1 = positions[chunk[1] as usize];
        let v2 = positions[chunk[2] as usize];
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let n = cross(e1, e2);
        for &i in chunk {
            let i = i as usize;
            normals[i][0] += n[0];
            normals[i][1] += n[1];
            normals[i][2] += n[2];
        }
    }
    for n in &mut normals {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        n[0] /= l;
        n[1] /= l;
        n[2] /= l;
    }
    normals
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

/// Same as [`build_gltf`] but also appends a single extra triangle
/// mesh (positions + normals + indices, one shared material). Used to
/// ship the icosphere or Stanford bunny alongside the room. UVs for
/// the extra mesh default to `(0, 0)` per-vertex — `PT-textures`
/// doesn't texture either of them.
fn build_gltf_with_extra_mesh(
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
        uvs: vec![[0.0, 0.0]; sphere_positions.len()],
        indices: sphere_indices.to_vec(),
    });
    let no_textures = vec![None; palette.len()];
    emit_gltf(&palette, &no_textures, &batches, &[])
}

fn build_gltf(quads: &[GpuQuad], materials: &[GpuMaterial], subdiv: usize) -> Vec<u8> {
    let (palette, quad_material) = unique_materials(materials);
    let batches = triangulate(quads, &quad_material, palette.len(), subdiv);
    let no_textures = vec![None; palette.len()];
    emit_gltf(&palette, &no_textures, &batches, &[])
}

/// `data/gltf/cornell_textured_floor.gltf` — same Cornell as
/// `cornell_quads`, but the floor's material is replaced by a textured
/// Lambertian that samples `uv_checker_color.png` (PNG bytes embedded
/// in the glTF as a base64 data URI). The publishable artifact for
/// `PT-textures`.
fn build_cornell_textured_floor() -> Vec<u8> {
    let scene = cornell_box();
    let quads = scene.quads.clone();
    let materials = scene.materials.clone();

    let (mut palette, mut quad_material) = unique_materials(&materials);

    // Append a textured-floor material. Same albedo as the existing
    // white material, but baseColorTexture is set so the WGSL shader
    // multiplies in the sampled checker.
    let floor_mat_idx = palette.len();
    palette.push(GpuMaterial {
        albedo: [1.0, 1.0, 1.0],
        roughness: 1.0,
        emission: [0.0; 3],
        metallic: 0.0,
        ior: 0.0,
        absorption: [0.0, 0.0, 0.0],
        scattering: [0.0, 0.0, 0.0],
    });
    // Quad 0 is the floor (per `cornell_box`); rebind it to the new
    // textured material.
    quad_material[0] = floor_mat_idx;

    let batches = triangulate(&quads, &quad_material, palette.len(), 1);

    let mut material_textures: Vec<Option<u32>> = vec![None; palette.len()];
    material_textures[floor_mat_idx] = Some(0);
    let textures: &[&[u8]] = &[include_bytes!("../data/textures/uv_checker_color.png")];

    emit_gltf(&palette, &material_textures, &batches, textures)
}

/// `material_textures[i] = Some(layer)` means material `i` uses
/// `textures[layer]` as its `baseColorTexture`. `textures` is a slice
/// of raw PNG byte slices; each gets embedded as a base64 data URI.
fn emit_gltf(
    palette: &[GpuMaterial],
    material_textures: &[Option<u32>],
    batches: &[PrimitiveBatch],
    textures: &[&[u8]],
) -> Vec<u8> {
    assert_eq!(palette.len(), material_textures.len());
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

        // PT-textures: optional TEXCOORD_0 accessor. Skip entirely if
        // the batch carries no UVs — the glTF spec allows missing
        // texcoord sets, and a default attribute would just inflate
        // the file for nothing.
        let uv_acc: Option<usize> = if batch.uvs.len() == batch.positions.len() {
            let uv_offset = bin.len();
            for uv in &batch.uvs {
                for &v in uv {
                    bin.extend_from_slice(&v.to_le_bytes());
                }
            }
            let uv_len = bin.len() - uv_offset;
            let uv_bv = buffer_views_json.len();
            buffer_views_json.push(format!(
                r#"{{"buffer":0,"byteOffset":{uv_offset},"byteLength":{uv_len},"target":{tgt}}}"#,
                tgt = TARGET_ARRAY_BUFFER,
            ));
            let acc_idx = accessors_json.len();
            accessors_json.push(format!(
                r#"{{"bufferView":{uv_bv},"componentType":{ct},"count":{cnt},"type":"VEC2"}}"#,
                ct = COMPONENT_TYPE_F32,
                cnt = batch.uvs.len(),
            ));
            Some(acc_idx)
        } else {
            None
        };

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

        let attributes = match uv_acc {
            Some(uv) => format!(
                r#""POSITION":{pos_acc},"NORMAL":{nor_acc},"TEXCOORD_0":{uv}"#
            ),
            None => format!(r#""POSITION":{pos_acc},"NORMAL":{nor_acc}"#),
        };
        primitives_json.push(format!(
            r#"{{"attributes":{{{attributes}}},"indices":{idx_acc},"material":{mat_idx},"mode":{mode}}}"#,
            mat_idx = batch.material_idx,
            mode = PRIMITIVE_MODE_TRIANGLES,
        ));
    }

    // --- Materials JSON ---
    let materials_json: Vec<String> = palette
        .iter()
        .zip(material_textures.iter())
        .enumerate()
        .map(|(i, (m, tex_idx))| {
            let base_color_texture = match tex_idx {
                Some(idx) => format!(r#","baseColorTexture":{{"index":{idx}}}"#),
                None => String::new(),
            };
            // PT-dielectrics / PT-beer-lambert / PT-fog: non-standard
            // material fields ride in `extras` rather than per-
            // extension feature gates on the gltf crate. We collect
            // the non-default pieces and stitch them together. The
            // `match` style got noisy with three optional fields, so
            // we drop down to a builder. When all three are zero the
            // section stays empty and the on-disk JSON is byte-stable
            // with what PT-dielectrics shipped.
            let mut extras_parts: Vec<String> = Vec::new();
            if m.ior > 0.0 {
                extras_parts.push(format!(r#""ior":{}"#, m.ior));
            }
            if m.absorption.iter().any(|&c| c > 0.0) {
                extras_parts.push(format!(
                    r#""absorption":[{},{},{}]"#,
                    m.absorption[0], m.absorption[1], m.absorption[2],
                ));
            }
            if m.scattering.iter().any(|&c| c > 0.0) {
                extras_parts.push(format!(
                    r#""scattering":[{},{},{}]"#,
                    m.scattering[0], m.scattering[1], m.scattering[2],
                ));
            }
            let extras = if extras_parts.is_empty() {
                String::new()
            } else {
                format!(r#","extras":{{{}}}"#, extras_parts.join(","))
            };
            format!(
                r#"{{"name":"{name}","pbrMetallicRoughness":{{"baseColorFactor":[{ar},{ag},{ab},1.0],"metallicFactor":{met},"roughnessFactor":{rough}{tex}}},"emissiveFactor":[{er},{eg},{eb}]{extras}}}"#,
                name = material_label(m, i),
                ar = m.albedo[0], ag = m.albedo[1], ab = m.albedo[2],
                met = m.metallic, rough = m.roughness,
                er = m.emission[0], eg = m.emission[1], eb = m.emission[2],
                tex = base_color_texture,
            )
        })
        .collect();

    // --- Textures + images JSON (optional) ---
    //
    // Image data goes into the binary buffer as a bufferView; the
    // gltf crate's `import_slice` rejects glTF documents that
    // reference *image* data via separate data URIs (it treats those
    // as external resources), but it's happy to pull them out of the
    // single self-contained binary buffer the rest of the geometry
    // already uses.
    let textures_section = if textures.is_empty() {
        String::new()
    } else {
        let mut image_bvs: Vec<usize> = Vec::with_capacity(textures.len());
        for png in textures {
            let img_offset = bin.len();
            bin.extend_from_slice(png);
            let img_len = bin.len() - img_offset;
            let bv = buffer_views_json.len();
            buffer_views_json.push(format!(
                r#"{{"buffer":0,"byteOffset":{img_offset},"byteLength":{img_len}}}"#,
            ));
            image_bvs.push(bv);
        }
        let images_json: Vec<String> = image_bvs
            .iter()
            .map(|bv| format!(r#"{{"bufferView":{bv},"mimeType":"image/png"}}"#))
            .collect();
        let textures_json: Vec<String> = (0..textures.len())
            .map(|i| format!(r#"{{"source":{i}}}"#))
            .collect();
        format!(
            r#","textures":[{tex}],"images":[{imgs}]"#,
            tex = textures_json.join(","),
            imgs = images_json.join(","),
        )
    };

    // --- Assemble root document ---
    let total_bin_len = bin.len();
    let b64 = base64_encode(&bin);
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
"buffers":[{{"byteLength":{blen},"uri":"data:application/octet-stream;base64,{b64}"}}]{textures_section}
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
