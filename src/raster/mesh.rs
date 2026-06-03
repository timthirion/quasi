//! Triangle meshes for the rasterized pipeline.
//!
//! A small library: one `Vertex` layout (position + normal + color) and a
//! handful of procedural primitives — cube, sphere, cylinder — that cover
//! everything an articulated-robot demo needs to render (link bodies,
//! joint spheres, obstacle markers). Real mesh I/O (OBJ / glTF) lands when
//! a use case actually demands it, likely via the `morsel` sibling crate.

use bytemuck::{Pod, Zeroable};

/// Per-vertex attributes consumed by `forward.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    /// Byte stride of a `Vertex` — pinned at 36 (three 12-byte `f32x3`s).
    pub const STRIDE: u64 = 36;

    /// Vertex buffer layout consumed by the forward pipeline.
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: Self::STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 24,
                    shader_location: 2,
                },
            ],
        }
    }
}

/// CPU-side mesh: owned vertex + 16-bit index data.
#[derive(Clone, Debug)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

impl Mesh {
    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }
}

/// Axis-aligned cube centered at the origin with side length `size`.
///
/// 24 vertices (one per (face, corner) for hard per-face normals) and 36
/// indices (two triangles per face).
pub fn cube_mesh(size: f32, color: [f32; 3]) -> Mesh {
    let h = size * 0.5;

    let face = |normal: [f32; 3], corners: [[f32; 3]; 4]| {
        corners.map(|p| Vertex {
            position: p,
            normal,
            color,
        })
    };

    let mut vertices = Vec::with_capacity(24);
    // +X
    vertices.extend_from_slice(&face(
        [1.0, 0.0, 0.0],
        [[h, -h, -h], [h, h, -h], [h, h, h], [h, -h, h]],
    ));
    // -X
    vertices.extend_from_slice(&face(
        [-1.0, 0.0, 0.0],
        [[-h, -h, h], [-h, h, h], [-h, h, -h], [-h, -h, -h]],
    ));
    // +Y
    vertices.extend_from_slice(&face(
        [0.0, 1.0, 0.0],
        [[-h, h, -h], [-h, h, h], [h, h, h], [h, h, -h]],
    ));
    // -Y
    vertices.extend_from_slice(&face(
        [0.0, -1.0, 0.0],
        [[-h, -h, h], [-h, -h, -h], [h, -h, -h], [h, -h, h]],
    ));
    // +Z
    vertices.extend_from_slice(&face(
        [0.0, 0.0, 1.0],
        [[h, -h, h], [h, h, h], [-h, h, h], [-h, -h, h]],
    ));
    // -Z
    vertices.extend_from_slice(&face(
        [0.0, 0.0, -1.0],
        [[-h, -h, -h], [-h, h, -h], [h, h, -h], [h, -h, -h]],
    ));

    // Two CCW triangles per face: (0,1,2) and (0,2,3).
    let mut indices = Vec::with_capacity(36);
    for face_idx in 0..6u16 {
        let base = face_idx * 4;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    Mesh { vertices, indices }
}

/// UV-sphere centered at the origin with the given radius.
///
/// `lat_segments` is the number of horizontal rings (excluding poles);
/// `lon_segments` is the number of vertical wedges. Vertex normals point
/// outward from the sphere's center.
pub fn sphere_mesh(radius: f32, lat_segments: u32, lon_segments: u32, color: [f32; 3]) -> Mesh {
    use std::f32::consts::PI;

    let lat = lat_segments.max(2);
    let lon = lon_segments.max(3);

    let mut vertices = Vec::with_capacity(((lat + 1) * (lon + 1)) as usize);
    for i in 0..=lat {
        let v = i as f32 / lat as f32;
        let theta = v * PI; // 0 at +y pole, PI at -y pole
        let (st, ct) = theta.sin_cos();
        for j in 0..=lon {
            let u = j as f32 / lon as f32;
            let phi = u * 2.0 * PI;
            let (sp, cp) = phi.sin_cos();
            let n = [sp * st, ct, cp * st];
            vertices.push(Vertex {
                position: [radius * n[0], radius * n[1], radius * n[2]],
                normal: n,
                color,
            });
        }
    }

    let stride = lon + 1;
    let mut indices = Vec::with_capacity((lat * lon * 6) as usize);
    for i in 0..lat {
        for j in 0..lon {
            let a = (i * stride + j) as u16;
            let b = (i * stride + j + 1) as u16;
            let c = ((i + 1) * stride + j) as u16;
            let d = ((i + 1) * stride + j + 1) as u16;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    Mesh { vertices, indices }
}

/// Cylinder centered at the origin with its axis along +Y, with flat caps.
///
/// `height` is the total length along Y; `radius` is the side radius.
/// `segments` is the number of circumferential segments. Side normals point
/// radially outward; cap normals point along ±Y.
pub fn cylinder_mesh(radius: f32, height: f32, segments: u32, color: [f32; 3]) -> Mesh {
    use std::f32::consts::PI;

    let segs = segments.max(3);
    let half = height * 0.5;

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();

    // --- Side wall ---
    let side_base = vertices.len() as u16;
    for i in 0..=segs {
        let t = i as f32 / segs as f32;
        let phi = t * 2.0 * PI;
        let (sp, cp) = phi.sin_cos();
        let n = [sp, 0.0, cp];
        // Bottom ring
        vertices.push(Vertex {
            position: [radius * sp, -half, radius * cp],
            normal: n,
            color,
        });
        // Top ring
        vertices.push(Vertex {
            position: [radius * sp, half, radius * cp],
            normal: n,
            color,
        });
    }
    for i in 0..segs {
        let b0 = side_base + (i * 2) as u16;
        let t0 = b0 + 1;
        let b1 = b0 + 2;
        let t1 = b0 + 3;
        indices.extend_from_slice(&[b0, t0, b1, b1, t0, t1]);
    }

    // --- Top cap (+Y) ---
    let top_center = vertices.len() as u16;
    vertices.push(Vertex {
        position: [0.0, half, 0.0],
        normal: [0.0, 1.0, 0.0],
        color,
    });
    let top_ring_start = vertices.len() as u16;
    for i in 0..=segs {
        let t = i as f32 / segs as f32;
        let phi = t * 2.0 * PI;
        let (sp, cp) = phi.sin_cos();
        vertices.push(Vertex {
            position: [radius * sp, half, radius * cp],
            normal: [0.0, 1.0, 0.0],
            color,
        });
    }
    for i in 0..segs {
        let a = top_ring_start + i as u16;
        let b = a + 1;
        indices.extend_from_slice(&[top_center, a, b]);
    }

    // --- Bottom cap (-Y) ---
    let bot_center = vertices.len() as u16;
    vertices.push(Vertex {
        position: [0.0, -half, 0.0],
        normal: [0.0, -1.0, 0.0],
        color,
    });
    let bot_ring_start = vertices.len() as u16;
    for i in 0..=segs {
        let t = i as f32 / segs as f32;
        let phi = t * 2.0 * PI;
        let (sp, cp) = phi.sin_cos();
        vertices.push(Vertex {
            position: [radius * sp, -half, radius * cp],
            normal: [0.0, -1.0, 0.0],
            color,
        });
    }
    for i in 0..segs {
        let a = bot_ring_start + i as u16;
        let b = a + 1;
        // Reverse winding so bottom faces outward.
        indices.extend_from_slice(&[bot_center, b, a]);
    }

    Mesh { vertices, indices }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_stride_matches_const() {
        assert_eq!(std::mem::size_of::<Vertex>() as u64, Vertex::STRIDE);
    }

    #[test]
    fn cube_has_24_vertices_and_36_indices() {
        let m = cube_mesh(1.0, [1.0; 3]);
        assert_eq!(m.vertices.len(), 24);
        assert_eq!(m.indices.len(), 36);
        assert_eq!(m.index_count(), 36);
    }

    #[test]
    fn cube_normals_are_unit() {
        let m = cube_mesh(2.0, [0.5; 3]);
        for v in &m.vertices {
            let len: f32 = v.normal.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((len - 1.0).abs() < 1e-6, "non-unit normal {:?}", v.normal);
        }
    }

    #[test]
    fn sphere_has_expected_counts() {
        let m = sphere_mesh(1.0, 8, 16, [1.0; 3]);
        assert_eq!(m.vertices.len(), 9 * 17);
        assert_eq!(m.indices.len(), 8 * 16 * 6);
    }

    #[test]
    fn sphere_vertices_are_on_the_sphere() {
        let r = 2.5;
        let m = sphere_mesh(r, 8, 16, [1.0; 3]);
        for v in &m.vertices {
            let len: f32 = v.position.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (len - r).abs() < 1e-4,
                "vertex not on sphere: {:?}",
                v.position
            );
            let n: f32 = v.normal.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn cylinder_side_radii_match() {
        let r = 0.4;
        let m = cylinder_mesh(r, 1.0, 12, [1.0; 3]);
        // Side ring vertices are at exactly radius r in xz; first 26
        // are the side wall (12 segs + 1) * 2.
        for v in m.vertices.iter().take(26) {
            let rho = (v.position[0].powi(2) + v.position[2].powi(2)).sqrt();
            assert!((rho - r).abs() < 1e-5);
        }
    }
}
