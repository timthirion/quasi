//! Triangle meshes for the rasterized pipeline.
//!
//! Phase R1 keeps this tiny: one `Vertex` layout (position + normal + color),
//! one `Mesh` type (owned vertex + index data), and a procedural cube. R2
//! grows this with capsules / cylinders / spheres for representing robot
//! link geometry.

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
}
