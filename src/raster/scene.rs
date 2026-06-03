//! Scene description for the rasterized pipeline.
//!
//! The scene is a flat list of [`Instance`]s — each names a [`MeshHandle`]
//! into the renderer's geometry library and supplies a model transform and
//! a color tint. The renderer groups by mesh handle each frame and issues
//! one instanced draw call per group.
//!
//! There's no per-pipeline scene trait shared with `pathtrace`; the two
//! pipelines deliberately work with different abstractions (quads +
//! emissive lights for path tracing, meshes + instances + a forward sun
//! for rasterization).

use bytemuck::{Pod, Zeroable};

/// Index of a registered mesh in the renderer's geometry library.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u32);

/// CPU-side description of one drawable: a registered mesh transformed by a
/// model matrix and tinted by a color.
#[derive(Clone, Debug)]
pub struct Instance {
    pub mesh: MeshHandle,
    pub model: [[f32; 4]; 4],
    pub tint: [f32; 4],
}

/// Identity model with a white tint — convenient for tests and defaults.
impl Instance {
    pub fn identity(mesh: MeshHandle) -> Self {
        Self {
            mesh,
            model: IDENTITY_MAT4,
            tint: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

pub const IDENTITY_MAT4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// GPU-side per-instance attributes consumed by the vertex shader as
/// instance-stepped vertex attributes (locations 3..=7).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct InstanceRaw {
    pub model_col0: [f32; 4],
    pub model_col1: [f32; 4],
    pub model_col2: [f32; 4],
    pub model_col3: [f32; 4],
    pub tint: [f32; 4],
}

impl InstanceRaw {
    pub const STRIDE: u64 = 80;

    pub fn from(instance: &Instance) -> InstanceRaw {
        InstanceRaw {
            model_col0: instance.model[0],
            model_col1: instance.model[1],
            model_col2: instance.model[2],
            model_col3: instance.model[3],
            tint: instance.tint,
        }
    }

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: Self::STRIDE,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 5,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 6,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 64,
                    shader_location: 7,
                },
            ],
        }
    }
}

/// CPU-side scene: a flat instance list, plus convenience accessors. The
/// renderer reads this each frame and uploads what it needs.
#[derive(Clone, Debug, Default)]
pub struct Scene {
    instances: Vec<Instance>,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }

    pub fn clear(&mut self) {
        self.instances.clear();
    }

    pub fn push(&mut self, instance: Instance) {
        self.instances.push(instance);
    }

    pub fn extend<I: IntoIterator<Item = Instance>>(&mut self, iter: I) {
        self.instances.extend(iter);
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

/// Build a column-major translation matrix.
pub fn translation(tx: f32, ty: f32, tz: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [tx, ty, tz, 1.0],
    ]
}

/// Column-major scale matrix.
pub fn scale(sx: f32, sy: f32, sz: f32) -> [[f32; 4]; 4] {
    [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_raw_size_matches_const() {
        assert_eq!(
            std::mem::size_of::<InstanceRaw>() as u64,
            InstanceRaw::STRIDE
        );
    }

    #[test]
    fn instance_raw_round_trips_through_from() {
        let inst = Instance {
            mesh: MeshHandle(7),
            model: translation(1.0, 2.0, 3.0),
            tint: [0.5, 0.25, 1.0, 1.0],
        };
        let raw = InstanceRaw::from(&inst);
        assert_eq!(raw.model_col3, [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(raw.tint, inst.tint);
    }
}
