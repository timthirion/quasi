//! Analytic Cornell Box description + the path tracer's scene uniform.
//!
//! T0/M0-M3 used analytic quads packed into a fat uniform buffer. T1
//! moves geometry to storage buffers loaded from glTF
//! ([`crate::pathtrace::mesh`]), and `Uniforms` shrinks to just the
//! camera + a few scalars. The analytic Cornell description below
//! survives because [`examples/gen_cornell.rs`] uses it to produce the
//! triangulated glTF files in `data/gltf/`.
//!
//! The `Gpu*` structs are laid out to match WGSL uniform alignment: each
//! `vec3<f32>` lands on a 16-byte boundary (the trailing scalar in every
//! quartet provides the padding), so the same bytes are valid on both sides.

use bytemuck::{Pod, Zeroable};

pub const MAX_QUADS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuCamera {
    pub position: [f32; 3],
    pub fov: f32,
    pub direction: [f32; 3],
    pub aspect: f32,
    pub up: [f32; 3],
    pub _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuQuad {
    pub origin: [f32; 3],
    pub _p0: f32,
    pub u: [f32; 3],
    pub _p1: f32,
    pub v: [f32; 3],
    pub _p2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuMaterial {
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub emission: [f32; 3],
    pub metallic: f32,
    /// Index of refraction. `0.0` = "not a dielectric"; non-zero
    /// routes the BSDF onto the smooth-glass branch in `pathtrace.wgsl`.
    /// Kept out of the in-scene `cornell_box()` palette (all matte
    /// walls), but written by the example glTF emitter when set.
    pub ior: f32,
    /// Beer-Lambert absorption coefficient. `(0, 0, 0)` = clear.
    /// Cornell walls leave this zero; coloured-glass scenes set it.
    pub absorption: [f32; 3],
    /// Scattering coefficient. `(0, 0, 0)` = no scattering.
    /// Fog / smoke scenes set this; closed-form Beer-Lambert glass
    /// leaves it zero.
    pub scattering: [f32; 3],
}

/// Camera + scalars uniform — the only data the WGSL shader still reads
/// out of a uniform buffer in T1. Triangle geometry and materials live
/// in storage buffers bound alongside this uniform.
///
/// Layout: 80 bytes (camera 48 + 8 × u32 = 80). Must match `Uniforms`
/// in `pathtrace.wgsl` byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub camera: GpuCamera,
    pub triangle_count: u32,
    pub emissive_count: u32,
    pub frame_count: u32,
    pub viewport_width: u32,
    pub viewport_height: u32,
    /// Discriminant of [`crate::pathtrace::sampler::SamplerKind`]. The WGSL
    /// side reads it directly to dispatch between PCG / Halton / Sobol.
    pub sampler_kind: u32,
    /// Discriminant of [`crate::pathtrace::integrator::IntegratorKind`].
    /// Switches the WGSL path tracer between MIS+NEE and pure BSDF.
    pub integrator_kind: u32,
    /// 1 = walk the BVH (T2 default), 0 = linear scan (verification
    /// path retained behind `render --brute-force`).
    pub use_bvh: u32,
}

/// CPU description of the Cornell Box.
pub struct Scene {
    pub quads: Vec<GpuQuad>,
    pub materials: Vec<GpuMaterial>,
    pub light_index: u32,
}

type V3 = [f32; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn rotate_y(p: V3, deg: f32) -> V3 {
    let r = deg * std::f32::consts::PI / 180.0;
    let (s, c) = r.sin_cos();
    [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c]
}

fn quad(origin: V3, u: V3, v: V3) -> GpuQuad {
    GpuQuad {
        origin,
        _p0: 0.0,
        u,
        _p1: 0.0,
        v,
        _p2: 0.0,
    }
}

fn mat(albedo: V3, emission: V3) -> GpuMaterial {
    GpuMaterial {
        albedo,
        roughness: 1.0,
        emission,
        metallic: 0.0,
        ior: 0.0,
        absorption: [0.0, 0.0, 0.0],
        scattering: [0.0, 0.0, 0.0],
    }
}

/// Adds a box (5 visible faces, bottom omitted) rotated about Y, matching the
/// reference Cornell Box layout.
fn add_box(scene: &mut Scene, center: V3, size: V3, angle_y: f32, m: GpuMaterial) {
    let hw = size[0] * 0.5;
    let h = size[1];
    let hd = size[2] * 0.5;

    let local = [
        [-hw, 0.0, -hd],
        [hw, 0.0, -hd],
        [-hw, 0.0, hd],
        [hw, 0.0, hd],
        [-hw, h, -hd],
        [hw, h, -hd],
        [-hw, h, hd],
        [hw, h, hd],
    ];
    let mut p = [[0.0f32; 3]; 8];
    for i in 0..8 {
        p[i] = add(rotate_y(local[i], angle_y), center);
    }

    let faces = [
        (p[4], sub(p[5], p[4]), sub(p[6], p[4])), // top
        (p[0], sub(p[1], p[0]), sub(p[4], p[0])), // front
        (p[3], sub(p[2], p[3]), sub(p[7], p[3])), // back
        (p[2], sub(p[0], p[2]), sub(p[6], p[2])), // left
        (p[1], sub(p[3], p[1]), sub(p[5], p[1])), // right
    ];
    for (o, u, v) in faces {
        scene.quads.push(quad(o, u, v));
        scene.materials.push(m);
    }
}

/// Builds the standard Cornell Box: walls, ceiling light, and two boxes.
pub fn cornell_box() -> Scene {
    let white = mat([0.73, 0.73, 0.73], [0.0, 0.0, 0.0]);
    let red = mat([0.65, 0.05, 0.05], [0.0, 0.0, 0.0]);
    let green = mat([0.12, 0.45, 0.15], [0.0, 0.0, 0.0]);
    let light = mat([0.0, 0.0, 0.0], [15.0, 15.0, 15.0]);

    let mut scene = Scene {
        quads: Vec::new(),
        materials: Vec::new(),
        light_index: 0,
    };

    let push = |s: &mut Scene, q: GpuQuad, m: GpuMaterial| {
        s.quads.push(q);
        s.materials.push(m);
    };

    push(
        &mut scene,
        quad([-1.0, 0.0, -1.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]),
        white,
    ); // floor
    push(
        &mut scene,
        quad([-1.0, 2.0, 1.0], [2.0, 0.0, 0.0], [0.0, 0.0, -2.0]),
        white,
    ); // ceiling
    push(
        &mut scene,
        quad([-1.0, 0.0, -1.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]),
        white,
    ); // back
    push(
        &mut scene,
        quad([-1.0, 0.0, 1.0], [0.0, 0.0, -2.0], [0.0, 2.0, 0.0]),
        red,
    ); // left
    push(
        &mut scene,
        quad([1.0, 0.0, -1.0], [0.0, 0.0, 2.0], [0.0, 2.0, 0.0]),
        green,
    ); // right

    scene.light_index = scene.quads.len() as u32;
    push(
        &mut scene,
        quad([-0.25, 1.99, -0.25], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]),
        light,
    );

    add_box(&mut scene, [-0.35, 0.0, 0.3], [0.5, 1.2, 0.5], 15.0, white);
    add_box(
        &mut scene,
        [0.35, 0.0, -0.3],
        [0.55, 0.55, 0.55],
        -18.0,
        white,
    );

    scene
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // These sizes/offsets MUST match the WGSL `Uniforms` layout in
    // shaders/pathtrace.wgsl. A mismatch fails only at runtime ("buffer too
    // small"), so this test is the off-GPU guard for that class of bug.
    #[test]
    fn gpu_struct_layout_matches_wgsl() {
        assert_eq!(size_of::<GpuCamera>(), 48);
        assert_eq!(size_of::<GpuQuad>(), 48);
        // 32 bytes pre-PT-dielectrics; PT-dielectrics added `ior` (→ 36);
        // PT-beer-lambert added `absorption` (→ 48); PT-fog added
        // `scattering` (→ 60). The runtime `Material` in `mesh.rs`
        // lives separately and stays std430-padded at 80 bytes.
        assert_eq!(size_of::<GpuMaterial>(), 60);
        // camera (48) + 8 × u32 = 80. T1 dropped the per-quad arrays;
        // triangle data lives in storage buffers now. T2 added
        // `use_bvh` in what used to be the trailing pad slot.
        assert_eq!(size_of::<Uniforms>(), 80);
        assert_eq!(offset_of!(Uniforms, triangle_count), 48);
        assert_eq!(offset_of!(Uniforms, integrator_kind), 48 + 6 * 4);
        assert_eq!(offset_of!(Uniforms, use_bvh), 48 + 7 * 4);
    }

    #[test]
    fn cornell_box_geometry() {
        let s = cornell_box();
        // 5 walls + 1 light + 2 boxes * 5 visible faces.
        assert_eq!(s.quads.len(), 16);
        assert_eq!(s.quads.len(), s.materials.len());
        assert!(s.quads.len() <= MAX_QUADS);
        assert!((s.light_index as usize) < s.quads.len());
    }

    #[test]
    fn only_the_light_is_emissive() {
        let s = cornell_box();
        for (i, m) in s.materials.iter().enumerate() {
            let emissive = m.emission.iter().any(|&e| e > 0.0);
            assert_eq!(
                emissive,
                i == s.light_index as usize,
                "material {i} emissive={emissive} but light_index={}",
                s.light_index
            );
        }
    }

    #[test]
    fn rotate_y_quarter_turn() {
        // +X rotated 90° about Y lands on -Z.
        let r = rotate_y([1.0, 0.0, 0.0], 90.0);
        assert!(r[0].abs() < 1e-5);
        assert!(r[1].abs() < 1e-5);
        assert!((r[2] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn add_and_sub_basics() {
        assert_eq!(add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), [5.0, 7.0, 9.0]);
        assert_eq!(sub([3.0, 2.0, 1.0], [1.0, 1.0, 1.0]), [2.0, 1.0, 0.0]);
    }
}
