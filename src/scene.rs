//! Cornell Box scene and the GPU-packed uniform layout shared with the WGSL
//! path tracer.
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
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub camera: GpuCamera,
    pub quad_count: u32,
    pub frame_count: u32,
    pub light_index: u32,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub _pad: [u32; 3],
    pub quads: [GpuQuad; MAX_QUADS],
    pub materials: [GpuMaterial; MAX_QUADS],
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
