//! Forward shading with per-instance model transforms.
//!
//! Vertex attributes:
//!   loc 0..2  position / normal / color  (vertex-stepped, the mesh's data)
//!   loc 3..6  model matrix columns       (instance-stepped)
//!   loc 7     tint                       (instance-stepped, rgba)
//!
//! Lighting: a single directional sun + flat ambient, modulated by
//! `vertex_color * instance_tint`. Gamma-encoded for a non-sRGB swapchain.

struct FrameU {
    view_proj: mat4x4<f32>,
    sun_dir:   vec3<f32>,
    _pad0:     f32,
    sun_color: vec3<f32>,
    _pad1:     f32,
    ambient:   vec3<f32>,
    _pad2:     f32,
}
@group(0) @binding(0) var<uniform> frame: FrameU;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec3<f32>,
    @location(3) m0: vec4<f32>,
    @location(4) m1: vec4<f32>,
    @location(5) m2: vec4<f32>,
    @location(6) m3: vec4<f32>,
    @location(7) tint: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let world = model * vec4<f32>(in.position, 1.0);
    // For uniform-scale model matrices this is enough. Non-uniform scales
    // would want the transposed inverse for normals; we don't ship those.
    let world_normal = (model * vec4<f32>(in.normal, 0.0)).xyz;

    var out: VsOut;
    out.clip_pos = frame.view_proj * world;
    out.world_normal = world_normal;
    out.color = in.color * in.tint.rgb;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let lambert = max(dot(n, -frame.sun_dir), 0.0);
    let lit = in.color * (frame.ambient + lambert * frame.sun_color);
    let gamma = pow(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, 1.0);
}
