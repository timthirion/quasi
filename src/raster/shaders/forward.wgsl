//! Forward shading: lambert from a single directional sun plus a flat
//! ambient term. Vertex colors modulate the lit result.

struct FrameU {
    view_proj: mat4x4<f32>,
    sun_dir: vec3<f32>,
    _pad0: f32,
    sun_color: vec3<f32>,
    _pad1: f32,
    ambient: vec3<f32>,
    _pad2: f32,
}
@group(0) @binding(0) var<uniform> frame: FrameU;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec3<f32>,
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = frame.view_proj * vec4<f32>(in.position, 1.0);
    out.world_normal = in.normal;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    // `sun_dir` points from the sun toward the scene, so light contribution
    // is dot(normal, -sun_dir).
    let lambert = max(dot(n, -frame.sun_dir), 0.0);
    let lit = in.color * (frame.ambient + lambert * frame.sun_color);
    // Gamma-encode for non-sRGB swapchain (matches pathtrace's present).
    let gamma = pow(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, 1.0);
}
