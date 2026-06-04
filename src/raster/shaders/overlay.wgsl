//! Overlay primitives (lines + points) for the rasterized pipeline.
//!
//! Shares the forward pipeline's `FrameU` uniform layout — only the
//! view-projection matrix is actually read here; the lighting fields
//! are inert but pinned at the same offsets so one bind group covers
//! both shaders.
//!
//! Vertex attributes match `OverlayVertex` on the CPU side:
//!   loc 0  position: vec3<f32>  (stride 12, offset 0)
//!   loc 1  color:    vec4<f32>  (stride 16, offset 12)
//! Total per-vertex stride: 28 bytes.
//!
//! Topology is selected on the host side per pipeline
//! (`LineList` for lines, `PointList` for points). The shader itself is
//! topology-agnostic — same VS / FS for both.
//!
//! Gamma is applied here for parity with `forward.wgsl`, since the
//! swapchain is non-sRGB.

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
    @location(1) color:    vec4<f32>,
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = frame.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let gamma = pow(clamp(in.color.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, in.color.a);
}
