// Progressive accumulation: weighted average of the new sample into the running
// estimate. textureLoad with integer pixel coords keeps all passes pixel-aligned
// (no sampler, no flips).

struct AccumU {
    frame_count: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0) var<uniform> A: AccumU;
@group(0) @binding(1) var sample_tex: texture_2d<f32>;
@group(0) @binding(2) var accum_prev: texture_2d<f32>;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    out.uv = uv;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.position.xy);
    let s = textureLoad(sample_tex, coord, 0);
    let p = textureLoad(accum_prev, coord, 0);
    let weight = 1.0 / f32(A.frame_count + 1u);
    return mix(p, s, weight);
}
