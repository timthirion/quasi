// PT-bloom (plan 0029): final composite — radiance + intensity * bloom.
//
// Reads the accumulated radiance via `textureLoad` (Rgba32Float
// isn't filterable without an opt-in feature; we don't need
// filtering for a 1:1 pixel-for-pixel composite anyway) and the
// bloom mip 0 (which, after the upsample chain, holds the full
// bloom contribution at the frame's native resolution). Outputs
// `radiance + intensity * bloom` so the subsequent CPU tonemap
// sees the bloomed image.
//
// Runs as a normal opaque render pass — we read from one ping-
// pong slot and write to the other, then bump `read_idx` so the
// readback uses the bloomed result.

struct CompositeU {
    intensity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var radiance_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> U: CompositeU;

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
    let r = textureLoad(radiance_tex, coord, 0).rgb;
    let b = textureLoad(bloom_tex, coord, 0).rgb;
    return vec4<f32>(r + U.intensity * b, 1.0);
}
