// PT-bloom (plan 0029): Kawase 4-tap downsample.
//
// Bilinear-aware 4-corner sampling at ±0.5 source-texel offsets
// effectively gives a 4×4 box average (each `textureSample` is
// itself a 4-tap bilinear filter). Cheap and energy-preserving.
//
// Source: Marius Bjørge, "Bandwidth-Efficient Rendering"
// (SIGGRAPH 2015), §3.4 dual-filter blur.

struct DownsampleU {
    src_texel: vec2<f32>, // 1.0 / src_dimensions
    _pad: vec2<f32>,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> U: DownsampleU;

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
    let t = U.src_texel;
    let s0 = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(-t.x, -t.y)).rgb;
    let s1 = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(t.x, -t.y)).rgb;
    let s2 = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(-t.x, t.y)).rgb;
    let s3 = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(t.x, t.y)).rgb;
    let avg = (s0 + s1 + s2 + s3) * 0.25;
    return vec4<f32>(avg, 1.0);
}
