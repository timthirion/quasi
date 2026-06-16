// PT-bloom (plan 0029): 9-tap tent-kernel upsample.
//
// Samples a 3×3 neighborhood with tent weights (1,2,1; 2,4,2;
// 1,2,1) / 16. Combined with the downsample chain, this produces
// a smooth Gaussian-approximation kernel whose effective radius
// grows with each upsample level. The composite of all upsample
// levels is a sum of Gaussians at varying scales — matching how
// real lens-flare PSFs decompose.
//
// Rendered with additive blending into the destination mip (the
// previously-downsampled level) so each pass contributes its
// scale's bloom energy. The blend state is configured at pipeline
// creation; this shader just outputs the per-pixel contribution.

struct UpsampleU {
    src_texel: vec2<f32>, // 1.0 / src_dimensions
    _pad: vec2<f32>,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> U: UpsampleU;

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

    // 9-tap tent kernel sampled at integer texel offsets.
    let c = textureSample(src_tex, src_sampler, in.uv).rgb;
    let n = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(0.0, -t.y)).rgb;
    let s = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(0.0, t.y)).rgb;
    let e = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(t.x, 0.0)).rgb;
    let w = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(-t.x, 0.0)).rgb;
    let ne = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(t.x, -t.y)).rgb;
    let nw = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(-t.x, -t.y)).rgb;
    let se = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(t.x, t.y)).rgb;
    let sw = textureSample(src_tex, src_sampler, in.uv + vec2<f32>(-t.x, t.y)).rgb;

    // Tent weights: corners 1, edges 2, center 4. Sum = 16.
    let sum = c * 4.0 + (n + s + e + w) * 2.0 + (ne + nw + se + sw);
    return vec4<f32>(sum / 16.0, 1.0);
}
