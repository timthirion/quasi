// Progressive accumulation across five AOV channels:
//   @location(0) radiance, @location(1) albedo,
//   @location(2) normal,   @location(3) depth,
//   @location(4) mean_y2   (PT-adaptive, plan 0028 — running mean of
//                          luminance² for per-pixel variance derivation).
//
// Each channel is a weighted running average of (prev, new). textureLoad
// with integer pixel coords keeps every AOV pixel-aligned (no samplers,
// no flips, no edge tap).

struct AccumU {
    frame_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> A: AccumU;

@group(0) @binding(1) var sample_rad: texture_2d<f32>;
@group(0) @binding(2) var sample_alb: texture_2d<f32>;
@group(0) @binding(3) var sample_nor: texture_2d<f32>;
@group(0) @binding(4) var sample_dep: texture_2d<f32>;
@group(0) @binding(5) var sample_my2: texture_2d<f32>;

@group(0) @binding(6) var prev_rad: texture_2d<f32>;
@group(0) @binding(7) var prev_alb: texture_2d<f32>;
@group(0) @binding(8) var prev_nor: texture_2d<f32>;
@group(0) @binding(9) var prev_dep: texture_2d<f32>;
@group(0) @binding(10) var prev_my2: texture_2d<f32>;

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

struct AccumOut {
    @location(0) rad: vec4<f32>,
    @location(1) alb: vec4<f32>,
    @location(2) nor: vec4<f32>,
    @location(3) dep: vec4<f32>,
    @location(4) my2: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut) -> AccumOut {
    let coord = vec2<i32>(in.position.xy);
    let w = 1.0 / f32(A.frame_count + 1u);

    var out: AccumOut;
    out.rad = mix(textureLoad(prev_rad, coord, 0), textureLoad(sample_rad, coord, 0), w);
    out.alb = mix(textureLoad(prev_alb, coord, 0), textureLoad(sample_alb, coord, 0), w);
    out.nor = mix(textureLoad(prev_nor, coord, 0), textureLoad(sample_nor, coord, 0), w);
    out.dep = mix(textureLoad(prev_dep, coord, 0), textureLoad(sample_dep, coord, 0), w);
    out.my2 = mix(textureLoad(prev_my2, coord, 0), textureLoad(sample_my2, coord, 0), w);
    return out;
}
