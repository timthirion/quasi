// Tonemap the accumulated HDR estimate to the surface: Reinhard + gamma. We
// target a non-sRGB surface format and encode gamma here, so the result is
// correct regardless of whether an sRGB surface format was available.

@group(0) @binding(0) var accum_tex: texture_2d<f32>;

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
    let hdr = textureLoad(accum_tex, coord, 0).rgb;
    let mapped = hdr / (hdr + vec3<f32>(1.0)); // Reinhard
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, 1.0);
}
