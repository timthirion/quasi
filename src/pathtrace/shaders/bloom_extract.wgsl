// PT-bloom (plan 0029): soft-knee threshold extract pass.
//
// Reads the accumulated radiance via `textureLoad` (1:1 with the
// output mip-0; the Rgba32Float radiance texture isn't filterable
// in WebGPU without the optional `float32-filterable` feature, so
// integer-pixel reads keep us portable) and applies the Unity-
// correct soft-knee formula to isolate "bright" pixels. Writes
// them to the bloom mip chain's level 0.
//
// Pixels below `threshold - knee` are zeroed (no contribution to
// bloom); pixels above `threshold` pass through scaled by
// `(brightness - threshold) / brightness`; pixels in the knee
// region get a smooth quadratic ramp.
//
// Crucially: a sub-threshold pixel's weight is **clamped to 0**.
// An earlier draft of this shader (rev-1) had a `min(curve, b -
// threshold)` formulation that allowed negative weights for
// sub-threshold pixels — which would *subtract* dim radiance
// from the bloom chain and darken midtones near bright sources.
// The `max(weight, 0)` here is the load-bearing fix flagged by
// the plan-skeptic round.
//
// Source: Unity HDRP `Runtime/PostProcessing/Shaders/Builtins/
// Bloom.shader::fragPrefilter4`.

struct ExtractU {
    threshold: f32,
    knee: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> U: ExtractU;

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

fn soft_knee_extract(rgb: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    // Guard against firefly pixels (NaN, Inf, > 1e6) — these would
    // otherwise propagate through the entire mip chain and turn
    // every composited pixel into garbage.
    let is_finite = rgb.x == rgb.x && rgb.y == rgb.y && rgb.z == rgb.z;
    let is_bounded = all(rgb < vec3<f32>(1.0e6));
    let safe = select(vec3<f32>(0.0), rgb, is_finite && is_bounded);

    let brightness = max(safe.x, max(safe.y, safe.z));
    let b_safe = max(brightness, 1e-6);

    // Quadratic curve over [threshold - knee, threshold + knee]
    // (clamp ensures the knee window doesn't extend past it).
    let curve_x = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    let curve = curve_x * curve_x * 0.25 / max(knee, 1e-6);

    // Linear above threshold.
    let linear = brightness - threshold;

    // max-of-max-of-zero: below the knee both terms ≤ 0 → weight
    // 0. In knee: curve > 0, linear ≤ 0 → curve wins. Above:
    // linear dominates as brightness grows.
    let weight = max(max(curve, linear), 0.0) / b_safe;
    return safe * weight;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.position.xy);
    let rgb = textureLoad(src_tex, coord, 0).rgb;
    let extracted = soft_knee_extract(rgb, U.threshold, U.knee);
    return vec4<f32>(extracted, 1.0);
}
