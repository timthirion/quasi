// Tonemap the accumulated HDR estimate to the surface: Reinhard + gamma. We
// target a non-sRGB surface format and encode gamma here, so the result is
// correct regardless of whether an sRGB surface format was available.
//
// PT-adaptive (plan 0028) — display mode selector:
//   mode = 0 → radiance (default; Reinhard + gamma)
//   mode = 1 → variance map (log-scale viridis colour-map of the
//             per-pixel luminance standard deviation, same colour-map
//             that the offscreen `<base>_variance.png` ships).

struct PresentU {
    // 0 = radiance display, 1 = variance display.
    display_mode: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0) var accum_tex: texture_2d<f32>;
@group(0) @binding(1) var accum_my2: texture_2d<f32>;
@group(0) @binding(2) var<uniform> P: PresentU;

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

// 5-point piecewise-linear approximation of Matplotlib viridis,
// matching `viridis_lut` in src/pathtrace/output.rs so the
// on-screen variance display agrees with the saved variance PNG.
fn viridis(t: f32) -> vec3<f32> {
    let cp0 = vec3<f32>(0.267, 0.005, 0.329);
    let cp1 = vec3<f32>(0.282, 0.140, 0.457);
    let cp2 = vec3<f32>(0.220, 0.448, 0.535);
    let cp3 = vec3<f32>(0.500, 0.751, 0.230);
    let cp4 = vec3<f32>(0.993, 0.906, 0.144);
    let tc = clamp(t, 0.0, 1.0);
    let s = tc * 4.0;
    if (s < 1.0) { return mix(cp0, cp1, s); }
    if (s < 2.0) { return mix(cp1, cp2, s - 1.0); }
    if (s < 3.0) { return mix(cp2, cp3, s - 2.0); }
    return mix(cp3, cp4, s - 3.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.position.xy);
    let hdr = textureLoad(accum_tex, coord, 0).rgb;

    if (P.display_mode == 1u) {
        let y = dot(hdr, vec3<f32>(0.2126, 0.7152, 0.0722));
        let e_y2 = textureLoad(accum_my2, coord, 0).r;
        let var_y = max(e_y2 - y * y, 0.0);
        let std_y = sqrt(var_y);
        // Log-scale clamp to [1e-3, 1e0], same as output.rs.
        let lo = 1e-3;
        let log_v = log(max(std_y, lo)) / log(10.0);
        let mapped = clamp((log_v + 3.0) / 3.0, 0.0, 1.0);
        return vec4<f32>(viridis(mapped), 1.0);
    }

    let mapped = hdr / (hdr + vec3<f32>(1.0)); // Reinhard
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, 1.0);
}
