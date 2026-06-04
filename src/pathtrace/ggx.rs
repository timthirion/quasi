//! CPU mirror of the GGX microfacet helpers in `pathtrace.wgsl`.
//!
//! Kept Rust-side for two reasons. First, the tests in
//! `tests/ggx.rs` can hand the formulas the same inputs the GPU sees
//! and check normalisation by Monte-Carlo integration — keeping us
//! honest that the shader didn't drift from the analytic identities.
//! Second, when the path tracer eventually gains a CPU reference
//! integrator (PT-cpu-ref, on the roadmap), it can lean on these
//! exact functions instead of re-deriving them.
//!
//! Every formula here MUST stay byte-identical with the WGSL side —
//! including the `GGX_MIN_ALPHA` clamp. The shader-validation tests
//! pin the constants, and `tests/ggx.rs` pins the math.
//!
//! Vectors stay as `[f32; 3]` arrays to avoid pulling in a linalg
//! crate just for three dot products and a cross — the path tracer
//! proper is GPU-side anyway.

/// Minimum value `alpha²` is clamped to in [`alpha`]. Matches the
/// WGSL `GGX_MIN_ALPHA` constant. Stops the δ-spike at
/// `roughness = 0` from blowing up the importance-sampling pdf.
pub const MIN_ALPHA_SQUARED: f32 = 0.0064;

/// Maps a glTF `roughnessFactor` (perceptual roughness in `[0, 1]`)
/// to the GGX `alpha² = roughness⁴` parameter, clamped from below.
/// The shader-side helper squares `roughness` after clamping; this
/// returns the squared form directly.
pub fn alpha(roughness: f32) -> f32 {
    let r = roughness.max(MIN_ALPHA_SQUARED.sqrt());
    r * r
}

/// Trowbridge-Reitz GGX normal distribution function.
/// `n_dot_h ∈ [0, 1]`; `alpha = roughness²`. Returns `D(h)` such that
/// `∫ D(h) · (n · h) dω_h = 1` over the unit hemisphere (the
/// integration test exercises this identity).
pub fn d(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (std::f32::consts::PI * denom * denom)
}

/// Smith separable masking function, one direction.
pub fn smith_g1(n_dot_x: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = n_dot_x + (a2 + (1.0 - a2) * n_dot_x * n_dot_x).sqrt();
    2.0 * n_dot_x / denom.max(1e-8)
}

/// Smith separable masking-shadowing — product of `smith_g1` on both
/// directions. Cheap, slightly biased relative to a coupled-height
/// model, but matches the most common real-time PBR convention.
pub fn smith_g(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    smith_g1(n_dot_v, alpha) * smith_g1(n_dot_l, alpha)
}

/// Schlick conductor Fresnel. `f0` is the base reflectance at normal
/// incidence — for metals we feed in `Material::albedo` directly
/// (the PBR convention).
pub fn schlick_fresnel(v_dot_h: f32, f0: [f32; 3]) -> [f32; 3] {
    let s = (1.0 - v_dot_h).powi(5);
    [
        f0[0] + (1.0 - f0[0]) * s,
        f0[1] + (1.0 - f0[1]) * s,
        f0[2] + (1.0 - f0[2]) * s,
    ]
}

/// Solid-angle pdf of a GGX importance-sampled outgoing direction.
/// `pdf_l = D(h) · (n · h) / (4 |v · h|)`.
pub fn pdf(n_dot_h: f32, v_dot_h: f32, alpha: f32) -> f32 {
    d(n_dot_h, alpha) * n_dot_h / (4.0 * v_dot_h.max(1e-6))
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = dot3(v, v).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Importance-sample a half vector from `D(h) · (n · h)`, given two
/// canonical uniforms `(u, v) ∈ [0, 1)²`. Returns the world-space
/// half vector aligned with `normal`.
///
/// Matches the WGSL `sample_ggx_half` byte-for-byte, including the
/// basis-vector choice (swaps the up axis when the normal is close
/// to the world `±x`).
pub fn sample_half(normal: [f32; 3], alpha: f32, u: f32, v: f32) -> [f32; 3] {
    let a2 = alpha * alpha;
    let cos_theta_2 = (1.0 - u) / (u * (a2 - 1.0) + 1.0);
    let cos_theta = cos_theta_2.max(0.0).sqrt();
    let sin_theta = (1.0 - cos_theta_2).max(0.0).sqrt();
    let phi = 2.0 * std::f32::consts::PI * v;

    let nrm = normalize3(normal);
    let a = if nrm[0].abs() > 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let vv = normalize3(cross3(nrm, a));
    let uu = cross3(nrm, vv);
    let cp = phi.cos();
    let sp = phi.sin();
    normalize3([
        uu[0] * cp * sin_theta + vv[0] * sp * sin_theta + nrm[0] * cos_theta,
        uu[1] * cp * sin_theta + vv[1] * sp * sin_theta + nrm[1] * cos_theta,
        uu[2] * cp * sin_theta + vv[2] * sp * sin_theta + nrm[2] * cos_theta,
    ])
}

/// Dot product, exposed for tests that want to compute `n · h` after
/// `sample_half`.
pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    dot3(a, b)
}
