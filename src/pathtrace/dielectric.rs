//! CPU mirror of the smooth-dielectric helpers in `pathtrace.wgsl`.
//!
//! Same role as [`super::ggx`]: keeps the analytic formulas
//! testable from Rust, and gives a future CPU reference integrator
//! something to lean on. Every function here matches its WGSL
//! counterpart byte-for-byte — divergence is what the tests in
//! `tests/dielectric.rs` are pinning down.
//!
//! Conventions match [`crate::pathtrace::ggx`]: `[f32; 3]` arrays,
//! no linalg crate. `wo` and `n` are unit vectors pointing into the
//! incident half-space (the convention `record_hit` settles on).

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Unpolarised Fresnel reflectance for a dielectric interface.
/// `cos_theta_i` is the cosine of the angle between the incident
/// direction and the normal (clamped to `[0, 1]`). Returns 1.0 on
/// total internal reflection (no refraction possible).
pub fn fresnel(cos_theta_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    let cti = cos_theta_i.clamp(0.0, 1.0);
    let eta_ratio = eta_i / eta_t;
    let sin_t2 = eta_ratio * eta_ratio * (1.0 - cti * cti).max(0.0);
    if sin_t2 >= 1.0 {
        return 1.0;
    }
    let cos_theta_t = (1.0 - sin_t2).sqrt();
    let r_par = (eta_t * cti - eta_i * cos_theta_t) / (eta_t * cti + eta_i * cos_theta_t);
    let r_perp = (eta_i * cti - eta_t * cos_theta_t) / (eta_i * cti + eta_t * cos_theta_t);
    0.5 * (r_par * r_par + r_perp * r_perp)
}

/// Snell-refracted direction. `wo` and `n` are unit vectors both
/// pointing into the incident side; the result also points into the
/// **transmitted** side (i.e., on the opposite hemisphere from `n`).
/// Returns `None` on total internal reflection — the WGSL signals
/// this by returning a length-zero vector; Rust uses `Option` for a
/// cleaner test surface.
pub fn refract(wo: [f32; 3], n: [f32; 3], eta_i: f32, eta_t: f32) -> Option<[f32; 3]> {
    let cos_i = dot3(n, wo);
    let eta_ratio = eta_i / eta_t;
    let sin_t2 = eta_ratio * eta_ratio * (1.0 - cos_i * cos_i).max(0.0);
    if sin_t2 >= 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin_t2).sqrt();
    let k = eta_ratio * cos_i - cos_t;
    Some([
        -wo[0] * eta_ratio + n[0] * k,
        -wo[1] * eta_ratio + n[1] * k,
        -wo[2] * eta_ratio + n[2] * k,
    ])
}

/// Critical angle beyond which TIR happens (incidence in the denser
/// medium). Returns `None` when `eta_i <= eta_t` (no TIR possible —
/// light is always refracting *into* a denser medium).
pub fn critical_angle(eta_i: f32, eta_t: f32) -> Option<f32> {
    if eta_i <= eta_t {
        return None;
    }
    Some((eta_t / eta_i).asin())
}
