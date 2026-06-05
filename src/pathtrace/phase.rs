//! CPU mirror of the Henyey-Greenstein phase function helpers in
//! `pathtrace.wgsl`.
//!
//! Same role as [`super::ggx`] / [`super::dielectric`]: keeps the
//! analytic formulas pinned with tests so the WGSL side can't quietly
//! drift. Future CPU reference integrators can lean on these exact
//! functions instead of re-deriving them.
//!
//! `g = 0` collapses to isotropic; positive `g` peaks forward
//! (cos θ → 1), negative peaks backward. The phase function IS
//! its own importance-sampling pdf — zonal symmetry collapses the
//! problem to a 1-D cosine inversion.

/// `1 / (4π)`.
pub const ISOTROPIC: f32 = 0.079_577_47;

/// Henyey-Greenstein pdf at scattering-angle cosine `cos_theta`,
/// anisotropy `g`. `g = 0` short-circuits to the isotropic value.
pub fn eval(cos_theta: f32, g: f32) -> f32 {
    if g.abs() < 1e-4 {
        return ISOTROPIC;
    }
    let denom = 1.0 + g * g - 2.0 * g * cos_theta;
    (1.0 - g * g) / (4.0 * std::f32::consts::PI * denom * denom.max(1e-30).sqrt())
}

/// Inverse-CDF sample of the HG cosine given a uniform `xi ∈ [0, 1)`.
/// Matches the WGSL formula byte-for-byte. Returns `cos_theta`.
pub fn sample_cos_theta(g: f32, xi: f32) -> f32 {
    if g.abs() < 1e-4 {
        return 1.0 - 2.0 * xi;
    }
    let sqr = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi);
    ((1.0 + g * g - sqr * sqr) / (2.0 * g)).clamp(-1.0, 1.0)
}
