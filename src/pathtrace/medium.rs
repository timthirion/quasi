//! CPU mirror of the participating-media helpers in `pathtrace.wgsl`.
//!
//! For PT-beer-lambert the entire "medium model" is the closed-form
//! solution to the radiative transfer equation with no scattering
//! and a constant absorption coefficient: `T = exp(-σ_a · t)`. The
//! function is trivial, but the tests in `tests/medium.rs` pin a
//! handful of identities the WGSL side relies on (zero σ → identity,
//! chain rule across consecutive segments, etc.) so we don't lose
//! the math in future refactors.
//!
//! PT-fog will add inverse-CDF distance sampling and a phase
//! function; both land in this module.

/// Beer-Lambert transmittance for a single segment. Per-channel
/// absorption `sigma_a`; segment length `t` in world units. Returned
/// vector lives in `[0, 1]³`.
pub fn attenuation(sigma_a: [f32; 3], t: f32) -> [f32; 3] {
    [
        (-sigma_a[0] * t).exp(),
        (-sigma_a[1] * t).exp(),
        (-sigma_a[2] * t).exp(),
    ]
}

/// Sample a distance from an exponential medium with scalar
/// extinction `sigma_t`. The pdf is `σ_t · exp(-σ_t · t)`, so
/// `E[t] = 1 / σ_t` (the mean free path). Inverse-CDF of the
/// exponential with `xi ∈ [0, 1)`.
///
/// Used in `pathtrace.wgsl`'s `sample_volume_distance` against a
/// per-channel σ_t majorant (`max(σ_t.x, σ_t.y, σ_t.z)`); per-
/// channel transmittance correction rides on the returned weight.
pub fn sample_distance(sigma_t: f32, xi: f32) -> f32 {
    let u = xi.clamp(0.0, 1.0 - 1e-7);
    -(1.0 - u).ln() / sigma_t.max(1e-30)
}

/// Scalar extinction `σ_t = σ_a + σ_s`, per channel.
pub fn extinction(sigma_a: [f32; 3], sigma_s: [f32; 3]) -> [f32; 3] {
    [
        sigma_a[0] + sigma_s[0],
        sigma_a[1] + sigma_s[1],
        sigma_a[2] + sigma_s[2],
    ]
}
