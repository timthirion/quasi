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
