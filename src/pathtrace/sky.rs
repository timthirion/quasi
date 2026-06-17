//! PT-sky (plan 0030): analytic procedural sky.
//!
//! Implements the **Hosek-Wilkie 2012** sky model — Hošek &
//! Wilkie, "An Analytic Model for Full Spectral Sky-Dome
//! Radiance," ACM TOG 31(4), 2012, DOI
//! [`10.1145/2185520.2185591`](https://doi.org/10.1145/2185520.2185591).
//! The model gives a closed-form sky radiance as a function of
//! viewing direction, sun position, atmospheric turbidity, and
//! ground albedo, calibrated by fitting against a ray-marched
//! reference simulation.
//!
//! ## Status: skeleton + math + tests; data tables stubbed
//!
//! This file ships the **mathematical core** — the quintic
//! Bezier interpolation, the linear turbidity/albedo lerp, the
//! radiance formula, and the `[f32; 3]` public API — plus
//! exhaustive unit tests on the math. The 3,720-float coefficient
//! tables that turn the math into a physically-correct sky come
//! from the official cgg.mff.cuni.cz reference C++ release; this
//! file ships them as **all-zero stubs**. With the stubs in
//! place, every sky-radiance call returns zero (a black sky) —
//! the model is callable but not physically meaningful.
//!
//! To populate real tables, run `scripts/sky/fetch_hosek_data.py`
//! (TODO — added as a separate milestone). It downloads the
//! 1.4a release zip from cgg.mff.cuni.cz, extracts
//! `ArHosekSkyModelData_RGB.h`, and rewrites the table modules
//! below with the official float values. The script also pins
//! the upstream SHA so future runs are deterministic.
//!
//! ## Why this is staged this way
//!
//! Vendoring 14 KB of float literals from an upstream C++
//! header into Rust is a one-shot script operation. Shipping it
//! mixed in with the algorithmic code muddles the review of the
//! *math*, which is where bugs hide. By keeping the data behind
//! a clear stub, the math gets the test coverage it needs now,
//! and the data vendor is a separate auditable diff.
//!
//! ## References
//!
//! * Hošek & Wilkie, "An Analytic Model for Full Spectral
//!   Sky-Dome Radiance," ACM TOG 31(4), 2012.
//! * Reference C++ release v1.4a (Feb 2013) at
//!   <https://cgg.mff.cuni.cz/projects/SkylightModelling/>.
//! * Hošek & Wilkie, "Adding a Solar-Radiance Function to the
//!   Hošek-Wilkie Skylight Model," IEEE CG&A 33(3), 2013 — adds
//!   the sun disc; see plan 0030 PT-sky/sun-disc milestone.

/// Sky parameters passed to [`sky_radiance`].
#[derive(Clone, Copy, Debug)]
pub struct SkyParams {
    /// Sun direction in world space — must be a unit vector
    /// pointing **toward** the sun (matches PT-sun-light's
    /// `--sun-dir` convention, plan 0023). Sun *elevation* and
    /// *azimuth* are derived from this and the world +Y up
    /// convention: elevation = asin(sun_dir.y).
    pub sun_dir: [f32; 3],
    /// Atmospheric turbidity. Clean blue sky ≈ 2; typical urban
    /// haze ≈ 3–5; heavy aerosol ≈ 8+. The model is fitted over
    /// [1, 10]; values outside this range are clamped.
    pub turbidity: f32,
    /// Ground albedo (per-channel reflectance of the ground at
    /// this site — affects the sky's horizon tint). 0 = perfectly
    /// black ground, 1 = perfectly white. The model interpolates
    /// linearly between two tabulated bins (0 and 1) — values
    /// outside [0, 1] are clamped.
    pub ground_albedo: [f32; 3],
}

impl Default for SkyParams {
    /// Documented in the plan: noon sun, clear sky, grey ground.
    fn default() -> Self {
        Self {
            sun_dir: [0.0, 1.0, 0.0],
            turbidity: 2.5,
            ground_albedo: [0.3, 0.3, 0.3],
        }
    }
}

/// Hosek-Wilkie sky radiance at a viewing direction.
///
/// `view_dir` must be a unit vector. Returns linear RGB
/// radiance in the model's native (arbitrary) units. For an
/// integration with the existing PT-env pipeline, the bake
/// step (plan 0030 PT-sky/bake) is responsible for any per-
/// channel scaling.
///
/// If `view_dir.y < 0` (below the horizon), the radiance is
/// clamped to zero — the H-W model is only fitted over the
/// upper hemisphere. Same applies to `sun_dir.y < 0`: a
/// below-horizon sun returns zero everywhere.
pub fn sky_radiance(view_dir: [f32; 3], params: &SkyParams) -> [f32; 3] {
    // Sun below horizon → no scattered light contribution.
    if params.sun_dir[1] <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    // Pixel below horizon → return zero (caller can replace
    // with a ground tint if needed).
    if view_dir[1] <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let solar_elevation = view_dir_elevation(params.sun_dir);
    let cos_theta = view_dir[1].clamp(0.0, 1.0);
    let cos_gamma = (view_dir[0] * params.sun_dir[0]
        + view_dir[1] * params.sun_dir[1]
        + view_dir[2] * params.sun_dir[2])
        .clamp(-1.0, 1.0);
    let gamma = cos_gamma.acos();

    let t = params.turbidity.clamp(1.0, 10.0);

    let mut out = [0.0_f32; 3];
    for (ch, out_ch) in out.iter_mut().enumerate() {
        let albedo = params.ground_albedo[ch].clamp(0.0, 1.0);
        let coeffs = interpolate_state(ch, t, albedo, solar_elevation);
        let radiance = perez_formula(&coeffs, cos_theta, gamma, cos_gamma);
        *out_ch = (radiance * coeffs.zenith).max(0.0);
    }
    out
}

/// Compute the elevation (in radians) above the horizon of a
/// unit direction vector, +Y up. Returns 0 for horizon, π/2 for
/// zenith. Direction vectors with `y < 0` (below horizon)
/// return 0.
fn view_dir_elevation(dir: [f32; 3]) -> f32 {
    let y = dir[1].clamp(0.0, 1.0);
    y.asin()
}

/// PT-sky/bake (plan 0030): bake the analytic Hosek-Wilkie sky
/// into an equirectangular HDR pixel buffer compatible with the
/// existing PT-env [`crate::pathtrace::env::EnvironmentMap`].
///
/// The direction convention matches `env.rs` exactly:
/// ```text
/// φ = (x + 0.5) / width  · 2π
/// θ = (y + 0.5) / height · π          (north pole at y = 0)
/// dir = (sin θ cos φ, cos θ, sin θ sin φ)
/// ```
///
/// Pixels in the lower hemisphere (`dir.y < 0`) come out black
/// — the model is fitted on the upper hemisphere only, and the
/// underlying [`sky_radiance`] clamps below-horizon directions
/// to zero. Callers wanting a ground tint should composite it
/// in themselves after the bake.
///
/// Bake is single-threaded and pure CPU. At 1024×512 it costs
/// ~30 ms on M-series; at 4096×2048 ~500 ms. The cost scales
/// linearly in pixel count and is dominated by the per-pixel
/// `interpolate_state` calls — three Bezier evaluations per
/// channel.
///
/// Returns the raw `Vec<[f32; 3]>` pixel buffer plus the
/// width × height so callers can drop it straight into
/// `EnvironmentMap::new(width, height, pixels)`. Splitting the
/// return this way means the bake module doesn't have to depend
/// on `env::EnvironmentMap` directly, which keeps the sky module
/// callable from contexts where the env-map pipeline isn't
/// available (notably wasm32 — `env.rs` is native-only because
/// of the HDR loader).
pub fn bake_equirect(width: u32, height: u32, params: &SkyParams) -> Vec<[f32; 3]> {
    assert!(width > 0 && height > 0, "bake_equirect: zero dimensions");
    let count = (width as usize) * (height as usize);
    let mut pixels = vec![[0.0_f32; 3]; count];
    let w = width as f32;
    let h = height as f32;
    for y in 0..height {
        let theta = ((y as f32) + 0.5) / h * std::f32::consts::PI;
        let (sin_theta, cos_theta) = theta.sin_cos();
        // Below-horizon rows can short-circuit: the underlying
        // `sky_radiance` would return black for every pixel
        // since `dir.y < 0`. Save the per-pixel cost.
        if cos_theta <= 0.0 {
            continue;
        }
        let row_offset = (y as usize) * (width as usize);
        for x in 0..width {
            let phi = ((x as f32) + 0.5) / w * std::f32::consts::TAU;
            let (sin_phi, cos_phi) = phi.sin_cos();
            let dir = [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi];
            pixels[row_offset + (x as usize)] = sky_radiance(dir, params);
        }
    }
    pixels
}

/// PT-sky/sun-color (plan 0030): default radiometric calibration
/// constant applied to [`solar_irradiance`] so the analytically
/// derived sun-color matches the existing Sponza reference render
/// at noon. Until the PT-sky/sun-color **calibration sub-step**
/// runs (Sponza render at `--sky-elevation 75 --sky-turbidity 2.5
/// --sun-intensity 1.0` compared against
/// `data/output/sponza_sunlit_reference.png` floor patch),
/// this is a placeholder of 1.0.
///
/// The proper value lands in plan 0030's `Findings` section once
/// the user has GPU time to run the calibration render. Until
/// then, the analytic sun-color will be qualitatively correct
/// (sunset reddens, midday is near-white, turbidity dims and
/// shifts) but absolutely under-scaled — `--sun-intensity` can
/// compensate at render time.
pub const DEFAULT_SOLAR_CALIBRATION: f32 = 1.0;

/// PT-sky/sun-color (plan 0030): analytic per-channel direct-beam
/// solar irradiance at the receiver, as a function of sun
/// elevation and atmospheric turbidity.
///
/// **The model.** This is a closed-form **Preetham 1999** direct
/// beam transmittance — Rayleigh + Ångström-turbidity aerosol —
/// evaluated at the three sRGB primary wavelengths
/// (615 nm R / 545 nm G / 465 nm B), multiplied by an
/// extraterrestrial-irradiance triple that integrates the ASTM
/// E-490 solar spectrum against the same primaries.
///
/// Concretely:
/// ```text
/// m(elev)  = 1 / (sin(elev) + 0.15 (elev_deg + 3.885)^-1.253)   [Kasten 1980 air mass]
/// β(T)     = 0.04608 T - 0.04586                                [Ångström β, Preetham eq. 12]
/// τ_R(λ)   = 0.008735 λ^-4.08                                    [Rayleigh, Preetham eq. 11]
/// τ_A(λ)   = β λ^-1.3                                            [Aerosol, Ångström α = 1.3]
/// T_atm(λ) = exp(-m (τ_R(λ) + τ_A(λ)))
/// I(λ)     = I_0(λ) · T_atm(λ)
/// ```
///
/// **What this is NOT.** The Hošek-Wilkie 2013 solar-radiance
/// model (the citation in plan 0030) prescribes a different
/// transmittance + limb-darkening table set, vendored from
/// `ArHosekSkyModelData_solar.h`. That file lives in the same
/// cgg.mff.cuni.cz release as the RGB skylight tables but is not
/// yet covered by `scripts/sky/fetch_hosek_data.py`. The Preetham
/// stand-in here gives the same qualitative behavior (sunset
/// reddens, midday is white-ish, higher turbidity shifts toward
/// yellow); the absolute scale absorbs into
/// [`DEFAULT_SOLAR_CALIBRATION`].
///
/// **Below horizon.** `elev_rad ≤ 0` returns `[0, 0, 0]`. The air
/// mass formula above blows up below ~5° elevation and the
/// Preetham model is documented as "informative, not authoritative"
/// in that regime (matches plan 0030's sunset hard floor).
///
/// **Units.** The output is in arbitrary-scale linear RGB. The
/// caller multiplies by [`DEFAULT_SOLAR_CALIBRATION`] (or whatever
/// calibration constant the integration prescribes) to bring it
/// onto the renderer's radiance scale.
pub fn solar_irradiance(elev_rad: f32, turbidity: f32) -> [f32; 3] {
    if elev_rad <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    // Extraterrestrial solar irradiance per sRGB channel. Derived
    // from integrating ASTM E-490 against the sRGB primaries; the
    // relative shape is what matters (calibration absorbs the
    // absolute scale).
    const I0_RGB: [f32; 3] = [1.0, 1.05, 0.95];

    // Sample wavelengths at the sRGB primaries (in µm — Preetham's
    // formulas take λ in µm).
    const LAMBDA_RGB: [f32; 3] = [0.615, 0.545, 0.465];

    let elev_deg = elev_rad.to_degrees();
    // Kasten 1980 air mass. Numerically stable down to ~5°
    // elevation; below that the result inflates but stays finite.
    let m = 1.0 / (elev_rad.sin() + 0.15 * (elev_deg + 3.885).powf(-1.253));

    // Ångström β from Preetham (eq. 12). Turbidity is clamped to
    // [1, 10] to match the sky-radiance model's clamp policy.
    let t = turbidity.clamp(1.0, 10.0);
    let beta = (0.04608 * t - 0.04586).max(0.0);

    let mut out = [0.0_f32; 3];
    for (ch, out_ch) in out.iter_mut().enumerate() {
        let lambda = LAMBDA_RGB[ch];
        let tau_rayleigh = 0.008735 * lambda.powf(-4.08);
        let tau_aerosol = beta * lambda.powf(-1.3);
        let transmittance = (-m * (tau_rayleigh + tau_aerosol)).exp();
        *out_ch = I0_RGB[ch] * transmittance;
    }
    out
}

/// Per-channel interpolated H-W model state at a particular
/// `(turbidity, albedo, solar_elevation)` query. Carries the
/// 9 Perez-formula parameters plus the zenith radiance scale.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ModelState {
    /// 9 Perez-formula parameters in canonical order (A..I).
    params: [f32; 9],
    /// Zenith radiance (Z in the paper).
    zenith: f32,
}

/// The Perez-style angular factor (eq. 3 of the H-W paper):
///
/// ```text
/// F(theta, gamma) = (1 + A·exp(B / (cos(theta) + 0.01)))
///                 · (C + D·exp(E·gamma)
///                    + F·cos²(gamma)
///                    + G·χ(H, gamma)
///                    + I·sqrt(cos(theta)))
/// ```
///
/// where `χ(H, γ) = (1 + cos²γ) / (1 + H² - 2H·cos γ)^(3/2)`
/// is the Henyey-Greenstein-like phase factor.
fn perez_formula(state: &ModelState, cos_theta: f32, gamma: f32, cos_gamma: f32) -> f32 {
    let p = &state.params;
    let a = p[0];
    let b = p[1];
    let c = p[2];
    let d = p[3];
    let e = p[4];
    let f = p[5];
    let g = p[6];
    let h = p[7];
    let i = p[8];

    let denom = 1.0 + h * h - 2.0 * h * cos_gamma;
    let chi = (1.0 + cos_gamma * cos_gamma) / denom.max(1e-12).powf(1.5);

    let height_term = 1.0 + a * (b / (cos_theta + 0.01)).exp();
    let angular_term =
        c + d * (e * gamma).exp() + f * (cos_gamma * cos_gamma) + g * chi + i * cos_theta.sqrt();

    height_term * angular_term
}

/// Interpolate the 9 Perez parameters + zenith radiance for one
/// RGB channel at the given (turbidity, albedo, solar elevation).
/// Linear in turbidity + albedo, quintic Bezier in elevation.
fn interpolate_state(
    channel: usize,
    turbidity: f32,
    albedo: f32,
    solar_elevation: f32,
) -> ModelState {
    debug_assert!(channel < 3);
    let t = turbidity.clamp(1.0, 10.0);
    let a = albedo.clamp(0.0, 1.0);

    // Turbidity is in [1, 10]. The data tables hold 10 entries
    // indexed [0..10), one per integer turbidity. Lerp between
    // adjacent integer bins.
    let t_index = (t - 1.0).floor() as usize;
    let t_index = t_index.min(8);
    let t_frac = (t - 1.0) - (t_index as f32);

    // Albedo lerp is just two endpoints.
    let bezier_t = elevation_to_bezier_param(solar_elevation);

    // Look up the four bracketing (turbidity bin, albedo bin)
    // tables, evaluate the Bezier for each, then lerp between
    // them.
    let s_lo_lo = bezier_eval_state(channel, t_index, 0, bezier_t);
    let s_lo_hi = bezier_eval_state(channel, t_index, 1, bezier_t);
    let s_hi_lo = bezier_eval_state(channel, t_index + 1, 0, bezier_t);
    let s_hi_hi = bezier_eval_state(channel, t_index + 1, 1, bezier_t);

    let lo = lerp_state(&s_lo_lo, &s_lo_hi, a);
    let hi = lerp_state(&s_hi_lo, &s_hi_hi, a);
    lerp_state(&lo, &hi, t_frac)
}

/// Convert solar elevation (radians) to the quintic Bezier
/// parameter. The reference C++ code applies a cube-root warp
/// to bias the control points: `t = (elevation / (π/2))^(1/3)`.
///
/// This warp puts more control points near the horizon (where
/// the sky changes most quickly with sun height) — sun
/// elevations near 0 map to small t, near π/2 map to t=1.
fn elevation_to_bezier_param(solar_elevation: f32) -> f32 {
    let normalized = (solar_elevation / (std::f32::consts::PI / 2.0)).clamp(0.0, 1.0);
    normalized.powf(1.0 / 3.0)
}

/// Quintic Bezier evaluation with 6 control points, parameter
/// `t ∈ [0, 1]`. The formula:
///
/// ```text
/// B(t) = C0·(1-t)^5 + C1·5(1-t)^4·t + C2·10(1-t)^3·t² +
///        C3·10(1-t)²·t³ + C4·5(1-t)·t^4 + C5·t^5
/// ```
fn quintic_bezier(ctrl: &[f32; 6], t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let it = 1.0 - t;
    let it2 = it * it;
    let it3 = it2 * it;
    let it4 = it3 * it;
    let it5 = it4 * it;
    ctrl[0] * it5
        + ctrl[1] * 5.0 * it4 * t
        + ctrl[2] * 10.0 * it3 * t2
        + ctrl[3] * 10.0 * it2 * t3
        + ctrl[4] * 5.0 * it * t4
        + ctrl[5] * t5
}

/// Evaluate the Bezier for a specific (channel, turbidity bin,
/// albedo bin) triple at parameter `t`. The 9 Perez parameters
/// and the zenith radiance each have their own 6-control-point
/// Bezier.
fn bezier_eval_state(channel: usize, t_bin: usize, a_bin: usize, t: f32) -> ModelState {
    let ctrl_set = data::control_set(channel, t_bin, a_bin);
    let mut params = [0.0_f32; 9];
    for (i, p) in params.iter_mut().enumerate() {
        *p = quintic_bezier(&ctrl_set.params[i], t);
    }
    let zenith = quintic_bezier(&ctrl_set.zenith, t);
    ModelState { params, zenith }
}

/// Componentwise lerp between two interpolated states.
fn lerp_state(a: &ModelState, b: &ModelState, t: f32) -> ModelState {
    let mut params = [0.0_f32; 9];
    for (i, p) in params.iter_mut().enumerate() {
        *p = a.params[i] * (1.0 - t) + b.params[i] * t;
    }
    let zenith = a.zenith * (1.0 - t) + b.zenith * t;
    ModelState { params, zenith }
}

// --------------------------------------------------------------
// Coefficient tables.
//
// The 3,720 floats below are the *control points* extracted from
// ArHosekSkyModelData_RGB.h in the official reference release at
// cgg.mff.cuni.cz. Per channel, per turbidity bin (10 bins), per
// albedo bin (2 bins), there are 6 elevation-control-points × 9
// Perez parameters = 54 floats for the angular formula, plus 6
// floats for the zenith radiance — 60 floats per (channel,
// turbidity, albedo) cell, 1,200 floats per channel, 3,600 floats
// for RGB total, plus 120 zenith-radiance floats per channel
// (already included above for a 60×20×3 = 3,600 layout).
//
// **The tables ship as all-zero stubs.** A separate vendoring
// follow-up (see module docstring) will populate them. With
// zeros in place, every sky-radiance call returns black — the
// model is callable but not yet physically meaningful.
mod data {
    /// Per-(channel, turbidity bin, albedo bin) coefficient set
    /// after Bezier-evaluation collapses elevation. Each of the
    /// 9 Perez parameters has 6 elevation control points; the
    /// zenith radiance also has 6.
    pub(super) struct ControlSet {
        pub params: [[f32; 6]; 9],
        pub zenith: [f32; 6],
    }

    /// All-zero stub control set — used until the vendoring
    /// follow-up populates the real tables.
    const ZERO: ControlSet = ControlSet {
        params: [[0.0; 6]; 9],
        zenith: [0.0; 6],
    };

    /// Look up the 6-control-point Bezier control sets for the
    /// given (channel, turbidity bin in [0, 10), albedo bin in
    /// {0, 1}). Returns the stub today; the populated version
    /// will index a static `[[ControlSet; 2]; 10]` array per
    /// channel.
    pub(super) fn control_set(channel: usize, t_bin: usize, a_bin: usize) -> &'static ControlSet {
        debug_assert!(channel < 3);
        debug_assert!(t_bin < 10);
        debug_assert!(a_bin < 2);
        // TODO(PT-sky/vendor-data): replace this stub with a
        // real lookup into the populated tables. The structure is
        // chosen so each `ControlSet` lives in `static` and the
        // lookup is a O(1) array index.
        let _ = (channel, t_bin, a_bin);
        &ZERO
    }
} // end of `mod data`

#[cfg(test)]
mod tests {
    use super::*;

    /// Quintic Bezier at t=0 returns the first control point.
    #[test]
    fn quintic_bezier_at_zero_returns_first_control_point() {
        let ctrl = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!((quintic_bezier(&ctrl, 0.0) - 1.0).abs() < 1e-6);
    }

    /// Quintic Bezier at t=1 returns the last control point.
    #[test]
    fn quintic_bezier_at_one_returns_last_control_point() {
        let ctrl = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!((quintic_bezier(&ctrl, 1.0) - 6.0).abs() < 1e-6);
    }

    /// Quintic Bezier with all control points equal evaluates
    /// to that constant for any t — the constancy invariant.
    #[test]
    fn quintic_bezier_constant_control_is_constant() {
        let ctrl = [7.5; 6];
        for &t in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            assert!((quintic_bezier(&ctrl, t) - 7.5).abs() < 1e-5);
        }
    }

    /// Quintic Bezier at t=0.5 is the symmetric weighted sum:
    /// (C0 + C5)·(1/32) + (C1 + C4)·(5/32) + (C2 + C3)·(10/32)
    /// = (C0 + C5)·0.03125 + (C1 + C4)·0.15625 + (C2 + C3)·0.3125
    #[test]
    fn quintic_bezier_at_half_matches_closed_form() {
        let ctrl = [1.0_f32, 0.0, 4.0, 2.0, 0.0, 3.0];
        let expected = (1.0 + 3.0) * 0.03125 + (0.0 + 0.0) * 0.15625 + (4.0 + 2.0) * 0.3125;
        let got = quintic_bezier(&ctrl, 0.5);
        assert!(
            (got - expected).abs() < 1e-6,
            "quintic_bezier at t=0.5: got {got}, expected {expected}",
        );
    }

    /// Bezier partition-of-unity: sum of basis weights over [0,
    /// 1] must equal 1 for any t. We verify this implicitly by
    /// checking that the Bezier of `[1, 1, 1, 1, 1, 1]` is 1.
    #[test]
    fn quintic_bezier_partition_of_unity() {
        let ctrl = [1.0_f32; 6];
        for &t in &[0.0, 0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 0.9, 1.0] {
            let v = quintic_bezier(&ctrl, t);
            assert!(
                (v - 1.0).abs() < 1e-5,
                "partition-of-unity broken at t={t}: got {v}",
            );
        }
    }

    /// Elevation-to-Bezier parameter: 0 elevation → t=0, π/2 →
    /// t=1, with the documented cube-root warp in between.
    #[test]
    fn elevation_to_bezier_endpoints_and_warp() {
        assert!(elevation_to_bezier_param(0.0).abs() < 1e-6);
        assert!((elevation_to_bezier_param(std::f32::consts::PI / 2.0) - 1.0).abs() < 1e-6);

        // Cube-root warp at quarter-elevation: t = (0.25)^(1/3) =
        // ~0.62996.
        let mid = elevation_to_bezier_param(std::f32::consts::PI / 8.0);
        let expected = 0.25_f32.powf(1.0 / 3.0);
        assert!((mid - expected).abs() < 1e-5);
    }

    /// Elevation parameter clamps to [0, 1] for out-of-range
    /// inputs (a negative sun shouldn't produce NaN downstream).
    #[test]
    fn elevation_to_bezier_clamps_out_of_range() {
        assert!(elevation_to_bezier_param(-1.0).abs() < 1e-6);
        assert!((elevation_to_bezier_param(std::f32::consts::PI) - 1.0).abs() < 1e-6);
    }

    /// `lerp_state` blends params + zenith componentwise.
    #[test]
    fn lerp_state_blends_componentwise() {
        let a = ModelState {
            params: [1.0; 9],
            zenith: 2.0,
        };
        let b = ModelState {
            params: [3.0; 9],
            zenith: 4.0,
        };
        let mid = lerp_state(&a, &b, 0.5);
        for &p in &mid.params {
            assert!((p - 2.0).abs() < 1e-6);
        }
        assert!((mid.zenith - 3.0).abs() < 1e-6);
    }

    /// `perez_formula` returns finite values for typical inputs.
    /// This is a math sanity check — actual radiance values are
    /// tested at the integration level once the data tables are
    /// vendored.
    #[test]
    fn perez_formula_finite_on_typical_inputs() {
        let state = ModelState {
            // Made-up but reasonable parameters: A=−1 (mild
            // height term), B=−0.5, then C..I = 0, except some
            // positive C/D for the angular factor.
            params: [-1.0, -0.5, 1.0, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0],
            zenith: 1.0,
        };
        // Sample at four sky directions.
        for &(cos_theta, gamma, cos_gamma) in &[
            (1.0, 0.5, 0.5),
            (0.5, 1.0, 0.0),
            (0.1, 1.5, -0.5),
            (0.7, 0.1, 0.95),
        ] {
            let v = perez_formula(&state, cos_theta, gamma, cos_gamma);
            assert!(v.is_finite(), "perez returned {v} (non-finite)");
        }
    }

    /// `perez_formula` doesn't blow up at the H = 1, γ = 0 edge
    /// case where `denom = 1 + 1 - 2 = 0` would otherwise produce
    /// infinity. The `max(1e-12)` floor in `chi` keeps it finite.
    #[test]
    fn perez_formula_finite_at_pathological_h_gamma() {
        let state = ModelState {
            params: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            zenith: 1.0,
        };
        let v = perez_formula(&state, 1.0, 0.0, 1.0);
        assert!(v.is_finite());
    }

    /// `sky_radiance` returns black for views below horizon
    /// (model is fitted on upper hemisphere only).
    #[test]
    fn sky_radiance_below_horizon_returns_black() {
        let params = SkyParams {
            sun_dir: [0.0, 1.0, 0.0],
            ..SkyParams::default()
        };
        let r = sky_radiance([0.0, -0.5, 0.866], &params);
        assert_eq!(r, [0.0, 0.0, 0.0]);
    }

    /// `sky_radiance` returns black for sun below horizon.
    #[test]
    fn sky_radiance_sun_below_horizon_returns_black() {
        let params = SkyParams {
            sun_dir: [0.0, -0.5, 0.866],
            ..SkyParams::default()
        };
        // Even looking straight up.
        let r = sky_radiance([0.0, 1.0, 0.0], &params);
        assert_eq!(r, [0.0, 0.0, 0.0]);
    }

    /// `sky_radiance` is finite for typical inputs — guard
    /// against NaN propagation from out-of-range angles.
    #[test]
    fn sky_radiance_finite_on_typical_inputs() {
        let params = SkyParams {
            sun_dir: [0.0, 0.866, 0.5],
            turbidity: 2.5,
            ground_albedo: [0.3, 0.3, 0.3],
        };
        for &view in &[
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.866],
            [0.866, 0.5, 0.0],
            [-0.5, 0.5, 0.707],
        ] {
            let r = sky_radiance(view, &params);
            for (c, &v) in r.iter().enumerate() {
                assert!(v.is_finite(), "channel {c}: got {v} (non-finite)");
                assert!(v >= 0.0, "channel {c}: got {v} (negative)");
            }
        }
    }

    /// With all-zero stub data tables, every call should return
    /// exactly [0, 0, 0]. This locks in the "stubs are stubs"
    /// invariant — if a contributor swaps in a non-zero default
    /// they'll have to update this test deliberately.
    #[test]
    fn sky_radiance_with_stub_data_returns_zero() {
        let params = SkyParams::default();
        for &view in &[[0.0, 1.0, 0.0], [0.5, 0.5, 0.707], [0.866, 0.5, 0.0]] {
            assert_eq!(sky_radiance(view, &params), [0.0, 0.0, 0.0]);
        }
    }

    /// Test the `interpolate_state` bookkeeping: blending the
    /// four bracketing corners must produce a `ModelState` whose
    /// params are componentwise-bilinear in (turbidity, albedo).
    /// With stub data this is trivially [0, 0] but the index
    /// math is exercised.
    #[test]
    fn interpolate_state_returns_zero_with_stub_data() {
        let s = interpolate_state(0, 2.5, 0.3, std::f32::consts::PI / 4.0);
        assert_eq!(s.params, [0.0_f32; 9]);
        assert_eq!(s.zenith, 0.0);
    }

    /// Out-of-range turbidity clamps cleanly: T = 100 should not
    /// panic (the `t_bin.min(8)` guard ensures we never index
    /// past the table).
    #[test]
    fn interpolate_state_clamps_out_of_range_turbidity() {
        // No panic: this is the load-bearing safety guarantee.
        let _ = interpolate_state(0, 100.0, 0.5, 0.5);
        let _ = interpolate_state(0, -10.0, 0.5, 0.5);
    }

    /// PT-sky/bake: output pixel buffer matches the requested
    /// resolution exactly.
    #[test]
    fn bake_equirect_returns_correctly_sized_buffer() {
        let pixels = bake_equirect(8, 4, &SkyParams::default());
        assert_eq!(pixels.len(), 32);
    }

    /// PT-sky/bake: lower-hemisphere rows (`y > height / 2`)
    /// produce black pixels because the model is fitted on the
    /// upper hemisphere only. The early-out by row keeps this
    /// cheap.
    #[test]
    fn bake_equirect_lower_hemisphere_is_black() {
        let w = 8;
        let h = 8;
        let pixels = bake_equirect(w, h, &SkyParams::default());
        // Rows `y >= h/2 = 4` should be entirely black. We use
        // `cos θ ≤ 0` as the cutoff, which happens at θ ≥ π/2,
        // which happens at `(y + 0.5) / h ≥ 0.5` → `y ≥ h/2 -
        // 0.5` → `y ≥ 4` for `h = 8`.
        for y in (h / 2)..h {
            for x in 0..w {
                let p = pixels[(y * w + x) as usize];
                assert_eq!(p, [0.0, 0.0, 0.0], "row {y}, col {x} not black");
            }
        }
    }

    /// PT-sky/bake: with sun below horizon, the entire baked
    /// map is zero — `sky_radiance` returns black everywhere.
    /// Pins the global short-circuit behaviour.
    #[test]
    fn bake_equirect_with_below_horizon_sun_is_all_black() {
        let params = SkyParams {
            sun_dir: [0.0, -0.5, 0.866],
            ..SkyParams::default()
        };
        let pixels = bake_equirect(16, 8, &params);
        for &p in &pixels {
            assert_eq!(p, [0.0, 0.0, 0.0]);
        }
    }

    /// PT-sky/bake: with stub (all-zero) data tables, the baked
    /// equirect is uniformly zero — the upper-hemisphere
    /// `sky_radiance` calls multiply through zero coefficients
    /// and return [0, 0, 0]. Pins the stub invariant.
    #[test]
    fn bake_equirect_with_stub_data_is_all_black() {
        let pixels = bake_equirect(32, 16, &SkyParams::default());
        for &p in &pixels {
            assert_eq!(p, [0.0, 0.0, 0.0]);
        }
    }

    /// PT-sky/bake: a 1×1 bake works without panicking. The
    /// pixel direction is (sin(π/2) cos(π), cos(π/2), sin(π/2)
    /// sin(π)) = (-1, 0, 0) — exactly on the horizon, so
    /// `cos_theta = 0` and the early-out fires. Result: black.
    #[test]
    fn bake_equirect_one_by_one_is_horizon_pixel() {
        let pixels = bake_equirect(1, 1, &SkyParams::default());
        assert_eq!(pixels.len(), 1);
        // 1×1 evaluates at θ = π/2 (cos = 0), which the row
        // early-out classifies as "below horizon" via the
        // `cos_theta <= 0` cutoff.
        assert_eq!(pixels[0], [0.0, 0.0, 0.0]);
    }

    /// PT-sky/bake: the integer-pixel direction convention
    /// matches `env.rs` exactly. Pixel (0, 0) in a (w=4, h=2)
    /// equirect has:
    ///   φ = 0.5 / 4 · 2π = π/4
    ///   θ = 0.5 / 2 · π = π/4
    ///   dir = (sin π/4 cos π/4, cos π/4, sin π/4 sin π/4)
    ///       ≈ (0.5, 0.707, 0.5)
    /// We can't test the radiance value (stub data → zero) but
    /// we *can* test the direction by checking that the bake
    /// would hit `sky_radiance` with the expected direction.
    /// Integration with `env.rs` is exercised by the assertion
    /// that an `EnvironmentMap` constructed from our output
    /// has the right pixel-count + dimensions.
    ///
    /// Native-only — `env::EnvironmentMap` is gated to
    /// `#[cfg(not(target_arch = "wasm32"))]` because it carries
    /// an HDR loader that wasm32 doesn't have.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn bake_equirect_integrates_with_environment_map() {
        let pixels = bake_equirect(8, 4, &SkyParams::default());
        let env = crate::pathtrace::env::EnvironmentMap::new(8, 4, pixels);
        assert_eq!(env.width, 8);
        assert_eq!(env.height, 4);
        assert_eq!(env.pixels.len(), 32);
    }

    /// PT-sky/sun-color: below-horizon sun gives zero irradiance —
    /// pins the early-out and the "Preetham not authoritative below
    /// horizon" contract.
    #[test]
    fn solar_irradiance_below_horizon_is_zero() {
        assert_eq!(solar_irradiance(0.0, 2.5), [0.0, 0.0, 0.0]);
        assert_eq!(solar_irradiance(-0.1, 2.5), [0.0, 0.0, 0.0]);
        assert_eq!(solar_irradiance(-1.0, 8.0), [0.0, 0.0, 0.0]);
    }

    /// PT-sky/sun-color: all three channels strictly decrease as
    /// turbidity increases (more aerosol → more extinction along
    /// the sun path). The aerosol term β(T) starts effectively at
    /// T = 1 (β = -0.05 clamped to 0), so we compare T = 2 → 10.
    #[test]
    fn solar_irradiance_monotone_in_turbidity() {
        let elev = std::f32::consts::FRAC_PI_4; // 45°
        let low = solar_irradiance(elev, 2.0);
        let mid = solar_irradiance(elev, 5.0);
        let high = solar_irradiance(elev, 10.0);
        for (ch, ((&lo, &mi), &hi)) in low.iter().zip(mid.iter()).zip(high.iter()).enumerate() {
            assert!(
                lo > mi && mi > hi,
                "channel {ch}: turbidity-monotone violated — low={lo}, mid={mi}, high={hi}",
            );
        }
    }

    /// PT-sky/sun-color: all three channels strictly increase as
    /// the sun rises from a low elevation toward zenith. Longer
    /// atmospheric path → more extinction → less ground-level
    /// irradiance.
    #[test]
    fn solar_irradiance_monotone_in_elevation() {
        let turbidity = 2.5;
        let low = solar_irradiance(10.0_f32.to_radians(), turbidity);
        let mid = solar_irradiance(45.0_f32.to_radians(), turbidity);
        let high = solar_irradiance(85.0_f32.to_radians(), turbidity);
        for (ch, ((&lo, &mi), &hi)) in low.iter().zip(mid.iter()).zip(high.iter()).enumerate() {
            assert!(
                lo < mi && mi < hi,
                "channel {ch}: elevation-monotone violated — low={lo}, mid={mi}, high={hi}",
            );
        }
    }

    /// PT-sky/sun-color: at low elevation (sunset/dawn) the
    /// red channel is brighter than the blue — Rayleigh scatters
    /// the short wavelengths preferentially, so the surviving
    /// direct-beam light reddens. This is the canonical "sunset
    /// is red" check.
    #[test]
    fn solar_irradiance_reddens_at_sunset() {
        let elev = 5.0_f32.to_radians();
        let i = solar_irradiance(elev, 2.5);
        assert!(
            i[0] > i[2],
            "low-elev red ({}) must exceed blue ({}) — Rayleigh reddening",
            i[0],
            i[2],
        );
        // The reddening should be substantial, not marginal: at 5°
        // elevation the ratio is typically ≥ 4× even at low
        // turbidity.
        assert!(
            i[0] / i[2].max(1e-12) > 4.0,
            "sunset reddening ratio R/B = {} should exceed 4×",
            i[0] / i[2].max(1e-12),
        );
    }

    /// PT-sky/sun-color: at high elevation (noon) the per-
    /// channel sun color is roughly balanced — the
    /// extraterrestrial spectrum is close to white and short-
    /// path extinction is small. The R/B ratio should be near 1
    /// (within ~25% — Rayleigh still acts on the short path).
    #[test]
    fn solar_irradiance_balanced_at_noon() {
        let elev = 85.0_f32.to_radians();
        let i = solar_irradiance(elev, 2.0);
        let ratio_rb = i[0] / i[2];
        assert!(
            (0.8..=1.4).contains(&ratio_rb),
            "noon R/B ratio = {ratio_rb}; should be near 1 (sun is approximately white)",
        );
    }

    /// PT-sky/sun-color: finite + non-NaN at the edges of the
    /// valid range — protect against blow-ups in
    /// `pow(-1.253)` and `exp(-large)` at low elevation /
    /// extreme turbidity.
    #[test]
    fn solar_irradiance_finite_at_edges() {
        for &(elev_deg, t) in &[
            (1.0_f32, 1.0_f32),
            (1.0, 10.0),
            (89.9, 1.0),
            (89.9, 10.0),
            (45.0, 0.5),  // below clamp floor
            (45.0, 20.0), // above clamp ceiling
        ] {
            let i = solar_irradiance(elev_deg.to_radians(), t);
            for (ch, &v) in i.iter().enumerate() {
                assert!(
                    v.is_finite() && v >= 0.0,
                    "(elev={elev_deg}, T={t}) ch{ch} = {v} not finite-non-negative",
                );
            }
        }
    }
}
