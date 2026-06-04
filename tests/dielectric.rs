//! PT-dielectrics: integration tests for the CPU mirror of the smooth
//! dielectric BSDF helpers.
//!
//! The WGSL side lives in `pathtrace.wgsl` (`fresnel_dielectric` +
//! `refract_through`); the formulas here in
//! [`quasi::pathtrace::dielectric`] are intentionally a byte-for-byte
//! port. These tests pin the analytic identities:
//!
//! * Fresnel: at normal incidence we hit Schlick's classic
//!   `(η₁ - η₂)² / (η₁ + η₂)²`; at grazing we return 1.
//! * Snell: refracted direction satisfies `η_i · sin θ_i = η_t · sin θ_t`.
//! * TIR: kicks in past `asin(η_t / η_i)` going from denser to less
//!   dense (and never the other way).
//! * Energy conservation: 0 ≤ Fresnel ≤ 1 over the full angle range.

use quasi::pathtrace::dielectric;

const ETA_AIR: f32 = 1.0;
const ETA_GLASS: f32 = 1.5;

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn fresnel_at_normal_incidence_matches_classic_formula() {
    // F₀ = ((η_t - η_i) / (η_t + η_i))² for unpolarised light at
    // normal incidence — the same value Schlick's approximation
    // hits at θ_i = 0.
    let r0 = ((ETA_GLASS - ETA_AIR) / (ETA_GLASS + ETA_AIR)).powi(2);
    let got = dielectric::fresnel(1.0, ETA_AIR, ETA_GLASS);
    assert!(
        approx(got, r0, 1e-5),
        "Fresnel at normal incidence (air → glass): expected {r0}, got {got}",
    );
}

#[test]
fn fresnel_at_grazing_returns_one() {
    let got = dielectric::fresnel(0.0, ETA_AIR, ETA_GLASS);
    assert!(approx(got, 1.0, 1e-5), "Fresnel at θ_i = 90° must be 1; got {got}");
}

#[test]
fn fresnel_energy_conserves_over_full_range() {
    for k in 0..=20 {
        let cos = k as f32 / 20.0;
        let f = dielectric::fresnel(cos, ETA_AIR, ETA_GLASS);
        assert!(f >= 0.0 && f <= 1.0, "Fresnel out of range at cos={cos}: {f}");
    }
}

#[test]
fn refract_satisfies_snell_law() {
    // Half a dozen incidence angles entering air → glass. Snell:
    // η_i · sin θ_i = η_t · sin θ_t.
    let n = [0.0_f32, 1.0, 0.0];
    for k in 1..=9 {
        let theta_i = (k as f32 / 10.0) * std::f32::consts::FRAC_PI_2;
        let wo = [-theta_i.sin(), theta_i.cos(), 0.0]; // pointing away from surface, into +y hemisphere
        let wt = dielectric::refract(wo, n, ETA_AIR, ETA_GLASS).expect("no TIR going into denser medium");
        // wt should point into the *transmitted* hemisphere
        // (`dot(n, wt) < 0`).
        assert!(wt[1] < 0.0, "refracted direction must point into transmitted hemisphere");
        let sin_t = (wt[0] * wt[0] + wt[2] * wt[2]).sqrt();
        let expected_sin_t = ETA_AIR / ETA_GLASS * theta_i.sin();
        assert!(
            approx(sin_t, expected_sin_t, 1e-5),
            "Snell mismatch at θ_i = {theta_i:.3}: sin θ_t = {sin_t}, expected {expected_sin_t}",
        );
    }
}

#[test]
fn refraction_is_lossless_at_normal_incidence() {
    // At θ_i = 0 the refracted direction should be exactly -n
    // (straight through, no bending).
    let n = [0.0_f32, 1.0, 0.0];
    let wo = [0.0, 1.0, 0.0];
    let wt = dielectric::refract(wo, n, ETA_AIR, ETA_GLASS).unwrap();
    for (got, want) in wt.iter().zip([0.0_f32, -1.0, 0.0].iter()) {
        assert!(approx(*got, *want, 1e-5), "got {wt:?}");
    }
}

#[test]
fn tir_kicks_in_past_critical_angle() {
    // Glass → air: critical angle is `asin(1 / 1.5) ≈ 41.8°`.
    let theta_c = dielectric::critical_angle(ETA_GLASS, ETA_AIR).expect("TIR is possible here");
    let expected = (ETA_AIR / ETA_GLASS).asin();
    assert!(approx(theta_c, expected, 1e-5));

    let n = [0.0_f32, 1.0, 0.0];
    // Just below the critical angle: refraction succeeds.
    let theta_under = theta_c - 0.01;
    let wo_under = [-theta_under.sin(), theta_under.cos(), 0.0];
    assert!(
        dielectric::refract(wo_under, n, ETA_GLASS, ETA_AIR).is_some(),
        "expected refraction just below the critical angle",
    );
    // Just above the critical angle: TIR.
    let theta_over = theta_c + 0.01;
    let wo_over = [-theta_over.sin(), theta_over.cos(), 0.0];
    assert!(
        dielectric::refract(wo_over, n, ETA_GLASS, ETA_AIR).is_none(),
        "expected TIR past the critical angle",
    );

    // Fresnel must also report 1.0 past the critical angle (since
    // the WGSL signals TIR through Fresnel directly).
    let f_over = dielectric::fresnel(theta_over.cos(), ETA_GLASS, ETA_AIR);
    assert!(approx(f_over, 1.0, 1e-5), "Fresnel must hit 1.0 past TIR; got {f_over}");
}

#[test]
fn no_tir_possible_going_into_denser_medium() {
    assert!(
        dielectric::critical_angle(ETA_AIR, ETA_GLASS).is_none(),
        "air → glass has no critical angle",
    );
}
