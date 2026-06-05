//! PT-hg: integration tests for the CPU mirror of the
//! Henyey-Greenstein phase function.
//!
//! Identities the WGSL side leans on:
//!
//! * `eval(*, 0)` returns the isotropic constant `1 / (4π)`.
//! * Normalisation: `∫ p(cos θ) dω = 1` over the unit sphere for
//!   any `g`.
//! * Forward (`g > 0`) peaks at `cos θ = 1`; backward (`g < 0`)
//!   peaks at `cos θ = -1`.
//! * Importance sampling: `E[cos θ]` under HG equals `g` exactly.
//!   The MC test pins this within tolerance.

use quasi::pathtrace::phase;

#[test]
fn g_zero_returns_isotropic_at_every_cosine() {
    for k in -10..=10 {
        let cos = k as f32 / 10.0;
        let p = phase::eval(cos, 0.0);
        assert!(
            (p - phase::ISOTROPIC).abs() < 1e-6,
            "HG(cos={cos}, g=0) = {p} should equal isotropic = {}",
            phase::ISOTROPIC,
        );
    }
}

#[test]
fn forward_g_peaks_at_cos_one() {
    let g = 0.7_f32;
    let forward = phase::eval(1.0, g);
    let backward = phase::eval(-1.0, g);
    let middle = phase::eval(0.0, g);
    assert!(forward > middle && middle > backward);
}

#[test]
fn backward_g_peaks_at_cos_negative_one() {
    let g = -0.6_f32;
    let forward = phase::eval(1.0, g);
    let backward = phase::eval(-1.0, g);
    let middle = phase::eval(0.0, g);
    assert!(backward > middle && middle > forward);
}

#[test]
fn pdf_normalises_to_one_over_the_sphere() {
    // Numerical integration over the sphere using the zonal
    // structure: ∫ p(cos θ) sin θ dθ dφ = 2π ∫ p(c) dc
    // where c = cos θ. Simpson-ish trapezoid is enough at N=2k.
    for &g in &[-0.7_f32, -0.3, 0.0, 0.3, 0.7] {
        let n = 2048_usize;
        let dc = 2.0 / n as f32;
        let mut acc = 0.0_f64;
        for i in 0..n {
            let c0 = -1.0 + i as f32 * dc;
            let c1 = c0 + dc;
            // midpoint
            let cm = 0.5 * (c0 + c1);
            acc += (phase::eval(cm, g) as f64) * dc as f64;
        }
        let total = acc * 2.0 * std::f64::consts::PI;
        assert!(
            (total - 1.0).abs() < 0.01,
            "HG(g={g}) ∫ over sphere = {total}, expected 1",
        );
    }
}

#[test]
fn importance_sample_mean_cosine_matches_g() {
    // `E[cos θ]` under HG = `g`. Estimate by Monte Carlo with a
    // golden-ratio Kronecker sequence.
    let g_seq = 1.324_717_957_244_746_f64;
    let a = (1.0 / g_seq) as f32;
    for &g in &[-0.7_f32, -0.3, 0.0, 0.3, 0.6, 0.85] {
        let n = 100_000_u32;
        let mut acc = 0.0_f64;
        for i in 0..n {
            let xi = (0.5 + a * (i as f32)).fract();
            let cos = phase::sample_cos_theta(g, xi);
            acc += cos as f64;
        }
        let mean = (acc / n as f64) as f32;
        assert!(
            (mean - g).abs() < 0.01,
            "MC mean cos θ for g={g}: got {mean}, expected ~{g}",
        );
    }
}

#[test]
fn sample_cos_theta_stays_in_unit_range() {
    // Cover the canonical interval edges as well.
    let g_seq = 1.324_717_957_244_746_f64;
    let a = (1.0 / g_seq) as f32;
    for &g in &[-0.9_f32, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9] {
        for i in 0..1000_u32 {
            let xi = (0.5 + a * (i as f32)).fract();
            let cos = phase::sample_cos_theta(g, xi);
            assert!(
                (-1.0..=1.0).contains(&cos),
                "cos θ out of range for g={g}, xi={xi}: {cos}",
            );
        }
    }
}
