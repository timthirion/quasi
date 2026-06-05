//! PT-beer-lambert: integration tests for the CPU mirror of the
//! participating-media attenuation helper.
//!
//! `pathtrace.wgsl`'s per-segment Beer-Lambert step boils down to
//! `throughput *= exp(-σ_a · t)`. The math is trivial but easy to
//! quietly break in a future refactor — these tests pin the identities
//! the path tracer depends on:
//!
//! * `attenuation(0, t) = (1, 1, 1)`  — zero σ is the identity.
//! * Strictly monotone decreasing in `t` when any channel of `σ_a > 0`.
//! * Chain rule: `attenuation(σ, t1) · attenuation(σ, t2) =
//!   attenuation(σ, t1 + t2)` — the WGSL loop multiplies per-segment
//!   transmittances, so the closed-form must agree with the product.
//! * One-unit slab: `attenuation(σ, 1) = exp(-σ)` — the textbook
//!   Beer-Lambert reading.

use quasi::pathtrace::medium;

fn approx_eq3(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
    (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
}

#[test]
fn zero_absorption_is_identity_at_every_distance() {
    let sigma = [0.0_f32; 3];
    for &t in &[0.0_f32, 0.1, 1.0, 5.0, 100.0] {
        let a = medium::attenuation(sigma, t);
        assert!(
            approx_eq3(a, [1.0; 3], 1e-6),
            "zero σ at t={t}: expected (1,1,1), got {a:?}",
        );
    }
}

#[test]
fn attenuation_at_zero_distance_is_identity() {
    let sigma = [0.5_f32, 2.0, 4.0];
    let a = medium::attenuation(sigma, 0.0);
    assert!(approx_eq3(a, [1.0; 3], 1e-6), "got {a:?}");
}

#[test]
fn positive_absorption_strictly_decreases_with_distance() {
    let sigma = [0.7_f32, 0.3, 1.1];
    let t_steps = [0.0_f32, 0.25, 0.5, 1.0, 2.0, 4.0];
    let series: Vec<[f32; 3]> = t_steps
        .iter()
        .map(|&t| medium::attenuation(sigma, t))
        .collect();
    for w in series.windows(2) {
        for c in 0..3 {
            assert!(
                w[1][c] < w[0][c],
                "attenuation should strictly decrease (channel {c}, {:?} → {:?})",
                w[0],
                w[1],
            );
        }
    }
}

#[test]
fn chain_rule_holds_for_consecutive_segments() {
    let sigma = [0.3_f32, 0.8, 1.5];
    let t1 = 0.4_f32;
    let t2 = 1.1_f32;
    let a1 = medium::attenuation(sigma, t1);
    let a2 = medium::attenuation(sigma, t2);
    let product = [a1[0] * a2[0], a1[1] * a2[1], a1[2] * a2[2]];
    let combined = medium::attenuation(sigma, t1 + t2);
    assert!(
        approx_eq3(product, combined, 1e-6),
        "chain rule violated: {product:?} vs {combined:?}",
    );
}

#[test]
fn one_unit_slab_matches_classic_beer_lambert_reading() {
    let sigma = [0.5_f32, 1.0, 2.0];
    let a = medium::attenuation(sigma, 1.0);
    let want = [(-0.5_f32).exp(), (-1.0_f32).exp(), (-2.0_f32).exp()];
    assert!(approx_eq3(a, want, 1e-6), "got {a:?}, want {want:?}");
}

#[test]
fn sample_distance_at_xi_zero_returns_zero() {
    let t = medium::sample_distance(1.5, 0.0);
    assert!(t.abs() < 1e-6, "xi=0 should map to t=0; got {t}");
}

#[test]
fn sample_distance_at_xi_half_matches_log_two_over_sigma_t() {
    // Inverse-CDF of Exp(σ_t) at xi = 0.5 is `ln(2) / σ_t`.
    for &sigma_t in &[0.25_f32, 1.0, 4.0] {
        let t = medium::sample_distance(sigma_t, 0.5);
        let want = std::f32::consts::LN_2 / sigma_t;
        assert!(
            (t - want).abs() < 1e-5,
            "ξ=0.5, σ_t={sigma_t}: got {t}, want {want}",
        );
    }
}

#[test]
fn sample_distance_monte_carlo_mean_matches_one_over_sigma_t() {
    // E[t] under Exp(σ_t) is 1 / σ_t. Estimate over many samples.
    let g = 1.324_717_957_244_746_f64;
    let a1 = (1.0 / g) as f32;

    for &sigma_t in &[0.4_f32, 1.5, 3.0] {
        let n = 200_000_u32;
        let mut acc = 0.0_f64;
        for i in 0..n {
            let xi = (0.5 + a1 * i as f32).fract();
            let t = medium::sample_distance(sigma_t, xi);
            acc += t as f64;
        }
        let mean = (acc / n as f64) as f32;
        let want = 1.0_f32 / sigma_t;
        let rel_err = ((mean - want) / want).abs();
        // 3% tolerance: 1-D Kronecker sequence is reasonable but not
        // perfect for exponential heavy-tail estimation; finer
        // bounds would need a much larger N for the smaller σ_t
        // values, which makes the test annoyingly slow.
        assert!(
            rel_err < 0.03,
            "MC mean should match 1/σ_t (σ_t={sigma_t}): got {mean}, want {want}",
        );
    }
}

#[test]
fn extinction_is_componentwise_sum() {
    let sigma_a = [0.1_f32, 0.5, 1.2];
    let sigma_s = [0.3_f32, 0.2, 0.6];
    let want = [0.4_f32, 0.7, 1.8];
    let got = medium::extinction(sigma_a, sigma_s);
    assert!(approx_eq3(got, want, 1e-6), "got {got:?}, want {want:?}");
}
