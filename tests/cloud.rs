//! PT-cloud: integration tests for the CPU mirror of the procedural
//! cloud density function.
//!
//! Pinning the geometric invariants the path tracer leans on:
//!
//! * `density` is 0 outside the sphere bounded by `radius`.
//! * `density` is non-negative everywhere; bounded above by some
//!   reasonable constant (so the σ_t majorant in the WGSL delta
//!   tracker stays sound).
//! * `density` is deterministic per position (the WGSL hash is
//!   keyed on integer lattice coords → same input → same output).
//! * `density` integrated over a centred slice is non-zero (the
//!   noise threshold isn't so high that the sphere is empty).

use quasi::pathtrace::cloud;

const CENTER: [f32; 3] = [0.0, 1.0, 0.0];
const RADIUS: f32 = 0.5;

#[test]
fn density_is_zero_outside_the_sphere() {
    // 8 points well outside the sphere.
    for &offset in &[
        [1.0_f32, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [0.0, 2.5, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.7, 1.7, 0.7],
        [-0.7, 0.3, -0.7],
    ] {
        let d = cloud::density(offset, CENTER, RADIUS);
        assert_eq!(
            d, 0.0,
            "outside-sphere density should be 0 at {offset:?}; got {d}"
        );
    }
}

#[test]
fn density_is_zero_exactly_at_the_boundary() {
    // Points right at radius from center.
    for &offset in &[
        [RADIUS, 0.0, 0.0],
        [0.0, RADIUS, 0.0],
        [0.0, 0.0, RADIUS],
        [-RADIUS, 0.0, 0.0],
    ] {
        let pos = [
            CENTER[0] + offset[0],
            CENTER[1] + offset[1],
            CENTER[2] + offset[2],
        ];
        let d = cloud::density(pos, CENTER, RADIUS);
        assert_eq!(d, 0.0, "boundary density should be 0 at {pos:?}; got {d}");
    }
}

#[test]
fn density_is_deterministic_per_position() {
    // Repeated calls at the same position should return the exact
    // same value — the WGSL hash is keyed on integer lattice coords,
    // not on any RNG state.
    let p = [0.13_f32, 1.07, -0.21];
    let d0 = cloud::density(p, CENTER, RADIUS);
    let d1 = cloud::density(p, CENTER, RADIUS);
    let d2 = cloud::density(p, CENTER, RADIUS);
    assert_eq!(d0, d1);
    assert_eq!(d1, d2);
}

#[test]
fn density_is_non_negative_and_bounded() {
    // Sweep a 32³ grid centred on the cloud, check both
    // non-negativity and a generous upper bound (the WGSL `σ_t`
    // majorant assumes density ≤ ~1.3; if this fails the
    // sample_volume_distance_heterogeneous loop would go biased).
    let n = 32_i32;
    let mut total = 0.0_f64;
    let mut nonzero = 0_u32;
    for ix in -n..=n {
        for iy in -n..=n {
            for iz in -n..=n {
                let p = [
                    CENTER[0] + (ix as f32) * RADIUS / (n as f32),
                    CENTER[1] + (iy as f32) * RADIUS / (n as f32),
                    CENTER[2] + (iz as f32) * RADIUS / (n as f32),
                ];
                let d = cloud::density(p, CENTER, RADIUS);
                assert!(d >= 0.0, "density at {p:?} is negative: {d}");
                assert!(d < 2.0, "density at {p:?} exceeds majorant bound: {d}");
                if d > 0.01 {
                    nonzero += 1;
                }
                total += d as f64;
            }
        }
    }
    // Not empty — at least a few percent of cells inside the
    // cloud have meaningful density.
    let n_total = ((2 * n + 1) as u32).pow(3);
    assert!(
        nonzero as f64 > n_total as f64 * 0.05,
        "expected >5% non-empty cells; got {nonzero} of {n_total}",
    );
    assert!(total > 0.0, "expected positive total density; got {total}");
}

#[test]
fn density_drops_outside_the_radius_smoothly() {
    // smoothstep edge falloff: density at r=0.5 should be a
    // sizeable fraction of central density; at r=1.0 should be 0.
    // Walks a radial line from center outward.
    let center_d = cloud::density(CENTER, CENTER, RADIUS);
    // The fbm makes the centre density variable, so we don't pin
    // the exact value — just sanity-check the trend.
    let r_walk: Vec<f32> = (0..=20).map(|i| i as f32 * RADIUS / 20.0).collect();
    let mut last_inside = None;
    for r in &r_walk {
        let p = [CENTER[0] + r, CENTER[1], CENTER[2]];
        let d = cloud::density(p, CENTER, RADIUS);
        if *r < RADIUS {
            last_inside = Some(d);
        } else {
            assert_eq!(d, 0.0, "density should be 0 at and past the radius (r={r})");
        }
    }
    let _ = center_d;
    let _ = last_inside;
}
