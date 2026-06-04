//! Image-quality metrics for the verification harness.
//!
//! All metrics operate on the RGB channels of `[f32; 4]` pixels (the
//! storage shape of [`Aovs`](crate::pathtrace::offscreen::Aovs)), ignoring
//! alpha. Computations are accumulated in `f64` so a large image can't
//! lose precision to scalar underflow.
//!
//! Three flavours:
//!
//! - [`mse_rgb`] — straight mean squared error. The natural baseline.
//! - [`rmse_rgb`] — `sqrt(mse)`, in the same units as the image. Easier
//!   to reason about when comparing against a target threshold.
//! - [`rel_mse_rgb`] — per-pixel-normalised MSE following the PBRT
//!   convention `mean((a − b)^2 / (b^2 + ε))`. Less sensitive to bright
//!   regions dominating the overall error, which matters when an HDR
//!   image has a small emissive region next to large dim regions.

const REL_MSE_EPS: f64 = 1e-2;

/// Mean squared error over the RGB channels of two equal-sized images.
pub fn mse_rgb(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "metrics: image sizes differ: {} vs {}",
        a.len(),
        b.len(),
    );
    if a.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for (pa, pb) in a.iter().zip(b.iter()) {
        let d0 = (pa[0] - pb[0]) as f64;
        let d1 = (pa[1] - pb[1]) as f64;
        let d2 = (pa[2] - pb[2]) as f64;
        sum += d0 * d0 + d1 * d1 + d2 * d2;
    }
    // Divide by total scalar count: pixels * 3 channels.
    sum / (a.len() as f64 * 3.0)
}

/// Root mean squared error — in the same units as the image.
pub fn rmse_rgb(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    mse_rgb(a, b).sqrt()
}

/// Relative MSE: `mean((a − b)^2 / (b^2 + ε))`, treating `b` as the
/// reference. The `ε` keeps the metric defined for black reference pixels.
pub fn rel_mse_rgb(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "metrics: image sizes differ: {} vs {}",
        a.len(),
        b.len(),
    );
    if a.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for (pa, pb) in a.iter().zip(b.iter()) {
        for c in 0..3 {
            let av = pa[c] as f64;
            let bv = pb[c] as f64;
            let d = av - bv;
            sum += (d * d) / (bv * bv + REL_MSE_EPS);
        }
    }
    sum / (a.len() as f64 * 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: usize, h: usize, value: [f32; 4]) -> Vec<[f32; 4]> {
        vec![value; w * h]
    }

    #[test]
    fn mse_identical_is_zero() {
        let a = img(4, 4, [0.5, 0.5, 0.5, 1.0]);
        let b = a.clone();
        assert_eq!(mse_rgb(&a, &b), 0.0);
        assert_eq!(rmse_rgb(&a, &b), 0.0);
        assert_eq!(rel_mse_rgb(&a, &b), 0.0);
    }

    #[test]
    fn mse_empty_is_zero() {
        assert_eq!(mse_rgb(&[], &[]), 0.0);
        assert_eq!(rmse_rgb(&[], &[]), 0.0);
        assert_eq!(rel_mse_rgb(&[], &[]), 0.0);
    }

    #[test]
    fn mse_constant_offset() {
        // (1.0)^2 per channel × 3 channels / 3 = 1.0.
        let a = img(8, 8, [1.0, 1.0, 1.0, 1.0]);
        let b = img(8, 8, [0.0, 0.0, 0.0, 1.0]);
        assert!((mse_rgb(&a, &b) - 1.0).abs() < 1e-9);
        assert!((rmse_rgb(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rmse_is_sqrt_of_mse() {
        let a = img(4, 4, [2.0, 4.0, 6.0, 1.0]);
        let b = img(4, 4, [0.0, 0.0, 0.0, 1.0]);
        let m = mse_rgb(&a, &b);
        let r = rmse_rgb(&a, &b);
        assert!((r - m.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn rel_mse_handles_black_reference() {
        // (1.0 − 0.0)² / (0² + ε) = 1/ε. With ε = 1e-2 → 100 per channel.
        let a = img(2, 2, [1.0, 1.0, 1.0, 1.0]);
        let b = img(2, 2, [0.0, 0.0, 0.0, 1.0]);
        let r = rel_mse_rgb(&a, &b);
        assert!((r - 100.0).abs() < 1e-9, "got {r}");
    }

    #[test]
    fn rel_mse_grows_with_error_at_fixed_reference() {
        // Holding the reference fixed, doubling the per-pixel error should
        // ~4× the rel-MSE (squared term). The ε keeps a tiny perturbation
        // at low intensities; tolerance is generous.
        let b = img(4, 4, [0.5, 0.5, 0.5, 1.0]);
        let a1 = img(4, 4, [0.6, 0.6, 0.6, 1.0]); // err = 0.1
        let a2 = img(4, 4, [0.7, 0.7, 0.7, 1.0]); // err = 0.2
        let r1 = rel_mse_rgb(&a1, &b);
        let r2 = rel_mse_rgb(&a2, &b);
        let ratio = r2 / r1;
        assert!(
            (ratio - 4.0).abs() < 0.05,
            "expected ~4x scaling of rel-MSE, got ratio={ratio}",
        );
    }

    #[test]
    #[should_panic(expected = "image sizes differ")]
    fn mse_size_mismatch_panics() {
        let a = img(2, 2, [0.0; 4]);
        let b = img(3, 2, [0.0; 4]);
        let _ = mse_rgb(&a, &b);
    }
}
