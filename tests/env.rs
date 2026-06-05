//! PT-env: integration tests for the CPU mirror of the environment-
//! map importance-sampling helpers.
//!
//! These pin the analytic identities the WGSL side will rely on:
//!
//! * Marginal CDF is monotone and ends at 1.0.
//! * Each conditional CDF row is monotone and ends at 1.0.
//! * Per-pixel PDFs sum to 1 within tolerance.
//! * `sample → pdf_at_pixel` round-trips: the PDF reported at the
//!   sampled pixel matches the joint PDF the sampler used.
//! * Sample frequency follows the radiance distribution — pixels
//!   with k× higher luminance get hit k× as often (MC tolerance).

use quasi::pathtrace::env::{EnvironmentMap, ImportanceTables};

fn solid_env(width: u32, height: u32, color: [f32; 3]) -> EnvironmentMap {
    let pixels = vec![color; (width * height) as usize];
    EnvironmentMap::new(width, height, pixels)
}

#[test]
fn marginal_cdf_is_monotone_and_ends_at_one() {
    let env = solid_env(4, 4, [1.0, 1.0, 1.0]);
    let t = ImportanceTables::build(&env);
    assert_eq!(t.marginal_cdf.len(), 5);
    assert_eq!(t.marginal_cdf[0], 0.0);
    for w in t.marginal_cdf.windows(2) {
        assert!(w[0] <= w[1], "marginal cdf decreased: {w:?}");
    }
    assert!((t.marginal_cdf.last().unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn conditional_cdf_rows_are_monotone_and_end_at_one() {
    let mut pixels = vec![[0.0_f32; 3]; 16];
    // Sprinkle some non-uniform values across rows so each row's CDF
    // exercises the recurrence.
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = [i as f32 + 1.0, 0.0, 0.0];
    }
    let env = EnvironmentMap::new(4, 4, pixels);
    let t = ImportanceTables::build(&env);
    let w = t.width as usize + 1;
    for y in 0..t.height as usize {
        let row = &t.conditional_cdf[y * w..(y + 1) * w];
        assert_eq!(row[0], 0.0);
        for win in row.windows(2) {
            assert!(win[0] <= win[1], "conditional CDF row {y} decreased");
        }
        assert!(
            (row[row.len() - 1] - 1.0).abs() < 1e-6,
            "row {y} ends at {} not 1.0",
            row[row.len() - 1]
        );
    }
}

#[test]
fn per_pixel_pdfs_sum_to_one() {
    let pixels: Vec<[f32; 3]> = (0..64)
        .map(|i| {
            let v = ((i % 13) as f32 + 1.0) * 0.5;
            [v, v * 0.7, v * 0.3]
        })
        .collect();
    let env = EnvironmentMap::new(8, 8, pixels);
    let t = ImportanceTables::build(&env);
    let w = t.width as usize;
    let h = t.height as usize;
    let mut total = 0.0_f64;
    for y in 0..h {
        for x in 0..w {
            let p = t.marginal_pdf[y] * t.conditional_pdf[y * w + x];
            total += p as f64;
        }
    }
    assert!(
        (total - 1.0).abs() < 1e-4,
        "joint PDF sums to {total}, not 1.0",
    );
}

#[test]
fn sample_round_trip_matches_pdf_at_pixel() {
    // For any xi pair, the (x, y) the sampler returns should report
    // the same PDF as the sampler used internally. Pre-padded-Sobol
    // golden-ratio Kronecker stand-in.
    let pixels: Vec<[f32; 3]> = (0..256)
        .map(|i| {
            let v = ((i * 7 + 3) % 17) as f32 / 10.0;
            [v, v, v]
        })
        .collect();
    let env = EnvironmentMap::new(16, 16, pixels);
    let t = ImportanceTables::build(&env);
    let g = 1.324_717_957_244_746_f64;
    let a1 = (1.0 / g) as f32;
    let a2 = (1.0 / (g * g)) as f32;
    for i in 1..50 {
        let xi = [(0.5 + a1 * i as f32).fract(), (0.5 + a2 * i as f32).fract()];
        let ((x, y), p_from_sample) = t.sample(xi);
        let p_from_lookup = t.pdf_at_pixel(x, y);
        assert!(
            (p_from_sample - p_from_lookup).abs() < 1e-6,
            "round-trip mismatch at xi={xi:?}: sampled p={p_from_sample}, eval p={p_from_lookup}",
        );
    }
}

#[test]
fn sampling_concentrates_on_bright_pixels() {
    // Half the image is dark (luminance 0.01), half is bright
    // (luminance 1.0). The bright half should attract ~100× more
    // samples than the dark half (modulo MC noise + the sin θ
    // weighting evenly distributed across rows).
    let w = 16u32;
    let h = 16u32;
    let mut pixels = vec![[0.0_f32; 3]; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let bright = x >= w / 2;
            let v = if bright { 1.0 } else { 0.01 };
            pixels[(y * w + x) as usize] = [v, v, v];
        }
    }
    let env = EnvironmentMap::new(w, h, pixels);
    let t = ImportanceTables::build(&env);

    let g = 1.324_717_957_244_746_f64;
    let a1 = (1.0 / g) as f32;
    let a2 = (1.0 / (g * g)) as f32;
    let n = 8_000u32;
    let mut bright_hits = 0u32;
    for i in 0..n {
        let xi = [(0.5 + a1 * i as f32).fract(), (0.5 + a2 * i as f32).fract()];
        let ((x, _y), _) = t.sample(xi);
        if x >= w / 2 {
            bright_hits += 1;
        }
    }
    let bright_frac = bright_hits as f32 / n as f32;
    // True ratio: bright luminance × bright pixels / total luminance.
    // 1.0 × half / (1.0 × half + 0.01 × half) = 1 / 1.01 ≈ 0.990.
    assert!(
        bright_frac > 0.95,
        "expected ≥95% of samples in the bright half; got {bright_frac:.3}",
    );
}

#[test]
fn direction_sample_pdf_round_trip() {
    let pixels: Vec<[f32; 3]> = (0..256)
        .map(|i| {
            let v = ((i * 11 + 5) % 19) as f32 / 10.0 + 0.1;
            [v, v, v]
        })
        .collect();
    let env = EnvironmentMap::new(16, 16, pixels);
    let t = ImportanceTables::build(&env);
    let g = 1.324_717_957_244_746_f64;
    let a1 = (1.0 / g) as f32;
    let a2 = (1.0 / (g * g)) as f32;
    for i in 1..32 {
        let xi = [(0.5 + a1 * i as f32).fract(), (0.5 + a2 * i as f32).fract()];
        let (dir, pdf_sample) = t.sample_direction(xi);
        let pdf_eval = t.pdf_at_direction(dir);
        let dir_len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        assert!(
            (dir_len - 1.0).abs() < 1e-3,
            "sample_direction returned non-unit dir {dir:?} (len {dir_len})",
        );
        // Quantisation: pdf_at_direction snaps to a pixel via floor,
        // so the eval pdf can equal the sample pdf at a neighbouring
        // pixel boundary. Tolerance covers that snap.
        let rel = (pdf_sample - pdf_eval).abs() / pdf_sample.max(1e-6);
        assert!(
            rel < 0.05 || (pdf_sample - pdf_eval).abs() < 1e-3,
            "direction round-trip mismatch at xi={xi:?}: sample pdf={pdf_sample}, eval pdf={pdf_eval}",
        );
    }
}
