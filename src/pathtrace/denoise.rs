//! Edge-aware à-trous wavelet denoiser (PT-denoise).
//!
//! Pure-CPU post-process that consumes the four AOVs the path
//! tracer writes — radiance / albedo / normal / depth — and
//! produces a denoised radiance buffer. Runs five à-trous
//! iterations at step sizes `1, 2, 4, 8, 16` with a 5×5 B3-spline
//! kernel and SVGF-style edge-stopping weights (colour, normal,
//! depth).
//!
//! ## Algorithm
//!
//! 1. **Demodulate**: `radiance / max(albedo, ε)`. Pulls texture
//!    detail out of the signal so the wavelet preserves edges.
//! 2. **À-trous wavelet** (5 iterations, step 1 → 16). 5×5
//!    B3-spline kernel; per-pair weights are the product of a
//!    colour Gaussian, a normal-cosine power, and a depth Gaussian.
//! 3. **Remodulate**: `denoised_demod · albedo`. Restores the
//!    texture detail without re-introducing noise.
//!
//! Defaults work for the Cornell + outdoor showcase scenes
//! without per-scene knobs. Expose [`DenoiseParams`] on the CLI
//! when a future scene needs a tweak.
//!
//! ## Performance
//!
//! Single-threaded, `~50 ms` for 768² in `--release`. The
//! widget-targeted resolutions (320–512) clock in under 20 ms.
//! Adding rayon is a deps churn we don't take on at first; the
//! filter is offline anyway, so the wall-clock win is small.

/// Tunable σ values for the edge-stopping functions. Defaults
/// land on the showcase scenes — see plan 0017.
#[derive(Clone, Copy, Debug)]
pub struct DenoiseParams {
    /// `σ_c` — colour edge stop. Lower → more aggressive
    /// blurring across colour discontinuities.
    pub sigma_color: f32,
    /// `σ_n` — normal edge stop (exponent on `max(0, n_p · n_q)`).
    /// Higher → tighter edge response across normal seams.
    pub sigma_normal: f32,
    /// `σ_z` — depth edge stop. Lower → more aggressive blurring
    /// across depth discontinuities. Auto-scaling by depth
    /// magnitude (`σ_z · max(|z_p|, 1.0)`) keeps the filter
    /// behaving sensibly across foreground / background ranges.
    pub sigma_depth: f32,
    /// Number of à-trous passes. Each pass doubles the step
    /// size. Five passes cover up to ±16 px of support, which
    /// matches typical SVGF setups.
    pub passes: u32,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            sigma_color: 0.50,
            sigma_normal: 32.0,
            sigma_depth: 0.10,
            passes: 5,
        }
    }
}

/// 5×5 B3-spline kernel `(1, 4, 6, 4, 1) / 16` per axis. We
/// store the **outer product** as a 5×5 array of weights
/// summing to 1.
const KERNEL: [[f32; 5]; 5] = {
    let row = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let mut k = [[0.0; 5]; 5];
    let mut j = 0;
    while j < 5 {
        let mut i = 0;
        while i < 5 {
            k[j][i] = row[i] * row[j];
            i += 1;
        }
        j += 1;
    }
    k
};

/// Top-level denoise: demodulate by albedo → 5-pass à-trous →
/// remodulate by albedo. Returns a fresh `Vec<[f32; 4]>` the
/// same length as the inputs; the alpha channel passes through
/// unchanged from `radiance`.
///
/// All four AOVs must share the same `width × height`.
pub fn denoise(
    radiance: &[[f32; 4]],
    albedo: &[[f32; 4]],
    normal: &[[f32; 4]],
    depth: &[[f32; 4]],
    width: u32,
    height: u32,
    params: DenoiseParams,
) -> Vec<[f32; 4]> {
    let n = (width as usize) * (height as usize);
    assert_eq!(radiance.len(), n);
    assert_eq!(albedo.len(), n);
    assert_eq!(normal.len(), n);
    assert_eq!(depth.len(), n);

    // Step 1 — demodulate radiance by albedo. Pixels with
    // ≈0 albedo become pass-through (set them aside and re-
    // attach untouched after the filter).
    const ALBEDO_FLOOR: f32 = 1.0e-3;
    let mut demod = vec![[0.0_f32; 4]; n];
    let mut demod_valid = vec![true; n];
    for i in 0..n {
        let a = &albedo[i];
        let r = &radiance[i];
        // Sum the RGB intensity; near-zero → bypass.
        let a_mag = (a[0] + a[1] + a[2]).abs();
        if a_mag < 3.0 * ALBEDO_FLOOR {
            demod[i] = *r;
            demod_valid[i] = false;
            continue;
        }
        demod[i] = [
            r[0] / a[0].max(ALBEDO_FLOOR),
            r[1] / a[1].max(ALBEDO_FLOOR),
            r[2] / a[2].max(ALBEDO_FLOOR),
            r[3],
        ];
    }

    // Step 2 — five à-trous passes.
    let mut buf = demod.clone();
    let mut step = 1_i32;
    for _ in 0..params.passes {
        buf = atrous_pass(&buf, normal, depth, width, height, step, params);
        step *= 2;
    }

    // Step 3 — remodulate, restoring the bypass pixels as-is.
    let mut out = vec![[0.0_f32; 4]; n];
    for i in 0..n {
        if !demod_valid[i] {
            out[i] = radiance[i];
            continue;
        }
        let a = &albedo[i];
        let d = &buf[i];
        out[i] = [d[0] * a[0], d[1] * a[1], d[2] * a[2], d[3]];
    }
    out
}

/// A single à-trous pass at the given pixel `step`. Edge stops
/// read normal + depth in their native pixel layout; `input` is
/// the (demodulated) radiance being filtered.
pub fn atrous_pass(
    input: &[[f32; 4]],
    normal: &[[f32; 4]],
    depth: &[[f32; 4]],
    width: u32,
    height: u32,
    step: i32,
    params: DenoiseParams,
) -> Vec<[f32; 4]> {
    let w = width as i32;
    let h = height as i32;
    let mut out = vec![[0.0_f32; 4]; (width * height) as usize];
    let sigma_c2 = params.sigma_color * params.sigma_color;
    for y in 0..h {
        for x in 0..w {
            let p_idx = (y as usize) * width as usize + (x as usize);
            let c_p = input[p_idx];
            let n_p = normal[p_idx];
            let z_p = depth[p_idx][0];
            let sigma_z_scaled = (params.sigma_depth * z_p.abs().max(1.0)).max(1.0e-3);
            let mut sum = [0.0_f32; 3];
            let mut wsum = 0.0_f32;
            for kj in 0..5_i32 {
                for ki in 0..5_i32 {
                    let qx = x + (ki - 2) * step;
                    let qy = y + (kj - 2) * step;
                    if qx < 0 || qx >= w || qy < 0 || qy >= h {
                        continue;
                    }
                    let q_idx = (qy as usize) * width as usize + (qx as usize);
                    let c_q = input[q_idx];
                    let n_q = normal[q_idx];
                    let z_q = depth[q_idx][0];
                    // Colour weight (sum-of-squared-diff on RGB).
                    let dr = c_p[0] - c_q[0];
                    let dg = c_p[1] - c_q[1];
                    let db = c_p[2] - c_q[2];
                    let w_color = (-((dr * dr + dg * dg + db * db) / sigma_c2)).exp();
                    // Normal weight.
                    let cos_nn = (n_p[0] * n_q[0] + n_p[1] * n_q[1] + n_p[2] * n_q[2]).max(0.0);
                    let w_normal = cos_nn.powf(params.sigma_normal);
                    // Depth weight (auto-scaled).
                    let w_depth = (-((z_p - z_q).abs() / sigma_z_scaled)).exp();
                    let w_kernel = KERNEL[kj as usize][ki as usize];
                    let w = w_kernel * w_color * w_normal * w_depth;
                    sum[0] += w * c_q[0];
                    sum[1] += w * c_q[1];
                    sum[2] += w * c_q[2];
                    wsum += w;
                }
            }
            if wsum > 0.0 {
                out[p_idx] = [sum[0] / wsum, sum[1] / wsum, sum[2] / wsum, c_p[3]];
            } else {
                out[p_idx] = c_p;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize, v: [f32; 4]) -> Vec<[f32; 4]> {
        vec![v; n]
    }

    /// Kernel rows sum to 1 (B3-spline normalisation).
    #[test]
    fn b3_kernel_sums_to_one() {
        let total: f32 = KERNEL.iter().flatten().copied().sum();
        assert!((total - 1.0).abs() < 1e-6, "kernel sums to {total}");
    }

    /// A uniform input image returns a uniform output (modulo
    /// the alpha channel passing through unchanged).
    #[test]
    fn uniform_input_yields_uniform_output() {
        let w = 16;
        let h = 16;
        let n = (w * h) as usize;
        let radiance = uniform(n, [0.5, 0.4, 0.3, 1.0]);
        let albedo = uniform(n, [0.7, 0.7, 0.7, 1.0]);
        let normal = uniform(n, [0.0, 1.0, 0.0, 0.0]);
        let depth = uniform(n, [1.0, 0.0, 0.0, 1.0]);
        let out = denoise(
            &radiance,
            &albedo,
            &normal,
            &depth,
            w,
            h,
            DenoiseParams::default(),
        );
        for (i, p) in out.iter().enumerate() {
            assert!(
                (p[0] - 0.5).abs() < 1e-4,
                "pixel {i}: R = {} (expected 0.5)",
                p[0],
            );
            assert!((p[1] - 0.4).abs() < 1e-4);
            assert!((p[2] - 0.3).abs() < 1e-4);
        }
    }

    /// Two regions separated by a sharp normal discontinuity
    /// must not bleed across the seam after the filter.
    #[test]
    fn normal_edge_is_preserved() {
        let w = 32;
        let h = 8;
        let n = (w * h) as usize;
        // Left half: bright + facing +Y; right half: dim + facing
        // +X. Demodulate-then-filter: with a strong normal stop
        // they should remain at their respective colours.
        let mut radiance = vec![[0.0_f32; 4]; n];
        let mut normal = vec![[0.0_f32; 4]; n];
        let albedo = vec![[1.0_f32, 1.0, 1.0, 1.0]; n];
        let depth = vec![[1.0_f32, 0.0, 0.0, 1.0]; n];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                if x < w / 2 {
                    radiance[idx] = [1.0, 1.0, 1.0, 1.0];
                    normal[idx] = [0.0, 1.0, 0.0, 0.0];
                } else {
                    radiance[idx] = [0.0, 0.0, 0.0, 1.0];
                    normal[idx] = [1.0, 0.0, 0.0, 0.0];
                }
            }
        }
        let out = denoise(
            &radiance,
            &albedo,
            &normal,
            &depth,
            w,
            h,
            DenoiseParams::default(),
        );
        // Pixels well into each half (away from the seam by > 16
        // — the largest atrous step) should preserve their full
        // brightness / darkness within tolerance.
        for y in 0..h {
            // Left interior pixel at x = 1.
            let l_idx = (y * w + 1) as usize;
            assert!(
                out[l_idx][0] > 0.85,
                "left pixel y={y} R = {} expected ≈1",
                out[l_idx][0],
            );
            // Right interior pixel at x = w - 2.
            let r_idx = (y * w + w - 2) as usize;
            assert!(
                out[r_idx][0] < 0.15,
                "right pixel y={y} R = {} expected ≈0",
                out[r_idx][0],
            );
        }
    }

    /// White-Gaussian noise on top of a flat patch should reduce
    /// in RMSE after denoising (compared to the noisy input).
    #[test]
    fn noisy_flat_patch_rmse_drops() {
        let w = 64;
        let h = 64;
        let n = (w * h) as usize;
        // Deterministic pseudo-noise via xorshift so the test
        // doesn't depend on rand.
        let mut state = 0x12345678_u32;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) - 0.5
        };
        let mean_r = 0.5_f32;
        let mut radiance = vec![[0.0_f32; 4]; n];
        for r in radiance.iter_mut() {
            *r = [
                mean_r + 0.3 * rng(),
                mean_r + 0.3 * rng(),
                mean_r + 0.3 * rng(),
                1.0,
            ];
        }
        let albedo = vec![[1.0_f32, 1.0, 1.0, 1.0]; n];
        let normal = vec![[0.0_f32, 1.0, 0.0, 0.0]; n];
        let depth = vec![[1.0_f32, 0.0, 0.0, 1.0]; n];

        let mut rmse_before = 0.0_f64;
        for r in radiance.iter() {
            let e = (r[0] - mean_r) as f64;
            rmse_before += e * e;
        }
        rmse_before = (rmse_before / n as f64).sqrt();

        let out = denoise(
            &radiance,
            &albedo,
            &normal,
            &depth,
            w,
            h,
            DenoiseParams::default(),
        );
        let mut rmse_after = 0.0_f64;
        for r in out.iter() {
            let e = (r[0] - mean_r) as f64;
            rmse_after += e * e;
        }
        rmse_after = (rmse_after / n as f64).sqrt();

        assert!(
            rmse_after < rmse_before * 0.5,
            "denoise didn't tighten RMSE: before {rmse_before:.4}, after {rmse_after:.4}",
        );
    }
}
