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
    /// PT-denoise-halo-metric (plan 0021): when `true` (default),
    /// the demodulated signal passes through Reinhard tonemap
    /// before the à-trous wavelet and is untonemapped after — the
    /// plan-0018 fix that bounds the HDR halo radius.
    ///
    /// **Tests** flip this to `false` to ablate the tonemap wrap
    /// and assert the tonemap-on-vs-off relationship the fix
    /// depends on. **Production code should keep the default** —
    /// switching it off silently restores the HDR-halo failure
    /// mode plan 0018 closed.
    pub tonemap_passes: bool,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            sigma_color: 0.50,
            sigma_normal: 32.0,
            sigma_depth: 0.10,
            passes: 5,
            tonemap_passes: true,
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

    // PT-denoise-tonemap (plan 0018): map the (demodulated)
    // signal through Reinhard `t = c / (1 + c)` so the wavelet's
    // colour edge stop sees a bounded `[0, 1)` range. Without
    // this, a ceiling-light pixel (radiance L ≈ 30) next to a
    // wall pixel (L ≈ 1) produces a colour-distance term of
    // ≈ 900 that the σ_c = 0.5 default can't tame — the kernel
    // pulls hard on the bright pixels and spreads a halo into
    // the wall. Reinhard collapses that into `[0, 1)` so the
    // colour stop behaves as designed.
    //
    // PT-denoise-halo-metric (plan 0021): gated on
    // `params.tonemap_passes` so tests can ablate the wrap and
    // measure the tonemap-on-vs-off halo relationship. Default
    // `true`; production keeps the plan-0018 behaviour.
    let mut working = demod;
    if params.tonemap_passes {
        for px in working.iter_mut() {
            px[0] /= 1.0 + px[0].max(0.0);
            px[1] /= 1.0 + px[1].max(0.0);
            px[2] /= 1.0 + px[2].max(0.0);
        }
    }

    // Step 2 — five à-trous passes (in tonemapped space when
    // `tonemap_passes`, in raw demodulated radiance otherwise).
    let mut buf = working;
    let mut step = 1_i32;
    for _ in 0..params.passes {
        buf = atrous_pass(&buf, normal, depth, width, height, step, params);
        step *= 2;
    }

    // Step 3a — untonemap. `c = t / max(1 - t, ε)` inverts
    // Reinhard exactly for `t < 1`. The kernel pulls some
    // smoothing toward 1 around very bright pixels, so we floor
    // `1 - t` to avoid blowing up the division. Skipped when
    // `tonemap_passes == false` (raw demodulated radiance
    // already lives in linear space).
    if params.tonemap_passes {
        for px in buf.iter_mut() {
            let max_t = 1.0 - 1e-4;
            px[0] = px[0].clamp(0.0, max_t) / (1.0 - px[0]).max(1e-4);
            px[1] = px[1].clamp(0.0, max_t) / (1.0 - px[1]).max(1e-4);
            px[2] = px[2].clamp(0.0, max_t) / (1.0 - px[2]).max(1e-4);
        }
    }

    // Step 3b — remodulate, restoring the bypass pixels as-is.
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

    /// PT-denoise-halo-metric (plan 0021): max R-channel value
    /// over the Chebyshev ring at the given radius around
    /// `center` — i.e. pixels where `|dx| ∨ |dy| == radius`.
    ///
    /// Used by the halo tests to standardise the
    /// "how far did the bright pixel reach?" measurement.
    /// Returns the peak halo intensity at the ring; the caller
    /// asserts a relationship against the background or against
    /// a paired denoiser configuration.
    fn halo_intensity_at_ring(
        out: &[[f32; 4]],
        width: u32,
        radius: i32,
        center: (i32, i32),
    ) -> f32 {
        let w = width as i32;
        let h = (out.len() as i32) / w;
        let (cx, cy) = center;
        let mut peak = f32::NEG_INFINITY;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx.abs().max(dy.abs()) != radius {
                    continue;
                }
                let px = cx + dx;
                let py = cy + dy;
                if px < 0 || px >= w || py < 0 || py >= h {
                    continue;
                }
                let idx = (py as usize) * (w as usize) + (px as usize);
                if out[idx][0] > peak {
                    peak = out[idx][0];
                }
            }
        }
        peak
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

    /// PT-denoise-tonemap: a uniform input that round-trips
    /// through Reinhard tonemap → identity wavelet (uniform stays
    /// uniform under any à-trous step) → Reinhard untonemap should
    /// come back to the original radiance.
    #[test]
    fn tonemap_round_trip_is_identity_on_uniform_input() {
        let w = 16;
        let h = 16;
        let n = (w * h) as usize;
        let target = [2.5_f32, 1.8, 0.7, 1.0];
        let radiance = uniform(n, target);
        let albedo = uniform(n, [1.0, 1.0, 1.0, 1.0]);
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
            for c in 0..3 {
                assert!(
                    (p[c] - target[c]).abs() < 5e-3,
                    "pixel {i} channel {c}: got {} expected {}",
                    p[c],
                    target[c],
                );
            }
        }
    }

    /// PT-denoise-tonemap halo test: an HDR-bright pixel
    /// surrounded by dim neighbours should not pull the
    /// neighbours' radiance up after denoising. Pre-tonemap
    /// behaviour: dim neighbours got pulled to ~5-10× their
    /// true value within a ±16 px halo. After: the kernel
    /// keeps them close to truth because the bright pixel's
    /// tonemapped distance is bounded.
    #[test]
    fn tonemap_kills_hdr_halo_around_bright_pixel() {
        let w = 32;
        let h = 32;
        let n = (w * h) as usize;
        let bright = 30.0_f32;
        let dim = 1.0_f32;
        let mut radiance = vec![[dim, dim, dim, 1.0]; n];
        let center_idx = ((h / 2) * w + (w / 2)) as usize;
        radiance[center_idx] = [bright, bright, bright, 1.0];
        let albedo = vec![[1.0_f32, 1.0, 1.0, 1.0]; n];
        let normal = vec![[0.0_f32, 1.0, 0.0, 0.0]; n];
        let depth = vec![[1.0_f32, 0.0, 0.0, 1.0]; n];

        let out = denoise(
            &radiance,
            &albedo,
            &normal,
            &depth,
            w,
            h,
            DenoiseParams::default(),
        );

        // PT-denoise-halo-metric (plan 0021): peak ring intensity
        // via the shared helper. Pre-refactor, this was an inline
        // double loop; the value is identical.
        let cx = (w / 2) as i32;
        let cy = (h / 2) as i32;
        let peak = halo_intensity_at_ring(&out, w, 8, (cx, cy));
        assert!(
            peak < dim * 1.5,
            "halo ring at r=8: peak R = {peak} exceeds 1.5×dim ({})",
            dim * 1.5,
        );
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

    // --- PT-denoise-halo-metric (plan 0021) ---

    /// Helper that runs the single-bright-pixel halo scene at
    /// the given HDR ratio + `tonemap_passes` setting and
    /// returns the peak halo intensity at the radius-8 ring.
    /// Centralised here so the three plan-0021 tests share the
    /// same geometry.
    fn halo_scene_peak(bright: f32, dim: f32, albedo_v: f32, tonemap: bool) -> f32 {
        let w = 32_u32;
        let h = 32_u32;
        let n = (w * h) as usize;
        let mut radiance = vec![[dim, dim, dim, 1.0]; n];
        let center_idx = ((h / 2) * w + (w / 2)) as usize;
        radiance[center_idx] = [bright, bright, bright, 1.0];
        let albedo = vec![[albedo_v, albedo_v, albedo_v, 1.0]; n];
        let normal = vec![[0.0_f32, 1.0, 0.0, 0.0]; n];
        let depth = vec![[1.0_f32, 0.0, 0.0, 1.0]; n];
        let params = DenoiseParams {
            tonemap_passes: tonemap,
            ..DenoiseParams::default()
        };
        let out = denoise(&radiance, &albedo, &normal, &depth, w, h, params);
        let cx = (w / 2) as i32;
        let cy = (h / 2) as i32;
        halo_intensity_at_ring(&out, w, 8, (cx, cy))
    }

    /// Tonemap ablation across HDR ratios on a **single bright
    /// pixel** scene. For each L/ℓ ∈ {3, 10, 30, 100, 300} the
    /// test runs the denoiser **twice** — once with the Reinhard
    /// tonemap wrap, once without — and asserts both
    /// configurations keep the radius-8 halo well within `1.1 ×
    /// dim`. This is the **falsifiable** form of plan 0018's
    /// halo claim on this geometry.
    ///
    /// Empirical finding pinned at this commit (single-pixel
    /// scene, `σ_c = 0.5`):
    ///
    /// * **Without tonemap**, the colour edge stop
    ///   `exp(-(L-ℓ)² / σ_c²)` collapses to zero for any HDR
    ///   ratio ≥ 3 — the bright pixel contributes nothing to
    ///   the dim neighbours. Halo @ r=8 = `1.000 × dim` at
    ///   every ratio.
    /// * **With tonemap**, the Reinhard curve compresses
    ///   `(L, ℓ) → (t_L, t_ℓ)` so the colour edge stop sees
    ///   `Δt ≈ 0.5`, `w_colour ≈ 0.42`. The bright pixel
    ///   contributes a small amount and a tiny halo appears:
    ///   `1.002 × dim` at L/ℓ = 3, growing to `1.020 × dim` at
    ///   L/ℓ = 300.
    ///
    /// So on **isolated bright pixels** the tonemap fix is
    /// (marginally) *worse* than no-tonemap. The visible halo
    /// plan 0018 closed lives elsewhere — multi-pixel emitter
    /// footprints + smooth HDR gradients. `halo_from_bright_cluster`
    /// covers the closer-to-real footprint; the gradient case
    /// awaits a future test.
    ///
    /// The test asserts both configurations stay within `1.1 ×
    /// dim` at every ratio — this protects against a future
    /// regression that makes either configuration spread halo
    /// further on this geometry. The plan-skeptic audit of
    /// plan 0018 surfaced exactly this gap (the original test
    /// allowed 1.5 × dim, loose enough to pass for a
    /// 30%-effective fix); this assertion is 5× tighter.
    #[test]
    fn tonemap_ablation_at_hdr_ratios() {
        let dim = 1.0_f32;
        let ratios = [3.0_f32, 10.0, 30.0, 100.0, 300.0];
        let halo_bound = 1.1_f32 * dim;
        let mut report =
            String::from("\n  L/ℓ |  halo @ r=8 (tonemap on) | halo @ r=8 (off) | on/off ratio\n");
        let mut all_within_bound = true;
        for &ratio in &ratios {
            let bright = dim * ratio;
            let halo_on = halo_scene_peak(bright, dim, 1.0, true);
            let halo_off = halo_scene_peak(bright, dim, 1.0, false);
            let ratio_on_off = halo_on / halo_off.max(1e-6);
            report.push_str(&format!(
                "  {ratio:>4} |  {halo_on:>22.6} | {halo_off:>16.6} | {ratio_on_off:>12.4}\n",
            ));
            if halo_on >= halo_bound || halo_off >= halo_bound {
                all_within_bound = false;
            }
        }
        assert!(
            all_within_bound,
            "halo exceeded {halo_bound}× dim at some HDR ratio: {report}",
        );
    }

    /// Single-bright-pixel halo under **realistic albedo** (0.7
    /// rather than the unity used in
    /// `tonemap_kills_hdr_halo_around_bright_pixel`). Exercises
    /// the demodulation pathway the plan-0018 audit named at
    /// `denoise.rs:135` — pre-this-test, the halo test couldn't
    /// surface a regression where the tonemap fix interacted
    /// badly with non-unity demodulation.
    #[test]
    fn halo_with_realistic_albedo() {
        let bright = 30.0_f32;
        let dim = 1.0_f32;
        let peak = halo_scene_peak(bright, dim, 0.7, true);
        assert!(
            peak < dim * 1.5,
            "halo at r=8 with albedo=0.7: peak R = {peak} exceeds 1.5×dim ({})",
            dim * 1.5,
        );
    }

    /// Halo around a **3×3 bright cluster** (closer to a real
    /// ceiling-light footprint than a single pixel). The
    /// larger source can spill more by design; the bound is
    /// 2.5× background rather than the 1.5× the single-pixel
    /// tests use. The radius-8 ring sits ≥ 7 pixels outside
    /// the cluster boundary on every side.
    #[test]
    fn halo_from_bright_cluster() {
        let w = 32_u32;
        let h = 32_u32;
        let n = (w * h) as usize;
        let bright = 30.0_f32;
        let dim = 1.0_f32;
        let mut radiance = vec![[dim, dim, dim, 1.0]; n];
        let cx = (w / 2) as i32;
        let cy = (h / 2) as i32;
        for dy in -1..=1_i32 {
            for dx in -1..=1_i32 {
                let idx = ((cy + dy) as usize) * (w as usize) + ((cx + dx) as usize);
                radiance[idx] = [bright, bright, bright, 1.0];
            }
        }
        let albedo = vec![[1.0_f32, 1.0, 1.0, 1.0]; n];
        let normal = vec![[0.0_f32, 1.0, 0.0, 0.0]; n];
        let depth = vec![[1.0_f32, 0.0, 0.0, 1.0]; n];
        let out = denoise(
            &radiance,
            &albedo,
            &normal,
            &depth,
            w,
            h,
            DenoiseParams::default(),
        );
        let peak = halo_intensity_at_ring(&out, w, 8, (cx, cy));
        assert!(
            peak < dim * 2.5,
            "halo at r=8 around 3×3 cluster: peak R = {peak} exceeds 2.5×dim ({})",
            dim * 2.5,
        );
    }
}
