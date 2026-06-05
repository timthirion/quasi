//! Environment-map illumination — PT-env.
//!
//! Loads an HDR equirectangular environment map from disk
//! (Radiance `.hdr` format) and builds the 2-D inverse-CDF tables
//! the path tracer's NEE branch uses to importance-sample the
//! sky. The pixel data + tables get uploaded to GPU storage once at
//! scene-build time; the WGSL side reads them in `sample_env_at_dir`,
//! `sample_env_importance`, and `env_pdf_at_dir`.
//!
//! Equirectangular convention used throughout:
//!   `(x, y) ∈ [0, w) × [0, h)`, with `x` running longitude
//!   φ ∈ [0, 2π) and `y` running latitude θ ∈ [0, π) from north
//!   pole (`y = 0`) to south. Direction:
//!     φ = (x + 0.5) / w · 2π
//!     θ = (y + 0.5) / h · π
//!     dir = (sin θ cos φ, cos θ, sin θ sin φ)
//!
//! The latitudinal `sin θ` distortion factor is folded into both
//! the marginal-row weighting (so importance sampling correctly
//! biases toward bright equatorial bands) and the PDF returned
//! from `pdf_at_direction` (so the MIS power heuristic stays
//! consistent with the BSDF-side pdf).

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// HDR equirectangular environment map in CPU memory. Pixels are
/// linear-encoded RGB radiance, row-major top-to-bottom (north pole
/// at `y = 0`).
#[derive(Clone, Debug)]
pub struct EnvironmentMap {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 3]>,
}

impl EnvironmentMap {
    /// Build directly from a pre-decoded pixel buffer. Mainly for
    /// tests; the CLI path uses `from_hdr_file`.
    pub fn new(width: u32, height: u32, pixels: Vec<[f32; 3]>) -> Self {
        assert_eq!(
            (width as usize) * (height as usize),
            pixels.len(),
            "EnvironmentMap pixel buffer length ({}) does not match width*height ({}×{})",
            pixels.len(),
            width,
            height,
        );
        EnvironmentMap {
            width,
            height,
            pixels,
        }
    }

    /// Load a Radiance `.hdr` file from disk and decode into linear
    /// RGB. Native-only; the web target loads via `JsValue` glue
    /// later if/when env maps fly across the wasm boundary.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_hdr_file(path: &Path) -> Result<Self, String> {
        use image::ImageDecoder;
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let cursor = std::io::Cursor::new(bytes);
        let decoder = image::codecs::hdr::HdrDecoder::new(cursor)
            .map_err(|e| format!("hdr decode {}: {e}", path.display()))?;
        let (width, height) = decoder.dimensions();
        let total_bytes = decoder.total_bytes() as usize;
        let mut buf = vec![0u8; total_bytes];
        decoder
            .read_image(&mut buf)
            .map_err(|e| format!("hdr pixels {}: {e}", path.display()))?;
        // Rgb32F → 3 × f32 per pixel; reinterpret the byte buffer.
        let float_count = total_bytes / 4;
        let floats: &[f32] = bytemuck::cast_slice(&buf[..float_count * 4]);
        let pixels: Vec<[f32; 3]> = floats.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        Ok(EnvironmentMap::new(width, height, pixels))
    }
}

/// Pre-computed marginal + conditional cumulative distributions for
/// importance-sampling the environment radiance, weighted by the
/// latitudinal `sin θ` factor. The path tracer's NEE branch
/// inverse-samples these to pick `(x, y)` proportional to perceived
/// sky brightness.
///
/// Both CDFs are normalised — each ends at exactly 1.0 — and arranged
/// so `binary_search` finds the cell whose CDF first exceeds the
/// random `ξ`.
#[derive(Clone, Debug)]
pub struct ImportanceTables {
    pub width: u32,
    pub height: u32,
    /// `marginal_cdf[y]` is `Σ_{y' ≤ y} p_row[y']` after
    /// normalisation. Length `height + 1` so the last entry is 1.0.
    pub marginal_cdf: Vec<f32>,
    /// Per-row conditional CDFs flattened — row `y` lives at
    /// `[y * (width + 1) .. (y + 1) * (width + 1)]`. Each row ends
    /// at 1.0 after normalisation. Stored as `(width + 1) · height`
    /// floats; the trailing 1.0 makes inverse-CDF lookup branchless
    /// at the right edge.
    pub conditional_cdf: Vec<f32>,
    /// Marginal probability of each row (post-normalisation). Used
    /// at `pdf_at_direction` time.
    pub marginal_pdf: Vec<f32>,
    /// Per-row conditional PDFs (flattened, `width · height`).
    /// `conditional_pdf[y * width + x]` is `p(x | y)`.
    pub conditional_pdf: Vec<f32>,
    /// `Σ_xy luminance(x, y) · sin θ_y` before normalisation. The
    /// renderer can scale env contributions by `total_power` if it
    /// later wants to power-balance between env and triangle lights;
    /// for now NEE picks uniformly between the two sources.
    pub total_power: f32,
}

impl ImportanceTables {
    /// Build the marginal + conditional CDFs from a pixel buffer.
    /// `weight_by_sin_theta` should normally be true — the
    /// equirectangular projection compresses pixels near the poles
    /// so the importance weighting needs the inverse correction.
    pub fn build(env: &EnvironmentMap) -> Self {
        let w = env.width as usize;
        let h = env.height as usize;
        assert!(w > 0 && h > 0, "environment map must have positive dims");

        let mut row_weights = vec![0.0_f32; h];
        let mut col_weights = vec![0.0_f32; w * h];

        for y in 0..h {
            let theta = (y as f32 + 0.5) * std::f32::consts::PI / h as f32;
            let sin_theta = theta.sin().max(0.0);
            let mut row_sum = 0.0_f32;
            for x in 0..w {
                let p = env.pixels[y * w + x];
                let l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                let w_xy = (l.max(0.0)) * sin_theta;
                col_weights[y * w + x] = w_xy;
                row_sum += w_xy;
            }
            row_weights[y] = row_sum;
        }
        let total_power = row_weights.iter().sum::<f32>().max(1e-30);

        let mut marginal_pdf = vec![0.0_f32; h];
        let mut marginal_cdf = vec![0.0_f32; h + 1];
        let mut acc = 0.0_f32;
        for y in 0..h {
            marginal_pdf[y] = row_weights[y] / total_power;
            acc += marginal_pdf[y];
            marginal_cdf[y + 1] = acc;
        }
        // Numerical clamp — the last entry must be exactly 1.0 so
        // the inverse-CDF binary search lands on a valid index for
        // `ξ` very close to 1.
        if let Some(last) = marginal_cdf.last_mut() {
            *last = 1.0;
        }

        let mut conditional_pdf = vec![0.0_f32; w * h];
        let mut conditional_cdf = vec![0.0_f32; (w + 1) * h];
        for y in 0..h {
            let row_sum = row_weights[y].max(1e-30);
            let mut row_acc = 0.0_f32;
            let row_cdf_off = y * (w + 1);
            for x in 0..w {
                let p = col_weights[y * w + x] / row_sum;
                conditional_pdf[y * w + x] = p;
                row_acc += p;
                conditional_cdf[row_cdf_off + x + 1] = row_acc;
            }
            // Clamp the trailing 1.0 — same reason as above.
            conditional_cdf[row_cdf_off + w] = 1.0;
        }

        ImportanceTables {
            width: env.width,
            height: env.height,
            marginal_cdf,
            conditional_cdf,
            marginal_pdf,
            conditional_pdf,
            total_power,
        }
    }

    /// Inverse-CDF sample: take two uniform `[0, 1)` floats and
    /// return the chosen pixel `(x, y)` plus the joint PDF *in
    /// pixel-density terms* (so before the equirectangular Jacobian
    /// gets folded in).
    pub fn sample(&self, xi: [f32; 2]) -> ((u32, u32), f32) {
        let y = inverse_cdf(&self.marginal_cdf, xi[1]).min(self.height as usize - 1);
        let row_off = y * (self.width as usize + 1);
        let row_cdf = &self.conditional_cdf[row_off..row_off + self.width as usize + 1];
        let x = inverse_cdf(row_cdf, xi[0]).min(self.width as usize - 1);
        let pdf = self.marginal_pdf[y] * self.conditional_pdf[y * self.width as usize + x];
        ((x as u32, y as u32), pdf)
    }

    /// Pixel-density PDF at `(x, y)`. Used by MIS to evaluate the
    /// env PDF at a direction the BSDF sampled.
    pub fn pdf_at_pixel(&self, x: u32, y: u32) -> f32 {
        let w = self.width as usize;
        let y = y as usize;
        let x = x as usize;
        self.marginal_pdf[y] * self.conditional_pdf[y * w + x]
    }

    /// Solid-angle PDF at a *direction*, folding in the
    /// equirectangular Jacobian `1 / (2π² · sin θ)`. Pairs with
    /// `sample_direction` below.
    pub fn pdf_at_direction(&self, dir: [f32; 3]) -> f32 {
        // Direction → (φ, θ) → (x, y).
        let dir = normalize3(dir);
        let theta = dir[1].clamp(-1.0, 1.0).acos();
        let mut phi = dir[2].atan2(dir[0]);
        if phi < 0.0 {
            phi += 2.0 * std::f32::consts::PI;
        }
        let x = ((phi / (2.0 * std::f32::consts::PI)) * self.width as f32)
            .floor()
            .clamp(0.0, (self.width - 1) as f32) as u32;
        let y = ((theta / std::f32::consts::PI) * self.height as f32)
            .floor()
            .clamp(0.0, (self.height - 1) as f32) as u32;
        let p_pixel = self.pdf_at_pixel(x, y);
        let sin_theta = theta.sin().max(1e-6);
        let area_jacobian = (self.width as f32) * (self.height as f32)
            / (2.0 * std::f32::consts::PI * std::f32::consts::PI * sin_theta);
        p_pixel * area_jacobian
    }

    /// Importance-sample a world-space direction on the sphere
    /// together with its solid-angle PDF. Pairs with
    /// `pdf_at_direction`.
    pub fn sample_direction(&self, xi: [f32; 2]) -> ([f32; 3], f32) {
        let ((x, y), p_pixel) = self.sample(xi);
        let phi = (x as f32 + 0.5) / self.width as f32 * 2.0 * std::f32::consts::PI;
        let theta = (y as f32 + 0.5) / self.height as f32 * std::f32::consts::PI;
        let sin_theta = theta.sin().max(1e-6);
        let dir = [sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin()];
        let area_jacobian = (self.width as f32) * (self.height as f32)
            / (2.0 * std::f32::consts::PI * std::f32::consts::PI * sin_theta);
        (dir, p_pixel * area_jacobian)
    }
}

fn inverse_cdf(cdf: &[f32], xi: f32) -> usize {
    // Returns the first index `i` where `cdf[i + 1] > xi`. The
    // standard PBRT bisection — `cdf` has length `n + 1`, with
    // `cdf[0] == 0.0` and `cdf[n] == 1.0`.
    let n = cdf.len() - 1;
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cdf[mid + 1] <= xi {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-30);
    [v[0] / l, v[1] / l, v[2] / l]
}
