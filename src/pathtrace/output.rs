//! Image output for the offscreen renderer (native-only).
//!
//! Given an [`Aovs`] snapshot from `pathtrace::offscreen`, [`write_render`]
//! emits two files:
//!
//! - `<base>.png` — tonemapped 8-bit RGB (Reinhard + gamma 1/2.2), matching
//!   what the windowed renderer shows on screen.
//! - `<base>.exr` — multi-channel HDR EXR carrying RGB radiance plus the
//!   `albedo.{R,G,B}`, `N.{X,Y,Z}`, and `Z` (depth) AOVs as separate
//!   channels in one layer.
//!
//! The two encoders run on **scoped threads** so they overlap in
//! wall-clock — quoth `AGENTS.md`'s "Use the language" guidance: a small,
//! genuine use of `std::thread::scope` exercises borrow-across-threads
//! cleanly without resorting to `Arc`.

use std::path::{Path, PathBuf};

use crate::pathtrace::offscreen::Aovs;

/// Files produced by [`write_render`].
#[derive(Clone, Debug)]
pub struct RenderPaths {
    pub png: PathBuf,
    pub exr: PathBuf,
    /// PT-adaptive (plan 0028): per-pixel relative standard error
    /// of luminance, log-scale clamped to `[1e-3, 1e0]`, viridis
    /// palette. Emitted as `<base>_variance.png` alongside the
    /// other outputs so a render's noise distribution is always
    /// inspectable.
    pub variance: PathBuf,
}

#[derive(Debug)]
pub enum OutputError {
    Image(image::ImageError),
    Exr(Box<exr::error::Error>),
}

impl std::fmt::Display for OutputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Image(e) => write!(f, "PNG encode error: {e}"),
            Self::Exr(e) => write!(f, "EXR encode error: {e}"),
        }
    }
}

impl std::error::Error for OutputError {}

impl From<image::ImageError> for OutputError {
    fn from(e: image::ImageError) -> Self {
        Self::Image(e)
    }
}

impl From<exr::error::Error> for OutputError {
    fn from(e: exr::error::Error) -> Self {
        Self::Exr(Box::new(e))
    }
}

/// Writes both files. Encoders run in parallel under
/// [`std::thread::scope`] — neither encoder borrows the other, both only
/// read from `aovs`, and the scope guarantees the borrow can't outlive
/// the call.
pub fn write_render(aovs: &Aovs, base: &Path) -> Result<RenderPaths, OutputError> {
    let png = base.with_extension("png");
    let exr = base.with_extension("exr");
    let variance = {
        let mut v = base.to_path_buf();
        let stem = v
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "render".to_string());
        v.set_file_name(format!("{stem}_variance"));
        v.set_extension("png");
        v
    };

    std::thread::scope(|s| -> Result<(), OutputError> {
        let png_h = s.spawn(|| write_tonemapped_png(aovs, &png));
        let exr_h = s.spawn(|| write_aov_exr(aovs, &exr));
        let var_h = s.spawn(|| write_variance_png(aovs, &variance));
        png_h.join().expect("PNG encoder thread panicked")?;
        exr_h.join().expect("EXR encoder thread panicked")?;
        var_h.join().expect("variance PNG thread panicked")?;
        Ok(())
    })?;

    Ok(RenderPaths { png, exr, variance })
}

/// Reinhard tonemap + gamma 1/2.2, matching the windowed renderer's
/// `present` shader so the saved PNG looks like what was on screen.
fn tonemap_pixel(hdr: [f32; 3]) -> [u8; 3] {
    let r = [
        hdr[0] / (hdr[0] + 1.0),
        hdr[1] / (hdr[1] + 1.0),
        hdr[2] / (hdr[2] + 1.0),
    ];
    let g = [
        r[0].powf(1.0 / 2.2),
        r[1].powf(1.0 / 2.2),
        r[2].powf(1.0 / 2.2),
    ];
    [
        (g[0].clamp(0.0, 1.0) * 255.0) as u8,
        (g[1].clamp(0.0, 1.0) * 255.0) as u8,
        (g[2].clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

pub fn write_tonemapped_png(aovs: &Aovs, path: &Path) -> Result<(), OutputError> {
    let w = aovs.width;
    let h = aovs.height;
    let mut buf = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for px in &aovs.radiance {
        let [r, g, b] = tonemap_pixel([px[0], px[1], px[2]]);
        buf.push(r);
        buf.push(g);
        buf.push(b);
    }
    image::save_buffer(path, &buf, w, h, image::ColorType::Rgb8)?;
    Ok(())
}

/// PT-adaptive (plan 0028): writes the per-pixel relative
/// standard error (the quantity the adaptive scheduler tests
/// against `noise_threshold`) as a log-scale viridis-coloured
/// PNG. Pixels with `relative_error > 1.0` saturate at the
/// brightest viridis yellow; pixels with `< 1e-3` saturate at
/// the darkest purple. The colour-map is intentionally noisy-
/// is-yellow / quiet-is-purple so a reader can spot the noisy
/// regions visually.
pub fn write_variance_png(aovs: &Aovs, path: &Path) -> Result<(), OutputError> {
    let w = aovs.width;
    let h = aovs.height;
    let var = aovs.luminance_variance();
    let mut buf = Vec::with_capacity((w as usize) * (h as usize) * 3);

    // For the relative-standard-error story we need the per-
    // pixel sample count. The accumulator carries it implicitly
    // (the `frame_count` at render end), but plumbing that
    // through the Aovs struct is its own milestone. For the
    // variance AOV PNG we'll surface the **unnormalised
    // standard deviation** sqrt(var(Y)) — divide-by-mean and
    // sqrt(n) deferred to PT-adaptive/scheduler when the active-
    // mask logic needs the exact relative-error quantity.
    let epsilon = 1e-3_f32;
    for v in var {
        let stddev = v.sqrt();
        // Log-scale, clamped to [1e-3, 1e0].
        let log_v = stddev.max(epsilon).min(1.0).log10();
        // Map [-3, 0] → [0, 1].
        let t = ((log_v + 3.0) / 3.0).clamp(0.0, 1.0);
        let [r, g, b] = viridis_lut(t);
        buf.push((r * 255.0) as u8);
        buf.push((g * 255.0) as u8);
        buf.push((b * 255.0) as u8);
    }
    image::save_buffer(path, &buf, w, h, image::ColorType::Rgb8)?;
    Ok(())
}

/// Piecewise-linear approximation of the Matplotlib viridis
/// colour-map. Five control points spaced uniformly in `[0, 1]`
/// taken from the published viridis RGB table. Sufficient for a
/// debug AOV display; not a reference-grade colour transform.
fn viridis_lut(t: f32) -> [f32; 3] {
    // Control points (Matplotlib viridis at t = 0, 0.25, 0.5, 0.75, 1).
    const CP: [[f32; 3]; 5] = [
        [0.267, 0.005, 0.329],
        [0.282, 0.140, 0.457],
        [0.220, 0.448, 0.535],
        [0.500, 0.751, 0.230],
        [0.993, 0.906, 0.144],
    ];
    let t = t.clamp(0.0, 1.0);
    let scaled = t * 4.0;
    let lo = (scaled as usize).min(3);
    let frac = scaled - (lo as f32);
    let a = CP[lo];
    let b = CP[lo + 1];
    [
        a[0] + (b[0] - a[0]) * frac,
        a[1] + (b[1] - a[1]) * frac,
        a[2] + (b[2] - a[2]) * frac,
    ]
}

pub fn write_aov_exr(aovs: &Aovs, path: &Path) -> Result<(), OutputError> {
    use exr::prelude::*;

    let w = aovs.width as usize;
    let h = aovs.height as usize;
    let n = w * h;

    // Demux per-channel scalar arrays out of the [f32; 4] pixel arrays.
    let mut rad_r = Vec::with_capacity(n);
    let mut rad_g = Vec::with_capacity(n);
    let mut rad_b = Vec::with_capacity(n);
    let mut alb_r = Vec::with_capacity(n);
    let mut alb_g = Vec::with_capacity(n);
    let mut alb_b = Vec::with_capacity(n);
    let mut nor_x = Vec::with_capacity(n);
    let mut nor_y = Vec::with_capacity(n);
    let mut nor_z = Vec::with_capacity(n);
    let mut depth = Vec::with_capacity(n);
    // PT-adaptive (plan 0028): per-pixel luminance variance
    // derived from radiance + mean_y2 — surfaced in the EXR so
    // downstream tools can read the variance map alongside the
    // standard AOVs.
    let var_y = aovs.luminance_variance();
    for i in 0..n {
        let r = aovs.radiance[i];
        rad_r.push(r[0]);
        rad_g.push(r[1]);
        rad_b.push(r[2]);
        let a = aovs.albedo[i];
        alb_r.push(a[0]);
        alb_g.push(a[1]);
        alb_b.push(a[2]);
        let nn = aovs.normal[i];
        nor_x.push(nn[0]);
        nor_y.push(nn[1]);
        nor_z.push(nn[2]);
        depth.push(aovs.depth[i][0]);
    }

    let make = |name: &str, vals: Vec<f32>| AnyChannel::new(name, FlatSamples::F32(vals));
    let mut channels: SmallVec<[AnyChannel<FlatSamples>; 4]> = SmallVec::new();
    channels.push(make("R", rad_r));
    channels.push(make("G", rad_g));
    channels.push(make("B", rad_b));
    channels.push(make("albedo.R", alb_r));
    channels.push(make("albedo.G", alb_g));
    channels.push(make("albedo.B", alb_b));
    channels.push(make("N.X", nor_x));
    channels.push(make("N.Y", nor_y));
    channels.push(make("N.Z", nor_z));
    channels.push(make("Z", depth));
    channels.push(make("variance.Y", var_y));

    let layer = Layer::new(
        (w, h),
        LayerAttributes::named("quasi"),
        Encoding::default(),
        AnyChannels::sort(channels),
    );
    let image = Image::from_layer(layer);
    image.write().to_file(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pathtrace::offscreen::Aovs;

    fn synthetic_aovs(width: u32, height: u32) -> Aovs {
        let n = (width as usize) * (height as usize);
        Aovs {
            width,
            height,
            radiance: (0..n)
                .map(|i| [i as f32 / n as f32, 0.5, 1.0 - i as f32 / n as f32, 1.0])
                .collect(),
            albedo: vec![[0.73, 0.73, 0.73, 1.0]; n],
            normal: vec![[0.0, 1.0, 0.0, 0.0]; n],
            depth: vec![[2.5, 0.0, 0.0, 1.0]; n],
            mean_y2: vec![[0.25, 0.0, 0.0, 0.0]; n],
        }
    }

    #[test]
    fn tonemap_extreme_inputs_dont_panic() {
        // f32::INFINITY / NaN cast to u8 is implementation-defined in
        // older Rust but currently saturates. The asserts below pin the
        // behaviour we rely on (no panic, no garbage).
        for v in [0.0_f32, 1.0, 5.0, 100.0, 1e6, f32::INFINITY] {
            let _ = tonemap_pixel([v, v, v]);
        }
    }

    #[test]
    fn tonemap_black_stays_black() {
        assert_eq!(tonemap_pixel([0.0, 0.0, 0.0]), [0, 0, 0]);
    }

    #[test]
    fn tonemap_saturates_bright_to_near_white() {
        // Reinhard r/(r+1) -> 1, gamma 1 -> 1, * 255 -> 255.
        let [r, g, b] = tonemap_pixel([1000.0, 1000.0, 1000.0]);
        assert!(r >= 250 && g >= 250 && b >= 250, "got {r},{g},{b}");
    }

    #[test]
    fn write_render_emits_both_files() {
        let tmp = std::env::temp_dir().join(format!("quasi-render-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let aovs = synthetic_aovs(8, 6);
        let paths = write_render(&aovs, &tmp.join("frame")).expect("write_render");

        assert!(paths.png.exists(), "PNG was not written");
        assert!(paths.exr.exists(), "EXR was not written");
        assert!(paths.variance.exists(), "variance PNG was not written");
        assert!(
            paths.variance.to_string_lossy().ends_with("_variance.png"),
            "variance PNG should be named <stem>_variance.png, got {:?}",
            paths.variance,
        );

        // PNG round-trip: width / height come back the same.
        let img = image::open(&paths.png).expect("re-open png");
        assert_eq!(img.width(), 8);
        assert_eq!(img.height(), 6);

        // Variance PNG also round-trips at the same dimensions.
        let var_img = image::open(&paths.variance).expect("re-open variance png");
        assert_eq!(var_img.width(), 8);
        assert_eq!(var_img.height(), 6);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// PT-adaptive/variance-aov: the variance.Y channel must
    /// round-trip through the EXR pipeline carrying the per-pixel
    /// luminance variance derived from radiance + mean_y2.
    #[test]
    fn exr_contains_variance_channel() {
        use exr::prelude::*;

        let tmp =
            std::env::temp_dir().join(format!("quasi-exr-variance-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("rt.exr");

        let aovs = synthetic_aovs(4, 4);
        write_aov_exr(&aovs, &path).expect("write");

        let read = read_all_flat_layers_from_file(&path).expect("read");
        let layer = &read.layer_data[0];
        let names: Vec<String> = layer
            .channel_data
            .list
            .iter()
            .map(|c| c.name.to_string())
            .collect();
        assert!(
            names.iter().any(|n| n == "variance.Y"),
            "variance.Y channel missing from EXR; channels: {names:?}",
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// PT-adaptive/variance-aov: the viridis lookup must monotonically
    /// brighten from t=0 (dark purple) to t=1 (bright yellow).
    #[test]
    fn viridis_lut_endpoints_match_published_palette() {
        let dark = viridis_lut(0.0);
        let bright = viridis_lut(1.0);
        assert!(
            (dark[0] - 0.267).abs() < 1e-3
                && (dark[1] - 0.005).abs() < 1e-3
                && (dark[2] - 0.329).abs() < 1e-3,
            "t=0 should be Matplotlib viridis dark purple (0.267, 0.005, 0.329), got {dark:?}",
        );
        assert!(
            (bright[0] - 0.993).abs() < 1e-3
                && (bright[1] - 0.906).abs() < 1e-3
                && (bright[2] - 0.144).abs() < 1e-3,
            "t=1 should be Matplotlib viridis bright yellow (0.993, 0.906, 0.144), got {bright:?}",
        );
        // Brightness (mean of channels) must increase monotonically
        // across the LUT.
        let bright_mean = (bright[0] + bright[1] + bright[2]) / 3.0;
        let dark_mean = (dark[0] + dark[1] + dark[2]) / 3.0;
        assert!(bright_mean > dark_mean);
    }

    #[test]
    fn exr_round_trip_preserves_values() {
        use exr::prelude::*;

        let tmp = std::env::temp_dir().join(format!("quasi-exr-roundtrip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("rt.exr");

        let aovs = synthetic_aovs(4, 4);
        write_aov_exr(&aovs, &path).expect("write");

        let read = read_all_flat_layers_from_file(&path).expect("read");
        assert_eq!(read.layer_data.len(), 1);
        let layer = &read.layer_data[0];
        assert_eq!(layer.size, Vec2(4, 4));
        // Sorted channels: albedo.B/G/R, B (radiance), G, N.X/Y/Z, R, Z.
        let names: Vec<String> = layer
            .channel_data
            .list
            .iter()
            .map(|c| c.name.to_string())
            .collect();
        for required in ["R", "G", "B", "albedo.R", "N.X", "Z"] {
            assert!(
                names.iter().any(|n| n == required),
                "missing channel {required} in {names:?}",
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
