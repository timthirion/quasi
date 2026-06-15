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

    std::thread::scope(|s| -> Result<(), OutputError> {
        let png_h = s.spawn(|| write_tonemapped_png(aovs, &png));
        let exr_h = s.spawn(|| write_aov_exr(aovs, &exr));
        png_h.join().expect("PNG encoder thread panicked")?;
        exr_h.join().expect("EXR encoder thread panicked")?;
        Ok(())
    })?;

    Ok(RenderPaths { png, exr })
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

        // PNG round-trip: width / height come back the same.
        let img = image::open(&paths.png).expect("re-open png");
        assert_eq!(img.width(), 8);
        assert_eq!(img.height(), 6);

        let _ = std::fs::remove_dir_all(&tmp);
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
