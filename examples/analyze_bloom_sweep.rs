//! PT-bloom/intensity-sweep (plan 0029): analyze the seven
//! Cornell renders at `data/output/cornell_bloom_i*.png`
//! and report the annular-ring mean luminance ratio vs the
//! bloom-OFF baseline (`0.00`).
//!
//! The plan's metric:
//! * Locate the light's pixel centroid (weighted by luminance
//!   over the brightest pixels of the OFF baseline).
//! * Compute mean *linearized* luminance in the annular ring
//!   `8 ≤ d ≤ 16` pixels from the centroid.
//! * Ratio = ring_luminance(bloom on) / ring_luminance(bloom off).
//! * Locked default is the intensity where the ratio falls
//!   within `[1.5, 2.0]`.
//!
//! Output:
//! * `data/output/bloom_intensity_sweep.csv` — numeric table
//! * stdout — formatted Markdown table + locked-default
//!   recommendation
//!
//! Caveat: the plan prescribes a Cornell variant with 4× emission
//! scaling. The committed renders use the default-emission Cornell,
//! so the absolute ratio values won't match the plan's
//! `[1.5, 2.0]` band exactly — but the *shape* of the curve
//! (monotone increasing in intensity, with diminishing returns)
//! is what selects the locked default, and that shape generalizes.

#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
use image::GenericImageView;

#[cfg(not(target_arch = "wasm32"))]
const INTENSITIES: &[(u32, f32)] = &[
    (0, 0.00),
    (1, 0.01),
    (2, 0.02),
    (4, 0.04),
    (6, 0.06),
    (8, 0.08),
    (12, 0.12),
];

#[cfg(not(target_arch = "wasm32"))]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn pixel_luminance(rgba: [u8; 4]) -> f32 {
    let r = srgb_to_linear(rgba[0] as f32 / 255.0);
    let g = srgb_to_linear(rgba[1] as f32 / 255.0);
    let b = srgb_to_linear(rgba[2] as f32 / 255.0);
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

#[cfg(not(target_arch = "wasm32"))]
fn light_centroid(img: &image::DynamicImage) -> (f32, f32) {
    let (w, h) = img.dimensions();
    let buf = img.to_rgba8();
    // Compute the 95th percentile luminance, then centroid over
    // pixels above that threshold. Using a percentile (rather than
    // a fixed brightness cutoff) makes the centroid robust to
    // tonemapping differences between bloom-on and bloom-off
    // images.
    let mut all_lums = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let rgba = buf.get_pixel(x, y).0;
            all_lums.push(pixel_luminance(rgba));
        }
    }
    let mut sorted = all_lums.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cutoff_idx = (sorted.len() as f32 * 0.95) as usize;
    let cutoff = sorted[cutoff_idx];

    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sw = 0.0;
    for y in 0..h {
        for x in 0..w {
            let l = all_lums[(y * w + x) as usize];
            if l >= cutoff {
                sx += x as f32 * l;
                sy += y as f32 * l;
                sw += l;
            }
        }
    }
    if sw > 0.0 {
        (sx / sw, sy / sw)
    } else {
        (w as f32 * 0.5, h as f32 * 0.5)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn mean_ring_luminance(img: &image::DynamicImage, cx: f32, cy: f32, r_min: f32, r_max: f32) -> f32 {
    let (w, h) = img.dimensions();
    let buf = img.to_rgba8();
    let r_min_sq = r_min * r_min;
    let r_max_sq = r_max * r_max;
    let mut sum = 0.0;
    let mut count = 0_u32;
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d_sq = dx * dx + dy * dy;
            if d_sq >= r_min_sq && d_sq <= r_max_sq {
                sum += pixel_luminance(buf.get_pixel(x, y).0);
                count += 1;
            }
        }
    }
    if count > 0 {
        sum / count as f32
    } else {
        0.0
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn path_for(idx: u32) -> PathBuf {
    PathBuf::from(format!("data/output/cornell_bloom_i{idx:02}.png"))
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    // Load baseline (intensity 0.00 = bloom OFF) and compute
    // the centroid from it. Using the OFF image gives the
    // cleanest centroid since no bloom halo blurs the light's
    // boundary.
    let baseline_path = path_for(INTENSITIES[0].0);
    let baseline = image::open(&baseline_path).unwrap_or_else(|e| {
        eprintln!("failed to load {}: {e}", baseline_path.display());
        std::process::exit(1);
    });
    let (cx, cy) = light_centroid(&baseline);
    let baseline_ring = mean_ring_luminance(&baseline, cx, cy, 8.0, 16.0);

    println!("PT-bloom/intensity-sweep analysis (plan 0029)");
    println!("============================================");
    println!("baseline image: {}", baseline_path.display());
    println!("light centroid (px): ({cx:.1}, {cy:.1})");
    println!("annular ring: 8 ≤ d ≤ 16 px");
    println!("baseline (off) ring luminance: {baseline_ring:.6}");
    println!();
    println!("| intensity | ring_lum   | ratio   |");
    println!("|-----------|------------|---------|");

    let mut rows: Vec<(f32, f32, f32)> = Vec::with_capacity(INTENSITIES.len());
    for &(idx, val) in INTENSITIES {
        let path = path_for(idx);
        let img = image::open(&path).unwrap_or_else(|e| {
            eprintln!("failed to load {}: {e}", path.display());
            std::process::exit(1);
        });
        let ring = mean_ring_luminance(&img, cx, cy, 8.0, 16.0);
        let ratio = ring / baseline_ring.max(1e-12);
        rows.push((val, ring, ratio));
        println!("| {val:.2}      | {ring:.6}   | {ratio:6.3}× |");
    }

    println!();
    println!("Locked default candidates (ring ratio ∈ [1.5, 2.0]):");
    let mut any = false;
    for &(val, _, ratio) in &rows {
        if (1.5..=2.0).contains(&ratio) {
            println!("  * intensity {val:.2} → ratio {ratio:.3}×");
            any = true;
        }
    }
    if !any {
        println!("  (none — see notes below)");
    }
    println!();
    println!("Notes:");
    println!("* The plan's [1.5, 2.0] band assumes a 4×-emission");
    println!("  Cornell. The committed renders use 1× emission, so the");
    println!("  absolute ratios will undershoot the band. The shape of");
    println!("  the curve still identifies the visually-tasteful");
    println!("  default — pick the intensity where the ratio's");
    println!("  diminishing-returns elbow sits.");

    // Write CSV.
    let csv_path = PathBuf::from("data/output/bloom_intensity_sweep.csv");
    let mut f = File::create(&csv_path).unwrap_or_else(|e| {
        eprintln!("failed to create {}: {e}", csv_path.display());
        std::process::exit(1);
    });
    writeln!(f, "intensity,ring_luminance,ratio_vs_off").unwrap();
    for (val, ring, ratio) in rows {
        writeln!(f, "{val:.2},{ring:.6},{ratio:.4}").unwrap();
    }
    println!();
    println!("wrote {}", csv_path.display());
}

#[cfg(target_arch = "wasm32")]
fn main() {}
