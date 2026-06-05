//! Bakes a small procedural sky HDR into `data/env/synthetic_sky.hdr`.
//!
//! Used as a baked, deterministic fixture for the PT-env smoke
//! render. Three components, equirectangular:
//!
//! * Sky gradient — Preetham-ish: bright pale-yellow horizon, deeper
//!   blue zenith.
//! * Ground — solid muddy brown below the horizon, ≈1.5× dimmer than
//!   the horizon so NEE picks the sky.
//! * Sun disc — a single bright spot at φ = 80°, θ = 35° (mid-afternoon
//!   altitude), radiance ≈ 200×.
//!
//! A real PolyHaven HDR replaces this for the published showcase
//! render; this one exists so the loader → table → WGSL inverse-CDF
//! path can be exercised without a network fetch.

use std::fs;
use std::io::BufWriter;
use std::path::PathBuf;

use image::codecs::hdr::HdrEncoder;
use image::Rgb;

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn synth_sky(width: u32, height: u32) -> Vec<Rgb<f32>> {
    let mut out = Vec::with_capacity((width * height) as usize);

    let horizon = [1.10_f32, 0.95, 0.75];
    let zenith = [0.30_f32, 0.45, 0.85];
    let ground = [0.18_f32, 0.13, 0.08];

    let sun_dir = {
        let theta = 35.0_f32.to_radians();
        let phi = 80.0_f32.to_radians();
        let st = theta.sin();
        [st * phi.cos(), theta.cos(), st * phi.sin()]
    };
    let sun_radiance = [200.0_f32, 190.0, 160.0];
    let sun_cos_threshold = 0.997; // half-angle ≈ 4.4°

    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        let theta = v * std::f32::consts::PI;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let phi = u * 2.0 * std::f32::consts::PI;
            let dir = [sin_t * phi.cos(), cos_t, sin_t * phi.sin()];

            let mut rgb = if cos_t > 0.0 {
                let t = (1.0 - cos_t).powf(0.8);
                lerp3(zenith, horizon, t)
            } else {
                ground
            };

            let cos_sun = dir[0] * sun_dir[0] + dir[1] * sun_dir[1] + dir[2] * sun_dir[2];
            if cos_sun > sun_cos_threshold {
                rgb = sun_radiance;
            }

            out.push(Rgb(rgb));
        }
    }
    out
}

fn main() {
    let out_dir = PathBuf::from("data/env");
    fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("mkdir {}: {e}", out_dir.display()));

    let width = 512_u32;
    let height = 256_u32;
    let pixels = synth_sky(width, height);

    let path = out_dir.join("synthetic_sky.hdr");
    let file = fs::File::create(&path).unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    let writer = BufWriter::new(file);
    let encoder = HdrEncoder::new(writer);
    encoder
        .encode(&pixels, width as usize, height as usize)
        .unwrap_or_else(|e| panic!("encode HDR: {e}"));

    let max_lum = pixels
        .iter()
        .map(|p| 0.2126 * p.0[0] + 0.7152 * p.0[1] + 0.0722 * p.0[2])
        .fold(0.0_f32, f32::max);
    let mean_lum = pixels
        .iter()
        .map(|p| 0.2126 * p.0[0] + 0.7152 * p.0[1] + 0.0722 * p.0[2])
        .sum::<f32>()
        / pixels.len() as f32;
    println!(
        "wrote {} ({}×{}, mean lum {:.3}, peak lum {:.0})",
        path.display(),
        width,
        height,
        mean_lum,
        max_lum,
    );
}
