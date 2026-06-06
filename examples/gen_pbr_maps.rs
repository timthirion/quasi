//! Bakes procedural PBR texture maps used by the PT-pbr-maps showcase
//! scenes:
//!
//! * `data/textures/brushed_brass_mr.png` — glTF metallic-roughness
//!   map. R channel unused. **G channel = roughness**; horizontal
//!   brushed streaks (anisotropic 1-D noise) on a low roughness
//!   floor, modulated by low-frequency dirt. **B channel = metallic**;
//!   nearly all 1.0 (full brass) with a few patchy 0.4 zones suggesting
//!   tarnish.
//! * `data/textures/stone_tile_normal.png` — glTF normal map.
//!   Repeating cobble pattern (square cells with bevelled edges +
//!   per-cell pebble fbm). OpenGL convention: +Y up. Pure xyz →
//!   stored as (R, G, B) = (nx*0.5+0.5, ny*0.5+0.5, nz*0.5+0.5).
//!
//! Output is 256², R8G8B8A8 PNG. Deterministic seed → reproducible
//! bytes per `cargo run --example gen_pbr_maps`.

use std::fs;
use std::path::PathBuf;

use image::{ImageBuffer, Rgba};

/// PCG xor-shift: deterministic, no external dep.
fn pcg_hash(seed: u32) -> u32 {
    let state = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

/// Float in `[0, 1)` from a seed. Useful for procedural sampling.
fn frand(seed: u32) -> f32 {
    (pcg_hash(seed) as f32) / (u32::MAX as f32)
}

/// 2-D value noise via bilinear lerp of integer-grid frand samples.
fn value_noise_2d(x: f32, y: f32, seed: u32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let s = |ix: i32, iy: i32| -> f32 {
        let hx = (ix as u32).wrapping_mul(73856093);
        let hy = (iy as u32).wrapping_mul(19349663);
        frand(seed ^ hx ^ hy)
    };
    let s00 = s(x0, y0);
    let s10 = s(x0 + 1, y0);
    let s01 = s(x0, y0 + 1);
    let s11 = s(x0 + 1, y0 + 1);
    // Smoothstep for visual softness.
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let a = s00 * (1.0 - ux) + s10 * ux;
    let b = s01 * (1.0 - ux) + s11 * ux;
    a * (1.0 - uy) + b * uy
}

/// Multi-octave fbm.
fn fbm_2d(x: f32, y: f32, octaves: u32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut total_amp = 0.0;
    for o in 0..octaves {
        sum += amp * value_noise_2d(x * freq, y * freq, seed.wrapping_add(o));
        total_amp += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / total_amp
}

fn brushed_brass_mr(width: u32, height: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    // Roughness pattern: horizontal brushed streaks (1-D noise along U
    // with fine high-freq dither). Mean roughness ~0.45 — enough to
    // soften the GGX lobe and let the brushed direction read clearly,
    // but not so high that the metallic specular disappears.
    let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width, height);
    let seed_streaks = 0x9E37_79B9_u32;
    let seed_dirt = 0x5BCE_E2A1_u32;
    let seed_tarnish = 0x1F23_44C5_u32;
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;

            // 1-D streaks in U (horizontal), high-freq stripe noise +
            // a slow modulation so the streaks come and go. Two
            // octaves at very different frequencies give the
            // brushed-metal look — fine grain + coarse band.
            let stripe_fine = (value_noise_2d(fx * 96.0, fy * 4.0, seed_streaks) - 0.5) * 0.40;
            let stripe_med =
                (value_noise_2d(fx * 24.0, fy * 2.0, seed_streaks ^ 0xA341) - 0.5) * 0.35;
            let slow = (fbm_2d(fx * 2.0, fy * 4.0, 4, seed_streaks) - 0.5) * 0.20;
            let mut rough = 0.25 + stripe_fine + stripe_med + slow;
            // Low-freq dirt — bigger blotches that bump roughness up.
            let dirt = fbm_2d(fx * 5.0, fy * 5.0, 5, seed_dirt);
            if dirt > 0.55 {
                rough += (dirt - 0.55) * 0.9;
            }
            let rough_u8 = (rough.clamp(0.08, 0.95) * 255.0) as u8;

            // Metallic: mostly 1 (full brass) with a couple of low-freq
            // tarnish patches dropping it to ~0.4.
            let tarnish = fbm_2d(fx * 3.0, fy * 3.0, 4, seed_tarnish);
            let metal = if tarnish > 0.7 {
                let t = (tarnish - 0.7) / 0.3;
                1.0 - 0.6 * t
            } else {
                1.0
            };
            let metal_u8 = (metal.clamp(0.0, 1.0) * 255.0) as u8;

            // R unused (sometimes occlusion in some pipelines — we
            // ignore it). Alpha = 255 (PNG opaque).
            img.put_pixel(x, y, Rgba([0, rough_u8, metal_u8, 255]));
        }
    }
    img
}

/// Stone-tile normal map. 6 × 6 cells across the texture; each cell
/// has bevelled edges (height ramps up from the seam, plateaus in
/// the centre) plus per-cell pebble fbm so adjacent tiles look
/// distinct. Heights are converted to a tangent-space normal via
/// finite-difference partials, encoded as (R, G, B) = ((n+1)/2).
fn stone_tile_normal(width: u32, height: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let seed_tile = 0x4815_1623_u32;
    let cells_x = 4.0_f32;
    let cells_y = 4.0_f32;
    let bevel_width = 0.06_f32;
    let normal_strength = 3.5_f32;

    let height_at = |fx: f32, fy: f32| -> f32 {
        let cell_x = fx * cells_x;
        let cell_y = fy * cells_y;
        let ux = cell_x.fract();
        let uy = cell_y.fract();
        // Distance to nearest seam in this cell (in cell-units).
        let dx = ux.min(1.0 - ux);
        let dy = uy.min(1.0 - uy);
        let d = dx.min(dy);
        // Bevel ramp 0 at seam → 1 inside.
        let bevel = (d / bevel_width).min(1.0);
        // Smoothstep for a softer bevel.
        let bevel = bevel * bevel * (3.0 - 2.0 * bevel);
        // Per-cell pebble noise — modulated by a per-cell hash so
        // adjacent tiles read distinct.
        let cell_id = cell_x.floor() as u32 ^ (cell_y.floor() as u32).wrapping_mul(0x9E37);
        // Per-cell DC offset — small bumps in plateau height so
        // adjacent tiles read as cobbles with slightly different
        // surface heights. No per-texel noise: at the finite-
        // difference scale of 1 texel that would just produce
        // texel-grain in the normal map.
        let cell_dc = (frand(seed_tile.wrapping_add(cell_id)) - 0.5) * 0.06;
        bevel * (0.80 + cell_dc)
    };

    let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width, height);
    let eps = 1.0 / width as f32;
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            // Finite-difference partials of height with respect to UV.
            let hx = (height_at(fx + eps, fy) - height_at(fx - eps, fy)) / (2.0 * eps);
            let hy = (height_at(fx, fy + eps) - height_at(fx, fy - eps)) / (2.0 * eps);
            // Tangent-space normal: rotate (0, 0, 1) by the partials.
            let nx = -hx * normal_strength;
            let ny = -hy * normal_strength;
            let nz = 1.0;
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let nx_n = nx / len;
            let ny_n = ny / len;
            let nz_n = nz / len;
            let r = ((nx_n * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let g = ((ny_n * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let b = ((nz_n * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }
    img
}

/// Soft bunny-fur normal map. Multi-octave fbm height (low
/// amplitude) → finite-difference normal. Tiles seamlessly in
/// U via low-frequency band synthesis; no per-cell structure,
/// just organic high-freq noise.
fn bunny_fur_normal(width: u32, height: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let seed = 0x1234_5678_u32;
    let normal_strength = 1.6_f32;
    let height_at = |fx: f32, fy: f32| -> f32 {
        let h1 = fbm_2d(fx * 28.0, fy * 28.0, 4, seed);
        let h2 = fbm_2d(fx * 6.0, fy * 6.0, 3, seed.wrapping_add(0xa5a5));
        h1 * 0.05 + h2 * 0.03
    };
    let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width, height);
    let eps = 1.0 / width as f32;
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let hx = (height_at(fx + eps, fy) - height_at(fx - eps, fy)) / (2.0 * eps);
            let hy = (height_at(fx, fy + eps) - height_at(fx, fy - eps)) / (2.0 * eps);
            let nx = -hx * normal_strength;
            let ny = -hy * normal_strength;
            let nz = 1.0_f32;
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let nx_n = nx / len;
            let ny_n = ny / len;
            let nz_n = nz / len;
            let r = ((nx_n * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let g = ((ny_n * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            let b = ((nz_n * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            img.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }
    img
}

fn main() {
    let out_dir = PathBuf::from("data/textures");
    fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("mkdir {}: {e}", out_dir.display()));

    let mr_img = brushed_brass_mr(256, 256);
    let path = out_dir.join("brushed_brass_mr.png");
    mr_img
        .save(&path)
        .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));

    let total_metal: u64 = mr_img.pixels().map(|p| p.0[2] as u64).sum();
    let total_rough: u64 = mr_img.pixels().map(|p| p.0[1] as u64).sum();
    let n = (mr_img.width() * mr_img.height()) as u64;
    println!(
        "wrote {} ({}×{}, mean roughness {:.3}, mean metallic {:.3})",
        path.display(),
        mr_img.width(),
        mr_img.height(),
        (total_rough as f64 / n as f64) / 255.0,
        (total_metal as f64 / n as f64) / 255.0,
    );

    let normal_img = stone_tile_normal(256, 256);
    let stone_path = out_dir.join("stone_tile_normal.png");
    normal_img
        .save(&stone_path)
        .unwrap_or_else(|e| panic!("save {}: {e}", stone_path.display()));
    let stone_total_z: u64 = normal_img.pixels().map(|p| p.0[2] as u64).sum();
    let stone_n = (normal_img.width() * normal_img.height()) as u64;
    println!(
        "wrote {} ({}×{}, mean Z channel {:.3})",
        stone_path.display(),
        normal_img.width(),
        normal_img.height(),
        (stone_total_z as f64 / stone_n as f64) / 255.0,
    );

    // PT-vertex-tangent showcase: low-amplitude bunny-fur normal
    // map. Multi-octave fbm → finite-difference normal. Tiles
    // cleanly over the bunny's cylindrical UVs.
    let fur_img = bunny_fur_normal(256, 256);
    let fur_path = out_dir.join("bunny_fur_normal.png");
    fur_img
        .save(&fur_path)
        .unwrap_or_else(|e| panic!("save {}: {e}", fur_path.display()));
    let fur_total_z: u64 = fur_img.pixels().map(|p| p.0[2] as u64).sum();
    let fur_n = (fur_img.width() * fur_img.height()) as u64;
    println!(
        "wrote {} ({}×{}, mean Z channel {:.3})",
        fur_path.display(),
        fur_img.width(),
        fur_img.height(),
        (fur_total_z as f64 / fur_n as f64) / 255.0,
    );
}
