//! Composes a side-by-side raw-vs-denoised PNG strip for the
//! PT-denoise showcase.
//!
//! Renders `cornell_glass_bunny.gltf` at 64 spp twice (the offscreen
//! pass is cheap), denoises one copy with default parameters, then
//! stitches the two tone-mapped PNGs into a 2-column strip with a
//! 4-pixel divider. Writes to
//! `data/output/denoise_comparison.png`.

use std::path::Path;

use image::{ImageBuffer, Rgb, RgbImage};

use quasi::pathtrace::denoise::{denoise, DenoiseParams};
use quasi::pathtrace::offscreen::{render_offscreen, Aovs, RenderConfig};
use quasi::pathtrace::{default_triangle_scene, mesh};

fn aov_to_png(aovs: &Aovs) -> RgbImage {
    let w = aovs.width;
    let h = aovs.height;
    let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let r = aovs.radiance[idx];
            // Reinhard + linear→sRGB, same as `output::write_tonemapped_png`.
            let map = |c: f32| -> u8 {
                let t = c / (1.0 + c);
                let s = if t <= 0.0031308 {
                    12.92 * t
                } else {
                    1.055 * t.powf(1.0 / 2.4) - 0.055
                };
                (s.clamp(0.0, 1.0) * 255.0) as u8
            };
            img.put_pixel(x, y, Rgb([map(r[0]), map(r[1]), map(r[2])]));
        }
    }
    img
}

fn main() {
    let scene = mesh::load_glb(Path::new("data/gltf/cornell_glass_bunny.gltf"))
        .unwrap_or_else(|_| default_triangle_scene());
    let cfg = RenderConfig {
        width: 384,
        height: 384,
        samples: 64,
        ..RenderConfig::default()
    };

    let aovs = render_offscreen(cfg, &scene);
    let raw = aov_to_png(&aovs);

    let denoised_radiance = denoise(
        &aovs.radiance,
        &aovs.albedo,
        &aovs.normal,
        &aovs.depth,
        aovs.width,
        aovs.height,
        DenoiseParams::default(),
    );
    let denoised_aovs = Aovs {
        width: aovs.width,
        height: aovs.height,
        radiance: denoised_radiance,
        albedo: aovs.albedo.clone(),
        normal: aovs.normal.clone(),
        depth: aovs.depth.clone(),
    };
    let denoised = aov_to_png(&denoised_aovs);

    let divider = 4_u32;
    let w = raw.width();
    let h = raw.height();
    let mut strip = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(2 * w + divider, h);
    for y in 0..h {
        for x in 0..w {
            *strip.get_pixel_mut(x, y) = *raw.get_pixel(x, y);
            *strip.get_pixel_mut(w + divider + x, y) = *denoised.get_pixel(x, y);
        }
        for x in w..(w + divider) {
            *strip.get_pixel_mut(x, y) = Rgb([20, 20, 20]);
        }
    }
    let path = Path::new("data/output/denoise_comparison.png");
    strip
        .save(path)
        .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
    println!(
        "wrote {} ({}×{}, raw left | denoised right)",
        path.display(),
        strip.width(),
        strip.height(),
    );
}
