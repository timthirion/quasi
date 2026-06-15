//! PT-adaptive/bias-check measurement harness (plan 0028).
//!
//! Renders the Cornell glass-bunny three ways at increasing spp:
//!
//! - **Reference** at high spp (8192) — the convergence target both
//!   configurations are measured against.
//! - **Adaptive** at the same `--spp` ceiling but with the
//!   `--noise-threshold 0.01 --min-spp 64` adaptive scheduler turned
//!   on.
//! - **Fixed** at the same `--spp` budget, no scheduler.
//!
//! Reports RMSE-to-reference for each. The plan-rev-3 Done-when says
//! adaptive must achieve `≤ 0.7 ×` the fixed-spp RMSE at equal sample
//! budget on the bias-check scenes; this harness produces the
//! numbers that go into the plan's `Findings` section.
//!
//! Run manually with `cargo run --release --example gen_adaptive_bias`.
//! Not part of `cargo test` — the reference render at 8192 spp takes
//! ~30 s on M-series and would tank CI throughput.

use std::path::Path;
use std::time::Instant;

use quasi::pathtrace::offscreen::{render_offscreen, AdaptiveConfig, Aovs, RenderConfig};
use quasi::pathtrace::{default_triangle_scene, mesh};

/// L2 RMSE between two radiance buffers, restricted to the linear-RGB
/// HDR space (no tonemap). Lower = closer to reference.
fn rmse(a: &Aovs, b: &Aovs) -> f32 {
    assert_eq!(a.pixel_count(), b.pixel_count());
    let mut acc = 0.0_f64;
    for i in 0..a.pixel_count() {
        let p = a.radiance[i];
        let q = b.radiance[i];
        for c in 0..3 {
            let d = (p[c] - q[c]) as f64;
            acc += d * d;
        }
    }
    ((acc / (3 * a.pixel_count()) as f64).sqrt()) as f32
}

fn render(label: &str, cfg: RenderConfig, scene: &quasi::pathtrace::mesh::TriangleScene) -> Aovs {
    let t = Instant::now();
    let aovs = render_offscreen(cfg, scene);
    let secs = t.elapsed().as_secs_f32();
    eprintln!(
        "  rendered {label}: {}×{} / {} spp / adaptive {} — {:.2}s",
        cfg.width,
        cfg.height,
        cfg.samples,
        cfg.adaptive.is_some(),
        secs,
    );
    aovs
}

fn main() {
    let scene = mesh::load_glb(Path::new("data/gltf/cornell_glass_bunny.gltf"))
        .unwrap_or_else(|_| default_triangle_scene());

    // 192×192 is small enough that the 8192-spp reference fits in
    // ~30 s on M-series. Plenty for a bias-check measurement.
    let w = 192;
    let h = 192;
    let reference_spp = 8192;
    let test_spp_values = [256_u32, 1024, 2048];

    let base = RenderConfig {
        width: w,
        height: h,
        samples: reference_spp,
        ..RenderConfig::default()
    };

    eprintln!("[reference]");
    let reference = render("reference", base, &scene);

    eprintln!();
    eprintln!("[budget sweep]");
    println!();
    println!(
        "{:>8}  {:>14}  {:>14}  {:>10}",
        "spp", "fixed RMSE", "adaptive RMSE", "ratio (a/f)"
    );
    println!(
        "{:>8}  {:>14}  {:>14}  {:>10}",
        "--------", "----------", "-------------", "-----------"
    );
    for &spp in &test_spp_values {
        let mut fixed_cfg = base;
        fixed_cfg.samples = spp;
        fixed_cfg.adaptive = None;

        let mut adapt_cfg = base;
        adapt_cfg.samples = spp;
        adapt_cfg.adaptive = Some(AdaptiveConfig {
            noise_threshold: 0.01,
            min_spp: 64,
            max_spp: spp,
        });

        let fixed_aovs = render(&format!("fixed @ {spp} spp"), fixed_cfg, &scene);
        let adapt_aovs = render(&format!("adaptive @ {spp} spp"), adapt_cfg, &scene);

        let fixed_rmse = rmse(&fixed_aovs, &reference);
        let adapt_rmse = rmse(&adapt_aovs, &reference);
        let ratio = adapt_rmse / fixed_rmse.max(1e-9);
        println!("{spp:>8}  {fixed_rmse:>14.6}  {adapt_rmse:>14.6}  {ratio:>10.3}",);
    }

    println!();
    println!("Plan-0028 PT-adaptive/bias-check sample-efficiency target:");
    println!("  ratio ≤ 0.7 on ≥ 2 of 3 spp tiers indicates adaptive wins.");
    println!("  Numbers above go into plan 0028 `Findings` section.");
}
