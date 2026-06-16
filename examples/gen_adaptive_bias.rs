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

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Per-scene render preset. Cornell is the default; passing
/// "sponza" on the command line swaps in the iconic sun-lit
/// camera + lower spp budgets to keep wall-clock manageable on
/// a 262K-triangle scene.
struct Preset {
    label: &'static str,
    scene_path: &'static str,
    w: u32,
    h: u32,
    reference_spp: u32,
    test_spp_values: Vec<u32>,
    cfg: RenderConfig,
}

fn cornell_preset() -> Preset {
    Preset {
        label: "cornell-glass-bunny",
        scene_path: "data/gltf/cornell_glass_bunny.gltf",
        w: 192,
        h: 192,
        reference_spp: 8192,
        test_spp_values: vec![256, 1024, 2048],
        cfg: RenderConfig::default(),
    }
}

/// Cornell-glass-sphere preset: the classic Veach-style caustic
/// test scene. A glass sphere in a Cornell box concentrates the
/// area light into a sharp caustic on the floor. The bulk of the
/// frame is moderate-variance diffuse, but the caustic region is
/// pathologically high-variance (rare specular paths from light →
/// glass refraction → eye). This is the regime where PT-adaptive's
/// budget redirection should pay off — most pixels converge at
/// `min_spp` and free up the budget for the small caustic region.
fn caustic_preset() -> Preset {
    Preset {
        label: "cornell-glass-sphere",
        scene_path: "data/gltf/cornell_glass_sphere.gltf",
        // 128×128 keeps the reference fast; 4096-spp reference is
        // enough to see the caustic's true mean without massive
        // wall-clock cost.
        w: 128,
        h: 128,
        reference_spp: 4096,
        test_spp_values: vec![256, 1024, 2048],
        cfg: RenderConfig::default(),
    }
}

fn sponza_preset() -> Preset {
    let camera_pos = [-10.0, 2.0, 0.0];
    let look_at = [10.0, 4.0, 0.0];
    let dir = normalize3([
        look_at[0] - camera_pos[0],
        look_at[1] - camera_pos[1],
        look_at[2] - camera_pos[2],
    ]);
    let intensity = 18.0_f32;
    let sun_color = [1.0 * intensity, 0.95 * intensity, 0.82 * intensity];
    Preset {
        label: "sponza",
        scene_path: "data/gltf/sponza/Sponza.gltf",
        // 128×128 + reference 2048 spp keeps wall-clock under
        // ~3 minutes on M-series; enough to see variance
        // heterogeneity but not lock the machine.
        w: 128,
        h: 128,
        reference_spp: 2048,
        test_spp_values: vec![128, 512, 1024],
        cfg: RenderConfig {
            camera_pos,
            camera_dir: dir,
            fov: 55.0,
            sun_dir: Some([0.1, 1.0, 0.1]),
            sun_color,
            ..RenderConfig::default()
        },
    }
}

fn main() {
    let preset_name: String = std::env::args().nth(1).unwrap_or_else(|| "cornell".into());
    let preset = match preset_name.as_str() {
        "sponza" => sponza_preset(),
        "caustic" => caustic_preset(),
        _ => cornell_preset(),
    };
    eprintln!("[preset: {}]", preset.label);

    let scene =
        mesh::load_glb(Path::new(preset.scene_path)).unwrap_or_else(|_| default_triangle_scene());

    let w = preset.w;
    let h = preset.h;
    let reference_spp = preset.reference_spp;
    let test_spp_values = preset.test_spp_values.clone();

    let base = RenderConfig {
        width: w,
        height: h,
        samples: reference_spp,
        ..preset.cfg
    };

    eprintln!("[reference]");
    let reference = render("reference", base, &scene);

    eprintln!();
    eprintln!("[equal-sample-budget sweep]");
    println!();
    println!(
        "Equal-sample-budget comparison: each row picks an adaptive\n\
         max-spp ceiling, runs adaptive, reads back the total samples\n\
         drawn, then runs fixed-spp at that equivalent budget. The\n\
         ratio is the apples-to-apples efficiency the plan rev-3\n\
         Done-when targets (≤ 0.7 on ≥ 2 of 3 tiers).\n",
    );
    println!(
        "{:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>10}",
        "max-spp", "adapt-spp", "fixed-spp", "fixed RMSE", "adaptive RMSE", "ratio (a/f)",
    );
    println!(
        "{:>10}  {:>10}  {:>10}  {:>14}  {:>14}  {:>10}",
        "----------", "----------", "----------", "----------", "-------------", "-----------",
    );
    // PT-adaptive: noise threshold. The plan-default `0.01` is
    // tight enough that few pixels converge within typical
    // test-budget windows, so the redistribution gain is small.
    // Setting `ADAPT_BIAS_THRESHOLD` env var lets a measurement
    // run loosen the threshold to see if the win shows up under
    // more aggressive early-termination.
    let noise_threshold: f32 = std::env::var("ADAPT_BIAS_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.01);
    eprintln!("[noise_threshold: {noise_threshold}]");

    // PT-adaptive-budget-driven: lift max_spp well above the
    // average target so the budget extension can actually fire.
    // The scheduler is supposed to redistribute the budget freed
    // by converged pixels onto hard pixels — that only happens if
    // the per-pixel ceiling is above the target average.
    let max_spp_multiplier: u32 = std::env::var("ADAPT_BIAS_MAX_SPP_MULT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    for &spp in &test_spp_values {
        let mut adapt_cfg = base;
        adapt_cfg.samples = spp;
        adapt_cfg.adaptive = Some(AdaptiveConfig {
            noise_threshold,
            min_spp: 64,
            max_spp: spp.saturating_mul(max_spp_multiplier),
        });
        let adapt_aovs = render(&format!("adaptive @ max-spp {spp}"), adapt_cfg, &scene);
        let adapt_equiv = adapt_aovs.fixed_spp_equivalent().max(1);

        // Configure the fixed-spp control at the *adaptive equivalent*
        // budget — this is the equal-sample-budget comparison the
        // plan rev-3 specifies. The fixed control draws roughly the
        // same total samples as adaptive did across the frame.
        let mut fixed_cfg = base;
        fixed_cfg.samples = adapt_equiv;
        fixed_cfg.adaptive = None;
        let fixed_aovs = render(
            &format!("fixed @ {} spp (equiv)", adapt_equiv),
            fixed_cfg,
            &scene,
        );

        let fixed_rmse = rmse(&fixed_aovs, &reference);
        let adapt_rmse = rmse(&adapt_aovs, &reference);
        let ratio = adapt_rmse / fixed_rmse.max(1e-9);
        println!(
            "{spp:>10}  {:>10}  {adapt_equiv:>10}  {fixed_rmse:>14.6}  {adapt_rmse:>14.6}  {ratio:>10.3}",
            spp,
        );
    }

    println!();
    println!("Plan-0028 PT-adaptive/bias-check sample-efficiency target:");
    println!("  ratio ≤ 0.7 on ≥ 2 of 3 tiers indicates adaptive wins.");
    println!("  Numbers above go into plan 0028 `Findings` section.");
}
