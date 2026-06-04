//! Convergence study runner.
//!
//! Renders a high-spp reference, then sweeps each (sampler, integrator)
//! combination across a doubling spp schedule, writing one CSV row per
//! sample-count with RMSE and relative-MSE vs the reference. The
//! resulting CSV is the source data for the "watch convergence" blog
//! story.

use std::io::Write;
use std::path::Path;

use crate::pathtrace::default_triangle_scene;
use crate::pathtrace::integrator::IntegratorKind;
use crate::pathtrace::mesh::{load_glb, TriangleScene};
use crate::pathtrace::metrics;
use crate::pathtrace::offscreen::{render_offscreen, RenderConfig};
use crate::pathtrace::sampler::SamplerKind;

/// Knobs for the convergence sweep.
#[derive(Clone, Debug)]
pub struct ConvergeConfig {
    pub width: u32,
    pub height: u32,
    /// Largest spp the sweep renders.
    pub max_spp: u32,
    /// The high-spp ground truth that every series is scored against.
    /// Rendered once with the lowest-variance pair (PCG + MIS+NEE).
    pub reference_spp: u32,
    /// Optional custom glTF scene. `None` uses the embedded Cornell
    /// default.
    pub scene_path: Option<std::path::PathBuf>,
}

impl Default for ConvergeConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            max_spp: 1024,
            reference_spp: 4096,
            scene_path: None,
        }
    }
}

/// One row of the convergence CSV.
#[derive(Clone, Copy, Debug)]
pub struct ConvergeRow {
    pub sampler: SamplerKind,
    pub integrator: IntegratorKind,
    pub spp: u32,
    pub rmse: f64,
    pub rel_mse: f64,
}

/// Powers of two from 1 up to and including `max_spp`. If `max_spp`
/// isn't a power of two, it's still included as the last point — the
/// CSV shouldn't have a phantom gap between the last 2^k and the user's
/// requested cap.
pub fn spp_schedule(max_spp: u32) -> Vec<u32> {
    let mut v = Vec::new();
    let mut s = 1u32;
    while s < max_spp {
        v.push(s);
        s = s.saturating_mul(2);
    }
    v.push(max_spp);
    v.dedup();
    v
}

/// Runs the sweep and writes the CSV. Returns the number of data rows
/// written (i.e. excluding the header).
pub fn run(cfg: ConvergeConfig, out_path: &Path) -> std::io::Result<Vec<ConvergeRow>> {
    let scene: TriangleScene = match cfg.scene_path.as_deref() {
        Some(path) => load_glb(path).map_err(|e| std::io::Error::other(e.to_string()))?,
        None => default_triangle_scene(),
    };
    log::info!(
        "rendering reference: {}x{} @ {} spp (pcg + misnee, {} triangles)",
        cfg.width,
        cfg.height,
        cfg.reference_spp,
        scene.triangle_count(),
    );
    let ref_aovs = render_offscreen(
        RenderConfig {
            width: cfg.width,
            height: cfg.height,
            samples: cfg.reference_spp,
            sampler: SamplerKind::Pcg,
            integrator: IntegratorKind::MisNee,
            ..RenderConfig::default()
        },
        &scene,
    );

    let schedule = spp_schedule(cfg.max_spp);
    let combos = [SamplerKind::Pcg, SamplerKind::Halton, SamplerKind::Sobol];
    let integrators = [IntegratorKind::MisNee, IntegratorKind::Bsdf];
    let total = combos.len() * integrators.len() * schedule.len();
    log::info!(
        "running sweep: {} renders ({} samplers × {} integrators × {} spp checkpoints)",
        total,
        combos.len(),
        integrators.len(),
        schedule.len(),
    );

    let mut rows: Vec<ConvergeRow> = Vec::with_capacity(total);
    let mut done = 0;
    for sampler in combos {
        for integrator in integrators {
            for &spp in &schedule {
                let aovs = render_offscreen(
                    RenderConfig {
                        width: cfg.width,
                        height: cfg.height,
                        samples: spp,
                        sampler,
                        integrator,
                        ..RenderConfig::default()
                    },
                    &scene,
                );
                let rmse = metrics::rmse_rgb(&aovs.radiance, &ref_aovs.radiance);
                let rel_mse = metrics::rel_mse_rgb(&aovs.radiance, &ref_aovs.radiance);
                rows.push(ConvergeRow {
                    sampler,
                    integrator,
                    spp,
                    rmse,
                    rel_mse,
                });
                done += 1;
                log::info!(
                    "[{done}/{total}] {} / {} @ {spp} spp -> rmse={rmse:.6} rel_mse={rel_mse:.6}",
                    sampler.label(),
                    integrator.label(),
                );
            }
        }
    }

    let mut file = std::fs::File::create(out_path)?;
    writeln!(file, "sampler,integrator,spp,rmse,rel_mse")?;
    for r in &rows {
        writeln!(
            file,
            "{},{},{},{:.6},{:.6}",
            r.sampler.label(),
            r.integrator.label(),
            r.spp,
            r.rmse,
            r.rel_mse,
        )?;
    }
    log::info!("wrote {} rows to {}", rows.len(), out_path.display());
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_doubles_then_caps() {
        assert_eq!(spp_schedule(1), vec![1]);
        assert_eq!(spp_schedule(8), vec![1, 2, 4, 8]);
        assert_eq!(
            spp_schedule(1024),
            vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        );
    }

    #[test]
    fn schedule_includes_non_power_of_two_cap() {
        // 100 isn't a power of two — it should still appear as the
        // final checkpoint so the CSV ends at the requested spp.
        let s = spp_schedule(100);
        assert_eq!(s.last().copied(), Some(100));
        assert!(s.contains(&64));
        assert!(!s.contains(&128));
    }
}
