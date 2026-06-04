//! Native entry point. Initializes logging and runs one of Quasi's commands.
//!
//! Usage:
//!
//! ```text
//! cargo run                       # path tracer window (default)
//! cargo run -- raster             # rasterizer window
//! cargo run -- render [opts]      # offscreen render -> base.png + base.exr
//! cargo run -- converge [opts]    # convergence sweep -> runs.csv
//! ```
//!
//! `render` options: `--out <base>` `--width N` `--height N` `--spp N`
//! `--sampler pcg|halton|sobol` `--integrator misnee|bsdf`.
//!
//! `converge` options: `--out <csv>` `--width N` `--height N`
//! `--max-spp N` `--reference-spp N`.

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
use quasi::pathtrace::converge::{self, ConvergeConfig};
#[cfg(not(target_arch = "wasm32"))]
use quasi::pathtrace::default_triangle_scene;
#[cfg(not(target_arch = "wasm32"))]
use quasi::pathtrace::integrator::IntegratorKind;
#[cfg(not(target_arch = "wasm32"))]
use quasi::pathtrace::offscreen::{render_offscreen, RenderConfig};
#[cfg(not(target_arch = "wasm32"))]
use quasi::pathtrace::output::write_render;
#[cfg(not(target_arch = "wasm32"))]
use quasi::pathtrace::sampler::SamplerKind;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("render") => {
            let rest: Vec<String> = args.collect();
            run_render(&rest);
        }
        Some("converge") => {
            let rest: Vec<String> = args.collect();
            run_converge(&rest);
        }
        Some("raster") | Some("--raster") => quasi::run_raster(),
        Some("pathtrace") | Some("--pathtrace") | None => quasi::run(),
        Some(other) => {
            eprintln!("unknown command: {other}");
            eprintln!("usage: cargo run -- [pathtrace | raster | render [opts] | converge [opts]]");
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------
// `render` subcommand
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
struct RenderArgs {
    out: PathBuf,
    width: u32,
    height: u32,
    samples: u32,
    sampler: SamplerKind,
    integrator: IntegratorKind,
    /// `--scene path.gltf` to load a custom triangle scene; default
    /// uses the Cornell Box embedded in the binary.
    scene: Option<PathBuf>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for RenderArgs {
    fn default() -> Self {
        Self {
            out: PathBuf::from("frame"),
            width: 512,
            height: 512,
            samples: 256,
            sampler: SamplerKind::default(),
            integrator: IntegratorKind::default(),
            scene: None,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_render_args(args: &[String]) -> Result<RenderArgs, String> {
    let mut r = RenderArgs::default();
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--out" | "-o" => {
                r.out = PathBuf::from(
                    iter.next()
                        .ok_or_else(|| "--out needs a path".to_string())?,
                );
            }
            "--width" | "-w" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--width needs a number".to_string())?;
                r.width = v.parse().map_err(|e| format!("--width: {e}"))?;
            }
            "--height" | "-h" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--height needs a number".to_string())?;
                r.height = v.parse().map_err(|e| format!("--height: {e}"))?;
            }
            "--spp" | "--samples" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--spp needs a number".to_string())?;
                r.samples = v.parse().map_err(|e| format!("--spp: {e}"))?;
            }
            "--sampler" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--sampler needs a name".to_string())?;
                r.sampler = v.parse()?;
            }
            "--integrator" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--integrator needs a name".to_string())?;
                r.integrator = v.parse()?;
            }
            "--scene" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--scene needs a path".to_string())?;
                r.scene = Some(PathBuf::from(v));
            }
            "--help" | "-?" => {
                println!(
                    "render options:\n\
                     \t--out <base>        output basename (default: frame)\n\
                     \t--width N           image width  (default: 512)\n\
                     \t--height N          image height (default: 512)\n\
                     \t--spp N             samples per pixel (default: 256)\n\
                     \t--sampler NAME      pcg | halton | sobol (default: pcg)\n\
                     \t--integrator NAME   misnee | bsdf (default: misnee)\n\
                     \t--scene PATH        load a custom glTF scene (default: embedded Cornell)"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown render option: {other}")),
        }
    }
    Ok(r)
}

#[cfg(not(target_arch = "wasm32"))]
fn run_render(args: &[String]) {
    let cli = parse_render_args(args).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let cfg = RenderConfig {
        width: cli.width,
        height: cli.height,
        samples: cli.samples,
        sampler: cli.sampler,
        integrator: cli.integrator,
        ..RenderConfig::default()
    };
    log::info!(
        "rendering {}x{} @ {} spp ({:?} / {:?})",
        cfg.width,
        cfg.height,
        cfg.samples,
        cfg.sampler,
        cfg.integrator,
    );
    let scene = match cli.scene.as_deref() {
        Some(path) => quasi::pathtrace::mesh::load_glb(path).unwrap_or_else(|e| {
            eprintln!("failed to load --scene {}: {e}", path.display());
            std::process::exit(1);
        }),
        None => default_triangle_scene(),
    };
    log::info!(
        "scene: {} triangles, {} emissive",
        scene.triangle_count(),
        scene.emissive_triangles.len(),
    );
    let start = std::time::Instant::now();
    let aovs = render_offscreen(cfg, &scene);
    let render_dur = start.elapsed();
    log::info!(
        "render took {:.2}s ({} samples)",
        render_dur.as_secs_f64(),
        cfg.samples
    );

    let encode_start = std::time::Instant::now();
    let paths = write_render(&aovs, &cli.out).unwrap_or_else(|e| {
        eprintln!("output error: {e}");
        std::process::exit(1);
    });
    log::info!(
        "encoded PNG + EXR in {:.2}s",
        encode_start.elapsed().as_secs_f64()
    );
    println!("wrote {}", paths.png.display());
    println!("wrote {}", paths.exr.display());
}

// ---------------------------------------------------------------------------
// `converge` subcommand
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
struct ConvergeArgs {
    out: PathBuf,
    cfg: ConvergeConfig,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for ConvergeArgs {
    fn default() -> Self {
        Self {
            out: PathBuf::from("convergence.csv"),
            cfg: ConvergeConfig::default(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_converge_args(args: &[String]) -> Result<ConvergeArgs, String> {
    let mut r = ConvergeArgs::default();
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--out" | "-o" => {
                r.out = PathBuf::from(
                    iter.next()
                        .ok_or_else(|| "--out needs a path".to_string())?,
                );
            }
            "--width" | "-w" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--width needs a number".to_string())?;
                r.cfg.width = v.parse().map_err(|e| format!("--width: {e}"))?;
            }
            "--height" | "-h" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--height needs a number".to_string())?;
                r.cfg.height = v.parse().map_err(|e| format!("--height: {e}"))?;
            }
            "--max-spp" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--max-spp needs a number".to_string())?;
                r.cfg.max_spp = v.parse().map_err(|e| format!("--max-spp: {e}"))?;
            }
            "--reference-spp" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--reference-spp needs a number".to_string())?;
                r.cfg.reference_spp = v.parse().map_err(|e| format!("--reference-spp: {e}"))?;
            }
            "--help" | "-?" => {
                println!(
                    "converge options:\n\
                     \t--out <csv>            CSV output path (default: convergence.csv)\n\
                     \t--width N              image width  (default: 256)\n\
                     \t--height N             image height (default: 256)\n\
                     \t--max-spp N            largest spp in the sweep (default: 1024)\n\
                     \t--reference-spp N      spp for the ground-truth reference (default: 4096)"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown converge option: {other}")),
        }
    }
    Ok(r)
}

#[cfg(not(target_arch = "wasm32"))]
fn run_converge(args: &[String]) {
    let cli = parse_converge_args(args).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let start = std::time::Instant::now();
    let rows = converge::run(cli.cfg, &cli.out).unwrap_or_else(|e| {
        eprintln!("converge error: {e}");
        std::process::exit(1);
    });
    log::info!(
        "convergence sweep took {:.2}s ({} rows)",
        start.elapsed().as_secs_f64(),
        rows.len(),
    );
    println!("wrote {} rows to {}", rows.len(), cli.out.display());
}

#[cfg(target_arch = "wasm32")]
fn main() {}
