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
use quasi::pathtrace::offscreen::RenderConfig;
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
#[cfg_attr(test, derive(Debug))]
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
    /// `--cloud-grid path.qvg` to swap in a runtime-loaded cloud
    /// density grid (typically the output of `scripts/vdb_to_qvg.py`).
    /// Without this flag, the embedded procedural cumulus is used.
    cloud_grid: Option<PathBuf>,
    /// `--env-map path.hdr` (PT-env) attaches a Radiance HDR
    /// equirectangular environment map. Camera rays that miss the
    /// scene return the sky radiance, and NEE samples the dome via
    /// the luminance × sin θ importance tables.
    env_map: Option<PathBuf>,
    /// `--denoise` (PT-denoise) runs the analytic edge-aware
    /// à-trous wavelet denoiser as a post-process and writes
    /// `<basename>_denoised.png` alongside the raw output. The
    /// EXR write always carries the raw radiance.
    denoise: bool,
    /// `--brute-force` switches the WGSL fragment shader to a linear
    /// triangle scan; the default walks the BVH.
    brute_force: bool,
    /// `--camera-pos x,y,z` overrides the camera origin. Plan 0022
    /// adds this so Sponza-class scenes (which don't fit Cornell's
    /// (0,1,3.5)/look-down-z framing) can be aimed without
    /// recompiling.
    camera_pos: Option<[f32; 3]>,
    /// `--look-at x,y,z` overrides the camera target. Used together
    /// with `--camera-pos` to derive the view direction.
    look_at: Option<[f32; 3]>,
    /// `--fov degrees` overrides the camera vertical FOV.
    fov: Option<f32>,
    /// `--sun-dir x,y,z` (PT-sun-light, plan 0023) enables a delta-
    /// distribution directional sun. The vector is interpreted as
    /// pointing TOWARD the sun (i.e. surface normals with positive
    /// dot product are sun-facing). Disabled when None.
    sun_dir: Option<[f32; 3]>,
    /// `--sun-color r,g,b` (PT-sun-light) sets the sun's emitted
    /// radiance per steradian. Linear, can exceed 1.0. Defaults to
    /// white (1,1,1) when `--sun-dir` is provided.
    sun_color: [f32; 3],
    /// `--sun-intensity I` (PT-sun-light) is a scalar multiplier on
    /// `--sun-color`. Defaults to 1.0. Common values: 1-3 for soft
    /// ambient, 5-30 for strong direct sun, 50+ for blown-out sky.
    sun_intensity: f32,
    /// `--adaptive` (PT-adaptive, plan 0028) enables per-pixel
    /// adaptive sampling: pixels stop sampling once their per-pixel
    /// relative standard error drops below `--noise-threshold`. The
    /// `--spp` flag becomes the per-pixel ceiling (`max_spp`) when
    /// `--adaptive` is set; the actual total sample budget is
    /// determined by the scheduler. Off by default — pre-plan
    /// behaviour preserved.
    adaptive: bool,
    /// `--noise-threshold T` (PT-adaptive) is the per-pixel
    /// relative-standard-error threshold below which a pixel is
    /// considered converged. Default 0.01 (1% relative). Lower
    /// values are stricter and produce more samples on hard pixels.
    noise_threshold: f32,
    /// `--min-spp N` (PT-adaptive) is the per-pixel sample floor
    /// before the convergence check is trusted. Default 64. Heavy-
    /// tailed integrands need ≥ 64 samples before the sample
    /// variance is a reliable estimator.
    min_spp: u32,
    /// `--max-spp M` (PT-adaptive) is the per-pixel sample ceiling.
    /// When unset, defaults to `--spp`. Pixels that hit this
    /// ceiling without converging are flagged in the variance map.
    max_spp: Option<u32>,
    /// `--bloom` (PT-bloom, plan 0029) enables the HDR bloom
    /// post-process. A Kawase 4-tap dual-filter chain spreads
    /// bright pixels into a halo before the CPU tonemap, so saved
    /// PNGs show the iconic glow around sun glint, lamp filaments,
    /// emissives, etc. Off by default — pre-plan behaviour
    /// preserved.
    bloom: bool,
    /// `--bloom-intensity I` (PT-bloom): composite multiplier
    /// applied to the bloom contribution. Defaults to 0.04 — a
    /// conservative value that produces a visible halo without
    /// softening normal-bright objects.
    bloom_intensity: f32,
    /// `--bloom-threshold T` (PT-bloom): soft-knee threshold.
    /// Pixels brighter than this contribute to bloom; pixels
    /// dimmer than `threshold - knee` don't. Default 1.0 — just
    /// above what Reinhard tonemap maps to white.
    bloom_threshold: f32,
    /// `--bloom-knee K` (PT-bloom): width of the soft-knee
    /// quadratic ramp covering `[threshold - knee, threshold +
    /// knee]`. Default 0.5.
    bloom_knee: f32,
    /// `--sky` (PT-sky, plan 0030) enables the analytic
    /// Hosek-Wilkie procedural sky. At render start the sky is
    /// baked into an equirect HDR pixel buffer and routed through
    /// the same env-map path as `--env-map`. Mutually exclusive
    /// with `--env-map` and `--sun-dir`. Off by default.
    sky: bool,
    /// `--sky-elevation DEG` (PT-sky): sun elevation above the
    /// horizon, degrees. Valid range [0, 90]. Default 45.
    sky_elevation: f32,
    /// `--sky-azimuth DEG` (PT-sky): sun azimuth measured from
    /// +X toward +Z, degrees. Wraps freely. Default 180.
    sky_azimuth: f32,
    /// `--sky-turbidity T` (PT-sky): atmospheric turbidity. The
    /// model clamps to [1, 10] internally. Default 2.5.
    sky_turbidity: f32,
    /// `--sky-ground-albedo R,G,B` (PT-sky): ground albedo for
    /// horizon tint. The model clamps each channel to [0, 1]
    /// internally. Default 0.3,0.3,0.3.
    sky_ground_albedo: [f32; 3],
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
            cloud_grid: None,
            env_map: None,
            denoise: false,
            brute_force: false,
            camera_pos: None,
            look_at: None,
            fov: None,
            sun_dir: None,
            sun_color: [1.0, 1.0, 1.0],
            sun_intensity: 1.0,
            adaptive: false,
            noise_threshold: 0.01,
            min_spp: 64,
            max_spp: None,
            bloom: false,
            bloom_intensity: 0.04,
            bloom_threshold: 1.0,
            bloom_knee: 0.5,
            sky: false,
            sky_elevation: 45.0,
            sky_azimuth: 180.0,
            sky_turbidity: 2.5,
            sky_ground_albedo: [0.3, 0.3, 0.3],
        }
    }
}

/// PT-sky/wire (plan 0030): convert `(elevation_deg, azimuth_deg)` to
/// a unit direction vector under the plan's coordinate convention
/// (pinned against `env.rs` line 16):
///   φ = azimuth, measured from +X toward +Z
///   θ_zenith = π/2 - elevation
///   dir = (sin θ cos φ, cos θ, sin θ sin φ)
///       = (cos elev · cos azi, sin elev, cos elev · sin azi)
#[cfg(not(target_arch = "wasm32"))]
fn sky_dir_from_elev_azimuth(elevation_deg: f32, azimuth_deg: f32) -> [f32; 3] {
    let elev = elevation_deg.to_radians();
    let azi = azimuth_deg.to_radians();
    let (sin_elev, cos_elev) = elev.sin_cos();
    let (sin_azi, cos_azi) = azi.sin_cos();
    [cos_elev * cos_azi, sin_elev, cos_elev * sin_azi]
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_vec3(s: &str) -> Result<[f32; 3], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err(format!("expected 'x,y,z', got '{s}'"));
    }
    let x = parts[0].parse().map_err(|e| format!("x: {e}"))?;
    let y = parts[1].parse().map_err(|e| format!("y: {e}"))?;
    let z = parts[2].parse().map_err(|e| format!("z: {e}"))?;
    Ok([x, y, z])
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
            "--cloud-grid" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--cloud-grid needs a path".to_string())?;
                r.cloud_grid = Some(PathBuf::from(v));
            }
            "--env-map" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--env-map needs a path".to_string())?;
                r.env_map = Some(PathBuf::from(v));
            }
            "--denoise" => {
                r.denoise = true;
            }
            "--brute-force" => {
                r.brute_force = true;
            }
            "--camera-pos" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--camera-pos needs x,y,z".to_string())?;
                r.camera_pos = Some(parse_vec3(v).map_err(|e| format!("--camera-pos: {e}"))?);
            }
            "--look-at" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--look-at needs x,y,z".to_string())?;
                r.look_at = Some(parse_vec3(v).map_err(|e| format!("--look-at: {e}"))?);
            }
            "--fov" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--fov needs a number".to_string())?;
                r.fov = Some(v.parse().map_err(|e| format!("--fov: {e}"))?);
            }
            "--sun-dir" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--sun-dir needs x,y,z".to_string())?;
                r.sun_dir = Some(parse_vec3(v).map_err(|e| format!("--sun-dir: {e}"))?);
            }
            "--sun-color" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--sun-color needs r,g,b".to_string())?;
                r.sun_color = parse_vec3(v).map_err(|e| format!("--sun-color: {e}"))?;
            }
            "--sun-intensity" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--sun-intensity needs a number".to_string())?;
                r.sun_intensity = v.parse().map_err(|e| format!("--sun-intensity: {e}"))?;
            }
            "--adaptive" => {
                r.adaptive = true;
            }
            "--noise-threshold" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--noise-threshold needs a number".to_string())?;
                r.noise_threshold = v.parse().map_err(|e| format!("--noise-threshold: {e}"))?;
                if r.noise_threshold <= 0.0 {
                    return Err(format!(
                        "--noise-threshold must be > 0; got {}",
                        r.noise_threshold,
                    ));
                }
            }
            "--min-spp" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--min-spp needs a number".to_string())?;
                r.min_spp = v.parse().map_err(|e| format!("--min-spp: {e}"))?;
            }
            "--max-spp" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--max-spp needs a number".to_string())?;
                let n: u32 = v.parse().map_err(|e| format!("--max-spp: {e}"))?;
                r.max_spp = Some(n);
            }
            "--bloom" => {
                r.bloom = true;
            }
            "--bloom-intensity" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--bloom-intensity needs a number".to_string())?;
                r.bloom_intensity = v.parse().map_err(|e| format!("--bloom-intensity: {e}"))?;
                if r.bloom_intensity < 0.0 {
                    return Err(format!(
                        "--bloom-intensity must be ≥ 0; got {}",
                        r.bloom_intensity,
                    ));
                }
            }
            "--bloom-threshold" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--bloom-threshold needs a number".to_string())?;
                r.bloom_threshold = v.parse().map_err(|e| format!("--bloom-threshold: {e}"))?;
            }
            "--bloom-knee" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--bloom-knee needs a number".to_string())?;
                r.bloom_knee = v.parse().map_err(|e| format!("--bloom-knee: {e}"))?;
                if r.bloom_knee <= 0.0 {
                    return Err(format!("--bloom-knee must be > 0; got {}", r.bloom_knee));
                }
            }
            "--sky" => {
                r.sky = true;
            }
            "--sky-elevation" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--sky-elevation needs a number".to_string())?;
                r.sky_elevation = v.parse().map_err(|e| format!("--sky-elevation: {e}"))?;
            }
            "--sky-azimuth" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--sky-azimuth needs a number".to_string())?;
                r.sky_azimuth = v.parse().map_err(|e| format!("--sky-azimuth: {e}"))?;
            }
            "--sky-turbidity" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--sky-turbidity needs a number".to_string())?;
                r.sky_turbidity = v.parse().map_err(|e| format!("--sky-turbidity: {e}"))?;
            }
            "--sky-ground-albedo" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--sky-ground-albedo needs r,g,b".to_string())?;
                r.sky_ground_albedo =
                    parse_vec3(v).map_err(|e| format!("--sky-ground-albedo: {e}"))?;
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
                     \t--scene PATH        load a custom glTF scene (default: embedded Cornell)\n\
                     \t--cloud-grid PATH   load a runtime .qvg cloud density grid\n\
                     \t                    (default: embedded procedural cumulus)\n\
                     \t--env-map PATH      attach a Radiance .hdr environment map\n\
                     \t--denoise           run the PT-denoise à-trous post-process; writes\n\
                     \t                    <out>_denoised.png alongside the raw PNG\n\
                     \t--brute-force       skip the BVH and linear-scan triangles (verification)\n\
                     \t--camera-pos x,y,z  override camera origin\n\
                     \t--look-at x,y,z     override camera target (combine with --camera-pos)\n\
                     \t--fov degrees       override vertical FOV\n\
                     \t--sun-dir x,y,z     enable delta-distribution sun (vector toward sun)\n\
                     \t--sun-color r,g,b   sun radiance per steradian (default 1,1,1)\n\
                     \t--sun-intensity I   scalar multiplier on --sun-color (default 1.0)\n\
                     \t--adaptive          PT-adaptive: per-pixel adaptive sampling (default off)\n\
                     \t--noise-threshold T relative-error stop criterion (default 0.01)\n\
                     \t--min-spp N         per-pixel sample floor (default 64)\n\
                     \t--max-spp M         per-pixel sample ceiling (default = --spp)\n\
                     \t--sky               PT-sky: bake analytic Hosek-Wilkie sky as env map\n\
                     \t                    (mutually exclusive with --env-map and --sun-dir)\n\
                     \t--sky-elevation DEG sun elevation above horizon, [0, 90] (default 45)\n\
                     \t--sky-azimuth DEG   sun azimuth +X→+Z (default 180)\n\
                     \t--sky-turbidity T   atmospheric turbidity (default 2.5)\n\
                     \t--sky-ground-albedo r,g,b  ground albedo for horizon tint (default 0.3,0.3,0.3)"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown render option: {other}")),
        }
    }
    // PT-adaptive: --min-spp must not exceed --max-spp (or --spp if
    // --max-spp is not set). Catches user-intent confusion at parse
    // rather than at render time.
    let effective_max = r.max_spp.unwrap_or(r.samples);
    if r.adaptive && r.min_spp > effective_max {
        return Err(format!(
            "--min-spp {} exceeds --max-spp / --spp ceiling {}; the floor cannot \
             be above the ceiling",
            r.min_spp, effective_max,
        ));
    }
    // PT-sky/wire (plan 0030): --sky is the only env-light source
    // when set, and derives sun_dir from elevation+azimuth. Combining
    // it with --env-map or --sun-dir is a user-intent ambiguity —
    // force the user to choose at parse time rather than picking one
    // silently.
    if r.sky && r.env_map.is_some() {
        return Err(
            "--sky and --env-map are mutually exclusive; --sky bakes its own env".to_string(),
        );
    }
    if r.sky && r.sun_dir.is_some() {
        return Err(
            "--sky and --sun-dir are mutually exclusive; --sky derives sun_dir from \
             --sky-elevation + --sky-azimuth"
                .to_string(),
        );
    }
    if r.sky && !(0.0..=90.0).contains(&r.sky_elevation) {
        return Err(format!(
            "--sky-elevation must be in [0, 90]; got {}",
            r.sky_elevation,
        ));
    }
    Ok(r)
}

#[cfg(not(target_arch = "wasm32"))]
fn run_render(args: &[String]) {
    let cli = parse_render_args(args).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let mut cfg = RenderConfig {
        width: cli.width,
        height: cli.height,
        samples: cli.samples,
        sampler: cli.sampler,
        integrator: cli.integrator,
        use_bvh: !cli.brute_force,
        ..RenderConfig::default()
    };
    if let Some(p) = cli.camera_pos {
        cfg.camera_pos = p;
    }
    if let Some(target) = cli.look_at {
        let pos = cfg.camera_pos;
        let dx = target[0] - pos[0];
        let dy = target[1] - pos[1];
        let dz = target[2] - pos[2];
        let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);
        cfg.camera_dir = [dx / len, dy / len, dz / len];
    }
    if let Some(f) = cli.fov {
        cfg.fov = f;
    }
    // PT-sky/wire (plan 0030): when --sky is set, the sun direction
    // is derived from --sky-elevation + --sky-azimuth so the baked
    // sky and the delta sun light agree on solar position. Parse
    // guarantees --sky-dir and --env-map are not also set, so a
    // single Option<[f32;3]> resolves the "where is the sun" answer
    // ahead of the sun-light wire-up below.
    let derived_sun_dir = if cli.sky {
        Some(sky_dir_from_elev_azimuth(
            cli.sky_elevation,
            cli.sky_azimuth,
        ))
    } else {
        cli.sun_dir
    };
    if let Some(dir) = derived_sun_dir {
        cfg.sun_dir = Some(dir);
        cfg.sun_color = [
            cli.sun_color[0] * cli.sun_intensity,
            cli.sun_color[1] * cli.sun_intensity,
            cli.sun_color[2] * cli.sun_intensity,
        ];
    }
    // PT-adaptive (plan 0028): wire CLI flags into the optional
    // AdaptiveConfig. With `--adaptive` unset, `cfg.adaptive` stays
    // None and the offscreen pipeline takes the pre-plan
    // bit-identical path.
    if cli.adaptive {
        cfg.adaptive = Some(quasi::pathtrace::offscreen::AdaptiveConfig {
            noise_threshold: cli.noise_threshold,
            min_spp: cli.min_spp,
            max_spp: cli.max_spp.unwrap_or(cli.samples),
        });
    }
    // PT-bloom (plan 0029): wire CLI flags into the optional
    // BloomConfig. With `--bloom` unset, `cfg.bloom` stays None
    // and the offscreen pipeline skips the entire bloom pass.
    if cli.bloom {
        cfg.bloom = Some(quasi::pathtrace::offscreen::BloomConfig {
            intensity: cli.bloom_intensity,
            threshold: cli.bloom_threshold,
            knee: cli.bloom_knee,
        });
    }
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
        scene.emissive_lights.len(),
    );
    // Plan 0022: bounds logging so framing flags can be derived from
    // data, not guessed.
    {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in scene.vertices.iter() {
            for c in 0..3 {
                if v.position[c] < min[c] {
                    min[c] = v.position[c];
                }
                if v.position[c] > max[c] {
                    max[c] = v.position[c];
                }
            }
        }
        log::info!(
            "scene bounds: min=({:.2}, {:.2}, {:.2}) max=({:.2}, {:.2}, {:.2}) extents=({:.2}, {:.2}, {:.2})",
            min[0], min[1], min[2],
            max[0], max[1], max[2],
            max[0] - min[0], max[1] - min[1], max[2] - min[2],
        );
    }
    let cloud_grid = cli.cloud_grid.as_deref().map(|p| {
        quasi::pathtrace::grid::Grid3D::load_from_path(p).unwrap_or_else(|e| {
            eprintln!("failed to load --cloud-grid {}: {e}", p.display());
            std::process::exit(1);
        })
    });
    // PT-sky/wire (plan 0030): when --sky is set, bake the analytic
    // sky into an equirect pixel buffer and wrap it in the existing
    // EnvironmentMap so the downstream miss-shader + CDF path is
    // bit-identical to the --env-map flow. Bake resolution is 1024×512
    // per plan default — PT-sky/perf-measure may dial this down for
    // the live widget bake but the CLI render isn't latency-sensitive.
    // Parse guarantees --sky and --env-map are not both set.
    let env_map = if cli.sky {
        // SAFETY: derived_sun_dir is Some when cli.sky is set.
        let sun_dir = derived_sun_dir.expect("derived_sun_dir set when cli.sky is on");
        let params = quasi::pathtrace::sky::SkyParams {
            sun_dir,
            turbidity: cli.sky_turbidity,
            ground_albedo: cli.sky_ground_albedo,
        };
        let (sky_w, sky_h) = (1024_u32, 512_u32);
        log::info!(
            "baking PT-sky: elev={:.1}° azi={:.1}° turbidity={:.2} albedo=({:.2}, {:.2}, {:.2}) → {}×{}",
            cli.sky_elevation,
            cli.sky_azimuth,
            cli.sky_turbidity,
            cli.sky_ground_albedo[0],
            cli.sky_ground_albedo[1],
            cli.sky_ground_albedo[2],
            sky_w,
            sky_h,
        );
        let bake_start = std::time::Instant::now();
        let pixels = quasi::pathtrace::sky::bake_equirect(sky_w, sky_h, &params);
        log::info!("sky bake took {:.3}s", bake_start.elapsed().as_secs_f64());
        Some(quasi::pathtrace::env::EnvironmentMap::new(
            sky_w, sky_h, pixels,
        ))
    } else {
        cli.env_map.as_deref().map(|p| {
            quasi::pathtrace::env::EnvironmentMap::from_hdr_file(p).unwrap_or_else(|e| {
                eprintln!("failed to load --env-map {}: {e}", p.display());
                std::process::exit(1);
            })
        })
    };
    if let Some(env) = env_map.as_ref() {
        log::info!("env map: {} × {}", env.width, env.height);
    }
    let start = std::time::Instant::now();
    let mut bar = quasi::util::progress::Bar::new("render", "spp");
    let aovs = quasi::pathtrace::offscreen::render_offscreen_full(
        cfg,
        &scene,
        cloud_grid,
        env_map,
        Some(&mut bar),
    );
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
    println!("wrote {}", paths.variance.display());

    if cli.denoise {
        let denoise_start = std::time::Instant::now();
        let denoised = quasi::pathtrace::denoise::denoise(
            &aovs.radiance,
            &aovs.albedo,
            &aovs.normal,
            &aovs.depth,
            aovs.width,
            aovs.height,
            quasi::pathtrace::denoise::DenoiseParams::default(),
        );
        let denoised_aovs = quasi::pathtrace::offscreen::Aovs {
            width: aovs.width,
            height: aovs.height,
            radiance: denoised,
            albedo: aovs.albedo.clone(),
            normal: aovs.normal.clone(),
            depth: aovs.depth.clone(),
            mean_y2: aovs.mean_y2.clone(),
            sample_counts: aovs.sample_counts.clone(),
        };
        let mut denoise_out = cli.out.clone();
        let basename = denoise_out
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "frame".to_string());
        denoise_out.set_file_name(format!("{basename}_denoised"));
        let denoise_path = denoise_out.with_extension("png");
        quasi::pathtrace::output::write_tonemapped_png(&denoised_aovs, &denoise_path)
            .unwrap_or_else(|e| {
                eprintln!("denoise output error: {e}");
                std::process::exit(1);
            });
        log::info!(
            "denoise + encode in {:.2}s",
            denoise_start.elapsed().as_secs_f64()
        );
        println!("wrote {}", denoise_path.display());
    }
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
            "--scene" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--scene needs a path".to_string())?;
                r.cfg.scene_path = Some(PathBuf::from(v));
            }
            "--help" | "-?" => {
                println!(
                    "converge options:\n\
                     \t--out <csv>            CSV output path (default: convergence.csv)\n\
                     \t--width N              image width  (default: 256)\n\
                     \t--height N             image height (default: 256)\n\
                     \t--max-spp N            largest spp in the sweep (default: 1024)\n\
                     \t--reference-spp N      spp for the ground-truth reference (default: 4096)\n\
                     \t--scene PATH           load a custom glTF scene (default: embedded Cornell)"
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<RenderArgs, String> {
        let owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        parse_render_args(&owned)
    }

    /// PT-adaptive/cli: `--adaptive` flag toggles the adaptive
    /// scheduler on. Default off.
    #[test]
    fn adaptive_flag_defaults_off_and_parses() {
        let r = parse(&[]).expect("empty parse");
        assert!(!r.adaptive, "default should be --adaptive off");

        let r = parse(&["--adaptive"]).expect("--adaptive parse");
        assert!(r.adaptive);
    }

    /// PT-adaptive/cli: `--noise-threshold` accepts a positive
    /// float; rejects zero or negative.
    #[test]
    fn noise_threshold_accepts_positive_floats() {
        let r = parse(&["--noise-threshold", "0.005"]).expect("0.005 parse");
        assert!((r.noise_threshold - 0.005).abs() < 1e-9);

        assert!(parse(&["--noise-threshold", "0"]).is_err());
        assert!(parse(&["--noise-threshold", "-1"]).is_err());
        assert!(parse(&["--noise-threshold"]).is_err()); // missing value
    }

    /// PT-adaptive/cli: `--min-spp` and `--max-spp` parse into
    /// concrete values.
    #[test]
    fn min_max_spp_parse() {
        let r = parse(&["--min-spp", "32", "--max-spp", "4096"]).expect("parse");
        assert_eq!(r.min_spp, 32);
        assert_eq!(r.max_spp, Some(4096));

        // Defaults
        let r = parse(&[]).expect("empty parse");
        assert_eq!(r.min_spp, 64);
        assert!(r.max_spp.is_none()); // unset -> defaults to --spp at apply time
    }

    /// PT-adaptive/cli: `--min-spp` > `--max-spp` errors at parse
    /// when `--adaptive` is set. Without `--adaptive` the validation
    /// is skipped (the flags have no effect on fixed-spp render).
    #[test]
    fn min_spp_above_max_spp_errors_under_adaptive() {
        let err = parse(&["--adaptive", "--min-spp", "1000", "--max-spp", "100"])
            .expect_err("must error");
        assert!(
            err.contains("--min-spp") && err.contains("1000") && err.contains("100"),
            "error message must mention both flags + values; got: {err}",
        );

        // Without --adaptive the same combination is benign (the
        // flags are inert on a fixed-spp render).
        assert!(parse(&["--min-spp", "1000", "--max-spp", "100"]).is_ok());
    }

    /// PT-adaptive/cli: `--min-spp` > `--spp` (no explicit
    /// `--max-spp`) also errors under adaptive — `--spp` is the
    /// effective ceiling when `--max-spp` is unset.
    #[test]
    fn min_spp_above_spp_errors_under_adaptive() {
        let err = parse(&["--adaptive", "--spp", "32", "--min-spp", "64"]).expect_err("must error");
        assert!(err.contains("--min-spp"));
    }

    /// PT-sky/cli (plan 0030): `--sky` flag defaults off, parses
    /// when present, and the sky-* defaults match the plan.
    #[test]
    fn sky_flag_defaults_off_and_parses() {
        let r = parse(&[]).expect("empty parse");
        assert!(!r.sky, "default should be --sky off");
        assert_eq!(r.sky_elevation, 45.0);
        assert_eq!(r.sky_azimuth, 180.0);
        assert_eq!(r.sky_turbidity, 2.5);
        assert_eq!(r.sky_ground_albedo, [0.3, 0.3, 0.3]);

        let r = parse(&["--sky"]).expect("--sky parse");
        assert!(r.sky);
    }

    /// PT-sky/cli (plan 0030): the four sky parameter flags round-
    /// trip to the parsed `RenderArgs`.
    #[test]
    fn sky_params_parse() {
        let r = parse(&[
            "--sky",
            "--sky-elevation",
            "75",
            "--sky-azimuth",
            "270",
            "--sky-turbidity",
            "4.0",
            "--sky-ground-albedo",
            "0.2,0.5,0.7",
        ])
        .expect("parse");
        assert!(r.sky);
        assert_eq!(r.sky_elevation, 75.0);
        assert_eq!(r.sky_azimuth, 270.0);
        assert_eq!(r.sky_turbidity, 4.0);
        assert_eq!(r.sky_ground_albedo, [0.2, 0.5, 0.7]);
    }

    /// PT-sky/cli (plan 0030): `--sky` + `--env-map` must error.
    /// Both inhabit the env-light channel and combining them is
    /// user-intent ambiguity.
    #[test]
    fn sky_plus_env_map_errors() {
        let err = parse(&["--sky", "--env-map", "anywhere.hdr"]).expect_err("must error");
        assert!(
            err.contains("--sky") && err.contains("--env-map"),
            "error must mention both flags; got: {err}",
        );
    }

    /// PT-sky/cli (plan 0030): `--sky` + `--sun-dir` must error.
    /// `--sky` derives sun_dir from elevation+azimuth; an explicit
    /// `--sun-dir` would silently override it.
    #[test]
    fn sky_plus_sun_dir_errors() {
        let err = parse(&["--sky", "--sun-dir", "0,1,0"]).expect_err("must error");
        assert!(
            err.contains("--sky") && err.contains("--sun-dir"),
            "error must mention both flags; got: {err}",
        );
    }

    /// PT-sky/cli (plan 0030): `--sky-elevation` outside [0, 90]
    /// errors at parse time when `--sky` is also set. Without `--sky`,
    /// the value is inert.
    #[test]
    fn sky_elevation_out_of_range_errors() {
        let err = parse(&["--sky", "--sky-elevation", "-5"]).expect_err("negative must error");
        assert!(err.contains("--sky-elevation"), "got: {err}");

        let err = parse(&["--sky", "--sky-elevation", "120"]).expect_err(">90 must error");
        assert!(err.contains("--sky-elevation"), "got: {err}");

        // Without --sky the elevation is inert; out-of-range doesn't error.
        assert!(parse(&["--sky-elevation", "120"]).is_ok());
    }

    /// PT-sky/wire (plan 0030): `sky_dir_from_elev_azimuth` follows
    /// the env.rs coordinate convention pinned in the plan
    /// ("Coordinate convention" section). +Y is up; azimuth 0 sits
    /// on +X; azimuth π/2 sits on +Z.
    #[test]
    fn sky_dir_from_elev_azimuth_matches_convention() {
        // Zenith: elevation 90° → +Y, azimuth doesn't matter.
        let dir = sky_dir_from_elev_azimuth(90.0, 0.0);
        assert!((dir[0]).abs() < 1e-5);
        assert!((dir[1] - 1.0).abs() < 1e-5);
        assert!((dir[2]).abs() < 1e-5);

        // Horizon east (+X): elevation 0°, azimuth 0°.
        let dir = sky_dir_from_elev_azimuth(0.0, 0.0);
        assert!((dir[0] - 1.0).abs() < 1e-5);
        assert!((dir[1]).abs() < 1e-5);
        assert!((dir[2]).abs() < 1e-5);

        // Horizon "north" (+Z per the plan's convention):
        // elevation 0°, azimuth 90°.
        let dir = sky_dir_from_elev_azimuth(0.0, 90.0);
        assert!((dir[0]).abs() < 1e-5);
        assert!((dir[1]).abs() < 1e-5);
        assert!((dir[2] - 1.0).abs() < 1e-5);

        // Output is always unit length.
        for (elev, azi) in [(45.0, 30.0), (10.0, 215.0), (75.0, 350.0)] {
            let d = sky_dir_from_elev_azimuth(elev, azi);
            let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-5,
                "(elev={elev}, azi={azi}) → {d:?}, |d|={mag}",
            );
        }
    }
}
