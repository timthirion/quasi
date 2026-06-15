//! Offscreen path-traced render — produces accumulated AOVs as plain CPU
//! arrays for image output and the convergence harness.
//!
//! The windowed [`super::State`] is single-frame, surface-bound, and
//! display-paced. The offscreen renderer is the opposite: it builds its
//! own headless wgpu device, runs `cfg.samples` accumulated frames in a
//! loop, then copies the final accumulator textures back to host memory.
//!
//! Render targets here are `Rgba32Float`, not the windowed renderer's
//! `Rgba16Float`. The native limit allows it, the precision is fine for
//! reference-quality images, and — crucially — readback is just a
//! `bytemuck::cast_slice` instead of an `f16 -> f32` decode loop.

use bytemuck::Zeroable;

use crate::gpu;
use crate::pathtrace::integrator::IntegratorKind;
use crate::pathtrace::mesh::TriangleScene;
use crate::pathtrace::sampler::SamplerKind;
use crate::pathtrace::scene;
use crate::pathtrace::{
    build_accumulate_bgl, build_pathtrace_bg, build_pathtrace_bgl, build_scene_buffers,
    make_pipeline, mrt_pass, AccumUniform, MaskUniform, AOV_ALBEDO, AOV_DEPTH, AOV_MEAN_Y2,
    AOV_NORMAL, AOV_RADIANCE, NUM_AOVS,
};

/// `Rgba32Float`: 16 bytes per pixel; trivially round-trips through host
/// memory as `[f32; 4]`.
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const BYTES_PER_PIXEL: u32 = 16;

fn align_up(v: u32, align: u32) -> u32 {
    v.div_ceil(align) * align
}

/// Accumulated AOV channels read back from the GPU.
#[derive(Clone, Debug)]
pub struct Aovs {
    pub width: u32,
    pub height: u32,
    /// HDR radiance (no tonemap applied).
    pub radiance: Vec<[f32; 4]>,
    /// First-hit albedo per pixel (emission's chromaticity for emissives).
    pub albedo: Vec<[f32; 4]>,
    /// First-hit world-space normal; alpha is unused.
    pub normal: Vec<[f32; 4]>,
    /// First-hit `t` (camera-ray distance); alpha is the hit mask.
    pub depth: Vec<[f32; 4]>,
    /// PT-adaptive (plan 0028): per-pixel running mean of luminance²,
    /// i.e. `E[Y²]` over the accumulated samples. Channel R carries
    /// the value; G/B/A are unused. Used to derive per-pixel variance
    /// via [`Aovs::luminance_variance`].
    pub mean_y2: Vec<[f32; 4]>,
}

impl Aovs {
    /// Number of pixels — matches all five AOV vectors' lengths.
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Slice access by AOV index
    /// (`AOV_RADIANCE`, …, `AOV_DEPTH`, `AOV_MEAN_Y2`).
    pub fn aov(&self, idx: usize) -> &[[f32; 4]] {
        match idx {
            AOV_RADIANCE => &self.radiance,
            AOV_ALBEDO => &self.albedo,
            AOV_NORMAL => &self.normal,
            AOV_DEPTH => &self.depth,
            AOV_MEAN_Y2 => &self.mean_y2,
            _ => panic!("invalid aov index {idx}"),
        }
    }

    /// PT-adaptive: per-pixel luminance variance derived from the
    /// running mean of `Y` (via the radiance accumulator) and the
    /// running mean of `Y²` (via the `mean_y2` AOV):
    ///
    /// ```text
    /// var(Y) = E[Y²] - (E[Y])²
    /// ```
    ///
    /// where `Y = 0.2126·R + 0.7152·G + 0.0722·B` is the Rec. 709
    /// scalar luminance. Returns one f32 per pixel.
    ///
    /// Note this is the **population variance** at the accumulated
    /// sample count, not the **sample variance** with Bessel's
    /// correction. The plan-0028 termination criterion uses
    /// `var/n` (the variance of the sample mean), so a factor of
    /// `n/(n-1)` is irrelevant for thresholding decisions at the
    /// sample counts we care about (≥ 64). The unit test compares
    /// against the population-variance closed form.
    pub fn luminance_variance(&self) -> Vec<f32> {
        let pixels = self.pixel_count();
        let mut out = Vec::with_capacity(pixels);
        for i in 0..pixels {
            let r = self.radiance[i];
            let y = 0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2];
            let e_y2 = self.mean_y2[i][0];
            // Clamp to zero — finite-precision FMA can drive
            // E[Y²] - (E[Y])² slightly negative on near-constant
            // pixels even though the true variance is non-negative.
            out.push((e_y2 - y * y).max(0.0));
        }
        out
    }
}

/// PT-adaptive (plan 0028): per-pixel adaptive sampling parameters.
/// `None` on [`RenderConfig::adaptive`] disables adaptive scheduling
/// — pre-plan bit-identical behaviour. `Some(_)` enables the
/// scheduler: per-pixel relative standard error is computed at
/// checkpoint boundaries (every 64 samples after `min_spp`); pixels
/// whose error falls below `noise_threshold` are masked out and
/// skipped by the path-trace + accumulate passes.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveConfig {
    /// Per-pixel relative standard error below which the pixel is
    /// marked converged. Typical default 0.01 (1% relative).
    pub noise_threshold: f32,
    /// Per-pixel sample floor before the convergence test is
    /// trusted. Heavy-tailed integrands need ~64 samples before
    /// the sample variance is a reliable estimator.
    pub min_spp: u32,
    /// Per-pixel sample ceiling. Pixels that hit this without
    /// converging are flagged as "clamped" (state 2) in the
    /// active mask.
    pub max_spp: u32,
}

impl AdaptiveConfig {
    /// Plan-0028 documented defaults.
    pub const DEFAULT: Self = Self {
        noise_threshold: 0.01,
        min_spp: 64,
        max_spp: u32::MAX,
    };
}

/// Camera + sampling configuration for an offscreen render.
#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub sampler: SamplerKind,
    pub integrator: IntegratorKind,
    /// True (default) → walk the BVH. False → linear scan over all
    /// triangles. Both produce the same image; the linear scan is
    /// retained as a verification fallback (see `--brute-force` on the
    /// `render` CLI).
    pub use_bvh: bool,
    pub camera_pos: [f32; 3],
    pub camera_dir: [f32; 3],
    pub camera_up: [f32; 3],
    pub fov: f32,
    /// PT-sun-light (plan 0023): optional delta-distribution
    /// directional sun. Unit vector points TOWARD the sun (so
    /// surface normals with positive dot product see the sun).
    /// `None` keeps the sun disabled — bit-identical with pre-plan.
    pub sun_dir: Option<[f32; 3]>,
    /// PT-sun-light: emitted radiance per steradian, linear units.
    /// Defaults to `[1, 1, 1]` when `sun_dir` is set; override for
    /// warmer / cooler suns or stronger intensity.
    pub sun_color: [f32; 3],
    /// PT-adaptive (plan 0028): adaptive sampling parameters.
    /// `None` keeps fixed-spp behaviour (pre-plan bit-identical).
    pub adaptive: Option<AdaptiveConfig>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        // Looks straight at the Cornell Box, matching the windowed
        // renderer's default OrbitCamera (target [0,1,0], distance 3.5).
        Self {
            width: 512,
            height: 512,
            samples: 256,
            sampler: SamplerKind::default(),
            integrator: IntegratorKind::default(),
            use_bvh: true,
            camera_pos: [0.0, 1.0, 3.5],
            camera_dir: [0.0, 0.0, -1.0],
            camera_up: [0.0, 1.0, 0.0],
            fov: 40.0,
            sun_dir: None,
            sun_color: [1.0, 1.0, 1.0],
            adaptive: None,
        }
    }
}

fn create_aov_texture(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    label: &str,
    extra_usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OFFSCREEN_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | extra_usage,
        view_formats: &[],
    })
}

/// Renders `scene` at `cfg.samples` samples per pixel and reads back
/// the accumulated AOVs. Blocks the calling thread (intended for use
/// from a CLI render command).
pub fn render_offscreen(cfg: RenderConfig, scene: &TriangleScene) -> Aovs {
    pollster::block_on(render_offscreen_async(cfg, scene, None, None, None))
}

/// Same as [`render_offscreen`], but with a runtime-loaded cloud
/// density grid. Pass `None` to use the embedded procedural cumulus.
pub fn render_offscreen_with_grid(
    cfg: RenderConfig,
    scene: &TriangleScene,
    cloud_grid: Option<crate::pathtrace::grid::Grid3D>,
) -> Aovs {
    pollster::block_on(render_offscreen_async(cfg, scene, cloud_grid, None, None))
}

/// Full offscreen entry: optional cloud grid + optional environment
/// map. `render --env-map PATH` routes here so the HDR env contributes
/// to both miss-shader emission and NEE.
pub fn render_offscreen_with_grid_and_env(
    cfg: RenderConfig,
    scene: &TriangleScene,
    cloud_grid: Option<crate::pathtrace::grid::Grid3D>,
    env_map: Option<crate::pathtrace::env::EnvironmentMap>,
) -> Aovs {
    pollster::block_on(render_offscreen_async(
        cfg, scene, cloud_grid, env_map, None,
    ))
}

/// Like [`render_offscreen_with_grid_and_env`] but with an optional
/// generic progress sink (see [`crate::util::progress`]). Pass a
/// concrete sink for live CLI progress, or `None` for silent
/// rendering. Used by the `render` CLI subcommand to draw the
/// stderr progress bar; tests and examples use the bare entries
/// above.
pub fn render_offscreen_full(
    cfg: RenderConfig,
    scene: &TriangleScene,
    cloud_grid: Option<crate::pathtrace::grid::Grid3D>,
    env_map: Option<crate::pathtrace::env::EnvironmentMap>,
    progress: Option<&mut dyn crate::util::progress::ProgressSink>,
) -> Aovs {
    pollster::block_on(render_offscreen_async(
        cfg, scene, cloud_grid, env_map, progress,
    ))
}

async fn render_offscreen_async(
    cfg: RenderConfig,
    scene_data: &TriangleScene,
    cloud_grid: Option<crate::pathtrace::grid::Grid3D>,
    env_map: Option<crate::pathtrace::env::EnvironmentMap>,
    mut progress: Option<&mut dyn crate::util::progress::ProgressSink>,
) -> Aovs {
    assert!(
        cfg.width > 0 && cfg.height > 0,
        "render size must be non-zero"
    );
    assert!(cfg.samples > 0, "samples per pixel must be > 0");

    let instance = gpu::make_instance();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("no GPU adapter found for offscreen render");
    log::info!("offscreen adapter: {:?}", adapter.get_info());

    // Rgba32Float × 4 AOVs = 64 bytes per sample, which exceeds the
    // default `max_color_attachment_bytes_per_sample = 32`. Ask for the
    // adapter's actual limits — offscreen is native-only, where the real
    // ceiling is 64+ on every backend we target.
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("quasi-offscreen-device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("failed to create offscreen device");

    let pathtrace_bgl = build_pathtrace_bgl(&device);
    let accumulate_bgl = build_accumulate_bgl(&device);

    let aov_formats = [OFFSCREEN_FORMAT; NUM_AOVS];
    let pathtrace_pipeline = make_pipeline(
        &device,
        "offscreen-pathtrace",
        include_str!("shaders/pathtrace.wgsl"),
        &pathtrace_bgl,
        &aov_formats,
    );
    let accumulate_pipeline = make_pipeline(
        &device,
        "offscreen-accumulate",
        include_str!("shaders/accumulate.wgsl"),
        &accumulate_bgl,
        &aov_formats,
    );

    // --- Scene + uniforms ---
    let mut uniforms = scene::Uniforms::zeroed();
    uniforms.triangle_count = scene_data.triangle_count() as u32;
    uniforms.emissive_count = scene_data.emissive_lights.len() as u32;
    uniforms.camera.position = cfg.camera_pos;
    uniforms.camera.direction = cfg.camera_dir;
    uniforms.camera.up = cfg.camera_up;
    uniforms.camera.fov = cfg.fov;
    uniforms.camera.aspect = cfg.width as f32 / cfg.height as f32;
    uniforms.viewport_width = cfg.width;
    uniforms.viewport_height = cfg.height;
    uniforms.sampler_kind = cfg.sampler.as_u32();
    uniforms.integrator_kind = cfg.integrator.as_u32();
    uniforms.use_bvh = u32::from(cfg.use_bvh);

    // PT-env: flip the has_environment flag + pass width/height into
    // the inverse-CDF helpers. Stays at zero when no env map.
    if let Some(env) = env_map.as_ref() {
        uniforms.has_environment = 1;
        uniforms.env_width = env.width;
        uniforms.env_height = env.height;
    }
    // PT-light-vs-env: surface the triangle total power up front
    // (env total power lands after `build_scene_buffers_*` below).
    uniforms.triangle_total_power = scene_data.triangle_total_power;

    // PT-sun-light (plan 0023): pack the delta sun. `sun_dir.w` is
    // the on/off flag (1.0 to include in NEE). `None` → all zeros,
    // bit-identical with pre-plan.
    if let Some(dir) = cfg.sun_dir {
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
            .sqrt()
            .max(1e-6);
        uniforms.sun_dir = [dir[0] / len, dir[1] / len, dir[2] / len, 1.0];
        uniforms.sun_color = [cfg.sun_color[0], cfg.sun_color[1], cfg.sun_color[2], 0.0];
    }
    // PT-adaptive (plan 0028): flag the shader to read the active
    // mask. When `cfg.adaptive.is_none()` this stays 0 and the
    // shader skips the mask read entirely — pre-plan bit-identical.
    uniforms.adaptive_enabled = u32::from(cfg.adaptive.is_some());
    uniforms._pad_adaptive = [0; 3];

    let scene_buffers = match (cloud_grid, env_map) {
        (Some(g), None) => {
            crate::pathtrace::build_scene_buffers_with_grid(&device, &queue, scene_data, g)
        }
        (cg, env @ Some(_)) => {
            let grid = cg.unwrap_or_else(|| {
                crate::pathtrace::grid::from_bytes_or_empty(crate::pathtrace::CUMULUS_QVG)
            });
            crate::pathtrace::build_scene_buffers_with_grid_and_env(
                &device, &queue, scene_data, grid, env,
            )
        }
        (None, None) => build_scene_buffers(&device, &queue, scene_data),
    };
    uniforms.env_total_power = scene_buffers.env_total_power;
    let uniform_buf = scene_buffers.uniform.clone();
    let accum_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen-accum-uniform"),
        size: std::mem::size_of::<AccumUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // PT-adaptive (plan 0028): per-pixel active mask. R32Uint
    // texture sized to the framebuffer; 1 = active, 0 = converged,
    // 2 = clamped-at-max-spp. Initialized to all 1u. Updated only
    // when `cfg.adaptive.is_some()` via CPU-side variance computation
    // at checkpoint boundaries.
    let active_mask_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen-active-mask"),
        size: wgpu::Extent3d {
            width: cfg.width,
            height: cfg.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let active_mask_view = active_mask_texture.create_view(&Default::default());
    {
        let pixels = (cfg.width as usize) * (cfg.height as usize);
        let init: Vec<u32> = vec![1u32; pixels];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &active_mask_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&init),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(cfg.width * 4),
                rows_per_image: Some(cfg.height),
            },
            wgpu::Extent3d {
                width: cfg.width,
                height: cfg.height,
                depth_or_array_layers: 1,
            },
        );
    }

    let pathtrace_bg =
        build_pathtrace_bg(&device, &pathtrace_bgl, &scene_buffers, &active_mask_view);

    // --- Targets ---
    let sample_textures: [wgpu::Texture; NUM_AOVS] = std::array::from_fn(|i| {
        let name = [
            "sample-rad",
            "sample-alb",
            "sample-nor",
            "sample-dep",
            "sample-my2",
        ][i];
        create_aov_texture(
            &device,
            cfg.width,
            cfg.height,
            name,
            wgpu::TextureUsages::empty(),
        )
    });
    let sample_views: [wgpu::TextureView; NUM_AOVS] =
        std::array::from_fn(|i| sample_textures[i].create_view(&Default::default()));

    let accum_textures: [[wgpu::Texture; NUM_AOVS]; 2] = std::array::from_fn(|p| {
        std::array::from_fn(|i| {
            let name = [
                [
                    "accum0-rad",
                    "accum0-alb",
                    "accum0-nor",
                    "accum0-dep",
                    "accum0-my2",
                ],
                [
                    "accum1-rad",
                    "accum1-alb",
                    "accum1-nor",
                    "accum1-dep",
                    "accum1-my2",
                ],
            ][p][i];
            create_aov_texture(
                &device,
                cfg.width,
                cfg.height,
                name,
                wgpu::TextureUsages::COPY_SRC,
            )
        })
    });
    let accum_views: [[wgpu::TextureView; NUM_AOVS]; 2] = std::array::from_fn(|p| {
        std::array::from_fn(|i| accum_textures[p][i].create_view(&Default::default()))
    });

    let make_accum_bg = |prev: usize| {
        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(2 + NUM_AOVS * 2);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: accum_uniform_buf.as_entire_binding(),
        });
        for (i, view) in sample_views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 1 + i as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        for (i, view) in accum_views[prev].iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 1 + (NUM_AOVS + i) as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        // PT-adaptive: active_mask at binding 11.
        entries.push(wgpu::BindGroupEntry {
            binding: 1 + (2 * NUM_AOVS) as u32,
            resource: wgpu::BindingResource::TextureView(&active_mask_view),
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("offscreen-accumulate-bg"),
            layout: &accumulate_bgl,
            entries: &entries,
        })
    };
    let accumulate_bg = [make_accum_bg(0), make_accum_bg(1)];

    // PT-adaptive (plan 0028): compute pipeline + bind groups for
    // the active-mask update. Built unconditionally — when
    // `cfg.adaptive.is_none()` the pipeline is never dispatched.
    // Two bind groups (one per ping-pong accumulator) so we can
    // bind whichever holds the current accumulated state at the
    // checkpoint.
    let mask_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("adaptive-mask-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::R32Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let mask_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("adaptive-mask-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/adaptive_mask.wgsl").into()),
    });
    let mask_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("adaptive-mask-pipeline-layout"),
        bind_group_layouts: &[Some(&mask_bgl)],
        immediate_size: 0,
    });
    let mask_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("adaptive-mask-pipeline"),
        layout: Some(&mask_pipeline_layout),
        module: &mask_shader,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mask_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adaptive-mask-uniform"),
        size: std::mem::size_of::<MaskUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let make_mask_bg = |accum_idx: usize| -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adaptive-mask-bg"),
            layout: &mask_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mask_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &accum_views[accum_idx][AOV_RADIANCE],
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &accum_views[accum_idx][AOV_MEAN_Y2],
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&active_mask_view),
                },
            ],
        })
    };
    let mask_bg = [make_mask_bg(0), make_mask_bg(1)];

    // --- Render loop ---
    //
    // Progress ticks live here. The submit calls below queue GPU
    // work without blocking, so without a sync the entire loop body
    // returns in microseconds and the progress bar would race to
    // 100% before the GPU has done meaningful work. We block the
    // CPU on a `device.poll(wait)` once every `sync_every` frames so
    // the tick rate reflects actual rendered samples — at most 100
    // sync points per render, which keeps the throughput cost in the
    // noise on long jobs.
    let sync_every = (cfg.samples / 100).max(1);
    // PT-adaptive (plan 0028): checkpoint every 64 samples once
    // `min_spp` is reached. At each checkpoint the CPU reads back the
    // running radiance + mean_y2, derives per-pixel relative standard
    // error, and uploads an updated active-mask texture. With
    // `cfg.adaptive.is_none()` the entire branch is skipped — the
    // shader's `adaptive_enabled` flag also stays 0, so the per-frame
    // cost is exactly pre-plan.
    const CHECKPOINT_INTERVAL: u32 = 64;
    let accum_adaptive_flag: u32 = u32::from(cfg.adaptive.is_some());

    let mut read_idx = 0usize;
    for frame in 0..cfg.samples {
        uniforms.frame_count = frame;
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        queue.write_buffer(
            &accum_uniform_buf,
            0,
            bytemuck::bytes_of(&AccumUniform {
                frame_count: frame,
                adaptive_enabled: accum_adaptive_flag,
                _pad: [0; 2],
            }),
        );

        let src = read_idx;
        let dst = 1 - src;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen-frame-encoder"),
        });
        mrt_pass(&mut encoder, "offscreen-pathtrace", &sample_views, |rp| {
            rp.set_pipeline(&pathtrace_pipeline);
            rp.set_bind_group(0, &pathtrace_bg, &[]);
            rp.draw(0..3, 0..1);
        });
        mrt_pass(
            &mut encoder,
            "offscreen-accumulate",
            &accum_views[dst],
            |rp| {
                rp.set_pipeline(&accumulate_pipeline);
                rp.set_bind_group(0, &accumulate_bg[src], &[]);
                rp.draw(0..3, 0..1);
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        read_idx = dst;

        // PT-adaptive (plan 0028): every CHECKPOINT_INTERVAL samples
        // (after `min_spp` is reached), dispatch the mask-update
        // compute pass to mark converged / clamped pixels. The
        // shader reads the current accumulators (which now hold the
        // post-frame-`frame` running mean) via `accum_views[read_idx]`
        // and updates `active_mask_texture` in place.
        if let Some(adapt) = cfg.adaptive {
            let n = frame + 1;
            let max_spp_eff = adapt.max_spp.min(cfg.samples);
            if n >= adapt.min_spp && n.is_multiple_of(CHECKPOINT_INTERVAL) {
                let mask_u = MaskUniform {
                    sample_count: n,
                    min_spp: adapt.min_spp,
                    max_spp: max_spp_eff,
                    _pad: 0,
                    noise_threshold: adapt.noise_threshold,
                    eps_dark: 0.001,
                    _pad2: 0.0,
                    _pad3: 0.0,
                };
                queue.write_buffer(&mask_uniform_buf, 0, bytemuck::bytes_of(&mask_u));
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("offscreen-mask-update"),
                });
                {
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("offscreen-mask-compute"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(&mask_pipeline);
                    cp.set_bind_group(0, &mask_bg[read_idx], &[]);
                    cp.dispatch_workgroups(cfg.width.div_ceil(8), cfg.height.div_ceil(8), 1);
                }
                queue.submit(std::iter::once(enc.finish()));
            }
        }

        let last = frame + 1 == cfg.samples;
        if let Some(p) = progress.as_mut() {
            if (frame + 1) % sync_every == 0 || last {
                // Block until the GPU has finished everything
                // submitted so far. Without this the tick fires on
                // submission, not completion, and the bar reaches
                // 100% within milliseconds. The cost is ~1 sync per
                // `sync_every` frames; amortised across a long
                // render it's invisible.
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                p.tick((frame + 1) as u64, cfg.samples as u64);
            }
        }
    }
    if let Some(p) = progress.as_mut() {
        p.finish();
    }

    // --- Readback ---
    let bytes_per_row = align_up(
        cfg.width * BYTES_PER_PIXEL,
        wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
    );
    let buffer_size = (bytes_per_row as u64) * (cfg.height as u64);

    let staging: [wgpu::Buffer; NUM_AOVS] = std::array::from_fn(|i| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(
                [
                    "staging-rad",
                    "staging-alb",
                    "staging-nor",
                    "staging-dep",
                    "staging-my2",
                ][i],
            ),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("offscreen-readback"),
    });
    for i in 0..NUM_AOVS {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &accum_textures[read_idx][i],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging[i],
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(cfg.height),
                },
            },
            wgpu::Extent3d {
                width: cfg.width,
                height: cfg.height,
                depth_or_array_layers: 1,
            },
        );
    }
    queue.submit(std::iter::once(encoder.finish()));

    for buf in &staging {
        buf.slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map_async failed"));
    }
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device.poll failed");

    let read_aov = |buf: &wgpu::Buffer| -> Vec<[f32; 4]> {
        let unpadded = (cfg.width * BYTES_PER_PIXEL) as usize;
        let padded = bytes_per_row as usize;
        let raw = buf.slice(..).get_mapped_range();
        let mut out: Vec<[f32; 4]> = Vec::with_capacity(cfg.width as usize * cfg.height as usize);
        for y in 0..cfg.height as usize {
            let start = y * padded;
            let row = &raw[start..start + unpadded];
            let pixels: &[[f32; 4]] = bytemuck::cast_slice(row);
            out.extend_from_slice(pixels);
        }
        drop(raw);
        buf.unmap();
        out
    };

    let radiance = read_aov(&staging[AOV_RADIANCE]);
    let albedo = read_aov(&staging[AOV_ALBEDO]);
    let normal = read_aov(&staging[AOV_NORMAL]);
    let depth = read_aov(&staging[AOV_DEPTH]);
    let mean_y2 = read_aov(&staging[AOV_MEAN_Y2]);

    Aovs {
        width: cfg.width,
        height: cfg.height,
        radiance,
        albedo,
        normal,
        depth,
        mean_y2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(1024, 256), 1024);
    }

    #[test]
    fn aovs_indexed_access() {
        let n = 4;
        let a = Aovs {
            width: 2,
            height: 2,
            radiance: vec![[1.0; 4]; n],
            albedo: vec![[2.0; 4]; n],
            normal: vec![[3.0; 4]; n],
            depth: vec![[4.0; 4]; n],
            mean_y2: vec![[5.0, 0.0, 0.0, 0.0]; n],
        };
        assert_eq!(a.aov(AOV_RADIANCE)[0][0], 1.0);
        assert_eq!(a.aov(AOV_ALBEDO)[0][0], 2.0);
        assert_eq!(a.aov(AOV_NORMAL)[0][0], 3.0);
        assert_eq!(a.aov(AOV_DEPTH)[0][0], 4.0);
        assert_eq!(a.aov(AOV_MEAN_Y2)[0][0], 5.0);
        assert_eq!(a.pixel_count(), 4);
    }

    /// PT-adaptive (plan 0028) buffers milestone:
    /// `Aovs::luminance_variance` derives per-pixel variance from
    /// the running mean E[Y] (via the radiance accumulator) and the
    /// running mean E[Y²] (via the mean_y2 AOV). This test exercises
    /// the load-bearing arithmetic with a synthetic Monte Carlo
    /// sequence that has a known closed-form variance.
    #[test]
    fn luminance_variance_matches_closed_form_on_synthetic_sequence() {
        // Three pixels with three different ground-truth variance
        // structures. For each, we directly construct what the GPU
        // accumulator should hold after N samples and check the
        // derived variance.
        //
        // Pixel 0: constant pure-grey, variance = 0.
        // Pixel 1: alternates between (1, 0, 0) and (0, 1, 0) over
        //          large N — luminance alternates 0.2126 / 0.7152.
        // Pixel 2: uniform-random luminance in [0, 1] — variance
        //          asymptotes to 1/12 ≈ 0.0833.
        let rec709 = |rgb: [f32; 3]| 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];

        // Pixel 0: 64 samples of (0.5, 0.5, 0.5) → Y = 0.5,
        // E[Y] = 0.5, E[Y²] = 0.25, variance = 0.
        let p0_y = 0.5_f32;
        let p0_e_y = p0_y;
        let p0_e_y2 = p0_y * p0_y;

        // Pixel 1: alternates with equal probability between
        // Y_a = rec709((1,0,0)) and Y_b = rec709((0,1,0)).
        // E[Y] = (Y_a + Y_b) / 2; E[Y²] = (Y_a² + Y_b²) / 2.
        let ya = rec709([1.0, 0.0, 0.0]);
        let yb = rec709([0.0, 1.0, 0.0]);
        let p1_e_y = (ya + yb) / 2.0;
        let p1_e_y2 = (ya * ya + yb * yb) / 2.0;
        let p1_closed_form_var = p1_e_y2 - p1_e_y * p1_e_y;
        // Sanity-check the closed form against ((Y_a - Y_b) / 2)².
        // The two derivations are algebraically identical but
        // numerically diverge by ~1e-7 due to f32 precision in the
        // E[Y²] - (E[Y])² subtraction of similar-magnitude values.
        let expected_var = ((ya - yb) / 2.0).powi(2);
        let sanity_rel = (p1_closed_form_var - expected_var).abs() / expected_var;
        assert!(
            sanity_rel < 1e-5,
            "closed-form variance {} vs ((Y_a - Y_b) / 2)² = {} — \
             relative error {} exceeds 1e-5",
            p1_closed_form_var,
            expected_var,
            sanity_rel,
        );

        // Pixel 2: uniform distribution over [0, 1] luminance.
        // E[Y] = 0.5, E[Y²] = 1/3, variance = 1/12.
        let p2_e_y = 0.5_f32;
        let p2_e_y2 = 1.0_f32 / 3.0;

        // To represent these on the Aovs side, radiance carries E[Y]
        // in its luminance (any spectrum with rec709-luminance =
        // E[Y] works; we use a grey channel for simplicity), and
        // mean_y2 carries E[Y²] in channel R.
        //
        // For pixel 0 and 2 we pick grey radiance so rec709 = the
        // value directly. For pixel 1, we pick (E[Y], 0, 0) / 0.2126
        // so luminance ≈ E[Y].
        let aovs = Aovs {
            width: 3,
            height: 1,
            radiance: vec![
                [p0_e_y, p0_e_y, p0_e_y, 1.0],
                [p1_e_y / 0.2126, 0.0, 0.0, 1.0],
                [p2_e_y, p2_e_y, p2_e_y, 1.0],
            ],
            albedo: vec![[0.0; 4]; 3],
            normal: vec![[0.0; 4]; 3],
            depth: vec![[0.0; 4]; 3],
            mean_y2: vec![
                [p0_e_y2, 0.0, 0.0, 0.0],
                [p1_e_y2, 0.0, 0.0, 0.0],
                [p2_e_y2, 0.0, 0.0, 0.0],
            ],
        };

        let var = aovs.luminance_variance();
        assert_eq!(var.len(), 3);

        // Pixel 0: variance ≡ 0 (constant input).
        assert!(var[0] < 1e-7, "pixel 0 variance must be ~0, got {}", var[0]);

        // Pixel 1: matches closed form to within 1e-6 relative.
        let p1_rel = (var[1] - p1_closed_form_var).abs() / p1_closed_form_var;
        assert!(
            p1_rel < 1e-6,
            "pixel 1 variance {} vs closed form {} — relative error \
             {} exceeds 1e-6",
            var[1],
            p1_closed_form_var,
            p1_rel,
        );

        // Pixel 2: matches uniform-distribution variance 1/12.
        let p2_expected = 1.0_f32 / 12.0;
        let p2_rel = (var[2] - p2_expected).abs() / p2_expected;
        assert!(
            p2_rel < 1e-6,
            "pixel 2 variance {} vs closed form {} (1/12) — relative \
             error {} exceeds 1e-6",
            var[2],
            p2_expected,
            p2_rel,
        );
    }

    /// PT-adaptive (plan 0028): the variance estimator must be
    /// computed on **scalar luminance**, not on per-channel
    /// recombination. This test pins the difference by
    /// constructing a synthetic scene where per-channel
    /// recombination would over-estimate variance by a factor of
    /// (sum w_c²) / (sum w_c)² ≈ 0.6 vs the true scalar-luminance
    /// variance.
    #[test]
    fn variance_uses_scalar_luminance_not_per_channel_recombination() {
        // A pixel that alternates samples between (0, 0, 0) and
        // (1, 1, 1). Channels are PERFECTLY CORRELATED — exactly
        // the path-tracer integrand case.
        //
        // Scalar luminance: Y oscillates between 0 and 1 with
        // E[Y] = 0.5, E[Y²] = 0.5, Var(Y) = 0.25.
        //
        // Per-channel "variance" (each channel treated independently):
        // Var(R) = Var(G) = Var(B) = 0.25.
        // Per-channel recombined via 0.2126² + 0.7152² + 0.0722² ≈
        // 0.5621 ≠ 0.25.
        //
        // The right answer is 0.25 — the scalar-luminance
        // variance. This test asserts the implementation uses the
        // scalar form.
        let e_y = 0.5_f32;
        let e_y2 = 0.5_f32;
        let aovs = Aovs {
            width: 1,
            height: 1,
            radiance: vec![[e_y, e_y, e_y, 1.0]],
            albedo: vec![[0.0; 4]],
            normal: vec![[0.0; 4]],
            depth: vec![[0.0; 4]],
            mean_y2: vec![[e_y2, 0.0, 0.0, 0.0]],
        };
        let var = aovs.luminance_variance();
        let expected = 0.25_f32;
        let rel = (var[0] - expected).abs() / expected;
        assert!(
            rel < 1e-6,
            "scalar-luminance variance must be 0.25 (got {}); a per-\
             channel recombined estimator would give ~0.56 here, \
             which would indicate the implementation is wrong",
            var[0],
        );
    }
}
