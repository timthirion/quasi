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
    make_pipeline, mrt_pass, AccumUniform, AOV_ALBEDO, AOV_DEPTH, AOV_NORMAL, AOV_RADIANCE,
    NUM_AOVS,
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
}

impl Aovs {
    /// Number of pixels — matches all four AOV vectors' lengths.
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Slice access by AOV index (`AOV_RADIANCE`, …, `AOV_DEPTH`).
    pub fn aov(&self, idx: usize) -> &[[f32; 4]] {
        match idx {
            AOV_RADIANCE => &self.radiance,
            AOV_ALBEDO => &self.albedo,
            AOV_NORMAL => &self.normal,
            AOV_DEPTH => &self.depth,
            _ => panic!("invalid aov index {idx}"),
        }
    }
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
    pollster::block_on(render_offscreen_async(cfg, scene, None, None))
}

/// Same as [`render_offscreen`], but with a runtime-loaded cloud
/// density grid. Pass `None` to use the embedded procedural cumulus.
pub fn render_offscreen_with_grid(
    cfg: RenderConfig,
    scene: &TriangleScene,
    cloud_grid: Option<crate::pathtrace::grid::Grid3D>,
) -> Aovs {
    pollster::block_on(render_offscreen_async(cfg, scene, cloud_grid, None))
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
    pollster::block_on(render_offscreen_async(cfg, scene, cloud_grid, env_map))
}

async fn render_offscreen_async(
    cfg: RenderConfig,
    scene_data: &TriangleScene,
    cloud_grid: Option<crate::pathtrace::grid::Grid3D>,
    env_map: Option<crate::pathtrace::env::EnvironmentMap>,
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
    uniforms.emissive_count = scene_data.emissive_triangles.len() as u32;
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
    let uniform_buf = scene_buffers.uniform.clone();
    let accum_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen-accum-uniform"),
        size: std::mem::size_of::<AccumUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pathtrace_bg = build_pathtrace_bg(&device, &pathtrace_bgl, &scene_buffers);

    // --- Targets ---
    let sample_textures: [wgpu::Texture; NUM_AOVS] = std::array::from_fn(|i| {
        let name = ["sample-rad", "sample-alb", "sample-nor", "sample-dep"][i];
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
                ["accum0-rad", "accum0-alb", "accum0-nor", "accum0-dep"],
                ["accum1-rad", "accum1-alb", "accum1-nor", "accum1-dep"],
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
        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(1 + NUM_AOVS * 2);
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
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("offscreen-accumulate-bg"),
            layout: &accumulate_bgl,
            entries: &entries,
        })
    };
    let accumulate_bg = [make_accum_bg(0), make_accum_bg(1)];

    // --- Render loop ---
    let mut read_idx = 0usize;
    for frame in 0..cfg.samples {
        uniforms.frame_count = frame;
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        queue.write_buffer(
            &accum_uniform_buf,
            0,
            bytemuck::bytes_of(&AccumUniform {
                frame_count: frame,
                _pad: [0; 3],
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
    }

    // --- Readback ---
    let bytes_per_row = align_up(
        cfg.width * BYTES_PER_PIXEL,
        wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
    );
    let buffer_size = (bytes_per_row as u64) * (cfg.height as u64);

    let staging: [wgpu::Buffer; NUM_AOVS] = std::array::from_fn(|i| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(["staging-rad", "staging-alb", "staging-nor", "staging-dep"][i]),
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

    Aovs {
        width: cfg.width,
        height: cfg.height,
        radiance,
        albedo,
        normal,
        depth,
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
        };
        assert_eq!(a.aov(AOV_RADIANCE)[0][0], 1.0);
        assert_eq!(a.aov(AOV_ALBEDO)[0][0], 2.0);
        assert_eq!(a.aov(AOV_NORMAL)[0][0], 3.0);
        assert_eq!(a.aov(AOV_DEPTH)[0][0], 4.0);
        assert_eq!(a.pixel_count(), 4);
    }
}
