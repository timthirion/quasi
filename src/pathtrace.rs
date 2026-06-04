//! Path-traced renderer.
//!
//! Cornell Box megakernel path tracer (NEE + MIS). Each frame runs three
//! passes:
//!
//! 1. **path-trace** — fragment shader samples one new path per pixel and
//!    writes four AOV color attachments (radiance / albedo / normal /
//!    depth) into a set of sample textures.
//! 2. **accumulate** — ping-pong: blend the four sample textures into the
//!    previous accumulator with `weight = 1 / (frame + 1)`, writing the
//!    new accumulator into the four target views.
//! 3. **present** — tone-map the radiance AOV to the surface (Reinhard +
//!    gamma).
//!
//! Sampler choice (PCG / Halton / Sobol) is selected at runtime via
//! [`State::set_sampler`] — the WGSL shader dispatches on a single
//! `sampler_kind` uniform field. CPU sampler implementations + tests
//! live in [`sampler`].
//!
//! The [`gpu`](crate::gpu) module supplies the wgpu instance factory and
//! the `OrbitCamera`. The rasterized pipeline ([`raster`](crate::raster))
//! shares nothing with this code below the `gpu` seam.

pub mod sampler;
pub mod scene;

#[cfg(not(target_arch = "wasm32"))]
pub mod offscreen;

#[cfg(not(target_arch = "wasm32"))]
pub mod output;

#[cfg(target_arch = "wasm32")]
pub mod web;

use bytemuck::{Pod, Zeroable};

use crate::gpu::OrbitCamera;
use crate::pathtrace::sampler::SamplerKind;

/// HDR storage for AOV accumulation. Filterable, broadly supported, and
/// enough precision for the Cornell Box convergence story.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// AOV slot indices. Order matches the WGSL `PathTraceOut` /
/// `AccumOut` `@location(...)` attributes and the `Targets` arrays.
pub const AOV_RADIANCE: usize = 0;
pub const AOV_ALBEDO: usize = 1;
pub const AOV_NORMAL: usize = 2;
pub const AOV_DEPTH: usize = 3;
pub const NUM_AOVS: usize = 4;

/// Small uniform for the accumulate pass. 16 bytes — must match WGSL `AccumU`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct AccumUniform {
    pub frame_count: u32,
    pub _pad: [u32; 3],
}

/// Per-resolution render targets and their bind groups.
///
/// `sample_views` are the path-trace outputs; `accum_views[ping]` holds
/// the running average. `accumulate_bg[ping]` reads from
/// `accum_views[ping]` (the previous accumulator) and is bound when
/// writing into `accum_views[1 - ping]`. The optional `present_bg` only
/// exists for the windowed renderer; the offscreen renderer skips it.
pub(crate) struct Targets {
    pub sample_views: [wgpu::TextureView; NUM_AOVS],
    pub accum_views: [[wgpu::TextureView; NUM_AOVS]; 2],
    pub accumulate_bg: [wgpu::BindGroup; 2],
    pub present_bg: [wgpu::BindGroup; 2],
}

/// The path-tracer renderer: owns the surface, pipelines, and scene state.
pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    pathtrace_pipeline: wgpu::RenderPipeline,
    accumulate_pipeline: wgpu::RenderPipeline,
    present_pipeline: wgpu::RenderPipeline,

    accumulate_bgl: wgpu::BindGroupLayout,
    present_bgl: wgpu::BindGroupLayout,

    uniform_buf: wgpu::Buffer,
    accum_uniform_buf: wgpu::Buffer,
    pathtrace_bg: wgpu::BindGroup,

    targets: Targets,

    uniforms: scene::Uniforms,
    pub camera: OrbitCamera,
    pub frame_count: u32,
    read_idx: usize,
}

/// Creates one HDR render-attachment texture and returns its default
/// view. `extra_usage` lets callers add `COPY_SRC` for AOV readback.
pub(crate) fn create_hdr_texture(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    label: &str,
    extra_usage: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | extra_usage,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn make_aov_views(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    prefix: &str,
) -> [wgpu::TextureView; NUM_AOVS] {
    let names = ["radiance", "albedo", "normal", "depth"];
    std::array::from_fn(|i| {
        let (_tex, view) = create_hdr_texture(
            device,
            w,
            h,
            &format!("{prefix}-{}", names[i]),
            wgpu::TextureUsages::empty(),
        );
        view
    })
}

fn build_targets(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    accumulate_bgl: &wgpu::BindGroupLayout,
    present_bgl: &wgpu::BindGroupLayout,
    accum_uniform_buf: &wgpu::Buffer,
) -> Targets {
    let sample_views = make_aov_views(device, w, h, "sample");
    let accum_views = [
        make_aov_views(device, w, h, "accum0"),
        make_aov_views(device, w, h, "accum1"),
    ];

    let make_accumulate_bg = |prev: usize| {
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
            label: Some("accumulate-bg"),
            layout: accumulate_bgl,
            entries: &entries,
        })
    };

    let make_present_bg = |idx: usize| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present-bg"),
            layout: present_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&accum_views[idx][AOV_RADIANCE]),
            }],
        })
    };

    Targets {
        accumulate_bg: [make_accumulate_bg(0), make_accumulate_bg(1)],
        present_bg: [make_present_bg(0), make_present_bg(1)],
        sample_views,
        accum_views,
    }
}

pub(crate) fn build_accumulate_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let tex = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::with_capacity(1 + 2 * NUM_AOVS);
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    for i in 0..(2 * NUM_AOVS) as u32 {
        entries.push(tex(1 + i));
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("accumulate-bgl"),
        entries: &entries,
    })
}

pub(crate) fn make_pipeline(
    device: &wgpu::Device,
    label: &str,
    src: &str,
    bgl: &wgpu::BindGroupLayout,
    formats: &[wgpu::TextureFormat],
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    // `blend: None` (not BlendState::REPLACE) so the same helper works
    // for Rgba32Float, which wgpu rejects as non-blendable even when the
    // semantic is "no blend".
    let targets: Vec<Option<wgpu::ColorTargetState>> = formats
        .iter()
        .map(|f| {
            Some(wgpu::ColorTargetState {
                format: *f,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        })
        .collect();
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

impl State {
    /// Builds the renderer for an already-created surface of the given size.
    pub async fn new(
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> State {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found (does this browser support WebGPU?)");
        log::info!("adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("quasi-device"),
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                ..Default::default()
            })
            .await
            .expect("failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        // A non-sRGB surface: the present shader applies gamma itself.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // --- Bind group layouts ---
        let pathtrace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pathtrace-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let accumulate_bgl = build_accumulate_bgl(&device);
        let present_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        // --- Pipelines ---
        let aov_formats = [HDR_FORMAT; NUM_AOVS];
        let pathtrace_pipeline = make_pipeline(
            &device,
            "pathtrace",
            include_str!("pathtrace/shaders/pathtrace.wgsl"),
            &pathtrace_bgl,
            &aov_formats,
        );
        let accumulate_pipeline = make_pipeline(
            &device,
            "accumulate",
            include_str!("pathtrace/shaders/accumulate.wgsl"),
            &accumulate_bgl,
            &aov_formats,
        );
        let present_pipeline = make_pipeline(
            &device,
            "present",
            include_str!("pathtrace/shaders/present.wgsl"),
            &present_bgl,
            &[config.format],
        );

        // --- Buffers + scene ---
        let cornell = scene::cornell_box();
        let mut uniforms = scene::Uniforms::zeroed();
        let n = cornell.quads.len().min(scene::MAX_QUADS);
        uniforms.quads[..n].copy_from_slice(&cornell.quads[..n]);
        uniforms.materials[..n].copy_from_slice(&cornell.materials[..n]);
        uniforms.quad_count = n as u32;
        uniforms.light_index = cornell.light_index;
        uniforms.sampler_kind = SamplerKind::default().as_u32();

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<scene::Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let accum_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accum-uniform"),
            size: std::mem::size_of::<AccumUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pathtrace_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pathtrace-bg"),
            layout: &pathtrace_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let targets = build_targets(
            &device,
            config.width,
            config.height,
            &accumulate_bgl,
            &present_bgl,
            &accum_uniform_buf,
        );

        State {
            surface,
            device,
            queue,
            config,
            pathtrace_pipeline,
            accumulate_pipeline,
            present_pipeline,
            accumulate_bgl,
            present_bgl,
            uniform_buf,
            accum_uniform_buf,
            pathtrace_bg,
            targets,
            uniforms,
            camera: OrbitCamera::new(),
            frame_count: 0,
            read_idx: 0,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 && (width != self.config.width || height != self.config.height) {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.targets = build_targets(
                &self.device,
                width,
                height,
                &self.accumulate_bgl,
                &self.present_bgl,
                &self.accum_uniform_buf,
            );
            self.frame_count = 0;
            self.read_idx = 0;
        }
    }

    /// Selects the sampler used by the integrator. Restarts accumulation
    /// since the noise statistics change.
    pub fn set_sampler(&mut self, kind: SamplerKind) {
        if self.uniforms.sampler_kind != kind.as_u32() {
            self.uniforms.sampler_kind = kind.as_u32();
            self.frame_count = 0;
            self.read_idx = 0;
        }
    }

    pub fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Validation => {
                log::warn!("surface validation error acquiring frame");
                return;
            }
        };

        // Restart accumulation when the camera moved.
        if self.camera.dirty {
            self.camera.dirty = false;
            self.frame_count = 0;
            self.read_idx = 0;
        }

        let pos = self.camera.position();
        let dir = self.camera.direction();
        self.uniforms.camera.position = pos;
        self.uniforms.camera.direction = dir;
        self.uniforms.camera.up = [0.0, 1.0, 0.0];
        self.uniforms.camera.fov = self.camera.fov;
        self.uniforms.camera.aspect = self.config.width as f32 / self.config.height as f32;
        self.uniforms.frame_count = self.frame_count;
        self.uniforms.viewport_width = self.config.width;
        self.uniforms.viewport_height = self.config.height;
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniforms));
        self.queue.write_buffer(
            &self.accum_uniform_buf,
            0,
            bytemuck::bytes_of(&AccumUniform {
                frame_count: self.frame_count,
                _pad: [0; 3],
            }),
        );

        let src = self.read_idx;
        let dst = 1 - src;
        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });

        // 1. Path-trace pass: 4 MRT outputs into sample_views[0..4].
        mrt_pass(
            &mut encoder,
            "pathtrace",
            &self.targets.sample_views,
            |rp| {
                rp.set_pipeline(&self.pathtrace_pipeline);
                rp.set_bind_group(0, &self.pathtrace_bg, &[]);
                rp.draw(0..3, 0..1);
            },
        );
        // 2. Accumulate pass: 4 MRT outputs into accum_views[dst].
        mrt_pass(
            &mut encoder,
            "accumulate",
            &self.targets.accum_views[dst],
            |rp| {
                rp.set_pipeline(&self.accumulate_pipeline);
                rp.set_bind_group(0, &self.targets.accumulate_bg[src], &[]);
                rp.draw(0..3, 0..1);
            },
        );
        // 3. Present (radiance only) to the surface.
        single_attachment_pass(&mut encoder, "present", &surface_view, |rp| {
            rp.set_pipeline(&self.present_pipeline);
            rp.set_bind_group(0, &self.targets.present_bg[dst], &[]);
            rp.draw(0..3, 0..1);
        });

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        self.read_idx = dst;
        self.frame_count += 1;
    }
}

/// One render pass into a single color attachment.
pub(crate) fn single_attachment_pass<F: FnOnce(&mut wgpu::RenderPass)>(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    draw: F,
) {
    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    draw(&mut rp);
}

/// One render pass into the four AOV color attachments.
pub(crate) fn mrt_pass<F: FnOnce(&mut wgpu::RenderPass)>(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    targets: &[wgpu::TextureView; NUM_AOVS],
    draw: F,
) {
    let atts: [Option<wgpu::RenderPassColorAttachment>; NUM_AOVS] = std::array::from_fn(|i| {
        Some(wgpu::RenderPassColorAttachment {
            view: &targets[i],
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })
    });
    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &atts,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    draw(&mut rp);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accum_uniform_is_16_bytes() {
        assert_eq!(std::mem::size_of::<AccumUniform>(), 16);
    }

    #[test]
    fn aov_indices_are_distinct_and_packed() {
        // The order matters: shader `@location(N)` must match the array
        // index used in mrt_pass and the present bind group.
        assert_eq!(AOV_RADIANCE, 0);
        assert_eq!(AOV_ALBEDO, 1);
        assert_eq!(AOV_NORMAL, 2);
        assert_eq!(AOV_DEPTH, 3);
        assert_eq!(NUM_AOVS, 4);
    }
}
