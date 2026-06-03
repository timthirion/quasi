//! Path-traced renderer.
//!
//! M1 megakernel pipeline: a Cornell Box path tracer (NEE + MIS) over an
//! analytic scene of quads. Three passes per frame — path-trace one sample
//! into an HDR texture, accumulate into a ping-pong running average, then
//! tonemap the average to the surface.
//!
//! This module owns the path-tracer's `State`, scene, and WGSL shaders.
//! The [`gpu`](crate::gpu) module supplies the wgpu instance factory and
//! the `OrbitCamera`. The rasterized pipeline ([`raster`](crate::raster))
//! shares nothing with this code below the `gpu` seam.

pub mod scene;

use bytemuck::{Pod, Zeroable};

use crate::gpu::OrbitCamera;

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Small uniform for the accumulate pass. 16 bytes — must match WGSL `AccumU`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AccumUniform {
    frame_count: u32,
    _pad: [u32; 3],
}

/// Per-resolution render targets and their bind groups.
struct Targets {
    sample_view: wgpu::TextureView,
    accum_views: [wgpu::TextureView; 2],
    accumulate_bg: [wgpu::BindGroup; 2],
    present_bg: [wgpu::BindGroup; 2],
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

fn create_hdr_texture(device: &wgpu::Device, w: u32, h: u32, label: &str) -> wgpu::TextureView {
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

fn build_targets(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    accumulate_bgl: &wgpu::BindGroupLayout,
    present_bgl: &wgpu::BindGroupLayout,
    accum_uniform_buf: &wgpu::Buffer,
) -> Targets {
    let sample_view = create_hdr_texture(device, w, h, "sample");
    let accum_views = [
        create_hdr_texture(device, w, h, "accum0"),
        create_hdr_texture(device, w, h, "accum1"),
    ];

    let make_accumulate_bg = |prev: usize| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accumulate-bg"),
            layout: accumulate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: accum_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&sample_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&accum_views[prev]),
                },
            ],
        })
    };
    let make_present_bg = |idx: usize| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present-bg"),
            layout: present_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&accum_views[idx]),
            }],
        })
    };

    Targets {
        accumulate_bg: [make_accumulate_bg(0), make_accumulate_bg(1)],
        present_bg: [make_present_bg(0), make_present_bg(1)],
        sample_view,
        accum_views,
    }
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

        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let accumulate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("accumulate-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
            ],
        });
        let present_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present-bgl"),
            entries: &[tex_entry(0)],
        });

        // --- Pipelines ---
        let make_pipeline =
            |label: &str, src: &str, bgl: &wgpu::BindGroupLayout, format: wgpu::TextureFormat| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(bgl)],
                    immediate_size: 0,
                });
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
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };

        let pathtrace_pipeline = make_pipeline(
            "pathtrace",
            include_str!("pathtrace/shaders/pathtrace.wgsl"),
            &pathtrace_bgl,
            HDR_FORMAT,
        );
        let accumulate_pipeline = make_pipeline(
            "accumulate",
            include_str!("pathtrace/shaders/accumulate.wgsl"),
            &accumulate_bgl,
            HDR_FORMAT,
        );
        let present_pipeline = make_pipeline(
            "present",
            include_str!("pathtrace/shaders/present.wgsl"),
            &present_bgl,
            config.format,
        );

        // --- Buffers + scene ---
        let cornell = scene::cornell_box();
        let mut uniforms = scene::Uniforms::zeroed();
        let n = cornell.quads.len().min(scene::MAX_QUADS);
        uniforms.quads[..n].copy_from_slice(&cornell.quads[..n]);
        uniforms.materials[..n].copy_from_slice(&cornell.materials[..n]);
        uniforms.quad_count = n as u32;
        uniforms.light_index = cornell.light_index;

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

        fullscreen_pass(&mut encoder, &self.targets.sample_view, |rp| {
            rp.set_pipeline(&self.pathtrace_pipeline);
            rp.set_bind_group(0, &self.pathtrace_bg, &[]);
            rp.draw(0..3, 0..1);
        });
        fullscreen_pass(&mut encoder, &self.targets.accum_views[dst], |rp| {
            rp.set_pipeline(&self.accumulate_pipeline);
            rp.set_bind_group(0, &self.targets.accumulate_bg[src], &[]);
            rp.draw(0..3, 0..1);
        });
        fullscreen_pass(&mut encoder, &surface_view, |rp| {
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

/// Runs a single fullscreen render pass that clears then invokes `draw`.
fn fullscreen_pass<F: FnOnce(&mut wgpu::RenderPass)>(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    draw: F,
) {
    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("pass"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accum_uniform_is_16_bytes() {
        assert_eq!(std::mem::size_of::<AccumUniform>(), 16);
    }
}
