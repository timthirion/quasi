//! Quasi (Rust) — wgpu renderer core.
//!
//! M1: a Cornell Box path tracer (next-event estimation + MIS) that renders the
//! same in a native window and a browser canvas. Three passes per frame:
//!
//! 1. path trace one sample into an HDR texture,
//! 2. accumulate it into a ping-pong running average,
//! 3. tonemap the average to the surface.
//!
//! An orbit camera drives the view; moving it restarts accumulation.

mod scene;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use winit::{
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Small uniform for the accumulate pass.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AccumUniform {
    frame_count: u32,
    _pad: [u32; 3],
}

/// Orbit camera: spherical coordinates around a target.
struct OrbitCamera {
    target: [f32; 3],
    distance: f32,
    azimuth: f32,
    elevation: f32,
    fov: f32,
    dragging: bool,
    last_cursor: (f64, f64),
    dirty: bool,
}

impl OrbitCamera {
    fn new() -> Self {
        Self {
            target: [0.0, 1.0, 0.0],
            distance: 3.5,
            azimuth: 0.0,
            elevation: 0.0,
            fov: 40.0,
            dragging: false,
            last_cursor: (0.0, 0.0),
            dirty: true,
        }
    }

    fn position(&self) -> [f32; 3] {
        let ce = self.elevation.cos();
        [
            self.target[0] + self.distance * self.azimuth.sin() * ce,
            self.target[1] + self.distance * self.elevation.sin(),
            self.target[2] + self.distance * self.azimuth.cos() * ce,
        ]
    }

    fn direction(&self) -> [f32; 3] {
        let p = self.position();
        let d = [
            self.target[0] - p[0],
            self.target[1] - p[1],
            self.target[2] - p[2],
        ];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
        [d[0] / len, d[1] / len, d[2] / len]
    }

    fn on_cursor(&mut self, x: f64, y: f64) {
        if self.dragging {
            let dx = (x - self.last_cursor.0) as f32;
            let dy = (y - self.last_cursor.1) as f32;
            self.azimuth -= dx * 0.005;
            self.elevation = (self.elevation + dy * 0.005).clamp(-1.5, 1.5);
            self.dirty = true;
        }
        self.last_cursor = (x, y);
    }

    fn zoom(&mut self, amount: f32) {
        self.distance = (self.distance - amount * 0.15).clamp(1.0, 10.0);
        self.dirty = true;
    }
}

/// Per-resolution render targets and their bind groups.
struct Targets {
    sample_view: wgpu::TextureView,
    accum_views: [wgpu::TextureView; 2],
    accumulate_bg: [wgpu::BindGroup; 2],
    present_bg: [wgpu::BindGroup; 2],
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Arc<Window>,

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
    camera: OrbitCamera,
    frame_count: u32,
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

#[allow(clippy::too_many_arguments)]
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
    async fn new(window: Arc<Window>) -> State {
        // `mut` is only used on the wasm path below.
        #[allow(unused_mut)]
        let mut size = window.inner_size();
        #[cfg(target_arch = "wasm32")]
        if size.width == 0 || size.height == 0 {
            size = winit::dpi::PhysicalSize::new(720, 720);
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");
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
            width: size.width.max(1),
            height: size.height.max(1),
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
            include_str!("shaders/pathtrace.wgsl"),
            &pathtrace_bgl,
            HDR_FORMAT,
        );
        let accumulate_pipeline = make_pipeline(
            "accumulate",
            include_str!("shaders/accumulate.wgsl"),
            &accumulate_bgl,
            HDR_FORMAT,
        );
        let present_pipeline = make_pipeline(
            "present",
            include_str!("shaders/present.wgsl"),
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
            size,
            window,
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

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.targets = build_targets(
                &self.device,
                self.config.width,
                self.config.height,
                &self.accumulate_bgl,
                &self.present_bgl,
                &self.accum_uniform_buf,
            );
            self.frame_count = 0;
            self.read_idx = 0;
        }
    }

    fn render(&mut self) {
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

        // Update uniforms from the camera.
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

        // Pass 1: path-trace one sample into sample_view.
        pass(&mut encoder, &self.targets.sample_view, |rp| {
            rp.set_pipeline(&self.pathtrace_pipeline);
            rp.set_bind_group(0, &self.pathtrace_bg, &[]);
            rp.draw(0..3, 0..1);
        });

        // Pass 2: accumulate into accum[dst] reading accum[src].
        pass(&mut encoder, &self.targets.accum_views[dst], |rp| {
            rp.set_pipeline(&self.accumulate_pipeline);
            rp.set_bind_group(0, &self.targets.accumulate_bg[src], &[]);
            rp.draw(0..3, 0..1);
        });

        // Pass 3: tonemap accum[dst] to the surface.
        pass(&mut encoder, &surface_view, |rp| {
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
fn pass<F: FnOnce(&mut wgpu::RenderPass)>(
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

/// Creates the window and runs the event loop.
pub async fn run() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Quasi")
            .build(&event_loop)
            .expect("failed to create window"),
    );

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(720, 720));
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let host = doc.get_element_by_id("quasi-canvas")?;
                let canvas = window.canvas()?;
                canvas.set_width(720);
                canvas.set_height(720);
                let style = canvas.style();
                let _ = style.set_property("width", "720px");
                let _ = style.set_property("height", "720px");
                host.append_child(canvas.as_ref()).ok()?;
                Some(())
            })
            .expect("couldn't attach canvas to #quasi-canvas");
        log::info!("canvas attached (720x720)");
    }

    let mut state = State::new(window.clone()).await;

    let handler = move |event: Event<()>, elwt: &winit::event_loop::EventLoopWindowTarget<()>| {
        if let Event::WindowEvent { window_id, event } = event {
            if window_id != state.window.id() {
                return;
            }
            match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            state: ElementState::Pressed,
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            ..
                        },
                    ..
                } => elwt.exit(),
                WindowEvent::Resized(size) => state.resize(size),
                WindowEvent::MouseInput {
                    state: btn_state,
                    button: MouseButton::Left,
                    ..
                } => {
                    state.camera.dragging = btn_state == ElementState::Pressed;
                }
                WindowEvent::CursorMoved { position, .. } => {
                    state.camera.on_cursor(position.x, position.y);
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let dy = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05,
                    };
                    state.camera.zoom(dy);
                }
                WindowEvent::RedrawRequested => {
                    state.window.request_redraw();
                    state.render();
                }
                _ => {}
            }
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    event_loop.run(handler).expect("event loop error");

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn(handler);
    }
}

/// Web entry point: invoked automatically when the wasm module initializes.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).expect("failed to init logger");
    log::info!("quasi: starting web renderer");
    wasm_bindgen_futures::spawn_local(run());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-4, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn default_camera_looks_down_negative_z() {
        let c = OrbitCamera::new();
        close(c.position(), [0.0, 1.0, 3.5]);
        close(c.direction(), [0.0, 0.0, -1.0]);
    }

    #[test]
    fn direction_is_normalized() {
        let mut c = OrbitCamera::new();
        c.azimuth = 0.7;
        c.elevation = 0.4;
        let d = c.direction();
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn zoom_clamps_distance() {
        let mut c = OrbitCamera::new();
        c.zoom(1000.0);
        assert!((c.distance - 1.0).abs() < 1e-6);
        c.zoom(-1000.0);
        assert!((c.distance - 10.0).abs() < 1e-6);
    }

    #[test]
    fn rotation_only_while_dragging() {
        let mut c = OrbitCamera::new();
        c.on_cursor(10.0, 10.0); // not dragging: no rotation, just records cursor
        assert_eq!(c.azimuth, 0.0);
        c.dragging = true;
        c.on_cursor(10.0, 10.0); // baseline (zero delta)
        c.on_cursor(110.0, 10.0); // dx = 100 -> azimuth -= 100 * 0.005
        assert!((c.azimuth + 0.5).abs() < 1e-5);
    }

    #[test]
    fn elevation_is_clamped() {
        let mut c = OrbitCamera::new();
        c.dragging = true;
        c.on_cursor(0.0, 0.0);
        c.on_cursor(0.0, 1.0e6); // huge upward drag
        assert!((c.elevation - 1.5).abs() < 1e-4);
    }

    #[test]
    fn accum_uniform_is_16_bytes() {
        // Must match the 16-byte WGSL AccumU in shaders/accumulate.wgsl.
        assert_eq!(std::mem::size_of::<AccumUniform>(), 16);
    }
}
