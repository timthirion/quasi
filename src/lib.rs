//! Quasi (Rust) — wgpu renderer core.
//!
//! M1: a Cornell Box path tracer (next-event estimation + MIS) that renders the
//! same in a native window and a browser canvas. Three passes per frame:
//!
//! 1. path trace one sample into an HDR texture,
//! 2. accumulate it into a ping-pong running average,
//! 3. tonemap the average to the surface.
//!
//! `State` is platform-agnostic (no windowing). Native is driven by a single
//! winit event loop; the web is driven per-instance by `requestAnimationFrame`
//! (see the `web` module), which allows multiple independent canvases on one page
//! — something a single winit event loop cannot do.

mod scene;

use bytemuck::{Pod, Zeroable};

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Small uniform for the accumulate pass. 16 bytes — must match WGSL `AccumU`.
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

    /// Records the cursor without rotating (use on press to set the drag origin).
    fn press(&mut self, x: f64, y: f64) {
        self.dragging = true;
        self.last_cursor = (x, y);
    }

    fn release(&mut self) {
        self.dragging = false;
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

/// The platform-agnostic renderer: owns the surface, pipelines, and scene state.
struct State {
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
    camera: OrbitCamera,
    frame_count: u32,
    read_idx: usize,
}

fn make_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
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
    async fn new(
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

    fn resize(&mut self, width: u32, height: u32) {
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

        pass(&mut encoder, &self.targets.sample_view, |rp| {
            rp.set_pipeline(&self.pathtrace_pipeline);
            rp.set_bind_group(0, &self.pathtrace_bg, &[]);
            rp.draw(0..3, 0..1);
        });
        pass(&mut encoder, &self.targets.accum_views[dst], |rp| {
            rp.set_pipeline(&self.accumulate_pipeline);
            rp.set_bind_group(0, &self.targets.accumulate_bg[src], &[]);
            rp.draw(0..3, 0..1);
        });
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

// ---------------------------------------------------------------------------
// Native: a single winit window + event loop.
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub fn run() {
    use std::sync::Arc;
    use winit::{
        event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
        event_loop::EventLoop,
        keyboard::{KeyCode, PhysicalKey},
        window::WindowBuilder,
    };

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Quasi")
            .build(&event_loop)
            .expect("failed to create window"),
    );
    let size = window.inner_size();

    let instance = make_instance();
    let surface = instance
        .create_surface(window.clone())
        .expect("failed to create surface");
    let mut state = pollster::block_on(State::new(
        instance,
        surface,
        size.width.max(1),
        size.height.max(1),
    ));

    event_loop
        .run(move |event, elwt| {
            if let Event::WindowEvent { window_id, event } = event {
                if window_id != window.id() {
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
                    WindowEvent::Resized(s) => state.resize(s.width, s.height),
                    WindowEvent::MouseInput {
                        state: btn_state,
                        button: MouseButton::Left,
                        ..
                    } => {
                        if btn_state == ElementState::Pressed {
                            let (x, y) = state.camera.last_cursor;
                            state.camera.press(x, y);
                        } else {
                            state.camera.release();
                        }
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
                        window.request_redraw();
                        state.render();
                    }
                    _ => {}
                }
            }
        })
        .expect("event loop error");
}

// ---------------------------------------------------------------------------
// Web: one renderer per host element, driven by requestAnimationFrame. Multiple
// instances can coexist on a page (each owns its canvas, rAF loop, observer, and
// input listeners). Canvas size follows the host element's clientWidth/Height
// (× devicePixelRatio), updated via a ResizeObserver.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
mod web {
    use super::{make_instance, State};
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    type EventClosure = Closure<dyn FnMut(web_sys::Event)>;
    type RenderLoop = Rc<RefCell<Option<Closure<dyn FnMut()>>>>;

    /// One-time setup: panic hook + console logging.
    #[wasm_bindgen(start)]
    pub fn start() {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        let _ = console_log::init_with_level(log::Level::Info);
        log::info!("quasi: wasm module loaded");
    }

    fn web_window() -> web_sys::Window {
        web_sys::window().expect("no global window")
    }

    fn request_animation_frame(cb: &Closure<dyn FnMut()>) {
        web_window()
            .request_animation_frame(cb.as_ref().unchecked_ref())
            .expect("requestAnimationFrame failed");
    }

    /// Backing-store size for the canvas: the host's CSS size × device pixel ratio.
    fn backing_size(host: &web_sys::Element) -> (u32, u32) {
        let dpr = web_window().device_pixel_ratio().max(1.0);
        let w = (host.client_width().max(1) as f64 * dpr).round() as u32;
        let h = (host.client_height().max(1) as f64 * dpr).round() as u32;
        (w.max(1), h.max(1))
    }

    type IntersectionClosure = Closure<dyn FnMut(js_sys::Array)>;

    /// Stop accumulating after this many samples — the image has converged and
    /// further frames would just burn GPU. Camera interaction resets the count.
    const SAMPLE_BUDGET: u32 = 1024;

    /// State touched by observer callbacks. Kept separate from `RefCell<Inner>`
    /// (these are `Cell`s) so an observer firing can never collide with the
    /// mutable borrow held during a render.
    struct Shared {
        pending_resize: Cell<Option<(u32, u32)>>,
        visible: Cell<bool>,
    }

    struct Inner {
        state: State,
        canvas: web_sys::HtmlCanvasElement,
        shared: Rc<Shared>,
    }

    impl Inner {
        fn tick(&mut self) {
            // Off-screen widgets do no work.
            if !self.shared.visible.get() {
                return;
            }
            if let Some((w, h)) = self.shared.pending_resize.take() {
                self.canvas.set_width(w);
                self.canvas.set_height(h);
                self.state.resize(w, h);
            }
            // Render only while converging or right after a camera change;
            // once the image is stable, leave the GPU idle.
            if self.state.camera.dirty || self.state.frame_count < SAMPLE_BUDGET {
                self.state.render();
            }
        }
    }

    /// A live renderer bound to a host element. Keep this handle alive for the
    /// lifetime of the widget (dropping it detaches the input/resize listeners).
    #[wasm_bindgen]
    pub struct QuasiInstance {
        _inner: Rc<RefCell<Inner>>,
        _raf: RenderLoop,
        _resize_observer: web_sys::ResizeObserver,
        _resize_cb: Closure<dyn FnMut()>,
        _intersection_observer: web_sys::IntersectionObserver,
        _intersection_cb: IntersectionClosure,
        _listeners: Vec<(String, EventClosure)>,
    }

    /// Creates a renderer inside the element with the given id, sized to it.
    #[wasm_bindgen]
    pub async fn create(host_id: String) -> Result<QuasiInstance, JsValue> {
        let document = web_window()
            .document()
            .ok_or_else(|| JsValue::from_str("no document"))?;
        let host = document
            .get_element_by_id(&host_id)
            .ok_or_else(|| JsValue::from_str(&format!("no element #{host_id}")))?;

        let canvas: web_sys::HtmlCanvasElement = document.create_element("canvas")?.dyn_into()?;
        let style = canvas.style();
        style.set_property("width", "100%")?;
        style.set_property("height", "100%")?;
        style.set_property("display", "block")?;
        host.append_child(&canvas)?;

        let (w, h) = backing_size(&host);
        canvas.set_width(w);
        canvas.set_height(h);

        let instance = make_instance();
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| JsValue::from_str(&format!("create_surface: {e:?}")))?;
        let state = State::new(instance, surface, w, h).await;

        let shared = Rc::new(Shared {
            pending_resize: Cell::new(None),
            visible: Cell::new(true),
        });

        let inner = Rc::new(RefCell::new(Inner {
            state,
            canvas: canvas.clone(),
            shared: shared.clone(),
        }));

        // requestAnimationFrame loop (self-rescheduling). The tick is cheap when
        // the widget is idle or off-screen: it does no GPU work.
        let raf: RenderLoop = Rc::new(RefCell::new(None));
        {
            let raf2 = raf.clone();
            let inner2 = inner.clone();
            *raf.borrow_mut() = Some(Closure::wrap(Box::new(move || {
                inner2.borrow_mut().tick();
                if let Some(cb) = raf2.borrow().as_ref() {
                    request_animation_frame(cb);
                }
            }) as Box<dyn FnMut()>));
        }
        request_animation_frame(raf.borrow().as_ref().unwrap());

        // ResizeObserver: re-read the host size on layout changes.
        let resize_cb = {
            let shared = shared.clone();
            let host = host.clone();
            Closure::wrap(Box::new(move || {
                shared.pending_resize.set(Some(backing_size(&host)));
            }) as Box<dyn FnMut()>)
        };
        let resize_observer = web_sys::ResizeObserver::new(resize_cb.as_ref().unchecked_ref())?;
        resize_observer.observe(&host);

        // IntersectionObserver: pause rendering while the host is off-screen.
        let intersection_cb = {
            let shared = shared.clone();
            Closure::wrap(Box::new(move |entries: js_sys::Array| {
                if let Ok(entry) = entries
                    .get(0)
                    .dyn_into::<web_sys::IntersectionObserverEntry>()
                {
                    shared.visible.set(entry.is_intersecting());
                }
            }) as Box<dyn FnMut(js_sys::Array)>)
        };
        let intersection_observer =
            web_sys::IntersectionObserver::new(intersection_cb.as_ref().unchecked_ref())?;
        intersection_observer.observe(&host);

        // Pointer + wheel input on the canvas.
        let mut listeners: Vec<(String, EventClosure)> = Vec::new();
        let mut add = |event: &str, cb: EventClosure| -> Result<(), JsValue> {
            canvas.add_event_listener_with_callback(event, cb.as_ref().unchecked_ref())?;
            listeners.push((event.to_string(), cb));
            Ok(())
        };

        {
            let inner = inner.clone();
            add(
                "pointerdown",
                Closure::wrap(Box::new(move |e: web_sys::Event| {
                    if let Ok(m) = e.dyn_into::<web_sys::MouseEvent>() {
                        inner
                            .borrow_mut()
                            .state
                            .camera
                            .press(m.client_x() as f64, m.client_y() as f64);
                    }
                }) as Box<dyn FnMut(web_sys::Event)>),
            )?;
        }
        {
            let inner = inner.clone();
            add(
                "pointermove",
                Closure::wrap(Box::new(move |e: web_sys::Event| {
                    if let Ok(m) = e.dyn_into::<web_sys::MouseEvent>() {
                        inner
                            .borrow_mut()
                            .state
                            .camera
                            .on_cursor(m.client_x() as f64, m.client_y() as f64);
                    }
                }) as Box<dyn FnMut(web_sys::Event)>),
            )?;
        }
        {
            let inner = inner.clone();
            add(
                "pointerup",
                Closure::wrap(Box::new(move |_e: web_sys::Event| {
                    inner.borrow_mut().state.camera.release();
                }) as Box<dyn FnMut(web_sys::Event)>),
            )?;
        }
        {
            let inner = inner.clone();
            add(
                "wheel",
                Closure::wrap(Box::new(move |e: web_sys::Event| {
                    if let Ok(w) = e.dyn_into::<web_sys::WheelEvent>() {
                        w.prevent_default();
                        inner
                            .borrow_mut()
                            .state
                            .camera
                            .zoom(-(w.delta_y() as f32) * 0.01);
                    }
                }) as Box<dyn FnMut(web_sys::Event)>),
            )?;
        }

        Ok(QuasiInstance {
            _inner: inner,
            _raf: raf,
            _resize_observer: resize_observer,
            _resize_cb: resize_cb,
            _intersection_observer: intersection_observer,
            _intersection_cb: intersection_cb,
            _listeners: listeners,
        })
    }

    impl Drop for QuasiInstance {
        fn drop(&mut self) {
            self._resize_observer.disconnect();
            self._intersection_observer.disconnect();
            let inner = self._inner.borrow();
            for (event, cb) in &self._listeners {
                let _ = inner
                    .canvas
                    .remove_event_listener_with_callback(event, cb.as_ref().unchecked_ref());
            }
        }
    }
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
        c.press(10.0, 10.0); // begin drag at the current cursor
        c.on_cursor(110.0, 10.0); // dx = 100 -> azimuth -= 100 * 0.005
        assert!((c.azimuth + 0.5).abs() < 1e-5);
    }

    #[test]
    fn elevation_is_clamped() {
        let mut c = OrbitCamera::new();
        c.press(0.0, 0.0);
        c.on_cursor(0.0, 1.0e6); // huge upward drag
        assert!((c.elevation - 1.5).abs() < 1e-4);
    }

    #[test]
    fn accum_uniform_is_16_bytes() {
        assert_eq!(std::mem::size_of::<AccumUniform>(), 16);
    }
}
