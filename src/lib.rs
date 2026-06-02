//! Quasi (Rust) — wgpu renderer core.
//!
//! M0: prove the dual-target pipeline. A fullscreen-triangle gradient renders in
//! both a native `winit` window and a browser canvas (via `wasm-pack`), driven by
//! the same code. No path tracing yet — this is the scaffold everything builds on.

use std::sync::Arc;

use winit::{
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// All GPU state plus the window it renders to.
struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    pipeline: wgpu::RenderPipeline,
    window: Arc<Window>,
}

impl State {
    async fn new(window: Arc<Window>) -> State {
        // `mut` is only used on the wasm path below.
        #[allow(unused_mut)]
        let mut size = window.inner_size();
        // On the web, inner_size() can still be 0 here (sizing is deferred), which
        // would configure a 1x1 surface that never shows. Fall back to the canvas
        // size we set explicitly below.
        #[cfg(target_arch = "wasm32")]
        if size.width == 0 || size.height == 0 {
            size = winit::dpi::PhysicalSize::new(720, 720);
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        // Arc<Window> yields a 'static surface, sidestepping borrow lifetimes.
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
                // Keep within WebGL2-class limits so the same code is portable;
                // revisit when path tracing needs more (see plan 0001 M1).
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
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gradient-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline-layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gradient-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
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
        });

        surface.configure(&device, &config);

        State {
            surface,
            device,
            queue,
            config,
            size,
            pipeline,
            window,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                // Reconfigure and skip this frame; the next one will be fine.
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Validation => {
                log::warn!("surface validation error acquiring frame");
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gradient-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
            rpass.set_pipeline(&self.pipeline);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }
}

/// Creates the window and runs the event loop. Async because `wgpu` adapter and
/// device requests are futures (blocked on natively, spawned on the web).
pub async fn run() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Quasi")
            .build(&event_loop)
            .expect("failed to create window"),
    );

    // On the web, attach winit's canvas to the page and give it an explicit size.
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(720, 720));
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let host = doc.get_element_by_id("quasi-canvas")?;
                let canvas = window.canvas()?;
                // Set the backing-store and CSS size directly; don't rely on
                // request_inner_size having propagated yet.
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
                WindowEvent::RedrawRequested => {
                    // Keep the loop alive: queue the next frame.
                    state.window.request_redraw();
                    state.render();
                }
                _ => {}
            }
        }
    };

    // On native, run() drives the loop to completion. On the web, run() never
    // returns and unwinds via an exception ("control flow" sentinel in the
    // console); spawn() integrates with the browser's event loop instead.
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
