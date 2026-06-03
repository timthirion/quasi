//! Quasi (Rust) — wgpu renderer core.
//!
//! Quasi is a **dual-pipeline renderer**:
//!
//! - [`pathtrace`] — a Cornell Box path tracer (next-event estimation +
//!   MIS) with progressive accumulation. The track for offline-quality
//!   stills and the "watch convergence" blog story.
//! - [`raster`] — a real-time forward-shaded pipeline. The track for
//!   interactive widgets (e.g. live in-browser planner demos for the
//!   sibling [`motum`](https://github.com/timthirion/motum) project).
//!
//! The two pipelines are kept wholly separate at the renderer layer; they
//! share only the platform plumbing in [`gpu`] (wgpu instance factory,
//! orbit camera, and the per-instance browser driver).
//!
//! Native is driven by a single `winit` event loop ([`run`]); the web is
//! driven per-instance by `requestAnimationFrame` so multiple independent
//! canvases can coexist on one page (see the `web` module).

pub mod gpu;
pub mod pathtrace;
pub mod raster;

// ---------------------------------------------------------------------------
// Native: a single winit window + event loop, driving one of the two
// renderers. `run` (path tracer) is the default. `run_raster` opens a
// window for the rasterized pipeline; main.rs picks based on argv.
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub fn run_raster() {
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
            .with_title("Quasi — raster")
            .build(&event_loop)
            .expect("failed to create window"),
    );
    let size = window.inner_size();

    let instance = gpu::make_instance();
    let surface = instance
        .create_surface(window.clone())
        .expect("failed to create surface");
    let mut state = pollster::block_on(raster::State::new(
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

    let instance = gpu::make_instance();
    let surface = instance
        .create_surface(window.clone())
        .expect("failed to create surface");
    let mut state = pollster::block_on(pathtrace::State::new(
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
// Web: per-pipeline drivers live in `pathtrace::web` and `raster::web`. The
// only crate-level wasm piece is the one-time `start` shim — wasm-bindgen
// requires a single `#[wasm_bindgen(start)]` per cdylib.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    let _ = console_log::init_with_level(log::Level::Info);
    log::info!("quasi: wasm module loaded");
}
