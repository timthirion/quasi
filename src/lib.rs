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
// Web: per-host renderer driven by requestAnimationFrame. Multiple instances
// can coexist on a page (each owns its canvas, rAF loop, observer, and input
// listeners). Currently a path-tracer-only driver; the raster pipeline will
// add a parallel `RasterInstance` in R1 (the rAF + observer logic is small
// enough to grow either by duplication or by generification — decided then).
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
mod web {
    use crate::gpu::make_instance;
    use crate::pathtrace::State;
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
