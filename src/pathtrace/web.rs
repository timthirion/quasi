//! Path-tracer-specific browser driver.
//!
//! Owns the rAF loop, ResizeObserver, IntersectionObserver, and pointer
//! listeners for a single canvas. Path-tracer-specific: renders only while
//! the camera is dirty or the sample budget hasn't been hit (the image
//! converges over time).
//!
//! Each `QuasiInstance` is independent; multiple instances coexist on a
//! page without sharing a single global event loop.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use super::State;
use crate::gpu::make_instance;

type EventClosure = Closure<dyn FnMut(web_sys::Event)>;
type RenderLoop = Rc<RefCell<Option<Closure<dyn FnMut()>>>>;

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
        if !self.shared.visible.get() {
            return;
        }
        if let Some((w, h)) = self.shared.pending_resize.take() {
            self.canvas.set_width(w);
            self.canvas.set_height(h);
            self.state.resize(w, h);
        }
        if self.state.camera.dirty || self.state.frame_count < SAMPLE_BUDGET {
            self.state.render();
        }
    }
}

/// A live path-tracer renderer bound to a host element. Keep this handle
/// alive for the lifetime of the widget (dropping it detaches the
/// input/resize listeners).
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

/// Creates a path-tracer renderer inside the element with the given id.
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

    let resize_cb = {
        let shared = shared.clone();
        let host = host.clone();
        Closure::wrap(Box::new(move || {
            shared.pending_resize.set(Some(backing_size(&host)));
        }) as Box<dyn FnMut()>)
    };
    let resize_observer = web_sys::ResizeObserver::new(resize_cb.as_ref().unchecked_ref())?;
    resize_observer.observe(&host);

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
