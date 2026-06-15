//! Path-tracer-specific browser driver.
//!
//! Owns the rAF loop, observers, pointer listeners, and (optionally) the
//! injected DOM controls for a single canvas. Path-tracer-specific:
//! renders only while the camera is dirty or the sample budget hasn't
//! been hit (the image converges over time).
//!
//! Each `QuasiInstance` is independent; multiple instances coexist on a
//! page without sharing a single global event loop.
//!
//! ## Embedding modes
//!
//! - [`create`] — chrome mode (the default for blog posts). The host
//!   element gets a wrapper `<div>` containing the canvas and a small
//!   controls overlay (sampler + integrator selects, sample-count
//!   readout, reset button). Defaults look acceptable on a dark page;
//!   embedder CSS overrides via the `.quasi-*` class names.
//! - [`create_headless`] — bare-canvas mode. Same renderer, no chrome;
//!   the embedding page is free to call [`QuasiInstance::set_sampler`]
//!   et al. from its own UI.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use super::integrator::IntegratorKind;
use super::sampler::SamplerKind;
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

const DEFAULT_STYLES_ID: &str = "quasi-default-styles";
const DEFAULT_CSS: &str = r#"
.quasi-wrapper { position: relative; width: 100%; height: 100%; }
.quasi-wrapper canvas { display: block; width: 100%; height: 100%; }
.quasi-controls {
  position: absolute; top: 8px; right: 8px;
  display: flex; flex-direction: column; gap: 4px;
  padding: 8px 10px;
  font: 12px/1.4 ui-sans-serif, system-ui, sans-serif;
  color: #eee;
  background: rgba(0,0,0,0.55);
  border-radius: 6px;
  user-select: none;
}
.quasi-controls label { display: flex; gap: 6px; align-items: center; justify-content: space-between; }
.quasi-controls select, .quasi-controls button {
  font: inherit; color: inherit;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 4px;
  padding: 2px 6px;
}
.quasi-controls button { cursor: pointer; }
.quasi-controls button:hover { background: rgba(255,255,255,0.15); }
.quasi-spp { font-variant-numeric: tabular-nums; text-align: right; }
"#;

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
    /// Optional sample-count readout — present in chrome mode.
    readout: Option<web_sys::Element>,
    /// The most recent value written to `readout`, to skip identical DOM updates.
    last_readout: u32,
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
        // Update the readout when the displayed spp count has changed.
        if let Some(readout) = &self.readout {
            if self.state.frame_count != self.last_readout {
                self.last_readout = self.state.frame_count;
                readout.set_text_content(Some(&format!("{} spp", self.last_readout)));
            }
        }
    }
}

/// A listener kept alive for the lifetime of the widget. Removed cleanly
/// on drop so a tab survives wasm-module reloads.
struct ListenerHandle {
    target: web_sys::EventTarget,
    event: String,
    closure: EventClosure,
}

/// A live path-tracer renderer bound to a host element. Keep this handle
/// alive for the lifetime of the widget (dropping it detaches all
/// listeners and observers).
#[wasm_bindgen]
pub struct QuasiInstance {
    inner: Rc<RefCell<Inner>>,
    _raf: RenderLoop,
    _resize_observer: web_sys::ResizeObserver,
    _resize_cb: Closure<dyn FnMut()>,
    _intersection_observer: web_sys::IntersectionObserver,
    _intersection_cb: IntersectionClosure,
    _listeners: Vec<ListenerHandle>,
}

#[wasm_bindgen]
impl QuasiInstance {
    /// Selects the sampler. `name` is one of `"pcg" | "halton" | "sobol"`
    /// (case-insensitive). Accumulation resets if the sampler changed.
    #[wasm_bindgen(js_name = setSampler)]
    pub fn set_sampler(&self, name: &str) -> Result<(), JsValue> {
        let kind: SamplerKind = name.parse().map_err(|e: String| JsValue::from_str(&e))?;
        self.inner.borrow_mut().state.set_sampler(kind);
        Ok(())
    }

    /// Selects the integrator. `name` is one of `"misnee" | "bsdf"`
    /// (case-insensitive). Accumulation resets if the integrator changed.
    #[wasm_bindgen(js_name = setIntegrator)]
    pub fn set_integrator(&self, name: &str) -> Result<(), JsValue> {
        let kind: IntegratorKind = name.parse().map_err(|e: String| JsValue::from_str(&e))?;
        self.inner.borrow_mut().state.set_integrator(kind);
        Ok(())
    }

    /// Restarts accumulation from sample 0.
    pub fn reset(&self) {
        self.inner.borrow_mut().state.reset_accumulation();
    }

    /// PT-adaptive (plan 0028): switches the on-screen tonemap
    /// between radiance and the variance display. `name` is one of
    /// `"radiance"` (default) or `"variance"` (per-pixel luminance
    /// standard deviation, log-scaled, viridis colour-mapped).
    /// Case-insensitive. Does NOT reset accumulation.
    #[wasm_bindgen(js_name = setDisplayMode)]
    pub fn set_display_mode(&self, name: &str) -> Result<(), JsValue> {
        let mode = match name.to_ascii_lowercase().as_str() {
            "radiance" => 0u32,
            "variance" => 1u32,
            other => {
                return Err(JsValue::from_str(&format!(
                    "unknown display mode '{other}' — expected 'radiance' or 'variance'",
                )));
            }
        };
        self.inner.borrow_mut().state.set_display_mode(mode);
        Ok(())
    }

    /// Current accumulated sample count, for an embedder running its own UI.
    #[wasm_bindgen(js_name = frameCount)]
    pub fn frame_count(&self) -> u32 {
        self.inner.borrow().state.frame_count
    }
}

/// Creates a path-tracer renderer inside the element with the given id.
/// Injects the default chrome — controls panel + sample-count readout.
#[wasm_bindgen]
pub async fn create(host_id: String) -> Result<QuasiInstance, JsValue> {
    create_inner(host_id, true).await
}

/// Same renderer, no injected chrome. Use this when the embedder is
/// providing their own UI and just wants the canvas.
#[wasm_bindgen(js_name = createHeadless)]
pub async fn create_headless(host_id: String) -> Result<QuasiInstance, JsValue> {
    create_inner(host_id, false).await
}

async fn create_inner(host_id: String, chrome: bool) -> Result<QuasiInstance, JsValue> {
    let document = web_window()
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;
    let host = document
        .get_element_by_id(&host_id)
        .ok_or_else(|| JsValue::from_str(&format!("no element #{host_id}")))?;

    // Wrapper div: gives us a positioning context for the absolute
    // controls overlay without mutating the embedder's host styles.
    let wrapper = document.create_element("div")?;
    wrapper.set_attribute("class", "quasi-wrapper")?;
    host.append_child(&wrapper)?;

    let canvas: web_sys::HtmlCanvasElement = document.create_element("canvas")?.dyn_into()?;
    wrapper.append_child(&canvas)?;

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

    // Optional chrome.
    let chrome_dom = if chrome {
        inject_default_styles_once(&document)?;
        Some(build_controls(&document, &wrapper)?)
    } else {
        None
    };
    let readout = chrome_dom.as_ref().map(|c| c.readout.clone());

    let inner = Rc::new(RefCell::new(Inner {
        state,
        canvas: canvas.clone(),
        readout,
        last_readout: u32::MAX,
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

    // --- Canvas pointer listeners ---
    let mut listeners: Vec<ListenerHandle> = Vec::new();
    let canvas_target: web_sys::EventTarget = canvas.clone().into();
    attach(
        &canvas_target,
        "pointerdown",
        {
            let inner = inner.clone();
            Closure::wrap(Box::new(move |e: web_sys::Event| {
                if let Ok(m) = e.dyn_into::<web_sys::MouseEvent>() {
                    inner
                        .borrow_mut()
                        .state
                        .camera
                        .press(m.client_x() as f64, m.client_y() as f64);
                }
            }) as Box<dyn FnMut(web_sys::Event)>)
        },
        &mut listeners,
    )?;
    attach(
        &canvas_target,
        "pointermove",
        {
            let inner = inner.clone();
            Closure::wrap(Box::new(move |e: web_sys::Event| {
                if let Ok(m) = e.dyn_into::<web_sys::MouseEvent>() {
                    inner
                        .borrow_mut()
                        .state
                        .camera
                        .on_cursor(m.client_x() as f64, m.client_y() as f64);
                }
            }) as Box<dyn FnMut(web_sys::Event)>)
        },
        &mut listeners,
    )?;
    attach(
        &canvas_target,
        "pointerup",
        {
            let inner = inner.clone();
            Closure::wrap(Box::new(move |_e: web_sys::Event| {
                inner.borrow_mut().state.camera.release();
            }) as Box<dyn FnMut(web_sys::Event)>)
        },
        &mut listeners,
    )?;
    attach(
        &canvas_target,
        "wheel",
        {
            let inner = inner.clone();
            Closure::wrap(Box::new(move |e: web_sys::Event| {
                if let Ok(w) = e.dyn_into::<web_sys::WheelEvent>() {
                    w.prevent_default();
                    inner
                        .borrow_mut()
                        .state
                        .camera
                        .zoom(-(w.delta_y() as f32) * 0.01);
                }
            }) as Box<dyn FnMut(web_sys::Event)>)
        },
        &mut listeners,
    )?;

    // --- Chrome control listeners ---
    if let Some(c) = chrome_dom {
        let sampler_target: web_sys::EventTarget = c.sampler.clone().into();
        attach(
            &sampler_target,
            "change",
            {
                let inner = inner.clone();
                Closure::wrap(Box::new(move |e: web_sys::Event| {
                    if let Some(t) = e.target() {
                        if let Ok(s) = t.dyn_into::<web_sys::HtmlSelectElement>() {
                            if let Ok(kind) = s.value().parse::<SamplerKind>() {
                                inner.borrow_mut().state.set_sampler(kind);
                            }
                        }
                    }
                }) as Box<dyn FnMut(web_sys::Event)>)
            },
            &mut listeners,
        )?;

        let integrator_target: web_sys::EventTarget = c.integrator.clone().into();
        attach(
            &integrator_target,
            "change",
            {
                let inner = inner.clone();
                Closure::wrap(Box::new(move |e: web_sys::Event| {
                    if let Some(t) = e.target() {
                        if let Ok(s) = t.dyn_into::<web_sys::HtmlSelectElement>() {
                            if let Ok(kind) = s.value().parse::<IntegratorKind>() {
                                inner.borrow_mut().state.set_integrator(kind);
                            }
                        }
                    }
                }) as Box<dyn FnMut(web_sys::Event)>)
            },
            &mut listeners,
        )?;

        let reset_target: web_sys::EventTarget = c.reset.clone().into();
        attach(
            &reset_target,
            "click",
            {
                let inner = inner.clone();
                Closure::wrap(Box::new(move |_e: web_sys::Event| {
                    inner.borrow_mut().state.reset_accumulation();
                }) as Box<dyn FnMut(web_sys::Event)>)
            },
            &mut listeners,
        )?;
    }

    Ok(QuasiInstance {
        inner,
        _raf: raf,
        _resize_observer: resize_observer,
        _resize_cb: resize_cb,
        _intersection_observer: intersection_observer,
        _intersection_cb: intersection_cb,
        _listeners: listeners,
    })
}

fn attach(
    target: &web_sys::EventTarget,
    event: &str,
    closure: EventClosure,
    listeners: &mut Vec<ListenerHandle>,
) -> Result<(), JsValue> {
    target.add_event_listener_with_callback(event, closure.as_ref().unchecked_ref())?;
    listeners.push(ListenerHandle {
        target: target.clone(),
        event: event.to_string(),
        closure,
    });
    Ok(())
}

/// References to the DOM elements that the rAF tick and event listeners
/// need to talk to.
struct ChromeDom {
    sampler: web_sys::HtmlSelectElement,
    integrator: web_sys::HtmlSelectElement,
    reset: web_sys::Element,
    readout: web_sys::Element,
}

fn build_controls(
    document: &web_sys::Document,
    wrapper: &web_sys::Element,
) -> Result<ChromeDom, JsValue> {
    let panel = document.create_element("div")?;
    panel.set_attribute("class", "quasi-controls")?;

    let sampler = make_select(
        document,
        "quasi-sampler",
        "Sampler",
        &[("pcg", "PCG"), ("halton", "Halton"), ("sobol", "Sobol")],
        "pcg",
    )?;
    let integrator = make_select(
        document,
        "quasi-integrator",
        "Integrator",
        &[("misnee", "MIS+NEE"), ("bsdf", "Pure BSDF")],
        "misnee",
    )?;
    panel.append_child(&sampler.0)?;
    panel.append_child(&integrator.0)?;

    let readout = document.create_element("span")?;
    readout.set_attribute("class", "quasi-spp")?;
    readout.set_text_content(Some("0 spp"));
    panel.append_child(&readout)?;

    let reset = document.create_element("button")?;
    reset.set_attribute("class", "quasi-reset")?;
    reset.set_attribute("type", "button")?;
    reset.set_text_content(Some("Reset"));
    panel.append_child(&reset)?;

    wrapper.append_child(&panel)?;

    Ok(ChromeDom {
        sampler: sampler.1,
        integrator: integrator.1,
        reset,
        readout,
    })
}

/// Creates `<label>Title <select class="...">...</select></label>` and
/// returns the wrapping label plus the inner select.
fn make_select(
    document: &web_sys::Document,
    class: &str,
    title: &str,
    options: &[(&str, &str)],
    default_value: &str,
) -> Result<(web_sys::Element, web_sys::HtmlSelectElement), JsValue> {
    let label = document.create_element("label")?;
    label.set_text_content(Some(title));

    let select: web_sys::HtmlSelectElement = document.create_element("select")?.dyn_into()?;
    select.set_attribute("class", class)?;
    for (value, text) in options {
        let opt = document.create_element("option")?;
        opt.set_attribute("value", value)?;
        opt.set_text_content(Some(text));
        select.append_child(&opt)?;
    }
    select.set_value(default_value);

    label.append_child(&select)?;
    Ok((label, select))
}

/// Appends the default CSS to `<head>` if (and only if) it isn't already
/// there. Idempotent across multiple `QuasiInstance::create` calls on
/// the same page.
fn inject_default_styles_once(document: &web_sys::Document) -> Result<(), JsValue> {
    if document.get_element_by_id(DEFAULT_STYLES_ID).is_some() {
        return Ok(());
    }
    let head = document
        .query_selector("head")?
        .ok_or_else(|| JsValue::from_str("no <head>"))?;
    let style = document.create_element("style")?;
    style.set_attribute("id", DEFAULT_STYLES_ID)?;
    style.set_text_content(Some(DEFAULT_CSS));
    head.append_child(&style)?;
    Ok(())
}

impl Drop for QuasiInstance {
    fn drop(&mut self) {
        self._resize_observer.disconnect();
        self._intersection_observer.disconnect();
        for h in &self._listeners {
            let _ = h
                .target
                .remove_event_listener_with_callback(&h.event, h.closure.as_ref().unchecked_ref());
        }
    }
}
