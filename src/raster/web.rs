//! Raster-pipeline browser driver.
//!
//! Mirrors [`pathtrace::web`](crate::pathtrace::web) in structure (per-host
//! canvas + rAF + ResizeObserver + IntersectionObserver + pointer
//! listeners) but renders **every frame** while visible — there's no
//! convergence story to pause for. Multiple `RasterInstance`s coexist on
//! a page; each owns its own canvas, wgpu surface, and event listeners.
//!
//! ## R4 — motum-shaped scene API
//!
//! [`RasterInstance`] exposes four JSON setters from JS:
//!
//! - `setWorldState(json)` — instances at per-link poses
//!   ([`crate::raster::wire::WireWorldState`])
//! - `setTrajectory(json)` + `setTrajectoryTime(t)` — world-space
//!   waypoints; the renderer snaps to the closest waypoint at time `t`
//! - `setTreeOverlay(json)` — depth-tested overlay edges + points
//! - `setGoal(json)` — pickable, draggable goal handle
//!
//! Plus `onGoalChanged(callback)` — JS receives the new pose JSON when
//! the user finishes dragging the goal.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use super::picking::{camera_ray, ray_floor_hit, ray_hits_sphere, serialize_pose};
use super::scene::Instance;
use super::wire::{
    parse_goal, parse_trajectory, parse_tree_overlay, parse_world_state, WireGoal, WireTrajectory,
    WireTreeOverlay, WireWorldState,
};
use super::{DefaultMeshes, State};
use crate::gpu::make_instance;

type EventClosure = Closure<dyn FnMut(web_sys::Event)>;
type RenderLoop = Rc<RefCell<Option<Closure<dyn FnMut()>>>>;
type IntersectionClosure = Closure<dyn FnMut(js_sys::Array)>;

fn web_window() -> web_sys::Window {
    web_sys::window().expect("no global window")
}

fn request_animation_frame(cb: &Closure<dyn FnMut()>) {
    web_window()
        .request_animation_frame(cb.as_ref().unchecked_ref())
        .expect("requestAnimationFrame failed");
}

fn backing_size(host: &web_sys::Element) -> (u32, u32) {
    let dpr = web_window().device_pixel_ratio().max(1.0);
    let w = (host.client_width().max(1) as f64 * dpr).round() as u32;
    let h = (host.client_height().max(1) as f64 * dpr).round() as u32;
    (w.max(1), h.max(1))
}

struct Shared {
    pending_resize: Cell<Option<(u32, u32)>>,
    visible: Cell<bool>,
}

/// Live motum-shaped scene mirroring the JSON the JS host pushed in.
#[derive(Default)]
struct WireScene {
    world_state: Option<WireWorldState>,
    trajectory: Option<WireTrajectory>,
    trajectory_time: f64,
    tree_overlay: Option<WireTreeOverlay>,
    goal: Option<WireGoal>,
    goal_callback: Option<js_sys::Function>,
}

const GOAL_HANDLE_RADIUS: f32 = 0.18;

impl WireScene {
    /// Picks the world state to render at the current playhead. If a
    /// trajectory is set, it overrides any standalone `world_state`.
    /// Snaps to the most recent waypoint at or before `trajectory_time`.
    fn active_world_state(&self) -> Option<&WireWorldState> {
        if let Some(traj) = &self.trajectory {
            if !traj.waypoints.is_empty() {
                let mut chosen = &traj.waypoints[0];
                for wp in &traj.waypoints {
                    if wp.time <= self.trajectory_time {
                        chosen = wp;
                    } else {
                        break;
                    }
                }
                return Some(&chosen.world_state);
            }
        }
        self.world_state.as_ref()
    }

    fn goal_world_position(&self) -> Option<[f32; 3]> {
        self.goal.map(|g| {
            let v = g.pose.translation.vector;
            [v[0] as f32, v[1] as f32, v[2] as f32]
        })
    }
}

struct Inner {
    state: State,
    canvas: web_sys::HtmlCanvasElement,
    shared: Rc<Shared>,
    wire: WireScene,
    dragging_goal: bool,
}

impl Inner {
    /// Rebuilds the renderer's scene from the wire data once per frame.
    /// O(world_state.links + tree_overlay.edges/nodes + 2), which is
    /// trivial for motum-sized scenes.
    fn populate_scene(&mut self) {
        self.state.scene.clear();
        let defaults = self.state.default_meshes();

        if let Some(ws) = self.wire.active_world_state() {
            populate_world_state(&mut self.state, ws, defaults);
        }

        if let Some(tree) = self.wire.tree_overlay.clone() {
            populate_tree_overlay(&mut self.state, &tree);
        }

        if let Some(goal) = self.wire.goal {
            populate_goal_handle(&mut self.state, &goal, defaults, self.dragging_goal);
        }
    }

    fn tick(&mut self) {
        if !self.shared.visible.get() {
            return;
        }
        if let Some((w, h)) = self.shared.pending_resize.take() {
            self.canvas.set_width(w);
            self.canvas.set_height(h);
            self.state.resize(w, h);
        }
        self.populate_scene();
        self.state.render();
    }
}

fn populate_world_state(state: &mut State, ws: &WireWorldState, defaults: DefaultMeshes) {
    for (pose, geom) in ws.link_poses.iter().zip(ws.link_geometry.iter()) {
        let Some(handle) = *geom else { continue };
        // Wire `GeometryHandle` indexes the renderer's default-mesh
        // library: 0 = cube, 1 = sphere, 2 = cylinder. Unknown handles
        // are silently ignored — same forward-compatibility behaviour
        // a real plugin model would want.
        let mesh = match handle {
            0 => defaults.cube,
            1 => defaults.sphere,
            2 => defaults.cylinder,
            _ => continue,
        };
        state.scene.push(Instance {
            mesh,
            model: pose.to_model_matrix(),
            tint: [0.78, 0.80, 0.86, 1.0],
        });
    }
}

fn populate_tree_overlay(state: &mut State, tree: &WireTreeOverlay) {
    for edge in &tree.edges {
        let a = [
            edge.from[0] as f32,
            edge.from[1] as f32,
            edge.from[2] as f32,
        ];
        let b = [edge.to[0] as f32, edge.to[1] as f32, edge.to[2] as f32];
        state
            .scene
            .depth_tested_overlay
            .line(a, b, [0.45, 0.7, 1.0, 0.75]);
    }
    for n in &tree.nodes {
        let p = [n[0] as f32, n[1] as f32, n[2] as f32];
        state
            .scene
            .depth_tested_overlay
            .point(p, [0.95, 0.85, 0.35, 0.9]);
    }
}

fn populate_goal_handle(
    state: &mut State,
    goal: &WireGoal,
    defaults: DefaultMeshes,
    is_dragging: bool,
) {
    let mut model = goal.pose.to_model_matrix();
    // Scale the sphere instance to the handle radius. The default sphere
    // has radius 0.5, so multiply each rotated basis column by
    // 2 × radius.
    let s = GOAL_HANDLE_RADIUS * 2.0;
    for col in model.iter_mut().take(3) {
        for v in col.iter_mut().take(3) {
            *v *= s;
        }
    }
    let tint = if is_dragging {
        [1.0, 0.95, 0.4, 1.0]
    } else {
        [1.0, 0.8, 0.2, 1.0]
    };
    state.scene.push(Instance {
        mesh: defaults.sphere,
        model,
        tint,
    });
    // An on-top point marker over the goal so it stays visible even
    // when the camera is below it.
    let pos = [
        goal.pose.translation.vector[0] as f32,
        goal.pose.translation.vector[1] as f32,
        goal.pose.translation.vector[2] as f32,
    ];
    state.scene.on_top_overlay.point(pos, [1.0, 1.0, 0.3, 1.0]);
}

/// A live raster renderer bound to a host element. Keep this handle alive
/// for the lifetime of the widget (dropping it detaches the listeners).
#[wasm_bindgen]
pub struct RasterInstance {
    inner: Rc<RefCell<Inner>>,
    _raf: RenderLoop,
    _resize_observer: web_sys::ResizeObserver,
    _resize_cb: Closure<dyn FnMut()>,
    _intersection_observer: web_sys::IntersectionObserver,
    _intersection_cb: IntersectionClosure,
    _listeners: Vec<(String, EventClosure)>,
}

#[wasm_bindgen]
impl RasterInstance {
    /// Replaces the current world state. Pose JSON matches motum's
    /// `WorldState` (nalgebra Isometry3 serialisation).
    #[wasm_bindgen(js_name = setWorldState)]
    pub fn set_world_state(&self, json: &str) -> Result<(), JsValue> {
        let parsed = parse_world_state(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.inner.borrow_mut().wire.world_state = Some(parsed);
        Ok(())
    }

    /// Replaces the trajectory. The renderer snaps to the closest
    /// waypoint at the current `setTrajectoryTime`.
    #[wasm_bindgen(js_name = setTrajectory)]
    pub fn set_trajectory(&self, json: &str) -> Result<(), JsValue> {
        let parsed = parse_trajectory(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.inner.borrow_mut().wire.trajectory = Some(parsed);
        Ok(())
    }

    /// Sets the trajectory playhead. Re-rendering happens automatically
    /// next rAF tick.
    #[wasm_bindgen(js_name = setTrajectoryTime)]
    pub fn set_trajectory_time(&self, t: f64) {
        self.inner.borrow_mut().wire.trajectory_time = t;
    }

    /// Replaces the planner-tree overlay. Edges become depth-tested
    /// overlay lines, nodes become depth-tested overlay points.
    #[wasm_bindgen(js_name = setTreeOverlay)]
    pub fn set_tree_overlay(&self, json: &str) -> Result<(), JsValue> {
        let parsed = parse_tree_overlay(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.inner.borrow_mut().wire.tree_overlay = Some(parsed);
        Ok(())
    }

    /// Places the goal handle. Subsequent drags fire `onGoalChanged`.
    #[wasm_bindgen(js_name = setGoal)]
    pub fn set_goal(&self, json: &str) -> Result<(), JsValue> {
        let parsed = parse_goal(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.inner.borrow_mut().wire.goal = Some(parsed);
        Ok(())
    }

    /// Registers a JS callback fired after each goal-drag finishes
    /// (pointerup). Receives the new pose JSON as a string. Pass `null`
    /// or omit to unregister.
    #[wasm_bindgen(js_name = onGoalChanged)]
    pub fn on_goal_changed(&self, callback: JsValue) {
        let mut inner = self.inner.borrow_mut();
        if callback.is_null() || callback.is_undefined() {
            inner.wire.goal_callback = None;
        } else if let Ok(f) = callback.dyn_into::<js_sys::Function>() {
            inner.wire.goal_callback = Some(f);
        }
    }
}

/// Creates a raster renderer inside the element with the given id, sized
/// to it.
#[wasm_bindgen]
pub async fn create_raster(host_id: String) -> Result<RasterInstance, JsValue> {
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
    let mut state = State::new(instance, surface, w, h).await;
    // The R3 demo scene populated some instances + overlays for the
    // standalone widget. For the R4 wire-driven path the host pushes
    // its own data; start empty so duplicate axes don't appear.
    state.scene.clear();

    let shared = Rc::new(Shared {
        pending_resize: Cell::new(None),
        visible: Cell::new(true),
    });

    let inner = Rc::new(RefCell::new(Inner {
        state,
        canvas: canvas.clone(),
        shared: shared.clone(),
        wire: WireScene::default(),
        dragging_goal: false,
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

    // pointerdown: pick goal handle if any, else start camera orbit.
    {
        let inner = inner.clone();
        add(
            "pointerdown",
            Closure::wrap(Box::new(move |e: web_sys::Event| {
                let Ok(m) = e.dyn_into::<web_sys::MouseEvent>() else {
                    return;
                };
                let mut g = inner.borrow_mut();
                let canvas_w = g.canvas.client_width().max(1) as f32;
                let canvas_h = g.canvas.client_height().max(1) as f32;
                let mx = m.offset_x() as f32;
                let my = m.offset_y() as f32;

                if let Some(goal_pos) = g.wire.goal_world_position() {
                    let (origin, dir) = camera_ray(&g.state.camera, canvas_w, canvas_h, mx, my);
                    if ray_hits_sphere(origin, dir, goal_pos, GOAL_HANDLE_RADIUS * 1.4) {
                        g.dragging_goal = true;
                        return;
                    }
                }
                g.state
                    .camera
                    .press(m.client_x() as f64, m.client_y() as f64);
            }) as Box<dyn FnMut(web_sys::Event)>),
        )?;
    }

    // pointermove: drag goal or orbit camera.
    {
        let inner = inner.clone();
        add(
            "pointermove",
            Closure::wrap(Box::new(move |e: web_sys::Event| {
                let Ok(m) = e.dyn_into::<web_sys::MouseEvent>() else {
                    return;
                };
                let mut g = inner.borrow_mut();
                if g.dragging_goal {
                    let canvas_w = g.canvas.client_width().max(1) as f32;
                    let canvas_h = g.canvas.client_height().max(1) as f32;
                    let (origin, dir) = camera_ray(
                        &g.state.camera,
                        canvas_w,
                        canvas_h,
                        m.offset_x() as f32,
                        m.offset_y() as f32,
                    );
                    if let Some(hit) = ray_floor_hit(origin, dir) {
                        if let Some(goal) = g.wire.goal.as_mut() {
                            goal.pose.translation.vector =
                                [hit[0] as f64, hit[1] as f64, hit[2] as f64];
                        }
                    }
                } else {
                    g.state
                        .camera
                        .on_cursor(m.client_x() as f64, m.client_y() as f64);
                }
            }) as Box<dyn FnMut(web_sys::Event)>),
        )?;
    }

    // pointerup: finish drag (fire callback) or release camera.
    {
        let inner = inner.clone();
        add(
            "pointerup",
            Closure::wrap(Box::new(move |_e: web_sys::Event| {
                let mut g = inner.borrow_mut();
                if g.dragging_goal {
                    g.dragging_goal = false;
                    if let (Some(goal), Some(cb)) =
                        (g.wire.goal.as_ref(), g.wire.goal_callback.as_ref())
                    {
                        let payload = serialize_pose(&goal.pose);
                        let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(&payload));
                    }
                } else {
                    g.state.camera.release();
                }
            }) as Box<dyn FnMut(web_sys::Event)>),
        )?;
    }

    // wheel: zoom (unchanged from R0).
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

    Ok(RasterInstance {
        inner,
        _raf: raf,
        _resize_observer: resize_observer,
        _resize_cb: resize_cb,
        _intersection_observer: intersection_observer,
        _intersection_cb: intersection_cb,
        _listeners: listeners,
    })
}

impl Drop for RasterInstance {
    fn drop(&mut self) {
        self._resize_observer.disconnect();
        self._intersection_observer.disconnect();
        let inner = self.inner.borrow();
        for (event, cb) in &self._listeners {
            let _ = inner
                .canvas
                .remove_event_listener_with_callback(event, cb.as_ref().unchecked_ref());
        }
    }
}
