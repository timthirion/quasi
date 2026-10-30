//! Shared GPU and platform plumbing used by both pipelines.
//!
//! Quasi is a dual-pipeline renderer ([`pathtrace`](crate::pathtrace) for
//! offline-quality stills, [`raster`](crate::raster) for real-time scenes).
//! This module owns the bits both pipelines need — currently the wgpu
//! `Instance` factory and the `OrbitCamera` — without leaking either
//! pipeline's scene shape or shaders into the other.

mod camera;
#[cfg(test)]
pub(crate) mod wgsl_layout;

pub use camera::OrbitCamera;

/// Route wgpu validation errors to the browser console on the web target.
///
/// The native `wgpu_core` backend panics on an uncaptured error by default,
/// so a bad pipeline or bind group fails loudly there. The WebGPU backend
/// has no default handler: without this call the error goes to the
/// device's `uncapturederror` event, nothing in the page listens, and the
/// widget renders black with no message. That is how the plan 0028
/// regressions shipped unnoticed. Native keeps the panicking default, so
/// this is a no-op outside wasm32.
pub fn install_error_logger(device: &wgpu::Device) {
    #[cfg(target_arch = "wasm32")]
    device.on_uncaptured_error(std::sync::Arc::new(|err| {
        log::error!("wgpu uncaptured error: {err}");
    }));
    #[cfg(not(target_arch = "wasm32"))]
    let _ = device;
}

/// Run `f` inside a validation error scope on the web target and log any
/// error under `label`.
///
/// [`install_error_logger`] is not enough on its own: Safari does not
/// deliver `uncapturederror` events to wgpu's handler, so a pipeline that
/// fails validation there shows up only as a black canvas. Error scopes
/// report through a promise instead, and name the object that failed.
/// Native runs `f` directly and keeps the panicking default handler.
pub async fn validated<T>(device: &wgpu::Device, label: &str, f: impl FnOnce() -> T) -> T {
    #[cfg(target_arch = "wasm32")]
    {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let out = f();
        if let Some(err) = scope.pop().await {
            log::error!("wgpu validation error in {label}: {err}");
        }
        out
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (device, label);
        f()
    }
}

/// Construct a wgpu instance suitable for both native and browser targets.
pub fn make_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}
