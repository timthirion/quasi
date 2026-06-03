//! Shared GPU and platform plumbing used by both pipelines.
//!
//! Quasi is a dual-pipeline renderer ([`pathtrace`](crate::pathtrace) for
//! offline-quality stills, [`raster`](crate::raster) for real-time scenes).
//! This module owns the bits both pipelines need — currently the wgpu
//! `Instance` factory and the `OrbitCamera` — without leaking either
//! pipeline's scene shape or shaders into the other.

mod camera;

pub use camera::OrbitCamera;

/// Construct a wgpu instance suitable for both native and browser targets.
pub fn make_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}
