//! Real-time rasterized renderer (R0 stub).
//!
//! This is the second of Quasi's two pipelines (see plan
//! `0002-realtime-rasterization`). R0 lands the module split; R1 will add
//! a forward triangle pipeline that draws a single shaded mesh native +
//! web. The shape will be: a `State` similar to
//! [`pathtrace::State`](crate::pathtrace::State), owning its own surface
//! and pipelines, sharing nothing with the path tracer below the
//! [`gpu`](crate::gpu) seam.
