//! Native entry point. Initializes logging and runs one of Quasi's renderers.
//!
//! `cargo run`              → path tracer (default).
//! `cargo run -- raster`    → real-time rasterizer.

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    let raster = std::env::args()
        .skip(1)
        .any(|a| a == "raster" || a == "--raster");
    if raster {
        quasi::run_raster();
    } else {
        quasi::run();
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
