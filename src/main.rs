//! Native entry point. Initializes logging and runs the renderer's event loop.
//! (The web build is a cdylib driven from JavaScript; see the `web` module.)

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    quasi::run();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
