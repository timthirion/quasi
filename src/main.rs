//! Native entry point. Initializes logging and blocks on the renderer's event loop.

fn main() {
    env_logger::init();
    pollster::block_on(quasi::run());
}
