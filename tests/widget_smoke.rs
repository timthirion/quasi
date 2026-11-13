//! Widget smoke test: build the interactive path tracer exactly as the
//! window / browser widget does, render, read the presented frame back,
//! and assert it is not black.
//!
//! Why this exists: from plan 0028 (2026-06-15) until 2026-09-15 the
//! widget rendered nothing — an AOV-table panic, a pipeline rejected for
//! exceeding the colour-attachment byte limit, and a mis-sized present
//! uniform — while `cargo test` and the wasm32 check stayed green, because
//! nothing ever instantiated `pathtrace::State`. See R0003's findings.
//!
//! Unlike the other GPU tests this one is **not** `#[ignore]`d: CI runs it
//! on Mesa's software Vulkan (lavapipe). With no adapter at all it skips
//! with a message, unless `QUASI_REQUIRE_GPU=1` is set — CI sets it, so a
//! runner that lost its adapter fails instead of passing vacuously.
//!
//! A frame that passes here can still fail in a browser (naga accepts
//! WGSL that Safari rejects — R0003 `safari-black`), so this is a floor,
//! not a substitute for the browser check.

use quasi::gpu::make_instance;
use quasi::pathtrace::present::FrameStats;
use quasi::pathtrace::State;

/// Small enough for lavapipe to render in well under a second per frame.
const W: u32 = 64;
const H: u32 = 48;

fn adapter() -> Option<wgpu::Adapter> {
    let found = pollster::block_on(
        make_instance().request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }),
    )
    .ok();
    match &found {
        Some(a) => eprintln!("widget_smoke adapter: {:?}", a.get_info()),
        None if std::env::var_os("QUASI_REQUIRE_GPU").is_some() => {
            panic!("QUASI_REQUIRE_GPU is set but no wgpu adapter was found")
        }
        None => eprintln!("widget_smoke: no wgpu adapter, skipping"),
    }
    found
}

fn headless(w: u32, h: u32) -> Option<State> {
    let adapter = adapter()?;
    Some(pollster::block_on(State::new_headless(adapter, w, h)))
}

fn render_and_read(state: &mut State, frames: u32) -> (Vec<u8>, FrameStats) {
    for _ in 0..frames {
        state.render();
    }
    let rgba = state.read_presented().expect("headless state reads back");
    let stats = FrameStats::from_rgba8(&rgba);
    (rgba, stats)
}

/// Thresholds for "the default Cornell scene actually rendered". The box
/// fills the frame, so a working renderer lights nearly every pixel; a
/// no-op renderer produces exactly zero. Measured values are printed so a
/// drift is visible in the test log long before it trips these.
fn assert_rendered(stats: FrameStats, what: &str) {
    eprintln!("widget_smoke {what}: {stats:?}");
    assert!(
        stats.nonzero_frac > 0.5,
        "{what}: only {:.1}% of pixels are non-black — widget is not rendering",
        stats.nonzero_frac * 100.0
    );
    assert!(
        stats.mean > 10.0,
        "{what}: mean {:.2}/255 is near black",
        stats.mean
    );
}

/// One test, one `State`, scenarios in sequence. On lavapipe almost all of
/// the cost is device creation plus the LLVM compile of the megakernel —
/// five separate tests each paying that took 443 s of CI time, against a
/// few seconds for the renders themselves. Every assertion names its
/// scenario, so a failure still says which one broke.
#[test]
fn widget_renders_non_black() {
    let Some(mut state) = headless(W, H) else {
        return;
    };

    // The very first frame is what a reader sees on page load.
    let (rgba, stats) = render_and_read(&mut state, 1);
    assert_eq!(rgba.len(), (W * H * 4) as usize);
    assert_rendered(stats, "first frame");
    let first = stats;

    // Accumulation keeps drawing.
    let (_, stats) = render_and_read(&mut state, 7);
    assert_eq!(state.frame_count, 8);
    assert_rendered(stats, "8 frames");

    // A camera move restarts accumulation. The camera has not actually
    // moved, so the restarted frame must reproduce the first one exactly:
    // that proves the restart reset the accumulation, not just the counter.
    state.camera.dirty = true;
    let (_, stats) = render_and_read(&mut state, 1);
    assert_eq!(state.frame_count, 1, "a dirty camera restarts accumulation");
    assert_rendered(stats, "after camera move");
    assert_eq!(stats, first, "restarted frame differs from the first frame");

    // Display mode 1 shows the adaptive-sampling variance AOV — the AOV
    // whose addition broke the widget in plan 0028. Variance is noise, so
    // only require that something is drawn.
    state.set_display_mode(1);
    let (_, stats) = render_and_read(&mut state, 3);
    eprintln!("widget_smoke variance mode: {stats:?}");
    assert!(stats.max > 0, "variance display is entirely black");
    state.set_display_mode(0);

    // The widget resizes to its observed canvas size every tick; resizing
    // rebuilds every AOV target and bind group.
    state.resize(80, 60);
    assert_eq!(state.size(), (80, 60));
    assert_eq!(state.frame_count, 0, "resize restarts accumulation");
    let (rgba, stats) = render_and_read(&mut state, 4);
    assert_eq!(rgba.len(), 80 * 60 * 4);
    assert_rendered(stats, "after resize");
}
