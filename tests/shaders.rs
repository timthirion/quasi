//! Validates that the WGSL shaders parse and pass naga validation — the same
//! front-end wgpu uses at runtime — so shader errors are caught by `cargo test`
//! without needing a GPU or a display.

use naga::valid::{Capabilities, ValidationFlags, Validator};

fn validate(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{name}: WGSL parse error:\n{}", e.emit_to_string(src)));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: WGSL validation error: {e:?}"));
}

#[test]
fn pathtrace_shader_is_valid() {
    validate(
        "pathtrace",
        include_str!("../src/pathtrace/shaders/pathtrace.wgsl"),
    );
}

#[test]
fn accumulate_shader_is_valid() {
    validate(
        "accumulate",
        include_str!("../src/pathtrace/shaders/accumulate.wgsl"),
    );
}

#[test]
fn present_shader_is_valid() {
    validate(
        "present",
        include_str!("../src/pathtrace/shaders/present.wgsl"),
    );
}

#[test]
fn forward_shader_is_valid() {
    validate(
        "forward",
        include_str!("../src/raster/shaders/forward.wgsl"),
    );
}

#[test]
fn overlay_shader_is_valid() {
    validate(
        "overlay",
        include_str!("../src/raster/shaders/overlay.wgsl"),
    );
}

#[test]
fn pathtrace_sampler_constants_match_cpu_side() {
    // WGSL `SAMPLER_*` constants must match the discriminants of
    // `pathtrace::sampler::SamplerKind`. naga validation doesn't catch a
    // drift here (each side parses fine independently); this guard does.
    let src = include_str!("../src/pathtrace/shaders/pathtrace.wgsl");
    for (name, expected) in [
        ("SAMPLER_PCG", 0u32),
        ("SAMPLER_HALTON", 1u32),
        ("SAMPLER_SOBOL", 2u32),
    ] {
        let needle = format!("const {name}: u32 = {expected}u;");
        assert!(
            src.contains(&needle),
            "expected `{needle}` in pathtrace.wgsl",
        );
    }
}

#[test]
fn pathtrace_integrator_constants_match_cpu_side() {
    let src = include_str!("../src/pathtrace/shaders/pathtrace.wgsl");
    for (name, expected) in [("INTEGRATOR_MIS_NEE", 0u32), ("INTEGRATOR_BSDF", 1u32)] {
        let needle = format!("const {name}: u32 = {expected}u;");
        assert!(
            src.contains(&needle),
            "expected `{needle}` in pathtrace.wgsl",
        );
    }
}
