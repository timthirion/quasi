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
fn adaptive_mask_shader_is_valid() {
    validate(
        "adaptive_mask",
        include_str!("../src/pathtrace/shaders/adaptive_mask.wgsl"),
    );
}

#[test]
fn bloom_shaders_are_valid() {
    validate(
        "bloom_extract",
        include_str!("../src/pathtrace/shaders/bloom_extract.wgsl"),
    );
    validate(
        "bloom_downsample",
        include_str!("../src/pathtrace/shaders/bloom_downsample.wgsl"),
    );
    validate(
        "bloom_upsample",
        include_str!("../src/pathtrace/shaders/bloom_upsample.wgsl"),
    );
    validate(
        "bloom_composite",
        include_str!("../src/pathtrace/shaders/bloom_composite.wgsl"),
    );
}

/// Safari's WGSL compiler rejects an explicitly typed array constructor in
/// a module-scope `const` initializer, e.g.
/// `const T: array<array<u32, 32>, 32> = array<array<u32, 32>, 32>(...)`,
/// with "cannot use override value in constant expression". naga accepts
/// it, so the validation tests above can't see the problem; the pathtrace
/// pipeline failed in Safari and the widget rendered black. Write the
/// constructor as `array(...)` and let the declared type pin it.
#[test]
fn no_typed_array_constructor_in_module_const() {
    let shaders = [
        (
            "pathtrace",
            include_str!("../src/pathtrace/shaders/pathtrace.wgsl"),
        ),
        (
            "accumulate",
            include_str!("../src/pathtrace/shaders/accumulate.wgsl"),
        ),
        (
            "present",
            include_str!("../src/pathtrace/shaders/present.wgsl"),
        ),
        (
            "adaptive_mask",
            include_str!("../src/pathtrace/shaders/adaptive_mask.wgsl"),
        ),
        (
            "forward",
            include_str!("../src/raster/shaders/forward.wgsl"),
        ),
        (
            "overlay",
            include_str!("../src/raster/shaders/overlay.wgsl"),
        ),
    ];
    for (name, src) in shaders {
        for (i, line) in src.lines().enumerate() {
            let line = line.trim_start();
            if line.starts_with("const ") {
                if let Some((_, init)) = line.split_once('=') {
                    assert!(
                        !init.trim_start().starts_with("array<"),
                        "{name}.wgsl:{}: typed array constructor in a module const \
                         breaks Safari; use `array(...)`",
                        i + 1
                    );
                }
            }
        }
    }
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
