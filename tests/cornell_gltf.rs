//! Loads the committed Cornell glTF files and asserts the triangle /
//! material / emissive counts match what `examples/gen_cornell.rs`
//! produced. Catches any drift between the generator and the loader
//! (and any silent corruption of the committed `.gltf` files).
//!
//! The GPU regression at the bottom (`#[ignore]`) renders both files
//! through the triangle path tracer at moderate spp and compares the
//! radiance buffers — they should agree within sampling noise since
//! both describe the same Cornell geometry, just at different
//! tessellation densities. Run with `cargo test -- --ignored` on a
//! machine with a working GPU adapter.

use quasi::pathtrace::mesh::load_glb_bytes;

const CORNELL_QUADS: &[u8] = include_bytes!("../data/gltf/cornell_quads.gltf");
const CORNELL_TRIS: &[u8] = include_bytes!("../data/gltf/cornell_tris.gltf");

#[test]
fn cornell_quads_has_expected_topology() {
    let scene = load_glb_bytes(CORNELL_QUADS).expect("load cornell_quads.gltf");
    // 16 quads × 2 triangles per quad (subdiv = 1) = 32 triangles.
    assert_eq!(scene.triangle_count(), 32);
    // 1 default + 4 unique materials (white, red, green, light).
    assert_eq!(scene.materials.len(), 5);
    // Exactly one emissive material (the light).
    let emissive_materials: Vec<_> = scene.materials.iter().filter(|m| m.is_emissive()).collect();
    assert_eq!(emissive_materials.len(), 1);
    // Light is one quad → 2 triangles.
    assert_eq!(scene.emissive_triangles.len(), 2);
}

#[test]
fn cornell_tris_has_expected_topology() {
    let scene = load_glb_bytes(CORNELL_TRIS).expect("load cornell_tris.gltf");
    // 16 quads × 4×4 sub-quads × 2 triangles = 512 triangles.
    assert_eq!(scene.triangle_count(), 512);
    assert_eq!(scene.materials.len(), 5);
    // Light: 1 quad × 4×4 × 2 = 32 emissive triangles.
    assert_eq!(scene.emissive_triangles.len(), 32);
}

#[test]
fn both_files_share_the_same_material_palette() {
    let quads = load_glb_bytes(CORNELL_QUADS).expect("quads");
    let tris = load_glb_bytes(CORNELL_TRIS).expect("tris");
    assert_eq!(quads.materials.len(), tris.materials.len());
    for (a, b) in quads.materials.iter().zip(tris.materials.iter()) {
        assert_eq!(
            a, b,
            "material drift between cornell_quads and cornell_tris"
        );
    }
}

/// GPU regression: renders both Cornell glTFs through the triangle path
/// tracer and compares radiance. Same geometry → renders agree within
/// sampling noise. Ignored by default since `cargo test` on a headless
/// machine has no GPU adapter; run with `cargo test -- --ignored`.
/// Native-only because the offscreen renderer + metrics modules aren't
/// compiled for `wasm32-unknown-unknown`.
#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore]
fn cornell_quads_and_tris_render_to_the_same_image() {
    use quasi::pathtrace::integrator::IntegratorKind;
    use quasi::pathtrace::metrics::rmse_rgb;
    use quasi::pathtrace::offscreen::{render_offscreen, RenderConfig};
    use quasi::pathtrace::sampler::SamplerKind;

    let cfg = RenderConfig {
        width: 128,
        height: 128,
        samples: 256,
        sampler: SamplerKind::Pcg,
        integrator: IntegratorKind::MisNee,
        ..RenderConfig::default()
    };

    let quads_scene = load_glb_bytes(CORNELL_QUADS).expect("quads");
    let tris_scene = load_glb_bytes(CORNELL_TRIS).expect("tris");

    let quads_aovs = render_offscreen(cfg, &quads_scene);
    let tris_aovs = render_offscreen(cfg, &tris_scene);

    let rmse = rmse_rgb(&quads_aovs.radiance, &tris_aovs.radiance);
    eprintln!(
        "cornell_quads vs cornell_tris @ {}x{} / {} spp: rmse = {rmse:.6}",
        cfg.width, cfg.height, cfg.samples,
    );
    // PCG @ 256 spp leaves visible noise; both renders share the same
    // pixel jitter pattern, but they read different triangle counts,
    // so independent Monte-Carlo variance shows up. Threshold matches
    // what M3's PCG/MisNee numbers led us to expect at this spp.
    assert!(
        rmse < 0.05,
        "rmse {rmse:.6} exceeds threshold — geometry mismatch?",
    );
}
