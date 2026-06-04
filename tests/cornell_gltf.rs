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
const CORNELL_SPHERE: &[u8] = include_bytes!("../data/gltf/cornell_sphere.gltf");
const CORNELL_BUNNY: &[u8] = include_bytes!("../data/gltf/cornell_bunny.gltf");
const CORNELL_METAL_BUNNY: &[u8] = include_bytes!("../data/gltf/cornell_metal_bunny.gltf");
const CORNELL_GLASS_SPHERE: &[u8] = include_bytes!("../data/gltf/cornell_glass_sphere.gltf");
const CORNELL_GLASS_BUNNY: &[u8] = include_bytes!("../data/gltf/cornell_glass_bunny.gltf");
const CORNELL_TEXTURED_FLOOR: &[u8] =
    include_bytes!("../data/gltf/cornell_textured_floor.gltf");

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
fn cornell_sphere_has_expected_topology() {
    let scene = load_glb_bytes(CORNELL_SPHERE).expect("load cornell_sphere.gltf");
    // 6 room quads × 2 triangles + level-5 icosphere (20 × 4^5 = 20480).
    assert_eq!(scene.triangle_count(), 6 * 2 + 20480);
    // 1 default + 4 room (white, red, green, light) + 1 sphere material.
    assert_eq!(scene.materials.len(), 6);
    // Only the light quad is emissive — 1 quad × 2 triangles.
    assert_eq!(scene.emissive_triangles.len(), 2);
    // BVH built at load time — non-empty.
    assert!(
        scene.bvh.nodes.len() > 1,
        "20k-triangle scene should build a multi-node BVH",
    );
}

#[test]
fn cornell_bunny_has_expected_topology() {
    let scene = load_glb_bytes(CORNELL_BUNNY).expect("load cornell_bunny.gltf");
    // 6 room quads × 2 + Stanford bunny (4968 triangles in the morsel
    // OBJ; quads are fan-triangulated so any non-tri faces would bump
    // the count). The OBJ ships pure triangles, so 4968 is exact.
    assert_eq!(scene.triangle_count(), 6 * 2 + 4968);
    // 1 default + 4 room (white/red/green/light) + 1 bunny material.
    assert_eq!(scene.materials.len(), 6);
    assert_eq!(scene.emissive_triangles.len(), 2);
    // BVH must exist for the renderer.
    assert!(scene.bvh.nodes.len() > 1);
}

#[test]
fn cornell_metal_bunny_topology_matches_clay_bunny_but_material_is_metallic() {
    let clay = load_glb_bytes(CORNELL_BUNNY).expect("load cornell_bunny.gltf");
    let metal = load_glb_bytes(CORNELL_METAL_BUNNY).expect("load cornell_metal_bunny.gltf");

    // Same geometry as the clay bunny — only the bunny material differs.
    assert_eq!(metal.triangle_count(), clay.triangle_count());
    assert_eq!(metal.materials.len(), clay.materials.len());
    assert_eq!(metal.emissive_triangles.len(), clay.emissive_triangles.len());

    // The new scene must carry at least one fully metallic (metallic=1)
    // material — anything less and the WGSL `metallic > 0.5` dispatch
    // would never route into the GGX branch.
    let any_metal = metal
        .materials
        .iter()
        .any(|m| m.metallic >= 0.5 && m.emission.iter().all(|&e| e < 0.1));
    assert!(any_metal, "expected a metallic (non-emissive) material");

    // ...and the clay bunny scene has *no* metallic materials, by
    // contrast. Pins that the regression won't be hidden by an old
    // metallic value sneaking into the all-Lambertian scenes.
    let any_metal_in_clay = clay.materials.iter().any(|m| m.metallic >= 0.5);
    assert!(!any_metal_in_clay, "cornell_bunny should be pure Lambertian");
}

#[test]
fn cornell_glass_sphere_topology_matches_lambertian_sphere_but_carries_ior() {
    let lambertian = load_glb_bytes(CORNELL_SPHERE).expect("load cornell_sphere.gltf");
    let glass = load_glb_bytes(CORNELL_GLASS_SPHERE).expect("load cornell_glass_sphere.gltf");

    // Same icosphere geometry, just a different sphere material.
    assert_eq!(glass.triangle_count(), lambertian.triangle_count());
    assert_eq!(glass.materials.len(), lambertian.materials.len());
    assert_eq!(glass.emissive_triangles.len(), lambertian.emissive_triangles.len());

    // Round-trip the ior through `extras` and out — the glass
    // scene must carry exactly one dielectric material with
    // ior ≈ 1.5 and no emission.
    let glass_mats: Vec<_> = glass
        .materials
        .iter()
        .filter(|m| m.ior > 0.0 && m.emission.iter().all(|&e| e < 0.1))
        .collect();
    assert_eq!(glass_mats.len(), 1, "expected exactly one dielectric material");
    let m = glass_mats[0];
    assert!((m.ior - 1.5).abs() < 1e-3, "ior should round-trip to 1.5; got {}", m.ior);

    // ...and the Lambertian sphere scene has *no* dielectrics —
    // pins that the extras round-trip doesn't accidentally fire for
    // scenes that omit it.
    let any_dielectric = lambertian.materials.iter().any(|m| m.ior > 0.0);
    assert!(!any_dielectric, "cornell_sphere should carry no dielectrics");
}

#[test]
fn cornell_glass_bunny_has_a_dielectric_with_non_zero_absorption() {
    let scene = load_glb_bytes(CORNELL_GLASS_BUNNY).expect("load cornell_glass_bunny.gltf");
    // Same bunny geometry as the clay scene — just a different
    // material. Find the dielectric with absorption and pin its
    // extras round-trip.
    let dielectrics: Vec<_> = scene
        .materials
        .iter()
        .filter(|m| m.ior > 0.0 && m.absorption.iter().any(|&c| c > 0.0))
        .collect();
    assert_eq!(
        dielectrics.len(),
        1,
        "expected exactly one absorbing-dielectric material",
    );
    let m = dielectrics[0];
    assert!((m.ior - 1.5).abs() < 1e-3, "ior should be 1.5; got {}", m.ior);
    // Sanity: green-tinted glass means the green channel attenuates
    // less than the red and blue channels.
    assert!(
        m.absorption[1] < m.absorption[0] && m.absorption[1] < m.absorption[2],
        "green channel should absorb least (this is *green* glass); got {:?}",
        m.absorption,
    );
    // And the clay bunny carries no absorption, by contrast.
    let clay = load_glb_bytes(CORNELL_BUNNY).expect("load cornell_bunny.gltf");
    let any_absorbing = clay.materials.iter().any(|m| m.absorption.iter().any(|&c| c > 0.0));
    assert!(!any_absorbing, "cornell_bunny should be pure Lambertian");
}

#[test]
fn cornell_textured_floor_texture_contains_non_zero_pixels() {
    let scene = load_glb_bytes(CORNELL_TEXTURED_FLOOR).expect("load");
    let tex = &scene.textures[0];
    let non_zero_alpha = tex.rgba.chunks_exact(4).filter(|p| p[3] != 0).count();
    let non_white = tex
        .rgba
        .chunks_exact(4)
        .filter(|p| !(p[0] == 255 && p[1] == 255 && p[2] == 255))
        .count();
    let total = tex.rgba.len() / 4;
    eprintln!(
        "texture: {total} pixels; {non_zero_alpha} non-zero alpha; {non_white} non-white",
    );
    eprintln!("first 16 bytes: {:?}", &tex.rgba[..16]);
    assert!(non_zero_alpha > total / 2, "texture is mostly transparent");
    assert!(non_white > total / 2, "texture is mostly white — PNG decode failed?");
}

#[test]
fn cornell_textured_floor_carries_uvs_on_floor_vertices() {
    let scene = load_glb_bytes(CORNELL_TEXTURED_FLOOR).expect("load");
    // At least one vertex must have a non-zero UV. If every vertex
    // has uv = (0, 0), the texture sample collapses to a single texel.
    let with_uv = scene
        .vertices
        .iter()
        .filter(|v| v.uv != [0.0, 0.0])
        .count();
    assert!(
        with_uv > 0,
        "every vertex's UV is (0, 0); the loader didn't see TEXCOORD_0"
    );
    let uvs: Vec<[f32; 2]> = scene.vertices.iter().map(|v| v.uv).collect();
    eprintln!("sample UVs (first 8): {:?}", &uvs[..8.min(uvs.len())]);
}

#[test]
fn cornell_textured_floor_has_the_embedded_uv_checker() {
    use quasi::pathtrace::mesh::NO_TEXTURE;
    let scene = load_glb_bytes(CORNELL_TEXTURED_FLOOR).expect("load");
    // Same topology as cornell_quads (room + 2 boxes, subdiv 1).
    assert_eq!(scene.triangle_count(), 32);
    // 1 default + 4 base materials + 1 textured floor material.
    assert_eq!(scene.materials.len(), 6);
    // Exactly one material references the lone texture (layer 0).
    let with_texture: Vec<_> = scene
        .materials
        .iter()
        .filter(|m| m.base_color_texture_idx != NO_TEXTURE)
        .collect();
    assert_eq!(with_texture.len(), 1);
    assert_eq!(with_texture[0].base_color_texture_idx, 0);
    // One texture in the scene, 1024×1024 RGBA (the embedded
    // uv_checker_color.png).
    assert_eq!(scene.textures.len(), 1);
    assert_eq!(scene.textures[0].width, 1024);
    assert_eq!(scene.textures[0].height, 1024);
    assert_eq!(scene.textures[0].rgba.len(), 1024 * 1024 * 4);
    // The light quad's 2 emissive triangles still get picked up.
    assert_eq!(scene.emissive_triangles.len(), 2);
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

/// GPU regression: the BVH walk and the brute-force linear scan should
/// produce visually identical images. Both find the same closest
/// triangle hit; only the traversal order differs. RMSE is dominated
/// by floating-point ordering noise. Ignored / native-only for the
/// same reasons as the previous test.
#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore]
fn bvh_traversal_agrees_with_brute_force() {
    use quasi::pathtrace::integrator::IntegratorKind;
    use quasi::pathtrace::metrics::rmse_rgb;
    use quasi::pathtrace::offscreen::{render_offscreen, RenderConfig};
    use quasi::pathtrace::sampler::SamplerKind;

    let scene = load_glb_bytes(CORNELL_TRIS).expect("tris");

    let make_cfg = |use_bvh: bool| RenderConfig {
        width: 128,
        height: 128,
        samples: 64,
        sampler: SamplerKind::Pcg,
        integrator: IntegratorKind::MisNee,
        use_bvh,
        ..RenderConfig::default()
    };

    let bvh = render_offscreen(make_cfg(true), &scene);
    let brute = render_offscreen(make_cfg(false), &scene);
    let rmse = rmse_rgb(&bvh.radiance, &brute.radiance);
    eprintln!("bvh vs brute-force on cornell_tris: rmse = {rmse:.8}");
    // Identical RNG, identical scene, identical Möller-Trumbore — the
    // only divergence source is the order Möller-Trumbore is invoked
    // in, which can change which `t` value happens to be the closest
    // when two triangles are nearly coplanar. In practice the Cornell
    // box has no near-coplanar triangles, so RMSE should be tiny.
    assert!(
        rmse < 1e-3,
        "rmse {rmse:.8} too large — BVH traversal disagrees with brute-force?",
    );
}

/// GPU benchmark (also `#[ignore]`): renders cornell_tris.gltf with
/// the BVH and with the brute-force linear scan, prints the wall-clock
/// time of each, and asserts the BVH is **at least faster** than brute
/// force. The plan's 10× target is specifically scoped to the
/// Stanford-bunny-in-Cornell scene (T4, ~70k triangles); at 512
/// triangles the BVH's per-iteration stack overhead eats into the
/// savings. This test enforces the no-regression bar so we'd catch a
/// BVH that's silently broken or slower than the loop it replaces.
#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore]
fn bvh_is_faster_than_brute_force_at_512_triangles() {
    use quasi::pathtrace::offscreen::{render_offscreen, RenderConfig};
    use std::time::Instant;

    let scene = load_glb_bytes(CORNELL_TRIS).expect("tris");
    let make_cfg = |use_bvh: bool| RenderConfig {
        width: 256,
        height: 256,
        samples: 64,
        use_bvh,
        ..RenderConfig::default()
    };

    // Warm-up: GPU pipeline / shader compilation lands in the first
    // render and shouldn't count.
    let _ = render_offscreen(make_cfg(true), &scene);

    let t = Instant::now();
    let _ = render_offscreen(make_cfg(true), &scene);
    let bvh_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let _ = render_offscreen(make_cfg(false), &scene);
    let brute_ms = t.elapsed().as_secs_f64() * 1000.0;

    let speedup = brute_ms / bvh_ms;
    eprintln!(
        "cornell_tris (512 tris) @ 256x256 / 64 spp: bvh = {bvh_ms:.1} ms, brute = {brute_ms:.1} ms, speedup = {speedup:.1}x",
    );
    assert!(
        speedup >= 2.0,
        "bvh speedup {speedup:.1}x < 2x — the BVH should at least beat a 512-triangle linear scan",
    );
}

/// The plan's headline 10× target lives here, on the 20k-triangle
/// cornell_sphere.gltf (T4 publishable scene). Same #[ignore] gating.
#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore]
fn bvh_is_at_least_10x_faster_at_20k_triangles() {
    use quasi::pathtrace::offscreen::{render_offscreen, RenderConfig};
    use std::time::Instant;

    let scene = load_glb_bytes(CORNELL_SPHERE).expect("sphere");
    let make_cfg = |use_bvh: bool| RenderConfig {
        width: 256,
        height: 256,
        samples: 64,
        use_bvh,
        ..RenderConfig::default()
    };

    let _ = render_offscreen(make_cfg(true), &scene); // warm-up

    let t = Instant::now();
    let _ = render_offscreen(make_cfg(true), &scene);
    let bvh_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let _ = render_offscreen(make_cfg(false), &scene);
    let brute_ms = t.elapsed().as_secs_f64() * 1000.0;

    let speedup = brute_ms / bvh_ms;
    eprintln!(
        "cornell_sphere ({} tris) @ 256x256 / 64 spp: bvh = {bvh_ms:.0} ms, brute = {brute_ms:.0} ms, speedup = {speedup:.1}x",
        scene.triangle_count(),
    );
    assert!(
        speedup >= 10.0,
        "bvh speedup {speedup:.1}x < 10x — plan target missed at 20k triangles",
    );
}
