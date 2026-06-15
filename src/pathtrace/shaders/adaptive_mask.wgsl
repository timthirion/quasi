// PT-adaptive (plan 0028): per-checkpoint active-mask update.
//
// Runs once every CHECKPOINT_INTERVAL samples (CPU-driven dispatch
// from `offscreen.rs`). Reads the current running radiance and
// E[Y²] accumulators, derives per-pixel relative standard error,
// and writes the new mask state:
//
//   1 = active        (still being sampled)
//   0 = converged-OK  (relative std error < noise_threshold)
//   2 = clamped       (hit max_spp without converging)
//
// Pixels whose mask is already 0 or 2 are left alone — terminations
// are sticky.
//
// PT-adaptive-sample-count follow-up: the same compute pass now also
// writes the per-pixel sample count to a parallel R32Uint texture.
// For every pixel that is *currently active* at this checkpoint
// (mask == 1u), we write `U.sample_count` — the total samples this
// pixel has drawn so far. Pixels that converge or clamp this
// checkpoint inherit that count (sticky); pixels that stay active
// will have their count overwritten next checkpoint. Final readback
// at render end gives the exact per-pixel sample budget — the
// quantity the equal-sample-budget bias-check needs.

struct MaskU {
    // Total accumulated samples so far (== AccumU.frame_count + 1
    // at the checkpoint boundary).
    sample_count: u32,
    min_spp: u32,
    max_spp: u32,
    _pad: u32,
    noise_threshold: f32,
    eps_dark: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform> U: MaskU;
@group(0) @binding(1) var radiance: texture_2d<f32>;
@group(0) @binding(2) var mean_y2: texture_2d<f32>;
@group(0) @binding(3) var active_mask: texture_storage_2d<r32uint, read_write>;
@group(0) @binding(4) var sample_count_tex: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(active_mask);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let coord = vec2<i32>(i32(id.x), i32(id.y));

    let cur = textureLoad(active_mask, coord).r;
    if (cur != 1u) { return; }

    // PT-adaptive-sample-count: this pixel is still active at this
    // checkpoint, so it has drawn `sample_count` samples so far.
    // Record that. If we converge or clamp below, the count stays
    // at this checkpoint's value; if we remain active, next
    // checkpoint will overwrite.
    textureStore(
        sample_count_tex,
        coord,
        vec4<u32>(U.sample_count, 0u, 0u, 0u),
    );

    if (U.sample_count < U.min_spp) { return; }

    let r = textureLoad(radiance, coord, 0);
    let y = 0.2126 * r.r + 0.7152 * r.g + 0.0722 * r.b;
    let e_y2 = textureLoad(mean_y2, coord, 0).r;
    let var_y = max(e_y2 - y * y, 0.0);
    let denom = max(y, U.eps_dark);
    let std_err = sqrt(var_y / f32(U.sample_count)) / denom;

    if (std_err < U.noise_threshold) {
        textureStore(active_mask, coord, vec4<u32>(0u, 0u, 0u, 0u));
    } else if (U.sample_count >= U.max_spp) {
        textureStore(active_mask, coord, vec4<u32>(2u, 0u, 0u, 0u));
    }
}
