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

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(active_mask);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    let coord = vec2<i32>(i32(id.x), i32(id.y));

    let cur = textureLoad(active_mask, coord).r;
    if (cur != 1u) { return; }
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
