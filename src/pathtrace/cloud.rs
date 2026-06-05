//! CPU mirror of the procedural cloud density function in
//! `pathtrace.wgsl`.
//!
//! The WGSL evaluates this density per-position to drive delta
//! tracking + ratio tracking; keeping it Rust-side means the tests
//! in `tests/cloud.rs` can pin the geometric invariants (zero
//! outside the sphere, bounded inside, deterministic per position).
//! Future CPU-reference integrators can lean on these exact
//! functions instead of re-deriving them.
//!
//! Constants and hashing match the WGSL byte-for-byte — divergence
//! is what the tests are guarding.

/// Frequency the noise grid is sampled at — matches
/// `CLOUD_NOISE_FREQ` in `pathtrace.wgsl`.
pub const NOISE_FREQ: f32 = 4.0;

/// Number of fbm octaves — matches `CLOUD_OCTAVES`.
pub const OCTAVES: i32 = 4;

/// Threshold subtracted from the fbm value before clamping. Matches
/// `CLOUD_NOISE_THRESHOLD`.
pub const NOISE_THRESHOLD: f32 = 0.2;

/// Gain applied after thresholding. Matches `CLOUD_NOISE_GAIN`.
pub const NOISE_GAIN: f32 = 1.8;

fn hash3(p: [i32; 3]) -> u32 {
    let ux = (p[0].wrapping_add(73_856_093)) as u32;
    let uy = (p[1].wrapping_add(19_349_663)) as u32;
    let uz = (p[2].wrapping_add(83_492_791)) as u32;
    let mut h =
        ux.wrapping_mul(0x9e37_79b1) ^ uy.wrapping_mul(0x85eb_ca6b) ^ uz.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

fn value_at(p: [i32; 3]) -> f32 {
    hash3(p) as f32 / 4_294_967_296.0
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn value_noise_3d(pos: [f32; 3]) -> f32 {
    let pf = [pos[0].floor(), pos[1].floor(), pos[2].floor()];
    let pi = [pf[0] as i32, pf[1] as i32, pf[2] as i32];
    let frac = [pos[0] - pf[0], pos[1] - pf[1], pos[2] - pf[2]];
    let s = [
        frac[0] * frac[0] * (3.0 - 2.0 * frac[0]),
        frac[1] * frac[1] * (3.0 - 2.0 * frac[1]),
        frac[2] * frac[2] * (3.0 - 2.0 * frac[2]),
    ];

    let c000 = value_at([pi[0], pi[1], pi[2]]);
    let c100 = value_at([pi[0] + 1, pi[1], pi[2]]);
    let c010 = value_at([pi[0], pi[1] + 1, pi[2]]);
    let c110 = value_at([pi[0] + 1, pi[1] + 1, pi[2]]);
    let c001 = value_at([pi[0], pi[1], pi[2] + 1]);
    let c101 = value_at([pi[0] + 1, pi[1], pi[2] + 1]);
    let c011 = value_at([pi[0], pi[1] + 1, pi[2] + 1]);
    let c111 = value_at([pi[0] + 1, pi[1] + 1, pi[2] + 1]);

    let x00 = mix(c000, c100, s[0]);
    let x10 = mix(c010, c110, s[0]);
    let x01 = mix(c001, c101, s[0]);
    let x11 = mix(c011, c111, s[0]);
    let y0 = mix(x00, x10, s[1]);
    let y1 = mix(x01, x11, s[1]);
    mix(y0, y1, s[2])
}

/// fbm of value noise. Matches WGSL `cloud_fbm`.
pub fn fbm(pos: [f32; 3]) -> f32 {
    let mut sum = 0.0_f32;
    let mut freq = 1.0_f32;
    let mut amp = 0.5_f32;
    let mut norm = 0.0_f32;
    for _ in 0..OCTAVES {
        let p = [pos[0] * freq, pos[1] * freq, pos[2] * freq];
        sum += amp * value_noise_3d(p);
        norm += amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    sum / norm
}

/// Procedural cloud density at `pos`. Matches WGSL `cloud_density`.
/// Returns 0 outside the sphere; positive inside.
pub fn density(pos: [f32; 3], center: [f32; 3], radius: f32) -> f32 {
    let r2 =
        (pos[0] - center[0]).powi(2) + (pos[1] - center[1]).powi(2) + (pos[2] - center[2]).powi(2);
    let r = r2.sqrt() / radius.max(1e-6);
    if r >= 1.0 {
        return 0.0;
    }
    let edge = smoothstep(1.0, 0.5, r);
    let scaled = [
        pos[0] * NOISE_FREQ,
        pos[1] * NOISE_FREQ,
        pos[2] * NOISE_FREQ,
    ];
    let n = fbm(scaled);
    let body = ((n - NOISE_THRESHOLD) * NOISE_GAIN).max(0.0);
    edge * body
}
