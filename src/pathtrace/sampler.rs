//! Samplers — PCG, Halton, and Sobol.
//!
//! Each sampler produces a stream of 2-D points in `[0, 1)^2`. The same
//! mathematical definitions are mirrored in `shaders/pathtrace.wgsl`: the
//! CPU implementations here exist so the sequences can be tested against
//! canonical reference values off-GPU (per `AGENTS.md` testing guidance).
//!
//! Two QMC families are offered alongside the workhorse PRNG:
//!
//! - **Halton.** 2-D Halton uses prime bases (2, 3). To avoid identical
//!   sequences at every pixel, a per-pixel **Cranley–Patterson rotation**
//!   shifts each pixel's sequence by a hashed random offset.
//! - **Sobol.** Standard Sobol points in two dimensions with direction
//!   vectors derived from the canonical polynomials (dim 0: identity =
//!   van der Corput in base 2; dim 1: polynomial `x + 1`, `m_1 = 1`). A
//!   per-pixel XOR scramble (Owen-style on the raw `u32`) decorrelates
//!   pixels.
//!
//! Sequences are sized for one Cornell-Box bounce path. The integrator
//! draws ~2 two-dimensional points per bounce; the index advances each
//! call.

use std::str::FromStr;

/// First 16 primes — Halton bases for the first 16 dimensions. The path
/// tracer only ever asks for the first few, but the table is here so the
/// QMC math is uniform with the WGSL side.
pub const HALTON_PRIMES: [u32; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

/// Which sampler the integrator uses. Discriminants match the WGSL
/// `SAMPLER_*` constants so the `u32` round-trips through the uniform.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SamplerKind {
    #[default]
    Pcg = 0,
    Halton = 1,
    Sobol = 2,
}

impl SamplerKind {
    /// The discriminant exactly as written to the WGSL uniform.
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

impl FromStr for SamplerKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "pcg" => Ok(SamplerKind::Pcg),
            "halton" => Ok(SamplerKind::Halton),
            "sobol" => Ok(SamplerKind::Sobol),
            other => Err(format!(
                "unknown sampler: {other} (expected pcg|halton|sobol)"
            )),
        }
    }
}

/// A stream of 2-D points in `[0, 1)^2`.
pub trait Sampler {
    fn next_2d(&mut self) -> [f32; 2];
}

// ---------------------------------------------------------------------------
// PCG
// ---------------------------------------------------------------------------

/// PCG hash — used both to advance a stream and to derive per-pixel
/// scrambles for the QMC samplers. The exact mixing matches the WGSL
/// implementation.
pub fn pcg_hash(input: u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let shift = (state >> 28).wrapping_add(4);
    let word = ((state >> shift) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

#[inline]
fn u32_to_unit_float(x: u32) -> f32 {
    // 2^32 as f32. Maps the full u32 range to [0, 1).
    (x as f32) / 4_294_967_296.0_f32
}

/// Stateful PCG sampler — independent uniform pairs.
#[derive(Clone, Debug)]
pub struct Pcg {
    state: u32,
}

impl Pcg {
    /// Seeds the stream by hashing once so adjacent seeds decorrelate.
    pub fn new(seed: u32) -> Self {
        Pcg {
            state: pcg_hash(seed),
        }
    }
}

impl Sampler for Pcg {
    fn next_2d(&mut self) -> [f32; 2] {
        self.state = pcg_hash(self.state);
        let a = u32_to_unit_float(self.state);
        self.state = pcg_hash(self.state);
        let b = u32_to_unit_float(self.state);
        [a, b]
    }
}

// ---------------------------------------------------------------------------
// Halton
// ---------------------------------------------------------------------------

/// Radical inverse of `n` in the given prime base. The canonical low-
/// discrepancy 1-D building block; Halton is just radical inverses across
/// successive prime bases.
pub fn radical_inverse(base: u32, mut n: u32) -> f32 {
    let inv_base = 1.0 / base as f32;
    let mut inv_base_n = inv_base;
    let mut result = 0.0;
    while n > 0 {
        let digit = (n % base) as f32;
        result += digit * inv_base_n;
        inv_base_n *= inv_base;
        n /= base;
    }
    result
}

/// 2-D Halton sampler with per-pixel Cranley–Patterson rotation.
#[derive(Clone, Debug)]
pub struct Halton {
    index: u32,
    dim: u32,
    scramble: [f32; 2],
}

impl Halton {
    /// `index` is the sequence position (typically the frame count);
    /// `pixel_seed` produces the per-pixel rotation so neighbouring pixels
    /// don't see identical sequences.
    pub fn new(index: u32, pixel_seed: u32) -> Self {
        let sx = pcg_hash(pixel_seed);
        let sy = pcg_hash(sx);
        Halton {
            index,
            dim: 0,
            scramble: [u32_to_unit_float(sx), u32_to_unit_float(sy)],
        }
    }
}

impl Sampler for Halton {
    fn next_2d(&mut self) -> [f32; 2] {
        let bx = HALTON_PRIMES[(self.dim as usize) % HALTON_PRIMES.len()];
        let by = HALTON_PRIMES[(self.dim as usize + 1) % HALTON_PRIMES.len()];
        self.dim = self.dim.wrapping_add(2);
        let x = (radical_inverse(bx, self.index) + self.scramble[0]).fract();
        let y = (radical_inverse(by, self.index) + self.scramble[1]).fract();
        [x, y]
    }
}

// ---------------------------------------------------------------------------
// Sobol
// ---------------------------------------------------------------------------

/// Direction vectors for Sobol dimensions 0 and 1. Computed at compile
/// time from the canonical polynomial recurrences so the test suite can
/// pin the numerical output without hand-transcribing a table.
const SOBOL_DIRECTIONS: [[u32; 32]; 2] = build_sobol_directions();

const fn build_sobol_directions() -> [[u32; 32]; 2] {
    let mut t = [[0u32; 32]; 2];

    // Dim 0: v_i = 1 << (W-i), W = 32. Equivalent to van der Corput in
    // base 2 (bit-reversed binary).
    let mut i = 0;
    while i < 32 {
        t[0][i] = 1u32 << (31 - i);
        i += 1;
    }

    // Dim 1: primitive polynomial p(x) = x + 1 (degree 1), initial m_1 = 1.
    // The recurrence collapses to m_{i+1} = (m_i << 1) XOR m_i.
    // v_i = m_i << (W - i).
    let mut m: u32 = 1;
    let mut i = 0;
    while i < 32 {
        t[1][i] = m << (31 - i);
        if i < 31 {
            // u32 shift-by-32 is UB; the last m_i isn't needed.
            m = (m << 1) ^ m;
        }
        i += 1;
    }

    t
}

/// Raw 1-D Sobol value as a `u32` (top bits are the most-significant
/// fractional bits). Bit-by-bit XOR of selected direction vectors.
pub fn sobol_1d_raw(dim: usize, index: u32) -> u32 {
    let dirs = &SOBOL_DIRECTIONS[dim];
    let mut acc = 0u32;
    let mut i = 0;
    while i < 32 {
        if (index >> i) & 1 == 1 {
            acc ^= dirs[i];
        }
        i += 1;
    }
    acc
}

/// 2-D Sobol point in `[0, 1)^2` with an Owen-style XOR scramble.
pub fn sobol_2d(index: u32, scramble: [u32; 2]) -> [f32; 2] {
    let x = sobol_1d_raw(0, index) ^ scramble[0];
    let y = sobol_1d_raw(1, index) ^ scramble[1];
    [u32_to_unit_float(x), u32_to_unit_float(y)]
}

/// 2-D Sobol sampler with per-pixel XOR scramble.
#[derive(Clone, Debug)]
pub struct Sobol {
    index: u32,
    scramble: [u32; 2],
}

impl Sobol {
    pub fn new(index: u32, pixel_seed: u32) -> Self {
        let sx = pcg_hash(pixel_seed);
        let sy = pcg_hash(sx);
        Sobol {
            index,
            scramble: [sx, sy],
        }
    }
}

impl Sampler for Sobol {
    fn next_2d(&mut self) -> [f32; 2] {
        let p = sobol_2d(self.index, self.scramble);
        self.index = self.index.wrapping_add(1);
        p
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn radical_inverse_base_2_canonical() {
        assert!(close(radical_inverse(2, 1), 0.5));
        assert!(close(radical_inverse(2, 2), 0.25));
        assert!(close(radical_inverse(2, 3), 0.75));
        assert!(close(radical_inverse(2, 4), 0.125));
        assert!(close(radical_inverse(2, 7), 0.875));
    }

    #[test]
    fn radical_inverse_base_3_canonical() {
        assert!(close(radical_inverse(3, 1), 1.0 / 3.0));
        assert!(close(radical_inverse(3, 2), 2.0 / 3.0));
        assert!(close(radical_inverse(3, 3), 1.0 / 9.0));
        assert!(close(radical_inverse(3, 4), 4.0 / 9.0));
    }

    #[test]
    fn sobol_dim_0_matches_van_der_corput() {
        // Without scramble, Sobol dim 0 IS van der Corput in base 2.
        for n in 1u32..32 {
            let s = u32_to_unit_float(sobol_1d_raw(0, n));
            let r = radical_inverse(2, n);
            assert!((s - r).abs() < 1e-6, "n={n} sobol={s} radical_inverse={r}",);
        }
    }

    #[test]
    fn sobol_dim_1_first_points() {
        // Dim 1 with polynomial x+1, m_1 = 1.
        // First few Sobol-dim-1 fractions: 0.5, 0.75, 0.25, 0.625.
        let s = |n| u32_to_unit_float(sobol_1d_raw(1, n));
        assert!(close(s(1), 0.5));
        assert!(close(s(2), 0.75));
        assert!(close(s(3), 0.25));
        assert!(close(s(4), 0.625));
    }

    #[test]
    fn sobol_directions_first_vectors() {
        // Pinning a few values keeps a refactor of the direction-vector
        // table from silently changing the sequence.
        assert_eq!(SOBOL_DIRECTIONS[0][0], 0x80000000);
        assert_eq!(SOBOL_DIRECTIONS[0][1], 0x40000000);
        assert_eq!(SOBOL_DIRECTIONS[0][31], 0x00000001);
        assert_eq!(SOBOL_DIRECTIONS[1][0], 0x80000000);
        assert_eq!(SOBOL_DIRECTIONS[1][1], 0xC0000000);
        assert_eq!(SOBOL_DIRECTIONS[1][2], 0xA0000000);
        assert_eq!(SOBOL_DIRECTIONS[1][3], 0xF0000000);
    }

    #[test]
    fn pcg_is_deterministic() {
        let mut a = Pcg::new(42);
        let mut b = Pcg::new(42);
        assert_eq!(a.next_2d(), b.next_2d());
        assert_eq!(a.next_2d(), b.next_2d());
    }

    #[test]
    fn pcg_outputs_are_in_unit_square() {
        let mut s = Pcg::new(1);
        for _ in 0..1024 {
            let [u, v] = s.next_2d();
            assert!((0.0..1.0).contains(&u), "u out of range: {u}");
            assert!((0.0..1.0).contains(&v), "v out of range: {v}");
        }
    }

    #[test]
    fn halton_points_are_in_unit_square() {
        let mut s = Halton::new(0, 7);
        for _ in 0..512 {
            let [u, v] = s.next_2d();
            assert!((0.0..1.0).contains(&u));
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn sobol_points_are_in_unit_square() {
        let mut s = Sobol::new(0, 7);
        for _ in 0..1024 {
            let [u, v] = s.next_2d();
            assert!((0.0..1.0).contains(&u));
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn sobol_mean_is_near_one_half() {
        // 2-D Sobol over many samples is unbiased; its empirical mean
        // converges to the analytic 1/2 much faster than i.i.d. sampling.
        let mut s = Sobol::new(0, 0);
        let n = 4096;
        let mut sum = [0.0f64; 2];
        for _ in 0..n {
            let [u, v] = s.next_2d();
            sum[0] += u as f64;
            sum[1] += v as f64;
        }
        let mu = [sum[0] / n as f64, sum[1] / n as f64];
        // i.i.d. would need O(1/sqrt(N)); a QMC bound is much tighter.
        assert!((mu[0] - 0.5).abs() < 5e-3, "mean_x={}", mu[0]);
        assert!((mu[1] - 0.5).abs() < 5e-3, "mean_y={}", mu[1]);
    }

    #[test]
    fn kind_from_str_is_case_insensitive() {
        assert_eq!("pcg".parse::<SamplerKind>().unwrap(), SamplerKind::Pcg);
        assert_eq!("PCG".parse::<SamplerKind>().unwrap(), SamplerKind::Pcg);
        assert_eq!(
            "Halton".parse::<SamplerKind>().unwrap(),
            SamplerKind::Halton
        );
        assert_eq!("sobol".parse::<SamplerKind>().unwrap(), SamplerKind::Sobol);
        assert!("xxx".parse::<SamplerKind>().is_err());
    }

    #[test]
    fn kind_discriminants_match_wgsl_constants() {
        // SAMPLER_PCG / SAMPLER_HALTON / SAMPLER_SOBOL in pathtrace.wgsl
        // must match these values.
        assert_eq!(SamplerKind::Pcg.as_u32(), 0);
        assert_eq!(SamplerKind::Halton.as_u32(), 1);
        assert_eq!(SamplerKind::Sobol.as_u32(), 2);
    }
}
