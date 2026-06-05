//! Samplers — PCG, Halton, and Sobol.
//!
//! Each sampler produces a stream of 2-D points in `[0, 1)^2`. The same
//! mathematical definitions are mirrored in `shaders/pathtrace.wgsl`: the
//! CPU implementations here exist so the sequences can be tested against
//! canonical reference values off-GPU (per `AGENTS.md` testing guidance).
//!
//! Two QMC families are offered alongside the workhorse PRNG:
//!
//! - **Halton.** Uses prime bases (2, 3, 5, 7, …). Each `next_2d` call
//!   advances the dimension counter by 2, so multi-2D draws within one
//!   path consume *independent* Halton dimensions rather than re-reading
//!   the same `(base 2, base 3)` pair. A per-pixel Cranley–Patterson
//!   rotation decorrelates neighbours.
//! - **Sobol.** Joe-Kuo direction numbers for dimensions 0 through 31
//!   ([Joe & Kuo's published table](https://web.maths.unsw.edu.au/~fkuo/sobol/)
//!   `new-joe-kuo-6.21201.txt`). `next_2d` advances a dimension counter
//!   the same way Halton does, so each `(dim, dim + 1)` pair samples
//!   independent Sobol axes — this is the "padded Sobol" pattern. A
//!   per-dimension XOR scramble (`pcg_hash(pixel_seed + dim)`) makes the
//!   wrap-around past dim 31 cheap and unbiased rather than catastrophic.
//!
//! **Dimension budget.** The integrator draws roughly two 2-D points per
//! bounce — at the default `MAX_BOUNCES = 5` plus camera jitter plus
//! light-pick scratch, a path can consume ~36 dimensions. We size the
//! Sobol table at 32; paths that overrun fall back on the per-dim
//! scramble's decorrelation. Long paths in dense clouds may want
//! `MAX_SOBOL_DIM` bumped — easy when we have a scene that needs it.

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

    /// Short label used in CSV output and CLI args.
    pub fn label(self) -> &'static str {
        match self {
            SamplerKind::Pcg => "pcg",
            SamplerKind::Halton => "halton",
            SamplerKind::Sobol => "sobol",
        }
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
// Sobol — padded high-dimensional, Joe-Kuo direction numbers
// ---------------------------------------------------------------------------

/// Maximum Sobol dimension carried in the direction-number table. Paths
/// that consume more than this wrap modulo `MAX_SOBOL_DIM`; with per-dim
/// scrambling the wrap is unbiased — just a loss of QMC benefit on the
/// repeated dimensions. Mirrored as `MAX_SOBOL_DIM` in `pathtrace.wgsl`.
pub const MAX_SOBOL_DIM: usize = 32;

/// Maximum polynomial degree across the Joe-Kuo table for dims 1..32.
/// Used to fix-size the `m` initial-direction-number array so the table
/// can live in const memory.
const MAX_S: usize = 7;

/// One row of the Joe-Kuo direction-number table — polynomial degree
/// `s`, polynomial-coefficient encoding `a`, and the first `s` initial
/// direction numbers `m_1..m_s`. Entries with `s < MAX_S` left-fill
/// `m`; the recurrence ignores the rest.
struct DimInit {
    s: u32,
    a: u32,
    m: [u32; MAX_S],
}

/// Joe-Kuo direction numbers for dimensions 1 through 31 (Joe-Kuo
/// 1-indexed dims 2 through 32). Dim 0 is the identity polynomial
/// (van der Corput in base 2), generated separately. Transcribed from
/// the public `new-joe-kuo-6.21201.txt` table.
const SOBOL_DIM_INITS: [DimInit; MAX_SOBOL_DIM - 1] = [
    DimInit {
        s: 1,
        a: 0,
        m: [1, 0, 0, 0, 0, 0, 0],
    }, // dim 1 (J-K d=2)
    DimInit {
        s: 2,
        a: 1,
        m: [1, 3, 0, 0, 0, 0, 0],
    }, // dim 2
    DimInit {
        s: 3,
        a: 1,
        m: [1, 3, 1, 0, 0, 0, 0],
    }, // dim 3
    DimInit {
        s: 3,
        a: 2,
        m: [1, 1, 1, 0, 0, 0, 0],
    }, // dim 4
    DimInit {
        s: 4,
        a: 1,
        m: [1, 1, 3, 3, 0, 0, 0],
    }, // dim 5
    DimInit {
        s: 4,
        a: 4,
        m: [1, 3, 5, 13, 0, 0, 0],
    }, // dim 6
    DimInit {
        s: 5,
        a: 2,
        m: [1, 1, 5, 5, 17, 0, 0],
    }, // dim 7
    DimInit {
        s: 5,
        a: 4,
        m: [1, 1, 5, 5, 5, 0, 0],
    }, // dim 8
    DimInit {
        s: 5,
        a: 7,
        m: [1, 1, 7, 11, 19, 0, 0],
    }, // dim 9
    DimInit {
        s: 5,
        a: 11,
        m: [1, 1, 5, 1, 1, 0, 0],
    }, // dim 10
    DimInit {
        s: 5,
        a: 13,
        m: [1, 1, 1, 3, 11, 0, 0],
    }, // dim 11
    DimInit {
        s: 5,
        a: 14,
        m: [1, 3, 5, 5, 31, 0, 0],
    }, // dim 12
    DimInit {
        s: 6,
        a: 1,
        m: [1, 3, 3, 9, 7, 49, 0],
    }, // dim 13
    DimInit {
        s: 6,
        a: 13,
        m: [1, 1, 1, 15, 21, 21, 0],
    }, // dim 14
    DimInit {
        s: 6,
        a: 16,
        m: [1, 3, 1, 13, 27, 49, 0],
    }, // dim 15
    DimInit {
        s: 6,
        a: 19,
        m: [1, 1, 1, 15, 7, 5, 0],
    }, // dim 16
    DimInit {
        s: 6,
        a: 22,
        m: [1, 3, 1, 15, 13, 25, 0],
    }, // dim 17
    DimInit {
        s: 6,
        a: 25,
        m: [1, 1, 5, 5, 19, 61, 0],
    }, // dim 18
    DimInit {
        s: 7,
        a: 1,
        m: [1, 3, 7, 11, 23, 15, 103],
    }, // dim 19
    DimInit {
        s: 7,
        a: 4,
        m: [1, 3, 7, 13, 13, 15, 69],
    }, // dim 20
    DimInit {
        s: 7,
        a: 7,
        m: [1, 1, 3, 13, 7, 35, 63],
    }, // dim 21
    DimInit {
        s: 7,
        a: 8,
        m: [1, 3, 5, 9, 1, 25, 53],
    }, // dim 22
    DimInit {
        s: 7,
        a: 14,
        m: [1, 3, 1, 13, 9, 35, 107],
    }, // dim 23
    DimInit {
        s: 7,
        a: 19,
        m: [1, 3, 1, 5, 27, 61, 31],
    }, // dim 24
    DimInit {
        s: 7,
        a: 21,
        m: [1, 1, 5, 11, 19, 41, 61],
    }, // dim 25
    DimInit {
        s: 7,
        a: 28,
        m: [1, 3, 5, 3, 3, 13, 69],
    }, // dim 26
    DimInit {
        s: 7,
        a: 31,
        m: [1, 1, 5, 13, 21, 15, 61],
    }, // dim 27
    DimInit {
        s: 7,
        a: 32,
        m: [1, 3, 1, 15, 5, 49, 119],
    }, // dim 28
    DimInit {
        s: 7,
        a: 37,
        m: [1, 1, 3, 15, 17, 19, 61],
    }, // dim 29
    DimInit {
        s: 7,
        a: 41,
        m: [1, 3, 1, 3, 13, 59, 57],
    }, // dim 30
    DimInit {
        s: 7,
        a: 42,
        m: [1, 3, 3, 3, 25, 31, 113],
    }, // dim 31
];

/// Direction vectors `v_i` for every Sobol dimension we carry. Computed
/// at compile time so the table is just baked into the binary.
pub const SOBOL_DIRECTIONS: [[u32; 32]; MAX_SOBOL_DIM] = build_sobol_directions();

const fn build_sobol_directions() -> [[u32; 32]; MAX_SOBOL_DIM] {
    let mut t = [[0u32; 32]; MAX_SOBOL_DIM];
    // Dim 0: identity polynomial → `v_i = 1 << (W - i)`. Equivalent to
    // van der Corput in base 2.
    let mut i = 0;
    while i < 32 {
        t[0][i] = 1u32 << (31 - i);
        i += 1;
    }
    // Dims 1..MAX_SOBOL_DIM: Joe-Kuo recurrence over the `(s, a, m)`
    // tuples above.
    let mut d = 1;
    while d < MAX_SOBOL_DIM {
        let init = &SOBOL_DIM_INITS[d - 1];
        let s = init.s as usize;
        let a = init.a;
        // First `s` direction numbers come straight from `m`.
        let mut i = 0;
        while i < s {
            t[d][i] = init.m[i] << (31 - i);
            i += 1;
        }
        // For i > s, apply Joe-Kuo's recurrence:
        //   v_i = v_{i-s} ^ (v_{i-s} >> s)
        //         ^ XOR over k = 1..s-1 of ((a >> (s-1-k)) & 1) * v_{i-k}
        while i < 32 {
            let prev = t[d][i - s];
            let mut new_v = prev ^ (prev >> s as u32);
            let mut k: usize = 1;
            while k < s {
                let bit = (a >> ((s - 1 - k) as u32)) & 1;
                if bit == 1 {
                    new_v ^= t[d][i - k];
                }
                k += 1;
            }
            t[d][i] = new_v;
            i += 1;
        }
        d += 1;
    }
    t
}

/// Raw 1-D Sobol value as a `u32` (top bits are the most-significant
/// fractional bits). Bit-by-bit XOR of selected direction vectors. The
/// `dim` index wraps modulo `MAX_SOBOL_DIM`.
pub fn sobol_1d_raw(dim: usize, index: u32) -> u32 {
    let dirs = &SOBOL_DIRECTIONS[dim % MAX_SOBOL_DIM];
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

/// 2-D Sobol point at dimensions `(dim, dim + 1)`, with per-dimension
/// XOR scrambles. The scramble seeds typically come from
/// `pcg_hash(pixel_seed + dim)` so adjacent pixels and adjacent
/// dimensions both decorrelate.
pub fn sobol_2d(dim: u32, index: u32, scramble: [u32; 2]) -> [f32; 2] {
    let x = sobol_1d_raw(dim as usize, index) ^ scramble[0];
    let y = sobol_1d_raw((dim + 1) as usize, index) ^ scramble[1];
    [u32_to_unit_float(x), u32_to_unit_float(y)]
}

/// Padded 2-D Sobol sampler. Each `next_2d` reads a fresh `(dim, dim+1)`
/// pair, so the integrator's multi-2D draws within one path consume
/// independent Sobol axes rather than re-reading dims 0–1.
#[derive(Clone, Debug)]
pub struct Sobol {
    index: u32,
    dim: u32,
    pixel_seed: u32,
}

impl Sobol {
    /// `index` is the sample-point index (typically the frame number);
    /// `pixel_seed` feeds the per-dimension scramble hash.
    pub fn new(index: u32, pixel_seed: u32) -> Self {
        Sobol {
            index,
            dim: 0,
            pixel_seed,
        }
    }
}

impl Sampler for Sobol {
    fn next_2d(&mut self) -> [f32; 2] {
        let dim = self.dim;
        let scramble = [
            pcg_hash(self.pixel_seed.wrapping_add(dim)),
            pcg_hash(self.pixel_seed.wrapping_add(dim.wrapping_add(1))),
        ];
        let p = sobol_2d(dim, self.index, scramble);
        self.dim = self.dim.wrapping_add(2);
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
        // table from silently changing the sequence — and pins the
        // exact bytes that get copied into the WGSL const table.
        assert_eq!(SOBOL_DIRECTIONS[0][0], 0x80000000);
        assert_eq!(SOBOL_DIRECTIONS[0][1], 0x40000000);
        assert_eq!(SOBOL_DIRECTIONS[0][31], 0x00000001);
        assert_eq!(SOBOL_DIRECTIONS[1][0], 0x80000000);
        assert_eq!(SOBOL_DIRECTIONS[1][1], 0xC0000000);
        assert_eq!(SOBOL_DIRECTIONS[1][2], 0xA0000000);
        assert_eq!(SOBOL_DIRECTIONS[1][3], 0xF0000000);
        // Joe-Kuo dims 2 & 3: first-row vectors come straight from the
        // `m_init` shifts so they're easy to pin without running the
        // recurrence.
        assert_eq!(SOBOL_DIRECTIONS[2][0], 0x80000000); // m_1 = 1 → 1 << 31
        assert_eq!(SOBOL_DIRECTIONS[2][1], 0xC0000000); // m_2 = 3 → 3 << 30
        assert_eq!(SOBOL_DIRECTIONS[3][0], 0x80000000);
        assert_eq!(SOBOL_DIRECTIONS[3][1], 0xC0000000);
        assert_eq!(SOBOL_DIRECTIONS[3][2], 0x20000000); // m_3 = 1 → 1 << 29
    }

    #[test]
    fn sobol_directions_table_size() {
        // Catches any drift between the `MAX_SOBOL_DIM` constant and
        // the actual table the CPU + WGSL share.
        assert_eq!(SOBOL_DIRECTIONS.len(), MAX_SOBOL_DIM);
        assert_eq!(MAX_SOBOL_DIM, 32);
    }

    #[test]
    fn sobol_high_dim_points_are_in_unit_square() {
        // Walk a path's worth of dimensions on a single sample point.
        // Pre-padded-Sobol this would call `sobol_2d(index, scramble)`
        // 16 times with the same `(dim 0, dim 1)` pair — the *whole*
        // bug this plan addresses. Post-fix, each call reads a fresh
        // dimension pair, so the points should still be in
        // `[0, 1)^2` but actually look uncorrelated.
        let mut s = Sobol::new(7, 42);
        for _ in 0..16 {
            let [u, v] = s.next_2d();
            assert!((0.0..1.0).contains(&u));
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn sobol_independent_dims_are_uncorrelated() {
        // Coarse 4×4 bucket-occupancy test: 1024 Sobol points at
        // dimensions (2, 3) should hit every cell at least once. The
        // pre-padded sampler would have failed this if dims (2, 3)
        // didn't exist; this guards against any high-dim table row
        // collapsing to a degenerate sequence.
        let mut counts = [[0u32; 4]; 4];
        for index in 0..1024_u32 {
            let scramble = [
                pcg_hash(7u32.wrapping_add(2)),
                pcg_hash(7u32.wrapping_add(3)),
            ];
            let [u, v] = sobol_2d(2, index, scramble);
            let bx = ((u * 4.0) as usize).min(3);
            let by = ((v * 4.0) as usize).min(3);
            counts[by][bx] += 1;
        }
        let mut zeros = 0;
        for row in &counts {
            for &c in row {
                if c == 0 {
                    zeros += 1;
                }
            }
        }
        assert_eq!(
            zeros, 0,
            "expected every 4x4 cell to be hit; counts = {counts:?}",
        );
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
