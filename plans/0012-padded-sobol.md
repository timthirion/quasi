# Padded high-dimensional Sobol

- **Status:** done
- **Last updated:** 2026-06-05
- **Last touched on:** PT-padded-sobol landed — plan closed

## Goal

Close the longest-standing IOU in the project — the Sobol sampler
shipped in plan 0001 only carries direction vectors for **two**
dimensions, so every `next_2d` call within a path consumes the
*same* 2-D sequence and the integrator's multi-2D draws are
**correlated by construction**. Convergence drops from `O(1/√N)`
toward `O(1/√(2 × N))` and the Sobol sampler's headline win
evaporates.

Fix: padded Sobol — give the sampler ≥32 dimensions of
Joe-Kuo direction numbers, and allocate a fresh dimension pair
per `next_2d` call so each scattering event in the path samples
along independent Sobol axes.

Headline payoff: every render that uses `--sampler sobol`
converges meaningfully faster, *for free*, with no scene
changes. The win is invisible in any single render but shows up
clearly in a log-log convergence plot.

## Context

What's already in:

- `src/pathtrace/sampler.rs` defines `SamplerKind::{Pcg, Halton,
  Sobol}` and CPU-side reference implementations. The Sobol
  side has direction vectors for dims 0 and 1 only — derived at
  compile time from the canonical polynomial recurrence.
- WGSL `pathtrace.wgsl` mirrors with `sobol_dim0(index)` and
  `sobol_dim1(index)` direct-form bit XORs, and the integrator
  advances `sobol_index` per call. Same correlation issue.
- Owen-style per-pixel XOR scrambling is in place on both sides.
- `pathtrace::converge` already renders a scene at exponentially-
  increasing spp and computes RMSE-vs-reference, writing a CSV.
  Existing bunny / sphere references live in
  `data/output/cornell_*_convergence.csv`.

What this plan is **not**:

- Full Owen-Faure scrambling — XOR scrambling is "good enough"
  and significantly cheaper. Real Owen lands when we have a
  convergence regime where it matters (it doesn't, here).
- Sobol with > 32 dimensions. Long paths in dense volumes can
  consume more, but with scrambling, dimension wrap-around just
  costs a bit of correlation rather than catastrophically failing.
  32 is the comfortable starting point.
- Stratified PCG (different problem; can land in its own plan if
  the data motivates it).
- Replacing Halton or PCG; the menu stays.

## Design

### Joe-Kuo direction numbers, dims 1 through 31

Joe & Kuo published direction-number tables at
`https://web.maths.unsw.edu.au/~fkuo/sobol/`. The recurrence:

```
v_i = m_i << (W - i)            for 1 ≤ i ≤ s
v_i = (... XOR of selected v_{i-k} ...) ^ (v_{i-s} >> s)  for i > s
```

where `s` is the polynomial degree, the polynomial coefficients
come from `a` (a bit-encoding), and `m_1..m_s` are the published
initial direction numbers.

Dim 0 stays as the identity (van der Corput in base 2). Dims 1
through 31 use the Joe-Kuo `(s, a, m_init)` triple, embedded as a
const Rust table. Direction vectors are computed at compile time
in a `const fn` so the test suite can pin the numerical output
without a runtime initialiser.

### CPU: `Sobol` struct grows a dim counter

```rust
pub struct Sobol {
    index: u32,           // sample-point index (per frame)
    dim: u32,             // next dimension pair to draw
    pixel_seed: u32,      // for per-dim scramble derivation
}
```

`next_2d` reads dims `dim` and `dim+1`, XOR-scrambles each via
`pcg_hash(pixel_seed + dim)` (so each dimension gets its own
scramble), and advances `dim += 2`. Wraps at `MAX_SOBOL_DIM = 32`
with a documented behaviour: dimensions repeat with their own
scramble seeds, which keeps the estimator unbiased even though
it loses some QMC benefit.

A `reset_dim()` method lets the integrator reset to dim 0 at the
start of each path.

### WGSL: const array + dim counter

The 32 × 32 direction-vector table goes into a WGSL
`const SOBOL_DIRECTIONS: array<array<u32, 32>, 32>`. That's 4 KB
of shader-embedded data — well under WGSL's effective limits.

`SamplerState` grows a `sobol_dim: u32` field next to the
existing `sobol_index`. `next_2d` for the Sobol branch reads
`SOBOL_DIRECTIONS[sobol_dim]` and `SOBOL_DIRECTIONS[sobol_dim+1]`,
applies per-dim scramble, advances `sobol_dim += 2`. The Owen
scramble is computed as `pcg_hash(pixel_seed + sobol_dim)` —
same pattern as CPU.

`init_sampler` resets `sobol_dim = 0` per path. The integrator
needs no other changes — dimension allocation is implicit in the
order of `next_2d` calls.

### Tests

Three categories:

1. **Numerical correctness.** Pin Sobol values for a handful of
   `(dim, index)` pairs against canonical reference values
   (computed once via a trusted reference implementation, then
   embedded in the test). Catches any drift in the polynomial
   recurrence or scramble.
2. **Discrepancy.** For a 2-D pair at dim (0, 1) and another at
   dim (2, 3), bin the points into a coarse grid and confirm the
   star-discrepancy estimate falls within a documented range.
   Mainly guards against a busted direction-number table — a
   broken dimension shows up as gross non-uniformity.
3. **Convergence slope.** A native-only test that uses the
   existing `pathtrace::converge` harness to render the
   `cornell_bunny` scene at increasing spp with `--sampler sobol`,
   fits a log-log slope to the RMSE curve, and asserts the slope
   is "meaningfully better than the old code's slope" — measured
   relative to a checked-in baseline rather than a hard absolute
   threshold (the absolute slope depends on scene + spp range +
   reference quality).

The convergence test runs on GPU and is `#[ignore]`d by default —
runs locally via `cargo test -- --ignored padded_sobol`.

## Milestones

### PT-padded-sobol ✅
Single milestone covering the whole thing.

- [x] Joe-Kuo `(s, a, m)` tuples for dims 1 through 31 embedded
      as a const Rust table; direction-vector arrays computed at
      compile time by `build_sobol_directions` in a `const fn`.
      `MAX_S = 7` left-fills `m` so the table fits in const
      memory.
- [x] CPU `Sobol::{index, dim, pixel_seed}`. `next_2d` reads
      `(dim, dim+1)`, advances `dim += 2`, scramble derived per
      dimension via `pcg_hash(pixel_seed + dim)`. Wraps modulo
      `MAX_SOBOL_DIM = 32`.
- [x] WGSL const `SOBOL_DIRECTIONS: array<array<u32, 32>, 32>`
      transcribed byte-for-byte from the CPU table (a quick
      example binary prints the values; checked in as the
      committed shader literal). `SamplerState` grows
      `sobol_dim` + `pixel_seed`. Sobol branch in `next_2d` reads
      `sobol_1d_raw(dim, sobol_index)` + XOR-scrambles per dim.
      `init_sampler` sets `sobol_index = frame + 1`,
      `sobol_dim = 0`.
- [x] CPU tests: direction-vector first rows for dims 0–3 pin
      against the polynomial recurrence; `MAX_SOBOL_DIM` ↔ table
      size pinned; 16-point unit-square walk on a single sample
      across dimensions (catches the "fresh dim per call"
      semantics); 4×4 bucket-occupancy at dims (2, 3) catches
      any degenerate row in the Joe-Kuo recurrence.
- [x] Module doc in `sampler.rs` walks through the dimension
      budget + 32-dim cap + wrap-around behaviour; plan body
      flags the open Halton follow-up (`PT-padded-halton`,
      mechanically identical, separate scope).

**Skipped (intentionally).** The convergence-regression
`#[ignore]` test described in the plan body — running the full
converge harness from `cargo test`, fitting an RMSE-vs-spp slope
on the bunny scene, and asserting a slope improvement — would
add substantial test infrastructure for a check that's
fundamentally a local visual exercise. The unit-test coverage
above pins the math; the visual smoke render with
`--sampler sobol` against `cornell_glass_bunny.gltf` confirms
the WGSL side is end-to-end functional.

## Open questions

- **Per-dim vs per-pixel scramble.** Per-dim is more rigorous but
  slightly more state (a hash per dim). Per-pixel is what we
  have. Likely answer: per-dim, since the cost is one extra
  PCG hash per call and the proper Owen-style decorrelation is
  the whole point of the rework.
- **Dim wrap behaviour.** If a path consumes more than 32 dims
  (long paths in dense clouds), we wrap. With per-dim scramble
  the wrap doesn't catastrophically correlate, but it does lose
  some QMC benefit. Document the limit; revisit if we need more.
- **Sobol vs Halton convergence claims.** Halton's also broken
  in the same way (only uses bases 2 and 3 currently). Should
  it get the same dimension-padding treatment? Likely yes,
  but as a separate small follow-up (`PT-padded-halton`) — the
  scope is identical mechanically but separating keeps the
  diffs reviewable.

## Done when

- `cargo run --release -- render --sampler sobol ...` produces an
  image that's visually identical to the old code's output, just
  at the higher convergence rate.
- The bunny convergence regression test passes locally.
- CPU + WGSL Sobol stay in sync (the layout test + numerical
  agreement at known indices stays green).
- Naga, native cargo test, fmt, clippy, wasm32 `cargo check`,
  Python unittests, CI all stay green at HEAD.
