//! PT-sky/perf-measure (plan 0030): wall-clock benchmark for the
//! sky bake + env-CDF build pipeline at the two resolutions the
//! widget milestone cares about.
//!
//! For each resolution, measures:
//! * `sky::bake_equirect` — single-threaded CPU bake of the
//!   analytic sky into an equirect HDR pixel buffer.
//! * `env::ImportanceTables::build` — luminance × sin θ CDF build
//!   from the baked pixel buffer (the same CDF the path-tracer's
//!   NEE-on-env branch consumes).
//! * Combined bake + build — the "end-to-end CPU re-bake" cost
//!   the widget pays on a slider release (excluding GPU upload
//!   and accumulation reset, which live on the wgpu side).
//!
//! Reports min/median/max wall-clock over a warm-up + N=5 repeats.
//!
//! ## Why this lives in `examples/`, not in `cargo bench`
//!
//! Quasi has no `criterion` dependency and the project's
//! one-binary-per-measurement pattern (see `examples/gen_*.rs`)
//! works fine for a low-frequency latency check like this.
//! Running it is a single command (`cargo run --example
//! bench_sky_bake --release`) and the numbers print to stdout in
//! a stable plan-friendly format that pastes directly into
//! `plans/0030-pt-sky.md`'s Findings.
//!
//! ## Stub-data caveat
//!
//! Until `scripts/sky/fetch_hosek_data.py` runs and replaces the
//! zero-coefficient table stubs with the real Hosek-Wilkie 2012
//! data, `bake_equirect` produces a black equirect. The bake's
//! cost is **data-independent** — the per-pixel Bezier eval +
//! lerp do the same arithmetic regardless of whether the inputs
//! are zeros or real coefficients — so the timings here are
//! representative of the post-vendor cost. The CDF-build cost,
//! by contrast, scans pixel luminances; on an all-black equirect
//! the inner loop still runs but the row-sum accumulation
//! degenerates to zeros. The CDF build's cost is also data-
//! independent (per-pixel sin θ + RGB→Y arithmetic happens
//! unconditionally), so the timings stay representative.
//!
//! ## Safari / wasm numbers
//!
//! The plan specifies "Safari at 512×256 and 1024×512" as the
//! decision point for the widget bake resolution. Compiling this
//! same example to wasm32 and running it from a JS harness via
//! `performance.now()` is the natural extension; PT-sky/widget
//! will land that harness as part of the slider plumbing. Until
//! then, the native numbers establish a per-pixel lower-bound
//! that the browser-JIT'd wasm typically stays within ~3-5× of
//! for hot scalar loops.

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
use quasi::pathtrace::env::{EnvironmentMap, ImportanceTables};
#[cfg(not(target_arch = "wasm32"))]
use quasi::pathtrace::sky::{bake_equirect, SkyParams};

#[cfg(not(target_arch = "wasm32"))]
const REPEATS: usize = 5;

#[cfg(not(target_arch = "wasm32"))]
fn time_once<R>(mut f: impl FnMut() -> R) -> (Duration, R) {
    let t0 = Instant::now();
    let r = f();
    (t0.elapsed(), r)
}

#[cfg(not(target_arch = "wasm32"))]
fn stats(samples: &mut [Duration]) -> (Duration, Duration, Duration) {
    samples.sort();
    let min = samples[0];
    let median = samples[samples.len() / 2];
    let max = samples[samples.len() - 1];
    (min, median, max)
}

#[cfg(not(target_arch = "wasm32"))]
fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_at(width: u32, height: u32, params: &SkyParams) {
    // Warm-up run — pulls the code into i-cache and the table
    // data into d-cache so the timed runs are representative of
    // a hot-cache widget re-bake.
    let _ = bake_equirect(width, height, params);

    let mut bake_samples = vec![Duration::ZERO; REPEATS];
    let mut build_samples = vec![Duration::ZERO; REPEATS];
    let mut combined_samples = vec![Duration::ZERO; REPEATS];

    for i in 0..REPEATS {
        let (bake_dur, pixels) = time_once(|| bake_equirect(width, height, params));
        let env = EnvironmentMap::new(width, height, pixels);
        let (build_dur, _) = time_once(|| ImportanceTables::build(&env));
        bake_samples[i] = bake_dur;
        build_samples[i] = build_dur;
        combined_samples[i] = bake_dur + build_dur;
    }

    let (bake_min, bake_med, bake_max) = stats(&mut bake_samples);
    let (build_min, build_med, build_max) = stats(&mut build_samples);
    let (comb_min, comb_med, comb_max) = stats(&mut combined_samples);

    println!(
        "| {w:4}×{h:<4} | bake        | {bmin:7.2} | {bmed:7.2} | {bmax:7.2} |",
        w = width,
        h = height,
        bmin = ms(bake_min),
        bmed = ms(bake_med),
        bmax = ms(bake_max),
    );
    println!(
        "| {w:4}×{h:<4} | CDF build   | {bmin:7.2} | {bmed:7.2} | {bmax:7.2} |",
        w = width,
        h = height,
        bmin = ms(build_min),
        bmed = ms(build_med),
        bmax = ms(build_max),
    );
    println!(
        "| {w:4}×{h:<4} | bake + CDF  | {cmin:7.2} | {cmed:7.2} | {cmax:7.2} |",
        w = width,
        h = height,
        cmin = ms(comb_min),
        cmed = ms(comb_med),
        cmax = ms(comb_max),
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    // Plan-prescribed noon params (see PT-sky/time-of-day):
    // elevation 75°, azimuth 180°, turbidity 2.5, default
    // ground albedo. Cost is direction-and-turbidity-invariant
    // (the per-pixel Bezier eval doesn't branch on values) but
    // pin a representative configuration so the table is
    // reproducible.
    let elev = 75.0_f32.to_radians();
    let azi = 180.0_f32.to_radians();
    let (sin_elev, cos_elev) = elev.sin_cos();
    let (sin_azi, cos_azi) = azi.sin_cos();
    let params = SkyParams {
        sun_dir: [cos_elev * cos_azi, sin_elev, cos_elev * sin_azi],
        turbidity: 2.5,
        ground_albedo: [0.3, 0.3, 0.3],
    };

    println!("PT-sky/perf-measure (plan 0030) — native, single-threaded");
    println!("REPEATS = {REPEATS} (plus one warm-up, discarded)");
    println!();
    println!("| resolution | stage       | min(ms) | med(ms) | max(ms) |");
    println!("|------------|-------------|---------|---------|---------|");
    for &(w, h) in &[(512_u32, 256_u32), (1024, 512)] {
        bench_at(w, h, &params);
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // The wasm path would call `performance.now()` from a JS
    // shim; building this example to wasm32 is not part of the
    // current run flow. PT-sky/widget will pick up the browser-
    // side measurement.
}
