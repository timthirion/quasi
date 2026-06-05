# AGENTS.md

Guidance for AI agents working on this repository.

## Project Goal

Quasi is a high-quality global illumination renderer. The end goal is publishable
output — polished technical blog posts and, ideally, novel research — so the work
prioritizes physical correctness, measurability (convergence / variance /
MSE-vs-reference), and modern-API techniques over breadth of features.

This repository is the **Rust** implementation. Its distinctive purpose: run in
the browser via WebAssembly (`wasm-pack`) so blog posts can embed **interactive,
live renders** — orbit the camera, flip the integrator, watch convergence — not
just static images. It must therefore build for both native (desktop) and web
(wasm) from one codebase.

See `plans/ROADMAP.md` for direction and `plans/` for current, machine-portable
plans.

## Tech Stack

- **Language:** Rust (edition 2021).
- **GPU:** [`wgpu`](https://wgpu.rs) — the renderer codes against **one API
  surface: WebGPU**, and one shading language: WGSL. `wgpu` *implements* WebGPU and
  maps it down to a native backend (Metal/Vulkan/DX12) automatically, or to WebGPU
  (with a WebGL2 fallback) in the browser.
- **Shaders:** WGSL (wgpu's native language; runs unmodified in the browser).
- **Windowing/input:** `winit` (supports a native window and an HTML canvas).
- **Async:** `pollster` to block on `wgpu` futures natively; `wasm-bindgen-futures`
  on web.
- **Web packaging:** `wasm-bindgen` + `wasm-pack`, with a minimal HTML/JS harness
  for embedding in posts.
- **Image I/O (native):** `image` (PNG) and `exr` (HDR EXR) for output and for the
  verification harness.

## Scope: single API, no backend abstraction

This is a deliberate divergence worth stating plainly: **Quasi (Rust) is
WebGPU-only at the API level, and has no GPU-backend abstraction layer.** We write
WebGPU/WGSL once and let `wgpu` choose the native backend.

- This is *not* "Metal-only" or "Vulkan-only" — running natively, `wgpu` still
  talks to Metal under the hood on macOS, Vulkan on Linux, etc. We just never
  write those APIs; the abstraction is `wgpu`'s job, not ours.
- We do **not** build a pluggable multi-backend system. Targeting a single API
  with a single shading language is precisely what makes the same source drop into
  a blog post as a WebAssembly widget — that one-source-to-browser story is the
  reason this implementation exists, and a backend abstraction would work against
  it.

(For contrast: a separate native implementation of Quasi is free to be
backend-agnostic and write Metal/Vulkan/etc. directly. This repo is not.)

## Use the language

A secondary goal: exercise the breadth of Rust. When a design has multiple
reasonable shapes and one of them puts `async`, parallelism (`rayon`, channels),
traits, type-state, or lifetimes to genuine use, prefer that shape. This is a
long-term project and a learning surface as well as a research vehicle; using
the language well is part of the point.

Concretely: design CPU-side systems around async / parallel APIs where they're
a natural fit — async asset and scene loaders, parallel acceleration-structure
builds, worker pools for the convergence / verification harness, channel-driven
pipelines — but **don't fake it**. A fundamentally sequential or GPU-bound
stage stays sequential. Architectural fit first; language breadth second.

This is CPU-side guidance. GPU work stays on the single WebGPU surface per the
section above.

## Build & Run

```bash
# Native
cargo run                      # desktop window
cargo test                     # unit tests (metrics, samplers, ...)
cargo clippy --all-targets     # lint
cargo fmt                      # format

# Web (once the wasm entry point exists)
wasm-pack build --target web   # produces pkg/ for the HTML harness
```

Keep the native and web builds working in lockstep — a change that only compiles
natively is half-done. Guard platform-specific code with `#[cfg(target_arch =
"wasm32")]` / `#[cfg(not(target_arch = "wasm32"))]`.

## Architecture (intended)

- A core renderer crate that owns the `wgpu` device/queue, scene, and the WGSL
  path-tracing pipeline; platform-agnostic.
- Thin native and web entry points (winit window vs. canvas) that drive the core.
- Path tracing as a WGSL megakernel rendering to an HDR texture, progressively
  accumulated across frames (ping-pong textures), then tonemapped to the surface.
- Verification (metrics, convergence) lives in native-only code and tests.
- Asset-pipeline utilities that need a Python toolchain (currently the OpenVDB
  `.vdb` → `.qvg` converter) live under `scripts/` at the repo root, outside
  the Rust crate so the build stays portable. See `scripts/README.md`.

## Coding Style

- `rustfmt` defaults; keep `cargo clippy` clean (no warnings).
- snake_case items, CamelCase types, SCREAMING_SNAKE_CASE consts.
- Errors via `Result` with a typed error enum (`thiserror`); avoid `unwrap()`
  outside tests and clearly-infallible setup.
- Document public items with `///`; module overviews with `//!`.
- One responsibility per module; keep WGSL shaders in their own `.wgsl` files
  (include via `include_str!`) rather than inline string literals.
- All features must have automated tests where they can run off-GPU (sampler
  sequences, image metrics, scene math).

## Testing

Automated tests are a **first-class, non-negotiable** priority. A strong test
suite is part of how this project earns the right to publish quality claims —
convergence numbers, BSDF correctness, MSE-vs-reference — and the bar applies
to every change.

**Rules of the road:**
- **No new module without tests.** Land code and its tests in the same change.
  A PR that bumps a module without exercising its public surface is incomplete.
- **No drift from green.** `cargo test` (native) and `cargo check --target
  wasm32-unknown-unknown` both stay green at every commit. `cargo clippy
  --all-targets -- -D warnings` clean too.
- **Test what you can; document what you can't.** GPU pipeline / binding
  validation needs hardware; say so explicitly when reporting work as done,
  and prefer landing a CPU-runnable regression alongside (e.g. an RMSE-vs-
  reference metric over a known scene).

**Categories that have an obligatory test, in priority order:**
1. **CPU↔GPU struct layout** — every uniform/storage struct used by WGSL gets
   a `size_of` / `offset_of` assertion (e.g. `vertex_is_32_bytes`). This
   class of bug — `vec3` forcing 16-byte alignment, scalar pads in the wrong
   place — fails only at runtime ("buffer too small") otherwise; pin it with
   a test.
2. **Cross-language constants** — when a Rust enum's discriminant is read by
   WGSL (`SamplerKind`, `IntegratorKind`, `AOV_*`), pin both sides:
   `as_u32()` returns the expected number AND the WGSL source literally
   contains `const NAME: u32 = N;` (see `tests/shaders.rs`).
3. **WGSL parses + validates** — every `.wgsl` file is covered by a naga
   validation test in `tests/shaders.rs`. The M1 `from` / `target` reserved-
   keyword save is what these earn.
4. **Pure math, samplers, metrics, transforms** — these are inherently
   testable; cover them directly, including canonical reference values
   (van der Corput, Halton bases 2/3, Sobol dim 0/1, identity / rotation
   transforms, identical-images = 0, black-reference rel-MSE = `1/ε`).
5. **File-format round-trips** — EXR round-trip, PNG header check, glTF
   round-trip (vertex / material / emissive counts). External format
   compat regressions get caught at `cargo test` instead of at "load the
   bunny" time.
6. **Error paths** — every typed error variant gets a test that triggers it
   (`MeshError::NoNormals`, size-mismatch panics in metrics).

**For GPU work that genuinely can't run headlessly:**
- Provide a CPU-runnable regression where any is feasible (e.g. the M3
  convergence runner scores RMSE against a high-spp reference — that
  works as a render regression at any time).
- Land a smoke command (`cargo run -- render --spp 32 …`) and reference
  it in the plan's "Done when" so a future contributor knows what to run.
- Note explicitly in the PR / commit / plan update what remained
  manual — never claim success on something the test suite didn't see.

**Symptoms of suite rot to watch for:**
- A new file lands with zero tests "because it's just glue". Glue has bugs.
- Tests are commented out or marked `#[ignore]` "for now".
- A failing test gets its assertion loosened instead of the underlying
  behaviour fixed.
- The cross-language constant-pinning test starts skipping new variants.
If any of these surface, the right move is to stop and refit the test
suite before adding more surface area.

## Git Workflow

- Solo repo: commit directly to `main` and push freely.
- End commit messages with the standard Co-Authored-By trailer.
- Use `git mv` for moves/renames to preserve history.
