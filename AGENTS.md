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

Automated tests are a first-class priority, not an afterthought. Aim for broad,
fast, deterministic coverage — heavy on unit tests — and land new code together
with tests for it in the same change.

Test everything that can run off-GPU/off-hardware:
- **Scene & geometry math** — Cornell Box construction, transforms, helpers.
- **CPU↔GPU struct layout** — assert `size_of`/`offset_of` for every uniform/buffer
  struct against the WGSL layout. This class of bug (e.g. `vec3` forcing 16-byte
  alignment) only fails at runtime otherwise; pin it with a test.
- **Sampler sequences**, **image metrics**, camera math — pure, so test directly.
- **WGSL shaders** — validate with `naga` in `cargo test` (`tests/shaders.rs`).

For anything that genuinely needs the GPU (pipeline/binding validation, real
renders), keep a headless validation path where feasible and note what remains
manual. `cargo test` should stay green and meaningful at every commit.

## Git Workflow

- Solo repo: commit directly to `main` and push freely.
- End commit messages with the standard Co-Authored-By trailer.
- Use `git mv` for moves/renames to preserve history.
