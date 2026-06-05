# quasi

[![CI](https://github.com/timthirion/quasi/actions/workflows/ci.yml/badge.svg)](https://github.com/timthirion/quasi/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust 2021](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2021/index.html)
[![WebGPU](https://img.shields.io/badge/runs_in-WebGPU-purple.svg)](https://wgpu.rs)

The **Rust** implementation of Quasi, a high-quality global illumination renderer.
It targets one API — WebGPU, via [`wgpu`](https://wgpu.rs) and WGSL — so the same
code runs natively and in the browser, letting blog posts embed live, interactive
renders. See [`plans/ROADMAP.md`](plans/ROADMAP.md) for direction and `AGENTS.md`
for the stack and conventions.

## Running

### Native

```bash
cargo run        # opens a window (Esc to quit)
```

### Web (WebGPU)

```bash
wasm-pack build --target web      # builds pkg/
python3 -m http.server            # serve the repo root
# open http://localhost:8000/ in a WebGPU-capable browser
```

Current state (plan 0001, M0): a fullscreen gradient renders in both targets,
proving the dual-target pipeline.
