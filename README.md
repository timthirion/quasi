# quasi

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
