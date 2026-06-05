# VDB ingest (cloud asset pipeline)

- **Status:** done
- **Last updated:** 2026-06-05
- **Last touched on:** PT-vdb-ingest landed — plan closed

## Goal

Close the asset pipeline for `PT-vdb`: ingest real OpenVDB files
and render them as cloud volumes. Plan 0008 left the `.qvg` format
+ GPU upload + sampler in place; this plan adds (a) a Python
converter that turns `.vdb` files into `.qvg`, and (b) a renderer
CLI flag so a non-embedded grid can be loaded at runtime.

Once this lands, swapping the procedural cumulus for a real
production cloud (e.g. the Disney Clouds dataset's CC-licensed
Cloud A) is a one-line CLI invocation.

## Context

What's already in (as of plan `0008`):

- `pathtrace::grid::Grid3D` knows the `.qvg` format end-to-end:
  save, load, trilinear sample, bounds.
- `examples/gen_cloud.rs` bakes a procedural cumulus to
  `data/grids/cumulus_64.qvg`; that file is `include_bytes!`'d
  into the binary and uploaded as the default cloud grid.
- The path tracer's WGSL `cloud_density` reads from the 3-D
  texture without any knowledge of where the data came from.

What this plan is **not**:

- Sparse-grid representations (NanoVDB, OpenVDB's native sparse
  layout). The Python side reads the sparse VDB, but we always
  write dense `.qvg`. Sparse storage on the GPU is a separate
  plan if scenes get big enough to need it.
- Multi-grid scenes (more than one cloud in a scene). One grid
  per render; this matches the current bind-group layout.
- Wavelength-dependent density grids (e.g. coloured smoke). Density
  is scalar `R8`.
- Animated grids. One static grid per render.

## Design

### Repository layout — `scripts/`

A new top-level `scripts/` directory hosts Python utilities that
sit alongside (not inside) the Rust crate. Convention:

```
scripts/
  README.md          # install + usage
  vdb_to_qvg.py      # the converter
```

`scripts/` stays out of `Cargo.toml`; it's deliberately separate
from the build pipeline so the Rust crate stays portable to
machines without a Python toolchain.

### `vdb_to_qvg.py`

CLI:

```
python scripts/vdb_to_qvg.py INPUT.vdb OUTPUT.qvg \
    [--resolution N | --resolution X Y Z]   # default 64
    [--grid-name NAME]                       # default "density"
    [--bounds-min X Y Z]                     # default: VDB grid bbox min
    [--bounds-max X Y Z]                     # default: VDB grid bbox max
    [--normalize]                            # rescale max → 1.0
```

Reads a `.vdb` via `pyopenvdb`, resamples the selected scalar
grid into a dense `dims × dims × dims` array via OpenVDB's
built-in resampling (nearest or trilinear — TBD; trilinear is the
sensible default for visual quality), normalises into `[0, 1]`,
quantises to `R8`, and writes the QVG1 binary that matches the
Rust `Grid3D::load` layout byte-for-byte.

### Renderer CLI: `--cloud-grid <path>`

`render` and `converge` grow a `--cloud-grid <path>` option that
loads a `.qvg` from disk and uploads it instead of the embedded
default. Without the flag, behaviour is unchanged — the embedded
cumulus is used.

Implementation:

- `pathtrace::build_cloud_grid_texture` already loads from bytes;
  refactor to accept either the embedded bytes or a runtime
  `&[u8]`.
- `main.rs::parse_render_args` adds the flag; `main.rs::run_render`
  threads the optional path down into `offscreen::render_offscreen`.
- A small `pathtrace::grid::load_or_default(path)` helper handles
  the "file missing or invalid → fall back to procedural" case
  with a warning.

### Scene-description integration: deferred

Long-term, a glTF scene should be able to reference a `.qvg` file
itself (e.g. via `Material.extras.cloud_grid`). For this milestone
we keep it CLI-only — adds the visual payoff without scope creep.

## Milestones

### PT-vdb-ingest ✅
Single milestone covering script + CLI flag + docs.

- [x] `scripts/` directory created with `README.md` documenting
      both the brew+pip and the conda-forge `pyopenvdb` install
      paths, sources of CC-licensed VDB clouds (Disney Clouds,
      EmberGen), and the end-to-end conversion + render workflow.
- [x] `scripts/vdb_to_qvg.py`: argparse CLI; reads VDB via
      `pyopenvdb` (lazy-imported so `--help` works without it);
      uses `GridSampler.wsSample` for trilinear world-space
      resampling at the requested resolution; quantises to `R8`
      via `qvg_writer.encode_density` and writes through
      `qvg_writer.write_qvg_path`.
- [x] `scripts/qvg_writer.py`: pure-Python no-deps module that
      writes the QVG1 binary. Mirrors the Rust `Grid3D::save`
      output byte-for-byte. Defensive `ValueError`s on dim/voxel
      mismatches.
- [x] `scripts/test_qvg_writer.py`: 10 unittest assertions
      pinning exact bytes for a known input, length mismatch →
      error, zero-dim → error, encode-density clamping +
      normalisation, parent-dir creation in `write_qvg_path`.
- [x] `pathtrace::grid::from_bytes_or_empty` +
      `load_from_path_or_default` helpers for the runtime-load
      path. The latter `log::warn!`s + falls back to the embedded
      default on failure.
- [x] `render` CLI grows `--cloud-grid PATH`. `main.rs::run_render`
      loads via `Grid3D::load_from_path` (fatal-error on bad path
      — keeps the explicit failure visible during scene setup).
      `render_offscreen_with_grid` threads the optional grid down
      to `build_scene_buffers_with_grid` →
      `build_cloud_grid_texture_from`. Without the flag, behaviour
      is unchanged (embedded cumulus).
- [x] AGENTS.md gets a one-line pointer under "Architecture
      (intended)" so future tooling that needs Python lands in
      `scripts/`.
- [x] Cross-language byte-equality test: `tests/grid.rs` pins
      exact bytes for the same canonical input
      `scripts/test_qvg_writer.py` tests, so if either side's
      writer drifts the corresponding test fails first.

**Visual payoff.** Now a one-line CLI invocation:

```sh
python scripts/vdb_to_qvg.py disney_cloud.vdb data/grids/cloud.qvg
cargo run --release -- render \
    --scene data/gltf/cornell_cloud.gltf \
    --cloud-grid data/grids/cloud.qvg \
    --out data/output/cornell_real_cloud
```

We don't ship a real `.vdb` file in the repo — those are 50+ MB
each and live with their original CC-licensed publishers — but
the pipeline accepts any density grid that comes out of OpenVDB.

**Out of scope here:** an integration test that runs the full VDB
→ QVG → render pipeline. That needs `pyopenvdb` installed *and* a
real `.vdb` file; both are environmental requirements we don't
want CI to depend on. The pure-format test + manual smoke render
cover the practical case.

## Open questions

- **Resampling filter.** OpenVDB's resampling supports nearest,
  trilinear, and triquadratic. Trilinear is the default; document
  the trade-off briefly in `scripts/README.md`.
- **Auto-detect grid name.** "density" is the most common name
  but VDB files in the wild also use "ws_density", "scalar", or
  per-renderer conventions. Default to "density" with a clear
  error message + `--grid-name` override.
- **Resolution sweet spot.** 64³ = 256 KB is comfortable. 128³ =
  2 MB is still fine for in-binary embedding. Document both.

## Done when

- A user with `pyopenvdb` installed can run
  `python scripts/vdb_to_qvg.py cloud.vdb cloud.qvg` and get a
  valid `.qvg` that `cargo run --release -- render --cloud-grid
  cloud.qvg ...` accepts and renders.
- The pure-Python QVG writer ships with pinned tests.
- `scripts/README.md` documents install, conversion, and renderer
  invocation end-to-end.
- Existing render workflows (no `--cloud-grid` flag) are
  unchanged.
- Naga, full Rust unit test suite all green.
