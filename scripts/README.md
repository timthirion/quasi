# scripts/

Python utilities that sit alongside (not inside) the Rust crate.
Anything that needs an external Python dependency lives here so
the Rust build stays portable to machines without a Python
toolchain.

## Index

| File | Purpose |
| --- | --- |
| `qvg_writer.py` | Pure-Python writer for the `.qvg` density-grid format (no third-party deps). |
| `vdb_to_qvg.py` | Converts an OpenVDB `.vdb` file to `.qvg` via `pyopenvdb`. |
| `test_qvg_writer.py` | Pure-Python tests for `qvg_writer`. Run with `python -m unittest discover scripts -p 'test_*.py'`. |

## VDB ingest: end-to-end

The Rust path tracer's PT-vdb pipeline reads dense 3-D density
grids in our own `.qvg` ("quasi volume grid") format. Real-world
clouds typically ship as sparse OpenVDB `.vdb` files; `vdb_to_qvg.py`
resamples them to dense and quantises to `R8`.

### 1. Install `pyopenvdb`

OpenVDB has Python bindings that are notoriously hard to install
because the C++ library has to match the Python module's
`pybind11` revision. Two paths that tend to work:

**macOS (homebrew + system Python):**

```sh
brew install openvdb
pip install pyopenvdb
```

If `pip install` can't find the C++ headers, point it at homebrew:

```sh
CPLUS_INCLUDE_PATH="$(brew --prefix openvdb)/include" \
LIBRARY_PATH="$(brew --prefix openvdb)/lib" \
pip install pyopenvdb
```

**Cross-platform (recommended for CI / new machines):**

```sh
conda create -n quasi-vdb python=3.11
conda activate quasi-vdb
conda install -c conda-forge openvdb
```

`conda-forge`'s `openvdb` package bundles a Python module that's
known to be ABI-compatible with the included Python.

Verify the install:

```sh
python -c "import pyopenvdb; print(pyopenvdb.__file__)"
```

### 2. Pick a `.vdb` cloud

Recommended sources of CC-licensed VDB clouds:

- [Walt Disney Animation Studios — Cloud Data Set](https://disneyanimation.com/data-sets/?drawer=/resources/clouds/)
  — three production-quality cloud volumes (the small "Cloud A" is
  a great starting point — ~50 MB compressed).
- [JangaFX EmberGen samples](https://embergen.com/) — various
  smoke + cloud `.vdb` files.

### 3. Convert

```sh
python scripts/vdb_to_qvg.py path/to/cloud.vdb data/grids/cloud.qvg
```

Useful options:

- `--resolution N` — output dims `N×N×N` (default 64; 128 is a
  good "publishable cloud" size).
- `--resolution X Y Z` — non-uniform resolution if the cloud's
  bounding box isn't roughly cubic.
- `--grid-name NAME` — name of the scalar grid inside the VDB
  (default `"density"`; some files use `"ws_density"`,
  `"scalar"`, etc.).
- `--normalize` — rescale so the maximum voxel value maps to 1.0
  (recommended for arbitrary input).

### 4. Render with the new grid

```sh
cargo run --release -- render \
    --scene data/gltf/cornell_cloud.gltf \
    --width 512 --height 512 --spp 1024 \
    --cloud-grid data/grids/cloud.qvg \
    --out data/output/cornell_real_cloud
```

Without `--cloud-grid`, the renderer uses the embedded procedural
cumulus at `data/grids/cumulus_64.qvg`.

## `.qvg` format reference

See `src/pathtrace/grid.rs` for the canonical Rust types. Layout
(little-endian throughout):

| Offset | Type | Field |
| --- | --- | --- |
| 0 | `u8[4]` | magic = `b"QVG1"` |
| 4 | `u32 × 3` | dims `(w, h, d)` |
| 16 | `f32 × 3` | bounds_min `(x, y, z)` |
| 28 | `f32 × 3` | bounds_max |
| 40 | `u32` | voxel_count (must equal `w * h * d`) |
| 44 | `u8[voxel_count]` | row-major voxels, x varies fastest |

Voxels are `R8Unorm` (`u8 / 255.0` is the density), bounds are in
the world-space frame the path tracer renders in. The path tracer
maps world-space samples into `[0, 1]³` via the material's
`cloud_center ± cloud_radius` AABB — the grid file's `bounds_min /
bounds_max` are advisory and currently ignored by the renderer
(they're round-tripped for future scene-aware loaders).
