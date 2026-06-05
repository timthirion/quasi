# scripts/

Python utilities that sit alongside (not inside) the Rust crate.
Anything that needs an external Python dependency lives here so
the Rust build stays portable to machines without a Python
toolchain.

## Index

| File | Purpose |
| --- | --- |
| `qvg_writer.py` | Pure-Python writer for the `.qvg` density-grid format (no third-party deps). |
| `vdb_to_qvg.py` | Converts an OpenVDB `.vdb` file to `.qvg` via `pyopenvdb`. **Caveat**: `pyopenvdb` isn't on PyPI and needs to be built against OpenVDB headers — see the install section. |
| `vdb_to_qvg.cpp` | **Recommended path** on macOS. C++ converter that links directly against Homebrew's `openvdb`. Avoids the `pyopenvdb` install problem entirely. |
| `test_qvg_writer.py` | Pure-Python tests for `qvg_writer`. Run with `python -m unittest discover scripts -p 'test_*.py'`. |

## VDB ingest: end-to-end

The Rust path tracer's PT-vdb pipeline reads dense 3-D density
grids in our own `.qvg` ("quasi volume grid") format. Real-world
clouds typically ship as sparse OpenVDB `.vdb` files; the converter
resamples them to dense and quantises to `R8`.

### 1. Pick a converter path

`vdb_to_qvg.cpp` (the C++ converter) is the recommended option on
macOS. It links against Homebrew's `openvdb` C++ library directly
and sidesteps the long-running headache that is `pyopenvdb`.

**Build it:**

```sh
brew install openvdb
c++ -std=c++17 -O2 \
    -I$(brew --prefix openvdb)/include \
    -I$(brew --prefix tbb)/include \
    -L$(brew --prefix openvdb)/lib \
    -L$(brew --prefix tbb)/lib \
    -lopenvdb -ltbb \
    scripts/vdb_to_qvg.cpp -o scripts/vdb_to_qvg
```

That produces a `scripts/vdb_to_qvg` binary. The CLI matches the
Python script's: `INPUT.vdb OUTPUT.qvg [--resolution N | --resolution
X Y Z] [--grid-name NAME] [--normalize] [--list-grids]`.

The Python script (`vdb_to_qvg.py`) still exists for environments
where you have `pyopenvdb` available (e.g. a conda-forge install).
The on-disk output of both is byte-identical.

**Why the C++ recommendation:** `pyopenvdb` isn't published on
PyPI. The PyPI listings under that name are unrelated. To get it
you either build OpenVDB from source with `OPENVDB_BUILD_PYTHON_MODULE=ON`,
or install via `conda-forge`'s `openvdb` package. Homebrew's
`openvdb` formula doesn't build the Python module. Hence: linking
the C++ converter against the brew library is the path of least
resistance on macOS.

If you DO want `pyopenvdb`:

```sh
# conda-forge bundles a working Python module
conda create -n quasi-vdb python=3.11
conda activate quasi-vdb
conda install -c conda-forge openvdb
python -c "import pyopenvdb; print(pyopenvdb.__file__)"
```

And note that on modern macOS Python installs you'll hit PEP 668
("externally-managed-environment") if you try `pip install` against
the system Python — use a venv (`python3 -m venv ~/.venvs/quasi-vdb`)
or pipx for that case.

### 2. Pick a `.vdb` cloud

The recommended demo file is the **Walt Disney Animation Studios
Cloud Data Set** — a CC-BY-SA 3.0 licensed production-quality
cumulus that Disney published in 2017. Direct download (3 GB zip):

```
https://assets.disneyanimation.com/wdas_cloud.zip
```

Read the cloud license at
`https://media.disneyanimation.com/uploads/production/data_set_asset/6/asset/License_Cloud.pdf`
before redistributing. The zip contains:

| File | Description |
| --- | --- |
| `wdas_cloud.vdb` | Full-resolution density (~2.7 GB) |
| `wdas_cloud_half.vdb` | Half resolution (~470 MB) |
| `wdas_cloud_quarter.vdb` | Quarter resolution (~65 MB) — best resolution-vs-size for our converter |
| `wdas_cloud_eighth.vdb` | Eighth (~10 MB) |
| `wdas_cloud_sixteenth.vdb` | Sixteenth (~1.6 MB) — fast iteration |

Other CC-licensed sources: JangaFX EmberGen sample packs,
OpenVDB.org's own example grids.

### 3. Convert

```sh
# Quarter resolution at 128³ — the recommended "publishable" pairing.
scripts/vdb_to_qvg \
    /path/to/wdas_cloud_quarter.vdb \
    data/grids/disney_quarter_128.qvg \
    --resolution 128 --normalize
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
  (recommended for arbitrary input — Disney's grid in particular
  has peak density ≈ 5, not 1).
- `--list-grids` — print the grid names in a file and exit (use
  this when `density` isn't the right name).

`.qvg` files derived from third-party data live under
`data/grids/disney_*.qvg` etc. and are **gitignored** — re-derive
them locally rather than redistributing. The procedural
`cumulus_64.qvg` is committed because it's ours.

### 4. Render with the new grid

```sh
cargo run --release -- render \
    --scene data/gltf/cornell_cloud.gltf \
    --width 512 --height 512 --spp 1024 \
    --cloud-grid data/grids/disney_quarter_128.qvg \
    --out data/output/cornell_disney_cloud_reference
```

Without `--cloud-grid`, the renderer uses the embedded procedural
cumulus at `data/grids/cumulus_64.qvg`.

### Attribution

When rendering with the Disney Cloud Data Set, credit the source
in any published image:

> Cloud volume: Walt Disney Animation Studios Cloud Data Set,
> available under CC BY-SA 3.0.

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
