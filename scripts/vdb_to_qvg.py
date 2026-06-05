#!/usr/bin/env python3
"""Convert an OpenVDB `.vdb` file to the dense `.qvg` density-grid
format the Quasi path tracer reads.

Usage:

    python scripts/vdb_to_qvg.py INPUT.vdb OUTPUT.qvg \
        [--resolution N | --resolution X Y Z]    (default 64)
        [--grid-name NAME]                       (default "density")
        [--bounds-min X Y Z]                     (default: VDB grid bbox min)
        [--bounds-max X Y Z]                     (default: VDB grid bbox max)
        [--normalize]                            (rescale max → 1.0)
        [--list-grids]                           (print grid names and exit)

The script reads the VDB through `pyopenvdb`, asks OpenVDB to
resample the selected scalar grid into a dense `dims_x × dims_y ×
dims_z` array (trilinear interpolation), normalises into `[0, 1]`,
quantises to `R8`, and writes the QVG1 binary that
`pathtrace::grid::Grid3D::load` reads on the Rust side.

`pyopenvdb` is an optional dependency. Installation guidance lives
in `scripts/README.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Import the format writer first so a Python-only smoke test
# (`python scripts/vdb_to_qvg.py --help`) works even if the
# environment doesn't have `pyopenvdb`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qvg_writer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vdb_to_qvg.py",
        description="Convert an OpenVDB .vdb file to the Quasi .qvg density-grid format.",
    )
    p.add_argument("input", type=Path, help="Path to the input .vdb file.")
    p.add_argument(
        "output", type=Path, nargs="?",
        help="Path to write the .qvg output. Required unless --list-grids.",
    )
    p.add_argument(
        "--resolution",
        type=int, nargs="+", default=[64],
        metavar="N",
        help="Output grid dims. Single N → N×N×N; three values → X Y Z. (Default 64.)",
    )
    p.add_argument(
        "--grid-name",
        type=str, default="density",
        help='Name of the scalar grid inside the .vdb to read. (Default "density".)',
    )
    p.add_argument(
        "--bounds-min",
        type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="World-space bounds_min written into the .qvg header. (Default: VDB grid bbox min.)",
    )
    p.add_argument(
        "--bounds-max",
        type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="World-space bounds_max written into the .qvg header. (Default: VDB grid bbox max.)",
    )
    p.add_argument(
        "--normalize", action="store_true",
        help="Rescale so the maximum voxel value maps to 1.0 before quantising.",
    )
    p.add_argument(
        "--list-grids", action="store_true",
        help="Print the names of all scalar grids in the input and exit.",
    )
    args = p.parse_args()

    if not args.list_grids and args.output is None:
        p.error("output path is required (unless --list-grids)")
    if len(args.resolution) not in (1, 3):
        p.error("--resolution accepts either 1 value (N → N×N×N) or 3 (X Y Z)")
    return args


def parse_resolution(values: Sequence[int]) -> tuple[int, int, int]:
    if len(values) == 1:
        n = values[0]
        return (n, n, n)
    return (values[0], values[1], values[2])


def main() -> int:
    args = parse_args()

    # Lazy import so `--help` works on machines without pyopenvdb.
    try:
        import pyopenvdb as vdb  # type: ignore[import]
    except ImportError as e:
        print(
            f"vdb_to_qvg: pyopenvdb is required for VDB ingest but is not installed: {e}\n"
            "See scripts/README.md for installation guidance.",
            file=sys.stderr,
        )
        return 2

    try:
        grids = vdb.readAllGridMetadata(str(args.input))
    except Exception as e:
        print(f"vdb_to_qvg: failed to open {args.input}: {e}", file=sys.stderr)
        return 1
    grid_names = [g.name for g in grids]

    if args.list_grids:
        print(f"{args.input}: {len(grid_names)} grid(s)")
        for name in grid_names:
            print(f"  {name}")
        return 0

    if args.grid_name not in grid_names:
        print(
            f"vdb_to_qvg: grid {args.grid_name!r} not found in {args.input}. "
            f"Available: {grid_names}",
            file=sys.stderr,
        )
        return 1

    grid = vdb.read(str(args.input), args.grid_name)
    bbox_min, bbox_max = grid.evalActiveVoxelBoundingBox()
    # Index-space → world-space via the grid's transform.
    ws_min = grid.transform.indexToWorld(bbox_min)
    ws_max = grid.transform.indexToWorld(bbox_max)
    bounds_min = tuple(args.bounds_min) if args.bounds_min is not None else tuple(ws_min)
    bounds_max = tuple(args.bounds_max) if args.bounds_max is not None else tuple(ws_max)

    dims = parse_resolution(args.resolution)
    w, h, d = dims

    # Sample the VDB at the centre of each output voxel. We walk
    # world-space, converting to grid index-space via the grid's
    # `transform`. Trilinear sampling avoids the boxy artifacts that
    # come from `tools.GridSampler`'s point sampler.
    sampler = vdb.GridSampler(grid)  # default = trilinear
    sx = (bounds_max[0] - bounds_min[0]) / w
    sy = (bounds_max[1] - bounds_min[1]) / h
    sz = (bounds_max[2] - bounds_min[2]) / d

    values: list[float] = []
    values_reserve = w * h * d
    values_list_extend = values.extend  # micro-opt for the inner loop
    for iz in range(d):
        z = bounds_min[2] + (iz + 0.5) * sz
        for iy in range(h):
            y = bounds_min[1] + (iy + 0.5) * sy
            row = [0.0] * w
            for ix in range(w):
                x = bounds_min[0] + (ix + 0.5) * sx
                row[ix] = float(sampler.wsSample((x, y, z)))
            values_list_extend(row)
    assert len(values) == values_reserve, (len(values), values_reserve)

    voxels = qvg_writer.encode_density(values, normalize=args.normalize)
    qvg_writer.write_qvg_path(args.output, dims, bounds_min, bounds_max, voxels)
    mean_density = sum(voxels) / len(voxels) / 255.0 if voxels else 0.0
    nonzero = sum(1 for v in voxels if v > 0)
    print(
        f"wrote {args.output} ({Path(args.output).stat().st_size} bytes, dims={dims}, "
        f"mean density={mean_density:.3f}, non-zero voxels={nonzero}/{len(voxels)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
