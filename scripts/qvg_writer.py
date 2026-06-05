"""Pure-Python writer for the .qvg ("quasi volume grid") format.

Mirrors the Rust `pathtrace::grid::Grid3D` save format byte-for-byte.
Kept dependency-free so it's testable in isolation from pyopenvdb
(the conversion script `vdb_to_qvg.py` imports this module + adds
the VDB ingest on top).

On-disk layout (little-endian throughout):

    offset 0..4    : magic = b"QVG1"
    offset 4..16   : dims u32 x 3 (w, h, d)
    offset 16..28  : bounds_min f32 x 3
    offset 28..40  : bounds_max f32 x 3
    offset 40..44  : voxel_count u32 (must equal w * h * d)
    offset 44..    : voxel data, w*h*d bytes (R8Unorm, x fastest)
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, Sequence


MAGIC = b"QVG1"


def write_qvg(
    out: BinaryIO,
    dims: Sequence[int],
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    voxels: bytes,
) -> None:
    """Write a `.qvg` blob to a binary file-like.

    `voxels` must be exactly `dims[0] * dims[1] * dims[2]` bytes
    laid out in row-major order with x varying fastest (then y,
    then z). Mismatched length raises `ValueError` rather than
    silently corrupting the file.
    """
    if len(dims) != 3 or len(bounds_min) != 3 or len(bounds_max) != 3:
        raise ValueError("dims, bounds_min, bounds_max must each have length 3")
    w, h, d = (int(dims[0]), int(dims[1]), int(dims[2]))
    if min(w, h, d) <= 0:
        raise ValueError(f"dims must be positive; got {(w, h, d)}")
    expected = w * h * d
    if len(voxels) != expected:
        raise ValueError(
            f"voxel buffer length {len(voxels)} does not match dims product {expected}"
        )
    out.write(MAGIC)
    out.write(struct.pack("<III", w, h, d))
    out.write(struct.pack("<fff", float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2])))
    out.write(struct.pack("<fff", float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2])))
    out.write(struct.pack("<I", expected))
    out.write(voxels)


def write_qvg_path(
    path: str | Path,
    dims: Sequence[int],
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    voxels: bytes,
) -> None:
    """Convenience wrapper that opens a file for binary write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        write_qvg(f, dims, bounds_min, bounds_max, voxels)


def encode_density(values: Sequence[float], normalize: bool = False) -> bytes:
    """Quantise a sequence of `[0, 1]`-style floats to `R8` bytes.

    If `normalize` is true, the input is first rescaled by its
    maximum so the largest sample maps to 255. Values outside
    `[0, 1]` after normalisation are clamped (`max < 0` is
    pathological — would mean the input is all negatives — but we
    still clamp rather than crash).
    """
    if normalize and values:
        m = max(values)
        if m > 0.0:
            scale = 1.0 / m
        else:
            scale = 0.0
    else:
        scale = 1.0
    out = bytearray(len(values))
    for i, v in enumerate(values):
        x = v * scale
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        out[i] = int(x * 255.0 + 0.5)
    return bytes(out)
