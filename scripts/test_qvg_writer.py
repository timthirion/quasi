"""Tests for `qvg_writer`.

Run from the repo root with:

    python -m unittest discover scripts -p 'test_*.py'

Mirrors the Rust-side `tests/grid.rs` `Grid3D::load` round-trip so
both ends of the format are pinned independently.
"""

from __future__ import annotations

import io
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

# Make `qvg_writer` importable when this file is run via unittest
# discovery from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import qvg_writer  # noqa: E402


class WriteQvgTests(unittest.TestCase):
    def test_header_bytes_round_trip(self) -> None:
        buf = io.BytesIO()
        dims = (3, 4, 5)
        bmin = (-1.0, -2.0, -3.0)
        bmax = (1.0, 2.0, 3.0)
        voxels = bytes(range(60))  # 3 * 4 * 5
        qvg_writer.write_qvg(buf, dims, bmin, bmax, voxels)
        data = buf.getvalue()
        self.assertEqual(data[0:4], b"QVG1")
        self.assertEqual(struct.unpack("<III", data[4:16]), dims)
        self.assertEqual(struct.unpack("<fff", data[16:28]), bmin)
        self.assertEqual(struct.unpack("<fff", data[28:40]), bmax)
        self.assertEqual(struct.unpack("<I", data[40:44])[0], 60)
        self.assertEqual(data[44:], voxels)

    def test_mismatched_voxel_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            qvg_writer.write_qvg(
                io.BytesIO(),
                (2, 2, 2),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                bytes(7),  # not 2*2*2 = 8
            )

    def test_zero_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            qvg_writer.write_qvg(
                io.BytesIO(),
                (0, 4, 4),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                b"",
            )

    def test_bad_dim_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            qvg_writer.write_qvg(
                io.BytesIO(),
                (1, 1),  # only 2 components
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                b"\x00",
            )

    def test_path_writer_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "deep" / "nested" / "out.qvg"
            qvg_writer.write_qvg_path(
                target,
                (2, 2, 2),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                bytes([10, 20, 30, 40, 50, 60, 70, 80]),
            )
            self.assertTrue(target.exists())
            self.assertEqual(target.stat().st_size, 44 + 8)


class EncodeDensityTests(unittest.TestCase):
    def test_zero_returns_zero(self) -> None:
        out = qvg_writer.encode_density([0.0, 0.0, 0.0])
        self.assertEqual(out, b"\x00\x00\x00")

    def test_one_returns_255(self) -> None:
        out = qvg_writer.encode_density([1.0, 1.0])
        self.assertEqual(out, b"\xff\xff")

    def test_out_of_range_clamps(self) -> None:
        out = qvg_writer.encode_density([-0.5, 0.5, 1.5])
        # -0.5 → 0, 0.5 → 128 (rounded), 1.5 → 255
        self.assertEqual(out[0], 0)
        self.assertAlmostEqual(out[1], 128, delta=1)
        self.assertEqual(out[2], 255)

    def test_normalize_scales_max_to_255(self) -> None:
        out = qvg_writer.encode_density([0.0, 0.25, 0.5], normalize=True)
        # max is 0.5 → scaled inputs become 0.0, 0.5, 1.0
        self.assertEqual(out[0], 0)
        self.assertAlmostEqual(out[1], 128, delta=1)
        self.assertEqual(out[2], 255)

    def test_normalize_with_all_zeros_yields_zeros(self) -> None:
        out = qvg_writer.encode_density([0.0, 0.0, 0.0], normalize=True)
        self.assertEqual(out, b"\x00\x00\x00")


if __name__ == "__main__":
    unittest.main()
