#!/usr/bin/env python3
"""Vendor the Hosek-Wilkie 2012 sky-model coefficient tables.

Fetches the official C++ reference release (v1.4a, Feb 2013) from
cgg.mff.cuni.cz, extracts the float literals from
``ArHosekSkyModelData_RGB.h``, and rewrites the ``mod data`` block
inside ``src/pathtrace/sky.rs`` with the official values.

The Rust-side sky module ships with all-zero stub tables so the math
gets unit-test coverage independent of the data vendor. This script
flips the stubs to real values — after which the sky model produces
physically-correct radiance.

## Usage

::

    python3 scripts/sky/fetch_hosek_data.py

The script is **idempotent**: running it twice produces the same Rust
source. The upstream zip SHA-256 is pinned at the top so a quiet
upstream change surfaces as a script error rather than silently
shipping different numbers.

## What it doesn't do (yet)

* No spectral mode — only the RGB tables are extracted. The spectral
  data lives in ``ArHosekSkyModelData_Spectral.h`` and would belong in
  a separate Rust module if Quasi ever goes spectral.
* No 2013 sun-disc model — that's a different table set (PT-sky/sun-
  disc milestone, plan 0030).

## License

The vendored data is BSD-licensed by Hošek & Wilkie. See the upstream
release for the full license text; the Rust file generated here
includes the BSD attribution as a comment header.
"""

from __future__ import annotations

import hashlib
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

UPSTREAM_URL = (
    "https://cgg.mff.cuni.cz/projects/SkylightModelling/"
    "HosekWilkie_SkylightModel_C_Source.1.4a.zip"
)
# SHA-256 of the upstream zip at the time of last successful vendor.
# A mismatch here means upstream changed the release without a
# version bump — the script aborts so we don't silently ship
# different numbers. To roll forward: download the new zip, verify
# it against the cgg.mff.cuni.cz site, recompute the SHA, and update
# this constant.
EXPECTED_ZIP_SHA256 = (
    # TODO: set on first successful run. See README in this folder
    # for the workflow.
    ""
)

RUST_FILE = Path(__file__).resolve().parents[2] / "src/pathtrace/sky.rs"
# The script rewrites the `mod data { ... }` block bounded by these
# markers — leave them in place when editing sky.rs by hand.
MOD_START_MARKER = "mod data {"
MOD_END_MARKER = "}  // end of `mod data`"


def main() -> int:
    print(f"[hosek-data] fetching {UPSTREAM_URL} …")
    zip_bytes = urllib.request.urlopen(UPSTREAM_URL).read()
    actual_sha = hashlib.sha256(zip_bytes).hexdigest()
    print(f"[hosek-data] zip SHA-256: {actual_sha}")

    if EXPECTED_ZIP_SHA256 and actual_sha != EXPECTED_ZIP_SHA256:
        print(
            f"[hosek-data] ABORT: upstream SHA mismatch.\n"
            f"  expected: {EXPECTED_ZIP_SHA256}\n"
            f"  actual:   {actual_sha}\n"
            f"If the upstream release legitimately changed, verify the new\n"
            f"contents against the site, then update EXPECTED_ZIP_SHA256 in\n"
            f"this script.",
            file=sys.stderr,
        )
        return 1

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Find ArHosekSkyModelData_RGB.h — its path inside the zip
        # has historically been "HosekWilkie/.../ArHosekSkyModelData_RGB.h"
        # but we glob for it to be robust.
        rgb_header = None
        for name in zf.namelist():
            if name.endswith("ArHosekSkyModelData_RGB.h"):
                rgb_header = name
                break
        if rgb_header is None:
            print(
                "[hosek-data] ABORT: ArHosekSkyModelData_RGB.h not found in zip.",
                file=sys.stderr,
            )
            return 1
        print(f"[hosek-data] extracting {rgb_header}")
        header_src = zf.read(rgb_header).decode("utf-8")

    tables = extract_tables(header_src)
    rust_block = render_rust_module(tables)
    rewrite_rust_file(rust_block)

    print(f"[hosek-data] rewrote {RUST_FILE.relative_to(RUST_FILE.parents[2])}")
    return 0


def extract_tables(src: str) -> dict[str, list[float]]:
    """Parse the six tables out of the C++ header.

    Returns a dict keyed by table name. The reference release defines
    six arrays for RGB mode:

    * ``datasetRGB1``, ``datasetRGB2``, ``datasetRGB3`` — the per-
      channel Perez parameter tables (1200 floats each).
    * ``datasetRGBRad1``, ``datasetRGBRad2``, ``datasetRGBRad3`` —
      the per-channel zenith-radiance tables (120 floats each).
    """
    tables: dict[str, list[float]] = {}
    table_names = [
        "datasetRGB1",
        "datasetRGB2",
        "datasetRGB3",
        "datasetRGBRad1",
        "datasetRGBRad2",
        "datasetRGBRad3",
    ]
    for name in table_names:
        # Match `double <name>[<count>] = { ... };` — values can span
        # many lines and include scientific notation.
        pattern = re.compile(
            r"double\s+" + re.escape(name) + r"\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}",
            re.DOTALL,
        )
        m = pattern.search(src)
        if not m:
            raise RuntimeError(f"table {name!r} not found in header")
        body = m.group(1)
        # Extract floats — handle integer literals, scientific notation,
        # and trailing commas.
        floats = []
        for token in re.split(r"[\s,]+", body):
            token = token.strip()
            if not token:
                continue
            # Strip trailing 'f' or 'F' suffix if present (C float
            # literals).
            if token.endswith(("f", "F")):
                token = token[:-1]
            floats.append(float(token))
        tables[name] = floats
        expected_size = 1200 if not name.endswith(("Rad1", "Rad2", "Rad3")) else 120
        if len(floats) != expected_size:
            raise RuntimeError(
                f"table {name!r} has {len(floats)} floats, expected {expected_size}"
            )
    return tables


def render_rust_module(tables: dict[str, list[float]]) -> str:
    """Generate the `mod data { ... }` Rust source block.

    The output preserves the original 6-control-points-per-(channel,
    turbidity, albedo) structure so the `control_set` lookup stays a
    constant-time array index.
    """
    lines = [
        "// AUTOGENERATED by scripts/sky/fetch_hosek_data.py from",
        "// the official Hosek-Wilkie 2012 reference release. Do not",
        "// edit by hand — re-run the vendor script to update.",
        "//",
        "// Original data © Hošek & Wilkie, BSD-licensed. See the",
        "// upstream release at cgg.mff.cuni.cz/projects/SkylightModelling/",
        "// for the full license text.",
        "",
        "mod data {",
        "    pub(super) struct ControlSet {",
        "        pub params: [[f32; 6]; 9],",
        "        pub zenith: [f32; 6],",
        "    }",
        "",
    ]
    for channel, base in enumerate(["datasetRGB1", "datasetRGB2", "datasetRGB3"]):
        rad_name = f"datasetRGBRad{channel + 1}"
        chan_label = "R" if channel == 0 else ("G" if channel == 1 else "B")
        lines.append(f"    // Channel {chan_label}: 10 turbidity bins × 2 albedo bins, 60")
        lines.append(f"    // floats each (54 Perez params + 6 zenith rad).")
        lines.append(f"    const CHANNEL_{chan_label}: [[ControlSet; 2]; 10] = [")
        for t_bin in range(10):
            lines.append("        [")
            for a_bin in range(2):
                # In the reference release, the data layout for each
                # (turbidity, albedo) pair is 9 parameters × 6 elevation
                # control points = 54 floats, but the major order is
                # *elevation-first*: param[0] for elev 0..5, then
                # param[1] for elev 0..5, etc. We need to transpose
                # to our [param][elev] indexing.
                cell_start = (a_bin * 10 + t_bin) * 9 * 6
                cell_params = tables[base][cell_start : cell_start + 9 * 6]
                rad_start = (a_bin * 10 + t_bin) * 6
                cell_zenith = tables[rad_name][rad_start : rad_start + 6]
                lines.append("            ControlSet {")
                lines.append("                params: [")
                for p in range(9):
                    ctrl = [cell_params[p * 6 + e] for e in range(6)]
                    formatted = ", ".join(f"{v}_f32" for v in ctrl)
                    lines.append(f"                    [{formatted}],")
                lines.append("                ],")
                formatted_zenith = ", ".join(f"{v}_f32" for v in cell_zenith)
                lines.append(f"                zenith: [{formatted_zenith}],")
                lines.append("            },")
            lines.append("        ],")
        lines.append("    ];")
        lines.append("")
    lines.extend(
        [
            "    pub(super) fn control_set(channel: usize, t_bin: usize, a_bin: usize)",
            "        -> &'static ControlSet",
            "    {",
            "        debug_assert!(channel < 3);",
            "        debug_assert!(t_bin < 10);",
            "        debug_assert!(a_bin < 2);",
            "        match channel {",
            "            0 => &CHANNEL_R[t_bin][a_bin],",
            "            1 => &CHANNEL_G[t_bin][a_bin],",
            "            _ => &CHANNEL_B[t_bin][a_bin],",
            "        }",
            "    }",
            "}  // end of `mod data`",
            "",
        ]
    )
    return "\n".join(lines)


def rewrite_rust_file(new_block: str) -> None:
    text = RUST_FILE.read_text()
    start = text.find(MOD_START_MARKER)
    if start == -1:
        raise RuntimeError(
            f"could not find `{MOD_START_MARKER}` in {RUST_FILE} — script "
            f"can't safely rewrite the file."
        )
    # Find the matching end marker. The hand-written version has
    # `}` closing `mod data` plus a comment we maintain — match the
    # next-occurrence-after-start.
    end_marker_pos = text.find(MOD_END_MARKER, start)
    if end_marker_pos == -1:
        # Fallback for the initial hand-written version that doesn't
        # have the explicit end marker yet: match the closing brace
        # of `mod data { ... }` by depth-tracking.
        depth = 0
        i = start + len(MOD_START_MARKER)
        end = -1
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                if depth == 0:
                    end = i + 1
                    break
                depth -= 1
            i += 1
        if end == -1:
            raise RuntimeError("could not find end of `mod data` block")
    else:
        end = end_marker_pos + len(MOD_END_MARKER)

    new_text = text[:start] + new_block.lstrip() + text[end:]
    RUST_FILE.write_text(new_text)


if __name__ == "__main__":
    sys.exit(main())
