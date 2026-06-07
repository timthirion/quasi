#!/usr/bin/env python3
"""
Fetch the Crytek Sponza glTF asset from the Khronos sample-models
repository into `data/gltf/sponza/`.

The asset is ~53 MB across 71 files (Sponza.gltf + Sponza.bin + 68
textures + 1 white placeholder), too large to commit alongside the
other Quasi data. `.gitignore` excludes `data/gltf/sponza/`; this
script lands the asset on the contributor's disk + CI runner on
demand.

Pure stdlib (urllib + json) so it slots into the same Python ≥ 3.8
constraint the existing `scripts/qvg_writer.py` ships under. No
external deps.

Usage:
    python3 scripts/fetch_sponza.py

Idempotent: existing files matching the manifest size are skipped.
On first run the download takes ~30 s on a residential connection.

The asset's licence is the same as the Khronos sample-models
distribution (CC-BY 4.0 for the textures; CC0 for the geometry).
Source: https://github.com/KhronosGroup/glTF-Sample-Models
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = "KhronosGroup/glTF-Sample-Models"
BRANCH = "main"
SUBDIR = "2.0/Sponza/glTF"
DEST = Path(__file__).resolve().parent.parent / "data" / "gltf" / "sponza"

LISTING_URL = f"https://api.github.com/repos/{REPO}/contents/{SUBDIR}?ref={BRANCH}"


def fetch_listing() -> list[dict]:
    """Hit the GitHub contents API to enumerate files + sizes."""
    print(f"Fetching listing: {LISTING_URL}", file=sys.stderr)
    req = urllib.request.Request(
        LISTING_URL, headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def fetch_one(entry: dict, dest_dir: Path) -> tuple[str, int, bool]:
    """Download one file from the listing entry. Returns (name, bytes, skipped)."""
    name = entry["name"]
    size = entry["size"]
    url = entry["download_url"]
    target = dest_dir / name
    if target.exists() and target.stat().st_size == size:
        return name, size, True
    with urllib.request.urlopen(url) as r:
        target.write_bytes(r.read())
    actual = target.stat().st_size
    if actual != size:
        raise RuntimeError(
            f"{name}: expected {size} bytes, got {actual} (download truncated?)"
        )
    return name, size, False


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    listing = fetch_listing()
    total_bytes = 0
    downloaded = 0
    skipped = 0
    for entry in listing:
        if entry["type"] != "file":
            continue
        name, size, was_skipped = fetch_one(entry, DEST)
        total_bytes += size
        if was_skipped:
            skipped += 1
        else:
            downloaded += 1
            print(f"  downloaded {name} ({size / 1024:.0f} KiB)", file=sys.stderr)
    print(
        f"\nSponza ready at {DEST}\n"
        f"  {downloaded} downloaded, {skipped} skipped (already current)\n"
        f"  {total_bytes / 1e6:.1f} MB total across {downloaded + skipped} files",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
