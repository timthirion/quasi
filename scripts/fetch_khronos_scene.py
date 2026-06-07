#!/usr/bin/env python3
"""
Fetch a Khronos glTF Sample-Assets scene into `data/gltf/<slug>/`.

Generalisation of `scripts/fetch_sponza.py`: pass the model name as
argv[1]. Hits the `glTF-Sample-Assets` repo (the newer "Assets" repo,
not the older "Models" repo Sponza used) under
`Models/<Name>/glTF/`. Both repos use the same listing layout.

Pure stdlib (urllib + json), idempotent (skips files matching size),
size-verifying.

Usage:
    python3 scripts/fetch_khronos_scene.py ABeautifulGame
    python3 scripts/fetch_khronos_scene.py FlightHelmet
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = "KhronosGroup/glTF-Sample-Assets"
BRANCH = "main"
ROOT = Path(__file__).resolve().parent.parent


def fetch_listing(name: str) -> list[dict]:
    url = (
        f"https://api.github.com/repos/{REPO}/contents/"
        f"Models/{name}/glTF?ref={BRANCH}"
    )
    print(f"Fetching listing: {url}", file=sys.stderr)
    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def fetch_one(entry: dict, dest_dir: Path) -> tuple[str, int, bool]:
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
            f"{name}: expected {size} bytes, got {actual} (truncated?)"
        )
    return name, size, False


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <ModelName>", file=sys.stderr)
        print("  e.g.  fetch_khronos_scene.py ABeautifulGame", file=sys.stderr)
        return 2
    model = argv[1]
    dest = ROOT / "data" / "gltf" / model.lower()
    dest.mkdir(parents=True, exist_ok=True)
    listing = fetch_listing(model)
    total_bytes = 0
    downloaded = 0
    skipped = 0
    for entry in listing:
        if entry["type"] != "file":
            continue
        name, size, was_skipped = fetch_one(entry, dest)
        total_bytes += size
        if was_skipped:
            skipped += 1
        else:
            downloaded += 1
            print(f"  downloaded {name} ({size / 1024:.0f} KiB)", file=sys.stderr)
    print(
        f"\n{model} ready at {dest}\n"
        f"  {downloaded} downloaded, {skipped} skipped (already current)\n"
        f"  {total_bytes / 1e6:.1f} MB total across {downloaded + skipped} files",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
