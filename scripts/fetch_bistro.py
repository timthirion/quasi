#!/usr/bin/env python3
"""
Fetch the Amazon Lumberyard Bistro glTF distribution into
`data/gltf/bistro/`.

Source: github.com/qian-o/GLTF-Assets (CC BY 4.0, matches Bistro
upstream). The `.bin` mesh data and `.hdr` env map are stored in
Git LFS; plain raw.githubusercontent.com returns only the LFS
pointer file. We hit media.githubusercontent.com/media/...
instead, which serves the real bytes.

For plan 0025 round one we fetch only the **Interior Wine**
variant plus the matching env map — under 250 MB total. The
622-file `.ktx2` texture directory (~515 MB) is skipped because
our renderer doesn't decode KTX2 / Basis Universal yet
(PT-ktx2 followup). The geometry renders without textures via
the glTF's `baseColorFactor` + scalar metallic-roughness
fallbacks.

Usage:
    python3 scripts/fetch_bistro.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO = "qian-o/GLTF-Assets"
BRANCH = "main"
DEST = Path(__file__).resolve().parent.parent / "data" / "gltf" / "bistro"

# Files we need for the Interior Wine hero. The .gltf is regular
# content (raw); the .bin and .hdr are LFS-stored (media). The
# Textures/ directory (~515 MB) is intentionally skipped.
RAW_FILES = [
    # The .gltf scene descriptions sit in regular git content.
    "Bistro/BistroExterior.gltf",
    "Bistro/BistroInterior_Wine.gltf",
    # The env map is under git's 100 MB threshold and lives at
    # raw.githubusercontent.com directly (no LFS).
    "Bistro/san_giuseppe_bridge_4k.hdr",
]
LFS_FILES = [
    # Mesh buffers are LFS-stored.
    "Bistro/BistroExterior.bin",
    "Bistro/BistroInterior_Wine.bin",
]


def fetch_url(url: str, target: Path, label: str) -> int:
    """Stream the URL into target; return bytes written."""
    if target.exists() and target.stat().st_size > 0:
        return target.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {label} ...", file=sys.stderr, flush=True)
    written = 0
    with urllib.request.urlopen(url) as r, target.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
    return written


def strip_textures(gltf_path: Path) -> bool:
    """Round-one PT-bistro decision: skip textures entirely. The
    qian-o distribution ships textures in `.ktx2` (Basis Universal
    compressed) which our loader doesn't decode (PT-ktx2 followup).
    `gltf::import` rejects any glTF whose `extensionsRequired`
    contains an unsupported entry, and even after demoting that it
    follows texture URIs and panics when the .ktx2 files aren't on
    disk.

    Surgically rewrite the glTF JSON so material extraction falls
    back to `baseColorFactor` + scalar metallic-roughness, which
    every material declares. Steps:

    * Drop `KHR_texture_basisu` from `extensionsRequired`.
    * Empty the `images` and `textures` arrays.
    * Strip `baseColorTexture` / `metallicRoughnessTexture` /
      `normalTexture` / `occlusionTexture` / `emissiveTexture`
      from every material.

    Round two (PT-ktx2) re-fetches the KTX2 textures and re-runs
    the strip with a `--keep-textures` flag.
    """
    import json
    data = json.loads(gltf_path.read_text())
    changed = False
    req = data.get("extensionsRequired", [])
    if "KHR_texture_basisu" in req:
        req.remove("KHR_texture_basisu")
        if not req:
            data.pop("extensionsRequired", None)
        else:
            data["extensionsRequired"] = req
        changed = True
    if data.get("images"):
        data["images"] = []
        changed = True
    if data.get("textures"):
        data["textures"] = []
        changed = True
    for mat in data.get("materials", []):
        pbr = mat.get("pbrMetallicRoughness", {})
        for key in ("baseColorTexture", "metallicRoughnessTexture"):
            if pbr.pop(key, None) is not None:
                changed = True
        for key in ("normalTexture", "occlusionTexture", "emissiveTexture"):
            if mat.pop(key, None) is not None:
                changed = True
    if changed:
        gltf_path.write_text(json.dumps(data, separators=(",", ":")))
    return changed


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    total = 0
    for path in RAW_FILES:
        url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}"
        target = DEST / Path(path).name
        n = fetch_url(url, target, Path(path).name)
        total += n
        print(f"  {target.name}: {n / 1e6:.1f} MB", file=sys.stderr)
    for path in LFS_FILES:
        url = f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/{path}"
        target = DEST / Path(path).name
        n = fetch_url(url, target, Path(path).name)
        total += n
        print(f"  {target.name}: {n / 1e6:.1f} MB", file=sys.stderr)
    # Post-process: demote KHR_texture_basisu so the gltf crate
    # accepts the file. See `strip_basisu_extension` for the why.
    for name in ("BistroInterior_Wine.gltf", "BistroExterior.gltf"):
        gltf = DEST / name
        if gltf.exists() and strip_textures(gltf):
            print(
                f"  stripped texture refs from {name} "
                "(round one: bake into baseColorFactor only)",
                file=sys.stderr,
            )
    print(
        f"\nBistro Interior Wine ready at {DEST}\n  {total / 1e6:.1f} MB total",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
