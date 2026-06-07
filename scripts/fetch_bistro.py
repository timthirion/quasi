#!/usr/bin/env python3
"""
Fetch the Amazon Lumberyard Bistro glTF distribution into
`data/gltf/bistro/`.

Source: github.com/qian-o/GLTF-Assets (CC BY 4.0, matches Bistro
upstream). The `.bin` mesh data is stored in Git LFS; plain
raw.githubusercontent.com returns only the LFS pointer file. We
hit media.githubusercontent.com/media/... instead, which serves
the real bytes. The .hdr env map + every .ktx2 texture live in
regular git content (under the 100 MB threshold).

The Bistro Exterior + Interior Wine variants are fetched
unconditionally (~270 MB across the .gltf + .bin + matching env
HDR). The 622-file Textures/ directory (~515 MB of .ktx2
files) is fetched only when `--with-textures` is passed.

When `--with-textures` is set, this script also DECODES each
.ktx2 to .png via the `basisu` CLI (Binomial Basis Universal,
install with `brew install basis_universal`) and rewrites the
glTF image URIs from `Textures/foo.ktx2` to `Textures/foo.png`.
This is the round-two pipeline behind plan 0026 PT-ktx2 — the
basis-universal-sys Rust bindings ship with KTX2 support
disabled at build time, so we pre-decode in the fetch step
rather than building a custom sys crate.

Usage:
    python3 scripts/fetch_bistro.py                  # round 1: no textures
    python3 scripts/fetch_bistro.py --with-textures  # round 2: full PBR
"""

from __future__ import annotations

import json
import shutil
import subprocess
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


def fetch_textures_dir(dest_dir: Path) -> int:
    """Read texture URIs straight from the two glTF files and
    download each from raw.githubusercontent.com. Avoids the
    GitHub contents API (60 req/hr unauthenticated limit blows
    up after the first ~60 textures). Returns the byte count
    fetched. Idempotent on existing non-zero-size files."""
    uris: set[str] = set()
    for gltf_name in ("BistroExterior.gltf", "BistroInterior_Wine.gltf"):
        gltf = dest_dir / gltf_name
        if not gltf.exists():
            continue
        data = json.loads(gltf.read_text())
        for img in data.get("images", []):
            uri = img.get("uri", "")
            if uri.endswith(".ktx2"):
                uris.add(uri)
    print(
        f"  Texture URIs from glTFs: {len(uris)} unique .ktx2 files",
        file=sys.stderr,
    )
    tex_dir = dest_dir / "Textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    downloaded = 0
    skipped = 0
    empty = 0
    for i, uri in enumerate(sorted(uris)):
        target = dest_dir / uri
        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            total += target.stat().st_size
            continue
        url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/Bistro/{uri}"
        try:
            with urllib.request.urlopen(url) as r, target.open("wb") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
        except Exception as e:  # noqa: BLE001
            print(f"    skip {uri}: {e}", file=sys.stderr)
            target.unlink(missing_ok=True)
            continue
        size = target.stat().st_size
        if size == 0:
            empty += 1
            target.unlink(missing_ok=True)
        else:
            downloaded += 1
            total += size
        if (i + 1) % 50 == 0:
            print(
                f"    {downloaded + skipped} / {len(uris)} "
                f"({total / 1e6:.0f} MB)",
                file=sys.stderr,
                flush=True,
            )
    print(
        f"  Textures/: {downloaded} downloaded, {skipped} already current, "
        f"{empty} empty (source 0 B), {total / 1e6:.0f} MB total",
        file=sys.stderr,
    )
    return total


def basisu_unpack(ktx2_path: Path, png_out: Path) -> bool:
    """Run `basisu -unpack` on `ktx2_path` and copy the best RGBA32
    PNG output into `png_out`. Idempotent: skips when `png_out`
    already exists with non-zero size. Returns True if the decode
    ran (i.e. we did work)."""
    if png_out.exists() and png_out.stat().st_size > 0:
        return False
    work_dir = ktx2_path.parent
    stem = ktx2_path.stem  # e.g. "curtainB1_BaseColor"
    # basisu writes outputs to CWD with the input stem as a prefix.
    # We cd into the texture directory so the outputs land alongside
    # the source.
    result = subprocess.run(
        ["basisu", "-unpack", ktx2_path.name],
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"basisu -unpack failed for {ktx2_path.name}: {result.stderr.strip()[:200]}"
        )
    # Prefer RGBA32 (full RGB + alpha). Fall back to RGB32.
    candidates = [
        work_dir / f"{stem}_unpacked_rgba_RGBA32_level_0_face_0_layer0000.png",
        work_dir / f"{stem}_unpacked_rgb_RGBA32_level_0_face_0_layer0000.png",
    ]
    pick = next((c for c in candidates if c.exists()), None)
    if pick is None:
        raise RuntimeError(
            f"basisu -unpack produced no RGBA32 PNG for {ktx2_path.name}"
        )
    shutil.copy(pick, png_out)
    # Sweep up basisu's intermediate outputs — they bloat the texture
    # directory by ~30 files per source.
    for sibling in work_dir.glob(f"{stem}_unpacked_*"):
        if sibling != png_out:
            sibling.unlink(missing_ok=True)
    for sibling in work_dir.glob(f"{stem}_transcoded_*"):
        sibling.unlink(missing_ok=True)
    return True


def decode_all_ktx2(dest_dir: Path) -> int:
    """Walk `dest_dir/Textures/` and unpack every .ktx2 to a sibling
    .png. Returns the number of files decoded this run."""
    tex_dir = dest_dir / "Textures"
    if not tex_dir.exists():
        return 0
    ktx2_files = sorted(tex_dir.glob("*.ktx2"))
    decoded = 0
    skipped = 0
    for i, k in enumerate(ktx2_files):
        png_out = k.with_suffix(".png")
        try:
            if basisu_unpack(k, png_out):
                decoded += 1
            else:
                skipped += 1
        except RuntimeError as e:
            # Some KTX2 files in the qian-o distro are empty (0
            # bytes) — basisu rejects those. Skip + warn so the
            # gltf rewrite later can fall back to the factor.
            print(f"    skip {k.name}: {e}", file=sys.stderr)
            continue
        if (i + 1) % 100 == 0:
            print(
                f"    decoded {decoded + skipped} / {len(ktx2_files)}",
                file=sys.stderr,
                flush=True,
            )
    print(
        f"  KTX2 → PNG: {decoded} decoded, {skipped} skipped (already current)",
        file=sys.stderr,
    )
    return decoded


def retarget_textures(gltf_path: Path) -> int:
    """Rewrite the glTF's image URIs from `Textures/foo.ktx2` to
    `Textures/foo.png`. Skips URIs whose .png target doesn't exist
    (because basisu rejected the source). Demotes
    `KHR_texture_basisu` out of `extensionsRequired` since the
    `gltf` crate doesn't whitelist it. Returns the number of image
    URIs rewritten."""
    data = json.loads(gltf_path.read_text())
    images = data.get("images", [])
    base = gltf_path.parent
    rewrites = 0
    drops: list[int] = []
    for i, img in enumerate(images):
        uri = img.get("uri", "")
        if uri.endswith(".ktx2"):
            png_uri = uri[:-5] + ".png"
            if (base / png_uri).exists():
                img["uri"] = png_uri
                # Drop the explicit mime type — it'd still say
                # image/ktx2 otherwise.
                img.pop("mimeType", None)
                rewrites += 1
            else:
                drops.append(i)
    # For images whose KTX2 couldn't be decoded (empty source), drop
    # the URI so downstream knows to fall back to the factor.
    for i in drops:
        images[i] = {}
    req = data.get("extensionsRequired", [])
    if "KHR_texture_basisu" in req:
        req.remove("KHR_texture_basisu")
        if not req:
            data.pop("extensionsRequired", None)
        else:
            data["extensionsRequired"] = req
    # Bistro materials use the specular-glossiness workflow via
    # KHR_materials_specular; the metallicFactor isn't set, so
    # gltf defaults it to 1.0 per spec → every material reads as
    # pure metal in the metalRoughness ingest. The asset's intent
    # is dielectric. Force metallicFactor = 0 on every material
    # that carries the specular extension. Roughness stays as the
    # explicit roughnessFactor (asset sets ~0.55 globally).
    coerced = 0
    for mat in data.get("materials", []):
        ext = mat.get("extensions", {})
        if "KHR_materials_specular" in ext or "KHR_materials_pbrSpecularGlossiness" in ext:
            pbr = mat.setdefault("pbrMetallicRoughness", {})
            if pbr.get("metallicFactor") != 0:
                pbr["metallicFactor"] = 0.0
                coerced += 1
    gltf_path.write_text(json.dumps(data, separators=(",", ":")))
    print(
        f"  retargeted {rewrites} image URIs ktx2 → png in {gltf_path.name} "
        f"({len(drops)} dropped — source ktx2 was empty), "
        f"coerced {coerced} materials to dielectric",
        file=sys.stderr,
    )
    return rewrites


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


def main(argv: list[str]) -> int:
    with_textures = "--with-textures" in argv[1:]
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

    if with_textures:
        if shutil.which("basisu") is None:
            print(
                "ERROR: --with-textures needs the `basisu` CLI on PATH.\n"
                "Install with `brew install basis_universal`.",
                file=sys.stderr,
            )
            return 3
        tex_bytes = fetch_textures_dir(DEST)
        total += tex_bytes
        decode_all_ktx2(DEST)
        for name in ("BistroInterior_Wine.gltf", "BistroExterior.gltf"):
            gltf = DEST / name
            if gltf.exists():
                retarget_textures(gltf)
    else:
        for name in ("BistroInterior_Wine.gltf", "BistroExterior.gltf"):
            gltf = DEST / name
            if gltf.exists() and strip_textures(gltf):
                print(
                    f"  stripped texture refs from {name} "
                    "(round one: bake into baseColorFactor only)",
                    file=sys.stderr,
                )

    print(
        f"\nBistro ready at {DEST}\n  {total / 1e6:.1f} MB total",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
