# `data/output/`

Reference render artifacts. The convention here is **tight on
purpose** — git was the wrong place to keep ~190 MB of HDR EXR
data that nobody but `cargo run` consumes.

## What's committed

| Pattern | Purpose |
|---------|---------|
| `*_reference.png` | Tonemapped PNG used by the README hero gallery, blog embeds, and visual diffs. Each is well under 1 MB at 768²/8-bit. |
| `cornell_bunny_convergence.csv` | RMSE-vs-spp convergence trace used by plan 0001 (PT-convergence). Small. |
| `denoise_comparison.png` | The PT-denoise raw-vs-denoised showcase strip. Used in the README's Denoising section. |

## What's NOT committed

| Pattern | Why |
|---------|-----|
| `*.exr` (any reference render's HDR companion) | ~17–23 MB each at 768²/2048 spp. Generatable on demand. See `.gitignore`. |
| `cornell_disney_*.{png,exr}` | Derived from the Disney Cloud Data Set (CC BY-SA 3.0) — license-incompatible with Apache 2.0. |

## Regenerating

The EXRs are produced as a side-effect of `quasi render`:

```bash
cargo run --release -- render \
    --scene data/gltf/cornell_glass_bunny.gltf \
    --width 768 --height 768 --spp 2048 \
    --out data/output/cornell_glass_bunny_reference
```

writes both `.png` and `.exr`. The EXR lands locally;
`.gitignore` keeps it out of commits.

## The broader policy

The project commits **PNGs for the gallery, OBJ + glTF for
scenes, procedurally-baked tiny textures (256² PBR maps,
synthetic env HDR, `cumulus_64.qvg`) for in-source assets**.
Anything bigger that the renderer can produce on demand — EXR
ground truth, 3D-density-grid bakes from VDB, convergence
plots above ~1 MB — stays out of git. A `git ls-tree` line
over ~5 MB earns explicit justification or doesn't land.

If you find yourself wanting to commit a chunky binary,
chances are it's an output, not an input — re-derive it from
what's already tracked.
