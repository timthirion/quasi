# README aesthetic rewrite

- **Status:** done
- **Last updated:** 2026-06-05
- **Last touched on:** README-revamp landed — plan closed

## Goal

Replace the stub `README.md` with something that earns the
project a second look: a hero gallery of the best renders, a
one-paragraph pitch built around the "runs in your browser"
differentiator, a feature list with mini-images, and clean
pointers to the per-plan documentation rather than a wall of
prose.

CI is already green (plan 0010), so the badge row at the top is
already in place. This plan is content + layout only — no code.

## Context

What's already in:

- Badge row (CI, license, Rust edition, WebGPU) added at the top
  of `README.md` as part of plan 0010.
- 10 ✅-closed plans covering BSDFs, media, density grids, VDB
  ingest, CI — all the milestones have shipped reference renders
  in `data/output/cornell_*_reference.png`.
- `AGENTS.md` carries the detailed tech-stack + testing doctrine
  (intended audience: future contributors / AI agents). The
  README should *point* there, not duplicate it.
- `scripts/README.md` documents the VDB ingest pipeline.
- `plans/ROADMAP.md` carries the forward direction.

What this plan is **not**:

- A new website / docs site / Cargo.toml description rewrite.
- Restructuring `AGENTS.md` or any of the plan files.
- Adding a tutorial / walkthrough — links to the plans + scripts
  README suffice.

## Design

### Section order

1. **Title + badge row** — keep as-is.
2. **Hero gallery** — 2 × 2 markdown table, 4 images, each
   captioned with the milestone slug and a one-line plain-English
   description.
3. **One-paragraph pitch** — what Quasi is + the "runs in your
   browser" hook (the differentiator vs. any other Rust path
   tracer).
4. **Feature highlights** — short bullets, grouped by area
   (BSDFs, media, geometry, integrator, runtime). One line per
   bullet.
5. **Quick start** — `cargo run` for the windowed renderer,
   `cargo run --release -- render --scene ... --out ...` for
   headless. The smallest useful one-liner.
6. **Architecture pointer** — 2-3 sentences max, then a link to
   `AGENTS.md`.
7. **Roadmap pointer** — link to `plans/ROADMAP.md` + a short
   note on the closed-plan history (`plans/0001-*` through
   `plans/0010-*`).
8. **Scripts pointer** — link to `scripts/README.md` for the VDB
   ingest pipeline.
9. **License + credits** — Apache-2.0 link + a short Disney
   attribution paragraph for anyone reproducing the Disney cloud
   render.

### Hero gallery picks

The four renders that best convey the renderer's range:

| Image | Caption |
| --- | --- |
| `cornell_glass_bunny_reference.png` | Green-glass Stanford bunny — Beer-Lambert absorption (PT-beer-lambert) |
| `cornell_foggy_room_reference.png` | God-rays through homogeneous fog — single scattering (PT-fog) |
| `cornell_cloud_reference.png` | Procedural cumulus — heterogeneous media via delta tracking (PT-cloud) |
| `cornell_glass_sphere_reference.png` | Glass icosphere — Snell refraction + caustic (PT-dielectrics) |

Sized at ~256 px each so the 2×2 grid fits a normal screen
without overwhelming the page. HTML `<img width="256">` inside
the markdown table — GitHub respects the width attribute.

### Feature highlights structure

Grouped:

- **BSDFs** — textured Lambertian, GGX conductors, smooth
  dielectrics with Snell + Fresnel + TIR.
- **Participating media** — Beer-Lambert absorption inside
  dielectrics, homogeneous fog with single scattering, procedural
  + VDB-loaded heterogeneous clouds via delta tracking.
- **Phase** — Henyey-Greenstein anisotropy.
- **Geometry** — glTF triangle scenes, SAH binned BVH.
- **Integrator** — MIS + NEE, PCG / Halton / Sobol samplers.
- **Runtime** — single WGSL megakernel; same source builds
  native (Metal/Vulkan/DX12 via `wgpu`) and web (WebGPU).

## Milestones

### README-revamp ✅
Single milestone — content + layout pass.

- [x] New `README.md` ships in commit `2488792`. Structure:
      title + badges, hero gallery (2×2 HTML `<table>` with
      `<img width="360">`), pitch, features grouped by area,
      quick-start, architecture pointer, roadmap pointer, scripts
      pointer, license + credits.
- [x] GitHub render verified via `gh api .../readme -H 'Accept:
      application/vnd.github.html'`: badge row resolves, the 2×2
      table renders, all four `cornell_*_reference.png` paths
      resolve to the repo's image files, all internal links
      (`AGENTS.md`, `plans/`, `plans/ROADMAP.md`,
      `scripts/README.md`, `LICENSE`, `examples/gen_cloud.rs`,
      `data/grids/`, `data/obj/`) resolve.

## Open questions

- **Disney attribution on the gallery.** The four gallery renders
  use *our own* materials and scenes — no Disney data involved.
  The Disney attribution belongs in the licensing section
  specifically for reproducing the WDAS render. Keep it scoped
  there.
- **Roadmap / closed-plans pointer.** Linking to all 10 closed
  plans by name would be ugly. Better: link `plans/` as a
  directory and let GitHub render the listing.

## Done when

- A first-time visitor to the repo can tell what Quasi is, see at
  least one striking render, and find the quick-start command
  within ~10 seconds of opening the page.
- The badge row + gallery render correctly on GitHub.
- Every link in the README points at an existing file.
- Naga, native cargo test, fmt, clippy, wasm32 `cargo check`,
  Python unittests, CI all stay green at HEAD.
