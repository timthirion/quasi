# PT-chess-showcase — Khronos ABeautifulGame chess scene

- **Status:** completed
- **Last updated:** 2026-06-07
- **Last touched on:** all four milestones ticked; chess hero in README + chess card added (gallery now 2×3)

## Goal

Ship a second complex-scene hero render after Sponza. Target:
the Khronos **ABeautifulGame** chess scene (1.5 M triangles,
multiple PBR materials, multi-mesh composition). Validate that
the renderer scales **6× past Sponza on triangle count** at
the same per-pixel budget without infrastructure work, and
that the sun + env combo from plan 0023 generalises beyond
Sponza-class geometry.

This plan is a **delivery plan**, not a feature plan. All
plumbing (PBR materials, glTF ingest, env, sun, BVH) is shipped
across plans 0001-0023. The only file-system additions are
the asset fetcher + the hero PNG.

## Context

What the user asked for, paraphrased: "render another large,
complex scene (cafe?)" — the question mark signalled scene
flexibility. Lumberyard Bistro (the literal cafe) is the
dream target but distributing it requires NVIDIA ORCA sign-up
and ~2 GB of FBX/material data that doesn't load cleanly into
our glTF-only pipeline today.

ABeautifulGame is the right intermediate hop:

* **1.5 M triangles** — 6× Sponza, exercises the BVH at a new
  scale.
* **Diverse PBR stack** — marble (board), polished marble
  pieces in two value-paired tints (green vs ivory), gold
  trim. Stresses the material-array + texture-array work from
  plan 0022 differently than Sponza's stone-and-banner mix.
* **Multi-mesh composition** — 32 chess pieces + board, each
  with its own normal + ORM textures.
* **CC-BY 4.0** via Khronos sample-assets, fetchable cleanly.

Deferred to a real future plan (call it PT-bistro):

* **Amazon Lumberyard Bistro** as the literal cafe. Real
  scale 3 M tris, real cafe scene, the canonical
  "renderer-flexes" benchmark. Sequencing note: PT-bistro
  needs (a) an FBX→glTF converter or NVIDIA ORCA's USD pipe,
  (b) verification of large-instance handling, (c) probably
  PT-instancing landed first. Not a one-session delivery.

## Design

Scene fetcher: extend the Sponza pattern to a generic
`scripts/fetch_khronos_scene.py` that takes the model name as
argv[1] and downloads it into `data/gltf/<slug>/`. Idempotent,
size-verifying, pure stdlib — same shape as `fetch_sponza.py`.

Render flags (no new CLI work, all from plan 0023 + 0022):

```
cargo run --release -- render \
    --scene data/gltf/abeautifulgame/ABeautifulGame.gltf \
    --env-map data/env/synthetic_sky.hdr \
    --camera-pos 0.55,0.4,0.55 --look-at 0,0.04,0 --fov 32 \
    --sun-dir 0.4,1.0,-0.2 --sun-color 1.0,0.95,0.82 \
    --sun-intensity 12 \
    --width 1024 --height 1024 --spp 2048 \
    --out data/output/chess_reference
```

Framing chosen from a 4-variation smoke pass at 384²/128 spp:

* Two-side composition: green + ivory both visible.
* Diagonal angle slightly above the board, shallow enough
  that the marble specularity reads from the camera.
* Sun from upper-right (`+x, +y, -z`) for cross-light on
  the piece silhouettes.

## Milestones

- [x] **[PT-chess-showcase/fetcher]** Generic
  `scripts/fetch_khronos_scene.py` lands; `data/gltf/<slug>/`
  added to `.gitignore` pattern. Tested by running it for
  ABeautifulGame (35 files / 29.8 MB).
- [x] **[PT-chess-showcase/smoke]** Smoke render at 384² /
  128 spp produces a recognisable chess composition with
  the framing flags above. Locked in for the hero.
- [x] **[PT-chess-showcase/hero]** 1024²/2048 spp hero
  render lands as `data/output/chess_reference.png`.
- [x] **[PT-chess-showcase/readme]** README hero gallery
  picks up the chess card.

## Done when

* All four milestones ticked
* `chess_reference.png` committed (PNG only; EXR stays
  gitignored)
* This plan moves to `Status: completed`

## Findings

### UV-pole tangent-space collapse (post-ship bug)

The chess hero shipped without an adversarial render review,
and the user caught a same-location dark patch on every white
pawn's top and bottom and a smaller one on each bishop's
crown. Root cause: Khronos ABeautifulGame ships only
`POSITION / NORMAL / TEXCOORD_0` — **no `TANGENT`** — so the
ingest path falls back to `compute_tangents`. At UV-sphere
poles where many radial triangles converge, the accumulated
tangent contributions cancel to ~zero. The old
`orthonormalize_tangent` then inflated the zero accumulator
to an arbitrary unit-length axis, the WGSL barycentric blend
inherited that axis as a unit-length but UV-meaningless
tangent, and the normal-map sample at the pole rotated the
shading normal through a meaningless direction. Shared mesh
+ shared UV-pole pixel → identical defect on every instance.

Fix is two-sided:

* `pathtrace::mesh::compute_tangents` writes a zero-length
  sentinel `[0,0,0,0]` when the projected accumulator
  collapses (length² < 1e-12), instead of inflating to the
  fallback axis. New unit test
  `compute_tangents_emits_sentinel_at_uv_sphere_pole` pins
  the behaviour with a UV-sphere-cap test geometry.
* WGSL `apply_normal_map` reads the interpolated raw
  tangent magnitude and smoothstep-fades the perturbed
  normal back to the geometric normal over
  `t_len ∈ [0.005, 0.04]`. Smooth fade avoids both the
  pinhole-too-tight failure (visible dark or bright dot at
  the pole) and the flatten-whole-triangle-too-loose
  failure (pole-fan triangles read as faceted plastic).
  Tuned by a `render-attacker` / `render-defender` pair on
  the close-up pawn renders.
* glTF-`TANGENT` ingest also propagates the sentinel when
  the supplied tangent is itself near-zero (defensive — the
  spec doesn't forbid degenerate suppliers).

The agent-tooling gap that allowed the bug to ship: the
`render-attacker` agent was defined for pair-mode (old vs
new of the same scene). A first render of a new scene has
no baseline, so pair-mode didn't apply, and I didn't invoke
it. The agent definition has been updated to spell out a
**single-image mode** ("attack the render alone for same-
location patterns across instances, pole patches, seams,
fireflies, banding") and "first render of a new scene" is
called out as the canonical trigger.

## Followups

* **PT-mikktspace** — the industry-standard fix for UV-pole
  tangent-space is **per-face tangents** (mikktspace
  convention) instead of per-vertex. Each triangle stores
  three corner tangents derived from its own UV gradient;
  no accumulation, no cancellation. Closes the smoothstep
  fade hack and the residual sub-pixel pinhole both. Scope:
  Vertex grows or we add a per-face-tangent buffer
  (storage-buffer cap is at 8 already — would need a
  repack). Deferred until a scene whose pole defects the
  smoothstep doesn't hide demands it.
* **PT-bistro** — Lumberyard Bistro as the real cafe
  target. Probably gated by an FBX→glTF converter and
  PT-instancing. Stays deferred until one of those
  prerequisites becomes a real ask.
* **PT-pbr-extension** — ABeautifulGame uses KHR_materials_*
  extensions (transmission, volume, ior) that our material
  loader currently ignores. The chess pieces render OK on
  the metalRoughness fallback, but a future plan that
  ingests those extensions would bring the marble's
  subsurface character forward.
