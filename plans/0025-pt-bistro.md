# PT-bistro — Amazon Lumberyard Bistro hero render

- **Status:** completed
- **Last updated:** 2026-06-07
- **Last touched on:** all five milestones ticked; **Exterior** variant chosen over Interior_Wine; sun-only no-env framing; README hero swap

## Goal

Land the canonical cafe-scene benchmark in the README hero
gallery: a render of the **Amazon Lumberyard Bistro Interior
Wine** variant. Original Bistro is the scene that
ray-tracing renderers traditionally use to flex; landing it
puts Quasi in the same conversation. The Interior Wine
variant is the right first target — at ~936 K triangles it
fits the current storage-buffer cap (the 2.8 M-tri Exterior
overflows the WebGPU 128 MB `max_storage_buffer_binding_size`
default at Vertex-stride 64 B). Exterior stays parked for a
future scope plan.

## Context

What's already in (post plans 0022–0024):

* glTF ingest with PBR materials (baseColor + MR + normal +
  per-vertex tangents)
* Texture-array layer-size resize at upload (plan 0022)
* Sun + env (plans 0014, 0023)
* Geometric-vs-shading normal split + hemisphere clamp
  (plan 0024 followups; commits `463ace1` + `8bb6c8e`)

What's specifically **not** in:

* **KTX2 / Basis Universal texture decoding.** The qian-o
  glTF distribution ships textures in `.ktx2` (Basis
  Universal compressed); our texture path only decodes PNG /
  JPG via the `image` crate. Solving this is its own
  followup; for the first render we **skip the textures**
  and let materials fall back to `baseColorFactor` + scalar
  metallic-roughness from the glTF JSON. The result reads as
  matte clay shading — ugly, but proves the geometry
  pipeline carries the scene.
* **KHR_materials_* extensions** beyond plain metalRoughness.
  Bistro uses some (specularGlossiness, transmission). We
  ignore extensions and fall back to metalRoughness.
* **glTF-LFS-aware fetcher.** The qian-o repo stores `.bin`
  files via Git LFS. Plain `raw.githubusercontent.com`
  returns the LFS pointer, not the binary. We hit
  `media.githubusercontent.com/media/...` instead, which
  serves the real bytes.

## Design

### Asset

* Source: `github.com/qian-o/GLTF-Assets` (CC BY 4.0,
  matches Bistro upstream).
* Files needed for round one:
  * `Bistro/BistroInterior_Wine.gltf` (~3.5 MB, scene
    description, regular content)
  * `Bistro/BistroInterior_Wine.bin` (the real geometry —
    served via the LFS media URL)
  * `Bistro/san_giuseppe_bridge_4k.hdr` (the matching env
    map, ~24 MB, also LFS)
* Textures (KTX2, ~515 MB) — **skipped for round one.**

### Fetcher

Extend `scripts/fetch_khronos_scene.py` pattern: add a small
`scripts/fetch_bistro.py` that hits the LFS media endpoint
for the few files we need. Idempotent + size-verifying like
the other fetchers. Keeps the asset off the repo tree;
gitignored at `data/gltf/bistro/`.

### Render

Same CLI surface as Sponza / chess: `--scene
data/gltf/bistro/BistroInterior_Wine.gltf --env-map ... 
--camera-pos ... --look-at ... --sun-dir ... --width ...
--spp ...`. Framing locked in after a smoke pass at
384²/128 spp.

### Followups (NOT in this plan)

* **PT-ktx2** — Basis Universal / KTX2 texture decoder.
  Needs a Rust crate (`basis-universal` or `ktx2` + manual
  Basis decode). Unlocks the full PBR texture stack on
  Bistro and on the many other modern glTFs that ship KTX2.
* **PT-bistro-exterior** — the 2.8 M-tri Exterior variant.
  Needs either (a) larger storage buffers via
  `request_device` limits negotiation, or (b) vertex
  compression to bring the per-vertex byte count down. Both
  flagged in plan 0022 followups.

## Milestones

- [x] **[PT-bistro/fetcher]** `scripts/fetch_bistro.py`
  lands; `data/gltf/bistro/` gitignored. Tested by
  downloading the Interior Wine variant + env map.
- [x] **[PT-bistro/load]** `cargo run --release -- render
  --scene data/gltf/bistro/BistroInterior_Wine.gltf
  --width 256 --height 256 --spp 16 --out /tmp/smoke` runs
  to completion without panic. Smoke render exists.
- [x] **[PT-bistro/frame]** Smoke render at 384²/128 spp
  shows a recognisable bistro interior with a framing that
  reads as a hero shot — flagged in the plan body.
- [x] **[PT-bistro/hero]** 1024² / 2048 spp hero render
  lands as `data/output/bistro_reference.png`.
- [x] **[PT-bistro/readme]** README hero gallery picks up
  the bistro card (likely replacing or joining the chess
  card).

## Done when

* All five milestones ticked
* Bistro hero PNG committed (EXR + textures stay
  gitignored)
* This plan moves to `Status: completed`

## Findings

### Pivot from Interior_Wine to Exterior

The plan's initial target was **Interior Wine** (~1.3 M tris,
under the storage-buffer cap by default). At execution time
that variant proved unrenderable as a hero render without
textures: only **7 unique baseColorFactors across 73
materials** and a **96 % mean metallic factor**, so most
interior surfaces collapse to a small set of mirror-like
greys that reflect whatever env-map happens to be loaded. Every
framing came back blue-blob soup because the env-map sky
dominated through every alpha-stripped foliage and window
opening. Without the KTX2 texture stack the Interior_Wine
scene has no visual story to tell.

The **Exterior** variant works at 2.8 M tris because outdoor
lighting carries the composition — sun + building geometry +
silhouettes read coherently without surface detail. The
offscreen renderer already requests `adapter.limits()` rather
than the WebGPU defaults, so the 180 MB vertex buffer
(2.8 M × 64 B) fits under Metal's much higher
`max_storage_buffer_binding_size` cap.

### Sun-only lighting, no env

Env-bleed through alpha-stripped foliage geometry blew out
every framing where the env map was enabled — the asset's
trees are billboards whose alpha-cutout textures we stripped,
so they render as solid coloured rectangles that punch the
sky through every line of sight. Sun-only lighting (no
`--env-map`) sidesteps the problem entirely; the geometry
shadowing carries the story.

### Tonemap-clipping pass

The first hero (`--sun-intensity 25`) blew the archway out
to clipped white — render-attacker flagged "the
compositional payoff is a hole, not a destination" as P0.
Re-rendered at `--sun-intensity 8`, which lands the archway
opening in a readable mid-range with building silhouettes
visible through it.

### Mesh root transform

Bistro raw mesh accessor mins/maxes are 100× larger than
the rendered scene bounds because the single root node
applies `scale: 0.016` and a 90° X-axis quaternion rotation
(Y-up to Z-up). Took two confused framings before tracing
the root transform; documenting here so the next scene
delivery doesn't repeat the mistake. The Vespa is at world
`(-7.4, -0.5, 1.4)`; the Paris awnings are around world
`(-13, 21, 12)`.

## Followups (NOT in this plan)

* **PT-ktx2** — Basis Universal / KTX2 texture decoder.
  Needs a Rust crate (`basis-universal` or `ktx2` + manual
  Basis decode) or a build-time `basisu -unpack` step that
  rewrites the glTF to point at `.png`. Unlocks the full
  Bistro PBR texture stack (~515 MB across 622 textures)
  and a "Bistro with textures" round-two hero. **Single
  highest-leverage texture-pipeline play available.**
* **PT-bistro-textured** — re-render after PT-ktx2 lands.
* **PT-alpha-mask** — alpha-masked geometry for foliage so
  env-bleed through stripped-texture billboards stops
  degrading every outdoor scene's framing. Could also live
  inside PT-ktx2 (foliage textures lose their meaning
  without their alpha channel).
* **PT-mikktspace** — per-face tangents to address the
  UV-seam vertical-line artifact that render-attacker
  flagged on the Bistro facades (same artifact class the
  chess scene exposed; carried forward unfixed).
