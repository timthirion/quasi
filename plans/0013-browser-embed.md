# Browser embed (blog widget pipeline)

- **Status:** active
- **Last updated:** 2026-06-05
- **Last touched on:** BROWSER-embed-page landed; deploy in flight

## Goal

Make Quasi the kind of thing a blog post can drop in via a single
`<iframe>` tag. Concretely: publish the wasm build to GitHub Pages
at `https://timthirion.github.io/quasi/`, expose URL-parameter
routing so each post can pin its own scene + sampler + integrator,
and ship a minimal `embed.html` whose chrome is sized for blog
typography rather than the development showcase.

This pays off the "runs in your browser" hook the new README
leans on, and lets future blog posts in
[`~/src/timthirion.github.io/`](https://github.com/timthirion/timthirion.github.io)
embed live, interactive renderers without copying any artifacts
between repos. The blog repo itself needs **zero** structural
changes — only the per-post `<iframe>` snippet when a post
actually embeds something.

## Context

What's already in:

- `src/lib.rs` + `src/pathtrace/web.rs` + `src/raster/web.rs`
  carry the wasm entry points. `wasm-pack build --target web`
  produces `pkg/` (gitignored).
- A development-time `index.html` at the repo root drives three
  Quasi path-trace instances (full chrome, default chrome,
  headless) plus a motum-shaped raster widget. The headless API
  (`createHeadless(id)`, `setSampler`, `setIntegrator`,
  `frameCount`, `reset`) is exactly what an embed needs.
- The README pitches "live, interactive renders embeddable in a
  blog post" — this plan finally makes that concrete.

What this plan is **not**:

- A blog framework / blog migration / blog editor.
- A separate quasi.js npm package — embedding is via the wasm-pack
  output served from a static URL.
- A redesign of the embedding API. The shapes
  (`create`, `createHeadless`) are already good; just need URL
  routing on top.
- Mobile optimisation. The widget is desktop-first for now.
- Backend / server logic. Pure static-hosting.

## Design

### Hosting model

GitHub Pages serves the two repos at distinct subpaths from the
same domain automatically:

- `timthirion/timthirion.github.io` → `https://timthirion.github.io/`
  (the blog)
- `timthirion/quasi` (with Pages enabled) → `https://timthirion.github.io/quasi/`
  (this widget)

A blog post that wants to embed a widget drops in an `<iframe>`
pointing at the hosted Quasi URL with the right query string. No
artifact copying. Updates to quasi propagate to every embed on
the next page load.

### URL-parameter routing

The hosted page parses `window.location.search` once at startup
and configures the widget accordingly. Supported keys (all
optional):

| Key | Values | Default | Effect |
| --- | --- | --- | --- |
| `scene` | one of the committed `cornell_*.gltf` slugs | `cornell_quads` | Loads that scene as the embedded TriangleScene. |
| `integrator` | `mis`, `bsdf` | `mis` | Picks the integrator. |
| `sampler` | `pcg`, `halton`, `sobol` | `pcg` | Picks the sampler. |
| `cloud_grid` | embedded grid slug (`cumulus_64` etc.) | unset | Optional density grid for cloud scenes. |
| `compact` | `1` | unset | Hides chrome — canvas + minimal SPP readout only, sized for iframe embed. |

Unknown keys are ignored. Invalid values fall back to defaults
with a console warning.

### Two page entry points

- **`index.html`** — full development / showcase page. Stays at
  the repo root. The duplicated `init().then(...)` block gets
  cleaned up. Adds a footer linking to the GitHub repo +
  attribution.
- **`embed.html`** — minimal page for blog iframes. Single
  canvas, optional thin SPP readout strip, no header, no docs,
  no extra widgets. Reads URL params; defaults to the cornell
  glass-bunny scene with `?compact=1`-ish behaviour built in.

Both pages share the same `pkg/` wasm bundle.

### Deploy workflow

New `.github/workflows/pages.yml`:

```yaml
on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - checkout
      - install Rust stable + wasm32 target
      - install wasm-pack
      - cache cargo
      - wasm-pack build --target web --out-dir pkg --release
      - copy index.html, embed.html, examples/web/*, data/gltf/*, data/grids/*, pkg/* into a deploy dir
      - actions/upload-pages-artifact
      - actions/deploy-pages
```

Runs on every push to `main` after the existing CI workflow.
First push after the workflow lands provisions GitHub Pages on
the repo; user enables the `gh-pages` source in repo settings
once.

### Preset pages

`examples/web/cornell_glass_bunny.html`,
`examples/web/cornell_cloud.html`, etc. — short HTML files that
include the wasm bundle + call `createHeadless` with hard-coded
scene + integrator. Doubles as known-good iframe URLs for the
blog and as local development entry points (open them with a
local static server).

These avoid the URL-parameter dance for the most common
configurations, and give a one-click shareable URL per scene.

### Blog-side touchpoints

Zero infrastructure changes in `timthirion/timthirion.github.io`.
Per blog post that embeds a widget:

```html
<iframe
  src="https://timthirion.github.io/quasi/embed.html?scene=cornell_glass_bunny&integrator=mis"
  width="640" height="640" frameborder="0"
  allow="cross-origin-isolated"
  loading="lazy">
</iframe>
```

`allow="cross-origin-isolated"` is required for WebGPU + SharedArrayBuffer access in some Chromium configs; harmless to include either way.

## Milestones

### BROWSER-embed-page ✅
Get URL routing + `embed.html` working locally.

- [x] Cleaned up `index.html`: removed the duplicated
      `init().then(...)` block at the bottom (the `const instances`
      re-declaration would have errored at runtime); added a small
      footer linking to the GitHub repo + `embed.html`.
- [x] Added URL-parameter routing to `embed.html` via an inline
      `<script type="module">`. Parses `sampler`, `integrator`,
      `compact`, `controls`. `integrator=mis` aliases to the Rust
      enum's `misnee` so blog URLs read naturally.
- [x] New `embed.html` ships minimal canvas + thin SPP / sampler /
      integrator / reset strip. Compact mode (`?compact=1`)
      collapses the strip so the canvas fills the iframe.
- [x] Verified locally: `wasm-pack build --target web --dev`
      builds clean, and the imported JS symbols
      (`createHeadless`, `setSampler`, `setIntegrator`,
      `frameCount`) all exist in `pkg/quasi.js`.

**Scene routing is deferred.** The plan body called out
`?scene=cornell_glass_bunny` as a goal; the existing
`pathtrace::web::create_inner` always loads the embedded Cornell
box quads and there's no wasm-bindgen API to load a glTF blob
from JS today. A proper follow-up either (a) adds a
`createSceneFromBytes` binding so JS can `fetch()` the glTF +
hand it in, or (b) embeds every Cornell scene as a static slug
the JS picks from. Either way it's its own little plan, and
the deploy + iframe story below works for the existing single
scene.

### BROWSER-embed-deploy
GitHub Pages deploy + verify hosted URLs.

- [ ] New `.github/workflows/pages.yml` — wasm-pack build,
      collect deploy artifacts, `actions/upload-pages-artifact`,
      `actions/deploy-pages`. Runs after CI (or alongside).
- [ ] One-time GitHub repo settings tweak (documented in plan):
      Settings → Pages → Source: GitHub Actions.
- [ ] First deploy lands; verify the hosted URLs work end-to-end.
      Maybe add a 1-line README badge linking to the live demo.
- [ ] Two-or-three preset pages in `examples/web/`:
      `cornell_glass_bunny.html`, `cornell_cloud.html`, plus
      perhaps the raster planner demo split out from `index.html`.

## Open questions

- **wasm-pack `pkg/` size.** Last `pkg/quasi_bg.wasm` was a few
  MB. If that's too heavy for cold loads, optimisation passes
  (LTO + `wasm-opt -O3`) can shave it; revisit only if it's a
  real problem.
- **GLTF + grid hosting.** The embed needs to fetch the scene
  glTF and cloud grids — committing them to the deploy artifact
  is the simplest path. Total deployed size adds up to maybe
  10-20 MB. Trim later if needed.
- **Iframe `allow` permissions.** WebGPU + the
  `SharedArrayBuffer`-style isolation requirements are
  ergonomically annoying. Document the exact `allow=` string
  blog posts need.

## Done when

- A blog post in `~/src/timthirion.github.io/` can include
  `<iframe src="https://timthirion.github.io/quasi/embed.html?..." ...>`
  and get a live, interactive Quasi widget that respects the
  URL parameters.
- The development `index.html` at the repo root still works
  (with the duplicate code removed).
- A push to `main` redeploys the hosted widget without manual
  intervention.
- Naga, native cargo test, fmt, clippy, wasm32 `cargo check`,
  Python unittests, CI all stay green at HEAD.
