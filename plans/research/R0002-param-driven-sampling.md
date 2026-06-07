# Mesh-parameterization-driven importance sampling

- **Status:** hypothesis
- **Last updated:** 2026-06-06
- **Last touched on:** drafting from the morsel + quasi crossover
- **Paper target:** Eurographics 2028 OR EGSR 2027; backup target
  Computer Graphics Forum.
- **Implementation foundation:** plan 0015 (PT-pbr-maps) +
  plan 0019 (PT-vertex-tangent) + the morsel `parameterize`
  CLI subcommand (committed `a975374` morsel-side).

## Hypothesis

Using LSCM or ARAP UV parameterization as a **structural prior
for surface-bound importance sampling** improves the variance
of path-traced renderings, at equal sample budget, vs both
BRDF-weighted importance sampling and uniform hemisphere
sampling, on textured surfaces under directional / environment
illumination.

**Mechanism:** the parameterization buckets the surface into
approximately-isometric chunks (LSCM minimises angle
distortion; ARAP additionally minimises area distortion);
a low-spp pilot pass splats per-texel incident-irradiance
proxies into the parameterization atlas; subsequent
samples draw direction proposals proportional to the
parameterized prior, with hardware-accelerated texture
sampling doing the inverse-CDF lookup.

The novelty is **not** the pilot-pass photon splat (well-known)
or the importance-sampled prior (well-known). It is the use of
**parameterization geometry as the bucketing structure**,
exploiting:

1. the approximate isometry that LSCM/ARAP provide
   (buckets-have-comparable-area = unweighted CDF terms);
2. the existing texture-sampling hardware on every GPU
   (inverse-CDF lookup = `textureSampleLevel` on a 2D atlas).

## Related work

Cited positively:

* **Lévy et al. 2002**, *Least Squares Conformal Maps for
  Automatic Texture Atlas Generation*. The LSCM
  parameterization morsel implements.
* **Liu et al. 2008**, *A Local/Global Approach to Mesh
  Parameterization* (ARAP). The other algorithm in morsel.
* **Jensen 1996**, *Global Illumination using Photon Maps*.
  The photon-splat prior.
* **Müller et al. 2017**, *Practical Path Guiding for Efficient
  Light-Transport Simulation*. The path-guiding mechanism we
  contrast with — uses spatial subdivision, not surface
  parameterization.
* **Bitterli et al. 2020**, *ReSTIR: Spatiotemporal Reservoir
  Resampling for Real-Time Ray Tracing with Dynamic Direct
  Lighting*. The other state-of-the-art importance-sampling
  approach. Different mechanism (reservoir over candidate
  samples), but the same problem space.
* **Wood et al. 2000**, *Surface Light Fields for 3D
  Photography*. Closest prior — uses surface parameterization
  to *store* radiance, not to *sample* it.

The gap our hypothesis addresses:

* Importance sampling for direct illumination is well-developed
  in **world** space (env CDFs, light BVHs, LTC, ReSTIR) and
  in **path-history** space (path guiding).
* Surface parameterization is well-developed for **storage**
  (texture maps, baked lighting, light fields).
* **The crossover — using a parameterization as a sampling
  prior — appears to be absent**, despite the GPU hardware
  literally being built around UV-space texture sampling
  (so the inverse-CDF lookup would be one `textureSampleLevel`
  call).

If this gap is real and the variance reduction is meaningful,
the paper writes itself: "you've been ignoring an algorithmic
asymmetry between surface parameterization and importance
sampling for 20 years."

## Experimental design

### Phase 1 — feasibility prototype

Single test scene: `outdoor_normal_bunny.gltf` (bunny under
HDR env). Cylindrical UVs already baked via `morsel parameterize`.

1. Run a **pilot pass** at 32 spp; for each first-hit
   intersection, write `radiance ÷ albedo` (the demodulated
   incident irradiance, mirroring what PT-denoise does) into a
   per-texel atlas — texel resolution matches the existing
   texture array.
2. Build a CDF in the atlas (same construction as the env
   importance tables).
3. In a **main pass**, at each surface NEE event, draw a
   direction proposal from the atlas instead of from the env
   CDF.
4. Measure variance vs spp on the main pass.

If feasibility passes (atlas-driven samples are at least as
good as the existing env CDF on this scene), proceed.

### Phase 2 — algorithm comparison

Same protocol, now across:

* **Sampling methods:** uniform-hemisphere, BRDF-weighted, env-NEE
  (current), our atlas-prior, ReSTIR, path guiding (Müller).
* **Parameterizations:** cylindrical (cheap), LSCM (angle-preserving),
  ARAP (angle + area). Cylindrical as a sanity check; the
  hypothesis says LSCM / ARAP should win because of better
  bucket-area uniformity.
* **Scenes:** the existing showcase (cornell_glass_bunny,
  outdoor_normal_bunny) + 3 new scenes designed to stress the
  hypothesis:
  * **Smooth dielectric on env** — focal caustic where the
    incident-irradiance prior is highly directional but
    parameterization is well-behaved.
  * **Matte under sun** — the easy case; baseline confirms
    nothing should win much.
  * **Rough metal on multi-light** — the case ReSTIR is built
    for; our method shouldn't tie but shouldn't lose by much.

For each (scene, parameterization, sampling-method) cell:

* Render at 16, 32, 64, 128, 256, 512 spp.
* Compute RMSE vs 8192-spp reference.
* Plot RMSE-vs-spp curves; report area-under-curve.

### Phase 3 — failure-mode analysis

For each scene where our method *doesn't* win, dissect why:

* Is the parameterization too distorted? (Measure
  conformal energy / area distortion.)
* Is the pilot pass too noisy? Sweep pilot spp.
* Is the texel resolution too coarse?

A research plan that ships a method without honest failure-mode
analysis is incomplete.

## Baselines

| Baseline | Why it's the right comparison |
|----------|-------------------------------|
| Uniform hemisphere | The naive case. Our method should crush this trivially. |
| BRDF-weighted importance sampling | The cheapest meaningful baseline. |
| Env-NEE (current quasi) | The closest existing analogue; same idea (importance-sampled CDF) but in env-direction space instead of surface-UV space. |
| ReSTIR | The state-of-the-art importance-sampling approach for real-time. Reasonable to lose to it on some scenes; we win on different ground. |
| Path guiding (Müller 2017) | The state-of-the-art for offline + spatial-subdivision. Closest in spirit to our prior. |

**Strongest baseline:** depends on the scene class. ReSTIR for
the multi-light case; path guiding for the smooth-glossy case.

## Milestones

1. **Feasibility prototype** — outdoor_normal_bunny with the
   atlas-prior method works at all + matches env-NEE at equal
   spp.
2. **LSCM + ARAP integration via morsel** — extend the
   pipeline so the parameterization choice is a runtime knob.
3. **Multi-scene sweep** — full 3-method × 3-parameterization
   × 5-scene comparison.
4. **Failure-mode analysis** — at least one written-up case
   where our method loses + an honest explanation.
5. **Write-up + figures** — RMSE plots, scene comparisons,
   failure cases.
6. **Submission.**

## Paper target

* **Primary:** Eurographics 2028. Algorithm + theory paper,
  full track.
* **Secondary:** EGSR 2027 (with feasibility + at least 2
  scenes worked out).
* **Fallback:** Computer Graphics Forum journal track.

Contribution narrative:
> "Importance sampling in path tracing has been studied
> extensively in world space (env CDFs, ReSTIR) and in
> path-history space (path guiding). We observe that **surface
> parameterization** — a well-developed area of mesh
> processing — provides a natural bucketing structure for
> importance distributions that exploits hardware texture
> sampling for inverse-CDF lookup. We build an LSCM/ARAP-driven
> importance-sampling pipeline; demonstrate variance reduction
> over uniform / BRDF / env-NEE baselines; analyse failure
> modes; identify when parameterization-driven sampling is
> the right tool vs ReSTIR / path guiding."

## Done when

**Accept criteria** (move to `writing`):

* Feasibility prototype works on `outdoor_normal_bunny.gltf`.
* Multi-scene sweep shows ≥1.5× variance reduction at fixed
  spp vs the strongest baseline on at least 3 scenes.
* Parameterization choice (LSCM vs ARAP vs cylindrical) shows
  measurable effect — the hypothesis only stands if the
  parameterization *matters*.
* Failure modes documented and explained on at least one
  losing scene.

**Abandon criteria** (move to `abandoned`, record why):

* The atlas-prior method ties with env-NEE on every scene
  (then we've discovered the env-CDF formulation is already
  using the same information differently — interesting but
  not paper-worthy).
* Parameterization choice doesn't matter (then the hypothesis
  is wrong — the variance reduction, if any, comes from the
  pilot pass alone, which is well-known).
* The texture-resolution / atlas-storage cost dominates the
  variance reduction at typical scene scales.

## Findings

*(none yet — move from `hypothesis` to `experimenting` when
the feasibility prototype runs.)*
