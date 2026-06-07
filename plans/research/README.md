# Research plans (`plans/research/`)

## Purpose

This directory holds **research plans** — hypotheses, experimental
designs, and writing roadmaps that aim at **paper-shaped
outcomes** rather than feature-shaped ones.

Research plans share DNA with the implementation plans in
`plans/NNNN-*.md` (a written-down goal, a "what's already in"
context block, milestones, "Done when" criteria) but are
structured to **track uncertainty** instead of feature
completion. The expected end state of an implementation plan is
"code lands and CI stays green." The expected end state of a
research plan is "we either understand the answer or we
recorded what we tried and why we stopped."

## Naming convention

Files are `RNNNN-<slug>.md`, four-digit zero-padded, monotone.
The R prefix differentiates research plans from implementation
plans without sharing their numbering — implementation plans
keep ticking up linearly through `0021`, `0022`, ..., and
research plans run their own counter `R0001`, `R0002`, ...

Slugs are short and topical (`R0001-tonemap-halo-bound.md`,
`R0002-param-driven-sampling.md`). No phase / track prefix is
required at this point — the `R` already distinguishes the
track. If we acquire enough research threads that they need
sub-organising, revisit then.

## Status states

```
hypothesis   — framed; no experiments run yet
experimenting — running baselines / sweeps
writing      — paper drafting
submitted    — under review at venue
accepted     — paper accepted (link the publication)
abandoned    — didn't pan out; leave the document as a record
                of what was tried and why we stopped
```

A research plan can move backwards (e.g. `experimenting →
hypothesis` if the framing turns out wrong and we need to
re-pose). Implementation plans cannot — once `completed` they
stay completed. This asymmetry reflects research's contingent
nature.

## Required sections

A research plan body should include:

```
## Hypothesis
The one-sentence claim we're trying to support / refute.

## Related work
What's known. Distinguish prior art we'll build on (cited
positively) from prior art our hypothesis disagrees with
(stated as the gap we're addressing).

## Experimental design
Concrete protocol. Data, baselines, metrics, sweep grids.

## Baselines
Each baseline gets a one-line description + a reason it's the
right comparison. "Strongest baseline" should be identified.

## Milestones
Like implementation plans, but with experiment-shaped tasks
(run sweep X; produce figure Y; derive bound Z) rather than
code-shaped ones.

## Paper target
Concrete venue + submission deadline if known. "EGSR 2027" or
"SIGGRAPH Short Papers 2026" — not "a paper" in the
abstract.

## Done when
The criteria that move the plan to `accepted` or `abandoned`.
Be specific about what counts as a positive result vs a
graceful abandonment.

## Findings
Accretes over time. Each finding is timestamped + brief. The
findings section is what makes a research plan a useful
historical record even if the plan ends up `abandoned` — it
records what was *learned*, not just what was *done*.
```

## Cross-referencing

Research plans **cite** implementation plans they build on:
> "Builds on plan 0017 PT-denoise's existing à-trous wavelet
> + AOV pipeline."

Implementation plans **point at** research follow-ups when a
sub-question is research-shaped rather than feature-shaped:
> "Variance-adaptive σ_c is the gold-standard SVGF approach;
> see `research/R0001-tonemap-halo-bound.md` for the
> theoretical alternative."

The same `[[name]]` linking convention used in auto-memory
files applies — link freely between research plans + implementation
plans + memory entries.

## Findings accretion

Unlike implementation milestones (which flip from `[ ]` to `[x]`
once and stay there), research findings **accumulate**. Each
finding is its own bullet, prefixed with the date you observed
it:

```
## Findings

- **2026-06-08** — Reinhard pre-tonemap reduces halo radius
  by 4× at L/ℓ = 100, σ_c = 0.5 across 5-scene sweep. Bound
  not yet derived but empirical pattern is consistent with
  the predicted scaling.
- **2026-06-15** — Bound derivation breaks down at k > 6
  iterations. Need to extend.
- ...
```

The most recent finding sits at the top.

## When a research plan reaches `writing`

Spawn a `papers/` directory at the repo root with:
- `papers/<plan-slug>/draft.tex` (or `.md`)
- `papers/<plan-slug>/figures/`
- `papers/<plan-slug>/experiments/` — scripts + run logs
- `papers/<plan-slug>/README.md` — pointer back to the plan

Keep `plans/research/RNNNN-*.md` as the **research-direction
record**; keep `papers/<slug>/` as the **paper-drafting
workspace**. Cross-link.

## Currently active

- [`R0001-tonemap-halo-bound.md`](R0001-tonemap-halo-bound.md)
  — closed-form bound on à-trous wavelet HDR halos; status:
  hypothesis.
- [`R0002-param-driven-sampling.md`](R0002-param-driven-sampling.md)
  — mesh-parameterization-driven importance sampling
  (LSCM/ARAP → UV-space prior → hardware texture sampling for
  inverse CDF); status: hypothesis.
